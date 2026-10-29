"""Sparse spectral pooling with historical occurrence or weighted dictionary MLPs."""
import math
from typing import Sequence
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .layers import build_mlp, without_dropout


def validate_pooling(pooling: str, normalization: str, pool_centering: str,
                     normalization_epsilon: float) -> None:
    """Reject ambiguous pooling semantics before constructing parameters."""
    if pooling not in {"occurrence_mlp", "intensity_weighted_dictionary"}:
        raise ValueError("pooling must be occurrence_mlp or intensity_weighted_dictionary.")
    allowed = ({"count"}, {"none", "zero_input"}) if pooling == "occurrence_mlp" else (
        {"l1", "l2"}, {"none", "dictionary_mean"})
    if normalization not in allowed[0] or pool_centering not in allowed[1]:
        raise ValueError(f"{pooling} requires normalization in {sorted(allowed[0])} and "
                         f"pool_centering in {sorted(allowed[1])}; changing pooling requires fresh training.")
    if (isinstance(normalization_epsilon, bool) or not isinstance(normalization_epsilon, (int, float))
            or not math.isfinite(normalization_epsilon) or normalization_epsilon <= 0):
        raise ValueError("normalization_epsilon must be finite and positive.")


class DeepSetsEncoder(nn.Module):
    """Sparse occurrence or intensity-weighted dictionary spectral pooling."""

    def __init__(
        self,
        num_peaks: int,
        embedding_dim: int,
        peak_hidden_dims: Sequence[int],
        peak_output_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        inference_peak_chunk_size: int = 262_144,
        *,
        initialize_embeddings: bool = True,
        peak_mlp_residual: bool = False,
        pool_centering: str = "none",
        pooling: str = "occurrence_mlp",
        normalization: str = "count",
        normalization_epsilon: float = 1.0e-8,
    ) -> None:
        """Initialize the learnable peak dictionary and Deep Sets MLPs.

        Args:
            num_peaks (int): Number of aligned m/z bins.
            embedding_dim (int): Peak dictionary width.
            peak_hidden_dims (Sequence[int]): Per-peak MLP hidden widths.
            peak_output_dim (int): Per-peak output width.
            activation (str): Hidden activation name.
            dropout (float): Hidden dropout probability.
            use_layer_norm (bool): Apply hidden LayerNorm.
            inference_peak_chunk_size (int): Maximum active peaks passed through
                the per-peak MLP at once when gradients are disabled.
            initialize_embeddings (bool): Defer the final embedding draw when a
                compatibility wrapper must first construct aggregation layers.
            peak_mlp_residual (bool): Enable normalized equal-width hidden skips.
            pool_centering (str): Subtract the dropout-free zero-input reference
                for nonempty occurrence pools with ``zero_input``, or subtract
                the transformed vocabulary mean with ``dictionary_mean``.
            pooling (str): Historical ``occurrence_mlp`` or
                ``intensity_weighted_dictionary`` sparse projection.
            normalization (str): Occurrence ``count`` or dictionary ``l1``/``l2``.
            normalization_epsilon (float): Positive dictionary row-norm floor.

        Returns:
            None: Module parameters are initialized.
        """

        super().__init__()
        validate_pooling(pooling, normalization, pool_centering, normalization_epsilon)
        self.pooling = pooling
        self.normalization = normalization
        self.normalization_epsilon = float(normalization_epsilon)
        self.pool_centering = pool_centering
        self.peak_mlp_residual = peak_mlp_residual
        self.num_peaks = int(num_peaks)
        self.embedding_dim = int(embedding_dim)
        self.inference_peak_chunk_size = int(inference_peak_chunk_size)
        if self.inference_peak_chunk_size <= 0:
            raise ValueError("inference_peak_chunk_size must be positive.")
        self.embedding = nn.Embedding(self.num_peaks, self.embedding_dim)
        self.peak_mlp = build_mlp(
            self.embedding_dim,
            peak_hidden_dims,
            peak_output_dim,
            activation,
            dropout,
            use_layer_norm,
            residual=peak_mlp_residual,
        )
        if initialize_embeddings:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def effective_dictionary(self) -> torch.Tensor:
        """Transform every peak once, sharing dropout across all rows in this call.

        The result is invocation-local: no detached cache can outlive an update.
        Occurrence pooling has intensity-conditioned responses, not a dictionary.
        """
        if self.pooling != "intensity_weighted_dictionary":
            raise ValueError("occurrence_mlp has no intensity-independent effective dictionary.")
        return self._transform_dictionary(self.embedding.weight)

    def _transform_dictionary(self, embeddings: torch.Tensor) -> torch.Tensor:
        # The explicit tensor argument also tells checkpoint which device RNG
        # to preserve. A closure-only call does not capture CUDA dropout state.
        values = self.peak_mlp(embeddings)
        if self.pool_centering == "dictionary_mean":
            values = values - values.mean(dim=0, keepdim=True)
        return values

    def _dictionary_pool(self, dictionary: torch.Tensor, peak_indices: torch.Tensor,
                         intensities: torch.Tensor, sample_indices: torch.Tensor,
                         peak_counts: torch.Tensor) -> torch.Tensor:
        # Intensities have already been transformed by the data loader. Coalesce
        # before norms so duplicate entries denote a single combined peak.
        matrix = torch.sparse_coo_tensor(
            torch.stack((sample_indices, peak_indices)), intensities,
            (peak_counts.numel(), self.num_peaks), device=dictionary.device,
            dtype=dictionary.dtype).coalesce()
        values = matrix.values()
        row_norm = values.new_zeros(peak_counts.numel())
        row_norm = row_norm.index_add(0, matrix.indices()[0],
                                     values.square() if self.normalization == "l2" else values)
        if self.normalization == "l2":
            denominator = row_norm.clamp_min(self.normalization_epsilon ** 2).sqrt()
        else:
            denominator = row_norm.clamp_min(self.normalization_epsilon)
        # No nnz-by-output-width occurrence tensor, including during backward.
        return torch.sparse.mm(matrix, dictionary) / denominator.unsqueeze(-1)

    def forward(
        self,
        peak_indices: torch.Tensor,
        intensities: torch.Tensor,
        sample_indices: torch.Tensor,
        peak_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Pool concatenated sparse peaks into one representation per pixel.

        Args:
            peak_indices (torch.Tensor): Flattened peak dictionary indices.
            intensities (torch.Tensor): Flattened nonnegative peak intensities.
            sample_indices (torch.Tensor): Pixel membership for each flattened peak.
            peak_counts (torch.Tensor): Number of active peaks in each pixel.

        Returns:
            torch.Tensor: Pooled representations with shape ``[batch, peak_output_dim]``.
        """

        if self.pooling == "intensity_weighted_dictionary":
            return self._dictionary_pool(self.effective_dictionary(), peak_indices,
                                         intensities, sample_indices, peak_counts)
        batch_size = int(peak_counts.numel())
        if peak_indices.numel() == 0:
            pooled = self.embedding.weight.new_zeros(
                (batch_size, self.peak_mlp[-1].out_features)
            )
        elif (
            torch.is_grad_enabled()
            or peak_indices.numel() <= self.inference_peak_chunk_size
        ):
            weighted_embeddings = self.embedding(peak_indices) * intensities.unsqueeze(-1)
            peak_features = self.peak_mlp(weighted_embeddings)
            pooled = peak_features.new_zeros((batch_size, peak_features.shape[-1]))
            pooled.index_add_(0, sample_indices, peak_features)
        else:
            pooled = self.embedding.weight.new_zeros(
                (batch_size, self.peak_mlp[-1].out_features)
            )
            for start in range(0, peak_indices.numel(), self.inference_peak_chunk_size):
                stop = min(
                    start + self.inference_peak_chunk_size,
                    peak_indices.numel(),
                )
                weighted_embeddings = self.embedding(peak_indices[start:stop])
                weighted_embeddings = (
                    weighted_embeddings * intensities[start:stop].unsqueeze(-1)
                )
                peak_features = self.peak_mlp(weighted_embeddings)
                pooled.index_add_(
                    0,
                    sample_indices[start:stop],
                    peak_features,
                )
        pooled = pooled / peak_counts.clamp_min(1).to(pooled.dtype).unsqueeze(-1)
        if self.pool_centering == "zero_input":
            reference = without_dropout(
                self.peak_mlp, self.embedding.weight.new_zeros((1, self.embedding_dim)))
            pooled = torch.where(peak_counts.unsqueeze(-1) > 0, pooled - reference,
                                 torch.zeros_like(pooled))
        return pooled



    def microbatched(self, peak_indices: torch.Tensor, intensities: torch.Tensor,
                     sample_indices: torch.Tensor, peak_counts: torch.Tensor,
                     peak_budget: int = 65536, checkpointing: bool = True, chunks=None) -> torch.Tensor:
        """Pool complete spectra in bounded groups; recomputation preserves RNG."""
        if chunks is None:
            from dann.execution import spectral_chunks
            chunks = spectral_chunks(peak_counts.detach().cpu().tolist(), peak_budget)
        dictionary = None
        if self.pooling == "intensity_weighted_dictionary":
            if checkpointing and self.training and torch.is_grad_enabled():
                dictionary = checkpoint(self._transform_dictionary, self.embedding.weight, use_reentrant=False,
                                        preserve_rng_state=True)
            else:
                dictionary = self.effective_dictionary()
        pieces = []
        for start, stop, offset, end in chunks:
            args = (peak_indices[offset:end], intensities[offset:end],
                    sample_indices[offset:end] - start, peak_counts[start:stop])
            if dictionary is not None:
                piece = self._dictionary_pool(dictionary, *args)
            elif checkpointing and self.training and torch.is_grad_enabled():
                piece = checkpoint(DeepSetsEncoder.forward, self, *args,
                                   use_reentrant=False, preserve_rng_state=True)
            else:
                piece = DeepSetsEncoder.forward(self, *args)
            pieces.append(piece)
            offset = end
        if not pieces:
            if dictionary is not None:
                return self._dictionary_pool(dictionary, peak_indices, intensities, sample_indices, peak_counts)
            return DeepSetsEncoder.forward(self, peak_indices, intensities, sample_indices, peak_counts)
        return torch.cat(pieces, dim=0)

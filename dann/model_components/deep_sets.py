"""Intensity-weighted Deep Sets through active-peak mean pooling."""
from typing import Sequence
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .layers import build_mlp

class DeepSetsEncoder(nn.Module):
    """PDF-defined Deep Sets encoder over only active MSI peaks."""

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

        Returns:
            None: Module parameters are initialized.
        """

        super().__init__()
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
        )
        if initialize_embeddings:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        peak_indices: torch.Tensor,
        intensities: torch.Tensor,
        sample_indices: torch.Tensor,
        peak_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool concatenated sparse peaks into one representation per pixel.

        Args:
            peak_indices (torch.Tensor): Flattened peak dictionary indices.
            intensities (torch.Tensor): Flattened nonnegative peak intensities.
            sample_indices (torch.Tensor): Pixel membership for each flattened peak.
            peak_counts (torch.Tensor): Number of active peaks in each pixel.

        Returns:
            torch.Tensor: Pooled representations with shape ``[batch, peak_output_dim]``.
        """

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
        return pooled



    def microbatched(self, peak_indices: torch.Tensor, intensities: torch.Tensor,
                     sample_indices: torch.Tensor, peak_counts: torch.Tensor,
                     peak_budget: int = 65536, checkpointing: bool = True) -> torch.Tensor:
        """Pool complete spectra in bounded groups; recomputation preserves RNG."""
        counts = peak_counts.detach().cpu().tolist()
        boundaries = [0]
        mass = 0
        for row, count in enumerate(counts):
            if mass + count > peak_budget and row > boundaries[-1]:
                boundaries.append(row)
                mass = 0
            mass += count
        boundaries.append(len(counts))
        pieces = []
        offset = 0
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            end = offset + sum(counts[start:stop])
            args = (peak_indices[offset:end], intensities[offset:end],
                    sample_indices[offset:end] - start, peak_counts[start:stop])
            if checkpointing and self.training and torch.is_grad_enabled():
                piece = checkpoint(DeepSetsEncoder.forward, self, *args,
                                   use_reentrant=False, preserve_rng_state=True)
            else:
                piece = DeepSetsEncoder.forward(self, *args)
            pieces.append(piece)
            offset = end
        return torch.cat(pieces, dim=0)

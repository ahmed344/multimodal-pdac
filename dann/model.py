"""Composition, stable ZILN transformation, and adversarial gradient reversal."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import torch
from torch import nn
from torch.nn import functional as F
from dann.model_components.layers import build_activation, build_mlp
from dann.model_components.deep_sets import DeepSetsEncoder
from dann.model_components.aggregation_mlp import MLPAggregation
from dann.model_components.aggregation_cnn import CNNAggregation
from dann.model_components.biology_mlp import MLPBiology
from dann.model_components.biology_cnn import CNNBiology
from dann.model_components.discriminator_mlp import MLPDiscriminator
from dann.model_components.discriminator_cnn import CNNDiscriminator


def transform_biology(raw: torch.Tensor, num_targets: int, sigma_min: float) -> dict[str, torch.Tensor]:
    """Apply the same ZILN parameterization to either biology component."""
    raw = raw.reshape(-1, num_targets, 3)
    return {"pi_logits": raw[..., 0], "pi": raw[..., 0].sigmoid(),
            "mu": raw[..., 1], "sigma": F.softplus(raw[..., 2]) + sigma_min}


class IntensityWeightedPeakEncoder(DeepSetsEncoder):
    """Compatibility wrapper around spectral pooling and MLP aggregation."""
    def __init__(self, num_peaks, embedding_dim, peak_hidden_dims, peak_output_dim,
                 aggregation_hidden_dims, latent_dim, activation="gelu", dropout=0.,
                 use_layer_norm=True, inference_peak_chunk_size=262144):
        super().__init__(num_peaks, embedding_dim, peak_hidden_dims, peak_output_dim,
                         activation, dropout, use_layer_norm, inference_peak_chunk_size,
                         initialize_embeddings=False)
        self.aggregation_mlp = MLPAggregation(
            peak_output_dim, aggregation_hidden_dims, latent_dim, activation=activation,
            dropout=dropout, use_layer_norm=use_layer_norm)
        # Preserve the original seeded draw order: peak MLP, aggregation, embedding.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, *args, **kwargs):
        return self.aggregation_mlp(super().forward(*args, **kwargs))


class BiologyPredictor(MLPBiology):
    """Legacy public pointwise predictor, including shared transformation."""
    def __init__(self, latent_dim, hidden_dims, num_targets, sigma_min,
                 activation, dropout, use_layer_norm):
        super().__init__(latent_dim, hidden_dims, num_targets * 3,
                         activation=activation, dropout=dropout, use_layer_norm=use_layer_norm)
        self.num_targets, self.sigma_min = num_targets, sigma_min

    def forward(self, latent):
        return transform_biology(super().forward(latent), self.num_targets, self.sigma_min)


class BatchDiscriminator(MLPDiscriminator):
    def __init__(self, latent_dim, hidden_dims, num_batches, activation, dropout, use_layer_norm):
        super().__init__(latent_dim, hidden_dims, num_batches, activation=activation,
                         dropout=dropout, use_layer_norm=use_layer_norm)


class _GradientReversalFunction(torch.autograd.Function):
    """Identity forward operation with sign-reversed backward gradients."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, strength: float) -> torch.Tensor:
        """Return inputs unchanged and retain the backward strength.

        Args:
            ctx (Any): PyTorch autograd context.
            inputs (torch.Tensor): Encoder latent tensor.
            strength (float): Gradient reversal multiplier.

        Returns:
            torch.Tensor: Identity view of ``inputs``.
        """

        ctx.strength = float(strength)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Reverse and scale the encoder-bound gradient.

        Args:
            ctx (Any): PyTorch autograd context holding the strength.
            gradient (torch.Tensor): Upstream discriminator gradient.

        Returns:
            tuple[torch.Tensor, None]: Reversed input gradient and no strength gradient.
        """

        return -ctx.strength * gradient, None


class GradientReversal(nn.Module):
    """Module wrapper for the gradient reversal autograd operation."""

    def forward(self, inputs: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """Apply identity forward and reversed backward behavior.

        Args:
            inputs (torch.Tensor): Shared latent tensor.
            strength (float): Nonnegative gradient reversal strength.

        Returns:
            torch.Tensor: Identity-valued latent tensor.
        """

        return _GradientReversalFunction.apply(inputs, strength)


class AdversarialLatentFusion(nn.Module):
    """Deep Sets, selectable aggregation, and two independently selected heads."""
    def __init__(self, encoder, biology_predictor, batch_discriminator):
        super().__init__()
        self.encoder = encoder
        self.biology_predictor = biology_predictor
        self.batch_discriminator = batch_discriminator
        self.gradient_reversal = GradientReversal()
        self.spatial = False
        self.peak_budget = 65536
        self.checkpointing = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any], num_batches: int,
                    num_targets: int) -> "AdversarialLatentFusion":
        """Construct selected components, interpreting flat legacy settings as MLPs."""
        from dann.config import component_settings
        values = config["model"]
        settings = component_settings(values)
        common = {"activation": values["activation"], "dropout": float(values["dropout"]),
                  "use_layer_norm": bool(values["use_layer_norm"])}
        spectral = settings["spectral_encoder"]["deep_sets"]
        encoder = IntensityWeightedPeakEncoder(
            int(values["num_peaks"]), **spectral,
            aggregation_hidden_dims=settings["aggregation"]["mlp"]["hidden_dims"],
            latent_dim=values["latent_dim"],
            inference_peak_chunk_size=values.get("inference_peak_chunk_size", 262144), **common)
        if settings["aggregation"]["type"] == "cnn":
            encoder.aggregation_mlp = CNNAggregation(spectral["peak_output_dim"],
                values["latent_dim"], **settings["aggregation"]["cnn"])
        bio = settings["heads"]["biology"]
        disc = settings["heads"]["discriminator"]
        biology = (BiologyPredictor(values["latent_dim"], bio["mlp"]["hidden_dims"],
                   num_targets, values["sigma_min"], **common) if bio["type"] == "mlp"
                   else CNNBiology(values["latent_dim"], num_targets * 3, **bio["cnn"]))
        discriminator = (BatchDiscriminator(values["latent_dim"], disc["mlp"]["hidden_dims"],
                         num_batches, **common) if disc["type"] == "mlp" else
                         CNNDiscriminator(values["latent_dim"], num_batches, **disc["cnn"]))
        model = cls(encoder, biology, discriminator)
        model.num_targets, model.sigma_min = num_targets, values["sigma_min"]
        model.latent_dim = int(values["latent_dim"])
        model.spatial = any(x["type"] == "cnn" for x in (settings["aggregation"], bio, disc))
        model.peak_budget = values.get("spectral_peak_budget", 65536)
        model.checkpointing = values.get("spectral_checkpointing", True)
        model.halo = encoder.aggregation_mlp.radius + max(biology.radius, discriminator.radius)
        return model

    @staticmethod
    def _on_map(component, value, mask):
        if component.radius:
            return component(value, mask)
        result = component(value.movedim(1, -1))
        return result.movedim(-1, 1) * mask

    def _latent_map(self, batch):
        if "occupancy" not in batch:
            raise ValueError("CNN components require spatial tiles with occupancy and core indices.")
        args = [batch[key] for key in ("peak_indices", "intensities", "sample_indices", "peak_counts")]
        pooled = self.encoder.microbatched(*args, peak_budget=self.peak_budget,
                                          checkpointing=self.checkpointing)
        mask = batch["occupancy"]
        n, _, h, w = mask.shape
        flat = pooled.new_zeros((n*h*w, pooled.shape[-1]))
        flat = flat.index_copy(0, batch["spatial_positions"], pooled)
        grid = flat.reshape(n, h, w, -1).movedim(-1, 1)
        return self._on_map(self.encoder.aggregation_mlp, grid, mask)

    @staticmethod
    def _select(grid, batch):
        return grid.movedim(1, -1).reshape(-1, grid.shape[1])[batch["core_positions"]]

    def encode(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return core aggregation latents, preserving context during aggregation."""
        if self.spatial:
            return self._select(self._latent_map(batch), batch)
        return self.encoder(*[batch[key] for key in
                            ("peak_indices", "intensities", "sample_indices", "peak_counts")])

    def forward(self, batch: Mapping[str, torch.Tensor],
                grl_strength: float = 1.0) -> dict[str, torch.Tensor]:
        """Predict aligned core outputs, retaining spatial context for both heads."""
        if not self.spatial:
            latent = self.encode(batch)
            return {"latent": latent, **self.biology_predictor(latent),
                    "batch_logits": self.batch_discriminator(self.gradient_reversal(latent, grl_strength))}
        grid = self._latent_map(batch)
        mask = batch["occupancy"]
        latent = self._select(grid, batch)
        # Keep the latent halo until both heads have consumed their neighborhoods.
        if self.biology_predictor.radius:
            biology = transform_biology(self._select(self.biology_predictor(grid, mask), batch),
                                        self.num_targets, self.sigma_min)
        else:
            biology = self.biology_predictor(latent)
        reversed_grid = self.gradient_reversal(grid, grl_strength)
        logits = self._select(self._on_map(self.batch_discriminator, reversed_grid, mask), batch)
        return {"latent": latent, **biology, "batch_logits": logits}

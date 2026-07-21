"""Neural modules for sparse adversarial latent fusion."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def build_activation(name: str) -> nn.Module:
    """Construct an activation module by configuration name.

    Args:
        name (str): Activation name.

    Returns:
        nn.Module: Stateless activation module.
    """

    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation {name!r}; choose from {sorted(activations)}.")
    return activations[name]()


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str,
    dropout: float,
    use_layer_norm: bool,
) -> nn.Sequential:
    """Build an MLP with normalized, activated hidden layers.

    Args:
        input_dim (int): Input feature width.
        hidden_dims (Sequence[int]): Hidden layer widths.
        output_dim (int): Linear output width.
        activation (str): Hidden activation name.
        dropout (float): Hidden dropout probability.
        use_layer_norm (bool): Add LayerNorm after each hidden linear layer.

    Returns:
        nn.Sequential: Configured feed-forward network.
    """

    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.append(nn.Linear(previous, int(width)))
        if use_layer_norm:
            layers.append(nn.LayerNorm(int(width)))
        layers.append(build_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        previous = int(width)
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class IntensityWeightedPeakEncoder(nn.Module):
    """PDF-defined Deep Sets encoder over only active MSI peaks."""

    def __init__(
        self,
        num_peaks: int,
        embedding_dim: int,
        peak_hidden_dims: Sequence[int],
        peak_output_dim: int,
        aggregation_hidden_dims: Sequence[int],
        latent_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        """Initialize the learnable peak dictionary and Deep Sets MLPs.

        Args:
            num_peaks (int): Number of aligned m/z bins.
            embedding_dim (int): Peak dictionary width.
            peak_hidden_dims (Sequence[int]): Per-peak MLP hidden widths.
            peak_output_dim (int): Per-peak output width.
            aggregation_hidden_dims (Sequence[int]): Post-pooling hidden widths.
            latent_dim (int): Shared latent width.
            activation (str): Hidden activation name.
            dropout (float): Hidden dropout probability.
            use_layer_norm (bool): Apply hidden LayerNorm.

        Returns:
            None: Module parameters are initialized.
        """

        super().__init__()
        self.num_peaks = int(num_peaks)
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(self.num_peaks, self.embedding_dim)
        self.peak_mlp = build_mlp(
            self.embedding_dim,
            peak_hidden_dims,
            peak_output_dim,
            activation,
            dropout,
            use_layer_norm,
        )
        self.aggregation_mlp = build_mlp(
            peak_output_dim,
            aggregation_hidden_dims,
            latent_dim,
            activation,
            dropout,
            use_layer_norm,
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        peak_indices: torch.Tensor,
        intensities: torch.Tensor,
        sample_indices: torch.Tensor,
        peak_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Encode concatenated sparse peaks into one latent vector per pixel.

        Args:
            peak_indices (torch.Tensor): Flattened peak dictionary indices.
            intensities (torch.Tensor): Flattened nonnegative peak intensities.
            sample_indices (torch.Tensor): Pixel membership for each flattened peak.
            peak_counts (torch.Tensor): Number of active peaks in each pixel.

        Returns:
            torch.Tensor: Latent matrix with shape ``[batch, latent_dim]``.
        """

        batch_size = int(peak_counts.numel())
        if peak_indices.numel() == 0:
            pooled = self.embedding.weight.new_zeros(
                (batch_size, self.peak_mlp[-1].out_features)
            )
        else:
            weighted_embeddings = self.embedding(peak_indices) * intensities.unsqueeze(-1)
            peak_features = self.peak_mlp(weighted_embeddings)
            pooled = peak_features.new_zeros((batch_size, peak_features.shape[-1]))
            pooled.index_add_(0, sample_indices, peak_features)
            pooled = pooled / peak_counts.clamp_min(1).to(pooled.dtype).unsqueeze(-1)
        return self.aggregation_mlp(pooled)


class BiologyPredictor(nn.Module):
    """Predict four zero-inflated logit-normal parameter triplets."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int],
        num_targets: int,
        sigma_min: float,
        activation: str,
        dropout: float,
        use_layer_norm: bool,
    ) -> None:
        """Initialize the biology prediction head.

        Args:
            latent_dim (int): Shared latent input width.
            hidden_dims (Sequence[int]): Predictor hidden widths.
            num_targets (int): Number of density targets.
            sigma_min (float): Positive standard-deviation floor.
            activation (str): Hidden activation name.
            dropout (float): Hidden dropout probability.
            use_layer_norm (bool): Apply hidden LayerNorm.

        Returns:
            None: Module parameters are initialized.
        """

        super().__init__()
        self.num_targets = int(num_targets)
        self.sigma_min = float(sigma_min)
        self.network = build_mlp(
            latent_dim,
            hidden_dims,
            self.num_targets * 3,
            activation,
            dropout,
            use_layer_norm,
        )

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map latent vectors to stable ZILN parameters.

        Args:
            latent (torch.Tensor): Shared latent matrix.

        Returns:
            dict[str, torch.Tensor]: ``pi_logits``, ``pi``, ``mu``, and ``sigma``.
        """

        raw = self.network(latent).reshape(-1, self.num_targets, 3)
        pi_logits = raw[..., 0]
        mu = raw[..., 1]
        sigma = F.softplus(raw[..., 2]) + self.sigma_min
        return {
            "pi_logits": pi_logits,
            "pi": torch.sigmoid(pi_logits),
            "mu": mu,
            "sigma": sigma,
        }


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


class BatchDiscriminator(nn.Module):
    """Predict slide/batch identity from gradient-reversed latents."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int],
        num_batches: int,
        activation: str,
        dropout: float,
        use_layer_norm: bool,
    ) -> None:
        """Initialize the batch classification MLP.

        Args:
            latent_dim (int): Shared latent width.
            hidden_dims (Sequence[int]): Discriminator hidden widths.
            num_batches (int): Number of batch classes.
            activation (str): Hidden activation name.
            dropout (float): Hidden dropout probability.
            use_layer_norm (bool): Apply hidden LayerNorm.

        Returns:
            None: Module parameters are initialized.
        """

        super().__init__()
        self.network = build_mlp(
            latent_dim,
            hidden_dims,
            num_batches,
            activation,
            dropout,
            use_layer_norm,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Produce unnormalized batch logits.

        Args:
            latent (torch.Tensor): Gradient-reversed latent matrix.

        Returns:
            torch.Tensor: Batch logits with shape ``[batch, num_batches]``.
        """

        return self.network(latent)


class AdversarialLatentFusion(nn.Module):
    """Shared sparse encoder with competing biology and batch heads."""

    def __init__(
        self,
        encoder: IntensityWeightedPeakEncoder,
        biology_predictor: BiologyPredictor,
        batch_discriminator: BatchDiscriminator,
    ) -> None:
        """Compose the encoder and two prediction branches.

        Args:
            encoder (IntensityWeightedPeakEncoder): Sparse Deep Sets encoder.
            biology_predictor (BiologyPredictor): Four-target ZILN head.
            batch_discriminator (BatchDiscriminator): Adversarial batch head.

        Returns:
            None: Child modules are registered.
        """

        super().__init__()
        self.encoder = encoder
        self.biology_predictor = biology_predictor
        self.batch_discriminator = batch_discriminator
        self.gradient_reversal = GradientReversal()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        num_batches: int,
        num_targets: int,
    ) -> "AdversarialLatentFusion":
        """Construct the complete model from YAML-derived configuration.

        Args:
            config (Mapping[str, Any]): Complete DANN configuration.
            num_batches (int): Number of encoded batch classes.
            num_targets (int): Number of IHC density targets.

        Returns:
            AdversarialLatentFusion: Configured model.
        """

        values = config["model"]
        common = {
            "activation": values["activation"],
            "dropout": float(values["dropout"]),
            "use_layer_norm": bool(values["use_layer_norm"]),
        }
        encoder = IntensityWeightedPeakEncoder(
            num_peaks=int(values["num_peaks"]),
            embedding_dim=int(values["embedding_dim"]),
            peak_hidden_dims=values["peak_hidden_dims"],
            peak_output_dim=int(values["peak_output_dim"]),
            aggregation_hidden_dims=values["aggregation_hidden_dims"],
            latent_dim=int(values["latent_dim"]),
            **common,
        )
        biology_predictor = BiologyPredictor(
            latent_dim=int(values["latent_dim"]),
            hidden_dims=values["biology_hidden_dims"],
            num_targets=num_targets,
            sigma_min=float(values["sigma_min"]),
            **common,
        )
        discriminator = BatchDiscriminator(
            latent_dim=int(values["latent_dim"]),
            hidden_dims=values["discriminator_hidden_dims"],
            num_batches=num_batches,
            **common,
        )
        return cls(encoder, biology_predictor, discriminator)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode one collated sparse batch.

        Args:
            batch (Mapping[str, torch.Tensor]): Sparse collator output.

        Returns:
            torch.Tensor: Shared latent matrix.
        """

        return self.encoder(
            batch["peak_indices"],
            batch["intensities"],
            batch["sample_indices"],
            batch["peak_counts"],
        )

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        grl_strength: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Run encoder, ZILN head, and adversarial discriminator.

        Args:
            batch (Mapping[str, torch.Tensor]): Sparse collator output.
            grl_strength (float): Encoder-bound discriminator gradient multiplier.

        Returns:
            dict[str, torch.Tensor]: Latent, biology parameters, and batch logits.
        """

        latent = self.encode(batch)
        biology = self.biology_predictor(latent)
        reversed_latent = self.gradient_reversal(latent, grl_strength)
        return {
            "latent": latent,
            **biology,
            "batch_logits": self.batch_discriminator(reversed_latent),
        }

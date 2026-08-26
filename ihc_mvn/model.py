"""Sparse MSI encoder and multivariate hurdle model with random intercepts."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def build_activation(name: str) -> nn.Module:
    """Build an activation module.

    Args:
        name (str): Case-insensitive activation name.

    Returns:
        nn.Module: A newly constructed activation module.
    """

    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
    }
    key = str(name).lower()
    if key not in activations:
        raise ValueError(f"Unknown activation {name!r}; choose from {sorted(activations)}.")
    return activations[key]()


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str = "gelu",
    dropout: float = 0.0,
    use_layer_norm: bool = True,
) -> nn.Sequential:
    """Build an MLP whose final layer is linear.

    Args:
        input_dim (int): Input feature width.
        hidden_dims (Sequence[int]): Hidden feature widths.
        output_dim (int): Output feature width.
        activation (str): Hidden-layer activation name.
        dropout (float): Hidden-layer dropout probability.
        use_layer_norm (bool): Whether to normalize each hidden layer.

    Returns:
        nn.Sequential: Configured feed-forward network.
    """

    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must lie in [0, 1).")
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for raw_width in hidden_dims:
        width = int(raw_width)
        if width <= 0:
            raise ValueError("All hidden dimensions must be positive.")
        layers.append(nn.Linear(previous, width))
        if use_layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(build_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        previous = width
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


def _inverse_softplus(value: float) -> float:
    """Compute the scalar inverse of softplus stably.

    Args:
        value (float): Strictly positive softplus output.

    Returns:
        float: Input whose softplus is approximately ``value``.
    """

    if value <= 0.0:
        raise ValueError("The inverse-softplus value must be positive.")
    return value + math.log(-math.expm1(-value))


def _config_value(
    values: Mapping[str, Any],
    names: Sequence[str],
    default: Any,
) -> Any:
    """Read the first present configuration alias.

    Args:
        values (Mapping[str, Any]): Configuration section.
        names (Sequence[str]): Keys in decreasing priority.
        default (Any): Value returned when no key is present.

    Returns:
        Any: Selected configuration value.
    """

    for name in names:
        if name in values:
            return values[name]
    return default


class IntensityWeightedPeakEncoder(nn.Module):
    """Deep Sets encoder over concatenated active MSI peaks."""

    def __init__(
        self,
        num_peaks: int,
        embedding_dim: int,
        peak_hidden_dims: Sequence[int],
        peak_output_dim: int,
        aggregation_hidden_dims: Sequence[int],
        latent_dim: int = 16,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        inference_peak_chunk_size: int = 262_144,
    ) -> None:
        """Initialize the sparse intensity-weighted encoder.

        Args:
            num_peaks (int): Number of aligned MSI peak bins.
            embedding_dim (int): Learnable peak embedding width.
            peak_hidden_dims (Sequence[int]): Per-peak MLP hidden widths.
            peak_output_dim (int): Per-peak representation width.
            aggregation_hidden_dims (Sequence[int]): Post-pooling MLP widths.
            latent_dim (int): Pixel latent width.
            activation (str): MLP activation name.
            dropout (float): MLP dropout probability.
            use_layer_norm (bool): Whether hidden layers use layer normalization.
            inference_peak_chunk_size (int): Maximum peaks processed per no-grad chunk.

        Returns:
            None: Module parameters are initialized.
        """

        super().__init__()
        if num_peaks <= 0:
            raise ValueError("num_peaks must be positive.")
        if inference_peak_chunk_size <= 0:
            raise ValueError("inference_peak_chunk_size must be positive.")
        self.num_peaks = int(num_peaks)
        self.peak_output_dim = int(peak_output_dim)
        self.latent_dim = int(latent_dim)
        self.inference_peak_chunk_size = int(inference_peak_chunk_size)
        self.embedding = nn.Embedding(self.num_peaks, int(embedding_dim))
        self.peak_mlp = build_mlp(
            int(embedding_dim),
            peak_hidden_dims,
            self.peak_output_dim,
            activation,
            dropout,
            use_layer_norm,
        )
        self.aggregation_mlp = build_mlp(
            self.peak_output_dim,
            aggregation_hidden_dims,
            self.latent_dim,
            activation,
            dropout,
            use_layer_norm,
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def _encode_peak_slice(
        self,
        peak_indices: torch.Tensor,
        intensities: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one contiguous slice of active peaks.

        Args:
            peak_indices (torch.Tensor): One-dimensional peak indices.
            intensities (torch.Tensor): Matching one-dimensional intensities.

        Returns:
            torch.Tensor: Per-peak features with shape ``[N, peak_output_dim]``.
        """

        weighted = self.embedding(peak_indices) * intensities.unsqueeze(-1)
        return self.peak_mlp(weighted)

    def forward(
        self,
        peak_indices: torch.Tensor,
        intensities: torch.Tensor,
        sample_indices: torch.Tensor,
        peak_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Encode concatenated sparse peaks using per-pixel mean pooling.

        Args:
            peak_indices (torch.Tensor): Flattened active peak indices, shape ``[N]``.
            intensities (torch.Tensor): Flattened active intensities, shape ``[N]``.
            sample_indices (torch.Tensor): Pixel index for each peak, shape ``[N]``.
            peak_counts (torch.Tensor): Active-peak count per pixel, shape ``[B]``.

        Returns:
            torch.Tensor: Pixel latent matrix with shape ``[B, latent_dim]``.
        """

        if peak_indices.ndim != 1 or intensities.ndim != 1 or sample_indices.ndim != 1:
            raise ValueError("Sparse peak inputs must be one-dimensional.")
        if not (
            peak_indices.numel() == intensities.numel() == sample_indices.numel()
        ):
            raise ValueError("Sparse peak inputs must have identical lengths.")
        if peak_counts.ndim != 1:
            raise ValueError("peak_counts must be one-dimensional.")
        batch_size = int(peak_counts.numel())
        pooled = self.embedding.weight.new_zeros((batch_size, self.peak_output_dim))
        num_active = int(peak_indices.numel())
        if num_active > 0:
            if torch.is_grad_enabled() or num_active <= self.inference_peak_chunk_size:
                features = self._encode_peak_slice(peak_indices, intensities)
                pooled.index_add_(0, sample_indices, features)
            else:
                for start in range(0, num_active, self.inference_peak_chunk_size):
                    stop = min(start + self.inference_peak_chunk_size, num_active)
                    features = self._encode_peak_slice(
                        peak_indices[start:stop],
                        intensities[start:stop],
                    )
                    pooled.index_add_(0, sample_indices[start:stop], features)
        denominator = peak_counts.clamp_min(1).to(pooled.dtype).unsqueeze(-1)
        return self.aggregation_mlp(pooled / denominator)


class MultivariateHurdleHead(nn.Module):
    """MSI-only hurdle, standardized mean, and marginal-scale head."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int],
        num_targets: int = 4,
        sigma_min: float = 1e-4,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        """Initialize the MSI prediction head.

        Args:
            latent_dim (int): Encoder latent width.
            hidden_dims (Sequence[int]): Prediction-head hidden widths.
            num_targets (int): Number of IHC targets.
            sigma_min (float): Strictly positive marginal-scale floor.
            activation (str): Hidden-layer activation.
            dropout (float): Hidden-layer dropout probability.
            use_layer_norm (bool): Whether hidden layers use layer normalization.

        Returns:
            None: Module parameters are initialized.
        """

        super().__init__()
        if num_targets <= 0:
            raise ValueError("num_targets must be positive.")
        if sigma_min <= 0.0:
            raise ValueError("sigma_min must be positive.")
        self.num_targets = int(num_targets)
        self.sigma_min = float(sigma_min)
        self.network = build_mlp(
            int(latent_dim),
            hidden_dims,
            3 * self.num_targets,
            activation,
            dropout,
            use_layer_norm,
        )

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict MSI-only hurdle logits, means, and positive scales.

        Args:
            latent (torch.Tensor): Pixel latent matrix with shape ``[B, d]``.

        Returns:
            dict[str, torch.Tensor]: MSI-only parameters, each shaped ``[B, T]``.
        """

        raw = self.network(latent).reshape(-1, self.num_targets, 3)
        return {
            "hurdle_logits_msi": raw[..., 0],
            "mean_msi": raw[..., 1],
            "scales": F.softplus(raw[..., 2]) + self.sigma_min,
        }


class NonCenteredRandomIntercepts(nn.Module):
    """Patient and slide random intercepts for hurdle logits and means."""

    def __init__(
        self,
        num_patients: int,
        num_slides: int,
        num_targets: int = 4,
        initial_scale: float = 0.05,
        scale_floor: float = 1e-6,
        scale_prior_std: float = 1.0,
    ) -> None:
        """Initialize non-centered random intercept parameters.

        Args:
            num_patients (int): Number of known patient codes.
            num_slides (int): Number of known slide codes.
            num_targets (int): Number of modeled targets.
            initial_scale (float): Initial prior scale for every intercept family.
            scale_floor (float): Positive numerical floor on learned prior scales.
            scale_prior_std (float): Half-normal prior standard deviation for scales.

        Returns:
            None: Random-effect parameters are initialized.
        """

        super().__init__()
        if num_patients < 0 or num_slides < 0:
            raise ValueError("Random-effect cardinalities cannot be negative.")
        if num_targets <= 0:
            raise ValueError("num_targets must be positive.")
        if scale_floor <= 0.0 or initial_scale <= scale_floor:
            raise ValueError("initial_scale must exceed a positive scale_floor.")
        if scale_prior_std <= 0.0:
            raise ValueError("scale_prior_std must be positive.")
        self.num_patients = int(num_patients)
        self.num_slides = int(num_slides)
        self.num_targets = int(num_targets)
        self.scale_floor = float(scale_floor)
        self.scale_prior_std = float(scale_prior_std)

        self.patient_hurdle_raw = nn.Embedding(self.num_patients, self.num_targets)
        self.patient_mean_raw = nn.Embedding(self.num_patients, self.num_targets)
        self.slide_hurdle_raw = nn.Embedding(self.num_slides, self.num_targets)
        self.slide_mean_raw = nn.Embedding(self.num_slides, self.num_targets)
        for embedding in (
            self.patient_hurdle_raw,
            self.patient_mean_raw,
            self.slide_hurdle_raw,
            self.slide_mean_raw,
        ):
            nn.init.zeros_(embedding.weight)

        raw_initial = _inverse_softplus(float(initial_scale) - self.scale_floor)
        self.patient_hurdle_scale_raw = nn.Parameter(
            torch.full((self.num_targets,), raw_initial)
        )
        self.patient_mean_scale_raw = nn.Parameter(
            torch.full((self.num_targets,), raw_initial)
        )
        self.slide_hurdle_scale_raw = nn.Parameter(
            torch.full((self.num_targets,), raw_initial)
        )
        self.slide_mean_scale_raw = nn.Parameter(
            torch.full((self.num_targets,), raw_initial)
        )

    def _positive_scale(self, raw_scale: torch.Tensor) -> torch.Tensor:
        """Transform an unconstrained scale parameter.

        Args:
            raw_scale (torch.Tensor): Unconstrained per-target scale.

        Returns:
            torch.Tensor: Strictly positive per-target scale.
        """

        return F.softplus(raw_scale) + self.scale_floor

    def _lookup(
        self,
        codes: torch.Tensor,
        embedding: nn.Embedding,
        raw_scale: torch.Tensor,
        label: str,
    ) -> torch.Tensor:
        """Look up scaled effects while mapping unknown code ``-1`` to zero.

        Args:
            codes (torch.Tensor): One-dimensional integer group codes.
            embedding (nn.Embedding): Standard-normal raw effect table.
            raw_scale (torch.Tensor): Unconstrained per-target prior scales.
            label (str): Group label used in validation errors.

        Returns:
            torch.Tensor: Scaled random effects with shape ``[B, T]``.
        """

        if codes.ndim != 1:
            raise ValueError(f"{label} codes must be one-dimensional.")
        if codes.is_floating_point() or codes.is_complex():
            raise TypeError(f"{label} codes must use an integer dtype.")
        known = codes >= 0
        if torch.any(codes < -1) or torch.any(codes[known] >= embedding.num_embeddings):
            raise ValueError(f"{label} codes must be -1 or valid encoded indices.")
        if embedding.num_embeddings == 0:
            if torch.any(known):
                raise ValueError(f"No known {label} levels were configured.")
            return raw_scale.new_zeros((codes.numel(), self.num_targets))
        safe_codes = codes.clamp_min(0)
        effects = embedding(safe_codes) * self._positive_scale(raw_scale)
        return effects * known.to(effects.dtype).unsqueeze(-1)

    def forward(
        self,
        patients: torch.Tensor,
        slides: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return patient and slide effects for both modeled branches.

        Args:
            patients (torch.Tensor): Patient codes with unknown values encoded as ``-1``.
            slides (torch.Tensor): Slide codes with unknown values encoded as ``-1``.

        Returns:
            dict[str, torch.Tensor]: Four scaled random-intercept tensors.
        """

        if patients.shape != slides.shape:
            raise ValueError("patients and slides must have identical shapes.")
        return {
            "patient_hurdle_effect": self._lookup(
                patients,
                self.patient_hurdle_raw,
                self.patient_hurdle_scale_raw,
                "patient",
            ),
            "patient_mean_effect": self._lookup(
                patients,
                self.patient_mean_raw,
                self.patient_mean_scale_raw,
                "patient",
            ),
            "slide_hurdle_effect": self._lookup(
                slides,
                self.slide_hurdle_raw,
                self.slide_hurdle_scale_raw,
                "slide",
            ),
            "slide_mean_effect": self._lookup(
                slides,
                self.slide_mean_raw,
                self.slide_mean_scale_raw,
                "slide",
            ),
        }

    def prior_nll(self) -> torch.Tensor:
        """Evaluate raw-effect Normal and scale half-Normal prior NLL.

        Args:
            None: This method uses registered module parameters.

        Returns:
            torch.Tensor: Scalar negative log-prior including normalization constants.
        """

        raw_embeddings = (
            self.patient_hurdle_raw.weight,
            self.patient_mean_raw.weight,
            self.slide_hurdle_raw.weight,
            self.slide_mean_raw.weight,
        )
        normal_constant = 0.5 * math.log(2.0 * math.pi)
        nll = sum(
            (0.5 * weight.square() + normal_constant).sum()
            for weight in raw_embeddings
        )
        raw_scales = (
            self.patient_hurdle_scale_raw,
            self.patient_mean_scale_raw,
            self.slide_hurdle_scale_raw,
            self.slide_mean_scale_raw,
        )
        half_normal_constant = math.log(self.scale_prior_std) + 0.5 * math.log(
            math.pi / 2.0
        )
        for raw_scale in raw_scales:
            scale = self._positive_scale(raw_scale)
            nll = nll + (
                0.5 * (scale / self.scale_prior_std).square()
                + half_normal_constant
            ).sum()
        return nll


class GlobalCorrelation(nn.Module):
    """Learn one shared positive-definite target correlation matrix."""

    def __init__(self, num_targets: int = 4, diagonal_floor: float = 1e-4) -> None:
        """Initialize an identity correlation parameterization.

        Args:
            num_targets (int): Correlation matrix dimension.
            diagonal_floor (float): Positive floor on the raw Cholesky diagonal.

        Returns:
            None: Correlation parameters are initialized.
        """

        super().__init__()
        if num_targets <= 0:
            raise ValueError("num_targets must be positive.")
        if not 0.0 < diagonal_floor < 1.0:
            raise ValueError("diagonal_floor must lie in (0, 1).")
        self.num_targets = int(num_targets)
        self.diagonal_floor = float(diagonal_floor)
        initial_diagonal = _inverse_softplus(1.0 - self.diagonal_floor)
        raw = torch.zeros((self.num_targets, self.num_targets))
        raw.diagonal().fill_(initial_diagonal)
        self.raw_lower = nn.Parameter(raw)

    def cholesky_factor(self) -> torch.Tensor:
        """Build the unconstrained covariance factor.

        Args:
            None: This method uses the learned lower-triangular parameter.

        Returns:
            torch.Tensor: Lower-triangular matrix with a positive diagonal.
        """

        lower = torch.tril(self.raw_lower, diagonal=-1)
        diagonal = F.softplus(self.raw_lower.diagonal()) + self.diagonal_floor
        return lower + torch.diag_embed(diagonal)

    def forward(self) -> torch.Tensor:
        """Construct the normalized shared correlation matrix.

        Args:
            None: This method uses registered module parameters.

        Returns:
            torch.Tensor: Positive-definite correlation matrix ``R``.
        """

        factor = self.cholesky_factor()
        covariance = factor @ factor.transpose(-1, -2)
        inverse_std = covariance.diagonal().rsqrt()
        return covariance * inverse_std.unsqueeze(0) * inverse_std.unsqueeze(1)

    @property
    def R(self) -> torch.Tensor:
        """Return the current shared correlation matrix.

        Args:
            None: This property uses registered module parameters.

        Returns:
            torch.Tensor: Positive-definite correlation matrix ``R``.
        """

        return self()


class IHCMultivariateModel(nn.Module):
    """Sparse MSI multivariate hurdle model with hierarchical intercepts."""

    def __init__(
        self,
        encoder: IntensityWeightedPeakEncoder,
        head: MultivariateHurdleHead,
        random_intercepts: NonCenteredRandomIntercepts,
        correlation: GlobalCorrelation,
    ) -> None:
        """Compose the encoder, head, random effects, and correlation.

        Args:
            encoder (IntensityWeightedPeakEncoder): Sparse MSI Deep Sets encoder.
            head (MultivariateHurdleHead): MSI-only prediction head.
            random_intercepts (NonCenteredRandomIntercepts): Hierarchical intercepts.
            correlation (GlobalCorrelation): Shared target correlation.

        Returns:
            None: Child modules are registered.
        """

        super().__init__()
        if head.num_targets != random_intercepts.num_targets:
            raise ValueError("Head and random effects must use the same target count.")
        if head.num_targets != correlation.num_targets:
            raise ValueError("Head and correlation must use the same target count.")
        self.encoder = encoder
        self.head = head
        self.random_intercepts = random_intercepts
        self.correlation = correlation
        self.num_targets = head.num_targets
        self.latent_dim = encoder.latent_dim

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        num_patients: int,
        num_slides: int,
        num_targets: int = 4,
    ) -> "IHCMultivariateModel":
        """Construct a model from a conventional YAML-derived mapping.

        Args:
            config (Mapping[str, Any]): Full config or its ``model`` section.
            num_patients (int): Number of encoded patient levels.
            num_slides (int): Number of encoded slide levels.
            num_targets (int): Number of IHC targets.

        Returns:
            IHCMultivariateModel: Configured multivariate hurdle model.
        """

        values = config.get("model", config)
        if not isinstance(values, Mapping):
            raise TypeError("The model configuration must be a mapping.")
        random_values = values.get("random_effects", {})
        if not isinstance(random_values, Mapping):
            raise TypeError("model.random_effects must be a mapping.")
        num_peaks = _config_value(
            values,
            ("num_peaks", "n_peaks"),
            config.get("data", {}).get("num_peaks")
            if isinstance(config.get("data", {}), Mapping)
            else None,
        )
        if num_peaks is None:
            raise KeyError("Model configuration requires num_peaks.")
        activation = str(values.get("activation", "gelu"))
        dropout = float(values.get("dropout", 0.0))
        use_layer_norm = bool(
            _config_value(values, ("use_layer_norm", "layer_norm"), True)
        )
        latent_dim = int(values.get("latent_dim", 16))
        encoder = IntensityWeightedPeakEncoder(
            num_peaks=int(num_peaks),
            embedding_dim=int(values.get("embedding_dim", 32)),
            peak_hidden_dims=values.get("peak_hidden_dims", (64, 64)),
            peak_output_dim=int(values.get("peak_output_dim", 64)),
            aggregation_hidden_dims=values.get("aggregation_hidden_dims", (64,)),
            latent_dim=latent_dim,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            inference_peak_chunk_size=int(
                values.get("inference_peak_chunk_size", 262_144)
            ),
        )
        head = MultivariateHurdleHead(
            latent_dim=latent_dim,
            hidden_dims=_config_value(
                values,
                ("head_hidden_dims", "biology_hidden_dims"),
                (64, 32),
            ),
            num_targets=num_targets,
            sigma_min=float(values.get("sigma_min", 1e-4)),
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        initial_scale = float(
            _config_value(
                random_values,
                ("initial_scale", "prior_scale_init", "scale_init"),
                _config_value(
                    values,
                    ("random_effect_initial_scale", "random_effect_scale_init"),
                    0.05,
                ),
            )
        )
        scale_floor = float(
            _config_value(
                random_values,
                ("scale_floor", "sigma_min"),
                values.get("random_effect_scale_floor", 1e-6),
            )
        )
        scale_prior_std = float(
            _config_value(
                random_values,
                ("scale_prior_std", "half_normal_scale", "prior_scale"),
                _config_value(
                    values,
                    ("random_effect_scale_prior_std", "random_effect_prior_scale"),
                    1.0,
                ),
            )
        )
        random_intercepts = NonCenteredRandomIntercepts(
            num_patients=num_patients,
            num_slides=num_slides,
            num_targets=num_targets,
            initial_scale=initial_scale,
            scale_floor=scale_floor,
            scale_prior_std=scale_prior_std,
        )
        correlation = GlobalCorrelation(
            num_targets=num_targets,
            diagonal_floor=float(values.get("correlation_diagonal_floor", 1e-4)),
        )
        return cls(encoder, head, random_intercepts, correlation)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode a collated sparse MSI batch.

        Args:
            batch (Mapping[str, torch.Tensor]): Sparse peak tensors.

        Returns:
            torch.Tensor: Latent matrix with shape ``[B, latent_dim]``.
        """

        peak_indices = batch.get("peak_indices", batch.get("feature_indices"))
        intensities = batch.get("intensities", batch.get("values"))
        peak_counts = batch.get("peak_counts", batch.get("feature_counts"))
        if peak_indices is None or intensities is None or peak_counts is None:
            raise KeyError(
                "Batch requires peak_indices/intensities/peak_counts or "
                "feature_indices/values/feature_counts."
            )
        return self.encoder(
            peak_indices,
            intensities,
            batch["sample_indices"],
            peak_counts,
        )

    @property
    def R(self) -> torch.Tensor:
        """Expose the learned shared correlation matrix.

        Args:
            None: This property reads the correlation submodule.

        Returns:
            torch.Tensor: Correlation matrix with shape ``[T, T]``.
        """

        return self.correlation.R

    def covariance(
        self,
        scales: torch.Tensor,
        correlation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Construct row-wise covariance matrices ``D R D``.

        Args:
            scales (torch.Tensor): Positive marginal scales with shape ``[B, T]``.
            correlation (torch.Tensor | None): Optional correlation matrix ``[T, T]``.

        Returns:
            torch.Tensor: Covariance matrices with shape ``[B, T, T]``.
        """

        if scales.ndim != 2 or scales.shape[-1] != self.num_targets:
            raise ValueError("scales must have shape [B, num_targets].")
        if torch.any(scales <= 0.0) or not torch.isfinite(scales).all():
            raise ValueError("scales must be finite and strictly positive.")
        matrix = self.R if correlation is None else correlation
        if matrix.shape != (self.num_targets, self.num_targets):
            raise ValueError("correlation has an incompatible shape.")
        return scales.unsqueeze(-1) * matrix.unsqueeze(0) * scales.unsqueeze(-2)

    def prior_nll(self) -> torch.Tensor:
        """Return the scalar hierarchical random-effect prior NLL.

        Args:
            None: This method uses registered random-effect parameters.

        Returns:
            torch.Tensor: Scalar negative log-prior.
        """

        return self.random_intercepts.prior_nll()

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Predict MSI-only, patient-adjusted, and total model parameters.

        Args:
            batch (Mapping[str, torch.Tensor]): Mapping containing sparse peak tensors
                plus one-dimensional ``patients`` and ``slides`` code tensors.

        Returns:
            dict[str, torch.Tensor]: Latent, hierarchical predictions, effects,
                marginal scales, shared correlation, and row-wise covariance.
        """

        latent = self.encode(batch)
        msi = self.head(latent)
        patients = batch.get("patients", batch.get("patient_codes"))
        slides = batch.get("slides", batch.get("slide_codes"))
        if patients is None or slides is None:
            raise KeyError(
                "Batch requires patients/slides or patient_codes/slide_codes."
            )
        effects = self.random_intercepts(patients, slides)
        hurdle_patient = (
            msi["hurdle_logits_msi"] + effects["patient_hurdle_effect"]
        )
        mean_patient = msi["mean_msi"] + effects["patient_mean_effect"]
        hurdle_total = hurdle_patient + effects["slide_hurdle_effect"]
        mean_total = mean_patient + effects["slide_mean_effect"]
        correlation = self.R
        covariance = self.covariance(msi["scales"], correlation)
        return {
            "latent": latent,
            **msi,
            **effects,
            "hurdle_logits_patient": hurdle_patient,
            "mean_patient": mean_patient,
            "hurdle_logits_total": hurdle_total,
            "mean_total": mean_total,
            "correlation": correlation,
            "R": correlation,
            "covariance": covariance,
            # Conventional aliases point to fully adjusted predictions.
            "hurdle_logits": hurdle_total,
            "pi_logits": hurdle_total,
            "mean": mean_total,
            "mu": mean_total,
            "sigma": msi["scales"],
        }


MultivariateHurdleModel = IHCMultivariateModel


__all__ = [
    "GlobalCorrelation",
    "IHCMultivariateModel",
    "IntensityWeightedPeakEncoder",
    "MultivariateHurdleHead",
    "MultivariateHurdleModel",
    "NonCenteredRandomIntercepts",
    "build_activation",
    "build_mlp",
]

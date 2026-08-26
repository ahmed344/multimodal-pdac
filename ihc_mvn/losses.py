"""Factorized hurdle and exactly masked multivariate Gaussian losses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MVNHurdleLossOutput:
    """Reduced loss components and observation counts."""

    total: torch.Tensor
    hurdle: torch.Tensor
    positive: torch.Tensor
    prior: torch.Tensor
    pixel_count: torch.Tensor
    hurdle_count: torch.Tensor
    positive_pixel_count: torch.Tensor
    positive_target_count: torch.Tensor


@dataclass(frozen=True)
class ConditionalGaussianOutput:
    """Conditional Gaussian moments and observed target excess."""

    conditional_mean: torch.Tensor
    conditional_variance: torch.Tensor
    excess: torch.Tensor
    conditioning_count: torch.Tensor


def _prediction(
    predictions: Mapping[str, torch.Tensor],
    names: tuple[str, ...],
) -> torch.Tensor:
    """Select a prediction tensor using conventional key aliases.

    Args:
        predictions (Mapping[str, torch.Tensor]): Model prediction mapping.
        names (tuple[str, ...]): Keys in decreasing priority.

    Returns:
        torch.Tensor: Tensor stored under the first available key.
    """

    for name in names:
        if name in predictions:
            return predictions[name]
    raise KeyError(f"Predictions require one of these keys: {names}.")


def _validate_distribution_shapes(
    hurdle_logits: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    correlation: torch.Tensor,
    targets: torch.Tensor,
    standardized_targets: torch.Tensor,
) -> None:
    """Validate multivariate hurdle prediction and target tensors.

    Args:
        hurdle_logits (torch.Tensor): Presence logits with shape ``[B, T]``.
        means (torch.Tensor): Standardized Gaussian means with shape ``[B, T]``.
        scales (torch.Tensor): Positive marginal scales with shape ``[B, T]``.
        correlation (torch.Tensor): Shared correlation with shape ``[T, T]``.
        targets (torch.Tensor): Targets used to define positive presence.
        standardized_targets (torch.Tensor): Gaussian coordinates for positive targets.

    Returns:
        None: Validation succeeds or an exception is raised.
    """

    if hurdle_logits.ndim != 2:
        raise ValueError("hurdle_logits must have shape [B, T].")
    if not (
        hurdle_logits.shape
        == means.shape
        == scales.shape
        == targets.shape
        == standardized_targets.shape
    ):
        raise ValueError(
            "hurdle_logits, means, scales, targets, and standardized_targets "
            "must have identical [B, T] shapes."
        )
    batch_size, num_targets = hurdle_logits.shape
    if batch_size == 0 or num_targets == 0:
        raise ValueError("Loss inputs must contain at least one pixel and target.")
    if correlation.shape != (num_targets, num_targets):
        raise ValueError("correlation must have shape [T, T].")
    if not torch.isfinite(hurdle_logits).all():
        raise ValueError("hurdle_logits must be finite.")
    if not torch.isfinite(means).all():
        raise ValueError("means must be finite.")
    if not torch.isfinite(scales).all() or torch.any(scales <= 0.0):
        raise ValueError("scales must be finite and strictly positive.")
    if not torch.isfinite(correlation).all():
        raise ValueError("correlation must be finite.")
    if not torch.isfinite(targets).all():
        raise ValueError("targets must be finite.")
    positive_mask = targets > 0.0
    if not torch.isfinite(standardized_targets[positive_mask]).all():
        raise ValueError("Standardized coordinates for positive targets must be finite.")


def masked_mvn_nll(
    values: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    correlation: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute exact per-row MVN NLL for arbitrary observed-coordinate masks.

    Excluded rows and columns are replaced by an identity block and their
    residuals are set to zero. Consequently, rows with no observed coordinates
    have exactly zero NLL, including no normalizing constant.

    Args:
        values (torch.Tensor): Standardized coordinates with shape ``[B, T]``.
        means (torch.Tensor): Gaussian means with shape ``[B, T]``.
        scales (torch.Tensor): Strictly positive marginal scales ``[B, T]``.
        correlation (torch.Tensor): Shared correlation matrix ``[T, T]``.
        mask (torch.Tensor): Boolean observed-coordinate mask ``[B, T]``.

    Returns:
        torch.Tensor: Exact masked Gaussian NLL for each row, shape ``[B]``.
    """

    if values.ndim != 2 or values.shape != means.shape or values.shape != scales.shape:
        raise ValueError("values, means, and scales must share shape [B, T].")
    if mask.shape != values.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean and have shape [B, T].")
    if correlation.shape != (values.shape[-1], values.shape[-1]):
        raise ValueError("correlation must have shape [T, T].")
    if not torch.isfinite(means).all():
        raise ValueError("means must be finite.")
    if not torch.isfinite(scales).all() or torch.any(scales <= 0.0):
        raise ValueError("scales must be finite and strictly positive.")
    if not torch.isfinite(correlation).all():
        raise ValueError("correlation must be finite.")
    if not torch.isfinite(values[mask]).all():
        raise ValueError("Observed Gaussian coordinates must be finite.")

    covariance = (
        scales.unsqueeze(-1)
        * correlation.unsqueeze(0)
        * scales.unsqueeze(-2)
    )
    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    identity = torch.eye(
        values.shape[-1],
        dtype=covariance.dtype,
        device=covariance.device,
    ).unsqueeze(0)
    padded_covariance = torch.where(pair_mask, covariance, identity)
    residual = torch.where(mask, values - means, torch.zeros_like(values))
    cholesky = torch.linalg.cholesky(padded_covariance)
    solved = torch.cholesky_solve(residual.unsqueeze(-1), cholesky).squeeze(-1)
    quadratic = (residual * solved).sum(dim=-1)
    log_determinant = 2.0 * torch.log(
        cholesky.diagonal(dim1=-2, dim2=-1)
    ).sum(dim=-1)
    observed_count = mask.sum(dim=-1).to(values.dtype)
    return 0.5 * (
        observed_count * math.log(2.0 * math.pi)
        + log_determinant
        + quadratic
    )


class MVNHurdleLoss(nn.Module):
    """Presence BCE plus exact masked multivariate Gaussian NLL."""

    def __init__(
        self,
        training_sample_count: int | None = None,
        prior_weight: float = 1.0,
    ) -> None:
        """Initialize the per-pixel objective.

        Args:
            training_sample_count (int | None): Number of training pixels used to
                normalize an unnormalized model prior NLL.
            prior_weight (float): Nonnegative multiplier on the normalized prior.

        Returns:
            None: Loss settings are initialized.
        """

        super().__init__()
        if training_sample_count is not None and training_sample_count <= 0:
            raise ValueError("training_sample_count must be positive when provided.")
        if prior_weight < 0.0:
            raise ValueError("prior_weight must be nonnegative.")
        self.training_sample_count = (
            None if training_sample_count is None else int(training_sample_count)
        )
        self.prior_weight = float(prior_weight)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MVNHurdleLoss":
        """Construct the loss from conventional config sections.

        Args:
            config (Mapping[str, Any]): Full config or its ``loss`` section.

        Returns:
            MVNHurdleLoss: Configured loss module.
        """

        values = config.get("loss", config)
        if not isinstance(values, Mapping):
            raise TypeError("The loss configuration must be a mapping.")
        training_values = config.get("training", {})
        if not isinstance(training_values, Mapping):
            training_values = {}
        sample_count = values.get(
            "training_sample_count",
            values.get(
                "num_training_samples",
                training_values.get(
                    "training_sample_count",
                    training_values.get("num_training_samples"),
                ),
            ),
        )
        return cls(
            training_sample_count=None if sample_count is None else int(sample_count),
            prior_weight=float(values.get("prior_weight", 1.0)),
        )

    def _prior_component(
        self,
        reference: torch.Tensor,
        prior_nll: torch.Tensor | float | None,
        normalized_prior: torch.Tensor | float | None,
    ) -> torch.Tensor:
        """Normalize and weight the model prior contribution.

        Args:
            reference (torch.Tensor): Tensor supplying output dtype and device.
            prior_nll (torch.Tensor | float | None): Unnormalized scalar model prior.
            normalized_prior (torch.Tensor | float | None): Already normalized scalar prior.

        Returns:
            torch.Tensor: Weighted scalar prior contribution.
        """

        if prior_nll is not None and normalized_prior is not None:
            raise ValueError("Pass prior_nll or normalized_prior, not both.")
        if normalized_prior is not None:
            value = torch.as_tensor(normalized_prior, dtype=reference.dtype, device=reference.device)
        elif prior_nll is not None:
            if self.training_sample_count is None:
                raise ValueError(
                    "training_sample_count is required for an unnormalized prior_nll."
                )
            value = torch.as_tensor(prior_nll, dtype=reference.dtype, device=reference.device)
            value = value / self.training_sample_count
        else:
            value = reference.new_zeros(())
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError("The prior contribution must be a finite scalar.")
        return self.prior_weight * value.reshape(())

    def forward(
        self,
        predictions: Mapping[str, torch.Tensor],
        targets: torch.Tensor,
        standardized_targets: torch.Tensor | None = None,
        prior_nll: torch.Tensor | float | None = None,
        normalized_prior: torch.Tensor | float | None = None,
    ) -> MVNHurdleLossOutput:
        """Evaluate factorized presence and masked MVN positive branches.

        Presence labels are exactly ``targets > 0``. If ``standardized_targets``
        is omitted, ``targets`` itself is treated as the Gaussian coordinate
        matrix; callers using separately standardized positive values should pass
        them explicitly.

        Args:
            predictions (Mapping[str, torch.Tensor]): Model output mapping containing
                total hurdle logits, total means, scales, and correlation.
            targets (torch.Tensor): Values defining positive presence, shape ``[B, T]``.
            standardized_targets (torch.Tensor | None): Standardized Gaussian
                coordinates, shape ``[B, T]``.
            prior_nll (torch.Tensor | float | None): Unnormalized scalar from
                ``model.prior_nll()``.
            normalized_prior (torch.Tensor | float | None): Scalar prior already
                divided by the configured training sample count.

        Returns:
            MVNHurdleLossOutput: Mean-per-pixel loss components and counts.
        """

        hurdle_logits = _prediction(
            predictions,
            ("hurdle_logits_total", "hurdle_logits", "pi_logits"),
        )
        means = _prediction(predictions, ("mean_total", "mean", "mu"))
        scales = _prediction(predictions, ("scales", "sigma"))
        correlation = _prediction(predictions, ("correlation", "R"))
        gaussian_targets = (
            targets if standardized_targets is None else standardized_targets
        )
        _validate_distribution_shapes(
            hurdle_logits,
            means,
            scales,
            correlation,
            targets,
            gaussian_targets,
        )

        positive_mask = targets > 0.0
        presence_labels = positive_mask.to(hurdle_logits.dtype)
        hurdle_per_target = F.binary_cross_entropy_with_logits(
            hurdle_logits,
            presence_labels,
            reduction="none",
        )
        hurdle = hurdle_per_target.sum(dim=-1).mean()
        positive_per_pixel = masked_mvn_nll(
            gaussian_targets,
            means,
            scales,
            correlation,
            positive_mask,
        )
        positive = positive_per_pixel.mean()
        prior = self._prior_component(
            hurdle_logits,
            prior_nll=prior_nll,
            normalized_prior=normalized_prior,
        )
        pixel_count = torch.as_tensor(
            targets.shape[0],
            dtype=torch.long,
            device=targets.device,
        )
        return MVNHurdleLossOutput(
            total=hurdle + positive + prior,
            hurdle=hurdle,
            positive=positive,
            prior=prior,
            pixel_count=pixel_count,
            hurdle_count=torch.as_tensor(
                targets.numel(),
                dtype=torch.long,
                device=targets.device,
            ),
            positive_pixel_count=positive_mask.any(dim=-1).sum(),
            positive_target_count=positive_mask.sum(),
        )


def conditional_gaussian_cd8(
    coordinates: torch.Tensor,
    means: torch.Tensor,
    covariance: torch.Tensor | None = None,
    *,
    scales: torch.Tensor | None = None,
    correlation: torch.Tensor | None = None,
    conditioning_mask: torch.Tensor | None = None,
    target_index: int = 3,
) -> ConditionalGaussianOutput:
    """Condition one coordinate (CD8 by default) on the other three.

    The default assumes four standardized coordinates with CD8 in position
    three and conditions on positions zero through two. Missing conditioning
    values can be represented by non-finite coordinates or a false mask.

    Args:
        coordinates (torch.Tensor): Observed standardized coordinates ``[B, 4]``.
        means (torch.Tensor): Joint Gaussian means ``[B, 4]``.
        covariance (torch.Tensor | None): Row-wise covariance ``[B, 4, 4]``.
        scales (torch.Tensor | None): Marginal scales ``[B, 4]`` used with correlation.
        correlation (torch.Tensor | None): Shared correlation ``[4, 4]``.
        conditioning_mask (torch.Tensor | None): Available-coordinate mask with
            shape ``[B, 3]`` or ``[B, 4]``.
        target_index (int): Index of the CD8 coordinate to predict.

    Returns:
        ConditionalGaussianOutput: Conditional mean, variance, observed excess,
            and number of conditioning coordinates per pixel.
    """

    if coordinates.ndim != 2 or means.shape != coordinates.shape:
        raise ValueError("coordinates and means must share shape [B, T].")
    batch_size, num_targets = coordinates.shape
    if num_targets != 4:
        raise ValueError("conditional_gaussian_cd8 expects exactly four coordinates.")
    if not 0 <= target_index < num_targets:
        raise ValueError("target_index is outside the coordinate range.")
    if not torch.isfinite(means).all():
        raise ValueError("means must be finite.")

    if covariance is None:
        if scales is None or correlation is None:
            raise ValueError(
                "Provide covariance, or provide both scales and correlation."
            )
        if scales.shape != coordinates.shape:
            raise ValueError("scales must have shape [B, 4].")
        if correlation.shape != (num_targets, num_targets):
            raise ValueError("correlation must have shape [4, 4].")
        if not torch.isfinite(scales).all() or torch.any(scales <= 0.0):
            raise ValueError("scales must be finite and strictly positive.")
        covariance = (
            scales.unsqueeze(-1)
            * correlation.unsqueeze(0)
            * scales.unsqueeze(-2)
        )
    if covariance.shape != (batch_size, num_targets, num_targets):
        raise ValueError("covariance must have shape [B, 4, 4].")
    if not torch.isfinite(covariance).all():
        raise ValueError("covariance must be finite.")

    conditioning_indices = [
        index for index in range(num_targets) if index != target_index
    ]
    conditioning_values = coordinates[:, conditioning_indices]
    finite_mask = torch.isfinite(conditioning_values)
    if conditioning_mask is None:
        available = finite_mask
    else:
        if conditioning_mask.dtype != torch.bool:
            raise ValueError("conditioning_mask must be boolean.")
        if conditioning_mask.shape == coordinates.shape:
            supplied_mask = conditioning_mask[:, conditioning_indices]
        elif conditioning_mask.shape == conditioning_values.shape:
            supplied_mask = conditioning_mask
        else:
            raise ValueError("conditioning_mask must have shape [B, 3] or [B, 4].")
        available = finite_mask & supplied_mask

    conditioning_mean = means[:, conditioning_indices]
    residual = torch.where(
        available,
        conditioning_values - conditioning_mean,
        torch.zeros_like(conditioning_values),
    )
    conditioning_covariance = covariance[:, conditioning_indices][
        :, :, conditioning_indices
    ]
    pair_mask = available.unsqueeze(-1) & available.unsqueeze(-2)
    identity = torch.eye(
        len(conditioning_indices),
        dtype=covariance.dtype,
        device=covariance.device,
    ).unsqueeze(0)
    padded_covariance = torch.where(pair_mask, conditioning_covariance, identity)
    target_cross_covariance = covariance[:, target_index, conditioning_indices]
    target_cross_covariance = torch.where(
        available,
        target_cross_covariance,
        torch.zeros_like(target_cross_covariance),
    )
    cholesky = torch.linalg.cholesky(padded_covariance)
    solved_residual = torch.cholesky_solve(
        residual.unsqueeze(-1),
        cholesky,
    ).squeeze(-1)
    conditional_mean = means[:, target_index] + (
        target_cross_covariance * solved_residual
    ).sum(dim=-1)
    solved_cross = torch.cholesky_solve(
        target_cross_covariance.unsqueeze(-1),
        cholesky,
    ).squeeze(-1)
    conditional_variance = covariance[:, target_index, target_index] - (
        target_cross_covariance * solved_cross
    ).sum(dim=-1)
    conditional_variance = conditional_variance.clamp_min(
        torch.finfo(conditional_variance.dtype).eps
    )
    observed_target = coordinates[:, target_index]
    excess = torch.where(
        torch.isfinite(observed_target),
        (observed_target - conditional_mean) / torch.sqrt(conditional_variance),
        torch.full_like(observed_target, torch.nan),
    )
    return ConditionalGaussianOutput(
        conditional_mean=conditional_mean,
        conditional_variance=conditional_variance,
        excess=excess,
        conditioning_count=available.sum(dim=-1),
    )


MultivariateHurdleLoss = MVNHurdleLoss
HurdleMVNLoss = MVNHurdleLoss


__all__ = [
    "ConditionalGaussianOutput",
    "HurdleMVNLoss",
    "MVNHurdleLoss",
    "MVNHurdleLossOutput",
    "MultivariateHurdleLoss",
    "conditional_gaussian_cd8",
    "masked_mvn_nll",
]

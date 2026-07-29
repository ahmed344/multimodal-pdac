"""Zero-inflated skew-logit-normal losses for the DANN biology head."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ZILNLossOutput:
    """Reduced total and component losses for metric reporting."""

    total: torch.Tensor
    hurdle: torch.Tensor
    positive: torch.Tensor
    positive_count: torch.Tensor


class ZILNLoss(nn.Module):
    """Zero-inflated skew-logit-normal negative log likelihood.

    The positive branch is a skew-normal on the logit scale rather than the
    PDF's plain normal: ``distributions_tests`` found the logit-transformed
    residuals of all four targets are consistently skewed, and skew-normal was
    the best- or near-best-fitting family. The extra ``alpha`` shape parameter
    collapses exactly to the PDF's Gaussian branch when ``alpha == 0``.
    """

    def __init__(
        self,
        target_weights: Sequence[float],
        logit_epsilon: float = 1e-6,
        reduction: str = "mean",
        include_normal_constant: bool = False,
    ) -> None:
        """Initialize stable four-target ZILN loss settings.

        Args:
            target_weights (Sequence[float]): Relative weight for each target.
            logit_epsilon (float): Clamp used before positive-value logits.
            reduction (str): ``"mean"`` or PDF-exact ``"sum"``.
            include_normal_constant (bool): Include ``0.5*log(2*pi)`` if true.

        Returns:
            None: Loss buffers and settings are initialized.
        """

        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("ZILN reduction must be 'mean' or 'sum'.")
        if not 0.0 < logit_epsilon < 0.5:
            raise ValueError("logit_epsilon must lie strictly between zero and 0.5.")
        weights = torch.as_tensor(target_weights, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() == 0 or torch.any(weights < 0.0):
            raise ValueError("target_weights must be a nonempty nonnegative sequence.")
        self.register_buffer("target_weights", weights)
        self.logit_epsilon = float(logit_epsilon)
        self.reduction = reduction
        self.include_normal_constant = bool(include_normal_constant)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ZILNLoss":
        """Construct the loss from YAML-derived configuration.

        Args:
            config (Mapping[str, Any]): Complete DANN configuration.

        Returns:
            ZILNLoss: Configured loss module.
        """

        values = config["loss"]
        return cls(
            target_weights=values["target_weights"],
            logit_epsilon=float(values["logit_epsilon"]),
            reduction=values["ziln_reduction"],
            include_normal_constant=bool(values["include_normal_constant"]),
        )

    def _reduce(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the configured reduction over all pixels and targets.

        Args:
            values (torch.Tensor): Elementwise weighted losses.

        Returns:
            torch.Tensor: Scalar reduced loss.
        """

        if self.reduction == "sum":
            return values.sum()
        weight_mass = self.target_weights.sum() * values.shape[0]
        return values.sum() / weight_mass.clamp_min(torch.finfo(values.dtype).eps)

    def forward(
        self,
        pi_logits: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        alpha: torch.Tensor,
        targets: torch.Tensor,
    ) -> ZILNLossOutput:
        """Evaluate hurdle and positive skew-logit-normal loss branches.

        Args:
            pi_logits (torch.Tensor): Structural-zero logits.
            mu (torch.Tensor): Positive-branch location on the logit-density scale.
            sigma (torch.Tensor): Positive-branch scale, strictly positive.
            alpha (torch.Tensor): Positive-branch skew-normal shape parameter.
            targets (torch.Tensor): Bounded density targets in ``[0, 1]``.

        Returns:
            ZILNLossOutput: Total loss and additive reduced components.
        """

        if not (
            pi_logits.shape == mu.shape == sigma.shape == alpha.shape == targets.shape
        ):
            raise ValueError(
                "pi_logits, mu, sigma, alpha, and targets must have identical shapes."
            )
        if targets.shape[-1] != self.target_weights.numel():
            raise ValueError(
                f"Expected {self.target_weights.numel()} targets, "
                f"observed {targets.shape[-1]}."
            )
        if torch.any((targets < 0.0) | (targets > 1.0)):
            raise ValueError("ZILN targets must lie inside [0, 1].")
        if torch.any(sigma <= 0.0):
            raise ValueError("ZILN sigma values must be strictly positive.")

        zero_mask = targets == 0.0
        positive_mask = ~zero_mask
        hurdle = F.binary_cross_entropy_with_logits(
            pi_logits, zero_mask.to(pi_logits.dtype), reduction="none"
        )
        safe_targets = targets.clamp(self.logit_epsilon, 1.0 - self.logit_epsilon)
        transformed = torch.logit(safe_targets)
        standardized = (transformed - mu) / sigma
        # log(2) and log_ndtr(alpha * z) both depend only on alpha, and cancel
        # exactly at alpha == 0 (log_ndtr(0) == -log(2)), so this reduces to the
        # plain Gaussian branch whenever alpha is zero.
        positive_nll = (
            0.5 * standardized.square()
            + torch.log(sigma)
            - math.log(2.0)
            - torch.special.log_ndtr(alpha * standardized)
        )
        if self.include_normal_constant:
            positive_nll = positive_nll + 0.5 * math.log(2.0 * math.pi)
        positive_nll = torch.where(
            positive_mask, positive_nll, torch.zeros_like(positive_nll)
        )
        weights = self.target_weights.to(targets).unsqueeze(0)
        reduced_hurdle = self._reduce(hurdle * weights)
        reduced_positive = self._reduce(positive_nll * weights)
        return ZILNLossOutput(
            total=reduced_hurdle + reduced_positive,
            hurdle=reduced_hurdle,
            positive=reduced_positive,
            positive_count=positive_mask.sum(),
        )

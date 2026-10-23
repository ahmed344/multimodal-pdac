"""Observational gradient and latent representation monitoring.

Gradients are weighted objective gradients on the encoder (including aggregation),
not AdamW updates. Only scalar summaries and sufficient statistics leave an epoch.
"""
from __future__ import annotations

import copy
import math
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dann.config import monitoring_settings
from dann.targets import target_contract

SIGNED_TOLERANCE = 1e-6


def metric_suffix(column: str) -> str:
    """Return the stable target suffix used in history columns."""
    return column.removeprefix("Density_").lower()


def sampled_updates(total: int, count: int | None = 5) -> set[int]:
    """Choose unique zero-based logical updates, independently of microbatches."""
    if total <= 0:
        return set()
    return set(np.linspace(0, total - 1, min(total, count or total), dtype=int).tolist())


def freeze_monitoring(config: Mapping[str, Any]) -> dict[str, Any]:
    """Record the reduced scope, accepting version-one monitoring on resume.

    Retained scientific definitions come from the saved run. Probability metadata
    is retired without reading targets or changing any historical observations.
    """
    config = copy.deepcopy(dict(config))
    settings = monitoring_settings(config)
    config["monitoring"] = settings
    saved = config.get("monitoring_definitions")
    if saved is not None:
        if saved.get("version") not in (1, 2) or saved.get("target_transform") != target_contract(config):
            raise ValueError("Incompatible saved monitoring definitions.")
    elif not settings["enabled"]:
        return config
    definitions = dict(
        version=2,
        scope=["training_gradients", "validation_latent_total_variance", "validation_latent_participation_ratio"],
        shared_parameters="encoder (including latent aggregation; excluding heads)",
        gradient_components="weighted hurdle/positive per target, H&E, reversed batch (actual GRL)",
        gradient_reduction="objective common denominator and effective-batch accumulation scaling",
        epoch_statistics="mean, population standard deviation, finite count over sampled updates",
        gradient_definitions=dict(magnitude="L2 norm of weighted component gradient",
                                  share="norm(component) / sum(component norms)",
                                  cosine="dot(a,b) / (norm(a)*norm(b)); conflict is cosine < 0",
                                  retained="norm(sum components) / sum(component norms)",
                                  signed="dot(component,sum components) / norm(sum components)^2",
                                  clipping="whole-model preclip norm; min(1, limit/(norm+1e-6)); fraction multiplier < 1"),
        participation_ratio="trace(sample covariance)^2 / sum(squared covariance entries); missing at zero variance",
        signed_contribution_relative_tolerance=SIGNED_TOLERANCE,
        covariance="sample covariance of all validation core latents, float64 centered merge",
        target_transform=target_contract(config),
    )
    if saved is not None:
        for key in definitions.keys() - {"version", "scope"}:
            if key in saved:
                definitions[key] = saved[key]
    config["monitoring_definitions"] = definitions
    return config


class ScalarSummaries:
    """Small scalar observations, reduced once at the epoch boundary."""

    def __init__(self) -> None:
        self.values: dict[str, list[float | torch.Tensor]] = {}

    def add(self, values: Mapping[str, float | torch.Tensor]) -> None:
        for key, value in values.items():
            self.values.setdefault(key, []).append(value.detach() if isinstance(value, torch.Tensor) else value)

    def finish(self) -> dict[str, float]:
        result = {}
        for key, values in self.values.items():
            if isinstance(values[0], torch.Tensor):
                array = torch.stack(values).double().cpu().numpy()
            else:
                array = np.asarray(values, dtype=np.float64)
            valid = array[np.isfinite(array)]
            result.update({f"{key}_mean": float(valid.mean()) if valid.size else math.nan,
                           f"{key}_std": float(valid.std()) if valid.size else math.nan,
                           f"{key}_count": int(valid.size)})
        return result


def gradient_statistics(gradients: Mapping[str, torch.Tensor], prefix: str) -> dict[str, float]:
    """Calculate all decomposition ratios with stable, bounded-memory dot products.

    Directly sum vectors before measuring the combined norm: subtracting Gram
    entries loses small residuals near cancellation. Genuine zero norms stay zero.
    """
    names = list(gradients)
    vectors = list(gradients.values())
    count = len(vectors)
    gram = torch.zeros((count, count), dtype=torch.float64, device=vectors[0].device)
    combined_square = gram.new_zeros(())
    contributions = gram.new_zeros(count)
    for start in range(0, vectors[0].numel(), 65536):
        chunk = torch.stack([v[start:start + 65536] for v in vectors]).double()
        combined = chunk.sum(0)
        gram += chunk @ chunk.T
        combined_square += combined.square().sum()
        contributions += chunk @ combined
    packed = torch.cat([gram.flatten(), combined_square.reshape(1), contributions]).cpu().numpy()
    gram = packed[:count * count].reshape(count, count)
    norms = np.sqrt(np.maximum(np.diag(gram), 0))
    combined_norm = math.sqrt(max(float(packed[count * count]), 0.))
    summed_norm = float(norms.sum())
    signed_valid = combined_norm > SIGNED_TOLERANCE * summed_norm
    result = {f"{prefix}_retained": combined_norm / summed_norm if summed_norm else math.nan,
              f"{prefix}_combined_norm": combined_norm}
    for i, name in enumerate(names):
        result[f"{prefix}_{name}_norm"] = float(norms[i])
        result[f"{prefix}_{name}_share"] = float(norms[i] / summed_norm) if summed_norm else math.nan
        result[f"{prefix}_{name}_signed"] = (float(packed[count * count + 1 + i] / combined_norm**2)
                                             if signed_valid else math.nan)
    for i, j in combinations(range(count), 2):
        denominator = norms[i] * norms[j]
        cosine = float(np.clip(gram[i, j] / denominator, -1., 1.)) if denominator else math.nan
        key = f"{prefix}_{names[i]}__{names[j]}"
        result[f"{key}_cosine"] = cosine
        result[f"{key}_conflict"] = float(cosine < 0) if math.isfinite(cosine) else math.nan
    return result


class GradientMonitor:
    """Accumulate component gradients without touching parameter .grad buffers."""

    def __init__(self, model: torch.nn.Module, columns: Sequence[str], total: int, count: int | None) -> None:
        self.parameters = tuple(p for p in model.encoder.parameters() if p.requires_grad)
        if not self.parameters:
            raise ValueError("Gradient monitoring requires trainable shared encoder parameters.")
        self.names = [metric_suffix(c) for c in columns]
        self.positions = sampled_updates(total, count)
        self.gradients: dict[str, torch.Tensor] = {}
        self.summary = ScalarSummaries()
        self.sample_count = 0

    def accumulate(self, components: Mapping[str, torch.Tensor]) -> None:
        """Use the existing forward graph; retain it for ordinary backward."""
        for name, value in components.items():
            grads = (torch.autograd.grad(value, self.parameters, retain_graph=True, allow_unused=True)
                     if value.requires_grad else (None,) * len(self.parameters))
            flat = torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                              for p, g in zip(self.parameters, grads)])
            if name in self.gradients:
                self.gradients[name].add_(flat)
            else:
                self.gradients[name] = flat

    def complete(self, strength: float) -> None:
        """Summarize one complete effective batch, then release gradient buffers."""
        order = [name for name in ("cd8", "tumor", "collagen", "stroma") if name in self.names]
        order += [name for name in self.names if name not in order]
        targets = {name: self.gradients[f"{name}_hurdle"] + self.gradients[f"{name}_positive"]
                   for name in order}
        ihc = sum(targets.values())
        overall = {"ihc": ihc}
        if "he" in self.gradients:
            overall["he"] = self.gradients["he"]
        overall["batch"] = self.gradients["batch"]
        values = gradient_statistics(overall, "gradient_overall")
        values.update(gradient_statistics(targets, "gradient_ihc"))
        for name in self.names:
            branches = {branch: self.gradients[f"{name}_{branch}"] for branch in ("hurdle", "positive")}
            values.update(gradient_statistics(branches, f"gradient_{name}"))
        values["gradient_sampled_grl_strength"] = strength
        self.summary.add(values)
        self.gradients.clear()
        self.sample_count += 1

    def clipping(self, norm: torch.Tensor, limit: float | None, strength: float) -> None:
        """Record whole-model clipping on every update, matching PyTorch's epsilon."""
        multiplier = torch.ones_like(norm) if limit is None else (limit / (norm + 1e-6)).clamp(max=1.)
        self.summary.add(dict(gradient_preclip_norm=norm, gradient_clip_multiplier=multiplier,
                              gradient_clipped=(multiplier < 1).to(norm.dtype), gradient_grl_strength=strength))

    def finish(self) -> dict[str, float]:
        return dict(self.summary.finish(), gradient_sampled_updates=self.sample_count)


class StreamingCovariance:
    """Centered float64 parallel covariance accumulation, without storing latents."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: torch.Tensor | None = None
        self.m2: torch.Tensor | None = None

    def update(self, latent: torch.Tensor) -> None:
        x = latent.detach().double()
        if not len(x):
            return
        mean = x.mean(0)
        centered = x - mean
        m2 = centered.T @ centered
        if self.count:
            delta = mean - self.mean
            total = self.count + len(x)
            self.m2 += m2 + torch.outer(delta, delta) * (self.count * len(x) / total)
            self.mean += delta * (len(x) / total)
        else:
            self.mean, self.m2 = mean, m2
        self.count += len(x)

    def finish(self) -> dict[str, float]:
        variance = participation = math.nan
        if self.count >= 2:
            covariance = self.m2 / (self.count - 1)
            variance = float(covariance.trace().cpu())
            square = float(covariance.square().sum().cpu())
            participation = variance**2 / square if square > 0 else math.nan
        return dict(latent_total_variance=variance,
                    latent_participation_ratio=participation)

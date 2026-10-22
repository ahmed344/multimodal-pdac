"""Observational gradient, representation, and CD8 distribution monitoring.

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
from dann.targets import CD8, target_contract

QUANTILES = (0.5, 0.75, 0.9)
SIGNED_TOLERANCE = 1e-6


def metric_suffix(column: str) -> str:
    """Return the stable target suffix used in history columns."""
    return column.removeprefix("Density_").lower()


def sampled_updates(total: int, count: int | None = 5) -> set[int]:
    """Choose unique zero-based logical updates, independently of microbatches."""
    if total <= 0:
        return set()
    return set(np.linspace(0, total - 1, min(total, count or total), dtype=int).tolist())


def freeze_monitoring(config: Mapping[str, Any], metadata: Any, rows: np.ndarray) -> dict[str, Any]:
    """Freeze definitions and training-only cutoffs, or reuse saved run semantics."""
    config = copy.deepcopy(dict(config))
    settings = monitoring_settings(config)
    config["monitoring"] = settings
    saved = config.get("monitoring_definitions")
    if saved is not None:
        if saved.get("version") != 1 or saved.get("target_transform") != target_contract(config):
            raise ValueError("Incompatible saved monitoring definitions.")
        return config
    if not settings["enabled"]:
        return config
    columns = config["data"]["target_columns"]
    values = np.empty(0, dtype=np.float64)
    if CD8 in columns:
        index = columns.index(CD8)
        values = np.asarray(metadata.targets[rows, index], dtype=np.float64)
        valid = getattr(metadata, "target_valid_mask", None)
        if valid is not None:
            values = values[np.asarray(valid[rows, index], dtype=bool)]
        values = values[np.isfinite(values) & (values > 0) & (values <= 1)]
    levels = list(config.get("analysis", {}).get("calibration_coverage_levels", [0.5, 0.9]))
    if (not levels or any(isinstance(v, bool) or not isinstance(v, (int, float))
                          or not 0 < v < 1 for v in levels) or len(set(levels)) != len(levels)):
        raise ValueError("calibration_coverage_levels must contain distinct levels in (0, 1).")
    cutoffs = [float(np.quantile(values, q)) if values.size else None for q in QUANTILES]
    config["monitoring_definitions"] = dict(
        version=1, shared_parameters="encoder (including latent aggregation; excluding heads)",
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
        quantile_method="linear",
        signed_contribution_relative_tolerance=SIGNED_TOLERANCE,
        covariance="sample covariance of all validation core latents, float64 centered merge",
        target_transform=target_contract(config), logit_epsilon=float(config["loss"]["logit_epsilon"]),
        cutoff_source="exploratory quantiles of valid positive transformed training targets after row selection",
        training_positive_count=int(values.size), quantiles=list(QUANTILES), cutoffs=cutoffs,
        # Duplicated cutoffs define the same valid event. Boundary cutoffs use exact 0/1 limits.
        cutoff_degenerate=[v is not None and (v in (0., 1.) or cutoffs.count(v) > 1) for v in cutoffs],
        event="density strictly greater than cutoff; unconditional probability includes hurdle",
        coverage_levels=levels, reliability_bins=10,
        reliability_intervals="[k/10, (k+1)/10), last bin includes 1",
    )
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
        return dict(latent_count=self.count, latent_total_variance=variance,
                    latent_participation_ratio=participation)


def coverage_key(level: float) -> str:
    """Encode arbitrary configured levels without integer-percent collisions."""
    return f"{level * 100:g}".replace(".", "p")


class ValidationMonitor:
    """All-observation CD8 scores, reliability sufficient statistics, and latents."""

    def __init__(self, definitions: Mapping[str, Any], cd8_index: int) -> None:
        self.definitions = definitions
        self.index = cd8_index
        self.covariance = StreamingCovariance()
        self.events = ["presence", "exceedance_q50", "exceedance_q75", "exceedance_q90"]
        self.sums: torch.Tensor | None = None

    def update(self, batch: Mapping[str, torch.Tensor], outputs: Mapping[str, torch.Tensor]) -> None:
        """Consume selected core predictions in eval/inference mode."""
        self.covariance.update(outputs["latent"])
        targets = batch["targets"][:, self.index].double()
        valid = batch.get("target_valid_mask", torch.ones_like(batch["targets"], dtype=torch.bool))[:, self.index]
        valid = valid & torch.isfinite(targets) & (targets >= 0) & (targets <= 1)
        targets = targets[valid]
        pi_logits, mu, sigma = [outputs[k][:, self.index][valid].double() for k in ("pi_logits", "mu", "sigma")]
        presence = torch.sigmoid(-pi_logits)
        # Each event: count, squared error, then count/prediction sum/event sum for ten bins.
        levels = self.definitions["coverage_levels"]
        if self.sums is None:
            self.sums = torch.zeros(4 * 32 + 1 + len(levels), dtype=torch.float64, device=targets.device)
        probabilities, truths = [presence], [targets > 0]
        for cutoff in self.definitions["cutoffs"]:
            if cutoff is None:
                probabilities.append(None)
                truths.append(None)
                continue
            if cutoff <= 0:
                probability = presence
            elif cutoff >= 1:
                probability = torch.zeros_like(presence)
            else:
                z = (math.log(cutoff / (1 - cutoff)) - mu) / sigma
                probability = presence * (0.5 * torch.erfc(z / math.sqrt(2.)))
            probabilities.append(probability)
            truths.append(targets > cutoff)
        for i, (probability, truth) in enumerate(zip(probabilities, truths)):
            if probability is None:
                continue
            start = i * 32
            self.sums[start] += len(targets)
            self.sums[start + 1] += (probability - truth.double()).square().sum()
            bins = (probability * 10).long().clamp(0, 9)
            table = self.sums[start + 2:start + 32].reshape(3, 10)
            for row, weights in enumerate((torch.ones_like(probability), probability, truth.double())):
                table[row].scatter_add_(0, bins, weights)
        positive = targets > 0
        self.sums[128] += positive.sum()
        epsilon = self.definitions["logit_epsilon"]
        z = (torch.logit(targets[positive].clamp(epsilon, 1-epsilon)) - mu[positive]) / sigma[positive]
        for i, level in enumerate(levels):
            limit = math.sqrt(2.) * torch.erfinv(z.new_tensor(level))
            self.sums[129 + i] += (z.abs() <= limit).sum()

    def finish(self) -> dict[str, float]:
        result = self.covariance.finish()
        levels = self.definitions["coverage_levels"]
        sums = (self.sums.cpu().numpy() if self.sums is not None else np.zeros(129 + len(levels)))
        for i, event in enumerate(self.events):
            count, error = sums[i*32:i*32+2]
            key = f"cd8_{event}"
            result.update({f"{key}_count": float(count), f"{key}_brier": float(error / count) if count else math.nan})
            table = sums[i*32+2:i*32+32].reshape(3, 10)
            for b in range(10):
                n, prediction, observed = table[:, b]
                result.update({f"{key}_bin{b}_count": float(n),
                               f"{key}_bin{b}_predicted": float(prediction/n) if n else math.nan,
                               f"{key}_bin{b}_observed": float(observed/n) if n else math.nan})
        for i, cutoff in enumerate(self.definitions["cutoffs"]):
            result[f"cd8_exceedance_q{int(QUANTILES[i]*100)}_cutoff"] = math.nan if cutoff is None else cutoff
        result["cd8_positive_count"] = float(sums[128])
        for i, level in enumerate(levels):
            result[f"cd8_coverage_{coverage_key(level)}"] = float(sums[129+i]/sums[128]) if sums[128] else math.nan
        return result

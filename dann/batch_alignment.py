"""Equal-slide direct alignment of exported aggregation latents.

These biased finite-sample dependence estimates do not guarantee preservation of
biology. Undefined geometry is unavailable, never a successful zero score.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import numpy as np
import torch

NAMES = ('nhsic', 'anova', 'soft_nn')
BANDWIDTHS = (0.05, 0.1, 0.2, 0.5)
NORM_FLOOR = 1e-8
DENOMINATOR_FLOOR = 1e-12


def alignment_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve and validate independent switches, including zero-weight diagnostics."""
    raw = config.get('loss', {}).get('batch_alignment', {})
    if not isinstance(raw, Mapping) or set(raw) - {*NAMES, 'bandwidths', 'temperature', 'train_per_slide', 'validation_per_slide'}:
        raise ValueError('Invalid loss.batch_alignment settings.')
    result = {}
    for name in NAMES:
        entry = raw.get(name, {})
        if not isinstance(entry, Mapping) or set(entry) - {'enabled', 'weight'}:
            raise ValueError(f'Invalid batch_alignment.{name} settings.')
        enabled, weight = entry.get('enabled', False), entry.get('weight', 0.)
        if not isinstance(enabled, bool) or isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f'batch_alignment.{name} requires boolean enabled and finite nonnegative weight.')
        result[name] = dict(enabled=enabled, weight=float(weight))
    bandwidths = raw.get('bandwidths', BANDWIDTHS)
    if not isinstance(bandwidths, (list, tuple)):
        raise ValueError('Alignment bandwidths must be a nonempty sequence.')
    result['bandwidths'] = list(bandwidths)
    result['temperature'] = raw.get('temperature', .1)
    if not result['bandwidths'] or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in [*result['bandwidths'], result['temperature']]):
        raise ValueError('Alignment bandwidths and temperature must be positive and finite.')
    for key, default in [('train_per_slide', 32), ('validation_per_slide', 64)]:
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= default:
            raise ValueError(f'batch_alignment.{key} must be an integer from 2 to {default}.')
        result[key] = value
    return result


def alignment_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze estimator and sampling semantics separately from architecture."""
    return dict(version=1, settings=alignment_settings(config), latent='exported aggregation, L2 normalized',
                distribution='equal slide: w_i=1/(S*n_slide)', conditioning='none',
                nhsic='mean exp(-cosine_distance/bandwidth); categorical label kernel; weighted centering and Frobenius norms; biased V statistic',
                anova='equal-slide between-centroid variance / total variance',
                soft_nn='weighted off-diagonal std; candidate weight w_j; self excluded; squared positive excess over leave-one-out chance',
                norm_variance_floor=NORM_FLOOR, denominator_floor=DENOMINATOR_FLOOR,
                sampling='sorted row identities, NumPy SeedSequence([seed, optimizer_update]); validation update=0')


def sample_rows(row_ids, labels, cap: int, seed: int, update: int = 0) -> np.ndarray:
    """Select deterministic identities, independent of row order and global RNG."""
    rows, labels = np.asarray(row_ids), np.asarray(labels)
    if rows.ndim != 1 or labels.ndim != 1 or len(rows) != len(labels):
        raise ValueError('Alignment sample identities and labels must be aligned vectors.')
    if cap < 2:
        raise ValueError('Alignment samples require at least two observations per slide.')
    if len(np.unique(rows)) != len(rows):
        raise ValueError('Alignment sampling requires unique row identities.')
    rng = np.random.default_rng(np.random.SeedSequence([seed, update]))
    chosen = []
    for label in np.unique(labels):
        candidates = np.sort(rows[labels == label])
        if len(candidates) >= 2:
            chosen.extend(rng.choice(candidates, min(cap, len(candidates)), replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=rows.dtype)


@dataclass
class Measurement:
    score: torch.Tensor | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


def _center(kernel: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    mean = kernel @ weights
    return kernel - mean[:, None] - mean[None, :] + weights @ mean


def evaluate_alignment(latent: torch.Tensor, labels: torch.Tensor, names=NAMES,
                       bandwidths=BANDWIDTHS, temperature: float = .1) -> dict[str, Measurement]:
    """Evaluate differentiable objectives on one complete mixed-slide sample."""
    names = tuple(names)
    if not names:
        return {}
    if set(names) - set(NAMES):
        raise ValueError('Unknown alignment objective.')
    if latent.ndim != 2 or labels.ndim != 1 or len(latent) != len(labels):
        raise ValueError('Alignment expects [rows, features] latents and [rows] labels.')
    _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    keep = counts[inverse] >= 2
    excluded = int((~keep).sum())
    x, labels = latent[keep], labels[keep]
    unique, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    coverage = dict(input_rows=len(latent), sample_rows=len(x), sample_slides=len(unique),
                    excluded_singleton_rows=excluded, row_coverage=len(x)/max(len(latent), 1))
    def invalid(reason):
        return {name: Measurement(None, dict(coverage), reason) for name in names}
    if len(unique) < 2:
        return invalid('insufficient_slides')
    # Double precision keeps centered near-constant kernels from cancellation.
    x = x.double()
    norms = torch.linalg.vector_norm(x, dim=1)
    if not torch.isfinite(x).all():
        return invalid('nonfinite_latents')
    coverage['near_zero_norm_rows'] = int((norms <= NORM_FLOOR).sum())
    if coverage['near_zero_norm_rows']:
        return invalid('near_zero_latent_norm')
    x = x / norms[:, None]
    weights = 1. / (len(unique) * counts[inverse].to(x))
    coverage.update(effective_rows=float(1 / weights.square().sum()), slide_weight_min=1/len(unique),
                    slide_weight_max=1/len(unique))
    mean = weights @ x
    variance = (weights * (x - mean).square().sum(1)).sum()
    coverage['normalized_total_variance'] = variance.detach()
    if variance <= NORM_FLOOR:
        return invalid('degenerate_latent_variance')
    result = {}
    if 'anova' in names:
        centroids = x.new_zeros((len(unique), x.shape[1])).index_add(0, inverse, x) / counts[:, None]
        between = (centroids - mean).square().sum(1).mean()
        result['anova'] = Measurement(between / variance.clamp_min(DENOMINATOR_FLOOR),
                                      dict(coverage, numerator=between, denominator=variance))
    if 'nhsic' in names or 'soft_nn' in names:
        similarity = (x @ x.T).clamp(-1, 1)
        same = labels[:, None] == labels[None, :]
        pair_weight = weights[:, None] * weights[None, :]
    if 'nhsic' in names:
        kernel = sum(torch.exp(-(1 - similarity) / width) for width in bandwidths) / len(bandwidths)
        zc, bc = _center(kernel, weights), _center(same.to(x), weights)
        numerator = (pair_weight * zc * bc).sum()
        znorm = (pair_weight * zc.square()).sum().sqrt()
        bnorm = (pair_weight * bc.square()).sum().sqrt()
        denominator = znorm * bnorm
        diagnostics = dict(coverage, numerator=numerator, denominator=denominator,
                           latent_kernel_norm=znorm, label_kernel_norm=bnorm)
        reason = 'degenerate_kernel_norm' if min(znorm, bnorm) <= NORM_FLOOR or denominator <= DENOMINATOR_FLOOR else None
        result['nhsic'] = Measurement(None if reason else numerator / denominator.clamp_min(DENOMINATOR_FLOOR), diagnostics, reason)
    if 'soft_nn' in names:
        diagonal = torch.eye(len(x), dtype=torch.bool, device=x.device)
        off_weights = pair_weight.masked_fill(diagonal, 0.)
        off_weights = off_weights / off_weights.sum()
        center = (off_weights * similarity).sum()
        sim_variance = (off_weights * (similarity - center).square()).sum()
        diagnostics = dict(coverage, similarity_std=sim_variance.sqrt())
        if sim_variance <= NORM_FLOOR:
            result['soft_nn'] = Measurement(None, diagnostics, 'degenerate_similarity_variance')
        else:
            logits = (similarity - center) / (sim_variance.sqrt() * temperature) + weights.log()[None, :]
            probability = logits.masked_fill(diagonal, -torch.inf).softmax(1)
            same_probability = (probability * same).sum(1)
            chance = (1 / len(unique) - weights) / (1 - weights)
            excess = (same_probability - chance).clamp_min(0)
            result['soft_nn'] = Measurement((weights * excess.square()).sum(), dict(diagnostics,
                same_slide_probability=(weights * same_probability).sum(), chance_probability=(weights * chance).sum()))
    return result


class AlignmentEvaluationError(ValueError):
    """A positive-weight objective lacks valid epoch evaluations; retain evidence."""
    def __init__(self, message: str, evidence: dict):
        super().__init__(message)
        self.evidence = evidence


class AlignmentSummary:
    """Reduce valid evaluations equally over optimizer updates and expose coverage."""
    def __init__(self, settings):
        self.settings = settings
        self.values = {n: [] for n in NAMES if settings[n]['enabled']}
        self.invalid = {n: Counter() for n in self.values}
        self.diagnostics = {n: {} for n in self.values}

    def add(self, measurements):
        for name, measurement in measurements.items():
            if measurement.score is None or not torch.isfinite(measurement.score):
                self.invalid[name][measurement.reason or 'nonfinite_score'] += 1
            else:
                self.values[name].append(float(measurement.score.detach()))
            for key, value in measurement.diagnostics.items():
                self.diagnostics[name].setdefault(key, []).append(float(value.detach() if isinstance(value, torch.Tensor) else value))

    def finish(self, *, strict: bool = True) -> dict:
        """Unavailable diagnostics contribute exactly zero when their weight is zero.

        Build the entire evidence record before rejecting an unavailable weighted
        objective. Missing scores stay NaN in history, never successful zeros.
        """
        result, failed = {}, []
        for name, values in self.values.items():
            weight = self.settings[name]['weight']
            score = float(np.mean(values)) if values else math.nan
            prefix = f'alignment_{name}'
            result.update({prefix: score, prefix+'_weighted_loss': weight * score if weight else 0.,
                           prefix+'_valid_count': len(values), prefix+'_invalid_count': sum(self.invalid[name].values())})
            result.update({prefix+'_invalid_'+reason: count for reason, count in self.invalid[name].items()})
            result.update({prefix+'_'+key: float(np.mean(v)) for key, v in self.diagnostics[name].items()})
            if not values and weight > 0:
                failed.append(name)
        if strict and failed:
            raise AlignmentEvaluationError(
                f'Enabled alignment objective {", ".join(failed)} has no valid evaluations in epoch: '
                f'{ {n: dict(self.invalid[n]) for n in failed} }', result)
        return result


class ValidationSample:
    """Gather a frozen identity sample, then evaluate once in canonical row order."""
    def __init__(self, row_ids):
        self.row_ids = np.asarray(row_ids, dtype=np.int64)
        self.parts = []
        self.slide_counts = Counter()
        self.total_rows = 0

    def add(self, rows, labels, latent):
        ids = rows.detach().cpu().numpy()
        self.total_rows += len(ids)
        self.slide_counts.update(labels.detach().cpu().tolist())
        positions = np.flatnonzero(np.isin(ids, self.row_ids))
        if len(positions):
            self.parts.append((ids[positions], labels[positions].detach(), latent[positions].detach()))

    def coverage(self):
        """Describe sampling relative to the complete observed validation pass."""
        return dict(available_rows=self.total_rows, available_slides=len(self.slide_counts),
                    sampled_fraction=len(self.row_ids)/max(self.total_rows, 1),
                    sampling_singleton_rows=sum(n for n in self.slide_counts.values() if n == 1))

    def tensors(self):
        if not self.parts:
            raise ValueError('Validation alignment sample was not observed.')
        rows = np.concatenate([p[0] for p in self.parts])
        if len(np.unique(rows)) != len(rows) or set(rows) != set(self.row_ids):
            raise ValueError('Validation alignment sample identities are missing or duplicated.')
        order = np.argsort(rows)
        return torch.cat([p[2] for p in self.parts])[order], torch.cat([p[1] for p in self.parts])[order]

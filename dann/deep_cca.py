"""Slide-centered Deep CCA with frozen training-only H&E standardization.

All covariance algebra is float64. Validation merges per-slide centered moments
across the complete pass; small physical batches never discard eligible slides.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn


DEFAULTS = dict(dimension=16, msi_hidden_dim=128, he_hidden_dim=256,
                biology_only_epochs=1, ramp_epochs=2)


def cca_settings(he: Mapping) -> dict:
    """Validate projector dimensions and the epoch-based auxiliary schedule."""
    supplied = he.get('deep_cca', {})
    if not isinstance(supplied, Mapping) or supplied.keys() - DEFAULTS.keys():
        raise ValueError('Invalid model.heads.he.deep_cca settings.')
    settings = {**DEFAULTS, **supplied}
    for key, value in settings.items():
        minimum = 0 if key.endswith('epochs') else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'deep_cca.{key} must be an integer >= {minimum}.')
    return settings


def cca_contract(config: Mapping) -> dict | None:
    """Record objective/preprocessing semantics separately from legacy raw MSE."""
    he = config.get('model', {}).get('heads', {}).get('he', {})
    if not he.get('enabled', False) or he.get('objective', 'raw_mse') != 'deep_cca':
        return None
    return dict(version=1, **cca_settings(he), centering='within_slide',
                covariance='equal_slide_unbiased', minimum_slide_rows=8,
                minimum_residual_df='dimension+1', ridge_relative=1e-3,
                ridge_absolute=1e-8, variance_weight=.1, variance_floor=1.,
                algebra_dtype='float64', normalization='training_feature_mean_population_std',
                scale_floor=1e-8, validation='complete_pass', projector='linear_layernorm_gelu_linear')


def effective_weight(weight: float, settings: Mapping, epoch: int, training: bool) -> float:
    """Use zero for the biology-only period, then a two-step linear epoch ramp."""
    if not training:
        return weight
    completed = epoch + 1 - settings['biology_only_epochs']
    return weight * min(1., max(0., completed / max(settings['ramp_epochs'], 1)))


class CCAProjectors(nn.Module):
    """Two pointwise projectors and portable fitted H&E preprocessing buffers."""
    def __init__(self, latent_dim: int, he_dim: int, settings: Mapping):
        super().__init__()
        self.settings = dict(settings)
        d = settings['dimension']
        def projector(width, hidden):
            return nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden),
                                 nn.GELU(), nn.Linear(hidden, d))
        self.msi = projector(latent_dim, settings['msi_hidden_dim'])
        self.he = projector(he_dim, settings['he_hidden_dim'])
        self.register_buffer('feature_mean', torch.zeros(he_dim, dtype=torch.float64))
        self.register_buffer('feature_scale', torch.ones(he_dim, dtype=torch.float64))
        self.register_buffer('fit_count', torch.zeros((), dtype=torch.int64))
        self.register_buffer('fit_rows_sha256', torch.zeros(32, dtype=torch.uint8))

    @torch.no_grad()
    def fit(self, reader, rows: np.ndarray, chunk_rows: int = 1024) -> dict:
        """Fit once from explicitly selected training rows, using stable moments."""
        if self.fit_count.item():
            raise ValueError('CCA preprocessing is already fitted; never refit on evaluation.')
        rows = np.sort(np.asarray(rows, dtype=np.int64))
        if len(rows) < 2 or len(np.unique(rows)) != len(rows):
            raise ValueError('CCA fitting requires at least two unique training rows.')
        mean = np.zeros(reader.width, dtype=np.float64)
        m2, count = mean.copy(), 0
        for start in range(0, len(rows), chunk_rows):
            x = reader.read(rows[start:start + chunk_rows]).astype(np.float64)
            n = len(x)
            delta = x.mean(0) - mean
            m2 += ((x - x.mean(0)) ** 2).sum(0) + delta ** 2 * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
        scale = np.maximum(np.sqrt(m2 / count), 1e-8)
        digest = hashlib.sha256(rows.astype('<i8').tobytes()).digest()
        self.feature_mean.copy_(torch.from_numpy(mean))
        self.feature_scale.copy_(torch.from_numpy(scale))
        self.fit_count.fill_(count)
        self.fit_rows_sha256.copy_(torch.tensor(list(digest), dtype=torch.uint8))
        return dict(fit_split='train', count=count, selected_rows_sha256=digest.hex(),
                    target_key=reader.key, source=str(reader.path), ddof=0,
                    constant_features=int(np.sum(m2 == 0)))

    def forward(self, latent: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.fit_count.item():
            raise ValueError('CCA H&E preprocessing must be fitted on training rows first.')
        standardized = ((targets.double() - self.feature_mean) / self.feature_scale).to(latent.dtype)
        return self.msi(latent), self.he(standardized)


@dataclass
class CCAResult:
    loss: torch.Tensor | None
    diagnostics: dict
    reason: str | None = None


def from_covariance(cov: torch.Tensor, dimension: int, diagnostics: dict) -> CCAResult:
    """Evaluate regularized canonical correlations and the variance penalty."""
    d = dimension
    cov = cov.double()
    u, v, uv = cov[:d, :d], cov[d:, d:], cov[:d, d:]
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    lu, lv = [torch.linalg.cholesky(x + (1e-3 * torch.trace(x) / d + 1e-8) * eye)
              for x in (u, v)]
    left = torch.linalg.solve_triangular(lu, uv, upper=False)
    t = torch.linalg.solve_triangular(lv, left.T, upper=False).T
    correlations = torch.linalg.svdvals(t)
    variance = torch.relu(1 - torch.sqrt(torch.diagonal(cov).clamp_min(0) + 1e-8)).mean()
    loss = 1 - correlations.mean() + .1 * variance
    with torch.no_grad():
        diagnostics = dict(diagnostics, correlation=float(correlations.mean()),
                           variance_penalty=float(variance), loss=float(loss),
                           msi_rank=int(torch.linalg.matrix_rank(u, rtol=1e-8, atol=1e-12)),
                           he_rank=int(torch.linalg.matrix_rank(v, rtol=1e-8, atol=1e-12)))
        diagnostics.update({f'canonical_{i+1}': float(value) for i, value in enumerate(correlations)})
    return CCAResult(loss, diagnostics)


def evaluate_cca(u: torch.Tensor, v: torch.Tensor, slides: torch.Tensor) -> CCAResult:
    """Use every eligible paired row in one complete optimizer batch."""
    if u.ndim != 2 or u.shape != v.shape or slides.shape != (len(u),):
        raise ValueError('CCA needs matching projection matrices and aligned slide labels.')
    moments = SlideCovariance(u.shape[1])
    moments.update(u, v, slides, detach=False)
    return moments.finish()


class SlideCovariance:
    """Merge float64 centered sufficient statistics with equal final slide weights."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.moments = {}
        self.rows = 0

    def update(self, u, v, slides, *, detach: bool = True) -> None:
        z = torch.cat((u, v), dim=1).double()
        if detach:
            z = z.detach()
        if not torch.isfinite(z).all():
            raise FloatingPointError('Nonfinite CCA projections.')
        self.rows += len(z)
        for slide in torch.unique(slides):
            x = z[slides == slide]
            n, mean = len(x), x.mean(0)
            centered = x - mean
            scatter = centered.T @ centered
            key = int(slide)
            if key in self.moments:
                old_n, old_mean, old_scatter = self.moments[key]
                delta = mean - old_mean
                scatter = old_scatter + scatter + torch.outer(delta, delta) * (old_n * n / (old_n + n))
                mean = old_mean + delta * (n / (old_n + n))
                n += old_n
            self.moments[key] = n, mean, scatter

    def finish(self) -> CCAResult:
        retained = [value for value in self.moments.values() if value[0] >= 8]
        rows = sum(n for n, _, _ in retained)
        df = rows - len(retained)
        diagnostics = dict(rows=self.rows, retained_rows=rows, slides=len(self.moments),
                           retained_slides=len(retained), residual_df=df,
                           coverage=rows / max(self.rows, 1))
        if not retained or df < self.dimension + 1:
            return CCAResult(None, diagnostics, 'insufficient_samples')
        cov = torch.stack([scatter / (n - 1) for n, _, scatter in retained]).mean(0)
        return from_covariance(cov, self.dimension, diagnostics)


class CCASummary:
    """Average valid optimizer evaluations, exposing unavailable batches explicitly."""
    def __init__(self, weight: float):
        self.weight = weight
        self.values, self.invalid, self.coverage = [], Counter(), []

    def add(self, result: CCAResult) -> None:
        self.coverage.append({k: v for k, v in result.diagnostics.items() if k in
                             ('rows', 'retained_rows', 'slides', 'retained_slides', 'residual_df', 'coverage')})
        if result.loss is None:
            self.invalid[result.reason] += 1
        else:
            self.values.append(result.diagnostics)

    def finish(self, *, strict: bool = True) -> dict:
        from dann.batch_alignment import AlignmentEvaluationError
        result = dict(cca_valid_count=len(self.values), cca_invalid_count=sum(self.invalid.values()),
                      cca_effective_weight=self.weight)
        result.update({f'cca_invalid_{k}': v for k, v in self.invalid.items()})
        keys = set().union(*(v.keys() for v in self.values)) if self.values else {'loss', 'correlation', 'variance_penalty'}
        result.update({f'cca_{key}': float(np.mean([v[key] for v in self.values])) if self.values else math.nan for key in keys})
        if self.coverage:
            result.update({f'cca_{k}': float(np.mean([v[k] for v in self.coverage])) for k in self.coverage[0]})
        result['cca_weighted_loss'] = self.weight * result['cca_loss'] if self.weight else 0.
        if strict and not self.values and self.weight > 0:
            raise AlignmentEvaluationError('Positive-weight CCA epoch has no valid evaluations.', result)
        return result

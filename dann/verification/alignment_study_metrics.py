"""Matched slide-mixing diagnostics and preregistered alignment-study decisions.

UMAP distances are Euclidean; input latent neighbors use cosine. Excess means
observed same-slide fraction minus its leave-one-out chance, in absolute units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dann.batch_alignment import sample_rows

SEED = 20260719
SEEDS = tuple(range(SEED, SEED + 5))
LIMITS = dict(probe_mse_relative=.10, probe_brier_absolute=.02,
              cd8_loss_relative=.10, brier_absolute=.02, pit_ks_absolute=.04,
              coverage_error_absolute=.04, latent_tie=.01)


def balanced_rows(rows: np.ndarray, labels: np.ndarray, cap: int, seed: int,
                  *, required: int | None = None) -> np.ndarray:
    """Sample equal numbers per slide, invariant to incoming row order."""
    rows, labels = np.asarray(rows), np.asarray(labels)
    if rows.ndim != 1 or labels.shape != rows.shape or len(np.unique(rows)) != len(rows):
        raise ValueError('Matched sampling needs aligned, unique row identities.')
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError('Matched sampling needs at least two slides with two rows each.')
    if required is not None and counts.min() < required:
        raise ValueError(f'Every slide must have at least {required} validation rows.')
    return sample_rows(rows, labels, min(cap, int(counts.min())), seed)


def geometry(values: np.ndarray, labels: np.ndarray, *, metric: str = 'cosine',
             neighbors: int = 30) -> tuple[dict, pd.DataFrame]:
    """Exact, self-excluded kNN with deterministic row-order tie breaking.

    Input rows must be in the same frozen order for every arm. Invalid geometry
    raises instead of producing a deceptively good zero mixing score.
    """
    x = torch.as_tensor(np.asarray(values), dtype=torch.float64)
    labels = np.asarray(labels)
    if (x.ndim != 2 or labels.shape != (len(x),) or len(x) <= neighbors
            or neighbors < 1 or not torch.isfinite(x).all()):
        raise ValueError('Invalid geometry: finite aligned rows and n > neighbors required.')
    if torch.mean((x - x.mean(0)).square()).item() <= 1e-16:
        raise ValueError('Invalid geometry: collapsed coordinates.')
    if len(np.unique(labels)) < 2:
        raise ValueError('Invalid geometry: fewer than two slides.')
    if metric == 'cosine':
        norms = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        if (norms <= 1e-8).any():
            raise ValueError('Invalid geometry: zero latent norm.')
        x = x / norms
        if torch.mean((x - x.mean(0)).square()).item() <= 1e-16:
            raise ValueError('Invalid geometry: collapsed cosine directions.')
        distances = (1 - x @ x.T).clamp(0, 2)
    elif metric == 'euclidean':
        distances = torch.cdist(x, x).square()
    else:
        raise ValueError('Unsupported neighbor metric.')
    distances.fill_diagonal_(torch.inf)
    nearest = distances.argsort(dim=1, stable=True)[:, :neighbors].numpy()
    fraction = (labels[nearest] == labels[:, None]).mean(1)
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    chance = (counts[inverse] - 1) / (len(x) - 1)
    per_row = pd.DataFrame(dict(slide=labels, same_slide=fraction, chance=chance,
                                excess=fraction-chance))
    return dict(rows=len(x), slides=len(unique), same_slide=float(fraction.mean()),
                chance=float(chance.mean()), excess=float((fraction-chance).mean())), per_row


def variance_spectrum(values: np.ndarray) -> dict:
    """Raw and row-normalized spectra, including explicit collapse eligibility."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        return dict(noncollapsed=False, reason='nonfinite or insufficient latent rows')
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms <= 1e-8):
        return dict(noncollapsed=False, reason='zero latent norm')
    result = {}
    for name, matrix in [('raw', x), ('normalized', x/norms[:, None])]:
        centered = matrix - matrix.mean(0)
        eig = np.maximum(np.linalg.eigvalsh(centered.T @ centered/(len(x)-1))[::-1], 0)
        total = float(eig.sum())
        result[name] = dict(total_variance=total, eigenvalues=eig.tolist(),
                            numerical_rank=int(np.sum(eig > max(1e-12, float(eig[0])*1e-8))),
                            participation_ratio=float(total**2/np.square(eig).sum()) if total > 0 else 0.)
    # Operational numerical-collapse floor, fixed before examining candidates.
    result['noncollapsed'] = all(result[n]['total_variance'] > 1e-8 and
                                result[n]['numerical_rank'] >= 2 for n in ('raw', 'normalized'))
    return result


def eligibility(candidate: dict, reference: dict) -> dict:
    """Inclusive bounds; negative log likelihood uses 10% of reference magnitude."""
    checks = []
    def bound(name, value, ref, allowance):
        finite = value is not None and ref is not None and np.isfinite(value) and np.isfinite(ref)
        checks.append(dict(metric=name, value=value, reference=ref, allowance=allowance,
                           passed=bool(finite and value <= ref + allowance + 1e-12)))
    checks.append(dict(metric='noncollapsed', passed=candidate['spectrum']['noncollapsed']))
    if not candidate['spectrum']['noncollapsed']:
        return dict(eligible=False, checks=checks)
    for name in ('stroma', 'collagen', 'tumor'):
        ref = reference['probes'][name]['mse']
        bound(f'probe_{name}_mse', candidate['probes'][name]['mse'], ref, LIMITS['probe_mse_relative']*abs(ref))
    for name in ('presence', 'exceedance_0', 'exceedance_1', 'exceedance_2'):
        bound(f'probe_{name}_brier', candidate['probes'][name]['brier'], reference['probes'][name]['brier'], .02)
    c, r = candidate['distributions']['cd8'], reference['distributions']['cd8']
    bound('cd8_loss', c['loss'], r['loss'], .10*abs(r['loss']))
    bound('presence_brier', c['presence']['brier'], r['presence']['brier'], .02)
    if set(c['exceedance']) != set(r['exceedance']) or len(r['exceedance']) != 3:
        raise ValueError('Exceedance thresholds must match the three frozen training quartiles.')
    for threshold in r['exceedance']:
        bound(f'exceedance_{threshold}_brier', c['exceedance'][threshold]['brier'], r['exceedance'][threshold]['brier'], .02)
    bound('positive_pit_ks', c['positive_pit_ks'], r['positive_pit_ks'], .04)
    for level in (.5, .9):
        bound(f'coverage_error_{level}', abs(c['intervals'][str(level)]['coverage']-level),
              abs(r['intervals'][str(level)]['coverage']-level), .04)
    for space in ('latent', 'umap'):
        bound(f'{space}_finite', candidate['mixing'][space]['excess'], 1., 0.)
    return dict(eligible=all(c['passed'] for c in checks), checks=checks)


def improves(candidate: dict, base: dict) -> bool:
    """An addition needs strictly lower excess in both spaces than its base."""
    return candidate['spectrum']['noncollapsed'] and all(candidate['mixing'][space]['excess'] < base['mixing'][space]['excess']
               for space in ('latent', 'umap'))


def choose(candidates: list[dict], reference: dict, *, base: dict | None = None) -> dict | None:
    """Eligible improvements, then latent excess; near ties use UMAP then CD8."""
    eligible = [c for c in candidates if eligibility(c, reference)['eligible']
                and improves(c, reference) and (base is None or improves(c, base))]
    if not eligible:
        return None
    minimum = min(c['mixing']['latent']['excess'] for c in eligible)
    tied = [c for c in eligible if c['mixing']['latent']['excess'] <= minimum + LIMITS['latent_tie'] + 1e-12]
    return min(tied, key=lambda c: (c['mixing']['umap']['excess'], c['distributions']['cd8']['loss'], c['id']))

"""Training-fitted geometry and observed-label diagnostics for latent experiments.

No model parameters, slide means or tumor directions are removed here. Discovery
uses the IHC-supervised latent only; the guided view explicitly adds predictions.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from dann.analysis_latents import covariance_pca, select_pc_count
from dann.ziln import exceedance_probability


def fit_preprocessing(training: np.ndarray) -> dict[str, np.ndarray]:
    """Fit one centered covariance decomposition and channel scales on training."""
    pca = covariance_pca(training)
    scale = np.sqrt(np.maximum(np.diag(pca['scatter']) / (len(training) - 1), 0))
    pca['feature_scale'] = np.where(scale > 0, scale, 1.)
    pca['training_count'] = np.array(len(training))
    return pca


def transform_geometry(values: np.ndarray, fitted: Mapping[str, np.ndarray], name: str,
                       settings: Mapping[str, Any]) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Apply frozen centering, channel standardization or truncated partial whitening.

    Whitening exponent alpha scales PC j by max(lambda_j, floor)^(-alpha/2).
    Alpha zero is unwhitened PCA; alpha one is full (regularized) whitening.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(fitted['mean']) or not np.isfinite(x).all():
        raise ValueError('Geometry requires finite latents of the fitted width.')
    centered = (x - fitted['origin']) - fitted['mean']
    metric = 'cosine' if name.endswith('cosine') else 'euclidean'
    info: dict[str, Any] = dict(geometry=name, metric=metric, fit_rows=int(fitted['training_count']))
    if name in ('raw_cosine', 'raw_euclidean'):
        result = x.copy()
    elif name == 'centered_cosine':
        result = centered
    elif name == 'standardized_euclidean':
        result = centered / fitted['feature_scale']
    elif name in ('pca_16', 'pca_32', 'pca_99', 'pca_32_partial'):
        eigenvalues = fitted['eigenvalues']
        total = float(eigenvalues.sum())
        if total <= 0:
            raise ValueError('PCA geometry is undefined for constant training latents.')
        requested = select_pc_count(eigenvalues)[0] if name == 'pca_99' else int(name.split('_')[1])
        count = min(requested, x.shape[1], int(fitted['training_count']) - 1)
        alpha = float(settings['whitening_power']) if name.endswith('partial') else 0.
        floor = float(settings['eigenvalue_floor'] * eigenvalues[0])
        scales = np.maximum(eigenvalues[:count], floor) ** (-alpha / 2)
        result = (centered @ fitted['components'][:, :count]) * scales
        info.update(requested_components=requested, components=count, whitening_power=alpha,
                    eigenvalue_floor=floor, retained_variance=float(eigenvalues[:count].sum()/total))
    else:
        raise ValueError(f'Unknown geometry: {name}')
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite transformed geometry.')
    if metric == 'cosine':
        tolerance = (16*np.finfo(float).eps * (np.linalg.norm(x, axis=1)
            + np.linalg.norm(fitted['origin']) + np.linalg.norm(fitted['mean']))) if name == 'centered_cosine' else 0.
        if np.any(np.linalg.norm(result, axis=1) <= tolerance):
            raise ValueError('Cosine geometry has zero-norm rows; do not silently change eligible rows.')
        directions = result/np.linalg.norm(result, axis=1, keepdims=True)
        if len(result) > 1 and np.max(np.ptp(directions, axis=0)) <= 32*np.finfo(float).eps:
            raise ValueError('Collapsed cosine directions cannot define informative neighborhoods.')
    elif len(result) > 1 and not np.any(result != result[0]):
        raise ValueError('Collapsed geometry cannot define informative neighborhoods.')
    return result, metric, info


def training_thresholds(labels: np.ndarray, quantiles: list[float], epsilon: float) -> dict[str, Any]:
    """Freeze exploratory thresholds from valid positive training labels only."""
    if not 0 < epsilon < .5:
        raise ValueError('Threshold epsilon must lie in (0, 0.5).')
    y = np.asarray(labels, dtype=float)
    if np.any(np.isfinite(y) & ((y < 0) | (y > 1))) or np.isinf(y).any():
        raise ValueError('CD8 labels must be in [0, 1] or missing.')
    positive = y[np.isfinite(y) & (y > 0)]
    raw = np.quantile(positive, quantiles) if len(positive) else np.array([])
    thresholds = np.unique(np.clip(raw, epsilon, 1-epsilon))
    return dict(quantiles=quantiles, raw_quantiles=raw.tolist(), thresholds=thresholds.tolist(),
                positive_training_labels=len(positive), epsilon=epsilon,
                boundary_policy='clip to [epsilon, 1-epsilon], then deduplicate',
                interpretation='exploratory summaries, not biological cutoffs',
                presence_only=not len(positive))


def probability_block(pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                      thresholds: list[float], epsilon: float) -> np.ndarray:
    """Bounded presence and unconditional exceedance probabilities, in order."""
    pi, mu, sigma = (np.asarray(x, dtype=float) for x in (pi, mu, sigma))
    if (pi.ndim != 1 or pi.shape != mu.shape or pi.shape != sigma.shape
            or not all(np.isfinite(x).all() for x in (pi, mu, sigma))
            or np.any((pi < 0) | (pi > 1)) or np.any(sigma <= 0)):
        raise ValueError('Invalid CD8 hurdle distribution parameters.')
    return np.column_stack([1-pi] + [exceedance_probability(pi, mu, sigma, t, epsilon) for t in thresholds])


def distance_scale(training: np.ndarray, pairs: int, seed: int) -> float:
    """RMS Euclidean distance of seeded, distinct training-row pairs (bounded RAM)."""
    x = np.asarray(training, dtype=float)
    if len(x) < 2:
        raise ValueError('Distance scaling requires at least two training rows.')
    rng = np.random.default_rng(seed)
    first = rng.integers(len(x), size=pairs)
    second = (first + rng.integers(1, len(x), size=pairs)) % len(x)
    squared = sum(float(np.square(x[a]-x[b]).sum()) for a, b in
                  zip(np.array_split(first, max(1, pairs//1024)), np.array_split(second, max(1, pairs//1024))))
    return float(np.sqrt(squared/pairs))


def guided_geometry(latent: np.ndarray, probabilities: np.ndarray, weight: float,
                    latent_scale: float, probability_scale: float) -> np.ndarray:
    """Realize a mixture of normalized squared distances as Euclidean features.

    A common latent-scale multiplier preserves the exact weight-zero baseline:
    d^2 / s_L^2 = (1-w) d_L^2 / s_L^2 + w d_Q^2 / s_Q^2.
    """
    if not 0 <= weight <= 1 or len(latent) != len(probabilities):
        raise ValueError('Invalid guided mixture weight or row alignment.')
    if not np.isfinite(latent_scale) or latent_scale <= 0:
        raise ValueError('Collapsed training latent distance scale.')
    if weight == 0:
        return np.asarray(latent).copy()
    if not np.isfinite(probability_scale) or probability_scale <= 0:
        raise ValueError('Constant training probability block cannot be distance-normalized.')
    return np.column_stack((np.sqrt(1-weight)*latent,
                            np.sqrt(weight)*latent_scale/probability_scale*probabilities))


def sample_panels(frame: pd.DataFrame, settings: Mapping[str, Any], seed: int) -> dict[str, np.ndarray]:
    """Identical representative, slide-balanced and grid-thinned rows for every arm.

    Thinning selects one pixel per slide/grid cell, not a guaranteed exclusion
    radius. Resampling is without replacement over slides or whole grid blocks.
    """
    rng = np.random.default_rng(seed)
    cap = settings['diagnostic_max_samples']
    def capped(rows):
        return np.sort(rng.choice(rows, min(len(rows), cap), replace=False))
    all_rows = np.arange(len(frame))
    groups = list(frame.groupby('slide', sort=True).indices.values())
    if cap < len(groups):
        raise ValueError('diagnostic_max_samples must allow at least one row per slide.')
    per_slide = min(min(map(len, groups)), max(1, cap // len(groups)))
    balanced = np.sort(np.concatenate([rng.choice(g, per_slide, replace=False) for g in groups]))
    cells = frame[['slide', 'x', 'y']].copy()
    cells[['x', 'y']] = np.floor(cells[['x', 'y']] / settings['spatial_stride']).astype(np.int64)
    blocks = list(cells.groupby(['slide', 'x', 'y'], sort=True).indices.values())
    thinned = np.array([rng.choice(g) for g in blocks])
    result = dict(representative=capped(all_rows), balanced=balanced, spatial_thinned=capped(thinned))
    fraction = settings['resample_fraction']
    for iteration in range(settings['resamples']):
        for name, units in (('slide', groups), ('spatial_block', blocks)):
            chosen = rng.choice(len(units), max(1, int(fraction*len(units))), replace=False)
            result[f'{name}_resample_{iteration}'] = capped(np.concatenate([units[i] for i in chosen]))
    return result


def neighbors(values: np.ndarray, k: int, metric: str,
              query: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Exact bounded-query kNN, removing self by index even for tied duplicates."""
    x = np.asarray(values, dtype=float)
    own = query is None
    k = min(k, len(x)-int(own))
    if k < 1:
        raise ValueError('Not enough observations for neighbors.')
    model = NearestNeighbors(metric=metric, algorithm='brute', n_jobs=1).fit(x)
    q = x if own else np.asarray(query, dtype=float)
    indices, distances = [], []
    for start in range(0, len(q), 256):
        d, idx = model.kneighbors(q[start:start+256], n_neighbors=k+int(own))
        if own:
            keep = idx != np.arange(start, start+len(idx))[:, None]
            # A tied self may be absent. Keep precisely the first k other rows.
            keep &= np.cumsum(keep, axis=1) <= k
            idx, d = idx[keep].reshape(-1, k), d[keep].reshape(-1, k)
        indices.append(idx)
        distances.append(d)
    return np.concatenate(indices), np.concatenate(distances)


def neighbor_overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Mean shared neighbor fraction for aligned rows with equal neighbor counts."""
    if first.shape != second.shape:
        raise ValueError('Neighborhoods must have equal shapes.')
    return float(np.mean([len(set(a) & set(b))/len(a) for a, b in zip(first, second)]))


def graph_diagnostics(values: np.ndarray, frame: pd.DataFrame, k: int, metric: str,
                      target_columns: list[str], baseline: np.ndarray | None = None
                      ) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Measure mixing, component support, edge lengths and observed-label locality."""
    ids, distances = neighbors(values, k, metric)
    n, k = ids.shape
    slides = frame.slide.to_numpy()
    same = slides[ids] == slides[:, None]
    _, codes, counts = np.unique(slides, return_inverse=True, return_counts=True)
    chance = (counts[codes]-1)/(n-1)
    graph = csr_matrix((np.ones(n*k), (np.repeat(np.arange(n), k), ids.ravel())), shape=(n, n))
    components, membership = connected_components(graph, directed=False)
    sizes = np.bincount(membership)
    cross = distances[~same]
    within = distances[same]
    records = pd.DataFrame(dict(row_id=frame.row_id.to_numpy(), slide=slides,
        same_slide=same.mean(axis=1), chance=chance, excess=same.mean(axis=1)-chance,
        component=membership))
    records['enrichment'] = np.divide(records.same_slide, chance,
        out=np.full(n, np.nan), where=chance > 0)
    metrics = dict(rows=n, neighbors=k, slides=len(counts), same_slide=float(same.mean()),
        chance=float(chance.mean()), same_slide_excess=float((same.mean(axis=1)-chance).mean()),
        same_slide_enrichment=float(same.mean()/chance.mean()) if chance.mean() else None,
        connected_components=int(components), largest_component_fraction=float(sizes.max()/n),
        cross_slide_edge_fraction=float((~same).mean()),
        cross_slide_distance_median=float(np.median(cross)) if len(cross) else None,
        cross_slide_distance_p95=float(np.quantile(cross, .95)) if len(cross) else None,
        within_slide_distance_median=float(np.median(within)) if len(within) else None)
    local = np.median(distances, axis=1)
    ratios = np.divide(distances, local[:, None], out=np.full_like(distances, np.nan), where=local[:, None] > 0)[~same]
    finite = ratios[np.isfinite(ratios)]
    metrics['cross_edge_to_local_radius_p95'] = float(np.quantile(finite, .95)) if len(finite) else None
    metrics['cross_edges_with_zero_local_radius'] = int(((~same) & (local[:, None] == 0) & (distances > 0)).sum())
    xy = frame[['x', 'y']].to_numpy(float)
    spatial = np.linalg.norm(xy[ids]-xy[:, None, :], axis=2)[same]
    metrics['within_slide_spatial_distance_median'] = float(np.median(spatial)) if len(spatial) else None
    for column in target_columns:
        y = frame[column].to_numpy(float)
        differences = np.abs(y[ids]-y[:, None])
        valid = np.isfinite(differences)
        metrics[f'{column}_neighbor_mae'] = float(differences[valid].mean()) if valid.any() else None
    if baseline is not None:
        metrics['raw_euclidean_neighbor_overlap'] = neighbor_overlap(ids, baseline)
    return metrics, records, ids


def heldout_probes(training: np.ndarray, validation: np.ndarray, train_frame: pd.DataFrame,
                   validation_frame: pd.DataFrame, target_columns: list[str], thresholds: list[float],
                   k: int, metric: str, tumor_edges: np.ndarray) -> pd.DataFrame:
    """Training-neighbor observed-label prediction, evaluated on held-out rows.

    These probes evaluate higher-dimensional geometry and never use validation
    observations as predictor inputs. They are study validation, not patient tests.
    """
    ids, _ = neighbors(training, k, metric, validation)
    tasks = [(name, train_frame[name].to_numpy(float), validation_frame[name].to_numpy(float), 'mse')
             for name in target_columns]
    for index, t in enumerate([0., *thresholds]):
        a, b = train_frame.Density_CD8.to_numpy(float), validation_frame.Density_CD8.to_numpy(float)
        tasks.append(('CD8_presence' if index == 0 else f'CD8_exceedance_{index}',
                      np.where(np.isfinite(a), a > t, np.nan), np.where(np.isfinite(b), b > t, np.nan), 'brier'))
    strata = np.searchsorted(tumor_edges, validation_frame.Density_Tumor.to_numpy(float), side='right')
    records = []
    for target, labels, truth, score in tasks:
        neighbor_labels = labels[ids]
        counts = np.isfinite(neighbor_labels).sum(axis=1)
        prediction = np.divide(np.nansum(neighbor_labels, axis=1), counts,
                               out=np.full(len(ids), np.nan), where=counts > 0)
        valid = np.isfinite(prediction) & np.isfinite(truth)
        for stratum in [-1, *np.unique(strata).tolist()]:
            mask = valid & ((strata == stratum) if stratum >= 0 else True)
            records.append(dict(target=target, metric=score, tumor_stratum='all' if stratum == -1 else str(stratum),
                rows=int(mask.sum()), score=float(np.mean((prediction[mask]-truth[mask])**2)) if mask.any() else np.nan,
                training_reference_rows=len(training), neighbors=ids.shape[1]))
    return pd.DataFrame(records)


def association_diagnostics(latent: np.ndarray, frame: pd.DataFrame,
                            fitted: Mapping[str, np.ndarray], columns: list[str]) -> pd.DataFrame:
    """Pooled and equal-slide Spearman diagnostics; correlations never become weights."""
    scores = ((latent-fitted['origin'])-fitted['mean']) @ fitted['components'][:, :8]
    values = {'latent_norm': np.linalg.norm(latent, axis=1),
              **{f'PC{i+1}': scores[:, i] for i in range(scores.shape[1])}}
    records = []
    for feature, x in values.items():
        for name in columns:
            y = frame[name].to_numpy(float)
            correlations = []
            for slide, rows in [('all', np.arange(len(frame))), *list(frame.groupby('slide').indices.items())]:
                valid = np.isfinite(x[rows]) & np.isfinite(y[rows])
                a, b = x[rows][valid], y[rows][valid]
                rho = float(spearmanr(a, b).statistic) if len(a) >= 3 and np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan
                records.append(dict(feature=feature, target=name, slide=slide, rows=len(a), spearman=rho))
                if slide != 'all' and np.isfinite(rho):
                    correlations.append(rho)
            records.append(dict(feature=feature, target=name, slide='equal_slide_mean', rows=len(correlations),
                                spearman=float(np.mean(correlations)) if correlations else np.nan))
    return pd.DataFrame(records)


def calibration_tables(frame: pd.DataFrame, probabilities: np.ndarray,
                       thresholds: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Presence/exceedance Brier and reliability on full eligible held-out rows."""
    metrics, bins = [], []
    labels = frame.Density_CD8.to_numpy(float)
    groups = [('all', np.arange(len(frame))), *list(frame.groupby('slide').indices.items())]
    for slide, rows in groups:
        for index, t in enumerate([0., *thresholds]):
            p, y = probabilities[rows, index], (labels[rows] > t).astype(float)
            event = 'presence' if index == 0 else f'exceedance_{index}'
            metrics.append(dict(slide=slide, event=event, threshold=t, rows=len(rows),
                brier=float(np.mean((p-y)**2)), mean_probability=float(p.mean()), observed_fraction=float(y.mean())))
            codes = np.minimum((p*10).astype(int), 9)
            for code in range(10):
                mask = codes == code
                bins.append(dict(slide=slide, event=event, threshold=t, bin=code, rows=int(mask.sum()),
                    mean_probability=float(p[mask].mean()) if mask.any() else np.nan,
                    observed_fraction=float(y[mask].mean()) if mask.any() else np.nan))
    return pd.DataFrame(metrics), pd.DataFrame(bins)

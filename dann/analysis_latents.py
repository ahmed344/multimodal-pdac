"""Shared lesion selection, full covariance PCA, variance plots and Scanpy UMAP."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING
import json
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from dann.analysis_execution import StageCache

THRESHOLDS = (
    (0.9, "pcs_90"),
    (0.99, "pcs_99"),
    (0.999, "pcs_99_9"),
    (0.9999, "pcs_99_99"),
    (0.99999, "pcs_99_999"),
)


def summarize_variance(
    scatter: np.ndarray,
    count: int,
    mean: np.ndarray,
    slide_counts: np.ndarray,
    slide_means: np.ndarray,
    eigenvalues: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute covariance PCA and pixel-weighted between-slide variance fractions."""
    covariance = (scatter + scatter.T) / (2 * (count - 1))
    if eigenvalues is None:
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    tolerance = 100 * np.finfo(float).eps * max(1, len(mean)) * np.max(np.abs(eigenvalues))
    if np.any(eigenvalues < -tolerance):
        raise ValueError("Covariance has materially negative eigenvalues.")
    eigenvalues = np.maximum(eigenvalues, 0)
    total = float(eigenvalues.sum())
    summary: dict[str, Any] = {
        "latent_width": len(mean),
        "n_observations": count,
        "n_slides": len(slide_counts),
        "total_variance": total,
        "participation_ratio": None,
        **{name: None for _, name in THRESHOLDS},
        "variance_explained_by_slide_means": None,
    }
    fractions = np.full(len(mean), np.nan)
    cumulative = fractions.copy()
    if total > 0:
        fractions = eigenvalues / total
        cumulative = np.minimum(np.cumsum(fractions), 1.0)
        cumulative[-1] = 1.0
        summary["participation_ratio"] = float(1.0 / np.dot(fractions, fractions))
        for threshold, name in THRESHOLDS:
            # Count exact threshold ties despite floating-point summation error.
            roundoff = 8 * np.finfo(float).eps * len(mean)
            summary[name] = int(np.searchsorted(cumulative, threshold - roundoff) + 1)
        between = float(np.sum(slide_counts[:, None] * (slide_means - mean) ** 2))
        summary["variance_explained_by_slide_means"] = float(
            np.clip(between / np.trace(scatter), 0, 1)
        )
    spectrum = pd.DataFrame({
        "pc": np.arange(1, len(mean) + 1),
        "eigenvalue": eigenvalues,
        "explained_variance_fraction": fractions,
        "cumulative_variance_fraction": cumulative,
    })
    return summary, spectrum


def plot_explained_variance(
    spectrum: pd.DataFrame, summary: Mapping[str, Any], path: Path, dpi: int = 300
) -> None:
    """Save stacked explained-variance panels under one global title.

    Args:
        spectrum (pd.DataFrame): One row per principal component, including ``pc``,
            ``explained_variance_fraction``, and ``cumulative_variance_fraction``.
        summary (Mapping[str, Any]): Variance summary with ``total_variance``,
            observation and width counts, and the PC count at each threshold.
        path (Path): Destination PNG path.
        dpi (int): Figure resolution. Defaults to 300.

    Returns:
        None
    """
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True, constrained_layout=True)
    try:
        axes[0].set_ylabel("Explained variance (%)")
        axes[0].set_yscale("log")
        axes[1].set_ylabel("Cumulative variance (%)")
        axes[1].set_xlabel("Principal component")
        component_count = len(spectrum)
        # Keep 0 and the last component inside the frame, away from the spines.
        margin = max(1.0, 0.03 * component_count)
        axes[1].set_xlim(-margin, component_count + margin)
        for ax in axes:
            ax.grid(alpha=0.25)
        if summary["total_variance"] == 0:
            for ax in axes:
                ax.text(0.5, 0.5, "Constant latents: variance fractions undefined",
                        ha="center", va="center", transform=ax.transAxes, wrap=True)
        else:
            per_component = 100 * spectrum["explained_variance_fraction"].to_numpy()
            cumulative = 100 * spectrum["cumulative_variance_fraction"].to_numpy()
            axes[0].plot(spectrum["pc"], np.ma.masked_less_equal(per_component, 0))
            axes[1].plot(spectrum["pc"], np.ma.masked_less_equal(cumulative, 0))
            axes[1].set_ylim(-5, 105)
            for (threshold, key), color in zip(THRESHOLDS, ("C1", "C2", "C3", "C4", "C5")):
                pc = summary[key]
                axes[1].axhline(100 * threshold, color=color, linestyle=":", alpha=0.6)
                axes[1].axvline(pc, color=color, linestyle="--", alpha=0.6,
                               label=f"{100 * threshold:g}%: {pc} PCs")
            axes[1].legend(loc="lower right")
        fig.suptitle(f"Latent variance | {summary['n_observations']:,} pixels | {summary['latent_width']} dimensions")
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)


def select_sample_rows(extracted: Mapping[str, Any], config: Mapping[str, Any],
                       batch_names: Sequence[str]) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    """Join metadata by saved source positions, preserving extraction order/context."""
    from dann.config import analysis_feature_settings
    analysis = config['analysis']
    features = analysis_feature_settings(analysis)
    scope = analysis.get('sample_scope', 'lesion' if any(features[n]['enabled'] for n in ('variance', 'pca_umap')) else 'split')
    rows = np.asarray(extracted['row_ids'])
    if rows.ndim != 1 or rows.dtype.kind not in 'iu' or (scope == 'lesion' and len(np.unique(rows)) != len(rows)):
        raise ValueError('Sample row identities must be unique integer positions.')
    with h5py.File(config['data']['path'], 'r') as handle:
        obs = handle['obs']
        batches = np.asarray(read_elem(obs[config['data']['batch_column']])).astype(str)
        slide_column = analysis.get('slide_column', config['data']['batch_column'])
        slides = np.asarray(read_elem(obs[slide_column]))
        if (rows < 0).any() or (rows >= len(batches)).any():
            raise ValueError('Sample row identities are outside source observations.')
        if not np.array_equal(batches[rows], np.asarray(batch_names)[extracted['batches']]):
            raise ValueError('Sample batch identities disagree with source observations.')
        selected = np.ones(len(rows), dtype=bool)
        if scope == 'lesion':
            column = analysis.get('lesion_column', 'Lesion')
            if column not in obs:
                raise ValueError(f'Missing lesion annotation: {column}')
            lesion = np.asarray(read_elem(obs[column]))
            if pd.isna(lesion).any() or not np.isin(lesion, [False, True, 0, 1]).all():
                raise ValueError('Lesion annotation must contain nonmissing boolean values.')
            selected = lesion[rows].astype(bool)
    labels = slides[rows[selected]]
    if pd.isna(labels).any() or any(not str(x).strip() for x in labels):
        raise ValueError('Slide labels cannot be missing or empty.')
    subset = {key: value[selected] if isinstance(value, np.ndarray) and value.ndim and len(value) == len(rows)
              else value for key, value in extracted.items()}
    return subset, labels.astype(str), dict(scope=scope, slide_column=slide_column,
        lesion_column=analysis.get('lesion_column', 'Lesion'), selected_rows=int(selected.sum()),
        split=analysis['split'], source_anndata=str(Path(config['data']['path']).resolve()))


def covariance_pca(latent: np.ndarray) -> dict[str, np.ndarray]:
    """Centered, unscaled float64 covariance with all eigenpairs, including zeros."""
    values = np.asarray(latent, dtype=np.float64)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('Latents must be a finite two-dimensional matrix.')
    if len(values) < 2:
        raise ValueError('At least two latent observations are required.')
    origin = values[0].copy()
    shifted = values - origin
    mean = shifted.mean(axis=0)
    centered = shifted - mean
    scatter = centered.T @ centered
    covariance = (scatter + scatter.T) / (2 * (len(values) - 1))
    eigenvalues, components = np.linalg.eigh(covariance)
    eigenvalues, components = eigenvalues[::-1], components[:, ::-1]
    tolerance = 100 * np.finfo(float).eps * values.shape[1] * np.max(np.abs(eigenvalues))
    if (eigenvalues < -tolerance).any():
        raise ValueError('Covariance has materially negative eigenvalues.')
    # Deterministic eigenvector signs keep cached scores reproducible.
    signs = np.sign(components[np.argmax(np.abs(components), axis=0), np.arange(values.shape[1])])
    components *= signs
    return dict(origin=origin, mean=mean, scatter=scatter,
                eigenvalues=np.maximum(eigenvalues, 0), components=components)


def variance_from_pca(latent: np.ndarray, slides: Sequence[str] | np.ndarray,
                      pca: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Reuse the PCA spectrum and pixel-weighted slide means for the summary."""
    codes, names = pd.factorize(np.asarray(slides), sort=False)
    values = np.asarray(latent, dtype=np.float64) - pca['origin']
    counts = np.bincount(codes, minlength=len(names))
    means = np.stack([values[codes == i].mean(axis=0) for i in range(len(names))])
    return summarize_variance(pca['scatter'], len(values), pca['mean'], counts, means, pca['eigenvalues'])


def select_pc_count(eigenvalues: np.ndarray, threshold: float = .99) -> tuple[int, float]:
    """Smallest count reaching at least the requested fraction; never cap at 50."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    total = eigenvalues.sum()
    if total <= 0:
        return 0, 0.
    cumulative = np.cumsum(eigenvalues) / total
    cumulative[-1] = 1.
    count = int(np.searchsorted(cumulative, threshold, side='left') + 1)
    return count, float(cumulative[count - 1])


def cached_pca(latent: np.ndarray, cache: StageCache) -> dict[str, np.ndarray]:
    from dann.analysis_execution import fingerprint
    key = dict(version=2, latent=fingerprint(latent), numpy=np.__version__)
    hit = cache.get('sample-pca', key)
    if hit is not None:
        return hit[0]
    values = covariance_pca(latent)
    count, _ = select_pc_count(values['eigenvalues'])
    values['scores_99'] = ((np.asarray(latent, dtype=np.float64) - values['origin']) - values['mean']) @ values['components'][:, :count]
    cache.put('sample-pca', key, values)
    return values


def fit_pca_umap(latent: np.ndarray, pca: Mapping[str, np.ndarray] | None,
                 cache: StageCache) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Explicit 99%-variance scores -> Scanpy neighbors -> separately cached UMAP."""
    from dann.analysis_execution import fingerprint
    import importlib.metadata
    metadata = dict(retained_variance=.99, selected_pc_count=0, achieved_variance=0.,
        n_neighbors=min(15, max(0, len(latent) - 1)), metric='euclidean',
        min_dist=.5, spread=1., n_components=2, init_pos='spectral', random_state=0,
        use_rep='X_pca', method='umap', knn=True)
    if len(latent) < 4:
        return None, dict(metadata, skipped='fewer than four observations')
    count, achieved = select_pc_count(pca['eigenvalues'])
    metadata.update(selected_pc_count=count, achieved_variance=achieved)
    if count == 0:
        return None, dict(metadata, skipped='zero total variance')
    import anndata as ad
    import scanpy as sc
    from scipy.sparse import csr_matrix
    scores = pca.get('scores_99')
    if scores is None:
        scores = ((np.asarray(latent, dtype=np.float64) - pca['origin']) - pca['mean']) @ pca['components'][:, :count]
    graph_key = dict(version=1, scores=fingerprint(scores), n_neighbors=metadata['n_neighbors'],
        metric='euclidean', random_state=0, use_rep='X_pca', knn=True, method='umap',
        versions={name: importlib.metadata.version(name) for name in ('scanpy', 'anndata', 'umap-learn', 'pynndescent', 'scipy', 'scikit-learn', 'numba')})
    sample = ad.AnnData(shape=(len(latent), 0))
    sample.obsm['X_pca'] = scores
    hit = cache.get('sample-pca-neighbors', graph_key)
    if hit is None:
        sc.pp.neighbors(sample, n_neighbors=metadata['n_neighbors'], use_rep='X_pca',
                        metric='euclidean', random_state=0, knn=True, method='umap')
        arrays = {}
        for name in ('distances', 'connectivities'):
            matrix = sample.obsp[name].tocsr()
            arrays.update({f'{name}_{key}': getattr(matrix, key) for key in ('data', 'indices', 'indptr')})
        cache.put('sample-pca-neighbors', graph_key, arrays, {'neighbors': sample.uns['neighbors']})
    else:
        arrays, info = hit
        for name in ('distances', 'connectivities'):
            sample.obsp[name] = csr_matrix(tuple(arrays[f'{name}_{key}'] for key in ('data', 'indices', 'indptr')), shape=(len(latent), len(latent)))
        sample.uns['neighbors'] = info['neighbors']
    embedding_key = dict(graph=graph_key, min_dist=.5, spread=1., n_components=2, init_pos='spectral', random_state=0)
    hit = cache.get('sample-pca-umap', embedding_key)
    if hit is None:
        sc.tl.umap(sample, min_dist=.5, spread=1., n_components=2, init_pos='spectral', random_state=0)
        coordinates = sample.obsm['X_umap'].copy()
        cache.put('sample-pca-umap', embedding_key, {'coordinates': coordinates})
    else:
        coordinates = hit[0]['coordinates']
    metadata['versions'] = graph_key['versions']
    return coordinates, metadata

"""Gap-aware neighborhoods and deterministic descriptive spatial envelopes."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .qc import adjacent_pairs


def neighborhoods(x: np.ndarray, y: np.ndarray) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Return unique one- and up-to-two-step four-neighbor paths, excluding self.

    Two-step paths require an occupied intermediate pixel, so they cannot jump
    across a tissue gap. Call separately for each slide and fitting subset.
    """
    n = len(x)
    if n == 0:
        return (sparse.csr_matrix((0, 0), dtype=float),) * 2
    edges = adjacent_pairs(x, y, 2 * n, np.random.default_rng(0))
    rows, cols = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
    one = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    two = one + one @ one
    two.setdiag(0)
    two.eliminate_zeros()
    two.data[:] = 1
    return one, two


def reference_envelopes(values: np.ndarray, draws: int, envelope: float,
                        seed: int) -> np.ndarray:
    """Fit detection and intensity-concordance q95 by neighborhood size 1..12.

    Each draw samples an independent center and twelve random fitting locations
    with replacement, including zero measurements. Concordance is detection
    fraction times exp(-abs(center log1p - positive-neighbor mean log1p)).
    Zero centers have score zero. These are descriptive envelopes, not tests.
    The same sampled locations are used for every feature and block.
    """
    p = values.shape[1]
    result = np.full((2, 13, p), np.nan)
    if len(values) < 2:
        return result
    locations = np.random.default_rng(seed).integers(len(values), size=(draws, 13))
    sample = values[locations]
    positive = sample > 0
    counts = positive[:, 1:].cumsum(axis=1)
    fraction = counts / np.arange(1, 13)[None, :, None]
    logs = np.log1p(sample)
    mean = np.divide(logs[:, 1:].cumsum(axis=1), counts,
                     out=np.zeros_like(fraction), where=counts > 0)
    concordance = fraction * np.exp(-np.abs(logs[:, :1] - mean)) * positive[:, :1]
    result[0, 1:] = np.quantile(fraction, envelope, axis=0)
    result[1, 1:] = np.quantile(concordance, envelope, axis=0)
    return result


def spatial_support(values: np.ndarray, graphs: tuple, envelopes: np.ndarray,
                    min_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Return supported and sufficiently assessed cells on either spatial scale."""
    present = values > 0
    logs = np.log1p(values)
    supported = np.zeros_like(present)
    assessed = np.zeros_like(present)
    for graph in graphs:
        sizes = np.diff(graph.indptr)
        count = graph @ present.astype(float)
        fraction = np.divide(count, sizes[:, None], out=np.zeros_like(count), where=sizes[:, None] > 0)
        mean = np.divide(graph @ logs, count, out=np.zeros_like(count), where=count > 0)
        concordance = fraction * np.exp(-np.abs(logs - mean)) * present
        det_limit, int_limit = envelopes[:, sizes, :]
        valid = ((sizes[:, None] >= min_neighbors) & np.isfinite(det_limit) & np.isfinite(int_limit))
        # Equality preserves constant fields. A zero envelope never supports an
        # isolated measurement: a strictly positive observed score is required.
        supported |= valid & (((fraction > 0) & (fraction >= det_limit))
                              | ((concordance > 0) & (concordance >= int_limit)))
        assessed |= valid
    return supported & present, assessed


def supported_patches(values: np.ndarray, graphs: tuple, envelopes: np.ndarray,
                      minimum: int, min_neighbors: int) -> np.ndarray:
    """Protect features with a connected component of spatially supported pixels."""
    support, _ = spatial_support(values, graphs, envelopes, min_neighbors)
    result = np.zeros(values.shape[1], bool)
    for j in range(values.shape[1]):
        indices = np.flatnonzero(support[:, j])
        if len(indices) < minimum:
            continue
        _, labels = connected_components(graphs[0][indices][:, indices], directed=False)
        result[j] = np.bincount(labels).max() >= minimum
    return result


def removal_mask(values: np.ndarray, graphs: tuple, envelopes: np.ndarray,
                 boundary: float, min_neighbors: int) -> np.ndarray:
    """Remove only weak, assessed cells with neither kind of spatial support."""
    supported, assessed = spatial_support(values, graphs, envelopes, min_neighbors)
    return (values > 0) & (values <= boundary) & assessed & ~supported

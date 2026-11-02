"""Sparse numerical routines for descriptive MSI QC, without model dependencies."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import h5py
import numpy as np
from scipy import sparse


def read_element(element: h5py.Dataset | h5py.Group, selection: Any = None) -> np.ndarray:
    """Read AnnData arrays, categorical values, or nullable string arrays."""
    key = slice(None) if selection is None else selection
    if isinstance(key, np.ndarray) and key.ndim == 1 and len(key) and np.all(np.diff(key) == 1):
        key = slice(int(key[0]), int(key[-1]) + 1)
    if isinstance(element, h5py.Dataset):
        result = element.asstr()[key] if h5py.check_string_dtype(element.dtype) else element[key]
        return np.asarray(result)
    if "categories" in element and "codes" in element:
        codes = np.asarray(element["codes"][key])
        categories = read_element(element["categories"])
        result = categories[np.maximum(codes, 0)].astype(object)
        result[codes < 0] = None
        return result
    if "values" in element and "mask" in element:
        result = read_element(element["values"], key).astype(object)
        result[np.asarray(element["mask"][key], dtype=bool)] = None
        return result
    raise ValueError(f"Unsupported AnnData encoding at {element.name}")


def canonical_block(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Validate and coalesce intensities before interpreting positivity."""
    matrix = matrix.astype(np.float64, copy=True)
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("MSI intensities must be finite and nonnegative.")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all():
        raise ValueError("Summing duplicate intensities overflowed.")
    return matrix


def read_csr_range(group: h5py.Group, start: int, stop: int) -> sparse.csr_matrix:
    """Read a contiguous range of CSR rows, preserving the original feature order."""
    ptr = np.asarray(group["indptr"][start:stop + 1], dtype=np.int64)
    lo, hi = int(ptr[0]), int(ptr[-1])
    indices = np.asarray(group["indices"][lo:hi])
    n_features = int(group.attrs["shape"][1])
    if (np.any(np.diff(ptr) < 0) or np.any(indices < 0) or np.any(indices >= n_features)
            or lo < 0 or hi > len(group["data"])):
        raise ValueError("Invalid CSR pointers or feature indices.")
    matrix = sparse.csr_matrix((group["data"][lo:hi], indices, ptr - lo),
                              shape=(stop - start, n_features))
    return canonical_block(matrix)


def iter_blocks(group: h5py.Group, rows: np.ndarray, block_rows: int
                ) -> Iterator[tuple[int, int, sparse.csr_matrix]]:
    """Yield bounded row blocks, including slide rows that are not contiguous."""
    for start in range(0, len(rows), block_rows):
        stop = min(start + block_rows, len(rows))
        selected = rows[start:stop]
        cuts = np.r_[0, np.flatnonzero(np.diff(selected) != 1) + 1, len(selected)]
        parts = [read_csr_range(group, int(selected[a]), int(selected[b - 1]) + 1)
                 for a, b in zip(cuts[:-1], cuts[1:])]
        yield start, stop, parts[0] if len(parts) == 1 else sparse.vstack(parts, format="csr")


def row_sums(values: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    """Sum a vector over CSR rows, including empty rows."""
    cumulative = np.r_[0., np.cumsum(values, dtype=np.float64)]
    return np.diff(cumulative[indptr])


def threshold_histograms(matrix: sparse.csr_matrix, thresholds: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Count and sum positive measurements by peak and inclusive cutoff bin."""
    nbins = len(thresholds) + 1
    bins = np.searchsorted(thresholds, matrix.data, side="left")
    flat = matrix.indices.astype(np.int64) * nbins + bins
    shape = (matrix.shape[1], nbins)
    counts = np.bincount(flat, minlength=np.prod(shape)).reshape(shape)
    totals = np.bincount(flat, weights=matrix.data, minlength=np.prod(shape)).reshape(shape)
    return counts, totals


def adjacent_pairs(x: np.ndarray, y: np.ndarray, maximum: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Sample unique four-neighbor edges within one slide, without crossing holes."""
    xy = np.column_stack((x, y))
    if not np.isfinite(xy).all() or not np.equal(xy, np.round(xy)).all():
        raise ValueError("Acquisition-grid coordinates must be finite integers.")
    x, y = x.astype(np.int64), y.astype(np.int64)
    x, y = x - x.min(), y - y.min()
    stride = int(y.max()) + 2
    key = x * stride + y
    order = np.argsort(key)
    sorted_key = key[order]
    if np.any(np.diff(sorted_key) == 0):
        raise ValueError("Duplicate acquisition-grid coordinates within a slide.")
    edges = []
    for offset in (1, stride):
        targets = key + offset
        positions = np.searchsorted(sorted_key, targets)
        valid = positions < len(key)
        valid[valid] &= sorted_key[positions[valid]] == targets[valid]
        edges.append(np.column_stack((np.flatnonzero(valid), order[positions[valid]])))
    pairs = np.vstack(edges)
    if len(pairs) > maximum:
        pairs = pairs[rng.choice(len(pairs), maximum, replace=False)]
    return pairs


def column_moments(matrix: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Compute sums and centered sums of squares without subtracting large moments."""
    n, p = matrix.shape
    sums = np.bincount(matrix.indices, weights=matrix.data, minlength=p)
    mean = sums / max(n, 1)
    centered = matrix.data - mean[matrix.indices]
    nnz = np.bincount(matrix.indices, minlength=p)
    variance = np.bincount(matrix.indices, weights=centered ** 2, minlength=p)
    variance += (n - nnz) * mean ** 2
    return sums, variance


def column_correlations(a: sparse.csr_matrix, b: sparse.csr_matrix,
                        moments: tuple | None = None) -> np.ndarray:
    """Pearson correlation per feature, with implicit zeros included."""
    n = a.shape[0]
    if moments is None:
        sa, va = column_moments(a)
        sb, vb = column_moments(b)
    else:
        sa, sb, va, vb = moments
    # Cancellation in E[x²] - E[x]² can otherwise make constant columns appear
    # variable after log1p. Use a relative floating-point error tolerance.
    tolerance = 128 * np.finfo(np.float64).eps
    va = np.where(va > tolerance * np.maximum(sa * sa / max(n, 1), 1.), va, 0.)
    vb = np.where(vb > tolerance * np.maximum(sb * sb / max(n, 1), 1.), vb, 0.)
    product = np.asarray(a.multiply(b).sum(axis=0)).ravel()
    denominator = np.sqrt(np.maximum(va, 0) * np.maximum(vb, 0))
    return np.divide(product - sa * sb / max(n, 1), denominator,
                     out=np.full(a.shape[1], np.nan), where=denominator > 1e-12).clip(-1, 1)


def spatial_diagnostics(matrix: sparse.csr_matrix, pairs: np.ndarray, tic: np.ndarray,
                        config: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Compare adjacent-pixel concordance with descriptive shuffled controls.

    Shuffle quantiles are reference envelopes, not calibrated p-values. Pairs may
    share pixels; no independent-replicate or spatial FDR claim is made.
    """
    p = matrix.shape[1]
    output = {"spatial_detections": np.asarray(matrix.getnnz(axis=0)).ravel()}
    if not len(pairs):
        for label in ("raw", "normalized", "binary"):
            for suffix in ("correlation", "null_q95", "excess"):
                output[f"{label}_{suffix}"] = np.full(p, np.nan)
        output["spatial_tested"] = np.zeros(p, bool)
        output["spatial_supported"] = np.zeros(p, bool)
        return output
    permutations = [rng.permutation(len(pairs)) for _ in range(config["shuffles"])]
    for label in ("raw", "normalized", "binary"):
        values = matrix.copy()
        if label == "binary":
            values.data[:] = 1.
        else:
            if label == "normalized":
                positive_tic = tic[tic > 0]
                reference = float(np.median(positive_tic)) if len(positive_tic) else 1.
                scales = np.divide(reference, tic, out=np.zeros_like(tic), where=tic > 0)
                values.data *= np.repeat(scales, np.diff(values.indptr))
            np.log1p(values.data, out=values.data)
        a, b = values[pairs[:, 0]], values[pairs[:, 1]]
        sa, va = column_moments(a)
        sb, vb = column_moments(b)
        moments = sa, sb, va, vb
        observed = column_correlations(a, b, moments)
        null = np.stack([column_correlations(a, b[perm], moments) for perm in permutations])
        valid = np.isfinite(null).all(axis=0) & np.isfinite(observed)
        q95 = np.full(p, np.nan)
        q95[valid] = np.quantile(null[:, valid], .95, axis=0)
        output[f"{label}_correlation"] = observed
        output[f"{label}_null_q95"] = q95
        output[f"{label}_excess"] = observed - q95
    output["spatial_tested"] = ((output["spatial_detections"] >= config["min_spatial_detections"])
                                  & np.isfinite(output["raw_excess"])
                                  & np.isfinite(output["normalized_excess"]))
    output["spatial_supported"] = (output["spatial_tested"]
                                   & (output["raw_excess"] >= config["spatial_support_excess"])
                                   & (output["normalized_excess"] >= config["spatial_support_excess"]))
    return output


def robust_limits(values: np.ndarray, multiplier: float) -> dict[str, float | bool]:
    """Compute three-scaled-MAD screens in log10(1+x) units; zero MAD disables them."""
    logged = np.log10(1. + values)
    center = float(np.median(logged))
    spread = float(1.4826 * np.median(np.abs(logged - center)))
    return {"median_log10p": center, "scaled_mad_log10p": spread,
            "enabled": spread > 0,
            "lower": max(0., float(10 ** (center - multiplier * spread) - 1)) if spread > 0 else 0.,
            "upper": float(10 ** (center + multiplier * spread) - 1) if spread > 0 else float("inf")}


def select_threshold(slide_losses: list[tuple[np.ndarray, np.ndarray]], thresholds: np.ndarray,
                     signal_min: float, measurements_min: float) -> float:
    """Select the largest cutoff meeting both retention budgets on every slide."""
    acceptable = np.ones(len(thresholds), dtype=bool)
    for count_loss, signal_loss in slide_losses:
        acceptable &= ((count_loss <= 1 - measurements_min + 1e-12)
                       & (signal_loss <= 1 - signal_min + 1e-12))
    return float(thresholds[np.flatnonzero(acceptable)[-1]]) if acceptable.any() else 0.

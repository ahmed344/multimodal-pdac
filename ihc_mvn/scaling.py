"""Frozen sparse log1p and within-slide unit-variance preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np


def matrix_group_path(matrix_key: str) -> str:
    """Translate an AnnData matrix key to its HDF5 group path.

    Args:
        matrix_key (str): ``"X"``, a layer name, or ``"layers/<name>"``.

    Returns:
        str: HDF5 path containing a CSR matrix.
    """

    if matrix_key == "X":
        return "X"
    if matrix_key.startswith("layers/"):
        return matrix_key
    return f"layers/{matrix_key}"


def _contiguous_runs(indices: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield half-open contiguous runs from sorted unique row indices.

    Args:
        indices (np.ndarray): Sorted unique integer row indices.

    Returns:
        Iterator[tuple[int, int]]: Half-open ``(start, stop)`` row runs.
    """

    if indices.size == 0:
        return
    boundaries = np.flatnonzero(np.diff(indices) != 1) + 1
    for run in np.split(indices, boundaries):
        yield int(run[0]), int(run[-1]) + 1


def _decode_attribute(value: Any) -> str:
    """Normalize an HDF5 text attribute to a Python string.

    Args:
        value (Any): HDF5 attribute value.

    Returns:
        str: Decoded attribute text.
    """

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@dataclass(frozen=True)
class SparseFeatureScaler:
    """Frozen per-slide sparse feature scales with a global fallback."""

    slide_names: tuple[str, ...]
    slide_to_code: dict[str, int]
    feature_names: tuple[str, ...]
    per_slide_scales: np.ndarray
    global_scales: np.ndarray
    slide_training_counts: np.ndarray
    scale_floor: float
    matrix_key: str
    transform: str
    center: bool
    num_features: int
    transform_metadata: dict[str, Any]

    def scales_for_slide(self, slide_code: int) -> np.ndarray:
        """Select frozen scales for a known or unknown slide.

        Args:
            slide_code (int): Encoded slide, with negative values denoting unknown slides.

        Returns:
            np.ndarray: Per-feature scale vector; unknown slides use global scales.
        """

        if (
            0 <= int(slide_code) < len(self.slide_names)
            and self.slide_training_counts[int(slide_code)] > 0
        ):
            return self.per_slide_scales[int(slide_code)]
        return self.global_scales

    def transform_sparse(
        self,
        feature_indices: np.ndarray,
        stored_values: np.ndarray,
        slide_code: int,
    ) -> np.ndarray:
        """Apply log1p and frozen scale division to stored CSR values only.

        Args:
            feature_indices (np.ndarray): Feature indices for one sparse row.
            stored_values (np.ndarray): Corresponding raw stored count values.
            slide_code (int): Encoded slide or ``-1`` for an unknown slide.

        Returns:
            np.ndarray: Transformed float32 stored values.
        """

        features = np.asarray(feature_indices, dtype=np.int64)
        values = np.asarray(stored_values, dtype=np.float64)
        if features.ndim != 1 or values.ndim != 1 or features.shape != values.shape:
            raise ValueError("feature_indices and stored_values must be matching 1D arrays.")
        if np.any((features < 0) | (features >= self.num_features)):
            raise IndexError("Sparse feature index is outside the fitted feature space.")
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("Raw sparse counts must be finite and nonnegative.")
        if self.transform != "log1p" or self.center:
            raise RuntimeError("Unsupported scaler transform metadata.")
        transformed = np.log1p(values)
        transformed /= self.scales_for_slide(int(slide_code))[features]
        return transformed.astype(np.float32)

    def to_state(self) -> dict[str, Any]:
        """Convert fitted scaler state to plain serialization values.

        Args:
            None.

        Returns:
            dict[str, Any]: Mapping containing only strings, booleans, numbers,
                dictionaries, and lists.
        """

        return {
            "slide_names": list(self.slide_names),
            "slide_to_code": dict(self.slide_to_code),
            "feature_names": list(self.feature_names),
            "per_slide_scales": self.per_slide_scales.astype(float).tolist(),
            "global_scales": self.global_scales.astype(float).tolist(),
            "slide_training_counts": self.slide_training_counts.astype(int).tolist(),
            "scale_floor": float(self.scale_floor),
            "matrix_key": self.matrix_key,
            "transform": self.transform,
            "center": bool(self.center),
            "num_features": int(self.num_features),
            "transform_metadata": dict(self.transform_metadata),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "SparseFeatureScaler":
        """Restore a fitted scaler from plain serialization state.

        Args:
            state (Mapping[str, Any]): State previously returned by ``to_state``.

        Returns:
            SparseFeatureScaler: Restored frozen scaler.
        """

        names = tuple(str(value) for value in state["slide_names"])
        feature_names = tuple(str(value) for value in state["feature_names"])
        mapping = {str(key): int(value) for key, value in state["slide_to_code"].items()}
        expected_mapping = {name: index for index, name in enumerate(names)}
        if mapping != expected_mapping:
            raise ValueError("Stored slide mapping is inconsistent with slide_names.")
        per_slide = np.asarray(state["per_slide_scales"], dtype=np.float64)
        global_scales = np.asarray(state["global_scales"], dtype=np.float64)
        counts = np.asarray(state["slide_training_counts"], dtype=np.int64)
        num_features = int(state["num_features"])
        if per_slide.shape != (len(names), num_features):
            raise ValueError("Stored per-slide scale array has an invalid shape.")
        if len(feature_names) != num_features or len(set(feature_names)) != num_features:
            raise ValueError("Stored feature names must be unique and match num_features.")
        if global_scales.shape != (num_features,) or counts.shape != (len(names),):
            raise ValueError("Stored global scales or slide counts have invalid shapes.")
        if np.any(per_slide <= 0.0) or np.any(global_scales <= 0.0):
            raise ValueError("All stored feature scales must be positive.")
        return cls(
            slide_names=names,
            slide_to_code=mapping,
            feature_names=feature_names,
            per_slide_scales=per_slide,
            global_scales=global_scales,
            slide_training_counts=counts,
            scale_floor=float(state["scale_floor"]),
            matrix_key=str(state["matrix_key"]),
            transform=str(state["transform"]),
            center=bool(state["center"]),
            num_features=num_features,
            transform_metadata=dict(state["transform_metadata"]),
        )


def _accumulate_window(
    group: h5py.Group,
    start_row: int,
    stop_row: int,
    selected_rows: set[int] | None,
    slide_codes: np.ndarray,
    sums: np.ndarray,
    sums_of_squares: np.ndarray,
    global_sums: np.ndarray,
    global_sums_of_squares: np.ndarray,
) -> None:
    """Accumulate transformed sparse moments from one contiguous CSR row window.

    Args:
        group (h5py.Group): CSR HDF5 group.
        start_row (int): Inclusive global row start.
        stop_row (int): Exclusive global row stop.
        selected_rows (set[int] | None): Selected rows in the window, or ``None``
            when every row is selected.
        slide_codes (np.ndarray): Slide code for every matrix row.
        sums (np.ndarray): Mutable per-slide feature sums.
        sums_of_squares (np.ndarray): Mutable per-slide feature squared sums.
        global_sums (np.ndarray): Mutable global feature sums.
        global_sums_of_squares (np.ndarray): Mutable global feature squared sums.

    Returns:
        None: Moment arrays are updated in place.
    """

    pointers = np.asarray(group["indptr"][start_row : stop_row + 1], dtype=np.int64)
    data_start = int(pointers[0])
    data_stop = int(pointers[-1])
    features = np.asarray(group["indices"][data_start:data_stop], dtype=np.int64)
    raw_values = np.asarray(group["data"][data_start:data_stop], dtype=np.float64)
    if not np.isfinite(raw_values).all() or np.any(raw_values < 0.0):
        raise ValueError(
            f"Raw sparse counts must be finite and nonnegative in rows "
            f"[{start_row}, {stop_row})."
        )
    values = np.log1p(raw_values)
    local_pointers = pointers - data_start
    for local_row, global_row in enumerate(range(start_row, stop_row)):
        if selected_rows is not None and global_row not in selected_rows:
            continue
        row_start = int(local_pointers[local_row])
        row_stop = int(local_pointers[local_row + 1])
        row_features = features[row_start:row_stop]
        row_values = values[row_start:row_stop]
        slide_code = int(slide_codes[global_row])
        np.add.at(sums[slide_code], row_features, row_values)
        np.add.at(sums_of_squares[slide_code], row_features, row_values * row_values)
        np.add.at(global_sums, row_features, row_values)
        np.add.at(global_sums_of_squares, row_features, row_values * row_values)


def fit_sparse_feature_scaler(
    path: str | Path,
    matrix_key: str,
    train_indices: np.ndarray,
    slide_codes: np.ndarray,
    slide_names: Sequence[str],
    feature_names: Sequence[str] | None = None,
    scale_floor: float = 1.0e-6,
    row_chunk_size: int = 4096,
) -> SparseFeatureScaler:
    """Fit log1p feature scales from selected training CSR rows.

    Variance is computed over every selected row in a slide, including implicit
    sparse zeros, as ``E[x^2] - E[x]^2``. No dense row-by-feature MSI matrix is
    materialized.

    Args:
        path (str | Path): Backed AnnData HDF5 path.
        matrix_key (str): CSR matrix key, normally ``"layers/counts"``.
        train_indices (np.ndarray): Actual selected global training row indices.
        slide_codes (np.ndarray): Slide code for every matrix row.
        slide_names (Sequence[str]): Ordered fitted slide category names.
        feature_names (Sequence[str] | None): Ordered training feature names. When
            omitted, names are read from the AnnData ``var`` index.
        scale_floor (float): Lower bound applied to feature standard deviations.
        row_chunk_size (int): Maximum contiguous rows read per dense-selection window.

    Returns:
        SparseFeatureScaler: Frozen fitted sparse preprocessing model.
    """

    file_path = Path(path)
    selected = np.unique(np.asarray(train_indices, dtype=np.int64))
    codes = np.asarray(slide_codes, dtype=np.int64)
    names = tuple(str(value) for value in slide_names)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("train_indices must contain at least one row.")
    if codes.ndim != 1:
        raise ValueError("slide_codes must be a 1D array.")
    if np.any((selected < 0) | (selected >= codes.size)):
        raise IndexError("train_indices contain rows outside slide_codes.")
    if np.any((codes < 0) | (codes >= len(names))):
        raise ValueError("Training metadata contains invalid slide codes.")
    if scale_floor <= 0.0:
        raise ValueError("scale_floor must be positive.")
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive.")

    group_path = matrix_group_path(matrix_key)
    with h5py.File(file_path, "r") as handle:
        if group_path not in handle:
            raise KeyError(f"Sparse matrix group {group_path!r} is absent from {file_path}.")
        group = handle[group_path]
        encoding = _decode_attribute(group.attrs.get("encoding-type", ""))
        if encoding != "csr_matrix":
            raise TypeError(
                f"{group_path!r} must be CSR encoded, observed {encoding!r}."
            )
        shape = tuple(int(value) for value in group.attrs["shape"])
        if len(shape) != 2 or shape[0] != codes.size:
            raise ValueError("CSR matrix shape disagrees with slide metadata.")
        num_features = shape[1]
        if feature_names is None:
            var_group = handle["var"]
            index_key = _decode_attribute(var_group.attrs.get("_index", "_index"))
            index_object = var_group[index_key]
            if isinstance(index_object, h5py.Dataset):
                feature_values = index_object.asstr()[:]
            elif "values" in index_object:
                feature_values = index_object["values"].asstr()[:]
            else:
                raise TypeError("Unsupported AnnData var index encoding.")
            fitted_feature_names = tuple(str(value) for value in feature_values)
        else:
            fitted_feature_names = tuple(str(value) for value in feature_names)
        if len(fitted_feature_names) != num_features:
            raise ValueError("Feature names do not match the CSR feature count.")
        if len(set(fitted_feature_names)) != num_features:
            raise ValueError("Feature names must be unique.")
        sums = np.zeros((len(names), num_features), dtype=np.float64)
        sums_of_squares = np.zeros_like(sums)
        global_sums = np.zeros(num_features, dtype=np.float64)
        global_sums_of_squares = np.zeros(num_features, dtype=np.float64)

        covered_span = int(selected[-1] - selected[0] + 1)
        selection_density = selected.size / covered_span
        if selection_density >= 0.10:
            selected_set = set(int(value) for value in selected)
            for start in range(int(selected[0]), int(selected[-1]) + 1, row_chunk_size):
                stop = min(start + row_chunk_size, int(selected[-1]) + 1)
                _accumulate_window(
                    group,
                    start,
                    stop,
                    selected_set,
                    codes,
                    sums,
                    sums_of_squares,
                    global_sums,
                    global_sums_of_squares,
                )
        else:
            for run_start, run_stop in _contiguous_runs(selected):
                for start in range(run_start, run_stop, row_chunk_size):
                    stop = min(start + row_chunk_size, run_stop)
                    _accumulate_window(
                        group,
                        start,
                        stop,
                        None,
                        codes,
                        sums,
                        sums_of_squares,
                        global_sums,
                        global_sums_of_squares,
                    )

    slide_counts = np.bincount(codes[selected], minlength=len(names)).astype(np.int64)
    global_count = int(selected.size)
    global_mean = global_sums / global_count
    global_variance = np.maximum(
        global_sums_of_squares / global_count - global_mean * global_mean,
        0.0,
    )
    global_scales = np.sqrt(global_variance)
    global_scales[global_scales < scale_floor] = 1.0

    per_slide_scales = np.empty_like(sums)
    for slide_code, count in enumerate(slide_counts):
        if count == 0:
            per_slide_scales[slide_code] = global_scales
            continue
        mean = sums[slide_code] / int(count)
        variance = np.maximum(
            sums_of_squares[slide_code] / int(count) - mean * mean,
            0.0,
        )
        fitted_scales = np.sqrt(variance)
        use_global = fitted_scales < scale_floor
        fitted_scales[use_global] = global_scales[use_global]
        per_slide_scales[slide_code] = fitted_scales

    mapping = {name: index for index, name in enumerate(names)}
    return SparseFeatureScaler(
        slide_names=names,
        slide_to_code=mapping,
        feature_names=fitted_feature_names,
        per_slide_scales=per_slide_scales,
        global_scales=global_scales,
        slide_training_counts=slide_counts,
        scale_floor=float(scale_floor),
        matrix_key=matrix_key,
        transform="log1p",
        center=False,
        num_features=num_features,
        transform_metadata={
            "input": "stored CSR counts",
            "nonlinear_transform": "log1p",
            "variance": "E[x^2] - E[x]^2 over selected training rows including zeros",
            "division": "per-slide per-feature standard deviation",
            "centering": False,
            "unknown_slide": "global per-feature training standard deviation",
            "constant_slide_feature": "global per-feature training standard deviation",
            "constant_global_feature": "unit scale",
            "frozen_at_inference": True,
        },
    )

"""Count construction and affine standardization for IHC density targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

TARGET_COLUMNS = (
    "Density_Tumor",
    "Density_Stroma",
    "Density_Collagen",
    "Density_CD8",
)


@dataclass(frozen=True)
class TargetArrays:
    """Derived count-space and coordinate-space target arrays."""

    densities: np.ndarray
    counts: np.ndarray
    extratumoral_counts: np.ndarray
    denominators: np.ndarray
    positive_mask: np.ndarray
    coordinates: np.ndarray


@dataclass(frozen=True)
class TargetStandardizer:
    """Per-target affine standardizer fitted on positive training rows only."""

    target_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    positive_counts: np.ndarray
    standard_deviation_floor: float

    @classmethod
    def fit(
        cls,
        coordinates: np.ndarray,
        positive_mask: np.ndarray,
        train_indices: np.ndarray,
        target_names: Sequence[str] = TARGET_COLUMNS,
        standard_deviation_floor: float = 1.0e-6,
    ) -> "TargetStandardizer":
        """Fit affine statistics using positive selected training rows only.

        Args:
            coordinates (np.ndarray): Haldane-Anscombe coordinates with shape
                ``(observations, targets)``.
            positive_mask (np.ndarray): Boolean positive-count mask of the same shape.
            train_indices (np.ndarray): Actual selected global training row indices.
            target_names (Sequence[str]): Ordered target names.
            standard_deviation_floor (float): Lower bound for fitted standard deviations.

        Returns:
            TargetStandardizer: Frozen training-fitted affine standardizer.
        """

        values = np.asarray(coordinates, dtype=np.float64)
        mask = np.asarray(positive_mask, dtype=bool)
        indices = np.asarray(train_indices, dtype=np.int64)
        names = tuple(str(name) for name in target_names)
        if values.ndim != 2 or mask.shape != values.shape:
            raise ValueError("coordinates and positive_mask must be matching 2D arrays.")
        if values.shape[1] != len(names):
            raise ValueError("The number of target names does not match coordinate columns.")
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("train_indices must be a non-empty 1D array.")
        if np.any((indices < 0) | (indices >= values.shape[0])):
            raise IndexError("train_indices contain rows outside the target arrays.")
        if standard_deviation_floor <= 0.0:
            raise ValueError("standard_deviation_floor must be positive.")

        means = np.empty(values.shape[1], dtype=np.float64)
        scales = np.empty(values.shape[1], dtype=np.float64)
        positive_counts = np.empty(values.shape[1], dtype=np.int64)
        for target_index in range(values.shape[1]):
            selected_mask = mask[indices, target_index]
            selected = values[indices[selected_mask], target_index]
            if selected.size == 0:
                raise ValueError(
                    f"Target {names[target_index]!r} has no positive selected training rows."
                )
            means[target_index] = selected.mean(dtype=np.float64)
            fitted_scale = selected.std(dtype=np.float64, ddof=0)
            scales[target_index] = max(float(fitted_scale), standard_deviation_floor)
            positive_counts[target_index] = selected.size

        return cls(
            target_names=names,
            means=means,
            scales=scales,
            positive_counts=positive_counts,
            standard_deviation_floor=float(standard_deviation_floor),
        )

    def transform(self, coordinates: np.ndarray) -> np.ndarray:
        """Apply the frozen per-target affine transformation.

        Args:
            coordinates (np.ndarray): Coordinates whose final dimension is targets.

        Returns:
            np.ndarray: Standardized float32 coordinates of the same shape.
        """

        values = np.asarray(coordinates, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != len(self.target_names):
            raise ValueError("Coordinate final dimension does not match fitted targets.")
        standardized = (values - self.means) / self.scales
        return standardized.astype(np.float32)

    def inverse_transform(self, standardized: np.ndarray) -> np.ndarray:
        """Undo the frozen per-target affine transformation.

        Args:
            standardized (np.ndarray): Standardized values with targets last.

        Returns:
            np.ndarray: Unstandardized float64 Haldane-Anscombe coordinates.
        """

        values = np.asarray(standardized, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != len(self.target_names):
            raise ValueError("Standardized final dimension does not match fitted targets.")
        return values * self.scales + self.means

    def to_state(self) -> dict[str, Any]:
        """Convert fitted statistics to plain serialization state.

        Args:
            None.

        Returns:
            dict[str, Any]: Mapping containing only strings, numbers, and lists.
        """

        return {
            "target_names": list(self.target_names),
            "means": self.means.astype(float).tolist(),
            "scales": self.scales.astype(float).tolist(),
            "positive_counts": self.positive_counts.astype(int).tolist(),
            "standard_deviation_floor": float(self.standard_deviation_floor),
            "fit_scope": "selected_training_positive_counts_only",
            "affine_transform": "(coordinate - mean) / scale",
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "TargetStandardizer":
        """Restore a standardizer from plain serialization state.

        Args:
            state (Mapping[str, Any]): State previously returned by ``to_state``.

        Returns:
            TargetStandardizer: Restored frozen standardizer.
        """

        names = tuple(str(value) for value in state["target_names"])
        means = np.asarray(state["means"], dtype=np.float64)
        scales = np.asarray(state["scales"], dtype=np.float64)
        positive_counts = np.asarray(state["positive_counts"], dtype=np.int64)
        if not (means.shape == scales.shape == positive_counts.shape == (len(names),)):
            raise ValueError("Target standardizer state arrays have inconsistent shapes.")
        if np.any(scales <= 0.0):
            raise ValueError("Target standardizer scales must be positive.")
        return cls(
            target_names=names,
            means=means,
            scales=scales,
            positive_counts=positive_counts,
            standard_deviation_floor=float(state["standard_deviation_floor"]),
        )


def density_to_counts(densities: np.ndarray, total_count: int = 36_100) -> np.ndarray:
    """Convert bounded density proportions to rounded integer counts.

    Args:
        densities (np.ndarray): Density array with targets in its final dimension.
        total_count (int): Pixel count ``N`` represented by a unit density.

    Returns:
        np.ndarray: Integer counts ``round(N * density)`` of matching shape.
    """

    values = np.asarray(densities, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != len(TARGET_COLUMNS):
        raise ValueError(f"densities must have {len(TARGET_COLUMNS)} targets last.")
    if not np.isfinite(values).all():
        raise ValueError("Density targets contain non-finite values.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Density targets must lie inside [0, 1].")
    if total_count <= 0:
        raise ValueError("total_count must be positive.")
    return np.rint(values * int(total_count)).astype(np.int64)


def target_denominators(counts: np.ndarray, total_count: int = 36_100) -> np.ndarray:
    """Construct tumor and residual-compartment target denominators.

    Args:
        counts (np.ndarray): Integer counts ordered as tumor, stroma, collagen, CD8.
        total_count (int): Tumor denominator ``N``.

    Returns:
        np.ndarray: Denominators with ``n_T=N`` and ``n_j=max(N-k_T, k_j)``.
    """

    values = np.asarray(counts, dtype=np.int64)
    residual = extratumoral_counts(values, total_count)
    denominators = np.empty_like(values)
    denominators[..., 0] = int(total_count)
    denominators[..., 1:] = np.maximum(residual[..., None], values[..., 1:])
    return denominators


def extratumoral_counts(
    counts: np.ndarray,
    total_count: int = 36_100,
) -> np.ndarray:
    """Compute residual extratumoral counts ``E = N - k_T``.

    Args:
        counts (np.ndarray): Integer counts ordered with tumor first.
        total_count (int): Total pixel count ``N``.

    Returns:
        np.ndarray: Integer residual count with the target dimension removed.
    """

    values = np.asarray(counts, dtype=np.int64)
    if values.ndim < 1 or values.shape[-1] != len(TARGET_COLUMNS):
        raise ValueError(f"counts must have {len(TARGET_COLUMNS)} targets last.")
    if np.any(values < 0) or np.any(values > total_count):
        raise ValueError("Counts must lie between zero and total_count.")
    return int(total_count) - values[..., 0]


def haldane_anscombe_coordinates(
    counts: np.ndarray,
    denominators: np.ndarray,
    correction: float = 0.5,
) -> np.ndarray:
    """Compute finite Haldane-Anscombe log-odds coordinates.

    Args:
        counts (np.ndarray): Target success counts.
        denominators (np.ndarray): Corresponding binomial denominators.
        correction (float): Pseudocount added to successes and failures.

    Returns:
        np.ndarray: Float64 values ``log((k+c)/(n-k+c))``.
    """

    successes = np.asarray(counts, dtype=np.float64)
    totals = np.asarray(denominators, dtype=np.float64)
    if successes.shape != totals.shape:
        raise ValueError("counts and denominators must have identical shapes.")
    if correction <= 0.0:
        raise ValueError("correction must be positive.")
    if np.any(successes < 0.0) or np.any(totals < successes):
        raise ValueError("Every denominator must be at least its count.")
    return np.log((successes + correction) / (totals - successes + correction))


def build_target_arrays(
    densities: np.ndarray,
    total_count: int = 36_100,
    correction: float = 0.5,
) -> TargetArrays:
    """Build counts, denominators, positive masks, and log-odds coordinates.

    Args:
        densities (np.ndarray): Bounded densities ordered according to ``TARGET_COLUMNS``.
        total_count (int): Pixel count ``N`` represented by a unit density.
        correction (float): Haldane-Anscombe pseudocount.

    Returns:
        TargetArrays: Complete immutable collection of derived target arrays.
    """

    density_values = np.asarray(densities, dtype=np.float32)
    counts = density_to_counts(density_values, total_count)
    residual = extratumoral_counts(counts, total_count)
    denominators = target_denominators(counts, total_count)
    positive_mask = counts > 0
    coordinates = haldane_anscombe_coordinates(counts, denominators, correction)
    return TargetArrays(
        densities=density_values,
        counts=counts,
        extratumoral_counts=residual,
        denominators=denominators,
        positive_mask=positive_mask,
        coordinates=coordinates,
    )

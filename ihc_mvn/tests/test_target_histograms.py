"""Positive-only histograms of the four IHC target transform stages."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from ihc_mvn.targets import (
    TARGET_COLUMNS,
    TargetStandardizer,
    build_target_arrays,
)

STAGE_LABELS = (
    "Density",
    "Count reconstruction",
    "Haldane–Anscombe log-odds",
    "Affine standardization",
)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def package_config() -> dict[str, object]:
    """Load the standalone IHC-MVN YAML configuration.

    Args:
        None.

    Returns:
        dict[str, object]: Parsed configuration mapping.
    """

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError("ihc_mvn/config.yaml must contain a mapping.")
    return loaded


def assembled_h5ad_path() -> Path:
    """Return the configured assembled training AnnData path.

    Args:
        None.

    Returns:
        Path: Absolute path to ``adata_assembled.h5ad``.
    """

    return Path(str(package_config()["data"]["path"]))


def default_histogram_output_path() -> Path:
    """Return the default PNG path under the package output directory.

    Args:
        None.

    Returns:
        Path: ``positive_target_transform_histograms.png`` under analysis/.
    """

    root = Path(str(package_config()["output"]["directory"]))
    return root / "analysis" / "positive_target_transform_histograms.png"


def load_density_matrix(path: Path) -> np.ndarray:
    """Load the four ordered density columns without reading the MSI matrix.

    Args:
        path (Path): AnnData ``.h5ad`` path containing ``TARGET_COLUMNS`` in ``obs``.

    Returns:
        np.ndarray: Float32 density matrix with shape ``(observations, 4)``.
    """

    adata = ad.read_h5ad(path, backed="r")
    try:
        missing = [name for name in TARGET_COLUMNS if name not in adata.obs]
        if missing:
            raise KeyError(f"Missing density columns: {missing}")
        densities = adata.obs[list(TARGET_COLUMNS)].to_numpy(
            dtype=np.float32, copy=True
        )
    finally:
        adata.file.close()
    if densities.ndim != 2 or densities.shape[1] != len(TARGET_COLUMNS):
        raise ValueError("Loaded densities must have four target columns.")
    if not np.isfinite(densities).all():
        raise ValueError("Density targets contain non-finite values.")
    return densities


def collect_positive_stages(
    densities: np.ndarray,
    *,
    total_count: int = 36_100,
    correction: float = 0.5,
    standard_deviation_floor: float = 1.0e-6,
) -> list[list[np.ndarray]]:
    """Build the four transform stages, keeping positive reconstructed counts only.

    Affine mean and scale are fitted on every positive row of ``densities`` so
    the histograms describe this file rather than a training split.

    Args:
        densities (np.ndarray): Bounded densities ordered as ``TARGET_COLUMNS``.
        total_count (int): Pixel count ``N`` used to reconstruct integer counts.
        correction (float): Haldane-Anscombe pseudocount.
        standard_deviation_floor (float): Lower bound for fitted scales.

    Returns:
        list[list[np.ndarray]]: Nested lists with one row per target and one
            column per stage. Each array is 1D and contains only positives.
    """

    arrays = build_target_arrays(
        densities, total_count=total_count, correction=correction
    )
    row_count = int(arrays.coordinates.shape[0])
    standardizer = TargetStandardizer.fit(
        arrays.coordinates,
        arrays.positive_mask,
        np.arange(row_count, dtype=np.int64),
        target_names=TARGET_COLUMNS,
        standard_deviation_floor=standard_deviation_floor,
    )
    standardized = standardizer.transform(arrays.coordinates)
    stages: list[list[np.ndarray]] = []
    for target_index in range(len(TARGET_COLUMNS)):
        mask = arrays.positive_mask[:, target_index]
        stages.append(
            [
                arrays.densities[mask, target_index].astype(np.float64, copy=False),
                arrays.counts[mask, target_index].astype(np.float64, copy=False),
                arrays.coordinates[mask, target_index],
                standardized[mask, target_index].astype(np.float64, copy=False),
            ]
        )
    return stages


def plot_positive_target_transform_histograms(
    densities: np.ndarray,
    output_path: Path,
    *,
    source_label: str,
    total_count: int = 36_100,
    correction: float = 0.5,
    standard_deviation_floor: float = 1.0e-6,
    bins: int = 80,
    dpi: int = 200,
) -> Path:
    """Write a 4x4 histogram grid of positive target transform stages.

    Args:
        densities (np.ndarray): Bounded densities ordered as ``TARGET_COLUMNS``.
        output_path (Path): Destination PNG path.
        source_label (str): File or dataset name shown in the figure title.
        total_count (int): Pixel count ``N`` used to reconstruct integer counts.
        correction (float): Haldane-Anscombe pseudocount.
        standard_deviation_floor (float): Lower bound for fitted scales.
        bins (int): Histogram bin count shared by every panel.
        dpi (int): Saved figure resolution.

    Returns:
        Path: The written PNG path.
    """

    if bins <= 1:
        raise ValueError("bins must be greater than one.")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")

    stages = collect_positive_stages(
        densities,
        total_count=total_count,
        correction=correction,
        standard_deviation_floor=standard_deviation_floor,
    )
    figure, axes = plt.subplots(
        nrows=len(TARGET_COLUMNS),
        ncols=len(STAGE_LABELS),
        figsize=(16, 14),
        constrained_layout=True,
    )
    for row_index, target in enumerate(TARGET_COLUMNS):
        short_name = target.removeprefix("Density_")
        for column_index, stage_name in enumerate(STAGE_LABELS):
            axis = axes[row_index, column_index]
            values = stages[row_index][column_index]
            axis.hist(values, bins=bins, color="#4C72B0", alpha=0.9)
            axis.grid(True, axis="y", alpha=0.3)
            if row_index == 0:
                axis.set_title(stage_name)
            if column_index == 0:
                axis.set_ylabel(short_name)
            axis.text(
                0.02,
                0.98,
                (
                    f"n={values.size:,}\n"
                    f"mean={values.mean():.3g}\n"
                    f"sd={values.std(ddof=0):.3g}"
                ),
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
    figure.suptitle(
        "Positive IHC target transforms\n"
        f"{source_label}  |  affine fit on all positive rows of this file"
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)
    return destination


def _target_settings() -> tuple[int, float, float]:
    """Return configured count, Haldane, and scale-floor constants.

    Args:
        None.

    Returns:
        tuple[int, float, float]: ``total_count``, Haldane correction, and
            standard-deviation floor.
    """

    targets = package_config()["targets"]
    return (
        int(targets["total_count"]),
        float(targets["haldane_correction"]),
        float(targets["standard_deviation_floor"]),
    )


def test_positive_stages_keep_positive_counts_only() -> None:
    """Verify zeros are dropped and stages follow the production transforms.

    Args:
        None.

    Returns:
        None: Assertions check masks, counts, Haldane values, and affine stats.
    """

    densities = np.asarray(
        [
            [0.0, 0.2, 0.3, 0.4],
            [0.1, 0.0, 0.3, 0.4],
            [0.2, 0.2, 0.0, 0.5],
            [0.9, 0.9, 0.9, 0.0],
        ],
        dtype=np.float32,
    )
    arrays = build_target_arrays(densities)
    stages = collect_positive_stages(densities)
    standardizer = TargetStandardizer.fit(
        arrays.coordinates,
        arrays.positive_mask,
        np.arange(densities.shape[0], dtype=np.int64),
    )
    standardized = standardizer.transform(arrays.coordinates)

    assert len(stages) == len(TARGET_COLUMNS)
    for target_index in range(len(TARGET_COLUMNS)):
        mask = arrays.positive_mask[:, target_index]
        density_values, count_values, haldane_values, affine_values = stages[
            target_index
        ]
        assert mask.sum() == density_values.size == count_values.size
        assert np.all(density_values > 0.0)
        np.testing.assert_array_equal(
            count_values, arrays.counts[mask, target_index]
        )
        np.testing.assert_allclose(
            haldane_values, arrays.coordinates[mask, target_index]
        )
        np.testing.assert_allclose(
            affine_values, standardized[mask, target_index]
        )
        np.testing.assert_allclose(affine_values.mean(), 0.0, atol=1.0e-6)
        np.testing.assert_allclose(affine_values.std(ddof=0), 1.0, atol=1.0e-5)


def test_positive_target_transform_histograms_on_synthetic(
    tiny_h5ad: Path, tmp_path: Path
) -> None:
    """Write a 4x4 histogram PNG from the synthetic labeled fixture.

    Args:
        tiny_h5ad (Path): Synthetic labeled CSR AnnData.
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: The expected nonempty PNG is produced.
    """

    densities = load_density_matrix(tiny_h5ad)
    output_path = tmp_path / "positive_target_transform_histograms.png"
    total_count, correction, scale_floor = _target_settings()
    written = plot_positive_target_transform_histograms(
        densities,
        output_path,
        source_label=tiny_h5ad.name,
        total_count=total_count,
        correction=correction,
        standard_deviation_floor=scale_floor,
        dpi=72,
    )
    assert written == output_path
    assert written.stat().st_size > 0


@pytest.mark.skipif(
    not assembled_h5ad_path().is_file(),
    reason="Configured assembled AnnData is not present.",
)
def test_positive_target_transform_histograms_on_assembled_data() -> None:
    """Write the 4x4 diagnostic figure from the assembled PDAC AnnData.

    Args:
        None.

    Returns:
        None: The configured analysis PNG is produced and nonempty.
    """

    input_path = assembled_h5ad_path()
    densities = load_density_matrix(input_path)
    total_count, correction, scale_floor = _target_settings()
    written = plot_positive_target_transform_histograms(
        densities,
        default_histogram_output_path(),
        source_label=str(input_path),
        total_count=total_count,
        correction=correction,
        standard_deviation_floor=scale_floor,
    )
    assert written.is_file()
    assert written.stat().st_size > 0

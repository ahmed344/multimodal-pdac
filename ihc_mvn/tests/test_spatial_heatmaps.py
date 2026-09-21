"""Contracts for ordered MVN spatial visualization."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ihc_mvn.spatial_heatmaps import (
    add_spatial_derived_columns,
    attach_predictions,
    plot_target_spatial_heatmap,
    validate_ordered_frame,
)
from ihc_mvn.targets import TARGET_COLUMNS, build_target_arrays


def spatial_adata() -> ad.AnnData:
    """Create a tiny two-slide spatial AnnData.

    Args:
        None.

    Returns:
        ad.AnnData: Four-row object with coordinates and one observed target.
    """

    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(["a", "a", "b", "b"]),
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "Density_Tumor": [0.1, 0.2, 0.3, 0.4],
        },
        index=[f"pixel_{index}" for index in range(4)],
    )
    return ad.AnnData(X=np.zeros((4, 1), dtype=np.float32), obs=obs)


def tumor_prediction_frame() -> pd.DataFrame:
    """Create aligned prediction columns needed by one spatial target.

    Args:
        None.

    Returns:
        pd.DataFrame: Ordered identity and Tumor prediction columns.
    """

    frame = pd.DataFrame(
        {
            "row_position": np.arange(4),
            "obs_name": [f"pixel_{index}" for index in range(4)],
            "prob_presence_Density_Tumor": [0.5, 0.6, 0.7, 0.8],
            "haldane_mean_msi_Density_Tumor": [-1.0, -0.5, 0.0, 0.5],
            "haldane_mean_patient_Density_Tumor": [-0.8, -0.3, 0.2, 0.7],
            "haldane_mean_total_Density_Tumor": [-0.6, -0.1, 0.4, 0.9],
            "positive_median_density_Density_Tumor": [0.35, 0.48, 0.60, 0.71],
            "q05_density_Density_Tumor": [0.1, 0.2, 0.3, 0.4],
            "q95_density_Density_Tumor": [0.6, 0.7, 0.8, 0.9],
        }
    )
    return frame


def test_validate_ordered_frame_rejects_name_mismatch() -> None:
    """Verify duplicate-safe observation identity validation.

    Args:
        None.

    Returns:
        None: A mismatched observation name raises ``ValueError``.
    """

    adata = spatial_adata()
    frame = tumor_prediction_frame()
    frame.loc[2, "obs_name"] = "wrong"

    with pytest.raises(ValueError, match="observation names"):
        validate_ordered_frame(adata, frame, "Predictions")


def test_spatial_derived_columns_and_plot(tmp_path: Path) -> None:
    """Verify MVN layer medians, uncertainty, residuals, and PNG output.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Derived values and nonempty spatial figure are asserted.
    """

    adata = spatial_adata()
    attach_predictions(adata, tumor_prediction_frame())
    created = add_spatial_derived_columns(adata, ("Density_Tumor",))
    output_path = tmp_path / "spatial.png"
    settings = {
        "batch_column": "batch",
        "x_column": "x",
        "y_column": "y",
        "figure_dpi": 72,
        "cmap": "viridis",
        "residual_cmap": "coolwarm",
        "include_hes": False,
    }

    plot_target_spatial_heatmap(
        adata,
        "Density_Tumor",
        ("a", "b"),
        output_path,
        settings,
    )

    assert "interval_width_Density_Tumor" in created
    assert "observed_haldane_Density_Tumor" in created
    assert np.allclose(adata.obs["interval_width_Density_Tumor"], 0.5)
    padded = np.zeros((4, len(TARGET_COLUMNS)), dtype=np.float32)
    padded[:, 0] = adata.obs["Density_Tumor"].to_numpy(dtype=np.float32)
    expected_haldane = build_target_arrays(padded).coordinates[:, 0]
    np.testing.assert_allclose(
        adata.obs["observed_haldane_Density_Tumor"].to_numpy(dtype=np.float64),
        expected_haldane,
    )
    assert output_path.stat().st_size > 0

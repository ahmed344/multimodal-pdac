"""Focused tests for spatial heatmap AnnData join and logit helpers."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from dann.spatial import build_output_table, prediction_column_names
from dann.spatial_heatmaps import add_logit_columns, attach_predictions


TARGETS = (
    "Density_CD8",
    "Density_Tumor",
    "Density_Collagen",
    "Density_Stroma",
)


def _tiny_adata(obs_names: list[str], densities: dict[str, np.ndarray]) -> ad.AnnData:
    """Build a small in-memory AnnData with density observation columns.

    Args:
        obs_names (list[str]): Observation names defining row order.
        densities (dict[str, np.ndarray]): Density column name to values.

    Returns:
        ad.AnnData: AnnData with a CSR placeholder matrix and density columns.
    """

    n_obs = len(obs_names)
    adata = ad.AnnData(X=sparse.csr_matrix(np.zeros((n_obs, 2), dtype=np.float32)))
    adata.obs_names = obs_names
    adata.var_names = ["500.1", "501.2"]
    for name, values in densities.items():
        adata.obs[name] = np.asarray(values, dtype=np.float32)
    return adata


def test_attach_predictions_joins_float32_columns_in_order() -> None:
    """Verify ordered prediction columns attach when identity matches.

    Args:
        None.

    Returns:
        None: Assertions validate joined prediction columns and dtypes.
    """

    obs_names = ["a", "b", "a"]
    adata = _tiny_adata(
        obs_names,
        {target: np.zeros(3, dtype=np.float32) for target in TARGETS},
    )
    table = build_output_table(
        row_positions=np.arange(3, dtype=np.int64),
        obs_names=np.asarray(obs_names, dtype=str),
        mu=np.arange(12, dtype=np.float32).reshape(3, 4),
        pi=np.full((3, 4), 0.25, dtype=np.float32),
        sigma=np.ones((3, 4), dtype=np.float32),
        target_columns=TARGETS,
    )
    predictions = table.to_pandas()
    attached = attach_predictions(adata, predictions)

    assert attached == prediction_column_names(TARGETS)
    for name in attached:
        assert name in adata.obs.columns
        assert adata.obs[name].dtype == np.float32
    np.testing.assert_allclose(
        adata.obs["prob_of_presence_Density_CD8"].to_numpy(),
        np.full(3, 0.75, dtype=np.float32),
    )


def test_attach_predictions_rejects_obs_name_mismatch() -> None:
    """Verify attach fails when Parquet obs names diverge from AnnData.

    Args:
        None.

    Returns:
        None: Assertions require an obs_name alignment error.
    """

    adata = _tiny_adata(
        ["a", "b"],
        {target: np.zeros(2, dtype=np.float32) for target in TARGETS},
    )
    table = build_output_table(
        row_positions=np.arange(2, dtype=np.int64),
        obs_names=np.asarray(["a", "c"], dtype=str),
        mu=np.zeros((2, 4), dtype=np.float32),
        pi=np.full((2, 4), 0.5, dtype=np.float32),
        sigma=np.ones((2, 4), dtype=np.float32),
        target_columns=TARGETS,
    )
    with pytest.raises(ValueError, match="obs_name"):
        attach_predictions(adata, table.to_pandas())


def test_attach_predictions_rejects_noncontiguous_row_position() -> None:
    """Verify attach fails when row_position is not AnnData order.

    Args:
        None.

    Returns:
        None: Assertions require a row_position alignment error.
    """

    adata = _tiny_adata(
        ["a", "b"],
        {target: np.zeros(2, dtype=np.float32) for target in TARGETS},
    )
    predictions = pd.DataFrame(
        {
            "row_position": np.asarray([1, 0], dtype=np.int64),
            "obs_name": ["b", "a"],
            "mean_Density_CD8": np.asarray([0.0, 1.0], dtype=np.float32),
        }
    )
    with pytest.raises(ValueError, match="row_position"):
        attach_predictions(adata, predictions)


def test_add_logit_columns_clips_and_transforms() -> None:
    """Verify logit columns use the clamped transform for each density.

    Args:
        None.

    Returns:
        None: Assertions validate logit values and created column names.
    """

    epsilon = 1e-4
    densities = ("Density_CD8", "Density_Tumor")
    adata = _tiny_adata(
        ["0", "1", "2"],
        {
            "Density_CD8": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            "Density_Tumor": np.asarray([0.25, 0.75, 0.0], dtype=np.float32),
        },
    )
    columns = add_logit_columns(adata, densities, epsilon)

    assert columns == ["logit_Density_CD8", "logit_Density_Tumor"]
    expected_cd8 = np.log(
        np.clip(np.asarray([0.0, 0.5, 1.0]), epsilon, 1.0 - epsilon)
        / (1.0 - np.clip(np.asarray([0.0, 0.5, 1.0]), epsilon, 1.0 - epsilon))
    ).astype(np.float32)
    np.testing.assert_allclose(
        adata.obs["logit_Density_CD8"].to_numpy(),
        expected_cd8,
        rtol=1e-5,
    )
    assert adata.obs["logit_Density_CD8"].dtype == np.float32


def test_add_logit_columns_rejects_missing_density() -> None:
    """Verify logit helper requires every requested density column.

    Args:
        None.

    Returns:
        None: Assertions require a missing-column KeyError.
    """

    adata = _tiny_adata(["0"], {"Density_CD8": np.asarray([0.5], dtype=np.float32)})
    with pytest.raises(KeyError, match="Density_Tumor"):
        add_logit_columns(adata, ["Density_CD8", "Density_Tumor"], 1e-4)

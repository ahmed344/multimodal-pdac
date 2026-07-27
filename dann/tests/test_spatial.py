"""Focused tests for ordered DANN spatial inference exports."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy import sparse

from dann.spatial import (
    build_output_table,
    output_schema,
    prediction_column_names,
    validate_input_schema,
    validate_output,
)


TARGETS = (
    "Density_CD8",
    "Density_Tumor",
    "Density_Collagen",
    "Density_Stroma",
)


def _write_tiny_anndata(
    path: Path,
    var_names: list[str],
    obs_names: list[str],
) -> Path:
    """Write a small CSR AnnData with the production target schema.

    Args:
        path (Path): Destination H5AD path.
        var_names (list[str]): Ordered feature names.
        obs_names (list[str]): Possibly duplicated observation names.

    Returns:
        Path: Written AnnData path.
    """

    matrix = sparse.csr_matrix(
        np.arange(len(obs_names) * len(var_names), dtype=np.float32).reshape(
            len(obs_names), len(var_names)
        )
    )
    adata = ad.AnnData(X=matrix)
    adata.obs_names = obs_names
    adata.var_names = var_names
    for target in TARGETS:
        adata.obs[target] = np.zeros(len(obs_names), dtype=np.float32)
    adata.write_h5ad(path)
    return path


def test_output_table_preserves_triplets_dtypes_and_duplicate_names() -> None:
    """Verify names, float32 channels, presence conversion, and duplicate IDs.

    Args:
        None.

    Returns:
        None: Assertions validate one in-memory output table.
    """

    rows = np.asarray([0, 1], dtype=np.int64)
    obs_names = np.asarray(["pixel", "pixel"], dtype=str)
    mu = np.arange(8, dtype=np.float32).reshape(2, 4)
    pi = np.asarray(
        [[0.0, 0.25, 0.5, 1.0], [1.0, 0.5, 0.25, 0.0]], dtype=np.float32
    )
    sigma = np.full((2, 4), 0.75, dtype=np.float32)
    table = build_output_table(rows, obs_names, mu, pi, sigma, TARGETS)

    assert table.schema.equals(output_schema(TARGETS))
    assert table.column_names == [
        "row_position",
        "obs_name",
        *prediction_column_names(TARGETS),
    ]
    assert table.column("obs_name").to_pylist() == ["pixel", "pixel"]
    assert table.column("row_position").type == output_schema(TARGETS).field(
        "row_position"
    ).type
    for name in prediction_column_names(TARGETS):
        assert str(table.column(name).type) == "float"
    np.testing.assert_allclose(
        table.column("prob_of_presence_Density_CD8").to_numpy(),
        np.asarray([1.0, 0.0], dtype=np.float32),
    )


def test_validate_input_schema_rejects_reordered_features(tmp_path: Path) -> None:
    """Verify inference refuses a tissue matrix with different feature order.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions require a feature-order validation error.
    """

    training = _write_tiny_anndata(
        tmp_path / "training.h5ad", ["500.1", "501.2"], ["0", "1"]
    )
    tissue = _write_tiny_anndata(
        tmp_path / "tissue.h5ad", ["501.2", "500.1"], ["0", "1"]
    )
    with pytest.raises(ValueError, match="feature order"):
        validate_input_schema(
            input_path=tissue,
            training_path=training,
            matrix_key="X",
            target_columns=TARGETS,
            num_peaks=2,
        )


def test_parquet_validation_uses_position_with_duplicate_obs_names(
    tmp_path: Path,
) -> None:
    """Verify full validation accepts duplicated names in exact positional order.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions validate a complete small Parquet artifact.
    """

    tissue = _write_tiny_anndata(
        tmp_path / "tissue.h5ad",
        ["500.1", "501.2"],
        ["0", "1", "0"],
    )
    rows = np.arange(3, dtype=np.int64)
    obs_names = np.asarray(["0", "1", "0"], dtype=str)
    mu = np.zeros((3, 4), dtype=np.float32)
    pi = np.full((3, 4), 0.25, dtype=np.float32)
    sigma = np.ones((3, 4), dtype=np.float32)
    table = build_output_table(rows, obs_names, mu, pi, sigma, TARGETS)
    output = tmp_path / "predictions.parquet"
    pq.write_table(table, output)

    report = validate_output(output, tissue, TARGETS, expected_rows=3, batch_size=2)
    assert report["rows"] == 3
    assert report["minima"]["prob_of_presence_Density_CD8"] == pytest.approx(0.75)
    assert report["maxima"]["sigma_Density_Stroma"] == pytest.approx(1.0)

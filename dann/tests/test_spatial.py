"""Focused tests for ordered DANN spatial inference exports."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse

from dann import ziln
from dann.spatial import (
    build_latent_output_table,
    build_output_table,
    latent_column_names,
    latent_output_schema,
    output_schema,
    prediction_column_names,
    validate_input_schema,
    validate_latent_output,
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


def test_output_table_preserves_parameters_dtypes_and_duplicate_names() -> None:
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
    alpha = np.arange(8, dtype=np.float32).reshape(2, 4) - 4.0
    table = build_output_table(rows, obs_names, mu, pi, sigma, alpha, TARGETS)

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
    np.testing.assert_allclose(
        table.column("alpha_Density_CD8").to_numpy(),
        np.asarray([-4.0, 0.0], dtype=np.float32),
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
    alpha = np.full((3, 4), -1.5, dtype=np.float32)
    table = build_output_table(rows, obs_names, mu, pi, sigma, alpha, TARGETS)
    output = tmp_path / "predictions.parquet"
    pq.write_table(table, output)

    report = validate_output(output, tissue, TARGETS, expected_rows=3, batch_size=2)
    assert report["rows"] == 3
    assert report["minima"]["prob_of_presence_Density_CD8"] == pytest.approx(0.75)
    assert report["maxima"]["sigma_Density_Stroma"] == pytest.approx(1.0)
    assert report["minima"]["alpha_Density_CD8"] == pytest.approx(-1.5)


def test_latent_parquet_preserves_dimension_order_and_duplicate_names(
    tmp_path: Path,
) -> None:
    """Verify latent export and validation are dimension-agnostic and ordered.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions validate a small three-dimensional latent artifact.
    """

    tissue = _write_tiny_anndata(
        tmp_path / "tissue.h5ad",
        ["500.1", "501.2"],
        ["pixel", "other", "pixel"],
    )
    rows = np.arange(3, dtype=np.int64)
    obs_names = np.asarray(["pixel", "other", "pixel"], dtype=str)
    latent = np.arange(9, dtype=np.float32).reshape(3, 3)
    table = build_latent_output_table(rows, obs_names, latent)
    output = tmp_path / "latent.parquet"
    pq.write_table(table, output)

    assert table.schema.equals(latent_output_schema(3))
    assert table.column_names == [
        "row_position",
        "obs_name",
        *latent_column_names(3),
    ]
    report = validate_latent_output(
        output,
        tissue,
        latent_dim=3,
        expected_rows=3,
        batch_size=2,
    )
    assert report["latent_dim"] == 3
    assert report["minimum"] == pytest.approx(0.0)
    assert report["maximum"] == pytest.approx(8.0)


def test_validate_latent_output_rejects_nonfinite_values(tmp_path: Path) -> None:
    """Verify latent validation rejects non-finite embedding coordinates.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions require a non-finite-value validation error.
    """

    tissue = _write_tiny_anndata(
        tmp_path / "tissue.h5ad",
        ["500.1", "501.2"],
        ["0", "1"],
    )
    latent = np.asarray([[0.0, 1.0], [np.nan, 2.0]], dtype=np.float32)
    table = build_latent_output_table(
        np.arange(2, dtype=np.int64),
        np.asarray(["0", "1"], dtype=str),
        latent,
    )
    output = tmp_path / "latent.parquet"
    pq.write_table(table, output)

    with pytest.raises(ValueError, match="non-finite"):
        validate_latent_output(output, tissue, latent_dim=2, expected_rows=2)


def test_output_table_derives_ziln_summaries_from_raw_parameters() -> None:
    """Verify derived columns match dann.ziln and stay quantile-ordered.

    Args:
        None.

    Returns:
        None: Assertions validate the derived summary columns.
    """

    rows = np.asarray([0, 1], dtype=np.int64)
    obs_names = np.asarray(["a", "b"], dtype=str)
    mu = np.asarray([[0.0, -1.0, 2.0, 0.5], [1.0, 0.0, -2.0, 3.0]], dtype=np.float32)
    pi = np.full((2, 4), 0.25, dtype=np.float32)
    sigma = np.asarray([[1.0, 0.5, 2.0, 1.5], [0.75, 1.0, 0.25, 2.5]], dtype=np.float32)
    alpha = np.asarray([[0.0, 3.0, -4.0, 1.0], [-2.0, 0.0, 5.0, -1.0]], dtype=np.float32)
    table = build_output_table(rows, obs_names, mu, pi, sigma, alpha, TARGETS)

    for index, target in enumerate(TARGETS):
        column_mu = mu[:, index].astype(np.float64)
        column_sigma = sigma[:, index].astype(np.float64)
        column_alpha = alpha[:, index].astype(np.float64)
        np.testing.assert_allclose(
            table.column(f"logit_mean_{target}").to_numpy(),
            ziln.positive_logit_mean(column_mu, column_sigma, column_alpha),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            table.column(f"logit_sd_{target}").to_numpy(),
            ziln.positive_logit_sd(column_sigma, column_alpha),
            rtol=1e-6,
        )
        lower = table.column(f"logit_q05_{target}").to_numpy()
        median = table.column(f"logit_median_{target}").to_numpy()
        upper = table.column(f"logit_q95_{target}").to_numpy()
        assert np.all(lower < median)
        assert np.all(median < upper)

    # A zero-skew target must leave mu untouched, which is the whole point of
    # keeping mu and logit_mean as separate columns.
    zero_skew = TARGETS[1]
    np.testing.assert_allclose(
        table.column(f"logit_mean_{zero_skew}").to_numpy()[1],
        table.column(f"mu_{zero_skew}").to_numpy()[1],
        rtol=1e-6,
    )


def test_validate_output_rejects_unordered_quantiles(tmp_path: Path) -> None:
    """Verify validation catches a corrupted quantile ordering.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions require an ordering ValueError.
    """

    tissue = _write_tiny_anndata(
        tmp_path / "tissue.h5ad",
        var_names=["mz_0", "mz_1"],
        obs_names=["a", "b"],
    )
    rows = np.arange(2, dtype=np.int64)
    table = build_output_table(
        rows,
        np.asarray(["a", "b"], dtype=str),
        np.zeros((2, len(TARGETS)), dtype=np.float32),
        np.full((2, len(TARGETS)), 0.5, dtype=np.float32),
        np.ones((2, len(TARGETS)), dtype=np.float32),
        np.zeros((2, len(TARGETS)), dtype=np.float32),
        TARGETS,
    )
    corrupted_name = f"logit_q05_{TARGETS[0]}"
    corrupted = table.set_column(
        table.schema.get_field_index(corrupted_name),
        pa.field(corrupted_name, pa.float32(), nullable=False),
        pa.array(np.full(2, 99.0, dtype=np.float32), type=pa.float32()),
    )
    output = tmp_path / "corrupted.parquet"
    pq.write_table(corrupted, output)

    with pytest.raises(ValueError, match="not ordered"):
        validate_output(output, tissue, TARGETS, expected_rows=2)

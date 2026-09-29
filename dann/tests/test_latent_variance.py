"""Numerical and artifact contracts for standalone latent variance diagnostics."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from scipy import sparse

from dann.config import latent_variance_settings, load_config
from dann.latent_variance import compute_latent_variance, latent_columns, main


def write_inputs(tmp_path, values, labels, extra_rows=0):
    """Create sparse source metadata and numerically reordered latent columns."""
    count, width = values.shape
    names = ["duplicate"] * (count + extra_rows)
    adata = ad.AnnData(sparse.csr_matrix((len(names), 2)))
    adata.obs_names = names
    adata.obs["batch"] = pd.Categorical([*labels, *(["unused"] * extra_rows)])
    input_path = tmp_path / "input.h5ad"
    adata.write_h5ad(input_path)
    frame = pd.DataFrame({"row_position": np.arange(count, dtype=np.int64), "obs_name": names[:count]})
    for i in reversed(range(width)):
        frame[f"latent_{i}"] = values[:, i].astype(np.float64)
    latent_path = tmp_path / "latent.parquet"
    frame.to_parquet(latent_path, index=False)
    return latent_path, input_path


@pytest.mark.parametrize("chunk_size", [1, 2, 10])
def test_known_spectrum_and_chunk_invariance(tmp_path, chunk_size):
    values = np.array([[3, 0], [-3, 0], [0, 1], [0, -1]], dtype=float)
    paths = write_inputs(tmp_path, values, ["a", "a", "b", "b"])
    summary, spectrum = compute_latent_variance(*paths, chunk_size=chunk_size)
    np.testing.assert_allclose(spectrum.eigenvalue, [6, 2 / 3])
    np.testing.assert_allclose(spectrum.explained_variance_fraction, [0.9, 0.1])
    assert summary["participation_ratio"] == pytest.approx(1 / 0.82)
    assert [summary[k] for k in ("pcs_90", "pcs_99", "pcs_99_9", "pcs_99_99", "pcs_99_999")] == [1, 2, 2, 2, 2]
    assert summary["variance_explained_by_slide_means"] == 0
    assert summary["latent_width"] == 2


@pytest.mark.parametrize("chunk_size", [1, 3, 8])
def test_pixel_weighted_slide_means_and_large_offset(tmp_path, chunk_size):
    values = np.array([[0, 2], [2, 0], [4, 1], [8, 3]], dtype=float) + 1e9
    labels = np.array(["a", "a", "a", "b"])
    paths = write_inputs(tmp_path, values, labels)
    summary, spectrum = compute_latent_variance(*paths, chunk_size=chunk_size)
    mean = values.mean(axis=0)
    total = np.sum((values - mean) ** 2)
    between = sum(np.sum(labels == s) * np.sum((values[labels == s].mean(axis=0) - mean) ** 2)
                  for s in np.unique(labels))
    assert summary["variance_explained_by_slide_means"] == pytest.approx(between / total)
    np.testing.assert_allclose(spectrum.eigenvalue, np.linalg.eigvalsh(np.cov(values.T))[::-1], rtol=1e-7)


@pytest.mark.parametrize("constant", [False, True])
def test_single_slide_and_constant_latents(tmp_path, constant):
    values = np.full((3, 2), 0.1) if constant else np.arange(6).reshape(3, 2)
    summary, spectrum = compute_latent_variance(*write_inputs(tmp_path, values, ["a"] * 3))
    if constant:
        assert summary["total_variance"] == 0
        assert summary["participation_ratio"] is None
        assert summary["pcs_90"] is None
        assert summary["variance_explained_by_slide_means"] is None
        assert spectrum.explained_variance_fraction.isna().all()
    else:
        assert summary["variance_explained_by_slide_means"] == 0
        assert summary["participation_ratio"] == pytest.approx(1)


def test_prefix_and_duplicate_names(tmp_path):
    paths = write_inputs(tmp_path, np.arange(36).reshape(3, 12), ["a", "b", "b"], extra_rows=2)
    summary, _ = compute_latent_variance(*paths)
    assert summary["inference_coverage_fraction"] == 0.6
    assert summary["is_prefix"]
    assert summary["n_slides"] == 2
    assert latent_columns(pq.read_schema(paths[0])) == [f"latent_{i}" for i in range(12)]


@pytest.mark.parametrize("problem,match", [
    ("positions", "row_position"), ("names", "obs_name order"),
    ("nan", "finite"), ("infinity", "finite"), ("gap", "contiguous latent"),
    ("null", "null"), ("dtype", "floating-point"),
])
def test_invalid_parquet(tmp_path, problem, match):
    paths = write_inputs(tmp_path, np.arange(6).reshape(3, 2), ["a"] * 3)
    frame = pd.read_parquet(paths[0])
    if problem == "positions":
        frame.loc[1, "row_position"] = 0
    elif problem == "names":
        frame.loc[1, "obs_name"] = "wrong"
    elif problem in ("nan", "infinity"):
        frame.loc[1, "latent_0"] = np.nan if problem == "nan" else np.inf
    elif problem == "gap":
        frame = frame.rename(columns={"latent_1": "latent_2"})
    elif problem == "null":
        frame.loc[1, "obs_name"] = None
    else:
        frame["latent_0"] = frame.latent_0.astype(int)
    # Preserve IEEE NaN rather than converting it into an Arrow null.
    table = pa.table({key: pa.array(frame[key].to_numpy(), from_pandas=(key == "obs_name")) for key in frame})
    pq.write_table(table, paths[0])
    with pytest.raises(ValueError, match=match):
        compute_latent_variance(*paths)


def test_missing_labels_and_inputs(tmp_path):
    paths = write_inputs(tmp_path, np.arange(6).reshape(3, 2), ["a", None, "b"])
    with pytest.raises(ValueError, match="Slide labels"):
        compute_latent_variance(*paths)
    with pytest.raises(ValueError, match="missing slide column"):
        compute_latent_variance(*paths, slide_column="absent")
    with pytest.raises(FileNotFoundError, match="dann.spatial"):
        compute_latent_variance(tmp_path / "missing", paths[1])


def test_too_few_rows(tmp_path):
    paths = write_inputs(tmp_path, np.ones((1, 2)), ["a"])
    with pytest.raises(ValueError, match="At least two"):
        compute_latent_variance(*paths)


@pytest.mark.parametrize("key,value", [("chunk_size", 0), ("chunk_size", True),
    ("figure_dpi", 1.5), ("slide_column", ""), ("latents", None)])
def test_config_validation(key, value):
    with pytest.raises(ValueError, match=f"latent_variance.{key}"):
        latent_variance_settings({key: value})


@pytest.mark.parametrize("constant", [False, True])
def test_cli_and_legacy_config(tmp_path, constant):
    values = np.ones((3, 2)) if constant else np.arange(6).reshape(3, 2)
    paths = write_inputs(tmp_path, values, ["a", "a", "b"])
    config = yaml.safe_load(Path("dann/config.yaml").read_text())
    config.pop("latent_variance")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert load_config(config_path)["latent_variance"]["chunk_size"] == 8192
    output = tmp_path / "results"
    assert main(["--config", str(config_path), "--latents", str(paths[0]),
                 "--input", str(paths[1]), "--output-dir", str(output)]) == 0
    assert {p.name for p in output.iterdir()} == {
        "explained_variance.png", "latent_variance_summary.csv", "pca_variance.csv"}
    assert (output / "explained_variance.png").stat().st_size > 1000
    summary = pd.read_csv(output / "latent_variance_summary.csv")
    assert summary.latent_width.iloc[0] == 2
    assert summary.n_observations.iloc[0] == 3
    if constant:
        assert pd.isna(summary.participation_ratio.iloc[0])

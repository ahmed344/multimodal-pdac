"""Row-removal, cohort provenance and downstream compatibility regressions."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import anndata as ad
from anndata.io import write_elem
import h5py
from matplotlib import colormaps
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image
import pytest
from scipy import sparse
import yaml

from msi_preprocess.apply import apply
from msi_preprocess.cohort import aligned_export_rows, decide_cohort
from msi_preprocess.comparison import render_heatmaps
from msi_preprocess.config import validate_config
from msi_preprocess.fit import fit, load_fit
from msi_preprocess.io import match_rows
from msi_preprocess.tests.test_pipeline import fixture
from msi_preprocess.verify import verify


def cohort_fixture(tmp_path, interleaved=True):
    config, dann = fixture(tmp_path)
    original = ad.read_h5ad(config["tissue_input"])
    base = original[original.obs.batch == "a"].copy()
    slides = []
    for index, name in enumerate("abcde"):
        slide = base.copy()
        slide.obs_names = [f"{name}_{i}" for i in range(len(slide))]
        slide.obs["batch"] = name
        slide.obs["patient"] = f"patient_{index // 2}"
        slide.layers["counts"] = slide.layers["counts"] * (index + 1)
        slide.X = slide.layers["counts"] * 3
        slides.append(slide)
    tissue = ad.concat(slides)
    tissue.obs["batch"] = pd.Categorical(tissue.obs.batch, categories=list("edcba") + ["unused"])
    tissue.obs["label"] = pd.Categorical(tissue.obs.batch.astype(str))
    tissue.obsm["spatial"] = tissue.obs[["x", "y"]].to_numpy()
    tissue.uns["spatial"] = {name: {"images": {"HES": np.zeros((16, 16, 3), dtype=np.uint8)},
                                  "scalefactors": {"tissue_HES_scalef": 1., "spot_diameter_fullres": 1.}} for name in "abcde"}
    tissue.obsp["sparse"] = sparse.diags([np.ones(len(tissue)), np.ones(len(tissue)-1)], [0, 1], format="csr")
    tissue.obsp["dense"] = np.arange(len(tissue)**2, dtype=float).reshape(len(tissue), len(tissue))
    if interleaved:
        tissue = tissue[np.random.default_rng(9).permutation(len(tissue))].copy()
    lesion = tissue[np.flatnonzero(tissue.obs.x.to_numpy() < 10)[::-1]].copy()
    # Deliberately keep unused labels, as AnnData slicing may prune categories.
    for data, key in ((tissue, "tissue_input"), (lesion, "lesion_input")):
        data.obs["batch"] = data.obs.batch.cat.add_categories(["unused"]) if "unused" not in data.obs.batch.cat.categories else data.obs.batch
        data.write_h5ad(config[key])
    config["exclude_slides"] = {name: "Manual synthetic QC" for name in "ace"}
    config["report"].update(pixels_per_slide=5, max_peaks=3)
    return config, dann


@pytest.mark.parametrize("interleaved", [False, True])
def test_export_removes_first_middle_last_slides_and_aligns_metadata(tmp_path, interleaved):
    config, _ = cohort_fixture(tmp_path, interleaved)
    fitted = fit(config)
    manifest, state = load_fit(fitted)
    apply(fitted, update_dann=True)
    root = Path(config["output"])
    assert manifest["schema"] == 3 and state["slides"].tolist() == ["b", "d"]
    for kind in ("lesion", "tissue"):
        raw = ad.read_h5ad(config[f"{kind}_input"])
        result = ad.read_h5ad(root / f"{kind}.h5ad")
        expected_rows = np.flatnonzero(raw.obs.batch.isin(["b", "d"]))
        np.testing.assert_array_equal(result.obs.msi_source_row, expected_rows)
        np.testing.assert_array_equal(result.obs_names, raw.obs_names[expected_rows])
        for key in ("hes_features", "spatial"):
            np.testing.assert_array_equal(result.obsm[key], raw.obsm[key][expected_rows])
        np.testing.assert_array_equal(result.obsp["dense"], raw.obsp["dense"][np.ix_(expected_rows, expected_rows)])
        np.testing.assert_array_equal(result.obsp["sparse"].toarray(), raw.obsp["sparse"][expected_rows][:, expected_rows].toarray())
        assert set(result.obs.batch.cat.categories) == set(result.obs.label.cat.categories) == {"b", "d"}
        assert set(result.uns["spatial"]) == {"b", "d"}
        aligned_export_rows(root / f"{kind}.h5ad", manifest, state, kind)
    assert verify(fitted)["all_retained_counts_match_sources"]
    records = pd.read_csv(root / "cohort.csv")
    assert records.included.tolist() == [False, True, False, True, False]


def test_excluded_measurements_cannot_change_any_retained_parameters(tmp_path):
    config, _ = cohort_fixture(tmp_path)
    _, before = load_fit(fit(config))
    for kind in ("lesion", "tissue"):
        path = config[f"{kind}_input"]
        data = ad.read_h5ad(path)
        values = data.layers["counts"].toarray()
        values[data.obs.batch.isin(list("ace"))] = np.arange(data.n_vars) + 1e6
        data.layers["counts"] = sparse.csr_matrix(values)
        data.write_h5ad(path)
    config["output"] = str(tmp_path / "refit")
    _, after = load_fit(fit(config))
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_raw_cleaned_heatmaps_use_original_rows_after_exclusion(tmp_path, monkeypatch):
    config, _ = cohort_fixture(tmp_path)
    fitted = fit(config)
    apply(fitted)
    manifest, state = load_fit(fitted)
    root = Path(config["output"])
    meta = render_heatmaps(root, manifest, state)
    pixels = pd.read_csv(root / "heatmap_pixels.csv")
    peaks = pd.read_csv(root / "heatmap_peaks.csv")
    raw = ad.read_h5ad(config["lesion_input"])
    cleaned = ad.read_h5ad(root / "lesion.h5ad")
    np.testing.assert_array_equal(raw.obs_names[pixels.source_row], cleaned.obs_names[pixels.export_row])
    assert not np.array_equal(pixels.source_row, pixels.export_row)
    for name, matrix in (("counts", raw.layers["counts"][pixels.source_row][:, peaks.original_index]),
                         ("old_X", raw.X[pixels.source_row][:, peaks.original_index]),
                         ("cleaned", cleaned.X[pixels.export_row][:, peaks.retained_index])):
        upper = meta["stages"][name]["upper_linear"]
        expected = colormaps["turbo"](Normalize(0, np.log1p(upper))(np.log1p(matrix.toarray())), bytes=True)
        with Image.open(root / "figures" / f"heatmap_{name}_pixels.png") as image:
            np.testing.assert_array_equal(image, expected)
    from msi_preprocess import report
    raw_tissue = ad.read_h5ad(config["tissue_input"])
    exported_tissue = ad.read_h5ad(root / "tissue.h5ad")
    seen = []
    def check_map(path, slide, coordinates, names, reasons, matrices):
        rows = np.flatnonzero(exported_tissue.obs.batch.astype(str) == slide)
        source = exported_tissue.obs.msi_source_row.to_numpy()[rows]
        columns = raw_tissue.var_names.get_indexer(names)
        np.testing.assert_array_equal(coordinates, raw_tissue.obs.iloc[source][["x", "y"]])
        np.testing.assert_array_equal(matrices["counts"], raw_tissue.layers["counts"][source][:, columns].toarray())
        np.testing.assert_array_equal(matrices["old_X"], raw_tissue.X[source][:, columns].toarray())
        seen.append(slide)
    monkeypatch.setattr(report, "plot_maps", check_map)
    report.render_maps(fitted)
    assert set(seen) == {"b", "d"}


def test_invalid_exclusions_and_empty_cohort(tmp_path):
    config, _ = cohort_fixture(tmp_path)
    lesion, tissue = [ad.read_h5ad(config[f"{kind}_input"]).obs for kind in ("lesion", "tissue")]
    for bad in ([], {"a": ""}, {1: "bad"}):
        config["exclude_slides"] = bad
        with pytest.raises(ValueError, match="exclude_slides"):
            validate_config(config)
    config["exclude_slides"] = {"missing": "QC"}
    with pytest.raises(ValueError, match="absent from lesion"):
        decide_cohort(lesion, tissue, config)
    config["exclude_slides"] = {"a": "QC"}
    with pytest.raises(ValueError, match="absent from tissue"):
        decide_cohort(lesion, tissue[tissue.batch != "a"], config)
    config["exclude_slides"] = dict.fromkeys("abcde", "QC")
    with pytest.raises(ValueError, match="No slides remain"):
        decide_cohort(lesion, tissue, config)


def test_dann_domains_splits_and_artifact_provenance(tmp_path):
    from dann.config import load_config
    from dann.data_loader import create_data_bundle
    from dann.preprocessing import artifact_preprocessing, parquet_preprocessing, validate_artifact_preprocessing, validate_preprocessing
    from dann.tiles import row_identity
    config, _ = cohort_fixture(tmp_path)
    fitted = fit(config)
    apply(fitted, update_dann=True)
    dann = load_config(Path(config["dann_config"]))
    path = Path(dann["data"]["path"])
    bundle = create_data_bundle(dann)
    assert bundle.metadata.batch_names == ("d", "b")
    assert len(bundle.batch_class_weights) == 2
    assert set(bundle.metadata.batch_codes) == {0, 1}
    for slide in (0, 1):
        count = np.sum(bundle.metadata.batch_codes == slide)
        assert np.sum(bundle.metadata.batch_codes[bundle.split_indices["train"]] == slide) == int(.9 * count)
    with pytest.raises(ValueError, match="matching preprocessing"):
        create_data_bundle(dann, bundle.split_indices)
    stale = row_identity(path, dann["data"])
    stale["preprocessing_fingerprint"] = "stale"
    with pytest.raises(ValueError, match="identities"):
        create_data_bundle(dann, bundle.split_indices, stale)
    provenance = artifact_preprocessing(path, dann["data"])
    validate_artifact_preprocessing(path, parquet_preprocessing(provenance))
    # Identical names, values and dimensions still require matching provenance.
    legacy = ad.read_h5ad(path)
    del legacy.uns["msi_preprocess"]
    legacy_path = tmp_path / "legacy_same_rows.h5ad"
    legacy.write_h5ad(legacy_path)
    with pytest.raises(ValueError, match="Artifact preprocessing/cohort"):
        validate_artifact_preprocessing(legacy_path, parquet_preprocessing(provenance))
    from dann.preprocessing import analysis_provenance
    sidecar = tmp_path / "saved_analysis"
    sidecar.mkdir()
    (sidecar / "preprocessing.json").write_text(json.dumps(dict(preprocessing=provenance)))
    legacy_data = dict(dann["data"], path=str(legacy_path), preprocessing_fingerprint=None)
    with pytest.raises(ValueError, match="legacy input"):
        analysis_provenance(sidecar, legacy_data)
    for wrong in ({}, {**provenance, "preprocessing_fingerprint": "stale"}, {**provenance, "retained_slides": ["a", "b"]}):
        with pytest.raises(ValueError, match="Artifact preprocessing/cohort"):
            validate_artifact_preprocessing(path, parquet_preprocessing(wrong))
    for dataset in bundle.datasets.values():
        dataset.close()
    manifest_text = Path(fitted).read_text()
    changed = json.loads(manifest_text)
    changed["cohort"]["decisions"][0]["reason"] = "different decision"
    Path(fitted).write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_fit(fitted)
    Path(fitted).write_text(manifest_text)
    with h5py.File(path, "r+") as handle:
        values = handle["obs/msi_source_row"][:]
        handle["obs/msi_source_row"][:] = values[::-1]
    with pytest.raises(ValueError, match="source-row"):
        validate_preprocessing(path, dann["data"])

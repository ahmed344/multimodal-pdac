"""Numerical, frozen-fit, sparse-export and DANN transformation contracts."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import yaml

from dann.config import read_config
from msi_preprocess.apply import apply, clean_slide, update_dann_config
from msi_preprocess.config import load_config, validate_config
from msi_preprocess.fit import fit, integration_parameters, load_fit, selection_decisions
from msi_preprocess.io import axes, check_disk, match_rows
from msi_preprocess.spatial import neighborhoods, reference_envelopes, removal_mask, supported_patches
from msi_preprocess.verify import verify


def defaults() -> dict:
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    config["exclude_slides"] = {}
    return config


def fixture(tmp_path: Path, *, interleaved=True) -> tuple[dict, dict]:
    rng = np.random.default_rng(5)
    x, y = np.meshgrid(np.arange(12), np.arange(12))
    x, y = x.ravel(), y.ravel()
    n = len(x)
    values = np.zeros((n * 2, 8), dtype=np.float32)
    for i in range(2):
        for j in range(4):
            values[i*n:(i+1)*n, j] = ((x >= 3) & (x <= 8)) * (j+1) * (i+1)
        values[i*n:(i+1)*n, 4] = ((y >= 4) & (y <= 6) & (x > 4)) * .15
        # Repeated weak isolated detections: adequate global assessment, no
        # adjacent supported patch, so selection must actually subset features.
        values[i*n:(i+1)*n, 5] = ((x % 3 == 0) & (y % 3 == 0)) * .08
        values[i*n+5, 6] = 5
    obs = pd.DataFrame(dict(batch=pd.Categorical(np.repeat(["b", "a"], n), categories=["b", "a"]),
                            patient=np.repeat(["p1", "p2"], n), x=np.tile(x, 2), y=np.tile(y, 2),
                            Lesion=True, Density_CD8=rng.uniform(0, .1, 2*n),
                            Density_Stroma=rng.uniform(0, .5, 2*n), Density_Tumor=.1, Density_Collagen=.2),
                       index=[f"cell_{i}" for i in range(2*n)])
    tissue = ad.AnnData(sparse.csr_matrix(values*3), obs=obs,
                        var=pd.DataFrame(index=[str(500+j) for j in range(8)]))
    tissue.layers["counts"] = sparse.csr_matrix(values)
    tissue.obsm["hes_features"] = rng.normal(size=(2*n, 3)).astype(np.float32)
    tissue.uns["original"] = "preserved"
    tissue.varm["annotation"] = np.arange(16).reshape(8, 2)
    tissue.varp["pairs"] = sparse.eye(8, format="csr")
    if interleaved:
        tissue = tissue[rng.permutation(2*n)].copy()
    lesion = tissue[rng.choice(2*n, 240, replace=False)].copy()
    tissue.write_h5ad(tmp_path / "tissue.h5ad")
    lesion.write_h5ad(tmp_path / "lesion.h5ad")
    dann = read_config(Path("dann/config.yaml"))
    dann["data"].pop("preprocessing_fingerprint", None)
    dann["data"].update(path=str(tmp_path / "lesion.h5ad"), context_path=str(tmp_path / "tissue.h5ad"),
                         train_fraction=.9, validation_fraction=.1, test_fraction=0, matrix_key="X", intensity_transform="log1p")
    dann["training"]["seed"] = 57
    dann["training"]["resume_checkpoint"] = None
    dann["model"]["num_peaks"] = 8
    dann["model"]["heads"]["he"]["enabled"] = False
    dann["performance"] = {}
    dann["training"]["pixel"].update(num_workers=0, batch_size=4, validation_batch_size=4)
    (tmp_path / "dann.yaml").write_text(yaml.safe_dump(dann, sort_keys=False))
    config = defaults()
    config.update(lesion_input=str(tmp_path / "lesion.h5ad"), tissue_input=str(tmp_path / "tissue.h5ad"),
                  output=str(tmp_path / "output"), dann_config=str(tmp_path / "dann.yaml"),
                  feature_block=3, block_rows=31, report_peaks=1)
    config["selection"].update(min_spatial_detections=5, shuffles=3, spatial_pairs=300)
    config["integration"]["min_anchors"] = 2
    config["local"]["draws"] = 128
    return config, dann


def test_local_structures_noise_isolated_constant_and_gaps():
    rng = np.random.default_rng(2)
    x, y = np.meshgrid(np.arange(40), np.arange(40))
    x, y = np.r_[x.ravel(), 100], np.r_[y.ravel(), 100]
    values = np.zeros((len(x), 5))
    region = (x >= 10) & (x < 18) & (y >= 10) & (y < 18)
    values[region, 0] = .2
    values[:, 1] = (rng.random(len(x)) < .025) * .1
    values[41, 2] = 5
    values[:, 3] = .2
    values[-1, 4] = .1
    graph = neighborhoods(x, y)
    env = reference_envelopes(values, 1024, .95, 42)
    remove = removal_mask(values, graph, env, .5, 2)
    assert not remove[region, 0].any()
    assert remove[:, 1].sum() > (values[:, 1] > 0).sum() * .6
    assert not remove[41, 2]
    assert not remove[:, 3].any()
    assert not remove[-1, 4]
    np.testing.assert_array_equal(env, reference_envelopes(values, 1024, .95, 42))
    for j in range(values.shape[1]):
        np.testing.assert_array_equal(env[:, :, j:j+1], reference_envelopes(values[:, j:j+1], 1024, .95, 42))
    one, two = neighborhoods(np.array([0, 2, 0, 0]), np.array([0, 0, 1, 2]))
    assert one[0, 1] == two[0, 1] == 0  # Cannot jump a missing intermediate pixel.
    assert one[0, 3] == 0 and two[0, 3] == 1
    assert np.all(two.diagonal() == 0)
    with pytest.raises(ValueError, match="Duplicate"):
        neighborhoods(np.array([0, 0]), np.array([0, 0]))


def test_four_connected_supported_pixels_veto_peak_removal():
    x, y = np.meshgrid(np.arange(20), np.arange(20))
    x, y = x.ravel(), y.ravel()
    values = np.zeros((len(x), 1))
    values[(x >= 8) & (x <= 9) & (y >= 8) & (y <= 9)] = .1
    env = reference_envelopes(values, 1024, .95, 42)
    assert supported_patches(values, neighborhoods(x, y), env, 4, 2)[0]
    tested = np.ones((4, 5), bool)
    tested[:, 4] = False
    weak = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], bool)
    support = np.zeros_like(tested)
    patch = np.zeros_like(tested)
    support[3, 1] = patch[3, 2] = True
    keep, reasons = selection_decisions(tested, weak, support, patch, True)
    assert keep.tolist() == [1, 2, 3, 4]
    assert reasons[0] == "weak_unstructured" and reasons[4] == "insufficient_assessment"


def test_patient_balance_known_multipliers_and_limits():
    base = np.arange(1, 41.)
    medians = np.array([base, base, 2*base])
    cfg = defaults()["integration"]
    params = integration_parameters(medians, np.ones_like(medians), np.ones_like(medians, bool),
                                    np.array(["a", "a", "b"]), np.arange(40), cfg, True)
    np.testing.assert_allclose(params["reference"], 1.5 * base)
    np.testing.assert_allclose(params["multipliers"], [2/3, 2/3, 4/3])
    np.testing.assert_allclose(medians / params["multipliers"][:, None], np.tile(1.5*base, (3, 1)))
    medians[-1] *= 10
    params = integration_parameters(medians, np.ones_like(medians), np.ones_like(medians, bool),
                                    np.array(["a", "a", "b"]), np.arange(40), cfg, True)
    assert params["multipliers"][0] == .5 and params["correction_limited"][0]
    with pytest.raises(ValueError, match="usable anchors"):
        integration_parameters(medians[:, :2], np.ones((3, 2)), np.ones((3, 2), bool),
                               np.array(["a", "a", "b"]), np.arange(2), cfg, True)


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=3)))
def test_all_stage_combinations_exports_linear_and_exact(tmp_path, flags):
    config, dann = fixture(tmp_path, interleaved=not all(flags))
    config["stages"] = dict(zip(("selection", "integration", "local"), flags))
    originals = {key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest() for key in ("lesion_input", "tissue_input")}
    fitted = fit(config)
    manifest, state = load_fit(fitted)
    if flags[0]:
        from msi_preprocess.report import example_peaks
        examples, reasons = example_peaks(state, str(state["slides"][0]), 6)
        assert 5 in examples and "removed peak" in reasons
    apply(fitted, update_dann=True)
    out = Path(config["output"])
    tissue, lesion = ad.read_h5ad(out / "tissue.h5ad"), ad.read_h5ad(out / "lesion.h5ad")
    source = ad.read_h5ad(config["tissue_input"])
    source_lesion = ad.read_h5ad(config["lesion_input"])
    retained = state["retained"]
    if flags[0]:
        assert 5 not in retained
    else:
        assert len(retained) == 8
    assert sparse.isspmatrix_csr(tissue.X) and sparse.isspmatrix_csr(tissue.layers["counts"])
    np.testing.assert_array_equal(tissue.layers["counts"].toarray(), source.layers["counts"][:, retained].toarray())
    np.testing.assert_array_equal(tissue.var_names, source.var_names[retained])
    np.testing.assert_array_equal(lesion.obs_names, source_lesion.obs_names)
    np.testing.assert_array_equal(tissue.obs_names, source.obs_names)
    np.testing.assert_array_equal(lesion.obsm["hes_features"], source_lesion.obsm["hes_features"])
    assert tissue.uns["original"] == "preserved"
    mapping = match_rows(lesion.obs, tissue.obs, dann["data"])
    np.testing.assert_array_equal(lesion.X.toarray(), tissue.X[mapping].toarray())
    np.testing.assert_array_equal(lesion.layers["counts"].toarray(), tissue.layers["counts"][mapping].toarray())
    expected = tissue.layers["counts"].toarray().astype(float)
    for i, slide in enumerate(state["slides"]):
        expected[tissue.obs.batch == slide] /= state["multipliers"][i]
    actual = tissue.X.toarray()
    np.testing.assert_allclose(actual[actual > 0], expected[actual > 0], rtol=1e-7)
    assert not np.any(actual[expected == 0])
    if not flags[2]:
        np.testing.assert_allclose(actual, expected, rtol=1e-7)
    if not any(flags):
        np.testing.assert_array_equal(actual, source.layers["counts"].toarray())
    for key, sha in originals.items():
        assert hashlib.sha256(Path(config[key]).read_bytes()).hexdigest() == sha
    updated = yaml.safe_load(Path(config["dann_config"]).read_text())
    for group in dann:
        for key in dann[group]:
            if (group, key) not in {( "data", "path"), ("data", "context_path"), ("model", "num_peaks"), ("results", "run_name")}:
                assert updated[group][key] == dann[group][key]
    assert updated["data"]["preprocessing_fingerprint"] == manifest["fingerprint"]
    from dann.config import resolve_execution
    from dann.data_loader import create_data_bundle
    bundle = create_data_bundle(resolve_execution(updated))
    dataset = bundle.datasets["train"]
    for k in range(min(20, len(dataset))):
        sample = dataset[k]
        row = bundle.split_indices["train"][k]
        raw = lesion.X[row]
        np.testing.assert_array_equal(sample["peak_indices"], raw.indices)
        np.testing.assert_allclose(sample["intensities"], np.log1p(raw.data), rtol=1e-7)
    for dataset in bundle.datasets.values():
        dataset.close()
    assert (out / "report.html").exists()
    assert (out / "local_sensitivity.csv").exists()
    assert tissue.uns["msi_preprocess"]["intensity_scale"] == "linear"
    assert verify(fitted, 8)["dann_log1p_exactly_once"]


def test_explicit_sparse_zeros_and_duplicates():
    from msi_preprocess.qc import canonical_block, column_correlations
    matrix = sparse.csr_matrix(([.2, 0, .3], [0, 1, 0], [0, 3, 3]), shape=(2, 3))
    counts = canonical_block(matrix)
    np.testing.assert_array_equal(counts.toarray(), [[.5, 0, 0], [0, 0, 0]])
    assert counts.nnz == 1
    assert np.isnan(column_correlations(sparse.csr_matrix((3, 2)), sparse.csr_matrix((3, 2)))).all()


def test_stream_writer_exceeds_int32_nonzero_offset(tmp_path):
    from msi_preprocess.io import CSRWriter
    offset = 2**31 + 100
    path = tmp_path / "large_logical_csr.h5"
    with h5py.File(path, "w") as h:
        writer = CSRWriter(h, "X", (3, 3), np.float32)
        # HDF5 chunked extents simulate a previously written huge prefix without
        # allocating billions of entries on disk or in memory.
        writer.data.resize(offset, axis=0)
        writer.indices.resize(offset, axis=0)
        writer.rows, writer.nnz = 1, offset
        writer.ptr[1] = offset
        block = sparse.csr_matrix([[.2, 0, 3], [0, 0, 0]], dtype=np.float32)
        assert block.indptr.dtype == np.int32
        writer.append(block)
        np.testing.assert_array_equal(writer.ptr[1:], [offset, offset+2, offset+2])
        np.testing.assert_array_equal(writer.data[offset:], block.data)
        assert writer.ptr.dtype == np.int64
    assert path.stat().st_size < 10_000_000


def test_local_feature_workers_are_exactly_equivalent():
    cfg = defaults()
    cfg["feature_block"] = 2
    x, y = np.meshgrid(np.arange(12), np.arange(12))
    graph = neighborhoods(x.ravel(), y.ravel())
    rng = np.random.default_rng(5)
    values = rng.exponential(.2, (x.size, 9))
    values[rng.random(values.shape) < .6] = 0
    env = reference_envelopes(values, 1024, .95, 42)
    outputs = []
    for workers in (1, 4):
        cfg["application_workers"] = workers
        outputs.append(clean_slide(sparse.csr_matrix(values), np.arange(9), 1., graph, env, cfg))
    for left, right in zip(*outputs):
        np.testing.assert_array_equal(left.toarray() if sparse.issparse(left) else left,
                                      right.toarray() if sparse.issparse(right) else right)


def test_full_lesion_fitting_is_split_independent_and_label_free(tmp_path):
    config, dann = fixture(tmp_path)
    fitted = fit(config)
    first, a = load_fit(fitted)
    lesion = ad.read_h5ad(config["lesion_input"])
    assert first["fitting"]["scope"] == "retained_lesion"
    np.testing.assert_array_equal(a["fitting_rows"], np.arange(lesion.n_obs))
    # Model split settings and smoke caps cannot affect any fitted parameter.
    dann["data"].update(train_fraction=.5, validation_fraction=.5)
    dann["training"]["seed"] = 999
    Path(config["dann_config"]).write_text(yaml.safe_dump(dann))
    config["output"] = str(tmp_path / "split_changed")
    _, b = load_fit(fit(config))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    lesion.obs["Density_CD8"] = 999
    lesion.obsm["hes_features"][:] = -999
    lesion.write_h5ad(config["lesion_input"])
    config["output"] = str(tmp_path / "labels_changed")
    _, b = load_fit(fit(config))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    # Previously held-out rows now contribute to the fit.
    from dann.data_loader import stratified_split_indices
    rows = stratified_split_indices(lesion.obs.batch.cat.codes.to_numpy(),
                train_fraction=.9, validation_fraction=.1, test_fraction=0, seed=57)["validation"]
    counts = lesion.layers["counts"].toarray()
    counts[rows] = 777
    lesion.layers["counts"] = sparse.csr_matrix(counts)
    lesion.write_h5ad(config["lesion_input"])
    config["output"] = str(tmp_path / "counts_changed")
    _, b = load_fit(fit(config))
    assert not np.array_equal(a["prevalence"], b["prevalence"])
    with pytest.raises(ValueError, match="changed since fitting"):
        apply(fitted)


def test_fingerprint_split_and_resume_guards(tmp_path):
    from dann.preprocessing import validate_preprocessing
    from dann.train import resume_training_config
    config, dann = fixture(tmp_path)
    fitted = fit(config)
    apply(fitted, update_dann=True)
    updated = yaml.safe_load(Path(config["dann_config"]).read_text())
    path = Path(updated["data"]["path"])
    with pytest.raises(ValueError, match="fresh training"):
        resume_training_config(dann, updated)
    bad = copy.deepcopy(updated["data"])
    bad["preprocessing_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_preprocessing(path, bad)
    bad = copy.deepcopy(updated["data"])
    bad["intensity_transform"] = "none"
    with pytest.raises(ValueError, match="exactly one"):
        validate_preprocessing(path, bad)
    assert validate_preprocessing(path, updated["data"], training_rows=np.arange(3))["fitting"]["scope"] == "retained_lesion"
    text = Path(fitted).read_text()
    changed = json.loads(text)
    changed["config"]["local"]["weak_boundary"] = 1.0
    Path(fitted).write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_fit(fitted)
    Path(fitted).write_text(text)
    state = Path(fitted).with_name("fit.npz")
    state.write_bytes(state.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_fit(fitted)


@pytest.mark.parametrize("group,key,value", [("stages", "local", "true"), ("local", "draws", 0),
        ("local", "sensitivity", [.5, .1]), ("integration", "prevalence", 2),
        ("selection", "spatial_weak_excess", .2), ("selection", "patch_pixels", 3),
        ("report", "max_peaks", 0), ("report", "max_peaks", True),
        ("report", "max_peaks", 2.5)])
def test_invalid_settings(group, key, value):
    cfg = defaults()
    cfg[group][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_disk_preflight_and_duplicate_keys(tmp_path, monkeypatch):
    config, dann = fixture(tmp_path)
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda path: type("Usage", (), {"free": 1})())
    with pytest.raises(OSError, match="Insufficient disk"):
        check_disk([config["tissue_input"]], tmp_path, 1)
    tissue = ad.read_h5ad(config["tissue_input"])
    tissue.obs.iloc[1, tissue.obs.columns.get_indexer(["batch", "x", "y"])] = tissue.obs.iloc[0][["batch", "x", "y"]].to_numpy()
    tissue.write_h5ad(config["tissue_input"])
    with pytest.raises(ValueError, match="Duplicate"):
        axes(config["tissue_input"], dann["data"], "patient")

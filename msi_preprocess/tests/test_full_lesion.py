"""Full-lesion fitting, numerical reporting, and transactional replacement tests."""
from __future__ import annotations

import copy
import hashlib
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

from msi_preprocess.tests.test_pipeline import defaults, fixture
from msi_preprocess.fit import fit, load_fit, integration_parameters
from msi_preprocess.apply import apply, clean_slide
from msi_preprocess.comparison import (
    RegionDiagnostics, sample_pixels, rank_retained_peaks, cluster_peaks,
    sampled_stages, render_report, tic_report_section,
)
from msi_preprocess.spatial import neighborhoods, reference_envelopes
from msi_preprocess.tic import render_tic_maps


def test_asymmetric_limits_and_disabled_integration():
    config = defaults()["integration"]
    base = np.arange(1., 41.)
    matrix = np.array([.1*base, base, 20*base])
    args = (matrix, np.ones_like(matrix), np.ones_like(matrix, bool), np.array(["a", "b", "c"]), np.arange(40), config)
    state = integration_parameters(*args, True)
    np.testing.assert_allclose(state["raw_multipliers"], [.1, 1, 20])
    np.testing.assert_array_equal(state["multipliers"], [.5, 1, 2.5])
    np.testing.assert_array_equal(state["correction_limited"], [True, False, True])
    np.testing.assert_array_equal(integration_parameters(*args, False)["multipliers"], np.ones(3))


def test_nonlesion_counts_never_enter_fitting(tmp_path):
    config, _ = fixture(tmp_path)
    _, first = load_fit(fit(config))
    lesion = ad.read_h5ad(config["lesion_input"])
    tissue = ad.read_h5ad(config["tissue_input"])
    mask = ~tissue.obs_names.isin(lesion.obs_names)
    values = tissue.layers["counts"].toarray()
    values[mask] = 999
    tissue.layers["counts"] = sparse.csr_matrix(values)
    tissue.write_h5ad(config["tissue_input"])
    config["output"] = str(tmp_path / "different_tissue")
    _, second = load_fit(fit(config))
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_tissue_neighbors_change_application_not_frozen_references():
    cfg = defaults()
    x, y = np.meshgrid(np.arange(8), np.arange(8))
    x, y = x.ravel(), y.ravel()
    graph = neighborhoods(x, y)
    values = np.zeros((len(x), 1))
    center = np.flatnonzero((x == 3) & (y == 3))[0]
    values[center] = .1
    env = reference_envelopes(values, 1024, .95, 42)
    frozen = env.copy()
    isolated = clean_slide(sparse.csr_matrix(values), np.array([0]), 1, graph, env, cfg, np.array([center]))
    values[(x >= 2) & (x <= 4) & (y >= 2) & (y <= 4)] = .1
    coherent = clean_slide(sparse.csr_matrix(values), np.array([0]), 1, graph, env, cfg, np.array([center]))
    assert isolated[2][center, 0] == 0 and coherent[2][center, 0] == .1
    assert isolated[-1][3, 0] == 1 and coherent[-1][3, 0] == 0
    np.testing.assert_array_equal(frozen, env)


def test_region_denominators_and_inclusive_thresholds(tmp_path):
    values = np.array([[0, .1, .5], [1, 2, 0]], dtype=float)
    cleaned = values.copy()
    cleaned[0, 1] = 0
    matrices = {key: sparse.csr_matrix(values) for key in ("counts", "selected", "integrated")}
    matrices["cleaned"] = sparse.csr_matrix(cleaned)
    diagnostics = RegionDiagnostics(tmp_path, np.array(["a", "b", "c"]), np.arange(3), [.1, .5, 1.])
    diagnostics.add("a", matrices, np.array([0]))
    diagnostics.finish()
    stats = pd.read_csv(tmp_path / "removal_summary.csv")
    whole = stats[(stats.region == "whole_tissue") & (stats.slide == "ALL")].iloc[0]
    lesion = stats[(stats.region == "lesion") & (stats.slide == "ALL")].iloc[0]
    assert whole.positive_before == 4 and whole.measurements_removed == 1
    assert whole.fraction_positive_removed == .25
    assert whole.fraction_all_removed == pytest.approx(1/6)
    assert lesion.fraction_positive_removed == .5
    table = pd.read_csv(tmp_path / "threshold_placement.csv")
    row = table[(table.region == "whole_tissue") & (table.stage == "integrated") & (table.boundary == .5)].iloc[0]
    assert row.positive_below == 2 and row.fraction_all_at_or_below == pytest.approx(4/6)
    assert row.intensity_below == pytest.approx(.6)


def test_pixel_sampling_clustering_and_retained_stage_mapping():
    obs = pd.DataFrame(dict(batch=["b"]*5+["a"]*7, x=list(range(5))+list(range(7)), y=0))
    data = defaults()["coordinate_columns"]
    rows = sample_pixels(obs, data, 4, 42)
    np.testing.assert_array_equal(rows, sample_pixels(obs, data, 4, 42))
    assert obs.iloc[rows].groupby("batch").size().tolist() == [4, 4]
    for _, group in obs.iloc[rows].groupby("batch"):
        assert group.x.is_monotonic_increasing
    counts = np.array([[0, 1, 2, 4], [0, 2, 3, 4], [0, 3, 1, 4]], dtype=float)
    order, tree, constant = cluster_peaks(counts)
    assert constant.tolist() == [0, 3] and order[-2:].tolist() == [0, 3]
    np.testing.assert_array_equal(order, cluster_peaks(counts)[0])
    retained, peaks = np.array([3, 0, 1]), np.array([1, 3])
    factors = np.array([1., 2., 4.])
    cleaned = counts[:, retained]/factors[:, None]
    cleaned[0, 2] = 0
    stages = sampled_stages(counts, cleaned, retained, factors, peaks)
    for name in ("counts", "selected"):
        np.testing.assert_array_equal(stages[name], counts[:, peaks])
    np.testing.assert_array_equal(stages["integrated"], counts[:, peaks]/factors[:, None])
    np.testing.assert_array_equal(stages["cleaned"], cleaned[:, [2, 0]])
    assert all(values.shape == (3, 2) for values in stages.values())
    with pytest.raises(ValueError, match="retained subset"):
        sampled_stages(counts, cleaned, retained, factors, np.array([2]))


@pytest.mark.parametrize("available", [1, 4, 502])
def test_retained_peak_ranking_ties_and_display_limit(available):
    counts = np.tile([0., 1., 0., 1.], (available+1, 1)).T
    counts[:, 0] *= 1000  # A removed peak must never compete for the display.
    retained = np.arange(1, available+1)[::-1]
    ranked, variances = rank_retained_peaks(counts, retained)
    np.testing.assert_array_equal(ranked, np.arange(1, min(500, available)+1))
    np.testing.assert_allclose(variances, np.log(2)**2/4)
    np.testing.assert_array_equal(ranked, rank_retained_peaks(counts, retained)[0])


def test_retained_peak_variance_includes_zeros_and_precedes_clustering():
    counts = np.array([[0., 0., 1., 0.], [1000., 1., 1., 3.],
                       [0., 0., 1., 0.], [1000., 1., 1., 3.]])
    ranked, variances = rank_retained_peaks(counts, np.array([1, 2, 3]), limit=2)
    np.testing.assert_array_equal(ranked, [3, 1])
    np.testing.assert_allclose(variances, np.array([np.log(4)**2, np.log(2)**2])/4)
    assert variances.dtype == np.float64


def test_complete_report_can_be_regenerated(tmp_path):
    config, _ = fixture(tmp_path)
    config["report"]["pixels_per_slide"] = 8
    config["report"]["max_peaks"] = 3
    fitted = fit(config)
    apply(fitted)
    manifest, state = load_fit(fitted)
    output = Path(fitted).parent
    protected = [output / name for name in ("fit.json", "fit.npz", "lesion.h5ad", "tissue.h5ad")]
    protected.append(Path(config["dann_config"]))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    summary_path = output / "validation_summary.json"
    preserved = dict(cohort={"slides": 2}, promotion={"passed": True}, tests={"passed": True})
    summary_path.write_text(json.dumps(dict(fingerprint=manifest["fingerprint"],
                                           retained_features=-1, **preserved)))
    tic = render_tic_maps(fitted)
    report = render_report(fitted)
    output = report.parent
    text = report.read_text()
    assert "Every row is one pixel" in text and "whole-tissue" in text
    assert "3 most-variable retained peaks" in text and "Turbo" in text
    assert "2 slides are clustered as independent observations" in text
    assert "Dendrograms are hidden" in text and "heatmap_slide_linkage.npy" in text
    assert "Whole-tissue spatial TIC" in text
    assert "tic_counts_all_slides.png" in text and "tic_cleaned_all_slides.png" in text
    assert tic_report_section(output, "different-fingerprint") == ""
    tissue = ad.read_h5ad(output / "tissue.h5ad")
    assert tic["pixels"] == tissue.n_obs == 288
    assert tic["retained_features"] == tissue.n_vars == len(state["retained"]) > 3
    assert tic["slide_order"] == ["a", "b"]
    counts_tic = np.asarray(tissue.layers["counts"].sum(axis=1, dtype=np.float64)).ravel()
    assert tic["upper_linear"] == pytest.approx(np.percentile(counts_tic[counts_tic > 0], 99.5))
    for stage, matrix in (("counts", tissue.layers["counts"]), ("cleaned", tissue.X)):
        statistics = tic["stages"][stage]
        assert statistics["upper_linear"] == tic["upper_linear"]
        for slide in statistics["slides"]:
            rows = np.flatnonzero(tissue.obs.batch == slide["slide"])
            expected = np.asarray(matrix[rows].sum(axis=1, dtype=np.float64)).ravel()
            assert slide["pixels"] == len(rows)
            assert slide["intensity_sum"] == pytest.approx(expected.sum())
            assert slide["zero_pixels"] == np.count_nonzero(expected == 0)
            assert slide["saturated_pixels"] == np.count_nonzero(expected > tic["upper_linear"])
        with Image.open(output / statistics["figure"]) as image:
            assert image.size == (5000, 750)
    assert "only DANN training rows" not in text
    summary = json.loads((output / "validation_summary.json").read_text())
    assert summary["fitting"]["scope"] == "retained_lesion"
    assert len(summary["removal"]) == 2
    assert summary["retained_features"] == len(state["retained"])
    assert {key: summary[key] for key in preserved} == preserved
    rows = pd.read_csv(output / "heatmap_pixels.csv")
    assert len(rows) == 16
    metadata = json.loads((output / "heatmap.json").read_text())
    assert metadata["raster_shape"] == [16, 3]
    assert metadata["retained_features"] == len(state["retained"]) > 3
    assert metadata["cmap"] == "turbo" and metadata["figure_inches"] == [14, 24]
    peaks = pd.read_csv(output / "heatmap_peaks.csv")
    original = peaks.original_index.to_numpy()
    source = ad.read_h5ad(config["lesion_input"])[rows.source_row].copy()
    counts = source.layers["counts"].toarray()
    expected_variances = np.log1p(counts.astype(np.float64)).var(axis=0)
    expected_rank = sorted(state["retained"], key=lambda j: (-expected_variances[j], j))[:3]
    assert peaks.sort_values("variance_rank").original_index.tolist() == expected_rank
    np.testing.assert_allclose(peaks.log1p_counts_variance, expected_variances[original])
    np.testing.assert_array_equal(state["retained"][peaks.retained_index], original)
    np.testing.assert_array_equal(peaks.cluster_position, np.arange(3))
    assert metadata["clustered_original_indices"] == original.tolist()
    cluster_input = metadata["linkage_input_original_indices"]
    np.testing.assert_allclose(np.load(output / "heatmap_linkage.npy"), cluster_peaks(counts[:, cluster_input])[1])
    assert original.tolist() == np.array(cluster_input)[cluster_peaks(counts[:, cluster_input])[0]].tolist()
    counts = counts[:, original]
    factors = source.obs.batch.astype(str).map(dict(zip(state["slides"], state["multipliers"]))).to_numpy()
    final = ad.read_h5ad(output / "lesion.h5ad")[rows.source_row, peaks.retained_index].X.toarray()
    values = dict(counts=counts, selected=counts, integrated=counts/factors[:, None],
                  cleaned=final, old_X=source.X[:, original].toarray())
    upper = np.percentile(counts[counts > 0], 99.5)
    assert metadata["upper_linear"] == pytest.approx(upper)
    for stage, matrix in values.items():
        vmax = np.percentile(matrix[matrix > 0], 99.5) if stage == "old_X" else upper
        assert metadata["stages"][stage]["upper_linear"] == pytest.approx(vmax)
        expected_pixels = colormaps["turbo"](Normalize(0, np.log1p(vmax))(np.log1p(matrix)), bytes=True)
        with Image.open(output / "figures" / f"heatmap_{stage}_pixels.png") as image:
            assert image.size == (3, 16)
            np.testing.assert_array_equal(np.asarray(image), expected_pixels)
        with Image.open(output / "figures" / f"heatmap_{stage}.png") as image:
            assert image.size == (2520, 4320)
    # Rerender from saved exact diagnostics without touching the fit or exports.
    render_report(fitted, heatmaps=False)
    assert "Whole-tissue spatial TIC" in report.read_text()
    assert {key: json.loads(summary_path.read_text())[key] for key in preserved} == preserved
    # Stale validation from another preprocessing generation must not survive.
    summary["fingerprint"] = "different-generation"
    summary_path.write_text(json.dumps(summary))
    render_report(fitted, heatmaps=False)
    assert not (preserved.keys() & json.loads(summary_path.read_text()).keys())
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected} == before


def test_legacy_split_contract_is_still_enforced(tmp_path):
    from dann.preprocessing import validate_preprocessing
    path = tmp_path / "legacy.h5ad"
    rows = np.arange(10, dtype=np.int64)
    manifest = dict(schema=1, split=dict(training_rows_sha256=hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()))
    with h5py.File(path, "w") as h:
        write_elem(h, "uns", dict(msi_preprocess=dict(fingerprint="legacy", intensity_scale="linear", manifest_json=json.dumps(manifest))))
    settings = dict(preprocessing_fingerprint="legacy", matrix_key="X", intensity_transform="log1p")
    validate_preprocessing(path, settings, training_rows=rows)
    with pytest.raises(ValueError, match="split differs"):
        validate_preprocessing(path, settings, training_rows=rows[:3])


@pytest.mark.parametrize("fail", [True, False])
def test_replacement_rolls_back_directory_and_configuration(tmp_path, monkeypatch, fail):
    import msi_preprocess.run as runner
    output, staging = tmp_path / "output", tmp_path / ".output.staging"
    output.mkdir(); staging.mkdir()
    (output / "fit.json").write_text('{"old": true}')
    (output / "tissue.h5ad").write_bytes(b"old")
    (staging / "tissue.h5ad").write_bytes(b"new")
    config = tmp_path / "dann.yaml"
    config.write_text("original")
    def update(path, *args):
        path.write_text("updated")
    def check(*args):
        if fail:
            raise ValueError("simulated contract failure")
    monkeypatch.setattr(runner, "update_dann_config", update)
    monkeypatch.setattr(runner, "check_promoted", check)
    if fail:
        with pytest.raises(ValueError, match="simulated"):
            runner.promote(staging, output, config, dict(fingerprint="new"))
        assert config.read_text() == "original"
        assert (output / "tissue.h5ad").read_bytes() == b"old"
        assert (staging / "tissue.h5ad").read_bytes() == b"new"
    else:
        runner.promote(staging, output, config, dict(fingerprint="new"))
        assert config.read_text() == "updated"
        assert (output / "tissue.h5ad").read_bytes() == b"new"
        assert (output / "previous_parameters_and_summaries.zip").exists()
        assert not staging.exists() and not (tmp_path / ".output.previous").exists()

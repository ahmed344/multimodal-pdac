"""Numerical, spatial, and artifact contracts for independent MSI QC."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import yaml

from msi_qc.analyze import run
from msi_qc.config import load_config
from msi_qc.core import (adjacent_pairs, canonical_block, column_correlations,
                         iter_blocks, read_element, robust_limits, row_sums,
                         select_threshold, spatial_diagnostics, threshold_histograms)


def defaults() -> dict:
    return load_config(Path(__file__).parents[1] / "config.yaml")


def test_coalesce_and_thresholds_match_dense_reference():
    raw = sparse.csr_matrix((np.array([.2, .3, 0., .5, 1., 2.]),
                             np.array([0, 0, 1, 2, 1, 2]), np.array([0, 4, 4, 6])), shape=(3, 4))
    matrix = canonical_block(raw)
    expected = np.array([[.5, 0., .5, 0.], [0., 0., 0., 0.], [0., 1., 2., 0.]])
    np.testing.assert_array_equal(matrix.toarray(), expected)
    thresholds = np.array([0., .25, .5, 1., 2.])
    counts, sums = threshold_histograms(matrix, thresholds)
    count_loss = counts.cumsum(axis=1)[:, :len(thresholds)]
    signal_loss = sums.cumsum(axis=1)[:, :len(thresholds)]
    for i, threshold in enumerate(thresholds):
        mask = (expected > 0) & (expected <= threshold)
        np.testing.assert_array_equal(count_loss[:, i], mask.sum(axis=0))
        np.testing.assert_allclose(signal_loss[:, i], np.where(mask, expected, 0).sum(axis=0))
    np.testing.assert_array_equal(row_sums(matrix.data <= .5, matrix.indptr), [2, 0, 0])
    assert np.all(np.diff(count_loss, axis=1) >= 0)


@pytest.mark.parametrize("bad", [-1., float("nan"), float("inf")])
def test_reject_invalid_intensities(bad):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        canonical_block(sparse.csr_matrix([[bad]]))


def test_grid_edges_do_not_cross_missing_pixels_or_duplicate_positions():
    x = np.array([0, 1, 0, 1, 10, 10])
    y = np.array([0, 0, 1, 1, 10, 11])
    pairs = adjacent_pairs(x, y, 99, np.random.default_rng(42))
    assert len(pairs) == 5
    assert np.all(abs(x[pairs[:, 0]] - x[pairs[:, 1]]) + abs(y[pairs[:, 0]] - y[pairs[:, 1]]) == 1)
    assert not np.any((pairs[:, 0] < 4) & (pairs[:, 1] >= 4))
    with pytest.raises(ValueError, match="Duplicate"):
        adjacent_pairs(np.array([0, 0]), np.array([0, 0]), 10, np.random.default_rng(42))


def test_sparse_correlations_include_zeros_and_constant_columns():
    a = np.array([[0, 2, 1], [1, 0, 1], [2, 4, 1], [3, 1, 1]], dtype=float)
    b = np.array([[1, 0, 1], [2, 1, 1], [3, 0, 1], [0, 4, 1]], dtype=float)
    actual = column_correlations(sparse.csr_matrix(a), sparse.csr_matrix(b))
    for j in (0, 1):
        assert actual[j] == pytest.approx(np.corrcoef(a[:, j], b[:, j])[0, 1])
    assert np.isnan(actual[2])


def test_spatial_structure_is_distinguished_from_shuffled_signal():
    rng = np.random.default_rng(7)
    x, y = np.meshgrid(np.arange(35), np.arange(35))
    x, y = x.ravel(), y.ravel()
    structure = ((x >= 8) & (x <= 25) & (y >= 8) & (y <= 25)).astype(float)
    signal = np.column_stack((structure, rng.permutation(structure), np.ones(len(x)), np.zeros(len(x))))
    matrix = sparse.csr_matrix(signal)
    pairs = adjacent_pairs(x, y, 5000, rng)
    config = defaults()
    config.update(shuffles=10, min_spatial_detections=100)
    result = spatial_diagnostics(matrix, pairs, signal.sum(axis=1), config, rng)
    assert result["spatial_supported"][0]
    assert not result["spatial_supported"][1]
    assert not result["spatial_tested"][2]
    assert not result["spatial_tested"][3]
    assert result["raw_correlation"][0] > .8
    assert abs(result["raw_correlation"][1]) < .1


def test_pixel_limits_zero_mad_and_outliers():
    limits = robust_limits(np.array([0., 0., 0.]), 3)
    assert limits["lower"] == 0 and not limits["enabled"]
    values = np.r_[np.linspace(80, 120, 200), 0., 10000.]
    limits = robust_limits(values, 3)
    assert 0 < limits["lower"] < 80
    assert 120 < limits["upper"] < 10000


def test_cutoff_must_meet_both_budgets_on_every_slide():
    thresholds = np.array([0., .1, .2])
    losses = [(np.array([0, .01, .04]), np.array([0, .001, .005])),
              (np.array([0, .02, .06]), np.array([0, .001, .002]))]
    assert select_threshold(losses, thresholds, .99, .95) == .1
    assert select_threshold([(np.array([0, .06, .2]), np.array([0, .001, .003]))], thresholds, .99, .95) == 0


@pytest.mark.parametrize("field,value", [("thresholds", [0, .5, .1]), ("thresholds", [.1, .5]),
                                        ("shuffles", 0), ("retained_signal_min", 1.2),
                                        ("weak_intensity_screen", 999), ("prevalence_screens", [0])])
def test_configuration_validation(tmp_path, field, value):
    config = defaults()
    config[field] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(path)


def write_fixture(path: Path) -> np.ndarray:
    rng = np.random.default_rng(42)
    x, y = np.meshgrid(np.arange(12), np.arange(12))
    x, y = x.ravel(), y.ravel()
    n = len(x)
    values = rng.exponential(.4, size=(n * 2, 7))
    values[rng.random(values.shape) < .35] = 0
    values[:, 0] = np.tile((x >= 4) & (x <= 8), 2) * .08
    values[:, 1] = np.tile((y >= 3) & (y <= 9), 2) * 3.
    values[:, 6] = 0.
    values[0] = 0.
    matrix = sparse.csr_matrix(values.astype(np.float32))
    string_dtype = h5py.string_dtype()
    with h5py.File(path, "w") as h:
        counts = h.create_group("layers/counts")
        counts.attrs["encoding-type"] = "csr_matrix"
        counts.attrs["shape"] = matrix.shape
        for key, array in (("data", matrix.data), ("indices", matrix.indices), ("indptr", matrix.indptr)):
            counts.create_dataset(key, data=array)
        obs = h.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.array([f"pixel_{i}" for i in range(n * 2)], dtype=object), dtype=string_dtype)
        for key, cats in (("batch", ["slide_a", "slide_b"]), ("patient", ["patient_a", "patient_b"])):
            group = obs.create_group(key)
            group.create_dataset("categories", data=np.asarray(cats, dtype=object), dtype=string_dtype)
            group.create_dataset("codes", data=np.repeat(np.arange(2), n))
        obs.create_dataset("x", data=np.tile(x, 2))
        obs.create_dataset("y", data=np.tile(y, 2))
        obs.create_dataset("Lesion", data=np.tile(x >= 6, 2))
        var = h.create_group("var")
        var.attrs["_index"] = "_index"
        index = var.create_group("_index")
        index.create_dataset("values", data=np.array([str(500.1 + i) for i in range(7)], dtype=object), dtype=string_dtype)
        index.create_dataset("mask", data=np.zeros(7, bool))
    return matrix.toarray()


def test_noncontiguous_sparse_reads_and_nullable_metadata(tmp_path):
    path = tmp_path / "input.h5ad"
    expected = write_fixture(path)
    rows = np.array([0, 1, 17, 18, 39, 45])
    with h5py.File(path, "r") as h:
        actual = sparse.vstack([b for _, _, b in iter_blocks(h["layers/counts"], rows, 3)])
        np.testing.assert_allclose(actual.toarray(), expected[rows])
        assert read_element(h["var/_index"])[0] == "500.1"
        assert read_element(h["obs/batch"], np.array([0, 144])).tolist() == ["slide_a", "slide_b"]


def test_full_workflow_exact_tables_artifacts_and_readonly_source(tmp_path):
    source = tmp_path / "input.h5ad"
    expected = write_fixture(source).astype(float)
    original = source.read_bytes()
    config = defaults()
    config.update(input=str(source), output=str(tmp_path / "output"), block_rows=31,
                  spatial_pairs=200, shuffles=3, min_spatial_detections=10,
                  example_slides=2, sample_values_per_slide=10000)
    report = run(config)
    assert report.exists()
    assert source.read_bytes() == original
    output = report.parent
    global_table = pd.read_csv(output / "threshold_tradeoffs.csv")
    for _, row in global_table.iterrows():
        mask = (expected > 0) & (expected <= row.threshold)
        assert row.measurements_removed == int(mask.sum())
        assert row.intensity_removed == pytest.approx(np.where(mask, expected, 0.).sum())
        assert row.pixels_empty_after == int(np.sum(np.max(expected, axis=1) <= row.threshold))
    region_table = pd.read_csv(output / "thresholds_by_slide_and_region.csv")
    for (_, threshold), frame in region_table.groupby(["slide", "threshold"]):
        all_rows = frame[frame.region == "all"].iloc[0]
        parts = frame[frame.region != "all"]
        assert parts.positive_measurements.sum() == all_rows.positive_measurements
        assert parts.measurements_removed.sum() == all_rows.measurements_removed
        assert parts.intensity_removed.sum() == pytest.approx(all_rows.intensity_removed)
    pixels = pd.read_parquet(output / "pixel_qc.parquet")
    np.testing.assert_allclose(pixels.summed_peak_intensity, expected.sum(axis=1))
    np.testing.assert_array_equal(pixels.detected_peaks, (expected > 0).sum(axis=1))
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["input_unchanged"]
    for href in re.findall(r'(?:href|src)="([^"]+)"', report.read_text()):
        if not href.startswith("http"):
            assert (output / href).exists(), href
    run(config, resume=True, render=False)
    assert source.read_bytes() == original
    changed = copy.deepcopy(config)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="Resume cache"):
        run(changed, resume=True, render=False)

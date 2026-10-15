"""Regression contracts for selective, parallel tissue/lesion heatmaps."""
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from PIL import Image

from dann import heatmap_execution as execution
from dann.config import spatial_heatmap_settings
from dann.spatial import build_output_table
from dann.spatial_heatmaps import run_spatial_heatmaps

DENSITY = "Density_Tumor"


@pytest.fixture
def inputs(tmp_path):
    # Lesion corners enclose non-lesion pixels and an unoccupied grid cell.
    obs = pd.DataFrame({"batch": pd.Categorical(["b"] * 5 + ["a"] * 2),
                        "x": [0, 1, 2, 0, 2, 0, 1], "y": [0, 0, 0, 1, 1, 0, 0],
                        "Lesion": [False, True, True, False, True, False, False],
                        DENSITY: np.array([.1, .2, 0, np.nan, .8, .2, .3], dtype=np.float32)},
                       index=["duplicate", "duplicate", "2", "3", "4", "5", "6"])
    data = ad.AnnData(obs=obs)
    data.obsm["spatial"] = data.obs[["x", "y"]].to_numpy(dtype=float) * 2
    data.obsm["hes_features"] = np.zeros((7, 8))
    data.uns["spatial"] = {batch: {"images": {"HES": np.zeros((8, 8, 3), dtype=np.uint8),
                                             "unrelated": np.ones((2, 2))},
                                  "scalefactors": {"tissue_HES_scalef": 1., "spot_diameter_fullres": 1.}}
                           for batch in ("a", "b")}
    source = tmp_path / "input.h5ad"
    data.write_h5ad(source)
    table = build_output_table(np.arange(7), np.asarray(data.obs_names),
                               np.arange(7, dtype=np.float32)[:, None],
                               np.full((7, 1), .25), np.ones((7, 1)), [DENSITY],
                               density_mean=np.linspace(0, .9, 7)[:, None])
    predictions = tmp_path / "predictions.parquet"
    pq.write_table(table, predictions)
    return source, predictions


def test_selective_loading_and_cached_grid(inputs, tmp_path, monkeypatch):
    actual = execution.read_elem
    accesses = []
    def read(item):
        accesses.append(item.name)
        return actual(item)
    monkeypatch.setattr(execution, "read_elem", read)
    settings = spatial_heatmap_settings()
    data = execution.load_plot_data(*inputs, [DENSITY], "batch", settings)
    assert data.X is None
    assert data.obs_names.tolist()[:2] == ["duplicate", "duplicate"]
    assert not any("hes_features" in path or "unrelated" in path or path.startswith("/X") for path in accesses)
    prepared = execution.prepare_grids(data, [DENSITY], 1e-4, "batch", settings, tmp_path)
    assert [slide["batch"] for slide in prepared["slides"]] == ["b", "a"]
    slide = prepared["slides"][0]
    grids = np.load(slide["grids"][DENSITY])
    np.testing.assert_array_equal(grids[1], [[0, 1, 2], [3, np.nan, 4]])
    assert slide["crop"] == [0, 2, 1, 3]
    cropped = execution.region_grid(grids, np.load(slide["lesion"]), slide["crop"], 1, "lesion")
    np.testing.assert_array_equal(cropped, [[1, 2], [np.nan, 4]])
    assert prepared["slides"][1]["crop"] is None
    # Observed zero/missing masks and predicted zero masks remain independent.
    assert np.isnan(grids[0][0, 2]) and np.isnan(grids[0][1, 0])
    assert np.isnan(grids[4][0, 0]) and np.isfinite(grids[4][0, 2])


def test_serial_parallel_identical_pngs(inputs, tmp_path):
    directories = []
    for workers in (1, 2):
        directory = tmp_path / f"workers-{workers}"
        run_spatial_heatmaps(*inputs, directory, [DENSITY], 1e-4, 8, workers=workers)
        directories.append(directory)
        names = {path.name for path in directory.rglob("*.png")}
        assert names == {f"spatial_heatmap_{prefix}{DENSITY}_{region}.png"
                         for prefix in ("", "diagnostics_") for region in ("tissue", "lesion")}
        assert not list(directory.glob(".heatmaps-*"))
        report = json.loads((directory / "heatmap_run.json").read_text())
        assert report["effective_workers"] == workers
        assert report["settings"]["cmap"] == "turbo"
        assert not list(directory.glob("*.png"))
        assert set(report["phase_seconds"]) == {"startup", "preparation", "rendering", "validation"}
    for path in directories[0].rglob("*.png"):
        with Image.open(path) as left, Image.open(directories[1] / path.relative_to(directories[0])) as right:
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


@pytest.mark.parametrize("value", ["missing", "string", "nullable"])
def test_invalid_lesion_annotations(inputs, value):
    source, predictions = inputs
    data = ad.read_h5ad(source)
    if value == "missing":
        del data.obs["Lesion"]
    elif value == "string":
        data.obs["Lesion"] = ["True"] * len(data)
    else:
        data.obs["Lesion"] = pd.array([True, None, False, True, False, False, False], dtype="boolean")
    data.write_h5ad(source)
    with pytest.raises(ValueError, match="[Ll]esion"):
        execution.load_plot_data(source, predictions, [DENSITY], "batch", spatial_heatmap_settings())
    # Explicit tissue-only mode needs no lesion metadata.
    execution.load_plot_data(source, predictions, [DENSITY], "batch", spatial_heatmap_settings({"regions": ["tissue"]}))


def test_identity_validation(inputs):
    source, predictions = inputs
    table = pq.read_table(predictions).to_pandas()
    table.loc[0, "row_position"] = 4
    table.to_parquet(predictions, index=False)
    with pytest.raises(ValueError, match="row_position"):
        execution.load_plot_data(source, predictions, [DENSITY], "batch", spatial_heatmap_settings())


def test_failure_preserves_outputs_and_cleans_temporary(inputs, tmp_path, monkeypatch):
    directory = tmp_path / "failed"
    directory.mkdir()
    old = directory / f"spatial_heatmap_{DENSITY}_tissue.png"
    old.write_bytes(b"existing figure")
    def fail(task):
        raise RuntimeError("renderer failed")
    monkeypatch.setattr(execution, "render_task", fail)
    with pytest.raises(RuntimeError, match="renderer failed"):
        run_spatial_heatmaps(*inputs, directory, [DENSITY], 1e-4, 8, workers=1)
    assert old.read_bytes() == b"existing figure"
    assert list(directory.iterdir()) == [old]


def test_budget_rejected_before_reading(inputs, tmp_path, monkeypatch):
    def fail(*args):
        pytest.fail("must reject before loading arrays")
    monkeypatch.setattr(execution, "load_plot_data", fail)
    with pytest.raises(MemoryError, match="cannot fit"):
        run_spatial_heatmaps(*inputs, tmp_path / "budget", [DENSITY], 1e-4, 300, host_budget_gib=.01)


@pytest.mark.parametrize("settings", [{"workers": 0}, {"workers": True}, {"regions": []},
                                      {"regions": ["lesion", "lesion"]}, {"regions": ["wrong"]},
                                      {"host_budget_gib": float("inf")}, {"png_compress_level": 10},
                                      {"lesion_column": ""}, {"unknown": 3}])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        spatial_heatmap_settings(settings)


def test_lesion_render_limits_and_hes_bounds(inputs, tmp_path, monkeypatch):
    from dann import spatial_heatmaps as plots
    settings = spatial_heatmap_settings()
    data = execution.load_plot_data(*inputs, [DENSITY], "batch", settings)
    manifest = execution.prepare_grids(data, [DENSITY], 1e-4, "batch", settings, tmp_path)
    execution._load_worker(manifest)
    captured = []
    original_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda figure: captured.append(figure))
    try:
        execution.render_task((DENSITY, "parameters", "lesion", str(tmp_path / "lesion.png"), 8, 1))
        figure = captured[-1]
        values = figure.axes[1].images[0].get_array()
        np.testing.assert_array_equal(values.compressed(), [1., 2., 4.])
        assert figure.axes[1].images[0].get_clim() == (1., 4.)
        assert any(text.get_text() == "No lesion" for axis in figure.axes for text in axis.texts)
        bounds = np.asarray(manifest["slides"][0]["lesion_hes"])
        assert bounds.tolist() == [[2., 0.], [4., 2.]]
    finally:
        for figure in captured:
            if hasattr(figure, "axes"):
                original_close(figure)
        execution._WORKER_DATA = None
        execution._THREAD_LIMIT.restore_original_limits()
        execution._THREAD_LIMIT = None


@pytest.mark.parametrize("kind", ["parameters", "diagnostics"])
def test_tissue_pixels_equal_reference_renderer(inputs, tmp_path, kind):
    from dann import spatial_heatmaps as plots
    settings = spatial_heatmap_settings()
    data = execution.load_plot_data(*inputs, [DENSITY], "batch", settings)
    manifest = execution.prepare_grids(data, [DENSITY], 1e-4, "batch", settings, tmp_path)
    reference = tmp_path / f"reference-{kind}.png"
    plot = plots.plot_density_heatmap if kind == "parameters" else plots.plot_diagnostics_density_heatmap
    plot(data, DENSITY, data.obs.batch.unique().tolist(), reference, dpi=12)
    execution._load_worker(manifest)
    result = tmp_path / f"optimized-{kind}.png"
    try:
        execution.render_task((DENSITY, kind, "tissue", str(result), 12, 1))
        with Image.open(reference) as left, Image.open(result) as right:
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    finally:
        execution._WORKER_DATA = None
        execution._THREAD_LIMIT.restore_original_limits()
        execution._THREAD_LIMIT = None


def test_spawned_worker_failure_cleanup(inputs, tmp_path):
    source, predictions = inputs
    # Invalid HES metadata is encountered inside the actual spawned renderers.
    with h5py.File(source, "r+") as handle:
        del handle["uns/spatial/b/scalefactors/spot_diameter_fullres"]
    directory = tmp_path / "worker-failure"
    with pytest.raises((ValueError, KeyError), match="spot"):
        run_spatial_heatmaps(source, predictions, directory, [DENSITY], 1e-4, 8, workers=2)
    assert list(directory.iterdir()) == []


def test_cli_execution_overrides(monkeypatch, tmp_path):
    from dann import config, spatial_heatmaps as plots
    values = {}
    monkeypatch.setattr(config, "load_config", lambda path: {
        "analysis": {"spatial_heatmaps": {"workers": 4}, "output_dir": str(tmp_path)},
        "spatial": {"output": "prediction.parquet"}})
    monkeypatch.setattr(config, "inference_input", lambda conf, path: Path("input.h5ad"))
    def capture(**arguments):
        values.update(arguments)
        return tmp_path
    monkeypatch.setattr(plots, "run_spatial_heatmaps", capture)
    plots.main(["--workers", "2", "--regions", "lesion", "--host-budget-gib", "64",
                "--lesion-column", "Lesion", "--png-compress-level", "3"])
    assert values["workers"] == 2 and values["regions"] == ["lesion"]
    assert values["host_budget_gib"] == 64 and values["png_compress_level"] == 3


def test_serial_initializer_failure_releases_maps_and_thread_limits(inputs, tmp_path, monkeypatch):
    original = execution._load_worker
    def fail(manifest):
        original(manifest)
        raise RuntimeError("initialization failed")
    monkeypatch.setattr(execution, "_load_worker", fail)
    directory = tmp_path / "initialization-failure"
    with pytest.raises(RuntimeError, match="initialization failed"):
        run_spatial_heatmaps(*inputs, directory, [DENSITY], 1e-4, 8, workers=1)
    assert execution._WORKER_DATA is None and execution._THREAD_LIMIT is None
    assert list(directory.iterdir()) == []


def test_unused_spatial_libraries_are_not_read(inputs, tmp_path):
    with h5py.File(inputs[0], "r+") as handle:
        handle["uns/spatial"].create_group("unused-library-without-images")
    directory = tmp_path / "unused-library"
    run_spatial_heatmaps(*inputs, directory, [DENSITY], 1e-4, 8, workers=1, regions=["tissue"])
    assert len(list(directory.rglob("*.png"))) == 2

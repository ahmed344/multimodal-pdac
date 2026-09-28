"""Focused tests for DANN analysis plot layouts and legends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection

from dann.analyze import plot_latent_umap_batch, plot_latent_umap_ziln_per_target


def _plot_config() -> dict[str, float | int | str]:
    """Build minimal analysis settings for plot tests.

    Args:
        None.

    Returns:
        dict[str, float | int | str]: Plot settings accepted by analysis helpers.
    """

    return {
        "point_size": 5.0,
        "figure_dpi": 20,
        "density_cmap": "viridis",
        "density_vmin_percentile": 1.0,
        "density_vmax_percentile": 99.0,
    }


def _assert_umap_axis_hidden(axis: Axes) -> None:
    """Assert a UMAP panel has no axis labels or visible spines.

    Args:
        axis (Axes): Matplotlib axes to inspect.

    Returns:
        None: Raises ``AssertionError`` when styling is incorrect.
    """

    assert axis.get_xlabel() == ""
    assert axis.get_ylabel() == ""
    assert all(not spine.get_visible() for spine in axis.spines.values())


def test_batch_umap_uses_one_column_larger_legend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the standalone batch UMAP legend layout and label size.

    Args:
        tmp_path (Path): Pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Fixture used to retain the plotted figure.

    Returns:
        None: Assertions inspect the generated legend.
    """

    coordinates = np.arange(12, dtype=np.float32).reshape(6, 2)
    batches = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    captured: list[plt.Figure] = []
    original_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)

    plot_latent_umap_batch(
        coordinates,
        batches,
        ["batch-a", "batch-b", "batch-c"],
        tmp_path / "batch.png",
        _plot_config(),
    )

    figure = captured[0]
    axis = figure.axes[0]
    legend = axis.get_legend()
    assert legend is not None
    assert legend._ncols == 1
    assert {text.get_fontsize() for text in legend.get_texts()} == {9.0}
    assert legend.get_title().get_text() == ""
    assert axis.get_title() == "Latent UMAP by batch (3 batches)"
    assert axis.title.get_fontsize() == 14.0
    _assert_umap_axis_hidden(axis)
    original_close(figure)


def test_ziln_umap_has_normal_and_batch_panels_without_legend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the per-density ZILN UMAP has six panels and no batch legend.

    Args:
        tmp_path (Path): Pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Fixture used to retain the plotted figure.

    Returns:
        None: Assertions inspect panel titles and legend absence.
    """

    coordinates = np.arange(12, dtype=np.float32).reshape(6, 2)
    densities = np.asarray([0.0, 0.1, 0.3, 0.0, 0.8, 1.0], dtype=np.float32)
    batches = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    captured: list[plt.Figure] = []
    original_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)

    plot_latent_umap_ziln_per_target(
        coordinates=coordinates,
        densities=densities,
        pi=np.linspace(0.1, 0.6, 6),
        mu=np.linspace(-2.0, 2.0, 6),
        sigma=np.linspace(0.5, 1.5, 6),
        batches=batches,
        batch_names=["batch-a", "batch-b", "batch-c"],
        target_column="Density_CD8",
        output_path=tmp_path / "ziln.png",
        config=_plot_config(),
        logit_epsilon=1e-4,
        density_mean=densities.copy(),
    )

    figure = captured[0]
    panel_axes = [axis for axis in figure.axes[:6] if axis.axison]
    titles = {axis.get_title() for axis in panel_axes}
    assert "Latent UMAP by mu_Density_CD8" in titles
    assert "logit(sampled mean density): Density_CD8" in titles
    mean_points = figure.axes[4].collections[1]
    observed_points = figure.axes[0].collections[1]
    np.testing.assert_array_equal(mean_points.get_array(), observed_points.get_array())
    assert mean_points.get_clim() == observed_points.get_clim()
    assert len(mean_points.get_offsets()) == np.count_nonzero(densities)
    assert len(figure.axes[4].collections[0].get_offsets()) == 2
    assert "Latent UMAP by batch" in titles
    assert all(axis.get_legend() is None for axis in panel_axes)
    assert all(axis.title.get_fontsize() == 14.0 for axis in panel_axes)
    for axis in panel_axes:
        _assert_umap_axis_hidden(axis)

    colorbar_axes = figure.axes[6:]
    assert colorbar_axes
    assert all(axis.get_ylabel() == "" for axis in colorbar_axes)
    original_close(figure)


@pytest.mark.parametrize("cap", [None, 3])
def test_analysis_sample_cap(cap: int | None) -> None:
    """An uncapped analysis preserves all held-out rows; a cap remains optional."""
    from dann.analyze import _analysis_config
    config = {"analysis": {"split": "test", "ziln_scatter_split": "validation",
                           "max_samples": cap, "num_workers": 0},
              "data": {"max_train_samples": None}, "training": {}}
    result = _analysis_config(config)
    assert result["data"]["max_test_samples"] == cap
    assert result["data"]["max_validation_samples"] == cap
    assert config["data"] == {"max_train_samples": None}


def test_normal_diagnostics_umap_panels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Diagnostics display observed logits, mean, presence, residual, width, batch."""
    from dann.analyze import plot_latent_umap_ziln_diagnostics_per_target
    captured = []
    original_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)
    plot_latent_umap_ziln_diagnostics_per_target(
        np.arange(12).reshape(6, 2), np.array([0., .2, .4, .6, .8, 1.]),
        np.full(6, .25), np.zeros(6), np.ones(6), np.zeros(6), ["a"],
        "Density_CD8", tmp_path / "diagnostics.png", _plot_config(), 1e-5)
    figure = captured[0]
    titles = [axis.get_title() for axis in figure.axes[:6]]
    assert titles == ["Latent UMAP by Logit_Density_CD8 (observed)",
                      "Latent UMAP by logit_mean_Density_CD8",
                      "Latent UMAP by 1-pi_Density_CD8",
                      "Latent UMAP by residual_Density_CD8",
                      "Latent UMAP by logit_q95-q05_Density_CD8",
                      "Latent UMAP by batch (positives)"]
    for axis in figure.axes[:6]:
        assert len(axis.collections[0].get_offsets()) == 5
    original_close(figure)


@pytest.mark.parametrize("cap", [None, 10, 0, -1, 1.5, True, "10"])
def test_config_validates_analysis_cap(tmp_path: Path, cap: object) -> None:
    """Only null and positive integer analysis limits are accepted."""
    import yaml
    from dann.config import load_config
    config = yaml.safe_load(Path("dann/config.yaml").read_text())
    config["analysis"]["max_samples"] = cap
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    if cap is None or type(cap) is int and cap > 0:
        assert load_config(path)["analysis"]["max_samples"] == cap
    else:
        with pytest.raises(ValueError, match="analysis.max_samples"):
            load_config(path)


def test_normal_analysis_export_schema_and_row_order() -> None:
    """Wide and scatter exports contain three parameters in target and row order."""
    from dann.analyze import build_latent_umap_frame, build_ziln_scatter_frame
    extracted = {"row_ids": np.array([7, 2, 9]), "batches": np.array([0, 1, 0]),
                 "targets": np.array([[0., .2], [.3, 0.], [.5, 1.]]),
                 "pi": np.full((3, 2), .2), "mu": np.zeros((3, 2)),
                 "sigma": np.ones((3, 2)),
                 "density_mean": np.arange(6).reshape(3, 2) / 10}
    targets = ["Density_CD8", "Density_Tumor"]
    wide = build_latent_umap_frame(extracted, np.zeros((3, 2)), targets, ["a", "b"], "test")
    assert wide.row_id.tolist() == [7, 2, 9]
    assert wide.columns[-10:].tolist() == [prefix + name for name in targets for prefix in ("", "pi_", "mu_", "sigma_", "density_mean_")]
    scatter = build_ziln_scatter_frame(extracted, targets, ["a", "b"], "validation", 1e-5)
    assert scatter.columns.tolist() == ["row_id", "split", "batch_id", "batch", "target", "true_density", "true_logit", "predicted_mu", "predicted_pi", "predicted_sigma", "predicted_density_mean"]
    assert scatter.row_id.tolist() == [2, 9, 7, 9]
    assert scatter.target.tolist() == [targets[0], targets[0], targets[1], targets[1]]
    assert np.isfinite(scatter.true_logit).all()

    np.testing.assert_array_equal(wide.density_mean_Density_CD8, extracted["density_mean"][:, 0])
    np.testing.assert_array_equal(scatter.predicted_density_mean, [.2, .4, .1, .5])


def test_combined_mean_umap_matches_observed_logit(tmp_path, monkeypatch) -> None:
    from dann.analyze import plot_latent_umap_densities
    captured = []
    original_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)
    means = np.array([[0., .5], [.2, 1.]])
    plot_latent_umap_densities(
        np.array([[0., 0.], [1., 1.]]), means, ["a", "b"],
        tmp_path / "means.png", _plot_config(), logit_epsilon=1e-3, sampled_means=True,
    )
    for index in range(2):
        points = captured[0].axes[index].collections[0]
        from dann.analyze import _robust_color_limits
        values = np.clip(means[means[:, index] > 0, index], 1e-3, 1 - 1e-3)
        expected = np.log(values / (1 - values))
        np.testing.assert_allclose(points.get_array(), expected)
        assert len(points.get_offsets()) == len(values)
        assert points.get_clim() == _robust_color_limits(expected, 1., 99.)
    original_close(captured[0])


@pytest.mark.parametrize("key,value", [
    ("density_mc_samples", 0), ("density_mc_samples", True),
    ("density_mc_samples", 1.5), ("density_mc_seed", -1),
    ("density_mc_seed", False), ("density_mc_seed", "12"),
])
def test_density_sampling_config_validation(key, value) -> None:
    from dann.config import density_sampling_settings
    with pytest.raises(ValueError, match=key):
        density_sampling_settings({key: value})


def test_density_sampling_legacy_defaults_and_cli() -> None:
    from dann.config import density_sampling_settings
    from dann.spatial import parse_args
    assert density_sampling_settings({}) == (1000, 20260719)
    args = parse_args(["--density-mc-samples", "40", "--density-mc-seed", "0"])
    assert density_sampling_settings(vars(args)) == (40, 0)

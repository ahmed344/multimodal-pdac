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


def test_ziln_umap_has_alpha_and_batch_panels_without_legend(
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
        alpha=np.linspace(-3.0, 3.0, 6),
        batches=batches,
        batch_names=["batch-a", "batch-b", "batch-c"],
        target_column="Density_CD8",
        output_path=tmp_path / "ziln.png",
        config=_plot_config(),
        logit_epsilon=1e-4,
    )

    figure = captured[0]
    panel_axes = figure.axes[:6]
    titles = {axis.get_title() for axis in panel_axes}
    assert "Latent UMAP by alpha_Density_CD8" in titles
    assert "Latent UMAP by batch" in titles
    assert all(axis.get_legend() is None for axis in panel_axes)
    assert all(axis.title.get_fontsize() == 14.0 for axis in panel_axes)
    for axis in panel_axes:
        _assert_umap_axis_hidden(axis)

    alpha_axis = next(
        axis
        for axis in panel_axes
        if axis.get_title() == "Latent UMAP by alpha_Density_CD8"
    )
    alpha_collections = [
        collection
        for collection in alpha_axis.collections
        if isinstance(collection, PathCollection)
    ]
    assert alpha_collections
    alpha_scatter = alpha_collections[0]
    assert alpha_scatter.cmap.name == "coolwarm"
    vmin, vmax = alpha_scatter.get_clim()
    assert vmin == pytest.approx(-vmax)
    assert vmax > 0.0
    colorbar_axes = figure.axes[6:]
    assert colorbar_axes
    assert all(axis.get_ylabel() == "" for axis in colorbar_axes)
    original_close(figure)

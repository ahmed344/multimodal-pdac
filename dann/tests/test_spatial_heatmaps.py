"""Focused tests for spatial heatmap AnnData join and logit helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from scipy import sparse

from dann import spatial_heatmaps
from dann.spatial import build_output_table, prediction_column_names
from dann.spatial_heatmaps import (
    add_interval_width_columns,
    add_logit_columns,
    add_residual_columns,
    attach_predictions,
    plot_diagnostics_density_heatmap,
    plot_density_heatmap,
)


TARGETS = (
    "Density_CD8",
    "Density_Tumor",
    "Density_Collagen",
    "Density_Stroma",
)


def _tiny_adata(obs_names: list[str], densities: dict[str, np.ndarray]) -> ad.AnnData:
    """Build a small in-memory AnnData with density observation columns.

    Args:
        obs_names (list[str]): Observation names defining row order.
        densities (dict[str, np.ndarray]): Density column name to values.

    Returns:
        ad.AnnData: AnnData with a CSR placeholder matrix and density columns.
    """

    n_obs = len(obs_names)
    adata = ad.AnnData(X=sparse.csr_matrix(np.zeros((n_obs, 2), dtype=np.float32)))
    adata.obs_names = obs_names
    adata.var_names = ["500.1", "501.2"]
    adata.obsm["spatial"] = np.arange(n_obs * 2).reshape(n_obs, 2)
    adata.uns["spatial"] = {
        batch: {"images": {"HES": np.zeros((8, 8, 3))},
                "scalefactors": {"tissue_HES_scalef": 1., "spot_diameter_fullres": 1.}}
        for batch in ("a", "batch-a")
    }
    for name, values in densities.items():
        adata.obs[name] = np.asarray(values, dtype=np.float32)
    return adata


def test_attach_predictions_joins_float32_columns_in_order() -> None:
    """Verify ordered prediction columns attach when identity matches.

    Args:
        None.

    Returns:
        None: Assertions validate joined prediction columns and dtypes.
    """

    obs_names = ["a", "b", "a"]
    adata = _tiny_adata(
        obs_names,
        {target: np.zeros(3, dtype=np.float32) for target in TARGETS},
    )
    table = build_output_table(
        row_positions=np.arange(3, dtype=np.int64),
        obs_names=np.asarray(obs_names, dtype=str),
        mu=np.arange(12, dtype=np.float32).reshape(3, 4),
        pi=np.full((3, 4), 0.25, dtype=np.float32),
        sigma=np.ones((3, 4), dtype=np.float32),
        target_columns=TARGETS,
    )
    predictions = table.to_pandas()
    attached = attach_predictions(adata, predictions)

    assert attached == prediction_column_names(TARGETS)
    for name in attached:
        assert name in adata.obs.columns
        assert adata.obs[name].dtype == np.float32
    np.testing.assert_allclose(
        adata.obs["prob_of_presence_Density_CD8"].to_numpy(),
        np.full(3, 0.75, dtype=np.float32),
    )


def test_attach_predictions_rejects_obs_name_mismatch() -> None:
    """Verify attach fails when Parquet obs names diverge from AnnData.

    Args:
        None.

    Returns:
        None: Assertions require an obs_name alignment error.
    """

    adata = _tiny_adata(
        ["a", "b"],
        {target: np.zeros(2, dtype=np.float32) for target in TARGETS},
    )
    table = build_output_table(
        row_positions=np.arange(2, dtype=np.int64),
        obs_names=np.asarray(["a", "c"], dtype=str),
        mu=np.zeros((2, 4), dtype=np.float32),
        pi=np.full((2, 4), 0.5, dtype=np.float32),
        sigma=np.ones((2, 4), dtype=np.float32),
        target_columns=TARGETS,
    )
    with pytest.raises(ValueError, match="obs_name"):
        attach_predictions(adata, table.to_pandas())


def test_attach_predictions_rejects_noncontiguous_row_position() -> None:
    """Verify attach fails when row_position is not AnnData order.

    Args:
        None.

    Returns:
        None: Assertions require a row_position alignment error.
    """

    adata = _tiny_adata(
        ["a", "b"],
        {target: np.zeros(2, dtype=np.float32) for target in TARGETS},
    )
    predictions = pd.DataFrame(
        {
            "row_position": np.asarray([1, 0], dtype=np.int64),
            "obs_name": ["b", "a"],
            "mu_Density_CD8": np.asarray([0.0, 1.0], dtype=np.float32),
        }
    )
    with pytest.raises(ValueError, match="row_position"):
        attach_predictions(adata, predictions)


def test_add_logit_columns_clips_and_transforms() -> None:
    """Verify logit columns use the clamped transform for each density.

    Args:
        None.

    Returns:
        None: Assertions validate logit values and created column names.
    """

    epsilon = 1e-4
    densities = ("Density_CD8", "Density_Tumor")
    adata = _tiny_adata(
        ["0", "1", "2"],
        {
            "Density_CD8": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            "Density_Tumor": np.asarray([0.25, 0.75, 0.0], dtype=np.float32),
        },
    )
    columns = add_logit_columns(adata, densities, epsilon)

    assert columns == ["logit_Density_CD8", "logit_Density_Tumor"]
    expected_cd8 = np.log(
        np.clip(np.asarray([0.0, 0.5, 1.0]), epsilon, 1.0 - epsilon)
        / (1.0 - np.clip(np.asarray([0.0, 0.5, 1.0]), epsilon, 1.0 - epsilon))
    ).astype(np.float32)
    np.testing.assert_allclose(
        adata.obs["logit_Density_CD8"].to_numpy(),
        expected_cd8,
        rtol=1e-5,
    )
    assert adata.obs["logit_Density_CD8"].dtype == np.float32


def test_add_logit_columns_rejects_missing_density() -> None:
    """Verify logit helper requires every requested density column.

    Args:
        None.

    Returns:
        None: Assertions require a missing-column KeyError.
    """

    adata = _tiny_adata(["0"], {"Density_CD8": np.asarray([0.5], dtype=np.float32)})
    with pytest.raises(KeyError, match="Density_Tumor"):
        add_logit_columns(adata, ["Density_CD8", "Density_Tumor"], 1e-4)


def test_plot_density_heatmap_requires_sigma_column(tmp_path: Path) -> None:
    """Verify spatial figures require the exported normal scale.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions require a missing-sigma KeyError before plotting.
    """

    density = "Density_CD8"
    adata = _tiny_adata(
        ["0", "1"],
        {density: np.asarray([0.25, 0.75], dtype=np.float32)},
    )
    adata.obs["logit_Density_CD8"] = np.zeros(2, dtype=np.float32)
    adata.obs["mu_Density_CD8"] = np.zeros(2, dtype=np.float32)
    adata.obs["prob_of_presence_Density_CD8"] = np.ones(2, dtype=np.float32)
    adata.obs["batch"] = ["batch-a", "batch-a"]
    adata.obs["x"] = [0, 1]
    adata.obs["y"] = [0, 0]

    with pytest.raises(KeyError, match="sigma_Density_CD8"):
        plot_density_heatmap(
            adata,
            density,
            ["batch-a"],
            tmp_path / "heatmap.png",
        )


def test_plot_density_heatmap_writes_six_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the spatial figure includes normal parameters before the HES panel.

    Args:
        tmp_path (Path): Pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Fixture used to isolate Scanpy rendering.

    Returns:
        None: Assertions inspect the six primary subplot titles.
    """

    density = "Density_CD8"
    adata = _tiny_adata(
        ["0", "1", "2", "3"],
        {density: np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32)},
    )
    adata.obs["logit_Density_CD8"] = np.arange(4, dtype=np.float32)
    adata.obs["mu_Density_CD8"] = np.arange(4, dtype=np.float32)
    adata.obs["prob_of_presence_Density_CD8"] = np.full(4, 0.75, dtype=np.float32)
    adata.obs["sigma_Density_CD8"] = np.ones(4, dtype=np.float32)
    adata.obs["density_mean_Density_CD8"] = [0., .2, .4, .6]
    adata.obs["batch"] = ["batch-a"] * 4
    adata.obs["x"] = [0, 1, 0, 1]
    adata.obs["y"] = [0, 0, 1, 1]
    captured: list[plt.Figure] = []
    original_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)
    monkeypatch.setattr(spatial_heatmaps.sc.pl, "spatial", Mock(return_value=None))

    plot_density_heatmap(
        adata,
        density,
        ["batch-a"],
        tmp_path / "heatmap.png",
        dpi=20,
    )

    figure = captured[0]
    titles = [axis.get_title() for axis in figure.axes[:6]]
    assert titles[4] == "logit(sampled mean density): Density_CD8"
    assert titles[5] == "HES"
    mean_image = figure.axes[4].images[0]
    np.testing.assert_array_equal(np.ma.getmaskarray(mean_image.get_array()), [[True, False], [False, False]])
    expected = np.log(np.array([.2, .4, .6]) / np.array([.8, .6, .4]))
    np.testing.assert_allclose(mean_image.get_array().compressed(), expected, rtol=1e-6)
    np.testing.assert_allclose(mean_image.get_clim(), [expected.min(), expected.max()])
    assert figure.get_size_inches()[0] == pytest.approx(36.0)
    original_close(figure)


def test_add_residual_columns_masks_zero_observations() -> None:
    """Verify residuals exist only where an observed density is positive.

    Args:
        None.

    Returns:
        None: Assertions validate NaN masking of structural zeros.
    """

    density = "Density_CD8"
    adata = _tiny_adata(
        ["0", "1", "2"],
        {density: np.asarray([0.0, 0.5, 0.25], dtype=np.float32)},
    )
    add_logit_columns(adata, [density], 1e-4)
    adata.obs[f"logit_mean_{density}"] = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

    created = add_residual_columns(adata, [density])
    assert created == [f"residual_{density}"]
    residual = adata.obs[f"residual_{density}"].to_numpy()
    assert np.isnan(residual[0])
    assert np.isfinite(residual[1]) and np.isfinite(residual[2])
    np.testing.assert_allclose(
        residual[1],
        np.float32(1.0) - adata.obs[f"logit_{density}"].to_numpy()[1],
        rtol=1e-6,
    )


def test_add_interval_width_columns_computes_span() -> None:
    """Verify the interval-width column is the q95 minus q05 span.

    Args:
        None.

    Returns:
        None: Assertions validate the computed width column.
    """

    density = "Density_CD8"
    adata = _tiny_adata(
        ["0", "1"],
        {density: np.asarray([0.5, 0.25], dtype=np.float32)},
    )
    adata.obs[f"logit_q05_{density}"] = np.asarray([-1.0, -2.0], dtype=np.float32)
    adata.obs[f"logit_q95_{density}"] = np.asarray([1.0, 3.0], dtype=np.float32)

    created = add_interval_width_columns(adata, [density])
    assert created == [f"interval_width_{density}"]
    np.testing.assert_allclose(
        adata.obs[f"interval_width_{density}"].to_numpy(),
        np.asarray([2.0, 5.0], dtype=np.float32),
    )


def test_plot_diagnostics_density_heatmap_titles_and_residual_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the diagnostics figure shows the expected five panels.

    Args:
        tmp_path (Path): Pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Fixture used to isolate Scanpy rendering.

    Returns:
        None: Assertions inspect the six primary subplot titles.
    """

    density = "Density_CD8"
    adata = _tiny_adata(
        ["0", "1", "2", "3"],
        {density: np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32)},
    )
    adata.obs[f"logit_{density}"] = np.arange(4, dtype=np.float32)
    adata.obs[f"logit_mean_{density}"] = np.arange(4, dtype=np.float32) + 0.5
    adata.obs[f"logit_median_{density}"] = np.arange(4, dtype=np.float32) + 0.25
    adata.obs[f"residual_{density}"] = np.linspace(-1.0, 1.0, 4, dtype=np.float32)
    adata.obs[f"interval_width_{density}"] = np.full(4, 2.0, dtype=np.float32)
    adata.obs["batch"] = ["batch-a"] * 4
    adata.obs["x"] = [0, 1, 0, 1]
    adata.obs["y"] = [0, 0, 1, 1]
    captured: list[plt.Figure] = []
    original_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)
    monkeypatch.setattr(spatial_heatmaps.sc.pl, "spatial", Mock(return_value=None))

    plot_diagnostics_density_heatmap(
        adata,
        density,
        ["batch-a"],
        tmp_path / "diagnostics.png",
        dpi=20,
    )

    figure = captured[0]
    titles = [axis.get_title() for axis in figure.axes[:5]]
    assert titles[0] == f"logit_{density} (observed)"
    assert titles[1] == f"logit_mean_{density}"
    assert titles[2] == f"residual_{density}"
    assert titles[3] == f"interval_width_{density}"
    assert titles[4] == "HES"
    residual_image = figure.axes[2].images[0]
    assert residual_image.cmap.name == "coolwarm"
    vmin, vmax = residual_image.get_clim()
    assert vmin == pytest.approx(-vmax)
    original_close(figure)


def test_plot_diagnostics_density_heatmap_requires_derived_columns(
    tmp_path: Path,
) -> None:
    """Verify the diagnostics figure refuses to plot without derived columns.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions require a missing-column KeyError.
    """

    density = "Density_CD8"
    adata = _tiny_adata(
        ["0", "1"],
        {density: np.asarray([0.25, 0.75], dtype=np.float32)},
    )
    adata.obs[f"logit_{density}"] = np.zeros(2, dtype=np.float32)
    adata.obs["batch"] = ["batch-a", "batch-a"]
    adata.obs["x"] = [0, 1]
    adata.obs["y"] = [0, 0]

    with pytest.raises(KeyError, match=f"logit_mean_{density}"):
        plot_diagnostics_density_heatmap(
            adata,
            density,
            ["batch-a"],
            tmp_path / "diagnostics.png",
        )


@pytest.mark.parametrize("values", [[0., 0., 0., 0.], [0., 0., 0., .5]])
@pytest.mark.parametrize("plot", [plot_density_heatmap, plot_diagnostics_density_heatmap])
def test_heatmaps_preserve_grid_with_masked_zeros(tmp_path, monkeypatch, values, plot) -> None:
    """Masking zeros must neither shrink the tissue grid nor fail on empty signal."""
    density = "Density_CD8"
    adata = _tiny_adata(["0", "1", "2", "3"], {density: np.asarray(values)})
    for prefix in ("logit_", "mu_", "prob_of_presence_", "sigma_", "density_mean_", "logit_mean_", "residual_", "interval_width_"):
        adata.obs[prefix + density] = np.ones(4)
    adata.obs["batch"] = ["a"] * 4
    adata.obs["x"] = [0, 1, 0, 1]
    adata.obs["y"] = [0, 0, 1, 1]
    captured = []
    original_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)
    monkeypatch.setattr(spatial_heatmaps.sc.pl, "spatial", Mock(return_value=None))
    plot(adata, density, ["a"], tmp_path / "masked.png", dpi=20)
    image = captured[0].axes[0].images[0].get_array()
    assert image.shape == (2, 2)
    np.testing.assert_array_equal(np.ma.getmaskarray(image), (np.asarray(values) == 0).reshape(2, 2))
    original_close(captured[0])


def test_hes_bounds_match_full_scanpy_image_limits() -> None:
    """Two invisible corners preserve Scanpy's crop, scale, image, and spot padding."""
    adata = _tiny_adata([str(i) for i in range(5)], {"Density_CD8": np.ones(5)})
    adata.obsm["spatial"] = np.array([[-2., 3.], [6., 2.], [0., 5.], [2., 9.], [6., -1.]])
    adata.uns["spatial"]["a"]["scalefactors"] = {
        "tissue_HES_scalef": 1.25, "spot_diameter_fullres": 4.}
    adata.uns["spatial"]["a"]["images"]["HES"] = np.random.default_rng(1).random((15, 20, 3))
    figure, axes = plt.subplots(1, 2)
    spatial_heatmaps.sc.pl.spatial(
        adata, library_id="a", img_key="HES", frameon=False, show=False, ax=axes[0])
    spatial_heatmaps._plot_hes_context(
        adata, pd.Series(True, index=adata.obs_names), "a", axes[1])
    np.testing.assert_allclose(axes[0].get_xlim(), axes[1].get_xlim())
    np.testing.assert_allclose(axes[0].get_ylim(), axes[1].get_ylim())
    np.testing.assert_array_equal(axes[0].images[0].get_array(), axes[1].images[0].get_array())
    np.testing.assert_allclose(axes[0].images[0].get_extent(), axes[1].images[0].get_extent())
    assert len(axes[1].collections[0].get_paths()) == 2
    assert axes[1].collections[0].get_facecolors()[0, 3] == 0.
    plt.close(figure)


@pytest.mark.parametrize("means", [[0., .2, 1.], [0., 0., 0.]])
def test_mean_heatmap_masks_own_zeros_and_preserves_grid(tmp_path, monkeypatch, means) -> None:
    density = "Density_CD8"
    adata = _tiny_adata(["0", "1", "2"], {density: np.zeros(3)})
    for prefix in ("logit_", "mu_", "prob_of_presence_", "sigma_"):
        adata.obs[prefix + density] = np.ones(3)
    adata.obs["density_mean_" + density] = means
    adata.obs["batch"] = ["a"] * 3
    adata.obs["x"] = [0, 1, 0]
    adata.obs["y"] = [0, 0, 1]
    captured = []
    original_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)
    monkeypatch.setattr(spatial_heatmaps.sc.pl, "spatial", Mock(return_value=None))
    plot_density_heatmap(adata, density, ["a"], tmp_path / "means.png", dpi=20, logit_epsilon=1e-3)
    image = captured[0].axes[4].images[0].get_array()
    expected_mask = np.r_[np.asarray(means) == 0., True].reshape(2, 2)
    np.testing.assert_array_equal(np.ma.getmaskarray(image), expected_mask)
    positive = np.asarray(means)[np.asarray(means) > 0]
    clipped = np.clip(positive, 1e-3, 1 - 1e-3)
    np.testing.assert_allclose(image.compressed(), np.log(clipped / (1 - clipped)), rtol=1e-6)
    original_close(captured[0])
    del adata.obs["density_mean_" + density]
    with pytest.raises(KeyError, match="rerun python -m dann.spatial"):
        plot_density_heatmap(adata, density, ["a"], tmp_path / "old.png", dpi=20)

"""Sparse TIC aggregation and native-coordinate, shared-scale rendering."""
from __future__ import annotations

from anndata.io import write_elem
import h5py
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np
import pytest
from scipy import sparse

from msi_preprocess.tic import plot_tic_grid, sum_tic


@pytest.mark.parametrize("block_rows", [1, 3, 20])
def test_tic_sums_all_features_in_float64_before_display(tmp_path, block_rows):
    values = np.zeros((7, 503), dtype=np.float32)
    values[1, [0, 501, 502]] = [2**24, 1, 1]
    values[3, [1, 502]] = [3, 4]
    values[6, 502] = .125
    with h5py.File(tmp_path / "values.h5", "w") as handle:
        write_elem(handle, "counts", sparse.csr_matrix(values))
        actual = sum_tic(handle["counts"], block_rows)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, [0, 2**24+2, 0, 7, 0, 0, .125])
    assert np.log1p(actual[3]) == pytest.approx(np.log(8))
    assert np.log1p(actual[3]) != pytest.approx(np.log1p(3) + np.log1p(4))


def test_tic_grids_preserve_interleaved_rows_holes_and_shared_scale(tmp_path, monkeypatch):
    coordinates = np.array([[10, 20], [-2, 5], [12, 21], [0, 5], [11, 20], [0, 6]])
    slide_rows = dict(a=np.array([1, 3, 5]), b=np.array([0, 2, 4]))
    counts = np.array([0., 1., 15., 7., 3., 0.])
    norm = Normalize(0, np.log1p(7), clip=True)
    captured = []

    def inspect_figure(figure, path, **kwargs):
        assert kwargs["dpi"] == 250
        np.testing.assert_array_equal(figure.get_size_inches(), [20, 3])
        assert len(figure.axes) == 3  # Two populated slide panels and one colorbar.
        panels = figure.axes[:2]
        assert [ax.get_title() for ax in panels] == ["a", "b"]
        stage = []
        for ax in panels:
            image = ax.images[0]
            assert image.norm is norm
            assert image.get_cmap().name == "turbo"
            np.testing.assert_array_equal(image.get_cmap().get_bad(), [1, 1, 1, 1])
            assert image.origin == "upper" and image.get_interpolation() == "nearest"
            assert ax.get_aspect() == 1
            np.testing.assert_array_equal(image.get_extent(), [-.5, 2.5, 1.5, -.5])
            stage.append(image.get_array().filled(np.nan))
        captured.append(stage)

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    for stage, values in (("counts", counts), ("cleaned", 2*counts)):
        plot_tic_grid(tmp_path / f"{stage}.png", coordinates, slide_rows, values, norm, stage)
    expected = [np.array([[1, np.nan, 7], [np.nan, np.nan, 0]]),
                np.array([[0, 3, np.nan], [np.nan, np.nan, 15]])]
    for index, stage in enumerate(captured):
        for actual, linear in zip(stage, expected):
            np.testing.assert_allclose(actual, np.log1p((index+1)*linear), equal_nan=True)
    assert norm(np.log1p(15)) == 1  # The same upper limit clips both figures.

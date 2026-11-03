"""Slide clustering and shared pixel ordering in preprocessing heatmaps."""
from __future__ import annotations

import anndata as ad
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PIL import Image
import pytest
from scipy import sparse
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from msi_preprocess import comparison


def test_slide_clustering_uses_mean_log_counts_and_groups_similar_profiles():
    profiles = np.array([[4., 3., 1., 0.], [0., 1., 3., 4.],
                         [.1, 1., 3., 4.], [4., 3., 1., .1]])
    names = np.repeat(list("abcd"), 2)
    # The mean of log1p(counts) is exactly the intended profile; log1p of
    # mean counts would give different correlations and linkage distances.
    counts = np.expm1(np.repeat(profiles, 2, axis=0) * np.tile([.5, 1.5], 4)[:, None])
    order, tree, constant = comparison.cluster_slides(counts, names)
    np.testing.assert_allclose(tree, linkage(pdist(profiles, metric="correlation"), method="average"), atol=1e-14)
    assert {frozenset(pair) for pair in order.reshape(2, 2)} == {frozenset("ad"), frozenset("bc")}
    assert not len(constant)
    shuffled = np.random.default_rng(8).permutation(len(names))
    again = comparison.cluster_slides(counts[shuffled], names[shuffled])
    np.testing.assert_array_equal(order, again[0])
    np.testing.assert_allclose(tree, again[1])


def test_slide_clustering_ties_use_alphabetical_input():
    names = np.array(list("dbac"))
    counts = np.tile([0., 1., 3., 2.], (4, 1))
    first = comparison.cluster_slides(counts, names)
    for permutation in (np.arange(4)[::-1], np.argsort(names)):
        again = comparison.cluster_slides(counts[permutation], names[permutation])
        for expected, actual in zip(first, again):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("counts,names,expected,constants", [
    ([[1, 1], [0, 1], [1, 0], [0, 0]], ["z", "c", "b", "a"], ["b", "c", "a", "z"], ["a", "z"]),
    ([[3, 3], [0, 0]], ["z", "a"], ["a", "z"], ["a", "z"]),
    ([[0, 0], [0, 1]], ["a", "b"], ["b", "a"], ["a"]),
    ([[0, 1], [0, 3]], ["a", "a"], ["a"], []),
    ([[0, 0], [2, 2]], ["a", "a"], ["a"], ["a"]),
    ([[1], [2]], ["z", "a"], ["a", "z"], ["a", "z"]),
])
def test_slide_clustering_constants_and_single_slide(counts, names, expected, constants):
    order, tree, constant = comparison.cluster_slides(np.asarray(counts, dtype=float), np.array(names))
    assert order.tolist() == expected
    assert constant.tolist() == constants
    assert tree.shape == (max(len(expected) - len(constants) - 1, 0), 4)


def test_heatmap_shared_slide_blocks_metadata_and_layout(tmp_path, monkeypatch):
    profiles = np.array([[4., 3., 1., 0.], [0., 1., 3., 4.],
                         [.1, 1., 3., 4.], [4., 3., 1., .1]])
    sizes = np.array([2, 3, 4, 5])
    batch = np.repeat(list("abcd"), sizes)
    values = np.expm1(np.repeat(profiles, sizes, axis=0))
    # Include a constant retained peak and a high-variance excluded peak.
    counts = np.column_stack([values, np.full(len(batch), 9.), np.arange(len(batch)) * 1000.])
    obs = pd.DataFrame(dict(batch=batch, x=np.concatenate([np.arange(n) for n in sizes]), y=0),
                       index=[f"pixel_{i}" for i in range(len(batch))])
    raw = ad.AnnData(sparse.csr_matrix(counts * 3), obs=obs)
    raw.layers["counts"] = sparse.csr_matrix(counts)
    source_path = tmp_path / "source.h5ad"
    raw.write_h5ad(source_path)
    # Export rows differ from source rows and interleave slides.
    source_rows = np.random.default_rng(19).permutation(len(batch))
    retained = np.array([3, 4, 1, 0, 2])
    multipliers = np.array([1., 2., 3., 4.])
    factors = obs.batch.map(dict(zip("abcd", multipliers))).to_numpy()
    cleaned = counts[:, retained] / factors[:, None]
    cleaned[::2, 2] = 0
    exported = ad.AnnData(sparse.csr_matrix(cleaned[source_rows]), obs=obs.iloc[source_rows].copy())
    exported.write_h5ad(tmp_path / "lesion.h5ad")
    (tmp_path / "figures").mkdir()
    monkeypatch.setattr(comparison, "aligned_export_rows", lambda *args: (exported.obs.copy(), source_rows))
    config = dict(lesion_input=str(source_path), seed=42, block_rows=32,
                  report=dict(pixels_per_slide=3, max_peaks=4, upper_percentile=99.5))
    data = dict(batch_column="batch", x_column="x", y_column="y")
    state = dict(retained=retained, features=raw.var_names.to_numpy(), anchors=np.array([0]),
                 slides=np.array(list("abcd")), multipliers=multipliers)
    sampled = comparison.sample_pixels(exported.obs, data, 3, 42)
    original_sample = source_rows[sampled]
    sampled_counts = counts[original_sample]
    ranked, _ = comparison.rank_retained_peaks(sampled_counts, retained, 4)
    column_order, peak_tree, _ = comparison.cluster_peaks(sampled_counts[:, np.sort(ranked)])
    expected_peaks = np.sort(ranked)[column_order]
    slide_order, slide_tree, _ = comparison.cluster_slides(sampled_counts[:, expected_peaks],
                                                          batch[original_sample])
    assert slide_order.tolist() != sorted(slide_order)
    expected_rows = np.concatenate([original_sample[batch[original_sample] == s] for s in slide_order])
    block_sizes = np.array([np.sum(batch[expected_rows] == s) for s in slide_order])
    starts = np.r_[0, np.cumsum(block_sizes)[:-1]]
    expected_stages = dict(counts=counts[expected_rows][:, expected_peaks],
        selected=counts[expected_rows][:, expected_peaks],
        integrated=counts[expected_rows][:, expected_peaks]/factors[expected_rows, None],
        cleaned=cleaned[expected_rows][:, pd.Index(retained).get_indexer(expected_peaks)],
        old_X=raw.X[expected_rows][:, expected_peaks].toarray())
    rendered = []

    def check_figure(fig, path, **kwargs):
        stage = path.stem.removeprefix("heatmap_")
        assert len(fig.axes) == 2  # Only the heatmap and its colorbar.
        ax, colorbar = fig.axes
        assert ax.get_subplotspec().get_gridspec().nrows == 1
        assert len(colorbar.images) == 0
        np.testing.assert_array_equal(fig.get_size_inches(), [14, 24])
        assert [label.get_text() for label in ax.get_yticklabels()] == slide_order.tolist()
        np.testing.assert_array_equal(ax.get_yticks(), starts + (block_sizes - 1)/2)
        np.testing.assert_array_equal([line.get_ydata()[0] for line in ax.lines], starts[1:] - .5)
        np.testing.assert_array_equal(ax.images[0].get_array(), np.log1p(expected_stages[stage]))
        rendered.append(stage)

    monkeypatch.setattr(Figure, "savefig", check_figure)
    meta = comparison.render_heatmaps(tmp_path, dict(config=config, coordinate_columns=data), state)
    pixels = pd.read_csv(tmp_path / "heatmap_pixels.csv")
    peaks = pd.read_csv(tmp_path / "heatmap_peaks.csv")
    np.testing.assert_array_equal(pixels.source_row, expected_rows)
    np.testing.assert_array_equal(source_rows[pixels.export_row], expected_rows)
    np.testing.assert_array_equal(peaks.original_index, expected_peaks)
    np.testing.assert_array_equal(np.load(tmp_path / "heatmap_linkage.npy"), peak_tree)
    np.testing.assert_array_equal(np.load(tmp_path / "heatmap_slide_linkage.npy"), slide_tree)
    assert meta["slide_order"] == slide_order.tolist()
    assert meta["slide_clustering"]["linkage_input_slides"] == list("abcd")
    assert meta["slide_clustering"]["constant_slides"] == []
    assert not meta["dendrogram_visible"]
    assert meta["slide_blocks"] == [dict(slide=s, start=int(start), size=int(size))
                                    for s, start, size in zip(slide_order, starts, block_sizes)]
    assert set(rendered) == set(expected_stages)
    for stage, matrix in expected_stages.items():
        upper = meta["stages"][stage]["upper_linear"]
        expected = colormaps["turbo"](Normalize(0, np.log1p(upper))(np.log1p(matrix)), bytes=True)
        with Image.open(tmp_path / "figures" / f"heatmap_{stage}_pixels.png") as image:
            np.testing.assert_array_equal(np.asarray(image), expected)

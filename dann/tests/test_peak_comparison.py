"""Numerical checks for peak comparison without real datasets or checkpoints."""
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import spearmanr, rankdata

from dann.verification.peak_comparison import (
    family_associations, family_match, family_profile_r2, measured_family_signals, nearest_indices,
    neighbor_coherence, rank_correlations,
)


def test_family_matching_is_invariant_to_arbitrary_family_numbers():
    left = np.array([1, 1, 2, 2, 3, 3, 3])
    right = np.array([8, 8, 4, 4, 2, 2, 2])
    scores, matched = family_match(left, right)
    assert scores['adjusted_rand'] == scores['matched_peak_fraction'] == 1
    assert (matched.jaccard == 1).all()
    assert dict(zip(matched.left_family, matched.right_family)) == {1:8, 2:4, 3:2}


def test_partial_rank_correlation_matches_independent_projection_reference():
    rng = np.random.default_rng(3)
    control = rng.normal(size=(200, 2))
    y = 3*control[:, 0] + rng.normal(size=200)
    x = np.column_stack((control[:, 0]+rng.normal(size=200), y+rng.normal(size=200)))
    np.testing.assert_allclose(rank_correlations(x, y), [spearmanr(x[:, j], y).statistic for j in range(2)])
    design = np.column_stack((np.ones(200), rankdata(control, axis=0)))
    projector = np.eye(200)-design@np.linalg.pinv(design)
    residual_x = projector@rankdata(x, axis=0)
    residual_y = projector@rankdata(y)
    expected = [np.corrcoef(residual_x[:, j], residual_y)[0, 1] for j in range(2)]
    np.testing.assert_allclose(rank_correlations(x, y, control), expected, atol=1e-12)


def test_family_signals_coalesce_raw_duplicates_and_preserve_missing_member_zeros():
    matrix = sparse.csr_matrix((np.array([1.,2.,0.,4.]), np.array([0,0,1,2]), np.array([0,3,4])), shape=(2,3))
    labels = {'a':np.array([1,1,2]), 'b':np.array([2,1,2])}
    signals, controls = measured_family_signals(matrix, labels, dict(nonzero_threshold=0., intensity_transform='log1p', intensity_clip_max=None))
    transformed = np.log1p(np.array([[3,0,0],[0,0,4]], dtype=np.float32)).astype(float)
    np.testing.assert_allclose(signals['a'], np.column_stack((transformed[:,:2].mean(1), transformed[:,2])))
    np.testing.assert_allclose(signals['b'], np.column_stack((transformed[:,1], transformed[:,[0,2]].mean(1))))
    np.testing.assert_allclose(controls[:,0], transformed.sum(1))
    np.testing.assert_array_equal(controls[:,1], [1,1])


def test_neighbor_controls_exclude_self_and_detect_perfect_profile_groups():
    neighbors = nearest_indices(np.ones((4,4)), 2)
    assert not np.any(neighbors == np.arange(4)[:,None])
    profile = np.array([.1,.1,.8,.8])
    mae, r2 = neighbor_coherence(profile, np.array([[1],[0],[3],[2]]))
    assert mae == 0 and r2 == 1
    assert family_profile_r2(profile, np.array([1,1,2,2])) == 1


def test_constant_family_correlations_stay_undefined_and_slide_names_are_pickle_free(tmp_path):
    rng = np.random.default_rng(8)
    signals = np.column_stack((np.zeros(60), rng.normal(size=60)))
    result = family_associations(signals, rng.normal(size=(60, 2)),
        np.array(['slide_a']*30+['slide_b']*30, dtype=object), rng.normal(size=(60, 2)))
    for kind in ('raw', 'partial'):
        assert np.isnan(result[kind][:, 0]).all()
        assert np.isfinite(result[kind][:, 1]).all()
    path = tmp_path/'associations.npz'
    np.savez(path, **result)
    with np.load(path, allow_pickle=False) as saved:
        assert saved['slides'].tolist() == ['slide_a', 'slide_b']

"""Frozen geometry, distribution summaries, sample identity and experiment contracts."""
import copy
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from dann import geometry as g
from dann.config import latent_geometry_settings
from dann.latent_geometry import eligible_samples, run_experiments, validate_splits


def test_pca_distances_partial_whitening_and_frozen_evaluation():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(60, 6))*[50, 10, 1, .1, .01, 0]+10
    heldout = rng.normal(size=(12, 6))+1000
    fitted = g.fit_preprocessing(train)
    original = copy.deepcopy(fitted)
    settings = latent_geometry_settings()
    unweighted, _, info = g.transform_geometry(train, fitted, 'pca_16', settings)
    np.testing.assert_allclose(pdist(unweighted), pdist(train), atol=1e-10)
    assert info['components'] == 6
    partial, _, _ = g.transform_geometry(train, fitted, 'pca_32_partial', settings)
    eigenvalues = fitted['eigenvalues']
    expected = eigenvalues / np.maximum(eigenvalues, eigenvalues[0]*settings['eigenvalue_floor'])**settings['whitening_power']
    np.testing.assert_allclose(partial.var(axis=0, ddof=1), expected, atol=1e-10)
    validation, _, _ = g.transform_geometry(heldout, fitted, 'standardized_euclidean', settings)
    np.testing.assert_allclose(validation, (heldout-train.mean(0))/fitted['feature_scale'], atol=1e-10)
    assert not np.allclose(validation.mean(0), 0)
    for name in original:
        np.testing.assert_array_equal(fitted[name], original[name])


def test_centered_cosine_is_global_centering_and_zero_norm_fails():
    x = np.array([[1., 2.], [3., 7.], [8., 4.]])
    fitted = g.fit_preprocessing(x)
    transformed, metric, _ = g.transform_geometry(x, fitted, 'centered_cosine', latent_geometry_settings())
    np.testing.assert_allclose(transformed, x-x.mean(0))
    assert metric == 'cosine'
    with pytest.raises(ValueError, match='zero-norm'):
        g.transform_geometry(x.mean(0)[None, :], fitted, 'centered_cosine', latent_geometry_settings())


def test_probability_summaries_thresholds_and_distance_mixture():
    info = g.training_thresholds(np.array([0, .1, .2, .4, .8, 1, np.nan]), [.25, .5, .75, .9], 1e-6)
    np.testing.assert_allclose(info['thresholds'], np.quantile([.1, .2, .4, .8, 1], [.25, .5, .75, .9]))
    q = g.probability_block(np.array([.2, .9]), np.array([0., 1.]), np.array([1., .5]), [.5, .8], 1e-6)
    assert q[0, 1] == pytest.approx(.4)
    assert np.all(np.diff(q, axis=1) <= 0) and np.all((q >= 0) & (q <= 1))
    rng = np.random.default_rng(1)
    x, q = rng.normal(size=(30, 4)), rng.uniform(size=(30, 5))
    sl, sq = g.distance_scale(x, 1000, 3), g.distance_scale(q, 1000, 3)
    for weight in (0, .1, .3):
        combined = g.guided_geometry(x, q, weight, sl, sq)
        np.testing.assert_allclose(pdist(combined)**2/sl**2,
                                   (1-weight)*pdist(x)**2/sl**2+weight*pdist(q)**2/sq**2, atol=1e-12)
    np.testing.assert_array_equal(g.guided_geometry(x, q, 0, sl, sq), x)
    with pytest.raises(ValueError, match='Constant training probability'):
        g.guided_geometry(x, q, .1, sl, 0)
    assert g.training_thresholds(np.zeros(4), [.25], 1e-6)['presence_only']
    assert g.training_thresholds(np.ones(4), [.25, .5], 1e-6)['thresholds'] == [1-1e-6]


def test_exact_neighbors_exclude_self_with_duplicates_and_cross_queries():
    x = np.array([[0., 0.], [0., 0.], [0., 0.], [2., 1.], [8., 3.]])
    ids, d = g.neighbors(x, 2, 'euclidean')
    assert not np.any(ids == np.arange(len(x))[:, None])
    full = squareform(pdist(x))
    np.fill_diagonal(full, np.inf)
    np.testing.assert_allclose(d, np.sort(full, axis=1)[:, :2])
    query_ids, query_d = g.neighbors(x, 1, 'euclidean', x[-1:])
    assert query_ids[0, 0] == 4 and query_d[0, 0] == 0


def test_collapsed_geometries_are_explicitly_rejected():
    x = np.ones((10, 4))
    fitted = g.fit_preprocessing(x)
    for name in ('raw_euclidean', 'raw_cosine', 'centered_cosine', 'pca_32_partial'):
        with pytest.raises(ValueError):
            g.transform_geometry(x, fitted, name, latent_geometry_settings())


def test_source_metadata_target_order_and_exclusion_reasons(tmp_path):
    import h5py
    from anndata.io import write_elem
    from dann.latent_geometry import source_observations
    path = tmp_path/'observations.h5ad'
    with h5py.File(path, 'w') as h:
        obs = h.create_group('obs')
        for name, values in dict(batch=np.array(['a']*4, dtype=object), Lesion=np.ones(4, bool),
            x=np.arange(4), y=np.zeros(4), Density_CD8=np.array([.1, .2, .8, .05]),
            Density_Tumor=np.array([.5, 1., .4, .3]), exposure=np.array([10., 20., 30., 40.])).items():
            write_elem(obs, name, values)
    c = dict(data=dict(path=str(path), target_columns=['Density_Tumor', 'Density_CD8'], batch_column='batch',
        x_column='x', y_column='y', cd8_normalization=dict(enabled=True, min_non_tumor_fraction=.01)),
        analysis=dict(latent_geometry=dict(technical_columns=['exposure'])))
    frame = source_observations(c)
    assert frame.valid_Density_CD8.tolist() == [True, False, False, True]
    assert frame.cd8_exclusion_reason.tolist() == ['valid', 'insufficient_non_tumor_area', 'normalized_ratio_above_one', 'valid']
    assert frame.Density_CD8[0] == pytest.approx(.2)
    np.testing.assert_array_equal(frame.exposure, [10, 20, 30, 40])
    assert frame.row_id.tolist() == [0, 1, 2, 3]


def fixture_frames():
    rng = np.random.default_rng(12)
    frames, arrays = {}, {}
    for split, n, offset in [('train', 80, 0), ('validation', 48, 1000)]:
        x = rng.normal(size=(n, 5))+2
        y = np.where(x[:, 0] > 2, .2, 0.)
        frame = pd.DataFrame(dict(row_id=np.arange(n)+offset, slide=np.repeat(['a', 'b', 'c', 'd'], n//4),
            x=np.tile(np.arange(n//4), 4), y=np.zeros(n), Density_CD8=y,
            Density_Tumor=rng.uniform(0, .8, n), Density_Collagen=x[:, 1]/10, Density_Stroma=x[:, 2]/10))
        arrays[split] = dict(latent=x, pi=rng.uniform(0, 1, (n, 4)), mu=rng.normal(size=(n, 4)), sigma=np.ones((n, 4)))
        frames[split] = frame
    return arrays, frames


def test_common_panels_mixing_and_graph_components():
    _, frames = fixture_frames()
    frame = frames['validation']
    settings = latent_geometry_settings(dict(diagnostic_max_samples=40, resamples=1))
    first, second = g.sample_panels(frame, settings, 12), g.sample_panels(frame, settings, 12)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
        assert len(np.unique(first[key])) == len(first[key])
    counts = frame.iloc[first['balanced']].groupby('slide').size()
    assert counts.nunique() == 1
    # Four well-separated slide components: mixing quality must not look good.
    x = np.repeat(np.arange(4)*100., 12)[:, None]+np.tile(np.arange(12), 4)[:, None]
    summary, records, _ = g.graph_diagnostics(x, frame, 3, 'euclidean', ['Density_CD8'])
    assert summary['connected_components'] == 4
    assert summary['same_slide'] == 1
    assert summary['chance'] == pytest.approx(11/47)
    assert summary['cross_slide_distance_median'] is None
    assert records.row_id.tolist() == frame.row_id.tolist()


def test_eligibility_audit_keeps_mixed_tumor_and_separate_invalid_causes():
    obs = pd.DataFrame(dict(row_id=np.arange(4), batch=['s']*4, slide=['s']*4, lesion=[True, True, True, False],
        valid_Density_CD8=[True, False, False, True], Density_Tumor=[.5, 1., .3, .1],
        cd8_exclusion_reason=['valid', 'insufficient_non_tumor_area', 'normalized_ratio_above_one', 'valid']))
    extracted = dict(row_ids=np.array([3, 2, 0, 1]), batches=np.zeros(4, int), latent=np.ones((4, 2)),
        pi=np.ones((4, 4))*.5, mu=np.ones((4, 4)), sigma=np.ones((4, 4)))
    samples, frame, audit = eligible_samples(extracted, obs, 'validation', ['s'])
    assert frame.row_id.tolist() == [0] and samples['row_ids'].tolist() == [0]
    assert audit.excluded_invalid_cd8.sum() == 2 and audit.excluded_non_lesion.sum() == 1
    assert audit.loc[audit.row_id == 2, 'cd8_exclusion_reason'].iloc[0] == 'normalized_ratio_above_one'
    splits = dict(train=np.arange(4), validation=np.arange(4, 8), test=np.array([], dtype=int))
    validate_splits(splits, 8)
    splits['validation'][0] = 0
    with pytest.raises(ValueError, match='overlapping'):
        validate_splits(splits, 8)


@pytest.mark.parametrize('bad', [dict(n_neighbors=[True]), dict(seeds=[-1]), dict(whitening_power=1.1),
    dict(eigenvalue_floor=0), dict(cd8_weights=[.1]), dict(threshold_quantiles=[.9, .5]),
    dict(guided_geometry='raw_cosine'), dict(layouts=1), dict(unknown=True), dict(max_train_samples=3),
    dict(resample_fraction=np.nan), dict(geometries=['invented'])])
def test_settings_reject_invalid_experiment_contracts(bad):
    with pytest.raises(ValueError):
        latent_geometry_settings(bad)


def test_end_to_end_numerics_and_training_only_layout_fit(tmp_path, monkeypatch):
    import umap
    arrays, frames = fixture_frames()
    calls = []
    class FakeUMAP:
        def __init__(self, **kwargs):
            assert kwargs['min_dist'] == .3 and kwargs['n_neighbors'] == 5
        def fit(self, x):
            assert len(x) == len(frames['train'])
            calls.append('fit')
            return self
        def transform(self, x):
            assert len(x) == len(frames['validation'])
            return x[:, :2]
    monkeypatch.setattr(umap, 'UMAP', FakeUMAP)
    monkeypatch.setattr('dann.latent_geometry.render_layout', lambda *args: None)
    settings = latent_geometry_settings(dict(geometries=['raw_euclidean'], guided_geometry='raw_euclidean',
        cd8_weights=[0., .1], n_neighbors=[5], seeds=[1, 2], diagnostic_max_samples=32, resamples=1))
    thresholds = g.training_thresholds(frames['train'].Density_CD8.to_numpy(), settings['threshold_quantiles'], 1e-6)
    summary = run_experiments(arrays, frames, thresholds, settings, tmp_path,
        ['Density_CD8', 'Density_Tumor', 'Density_Collagen', 'Density_Stroma'], 1e-6)
    assert len(calls) == 4  # Two seeds for discovery and nonzero guidance; zero reuses exactly.
    assert len(summary['completed_arms']) == 3
    np.testing.assert_array_equal(np.load(tmp_path/'raw_euclidean/representation.npz')['values'],
                                  np.load(tmp_path/'cd8_weight_0p0/representation.npz')['values'])
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path/'raw_euclidean/umap_k5_seed1/coordinates.csv'),
                                 pd.read_csv(tmp_path/'cd8_weight_0p0/umap_k5_seed1/coordinates.csv'))
    assert (tmp_path/'heldout_probes.csv').exists()
    assert pd.read_csv(tmp_path/'calibration.csv').query("slide == 'all'").rows.eq(48).all()
    assert (tmp_path/'preprocessing.npz').exists()

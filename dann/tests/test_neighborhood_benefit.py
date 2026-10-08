"""Frozen spatial intervention identities, numerical metrics and resource gates."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from dann.losses import ZILNLoss
from dann.model import BiologyPredictor
from dann.model_components.biology_cnn import CNNBiology
from dann.verification.neighborhood_benefit import (
    GIB, ResourceGuard, adjacent_pairs, bootstrap_delta, donor_pools,
    elementwise_metrics, neighbor_rows, patch_predictions, perturb_sources,
    quadrature_mean, select_donors, summarize,
)


@pytest.fixture
def grid():
    return pd.MultiIndex.from_tuples([(slide, x, y) for slide in ('a', 'b')
                                     for y in range(19) for x in range(19)
                                     if (x, y) != (1, 1)])


def test_donors_are_complete_separated_same_slide_and_reproducible(grid):
    rows = np.arange(0, len(grid), 13)
    pools = donor_pools(grid)
    a = select_donors(grid, rows, pools, np.random.default_rng(42))
    b = select_donors(grid, rows, pools, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    patches = neighbor_rows(grid, a)
    assert (patches >= 0).all()
    for recipient, donor, patch in zip(rows, a, patches):
        assert grid[recipient][0] == grid[donor][0]
        assert max(abs(grid[recipient][i] - grid[donor][i]) for i in (1, 2)) > 6
        assert not set(patch) & set(neighbor_rows(grid, np.array([recipient]))[0])


@pytest.mark.parametrize('condition', ['original', 'replicated', 'rearranged', 'distant'])
def test_center_occupancy_and_permutation_preserved(grid, condition):
    rows = np.array([0, 20, 170, 400])
    original = neighbor_rows(grid, rows)
    donors = select_donors(grid, rows, donor_pools(grid), np.random.default_rng(7))
    patches = neighbor_rows(grid, donors)
    a = perturb_sources(original, condition, np.random.default_rng(81), patches)
    b = perturb_sources(original, condition, np.random.default_rng(81), patches)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[:, 24], rows)
    np.testing.assert_array_equal(a < 0, original < 0)
    if condition == 'rearranged':
        for row, before in zip(a, original):
            np.testing.assert_array_equal(np.sort(row), np.sort(before))
    if condition == 'replicated':
        for i, row in enumerate(a):
            assert (row[row >= 0] == rows[i]).all()


@pytest.mark.parametrize('residual', [False, True])
def test_patch_matches_masked_normal_tile_with_holes_and_boundary(residual):
    torch.manual_seed(16)
    keys = pd.MultiIndex.from_tuples([('slide', x, y) for y in range(13) for x in range(13)
                                     if (x, y) not in ((4, 4), (8, 3))])
    features = np.random.default_rng(9).normal(size=(len(keys), 4)).astype('float32')
    head = CNNBiology(4, 6, channels=7, depth=3, dropout=.2, residual=residual).eval()
    model = SimpleNamespace(biology_predictor=head, num_targets=2, sigma_min=1e-4)
    grid = torch.zeros(1, 4, 13, 13)
    mask = torch.zeros(1, 1, 13, 13)
    for i, (_, x, y) in enumerate(keys):
        grid[0, :, y, x] = torch.from_numpy(features[i])
        mask[0, 0, y, x] = 1
    from dann.model import transform_biology
    with torch.inference_mode():
        out = head(grid, mask)
        raw = torch.stack([out[0, :, y, x] for _, x, y in keys])
        normal = transform_biology(raw, 2, 1e-4)
    expected = np.stack([normal[k].numpy() for k in ('pi_logits', 'mu', 'sigma')], -1)
    actual = patch_predictions(model, features, neighbor_rows(keys, np.arange(len(keys))), torch.device('cpu'))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_pointwise_mlp_is_exactly_invariant_to_every_intervention(grid):
    model = SimpleNamespace(biology_predictor=BiologyPredictor(4, [7], 2, 1e-4, 'gelu', .2, True).eval())
    features = np.random.default_rng(8).normal(size=(len(grid), 4)).astype('float32')
    rows = np.arange(0, len(grid), 23)
    original = neighbor_rows(grid, rows)
    donors = select_donors(grid, rows, donor_pools(grid), np.random.default_rng(8))
    expected = patch_predictions(model, features, original, torch.device('cpu'))
    for condition in ('distant', 'rearranged', 'replicated'):
        sources = perturb_sources(original, condition, np.random.default_rng(20), neighbor_rows(grid, donors))
        np.testing.assert_array_equal(patch_predictions(model, features, sources, torch.device('cpu')), expected)


def test_metrics_match_production_loss_masking_and_positive_r2():
    prediction = np.array([[[.5, -.2, .8], [.1, .4, 1.2]], [[-.7, -.1, .4], [.1, .5, .3]],
                           [[.4, -.3, .5], [.1, -.5, 1.]]], dtype=np.float32)
    y = np.array([[.2, np.nan], [0, .1], [.4, .6]], dtype=np.float32)
    valid = np.isfinite(y)
    config = {'loss': {'logit_epsilon': 1e-5, 'include_normal_constant': False}}
    density, audit = quadrature_mean(prediction)
    assert audit['converged']
    metrics = elementwise_metrics(prediction, density, y, valid, config)
    result = summarize(metrics, np.nan_to_num(y), valid, np.ones(3, bool), np.array([8., 1.]), 1e-5)
    loss = ZILNLoss([8., 1.], 1e-5)(*[torch.from_numpy(prediction[..., i]) for i in range(3)],
        torch.from_numpy(y), torch.from_numpy(valid))
    assert result['weighted_biology_loss'] == pytest.approx(loss.total.item(), abs=5e-7)
    assert all(v[0, 1] == 0 for v in metrics.values())
    assert np.isfinite(result['positive_logit_r2']).all()
    metrics['loss'][:, 1] = 0
    valid[:, 1] = False
    result = summarize(metrics, np.nan_to_num(y), valid, np.ones(3, bool), np.array([8., 1.]), 1e-5)
    assert np.isnan(result['loss'][1])


def test_quadrature_symmetry_and_unresolved_flag():
    prediction = np.array([[[0., 0., 1.], [0., 0., 30.], [0., 5., 50.]]])
    density, audit = quadrature_mean(prediction)
    np.testing.assert_allclose(density[0, :2], .25, atol=1e-14)
    assert audit['nodes'] == 512
    assert not audit['converged']


def test_slide_bootstrap_keeps_pairs_denominators_and_repetition_average():
    delta = np.array([[1., 9.], [3., 9.], [4., 5.]])
    valid = np.array([[True, False], [True, False], [True, True]])
    slides = np.array([0, 0, 1])
    result = bootstrap_delta(delta, valid, slides, np.array([8., 1.]))
    np.testing.assert_allclose(result['delta'], [8/3, 5])
    assert result['lower'][0] == 2
    assert result['upper'][0] == 4
    assert result['weighted_delta'] == pytest.approx(69/25)
    # Averaging row-level loss differences before bootstrapping is linear.
    repeated = bootstrap_delta(((delta-1) + (delta+1))/2, valid, slides, np.array([8., 1.]))
    np.testing.assert_array_equal(result['lower'], repeated['lower'])


def test_pause_thresholds_and_oom_wait_without_training_signals(tmp_path, monkeypatch):
    sequence = iter([15 * GIB, 16 * GIB, 11 * GIB, 12 * GIB])
    sleeps = []
    guard = ResourceGuard(tmp_path/'resources.jsonl', torch.device('cpu'),
        free_bytes=lambda: next(sequence), sleep=sleeps.append)
    guard.wait(initial=True)
    guard.wait()
    assert sleeps == [30, 30]
    training = iter([True, True, False])
    guard.free_bytes = lambda: 20 * GIB
    guard.training = lambda: next(training)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    attempts = []
    def work():
        attempts.append(1)
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError('synthetic')
        return 42
    assert guard.run(work) == 42
    assert len(sleeps) == 4
    assert 'oom_wait_training' in (tmp_path/'resources.jsonl').read_text()


def test_adjacencies_use_only_validation_endpoints_and_no_duplicate_edges(grid):
    rows = np.array([0, 1, 2, 19, 100, 400])
    pairs = adjacent_pairs(grid, rows)
    assert len({tuple(p) for p in pairs}) == len(pairs)
    for a, b in pairs:
        left, right = grid[rows[a]], grid[rows[b]]
        assert left[0] == right[0]
        assert abs(left[1]-right[1]) + abs(left[2]-right[2]) == 1


def test_sparse_full_forward_equals_independent_patch(spatial_config):
    from dann.config import resolve_execution
    from dann.model import AdversarialLatentFusion
    from dann.spatial import SparseInferenceDataset, sparse_inference_collate
    from dann.tiles import SpatialTileDataset, read_grid, tile_collate
    from pathlib import Path
    config = spatial_config
    config['model']['aggregation']['type'] = 'mlp'
    config['model']['heads']['biology']['cnn']['depth'] = 3
    resolve_execution(config)
    model = AdversarialLatentFusion.from_config(config, 2, 4).eval()
    path = Path(config['data']['path'])
    keys, _ = read_grid(path, config['data'])
    rows = np.arange(len(keys))
    data = config['data']
    spectra = SparseInferenceDataset(path, len(keys), data['matrix_key'], data['intensity_transform'],
                                     data.get('intensity_clip_max'), data['nonzero_threshold'])
    tiles = SpatialTileDataset(config, rows, input_path=path)
    try:
        cpu = sparse_inference_collate([spectra[i] for i in rows])
        with torch.inference_mode():
            features = model.encoder(*[cpu[k] for k in ('peak_indices', 'intensities', 'sample_indices', 'peak_counts')]).numpy()
            independent = patch_predictions(model, features, neighbor_rows(keys, rows), torch.device('cpu'))
            for index in range(len(tiles)):
                batch = tile_collate([tiles[index]])
                actual = model(batch)
                expected = independent[batch['row_ids'].numpy()]
                np.testing.assert_allclose(np.stack([actual[k].numpy() for k in ('pi_logits', 'mu', 'sigma')], -1),
                                           expected, atol=2e-6, rtol=2e-6)
    finally:
        spectra.close()
        tiles.close()


# Reuse the sparse synthetic fixture, without touching real datasets.
from dann.tests.test_components import spatial_config  # noqa: E402,F401


def test_metric_bootstrap_recomputes_rmse_and_r2():
    from dann.verification.neighborhood_benefit import bootstrap_metric_changes
    y = np.array([[.2], [.4], [.6], [.8]], dtype=np.float32)
    base = {k: np.ones_like(y) for k in ('brier', 'absolute_error', 'squared_error', 'positive_sse')}
    perturbed = {k: 4 * v for k, v in base.items()}
    result = bootstrap_metric_changes(base, perturbed, y, np.ones_like(y, bool), np.array([0, 0, 1, 1]), 1e-5)
    assert result['rmse']['delta'][0] == 1
    assert result['rmse']['lower'][0] == 1
    assert result['rmse']['upper'][0] == 1
    assert result['brier']['delta'][0] == 3
    assert result['positive_logit_r2']['delta'][0] < 0


def test_report_exports_all_metrics_and_bootstrap_artifacts(tmp_path, grid):
    from dann.verification.neighborhood_benefit import analyze_results, SEEDS
    rows = np.arange(len(grid))
    rng = np.random.default_rng(23)
    y = rng.uniform(.01, .5, (len(rows), 1)).astype('float32')
    valid = np.ones_like(y, bool)
    metadata = SimpleNamespace(targets=y, target_valid_mask=valid)
    config = {'data': {'target_columns': ['Density_CD8']},
              'loss': {'target_weights': [1.], 'logit_epsilon': 1e-5, 'include_normal_constant': False},
              'training': {'learning_rate': .001}}
    configs = {name: config for name in ('cnn', 'mlp')}
    manifests = {'cnn': {'epoch': 30}, 'mlp': {'epoch': 33}}
    frame = grid.to_frame(index=False)
    frame.columns = ['batch', 'x', 'y']
    frame.insert(0, 'row_id', rows)
    frame['group'] = np.where((neighbor_rows(grid, rows) >= 0).all(1), 'interior', 'boundary')
    conditions = [('original', None), ('replicated', None)] + [(c, s) for c in ('distant', 'rearranged') for s in SEEDS]
    conditions = [(c, s, c if s is None else f'{c}_{s}', None) for c, s in conditions]
    for name in ('cnn', 'mlp'):
        directory = tmp_path/name
        directory.mkdir()
        for condition, seed, label, _ in conditions:
            values = np.zeros((len(rows), 1, 3), dtype=np.float32)
            values[..., 1] = -2
            values[..., 2] = 1
            if condition != 'original' and name == 'cnn':
                values[..., 0] += .1
            np.save(directory/f'{label}_predictions.npy', values)
    analyze_results(configs, manifests, metadata, grid, rows, conditions, frame, tmp_path)
    assert (tmp_path/'effect_sizes.png').exists()
    effects = pd.read_csv(tmp_path/'paired_effects.csv')
    assert effects.loc[effects.model == 'mlp', 'delta'].eq(0).all()
    assert effects.loc[effects.model == 'cnn', 'lower'].gt(0).all()
    changes = pd.read_csv(tmp_path/'paired_metric_effects.csv')
    assert set(changes.metric) == {'brier', 'mae', 'rmse', 'positive_logit_r2'}
    assert '|---|---:|---:|---:|\n| cnn' in (tmp_path/'report.md').read_text()
    assert (tmp_path/'completion.json').exists()

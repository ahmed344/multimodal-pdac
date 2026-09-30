"""Contracts for modular spatial composition and frozen sparse supervision."""
import copy
import itertools
import math
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from dann.config import load_config, resolve_execution, execution_settings
from dann.data_loader import create_data_bundle
from dann.model import AdversarialLatentFusion
from dann.tiles import SpatialTileDataset, row_identity, tile_collate, read_grid
from dann.losses import ZILNLoss
from dann.train import _format_epoch_summary, load_training_checkpoint, save_checkpoint


@pytest.fixture
def spatial_config(tmp_path):
    rng = np.random.default_rng(31)
    keys = [(slide, x, y) for slide in ('a', 'b') for y in range(9) for x in range(9)
            if (x, y) not in ((3, 4), (4, 4), (7, 2))]
    rng.shuffle(keys)
    obs = pd.DataFrame(keys, columns=['batch', 'x', 'y'], index=['duplicate']*len(keys))
    for name in ('Density_CD8', 'Density_Tumor', 'Density_Collagen', 'Density_Stroma'):
        obs[name] = rng.uniform(0, .4, len(keys))
    obs.loc[obs.x == 0, 'Density_CD8'] = 0
    obs.loc[obs.x == 1, 'Density_Tumor'] = 1  # excluded CD8, other targets remain valid
    matrix = rng.uniform(0, 3, (len(keys), 5)).astype('float32')
    matrix[matrix < 1] = 0
    matrix[0] = 0
    tissue = ad.AnnData(csr_matrix(matrix), obs=obs)
    tissue.var_names = ['100', '101', '102', '103', '104']
    tissue.write_h5ad(tmp_path/'tissue.h5ad')
    tissue[np.arange(0, len(keys), 2)].copy().write_h5ad(tmp_path/'labels.h5ad')
    config = load_config(Path('dann/config.yaml'))
    config['data'].update(path=str(tmp_path/'labels.h5ad'), context_path=str(tmp_path/'tissue.h5ad'),
                          train_fraction=.6, validation_fraction=.2, test_fraction=.2)
    # Spatial tests must not inherit editable architecture selectors from config.yaml.
    config['model']['aggregation']['type'] = 'cnn'
    config['model']['heads']['biology']['type'] = 'cnn'
    config['model']['heads']['discriminator']['type'] = 'mlp'
    config['model'].update(num_peaks=5, latent_dim=6, dropout=0., spectral_peak_budget=9)
    config['model']['spectral_encoder']['deep_sets'].update(embedding_dim=4, peak_hidden_dims=[7], peak_output_dim=6)
    for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
        group['mlp']['hidden_dims'] = [7]
        group['cnn'].update(channels=5, depth=2, dropout=0.)
    config['training']['pixel'].update(batch_size=12, validation_batch_size=12, num_workers=0)
    config['training']['spatial'].update(core_size=4, tiles_per_batch=2, num_workers=0)
    config['training'].update(device='cpu', pin_memory=False)
    config['analysis'].update(num_workers=0, density_mc_samples=7)
    config['results']['root'] = str(tmp_path/'results')
    return resolve_execution(config)


@pytest.mark.parametrize('spatial', [False, True])
def test_test_loader_uses_execution_mode_batch_units(spatial_config, spatial):
    config = spatial_config
    for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
        group['type'] = 'cnn' if spatial else 'mlp'
    config['analysis']['batch_size'] = 17
    resolve_execution(config)
    bundle = create_data_bundle(config)
    expected = config['training']['tiles_per_batch'] if spatial else 17
    assert bundle.loaders['test'].batch_size == expected
    seen = torch.cat([batch['row_ids'] for batch in bundle.loaders['test']]).tolist()
    assert sorted(seen) == sorted(bundle.split_indices['test'].tolist())
    for dataset in bundle.datasets.values():
        dataset.close()


@pytest.mark.parametrize('choices', list(itertools.product(('mlp','cnn'), repeat=3)))
def test_combinations_train_reload_and_gradients(spatial_config, choices, tmp_path):
    c = spatial_config
    for group, choice in zip([c['model']['aggregation'], *c['model']['heads'].values()], choices):
        group['type'] = choice
    resolve_execution(c)
    bundle = create_data_bundle(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    batch = next(iter(bundle.loaders['train']))
    outputs = model(batch)
    loss = ZILNLoss.from_config(c)(outputs['pi_logits'], outputs['mu'], outputs['sigma'],
                                  batch['targets'], batch['target_valid_mask']).total
    loss += torch.nn.functional.cross_entropy(outputs['batch_logits'], batch['batches'])
    loss.backward()
    for component in (model.encoder.peak_mlp, model.encoder.embedding, model.encoder.aggregation_mlp,
                      model.biology_predictor, model.batch_discriminator):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())
    optimizer.step()
    model.eval()
    expected = model(batch)
    path = tmp_path/'checkpoint.pt'
    save_checkpoint(path, model, optimizer, 0, c, bundle, 1.)
    restored = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    opt = torch.optim.AdamW(restored.parameters())
    load_training_checkpoint(path, restored, opt, torch.device('cpu'), c)
    for name, value in restored(batch).items():
        torch.testing.assert_close(value, expected[name])
    assert set(batch['row_ids'].tolist()).issubset(set(bundle.split_indices['train']))
    if model.spatial:
        assert batch['peak_counts'].numel() > len(batch['row_ids'])
    for dataset in bundle.datasets.values():
        dataset.close()


def predict_tiles(config, model, rows):
    dataset = SpatialTileDataset(config, rows, input_path=config['data']['context_path'])
    results = {}
    with torch.no_grad():
        for item in range(len(dataset)):
            batch = tile_collate([dataset[item]])
            outputs = model(batch)
            for i, row in enumerate(batch['row_ids'].tolist()):
                results[row] = {key: value[i].clone() for key, value in outputs.items()}
    dataset.close()
    return results


@pytest.mark.parametrize('choices', list(itertools.product(('mlp','cnn'), repeat=3))[1:])
def test_combined_receptive_field_matches_reference(spatial_config, choices):
    c = spatial_config
    for group, choice in zip([c['model']['aggregation'], *c['model']['heads'].values()], choices):
        group['type'] = choice
    resolve_execution(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    rows = np.arange(156)
    small = predict_tiles(c, model, rows)
    c['training']['core_size'] = 32
    reference = predict_tiles(c, model, rows)
    for row in rows:
        for name in reference[row]:
            torch.testing.assert_close(small[row][name], reference[row][name], atol=2e-6, rtol=2e-5)
    capped = predict_tiles(c, model, rows[:7])
    for row in capped:
        for name in capped[row]:
            torch.testing.assert_close(capped[row][name], reference[row][name])


@pytest.mark.parametrize('dropout', [0., .3])
def test_checkpoint_microbatches_values_gradients_rng(spatial_config, dropout):
    c = spatial_config
    c['model']['dropout'] = dropout
    a = AdversarialLatentFusion.from_config(c, 2, 4).encoder
    b = copy.deepcopy(a)
    counts = torch.tensor([3, 0, 2, 4, 1])
    indices = torch.arange(10) % 5
    intensities = torch.linspace(.1, 1., 10)
    samples = torch.repeat_interleave(torch.arange(5), counts)
    args = (indices, intensities, samples, counts)
    torch.manual_seed(5)
    value_a = a.microbatched(*args, peak_budget=5, checkpointing=True)
    value_a.square().sum().backward()
    rng_a = torch.get_rng_state()
    torch.manual_seed(5)
    value_b = b.microbatched(*args, peak_budget=5, checkpointing=False)
    value_b.square().sum().backward()
    torch.testing.assert_close(value_a, value_b)
    assert torch.equal(rng_a, torch.get_rng_state())
    assert torch.equal(value_a[1], torch.zeros(6))
    for pa, pb in zip(a.parameters(), b.parameters()):
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad)
    if dropout == 0:
        from dann.model_components.deep_sets import DeepSetsEncoder
        b.zero_grad()
        whole = DeepSetsEncoder.forward(b, *args)
        torch.testing.assert_close(value_a, whole)
        whole.square().sum().backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            if pa.grad is not None:
                torch.testing.assert_close(pa.grad, pb.grad, atol=2e-6, rtol=2e-5)


def test_grid_validation_and_supervision_isolation(spatial_config):
    c = spatial_config
    bundle = create_data_bundle(c)
    dataset = bundle.datasets['train']
    original = tile_collate([dataset[0]])
    slide = dataset.tiles[0][0][0]
    tissue = ad.read_h5ad(c['data']['context_path'])
    other = (tissue.obs.batch != slide).to_numpy()
    tissue.X[other] *= 10
    dataset.close()
    tissue.write_h5ad(c['data']['context_path'])
    labels = ad.read_h5ad(c['data']['path'])
    held_out = np.concatenate([bundle.split_indices['validation'], bundle.split_indices['test']])
    labels.obs.iloc[held_out, labels.obs.columns.get_loc('Density_CD8')] = .9
    labels.write_h5ad(c['data']['path'])
    new_bundle = create_data_bundle(c, bundle.split_indices)
    changed = tile_collate([new_bundle.datasets["train"][0]])
    for d in new_bundle.datasets.values(): d.close()
    for key in original:
        torch.testing.assert_close(original[key], changed[key])
    for d in bundle.datasets.values():
        d.close()
    tissue.obs['x'] = tissue.obs['x'].astype(float)
    tissue.obs.iloc[0, tissue.obs.columns.get_loc('x')] = .5
    tissue.write_h5ad(c['data']['context_path'])
    with pytest.raises(ValueError, match='integers'):
        read_grid(Path(c['data']['context_path']), c['data'])


@pytest.mark.parametrize("choices", list(itertools.product(("mlp", "cnn"), repeat=3)))
def test_spatial_export_order_cap_and_cleanup(spatial_config, tmp_path, choices):
    from dann.spatial import run_inference
    import pyarrow.parquet as pq
    c = spatial_config
    for group, choice in zip([c['model']['aggregation'], *c['model']['heads'].values()], choices):
        group['type'] = choice
    resolve_execution(c)
    bundle = create_data_bundle(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    path = tmp_path/'checkpoint.pt'
    save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), 0, c, bundle, 1.)
    for name, cap in [('full', None), ('cap', 7)]:
        run_inference(Path(c['data']['context_path']), path, tmp_path/f'{name}.parquet',
                      latent_output_path=tmp_path/f'{name}-latent.parquet', max_rows=cap)
    for suffix in ('', '-latent'):
        full = pq.read_table(tmp_path/f'full{suffix}.parquet').to_pandas().iloc[:7]
        cap = pq.read_table(tmp_path/f'cap{suffix}.parquet').to_pandas()
        pd.testing.assert_frame_equal(full, cap, check_exact=False, atol=2e-6, rtol=2e-5)
    if model.spatial:
        support = pq.read_table(tmp_path/'full.support.parquet').to_pandas()
        assert len(support) == 156
        if execution_settings(c)['biology_radius']:
            assert support.biology_support.min() < 1
    assert not list(tmp_path.glob('.dann-stage-*'))
    for d in bundle.datasets.values():
        d.close()


def test_gradient_reversal_only_discriminator_upstream(spatial_config):
    c = spatial_config
    c['model']['heads']['discriminator']['type'] = 'cnn'
    model = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    bundle = create_data_bundle(c)
    batch = next(iter(bundle.loaders['train']))
    gradients = []
    for strength in (1., -1.):
        model.zero_grad()
        output = model(batch, grl_strength=strength)
        output['batch_logits'].square().sum().backward()
        gradients.append({name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None})
        assert all(p.grad is None for p in model.biology_predictor.parameters())
    for name in gradients[0]:
        sign = -1 if name.startswith('encoder.') else 1
        torch.testing.assert_close(gradients[0][name], sign*gradients[1][name])
    model.zero_grad()
    model(batch, grl_strength=1.)['mu'].sum().backward()
    assert all(p.grad is None for p in model.batch_discriminator.parameters())
    for d in bundle.datasets.values(): d.close()


def test_frozen_splits_and_incompatible_resume(spatial_config, tmp_path):
    c = spatial_config
    bundle = create_data_bundle(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path/'checkpoint.pt'
    save_checkpoint(path, model, optimizer, 0, c, bundle, 1.)
    payload = torch.load(path, weights_only=False)
    c['training']['seed'] += 21
    restored = create_data_bundle(c, payload['split_indices'], payload['data_identity'])
    for name in bundle.split_indices:
        np.testing.assert_array_equal(bundle.split_indices[name], restored.split_indices[name])
    bad = copy.deepcopy(c)
    bad['model']['heads']['discriminator']['type'] = 'cnn'
    with pytest.raises(ValueError, match='components'):
        load_training_checkpoint(path, model, optimizer, torch.device('cpu'), bad)
    bad = copy.deepcopy(c)
    bad['data']['target_columns'] = list(reversed(bad['data']['target_columns']))
    with pytest.raises(ValueError, match='target'):
        load_training_checkpoint(path, model, optimizer, torch.device('cpu'), bad)
    payload['optimizer_parameter_names'] = list(reversed(payload['optimizer_parameter_names']))
    torch.save(payload, path)
    with pytest.raises(ValueError, match='ordering'):
        load_training_checkpoint(path, model, optimizer, torch.device('cpu'), c)
    for b in (bundle, restored):
        for d in b.datasets.values(): d.close()
    adata = ad.read_h5ad(c['data']['path'])
    adata[np.arange(len(adata)-1, -1, -1)].copy().write_h5ad(c['data']['path'])
    with pytest.raises(ValueError, match='identities'):
        create_data_bundle(c, payload['split_indices'], payload['data_identity'])


def test_configuration_routes_and_invalid_selectors(spatial_config):
    c = spatial_config
    roots = set()
    for choices in itertools.product(('mlp', 'cnn'), repeat=3):
        for group, choice in zip([c['model']['aggregation'], *c['model']['heads'].values()], choices):
            group['type'] = choice
        resolve_execution(c)
        roots.add(c['results']['run_dir'])
        assert c['analysis']['checkpoint'] == str(Path(c['training']['output_dir'])/'best.pt')
        assert c['latent_variance']['latents'] == c['spatial']['latents']
    assert len(roots) == 8
    c['model']['aggregation']['cnn']['depth'] = 0
    with pytest.raises(ValueError, match='positive'):
        resolve_execution(c)
    c['model']['aggregation']['cnn']['depth'] = 2
    c['model']['aggregation']['type'] = 'dnn'
    with pytest.raises(ValueError, match='mlp or cnn'):
        resolve_execution(c)


def test_duplicate_grid_and_feature_order_rejected(spatial_config):
    c = spatial_config
    tissue = ad.read_h5ad(c['data']['context_path'])
    tissue.var_names = tissue.var_names[::-1]
    tissue.write_h5ad(c['data']['context_path'])
    bundle = create_data_bundle(c)
    for dataset in bundle.datasets.values():
        dataset.close()
    tissue.var_names = tissue.var_names[::-1]
    for key in ('batch', 'x', 'y'):
        tissue.obs.iloc[1, tissue.obs.columns.get_loc(key)] = tissue.obs.iloc[0][key]
    tissue.write_h5ad(c['data']['context_path'])
    with pytest.raises(ValueError, match='Duplicate'):
        read_grid(Path(c['data']['context_path']), c['data'])


def test_disk_staging_cleans_failure(spatial_config, tmp_path):
    from dann.tiles import ordered_spatial_predictions
    c = spatial_config
    model = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    dataset = SpatialTileDataset(c, np.arange(5), input_path=c['data']['context_path'])
    batch = tile_collate([dataset[0]])
    with pytest.raises(ValueError, match='Duplicate'):
        list(ordered_spatial_predictions(model, [batch, batch], 5, torch.device('cpu'), tmp_path))
    assert not list(tmp_path.glob('.dann-stage-*'))
    dataset.close()


def test_cd8_selection_is_mean_over_valid_observations():
    from dann.train import run_epoch
    class Fixed(torch.nn.Module):
        def forward(self, batch, grl_strength=1.):
            shape = batch['targets'].shape
            return {'pi_logits': torch.zeros(shape), 'mu': batch['mu'],
                    'sigma': torch.ones(shape), 'batch_logits': torch.zeros((shape[0], 2))}
    targets = torch.tensor([[0., .1, .3, .2], [.9, .2, .1, .4], [.2, .5, .2, .3]])
    valid = torch.ones_like(targets, dtype=torch.bool)
    valid[1,0] = False
    mu = torch.tensor([[.1, 0., 0., 0.], [30., 0., 0., 0.], [.4, 0., 0., 0.]])
    batches = [{'targets': targets[a:b], 'mu': mu[a:b], 'target_valid_mask': valid[a:b],
                'batches': torch.zeros(b-a, dtype=torch.long)} for a,b in ((0,1),(1,3))]
    metric = run_epoch(Fixed(), batches, ZILNLoss([8.,4.,2.,1.]), torch.device('cpu'),
                       5., 1., None, None, 0, 1, 'constant', .25, 10., None)
    expected = ZILNLoss([1.])(torch.zeros(3,1), mu[:,:1], torch.ones(3,1), targets[:,:1], valid[:,:1]).total
    assert metric['cd8_loss'] == pytest.approx(float(expected))


def test_backed_read_ignores_duplicate_observation_names(tmp_path):
    obs = pd.DataFrame(
        {"batch": ["a", "a"], "x": [0, 1], "y": [0, 0]},
        index=["spot", "spot"],
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Observation names are not unique\.",
            category=UserWarning,
        )
        adata = ad.AnnData(csr_matrix((2, 1), dtype=np.float32), obs=obs)
        adata.var_names = ["1.0"]
        path = tmp_path / "duplicate-names.h5ad"
        adata.write_h5ad(path)
    data = {"batch_column": "batch", "x_column": "x", "y_column": "y"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        keys, features = read_grid(path, data)
        identity = row_identity(path, data)
    assert not any("not unique" in str(item.message) for item in caught)
    assert len(keys) == 2
    assert features.tolist() == ["1.0"]
    assert identity["rows"] == 2


def test_epoch_summary_aligns_each_target():
    columns = ["Density_CD8", "Density_Tumor", "Density_Collagen", "Density_Stroma"]
    train = {
        "biology_loss": 1.234567,
        "batch_loss": 0.501234,
        "hurdle_loss": 0.401234,
        "positive_loss": 0.833333,
        "cd8_loss": 1.111111,
        "batch_accuracy": 0.8123,
        "hurdle_acc": 0.71,
        "mean_r2": 0.21,
        "hurdle_acc_cd8": 0.72,
        "mean_r2_cd8": 0.22,
        "hurdle_acc_tumor": 0.70,
        "mean_r2_tumor": 0.17,
        "hurdle_acc_collagen": 0.69,
        "mean_r2_collagen": 0.16,
        "hurdle_acc_stroma": 0.67,
        "mean_r2_stroma": 0.15,
    }
    validation = {
        "biology_loss": 1.345678,
        "batch_loss": 0.512345,
        "hurdle_loss": 0.412345,
        "positive_loss": 0.933333,
        "cd8_loss": 1.222222,
        "batch_accuracy": 0.7901,
        "hurdle_acc": 0.68,
        "mean_r2": 0.18,
        "hurdle_acc_cd8": 0.70,
        "mean_r2_cd8": 0.20,
        "hurdle_acc_tumor": 0.68,
        "mean_r2_tumor": 0.15,
        "hurdle_acc_collagen": 0.66,
        "mean_r2_collagen": 0.14,
        "hurdle_acc_stroma": 0.64,
        "mean_r2_stroma": 0.13,
    }
    summary = _format_epoch_summary(12, 300, train, validation, columns)
    expected = "\n".join([
        "epoch=12/300 | biology_loss = 1.234567, 1.345678 | batch_loss = 0.501234, 0.512345 | "
        "hurdle_loss = 0.401234, 0.412345 | mean_loss = 0.833333, 0.933333 | "
        "cd8_loss = 1.111111, 1.222222",
        "  batch_acc = 0.8123, 0.7901 | hurdle_acc = 0.7100, 0.6800 | mean_r2 = 0.2100, 0.1800",
        "  [CD8]      hurdle_acc = 0.7200, 0.7000 | mean_r2 = 0.2200, 0.2000",
        "  [Tumor]    hurdle_acc = 0.7000, 0.6800 | mean_r2 = 0.1700, 0.1500",
        "  [Collagen] hurdle_acc = 0.6900, 0.6600 | mean_r2 = 0.1600, 0.1400",
        "  [Stroma]   hurdle_acc = 0.6700, 0.6400 | mean_r2 = 0.1500, 0.1300",
    ])
    assert summary == expected
    assert "step=" not in summary
    missing = dict(train)
    missing["mean_r2_stroma"] = math.nan
    nan_summary = _format_epoch_summary(1, 2, missing, validation, columns)
    assert "mean_r2 = nan, 0.1300" in nan_summary.splitlines()[-1]


def test_context_requires_csr(spatial_config):
    c = spatial_config
    tissue = ad.read_h5ad(c['data']['path'])
    tissue.X = tissue.X.tocsc()
    tissue.write_h5ad(c['data']['path'])
    with pytest.raises(TypeError, match='CSR'):
        create_data_bundle(c)

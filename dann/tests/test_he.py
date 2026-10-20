"""Raw H&E supervision, compatibility, alignment, and bounded optional exports."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from dann.checkpoints import architecture_contract, validate_checkpoint
from dann.config import execution_settings, resolve_execution
from dann.data_loader import create_data_bundle
from dann.execution import evaluation_groups, build_validation_features, validation_execution
from dann.he import HETargetReader, HEWriter, extract_with_he, he_mse
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.tests.test_components import spatial_config
from dann.tests.test_execution import enable, epoch
from dann.tests.test_spatial_execution import checkpoint, options
from dann.tiles import tile_collate
from dann.train import load_training_checkpoint, resume_training_config


@pytest.fixture
def he_config(spatial_config):
    c = spatial_config
    he = copy.deepcopy(c['model']['heads']['biology'])
    he.update(enabled=True, output_dim=7)
    c['model']['heads']['he'] = he
    with h5py.File(c['data']['path'], 'r+') as f:
        n = len(f['X/indptr']) - 1
        f['obsm'].create_dataset('hes_features', data=np.arange(n*7, dtype=np.float32).reshape(n, 7)/100-2)
    return resolve_execution(c)


def test_exact_mse_and_gradient():
    x = torch.tensor([[1., -2.], [3., 4.]], requires_grad=True)
    target = torch.tensor([[0., 2.], [2., -1.]])
    loss = he_mse(x, target)
    assert loss.item() == (1 + 16 + 1 + 25) / 4
    loss.backward()
    torch.testing.assert_close(x.grad, 2*(x.detach()-target)/4)
    with pytest.raises(ValueError, match='matching'):
        he_mse(x, target[:1])


@pytest.mark.parametrize('kind', ['mlp', 'cnn', 'gatv2'])
def test_heads_he_only_spatial_and_encoder_gradients(he_config, kind):
    c = he_config
    for g in [c['model']['aggregation'], *c['model']['heads'].values()]:
        g['type'] = 'mlp'
    he = c['model']['heads']['he']
    he['type'] = kind
    he['cnn']['depth'] = 3
    he['gatv2'].update(depth=3, neighbor_radius=2.)
    resolve_execution(c)
    bundle = create_data_bundle(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    batch = next(iter(bundle.loaders['train']))
    result = model(batch)
    assert result['he_prediction'].shape == batch['he_targets'].shape
    assert model.spatial == (kind != 'mlp')
    assert model.halo == (0 if kind == 'mlp' else 3 if kind == 'cnn' else 6)
    assert execution_settings(c)['he_radius'] == model.halo
    he_mse(result['he_prediction'], batch['he_targets']).backward()
    assert model.encoder.embedding.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.batch_discriminator.parameters())
    assert 'he_prediction' not in model(batch, predict_he=False)
    # Outputs are unrestricted, including negative raw values.
    with torch.no_grad():
        for p in model.he_predictor.parameters():
            p.zero_()
        list(model.he_predictor.parameters())[-1].fill_(-3)
    torch.testing.assert_close(model(batch)['he_prediction'], torch.full_like(batch['he_targets'], -3))


def test_reader_schema_finiteness_order_and_threads(he_config):
    path = Path(he_config['data']['path'])
    with h5py.File(path, 'r') as f:
        values = f['obsm/hes_features'][:]
    reader = HETargetReader(path, 'hes_features', len(values), 7)
    rows = np.array([8, 2, 8, 0])
    with ThreadPoolExecutor(3) as pool:
        for observed in pool.map(reader.read, [rows]*6):
            np.testing.assert_array_equal(observed, values[rows])
    with pytest.raises(ValueError, match='shape'):
        HETargetReader(path, 'hes_features', len(values), 6)
    with pytest.raises(ValueError, match='Missing'):
        HETargetReader(path, 'absent', len(values), 7)
    with h5py.File(path, 'r+') as f:
        f['obsm/hes_features'][2, 1] = np.nan
        f['obsm'].create_dataset('strings', data=np.full((len(values), 7), b'x'))
    np.testing.assert_array_equal(reader.read(np.array([0])), values[:1])
    with pytest.raises(ValueError, match=r'positions: \[2\]'):
        reader.read(rows)
    with pytest.raises(TypeError, match='numeric'):
        HETargetReader(path, 'strings', len(values), 7)


def test_prepared_split_core_alignment_regrouping_and_epoch_mse(he_config):
    c = he_config
    c['model']['heads']['he']['type'] = 'gatv2'
    c['model']['heads']['he']['gatv2'].update(depth=3, neighbor_radius=2.)
    c['training']['spatial'].update(supervised_rows_per_batch=5)
    enable(c, validation_tiles=3, validation_feature_cache=True, validation_gat_tiles=2)
    bundle = create_data_bundle(c)
    with h5py.File(c['data']['path'], 'r') as f:
        expected = f['obsm/hes_features'][:]
    seen = []
    for batch in bundle.loaders['train']:
        rows = batch['row_ids'].numpy()
        seen.extend(rows)
        np.testing.assert_array_equal(batch['he_targets'].numpy(), expected[rows])
        assert len(rows) <= 5
    assert sorted(seen) == sorted(bundle.split_indices['train'])
    for item in bundle.datasets['train'].prepared_items:
        assert 'he_targets' not in item
    model = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    sse, count = 0., 0
    with torch.inference_mode():
        for _, batch, output in evaluation_groups(model, bundle.loaders['validation'], torch.device('cpu')):
            np.testing.assert_array_equal(batch['he_targets'], expected[batch['row_ids'].numpy()])
            sse += (output['he_prediction'].double()-batch['he_targets'].double()).square().sum().item()
            count += batch['he_targets'].numel()
    metrics = epoch(model, bundle.loaders['validation'], ZILNLoss.from_config(c))
    assert metrics['he_mse'] == pytest.approx(sse/count)
    assert metrics['he_weighted_loss'] == metrics['he_mse']
    validation_targets = expected[bundle.split_indices['validation']].astype(np.float64)
    centered_sum = np.square(validation_targets-validation_targets.mean(axis=0)).sum()
    assert metrics['he_r2'] == pytest.approx(1-sse/centered_sum)
    assert metrics['total_loss'] == pytest.approx(5*metrics['biology_loss']+metrics['batch_loss']+metrics['he_mse'])


def test_checkpoint_roundtrip_disabled_legacy_and_incompatibilities(he_config, tmp_path):
    c = he_config
    path = checkpoint(c, tmp_path)
    payload = torch.load(path, weights_only=False)
    assert payload['architecture']['version'] == 6
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    optimizer = torch.optim.AdamW(model.parameters())
    load_training_checkpoint(path, model, optimizer, torch.device('cpu'), c)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, payload['model_state'][key])
    for key, value in [('output_dim', 8), ('enabled', False), ('type', 'mlp')]:
        bad = copy.deepcopy(c)
        bad['model']['heads']['he'][key] = value
        with pytest.raises(ValueError, match='incompatible'):
            validate_checkpoint(payload, bad)
        with pytest.raises(ValueError, match='fresh training'):
            resume_training_config(c, bad)
    bad = copy.deepcopy(c)
    bad['data']['he_features_key'] = 'different'
    with pytest.raises(ValueError, match='incompatible'):
        validate_checkpoint(payload, bad)
    disabled = copy.deepcopy(c)
    disabled['model']['heads']['he'] = {'enabled': False}
    legacy = copy.deepcopy(disabled)
    legacy['model']['heads'].pop('he')
    assert architecture_contract(disabled) == architecture_contract(legacy)
    torch.manual_seed(12)
    a = AdversarialLatentFusion.from_config(disabled, 2, 4)
    torch.manual_seed(12)
    b = AdversarialLatentFusion.from_config(legacy, 2, 4)
    for k in a.state_dict():
        torch.testing.assert_close(a.state_dict()[k], b.state_dict()[k], rtol=0, atol=0)
    validate_checkpoint({'config': legacy, 'architecture': architecture_contract(legacy)}, disabled, a)
    with h5py.File(c['data']['path'], 'r+') as f:
        del f['obsm/hes_features']
    create_data_bundle(disabled)  # Disabled requires no H&E schema or labels.


@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('export', [False, True])
def test_analysis_metrics_and_streamed_export(he_config, tmp_path, prepared, export):
    c = he_config
    if prepared:
        enable(c, validation_feature_cache=True)
    c['analysis']['export_he_predictions'] = export
    bundle = create_data_bundle(c)
    loader = bundle.loaders['validation']
    model = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    extractor = None
    with torch.inference_mode():
        if prepared:
            from functools import partial
            from dann.analysis_execution import extract_prepared
            from dann.spatial_execution import InferenceProgress
            cache = build_validation_features(model, loader, torch.device('cpu'))
            extractor = partial(extract_prepared, cache=cache, progress=InferenceProgress())
        arrays = extract_with_he(model, loader, torch.device('cpu'), c, tmp_path, 'validation', extractor)
    metrics = json.loads((tmp_path/'he/validation_metrics.json').read_text())
    assert metrics['mse'] == pytest.approx(arrays['he_squared_error'].sum()/arrays['he_element_count'].sum())
    assert 'he_prediction' not in arrays
    output = tmp_path/'he/validation_predictions.parquet'
    assert output.exists() == export
    if export:
        table = pq.read_table(output)
        assert table['row_position'].to_pylist() == arrays['row_ids'].tolist()
        assert table['obs_name'].to_pylist() == ['duplicate']*len(arrays['row_ids'])
        predictions = np.stack(table['he_prediction'].to_pylist())
        with h5py.File(c['data']['path'], 'r') as f:
            targets = f['obsm/hes_features'][:][arrays['row_ids']]
        assert metrics['mse'] == pytest.approx(np.square(predictions-targets).mean())


@pytest.mark.parametrize('kind', ['mlp', 'gatv2'])
def test_spatial_export_without_observed_he_and_skip_branch(he_config, tmp_path, kind):
    from dann.spatial import run_inference
    c = he_config
    for g in [c['model']['aggregation'], *c['model']['heads'].values()]:
        g['type'] = 'mlp'
    c['model']['heads']['he']['type'] = kind
    resolve_execution(c)
    path = checkpoint(c, tmp_path)
    source = Path(c['data']['context_path'])  # Has no obsm H&E targets.
    for export in (False, True):
        output = tmp_path/str(export)/'predictions.parquet'
        run_inference(source, path, output, max_rows=13,
                      execution_overrides=options(export_he_predictions=export))
        he_path = output.parent/'he_predictions.parquet'
        assert he_path.exists() == export
        if export:
            table = pq.read_table(he_path)
            assert table['row_position'].to_pylist() == list(range(13))
            assert np.isfinite(np.stack(table['he_prediction'].to_pylist())).all()


def test_writer_rejects_incomplete_rows(he_config, tmp_path):
    path = tmp_path/'bad.parquet'
    with pytest.raises(ValueError, match='coverage'):
        with HEWriter(path, Path(he_config['data']['path']), 7, np.array([2, 1])) as writer:
            writer.write(np.array([1]), np.zeros((1, 7)))
    assert not path.exists()
    assert not list(tmp_path.glob('.*.tmp'))


@pytest.mark.parametrize('setting,value', [('output_dim', 0), ('output_dim', True),
                                           ('enabled', 'yes'), ('type', 'unknown')])
def test_he_configuration_errors(he_config, setting, value):
    he_config['model']['heads']['he'][setting] = value
    with pytest.raises(ValueError):
        resolve_execution(he_config)


@pytest.mark.parametrize('weight', [-1, float('nan'), True])
def test_he_weight_validation(he_config, weight):
    he_config['loss']['he_weight'] = weight
    with pytest.raises(ValueError, match='he_weight'):
        resolve_execution(he_config)


def test_pixel_worker_alignment_and_mse_batch_size_independence(he_config):
    from torch.utils.data import DataLoader
    from dann.data_loader import sparse_collate
    c = he_config
    for g in [c['model']['aggregation'], *c['model']['heads'].values()]:
        g['type'] = 'mlp'
    resolve_execution(c)
    bundle = create_data_bundle(c)
    dataset = bundle.datasets['validation']
    model = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    metrics = []
    with h5py.File(c['data']['path'], 'r') as f:
        targets = f['obsm/hes_features'][:]
    for size in (3, 7):
        loader = DataLoader(dataset, batch_size=size, num_workers=2, collate_fn=sparse_collate)
        for batch in loader:
            np.testing.assert_array_equal(batch['he_targets'], targets[batch['row_ids'].numpy()])
        metrics.append(epoch(model, loader, ZILNLoss.from_config(c)))
    for key in ('he_mse', 'he_r2'):
        assert metrics[0][key] == pytest.approx(metrics[1][key], rel=1e-7)


def test_optimized_analysis_cache_retains_mse_and_exports_on_request(he_config, tmp_path, monkeypatch):
    from dann import analyze, analysis_execution
    c = he_config
    enable(c)
    c['analysis'].update(split='validation', ziln_scatter_split='validation',
                         execution={'enabled': True, 'render_workers': 0})
    path = checkpoint(c, tmp_path)
    monkeypatch.setattr(analyze, 'fit_latent_umap', lambda latent, config: latent[:, :2].copy())
    def artifacts(*args, **kwargs):
        (args[9] / 'complete.json').write_text('{}')
    monkeypatch.setattr(analysis_execution, 'write_artifacts', artifacts)
    output = tmp_path/'analysis'
    for repeat in range(2):
        analyze.run_analysis(c, path, output_override=output)
        assert json.loads((output/'he/validation_metrics.json').read_text())['mse'] > 0
        assert not (output/'he/validation_predictions.parquet').exists()
        if repeat:
            assert 'extracted-validation' in json.loads((output/'analysis_execution.json').read_text())['cache_hits']
    c['analysis']['export_he_predictions'] = True
    analyze.run_analysis(c, path, output_override=output)
    table = pq.read_table(output/'he/validation_predictions.parquet')
    assert len(table) == len(torch.load(path, weights_only=False)['split_indices']['validation'])


def test_he_memory_estimate_tracks_optional_outputs(he_config):
    from dann.config import performance_settings
    from dann.spatial_execution import preflight
    c = he_config
    performance = performance_settings(c)
    yes = preflight(Path(c['data']['path']), c, 100, performance, predict_he=True)
    no = preflight(Path(c['data']['path']), c, 100, performance, predict_he=False)
    assert yes['he_staging_bytes'] == 100 * 7 * 4
    assert no['he_staging_bytes'] == 0

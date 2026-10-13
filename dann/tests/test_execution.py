"""Resident inputs, graph preparation, pass-local evaluation and resume contracts."""
import copy
import time

import h5py
import numpy as np
import pytest
import torch

from dann.config import performance_settings, resolve_execution
from dann.data_loader import create_data_bundle
from dann.execution import ResidentSpectra, spectral_chunks, tree_map
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.tests.test_components import spatial_config
from dann.train import run_epoch, resume_training_config


def enable(config, **kwargs):
    config['performance'].update(ram_cache=True, cache_geometry=True,
                                 preparation_workers=2, prepared_batches=2, **kwargs)
    return resolve_execution(config)


def epoch(model, loader, loss, optimizer=None):
    return run_epoch(model, loader, loss, torch.device('cpu'), 5., 1.,
                     torch.tensor([.7, 1.3]), optimizer, 0, 3, 'dann', .25, 10., 5.)


@pytest.mark.parametrize('transform', ['none', 'sqrt', 'log1p'])
@pytest.mark.parametrize('threshold', [0., .8])
def test_resident_entries(spatial_config, transform, threshold):
    config = spatial_config
    # Replace a stored entry by an explicit zero without eliminating it.
    with h5py.File(config['data']['path'], 'r+') as handle:
        handle['X/data'][0] = 0.
    config['data'].update(intensity_transform=transform, nonzero_threshold=threshold,
                          intensity_clip_max=.9)
    reference = create_data_bundle(config)
    actual = create_data_bundle(enable(config))
    for name in reference.datasets:
        expected = reference.datasets[name]
        observed = actual.datasets[name]
        assert observed.resident is actual.datasets['train'].resident
        for i in range(len(expected)):
            from dann.tiles import tile_collate
            a, b = tile_collate([expected[i]]), tile_collate([observed[i]])
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, equal_nan=True)
    cache = actual.datasets['train'].resident
    rows = np.array([0, 0, len(cache.indptr)-2])
    gathered = cache.gather(rows)
    assert gathered['peak_counts'].tolist() == np.diff(cache.indptr)[rows].tolist()
    assert not cache.values.flags.writeable


@pytest.mark.parametrize('checkpointing', [False, True])
def test_graph_preparation_gradients_and_order(spatial_config, checkpointing):
    config = spatial_config
    config['model']['aggregation']['type'] = 'gatv2'
    config['model']['heads']['biology']['type'] = 'gatv2'
    config['model'].update(gatv2_tile_budget=2, gatv2_checkpointing=checkpointing)
    config['training']['spatial']['supervised_rows_per_batch'] = 9
    resolve_execution(config)
    reference = create_data_bundle(config)
    actual = create_data_bundle(enable(config))
    loader = actual.loaders['train']
    prepare = loader.prepare
    def delayed(request):
        if request.row_ids[0] % 2:
            time.sleep(.01)
        return prepare(request)
    loader.prepare = delayed
    a = AdversarialLatentFusion.from_config(config, 2, 4)
    b = copy.deepcopy(a)
    for expected, observed in zip(reference.loaders['train'], loader):
        torch.testing.assert_close(expected['row_ids'], observed['row_ids'])
        out_a, out_b = a(expected), b(observed)
        for key in out_a:
            torch.testing.assert_close(out_a[key], out_b[key], atol=2e-6, rtol=2e-5)
        out_a['mu'].square().sum().backward()
        out_b['mu'].square().sum().backward()
        for p, q in zip(a.parameters(), b.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)
        a.zero_grad(); b.zero_grad()
    assert spectral_chunks([0, 8, 0, 2, 15, 0], 9) == ((0, 3, 0, 8), (3, 4, 8, 10), (4, 5, 10, 25), (5, 6, 25, 25))


@pytest.mark.parametrize('reduction', ['mean', 'sum'])
@pytest.mark.parametrize('physical', [1, 3, 32])
def test_validation_metrics_feature_invalidation(spatial_config, reduction, physical):
    config = spatial_config
    config['loss']['ziln_reduction'] = reduction
    config['model']['aggregation']['type'] = 'gatv2'
    resolve_execution(config)
    reference = create_data_bundle(config)
    model = AdversarialLatentFusion.from_config(config, 2, 4)
    loss = ZILNLoss.from_config(config)
    expected = epoch(model, reference.loaders['validation'], loss)
    actual = create_data_bundle(enable(config, validation_tiles=physical,
        validation_feature_cache=True, validation_peak_budget=17, validation_gat_tiles=3))
    observed = epoch(model, actual.loaders['validation'], loss)
    for key in expected:
        assert observed[key] == pytest.approx(expected[key], rel=2e-5, abs=2e-5, nan_ok=True)
    with torch.no_grad():
        model.encoder.peak_mlp[-1].bias.add_(.8)
    changed = epoch(model, actual.loaders['validation'], loss)
    expected_changed = epoch(model, reference.loaders['validation'], loss)
    for key in expected_changed:
        assert changed[key] == pytest.approx(expected_changed[key], rel=2e-5, abs=2e-5, nan_ok=True)
    assert changed['cd8_loss'] != observed['cd8_loss']
    batch = next(iter(actual.loaders['validation']))
    batch['pooled_features'] = torch.zeros(len(batch['peak_counts']), 6)
    model.train()
    with pytest.raises(ValueError, match='evaluation-only'):
        model(batch)
    assert model.peak_budget == config['model']['spectral_peak_budget']


@pytest.mark.parametrize('key,value', [('ram_cache', 1), ('prepared_batches', 0),
    ('preparation_workers', True), ('validation_tiles', -1), ('host_budget_gib', float('nan')),
    ('vram_budget_gib', 0), ('validation_feature_cache', True)])
def test_invalid_performance(key, value):
    with pytest.raises(ValueError):
        performance_settings({'performance': {key: value}})


def test_resume_execution_only(spatial_config):
    saved = copy.deepcopy(spatial_config)
    requested = enable(spatial_config, validation_tiles=32)
    requested['training']['supervised_rows_per_batch'] = 8192
    requested['model'].update(spectral_peak_budget=524288, gatv2_tile_budget=64, gatv2_checkpointing=False)
    resumed = resume_training_config(saved, requested)
    assert resumed['training']['supervised_rows_per_batch'] == saved['training']['supervised_rows_per_batch']
    assert resumed['performance']['validation_tiles'] == 32
    assert resumed['model']['gatv2_tile_budget'] == 64
    old_model = AdversarialLatentFusion.from_config(saved, 2, 4)
    new_model = AdversarialLatentFusion.from_config(resumed, 2, 4)
    new_model.load_state_dict(old_model.state_dict(), strict=True)


@pytest.mark.parametrize('strategy', ['shuffled_tiles', 'proportional_slide_tiles'])
@pytest.mark.parametrize('workers', [0, 2])
def test_sampler_order_repeated_epochs(spatial_config, strategy, workers):
    spatial_config['training']['spatial'].update(sampling_strategy=strategy,
        supervised_rows_per_batch=9, num_workers=workers)
    resolve_execution(spatial_config)
    reference = create_data_bundle(spatial_config)
    actual = create_data_bundle(enable(spatial_config))
    for epoch_index in range(2):
        for bundle in (reference, actual):
            sampler = bundle.loaders['train'].sampler
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch_index)
        expected = [b['row_ids'].tolist() for b in reference.loaders['train']]
        observed = [b['row_ids'].tolist() for b in actual.loaders['train']]
        assert observed == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA transfer stream required')
def test_cuda_prefetch_and_cached_validation(spatial_config):
    from dann.execution import device_batches
    config = enable(spatial_config, transfer_prefetch=True, validation_tiles=3,
                    validation_feature_cache=True)
    bundle = create_data_bundle(config)
    loader = bundle.loaders['validation']
    expected = list(loader)
    for cpu, gpu in device_batches(loader, torch.device('cuda')):
        assert cpu['targets'].is_pinned()
        torch.testing.assert_close(gpu['targets'].cpu(), cpu['targets'])
        assert gpu['prepared_graphs'] == {}  # CNN fixture has no graph radii.
    model = AdversarialLatentFusion.from_config(config, 2, 4).cuda()
    kwargs = (model, loader, ZILNLoss.from_config(config).cuda(), torch.device('cuda'),
              5., 1., None, None, 0, 1, 'constant', .25, 10., None)
    cached = run_epoch(*kwargs)
    loader.settings['validation_feature_cache'] = False
    fresh = run_epoch(*kwargs)
    for key in cached:
        if key.endswith('loss'):
            assert fresh[key] == pytest.approx(cached[key], abs=2e-5, rel=2e-5)


def test_cached_graphs_distinct_radii_and_validation_predictions(spatial_config):
    from dann.execution import evaluation_groups
    config = spatial_config
    for group, radius in zip([config['model']['aggregation'], *config['model']['heads'].values()], [1., 1.5, 2.]):
        group['type'] = 'gatv2'
        group['gatv2']['neighbor_radius'] = radius
    resolve_execution(config)
    reference = create_data_bundle(config)
    model = AdversarialLatentFusion.from_config(config, 2, 4).eval()
    actual = create_data_bundle(enable(config, validation_tiles=3, validation_gat_tiles=2,
                                      validation_feature_cache=True))
    with torch.inference_mode():
        expected = [model(batch) for batch in reference.loaders['validation']]
        observed = [outputs for _, _, outputs in evaluation_groups(model, actual.loaders['validation'], torch.device('cpu'))]
    assert len(expected) == len(observed)
    for left, right in zip(expected, observed):
        for key in left:
            torch.testing.assert_close(left[key], right[key], atol=2e-6, rtol=2e-5)


def test_cache_memory_budget_and_invalid_values(spatial_config):
    config = spatial_config
    with pytest.raises(MemoryError, match='host budget'):
        ResidentSpectra(config['data']['path'], config['data'], 1)
    with h5py.File(config['data']['path'], 'r+') as handle:
        handle['X/data'][0] = np.nan
    with pytest.raises(ValueError, match='Invalid transformed'):
        create_data_bundle(enable(config))


def test_empty_validation_and_pixel_residency(spatial_config):
    config = enable(spatial_config, validation_feature_cache=True)
    config['data'].update(validation_fraction=0., test_fraction=.4)
    bundle = create_data_bundle(config)
    model = AdversarialLatentFusion.from_config(config, 2, 4)
    assert np.isnan(epoch(model, bundle.loaders['validation'], ZILNLoss.from_config(config))['cd8_loss'])
    for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
        group['type'] = 'mlp'
    config['data'].update(validation_fraction=.2, test_fraction=.2)
    resolve_execution(config)
    bundle = create_data_bundle(config)
    model = AdversarialLatentFusion.from_config(config, 2, 4)
    assert np.isfinite(epoch(model, bundle.loaders['validation'], ZILNLoss.from_config(config))['cd8_loss'])


def test_resident_source_identity(spatial_config):
    from dann.tiles import SpatialTileDataset
    bundle = create_data_bundle(enable(spatial_config))
    source = bundle.datasets['train']
    changed = copy.deepcopy(spatial_config)
    changed['data']['intensity_transform'] = 'sqrt'
    with pytest.raises(ValueError, match='source or preprocessing'):
        SpatialTileDataset(changed, source.indices, bundle.metadata, resident=source.resident)


def test_independent_checkpoint_switches_preserve_dropout_gradients(spatial_config):
    from dann.model_components.layers_gatv2 import GATv2Network
    config = spatial_config
    config['model']['aggregation']['type'] = 'gatv2'
    config['model']['dropout'] = .2
    config['model']['aggregation']['gatv2'].update(dropout=.2, attention_dropout=.2)
    config['model'].update(spectral_checkpointing=True, gatv2_checkpointing=True)
    bundle = create_data_bundle(enable(config))
    batch = next(iter(bundle.loaders['train']))
    reference = AdversarialLatentFusion.from_config(config, 2, 4)
    for spectral, graph in [(True, False), (False, True), (False, False)]:
        model = copy.deepcopy(reference)
        model.checkpointing = spectral
        for module in model.modules():
            if isinstance(module, GATv2Network):
                module.checkpointing = graph
        reference.zero_grad()
        torch.manual_seed(71)
        expected = reference(batch)
        expected['mu'].square().sum().backward()
        torch.manual_seed(71)
        observed = model(batch)
        observed['mu'].square().sum().backward()
        torch.testing.assert_close(expected['mu'], observed['mu'], rtol=0, atol=0)
        for p, q in zip(reference.parameters(), model.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize('failure', [False, True])
def test_validation_inner_chunk_budget_restored(spatial_config, failure):
    from dann.execution import validation_execution
    config = spatial_config
    config['model'].update(spectral_peak_budget=19, inference_peak_chunk_size=7)
    model = AdversarialLatentFusion.from_config(config, 2, 4)
    with validation_execution(model, {'validation_peak_budget': None}):
        assert model.encoder.inference_peak_chunk_size == 7
    try:
        with validation_execution(model, {'validation_peak_budget': 23}):
            assert model.peak_budget == 23
            assert model.encoder.inference_peak_chunk_size == 23
            with validation_execution(model, {'validation_peak_budget': 31}):
                assert model.encoder.inference_peak_chunk_size == 31
            assert model.encoder.inference_peak_chunk_size == 23
            if failure:
                raise RuntimeError('simulate rejected candidate')
    except RuntimeError:
        assert failure
    assert model.peak_budget == 19
    assert model.encoder.inference_peak_chunk_size == 7


def test_larger_validation_chunk_predictions_and_partial_groups(spatial_config):
    from dann.execution import evaluation_groups
    config = enable(spatial_config, validation_tiles=3, validation_feature_cache=True,
                    validation_peak_budget=17, feature_cache_rows=3)
    config['model']['inference_peak_chunk_size'] = 2
    bundle = create_data_bundle(config)
    model = AdversarialLatentFusion.from_config(config, 2, 4).eval()
    loader = bundle.loaders['validation']
    with torch.inference_mode():
        reference = [(batch['row_ids'], outputs) for _, batch, outputs in evaluation_groups(model, loader, torch.device('cpu'))]
        loader.settings.update(validation_peak_budget=1000, feature_cache_rows=1000)
        observed = [(batch['row_ids'], outputs) for _, batch, outputs in evaluation_groups(model, loader, torch.device('cpu'))]
    for (left_ids, left), (right_ids, right) in zip(reference, observed, strict=True):
        torch.testing.assert_close(left_ids, right_ids)
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=5e-5, atol=5e-5)
    assert model.encoder.inference_peak_chunk_size == 2


def test_benchmark_candidate_resolution_and_memory_tie_break(spatial_config):
    from dann.verification.larger_batch_benchmark import changed, choose
    candidate = changed(spatial_config, rows=8192, peaks=1048576, validation_tiles=256)
    assert candidate['training']['spatial']['supervised_rows_per_batch'] == 8192
    assert candidate['training']['batch_size'] == 8192
    assert candidate['execution']['training_supervised_rows_per_batch'] == 8192
    def record(seconds, memory):
        return {'summary': {'seconds': seconds, 'peak_vram_gib': memory}}
    baseline = record(100., 5.)
    tiny_gain = record(98., 3.)
    assert choose([tiny_gain], baseline) is baseline
    fast, economical = record(80., 20.), record(82., 10.)
    assert choose([fast, economical, {'rejected': 'OOM'}], baseline) is economical

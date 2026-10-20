"""Analysis membership, numerical caches, exact RNG streams and isolated workers."""
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from dann.analysis_execution import (
    RenderQueue, StageCache, extract_prepared, sampled_density_mean, validate_outputs,
)
from dann.config import analysis_execution_settings, performance_settings, resolve_execution
from dann.data_loader import create_data_bundle, load_anndata_metadata
from dann.execution import PreparedLoader, build_validation_features, validation_execution
from dann.model import AdversarialLatentFusion
from dann.spatial_execution import InferenceProgress
from dann.tests.test_components import spatial_config
from dann.tests.test_spatial_execution import checkpoint


@pytest.mark.parametrize('encoding', ['categorical', 'string', 'nullable'])
def test_metadata_reads_only_required_fields(spatial_config, monkeypatch, encoding):
    from dann.tiles import open_backed_anndata
    config = spatial_config
    data = config['data']
    path = Path(data['path'])
    from anndata.io import read_elem, write_elem
    with h5py.File(path, 'r+') as handle:
        batch = pd.Series(read_elem(handle['obs/batch']))
        del handle['obs/batch']
        if encoding == 'categorical':
            values = pd.Categorical(batch, categories=['unused', 'b', 'a'], ordered=True)
        elif encoding == 'string':
            values = batch.astype(str).to_numpy()
        else:
            values = pd.array(batch.map({'a': 0, 'b': 1}), dtype='Int64')
        write_elem(handle['obs'], 'batch', values)
    backed = open_backed_anndata(path)
    expected = backed.obs[data['target_columns']].to_numpy(copy=True)
    batches = backed.obs[data['batch_column']].astype('category')
    backed.file.close()
    original = h5py.Group.__getitem__
    def selective(group, name):
        full = f'{group.name}/{name}'
        assert not any(part in full.split('/') for part in ('obsm', 'layers', 'HES', 'data', 'indices', 'indptr'))
        return original(group, name)
    monkeypatch.setattr(h5py.Group, '__getitem__', selective)
    metadata = load_anndata_metadata(path, data['target_columns'], data['batch_column'],
                                    cd8_normalization=data['cd8_normalization'])
    np.testing.assert_array_equal(metadata.raw_targets, expected)
    np.testing.assert_array_equal(metadata.batch_codes, batches.cat.codes)
    assert metadata.batch_names == tuple(str(value) for value in batches.cat.categories)


@pytest.mark.parametrize('workers', [0, 1, 2, 4])
@pytest.mark.parametrize('draws,seed', [(1, 0), (129, 5), (1000, 20260719)])
def test_sampling_bitwise_stream_and_partial_blocks(workers, draws, seed):
    from dann.ziln import sampled_density_mean as reference
    params = np.random.default_rng(51)
    pi = params.uniform(size=(521, 4))
    mu = params.normal(size=pi.shape)
    sigma = params.uniform(.01, 3, size=pi.shape)
    pi[0] = [0, 1, .5, .5]
    first, second = np.random.default_rng(seed), np.random.default_rng(seed)
    # Advance both streams as analysis does between different splits.
    for rows in (slice(None), slice(7, 19)):
        expected = reference(pi[rows], mu[rows], sigma[rows], num_samples=draws, rng=first)
        observed = sampled_density_mean(pi[rows], mu[rows], sigma[rows], num_samples=draws,
                                        rng=second, workers=workers)
        np.testing.assert_array_equal(observed, expected)
        assert first.bit_generator.state == second.bit_generator.state


def test_cache_validation_rng_state_and_invalidation(tmp_path):
    progress = InferenceProgress()
    cache = StageCache(tmp_path, 'reuse', progress)
    rng = np.random.default_rng(71)
    rng.standard_normal(333)
    arrays = {'latent': np.arange(12, dtype='float32').reshape(3, 4)}
    cache.put('test', {'source': 1}, arrays, {'rng_state': rng.bit_generator.state})
    hit, metadata = cache.get('test', {'source': 1})
    np.testing.assert_array_equal(hit['latent'], arrays['latent'])
    assert not hit['latent'].flags.writeable
    restored = np.random.default_rng()
    restored.bit_generator.state = metadata['rng_state']
    np.testing.assert_array_equal(rng.random(100), restored.random(100))
    assert cache.get('test', {'source': 2}) is None
    path = next(tmp_path.rglob('*.npy'))
    with path.open('r+b') as handle:
        handle.seek(-1, 2)
        handle.write(b'!')
    assert cache.get('test', {'source': 1}) is None
    cache.put('test', {'source': 1}, arrays)
    assert cache.get('test', {'source': 1}) is not None
    assert StageCache(tmp_path, 'refresh', progress).get('test', {'source': 1}) is None
    assert StageCache(tmp_path, 'off', progress).get('test', {'source': 1}) is None


@pytest.mark.parametrize('kind,radius', [('cnn', 1), ('gatv2', 1), ('gatv2', 2)])
@pytest.mark.parametrize('cap', [None, 7])
def test_extraction_matches_legacy_with_union_context(spatial_config, kind, radius, cap):
    from torch.utils.data import DataLoader
    from dann.analyze import extract_latent_predictions
    from dann.tiles import tile_collate
    config = spatial_config
    for group in [config['model']['aggregation'], config['model']['heads']['biology']]:
        group['type'] = kind
        group['gatv2']['neighbor_radius'] = radius
    config['data'].update(max_test_samples=cap, max_validation_samples=cap)
    resolve_execution(config)
    # Explicit stored zero and an empty spectrum are both present.
    with h5py.File(config['data']['path'], 'r+') as handle:
        handle['X/data'][0] = 0.
    legacy = create_data_bundle(config, requested_splits=['test', 'validation'])
    assert set(legacy.datasets) == {'test', 'validation'}
    model = AdversarialLatentFusion.from_config(config, 2, 4).eval()
    expected = {name: extract_latent_predictions(model, loader, torch.device('cpu'))
                for name, loader in legacy.loaders.items()}
    performance = performance_settings({'performance': dict(ram_cache=True, cache_geometry=True,
        preparation_workers=2, prepared_batches=4, validation_tiles=3, validation_peak_budget=17,
        validation_gat_tiles=3, validation_feature_cache=True, feature_cache_rows=5)})
    config['performance'] = performance
    bundle = create_data_bundle(config, requested_splits=['test', 'validation'])
    loaders = {name: PreparedLoader(DataLoader(dataset, batch_size=3, collate_fn=tile_collate),
                performance, config['model'], validation=True) for name, dataset in bundle.datasets.items()}
    with torch.inference_mode(), validation_execution(model, performance):
        cache = build_validation_features(model, loaders['test'], torch.device('cpu'),
                                         context_datasets=list(bundle.datasets.values()))
        expected_context = set(np.concatenate([g['context_rows'] for dataset in bundle.datasets.values()
                                              for g in dataset.context_rows]).tolist())
        assert set(np.flatnonzero(cache[0] >= 0)) == expected_context
        if cap:
            assert expected_context - set(bundle.split_indices['test']) - set(bundle.split_indices['validation'])
        for name, loader in loaders.items():
            actual = extract_prepared(model, loader, torch.device('cpu'), cache, InferenceProgress())
            assert actual.keys() == expected[name].keys()
            for key in actual:
                if key in ('latent', 'pi', 'mu', 'sigma'):
                    np.testing.assert_allclose(actual[key], expected[name][key], rtol=5e-5, atol=5e-5)
                else:
                    np.testing.assert_array_equal(actual[key], expected[name][key])
        missing = PreparedLoader(DataLoader(bundle.datasets['test'], batch_size=10000,
                                 collate_fn=tile_collate, drop_last=True), performance, config['model'])
        with pytest.raises(ValueError, match='Missing or duplicate'):
            extract_prepared(model, missing, torch.device('cpu'), cache, InferenceProgress())
    with pytest.raises(ValueError, match='disabled gradients'):
        build_validation_features(model, loaders['test'], torch.device('cpu'))
    for item in (legacy, bundle):
        for dataset in item.datasets.values():
            dataset.close()


@pytest.mark.parametrize('key,value', [('enabled', 1), ('sampling_workers', -1), ('render_workers', True),
    ('cache_policy', 'maybe'), ('performance', {'host_budget_gib': -1}), ('tiles_per_batch', 0)])
def test_invalid_analysis_controls(key, value):
    with pytest.raises(ValueError):
        analysis_execution_settings({key: value})
    assert not analysis_execution_settings()['enabled']


def test_spawn_renderer_matches_serial_and_failure_cleanup(tmp_path):
    from dann.analyze import plot_latent_umap_batch
    from PIL import Image
    coordinates = np.random.default_rng(7).normal(size=(20, 2))
    batches = np.arange(20) % 2
    settings = dict(figure_dpi=25, point_size=1.)
    direct = tmp_path / 'direct.png'
    plot_latent_umap_batch(coordinates, batches, ['a', 'b'], direct, settings)
    queue = RenderQueue(tmp_path / 'arrays')
    queue.submit(plot_latent_umap_batch, coordinates, batches, ['a', 'b'], tmp_path / 'spawn.png', settings)
    queue.run(2, performance_settings({}), InferenceProgress())
    with Image.open(direct) as a, Image.open(tmp_path / 'spawn.png') as b:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    queue = RenderQueue(tmp_path / 'bad-arrays')
    queue.submit(plot_latent_umap_batch, coordinates[:, :1], batches, ['a', 'b'], tmp_path / 'bad.png', settings)
    with pytest.raises(IndexError):
        queue.run(1, performance_settings({}), InferenceProgress())


@pytest.mark.parametrize('split', ['test', 'validation'])
def test_end_to_end_cache_reuse_and_failure_preserves_outputs(spatial_config, tmp_path, monkeypatch, split):
    from dann import analyze
    from dann import analysis_execution as execution
    config = spatial_config
    config['analysis'].update(split=split, figure_dpi=15, family_count=3, heatmap_top_peaks=4)
    if split == 'validation':
        config['data'].update(train_fraction=.9, validation_fraction=.1, test_fraction=0.)
    config['analysis']['execution'].update(render_workers=0, sampling_workers=1,
                                         feature_cache_rows=7, spectral_peak_budget=17)
    path = checkpoint(config, tmp_path)
    # UMAP numerical reproducibility is tested separately; avoid JIT in orchestration tests.
    monkeypatch.setattr(analyze, 'fit_latent_umap', lambda latent, config: latent[:, :2].copy())
    calls = []
    def compact(config, path, epoch, metadata, extracted, scatter, coords, peptides, activity, output, render, **kwargs):
        assert kwargs['checkpoint']['config']['loss'] == torch.load(path, weights_only=False)['config']['loss']
        calls.append((extracted['density_mean'].copy(), scatter['density_mean'].copy()))
        analyze.build_latent_umap_frame(extracted, coords, config['data']['target_columns'],
            metadata.batch_names, config['analysis']['split']).to_csv(output / 'latent.csv', index=False)
    monkeypatch.setattr(execution, 'write_artifacts', compact)
    out = tmp_path / 'optimized'
    analyze.run_analysis(config, path, output_override=out)
    before = (out / 'latent.csv').read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('A valid cache hit must skip model loading, UMAP and sampling.')
    monkeypatch.setattr(analyze, 'load_model', forbidden)
    monkeypatch.setattr(analyze, 'fit_latent_umap', forbidden)
    monkeypatch.setattr(execution, 'sampled_density_mean', forbidden)
    analyze.run_analysis(config, path, output_override=out)
    assert (out / 'latent.csv').read_bytes() == before
    assert set(json.loads((out / 'analysis_execution.json').read_text())['cache_hits']) == {
        'extracted-validation', 'density-validation', 'umap', 'peptides', 'activity'
    } | ({'extracted-test', 'density-test'} if split == 'test' else set())
    for a, b in zip(*calls):
        np.testing.assert_array_equal(a, b)
    # A cached first split must advance the RNG before a missing second split.
    manifest = next((out / '.analysis-cache/density-validation').glob('*/manifest.json'))
    manifest.unlink()
    monkeypatch.setattr(execution, 'sampled_density_mean', sampled_density_mean)
    analyze.run_analysis(config, path, output_override=out)
    for first, last in zip(calls[0], calls[-1]):
        np.testing.assert_array_equal(first, last)
    monkeypatch.setattr(execution, 'validate_outputs', lambda directory, **kwargs: (_ for _ in ()).throw(ValueError('bad output')))
    with pytest.raises(ValueError, match='bad output'):
        analyze.run_analysis(config, path, output_override=out)
    assert (out / 'latent.csv').read_bytes() == before
    assert not list(out.glob('.analysis-work-*'))
    # Matrix edits with unchanged observation/feature identities still invalidate
    # extraction, since the source file identity is part of every cache key.
    with h5py.File(config['data']['path'], 'r+') as handle:
        handle['X/data'][0] += .25
    with pytest.raises(AssertionError, match='valid cache hit'):
        analyze.run_analysis(config, path, output_override=out)
    assert (out / 'latent.csv').read_bytes() == before


def test_output_validation_detects_corrupt_image(tmp_path):
    (tmp_path / 'broken.png').write_bytes(b'bad')
    with pytest.raises(OSError):
        validate_outputs(tmp_path)


@pytest.mark.parametrize('block_rows', [1, 7, 10000])
def test_partitioned_csv_bytes_match_legacy(tmp_path, block_rows):
    from dann import analyze as a
    rng = np.random.default_rng(1)
    extracted = {name: rng.random((31, 2)).astype('float32') for name in
                 ('pi', 'mu', 'sigma', 'targets', 'raw_targets')}
    extracted.update(row_ids=np.arange(31)[::-1], batches=np.arange(31) % 2,
                     latent=rng.normal(size=(31, 128)).astype('float32'),
                     density_mean=rng.random((31, 2)))
    extracted['targets'][:7, 0] = 0
    extracted['targets'][10, 1] = np.nan
    coordinates = rng.normal(size=(31, 2)).astype('float32')
    targets, batches = ['Density_CD8', 'Density_Tumor'], ['a', 'b']
    frames = dict(latent=a.build_latent_embeddings_frame(extracted, batches, 'test'),
        umap=a.build_latent_umap_frame(extracted, coordinates, targets, batches, 'test'),
        scatter=a.build_ziln_scatter_frame(extracted, targets, batches, 'test', 1e-6))
    queue = RenderQueue(tmp_path / 'parts')
    for kind, frame in frames.items():
        frame.to_csv(tmp_path / f'expected-{kind}.csv', index=False,
                     float_format=None if kind == 'latent' else '%.17g')
        queue.table(kind, extracted, coordinates, targets, batches, 'test', 1e-6,
                    tmp_path / f'{kind}.csv', block_rows=block_rows)
    queue.run(0, performance_settings({}), InferenceProgress())
    for kind in frames:
        assert (tmp_path / f'{kind}.csv').read_bytes() == (tmp_path / f'expected-{kind}.csv').read_bytes()


def test_legacy_dispatch_and_cli_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr('dann.artifact_layout.organize_analysis', lambda output: None)
    from dann import analyze
    from dann import analysis_execution
    monkeypatch.setattr(analyze, '_run_legacy_analysis', lambda *args, **kwargs: 'legacy')
    for section in ({}, {'execution': None}, {'execution': {'enabled': False}}):
        assert analyze.run_analysis({'analysis': section}, None) == 'legacy'
    seen = []
    monkeypatch.setattr(analysis_execution, 'run_optimized_analysis',
                        lambda config, checkpoint, settings, output: seen.append((settings, output)))
    config = {'analysis': {'execution': {'enabled': True, 'performance': {'prepared_batches': 7}}}}
    analyze.run_analysis(config, None, execution_overrides={'performance': {'host_budget_gib': 33}},
                         output_override=tmp_path)
    assert seen[0][0]['performance'] == {'prepared_batches': 7, 'host_budget_gib': 33}
    assert seen[0][1] == tmp_path
    args = analyze.parse_args(['--output-dir', str(tmp_path), '--cache-policy', 'off'])
    assert args.output_dir == tmp_path and args.cache_policy == 'off'


def test_budget_fails_before_resident_load(spatial_config, tmp_path, monkeypatch):
    from dann import analyze, execution
    config = spatial_config
    path = checkpoint(config, tmp_path)
    config['analysis']['execution']['performance']['host_budget_gib'] = .001
    def forbidden(*args, **kwargs):
        raise AssertionError('Budget must fail before loading spectra.')
    monkeypatch.setattr(execution, 'ResidentSpectra', forbidden)
    with pytest.raises(MemoryError, match='host budget'):
        analyze.run_analysis(config, path, output_override=tmp_path / 'budget')
    assert not list((tmp_path / 'budget').glob('.analysis-work-*'))


@pytest.mark.parametrize('mode', ['pixel', 'train'])
def test_legacy_extraction_fallback_without_residency(spatial_config, tmp_path, monkeypatch, mode):
    from dann import analyze, analysis_execution
    config = spatial_config
    if mode == 'pixel':
        for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
            group['type'] = 'mlp'
        resolve_execution(config)
    else:
        config['analysis']['split'] = 'train'
    config['analysis']['execution'] = {'enabled': True, 'render_workers': 0}
    path = checkpoint(config, tmp_path)
    used = []
    original = analyze.extract_latent_predictions
    def extraction(*args):
        result = original(*args)
        used.append(result['row_ids'].copy())
        return result
    monkeypatch.setattr(analyze, 'extract_latent_predictions', extraction)
    monkeypatch.setattr(analyze, 'fit_latent_umap', lambda latent, config: latent[:, :2].copy())
    def artifacts(*args, **kwargs):
        (args[9] / 'complete.json').write_text('{}')
    monkeypatch.setattr(analysis_execution, 'write_artifacts', artifacts)
    out = tmp_path / 'fallback'
    analyze.run_analysis(config, path, output_override=out)
    assert len(used) == (1 if mode == 'pixel' else 2)
    assert not json.loads((out / 'analysis_execution.json').read_text())['feature_cache_active']


def test_checkpoint_snapshot_identity_and_concurrent_write(tmp_path, monkeypatch):
    from dann.analysis_execution import digest_file, load_checkpoint_snapshot
    path = tmp_path / 'checkpoint.pt'
    torch.save({'weights': torch.arange(3)}, path)
    checkpoint, digest = load_checkpoint_snapshot(path)
    assert digest == digest_file(path)
    torch.testing.assert_close(checkpoint['weights'], torch.arange(3))
    original = torch.load
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with path.open('ab') as handle:
            handle.write(b'!')
        return result
    monkeypatch.setattr(torch, 'load', changed)
    with pytest.raises(ValueError, match='changed while'):
        load_checkpoint_snapshot(path)

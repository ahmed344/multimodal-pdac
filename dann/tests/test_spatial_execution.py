"""Inference cache, metadata, deterministic spawned sampling and publication contracts."""
import copy
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch

from dann.config import spatial_execution_settings, resolve_execution
from dann.spatial import run_inference
from dann.spatial_execution import density_blocks, sample_rows
from dann.tests.test_components import spatial_config
from dann.tiles import open_backed_anndata, read_grid, row_identity


def test_selective_metadata_and_fingerprint(spatial_config, monkeypatch):
    path = Path(spatial_config['data']['context_path'])
    data = spatial_config['data']
    with h5py.File(path, 'r+') as handle:
        # Nullable coordinates are supported without loading unrelated fields.
        values = handle['obs/x'][:]
        del handle['obs/x']
        group = handle['obs'].create_group('x')
        group.attrs.update({'encoding-type': 'nullable-integer', 'encoding-version': '0.1.0'})
        group.create_dataset('values', data=values)
        group.create_dataset('mask', data=np.zeros(len(values), dtype=bool))
    adata = open_backed_anndata(path)
    columns = [data[k] for k in ('batch_column', 'x_column', 'y_column')]
    expected_keys = pd.MultiIndex.from_frame(adata.obs[columns].assign(batch=adata.obs.batch.astype(str), x=adata.obs.x.astype('int64'), y=adata.obs.y.astype('int64')))
    expected_digest = hashlib.sha256(pd.util.hash_pandas_object(adata.obs[columns].astype(str), index=True).values.tobytes()).hexdigest()
    expected_features = np.asarray(adata.var_names, dtype=str)
    adata.file.close()
    original = h5py.Group.__getitem__
    def selective(group, name):
        full = f'{group.name}/{name}'
        assert not any(part in full.split('/') for part in ('layers', 'obsm', 'X', 'HES'))
        return original(group, name)
    monkeypatch.setattr(h5py.Group, '__getitem__', selective)
    keys, features = read_grid(path, data)
    assert keys.equals(expected_keys)
    np.testing.assert_array_equal(features, expected_features)
    assert row_identity(path, data)['row_digest'] == expected_digest


@pytest.mark.parametrize('samples,seed', [(1, 0), (129, 17), (1000, 20260719)])
def test_spawned_sampling_bitwise_and_block_invariant(tmp_path, samples, seed):
    rng = np.random.default_rng(21)
    n, width, targets = 17, 3, 4
    values = rng.uniform(size=(n, width+3*targets+2)).astype('float32')
    path = tmp_path/'parameters.bin'
    values.tofile(path)
    pi, mu, sigma = np.split(values[:, width:width+3*targets], 3, axis=1)
    expected = sample_rows(pi, mu, sigma, 0, samples, seed)
    for workers, block in [(0, 1), (0, 7), (2, 3), (2, 8)]:
        result = list(density_blocks(path, values.shape, width, targets, block, workers, samples, seed))
        assert [start for start, _ in result] == list(range(0, n, block))
        actual = np.concatenate([v for _, v in result])
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(actual, expected)


def test_density_worker_failure(tmp_path):
    values = np.full((5, 9), np.nan, dtype='float32')
    path = tmp_path/'bad.bin'
    values.tofile(path)
    with pytest.raises(ValueError):
        list(density_blocks(path, values.shape, 1, 2, 2, 2, 7, 31))


def checkpoint(config, tmp_path):
    from dann.data_loader import create_data_bundle
    from dann.model import AdversarialLatentFusion
    from dann.train import save_checkpoint
    bundle = create_data_bundle(config)
    model = AdversarialLatentFusion.from_config(config, 2, 4)
    path = tmp_path/'model.pt'
    save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), 0, config, bundle, 1.)
    for dataset in bundle.datasets.values():
        dataset.close()
    return path


def options(**changes):
    return dict(spatial_execution=dict(enabled=True, output_rows=7, density_workers=2,
                                      tiles_per_batch=3, spectral_peak_budget=17, gat_tiles=2,
                                      feature_cache_rows=5) | changes,
                performance=dict(ram_cache=True, cache_geometry=True, preparation_workers=2,
                                 prepared_batches=4, torch_threads=2))


@pytest.mark.parametrize('kind', ['cnn', 'gatv2'])
def test_cached_export_checkpoint_cap_context_and_support(spatial_config, tmp_path, kind):
    config = spatial_config
    config['analysis'].pop('spatial_execution', None)
    config['model']['aggregation']['type'] = kind
    config['model']['heads']['biology']['type'] = kind
    config['model']['aggregation']['gatv2']['neighbor_radius'] = 1.
    config['model']['heads']['biology']['gatv2']['neighbor_radius'] = 2.
    resolve_execution(config)
    path = checkpoint(config, tmp_path)
    source = Path(config['data']['context_path'])
    # Retain a stored zero and an empty row. Fixture also has two slides and tissue gaps.
    with h5py.File(source, 'r+') as handle:
        handle['X/data'][0] = 0.
    for name, overrides, cap in [('legacy', {}, None), ('fast', options(), 23)]:
        run_inference(source, path, tmp_path/name/'prediction.parquet', max_rows=cap,
                      execution_overrides=overrides)
    for file in ('prediction.parquet', 'spatial_latent.parquet', 'prediction.support.parquet'):
        a = pq.read_table(tmp_path/'legacy'/file).slice(0, 23)
        b = pq.read_table(tmp_path/'fast'/file)
        assert a.schema.equals(b.schema)
        for key in a.column_names:
            left, right = a[key].to_numpy(), b[key].to_numpy()
            if key in ('row_position', 'obs_name') or 'support' in key:
                np.testing.assert_array_equal(left, right)
            else:
                np.testing.assert_allclose(left, right, rtol=5e-5, atol=5e-5)
    import json
    report = json.loads((tmp_path/'fast'/'prediction.provenance.json').read_text())['execution']
    assert report['feature_cache_rows'] > 23
    assert 'feature encoding' in report['phase_seconds']
    assert 'density sampling/writing' in report['phase_seconds']
    assert report['physical_tiles_per_batch'] == 3
    assert not list((tmp_path/'fast').glob('.*tmp'))
    assert not list((tmp_path/'fast').glob('.dann-stage-*'))


def test_budget_failure_cleanup_and_overwrite(spatial_config, tmp_path, monkeypatch):
    config = spatial_config
    config['analysis'].pop('spatial_execution', None)
    path = checkpoint(config, tmp_path)
    source = Path(config['data']['context_path'])
    output = tmp_path/'out'/'prediction.parquet'
    opts = options()
    opts['performance']['host_budget_gib'] = .00001
    with pytest.raises(MemoryError, match='host budget'):
        run_inference(source, path, output, execution_overrides=opts)
    assert not output.exists()
    import dann.spatial as spatial
    def fail(*args, **kwargs):
        raise ValueError('validation injected')
    monkeypatch.setattr(spatial, 'validate_support_output', fail)
    with pytest.raises(ValueError, match='validation injected'):
        run_inference(source, path, output, max_rows=7, execution_overrides=options())
    assert list(output.parent.iterdir()) == []
    output.with_suffix('.support.parquet').write_bytes(b'protected')
    with pytest.raises(FileExistsError):
        run_inference(source, path, output, execution_overrides=options())
    assert output.with_suffix('.support.parquet').read_bytes() == b'protected'


@pytest.mark.parametrize('key,value', [('enabled', 1), ('feature_cache', 0), ('tiles_per_batch', 0),
                                     ('density_workers', -1), ('output_rows', True), ('typo', 1)])
def test_execution_validation(key, value):
    with pytest.raises(ValueError):
        spatial_execution_settings({key: value})
    assert not spatial_execution_settings()['enabled']


def test_cli_tile_precedence(spatial_config, tmp_path):
    config = spatial_config
    config['analysis'].pop('spatial_execution', None)
    path = checkpoint(config, tmp_path)
    output = tmp_path/'precedence'/'prediction.parquet'
    run_inference(Path(config['data']['context_path']), path, output, max_rows=3,
                  batch_size_override=7, tiles_per_batch=1, execution_overrides=options())
    import json
    report = json.loads(output.with_suffix('.provenance.json').read_text())
    assert report['execution']['physical_tiles_per_batch'] == 1


def test_categorical_axis_names_and_nullable_missing(spatial_config):
    from dann.spatial import read_axis_names, ObservationNameReader
    from anndata.io import write_elem
    source = Path(spatial_config['data']['context_path'])
    names = read_axis_names(source, 'obs')
    with h5py.File(source, 'r+') as handle:
        del handle['obs/_index']
        write_elem(handle['obs'], '_index', pd.Categorical(names))
    np.testing.assert_array_equal(read_axis_names(source, 'obs'), names)
    with ObservationNameReader(source) as reader:
        np.testing.assert_array_equal(reader.read(2, 7), names[2:7])
    with h5py.File(source, 'r+') as handle:
        values = handle['obs/x'][:]
        del handle['obs/x']
        write_elem(handle['obs'], 'x', pd.array(values, dtype='Int64'))
        handle['obs/x/mask'][0] = True
    with pytest.raises(ValueError, match='missing'):
        read_grid(source, spatial_config['data'])


def test_feature_cache_requires_inference_and_rebuilds(spatial_config, tmp_path):
    from dann.execution import PreparedLoader, ResidentSpectra, build_validation_features
    from dann.model import AdversarialLatentFusion
    from dann.spatial_execution import inference_performance
    from dann.tiles import SpatialTileDataset, tile_collate
    from torch.utils.data import DataLoader
    config = spatial_config
    settings = spatial_execution_settings(options()['spatial_execution'])
    performance = inference_performance(config, options(), settings)
    config['performance'] = performance
    path = Path(config['data']['context_path'])
    resident = ResidentSpectra(path, config['data'], 2**30)
    dataset = SpatialTileDataset(config, np.arange(5), input_path=path, resident=resident)
    loader = PreparedLoader(DataLoader(dataset, batch_size=3, collate_fn=tile_collate), performance, config['model'], validation=True)
    model = AdversarialLatentFusion.from_config(config, 2, 4).eval()
    with pytest.raises(ValueError, match='evaluation-only'):
        build_validation_features(model, loader, torch.device('cpu'))
    with torch.inference_mode():
        ids, first = build_validation_features(model, loader, torch.device('cpu'))
        model.encoder.peak_mlp[-1].bias.add_(1.)
        second_ids, second = build_validation_features(model, loader, torch.device('cpu'))
    np.testing.assert_array_equal(ids, second_ids)
    assert not torch.equal(first, second)
    dataset.close()


def test_inference_without_feature_cache_and_saved_training_annotations(spatial_config, tmp_path):
    config = spatial_config
    config['analysis'].pop('spatial_execution', None)
    # Training annotations alone must not activate inference caches.
    config['performance'].update(ram_cache=True, cache_geometry=True, validation_feature_cache=True)
    path = checkpoint(config, tmp_path)
    source = Path(config['data']['context_path'])
    run_inference(source, path, tmp_path/'legacy'/'predictions.parquet', max_rows=7)
    run_inference(source, path, tmp_path/'uncached'/'predictions.parquet', max_rows=7,
                  execution_overrides=options(feature_cache=False))
    import json
    legacy = json.loads((tmp_path/'legacy'/'predictions.provenance.json').read_text())
    assert not legacy['execution']['spatial_execution']['enabled']
    assert legacy['execution']['performance'] == {}
    for name in ('predictions.parquet', 'spatial_latent.parquet'):
        a, b = [pq.read_table(tmp_path/d/name) for d in ('legacy', 'uncached')]
        for column in a.column_names[2:]:
            np.testing.assert_allclose(a[column].to_numpy(), b[column].to_numpy(), atol=5e-5, rtol=5e-5)


def test_saved_validation_settings_do_not_constrain_inference():
    from dann.spatial_execution import inference_performance
    saved = {'performance': {'ram_cache': True, 'cache_geometry': True,
                             'validation_feature_cache': True, 'validation_peak_budget': 12}}
    runtime = {'performance': {'ram_cache': False, 'cache_geometry': False}}
    spatial = spatial_execution_settings({'enabled': True, 'feature_cache': False})
    settings = inference_performance(saved, runtime, spatial)
    assert not settings['ram_cache']
    assert not settings['validation_feature_cache']
    assert settings['validation_peak_budget'] == spatial['spectral_peak_budget']


def test_spawn_failure_keeps_published_outputs(spatial_config, tmp_path, monkeypatch):
    config = spatial_config
    config['analysis'].pop('spatial_execution', None)
    path = checkpoint(config, tmp_path)
    source = Path(config['data']['context_path'])
    output = tmp_path/'protected'/'predictions.parquet'
    run_inference(source, path, output, max_rows=7, execution_overrides=options(density_workers=0))
    original = {p.name: p.read_bytes() for p in output.parent.iterdir()}
    import dann.spatial_execution as execution
    original_blocks = execution.density_blocks
    def corrupt(staging, shape, width, *args):
        values = np.memmap(staging, mode='r+', dtype=np.float32, shape=shape)
        values[0, width] = np.nan
        values.flush()
        values._mmap.close()
        yield from original_blocks(staging, shape, width, *args)
    monkeypatch.setattr(execution, 'density_blocks', corrupt)
    with pytest.raises(ValueError):
        run_inference(source, path, output, max_rows=7, overwrite=True,
                      execution_overrides=options(density_workers=2))
    assert {p.name: p.read_bytes() for p in output.parent.iterdir()} == original


def test_feature_budget_rejected_before_resident_read(spatial_config, tmp_path, monkeypatch):
    import dann.execution as execution
    config = spatial_config
    config['analysis'].pop('spatial_execution', None)
    path = checkpoint(config, tmp_path)
    # Emulate an already allocated CUDA model without touching the active GPU.
    from dann.spatial_execution import check_feature_budget
    from dann.model import AdversarialLatentFusion
    model = AdversarialLatentFusion.from_config(config, 2, 4)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda *args: 0)
    with pytest.raises(MemoryError, match='feature budget'):
        check_feature_budget(model, 1_000_000, torch.device('cuda'),
                             {'vram_budget_gib': .001, 'validation_peak_budget': 17})


def test_writer_close_failure_removes_temporary_outputs(spatial_config, tmp_path, monkeypatch):
    config = spatial_config
    config['analysis'].pop('spatial_execution', None)
    path = checkpoint(config, tmp_path)
    output = tmp_path/'close_failure'/'predictions.parquet'
    original = pq.ParquetWriter
    class FailingWriter:
        def __init__(self, *args, **kwargs):
            self.writer = original(*args, **kwargs)
        def write_table(self, *args, **kwargs):
            self.writer.write_table(*args, **kwargs)
        def close(self):
            self.writer.close()
            raise OSError('injected close failure')
    monkeypatch.setattr(pq, 'ParquetWriter', FailingWriter)
    with pytest.raises(OSError, match='injected close failure'):
        run_inference(Path(config['data']['context_path']), path, output, max_rows=3,
                      execution_overrides=options(density_workers=0))
    assert list(output.parent.iterdir()) == []


def test_density_rounding_audit_explains_one_draw_and_rejects_corruption(tmp_path):
    from dann.spatial import build_output_table
    from dann.verification.spatial_density_audit import audit
    seed, samples = 123, 1000
    rng = np.random.default_rng([seed, 0])
    rng.standard_normal((4, 128))
    threshold = np.float32(rng.random((4, 128))[0, 0])
    mu, sigma = np.zeros((1, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32)
    targets = ['a', 'b', 'c', 'd']
    paths = [tmp_path/'a.parquet', tmp_path/'b.parquet']
    for direction, path in zip((-np.inf, np.inf), paths):
        pi = np.full((1, 4), .25, dtype=np.float32)
        pi[0, 0] = np.nextafter(threshold, np.float32(direction))
        means = sample_rows(pi, mu, sigma, 0, samples, seed).astype(np.float32)
        table = build_output_table(np.array([0]), np.array(['row']), mu, pi, sigma, targets, density_mean=means)
        pq.write_table(table, path)
    report = audit(*paths, samples, seed)
    assert report['all_explained']
    assert report['exceptions_by_target']['a'] == 1
    assert report['max_possible_hurdle_switches'] == 1
    means[0, 0] += .01
    table = build_output_table(np.array([0]), np.array(['row']), mu, pi, sigma, targets, density_mean=means)
    pq.write_table(table, paths[1])
    assert audit(*paths, samples, seed)['unexplained_rows'] == [0]


def test_inverse_presence_rounding_interval():
    from dann.verification.spatial_density_audit import pi_interval
    pi = np.r_[np.zeros(1), np.ones(1), np.random.default_rng(12).uniform(size=10000)].astype(np.float32)
    low, high = pi_interval(np.float32(1)-pi)
    assert np.all(pi.astype(np.float64) >= low)
    assert np.all(pi.astype(np.float64) <= high)

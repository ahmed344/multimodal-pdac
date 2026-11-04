"""Storage-aware initialization and shared resident H&E supervision."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pytest

from dann.config import performance_settings, resolve_execution
from dann.data_loader import create_data_bundle
from dann.execution import ResidentSpectra
from dann.he import HETargetReader
from dann.initialization import final_projection, initialize_supervised_heads
from dann.model import AdversarialLatentFusion
from dann.tests.test_components import spatial_config
from dann.tests.test_he import he_config


@pytest.mark.parametrize('chunks', [None, (6, 2)])
@pytest.mark.parametrize('dtype', [np.float64, np.int32])
def test_resident_values_order_threads_and_no_disk(tmp_path, monkeypatch, chunks, dtype):
    path = tmp_path / 'features.h5'
    values = np.arange(119).reshape(17, 7).astype(dtype)
    with h5py.File(path, 'w') as f:
        f.create_dataset('obsm/he', data=values, chunks=chunks)
    reader = HETargetReader(path, 'he', 17, 7)
    rows = np.array([16, 6, 0, 6, 5, 12])
    expected = reader.read(rows)
    reader.cache_in_memory(128 * 2**30)
    assert reader.nbytes == 17 * 7 * 4
    assert not reader.values.flags.writeable
    def no_io(*args, **kwargs):
        raise AssertionError('Cached supervision must not open HDF5.')
    monkeypatch.setattr(h5py, 'File', no_io)
    with ThreadPoolExecutor(4) as pool:
        for actual in pool.map(reader.read, [rows] * 8):
            np.testing.assert_array_equal(actual, expected)
    assert reader.read(np.array([], dtype=int)).shape == (0, 7)
    for workers in (1, 4):
        np.testing.assert_allclose(reader.mean(rows, 3, workers=workers), expected.mean(0, dtype=np.float64))
    np.testing.assert_array_equal(reader.mean(np.array([], dtype=int)), np.zeros(7))


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('chunks', [None, (6, 2)])
def test_training_only_mean_and_errors(tmp_path, cached, chunks):
    path = tmp_path / 'features.h5'
    rng = np.random.default_rng(2)
    values = rng.normal(size=(17, 7))
    rows = np.array([16, 6, 0, 6, 5, 12])
    values[[1, 7, 10]] = np.nan  # Other rows in the same storage chunks.
    values[9] = 1e100  # Float32 overflow in an unselected row is also ignored.
    with h5py.File(path, 'w') as f:
        f.create_dataset('obsm/he', data=values, chunks=chunks)
    reader = HETargetReader(path, 'he', 17, 7)
    if cached:
        reader.cache_in_memory(128 * 2**30)
    expected = values[rows].astype(np.float32).mean(0, dtype=np.float64)
    a = reader.mean(rows, 3, workers=1)
    b = reader.mean(rows, 3, workers=4)
    np.testing.assert_allclose(a, expected, atol=1e-15)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(rows, [16, 6, 0, 6, 5, 12])
    for selected in (np.array([7]), np.array([9])):
        with pytest.raises(ValueError, match='source row positions'):
            reader.mean(selected, workers=4)
        with pytest.raises(ValueError, match='source row positions'):
            reader.read(selected)
    with pytest.raises(ValueError, match='within the source'):
        reader.read(np.array([-1]))


def test_backed_mean_reads_each_selected_chunk_once(tmp_path, monkeypatch):
    path = tmp_path / 'features.h5'
    values = np.arange(96, dtype=np.float32).reshape(24, 4)
    with h5py.File(path, 'w') as f:
        f.create_dataset('obsm/he', data=values, chunks=(6, 2))
    reader = HETargetReader(path, 'he', 24, 4)
    original = h5py.Dataset.__getitem__
    selections = []
    def record(dataset, selected):
        if dataset.name == '/obsm/he':
            selections.append(selected)
        return original(dataset, selected)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', record)
    rows = np.array([23, 1, 18, 0, 5, 23])
    np.testing.assert_array_equal(reader.mean(rows, 2), values[rows].mean(0, dtype=np.float64))
    assert selections == [slice(0, 6), slice(18, 24)]


def test_budget_rejected_before_allocation(tmp_path, monkeypatch):
    path = tmp_path / 'features.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('obsm/he', data=np.ones((10, 4)))
    reader = HETargetReader(path, 'he', 10, 4)
    def no_allocation(*args, **kwargs):
        raise AssertionError('Budget must be checked before allocating the cache.')
    monkeypatch.setattr(np, 'empty', no_allocation)
    with pytest.raises(MemoryError, match='H&E cache.*host budget'):
        reader.cache_in_memory(1)
    assert reader.values is None


@pytest.mark.parametrize('kind', ['mlp', 'gatv2'])
def test_bundle_shares_cache_and_initializes_without_io(he_config, monkeypatch, kind):
    c = he_config
    for group in [c['model']['aggregation'], *c['model']['heads'].values()]:
        group['type'] = kind
    c['performance'].update(ram_cache=True, he_ram_cache=True, startup_workers=4,
                            preparation_workers=4, prepared_batches=8)
    c['training']['head_initialization']['enabled'] = True
    resolve_execution(c)
    bundle = create_data_bundle(c)
    reader = bundle.he_reader
    assert reader.nbytes > 0
    assert all(ds.he_reader is reader for ds in bundle.datasets.values())
    train = bundle.split_indices['train']
    expected = reader.values[train].mean(0, dtype=np.float64)
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    def no_io(*args, **kwargs):
        raise AssertionError('Initialization and cached batches must not read HDF5.')
    monkeypatch.setattr(h5py, 'File', no_io)
    report = initialize_supervised_heads(model, bundle, c, chunk_rows=3)
    np.testing.assert_allclose(report['he']['mean'], expected)
    np.testing.assert_allclose(final_projection(model.he_predictor).bias.detach(), expected, atol=1e-6)
    for batch in bundle.loaders['train']:
        np.testing.assert_array_equal(batch['he_targets'], reader.values[batch['row_ids'].numpy()])


@pytest.mark.parametrize('transform', ['none', 'sqrt', 'log1p'])
@pytest.mark.parametrize('threshold', [0., .8])
def test_parallel_spectra_preserve_entries(spatial_config, transform, threshold):
    data = dict(spatial_config['data'], intensity_transform=transform,
                nonzero_threshold=threshold, intensity_clip_max=.9)
    a = ResidentSpectra(Path(data['path']), data, 2**30, workers=1)
    b = ResidentSpectra(Path(data['path']), data, 2**30, workers=4)
    for name in ('indptr', 'indices', 'values'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_enabled_cache_with_disabled_head_needs_no_he(spatial_config):
    c = spatial_config
    c['performance'].update(ram_cache=True, he_ram_cache=True)
    bundle = create_data_bundle(resolve_execution(c))
    assert bundle.he_reader is None
    assert all(dataset.he_reader is None for dataset in bundle.datasets.values())


def test_preparation_budget_counts_shared_he_once(he_config):
    c = he_config
    for group in [c['model']['aggregation'], *c['model']['heads'].values()]:
        group['type'] = 'mlp'
    c['performance'].update(ram_cache=True, he_ram_cache=True)
    bundle = create_data_bundle(resolve_execution(c))
    loader = bundle.loaders['train']
    request = [0, 1]
    loader.prepare(request)
    spectra_bytes = bundle.datasets['train'].resident.nbytes
    batch_reservation = loader.max_batch_bytes * (3 * loader.settings['prepared_batches'] + 6)
    loader.settings = dict(loader.settings, host_budget_gib=(spectra_bytes + batch_reservation) / 2**30)
    with pytest.raises(MemoryError, match='host budget'):
        loader.prepare(request)
    loader.settings['host_budget_gib'] += bundle.he_reader.nbytes / 2**30
    loader.prepare(request)


@pytest.mark.parametrize('settings', [dict(startup_workers=0), dict(startup_workers=True),
                                     dict(he_ram_cache=1), dict(he_ram_cache=True)])
def test_invalid_startup_configuration(settings):
    with pytest.raises(ValueError):
        performance_settings({'performance': settings})

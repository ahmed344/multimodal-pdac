"""Whole-pixel batches preserve the scalar sparse loader's numerical contract."""
import copy
import threading
import time

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from dann.config import performance_settings, resolve_execution
from dann.data_loader import AnnDataMetadata, SparseAnnDataDataset, create_data_bundle, sparse_collate
from dann.execution import PreparedLoader, ResidentSpectra
from dann.he import HETargetReader, he_mse
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.tests.test_components import spatial_config
from dann.tests.test_he import he_config


class ScalarDataset(Dataset):
    """Keep the pre-optimization fetch path as an independent reference."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


def assert_batch(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)


@pytest.fixture
def pixel_dataset(tmp_path):
    path = tmp_path / 'pixels.h5ad'
    # Empty rows at both ends and between runs; unsorted/duplicate peaks and zeros.
    with h5py.File(path, 'w') as handle:
        group = handle.create_group('layers/msi')
        group.attrs['shape'] = (8, 7)
        group.create_dataset('indptr', data=[0, 0, 4, 4, 6, 7, 7, 10, 10])
        group.create_dataset('indices', data=[4, 1, 4, 0, 2, 1, 5, 6, 2, 0])
        group.create_dataset('data', data=[0., .4, 3., 2., .8, 4., 7., 1., 0., 8.])
        handle.create_dataset('obsm/he', data=np.arange(40).reshape(8, 5) / 7)
    metadata = AnnDataMetadata(
        targets=np.arange(24, dtype=np.float64).reshape(8, 3) / 19,
        raw_targets=np.arange(24, dtype=np.float64).reshape(8, 3) / 23,
        target_valid_mask=np.arange(24).reshape(8, 3) % 2 == 0,
        batch_codes=np.arange(8, dtype=np.int32) % 2, batch_names=('a', 'b'),
        mz_values=np.arange(7), num_observations=8, num_peaks=7)

    def make(resident=False, he=True, transform='none', threshold=0., clip=None, fallback=False):
        settings = dict(matrix_key='layers/msi', intensity_transform=transform,
                        nonzero_threshold=threshold, intensity_clip_max=clip)
        cache = ResidentSpectra(path, settings, 2**20) if resident else None
        labels = metadata
        if fallback:
            from dataclasses import replace
            labels = replace(metadata, raw_targets=None, target_valid_mask=None)
        return SparseAnnDataDataset(path, np.array([6, 0, 3, 1, 7, 4, 2, 5]), labels,
            resident=cache, he_reader=HETargetReader(path, 'he', 8, 5) if he else None, **settings)
    return make


@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('he', [False, True])
@pytest.mark.parametrize('transform', ['none', 'sqrt', 'log1p'])
@pytest.mark.parametrize('threshold,clip', [(0., None), (.8, .9)])
def test_batch_scalar_equivalence(pixel_dataset, resident, he, transform, threshold, clip):
    dataset = pixel_dataset(resident, he, transform, threshold, clip)
    for items in ([0, 3, 1, 0, 5, 2, 4], [7], [1, 4, 6, 7], [-1, 0]):
        samples = [dataset[i] for i in items]
        expected = sparse_collate(samples)
        assert_batch(dataset.get_batch(items), expected)
        observed = dataset.__getitems__(items)
        assert_batch(sparse_collate(observed), expected)
        for left, right in zip(samples, observed, strict=True):
            assert left.keys() == right.keys()
            for key in left:
                assert type(left[key]) is type(right[key])
                if isinstance(left[key], np.ndarray):
                    assert left[key].dtype == right[key].dtype
                np.testing.assert_array_equal(left[key], right[key])
    dataset.close()


@pytest.mark.parametrize('resident', [False, True])
def test_fallback_labels_empty_requests_and_splits(pixel_dataset, resident):
    dataset = pixel_dataset(resident, fallback=True)
    assert_batch(dataset.get_batch([0, 1]), sparse_collate([dataset[0], dataset[1]]))
    with pytest.raises(ValueError, match='empty sample'):
        dataset.get_batch([])
    assert dataset.__getitems__([]) == []
    dataset.indices = np.array([], dtype=np.int64)
    loader = DataLoader(dataset, batch_size=3, collate_fn=sparse_collate)
    assert list(loader) == []
    assert list(PreparedLoader(loader, performance_settings({}), {})) == []
    dataset.close()


@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('workers,context', [(0, None), (2, None), (2, 'spawn')])
def test_dataloader_repeated_epochs(pixel_dataset, resident, workers, context):
    dataset = pixel_dataset(resident)
    kwargs = dict(batch_size=3, shuffle=True, collate_fn=sparse_collate,
                  num_workers=workers, persistent_workers=bool(workers))
    if context is not None:
        kwargs['multiprocessing_context'] = context
    expected = DataLoader(ScalarDataset(dataset), generator=torch.Generator().manual_seed(19), **kwargs)
    actual = DataLoader(dataset, generator=torch.Generator().manual_seed(19), **kwargs)
    try:
        for _ in range(2):
            for left, right in zip(expected, actual, strict=True):
                assert_batch(left, right)
    finally:
        for loader in (expected, actual):
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()
        dataset.close()


@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('threads', [0, 2])
def test_direct_preparation_order_and_one_read(pixel_dataset, resident, threads, monkeypatch):
    dataset = pixel_dataset(resident)
    reference = [sparse_collate([dataset[i] for i in request])
                 for request in ([0, 3, 1], [0, 5, 2], [4])]
    dataset.close()
    reads, gathers = [], []
    original_read = dataset.he_reader.read
    def read(rows):
        reads.append(rows.copy())
        if rows[0] == 6:
            time.sleep(.01)
        return original_read(rows)
    monkeypatch.setattr(dataset.he_reader, 'read', read)
    if resident:
        gather = dataset.resident.gather
        def tracked(rows):
            gathers.append(rows.copy())
            return gather(rows)
        monkeypatch.setattr(dataset.resident, 'gather', tracked)
    def forbidden(*args):
        raise AssertionError('Direct preparation must not fetch scalar/sample dictionaries.')
    monkeypatch.setattr(SparseAnnDataDataset, '__getitem__', forbidden)
    monkeypatch.setattr(SparseAnnDataDataset, '__getitems__', forbidden)
    loader = DataLoader(dataset, batch_sampler=[[0, 3, 1], [0, 5, 2], [4]], collate_fn=sparse_collate)
    settings = {**performance_settings({}), 'preparation_workers': threads}
    prepared = PreparedLoader(loader, settings, {})
    for batch, expected in zip(prepared, reference, strict=True):
        assert batch.pop('spectral_chunks')
        assert_batch(batch, expected)
    assert len(reads) == 3
    assert len(gathers) == (3 if resident else 0)
    assert dataset._handle is None
    prepared.settings['host_budget_gib'] = 0
    with pytest.raises(MemoryError, match='host budget'):
        prepared.prepare([0])


def numpy_collator(samples):
    for sample in samples:
        assert isinstance(sample['row_id'], int)
        assert isinstance(sample['batch'], int)
        assert isinstance(sample['targets'], np.ndarray)
        assert sample['targets'].dtype == np.float64
    return sparse_collate(samples)


def test_custom_collator_and_prepared_rng(pixel_dataset):
    dataset = pixel_dataset()
    kwargs = dict(batch_size=3, shuffle=True, collate_fn=numpy_collator)
    reference = DataLoader(ScalarDataset(dataset), generator=torch.Generator().manual_seed(5), **kwargs)
    source = DataLoader(dataset, generator=torch.Generator().manual_seed(5), **kwargs)
    prepared = PreparedLoader(source, {**performance_settings({}), 'preparation_workers': 2}, {})
    for _ in range(2):
        for left, right in zip(reference, prepared, strict=True):
            right.pop('spectral_chunks')
            assert_batch(left, right)
        assert torch.equal(reference.generator.get_state(), source.generator.get_state())
    dataset.close()


@pytest.mark.parametrize('resident', [False, True])
def test_ordinary_loader_reads_he_once_per_batch(pixel_dataset, resident, monkeypatch):
    dataset = pixel_dataset(resident)
    reads = []
    original = dataset.he_reader.read
    def read(rows):
        reads.append(rows.copy())
        return original(rows)
    def forbidden(*args):
        raise AssertionError('PyTorch must use the dataset batch-fetch interface.')
    monkeypatch.setattr(dataset.he_reader, 'read', read)
    monkeypatch.setattr(SparseAnnDataDataset, '__getitem__', forbidden)
    batches = list(DataLoader(dataset, batch_size=3, collate_fn=numpy_collator))
    assert len(reads) == len(batches) == 3
    for rows, batch in zip(reads, batches, strict=True):
        np.testing.assert_array_equal(rows, batch['row_ids'].numpy())


def test_backed_group_reads_and_handle_cleanup(pixel_dataset, monkeypatch):
    dataset = pixel_dataset(he=False)
    opened, reads = [], []
    original_file, original_get = h5py.File, h5py.Dataset.__getitem__
    def track_file(*args, **kwargs):
        handle = original_file(*args, **kwargs)
        opened.append(handle)
        return handle
    def track_get(array, key):
        reads.append((array.name, key))
        return original_get(array, key)
    monkeypatch.setattr(h5py, 'File', track_file)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', track_get)
    dataset.get_batch([0, 3, 1, 0])  # global [6, 1, 0, 6]: two runs
    assert len(opened) == 1 and not opened[0].id.valid
    assert reads == [('/layers/msi/indptr', slice(0, 3)),
                     ('/layers/msi/indices', slice(0, 4)), ('/layers/msi/data', slice(0, 4)),
                     ('/layers/msi/indptr', slice(6, 8)),
                     ('/layers/msi/indices', slice(7, 10)), ('/layers/msi/data', slice(7, 10))]
    def failing_get(array, key):
        if array.name.endswith('/data'):
            raise OSError('simulated read failure')
        return original_get(array, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', failing_get)
    with pytest.raises(OSError, match='simulated'):
        dataset.get_batch([0])
    assert all(not handle.id.valid for handle in opened)


@pytest.mark.parametrize('failure', [False, True])
def test_thread_cleanup_after_early_close_or_failure(pixel_dataset, failure, monkeypatch):
    dataset = pixel_dataset(he=False)
    original_file = h5py.File
    opened = []
    def tracked(*args, **kwargs):
        handle = original_file(*args, **kwargs)
        opened.append(handle)
        return handle
    monkeypatch.setattr(h5py, 'File', tracked)
    loader = PreparedLoader(DataLoader(dataset, batch_size=2, collate_fn=sparse_collate),
        {**performance_settings({}), 'preparation_workers': 2}, {})
    if failure:
        def fail(values):
            raise RuntimeError('transformation failure')
        monkeypatch.setattr(dataset, '_transform_values', fail)
    iterator = iter(loader)
    try:
        if failure:
            with pytest.raises(RuntimeError, match='transformation'):
                next(iterator)
        else:
            next(iterator)
    finally:
        iterator.close()
    assert opened and all(not handle.id.valid for handle in opened)
    assert not any(t.name.startswith('dann-prepare') for t in threading.enumerate())


@pytest.mark.parametrize('value', [np.nan, np.inf, -1.])
def test_invalid_msi_source_row(pixel_dataset, value):
    dataset = pixel_dataset(he=False)
    with h5py.File(dataset.path, 'r+') as handle:
        handle['layers/msi/data'][7] = value
    for get in (lambda: dataset[0], lambda: dataset.get_batch([1, 0, 0])):
        with pytest.raises(ValueError, match='row 6'):
            get()
        dataset.close()
    with pytest.raises(ValueError, match='row 6'):
        pixel_dataset(resident=True, he=False)


@pytest.mark.parametrize('resident', [False, True])
def test_invalid_he_source_row(pixel_dataset, resident):
    dataset = pixel_dataset(resident)
    with h5py.File(dataset.path, 'r+') as handle:
        handle['obsm/he'][6, 0] = np.nan
    with pytest.raises(ValueError, match=r'positions: \[6\]'):
        dataset.get_batch([1, 0, 0])
    assert dataset._handle is None


@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_model_loss_gradients_update_and_source_isolation(he_config, resident, device, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    config = he_config
    for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
        group['type'] = 'mlp'
    config['model']['dropout'] = .2
    config['performance'] = {'ram_cache': resident, 'torch_threads': 2}
    resolve_execution(config)
    # An unreadable inference source must not participate in supervised loading.
    config['data']['context_path'] = '/absent/inference-only.h5ad'
    bundle = create_data_bundle(config)
    dataset = bundle.datasets['train']
    items = [5, 1, 5, 0, 7]
    expected, actual = sparse_collate([dataset[i] for i in items]), dataset.get_batch(items)
    assert_batch(actual, expected)
    torch.manual_seed(41)
    model = AdversarialLatentFusion.from_config(config, 2, 4).to(device)
    other = copy.deepcopy(model)
    losses, outputs = [], []
    for net, batch in ((model, expected), (other, actual)):
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
        batch = {key: value.to(device) for key, value in batch.items()}
        torch.manual_seed(73)
        result = net(batch, grl_strength=.1)
        loss = ZILNLoss.from_config(config).to(device)(result['pi_logits'], result['mu'],
            result['sigma'], batch['targets'], batch['target_valid_mask']).total
        loss += torch.nn.functional.cross_entropy(result['batch_logits'], batch['batches'])
        loss += he_mse(result['he_prediction'], batch['he_targets'])
        loss.backward()
        optimizer.step()
        losses.append(loss)
        outputs.append(result)
    torch.testing.assert_close(*losses)
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key])
    for left, right in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad)
        torch.testing.assert_close(left, right)
    dataset.close()


def test_benchmark_replays_requests_and_compares_training(he_config, tmp_path):
    from dann.verification.pixel_loader_benchmark import freeze_requests, run_pair, save_requests
    config = he_config
    for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
        group['type'] = 'mlp'
    config['performance'] = {'ram_cache': True, 'preparation_workers': 2, 'torch_threads': 2}
    resolve_execution(config)
    bundle = create_data_bundle(config)
    requests = freeze_requests(bundle, 1, 1, 1)
    save_requests(tmp_path, requests, bundle.split_indices)
    saved = np.load(tmp_path / 'row_requests.npz')
    assert saved['train_full_items'].dtype == np.int64
    assert sorted(saved['train_full_items']) == list(range(len(bundle.datasets['train'])))
    assert len(requests['stress'][-1]) < config['training']['batch_size']
    model = AdversarialLatentFusion.from_config(config, 2, 4)
    result = run_pair(bundle, config, requests, model.state_dict(), tmp_path, torch.device('cpu'), 0)
    assert result['state_close'] and result['max_state_absolute_error'] == 0
    assert result['train']['loss_absolute_difference'] == 0
    assert result['validation']['loss_absolute_difference'] == 0

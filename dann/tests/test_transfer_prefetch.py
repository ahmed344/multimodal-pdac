"""Persistent CUDA transfer streams, allocator reuse, and iterator lifetimes."""
import copy
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from dann.data_loader import create_data_bundle
from dann.execution import PreparedLoader, _transfer_stream, device_batches, tree_map
from dann.model import AdversarialLatentFusion
from dann.tests.test_components import spatial_config
from dann.tests.test_execution import enable


class ReplayLoader:
    """Replay fixed CPU batches and expose source cleanup, including failures."""

    def __init__(self, batches, prefetch=True):
        self.batches = batches
        self.settings = {'transfer_prefetch': prefetch}
        self.closed = 0
        self.fail_after = None

    def __iter__(self):
        try:
            for index, batch in enumerate(self.batches):
                if index == self.fail_after:
                    raise RuntimeError('source failure')
                yield batch
        finally:
            self.closed += 1


def test_stream_cache_device_aliases_and_loader_ownership(monkeypatch):
    """Exercise multi-device keys even on hosts with only one (or no) GPU."""
    created = []

    def stream(device):
        result = SimpleNamespace(device=device)
        created.append(result)
        return result

    monkeypatch.setattr(torch.cuda, 'Stream', stream)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    settings = {'validation_peak_budget': None, 'validation_gat_tiles': None}
    source = SimpleNamespace(dataset=[], sampler=[])
    first = PreparedLoader(source, settings, {})
    second = PreparedLoader(first, settings, {}, validation=True)
    assert created == []  # Construction never initializes CUDA.
    zero = _transfer_stream(first, torch.device('cuda'))
    assert _transfer_stream(first, torch.device('cuda:0')) is zero
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 1)
    one = _transfer_stream(first, torch.device('cuda'))
    assert _transfer_stream(first, torch.device('cuda:1')) is one
    assert one is not zero
    assert _transfer_stream(first, torch.device('cuda:0')) is zero
    assert _transfer_stream(second, torch.device('cuda:0')) is not zero
    assert len(created) == 3
    assert first._transfer_streams is not second._transfer_streams
    assert '_transfer_streams' not in settings


def test_cpu_transfer_does_not_initialize_cuda(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('CPU execution must not initialize a transfer stream')

    monkeypatch.setattr(torch.cuda, 'Stream', unexpected)
    loader = ReplayLoader([{'value': torch.arange(5)}])
    for cpu, batch in device_batches(loader, torch.device('cpu')):
        torch.testing.assert_close(batch, cpu)
    assert not hasattr(loader, '_transfer_streams')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA transfer stream required')
@pytest.mark.parametrize('loader_count', [1, 2])
def test_cuda_reserved_memory_stabilizes_across_40_passes(loader_count):
    """An old stream-per-pass implementation fills the 32-stream pool here."""
    # Isolate this measurement from allocations cached by earlier tests. Never
    # flush between passes: that would hide the stream ownership regression.
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    batch = {'value': torch.ones(2**21)}  # 8 MiB, outside the small-block pool.
    loaders = [ReplayLoader([batch] * 3) for _ in range(loader_count)]
    consumer = torch.cuda.Stream()

    def consume(loader, device):
        with torch.cuda.stream(consumer):
            for _, gpu in device_batches(loader, device):
                # Work on a different stream also exercises record_stream().
                gpu['value'].add_(1)

    reserved = []
    for index in range(40):
        loader = loaders[index % loader_count]
        device = torch.device('cuda' if index % 3 else f'cuda:{torch.cuda.current_device()}')
        consume(loader, device)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == baseline
        reserved.append(torch.cuda.memory_reserved())
    # Allow one allocator segment of variation after both loaders warm up.
    assert max(reserved[4:]) - min(reserved[4:]) <= 2 * 2**20, reserved
    assert all(len(loader._transfer_streams) == 1 for loader in loaders)
    assert sum(loader.closed for loader in loaders) == 40


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA transfer stream required')
@pytest.mark.parametrize('ending', ['empty', 'complete', 'close', 'source', 'transfer', 'consumer'])
def test_cuda_cleanup_and_subsequent_stream_reuse(ending, monkeypatch):
    batch = {'a': torch.arange(4096), 'b': torch.ones(4096)}
    loader = ReplayLoader([] if ending == 'empty' else [batch] * 4)
    device = torch.device('cuda')
    pinned_refs = []
    pin_memory = torch.Tensor.pin_memory

    def track_pin(tensor, *args, **kwargs):
        pinned = pin_memory(tensor, *args, **kwargs)
        pinned_refs.append(weakref.ref(pinned))
        return pinned

    monkeypatch.setattr(torch.Tensor, 'pin_memory', track_pin)
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()

    def run():
        iterator = device_batches(loader, device)
        if ending == 'source':
            loader.fail_after = 2  # Fail with both a current and prefetched batch.
        if ending == 'transfer':
            transfer = torch.Tensor.to
            calls = 0

            def fail_transfer(tensor, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError('transfer failure')
                return transfer(tensor, *args, **kwargs)

            monkeypatch.setattr(torch.Tensor, 'to', fail_transfer)
        if ending in ('close', 'consumer'):
            cpu, gpu = next(iterator)
            assert all(ref() is not None for ref in pinned_refs)
            if ending == 'consumer':
                with pytest.raises(RuntimeError, match='consumer failure'):
                    iterator.throw(RuntimeError('consumer failure'))
            else:
                iterator.close()
            torch.testing.assert_close(gpu['a'].cpu(), cpu['a'])
        elif ending in ('source', 'transfer'):
            with pytest.raises(RuntimeError, match=f'{ending} failure'):
                for _ in iterator:
                    pass
        else:
            for _ in iterator:
                pass
        assert loader.closed == 1
        stream = _transfer_stream(loader, device)
        assert stream.query()  # Cleanup has finished all DMA, even on failure.
        return stream

    stream = run()
    monkeypatch.undo()
    gc.collect()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == baseline
    assert all(ref() is None for ref in pinned_refs)
    loader.fail_after = None
    loader.batches = [batch] * 2
    for cpu, gpu in device_batches(loader, device):
        torch.testing.assert_close(tree_map(gpu, lambda t: t.cpu()), cpu)
    assert _transfer_stream(loader, device) is stream
    assert loader.closed == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA transfer stream required')
@pytest.mark.parametrize('component', ['cnn', 'gatv2'])
def test_cuda_prefetch_matches_data_outputs_and_gradients(spatial_config, component):
    config = enable(spatial_config, transfer_prefetch=True)
    config['model']['aggregation']['type'] = component
    config['model']['heads']['biology']['type'] = component
    config['model']['dropout'] = .2
    bundle = create_data_bundle(config)
    # Freeze inputs and order, including nested prepared graphs for GATv2.
    batches = list(bundle.loaders['train'])[:3]
    torch.manual_seed(61)
    reference = AdversarialLatentFusion.from_config(config, 2, 4).cuda().train()
    actual = copy.deepcopy(reference)

    def execute(model, loader):
        torch.manual_seed(92)
        results = []
        for index, (cpu, gpu) in enumerate(device_batches(loader, torch.device('cuda'))):
            torch.testing.assert_close(tree_map(gpu, lambda t: t.cpu()), batches[index],
                                       rtol=0, atol=0, equal_nan=True)
            model.zero_grad(set_to_none=True)
            output = model(gpu, grl_strength=.1)
            sum(value.square().mean() for value in output.values()).backward()
            results.append((tree_map(output, lambda t: t.detach().cpu()),
                            {name: p.grad.cpu().clone() for name, p in model.named_parameters()
                             if p.grad is not None}))
        return results

    expected = execute(reference, ReplayLoader(batches, prefetch=False))
    loader = ReplayLoader(batches)
    for _ in range(3):
        observed = execute(actual, loader)
        torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-6)

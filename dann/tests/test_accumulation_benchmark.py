"""Benchmark resets, real loader coverage, and comparison/failure contracts."""
import copy
from types import SimpleNamespace

import pytest
import torch

from dann.config import resolve_execution
from dann.data_loader import create_data_bundle
from dann.tests.test_components import spatial_config
from dann.tests.test_he import he_config
from dann.train import TrainingProgress
from dann.verification.core_batch_pilot import pilot_config
from dann.verification.accumulation_benchmark import (
    AuditedLoader, benchmark_config, execute_epoch, failure_kind, reset_training,
    run, state_hash, summarize, verify_epoch,
)


@pytest.mark.parametrize('steps,micro', [(1, None), (2, 16384), (4, 8192), (8, 4096)])
def test_settings_change_only_accumulation(he_config, tmp_path, steps, micro):
    base = pilot_config(he_config, 'candidate', tmp_path)
    saved = copy.deepcopy(base)
    config = benchmark_config(base, steps)
    assert base == saved
    assert config['training']['microbatch_rows'] == micro
    assert config['training']['supervised_rows_per_batch'] == 32768
    assert config['model'] == base['model']
    assert config['loss'] == base['loss']
    assert config['training']['grl_horizon_updates'] == 500
    with pytest.raises(ValueError):
        benchmark_config(base, 3)


def test_warmup_reset_restores_parameters_optimizer_rng_and_counters():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    source = dict(model_state=copy.deepcopy(model.state_dict()),
                  optimizer_state=copy.deepcopy(optimizer.state_dict()),
                  config={'training': {'deterministic_algorithms': False}})
    initial = state_hash(model.state_dict())
    reset_training(model, optimizer, source, 17, 2)
    draws = torch.randn(4)
    model(torch.randn(5, 3)).square().sum().backward()
    optimizer.step()
    assert optimizer.state and state_hash(model.state_dict()) != initial
    progress = reset_training(model, optimizer, source, 17, 2)
    assert progress == TrainingProgress(epoch=2)
    assert state_hash(model.state_dict()) == initial
    assert not optimizer.state and all(p.grad is None for p in model.parameters())
    torch.testing.assert_close(draws, torch.randn(4))


@pytest.mark.parametrize('steps', [1, 2, 4, 8])
def test_real_loader_coverage_and_partial_group(he_config, steps):
    he_config['training']['spatial'].update(
        supervised_rows_per_batch=24, microbatch_rows=None if steps == 1 else 24 // steps)
    resolve_execution(he_config)
    bundle = create_data_bundle(he_config, requested_splits=('train',))
    loader = bundle.loaders['train']
    expected = list(loader.sampler.logical_requests())
    audited = AuditedLoader(loader, expected)
    try:
        list(audited)
        count = len(expected)
        progress = TrainingProgress(count, len(bundle.split_indices['train']), 1, 0)
        verified = verify_epoch(audited.records, expected, bundle.split_indices['train'],
                                progress, count, 0)
        assert verified['passed']
        assert verified['final_partial_rows'] < 24
        assert audited.records[0]['microbatches'] == steps
        with pytest.raises(ValueError, match='counters'):
            verify_epoch(audited.records, expected, bundle.split_indices['train'], progress, count - 1, 0)
        audited.records[0]['row_ids_sha256'] = 'corrupted'
        with pytest.raises(ValueError, match='hashes'):
            verify_epoch(audited.records, expected, bundle.split_indices['train'], progress, count, 0)
    finally:
        for dataset in bundle.datasets.values():
            dataset.close()


def trial(steps, repetition, seconds, allocated, status='passed'):
    return dict(steps=steps, repetition=repetition, status=status,
                epoch_seconds=seconds, seconds_per_update=seconds / 47,
                rows_per_second=1528567 / seconds, peak_allocated_gib=allocated,
                peak_reserved_gib=allocated + 1, peak_host_gib=20,
                warmup_memory={'peak_allocated_gib': allocated, 'peak_reserved_gib': allocated + 1},
                coverage={'row_order_sha256': str(repetition)}, initial_state_sha256='same')


def test_selection_rejects_incomplete_or_over_budget_and_prefers_memory_within_three_percent():
    records = [trial(s, r, time, memory) for r in range(3)
               for s, time, memory in [(1, 100, 22), (2, 102, 12), (4, 110, 8), (8, 120, 5)]]
    result = summarize(records, 3)
    assert result['recommended_steps'] == 2
    assert result['minimum_passing_steps'] == 1
    assert result['settings'][0]['speedup_over_8'] == 1.2
    records[0]['status'] = 'budget_exceeded'
    assert summarize(records, 3)['minimum_passing_steps'] == 2
    assert summarize(records[:4], 3)['recommended_steps'] is None
    records[1]['coverage']['row_order_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='row sequences'):
        summarize(records, 3)


def test_cuda_oom_and_configured_budget_are_distinct(monkeypatch):
    assert failure_kind(torch.cuda.OutOfMemoryError('allocation')) == 'cuda_oom'
    assert failure_kind(MemoryError('GPU budget exceeded')) == 'memory_budget'
    assert failure_kind(MemoryError('allocation')) == 'host_oom'

    def fail(**kwargs):
        raise MemoryError('Measured execution exceeded VRAM budget.')

    monkeypatch.setattr('dann.verification.accumulation_benchmark.run_epoch', fail)
    assert execute_epoch({}) == {'configured_gpu_budget_exceeded': True}

    def oom(**kwargs):
        raise torch.cuda.OutOfMemoryError('allocation')

    monkeypatch.setattr('dann.verification.accumulation_benchmark.run_epoch', oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        execute_epoch({})


def test_parent_uses_module_and_records_child_failure_without_fallback(tmp_path, monkeypatch):
    commands = []

    def child(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr('dann.verification.accumulation_benchmark.subprocess.run', child)
    args = SimpleNamespace(output=tmp_path / 'bench', checkpoint=tmp_path / 'initial.pt',
                           minutes=1, repeats=3, warmup=5)
    result = run(args)
    assert len(commands) == 4  # Failed settings are not retried with different accumulation.
    assert all(c[3] == 'dann.verification.accumulation_benchmark' for c in commands)
    assert [r['steps'] for r in result['trials']] == [1, 2, 4, 8]
    assert all(r['status'] == 'process_error' for r in result['trials'])
    assert result['summary']['recommended_steps'] is None

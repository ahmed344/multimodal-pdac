"""Snapshot isolation, concurrent publication, and architecture-wide resumes."""
import copy
from functools import partial
import itertools
import multiprocessing
import os
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import pytest
import torch

from dann import train
from dann import training_artifacts as artifacts_module
from dann.config import load_config, performance_settings, resolve_execution
from dann.data_loader import create_data_bundle
from dann.model import AdversarialLatentFusion
from dann.tests.test_components import spatial_config
from dann.tests.test_he import he_config
from dann.training_artifacts import TrainingArtifacts, atomic_output, cpu_snapshot


def gated_write(started, release, job):
    started.set()
    if not release.wait(30):
        raise TimeoutError('Writer gate was not released.')
    return artifacts_module._write_job(job)


def gated_plot(started, release, job):
    started.set()
    if not release.wait(30):
        raise TimeoutError('Plot gate was not released.')
    return artifacts_module._plot_job(job)


def failed_job(job):
    raise OSError('deliberate artifact failure')


def crashed_job(job):
    os._exit(7)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.fixture(scope='module')
def worker(tmp_path_factory):
    config = load_config(Path('dann/config.yaml'))
    config['model']['heads']['he']['enabled'] = False
    config['analysis']['figure_dpi'] = 12
    manager = TrainingArtifacts(tmp_path_factory.mktemp('artifact-workers'), config)
    yield manager
    manager.close()


@pytest.mark.parametrize('choices', list(itertools.product(('mlp', 'cnn', 'gatv2'), repeat=3)))
def test_all_architectures_snapshot_reload_next_update(spatial_config, choices, worker, tmp_path):
    config = spatial_config
    for group, kind in zip([config['model']['aggregation'], *config['model']['heads'].values()], choices):
        group['type'] = kind
    resolve_execution(config)
    bundle = create_data_bundle(config)
    try:
        model = AdversarialLatentFusion.from_config(config, 2, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        batch = next(iter(bundle.loaders['train']))

        def update(network, opt):
            network.train()
            opt.zero_grad()
            outputs = network(batch)
            sum(value.square().mean() for value in outputs.values()).backward()
            opt.step()

        update(model, optimizer)
        sync = tmp_path / 'sync.pt'
        train.save_checkpoint(sync, model, optimizer, 0, config, bundle, 1., best_epoch=1)
        state = worker.snapshot(model, optimizer)
        payload = train.checkpoint_payload(train.checkpoint_metadata(model, config, bundle), state, 0, 1., best_epoch=1)
        worker.submit_epoch([dict(epoch=1, train_total_loss=1.)], ['epoch'], {},
                            dict(best_epoch=1, selection_metric='supervised_objective'), payload, ['best.pt', 'latest.pt'])
        # Updating live CPU parameters and Adam state must not change the submitted epoch.
        update(model, optimizer)
        worker.flush()
        saved = torch.load(worker.directory / 'best.pt', weights_only=False)
        assert_nested_equal(saved, torch.load(sync, weights_only=False))
        assert_nested_equal(saved, torch.load(worker.directory / 'latest.pt', weights_only=False))
        restored = AdversarialLatentFusion.from_config(config, 2, 4)
        restored_opt = torch.optim.AdamW(restored.parameters())
        assert train.load_training_checkpoint(worker.directory / 'best.pt', restored, restored_opt,
                                              torch.device('cpu'), config) == (1, 1.)
        update(restored, restored_opt)
        assert_nested_equal(model.state_dict(), restored.state_dict())
        assert_nested_equal(optimizer.state_dict(), restored_opt.state_dict())
        model.eval()
        restored.eval()
        assert_nested_equal(model(batch), restored(batch))
    finally:
        for dataset in bundle.datasets.values():
            dataset.close()


@pytest.mark.parametrize('kind', ['mlp', 'cnn', 'gatv2'])
def test_he_architectures_snapshot(he_config, kind, worker):
    config = he_config
    for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
        group['type'] = kind
    resolve_execution(config)
    bundle = create_data_bundle(config)
    try:
        model = AdversarialLatentFusion.from_config(config, 2, 4)
        optimizer = torch.optim.AdamW(model.parameters())
        batch = next(iter(bundle.loaders['train']))
        model(batch)['he_prediction'].square().mean().backward()
        optimizer.step()
        frozen = worker.snapshot(model, optimizer)
        expected = copy.deepcopy(frozen)
        payload = train.checkpoint_payload(train.checkpoint_metadata(model, config, bundle), frozen, 2, 1.)
        worker.submit_epoch([dict(epoch=3)], ['epoch'], {}, {}, payload, ['latest.pt'])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(100.)
        worker.flush()
        saved = torch.load(worker.directory / 'latest.pt', weights_only=False)
        assert_nested_equal(saved['model_state'], expected['model_state'])
        assert_nested_equal(saved['optimizer_state'], expected['optimizer_state'])
    finally:
        for dataset in bundle.datasets.values():
            dataset.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_snapshot_copies_cpu_and_cuda_and_preserves_metadata(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model = torch.nn.Linear(3, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2, 3, device=device)).sum().backward()
    optimizer.step()
    source = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict())
    frozen = cpu_snapshot(source)
    assert frozen['model_state']._metadata == source['model_state']._metadata
    expected = copy.deepcopy(frozen)
    for tensor in model.parameters():
        tensor.data.add_(10.)
    optimizer.step()
    assert_nested_equal(expected, frozen)
    assert all(value.device.type == 'cpu' for value in frozen['model_state'].values())


def test_backlog_coalesces_but_keeps_best_history_and_audits(tmp_path, monkeypatch):
    config = load_config(Path('dann/config.yaml'))
    config['analysis']['figure_dpi'] = 12
    with multiprocessing.get_context('spawn').Manager() as gates:
        write_started, plot_started, release = gates.Event(), gates.Event(), gates.Event()
        monkeypatch.setattr(artifacts_module, '_write_job', partial(gated_write, write_started, release))
        monkeypatch.setattr(artifacts_module, '_plot_job', partial(gated_plot, plot_started, release))
        manager = TrainingArtifacts(tmp_path, config)
        history = []
        try:
            for epoch in range(1, 5):
                history.append(dict(epoch=epoch, train_total_loss=1 / epoch))
                state = dict(epoch=epoch-1, value=torch.tensor([float(epoch)]))
                targets = ['latest.pt', 'best.pt'] if epoch <= 2 else ['latest.pt']
                manager.submit_epoch(history, ['epoch', 'train_total_loss'], {},
                                     dict(best_epoch=min(epoch, 2), selection_metric='supervised_objective'), state, targets)
                if epoch == 1:
                    assert write_started.wait(30) and plot_started.wait(30)
                manager.submit_audit(tmp_path / 'sampling', epoch-1, [dict(epoch=epoch)], dict(epoch=epoch))
            assert not release.is_set()  # All subsequent epochs could submit while both workers were blocked.
            assert manager.report['coalesced_checkpoints'] >= 2
            assert manager.report['coalesced_plots'] >= 2
        finally:
            release.set()
            manager.close()
        assert torch.load(tmp_path / 'best.pt', weights_only=False)['epoch'] == 1
        assert torch.load(tmp_path / 'latest.pt', weights_only=False)['epoch'] == 3
        assert pd.read_csv(tmp_path / 'history.csv').epoch.tolist() == [1, 2, 3, 4]
        assert len(list((tmp_path / 'sampling').glob('*_summary.json'))) == 4
        assert [job['history_epoch'] for job in manager.report['jobs'] if job['kind'] == 'plot'] == [1, 4]
        assert not list(tmp_path.glob('.*.pt'))


@pytest.mark.parametrize('failed_kind', ['write', 'plot'])
@pytest.mark.parametrize('failure', [failed_job, crashed_job])
def test_worker_failure_is_reported_and_shutdown(tmp_path, monkeypatch, failed_kind, failure):
    config = load_config(Path('dann/config.yaml'))
    config['analysis']['figure_dpi'] = 12
    monkeypatch.setattr(artifacts_module, f'_{failed_kind}_job', failure)
    manager = TrainingArtifacts(tmp_path, config)
    manager.submit_epoch([dict(epoch=1)], ['epoch'], {}, {})
    with pytest.raises(RuntimeError, match='Background training artifact output failed'):
        manager.close()
    assert manager.closed and not manager.thread.is_alive()


def test_memory_pressure_waits_for_required_output_without_dropping_it(tmp_path, monkeypatch):
    config = load_config(Path('dann/config.yaml'))
    config['analysis']['figure_dpi'] = 12
    with multiprocessing.get_context('spawn').Manager() as gates:
        started, release = gates.Event(), gates.Event()
        monkeypatch.setattr(artifacts_module, '_write_job', partial(gated_write, started, release))
        manager = TrainingArtifacts(tmp_path, config)
        entered, admitted = threading.Event(), threading.Event()
        errors = []
        try:
            manager.submit_audit(tmp_path / 'sampling', 0, [dict(epoch=0)], dict(epoch=0))
            assert started.wait(30)
            # Simulate a full host budget until the required active audit finishes.
            monkeypatch.setattr(manager, '_rss', lambda: manager.budget if manager.active else 0)

            def reserve():
                try:
                    entered.set()
                    manager.reserve(1024)
                    admitted.set()
                except BaseException as error:
                    errors.append(error)

            waiter = threading.Thread(target=reserve)
            waiter.start()
            assert entered.wait(5)
            assert not admitted.is_set()
            release.set()
            waiter.join(30)
            assert not waiter.is_alive() and admitted.is_set() and not errors
        finally:
            release.set()
            manager.close()
        assert (tmp_path / 'sampling' / 'epoch_0000_summary.json').exists()


@pytest.mark.parametrize('metric, expected_epoch', [('supervised_objective', 2), ('cd8_loss', 3)])
def test_async_selection_and_resume(spatial_config, monkeypatch, metric, expected_epoch):
    from dann.tests.test_objectives import test_training_selection_roundtrip_and_resume
    spatial_config['performance']['async_artifacts'] = True
    spatial_config['analysis']['figure_dpi'] = 12
    test_training_selection_roundtrip_and_resume(spatial_config, monkeypatch, metric, expected_epoch)


@pytest.mark.parametrize('interruption', [KeyboardInterrupt, RuntimeError])
def test_next_epoch_starts_before_outputs_finish_and_interruption_drains(spatial_config, monkeypatch, interruption):
    config = spatial_config
    config['performance']['async_artifacts'] = True
    config['analysis']['figure_dpi'] = 12
    config['training']['epochs'] = 3
    monkeypatch.setattr(train, '_format_epoch_summary', lambda *args: 'synthetic epoch')
    with multiprocessing.get_context('spawn').Manager() as gates:
        write_started, plot_started, release = gates.Event(), gates.Event(), gates.Event()
        monkeypatch.setattr(artifacts_module, '_write_job', partial(gated_write, write_started, release))
        monkeypatch.setattr(artifacts_module, '_plot_job', partial(gated_plot, plot_started, release))

        def epoch(**kwargs):
            if kwargs['epoch_index'] == 1:
                assert write_started.wait(30) and plot_started.wait(30)
                assert not release.is_set()
                release.set()
                raise interruption('stop during the next epoch')
            return dict(supervised_objective=1., cd8_loss=1.)

        monkeypatch.setattr(train, 'run_epoch', epoch)
        try:
            with pytest.raises(interruption, match='stop during the next epoch'):
                train.train_model(config)
        finally:
            release.set()
        output = Path(config['training']['output_dir'])
        assert torch.load(output / 'latest.pt', weights_only=False)['epoch'] == 0
        assert torch.load(output / 'best.pt', weights_only=False)['epoch'] == 0
        assert pd.read_csv(output / 'history.csv').epoch.tolist() == [1]
        assert (output / 'loss_curves.png').exists()


def test_atomic_failure_retains_previous_file(tmp_path):
    path = tmp_path / 'best.pt'
    path.write_bytes(b'previous')
    with pytest.raises(OSError):
        with atomic_output(path) as temporary:
            temporary.write_bytes(b'incomplete')
            raise OSError('disk failure')
    assert path.read_bytes() == b'previous'
    assert list(tmp_path.iterdir()) == [path]


def test_async_setting_validation_and_memory_limit(tmp_path):
    assert not performance_settings({})['async_artifacts']
    with pytest.raises(ValueError, match='async_artifacts must be boolean'):
        performance_settings(dict(performance=dict(async_artifacts=1)))
    config = load_config(Path('dann/config.yaml'))
    assert config['performance']['async_artifacts']
    config['performance']['host_budget_gib'] = .001
    with pytest.raises(MemoryError, match='host_budget_gib'):
        TrainingArtifacts(tmp_path, config)

"""Progress timing, execution-only settings, and unavailable alignment diagnostics."""
import copy
import math
from types import SimpleNamespace

import pytest
import torch

from dann.batch_alignment import AlignmentSummary, alignment_settings, evaluate_alignment
from dann.config import load_config, resolve_execution
from dann.progress import EpochProgress
from dann.tests.test_conditional_alignment import fixture
from dann.tests.test_batch_alignment import alignment_epoch_fixture
from dann.train import resume_training_config, run_epoch


def test_progress_first_update_interval_boundaries_and_resume(monkeypatch, capsys):
    clock = [0.]
    monkeypatch.setattr('dann.progress.perf_counter', lambda: clock[0])
    progress = EpochProgress(2, 300, 30.)
    progress.set_phase('training', 10, unit='updates', completed=4)
    clock[0] = 2.
    progress.update(5, rows=4096)
    first = capsys.readouterr().out
    assert 'starting (4/10 updates)' in first
    assert '5/10 updates, 0.50 updates/s, 2048.0 rows/s' in first
    assert 'ETA 10.0s' in first
    clock[0] = 31.
    progress.update(6, rows=8192)
    assert not capsys.readouterr().out
    clock[0] = 32.
    progress.update(7, rows=12288)
    assert '7/10 updates' in capsys.readouterr().out
    progress.set_phase('feature encoding', 1024)
    boundary = capsys.readouterr().out
    assert 'training: finished 7/10' in boundary
    assert 'feature encoding: starting' in boundary
    progress.update(1024)
    progress.set_phase('validation', 100)
    progress.update(100)
    progress.finish_phase()
    assert 'validation: finished 100/100 rows' in capsys.readouterr().out


@pytest.mark.parametrize('value', [0, -1, True, None, '30', math.inf, math.nan])
def test_progress_interval_rejects_invalid_values(value):
    config = load_config('dann/config.yaml')
    config['training']['progress_interval_seconds'] = value
    with pytest.raises(ValueError, match='progress_interval_seconds'):
        resolve_execution(config)


def test_progress_interval_is_defaulted_and_can_change_on_resume():
    saved = load_config('dann/config.yaml')
    del saved['training']['progress_interval_seconds']
    assert resolve_execution(saved)['training']['progress_interval_seconds'] == 30.
    requested = copy.deepcopy(saved)
    requested['training']['progress_interval_seconds'] = 5.5
    assert resume_training_config(saved, requested)['training']['progress_interval_seconds'] == 5.5


@pytest.mark.parametrize('training', [False, True])
def test_progress_preserves_epoch_results_gradients_and_rng(training, monkeypatch, capsys):
    model, batch, kwargs = alignment_epoch_fixture()
    kwargs['loader'].dataset = SimpleNamespace(indices=batch['row_ids'])
    kwargs['validation_alignment_rows'] = batch['row_ids'].tolist()
    results = []
    # A CPU epoch has no reason to synchronize CUDA, even when reporting.
    def unexpected_sync(*args, **kw):
        raise AssertionError('Progress must not synchronize CUDA.')
    monkeypatch.setattr(torch.cuda, 'synchronize', unexpected_sync)
    for interval in (None, 30.):
        candidate = copy.deepcopy(model)
        optimizer = torch.optim.SGD(candidate.parameters(), lr=.01) if training else None
        torch.manual_seed(31)
        metrics = run_epoch(model=candidate, optimizer=optimizer,
                            progress_interval_seconds=interval, **kwargs)
        results.append((candidate, metrics, torch.get_rng_state()))
    for key, value in results[0][1].items():
        assert results[1][1][key] == pytest.approx(value, nan_ok=True)
    for left, right in zip(results[0][0].parameters(), results[1][0].parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        if training:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
    assert torch.equal(results[0][2], results[1][2])
    phase = 'training' if training else 'validation'
    assert f'{phase}: finished' in capsys.readouterr().out


@pytest.mark.parametrize('conditioned', [False, True])
def test_undefined_soft_nn_keeps_weighted_objectives_and_gradients_valid(conditioned):
    # Orthonormal rows have zero off-diagonal similarity variance but valid
    # NHSIC/ANOVA. Exercise the precise validation failure from the prior run.
    config, _, _, features, _ = fixture(n=40)
    config['loss']['batch_alignment']['soft_nn']['weight'] = 0.
    settings = alignment_settings(config)
    latent = torch.eye(40, dtype=torch.double, requires_grad=True)
    measurements = evaluate_alignment(latent, torch.arange(40) % 3,
                                      conditioning=features if conditioned else None)
    assert measurements['soft_nn'].reason == 'degenerate_similarity_variance'
    summary = AlignmentSummary(settings)
    summary.add(measurements)
    result = summary.finish()
    assert math.isnan(result['alignment_soft_nn'])
    assert result['alignment_soft_nn_weighted_loss'] == 0.
    assert result['alignment_soft_nn_invalid_degenerate_similarity_variance'] == 1
    objective = sum(settings[n]['weight'] * measurements[n].score for n in ('nhsic', 'anova'))
    assert torch.isfinite(objective)
    objective.backward()
    assert torch.isfinite(latent.grad).all()
    settings['soft_nn']['weight'] = .001
    with pytest.raises(ValueError, match='no valid evaluations'):
        summary.finish()

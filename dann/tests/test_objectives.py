"""Weighted reporting, selection boundaries, and historical curve contracts."""

import copy
import math
from pathlib import Path
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from dann import train
from dann.config import load_config, resolve_execution
from dann.losses import ZILNLoss
from dann.objective_curves import plot_loss_curves, weighted_history
from dann.selection import checkpoint_selection, selection_metric
from dann.tests.test_components import spatial_config


TARGETS = ["Density_CD8", "Density_Tumor", "Density_Collagen", "Density_Stroma"]
WEIGHTS = [8., 3., .5, 2.]


def reference_components(outputs, targets, valid):
    """Independent elementwise likelihood reference before reduction."""
    observed = torch.where(valid, targets, torch.zeros_like(targets))
    hurdle = F.binary_cross_entropy_with_logits(outputs['pi_logits'], (observed == 0).double(), reduction='none')
    positive = .5 * ((torch.logit(observed.clamp(1e-6, 1-1e-6)) - outputs['mu']) / outputs['sigma']).square()
    positive = positive + outputs['sigma'].log()
    weights = torch.tensor(WEIGHTS, dtype=torch.float64)
    return (torch.where(valid, hurdle, 0.) * weights,
            torch.where(valid & (observed > 0), positive, 0.) * weights)


class TableModel(torch.nn.Module):
    def __init__(self, he):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.linspace(-1, 1, 24, dtype=torch.float64).reshape(6, 4))
        self.mu = torch.nn.Parameter(torch.zeros(6, 4, dtype=torch.float64))
        self.log_sigma = torch.nn.Parameter(torch.full((6, 4), math.log(.45), dtype=torch.float64))
        self.batch_logits = torch.nn.Parameter(torch.tensor([[1., -1.]] * 6, dtype=torch.float64))
        self.he_predictor = torch.nn.Identity() if he else None
        self.he = torch.nn.Parameter(torch.zeros(6, 2, dtype=torch.float64)) if he else None

    def forward(self, batch, **kwargs):
        rows = batch['rows']
        output = dict(pi_logits=self.logits[rows], mu=self.mu[rows], sigma=self.log_sigma[rows].exp(),
                      batch_logits=self.batch_logits[rows])
        if self.he is not None:
            output['he_prediction'] = self.he[rows]
        return output


class Batches(list):
    sampler = None


def objective_batches():
    targets = torch.tensor([[0., .5, .6, 0.], [.5, .4, 0., .5], [.6, 0., .5, .4],
                            [0., .6, .4, .5], [.5, .5, .6, .4], [.4, 0., .5, .6]], dtype=torch.float64)
    valid = torch.ones_like(targets, dtype=torch.bool)
    valid[0, :] = False
    valid[2, 0] = False
    valid[3, 2:] = False
    targets[~valid] = math.nan
    return Batches(dict(rows=torch.arange(a, b), targets=targets[a:b], target_valid_mask=valid[a:b],
                        batches=torch.arange(a, b) % 2,
                        he_targets=torch.arange(a*2, b*2, dtype=torch.float64).reshape(-1, 2) / 10)
                   for a, b in ((0, 2), (2, 5), (5, 6)))


@pytest.mark.parametrize('reduction', ['mean', 'sum'])
@pytest.mark.parametrize('he', [False, True])
def test_weighted_epoch_decomposition_and_unchanged_gradients(reduction, he):
    batches = objective_batches()
    model = TableModel(he)
    reference = copy.deepcopy(model)
    loss = ZILNLoss(WEIGHTS, reduction=reduction)
    components, masses = [], []
    for batch in batches:
        outputs = model(batch)
        hurdle, positive = reference_components(outputs, batch['targets'], batch['target_valid_mask'])
        mass = (batch['target_valid_mask'] * torch.tensor(WEIGHTS)).sum()
        denominator = mass if reduction == 'mean' else 1.
        result = loss(outputs['pi_logits'], outputs['mu'], outputs['sigma'], batch['targets'], batch['target_valid_mask'])
        expected = hurdle.sum() / denominator + positive.sum() / denominator
        torch.testing.assert_close(result.total, expected, rtol=0, atol=1e-14)
        torch.testing.assert_close(result.hurdle_per_target, hurdle.sum(0) / denominator)
        torch.testing.assert_close(result.positive_per_target, positive.sum(0) / denominator)
        assert not result.hurdle_per_target.requires_grad and not result.positive_per_target.requires_grad
        actual_grad = torch.autograd.grad(result.total, (model.logits, model.mu, model.log_sigma), retain_graph=True)
        expected_grad = torch.autograd.grad(expected, (model.logits, model.mu, model.log_sigma))
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)
        reporting_mass = mass if reduction == 'mean' else len(batch['rows'])
        components.append(torch.stack([hurdle.sum(0), positive.sum(0)]).detach() / denominator * reporting_mass)
        masses.append(reporting_mass)

    kwargs = dict(loader=batches, ziln_loss=loss, device=torch.device('cpu'), biology_weight=5.,
                  batch_weight=.3, he_weight=2., class_weights=None, epoch_index=0, total_epochs=1,
                  grl_schedule='constant', grl_max_lambda=.2, grl_gamma=10., gradient_clip_norm=None,
                  target_columns=TARGETS)
    metrics = train.run_epoch(model=model, optimizer=None, **kwargs)
    expected_components = 5 * sum(components) / sum(masses)
    for index, target in enumerate(('cd8', 'tumor', 'collagen', 'stroma')):
        for branch, name in enumerate(('hurdle', 'positive')):
            assert metrics[f'{name}_weighted_loss_{target}'] == pytest.approx(expected_components[branch, index].item())
    assert metrics['biology_weighted_loss'] == pytest.approx(expected_components.sum().item())
    assert metrics['hurdle_weighted_loss'] + metrics['positive_weighted_loss'] == pytest.approx(metrics['biology_weighted_loss'])
    assert metrics['positive_weighted_loss'] < 0
    assert metrics['supervised_objective'] == pytest.approx(metrics['biology_weighted_loss'] + metrics.get('he_weighted_loss', 0.))
    assert metrics['total_loss'] == pytest.approx(metrics['supervised_objective'] + metrics['batch_weighted_loss'])
    assert ('he_weighted_loss' in metrics) == he
    assert ('he_r2' in metrics) == he
    if he:
        he_targets = torch.cat([batch['he_targets'] for batch in batches])
        expected_r2 = 1 - he_targets.square().sum() / (he_targets - he_targets.mean(0)).square().sum()
        assert metrics['he_r2'] == pytest.approx(expected_r2.item())

    # Compare real optimizer updates with the original weighted scalar objective.
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
    train.run_epoch(model=model, optimizer=optimizer, **kwargs)
    for batch in batches:
        reference_optimizer.zero_grad()
        outputs = reference(batch)
        hurdle, positive = reference_components(outputs, batch['targets'], batch['target_valid_mask'])
        denominator = (batch['target_valid_mask'] * torch.tensor(WEIGHTS)).sum() if reduction == 'mean' else 1.
        objective = 5 * (hurdle.sum() / denominator + positive.sum() / denominator)
        objective += .3 * F.cross_entropy(outputs['batch_logits'], batch['batches'])
        if he:
            objective += 2 * F.mse_loss(outputs['he_prediction'], batch['he_targets'])
        objective.backward()
        reference_optimizer.step()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)


def test_selection_config_defaults_and_validation():
    config = load_config(Path('dann/config.yaml'))
    assert selection_metric(config) == 'supervised_objective'
    del config['training']['early_stopping_metric']
    assert resolve_execution(config)['training']['early_stopping_metric'] == 'cd8_loss'
    config['training']['early_stopping_metric'] = 'total_loss'
    with pytest.raises(ValueError, match='early_stopping_metric'):
        resolve_execution(config)


@pytest.mark.parametrize('metric, expected_epoch', [('supervised_objective', 2), ('cd8_loss', 3)])
def test_training_selection_roundtrip_and_resume(spatial_config, monkeypatch, metric, expected_epoch):
    config = spatial_config
    config['training'].update(epochs=3, early_stopping_patience=1, early_stopping_metric=metric,
                              checkpoint_every=10)
    monkeypatch.setattr(train, 'refresh_loss_curves', Mock())
    monkeypatch.setattr(train, '_format_epoch_summary', lambda *args: 'synthetic epoch')
    # Supervised prefers epoch 2; CD8 and total prefer epoch 3; discriminator prefers epoch 1.
    rows = [dict(cd8_loss=c, supervised_objective=s, batch_weighted_loss=b, total_loss=s+b)
            for c, s, b in [(3., 5., 0.), (2., 3., 2.), (1., 4., .5)]]
    epoch = Mock(side_effect=[row for row in rows for _ in range(2)])
    monkeypatch.setattr(train, 'run_epoch', epoch)
    best = train.train_model(config)
    payload = torch.load(best, weights_only=False)
    latest = torch.load(best.with_name('latest.pt'), weights_only=False)
    assert payload['epoch'] + 1 == expected_epoch
    assert payload['best_epoch'] == latest['best_epoch'] == expected_epoch
    assert latest['selection_metric'] == metric
    assert latest['best_selection_score'] == rows[expected_epoch-1][metric]
    assert 'best_validation_cd8' not in latest and 'best_validation_biology' not in latest
    assert latest['early_stopping_state']['stale_epochs'] == (1 if metric == 'supervised_objective' else 0)
    assert latest['selection_start_epoch'] == 1

    requested = copy.deepcopy(config)
    requested['training'].update(resume_checkpoint=str(best.with_name('latest.pt')), epochs=4)
    # State must survive even without history and when min_delta changes.
    history_path = best.with_name('history.csv')
    history_text = history_path.read_text()
    history_path.unlink()
    requested['training']['early_stopping_min_delta'] = .01
    if metric == 'supervised_objective':
        train.train_model(requested)
        assert epoch.call_count == 6
    history_path.write_text(history_text)
    switched = 'cd8_loss' if metric == 'supervised_objective' else 'supervised_objective'
    requested['training'].update(early_stopping_metric=switched, epochs=3)
    with pytest.raises(ValueError, match='at least one additional epoch'):
        train.train_model(requested)
    requested['training']['epochs'] = 4
    fresh_row = dict(cd8_loss=100., supervised_objective=100.)
    epoch.side_effect = [fresh_row, fresh_row]
    train.train_model(requested)
    switched_payload = torch.load(best, weights_only=False)
    assert switched_payload['best_epoch'] == switched_payload['selection_start_epoch'] == 4
    assert switched_payload['selection_metric'] == switched
    assert switched_payload['best_selection_score'] == 100.
    assert switched_payload['early_stopping_state']['stale_epochs'] == 0
    history = pd.read_csv(history_path)
    assert history.epoch.tolist() == [1, 2, 3, 4]
    assert history.selected_epoch.iloc[-1] == 4


def test_legacy_selection_restoration():
    checkpoint = dict(epoch=3, selection_metric='mean_valid_cd8_ziln', best_validation_biology=.4)
    history = [dict(epoch=i, validation_cd8_loss=v) for i, v in enumerate([1., .4, .5, .6], 1)]
    stopper = train.EarlyStopping(5, .001)
    assert train.restore_selection_state(stopper, checkpoint, 'cd8_loss', history) == (2., 1)
    assert stopper.best == .4 and stopper.stale_epochs == 2
    assert checkpoint_selection(checkpoint)['selection_metric'] == 'cd8_loss'
    assert train.restore_selection_state(stopper, checkpoint, 'supervised_objective', history) == (None, 5)
    assert math.isinf(stopper.best) and stopper.stale_epochs == 0


def test_resume_uses_saved_progress_and_legacy_threshold():
    # A new min_delta applies only to future epochs, even on a legacy resume.
    history = [dict(epoch=i, validation_cd8_loss=v) for i, v in enumerate([1., .95, .94], 1)]
    legacy = dict(epoch=2, selection_metric='mean_valid_cd8_ziln', best_validation_cd8=1.,
                  config=dict(training=dict(early_stopping_min_delta=.1)))
    stopper = train.EarlyStopping(4, .001)
    assert train.restore_selection_state(stopper, legacy, 'cd8_loss', history) == (1, 1)
    assert stopper.best == 1. and stopper.stale_epochs == 2 and stopper.min_delta == .001
    assert stopper.update(.96) == (True, False)
    checkpoint = dict(epoch=7, selection_metric='supervised_objective', best_selection_score=4.,
                      best_epoch=5, selection_start_epoch=3, early_stopping_state=dict(stale_epochs=3))
    assert train.restore_selection_state(stopper, checkpoint, 'supervised_objective', history) == (5, 3)
    assert stopper.best == 4. and stopper.stale_epochs == 3
    assert stopper.update(5.) == (False, True)


@pytest.mark.parametrize('he', [False, True])
def test_four_column_curve_order_gaps_and_selected_epoch(tmp_path, monkeypatch, he):
    config = load_config(Path('dann/config.yaml'))
    config['model']['heads']['he']['enabled'] = he
    config['data']['target_columns'] = TARGETS[::-1]
    config['loss']['target_weights'] = WEIGHTS[::-1]
    rows = []
    for epoch in range(1, 4):
        row = dict(epoch=epoch)
        for split in ('train', 'validation'):
            row.update({f'{split}_{key}': value for key, value in dict(
                total_loss=epoch, biology_loss=epoch, hurdle_loss=1., positive_loss=epoch-1,
                batch_loss=.5, batch_accuracy=.4, hurdle_acc=.6, mean_r2=-2.,
                he_mse=3., mean_r2_cd8=-3., positive_weighted_loss_cd8=-.3,
            ).items()})
            row[f'{split}_positive_weighted_loss_cd8'] = math.nan if epoch == 1 else -.3
        row.update(train_gradient_cd8_hurdle_norm_mean=[0., math.nan, 2.][epoch-1],
                   train_gradient_cd8_positive_norm_mean=[1., math.nan, 3.][epoch-1],
                   train_gradient_cd8_hurdle__positive_cosine_mean=[-.5, math.nan, .3][epoch-1],
                   train_gradient_ihc_cd8_signed_mean=[-.4, math.nan, 1.4][epoch-1])
        rows.append(row)
    path = tmp_path / 'history.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    captured = []
    monkeypatch.setattr(plt, 'close', lambda figure: captured.append(figure))
    plot_loss_curves(path, tmp_path / 'curves.png', 20, config=config,
                     selection=dict(selection_metric='supervised_objective', best_epoch=2, selection_start_epoch=1))
    ihc, overall = captured[-2:]
    assert (tmp_path / 'ihc_curves.png').is_file()
    from dann.curve_layout import figure_size, render_reservation_bytes
    from dann.objective_curves import COLORS
    assert len(ihc.axes) == 38 and len(overall.axes) == 22
    title = lambda axis: axis.get_title().replace('\n', ' ')
    position = lambda axis: (axis.get_subplotspec().rowspan.start, axis.get_subplotspec().colspan.start)
    assert [title(axis) for axis in overall.axes[:8]] == [
        'IHC balanced hurdle accuracy (macro average)', 'IHC positive-logit R² (macro average)',
        'H&E R² (variance-weighted features)' if he else '', 'Discriminator accuracy',
        'IHC hurdle objective', 'IHC positive objective (location and scale)',
        'H&E objective' if he else '', 'Batch discriminator objective']
    for column in range(4):
        assert position(overall.axes[column]) == (0, column)
        assert position(overall.axes[4+column]) == (1, column)
    assert title(overall.axes[8]) == 'Total objective'
    assert title(overall.axes[9]) == 'Supervised objective — early stopping'
    assert overall.axes[2].get_visible() == he
    assert overall.axes[6].get_visible() == he
    for index, label in enumerate(('CD8', 'Tumor', 'Collagen', 'Stroma')):
        for row, metric in enumerate(('balanced hurdle accuracy', 'hurdle objective',
                                      'positive-logit R²', 'positive objective (location and scale)')):
            axis = ihc.axes[row*4+index]
            assert title(axis) == f'{label} {metric}'
            assert position(axis) == (row, index)
        for row, offset in enumerate((22, 26, 30, 34), 6):
            assert position(ihc.axes[offset+index]) == (row, index)
            assert title(ihc.axes[offset+index]).startswith(label + ':')
    for axis, column in zip(ihc.axes[20:22], (0, 2)):
        assert position(axis) == (5, column)
        assert axis.get_subplotspec().colspan.stop == column + 2
    for row, index, label in ((3, 15, 'participation ratio'), (4, 19, 'total variance')):
        axis = overall.axes[index]
        assert title(axis) == f'Validation latent {label}'
        assert position(axis) == (row, 3)
        assert axis.get_subplotspec().colspan.stop == 4
    for row, index, label in ((3, 14, 'pairwise cosine'), (4, 18, 'conflict frequency')):
        axis = overall.axes[index]
        assert title(axis).startswith(f'Objectives: {label}')
        assert position(axis) == (row, 2)
        assert axis.get_subplotspec().colspan.stop == 3
    assert ihc.axes[8].get_ylim()[0] < -3
    assert ihc.axes[12].get_ylim()[0] < -.3
    assert np.isnan(ihc.axes[12].lines[0].get_ydata()[0])
    assert all(np.isnan(ihc.axes[4].lines[0].get_ydata()))
    assert 'Unavailable' in ihc.axes[4].texts[0].get_text()
    assert ihc.axes[22].get_yscale() == 'log'
    np.testing.assert_allclose(ihc.axes[22].lines[0].get_ydata(), [math.nan, math.nan, 2.], equal_nan=True)
    assert 'Zeros omitted' in ihc.axes[22].texts[0].get_text()
    np.testing.assert_allclose(ihc.axes[30].lines[0].get_ydata(), [-.5, math.nan, .3], equal_nan=True)
    np.testing.assert_allclose(ihc.axes[21].lines[0].get_ydata(), [-.4, math.nan, 1.4], equal_nan=True)
    assert ihc.axes[21].get_ylim()[0] < -.4 and ihc.axes[21].get_ylim()[1] > 1.4
    for axis in (ihc.axes[30], ihc.axes[21]):
        np.testing.assert_equal(axis.lines[-2].get_ydata(), [0, 0])
    assert COLORS['batch'] != COLORS['positive'] != COLORS['hurdle']
    assert [line.get_color() for line in ihc.axes[22].lines[:2]] == [COLORS['hurdle'], COLORS['positive']]
    for figure, kind in ((ihc, 'ihc'), (overall, 'loss')):
        np.testing.assert_equal(figure.get_size_inches(), figure_size(kind))
        assert figure._suptitle.get_fontsize() == 18
        assert len(figure.legends) == 2
        legend_labels = [text.get_text() for legend in figure.legends for text in legend.get_texts()]
        assert {'Train', 'Validation', 'Selected checkpoint: epoch 2', 'Hurdle', 'Positive'} <= set(legend_labels)
        assert all(text.get_fontsize() == 12 for legend in figure.legends for text in legend.get_texts())
        assert all(a.get_subplotspec().get_gridspec().ncols == 4 for a in figure.axes)
        for axis in figure.axes:
            assert axis.get_xlabel() == ''
            assert not any(word in title(axis).lower() for word in ('brier', 'coverage', 'reliability', 'observations'))
            if axis.get_visible():
                np.testing.assert_equal(axis.lines[-1].get_xdata(), [2, 2])
                assert axis.title.get_fontsize() == 14
                assert all(t.get_fontsize() == 12 for t in axis.get_xticklabels() + axis.get_yticklabels())
                assert (axis.get_legend() is not None) == (title(axis).startswith(('Actual GRL', 'Clipping activity')))
        plt.Figure.clear(figure)
    assert render_reservation_bytes(20) == int(np.prod(figure_size('ihc')) * 20**2 * 16)
    # Legacy histories preserve missing values and never infer a minimum marker.
    pd.DataFrame(dict(epoch=[1], train_total_loss=[3.])).to_csv(path, index=False)
    plot_loss_curves(path, tmp_path / 'partial.png', 12)
    figure = captured[-1]
    assert len(figure.axes[8].lines) == 2
    assert figure.axes[8].lines[0].get_marker() == '.'
    assert all(np.isnan(figure.axes[9].lines[0].get_ydata()))
    for figure in captured[-2:]:
        plt.Figure.clear(figure)


@pytest.mark.parametrize('batch_size', [2, 4, 6])
@pytest.mark.parametrize('scenario', ['variable', 'constant_feature', 'constant', 'single', 'large_offset', 'perfect', 'mean'])
def test_he_r2_matches_epoch_reference(batch_size, scenario):
    targets = torch.arange(6, dtype=torch.float64)[:, None] * torch.tensor([[3., 6.]])
    if scenario == 'constant_feature':
        targets[:, 0] = 1000.
    elif scenario == 'constant':
        targets[:] = torch.tensor([1000., -2000.])
    elif scenario == 'single':
        targets = targets[:1]
    elif scenario == 'large_offset':
        targets += torch.tensor([1e12, -1e12], dtype=torch.float64)
    predictions = targets + torch.tensor([1., -2.])
    if scenario == 'perfect':
        predictions = targets.clone()
    elif scenario == 'mean':
        predictions = targets.mean(0).expand_as(targets).clone()
    combined = {key: torch.cat([batch[key] for batch in objective_batches()]) for key in objective_batches()[0]}
    combined['he_targets'] = targets
    batches = Batches({key: value[start:min(start+batch_size, len(targets))] for key, value in combined.items()}
                      for start in range(0, len(targets), batch_size))
    model = TableModel(True)
    model.he = torch.nn.Parameter(predictions)
    metrics = train.run_epoch(model, batches, ZILNLoss(WEIGHTS), torch.device('cpu'), 5., .3, None,
                               None, 0, 1, 'constant', .2, 10., None, he_weight=2., target_columns=TARGETS)
    observed, predicted = targets.numpy(), predictions.numpy()
    sse = np.square(observed-predicted).sum()
    sst = np.square(observed-observed.mean(axis=0)).sum()
    assert metrics['he_mse'] == pytest.approx(sse / observed.size)
    if len(targets) < 2 or sst == 0:
        assert math.isnan(metrics['he_r2'])
    else:
        assert metrics['he_r2'] == pytest.approx(1-sse/sst, abs=1e-12)


def test_he_r2_curve_preserves_negative_values_and_gaps(tmp_path, monkeypatch):
    path = tmp_path / 'history.csv'
    pd.DataFrame(dict(epoch=[1, 2, 3], train_he_r2=[math.nan, -.4, .2],
                      validation_he_r2=[math.nan, -.6, .1])).to_csv(path, index=False)
    captured = []
    monkeypatch.setattr(plt, 'close', lambda figure: captured.append(figure))
    plot_loss_curves(path, tmp_path / 'curves.png', 20)
    figure = captured[-1]
    axis = figure.axes[2]
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [math.nan, -.4, .2], equal_nan=True)
    assert axis.get_ylim()[0] < -.6
    assert axis.get_legend() is None
    assert figure.legends
    plt.Figure.clear(figure)


def test_old_history_weights_use_saved_configuration_without_inventing_signal_losses():
    config = dict(loss=dict(biology_weight=7., batch_weight=.2, he_weight=3.),
                  model=dict(heads=dict(he=dict(enabled=True))))
    frame = pd.DataFrame(dict(train_biology_loss=[2.], train_hurdle_loss=[.5], train_positive_loss=[1.5],
                              train_batch_loss=[4.], train_he_mse=[5.]))
    resolved = weighted_history(frame, config)
    assert resolved.train_biology_weighted_loss.iloc[0] == 14.
    assert resolved.train_batch_weighted_loss.iloc[0] == .8
    assert resolved.train_supervised_objective.iloc[0] == 29.
    assert not any('_cd8' in column for column in resolved)
    frame['train_biology_weighted_loss'] = math.nan
    assert weighted_history(frame, config).train_biology_weighted_loss.isna().all()

"""Independent references and observational contracts for training diagnostics."""
import copy
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from dann.config import monitoring_settings
from dann.losses import ZILNLoss
from dann.model import GradientReversal
from dann.monitoring import (GradientMonitor, ScalarSummaries, StreamingCovariance,
                             freeze_monitoring, gradient_statistics, sampled_updates)
from dann.tests.test_objectives import Batches, TARGETS, WEIGHTS, objective_batches
from dann.tests.test_training_artifacts import assert_nested_equal
from dann.train import accumulation_scales, run_epoch


class SharedModel(torch.nn.Module):
    """Small shared encoder with stochastic recomputation and independent heads."""
    def __init__(self, he=True, checkpointing=False):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Dropout(.2),
                                           torch.nn.Linear(8, 4), torch.nn.Tanh())
        self.biology_predictor = torch.nn.Linear(4, 12)
        self.batch_discriminator = torch.nn.Linear(4, 2)
        self.he_predictor = torch.nn.Linear(4, 2) if he else None
        self.checkpointing = checkpointing
        self.grl = GradientReversal()

    def forward(self, batch, grl_strength=1.):
        latent = (checkpoint(self.encoder, batch['he_targets'], use_reentrant=False, preserve_rng_state=True)
                  if self.training and self.checkpointing else self.encoder(batch['he_targets']))
        pi, mu, raw = self.biology_predictor(latent).chunk(3, dim=1)
        outputs = dict(latent=latent, pi_logits=pi, mu=mu, sigma=F.softplus(raw)+.1,
                       batch_logits=self.batch_discriminator(self.grl(latent, grl_strength)))
        if self.he_predictor is not None:
            outputs['he_prediction'] = self.he_predictor(latent)
        return outputs


def definitions():
    config = dict(monitoring=dict(enabled=True), data=dict(target_columns=TARGETS),
                  loss=dict(logit_epsilon=1e-6), analysis=dict(calibration_coverage_levels=[.5, .9]))
    return freeze_monitoring(config)


@pytest.mark.parametrize('bad', [None, True, [], dict(enabled=1), dict(typo=True),
                                dict(gradient_updates_per_epoch=0), dict(gradient_updates_per_epoch=True),
                                dict(gradient_updates_per_epoch=1.5)])
def test_monitoring_validation(bad):
    with pytest.raises(ValueError, match='monitoring'):
        monitoring_settings(dict(monitoring=bad))


def test_sample_positions_and_legacy_default():
    assert monitoring_settings({}) == dict(enabled=False, gradient_updates_per_epoch=5)
    assert sampled_updates(11) == {0, 2, 5, 7, 10}
    assert sampled_updates(3) == {0, 1, 2}
    assert sampled_updates(11, None) == set(range(11))
    assert sampled_updates(0) == set()
    assert sampled_updates(20, 1) == {0}


@pytest.mark.parametrize('a,b,cosine,retained,signed', [
    ([1., 0.], [2., 0.], 1., 1., [1/3, 2/3]),
    ([1., 0.], [-2., 0.], -1., 1/3, [-1., 2.]),
    ([1., 0.], [0., 1.], 0., math.sqrt(2)/2, [.5, .5]),
    ([1., 0.], [-1., 0.], -1., 0., [math.nan, math.nan]),
    ([0., 0.], [0., 0.], math.nan, math.nan, [math.nan, math.nan]),
    ([0., 0.], [1., 0.], math.nan, 1., [0., 1.]),
    ([1., 0.], [-1.+1e-8, 0.], -1., 1e-8/(2-1e-8), [math.nan, math.nan]),
])
def test_gradient_geometry(a, b, cosine, retained, signed):
    metrics = gradient_statistics(dict(a=torch.tensor(a, dtype=torch.float64), b=torch.tensor(b, dtype=torch.float64)), 'g')
    np.testing.assert_allclose(metrics['g_a__b_cosine'], cosine, atol=1e-14, equal_nan=True)
    np.testing.assert_allclose(metrics['g_retained'], retained, atol=1e-14, equal_nan=True)
    np.testing.assert_allclose([metrics['g_a_signed'], metrics['g_b_signed']], signed, equal_nan=True)
    assert metrics['g_a_norm'] == np.linalg.norm(a)
    summary = ScalarSummaries()
    summary.add(metrics)
    summary.add({k: math.nan for k in metrics})
    result = summary.finish()
    assert result['g_a_norm_count'] == 1
    assert result['g_a_norm_std'] == 0
    assert result['g_a__b_cosine_count'] == int(math.isfinite(cosine))


@pytest.mark.parametrize('reduction', ['mean', 'sum'])
@pytest.mark.parametrize('strength', [0., .4])
@pytest.mark.parametrize('he', [False, True])
def test_weighted_components_grl_masks_and_effective_batch(reduction, strength, he):
    torch.manual_seed(12)
    model = SharedModel(he).double().eval()
    batches = objective_batches()
    combined = {key: torch.cat([b[key] for b in batches]) for key in batches[0]}
    # An entirely missing target and an entirely missing microbatch are genuine zeros.
    combined['target_valid_mask'][:, 2] = False
    loss = ZILNLoss(WEIGHTS, reduction=reduction).double()
    weights = torch.tensor([.3, 1.7], dtype=torch.float64)
    monitor = GradientMonitor(model, TARGETS, 1, 5)
    expected = torch.zeros(sum(p.numel() for p in monitor.parameters), dtype=torch.float64)
    for index, (start, stop) in enumerate(((0, 2), (2, 5), (5, 6))):
        batch = {k: v[start:stop] for k, v in combined.items()}
        batch['accumulation'] = dict(index=index, count=3, rows=6,
                                    valid_counts=combined['target_valid_mask'].sum(0),
                                    class_counts=combined['batches'].bincount(minlength=2))
        out = model(batch, grl_strength=strength)
        result = loss(out['pi_logits'], out['mu'], out['sigma'], batch['targets'], batch['target_valid_mask'], return_components=True)
        assert not result.hurdle_per_target.requires_grad
        b, c, h = accumulation_scales(batch, result, loss, weights)
        parts = {f'{name}_{branch}': 5 * values[i] * b
                 for branch, values in zip(('hurdle', 'positive'), result.differentiable_components)
                 for i, name in enumerate(monitor.names)}
        parts['batch'] = .3 * F.cross_entropy(out['batch_logits'], batch['batches'], weight=weights) * c
        unreversed = .3 * F.cross_entropy(model.batch_discriminator(out['latent']), batch['batches'], weight=weights) * c
        reverse = torch.autograd.grad(parts['batch'], monitor.parameters, retain_graph=True)
        ordinary = torch.autograd.grad(unreversed, monitor.parameters, retain_graph=True)
        for x, y in zip(reverse, ordinary):
            torch.testing.assert_close(x, -strength * y, atol=1e-14, rtol=1e-12)
        total = 5 * result.total * b + parts['batch']
        if he:
            parts['he'] = .7 * F.mse_loss(out['he_prediction'], batch['he_targets']) * h
            total += parts['he']
        reference = torch.autograd.grad(total, monitor.parameters, retain_graph=True)
        expected += torch.cat([v.flatten() for v in reference])
        monitor.accumulate(parts)
        assert all(p.grad is None for p in model.parameters())
        # Additive branch -> target -> IHC identity, including the common denominator.
        branch_total = sum(result.differentiable_components).sum()
        torch.testing.assert_close(branch_total, result.total)
    torch.testing.assert_close(sum(monitor.gradients.values()), expected, atol=1e-13, rtol=1e-12)
    assert monitor.gradients['collagen_hurdle'].count_nonzero() == 0
    assert monitor.gradients['collagen_positive'].count_nonzero() == 0
    monitor.complete(strength)
    values = monitor.finish()
    assert values['gradient_sampled_updates'] == 1
    assert values['gradient_ihc_collagen_norm_mean'] == 0
    if strength == 0:
        assert values['gradient_overall_batch_norm_mean'] == 0


@pytest.mark.parametrize('checkpointing,clip,he', [(False, None, False), (False, .5, True), (True, .5, True)])
def test_monitoring_does_not_change_rng_gradients_or_adamw(checkpointing, clip, he):
    torch.manual_seed(23)
    source = SharedModel(he, checkpointing).double()
    snapshots, metrics = [], []
    for enabled in (False, True):
        model = copy.deepcopy(source)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        torch.manual_seed(17)
        random.seed(17)
        np.random.seed(17)
        values = run_epoch(model, objective_batches(), ZILNLoss(WEIGHTS).double(), torch.device('cpu'),
                           5., .3, None, optimizer, 0, 1, 'constant', .2, 10., clip,
                           target_columns=TARGETS, monitoring=dict(enabled=enabled))
        metrics.append(values)
        snapshots.append(dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                              grads=[p.grad.clone() for p in model.parameters()], torch_rng=torch.get_rng_state(),
                              numpy_rng=np.random.get_state(), python_rng=random.getstate()))
    assert_nested_equal(*snapshots)
    assert metrics[1]['gradient_sampled_updates'] == 3
    assert metrics[1]['gradient_preclip_norm_count'] == 3
    assert metrics[0]['total_loss'] == metrics[1]['total_loss']


@pytest.mark.parametrize('constant,offset', [(False, 0.), (False, 1e9), (True, 1e9)])
def test_streamed_covariance_reference(constant, offset):
    values = np.random.default_rng(3).normal(size=(101, 7)) + offset
    if constant:
        values[:] = offset
    monitor = StreamingCovariance()
    for part in np.array_split(values, 7):
        monitor.update(torch.from_numpy(part))
    result = monitor.finish()
    covariance = np.cov(values, rowvar=False)
    assert monitor.count == len(values)
    assert set(result) == {'latent_total_variance', 'latent_participation_ratio'}
    assert result['latent_total_variance'] == pytest.approx(np.trace(covariance), rel=2e-8, abs=1e-12)
    if constant:
        assert math.isnan(result['latent_participation_ratio'])
    else:
        assert result['latent_participation_ratio'] == pytest.approx(np.trace(covariance)**2 / np.square(covariance).sum(), rel=2e-8)


@pytest.mark.parametrize('count', [0, 1])
def test_covariance_without_enough_observations(count):
    monitor = StreamingCovariance()
    monitor.update(torch.zeros(count, 3))
    assert monitor.count == count
    assert all(math.isnan(value) for value in monitor.finish().values())


def test_reduced_definitions_need_no_targets_or_probability_settings(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Monitoring must not calculate training quantiles')
    monkeypatch.setattr(np, 'quantile', forbidden)
    config = definitions()
    frozen = copy.deepcopy(config['monitoring_definitions'])
    assert frozen['version'] == 2
    assert frozen['scope'] == ['training_gradients', 'validation_latent_total_variance',
                               'validation_latent_participation_ratio']
    config['analysis']['calibration_coverage_levels'] = 'standalone analysis only'
    config['monitoring']['gradient_updates_per_epoch'] = None
    changed = freeze_monitoring(config)
    assert changed['monitoring_definitions'] == frozen
    assert changed['monitoring']['gradient_updates_per_epoch'] is None


@pytest.mark.parametrize('enabled', [False, True])
def test_old_monitoring_definitions_migrate_without_mutation(enabled):
    config = definitions()
    expected = copy.deepcopy(config['monitoring_definitions'])
    legacy = config['monitoring_definitions']
    legacy.pop('scope')
    legacy.update(version=1, cutoffs=[.1, .2, .3], training_positive_count=123,
                  quantiles=[.5, .75, .9], coverage_levels=[.5, .9], reliability_bins=10)
    config['monitoring']['enabled'] = enabled
    saved = copy.deepcopy(config)
    assert freeze_monitoring(config)['monitoring_definitions'] == expected
    assert config == saved
    config['monitoring_definitions']['version'] = 999
    with pytest.raises(ValueError, match='Incompatible saved monitoring'):
        freeze_monitoring(config)
    config['monitoring_definitions']['version'] = 1
    config['data']['target_columns'] = TARGETS[::-1]
    with pytest.raises(ValueError, match='Incompatible saved monitoring'):
        freeze_monitoring(config)


@pytest.mark.parametrize('he', [False, True])
def test_validation_collects_only_latent_statistics_and_preserves_fits(he):
    torch.manual_seed(23)
    model = SharedModel(he=he).double()
    kwargs = dict(model=model, loader=objective_batches(), ziln_loss=ZILNLoss(WEIGHTS),
                  device=torch.device('cpu'), biology_weight=5., batch_weight=.3, class_weights=None,
                  optimizer=None, epoch_index=0, total_epochs=1, grl_schedule='constant',
                  grl_max_lambda=.2, grl_gamma=10., gradient_clip_norm=None, target_columns=TARGETS)
    plain = run_epoch(**kwargs)
    observed = run_epoch(**kwargs, monitoring=dict(enabled=True))
    assert observed.keys() - plain.keys() == {'latent_total_variance', 'latent_participation_ratio'}
    for key, value in plain.items():
        np.testing.assert_allclose(observed[key], value, equal_nan=True)
    with torch.inference_mode():
        latents = torch.cat([model(batch)['latent'] for batch in objective_batches()]).numpy()
    covariance = np.cov(latents, rowvar=False)
    assert observed['latent_total_variance'] == pytest.approx(np.trace(covariance))
    assert observed['latent_participation_ratio'] == pytest.approx(np.trace(covariance)**2 / np.square(covariance).sum())


from dann.tests.test_components import spatial_config
from dann.tests.test_he import he_config


def test_real_gatv2_dropout_checkpointing_accumulation_is_observational(he_config):
    from dann.tests.test_accumulation import settings, execute
    from dann.data_loader import create_data_bundle
    from dann.model import AdversarialLatentFusion
    from dann.train import TrainingProgress
    config = settings(he_config, 5)
    config['model'].update(dropout=.15, spectral_checkpointing=True, gatv2_checkpointing=True)
    for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
        group['gatv2'].update(dropout=.1, attention_dropout=.1)
    bundle = create_data_bundle(config)
    try:
        torch.manual_seed(12)
        source = AdversarialLatentFusion.from_config(config, 2, 4).double()
        snapshots = []
        for enabled in (False, True):
            model = copy.deepcopy(source)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
            torch.manual_seed(41)
            metrics = execute(model, bundle, config, optimizer, TrainingProgress(), max_updates=2,
                              monitoring=dict(enabled=enabled, gradient_updates_per_epoch=None))
            snapshots.append(dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                                  grads=[p.grad.clone() for p in model.parameters()], rng=torch.get_rng_state()))
        assert_nested_equal(*snapshots)
        assert metrics['gradient_sampled_updates'] == metrics['gradient_preclip_norm_count'] == 2
    finally:
        for dataset in bundle.datasets.values():
            dataset.close()


def test_resume_starts_monitoring_and_keeps_frozen_definitions(spatial_config, monkeypatch):
    from dann import train
    import yaml
    config = spatial_config
    config.pop('monitoring', None)
    config['training'].update(epochs=1, early_stopping_enabled=False)
    monkeypatch.setattr(train, 'refresh_loss_curves', lambda *args, **kwargs: None)
    train.train_model(config)
    output = Path(config['training']['output_dir'])
    first = torch.load(output/'latest.pt', weights_only=False)
    # Exercise a checkpoint predating the monitoring feature.
    first['config'].pop('monitoring', None)
    first.pop('monitoring_definitions', None)
    torch.save(first, output/'latest.pt')
    config['training'].update(epochs=2, resume_checkpoint=str(output/'latest.pt'))
    config['monitoring'] = dict(enabled=True, gradient_updates_per_epoch=5)
    train.train_model(config)
    second = torch.load(output/'latest.pt', weights_only=False)
    definitions = second['monitoring_definitions']
    assert definitions == second['config']['monitoring_definitions']
    assert definitions['version'] == 2
    config['training']['epochs'] = 3
    config['monitoring']['gradient_updates_per_epoch'] = None
    config['analysis']['calibration_coverage_levels'] = [.8]
    config['monitoring_definitions'] = {'version': 999}  # Requested definitions never replace saved semantics.
    train.train_model(config)
    third = torch.load(output/'latest.pt', weights_only=False)
    assert third['monitoring_definitions'] == definitions
    assert third['config']['monitoring']['gradient_updates_per_epoch'] is None
    resolved = yaml.safe_load((output/'resolved_config.yaml').read_text())
    assert resolved['monitoring_definitions'] == definitions
    history = pd.read_csv(output/'history.csv')
    assert math.isnan(history.train_gradient_sampled_updates.iloc[0])
    assert history.train_gradient_sampled_updates.iloc[1] == min(5, history.train_optimizer_updates.iloc[1])
    assert history.train_gradient_sampled_updates.iloc[2] == history.train_optimizer_updates.iloc[2]
    assert np.isfinite(history.validation_latent_total_variance.iloc[2])
    assert np.isfinite(history.validation_latent_participation_ratio.iloc[2])
    assert 'validation_latent_count' not in history
    assert not any(key.startswith('validation_cd8_coverage_') for key in history)

    legacy = copy.deepcopy(third)
    legacy_definitions = legacy['config']['monitoring_definitions']
    legacy_definitions.pop('scope')
    legacy_definitions.update(version=1, cutoffs=[.1, .2, .3], training_positive_count=7,
                              coverage_levels=[.5, .9], reliability_bins=10)
    legacy['monitoring_definitions'] = copy.deepcopy(legacy_definitions)
    torch.save(legacy, output/'latest.pt')
    removed = dict(validation_latent_count=7., validation_cd8_presence_brier=.25,
                   validation_cd8_presence_count=7., validation_cd8_positive_count=4.,
                   validation_cd8_coverage_50=.5, validation_cd8_exceedance_q50_cutoff=.1,
                   validation_cd8_presence_bin5_count=2., validation_cd8_presence_bin5_predicted=.5,
                   validation_cd8_presence_bin5_observed=.4)
    for key, value in removed.items():
        history[key] = value
    history.to_csv(output/'history.csv', index=False)
    old_lines = (output/'history.csv').read_text().splitlines()
    config['training']['epochs'] = 4
    train.train_model(config)
    resumed = torch.load(output/'latest.pt', weights_only=False)
    assert resumed['monitoring_definitions'] == definitions
    assert (output/'history.csv').read_text().splitlines()[:len(old_lines)] == old_lines
    appended = pd.read_csv(output/'history.csv').iloc[-1]
    assert appended[list(removed)].isna().all()
    assert np.isfinite(appended.validation_latent_total_variance)


def test_figures_sync_async_and_analysis_share_snapshot(tmp_path, monkeypatch):
    import matplotlib as mpl
    monkeypatch.setitem(mpl.rcParams, "font.size", 23)
    monkeypatch.setitem(mpl.rcParams, "axes.facecolor", ".8")
    from dann.config import load_config
    from dann.objective_curves import plot_loss_curves
    from dann.train import refresh_loss_curves, write_training_history
    from dann.training_artifacts import TrainingArtifacts
    config = load_config(Path('dann/config.yaml'))
    config['analysis']['figure_dpi'] = 12
    rows = [dict(epoch=1, train_total_loss=1., train_gradient_overall_ihc_norm_mean=0.,
                 validation_cd8_presence_brier=.25, validation_cd8_presence_bin5_count=5.,
                 validation_cd8_presence_bin5_predicted=.5, validation_cd8_presence_bin5_observed=.4),
            dict(epoch=2, train_total_loss=.5, train_gradient_overall_ihc_norm_mean=1.,
                 validation_cd8_presence_brier=.2, validation_cd8_presence_bin5_count=5.,
                 validation_cd8_presence_bin5_predicted=.5, validation_cd8_presence_bin5_observed=.6)]
    frozen = copy.deepcopy(rows)
    sync, asynchronous, analysis = [tmp_path/p for p in ('sync', 'async', 'analysis')]
    for folder in (sync, asynchronous, analysis):
        folder.mkdir()
    selection = dict(best_epoch=2, selection_metric='supervised_objective')
    write_training_history(sync/'history.csv', rows, list(rows[0]), {})
    refresh_loss_curves(sync/'history.csv', sync/'loss_curves.png', 12, 2, config, selection)
    manager = TrainingArtifacts(asynchronous, config)
    try:
        manager.submit_epoch(rows, list(rows[0]), {}, selection)
        rows[-1]['train_total_loss'] = 999  # Worker must own an immutable snapshot.
    finally:
        manager.close()
    plot_loss_curves(sync/'history.csv', analysis/'loss_curves.png', 12, config=config, selection=selection)
    for name in ('loss_curves.png', 'ihc_curves.png'):
        assert (sync/name).read_bytes() == (asynchronous/name).read_bytes() == (analysis/name).read_bytes()
    assert pd.read_csv(asynchronous/'history.csv').train_total_loss.iloc[-1] == frozen[-1]['train_total_loss']


def test_sampling_uses_logical_updates_and_keeps_positions_after_resume():
    from dann.train import TrainingProgress

    class LogicalBatches:
        def __init__(self):
            self.sampler = SimpleNamespace(logical_updates=9, start_update=0, stop_update=None,
                                           set_epoch=lambda epoch: None)
            batches = objective_batches()
            combined = {key: torch.cat([b[key] for b in batches]) for key in batches[0]}
            self.microbatches = []
            for i, batch in enumerate(batches):
                batch = dict(batch)
                batch['accumulation'] = dict(index=i, count=3, rows=6,
                                            valid_counts=combined['target_valid_mask'].sum(0),
                                            class_counts=combined['batches'].bincount(minlength=2))
                self.microbatches.append(batch)

        def __len__(self):
            return 27

        def __iter__(self):
            for _ in range(self.sampler.start_update, self.sampler.stop_update or 9):
                yield from self.microbatches

    torch.manual_seed(21)
    source = SharedModel(checkpointing=True).double()
    outcomes = []
    for split in (False, True):
        torch.manual_seed(91)
        model = copy.deepcopy(source)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        state = TrainingProgress()
        kwargs = dict(model=model, loader=LogicalBatches(), ziln_loss=ZILNLoss(WEIGHTS).double(),
                      device=torch.device('cpu'), biology_weight=5., batch_weight=.3, class_weights=None,
                      optimizer=optimizer, epoch_index=0, total_epochs=1, grl_schedule='dann',
                      grl_max_lambda=.2, grl_gamma=10., gradient_clip_norm=1., target_columns=TARGETS,
                      monitoring=dict(enabled=True), progress_state=state)
        if split:
            first = run_epoch(max_updates=4, **kwargs)
            second = run_epoch(**kwargs)
            assert first['gradient_sampled_updates'] == 2
            assert second['gradient_sampled_updates'] == 3
            assert first['gradient_preclip_norm_count'] == 4 and second['gradient_preclip_norm_count'] == 5
            key = 'gradient_overall_ihc_norm_mean'
            combined = (first[key]*2 + second[key]*3)/5
        else:
            complete = run_epoch(**kwargs)
            assert complete['gradient_sampled_updates'] == 5
            assert complete['gradient_preclip_norm_count'] == 9
            combined = complete['gradient_overall_ihc_norm_mean']
        assert state.update_in_epoch == 0 and state.epoch == 1
        outcomes.append((model.state_dict(), combined))
    assert_nested_equal(outcomes[0][0], outcomes[1][0])
    assert outcomes[0][1] == pytest.approx(outcomes[1][1], abs=1e-14)

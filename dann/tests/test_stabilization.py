"""Residual, centering, marginal initialization, and warmup contracts."""
import copy
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from dann.checkpoints import architecture_contract, checkpoint_config, validate_checkpoint
from dann.config import resolve_execution, warmup_settings, head_initialization_settings
from dann.data_loader import create_data_bundle
from dann.execution import build_validation_features, evaluation_groups
from dann.initialization import biology_marginals, final_projection, initialize_supervised_heads
from dann.lr_scheduler import UpdateWarmup, create_lr_scheduler, step_lr_scheduler, validate_warmup_checkpoint
from dann.model import AdversarialLatentFusion
from dann.model_components.deep_sets import DeepSetsEncoder
from dann.model_components.layers import NormalizedResidual, build_mlp, without_dropout
from dann.tests.test_components import spatial_config
from dann.tests.test_he import he_config
from dann.tests.test_execution import enable, epoch
from dann.train import save_checkpoint, load_training_checkpoint, train_model
from dann.training_artifacts import cpu_snapshot, training_state


def test_residual_formula_widths_output_and_legacy_draws():
    args = (4, [4, 6, 6], 6, 'gelu', 0., True)
    torch.manual_seed(31)
    old = build_mlp(*args)
    torch.manual_seed(31)
    new = build_mlp(*args, residual=True)
    for a, b in zip(old.parameters(), new.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert list(old.state_dict()) == list(build_mlp(*args, residual=False).state_dict())
    assert isinstance(new[0], NormalizedResidual)
    assert isinstance(new[1], nn.Linear) and new[1].out_features == 6
    assert isinstance(new[-2], NormalizedResidual)
    assert isinstance(new[-1], nn.Linear)
    x = torch.randn(3, 4)
    torch.testing.assert_close(new[0](x), (x + nn.Sequential(*list(new[0]))(x))/math.sqrt(2))


def encoder(dropout=0.):
    return DeepSetsEncoder(5, 4, [4, 4], 6, dropout=dropout, peak_mlp_residual=True,
                           pool_centering='zero_input', inference_peak_chunk_size=2)


def spectra():
    return (torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0., 0., 1., 2., 3.]),
            torch.tensor([1, 1, 2, 3, 3]), torch.tensor([0, 2, 1, 2]))


def test_centering_empty_zero_and_mixed_counts_backward():
    model = encoder().double()
    ids, values, rows, counts = spectra()
    values = values.double()
    output = model(ids, values, rows, counts)
    torch.testing.assert_close(output[:2], torch.zeros_like(output[:2]), atol=1e-15, rtol=0)
    reference = model.peak_mlp(torch.zeros(1, 4, dtype=torch.double))
    raw = model.peak_mlp(model.embedding(ids) * values[:, None])
    expected = torch.stack([torch.zeros(6, dtype=torch.double), raw[:2].mean(0)-reference[0],
                            raw[2]-reference[0], raw[3:].mean(0)-reference[0]])
    torch.testing.assert_close(output, expected)
    output.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.embedding.weight.grad.norm() > 0
    empty = model(ids[:0], values[:0], rows[:0], torch.tensor([0, 0]))
    assert torch.equal(empty, torch.zeros_like(empty))
    with torch.no_grad():
        torch.testing.assert_close(model(ids, values, rows, counts), output)


def test_reference_skips_dropout_rng_and_preserves_modes_and_gradients():
    model = encoder(.5).train()
    before = torch.get_rng_state().clone()
    value = without_dropout(model.peak_mlp, torch.zeros(1, 4))
    assert torch.equal(before, torch.get_rng_state())
    assert all(m.training for m in model.modules())
    value.sum().backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.peak_mlp.parameters())
    model.eval()
    torch.testing.assert_close(value, model.peak_mlp(torch.zeros(1, 4)))


@pytest.mark.parametrize('dropout', [0., .3])
def test_chunks_checkpoint_outputs_gradients_and_rng(dropout):
    a = encoder(dropout).train()
    b = copy.deepcopy(a)
    torch.manual_seed(6)
    x = a.microbatched(*spectra(), peak_budget=2, checkpointing=False)
    x.square().sum().backward()
    rng = torch.get_rng_state()
    torch.manual_seed(6)
    y = b.microbatched(*spectra(), peak_budget=2, checkpointing=True)
    y.square().sum().backward()
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(x, y)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    if not dropout:
        torch.testing.assert_close(x, a(*spectra()))


@pytest.mark.parametrize('kind', ['mlp', 'cnn', 'gatv2'])
def test_training_marginals_masks_streaming_and_head_order(he_config, kind):
    c = he_config
    c['training']['head_initialization']['enabled'] = True
    for group in [c['model']['aggregation'], *c['model']['heads'].values()]:
        group['type'] = kind
    resolve_execution(c)
    bundle = create_data_bundle(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    original = copy.deepcopy(model.state_dict())
    initial_weights = [final_projection(h).weight.detach().clone() for h in (model.biology_predictor, model.he_predictor)]
    # Held-out targets and H&E must not affect fitted marginal statistics.
    train = bundle.split_indices['train']
    held = np.concatenate([bundle.split_indices[s] for s in ('validation', 'test')])
    bundle.metadata.targets[held] = np.nan
    with h5py.File(c['data']['path'], 'r+') as f:
        he = f['obsm/hes_features'][:]
        he[held] = np.nan
        f['obsm/hes_features'][:] = he
    report = initialize_supervised_heads(model, bundle, c, chunk_rows=3)
    expected, records = biology_marginals(bundle.metadata.targets[train], bundle.metadata.target_valid_mask[train],
        c['data']['target_columns'], c['loss']['logit_epsilon'], .1, c['model']['sigma_min'])
    np.testing.assert_allclose(final_projection(model.biology_predictor).bias.detach(), expected)
    assert report['biology'] == records
    np.testing.assert_allclose(final_projection(model.he_predictor).bias.detach(), he[train].mean(0), atol=1e-6)
    assert records[0]['valid_count'] == int(bundle.metadata.target_valid_mask[train, 0].sum())
    assert records[0]['valid_count'] < len(train)  # CD8 area/ratio masks survive initialization.
    for weight, head in zip(initial_weights, (model.biology_predictor, model.he_predictor)):
        torch.testing.assert_close(final_projection(head).weight, weight * .01)
    for name, value in model.state_dict().items():
        if name.startswith('batch_discriminator.') or name.startswith('encoder.'):
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    batch = next(iter(bundle.loaders['train']))
    model(batch)['mu'].sum().backward()
    assert model.encoder.embedding.weight.grad.norm() > 0
    for ds in bundle.datasets.values():
        ds.close()


def test_degenerate_marginals_and_target_order():
    y = np.array([[0., 0., .5], [0., 1., .5]])
    mask = np.array([[False, True, True], [False, True, True]])
    bias, records = biology_marginals(y, mask, ['missing', 'mixed', 'constant'], 1e-5, .1, 1e-4)
    assert [r['target'] for r in records] == ['missing', 'mixed', 'constant']
    assert records[0]['zero_probability'] == .5 and records[0]['mu'] == 0 and records[0]['sigma'] == 1
    assert len(records[0]['fallbacks']) == 2
    assert records[2]['sigma'] == .1
    assert np.isfinite(bias).all()
    bias, records = biology_marginals(y[:,:1], np.ones((2,1), bool), ['zero'], 1e-5, .1, 1e-4)
    assert records[0]['zero_probability'] == 1-1e-5 and records[0]['positive_count'] == 0


@pytest.mark.parametrize('version', range(2, 8))
def test_version_roundtrip_and_new_settings_rejected(spatial_config, version):
    c = spatial_config
    for group in [c['model']['aggregation'], *c['model']['heads'].values()]:
        group['type'] = 'mlp'
    resolve_execution(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    contract = architecture_contract(c)
    contract['version'] = version
    if version < 7:
        spectral = contract['components']['spectral_encoder']['deep_sets']
        spectral.pop('pool_centering'); spectral.pop('peak_mlp_residual')
        for name, group in contract['components'].items():
            if name != 'spectral_encoder':
                group['mlp'].pop('residual')
    payload = dict(config=c, architecture=contract)
    restored = checkpoint_config(payload)
    other = AdversarialLatentFusion.from_config(restored, 2, 4)
    other.load_state_dict(model.state_dict())
    assert list(model.state_dict()) == list(other.state_dict())
    validate_checkpoint(payload, restored, other)
    changed = copy.deepcopy(c)
    changed['model']['spectral_encoder']['deep_sets']['pool_centering'] = 'zero_input'
    with pytest.raises(ValueError, match='incompatible'):
        validate_checkpoint(payload, changed)
    changed = copy.deepcopy(c)
    changed['model']['heads']['biology']['mlp']['residual'] = True
    with pytest.raises(ValueError, match='incompatible'):
        validate_checkpoint(payload, changed)


@pytest.mark.parametrize('epochs, updates', [(0, 4), (1, 1), (1, 4), (2, 4)])
def test_warmup_endpoints_plateau_and_resume(epochs, updates):
    c = dict(training=dict(learning_rate=.001, warmup=dict(epochs=epochs, start_factor=.1),
                          lr_scheduler=dict(enabled=True, patience=0, cooldown=0)))
    def pair():
        opt = torch.optim.AdamW([nn.Parameter(torch.ones(1))], lr=.001)
        return opt, create_lr_scheduler(opt, c), UpdateWarmup(opt, c, updates)
    opt, scheduler, warmup = pair()
    rates = []
    for i in range(12):
        rates.append(opt.param_groups[0]['lr'])
        warmup.step()
        if i >= warmup.horizon:
            step_lr_scheduler(scheduler, dict(supervised_objective=1.))
        if i == 1:
            state = cpu_snapshot(training_state(nn.Linear(1, 1), opt, scheduler, warmup))
    assert rates[0] == (.0001 if epochs else .001)
    if warmup.horizon > 1:
        assert rates[warmup.horizon-1] == .001
    opt2, sched2, warm2 = pair()
    opt2.load_state_dict(state['optimizer_state']); sched2.load_state_dict(state['lr_scheduler_state'])
    warm2.load_state_dict(state['warmup_state'])
    for i in range(2, 12):
        assert opt2.param_groups[0]['lr'] == rates[i]
        warm2.step()
        if i >= warm2.horizon:
            step_lr_scheduler(sched2, dict(supervised_objective=1.))
    assert opt2.param_groups[0]['lr'] >= .001*.01


@pytest.mark.parametrize('block, value', [('warmup', {'epochs':-1}), ('warmup', {'start_factor':0}),
    ('warmup', {'epochs':True}), ('warmup', {'start_factor':float('nan')}),
    ('head_initialization', {'enabled':1}), ('head_initialization', {'output_weight_scale':0}),
    ('head_initialization', {'min_positive_sigma':float('inf')})])
def test_invalid_settings(block, value):
    with pytest.raises(ValueError):
        (warmup_settings if block == 'warmup' else head_initialization_settings)({block:value})


@pytest.mark.parametrize('kind', ['cnn', 'gatv2'])
def test_centering_cached_direct_predictions(he_config, kind):
    c = he_config
    c['model']['spectral_encoder']['deep_sets'].update(pool_centering='zero_input', peak_mlp_residual=True)
    c['model']['aggregation']['type'] = kind
    resolve_execution(c)
    direct = create_data_bundle(c)
    cached = create_data_bundle(enable(c, validation_tiles=3, validation_feature_cache=True,
                                       validation_peak_budget=17, validation_gat_tiles=3))
    model = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    expected, actual = {}, {}
    with torch.no_grad():
        for batch in direct.loaders['validation']:
            out = model(batch)
            expected.update({int(row): {k:v[i].clone() for k,v in out.items()}
                             for i,row in enumerate(batch['row_ids'])})
        for cpu, batch, out in evaluation_groups(model, cached.loaders['validation'], torch.device('cpu')):
            actual.update({int(row): {k:v[i].clone() for k,v in out.items()}
                           for i,row in enumerate(batch['row_ids'])})
    assert expected.keys() == actual.keys()
    for row in expected:
        for key in expected[row]:
            torch.testing.assert_close(actual[row][key], expected[row][key], atol=2e-6, rtol=2e-5)
    for bundle in (direct, cached):
        for dataset in bundle.datasets.values():
            dataset.close()


def test_warmup_accumulation_boundaries(he_config):
    from dann.tests.test_accumulation import settings, execute
    from dann.train import TrainingProgress
    configs = [settings(he_config, micro) for micro in (None, 5)]
    outputs, rates = [], []
    for c in configs:
        c['training']['warmup'] = dict(epochs=1, start_factor=.1)
        bundle = create_data_bundle(c)
        torch.manual_seed(41)
        model = AdversarialLatentFusion.from_config(c, 2, 4).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        warmup = UpdateWarmup(optimizer, c, bundle.loaders['train'].sampler.logical_updates)
        current = []
        def capture(module, args):
            current.append((args[0].get('accumulation', {}).get('index', 0), optimizer.param_groups[0]['lr']))
        model.register_forward_pre_hook(capture)
        metrics = execute(model, bundle, c, optimizer, TrainingProgress(), warmup=warmup)
        rates.append([lr for i,lr in current if i == 0])
        active = None
        for i, lr in current:
            if i == 0:
                active = lr
            assert active == lr
        assert metrics['learning_rate_min'] == .0001 and metrics['learning_rate_max'] == .001
        assert metrics['warmup_updates'] == warmup.horizon == warmup.completed_updates
        outputs.append(copy.deepcopy(model.state_dict()))
        for dataset in bundle.datasets.values():
            dataset.close()
    assert rates[0] == rates[1]
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=2e-8, rtol=2e-6)


@pytest.mark.parametrize('asynchronous', [False, True])
def test_production_warmup_snapshot_resume_and_no_reinitialization(he_config, tmp_path, monkeypatch, asynchronous):
    import pandas as pd
    from dann import train
    c = he_config
    c['performance']['async_artifacts'] = asynchronous
    c['training'].update(epochs=1, early_stopping_enabled=False, warmup=dict(epochs=2, start_factor=.1))
    c['training']['head_initialization']['enabled'] = True
    for g in [c['model']['aggregation'], *c['model']['heads'].values()]:
        g['type'] = 'mlp'
    resolve_execution(c)
    c['analysis']['figure_dpi'] = 30
    path = train_model(c)
    saved = torch.load(path, weights_only=False)
    validate_warmup_checkpoint(saved)
    assert saved['head_initialization']['fit_split'] == 'train'
    assert saved['lr_scheduler_state']['last_epoch'] == 0
    horizon = saved['warmup_state']['horizon']
    assert saved['warmup_state']['completed_updates'] * 2 == horizon
    for mutation in ('missing', 'horizon', 'progress', 'base'):
        bad = copy.deepcopy(saved)
        if mutation == 'missing':
            bad.pop('warmup_state')
        elif mutation == 'horizon':
            bad['warmup_state']['horizon'] += 1
        elif mutation == 'progress':
            bad['warmup_state']['completed_updates'] += 1
        else:
            bad['warmup_state']['base_lrs'][0] *= 2
        with pytest.raises(ValueError, match='warmup'):
            validate_warmup_checkpoint(bad)
    def forbidden(*args, **kwargs):
        raise AssertionError('Resume must never refit marginals')
    monkeypatch.setattr(train, 'initialize_supervised_heads', forbidden)
    c['training'].update(epochs=3, resume_checkpoint=str(path))
    train_model(c)
    latest = torch.load(Path(c['training']['output_dir'])/'latest.pt', weights_only=False)
    validate_warmup_checkpoint(latest)
    assert latest['warmup_state']['completed_updates'] * 2 == horizon * 3
    assert latest['lr_scheduler_state']['last_epoch'] == 1
    assert latest['head_initialization'] == saved['head_initialization']
    history = pd.read_csv(Path(c['training']['output_dir'])/'history.csv')
    assert history.train_learning_rate_min.iloc[0] == pytest.approx(.0001)
    assert history.train_learning_rate_max.iloc[1] == pytest.approx(.001)
    assert history.learning_rate.iloc[2] == pytest.approx(.001)


def test_three_epoch_runner_contract_and_artifacts(he_config, tmp_path):
    import json
    import yaml
    from dann.verification.stabilization import arm_config, run_arm, summarize
    source = he_config
    source['analysis']['peaks'] = dict(enabled=False)
    root = tmp_path/'experiment'
    (root/'configs').mkdir(parents=True)
    configurations = [arm_config(source, 'mlp', 8, ablation, .001, root)
                      for ablation in ('baseline', 'combined')]
    for c in configurations:
        name = c['results']['run_name']
        (root/'configs'/f'{name}.yaml').write_text(yaml.safe_dump(c))
        run_arm(root, name)
        directory = root/'arms'/name
        manifest = json.loads((directory/'manifest.json').read_text())
        history = json.loads((directory/'history.json').read_text())
        assert [r['epoch'] for r in history] == [0, 1, 2, 3]
        assert manifest['grl_horizon_updates'] == 300 * manifest['logical_updates_per_epoch']
        payload = torch.load(directory/'epoch_3.pt', weights_only=False)
        validate_warmup_checkpoint(payload)
        assert payload['epoch'] == 2
        assert len(json.loads((directory/'early_updates.json').read_text())) == 10
    summarize(root)
    assert (root/'comparison_epoch_3.csv').exists() and (root/'comparison.png').exists()
    manifests = [json.loads((root/'arms'/c['results']['run_name']/'manifest.json').read_text()) for c in configurations]
    assert manifests[0]['initial_parameters_sha256'] == manifests[1]['initial_parameters_sha256']
    assert manifests[0]['thresholds'] == manifests[1]['thresholds']


def test_resume_rejects_changed_centering_and_mlp_residuals(spatial_config):
    from dann.train import resume_training_config
    c = spatial_config
    requested = copy.deepcopy(c)
    requested['model']['spectral_encoder']['deep_sets']['pool_centering'] = 'zero_input'
    with pytest.raises(ValueError, match='centering'):
        resume_training_config(c, requested)
    requested = copy.deepcopy(c)
    requested['model']['heads']['discriminator']['mlp']['residual'] = True
    with pytest.raises(ValueError, match='residual'):
        resume_training_config(c, requested)


@pytest.mark.parametrize('checkpointing', [False, True])
def test_pixel_spectral_microbatching_equivalence(he_config, checkpointing):
    from dann.execution import tree_map
    c = he_config
    for group in [c['model']['aggregation'], *c['model']['heads'].values()]:
        group['type'] = 'mlp'
        group['mlp']['residual'] = True
    c['model']['spectral_encoder']['deep_sets'].update(pool_centering='zero_input', peak_mlp_residual=True)
    resolve_execution(c)
    bundle = create_data_bundle(c)
    batch = tree_map(next(iter(bundle.loaders['train'])), lambda t: t.double() if t.is_floating_point() else t)
    direct = AdversarialLatentFusion.from_config(c, 2, 4).double()
    updated = copy.deepcopy(c)
    updated['model'].update(pixel_spectral_microbatching=True, spectral_checkpointing=checkpointing, spectral_peak_budget=9)
    assert architecture_contract(c) == architecture_contract(updated)
    chunked = AdversarialLatentFusion.from_config(updated, 2, 4).double()
    chunked.load_state_dict(direct.state_dict())
    a, b = direct(batch), chunked(batch)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=1e-12, rtol=1e-10)
    for out in (a, b):
        sum(v.square().sum() for v in out.values()).backward()
    for p, q in zip(direct.parameters(), chunked.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-8)
    for ds in bundle.datasets.values():
        ds.close()

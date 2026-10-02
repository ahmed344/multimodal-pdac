"""Synthetic contracts for the isolated, matched three-epoch biology diagnosis."""
import copy

import numpy as np
import pytest
import torch
from torch import nn

from dann.config import resolve_execution
from dann.data_loader import create_data_bundle
from dann.losses import ZILNLoss
from dann.tests.test_components import spatial_config
from dann.tiles import tile_collate
from dann.train import grl_strength
from dann.verification.biology_variants import (VARIANTS, RADII, make_variant, audit,
                                               optimizer_for, state_hash)
from dann.verification.cnn_biology_diagnosis import (classify, followups, route, select_candidate,
    epoch_arguments, CheckedLoader, Deadline, SEEDS, run_fit)
from dann.verification.production_fits import close_bundle, fit_baseline, sampler_evidence
from dann.verification.extended_batch_diagnosis import admit


@pytest.fixture
def diagnostic_config(spatial_config):
    c = spatial_config
    c['model']['aggregation']['type'] = 'mlp'
    c['model']['heads']['biology']['cnn'].update(depth=3, dropout=.1)
    c['training']['spatial'].update(supervised_rows_per_batch=7, sampling_strategy='proportional_slide_tiles')
    return resolve_execution(c)


def test_pointwise_linear_forward_and_gradient_equivalence():
    linear = nn.Linear(7, 5).double()
    conv = nn.Conv2d(7, 5, 1).double()
    conv.weight.data.copy_(linear.weight[:, :, None, None]); conv.bias.data.copy_(linear.bias)
    x = torch.randn(2, 7, 6, 6, dtype=torch.float64, requires_grad=True)
    a = conv(x); b = linear(x.movedim(1,-1)).movedim(-1,1)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(torch.autograd.grad(a.square().sum(), x)[0],
                               torch.autograd.grad(b.square().sum(), x)[0])


@pytest.mark.parametrize('variant', VARIANTS)
def test_initialization_shared_and_fixed_spatial_execution(diagnostic_config, variant):
    c = diagnostic_config
    rng = torch.get_rng_state().clone()
    model, meta = make_variant(c, 2, 19, variant)
    baseline, reference = make_variant(c, 2, 19, 'C')
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert meta['shared_sha256'] == reference['shared_sha256']
    assert model.spatial and model.halo == 3 and meta['receptive_radius'] == RADII[variant]
    assert not any(p.requires_grad for p in model.batch_discriminator.parameters())
    assert all(p.requires_grad for p in model.encoder.parameters())
    from dann.model import AdversarialLatentFusion
    with torch.random.fork_rng():
        torch.manual_seed(19)
        production = AdversarialLatentFusion.from_config(c, 2, 4)
    assert state_hash(baseline.state_dict()) == state_hash(production.state_dict())
    opt = optimizer_for(model, c, variant)
    assert [g['lr'] for g in opt.param_groups] == [.001, .0001 if variant=='low_lr' else .001]


def test_pointwise_initialized_3x3_and_masked_halo_gradients(diagnostic_config):
    p, _ = make_variant(diagnostic_config, 2, 19, 'P')
    q, _ = make_variant(diagnostic_config, 2, 19, 'pointwise_init')
    p.eval(); q.eval()
    x = torch.randn(1, 6, 9, 9, requires_grad=True)
    mask = torch.ones(1, 1, 9, 9); mask[:,:,2,3] = 0
    a = p.biology_map(x, mask); b = q.biology_map(x, mask)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    ga = torch.autograd.grad(a[:,:,4,4].sum(), x)[0]
    gb = torch.autograd.grad(b[:,:,4,4].sum(), x)[0]
    torch.testing.assert_close(ga, gb)
    assert (b[:,:,2,3] == 0).all()
    c, _ = make_variant(diagnostic_config, 2, 19, 'C'); c.eval()
    out = c.biology_map(x, mask)
    grad = torch.autograd.grad(out[:,:,4,4].sum(), x)[0]
    assert grad[:,:,2,3].abs().sum() == 0
    assert grad[:,:,:,0].abs().sum() == 0 and grad[:,:,:,8].abs().sum() == 0
    assert grad[:,:,1,4].abs().sum() > 0  # three-pixel halo contributes gradients
    assert grad[:,:,4,3].abs().sum() > 0  # immediate occupied neighbor
    changed = x.detach().clone(); changed[:,:,2,3] = 10000
    torch.testing.assert_close(c.biology_map(changed,mask), out)


def test_residual_formula(diagnostic_config):
    c, _ = make_variant(diagnostic_config, 2, 19, 'C')
    r, _ = make_variant(diagnostic_config, 2, 19, 'residual')
    c.eval(); r.eval()
    h = c.biology_predictor
    x, mask = torch.randn(1,6,7,7), torch.ones(1,1,7,7)
    mask[:,:,2,3] = 0
    value = h.hidden(h.input(x*mask),h.norms[0],mask)
    for conv, norm in zip(h.convolutions,h.norms[1:]):
        value = (value+h.hidden(conv(value),norm,mask))/2**.5*mask
    torch.testing.assert_close(r.biology_map(x,mask), h.output(value)*mask)


def test_audit_preserves_rng_modes_gradients_optimizer(diagnostic_config):
    bundle = create_data_bundle(diagnostic_config)
    try:
        batch = tile_collate([bundle.datasets['train'][0],bundle.datasets['train'][1]])
        model, _ = make_variant(diagnostic_config, 2, 19, 'C')
        model.train(); model.batch_discriminator.eval()
        opt = optimizer_for(model, diagnostic_config, 'C')
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        state, rng = copy.deepcopy(opt.state_dict()), torch.get_rng_state().clone()
        old = state_hash(model.state_dict())
        result = audit(model, batch, ZILNLoss.from_config(diagnostic_config), 5.)
        assert model.training and not model.batch_discriminator.training
        assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
        torch.testing.assert_close(torch.get_rng_state(), rng)
        assert opt.state_dict() == state and state_hash(model.state_dict()) == old
        assert result['sensitivity']['mu']['missing_tissue_max'] == 0
        assert result['sensitivity']['mu']['outside_halo_max'] == 0
    finally:
        close_bundle(bundle)


def test_three_complete_epochs_and_schedule(diagnostic_config, tmp_path):
    bundle = create_data_bundle(diagnostic_config)
    deadline = Deadline(tmp_path/'deadline.json',240); deadline.start()
    try:
        model, _ = make_variant(diagnostic_config,2,19,'M',True)
        common = epoch_arguments(diagnostic_config,bundle,model,ZILNLoss.from_config(diagnostic_config),True)
        assert common['total_epochs'] == 300
        steps = 664
        strength = grl_strength((3*steps-1)/(300*steps-1), 'dann', .25, 10.)
        assert .012 < strength < .013
        for epoch in range(3):
            bundle.loaders['train'].sampler.set_epoch(epoch)
            plan = sampler_evidence(bundle)
            checked = CheckedLoader(bundle.loaders['train'],deadline)
            sizes = [len(batch['row_ids']) for batch in checked]
            checked.verify(bundle.split_indices['train'],plan)
            assert all(size == 7 for size in sizes[:-1])
            assert sizes[-1] == (len(bundle.split_indices['train'])-1)%7+1
        assert common['grl_max_lambda'] == diagnostic_config['training']['grl_max_lambda']
        assert epoch_arguments(diagnostic_config,bundle,model,ZILNLoss.from_config(diagnostic_config),False)['grl_max_lambda']==0
    finally:
        close_bundle(bundle)


def test_epoch3_only_classification():
    baseline = dict(cd8_loss=1.,biology_loss=2.)
    good = dict(cd8_loss=.9,biology_loss=1.9,mean_r2_cd8=.1)
    bad = dict(good,mean_r2_cd8=-.1)
    history = [dict(epoch=e,validation=good) for e in (1,2,3)]
    assert not classify(history,bad,baseline)['passed']
    assert not classify(history[:2],good,baseline)['passed']
    assert classify(history,good,baseline)['passed']


@pytest.mark.parametrize('m,c,p',[(False,False,False),(True,True,True),(True,False,True),(True,False,False)])
def test_routing(m,c,p):
    calls, results = [], {}
    def group(name,specs,reserve_fits=0):
        calls.append((name,specs,reserve_fits))
        for seed,v,a in specs:
            passed = {'M':m,'C':c,'P':p}.get(v, v==followups(p)[0])
            results[seed,v,a] = dict(assessment=dict(passed=passed),validation=dict(cd8_loss=.9,biology_loss=1.))
        return True
    route(group,lambda s,v,a=False: results[s,v,a])
    if not m:
        assert [c[0] for c in calls] == ['screen','positive_control_confirmation']
    elif c:
        assert [c[0] for c in calls] == ['screen','current_cnn_restoration_screen','current_cnn_confirmation']
    else:
        assert tuple(v for _,v,_ in calls[1][1]) == followups(p)
        assert calls[-1][0] == 'restoration'
        assert calls[1][2] == 7


def test_deadline_failure_and_mixed_seed_routing(tmp_path):
    deadline = Deadline(tmp_path/'deadline.json',240); deadline.start()
    assert not admit(deadline,[100000],1200)
    saved = deadline.record.copy(); again = Deadline(deadline.path,240); again.start()
    assert again.record == saved
    again.record['deadline'] = 0
    with pytest.raises(TimeoutError):
        again.check()
    assert 'incomplete' in route(lambda *args,**kwargs:False,lambda *args:None).lower()
    calls, results = [], {}
    def group(name,specs,reserve_fits=0):
        calls.append(name)
        for s,v,a in specs:
            results[s,v,a] = dict(assessment=dict(passed=v!='C' and not (s==SEEDS[2] and v=='shallow')),
                                 validation=dict(cd8_loss=.9,biology_loss=1.))
        return True
    conclusion = route(group,lambda s,v,a=False:results[s,v,a])
    assert 'mixed seed' in conclusion and 'restoration' not in calls
    def broken(*args,**kwargs):
        raise RuntimeError('OOM')
    with pytest.raises(RuntimeError,match='OOM'):
        route(broken,lambda *args:None)


def test_synthetic_fit_three_checkpoints_independent_rescore(diagnostic_config,tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    bundle = create_data_bundle(diagnostic_config)
    deadline = Deadline(tmp_path/'deadline.json',240); deadline.start()
    try:
        seed = SEEDS[0]
        plans = {seed:[]}
        bundle.loaders['train'].sampler.seed=seed
        for epoch in range(3):
            bundle.loaders['train'].sampler.set_epoch(epoch)
            plans[seed].append(sampler_evidence(bundle))
        batch = tile_collate([bundle.datasets['train'][0],bundle.datasets['train'][1]])
        result = run_fit(tmp_path,diagnostic_config,bundle,plans,batch,
                         fit_baseline(bundle,diagnostic_config),deadline,seed,'P')
        assert result['status']=='completed', result
        directory = tmp_path/f'{seed}_P'/'disabled'
        assert all((directory/f'epoch_{e}.pt').exists() for e in (1,2,3))
        assert result['updates']==3*len(bundle.loaders['train'])
        assert result['validation_coverage']['rows']==len(bundle.split_indices['validation'])
        before = torch.load(directory/'initial.pt',weights_only=True)
        after = torch.load(directory/'epoch_3.pt',weights_only=False)['model_state']
        for key in before:
            if key.startswith('batch_discriminator'):
                torch.testing.assert_close(before[key],after[key],atol=0,rtol=0)
    finally:
        close_bundle(bundle)


def test_constructor_does_not_reseed_cuda(diagnostic_config, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', calls.append)
    _, metadata = make_variant(diagnostic_config, 2, 19, 'P')
    assert calls == []
    assert metadata['cpu_dropout_seed'] == 19
    assert metadata['cuda_dropout_seed'] == metadata['shape_initialization_seed'] == 1019


def test_report_distinguishes_incomplete_from_scientific_failure(tmp_path):
    from types import SimpleNamespace
    from pathlib import Path
    from dann.verification.cnn_biology_diagnosis import report
    from dann.verification.epoch_comparison import file_hash
    deadline = Deadline(tmp_path/'deadline.json', 240)
    deadline.start()
    results = {
        'completed': dict(seed=19, variant='C', adversarial=False, status='completed',
                          assessment=dict(passed=False), seconds=1., directory=str(tmp_path/'C'),
                          validation=dict(cd8_loss=1.3, biology_loss=1.3, mean_r2_cd8=-.1)),
        'oom': dict(seed=20, variant='C', adversarial=False, status='oom',
                    assessment=dict(passed=False), seconds=1., directory=str(tmp_path/'oom')),
    }
    study = SimpleNamespace(root=tmp_path, results=results, deadline=deadline,
        baseline=dict(validation=dict(cd8_loss=1.2, biology_loss=1.2, mean_r2_cd8=0.)),
        source_hash=file_hash(Path('dann/config.yaml')), archive_hashes={})
    report(study, 'Incomplete comparison.')
    text = (tmp_path/'report.md').read_text()
    assert '19: fail' in text and '20: incomplete (oom)' in text
    assert '| Seed | Biology head |' in text and '| oom |' in text

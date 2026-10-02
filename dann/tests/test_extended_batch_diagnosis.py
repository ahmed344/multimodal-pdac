"""Contracts for the 1,000-update follow-up sampler and group admission."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from dann.tests.test_components import spatial_config
from dann.tests.test_discriminator_controls import setup_control
from dann.verification import batch_execution_diagnosis as b
from dann.verification.extended_batch_diagnosis import proportional_plan, dispersed_plan, admit, progression


def test_proportional_queues_preserve_coverage_and_spread_small_slides():
    tiles=[];row=0
    for slide,n in [('a',40),('b',4),('c',12)]:
        for t in range(n):
            tiles.append(((slide,t,0),[(row,row),(row+1,row+1)]));row+=2
    source=SimpleNamespace(tiles=tiles)
    plan=proportional_plan(source,np.arange(row),719,4,batch_size=9)
    b.verify_plan(plan,np.arange(row))
    assert len(plan.passes)==5 and plan.boundaries[plan.passes[1]]==row
    assert np.diff(plan.boundaries)[plan.passes[1]-1]==row%9
    first=plan.rows[:row]; positions=np.flatnonzero((first>=80)&(first<88))/row
    assert positions.min()<.2 and positions.max()>.8
    assert plan.sha256==proportional_plan(source,np.arange(row),719,4,9).sha256
    assert plan.sha256!=proportional_plan(source,np.arange(row),720,4,9).sha256


def test_proportional_partial_supervision_and_dispersion(spatial_config):
    bundle,_,_,_=setup_control(spatial_config)
    tiles=b.SpatialTileDataset(b.geometry_config(spatial_config,core_size=4),bundle.split_indices['train'],bundle.metadata)
    plan=proportional_plan(tiles,bundle.split_indices['train'],719,3,3)
    a=b.PlannedDataset(tiles,plan,False);z=b.PlannedDataset(tiles,plan,True)
    for i in range(plan.passes[1]):
        flat=a[i];other=b.core_flat(z[i])
        for k in flat:torch.testing.assert_close(flat[k],other[k],equal_nan=True)
    dispersed=dispersed_plan(plan,bundle.split_indices['train'],bundle.metadata.batch_codes,719)
    np.testing.assert_array_equal(dispersed.boundaries,plan.boundaries)
    np.testing.assert_array_equal(bundle.metadata.batch_codes[dispersed.rows],bundle.metadata.batch_codes[plan.rows])
    b.verify_plan(dispersed,bundle.split_indices['train']);tiles.close()


def test_plans_extend_beyond_two_passes_without_prefix_changes(spatial_config):
    bundle,_,_,_=setup_control(spatial_config)
    tiles=b.SpatialTileDataset(b.geometry_config(spatial_config,core_size=4),bundle.split_indices['train'],bundle.metadata)
    a=b.make_plans(tiles,bundle.split_indices['train'],bundle.metadata.batch_codes,719,passes=2,small=3,tiles_per_batch=2)
    z=b.make_plans(tiles,bundle.split_indices['train'],bundle.metadata.batch_codes,719,passes=4,small=3,tiles_per_batch=2)
    for name in a:
        np.testing.assert_array_equal(a[name].rows,z[name].rows[:len(a[name].rows)])
        np.testing.assert_array_equal(a[name].boundaries,z[name].boundaries[:len(a[name].boundaries)])
    tiles.close()


def test_group_deadline_and_finite_estimates(tmp_path):
    deadline=b.Deadline(tmp_path/'clock.json',10)
    assert admit(deadline,[100,100],reserve=100)
    assert not admit(deadline,[201,201],reserve=100)
    assert not admit(deadline,[100,100],reserve=400)
    with pytest.raises(ValueError):admit(deadline,[float('nan')])
    with pytest.raises(ValueError):admit(deadline,[0])
    deadline.start();other=b.Deadline(tmp_path/'clock.json',10)
    assert deadline.record==other.record


@pytest.mark.parametrize('passes,size',[(0,2),(1,0),(-1,1)])
def test_invalid_sampler_contract(passes,size):
    with pytest.raises(ValueError):proportional_plan(SimpleNamespace(tiles=[]),np.array([]),719,passes,size)


def test_progression_requires_every_criterion():
    baseline={'cd8':1.25,'biology':1.27}
    good={'cd8':1.24,'biology':1.26,'positive_cd8_r2':.01}
    assert progression(good,baseline)
    for key,value in [('cd8',1.2495),('biology',1.28),('positive_cd8_r2',0),('positive_cd8_r2',None)]:
        assert not progression(dict(good,**{key:value}),baseline)


def test_spatial_correction_gate_does_not_require_flat_success():
    from dann.verification.extended_batch_diagnosis import correction_supported
    assert correction_supported([True]*3,[.08,.06,.008],[.8,.7,.2],True)
    assert not correction_supported([True,False,True],[.08,.06,.008],[.8,.7,.2],True)
    assert not correction_supported([True]*3,[.08,.06,-.002],[.8,.7,.2],True)
    assert not correction_supported([True]*3,[.0005]*3,[.8,.7,.2],True)
    assert not correction_supported([True]*3,[.08,.06,.008],[.8,.7,1e-5],True)
    assert not correction_supported([True]*3,[.08,.06,.008],[.8,.7,.2],False)
    assert not correction_supported([True]*2,[.08,.06],[.8,.7],True)
    assert not correction_supported([True]*3,[.08,float('nan'),.008],[.8,.7,.2],True)


def test_full_batch_precision_audit_preserves_initial_reference(spatial_config,tmp_path):
    from dann.verification.extended_batch_diagnosis import checkpoint_precision_audit
    import copy,json
    c=copy.deepcopy(spatial_config);c['training']['device']='cpu';c['model']['dropout']=.1
    bundle,model,_,_=setup_control(c);model.peak_budget=3;model.eval()
    tiles=b.SpatialTileDataset(b.geometry_config(c,core_size=4),bundle.split_indices['train'],bundle.metadata)
    plan=proportional_plan(tiles,bundle.split_indices['train'],20260719,2,7)
    study=SimpleNamespace(root=tmp_path,config=c,plans={20260719:{'Q':plan}},small=tiles,refs={20260719:model})
    before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    checkpoint_precision_audit(study)
    result=json.loads((tmp_path/'full_batch_checkpoint_precision.json').read_text())
    assert result['float64_strict']['passed'] and result['rows']==min(7,len(bundle.split_indices['train']))
    assert result['float32']['checkpoint_toggle']['rng_equal']
    assert result['float32']['same_mode_repeat']['rng_equal']
    assert not model.training and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in model.parameters())
    recorded=(tmp_path/'full_batch_checkpoint_precision.json').read_bytes()
    checkpoint_precision_audit(study)
    assert (tmp_path/'full_batch_checkpoint_precision.json').read_bytes()==recorded
    tiles.close()


@pytest.mark.parametrize('arguments',[
    ['--seeds','20260719'],
    ['--controls','QX2','QX2'],
    ['--controls','QX2','--seeds','20260719','20260719'],
    ['--controls','unknown'],
])
def test_replay_cli_rejects_ambiguous_selections(arguments,tmp_path):
    from dann.verification.extended_batch_diagnosis import main
    with pytest.raises(SystemExit):main(['--output',str(tmp_path/'new'),*arguments])
    assert not (tmp_path/'new').exists()

"""Scientific contracts for the isolated batch/execution runner."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dann.tests.test_components import spatial_config
from dann.tests.test_discriminator_controls import setup_control
from dann.losses import ZILNLoss
from dann.tiles import SpatialTileDataset, tile_collate
from dann.verification.batch_execution_diagnosis import (
    Deadline, Plan, PlannedDataset, adapt, audit, completed_matches, core_flat,
    equivalence, evaluate, file_hash, geometry_config, make_plans, objective,
    permutations, select_labels, verify_plan,
)


def test_execution_equivalence_and_excluded_labels(spatial_config):
    c = spatial_config
    bundle, model, flat, tiled = setup_control(c)
    model.peak_budget = 3  # spectra crossing a chunk boundary, including oversized spectra
    result = equivalence(model, flat, tiled, c)
    assert result['passed']
    for key in flat:
        torch.testing.assert_close(flat[key], core_flat(tiled)[key], equal_nan=True)
    selected = select_labels(tiled, torch.arange(0, len(tiled['row_ids']), 2))
    changed = copy.deepcopy(selected)
    changed['targets'][~changed['target_valid_mask']] = 1234.
    loss = ZILNLoss.from_config(c)
    m = adapt(model, 'X4', 'off')
    torch.testing.assert_close(objective(m, selected, loss, c)[2], objective(m, changed, loss, c)[2])


def test_checkpoint_dropout_rng_gradients_and_adamw(spatial_config):
    c = spatial_config
    c['model']['dropout'] = .1
    _, model, flat, tiled = setup_control(c)
    model.peak_budget = 3
    assert equivalence(model, flat, tiled, c, ('X1','X2'), True)['passed']


def test_all_schedules_coverage_slide_preservation_and_boundaries(spatial_config, tmp_path):
    c = spatial_config
    bundle, _, _, _ = setup_control(c)
    tiles = SpatialTileDataset(geometry_config(c, core_size=4), bundle.split_indices['train'], bundle.metadata)
    plans = make_plans(tiles, bundle.split_indices['train'], bundle.metadata.batch_codes, 91, small=7, tiles_per_batch=2)
    for name, plan in plans.items():
        verify_plan(plan, bundle.split_indices['train'])
        path = tmp_path/f'{name}.npz'
        plan.save(path); plan.save(path)
        assert Plan.load(path).sha256 == plan.sha256
    a, b = plans['F-native'], plans['F-dispersed']
    np.testing.assert_array_equal(a.boundaries, b.boundaries)
    np.testing.assert_array_equal(bundle.metadata.batch_codes[a.rows], bundle.metadata.batch_codes[b.rows])
    np.testing.assert_array_equal(plans['G-native'].rows, plans['G-small'].rows)
    np.testing.assert_array_equal(a.rows, plans['F-small'].rows)
    assert plans['G-native'].sha256 != plans['G-small'].sha256
    broken = copy.deepcopy(a); broken.rows[0] = broken.rows[1]
    with pytest.raises(AssertionError):
        verify_plan(broken, bundle.split_indices['train'])
    with pytest.raises(ValueError):
        broken.save(tmp_path/'F-native.npz')
    tiles.close()


def test_archived_persistent_loader_permutations():
    loader = DataLoader(range(17), batch_size=4, shuffle=True, num_workers=1,
        persistent_workers=True, generator=torch.Generator().manual_seed(19))
    expected = [torch.cat(list(loader)).numpy() for _ in range(2)]
    for a, b in zip(expected, permutations(17,19)):
        np.testing.assert_array_equal(a,b)


def test_partial_tiles_keep_complete_context_and_order(spatial_config):
    c = spatial_config
    bundle, _, _, _ = setup_control(c)
    tiles = SpatialTileDataset(geometry_config(c, core_size=4), bundle.split_indices['train'], bundle.metadata)
    plans = make_plans(tiles, bundle.split_indices['train'], bundle.metadata.batch_codes, 31, small=3, tiles_per_batch=2)
    plan = plans['F-small']
    spatial, flat = PlannedDataset(tiles,plan,True), PlannedDataset(tiles,plan,False)
    seen = []
    for i in range(plan.passes[1]):
        a,b = spatial[i],flat[i]
        assert a['peak_counts'].numel() >= len(a['row_ids'])
        for key in b:
            torch.testing.assert_close(core_flat(a)[key],b[key],equal_nan=True)
        seen.extend(a['row_ids'].tolist())
        for t in dict.fromkeys(spatial.row_tile[int(r)] for r in a['row_ids']):
            tile = tiles[t]
            slide = tiles.tiles[t][0][0]
            # Every occupied context site belongs to this slide, preserving gaps.
            assert all(tiles.keys[int(s['row_id'])][0] == slide for s in tile['samples'])
    np.testing.assert_array_equal(np.sort(seen), np.sort(bundle.split_indices['train']))
    tiles.close()


def test_audit_preserves_rng_mode_and_existing_gradients(spatial_config):
    c = spatial_config
    c['model']['dropout'] = .1
    _, model, flat, _ = setup_control(c)
    model = adapt(model,'X2').train()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = torch.get_rng_state().clone()
    result = audit(model,flat,ZILNLoss.from_config(c),c,12)
    assert torch.equal(before,torch.get_rng_state()) and model.training
    assert all(torch.equal(p.grad,torch.ones_like(p)) for p in model.parameters())
    assert set(result)=={'eval','training'}
    assert set(result['eval']['gradient_norms'])=={'embedding','peak_mlp','aggregation','biology','discriminator'}


def test_cnn_shortcut_rejected(spatial_config):
    from dann.model import AdversarialLatentFusion
    model = AdversarialLatentFusion.from_config(spatial_config,2,4)
    with pytest.raises(ValueError,match='CNN'):
        adapt(model,'X3')
    with pytest.raises(ValueError,match='CNN'):
        evaluate(model,None,None,None,torch.device('cpu'),np.array([]),False)


def test_persistent_deadline_and_immutable_resume(tmp_path,monkeypatch):
    now = [100.]
    monkeypatch.setattr('dann.verification.batch_execution_diagnosis.time.time',lambda:now[0])
    deadline = Deadline(tmp_path/'deadline.json',10)
    assert not deadline.path.exists()
    deadline.start(); now[0]=450.
    restored = Deadline(deadline.path,10)
    assert restored.remaining()==250.
    with pytest.raises(TimeoutError):
        restored.check(300.)
    with pytest.raises(ValueError):
        Deadline(deadline.path,20)
    directory = tmp_path/'run'; directory.mkdir()
    contract={'mode':'X4','plan':'abc'}
    assert not completed_matches(directory,contract)
    final=directory/'final.pt';final.write_bytes(b'final')
    (directory/'completion.json').write_text(json.dumps(dict(contract=contract,hashes={'final.pt':file_hash(final)})))
    assert completed_matches(directory,contract)
    with pytest.raises(ValueError):
        completed_matches(directory,dict(contract,mode='X0'))
    final.write_bytes(b'changed')
    with pytest.raises(ValueError):
        completed_matches(directory,contract)


def test_fit_deadline_saves_incomplete_and_refuses_overwrite(spatial_config,tmp_path):
    from dann.verification.batch_execution_diagnosis import fit
    c = spatial_config
    bundle, model, _, _ = setup_control(c)
    tiles = SpatialTileDataset(geometry_config(c,core_size=4),bundle.split_indices['train'],bundle.metadata)
    plan = make_plans(tiles,bundle.split_indices['train'],bundle.metadata.batch_codes,17,small=7)['G-small']
    audit_cpu = tile_collate([tiles[0]])
    tiles.close()
    torch.save(model.state_dict(),tmp_path/f'initial_{c["training"]["seed"]}.pt')
    deadline = Deadline(tmp_path/'deadline.json',.001);deadline.start()
    dest=tmp_path/'interrupted'
    with pytest.raises(RuntimeError,match='deadline'):
        fit(dest,model,c,bundle,tiles,plan,'X0','unchanged',1,deadline,audit_cpu)
    saved=torch.load(dest/'final.pt',weights_only=False)
    assert saved['status']=='incomplete' and saved['completed_update']==0
    assert not (dest/'completion.json').exists()
    before=file_hash(dest/'final.pt')
    with pytest.raises(FileExistsError):
        fit(dest,model,c,bundle,tiles,plan,'X0','unchanged',1,deadline,audit_cpu,resume=True)
    assert file_hash(dest/'final.pt')==before


def test_round_sampler_no_oversampling_and_partial_carry(spatial_config):
    from dann.verification.batch_execution_diagnosis import round_plan
    c=spatial_config
    bundle,_,_,_=setup_control(c)
    tiles=SpatialTileDataset(geometry_config(c,core_size=2),bundle.split_indices['train'],bundle.metadata)
    native=make_plans(tiles,bundle.split_indices['train'],bundle.metadata.batch_codes,31,small=3)['F-native']
    for size in (None,3):
        plan=round_plan(tiles,native,bundle.split_indices['train'],31,size)
        verify_plan(plan,bundle.split_indices['train'])
        assert len(plan.rows)==2*len(bundle.split_indices['train'])
        if size is None:
            np.testing.assert_array_equal(plan.boundaries,native.boundaries)
        else:
            assert np.max(np.diff(plan.boundaries))<=3
    tiles.close()


def test_eval_gradient_audit_checkpoints_without_enabling_dropout(spatial_config):
    c=spatial_config;c['model']['dropout']=.1
    _,model,_,tiled=setup_control(c)
    model=adapt(model,'X4').eval();model.peak_budget=3
    loss=ZILNLoss.from_config(c)
    expected=torch.autograd.grad(objective(model,tiled,loss,c)[2],list(model.parameters()),allow_unused=True)
    result=audit(model,tiled,loss,c,91)
    from dann.verification.fit_diagnosis import groups
    by_id={id(p):g for p,g in zip(model.parameters(),expected)}
    for name,params in groups(model).items():
        norm=sum(float(by_id[id(p)].double().square().sum()) for p in params if by_id[id(p)] is not None)**.5
        assert result['eval']['gradient_norms'][name]==pytest.approx(norm,rel=1e-6,abs=1e-8)
    assert not model.training and all(not m.training for m in model.modules())


@pytest.mark.parametrize('stage,controls,dropout,placement,sampler', [
    ('C1',['S'],'unchanged','none','native'),
    ('C2',['P'],'unchanged','none','native'),
    ('D',['P'],'unchanged','none','rounds'),
    ('B',['P'],'off','none','native'),
    ('B',['S'],'unchanged','discriminator','native'),
    ('D',['S'],'spectral-off','none','rounds'),
    ('D',['S'],'off','none','native'),
])
def test_incompatible_scientific_selections_rejected(stage,controls,dropout,placement,sampler):
    from dann.verification.batch_execution_diagnosis import validate_selection
    with pytest.raises(ValueError):
        validate_selection(stage,controls,dropout,placement,sampler)


def test_preflight_is_seed_and_schedule_specific(spatial_config,tmp_path,monkeypatch):
    import h5py
    from dann.verification import batch_execution_diagnosis as diagnosis
    c=spatial_config
    bundle,model,_,tiled=setup_control(c)
    tiles=SpatialTileDataset(geometry_config(c,core_size=4),bundle.split_indices['train'],bundle.metadata)
    plans=make_plans(tiles,bundle.split_indices['train'],bundle.metadata.batch_codes,31,small=7,tiles_per_batch=2)
    with h5py.File(c['data']['path']) as f:counts=np.diff(f['X/indptr'][:])
    plan=plans['F-native'];mass=np.add.reduceat(counts[plan.rows],plan.boundaries[:-1])
    blocked=torch.from_numpy(plan.batch(int(mass.argmax())))
    actual=diagnosis.objective
    def simulated_bound(model,batch,*args,**kwargs):
        if torch.equal(batch['row_ids'],blocked):
            raise torch.cuda.OutOfMemoryError('synthetic monolithic capacity')
        return actual(model,batch,*args,**kwargs)
    monkeypatch.setattr(diagnosis,'objective',simulated_bound)
    monkeypatch.setattr(diagnosis,'equivalence',lambda *a,**k:dict(passed=True))
    deadline=Deadline(tmp_path/'deadline.json',10)
    result=diagnosis.preflight(tmp_path,model,c,tiles,plans,tiled,deadline)
    assert not result['flat_feasibility']['F-native']['passed']
    assert result['flat_feasibility']['G-small']['passed']
    assert diagnosis.preflight(tmp_path,model,c,tiles,plans,tiled,deadline)==result
    changed=copy.deepcopy(plans);changed['F-native'].rows=changed['F-native'].rows[::-1].copy()
    with pytest.raises(ValueError,match='contract'):
        diagnosis.preflight(tmp_path,model,c,tiles,changed,tiled,deadline)
    old_seed=c['training']['seed'];c['training']['seed']+=1
    next_seed=diagnosis.preflight(tmp_path,model,c,tiles,plans,tiled,deadline)
    assert next_seed['contract']['seed']==old_seed+1
    assert (tmp_path/f'preflight_{old_seed}.json').exists()
    assert (tmp_path/f'preflight_{old_seed+1}.json').exists()
    tiles.close()

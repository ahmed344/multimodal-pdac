"""H&E-disabled selection, spatial extraction, immutable stages and safe execution."""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest
import torch

from dann.config import load_config, resolve_execution, execution_settings
from dann.data_loader import create_data_bundle
from dann.model import AdversarialLatentFusion
from dann.sampling import SamplingEpochAudit
from dann.tests.test_components import spatial_config
from dann.tests.test_alignment_sweep import metric_record
from dann.tests.test_training_artifacts import assert_nested_equal
from dann.verification import alignment_no_he as study
from dann.verification import alignment_sweep as sweep
from dann.verification.dictionary_comparison import close


@pytest.mark.parametrize('architecture',['mlp','cnn','gatv2'])
def test_configs_preserve_science_and_disable_every_workflow(tmp_path,architecture):
    source = load_config(Path('dann/config.yaml'))
    before = copy.deepcopy(source)
    c = study.make_config(source,tmp_path,'test',(.03,.01,.3),architecture)
    assert source == before
    study.assert_disabled(c)
    assert c['training']['epochs'] == 300
    assert c['training']['seed'] == 20260719
    assert c['training']['archive_checkpoint_every'] == 5
    assert c['training']['lr_scheduler'] == source['training']['lr_scheduler']
    assert c['data'] == source['data']
    assert c['loss']['target_weights'] == source['loss']['target_weights']
    if architecture != 'mlp':
        assert execution_settings(c)['halo'] == 6
        assert c['training']['core_size'] == source['training']['spatial']['core_size']
        assert c['training']['supervised_rows_per_batch'] == source['training']['spatial']['supervised_rows_per_batch']
        assert c['model']['aggregation'][architecture]['channels'] == 128
    assert all(c['loss']['batch_alignment'][n]['enabled'] for n in study.NAMES)


def test_initial_choice_does_not_require_mixing_improvement_and_uses_both_limits():
    control,reference = metric_record('control'),metric_record('reference')
    a = metric_record('a',.9,.95)
    assert study.choose([a],control,reference)['id'] == 'a'
    assert study.choose([a],control,reference,base=control) is None
    control['probes']['stroma']['mse'] = .08
    assert not study.dual_eligibility(a,control,reference)['eligible']
    control = metric_record('control')
    reference['probes']['stroma']['mse'] = .08
    assert study.choose([a],control,reference) is None


def test_ties_boundaries_and_stable_checkpoint_id():
    control = metric_record('control')
    a,b = metric_record('a',.2,.4),metric_record('b',.21,.3)
    assert study.choose([a,b],control,control)['id'] == 'b'
    b['mixing']['latent']['excess'] += 1e-7
    assert study.choose([a,b],control,control)['id'] == 'a'
    b = copy.deepcopy(a); b['id'] = 'b'
    assert study.choose([b,a],control,control)['id'] == 'a'
    a['probes']['tumor']['mse'] = .11
    assert study.dual_eligibility(a,control,control)['eligible']
    a['probes']['tumor']['mse'] += 1e-8
    assert not study.dual_eligibility(a,control,control)['eligible']
    b['selectable'] = False
    assert study.choose([b],control,control) is None


@pytest.mark.parametrize('role',['control','reference'])
@pytest.mark.parametrize('failure',['collapsed','missing_metric'])
def test_invalid_reference_cannot_select_candidates(role,failure):
    references = {name:metric_record(name) for name in ('control','reference')}
    invalid = references[role]
    if failure == 'collapsed':
        invalid.update(spectrum={'noncollapsed':False},probes={},distributions={},mixing={})
    else:
        invalid['probes']['stroma']['mse'] = None
    candidate = metric_record('candidate')
    result = study.dual_eligibility(candidate,**references)
    assert not result['eligible']
    assert result['references'][role]['checks'] == [dict(metric='reference_valid',passed=False)]
    assert study.choose([candidate],**references) is None


@pytest.mark.parametrize('actual',[[1,1,3],[1,2],[1,2,4],[1,2,3,4]])
def test_extraction_rejects_duplicate_missing_or_extra_rows(actual):
    with pytest.raises(ValueError,match='identities'):
        sweep.restore_row_order(dict(rows=np.array(actual),latent=np.zeros((len(actual),2))),np.array([1,2,3]))


@pytest.mark.parametrize('architecture',['mlp','cnn','gatv2'])
def test_disabled_he_extraction_uses_normal_context_and_restores_order(spatial_config,architecture,tmp_path):
    c = study.make_config(spatial_config,tmp_path,'extract',(0.,0.,0.),architecture)
    # Fixture deliberately contains no H&E obsm and has different tissue/lesion rows.
    bundle = create_data_bundle(c)
    try:
        assert bundle.he_reader is None
        model = AdversarialLatentFusion.from_config(c,2,4).eval()
        rows = bundle.split_indices['validation'][::-1].copy()
        data,he = sweep.extract_rows(model,c,bundle,rows,he=True)
        np.testing.assert_array_equal(data['rows'],rows)
        assert he == dict(available=False,reason='head_disabled')
        expected = {}
        with torch.inference_mode():
            for batch in bundle.loaders['validation']:
                assert 'he_targets' not in batch
                assert set(batch['row_ids'].tolist()) <= set(rows)
                output = model(batch)
                for i,row in enumerate(batch['row_ids'].tolist()):
                    expected[row] = output['latent'][i].numpy()
        np.testing.assert_allclose(data['latent'],np.stack([expected[r] for r in rows]),rtol=1e-5,atol=1e-6)
        if architecture != 'mlp':
            assert bundle.datasets['validation'].path == Path(c['data']['path'])
            training_rows = set(bundle.split_indices['train'])
            validation_rows = set(bundle.split_indices['validation'])
            for batch in bundle.loaders['train']:
                assert set(batch['row_ids'].tolist()) <= training_rows
                assert not (set(batch['row_ids'].tolist()) & validation_rows)
                assert len(batch['targets']) == len(batch['row_ids'])
    finally:
        close(bundle)


def test_coverage_hash_is_order_independent_and_exposes_duplicates():
    def audit(rows):
        obj = SamplingEpochAudit(0,4)
        obj.update(dict(row_ids=torch.tensor(rows),batches=torch.tensor([0,0,1,1])))
        return obj.summary()
    a,b = audit([1,3,2,4]),audit([4,2,3,1])
    assert a['row_coverage_sha256'] == b['row_coverage_sha256']
    assert a['row_order_sha256'] != b['row_order_sha256']
    assert a['unique_rows'] == 4
    assert audit([1,1,2,4])['unique_rows'] == 3


def test_runtime_guards_cover_all_entry_points():
    from dann import workflow, spatial, spatial_heatmaps
    with study.training_validation_only():
        for fn in (workflow.run_post_training_workflow,spatial.run_inference,
                   spatial._run_inference,spatial_heatmaps.run_spatial_heatmaps):
            with pytest.raises(RuntimeError,match='Forbidden'):
                fn()


def test_registration_deduplicates_and_rejects_changed_config(tmp_path):
    for name in ('configs','arms','decisions'):
        (tmp_path/name).mkdir()
    source = load_config(Path('dann/config.yaml'))
    import yaml
    (tmp_path/'source_config.yaml').write_text(yaml.safe_dump(source))
    first = study.register_arm(tmp_path,(0.,0.,0.))
    assert study.register_arm(tmp_path,(0.,0.,0.)) == first
    assert len(list((tmp_path/'arms').glob('*.json'))) == 1
    path = tmp_path/'configs'/f'{first}.yaml'
    path.write_text(path.read_text()+'# changed\n')
    with pytest.raises(ValueError,match='configuration changed'):
        study.register_arm(tmp_path,(0.,0.,0.))


def test_completed_stage_reuses_without_training(tmp_path,monkeypatch):
    (tmp_path/'stages').mkdir(); (tmp_path/'manifest.json').write_text('{}')
    c = study.make_config(load_config(Path('dann/config.yaml')),tmp_path,'test',(0.,0.,0.))
    monkeypatch.setattr(study,'manifest_for',lambda root:{})
    monkeypatch.setattr(study,'config_for',lambda root,arm:copy.deepcopy(c))
    output = tmp_path/'checkpoint.pt'; output.write_bytes(b'checkpoint')
    identity = dict(manifest_sha256=study.digest_file(tmp_path/'manifest.json'),arm='test',stage='train',config=study.fingerprint(c))
    stamp = tmp_path/'stages/test-train.json'
    sweep.complete(stamp,identity,dict(files=[str(output)]),time.monotonic())
    monkeypatch.setattr(study,'train_model',lambda *a,**kw:pytest.fail('Retrained completed stage'))
    before = stamp.read_bytes()
    study.execute_stage(tmp_path,'test','train')
    assert stamp.read_bytes() == before


@pytest.mark.parametrize('architecture',['mlp','cnn','gatv2'])
def test_disabled_he_resume_equivalence(spatial_config,tmp_path,monkeypatch,architecture):
    from dann import train
    c = study.make_config(spatial_config,tmp_path,'continuous',(0.,0.,0.),architecture)
    c['training'].update(epochs=2,early_stopping_enabled=False,archive_checkpoint_every=1)
    c['analysis']['figure_dpi'] = 12
    c['performance']['async_artifacts'] = False
    monkeypatch.setattr(train,'refresh_loss_curves',lambda *a,**kw:None)
    # Run under the same forbidden-entry guards used by normal/resumed/smoke stages.
    with study.training_validation_only():
        continuous = train.train_model(c)
        expected = torch.load(continuous.parent/'latest.pt',weights_only=False)
        c['training']['output_dir'] = str(tmp_path/'resumed')
        train.train_model(c,stop_after_epoch=1)
        c['training']['resume_checkpoint'] = str(tmp_path/'resumed/latest.pt')
        train.train_model(c)
    actual = torch.load(tmp_path/'resumed/latest.pt',weights_only=False)
    for key in ('model_state','optimizer_state','training_rng_state','early_stopping_state',
                'sampler_state','training_progress','lr_scheduler_state','warmup_state'):
        if key in expected:
            assert_nested_equal(expected[key],actual[key])
    history = pd.read_csv(tmp_path/'resumed/history.csv')
    assert not any('he_mse' in k or 'he_r2' in k for k in history)
    np.testing.assert_allclose(history.validation_supervised_objective,history.validation_biology_weighted_loss)


@pytest.mark.parametrize('invalid_control',[False,True])
def test_no_eligible_nhsic_stops_before_architecture_transfer(tmp_path,monkeypatch,invalid_control):
    for name in ('stages','diagnostics','decisions','logs','arms'):
        (tmp_path/name).mkdir()
    ref = dict(metric_record(),arm='reference',supervised_selected=True)
    monkeypatch.setattr(study,'manifest_for',lambda root:dict(historical=dict(reference=[ref])))
    monkeypatch.setattr(study,'verify_suite',lambda root:None)
    monkeypatch.setattr(study,'register_arm',lambda root,w,architecture='mlp':study.arm_name(architecture,w))
    monkeypatch.setattr(study,'report',lambda *a,**kw:None)
    calls = []
    monkeypatch.setattr(study,'run_subprocess',lambda root,arm,stage:calls.append((arm,stage)))
    def records(root,arm):
        r = dict(metric_record(arm),arm=arm,supervised_selected=True)
        r['probes']['tumor']['mse'] = .3
        if invalid_control:
            r.update(spectrum={'noncollapsed':False},probes={},distributions={},mixing={})
        return [r]
    monkeypatch.setattr(study,'records_for',records)
    finished = []
    monkeypatch.setattr(study,'finish',lambda root,m,control,reference,winner,reason:finished.append((winner,reason)))
    study.run_all(tmp_path)
    assert finished[0][0] is None
    fits = [arm for arm,stage in calls if stage=='train']
    assert len(fits) == (1 if invalid_control else 6)
    assert all(a.startswith('mlp_') for a in fits)
    assert [s for a,s in calls[:3]] == ['smoke']*3


def test_invalid_control_finishes_with_explicit_unavailable_umap(tmp_path,monkeypatch):
    for name in ('arms','decisions'):
        (tmp_path/name).mkdir()
    arm = study.arm_name('mlp',(0.,0.,0.))
    control = dict(metric_record(arm),arm=arm,supervised_selected=True,
                   spectrum={'noncollapsed':False},probes={},distributions={},mixing={})
    reference = metric_record('reference')
    (tmp_path/'arms'/f'{arm}.json').write_text('{}')
    monkeypatch.setattr(study,'records_for',lambda root,name:[control])
    calls = []
    monkeypatch.setattr(study,'run_subprocess',lambda *args:calls.append(args))
    monkeypatch.setattr(study,'report',lambda *args,**kwargs:None)
    monkeypatch.setattr(study,'audit',lambda *args:None)
    study.finish(tmp_path,{'anchor':reference},control,reference,None,'Invalid control')
    assert not calls
    assert json.loads((tmp_path/'decisions/selected_epochs.json').read_text()) == {}
    assert arm in json.loads((tmp_path/'decisions/unavailable_umaps.json').read_text())
    recommendation = json.loads((tmp_path/'decisions/recommendation.json').read_text())
    assert not recommendation['architecture_transfer']
    assert not recommendation['replaced_combined']


@pytest.mark.parametrize('architecture_control',['valid','collapsed','excluded'])
def test_decisions_normalize_tuples_and_restart_complete_sequence(tmp_path,monkeypatch,architecture_control):
    for name in ('stages','diagnostics','decisions','logs','arms','configs'):
        (tmp_path/name).mkdir()
    ref = dict(metric_record(),arm='reference',supervised_selected=True)
    monkeypatch.setattr(study,'manifest_for',lambda root:dict(historical=dict(reference=[ref])))
    monkeypatch.setattr(study,'verify_suite',lambda root:None)
    monkeypatch.setattr(study,'report',lambda *a,**kw:None)
    records, calls, arm_weights = {}, [], {}
    def register(root,w,architecture='mlp'):
        arm = study.arm_name(architecture,w)
        arm_weights[arm] = w
        (root/'arms'/f'{arm}.json').write_text(json.dumps(dict(weights=w)))
        return arm
    monkeypatch.setattr(study,'register_arm',register)
    def stage(root,arm,stage):
        if stage == 'train':
            if arm not in records:
                calls.append(arm)
                weights = arm_weights[arm]
                # nHSIC .03 wins; all later additions improve both scores.
                score = .1+abs(weights[0]-.03)-weights[1]*.1-weights[2]*.01
                records[arm] = [dict(metric_record(arm,score,score+.1),arm=arm,supervised_selected=True)]
                if arm == study.arm_name('cnn',(0.,0.,0.)):
                    if architecture_control == 'collapsed':
                        records[arm][0].update(spectrum={'noncollapsed':False},
                            probes={},distributions={},mixing={})
                    elif architecture_control == 'excluded':
                        records[arm] = []
    monkeypatch.setattr(study,'run_subprocess',stage)
    monkeypatch.setattr(study,'records_for',lambda root,arm:records.get(arm,[]))
    outcomes = []
    monkeypatch.setattr(study,'finish',lambda root,m,control,reference,winner,reason:outcomes.append(winner))
    study.run_all(tmp_path)
    before = list(calls)
    study.run_all(tmp_path)
    assert calls == before and len(calls) == 18
    assert outcomes[0] == outcomes[1]
    if architecture_control != 'valid':
        comparison = json.loads((tmp_path/'decisions/cnn.json').read_text())
        assert not comparison['comparison_available']
        assert comparison['selected'] is None
        assert not comparison['improves_own_control']
        assert 'control' in comparison['unavailable_reason']
    frozen = json.loads((tmp_path/'decisions/frozen_mlp.json').read_text())
    for arm,w in arm_weights.items():
        if not arm.startswith('mlp_'):
            assert list(w) in ([0.,0.,0.],frozen['weights'])


def test_real_smoke_fixture_reads_neither_he_nor_tissue(spatial_config,tmp_path,monkeypatch):
    import h5py
    c = copy.deepcopy(spatial_config)
    c['data']['he_features_key'] = 'absent_he_targets'
    c['data']['context_path'] = '/nonexistent/forbidden_tissue.h5ad'
    original = h5py.File.__init__
    def guarded(self,name,*args,**kwargs):
        assert str(name) != c['data']['context_path']
        return original(self,name,*args,**kwargs)
    monkeypatch.setattr(h5py.File,'__init__',guarded)
    sweep.smoke_inputs(tmp_path,c,include_he=False,lesion_only=True,spatial_regions=True)
    with h5py.File(tmp_path/'smoke_data/path.h5ad','r') as handle:
        assert 'absent_he_targets' not in handle['obsm']
        assert 'msi_preprocess' not in handle['uns']
    provenance = json.loads((tmp_path/'smoke_data/provenance.json').read_text())
    assert set(provenance['inputs']) == {'path'}
    assert not provenance['preprocessing_refitted']


@pytest.mark.parametrize('field,value',[('seed',5),('weight_decay',.3),('epochs',200),('early_stopping_patience',8)])
def test_resume_rejects_scientific_configuration_changes(field,value):
    c = load_config(Path('dann/config.yaml'))
    saved = copy.deepcopy(c)
    saved['training']['resume_checkpoint'] = '/old/latest.pt'
    study.validate_scientific_config(saved,c)
    saved['training'][field] = value
    with pytest.raises(ValueError,match='scientific configuration mismatch'):
        study.validate_scientific_config(saved,c)


@pytest.mark.parametrize('mixing_eligible,invalid_control',[(True,False),(False,False),(False,True)])
def test_report_preserves_reference_and_selection_labels(tmp_path,monkeypatch,mixing_eligible,invalid_control):
    for name in ('arms','diagnostics','decisions','model'):
        (tmp_path/name).mkdir()
    history = dict(epoch=5,validation_cd8_loss=.9,validation_biology_loss=1.,validation_biology_weighted_loss=2.)
    pd.DataFrame([history]).to_csv(tmp_path/'model/history.csv',index=False)
    def record(arm):
        return dict(metric_record(arm),arm=arm,epoch=5,supervised_selected=True,
                    checkpoint=str(tmp_path/'model/best.pt'),training_history=[history])
    own = record(study.arm_name('mlp',(0.,0.,0.)))
    if invalid_control:
        own.update(spectrum={'noncollapsed':False},probes={},distributions={},mixing={})
    old = {name:[record(name)] for name in study.HISTORICAL}
    (tmp_path/'arms'/f'{own["arm"]}.json').write_text('{}')
    (tmp_path/'decisions/selected_epochs.json').write_text(json.dumps({own['arm']:own}))
    (tmp_path/'decisions/mixing_epochs.json').write_text(json.dumps({own['arm']:own if mixing_eligible else None}))
    monkeypatch.setattr(study,'manifest_for',lambda root:dict(previous=str(tmp_path/'old'),historical=old))
    monkeypatch.setattr(study,'records_for',lambda root,arm:[own])
    monkeypatch.setattr(study,'plot_curves',lambda *a:None)
    study.report(tmp_path)
    table = pd.read_csv(tmp_path/'eligibility_limits.csv')
    assert set(table.reference_name) == {'control','reference'}
    assert table.loc[table.metric=='probe_stroma_mse','reference'].eq(.1).all()
    comparison = pd.read_csv(tmp_path/'checkpoint_comparison.csv')
    displayed = comparison[comparison.arm==own['arm']].iloc[0]
    assert displayed.umap_selected
    assert bool(displayed.mixing_selected) == mixing_eligible
    if invalid_control:
        assert not displayed.eligible
    assert 'he_component_comparisons.csv' in (tmp_path/'report.html').read_text()


@pytest.mark.parametrize('collapsed',['control','aligned'])
def test_architecture_comparison_marks_invalid_evidence_unavailable(tmp_path,monkeypatch,collapsed):
    for name in ('arms','diagnostics','decisions','model'):
        (tmp_path/name).mkdir()
    history = dict(epoch=5,validation_cd8_loss=.9,validation_biology_loss=1.)
    pd.DataFrame([history]).to_csv(tmp_path/'model/history.csv',index=False)
    def record(arm):
        return dict(metric_record(arm),arm=arm,epoch=5,supervised_selected=True,
                    checkpoint=str(tmp_path/'model/best.pt'),training_history=[history])
    weights = (.1,.003,.1)
    control = record(study.arm_name('cnn',(0.,0.,0.)))
    aligned = record(study.arm_name('cnn',weights))
    invalid = control if collapsed=='control' else aligned
    invalid.update(spectrum={'noncollapsed':False},probes={},distributions={},mixing={})
    own = record(study.arm_name('mlp',(0.,0.,0.)))
    records = {r['arm']:r for r in (own,control,aligned)}
    for arm in records:
        (tmp_path/'arms'/f'{arm}.json').write_text('{}')
    (tmp_path/'decisions/frozen_mlp.json').write_text(json.dumps(dict(weights=weights)))
    old = {name:[record(name)] for name in study.HISTORICAL}
    monkeypatch.setattr(study,'manifest_for',lambda root:dict(previous=str(tmp_path/'old'),historical=old))
    monkeypatch.setattr(study,'records_for',lambda root,arm:[records[arm]])
    monkeypatch.setattr(study,'plot_curves',lambda *a:None)
    study.report(tmp_path)
    table = pd.read_csv(tmp_path/'architecture_comparisons.csv')
    assert len(table) == 2
    assert not table.comparison_available.any()
    assert table[['latent_excess_delta','umap_excess_delta','cd8_loss_delta']].isna().all().all()
    reason = 'invalid_control' if collapsed=='control' else 'invalid_aligned_checkpoint'
    assert table.unavailable_reason.eq(reason).all()


def test_recovery_snapshot_is_checked_and_base_is_immutable(tmp_path):
    (tmp_path/'source').mkdir()
    (tmp_path/'manifest.json').write_text('{}')
    original = tmp_path/'source/original.py';original.write_text('old source')
    revision = tmp_path/'recoveries/0001';(revision/'source').mkdir(parents=True)
    fixed = revision/'source/original.py';fixed.write_text('fixed source')
    (revision/'changes.patch').write_text('reviewable diff')
    provenance = dict(parent='base',manifest_sha256=study.digest_file(tmp_path/'manifest.json'),
        sources={'original.py':study.digest_file(fixed)},evidence={'changes.patch':study.digest_file(revision/'changes.patch')})
    (revision/'provenance.json').write_text(json.dumps(provenance))
    assert study.active_source(tmp_path) == revision/'source'
    assert original.read_text() == 'old source'
    fixed.write_text('silently changed')
    with pytest.raises(ValueError,match='Recovery source changed'):
        study.active_source(tmp_path)


def test_execution_receipt_keeps_loaded_revision(tmp_path,monkeypatch):
    loaded = tmp_path/'recoveries/0001/source'
    loaded.mkdir(parents=True)
    provenance = loaded.parent/'provenance.json'
    provenance.write_text('{"revision": 1}')
    monkeypatch.setattr(study,'__file__',str(loaded/'dann/verification/alignment_no_he.py'))
    monkeypatch.setattr(study,'active_source',lambda root:tmp_path/'recoveries/0002/source')
    receipt = study.execution_provenance(tmp_path)
    assert receipt['source'] == str(loaded)
    assert receipt['revision_sha256'] == study.digest_file(provenance)


@pytest.mark.parametrize('change',['function','constant'])
def test_reporting_recovery_rejects_scientific_edits(tmp_path,monkeypatch,change):
    root = tmp_path/'study'
    repository = tmp_path/'repository'
    source = root/'source/dann/verification/alignment_no_he.py'
    target = repository/'dann/verification/alignment_no_he.py'
    for p in (source,target):
        p.parent.mkdir(parents=True)
        p.write_text('WEIGHT = .1\n\ndef choose():\n    return 1\n\ndef report():\n    return "old"\n')
    for directory in (root/'source',repository):
        tests = directory/'dann/tests/test_alignment_no_he.py'
        tests.parent.mkdir(parents=True)
        tests.write_text('# tests\n')
    (root/'manifest.json').write_text('{}')
    original = source.read_bytes()
    target.write_text(target.read_text().replace('"old"','"fixed"').replace(
        'def choose():\n','def choose():\n    """Documented selection."""\n'))
    monkeypatch.setattr(study,'__file__',str(target))
    monkeypatch.setattr(study,'manifest_for',lambda root:{})
    study.freeze_recovery(root,'reporting regression')
    assert source.read_bytes() == original
    assert study.active_source(root) == root/'recoveries/0001/source'
    with (root/'runner.lock').open('a') as lock:
        study.fcntl.flock(lock,study.fcntl.LOCK_EX|study.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='runner is active'):
            study.freeze_recovery(root,'unsafe concurrent recovery')
    old,new = ('return 1','return 2') if change=='function' else ('WEIGHT = .1','WEIGHT = .3')
    target.write_text(target.read_text().replace(old,new))
    with pytest.raises(ValueError,match='may not change'):
        study.freeze_recovery(root,'invalid scientific edit')
    assert not (root/'recoveries/0002').exists()

"""Diagnostic isolation, true pilot pauses, safe stages and historical reuse."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest
import torch

from dann import train
from dann.batch_alignment import AlignmentEvaluationError, AlignmentSummary, Measurement, alignment_settings
from dann.checkpoints import checkpoint_config, validate_checkpoint
from dann.config import load_config, resolve_execution
from dann.tests.test_components import spatial_config
from dann.tests.test_training_artifacts import assert_nested_equal
from dann.tests.test_alignment_sweep import metric_record
from dann.verification import alignment_refinement as refinement
from dann.verification.alignment_study_metrics import choose


def test_zero_weight_unavailable_has_finite_contribution_and_complete_evidence():
    settings=alignment_settings({'loss':{'batch_alignment':{
        'nhsic':dict(enabled=True,weight=10.),'soft_nn':dict(enabled=True,weight=0.)}}})
    summary=AlignmentSummary(settings)
    summary.add({'nhsic':Measurement(torch.tensor(.2)),
                 'soft_nn':Measurement(None,{'similarity_std':1e-6},'degenerate_similarity_variance')})
    values=summary.finish()
    assert np.isnan(values['alignment_soft_nn'])
    assert values['alignment_soft_nn_valid_count']==0
    assert values['alignment_soft_nn_invalid_degenerate_similarity_variance']==1
    assert values['alignment_soft_nn_weighted_loss']==0.
    assert np.isfinite(3.+sum(v for k,v in values.items() if k.endswith('_weighted_loss')))
    settings['soft_nn']['weight']=1.
    with pytest.raises(AlignmentEvaluationError) as error:
        summary.finish()
    assert error.value.evidence['alignment_nhsic_valid_count']==1
    assert error.value.evidence['alignment_soft_nn_invalid_count']==1


def test_nonfinite_diagnostic_is_unavailable_not_valid():
    settings=alignment_settings({'loss':{'batch_alignment':{'nhsic':dict(enabled=True,weight=1.)}}})
    summary=AlignmentSummary(settings)
    summary.add({'nhsic':Measurement(torch.tensor(float('nan')))})
    with pytest.raises(AlignmentEvaluationError) as error:
        summary.finish()
    assert error.value.evidence['alignment_nhsic_invalid_nonfinite_score']==1


def small_config(c,tmp_path):
    c=copy.deepcopy(c)
    c['model']['aggregation']['type']='mlp'
    for name in ('biology','discriminator'):
        c['model']['heads'][name]['type']='mlp'
    c['model']['dropout']=.2
    c['training'].update(epochs=6,early_stopping_enabled=False,archive_checkpoint_every=5)
    c['training']['pixel'].update(batch_size=32,num_workers=2)
    c['performance'].update(ram_cache=True,preparation_workers=1,async_artifacts=False)
    c['analysis']['figure_dpi']=12
    c=resolve_execution(c)
    c['training']['output_dir']=str(tmp_path)
    return c


def test_five_epoch_pilot_matches_uninterrupted_horizon_and_rng(spatial_config,tmp_path,monkeypatch):
    monkeypatch.setattr(train,'refresh_loss_curves',lambda *a,**k:None)
    c=small_config(spatial_config,tmp_path/'continuous')
    train.train_model(c)
    expected=torch.load(tmp_path/'continuous/latest.pt',weights_only=False)
    c['training']['output_dir']=str(tmp_path/'pilot')
    train.train_model(c,stop_after_epoch=5)
    paused=torch.load(tmp_path/'pilot/latest.pt',weights_only=False)
    assert paused['epoch']==4 and paused['config']['training']['epochs']==6
    assert paused['early_stopping_state']['stopped'] is False
    assert (tmp_path/'pilot/archives/epoch_0005.pt').exists()
    c['training']['resume_checkpoint']=str(tmp_path/'pilot/latest.pt')
    train.train_model(c)
    actual=torch.load(tmp_path/'pilot/latest.pt',weights_only=False)
    for key in ('model_state','optimizer_state','training_rng_state','early_stopping_state',
                'sampler_state','training_progress','lr_scheduler_state','warmup_state'):
        if key in expected:
            assert_nested_equal(expected[key],actual[key])
    left=pd.read_csv(tmp_path/'continuous/history.csv')
    right=pd.read_csv(tmp_path/'pilot/history.csv')
    keys=[k for k in left if k.startswith('validation_') and 'seconds' not in k and 'gib' not in k and 'per_second' not in k and 'prepared_batch' not in k]
    pd.testing.assert_frame_equal(left[keys],right[keys])


@pytest.mark.parametrize('kind',['alignment','outputs'])
def test_failure_snapshot_cannot_replace_resume_or_selection(spatial_config,tmp_path,monkeypatch,kind):
    monkeypatch.setattr(train,'refresh_loss_curves',lambda *a,**k:None)
    c=small_config(spatial_config,tmp_path/'run')
    original=train.run_epoch
    def fail(**kw):
        if kw['epoch_index']==1:
            if kind=='alignment':
                raise AlignmentEvaluationError('Enabled alignment objective nhsic has no valid evaluations',
                    {'alignment_nhsic_invalid_degenerate_kernel_norm':1})
            error=FloatingPointError('Non-finite model output')
            error.evidence={'nonfinite_outputs':{'latent':3}}
            raise error
        return original(**kw)
    monkeypatch.setattr(train,'run_epoch',fail)
    with pytest.raises((AlignmentEvaluationError,FloatingPointError)):
        train.train_model(c)
    latest=tmp_path/'run/latest.pt'
    assert torch.load(latest,weights_only=False)['epoch']==0
    failures=list((tmp_path/'run/failures').glob('*/snapshot.pt'))
    assert len(failures)==1
    snapshot=torch.load(failures[0],weights_only=False)
    assert snapshot['checkpoint_kind']=='failure' and snapshot['selectable'] is False
    evidence=json.loads(failures[0].with_name('diagnostics.json').read_text())
    assert evidence['diagnostics'] and evidence['resumable'] is False
    assert evidence['last_completed_checkpoint']==str(latest)
    with pytest.raises(ValueError,match='Failure snapshots'):
        checkpoint_config(snapshot)
    with pytest.raises(ValueError,match='Failure snapshots'):
        validate_checkpoint(snapshot,c)
    c['training']['resume_checkpoint']=str(failures[0])
    with pytest.raises(ValueError,match='Failure snapshots'):
        train.train_model(c)
    monkeypatch.setattr(train,'run_epoch',original)
    c['training']['resume_checkpoint']=str(latest)
    train.train_model(c,stop_after_epoch=2)
    assert torch.load(latest,weights_only=False)['epoch']==1
    ref=metric_record(); candidate=metric_record('failure',.01,.01)
    candidate['selectable']=False
    assert choose([candidate],ref) is None


def test_pilot_gate_does_not_require_early_mixing_improvement():
    ref=metric_record()
    candidate=dict(metric_record('pilot',.9,.95),epoch=5,checkpoint='epoch_0005.pt',training_history=[{
        'train_alignment_nhsic':.02,'validation_alignment_nhsic':.02,
        'train_alignment_nhsic_valid_count':20,'validation_alignment_nhsic_valid_count':1}])
    c={'loss':{'batch_alignment':{n:dict(weight=10. if n=='nhsic' else 0.) for n in refinement.NAMES}}}
    assert refinement.pilot_gate(candidate,ref,c)['extend']
    candidate['training_history'][0]['validation_alignment_nhsic_valid_count']=0
    assert not refinement.pilot_gate(candidate,ref,c)['extend']
    candidate['training_history'][0]['validation_alignment_nhsic_valid_count']=1
    candidate['probes']['stroma']['mse']=.111
    assert not refinement.pilot_gate(candidate,ref,c)['extend']


def test_historical_dependency_mismatch_and_unchanged_artifacts(tmp_path,monkeypatch):
    old=tmp_path/'old.pt'; old.write_bytes(b'original checkpoint')
    digest=refinement.digest_file(old)
    m={'dependencies':{str(old):digest}}
    monkeypatch.setattr(refinement.sweep,'manifest_for',lambda root:m)
    assert refinement.manifest_for(tmp_path)==m
    assert old.read_bytes()==b'original checkpoint'
    old.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError,match='Historical dependency changed'):
        refinement.manifest_for(tmp_path)


def test_stages_and_smoke_never_invoke_downstream_workflow(tmp_path,monkeypatch):
    from dann import workflow
    from dann.verification import dictionary_comparison
    def forbidden(*a,**k):
        pytest.fail('Tissue/spatial/downstream workflow invoked')
    monkeypatch.setattr(workflow,'run_post_training_workflow',forbidden)
    monkeypatch.setattr(refinement.sweep,'recommendation_workflow',forbidden)
    monkeypatch.setattr(dictionary_comparison,'smoke',forbidden)
    monkeypatch.setattr(refinement.sweep,'smoke_inputs',lambda root,c:None)
    c=load_config(Path('dann/config.yaml'))
    c['results'].update(root=str(tmp_path/'runs'),run_name='smoke_isolated')
    def fit(config):
        assert not any(config['post_training'][k] for k in ('enabled','run_inference','run_analysis','run_spatial_heatmaps'))
        directory=Path(config['training']['output_dir']);(directory/'archives').mkdir(parents=True)
        ck=dict(config=config,epoch=0,training_rng_state={'yes':True},split_indices={'train':np.arange(2),'validation':np.arange(2)})
        torch.save(ck,directory/'best.pt');torch.save(ck,directory/'archives/epoch_0001.pt')
        pd.DataFrame([dict(train_total_loss=1.,validation_total_loss=1.,validation_supervised_objective=1.)]).to_csv(directory/'history.csv',index=False)
        return directory/'best.pt'
    monkeypatch.setattr(refinement,'train_model',fit)
    monkeypatch.setattr(refinement,'validate_checkpoint',lambda *a:None)
    result=refinement.bounded_smoke(tmp_path,c)
    assert result['inference'] is False and result['spatial_exports'] is False
    assert set(refinement.STAGES)=={'smoke','pilot','pilot_evaluate','train','evaluate','full_umap'}


def test_run_order_restart_and_no_recommendation_workflow(tmp_path,monkeypatch):
    for name in ('stages','diagnostics','decisions','logs'):(tmp_path/name).mkdir()
    reference=dict(metric_record(),arm='reference',epoch=29,checkpoint='reference.pt',supervised_selected=True)
    anchor=dict(metric_record('combined',.15,.24),arm='combined',epoch=39,checkpoint='combined.pt',supervised_selected=False)
    monkeypatch.setattr(refinement,'manifest_for',lambda root:dict(anchor=anchor))
    monkeypatch.setattr(refinement,'report',lambda *a,**k:None)
    monkeypatch.setattr(refinement.sweep,'recommendation_workflow',lambda *a:pytest.fail('forbidden workflow'))
    records={'reference':[reference],'combined':[anchor]}
    calls=[]
    def stage(root,arm,stage):
        calls.append((arm,stage))
        if stage=='pilot_evaluate':
            candidate=dict(metric_record(arm,.9,.95),arm=arm,epoch=5,checkpoint=arm+'.pt',supervised_selected=True,training_history=[{
                'train_alignment_nhsic':.1,'validation_alignment_nhsic':.1,
                'train_alignment_nhsic_valid_count':1,'validation_alignment_nhsic_valid_count':1}])
            candidate['probes']['stroma']['mse']=.5  # biological failure, no continuation
            (root/'diagnostics'/f'{arm}_pilot.json').write_text(json.dumps([candidate]))
        if stage=='evaluate':
            records[arm]=[dict(metric_record(arm,.16,.23),arm=arm,epoch=5,checkpoint=arm+'.pt',supervised_selected=True)]
    monkeypatch.setattr(refinement,'run_subprocess',stage)
    monkeypatch.setattr(refinement.sweep,'records_for',lambda root,arm:records.get(arm,[]))
    monkeypatch.setattr(refinement,'config_for',lambda root,arm:dict(loss=dict(batch_alignment={n:dict(weight=w) for n,w in zip(refinement.NAMES,refinement.PILOTS[arm])})))
    refinement.run_all(tmp_path)
    assert calls[0]==('both_stronger','smoke')
    assert [a for a,s in calls if s=='train']==list(refinement.FULL_ARMS)
    assert [a for a,s in calls if s=='pilot']==list(refinement.PILOTS)
    assert set(s for _,s in calls)<=set(refinement.STAGES)
    assert json.loads((tmp_path/'decisions/recommendation.json').read_text())['selected']['id']=='combined'


def test_frozen_source_and_configuration_mismatch_rejected(tmp_path):
    source=tmp_path/'source/dann/example.py';source.parent.mkdir(parents=True);source.write_text('original source')
    config=tmp_path/'source_config.yaml';config.write_text('data: {}\n')
    baseline=tmp_path/'baseline.pt';baseline.write_bytes(b'original checkpoint')
    arms=tmp_path/'arms';arms.mkdir()
    arm_config=tmp_path/'arm.yaml';arm_config.write_text('weight: 1\n')
    (arms/'arm.json').write_text(json.dumps(dict(config=str(arm_config),config_sha256=refinement.digest_file(arm_config))))
    manifest=dict(sources={'dann/example.py':refinement.digest_file(source)},
        files={'source_config.yaml':refinement.digest_file(config)},inputs={},
        baseline=dict(checkpoint=str(baseline),sha256=refinement.digest_file(baseline)),dependencies={})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    assert refinement.manifest_for(tmp_path)==manifest
    source.write_text('modified source')
    with pytest.raises(ValueError,match='Frozen source changed'):
        refinement.manifest_for(tmp_path)
    source.write_text('original source')
    arm_config.write_text('weight: 2\n')
    with pytest.raises(ValueError,match='Arm configuration changed'):
        refinement.manifest_for(tmp_path)


def test_completed_training_stage_reuses_without_mutation(tmp_path,monkeypatch):
    (tmp_path/'stages').mkdir();(tmp_path/'manifest.json').write_text('{}')
    c=load_config(Path('dann/config.yaml'))
    c=refinement.no_exports(c)
    monkeypatch.setattr(refinement,'manifest_for',lambda root:{})
    monkeypatch.setattr(refinement,'config_for',lambda root,arm:copy.deepcopy(c))
    artifact=tmp_path/'epoch.pt';artifact.write_bytes(b'existing')
    identity=dict(manifest_sha256=refinement.digest_file(tmp_path/'manifest.json'),arm='soft_nn30',stage='train',config=refinement.fingerprint(c))
    stamp=tmp_path/'stages/soft_nn30-train.json'
    refinement.sweep.complete(stamp,identity,dict(files=[str(artifact)]),time.monotonic())
    before=stamp.read_bytes()
    monkeypatch.setattr(refinement,'train_model',lambda *a,**k:pytest.fail('retrained completed stage'))
    refinement.execute_stage(tmp_path,'soft_nn30','train')
    assert stamp.read_bytes()==before and artifact.read_bytes()==b'existing'

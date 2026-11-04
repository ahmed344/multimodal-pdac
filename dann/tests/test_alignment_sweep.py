"""Selection limits, matched geometry, immutable completion, archives and restart."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from dann import train
from dann.config import load_config, resolve_execution
from dann.tests.test_components import spatial_config
from dann.tests.test_training_artifacts import assert_nested_equal
from dann.verification.alignment_study_metrics import (
    balanced_rows, choose, eligibility, geometry, improves, variance_spectrum,
)
from dann.verification.alignment_sweep import complete, reusable, make_config, decision


def metric_record(name='reference', latent=.30, umap=.80):
    return dict(id=name, spectrum=dict(noncollapsed=True),
        mixing=dict(latent=dict(excess=latent),umap=dict(excess=umap)),
        probes={**{n:dict(mse=.1) for n in ('stroma','collagen','tumor')},
                **{n:dict(brier=.15) for n in ('presence','exceedance_0','exceedance_1','exceedance_2')}},
        distributions=dict(cd8=dict(loss=-.5,presence=dict(brier=.15),positive_pit_ks=.1,
            exceedance={str(t):dict(brier=.1) for t in (.1,.2,.3)},
            intervals={'0.5':dict(coverage=.48),'0.9':dict(coverage=.88)})))


def test_matched_sampling_is_balanced_order_invariant_and_requires_coverage():
    rows = np.arange(500); labels = rows % 4
    a = balanced_rows(rows, labels, 64, 123, required=64)
    b = balanced_rows(rows[::-1], labels[::-1], 64, 123, required=64)
    np.testing.assert_array_equal(a,b)
    assert np.unique(labels[a],return_counts=True)[1].tolist() == [64]*4
    assert len(np.unique(a)) == len(a)
    assert len(balanced_rows(rows,labels,512,123)) == 500
    with pytest.raises(ValueError,match='at least 128'):
        balanced_rows(rows,labels,128,123,required=128)
    with pytest.raises(ValueError,match='unique'):
        balanced_rows(np.zeros(500),labels,64,123)


def test_exact_neighbors_exclude_self_and_correct_leave_one_out_chance():
    # Duplicate coordinates force deterministic ties, including a self at index 0.
    x = np.array([[1.,0.],[1.,0.],[1.,.001],[0.,1.]])
    labels = np.array([0,1,1,0])
    summary, rows = geometry(x,labels,neighbors=1)
    assert rows.same_slide.tolist() == [0.,0.,0.,0.]
    assert summary['chance'] == pytest.approx(1/3)
    assert summary['excess'] == pytest.approx(-1/3)
    # k=n-1 must equal chance, even with unequal slide sizes.
    summary,_ = geometry(x,np.array([0,0,0,1]),neighbors=3,metric='euclidean')
    assert summary['same_slide'] == pytest.approx(.5)
    assert summary['excess'] == pytest.approx(0.)


@pytest.mark.parametrize('x', [np.ones((40,3)),np.zeros((40,3)),np.full((40,3),np.nan),np.ones((10,3))])
def test_invalid_geometry_never_looks_mixed(x):
    with pytest.raises(ValueError,match='Invalid geometry'):
        geometry(x,np.arange(len(x))%2)
    assert not variance_spectrum(x)['noncollapsed']


def test_zero_norm_and_rank_one_representations_are_rejected():
    x = np.random.default_rng(1).normal(size=(100,8)); x[0] = 0
    with pytest.raises(ValueError,match='zero latent norm'):
        geometry(x,np.arange(100)%2)
    assert not variance_spectrum(x)['noncollapsed']
    assert not variance_spectrum(np.arange(1,101)[:,None]*np.ones((1,8)))['noncollapsed']
    assert variance_spectrum(np.random.default_rng(2).normal(size=(100,8)))['noncollapsed']


@pytest.mark.parametrize('kind', ['mse','probe_brier','loss','presence','exceedance','pit','coverage50','coverage90'])
def test_inclusive_selection_limit_boundaries(kind):
    reference = metric_record(); candidate = copy.deepcopy(reference)
    if kind == 'mse':
        obj,key,bound = candidate['probes']['stroma'],'mse',.11
    elif kind == 'probe_brier':
        obj,key,bound = candidate['probes']['exceedance_2'],'brier',.17
    elif kind == 'loss':
        obj,key,bound = candidate['distributions']['cd8'],'loss',-.45
    elif kind == 'presence':
        obj,key,bound = candidate['distributions']['cd8']['presence'],'brier',.17
    elif kind == 'exceedance':
        obj,key,bound = candidate['distributions']['cd8']['exceedance']['0.3'],'brier',.12
    elif kind == 'pit':
        obj,key,bound = candidate['distributions']['cd8'],'positive_pit_ks',.14
    else:
        level = '0.5' if kind == 'coverage50' else '0.9'
        obj,key,bound = candidate['distributions']['cd8']['intervals'][level],'coverage',float(level)+.06
    obj[key] = bound
    assert eligibility(candidate,reference)['eligible']
    obj[key] += 1e-8
    assert not eligibility(candidate,reference)['eligible']
    obj[key] = float('nan')
    assert not eligibility(candidate,reference)['eligible']


def test_selection_requires_both_improvements_and_handles_near_ties():
    ref = metric_record()
    a = metric_record('a',.1,.5); b = metric_record('b',.11,.4)
    assert choose([a,b],ref)['id'] == 'b'
    b['mixing']['latent']['excess'] += 1e-7
    assert choose([a,b],ref)['id'] == 'a'
    a['mixing']['umap']['excess'] = .80
    assert not improves(a,ref)
    assert choose([a],ref) is None
    a['spectrum']['noncollapsed'] = False; a['mixing'] = {}
    assert not eligibility(a,ref)['eligible'] and not improves(a,ref)
    base = metric_record('base',.05,.3)
    assert choose([b],ref,base=base) is None


def test_stale_completion_and_changed_adaptive_decisions_fail_closed(tmp_path):
    artifact = tmp_path/'metrics.json'; artifact.write_text('{}')
    stamp = tmp_path/'complete.json'; identity = {'checkpoint':'a'}
    assert not reusable(stamp,identity)
    stamp.write_text(json.dumps(dict(status='running',identity=identity)))
    assert not reusable(stamp,identity)
    complete(stamp,identity,dict(files=[str(artifact)]),time.monotonic())
    assert reusable(stamp,identity)
    with pytest.raises(ValueError,match='identity changed'):
        reusable(stamp,{'checkpoint':'b'})
    artifact.write_text('{"corrupted":true}')
    with pytest.raises(ValueError,match='artifact changed'):
        reusable(stamp,identity)
    (tmp_path/'decisions').mkdir()
    decision(tmp_path,'base',{'weight':1})
    decision(tmp_path,'base',{'weight':1})
    with pytest.raises(ValueError,match='adaptive decision changed'):
        decision(tmp_path,'base',{'weight':10})


@pytest.mark.parametrize('value', [0,-1,True,1.5,'5'])
def test_archive_interval_validation(value):
    c = load_config(Path('dann/config.yaml'))
    c['training']['archive_checkpoint_every'] = value
    with pytest.raises(ValueError,match='archive_checkpoint_every'):
        resolve_execution(c)


def test_study_configs_do_not_change_optimizer_architecture_or_sampling(tmp_path):
    c = load_config(Path('dann/config.yaml'))
    assert c['training']['archive_checkpoint_every'] is None
    study = make_config(c,tmp_path,'nhsic10',(10.,0.,0.))
    assert study['model'] == c['model'] and study['data'] == c['data']
    assert study['training']['pixel'] == c['training']['pixel']
    assert study['training']['head_initialization'] == c['training']['head_initialization']
    assert study['training']['archive_checkpoint_every'] == 5
    assert all(study['loss']['batch_alignment'][n]['enabled'] for n in ('nhsic','anova','soft_nn'))
    assert c['loss']['batch_alignment']['nhsic']['weight'] == .1


@pytest.mark.parametrize('asynchronous', [False,True])
def test_epoch_archives_and_exact_interrupted_resume(spatial_config,monkeypatch,tmp_path,asynchronous):
    c = spatial_config
    c['model']['aggregation']['type'] = 'mlp'
    c['model']['heads']['biology']['type'] = 'mlp'
    c['model']['heads']['discriminator']['type'] = 'mlp'
    c['model']['dropout'] = .2
    c['training'].update(epochs=3,early_stopping_enabled=False,archive_checkpoint_every=1)
    c['training']['pixel'].update(batch_size=32,num_workers=2)
    c['performance'].update(ram_cache=True,preparation_workers=1,async_artifacts=asynchronous)
    c['analysis']['figure_dpi'] = 12
    resolve_execution(c)
    monkeypatch.setattr(train,'refresh_loss_curves',lambda *a,**k:None)
    c['training']['output_dir'] = str(tmp_path/'continuous')
    train.train_model(c)
    reference = torch.load(tmp_path/'continuous/latest.pt',weights_only=False)
    c['training']['output_dir'] = str(tmp_path/'interrupted')
    original = train.run_epoch
    def interrupt(**kwargs):
        if kwargs['optimizer'] is not None and kwargs['epoch_index'] == 1:
            raise RuntimeError('deliberate epoch interruption')
        return original(**kwargs)
    monkeypatch.setattr(train,'run_epoch',interrupt)
    with pytest.raises(RuntimeError,match='deliberate epoch interruption'):
        train.train_model(c)
    latest = tmp_path/'interrupted/latest.pt'
    assert torch.load(latest,weights_only=False)['epoch'] == 0
    monkeypatch.setattr(train,'run_epoch',original)
    c['training']['resume_checkpoint'] = str(latest)
    train.train_model(c)
    resumed = torch.load(latest,weights_only=False)
    for key in ('model_state','optimizer_state','training_rng_state','early_stopping_state'):
        assert_nested_equal(reference[key],resumed[key])
    for epoch in range(1,4):
        name = f'archives/epoch_{epoch:04d}.pt'
        a = torch.load(tmp_path/'continuous'/name,weights_only=False)
        b = torch.load(tmp_path/'interrupted'/name,weights_only=False)
        assert a['epoch'] == b['epoch'] == epoch-1
        assert_nested_equal(a['model_state'],b['model_state'])
    assert (tmp_path/'interrupted/best.pt').is_file()


@pytest.mark.parametrize('helpful_soft,expected_fits',[(False,8),(True,9)])
def test_adaptive_run_order_and_conditional_combination(tmp_path,monkeypatch,helpful_soft,expected_fits):
    from dann.verification import alignment_sweep as sweep
    for name in ('logs','decisions'):
        (tmp_path/name).mkdir()
    reference = dict(metric_record(),arm='reference',epoch=29,checkpoint='reference.pt',supervised_selected=True)
    weights = dict(sweep.INITIAL_ARMS)
    def config(root,arm):
        w = weights.get(arm,(.1,0.,0.))
        return dict(loss=dict(batch_alignment={name:dict(weight=value) for name,value in zip(sweep.NAMES,w)}))
    def register(root,arm,w):
        weights[arm] = w
    def records(root,arm):
        if arm=='reference':
            return [reference]
        latent,umap = {'control':(.4,.9),'nhsic1':(.15,.5),'nhsic10':(.2,.6),'nhsic100':(.25,.7),
                      'anova0p1':(.13,.45),'anova1':(.12,.4),'soft_nn1':(.13,.43),
                      'soft_nn10':(.12,.39),'combined':(.08,.3)}[arm]
        if arm.startswith('soft_nn') and not helpful_soft:
            latent,umap = .16,.51
        return [dict(metric_record(arm,latent,umap),arm=arm,epoch=10,checkpoint=f'{arm}.pt',supervised_selected=True)]
    calls=[]
    monkeypatch.setattr(sweep,'manifest_for',lambda root:{})
    monkeypatch.setattr(sweep,'report',lambda *a,**k:None)
    monkeypatch.setattr(sweep,'config_for',config)
    monkeypatch.setattr(sweep,'register_arm',register)
    monkeypatch.setattr(sweep,'records_for',records)
    monkeypatch.setattr(sweep,'run_subprocess',lambda root,arm,stage:calls.append((arm,stage)))
    monkeypatch.setattr(sweep.subprocess,'run',lambda *a,**k:None)
    sweep.run_all(tmp_path)
    assert calls[:2] == [('control','smoke'),('reference','evaluate')]
    fits=[arm for arm,stage in calls if stage=='train']
    assert fits[:4] == list(sweep.INITIAL_ARMS) and len(fits)==expected_fits
    base=json.loads((tmp_path/'decisions/nhsic_base.json').read_text())
    assert base['selected']['arm']=='nhsic1' and base['weight']==1
    assert weights['anova1']==(1,1.,0.)
    assert ('combined' in fits)==helpful_soft
    selected=json.loads((tmp_path/'decisions/recommendation.json').read_text())['selected']['arm']
    assert selected == ('combined' if helpful_soft else 'anova1')


def test_cosine_direction_collapse_and_single_slide_have_no_mixing_score():
    values = np.arange(1,41)[:,None]*np.ones((1,8))
    with pytest.raises(ValueError,match='collapsed cosine directions'):
        geometry(values,np.arange(40)%2)
    with pytest.raises(ValueError,match='fewer than two slides'):
        geometry(np.random.default_rng(1).normal(size=(40,8)),np.zeros(40))


@pytest.mark.parametrize('error,excluded',[(FloatingPointError('nonfinite objective'),True),
    (ValueError('Enabled alignment objective nhsic has no valid evaluations in epoch'),True),
    (OSError('disk full'),False)])
def test_numerical_exclusion_is_terminal_but_infrastructure_errors_are_not(tmp_path,monkeypatch,error,excluded):
    from dann.verification import alignment_sweep as sweep
    for name in ('stages','diagnostics'):
        (tmp_path/name).mkdir()
    (tmp_path/'manifest.json').write_text('{}')
    monkeypatch.setattr(sweep,'manifest_for',lambda root:{})
    monkeypatch.setattr(sweep,'config_for',lambda root,arm:dict(training=dict(output_dir=str(tmp_path/'model'))))
    calls=[]
    def fail(*a,**k):
        calls.append(1)
        raise error
    monkeypatch.setattr(sweep,'train_model',fail)
    if excluded:
        sweep.execute_stage(tmp_path,'nhsic100','train')
        sweep.execute_stage(tmp_path,'nhsic100','train')
        assert calls==[1] and sweep.records_for(tmp_path,'nhsic100')==[]
        assert json.loads((tmp_path/'stages/nhsic100-train.json').read_text())['result']['excluded']
    else:
        with pytest.raises(OSError,match='disk full'):
            sweep.execute_stage(tmp_path,'nhsic100','train')
        assert json.loads((tmp_path/'stages/nhsic100-train.json').read_text())['status']=='failed'


def test_recommendation_restart_reuses_validated_inference_in_isolated_tree(tmp_path,monkeypatch):
    from dann.verification import alignment_sweep as sweep
    from dann import workflow
    (tmp_path/'stages').mkdir()
    (tmp_path/'manifest.json').write_text('{}')
    checkpoint = tmp_path/'chosen.pt'; torch.save({},checkpoint)
    config = load_config(Path('dann/config.yaml'))
    monkeypatch.setattr(sweep,'manifest_for',lambda root:{})
    monkeypatch.setattr(sweep,'config_for',lambda root,arm:copy.deepcopy(config))
    marker=tmp_path/'recommendation/export.txt'
    calls=[]
    def validate(c,*a,**k):
        assert Path(c['spatial']['output']).is_relative_to(tmp_path/'recommendation')
        if not marker.exists():
            raise ValueError('No completed export')
    def run(c,path):
        assert Path(c['analysis']['output_dir']).is_relative_to(tmp_path/'recommendation')
        calls.append(c['post_training']['run_inference'])
        if len(calls)==1:
            marker.parent.mkdir();marker.write_text('complete export')
            raise RuntimeError('render interrupted')
    monkeypatch.setattr(workflow,'_validate_exports',validate)
    monkeypatch.setattr(workflow,'run_post_training_workflow',run)
    selected=dict(checkpoint=str(checkpoint),arm='nhsic1')
    with pytest.raises(RuntimeError,match='render interrupted'):
        sweep.recommendation_workflow(tmp_path,selected)
    sweep.recommendation_workflow(tmp_path,selected)
    sweep.recommendation_workflow(tmp_path,selected)
    assert calls == [True,False]
    assert config['analysis']['output_dir'] != str(tmp_path/'recommendation/analysis')


def test_smoke_fixtures_preserve_frozen_values_without_false_cohort_provenance(tmp_path):
    import anndata as ad
    import pandas as pd
    from scipy.sparse import csr_matrix
    from dann.analysis_execution import digest_file
    from dann.verification.alignment_sweep import smoke_inputs
    source=tmp_path/'original.h5ad'
    matrix=csr_matrix(np.arange(32,dtype=np.float32).reshape(8,4))
    he=np.arange(24,dtype=np.float32).reshape(8,3)
    original=ad.AnnData(matrix,obs=pd.DataFrame({'batch':['a']*4+['b']*4},index=[f'r{i}' for i in range(8)]),
        obsm={'he':he,'spatial':np.arange(16).reshape(8,2)},uns={'msi_preprocess':{'manifest_json':'original exact-cohort claim'},
        'spatial':{slide:{'images':{'HES':np.ones((8,8,3),dtype=np.uint8)}} for slide in ('a','b')},'unrelated':np.ones(100)})
    original.write_h5ad(source)
    before=digest_file(source)
    c=dict(data=dict(path=str(source),context_path=str(source),batch_column='batch',he_features_key='he',preprocessing_fingerprint='frozen'))
    smoke_inputs(tmp_path,c)
    assert c['data']['preprocessing_fingerprint'] is None
    assert digest_file(source)==before
    fixture=ad.read_h5ad(tmp_path/'smoke_data/path.h5ad')
    np.testing.assert_array_equal(fixture.X.toarray(),matrix.toarray())
    np.testing.assert_array_equal(fixture.obsm['he'],he)
    np.testing.assert_array_equal(fixture.obsm['spatial'],original.obsm['spatial'])
    assert set(fixture.uns['spatial']) == {'a','b'}
    assert 'msi_preprocess' not in fixture.uns and 'unrelated' not in fixture.uns
    provenance=json.loads((tmp_path/'smoke_data/provenance.json').read_text())
    assert provenance['fixture_only'] and not provenance['preprocessing_refitted']
    assert provenance['original_preprocessing_fingerprint']=='frozen'


def test_dominant_variance_direction_is_not_numerical_collapse():
    x=np.random.default_rng(8).normal(size=(200,8))
    x[:,0]*=100
    spectrum=variance_spectrum(x)
    assert spectrum['noncollapsed']
    assert spectrum['raw']['participation_ratio'] < 2
    assert spectrum['raw']['numerical_rank'] == 8

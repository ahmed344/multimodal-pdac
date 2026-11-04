"""Independent references and contract tests for direct slide alignment."""
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import softmax

from dann.batch_alignment import (NAMES, BANDWIDTHS, AlignmentSummary, ValidationSample,
    alignment_settings, evaluate_alignment, sample_rows)
from dann.tests.test_components import spatial_config


def reference(x, b):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    unique, inverse, counts = np.unique(b, return_inverse=True, return_counts=True)
    w = 1 / (len(unique) * counts[inverse])
    h = np.eye(len(x)) - np.ones((len(x), 1)) @ w[None, :]
    dot = np.clip(x @ x.T, -1, 1)
    k = np.mean([np.exp(-(1-dot)/s) for s in BANDWIDTHS], axis=0)
    labels = (b[:, None] == b).astype(float)
    a, c = h @ k @ h.T, h @ labels @ h.T
    pair = np.outer(w, w)
    numerator = np.sum(pair*a*c)
    anorm, cnorm = np.sqrt(np.sum(pair*a*a)), np.sqrt(np.sum(pair*c*c))
    mean = w @ x
    centroids = np.array([x[b==v].mean(0) for v in unique])
    anova = np.mean(np.sum((centroids-mean)**2, axis=1)) / np.sum(w[:, None]*(x-mean)**2)
    mask = ~np.eye(len(x), dtype=bool)
    off = pair*mask / np.sum(pair*mask)
    average = np.sum(off*dot)
    std = np.sqrt(np.sum(off*(dot-average)**2))
    logits = (dot-average)/std/.1 + np.log(w)[None, :]
    logits[~mask] = -np.inf
    probs = softmax(logits, axis=1)
    chance = np.sum((labels*mask)*w[None, :], axis=1) / np.sum(mask*w[None, :], axis=1)
    snn = np.sum(w*np.maximum(np.sum(probs*labels, axis=1)-chance, 0)**2)
    return dict(nhsic=numerator/(anorm*cnorm), anova=anova, soft_nn=snn), (numerator, anorm, cnorm)


@pytest.mark.parametrize('counts', [(4, 4, 4), (2, 5, 9)])
def test_independent_numpy_reference_and_permutations(counts):
    rng = np.random.default_rng(30)
    x = rng.normal(size=(sum(counts), 5)); labels = np.repeat(np.arange(len(counts)), counts)
    expected, terms = reference(x, labels)
    result = evaluate_alignment(torch.tensor(x), torch.tensor(labels))
    for name in NAMES:
        assert float(result[name].score) == pytest.approx(expected[name], abs=1e-12)
    for key, value in zip(['numerator', 'latent_kernel_norm', 'label_kernel_norm'], terms):
        assert float(result['nhsic'].diagnostics[key]) == pytest.approx(value, abs=1e-12)
    perm = rng.permutation(len(x))
    transformed = evaluate_alignment(torch.tensor(x[perm]), torch.tensor((labels[perm]+1)*7))
    for name in NAMES:
        torch.testing.assert_close(result[name].score, transformed[name].score)


@pytest.mark.parametrize('name', NAMES)
def test_autograd(name):
    x = torch.randn(8, 3, generator=torch.Generator().manual_seed(7), dtype=torch.double, requires_grad=True)
    b = torch.tensor([0]*3+[1]*5)
    assert torch.autograd.gradcheck(lambda value: evaluate_alignment(value, b, (name,))[name].score, (x,), atol=1e-5)


def test_weighted_duplication_invariance():
    x = torch.randn(7, 4, dtype=torch.double, generator=torch.Generator().manual_seed(1))
    b = torch.tensor([0,0,0,1,1,1,1])
    indices = torch.tensor([0,1,2,0,1,2,0,1,2,3,4,5,6])
    a, c = evaluate_alignment(x,b), evaluate_alignment(x[indices],b[indices])
    for name in ('nhsic','anova'):
        torch.testing.assert_close(a[name].score,c[name].score)


@pytest.mark.parametrize('kind,reason', [('zero','near_zero_latent_norm'),('constant','degenerate_latent_variance'),
    ('one_slide','insufficient_slides'),('singletons','insufficient_slides'),('nonfinite','nonfinite_latents')])
def test_degeneracy_never_reports_zero_success(kind, reason):
    x = torch.randn(6,4); b = torch.tensor([0,0,0,1,1,1])
    if kind=='zero': x[0].zero_()
    if kind=='constant': x[:] = 1.
    if kind=='one_slide': b.zero_()
    if kind=='singletons': b = torch.arange(6)
    if kind=='nonfinite': x[0,0] = math.nan
    values = evaluate_alignment(x,b)
    assert all(m.score is None and m.reason == reason for m in values.values())
    s = alignment_settings({'loss': {'batch_alignment': {'nhsic': {'enabled': True, 'weight': 0.}}}})
    summary = AlignmentSummary(s); summary.add({'nhsic': values['nhsic']})
    with pytest.raises(ValueError, match='no valid evaluations'):
        summary.finish()


def test_singleton_exclusion_and_soft_nn_self_exclusion_below_chance():
    # The nearest candidate is always on the other slide. Below-chance mixing has no penalty.
    x = torch.tensor([[1.,0.],[-1.,0.],[1.,.001],[-1.,.001]], dtype=torch.double)
    b = torch.tensor([0,0,1,1])
    result = evaluate_alignment(x,b)
    assert float(result['soft_nn'].score) < 1e-12
    assert float(result['soft_nn'].diagnostics['chance_probability']) == pytest.approx(1/3)
    singleton = evaluate_alignment(torch.cat([x, x[:1]]), torch.tensor([0,0,1,1,2]))
    for name in NAMES:
        torch.testing.assert_close(result[name].score, singleton[name].score)
        assert singleton[name].diagnostics['excluded_singleton_rows'] == 1
    # All off-diagonal similarities identical: undefined Soft-NN standardization.
    m = evaluate_alignment(torch.eye(4), b)['soft_nn']
    assert m.score is None and m.reason == 'degenerate_similarity_variance'


def test_independence_is_positive_finite_sample_baseline_and_biology_can_be_deleted():
    generator = torch.Generator().manual_seed(15)
    x = torch.randn(128, 10, generator=generator, dtype=torch.double)
    b = torch.arange(4).repeat_interleave(32)
    score = evaluate_alignment(x,b)['nhsic'].score
    assert 0 < score < .2  # Biased finite-sample statistic, not exact zero at independence.
    biology = torch.cat([3*torch.nn.functional.one_hot(b,4).double(), x],1)
    assert evaluate_alignment(biology,b)['nhsic'].score > score
    # Deleting genuinely slide-associated biology improves the objective.


def test_sampling_is_order_independent_rng_isolated_and_capped():
    rows = np.arange(125); b = np.repeat([0,1,2], [60,64,1])
    state = np.random.get_state(); torch_state = torch.random.get_rng_state().clone()
    a = sample_rows(rows,b,32,21,4)
    c = sample_rows(rows[::-1],b[::-1],32,21,4)
    np.testing.assert_array_equal(a,c)
    assert len(a)==64 and 124 not in a
    assert not np.array_equal(a,sample_rows(rows,b,32,21,5))
    assert all(np.array_equal(x,y) for x,y in zip(state,np.random.get_state()))
    assert torch.equal(torch_state,torch.random.get_rng_state())


def test_validation_sample_batching_and_order_invariance():
    rows = torch.arange(60); labels = rows % 3
    x = torch.randn(60,5,generator=torch.Generator().manual_seed(2))
    selected = sample_rows(rows.numpy(),labels.numpy(),12,9)
    scores=[]
    for size,order in [(1, rows),(11, rows.flip(0)),(60,rows)]:
        collector=ValidationSample(selected)
        for chunk in order.split(size): collector.add(rows[chunk],labels[chunk],x[chunk])
        xx,bb=collector.tensors();scores.append(evaluate_alignment(xx,bb))
    for name in NAMES:
        for result in scores[1:]: torch.testing.assert_close(scores[0][name].score,result[name].score)


@pytest.mark.parametrize('kind', ['mlp','cnn','gatv2'])
def test_discriminator_free_forward_and_checkpoint(spatial_config,kind,tmp_path):
    from dann.config import resolve_execution
    from dann.model import AdversarialLatentFusion
    from dann.data_loader import create_data_bundle
    from dann.checkpoints import architecture_contract, validate_checkpoint, checkpoint_config
    c=spatial_config
    c['model']['heads']['discriminator'].update(enabled=False,type='gatv2')
    c['model']['aggregation']['type']=kind;c['model']['heads']['biology']['type']=kind
    resolve_execution(c)
    assert 'disc-none' in c['execution']['combination']
    if kind=='mlp': assert c['execution']['mode']=='pixel' and c['execution']['halo']==0
    bundle=create_data_bundle(c);model=AdversarialLatentFusion.from_config(c,2,4)
    out=model(next(iter(bundle.loaders['train'])))
    assert 'batch_logits' not in out and model.gradient_reversal is None
    assert not any('discriminator' in n for n,_ in model.named_parameters())
    out['mu'].sum().backward()
    payload={'config':c,'architecture':architecture_contract(c),'model_state':model.state_dict()}
    path=tmp_path/'headless.pt';torch.save(payload,path);saved=torch.load(path,weights_only=False)
    validate_checkpoint(saved,c,model)
    restored=AdversarialLatentFusion.from_config(checkpoint_config(saved),2,4);restored.load_state_dict(saved['model_state'])
    changed=copy.deepcopy(c);changed['model']['heads']['discriminator']['enabled']=True
    with pytest.raises(ValueError,match='fresh training'): validate_checkpoint(saved,changed)
    for dataset in bundle.datasets.values(): dataset.close()


def test_archive_identity_join_and_rejections():
    from dann.verification.batch_alignment_comparison import matched_metadata
    table=pd.DataFrame(dict(row_id=range(8),batch=['a']*4+['b']*4,batch_id=[0]*4+[1]*4,split='validation')).set_index('row_id')
    assert matched_metadata(table,table.iloc[::-1],2,4).equals(table)
    bad=table.copy();bad.loc[0,'batch']='b'
    with pytest.raises(ValueError,match='disagree'):matched_metadata(table,bad,2,4)
    with pytest.raises(ValueError,match='Require'):matched_metadata(table,table,43,64)


def alignment_epoch_fixture():
    from dann.tests.test_monitoring import SharedModel
    from dann.tests.test_objectives import Batches, TARGETS
    from dann.losses import ZILNLoss
    class Headless(SharedModel):
        def __init__(self):
            super().__init__(he=False)
            self.batch_discriminator = None
            self.encoder[1] = torch.nn.Identity()
        def forward(self, batch, **kwargs):
            latent = self.encoder(batch['he_targets'])
            pi, mu, raw = self.biology_predictor(latent).chunk(3, dim=1)
            return dict(latent=latent, pi_logits=pi, mu=mu, sigma=torch.nn.functional.softplus(raw)+.1)
    torch.manual_seed(15)
    model=Headless().double()
    b=dict(row_ids=torch.arange(12), batches=torch.arange(12)%3,
           targets=torch.rand(12,4,dtype=torch.double)*.7,
           he_targets=torch.randn(12,2,dtype=torch.double))
    b['targets'][::3,0]=0
    kwargs=dict(loader=Batches([b]),ziln_loss=ZILNLoss([1]*4),device=torch.device('cpu'),biology_weight=2.,
                batch_weight=1.,class_weights=None,epoch_index=0,total_epochs=1,grl_schedule='constant',
                grl_max_lambda=.25,grl_gamma=10.,gradient_clip_norm=None,target_columns=TARGETS,
                monitoring=dict(enabled=True),alignment_seed=11,
                batch_alignment=dict(nhsic=dict(enabled=True,weight=.1),anova=dict(enabled=True,weight=0.)))
    return model,b,kwargs


def test_epoch_loss_gradient_accounting_diagnostics_and_disabled_components():
    from dann.train import run_epoch
    model,b,kwargs=alignment_epoch_fixture();reference=copy.deepcopy(model)
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    expected_optimizer=torch.optim.SGD(reference.parameters(),lr=.01)
    out=reference(b);bio=kwargs['ziln_loss'](out['pi_logits'],out['mu'],out['sigma'],b['targets']).total
    measurement=evaluate_alignment(out['latent'],b['batches'])
    expected=2*bio+.1*measurement['nhsic'].score
    expected.backward();expected_optimizer.step()
    metrics=run_epoch(model=model,optimizer=optimizer,**kwargs)
    assert metrics['total_loss']==pytest.approx(float(expected.detach()))
    assert metrics['alignment_anova']==pytest.approx(float(measurement['anova'].score.detach()))
    assert metrics['alignment_anova_weighted_loss']==0
    assert not any('batch_' in k or 'grl' in k or 'soft_nn' in k for k in metrics)
    assert not any('gradient_overall_anova' in k for k in metrics)
    assert metrics['gradient_overall_nhsic_norm_mean']>0
    assert metrics['gradient_overall_ihc_share_mean']+metrics['gradient_overall_nhsic_share_mean']==pytest.approx(1)
    for a,c in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a,c)


def test_validation_epoch_is_independent_of_batching_and_tile_order():
    from dann.train import run_epoch
    from dann.tests.test_objectives import Batches
    model,b,kwargs=alignment_epoch_fixture()
    kwargs['validation_alignment_rows']=b['row_ids'].tolist()
    results=[]
    for size, order in [(1,torch.arange(12)),(5,torch.arange(12).flip(0)),(12,torch.arange(12))]:
        kwargs['loader']=Batches({k:v[indices] for k,v in b.items()} for indices in order.split(size))
        results.append(run_epoch(model=model,optimizer=None,**kwargs))
    for key in ['alignment_nhsic','alignment_nhsic_numerator','alignment_anova','total_loss']:
        assert results[0][key]==pytest.approx(results[1][key],abs=1e-10)
        assert results[0][key]==pytest.approx(results[2][key],abs=1e-10)


def test_split_optimizer_batches_rejected_and_invalid_diagnostics_fail_epoch():
    from dann.train import run_epoch
    model,b,kwargs=alignment_epoch_fixture()
    b['accumulation']=dict(index=0,count=2)
    with pytest.raises(ValueError,match='unsplit optimizer'):
        run_epoch(model=model,optimizer=torch.optim.SGD(model.parameters(),lr=.01),**kwargs)
    b.pop('accumulation'); b['batches'].zero_()
    with pytest.raises(ValueError,match='no valid evaluations'):
        run_epoch(model=model,optimizer=torch.optim.SGD(model.parameters(),lr=.01),**kwargs)


def test_alignment_contract_changes_require_fresh_training(spatial_config):
    from dann.train import resume_training_config
    c=spatial_config
    changed=copy.deepcopy(c);changed['loss']['batch_alignment']=dict(nhsic=dict(enabled=True,weight=.1))
    with pytest.raises(ValueError,match='fresh training'):resume_training_config(c,changed)
    changed=copy.deepcopy(c);changed['model']['heads']['discriminator']['enabled']=False
    with pytest.raises(ValueError,match='fresh training'):resume_training_config(c,changed)


@pytest.mark.parametrize('diagnostics',[False,True])
def test_alignment_plot_layout_legends_and_memory(tmp_path,monkeypatch,diagnostics):
    import matplotlib.pyplot as plt
    from dann.objective_curves import plot_objective_history, objective_layout
    from dann.curve_layout import figure_size,render_reservation_bytes
    c=dict(model=dict(heads=dict(discriminator=dict(enabled=False))),loss=dict(
        batch_alignment=dict(nhsic=dict(enabled=True,weight=.1),anova=dict(enabled=diagnostics,weight=0.))))
    row=dict(epoch=1, train_total_loss=2.,train_alignment_nhsic=.2,validation_alignment_nhsic=.15,
             train_alignment_nhsic_weighted_loss=.02,train_alignment_nhsic_numerator=.001,
             train_alignment_nhsic_denominator=.01,train_alignment_anova=.3,
             train_batch_loss=2.,train_gradient_grl_strength_mean=.2,
             train_gradient_overall_ihc_norm_mean=1.,train_gradient_overall_nhsic_norm_mean=.1,
             train_gradient_overall_ihc_share_mean=.9,train_gradient_overall_nhsic_share_mean=.1)
    frame=pd.DataFrame([row]);captured=[]
    monkeypatch.setattr(plt,'close',lambda f:captured.append(f))
    plot_objective_history(frame,tmp_path/'loss.png',12,config=c)
    f=captured[-1];titles=' '.join(a.get_title() for a in f.axes)
    labels=[t.get_text() for legend in f.legends for t in legend.get_texts()]
    assert 'Adversarial' not in labels and 'Soft-NN' not in labels
    assert ('Cosine ANOVA' in labels)==diagnostics
    assert 'GRL' not in titles and 'discriminator' not in titles.lower()
    layout=objective_layout(frame,c)
    np.testing.assert_equal(f.get_size_inches(),figure_size('loss',len(layout[0])))
    assert render_reservation_bytes(12,{'loss':len(layout[0]),'ihc':len(layout[1])}) >= np.prod(f.get_size_inches())*12**2*16
    for f in captured: plt.Figure.clear(f)


@pytest.mark.parametrize('settings', [None, [], {'typo': True}, {'nhsic': {'enabled':1}},
    {'nhsic': {'weight':math.nan}}, {'nhsic':{'weight':-1}}, {'bandwidths':None},
    {'bandwidths':[]}, {'bandwidths':[0.]}, {'temperature':0.}, {'train_per_slide':33},
    {'validation_per_slide':65}, {'train_per_slide':1}])
def test_settings_validation(settings):
    with pytest.raises(ValueError):alignment_settings({'loss':{'batch_alignment':settings}})


def test_kernel_denominator_degenerate_and_disabled_work_skipped():
    x=torch.randn(8,4,dtype=torch.double);b=torch.arange(8)%2
    assert evaluate_alignment(x,b,names=())=={}
    result=evaluate_alignment(x,b,('nhsic',),bandwidths=[1e15])['nhsic']
    assert result.score is None and result.reason=='degenerate_kernel_norm'


def test_archive_provenance_and_column_checks(tmp_path):
    import json
    from dann.verification.batch_alignment_comparison import read_archive,file_hash
    (tmp_path/'tables').mkdir();(tmp_path/'latent').mkdir()
    path=tmp_path/'tables'/'latent_embeddings.csv'
    table=pd.DataFrame(dict(row_id=[1,2,3,4],batch=['a','a','b','b'],batch_id=[0,0,1,1],split='validation',
                            latent_0=[1,2,3,4],latent_1=[4,3,2,1]))
    table.to_csv(path,index=False)
    provenance=dict(checkpoint_epoch=44,checkpoint='archived.pt',source_identity=dict(checkpoint='a'*64,identity=dict(row_digest='b'*64)))
    (tmp_path/'analysis_execution.json').write_text(json.dumps(provenance))
    record=tmp_path/'latent'/'latent_run.json'
    record.write_text(json.dumps(dict(source_hashes={str(path):file_hash(path)})))
    meta,info=read_archive(tmp_path)
    assert meta.index.tolist()==[1,2,3,4] and info['epoch']==44
    table.loc[0,'latent_0']=99;table.to_csv(path,index=False)
    with pytest.raises(ValueError,match='checksum'):read_archive(tmp_path)
    table.loc[0,'row_id']=2;table.to_csv(path,index=False)
    record.write_text(json.dumps(dict(source_hashes={str(path):file_hash(path)})))
    with pytest.raises(ValueError,match='unique'):read_archive(tmp_path)
    table=table[['row_id','batch','batch_id','split','latent_1','latent_0']];table.to_csv(path,index=False)
    record.write_text(json.dumps(dict(source_hashes={str(path):file_hash(path)})))
    with pytest.raises(ValueError,match='order'):read_archive(tmp_path)


@pytest.mark.parametrize('key,label',[('validation_he_r2','H&E'),('validation_batch_accuracy','Adversarial')])
def test_legacy_sparse_history_legends_only_include_available_objectives(tmp_path,monkeypatch,key,label):
    import matplotlib.pyplot as plt
    from dann.objective_curves import plot_objective_history
    captured=[];monkeypatch.setattr(plt,'close',lambda f:captured.append(f))
    plot_objective_history(pd.DataFrame([{'epoch':1,key:.5}]),tmp_path/'history.png',12)
    labels=[t.get_text() for legend in captured[-1].legends for t in legend.get_texts()]
    assert labels==['Train','Validation',label]
    assert len(captured[-1].axes)==1
    for f in captured:plt.Figure.clear(f)


def test_alignment_training_resume_retains_sample_and_rejects_tampering(spatial_config,monkeypatch):
    from dann import train
    from dann.config import resolve_execution
    c=spatial_config
    c['model']['heads']['discriminator']['enabled']=False
    c['model']['aggregation']['type']='mlp';c['model']['heads']['biology']['type']='mlp'
    c['loss']['batch_alignment']=dict(nhsic=dict(enabled=True,weight=.1))
    c['training'].update(epochs=1,early_stopping_enabled=False)
    c['training']['pixel']['batch_size']=64
    resolve_execution(c)
    monkeypatch.setattr(train,'refresh_loss_curves',lambda *args,**kwargs:None)
    path=train.train_model(c)
    first=torch.load(path,weights_only=False)
    assert first['objective_contract']['settings']['nhsic']==dict(enabled=True,weight=.1)
    assert len(first['validation_alignment_sample']['row_ids'])>0
    latest=path.with_name('latest.pt')
    c['training'].update(epochs=2,resume_checkpoint=str(latest))
    train.train_model(c)
    second=torch.load(latest,weights_only=False)
    assert first['validation_alignment_sample']==second['validation_alignment_sample']
    assert second['epoch']==1  # Checkpoint epochs remain zero based.
    second['validation_alignment_sample']['row_ids'][0]=-1
    torch.save(second,latest)
    c['training']['epochs']=3
    with pytest.raises(ValueError,match='Validation alignment sample changed'):
        train.train_model(c)

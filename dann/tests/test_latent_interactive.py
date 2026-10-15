"""Numerical contracts and display-only changes in offline latent analysis."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from dann.html_display import format_number
from dann.latent_analysis import slide_correlations, feature_layout, align_exports
from dann.artifact_layout import migrate_latent_variance


@pytest.mark.parametrize('value,kind,expected',[(12,'integer','12'),(100.123456,'mz','100.1235'),
    (1.23456,'float','1.23'),(.123456,'float','0.123'),(.00123456,'float','0.0012'),
    (.0000123456,'float','1.23e-05'),(-0.,'float','0'),(np.nan,'float','Unavailable'),(None,'float','Unavailable')])
def test_number_display(value,kind,expected):
    assert format_number(value,kind)==expected


def test_equal_slide_correlations_reference_and_missing():
    rng=np.random.default_rng(8)
    x=rng.normal(size=(90,4));x[:,3]=1;x[0,0]=np.nan
    y=rng.normal(size=(90,2));y[:30,1]=np.nan
    slides=np.array(['a']*30+['b']*60)
    result=slide_correlations(x,y,slides)
    for i in range(4):
        for j in range(4):
            reference=[]
            for slide in ['a','b']:
                valid=(slides==slide)&np.isfinite(x[:,i])&np.isfinite(x[:,j])
                if np.ptp(x[valid,i]) and np.ptp(x[valid,j]):reference.append(spearmanr(x[valid,i],x[valid,j]).statistic)
            np.testing.assert_allclose(result['correlation'][i,j],np.mean(reference) if reference else np.nan,equal_nan=True,atol=1e-14)
    assert result['target_counts'][0,1]==1
    assert result['contributing'][3,0]==0
    expected=np.mean([spearmanr(x[(slides==s)&np.isfinite(x[:,0]),0],y[(slides==s)&np.isfinite(x[:,0]),0]).statistic for s in ['a','b']])
    np.testing.assert_allclose(result['association'][0,0],expected,atol=1e-14)


def test_empty_singleton_insufficient_and_tied_features():
    settings=dict(umap_neighbors=30,umap_min_dist=.3,umap_random_state=1)
    empty=slide_correlations(np.empty((0,2)),np.empty((0,1)),np.array([],dtype=str))
    assert np.isnan(empty['correlation']).all()
    assert np.isnan(feature_layout(empty['correlation'],settings)['coordinates']).all()
    layout=feature_layout(np.ones((1,1)),settings)
    np.testing.assert_array_equal(layout['coordinates'],[[0,0]])
    tied=np.repeat(np.arange(10),3)[:,None]
    r=slide_correlations(np.c_[tied,-tied],tied,np.array(['a']*30))
    np.testing.assert_allclose(r['correlation'],[[1,-1],[-1,1]])
    assert np.isnan(slide_correlations(tied[:10],tied[:10],np.array(['a']*10))['correlation']).all()


def test_alignment_uses_ids_not_position():
    latent=pd.DataFrame(dict(row_id=[1,2],split=['test']*2,batch_id=[0,1],batch=['a','b'],latent_0=[1.,2.]))
    frame=latent.drop(columns='latent_0').iloc[::-1]
    _,aligned=align_exports(latent,frame)
    assert aligned.row_id.tolist()==[1,2]
    with pytest.raises(ValueError,match='unique'):
        align_exports(latent,pd.concat([frame,frame]))


def test_variance_migration_is_idempotent_and_preserves_unknown(tmp_path):
    root=tmp_path/'analysis';root.mkdir();old=tmp_path/'latent_variance';old.mkdir()
    (old/'pca_variance.csv').write_text('same');(old/'notes.txt').write_text('keep')
    migrate_latent_variance(root);migrate_latent_variance(root)
    assert (root/'latent_variance/pca_variance.csv').read_text()=='same'
    assert (old/'notes.txt').read_text()=='keep'
    (old/'pca_variance.csv').write_text('different')
    (old/'latent_variance_summary.csv').write_text('do not move yet')
    with pytest.raises(ValueError,match='Conflicting'):migrate_latent_variance(root)
    assert (old/'latent_variance_summary.csv').exists()


def test_export_pipeline_cache_reuse_and_failure_cleanup(tmp_path,monkeypatch):
    import anndata as ad
    from scipy import sparse
    from dann.latent_analysis import run_latent_analysis
    from dann.analysis_execution import digest_file
    rng=np.random.default_rng(15);n=30
    root=tmp_path/'analysis';(root/'tables').mkdir(parents=True);(root/'umap/tables').mkdir(parents=True)
    obs=pd.DataFrame({'batch':['a']*n,'Lesion':[True]*n},index=[f'row-{i}' for i in range(n)])
    source=tmp_path/'labels.h5ad';ad.AnnData(sparse.csr_matrix((n,1)),obs=obs).write_h5ad(source)
    identity=dict(row_id=np.arange(n),batch_id=np.zeros(n,dtype=int),batch=['a']*n,split=['test']*n)
    latent=pd.DataFrame(dict(**identity,latent_0=rng.normal(size=n)))
    frame=pd.DataFrame(dict(**identity,umap_1=np.linspace(0,1,n),umap_2=np.zeros(n),Density_CD8=rng.uniform(.01,.8,n),
        valid_Density_CD8=np.ones(n,dtype=bool),pi_Density_CD8=np.full(n,.1),mu_Density_CD8=rng.normal(size=n),
        sigma_Density_CD8=np.ones(n),density_mean_Density_CD8=np.full(n,.2)))
    latent.to_csv(root/'tables/latent_embeddings.csv',index=False);frame.to_csv(root/'umap/tables/latent_umap.csv',index=False)
    pd.DataFrame(dict(**identity,target=['Density_CD8']*n,true_logit=np.log(frame.Density_CD8/(1-frame.Density_CD8)),
        predicted_mu=frame.mu_Density_CD8,predicted_sigma=np.ones(n))).to_csv(root/'tables/ziln_density_scatter.csv',index=False)
    config=dict(analysis=dict(output_dir=str(root),execution={'cache_policy':'reuse'},peaks={}),
        data=dict(path=str(source),batch_column='batch',target_columns=['Density_CD8']),loss={'logit_epsilon':1e-6})
    out=run_latent_analysis(config)
    first={p:digest_file(p) for p in (root/'.analysis-cache').rglob('*.npy')}
    assert len(list(out.glob('*.html')))==5
    config['analysis']['peaks'].update(ihc_percentile=95,ihc_gamma=.75)
    run_latent_analysis(config)
    report=json.loads((out/'latent_run.json').read_text())
    assert set(report['cache_hits'])=={'latent-feature-associations','latent-feature-umap'}
    assert all(digest_file(p)==digest for p,digest in first.items())
    before=digest_file(out/'sample_umap.html')
    monkeypatch.setattr('dann.latent_analysis.sample_viewers',lambda *args: (_ for _ in ()).throw(RuntimeError('render failed')))
    with pytest.raises(RuntimeError,match='render failed'):run_latent_analysis(config)
    assert digest_file(out/'sample_umap.html')==before
    assert not list(root.glob('.latent-work-*'))


def test_normal_analysis_calls_latent_generator(tmp_path,monkeypatch):
    from dann.analyze import run_analysis
    calls=[]
    monkeypatch.setattr('dann.analyze._run_legacy_analysis',lambda *args,**kwargs:tmp_path)
    monkeypatch.setattr('dann.latent_analysis.run_latent_analysis',lambda c,o:calls.append(o))
    run_analysis({'analysis':{'latent_interactive':True}},None)
    assert calls==[tmp_path]

"""Independent output controls, inference dependencies and invocation-owned artifacts."""
from pathlib import Path
import copy
import json
from types import SimpleNamespace
import h5py
import numpy as np
import pandas as pd
import pytest
from anndata.io import write_elem

from dann.config import ANALYSIS_FEATURE_DEFAULTS, analysis_feature_settings
from dann.analysis_outputs import analysis_needs, write_artifacts
from dann.tests.test_components import spatial_config
from dann.tests.test_spatial_execution import checkpoint


def switches(**enabled):
    values = {name: {'enabled': False} for name in ANALYSIS_FEATURE_DEFAULTS}
    values.update(enabled)
    return values


@pytest.mark.parametrize('name', ANALYSIS_FEATURE_DEFAULTS)
def test_feature_validation_and_family_override(name):
    assert not analysis_feature_settings({'features':switches()})[name]['enabled']
    with pytest.raises(ValueError, match=name):
        analysis_feature_settings({'features':{name:{'enabled':1}}})
    with pytest.raises(ValueError, match=name):
        analysis_feature_settings({'features':{name:{'typo':True}}})
    assert not any(analysis_feature_settings({'enabled':False})[name].values())


def test_omitted_switches_and_no_output_children():
    f = analysis_feature_settings({})
    assert f['direct_umap']['enabled'] and f['calibration']['tables']
    assert not f['variance']['enabled'] and not f['pca_umap']['enabled']
    all_off = {n: dict.fromkeys(d, False) for n,d in ANALYSIS_FEATURE_DEFAULTS.items()}
    for values in all_off.values(): values['enabled'] = True
    assert not analysis_needs({'features':all_off})['inference']


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('global_disable', [False, True])
def test_all_disabled_reads_no_checkpoint_or_data(tmp_path, monkeypatch, optimized, global_disable):
    from dann.analyze import run_analysis
    from dann import analysis_execution
    monkeypatch.setattr(analysis_execution, 'load_checkpoint_snapshot', lambda *a: pytest.fail('checkpoint loaded'))
    analysis = dict(output_dir=str(tmp_path), enabled=not global_disable,
                    execution={'enabled':optimized}, features=switches())
    old = tmp_path/'old.png'; old.write_bytes(b'preserve disabled output')
    run_analysis({'analysis':analysis}, Path('missing'))
    assert old.read_bytes() == b'preserve disabled output'
    assert json.loads((tmp_path/'analysis_execution.json').read_text())['artifacts'] == []


@pytest.mark.parametrize('optimized', [False, True])
def test_variance_only_preserves_context_and_skips_density_and_peaks(spatial_config, tmp_path, monkeypatch, optimized):
    from dann import analyze, analysis_execution
    c = spatial_config
    with h5py.File(c['data']['path'], 'r+') as h:
        write_elem(h['obs'], 'Lesion', np.arange(len(h['obs']['x'])) % 2 == 0)
    c['analysis'].update(features=switches(variance={'enabled':True}), figure_dpi=20, max_samples=12)
    c['analysis']['execution'].update(enabled=optimized, render_workers=0)
    path = checkpoint(c, tmp_path)
    for name in ('compute_peak_activity','fit_latent_umap','embedding_cosine_families'):
        monkeypatch.setattr(analyze,name,lambda *a,**kw: pytest.fail('disabled numerical stage'))
    monkeypatch.setattr(analysis_execution,'sampled_density_mean',lambda *a,**kw: pytest.fail('sampling'))
    output = analyze.run_analysis(c,path,output_override=tmp_path/'variance')
    report = json.loads((output/'latent_variance/provenance.json').read_text())
    assert report['scope'] == 'lesion' and report['split'] == 'validation'
    assert 2 <= report['n_observations'] <= 12
    assert not (output/'spatial').exists()
    assert not (output/'umap').exists()
    assert not (output/'peaks').exists()
    assert not list(output.rglob('*density*.csv'))
    manifest = json.loads((output/'analysis_execution.json').read_text())
    assert 'latent_variance/explained_variance.png' in manifest['artifacts']


def test_viewers_without_tables_and_diagnostic_suppression(tmp_path, monkeypatch):
    from dann import analyze
    from dann.analysis_outputs import write_artifacts
    from dann.config import load_config
    c = load_config(Path('dann/config.yaml'))
    c['analysis']['features'] = switches(
        direct_umap={'enabled':True,'tables':False,'batch':False,'densities':False,
            'logit_densities':False,'density_means':False,'parameters':False,'diagnostics':False,'interactive':True},
        pca_umap={'enabled':True,'tables':False,'batch':False,'densities':False,
            'logit_densities':False,'density_means':False,'parameters':False,'diagnostics':False,'interactive':True},
        prediction_scatter={'enabled':True,'tables':False,'plot':False,'interactive':True})
    n,t=5,4
    extracted = dict(row_ids=np.arange(n),batches=np.zeros(n,dtype=int),latent=np.ones((n,3)),
        targets=np.full((n,t),.2),pi=np.full((n,t),.2),mu=np.ones((n,t)),sigma=np.ones((n,t)),density_mean=np.full((n,t),.3))
    coordinates=np.arange(10).reshape(n,2)
    render = lambda *a,**kw: pytest.fail('disabled static figure')
    write_artifacts(c, Path('missing'),1,SimpleNamespace(batch_names=['slide']),extracted,extracted,
        coordinates,None,None,tmp_path,render,checkpoint={},pca_coordinates=coordinates,pca_metadata={})
    assert (tmp_path/'latent/sample_umap.html').exists()
    assert (tmp_path/'umap_pca/sample_umap.html').exists()
    assert (tmp_path/'latent/prediction_scatter.html').exists()
    assert not list(tmp_path.rglob('*.csv'))
    assert not list(tmp_path.rglob('*.png'))


def test_loss_only_never_infers(tmp_path, monkeypatch):
    from dann import analyze, analysis_execution
    from dann.config import load_config
    c = load_config(Path('dann/config.yaml'))
    c['analysis']['features'] = switches(loss_curves={'enabled':True})
    c['analysis']['peaks']['enabled']=False
    c['analysis']['output_dir']=str(tmp_path)
    c['training']['output_dir']=str(tmp_path)
    c['analysis']['figure_dpi']=12
    pd.DataFrame(dict(epoch=[1,2],train_total_loss=[2,1])).to_csv(tmp_path/'history.csv',index=False)
    monkeypatch.setattr(analysis_execution,'load_checkpoint_snapshot',lambda *a: ({'config':c,'epoch':1},'hash'))
    monkeypatch.setattr(analyze,'load_model',lambda *a,**kw:pytest.fail('inference'))
    analyze.run_analysis(c,Path('missing'))
    assert (tmp_path/'loss_curves.png').exists()
    assert (tmp_path/'ihc_curves.png').exists()


def test_no_output_peak_and_heatmap_entrypoints_need_no_inputs(tmp_path):
    from dann.config import analysis_execution_settings
    from dann.peak_analysis import run_peak_analysis
    from dann.spatial_heatmaps import run_spatial_heatmaps
    peak_config = dict(analysis=dict(output_dir=str(tmp_path), peaks=dict(enabled=True,
        tables=False, plots=False, interactive=False)))
    assert run_peak_analysis(peak_config,Path('missing'),analysis_execution_settings()) == tmp_path/'peaks'
    assert run_spatial_heatmaps(Path('missing'),Path('missing'),tmp_path/'maps',[],1e-6,30,
        enabled=False) == tmp_path/'maps'
    assert not (tmp_path/'maps').exists()


def test_support_only_needs_no_density_or_grid(tmp_path):
    from dann.diagnostics import write_density_diagnostics
    c=dict(analysis=dict(features=switches(density_diagnostics={'enabled':True,'errors':False,'contrasts':False})))
    assert not analysis_needs(c['analysis'])['density']
    write_density_diagnostics(dict(targets=np.zeros((2,1)),row_ids=np.array([3,7]),
        support=np.ones(2),latent_support=np.ones(2)),c,tmp_path,'validation')
    assert [p.name for p in tmp_path.iterdir()] == ['validation_neighborhood_support.csv']


def test_only_prediction_scatter_requests_only_its_split():
    needs=analysis_needs(dict(features=switches(prediction_scatter={'enabled':True,'tables':False})))
    assert needs['scatter'] and not needs['main']
    assert not needs['density']

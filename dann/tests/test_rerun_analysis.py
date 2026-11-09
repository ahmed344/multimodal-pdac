"""Safety contracts for archival and resumable downstream execution."""
import json
from pathlib import Path

import pytest

from dann.verification import rerun_analysis as batch


def fixture_manifest(tmp_path):
    directory = tmp_path / 'results' / 'combination' / 'run'
    model = directory / 'model'
    model.mkdir(parents=True)
    (model / 'best.pt').write_bytes(b'immutable checkpoint')
    evidence = tmp_path / 'archive'
    evidence.mkdir()
    config = evidence / 'config.yaml'
    config.write_text('configuration')
    run = dict(name='combination/run', directory=str(directory), config=str(config),
               config_sha256=batch.digest(config), model=batch.inventory(model, hashes=True), stages={})
    manifest = dict(inputs={}, runs=[run], archives=[], archived=True)
    batch.save_json(evidence / 'manifest.json', manifest)
    return evidence, manifest, run


def add_archive(tmp_path, manifest, name):
    source = tmp_path / 'source' / name
    source.mkdir(parents=True)
    (source / 'result').write_text(name)
    item = dict(source=str(source), destination=str(tmp_path / 'destination' / name),
                files=batch.inventory(source), state='pending')
    manifest['archives'].append(item)
    manifest['archived'] = False
    return item


def test_archive_collision_is_checked_before_any_move(tmp_path):
    evidence, manifest, _ = fixture_manifest(tmp_path)
    first = add_archive(tmp_path, manifest, 'first')
    second = add_archive(tmp_path, manifest, 'second')
    Path(second['destination']).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        batch.archive_outputs(evidence, manifest)
    assert Path(first['source']).exists()
    assert not Path(first['destination']).exists()


def test_archive_recovers_rename_before_manifest_write(tmp_path):
    evidence, manifest, run = fixture_manifest(tmp_path)
    item = add_archive(tmp_path, manifest, 'analysis')
    destination = Path(item['destination'])
    destination.parent.mkdir()
    Path(item['source']).rename(destination)
    batch.archive_outputs(evidence, manifest)
    assert manifest['archived']
    assert destination.joinpath('result').read_text() == 'analysis'
    assert batch.inventory(Path(run['directory']) / 'model', hashes=True) == run['model']


def test_archive_rejects_changed_inventory(tmp_path):
    evidence, manifest, _ = fixture_manifest(tmp_path)
    item = add_archive(tmp_path, manifest, 'analysis')
    Path(item['source'], 'result').write_text('changed')
    with pytest.raises(ValueError, match='inventory changed'):
        batch.archive_outputs(evidence, manifest)


@pytest.mark.parametrize('changed', ['model', 'config'])
def test_resume_rejects_changed_immutable_inputs(tmp_path, changed):
    evidence, _, run = fixture_manifest(tmp_path)
    path = Path(run['config']) if changed == 'config' else Path(run['directory']) / 'model/best.pt'
    path.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        batch.orchestrate(evidence)


def test_commands_never_train_and_route_loss_curves_inside_analysis(tmp_path):
    _, _, run = fixture_manifest(tmp_path)
    commands = dict(batch.commands(run))
    assert all('dann.train' not in command for command in commands.values())
    assert commands['analyze'][-2:] == ['--output-dir', str(Path(run['directory']) / 'analysis')]
    assert 'latent_variance' not in commands


def test_resume_skips_valid_stages_and_invalidates_dependents(tmp_path, monkeypatch):
    evidence, manifest, run = fixture_manifest(tmp_path)
    root = Path(run['directory']) / 'analysis'
    root.mkdir()
    for stage in ('first', 'second', 'third'):
        (root / stage).write_text(stage)
        run['stages'][stage] = dict(state='completed', artifacts={stage: batch.inventory(root)[stage]})
    batch.save_json(evidence / 'manifest.json', manifest)
    monkeypatch.setattr(batch, 'commands', lambda run: [(s, []) for s in ('first', 'second', 'third')])
    calls = []

    def execute(evidence, manifest, run, stage, arguments):
        calls.append(stage)
        run['stages'][stage] = dict(state='failed')
        return False

    monkeypatch.setattr(batch, 'execute', execute)
    assert batch.orchestrate(evidence)['status'] == 'completed'
    assert calls == []
    (root / 'second').write_text('damaged artifact')
    assert batch.orchestrate(evidence)['status'] == 'failed'
    saved = json.loads((evidence / 'manifest.json').read_text())
    assert calls == ['second']
    assert saved['runs'][0]['stages']['third']['state'] == 'stale'


def test_latest_does_not_reuse_best_stage_receipts(tmp_path, monkeypatch):
    evidence, manifest, run = fixture_manifest(tmp_path)
    root = Path(run['directory'])/'analysis'
    root.mkdir()
    (root/'result').write_text('best checkpoint output')
    run.update(checkpoint='latest', stages={'analyze': dict(state='completed', artifacts=batch.inventory(root))})
    batch.save_json(evidence/'manifest.json', manifest)
    monkeypatch.setattr(batch, 'commands', lambda run: [('analyze', [])])
    calls = []
    def execute(batch_path, manifest, run, stage, arguments):
        calls.append(stage)
        run['stages'][stage].update(state='completed', checkpoint='latest')
        return True
    monkeypatch.setattr(batch, 'execute', execute)
    assert batch.orchestrate(evidence)['status'] == 'completed'
    assert calls == ['analyze']
    assert batch.orchestrate(evidence)['status'] == 'completed'
    assert calls == ['analyze']


def test_stage_failure_is_recorded_without_touching_model(tmp_path):
    evidence, manifest, run = fixture_manifest(tmp_path)
    assert not batch.execute(evidence, manifest, run, 'artifacts',
                             ['dann.verification.rerun_analysis', '--audit', str(tmp_path / 'missing.yaml')])
    entry = run['stages']['artifacts']
    assert entry['state'] == 'failed' and entry['returncode'] != 0
    assert batch.inventory(Path(run['directory']) / 'model', hashes=True) == run['model']


def test_stage_inventory_isolated_from_later_stages(tmp_path):
    for name in ('tables/data.csv', 'spatial/predictions.parquet', 'spatial/heatmaps/plot.png',
                 'latent_variance/summary.csv', 'peaks/reports/peak_browser_audit.json'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('result')
    assert set(batch.stage_inventory(tmp_path, 'analyze')) == {'tables/data.csv', 'latent_variance/summary.csv'}
    assert set(batch.stage_inventory(tmp_path, 'spatial')) == {'spatial/predictions.parquet'}
    assert set(batch.stage_inventory(tmp_path, 'peak_browser')) == {'peaks/reports/peak_browser_audit.json'}


def test_interruption_terminates_worker_and_records_failure(tmp_path, monkeypatch):
    import psutil
    from types import SimpleNamespace
    evidence, manifest, run = fixture_manifest(tmp_path)
    monkeypatch.setattr(batch.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match='disk headroom'):
        batch.execute(evidence, manifest, run, 'artifacts',
                      ['dann.verification.rerun_analysis', '--audit', str(tmp_path / 'missing.yaml')])
    entry = run['stages']['artifacts']
    assert entry['state'] == 'interrupted'
    assert not psutil.pid_exists(entry['pid'])


@pytest.mark.parametrize('width,slide_count', [(3,1),(256,2),(512,3)])
def test_audit_uses_checkpoint_width_input_slides_and_generated_manifests(tmp_path, monkeypatch, width, slide_count):
    import anndata as ad
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from scipy import sparse
    from PIL import Image
    from dann.config import load_config, resolve_execution, spatial_heatmap_settings
    from dann.spatial import _file_identity, build_output_table, build_latent_output_table
    from dann.targets import target_contract
    source=tmp_path/'input.h5ad'
    names=np.array(['a','b','c'])
    slides=np.array([f'slide_{i%slide_count}' for i in range(3)])
    data=ad.AnnData(sparse.csr_matrix(np.ones((3,2),dtype=np.float32)),obs=pd.DataFrame(index=names))
    data.obs['batch']=slides
    data.obs['Lesion']=True
    config=load_config(Path('dann/config.yaml'))
    config['model']['heads']['he']['enabled']=False  # Synthetic audit has no H&E artifacts.
    config['data'].pop('preprocessing_fingerprint',None)
    for target in config['data']['target_columns']:
        data.obs[target]=np.zeros(3,dtype=np.float32)
    data.write_h5ad(source)
    config['results'].update(root=str(tmp_path/'results'),run_name='audit')
    config['data'].update(path=str(source),context_path=str(source),matrix_key='X')
    config['model'].update(num_peaks=2,latent_dim=width)
    config['analysis'].update(max_samples=None,latent_interactive=True,figure_dpi=1)
    config['analysis']['features']['pca_umap']['enabled'] = False  # This fixture represents the existing direct UMAP.
    config['analysis']['peaks']['enabled'] = True
    config['analysis']['spatial_heatmaps'].update(enabled=True, diagnostics=True)
    config['post_training'].update(run_inference=True, run_spatial_heatmaps=True)
    resolve_execution(config)
    root=Path(config['analysis']['output_dir']);root.mkdir(parents=True)
    checkpoint=dict(config=config,epoch=24,split_indices={'validation':np.arange(3)},target_columns=config['data']['target_columns'],target_transform=target_contract(config))
    best=Path(config['analysis']['checkpoint']);best.parent.mkdir();torch.save(checkpoint,best)
    monkeypatch.setattr('dann.config.load_config',lambda p: config)
    spatial=root/'spatial';spatial.mkdir()
    shape=(3,len(config['data']['target_columns']))
    predictions=build_output_table(np.arange(3),names,np.zeros(shape),np.zeros(shape),np.ones(shape),config['data']['target_columns'],np.zeros(shape))
    predictions=predictions.replace_schema_metadata({b'dann_target_transform':json.dumps(target_contract(config)).encode()})
    prediction_path=Path(config['spatial']['output']);latent_path=Path(config['spatial']['latents'])
    pq.write_table(predictions,prediction_path)
    pq.write_table(build_latent_output_table(np.arange(3),names,np.zeros((3,width))),latent_path)
    provenance=dict(selected_rows=3,context_rows=3,density_mc_samples=config['analysis']['density_mc_samples'],density_mc_seed=config['analysis']['density_mc_seed'])
    for key,path in [('checkpoint',best),('source',source),('prediction',prediction_path),('latent',latent_path)]:
        provenance[key+'_identity']=_file_identity(path)
    batch.save_json(prediction_path.with_suffix('.provenance.json'),provenance)
    (root/'tables').mkdir()
    pd.DataFrame(dict(row_id=np.arange(3),**{f'latent_{i}':np.zeros(3) for i in range(width)})).to_csv(root/'tables/latent_embeddings.csv',index=False)
    (root/'umap/tables').mkdir(parents=True)
    pd.DataFrame(dict(row_id=np.arange(3),umap_1=np.zeros(3),umap_2=np.zeros(3),
                      **{f'{prefix}_{target}':np.ones(3) for target in config['data']['target_columns']
                         for prefix in ('pi','mu','sigma','density_mean')})).to_csv(root/'umap/tables/latent_umap.csv',index=False)
    batch.save_json(root/'analysis_execution.json',dict(checkpoint_epoch=25,
        source_identity=dict(checkpoint=batch.digest(best)),
        artifacts=['tables/latent_embeddings.csv','umap/tables/latent_umap.csv']))
    (root/'peaks/settings').mkdir(parents=True)
    (root/'peaks/spatial/families').mkdir(parents=True)
    Image.new('RGB',(2,2)).save(root/'peaks/spatial/families/family_01_spatial.png')
    batch.save_json(root/'peaks/settings/cd8_family_comparison_settings.json',dict(slides=slides.tolist(),family_ids=[1]))
    batch.save_json(root/'peaks/peak_run.json',dict(artifacts=['settings/cd8_family_comparison_settings.json','spatial/families/family_01_spatial.png']))
    (root/'latent').mkdir()
    (root/'latent/view.html').write_text('<html></html>')
    batch.save_json(root/'latent/latent_run.json',dict(artifacts=['view.html']))
    settings=spatial_heatmap_settings(config['analysis'].get('spatial_heatmaps'))
    outputs=[]
    for region in settings['regions']:
        directory=spatial/'heatmaps'/region;directory.mkdir(parents=True)
        for target in config['data']['target_columns']:
            for prefix in ('spatial_heatmap','spatial_heatmap_diagnostics'):
                path=directory/f'{prefix}_{target}_{region}.png'
                size=[30 if prefix.endswith('diagnostics') else 36,4*slide_count]
                Image.new('RGB',tuple(size)).save(path);outputs.append(dict(path=str(path),size=size))
    batch.save_json(spatial/'heatmaps/heatmap_run.json',dict(outputs=outputs,slides=slides.tolist(),settings=settings,densities=list(reversed(config['data']['target_columns'])),dpi=1))
    variance=root/'latent_variance';variance.mkdir()
    pd.DataFrame([dict(latent_width=width,n_observations=3,n_slides=slide_count)]).to_csv(variance/'latent_variance_summary.csv',index=False)
    pd.DataFrame({'pc':np.arange(width)}).to_csv(variance/'pca_variance.csv',index=False)
    Image.new('RGB',(2,2)).save(variance/'explained_variance.png')
    report=batch.audit_outputs(tmp_path/'config.yaml')
    assert report['latent_width']==width and len(report['slides'])==slide_count
    assert report['validation_rows']==report['tissue_rows']==3
    assert report['counts']['peak_png']==report['counts']['latent_html']==1
    identity=batch.inventory(root)['artifact_audit.json']
    batch.audit_outputs(tmp_path/'config.yaml',write_report=False)
    assert batch.inventory(root)['artifact_audit.json']==identity
    execution_path = root/'analysis_execution.json'
    execution = json.loads(execution_path.read_text())
    execution['checkpoint_epoch'] = 35
    batch.save_json(execution_path, execution)
    with pytest.raises(ValueError, match='checkpoint provenance mismatch'):
        batch.audit_outputs(tmp_path/'config.yaml')
    execution['checkpoint_epoch'] = 25
    batch.save_json(execution_path, execution)
    original=(root/'tables/latent_embeddings.csv').read_text()
    original_umap=(root/'umap/tables/latent_umap.csv').read_text()
    pd.read_csv(root/'umap/tables/latent_umap.csv').iloc[::-1].to_csv(root/'umap/tables/latent_umap.csv',index=False)
    frame=pd.read_csv(root/'tables/latent_embeddings.csv')
    frame.iloc[::-1].to_csv(root/'tables/latent_embeddings.csv',index=False)
    with pytest.raises(AssertionError):
        batch.audit_outputs(tmp_path/'config.yaml')
    (root/'tables/latent_embeddings.csv').write_text(original)
    (root/'umap/tables/latent_umap.csv').write_text(original_umap)
    damaged=np.zeros((3,width));damaged[-1,-1]=np.nan
    pq.write_table(build_latent_output_table(np.arange(3),names,damaged),latent_path)
    provenance['latent_identity']=_file_identity(latent_path)
    batch.save_json(prediction_path.with_suffix('.provenance.json'),provenance)
    with pytest.raises(ValueError,match='non-finite'):
        batch.audit_outputs(tmp_path/'config.yaml')


def test_recovery_does_not_signal_a_reused_pid(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import psutil
    _,_,run=fixture_manifest(tmp_path)
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:2.))
    monkeypatch.setattr(batch.os,'killpg',lambda *args:pytest.fail('Signaled an unrelated process'))
    batch.recover_worker(run,'analyze',dict(pid=123,process_created=1.))


def test_recovery_terminates_only_matching_worker_group(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import psutil
    _,_,run=fixture_manifest(tmp_path)
    previous=dict(pid=123,process_created=1.,attempt=2,command=[batch.sys.executable,'-m','dann.analyze'])
    expected=[batch.sys.executable,'-m','dann.verification.production_fits','--worker',
              str(Path(run['config']).parent/'analyze.2.resources.json'),'dann.analyze']
    calls=[]
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:1.,cmdline=lambda:expected,wait=lambda timeout:calls.append('wait')))
    monkeypatch.setattr(batch.os,'getpgid',lambda pid:123)
    monkeypatch.setattr(batch.os,'killpg',lambda pid,sig:calls.append((pid,sig)))
    batch.recover_worker(run,'analyze',previous)
    assert calls==[(123,batch.signal.SIGTERM),'wait']


def test_html_link_audit_ignores_embedded_script_strings(tmp_path):
    page=tmp_path/'index.html'
    (tmp_path/'asset.js').write_text('')
    page.write_text('<html><script>let text = `src="not-a-resource"`;</script><script src="asset.js"></script><a href="#local">anchor</a></html>')
    batch.validate_html_links(page)
    (tmp_path/'asset.js').unlink()
    with pytest.raises(ValueError,match='Broken report link'):
        batch.validate_html_links(page)

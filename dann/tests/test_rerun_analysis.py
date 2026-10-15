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
    assert commands['latent_variance'][-1].endswith('/analysis/latent_variance')


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
    assert set(batch.stage_inventory(tmp_path, 'analyze')) == {'tables/data.csv'}
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


def test_audit_reads_categorical_observation_index_and_checks_row_order(tmp_path, monkeypatch):
    import h5py
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from anndata.io import write_elem
    from PIL import Image
    import dann.config
    from dann.targets import target_contract
    source = tmp_path / 'input.h5ad'
    with h5py.File(source, 'w') as handle:
        obs = handle.create_group('obs')
        obs.attrs['_index'] = '_index'
        write_elem(obs, '_index', pd.Categorical(['a', 'b', 'c']))
    root = tmp_path / 'analysis'
    config = dict(data=dict(context_path=str(source), target_columns=['Density_CD8']),
                  analysis=dict(output_dir=str(root)))
    monkeypatch.setattr(dann.config, 'load_config', lambda path: config)
    for folder, pngs, htmls in [('peaks', 78, 38), ('latent', 0, 5), ('spatial/heatmaps', 16, 0)]:
        destination = root / folder
        destination.mkdir(parents=True)
        for i in range(pngs):
            Image.new('RGB', (2, 2)).save(destination / f'{i}.png')
        for i in range(htmls):
            (destination / f'{i}.html').write_text('<html></html>')
    variance = root / 'latent_variance'
    variance.mkdir()
    Image.new('RGB', (2, 2)).save(variance / 'explained_variance.png')
    for name in ('latent_variance_summary.csv', 'pca_variance.csv'):
        (variance / name).write_text('value\n1\n')
    predictions = pa.table(dict(row_position=[0, 1, 2], mu_Density_CD8=[0., 0., 0.]))
    predictions = predictions.replace_schema_metadata({b'dann_target_transform': json.dumps(target_contract(config)).encode()})
    pq.write_table(predictions, root / 'spatial/spatial_predictions.parquet')
    latents = dict(row_position=[0, 1, 2], **{f'latent_{i}': [0., 0., 0.] for i in range(256)})
    pq.write_table(pa.table(latents), root / 'spatial/spatial_latent.parquet')
    assert batch.audit_outputs(tmp_path / 'config.yaml')['tissue_rows'] == 3
    latents['row_position'] = [0, 2, 1]
    pq.write_table(pa.table(latents), root / 'spatial/spatial_latent.parquet')
    with pytest.raises(AssertionError):
        batch.audit_outputs(tmp_path / 'config.yaml')

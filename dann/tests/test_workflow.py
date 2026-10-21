"""Synthetic orchestration, checkpoint routing, and export-reuse contracts."""

import copy
import json
from pathlib import Path
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from scipy import sparse

from dann import workflow
from dann.checkpoints import architecture_contract
from dann.config import apply_smoke_overrides, load_config, post_training_settings, resolve_execution
from dann.spatial import _file_identity, build_latent_output_table, build_output_table
from dann.targets import target_contract


@pytest.fixture
def run(tmp_path):
    config = load_config(Path('dann/config.yaml'))
    config['results'].update(root=str(tmp_path / 'requested'), run_name='fit')
    config['model']['aggregation']['type'] = 'mlp'
    for head in config['model']['heads'].values():
        head['type'] = 'mlp'
    config['model'].update(num_peaks=2, latent_dim=2)
    config['training']['device'] = 'cpu'
    resolve_execution(config)
    tissue = tmp_path / 'tissue.h5ad'
    adata = ad.AnnData(sparse.csr_matrix(np.ones((3, 2), dtype=np.float32)))
    adata.obs['batch'] = ['a', 'b', 'a']
    for target in config['data']['target_columns']:
        adata.obs[target] = np.zeros(3, dtype=np.float32)
    adata.write_h5ad(tissue)
    config['data'].update(path=str(tissue), context_path=str(tissue), matrix_key='X')
    checkpoint_path = tmp_path / 'actual' / 'model' / 'best.pt'
    checkpoint_path.parent.mkdir(parents=True)
    saved = copy.deepcopy(config)
    saved['post_training'] = {key: False for key in post_training_settings()}
    saved['analysis'].update(density_mc_samples=7, latent_interactive=False)
    checkpoint = dict(config=saved, target_columns=saved['data']['target_columns'],
                      target_transform=target_contract(saved), split_indices={'train': np.array([0, 1]),
                      'validation': np.array([2])}, data_identity={'feature_order': ['0', '1']})
    torch.save(checkpoint, checkpoint_path)
    return config, checkpoint_path, checkpoint


@pytest.fixture
def stages(monkeypatch):
    calls = []
    mocks = {}
    for name in ('analysis', 'inference', 'spatial_heatmaps', 'latent_variance'):
        def stage(*args, _name=name, **kwargs):
            calls.append(_name)
            if _name == 'analysis':
                return kwargs['output_override']
            if _name == 'latent_variance':
                return Path(args[0]['latent_variance']['output_dir'])
            return kwargs['output_path' if _name == 'inference' else 'output_dir']
        mocks[name] = Mock(side_effect=stage)
        monkeypatch.setattr(workflow, f'run_{name}', mocks[name])
    return calls, mocks


def test_settings_defaults_partial_and_smoke():
    assert all(post_training_settings().values())
    assert post_training_settings({'run_analysis': False}) == {
        **post_training_settings(), 'run_analysis': False}
    config = load_config(Path('dann/config.yaml'))
    smoke = apply_smoke_overrides(config)
    assert not smoke['post_training']['enabled']
    assert config['post_training']['enabled']
    assert config['analysis']['peaks']['enabled'] and config['analysis']['latent_interactive']


@pytest.mark.parametrize('supplied', [False, [], 'yes', {'oops': True},
                                      *[{key: value} for key in post_training_settings()
                                        for value in (1, None, 'false')]])
def test_invalid_settings(supplied):
    with pytest.raises(ValueError, match='post_training'):
        post_training_settings(supplied)


@pytest.mark.parametrize('supplied', ['omitted', {}, {'run_analysis': False}, None])
def test_yaml_settings(tmp_path, supplied):
    config = yaml.safe_load(Path('dann/config.yaml').read_text())
    if supplied == 'omitted':
        config.pop('post_training')
    else:
        config['post_training'] = supplied
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    if supplied is None:
        with pytest.raises(ValueError, match='post_training'):
            load_config(path)
    else:
        assert load_config(path)['post_training'] == post_training_settings(
            None if supplied == 'omitted' else supplied)


def test_order_current_settings_frozen_contracts_and_resumed_paths(run, stages):
    config, best, checkpoint = run
    original = copy.deepcopy(config)
    config['data']['path'] = '/changed/training.h5ad'
    config['data']['intensity_transform'] = 'changed'
    config['data']['target_columns'] = ['changed']
    config['model']['latent_dim'] = 99
    config['training']['resume_checkpoint'] = str(best.with_name('latest.pt'))
    calls, mocks = stages
    outputs = workflow.run_post_training_workflow(config, best)
    assert calls == ['analysis', 'inference', 'spatial_heatmaps', 'latent_variance']
    effective, actual_checkpoint = mocks['analysis'].call_args.args
    assert actual_checkpoint == best
    assert effective['data']['path'] == checkpoint['config']['data']['path']
    assert effective['data']['intensity_transform'] == checkpoint['config']['data']['intensity_transform']
    assert effective['model']['latent_dim'] == 2
    assert effective['data']['target_columns'] == checkpoint['target_columns']
    assert effective['analysis']['latent_interactive']
    assert effective['analysis']['peaks']['enabled']
    assert config['analysis'] == original['analysis']
    root = best.parent.parent / 'analysis'
    assert outputs == dict(analysis=root, inference=root / 'spatial/spatial_predictions.parquet',
                          latents=root / 'spatial/spatial_latent.parquet',
                          spatial_heatmaps=root / 'spatial/heatmaps', latent_variance=root / 'latent_variance')
    inference = mocks['inference'].call_args.kwargs
    assert inference['checkpoint_path'] == best
    assert inference['input_path'] == Path(config['data']['context_path'])
    assert inference['density_mc_samples'] == config['analysis']['density_mc_samples']
    assert inference['density_mc_seed'] == config['analysis']['density_mc_seed']
    assert inference['execution_overrides']['spatial_execution'] == config['analysis']['spatial_execution']
    assert inference['overwrite']
    assert mocks['spatial_heatmaps'].call_args.kwargs['workers'] == config['analysis']['spatial_heatmaps']['workers']
    assert mocks['latent_variance'].call_args.args[0]['latent_variance']['latents'] == str(outputs['latents'])


@pytest.mark.parametrize('saved_kind', ['mlp', 'cnn', 'gatv2'])
@pytest.mark.parametrize('current_kind', ['mlp', 'gatv2'])
def test_inference_batch_units_follow_checkpoint(run, stages, saved_kind, current_kind):
    config, best, checkpoint = run
    saved = checkpoint['config']
    saved['model']['aggregation']['type'] = saved_kind
    resolve_execution(saved)
    checkpoint['architecture'] = architecture_contract(saved)
    torch.save(checkpoint, best)
    config['model']['aggregation']['type'] = current_kind
    resolve_execution(config)
    config['analysis']['batch_size'] = 4096
    config['analysis']['spatial_execution']['tiles_per_batch'] = 64
    config['post_training'].update(run_analysis=False, run_spatial_heatmaps=False,
                                   run_latent_variance=False)

    workflow.run_post_training_workflow(config, best)

    inference = stages[1]['inference'].call_args.kwargs
    assert inference['batch_size_override'] == (4096 if saved_kind == 'mlp' else None)
    assert inference['execution_overrides']['spatial_execution']['tiles_per_batch'] == 64


@pytest.mark.parametrize('switch', ['enabled', 'run_analysis', 'run_inference',
                                    'run_spatial_heatmaps', 'run_latent_variance'])
def test_individual_switches(run, stages, monkeypatch, switch):
    config, best, _ = run
    config['post_training'][switch] = False
    validation = Mock()
    monkeypatch.setattr(workflow, '_validate_exports', validation)
    workflow.run_post_training_workflow(config, best)
    expected = ['analysis', 'inference', 'spatial_heatmaps', 'latent_variance']
    expected = [] if switch == 'enabled' else [s for s in expected if f'run_{s}' != switch]
    assert stages[0] == expected
    assert validation.call_count == (2 if switch == 'run_inference' else 0)


def test_disabled_needs_no_checkpoint_or_tissue(stages):
    assert workflow.run_post_training_workflow({'post_training': {'enabled': False}}, Path('missing')) == {}
    assert workflow.run_post_training_workflow({'post_training': {
        key: False for key in post_training_settings() if key.startswith('run_')}}, Path('missing')) == {}
    assert not stages[0]


@pytest.mark.parametrize('failed_stage', ['analysis', 'inference', 'spatial_heatmaps', 'latent_variance'])
def test_immediate_failure_preserves_checkpoint(run, stages, failed_stage):
    config, best, _ = run
    calls, mocks = stages
    mocks[failed_stage].side_effect = OSError('failure sentinel')
    with pytest.raises(workflow.PostTrainingError, match=f"'{failed_stage}' failed") as caught:
        workflow.run_post_training_workflow(config, best)
    assert isinstance(caught.value.__cause__, OSError)
    names = list(mocks)
    assert calls == names[:names.index(failed_stage)]
    assert best.is_file()
    for name in names[names.index(failed_stage) + 1:]:
        mocks[name].assert_not_called()


def test_overwrite_refusal_propagates(run, stages):
    config, best, _ = run
    config['post_training']['overwrite_inference'] = False
    stages[1]['inference'].side_effect = FileExistsError('Output already exists')
    with pytest.raises(workflow.PostTrainingError, match='inference.*already exists'):
        workflow.run_post_training_workflow(config, best)
    assert stages[1]['inference'].call_args.kwargs['overwrite'] is False
    stages[1]['spatial_heatmaps'].assert_not_called()


def write_exports(config, best, checkpoint):
    effective = workflow._workflow_config(config, checkpoint, best)
    predictions = Path(effective['spatial']['output'])
    latents = Path(effective['spatial']['latents'])
    predictions.parent.mkdir(parents=True)
    positions = np.arange(3, dtype=np.int64)
    names = positions.astype(str)
    shape = (3, len(checkpoint['target_columns']))
    table = build_output_table(positions, names, np.zeros(shape, dtype=np.float32),
                               np.full(shape, .5, dtype=np.float32), np.ones(shape, dtype=np.float32),
                               checkpoint['target_columns'])
    pq.write_table(table.replace_schema_metadata({
        b'dann_target_transform': json.dumps(target_contract(effective)).encode()}), predictions)
    pq.write_table(build_latent_output_table(positions, names, np.ones((3, 2), dtype=np.float32)), latents)
    provenance_path = predictions.with_suffix('.provenance.json')
    provenance = dict(checkpoint_identity=_file_identity(best),
                      source_identity=_file_identity(Path(config['data']['context_path'])),
                      prediction_identity=_file_identity(predictions), latent_identity=_file_identity(latents),
                      selected_rows=3, context_rows=3, density_mc_samples=config['analysis']['density_mc_samples'],
                      density_mc_seed=config['analysis']['density_mc_seed'])
    provenance_path.write_text(json.dumps(provenance))
    return predictions, latents, provenance_path, provenance


@pytest.mark.parametrize('consumer', ['both', 'spatial_heatmaps', 'latent_variance'])
def test_reuse_valid_exports(run, stages, consumer):
    config, best, checkpoint = run
    predictions, latents, _, _ = write_exports(config, best, checkpoint)
    config['post_training'].update(run_analysis=False, run_inference=False)
    if consumer != 'both':
        other = 'latent_variance' if consumer == 'spatial_heatmaps' else 'spatial_heatmaps'
        config['post_training'][f'run_{other}'] = False
        (latents if other == 'latent_variance' else predictions).unlink()
    outputs = workflow.run_post_training_workflow(config, best)
    assert set(outputs) == ({'spatial_heatmaps', 'latent_variance'} if consumer == 'both' else {consumer})
    stages[1]['inference'].assert_not_called()


@pytest.mark.parametrize('problem', ['missing_predictions', 'missing_latents', 'missing_provenance',
    'checkpoint_identity', 'source_identity', 'prediction_identity', 'latent_identity', 'density_mc_seed',
    'prefix', 'prediction_schema', 'latent_schema', 'prediction_order', 'latent_order', 'target_transform'])
def test_reuse_rejects_missing_incompatible_or_misaligned_exports(run, stages, problem):
    config, best, checkpoint = run
    predictions, latents, provenance_path, provenance = write_exports(config, best, checkpoint)
    config['post_training'].update(run_analysis=False, run_inference=False)
    if problem.startswith('missing_'):
        {'missing_predictions': predictions, 'missing_latents': latents,
         'missing_provenance': provenance_path}[problem].unlink()
    elif problem in provenance:
        provenance[problem] = None
    elif problem == 'prefix':
        provenance['selected_rows'] = 2
    else:
        path = latents if problem.startswith('latent_') else predictions
        table = pq.read_table(path)
        if problem.endswith('_schema'):
            table = table.drop([table.column_names[-1]])
        elif problem.endswith('_order'):
            table = table.take([2, 1, 0])
        else:
            table = table.replace_schema_metadata({})
        pq.write_table(table, path)
        provenance['latent_identity' if path == latents else 'prediction_identity'] = _file_identity(path)
    if problem != 'missing_provenance':
        provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(workflow.PostTrainingError, match='Regenerate them.*run_inference: true'):
        workflow.run_post_training_workflow(config, best)
    stages[1]['inference'].assert_not_called()
    stages[1]['latent_variance'].assert_not_called()


def test_missing_runtime_input_names_failed_stage(run, stages):
    config, best, _ = run
    config['data']['context_path'] = '/missing/tissue.h5ad'
    with pytest.raises(workflow.PostTrainingError, match="'inference'.*input does not exist"):
        workflow.run_post_training_workflow(config, best)
    assert stages[0] == ['analysis']


@pytest.mark.parametrize('smoke,training_failure', [(False, False), (True, False), (False, True)])
def test_training_cli_dispatch(run, monkeypatch, smoke, training_failure):
    from dann import train
    config, best, _ = run
    monkeypatch.setattr(train, 'load_config', lambda _: config)
    training = Mock(return_value=best, side_effect=RuntimeError('training failed') if training_failure else None)
    monkeypatch.setattr(train, 'train_model', training)
    downstream = Mock(return_value={})
    monkeypatch.setattr(workflow, 'run_post_training_workflow', downstream)
    args = ['--smoke-test'] if smoke else []
    if training_failure:
        with pytest.raises(RuntimeError, match='training failed'):
            train.main(args)
        downstream.assert_not_called()
    else:
        assert train.main(args) == 0
        downstream.assert_called_once()
        assert downstream.call_args.args[1] == best
        assert downstream.call_args.args[0]['post_training']['enabled'] is not smoke


def test_cli_workflow_failure_returns_nonzero(run, monkeypatch, capsys):
    from dann import train
    config, best, _ = run
    monkeypatch.setattr(train, 'load_config', lambda _: config)
    monkeypatch.setattr(train, 'train_model', lambda _: best)
    monkeypatch.setattr(workflow, 'run_post_training_workflow', Mock(
        side_effect=workflow.PostTrainingError('analysis', ValueError('bad input'))))
    assert train.main([]) == 1
    assert "'analysis' failed" in capsys.readouterr().out
    assert best.is_file()


def test_analysis_nested_reports_execute_once(run, monkeypatch):
    from dann import analyze, analysis_execution, artifact_layout, latent_analysis, peak_analysis
    config, best, _ = run
    config['post_training'].update(run_inference=False, run_spatial_heatmaps=False, run_latent_variance=False)
    output = best.parent.parent / 'analysis'
    numeric = Mock(return_value=output)
    interactive = Mock()
    peaks = Mock()
    monkeypatch.setattr(analysis_execution, 'run_optimized_analysis', numeric)
    monkeypatch.setattr(artifact_layout, 'organize_analysis', Mock())
    monkeypatch.setattr(latent_analysis, 'run_latent_analysis', interactive)
    monkeypatch.setattr(peak_analysis, 'run_peak_analysis', peaks)
    # Use the actual analysis wrapper so its nested report dispatch is exercised.
    monkeypatch.setattr(workflow, 'run_analysis', analyze.run_analysis)
    assert workflow.run_post_training_workflow(config, best) == {'analysis': output}
    numeric.assert_called_once()
    interactive.assert_called_once()
    peaks.assert_called_once()
    assert peaks.call_args.args[1] == best


def test_configured_destinations_are_rebased(run, stages):
    config, best, _ = run
    root = Path(config['results']['run_dir'])
    config['analysis']['output_dir'] = str(root / 'reports')
    config['spatial'] = dict(output=str(root / 'exports/predictions.parquet'),
                             latents=str(root / 'exports/embeddings.parquet'))
    config['latent_variance']['output_dir'] = str(root / 'variance')
    outputs = workflow.run_post_training_workflow(config, best)
    actual = best.parent.parent
    assert outputs == dict(analysis=actual / 'reports', inference=actual / 'exports/predictions.parquet',
                          latents=actual / 'exports/embeddings.parquet',
                          spatial_heatmaps=actual / 'reports/spatial/heatmaps', latent_variance=actual / 'variance')


def test_training_only_and_resume_early_stopping(tmp_path, monkeypatch):
    from dann import train
    from dann.tests.test_components import spatial_config
    # Reuse the small CSR fixture without running real-data training.
    config = spatial_config.__wrapped__(tmp_path)
    config['training'].update(epochs=3, early_stopping_patience=1, early_stopping_metric='cd8_loss')
    original_bundle = train.create_data_bundle
    datasets = []

    def bundle(*args, **kwargs):
        result = original_bundle(*args, **kwargs)
        for dataset in result.datasets.values():
            dataset.close = Mock(wraps=dataset.close)
            datasets.append(dataset)
        return result

    monkeypatch.setattr(train, 'create_data_bundle', bundle)
    monkeypatch.setattr(train, 'refresh_loss_curves', Mock())
    monkeypatch.setattr(train, '_format_epoch_summary', lambda *args: 'synthetic epoch')
    epochs = Mock(side_effect=[{'cd8_loss': 1., 'supervised_objective': 1.},
                               {'cd8_loss': 1., 'supervised_objective': 1.},
                               {'cd8_loss': 2., 'supervised_objective': 2.},
                               {'cd8_loss': 2., 'supervised_objective': 2.}])
    monkeypatch.setattr(train, 'run_epoch', epochs)
    downstream = Mock(return_value={})
    monkeypatch.setattr(workflow, 'run_post_training_workflow', downstream)
    best = train.train_model(config)
    assert best.name == 'best.pt'
    assert torch.load(best, weights_only=False)['epoch'] == 0
    assert torch.load(best.with_name('latest.pt'), weights_only=False)['epoch'] == 1
    assert epochs.call_count == 4
    assert all(dataset.close.called for dataset in datasets)
    downstream.assert_not_called()
    requested = copy.deepcopy(config)
    requested['training']['resume_checkpoint'] = str(best.with_name('latest.pt'))
    requested['post_training']['run_analysis'] = False
    resumed_config = train.resume_training_config(config, requested)
    assert resumed_config['post_training'] == requested['post_training']
    assert not any(key.startswith('post_training') for key in
                   train.ignored_resume_settings(requested, resumed_config))
    requested['results']['run_name'] = 'different-run'
    resolve_execution(requested)
    monkeypatch.setattr(train, 'load_config', lambda _: requested)
    assert train.main([]) == 0
    # Resumed early stopping returns the original best without another epoch.
    assert epochs.call_count == 4
    assert all(dataset.close.called for dataset in datasets)
    downstream.assert_called_once_with(requested, best)


def test_generated_inference_provenance_supports_reuse(tmp_path, monkeypatch):
    from dann.tests.test_components import spatial_config
    from dann.tests.test_spatial_execution import checkpoint as make_checkpoint
    config = spatial_config.__wrapped__(tmp_path)
    best = make_checkpoint(config, tmp_path)
    saved = torch.load(best, weights_only=False)
    output = tmp_path / 'prediction.parquet'
    latent = tmp_path / 'latent.parquet'
    workflow.run_inference(Path(config['data']['context_path']), best, output,
                           latent_output_path=latent)
    effective = workflow._workflow_config(config, saved, best)
    effective['spatial'] = dict(output=str(output), latents=str(latent))
    workflow._validate_exports(effective, saved, best, Path(config['data']['context_path']),
                               predictions=True, latents=True)
    with pytest.raises(FileExistsError):
        workflow.run_inference(Path(config['data']['context_path']), best, output,
                               latent_output_path=latent, overwrite=False)


def test_smoke_cli_runs_no_stages(run, stages, monkeypatch):
    from dann import train
    config, best, _ = run
    monkeypatch.setattr(train, 'load_config', lambda _: config)
    monkeypatch.setattr(train, 'train_model', lambda _: best)
    assert train.main(['--smoke-test']) == 0
    assert stages[0] == []


def test_legacy_run_paths(run, stages):
    config, best, checkpoint = run
    config.pop('results')
    config.pop('latent_variance')
    config.pop('spatial')
    requested = best.parent / 'requested'
    config['training']['output_dir'] = str(requested)
    config['analysis']['output_dir'] = str(requested / 'analysis')
    checkpoint['config'].pop('results')
    torch.save(checkpoint, best)
    outputs = workflow.run_post_training_workflow(config, best)
    assert outputs['analysis'] == best.parent / 'analysis'
    assert outputs['latent_variance'] == best.parent / 'analysis/latent_variance'
    assert outputs['inference'] == best.parent / 'analysis/spatial/spatial_predictions.parquet'

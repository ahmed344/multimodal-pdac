"""Promotion preserves scientific state and never writes into the frozen study."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from dann.config import load_config, resolve_execution
from dann.targets import target_contract
from dann.verification import promote_alignment as promotion
from dann.verification import rerun_analysis as downstream


@pytest.fixture
def study(tmp_path):
    source = tmp_path/'experiment'
    results = tmp_path/'results'
    source.mkdir()
    data = tmp_path/'data.h5ad'
    data.write_bytes(b'unchanged input identity')
    (source/'status.json').write_text('{"paused": true}')
    for name, weights in promotion.RUNS.items():
        config = load_config(Path('dann/config.yaml'))
        config['results'].update(root=str(source/'runs'), run_name=name)
        config['model']['aggregation']['type'] = 'mlp'
        config['model']['heads']['biology']['type'] = 'mlp'
        config['model']['heads']['he']['enabled'] = False
        config['model']['heads']['discriminator']['enabled'] = False
        config['model']['latent_dim'] = 512
        config['loss']['he_weight'] = 0.
        config['analysis'].update(max_samples=None, density_mc_samples=1000, exceedance_thresholds=[.01,.02,.03])
        config['data'].update(path=str(data), context_path=str(data))
        for key, weight in zip(('nhsic','anova','soft_nn'), weights):
            config['loss']['batch_alignment'][key].update(weight=weight, enabled=weight>0)
        resolve_execution(config)
        directory = Path(config['training']['output_dir'])
        directory.mkdir(parents=True)
        (directory/'resolved_config.yaml').write_text(yaml.safe_dump(config))
        pd.DataFrame({'epoch':range(1,36),'value':np.arange(35)}).to_csv(directory/'history.csv',index=False)
        (directory/'sampling').mkdir()
        (directory/'sampling/epoch_0000_summary.json').write_text('{"complete": true}')
        checkpoint = dict(config=config, epoch=24, best_epoch=25, target_columns=config['data']['target_columns'],
                          target_transform=target_contract(config), split_indices={'validation':np.arange(166598)},
                          model_state={'weight':torch.tensor([.1,.2])}, optimizer_state={'state':torch.ones(2)},
                          training_rng_state={'numpy':np.random.default_rng(0).bit_generator.state})
        torch.save(checkpoint,directory/'best.pt')
        checkpoint['epoch']=34
        torch.save(checkpoint,directory/'latest.pt')
        (directory/'archives').mkdir()
        for epoch in range(5,36,5):
            checkpoint['epoch']=epoch-1
            torch.save(checkpoint,directory/f'archives/epoch_{epoch:04d}.pt')
    return source, results


def test_complete_promotion_changes_only_paths_and_routes_ordinary_commands(study):
    source, results = study
    before = downstream.inventory(source,hashes=True)
    batch = promotion.prepare(source,results)
    manifest = json.loads((batch/'manifest.json').read_text())
    assert [r['name'] for r in manifest['runs']] == list(promotion.RUNS)
    assert before == downstream.inventory(source,hashes=True)
    for run in manifest['runs']:
        original = source/'runs'/promotion.COMBINATION/run['name']
        destination = Path(run['directory'])
        for path in (destination/'model').rglob('*.pt'):
            old = torch.load(original/'model'/path.relative_to(destination/'model'),weights_only=False)
            new = torch.load(path,weights_only=False)
            expected, _ = promotion.rebase_config(old['config'],original,destination)
            promotion.assert_equal(expected,new['config'])
            new['config']=old['config']
            promotion.assert_equal(old,new)
        best = torch.load(destination/'model/best.pt',weights_only=False)
        promotion.validate_routing(load_config(destination/'model/resolved_config.yaml'),best,destination)
        assert best['epoch'] == 24
        assert len(pd.read_csv(destination/'model/history.csv')) == 35
        assert all('dann.train' not in args for _,args in downstream.commands(run))
    model_before = {r['name']:downstream.inventory(Path(r['directory'])/'model',hashes=True) for r in manifest['runs']}
    assert promotion.prepare(source,results) == batch
    assert model_before == {r['name']:downstream.inventory(Path(r['directory'])/'model',hashes=True) for r in manifest['runs']}


@pytest.mark.parametrize('checkpoint', ['best', 'latest'])
def test_collision_preflight_leaves_first_run_unpublished(study, checkpoint):
    source,results=study
    first,second=list(promotion.RUNS)
    suffix = '__latest_epoch35' if checkpoint == 'latest' else ''
    (results/promotion.COMBINATION/(second+suffix)).mkdir(parents=True)
    with pytest.raises(FileExistsError,match='Conflicting destination'):
        promotion.prepare(source,results,checkpoint)
    assert not (results/promotion.COMBINATION/(first+suffix)).exists()


@pytest.mark.parametrize('checkpoint', ['best', 'latest'])
def test_resume_after_publish_before_manifest_write(study,monkeypatch,checkpoint):
    source,results=study
    real=promotion.publish_model
    calls=[]
    def interrupted(*args):
        value=real(*args)
        calls.append(args[1])
        raise InterruptedError('after atomic publication')
    monkeypatch.setattr(promotion,'publish_model',interrupted)
    with pytest.raises(InterruptedError):
        promotion.prepare(source,results,checkpoint)
    before=downstream.inventory(calls[0]/'model',hashes=True)
    monkeypatch.setattr(promotion,'publish_model',real)
    batch=promotion.prepare(source,results,checkpoint)
    assert downstream.inventory(calls[0]/'model',hashes=True)==before
    assert len(json.loads((batch/'manifest.json').read_text())['runs'])==2


def test_reuse_rejects_changed_source_or_destination(study):
    source,results=study
    batch=promotion.prepare(source,results)
    manifest=json.loads((batch/'manifest.json').read_text())
    model=Path(manifest['runs'][0]['directory'])/'model'
    (model/'history.csv').write_text('changed')
    with pytest.raises(ValueError,match='Promoted model changed'):
        promotion.prepare(source,results)
    (source/'status.json').write_text('{"paused": false}')
    with pytest.raises(ValueError,match='Original experiment changed'):
        promotion.prepare(source,results)


def test_exact_state_check_detects_optimizer_change():
    old={'model':torch.zeros(2),'optimizer':{'momentum':np.zeros(2,dtype=np.float32)}}
    new=copy.deepcopy(old)
    new['optimizer']['momentum'][0]=1
    with pytest.raises(AssertionError):
        promotion.assert_equal(old,new)


def test_latest_preserves_best_results_and_all_scientific_state(study, monkeypatch):
    source, results = study
    best_batch = promotion.prepare(source, results)
    before = downstream.inventory(results, hashes=True)
    # The new batch takes today's root configuration baseline independently.
    original_digest = downstream.digest
    root = Path('dann/config.yaml').resolve()
    monkeypatch.setattr(downstream, 'digest', lambda path:
                        'new-root-baseline' if path.resolve() == root else original_digest(path))
    latest_batch = promotion.prepare(source, results, 'latest')
    manifest = json.loads((latest_batch/'manifest.json').read_text())
    assert manifest['root_config_sha256'] == 'new-root-baseline'
    assert latest_batch.name == '_promotion_alignment_latest_epoch35'
    assert manifest['checkpoint'] == 'latest'
    assert [r['name'] for r in manifest['runs']] == list(promotion.RUNS)
    after = downstream.inventory(results, hashes=True)
    assert all(after[key] == value for key, value in before.items())
    assert json.loads((best_batch/'manifest.json').read_text())['root_config_sha256'] != 'new-root-baseline'
    for run in manifest['runs']:
        destination = Path(run['directory'])
        assert destination.name == run['name']+'__latest_epoch35'
        assert run['evaluated_epoch'] == 35 and run['best_epoch'] == 25
        original = source/'runs'/promotion.COMBINATION/run['name']
        for path in (destination/'model').rglob('*.pt'):
            old = torch.load(original/'model'/path.relative_to(destination/'model'), weights_only=False)
            new = torch.load(path, weights_only=False)
            expected, _ = promotion.rebase_config(old['config'], original, destination, 'latest')
            promotion.assert_equal(expected, new['config'])
            new['config'] = old['config']
            promotion.assert_equal(old, new)
        config = load_config(Path(run['config']))
        selected = promotion.validate_selected(destination/'model', run['name'], 'latest')
        promotion.validate_routing(config, selected, destination, 'latest')
        commands = dict(downstream.commands(run))
        assert 'analyze' in commands
        assert ('spatial' in commands) == config['post_training']['run_inference']
        for name in ('analyze', *(['spatial'] if config['post_training']['run_inference'] else [])):
            args = commands[name]
            assert args[args.index('--checkpoint')+1] == str(destination/'model/latest.pt')
    assert promotion.prepare(source, results, 'latest') == latest_batch
    preserved = results/promotion.COMBINATION/next(iter(promotion.RUNS))/'model/history.csv'
    preserved.write_text('changed previous result')
    with pytest.raises(ValueError, match='Previous promotion results changed'):
        promotion.prepare(source, results, 'latest')


@pytest.mark.parametrize('selected', ['best', 'latest'])
def test_results_checkpoint_routing_and_validation(selected):
    from dann.targets import checkpoint_analysis_config
    config = load_config(Path('dann/config.yaml'))
    config['model']['heads']['he']['enabled'] = False  # Legacy biology-only checkpoint.
    config['results'].pop('analysis_checkpoint')
    resolve_execution(config)
    assert Path(config['analysis']['checkpoint']).name == 'best.pt'
    saved = copy.deepcopy(config)
    checkpoint = dict(config=saved, target_columns=saved['data']['target_columns'],
                      target_transform=target_contract(saved))
    config['results']['analysis_checkpoint'] = selected
    resolve_execution(config)
    assert Path(config['analysis']['checkpoint']).name == selected+'.pt'
    effective = checkpoint_analysis_config(config, checkpoint)
    assert Path(effective['analysis']['checkpoint']).name == selected+'.pt'
    config['results']['analysis_checkpoint'] = 'epoch35'
    with pytest.raises(ValueError, match='results.analysis_checkpoint'):
        resolve_execution(config)


@pytest.mark.parametrize('inference', [False, True])
def test_bounded_checks_use_latest_and_do_not_reuse_best_stages(study, monkeypatch, inference):
    batch = promotion.prepare(*study, 'latest')
    original_load = promotion.load_config
    def configured(path):
        config = original_load(path)
        config['post_training']['run_inference'] = inference
        return config
    monkeypatch.setattr(promotion, 'load_config', configured)
    calls = []
    def execute(batch, manifest, run, stage, args):
        calls.append((run['name'], stage, args))
        directory = Path(run['directory'])/'analysis'
        directory.mkdir(exist_ok=True)
        artifact = directory/('spatial/spatial_predictions.support.parquet' if stage == 'spatial' else 'table.csv')
        artifact.parent.mkdir(exist_ok=True)
        artifact.write_text('bounded')
        run['stages'][stage] = dict(state='completed', checkpoint=run['checkpoint'],
                                   artifacts=downstream.inventory(directory))
        downstream.save_json(batch/'manifest.json', manifest)
        return True
    monkeypatch.setattr(downstream, 'execute', execute)
    promotion.bounded_checks(batch)
    expected = 4 if inference else 2
    assert len(calls) == expected
    for _, _, args in calls:
        assert args[args.index('--checkpoint')+1].endswith('/model/latest.pt')
        assert Path(args[args.index('--checkpoint')+1]).is_file()
    promotion.bounded_checks(batch)
    assert len(calls) == expected
    manifest = json.loads((batch/'manifest.json').read_text())
    for run in manifest['runs']:
        run['bounded_stages']['analyze']['checkpoint'] = 'best'
    downstream.save_json(batch/'manifest.json', manifest)
    promotion.bounded_checks(batch)
    assert len(calls) == 2 * expected

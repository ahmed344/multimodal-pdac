"""Promote exactly the two selected no-H&E MLP fits; never train or resume a sweep.

python -m dann.verification.promote_alignment --prepare
python -m dann.verification.promote_alignment --run  # also resumes valid stages
python -m dann.verification.promote_alignment --audit
python -m dann.verification.promote_alignment --checkpoint latest --run
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import html
import os
import json
from pathlib import Path
import shutil
import signal
from typing import Any

import numpy as np
import torch
import yaml

from dann.config import load_config
from dann.targets import checkpoint_analysis_config
from dann.verification import rerun_analysis as downstream

COMBINATION = 'deep_sets__agg-mlp__bio-mlp__disc-none'
RUNS = {'mlp_nhsic0p1_anova0p003_soft_nn0p1': (.1, .003, .1),
        'mlp_nhsic0p1_anova0_soft_nn0': (.1, 0., 0.)}
SOURCE = Path('data/verification/alignment_no_he_2026-11-04')
RESULTS = Path('data/PDAC/Results/dann')


def assert_equal(left: Any, right: Any, path: str = '') -> None:
    """Require exact scientific state, including tensors, RNG and optimizer arrays."""
    if type(left) is not type(right):
        raise ValueError(f'State type changed: {path}')
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f'State keys changed: {path}')
        for key in left:
            assert_equal(left[key], right[key], f'{path}/{key}')
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f'State length changed: {path}')
        for i, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, f'{path}/{i}')
    elif isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise ValueError(f'Tensor changed: {path}')
    elif isinstance(left, np.ndarray):
        if left.dtype != right.dtype:
            raise ValueError(f'Array dtype changed: {path}')
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif left != right and not (isinstance(left, float) and np.isnan(left) and np.isnan(right)):
        raise ValueError(f'State value changed: {path}')


def rebase_config(config: dict, source: Path, destination: Path,
                  checkpoint: str = 'best') -> tuple[dict, list[dict]]:
    """Change only explicit output routing; retain every scientific/runtime setting."""
    result = copy.deepcopy(config)
    if checkpoint not in ('best', 'latest'):
        raise ValueError('Checkpoint must be best or latest.')
    changes = []
    fields = [('results', 'root', destination.parent.parent),
              ('results', 'run_name', destination.name), ('results', 'run_dir', destination),
              ('training', 'output_dir', destination/'model'),
              ('analysis', 'output_dir', destination/'analysis'),
              ('analysis', 'checkpoint', destination/f'model/{checkpoint}.pt'),
              ('spatial', 'output', destination/'analysis/spatial/spatial_predictions.parquet'),
              ('spatial', 'latents', destination/'analysis/spatial/spatial_latent.parquet'),
              ('latent_variance', 'latents', destination/'analysis/spatial/spatial_latent.parquet'),
              ('latent_variance', 'output_dir', destination/'analysis/latent_variance')]
    for section, key, value in fields:
        old = result[section][key]
        if section != 'results' and not Path(old).resolve().is_relative_to(source):
            raise ValueError(f'Unexpected source routing: {section}.{key}: {old}')
        new = str(value)
        if old != new:
            changes.append(dict(field=f'{section}.{key}', original=old, promoted=new))
            result[section][key] = new
    old = result['results'].get('analysis_checkpoint', 'best')
    if old != checkpoint:
        changes.append(dict(field='results.analysis_checkpoint', original=old, promoted=checkpoint))
    result['results']['analysis_checkpoint'] = checkpoint
    return result, changes


def validate_routing(config: dict, checkpoint: dict, destination: Path,
                     selected: str = 'best') -> None:
    """Check ordinary CLI configuration resolution and checkpoint-derived routing."""
    effective = checkpoint_analysis_config(config, checkpoint)
    for candidate in (config, effective):
        expected = {'training': {'output_dir': destination/'model'},
                    'analysis': {'checkpoint': destination/f'model/{selected}.pt', 'output_dir': destination/'analysis'},
                    'spatial': {'output': destination/'analysis/spatial/spatial_predictions.parquet',
                                'latents': destination/'analysis/spatial/spatial_latent.parquet'},
                    'latent_variance': {'output_dir': destination/'analysis/latent_variance',
                                        'latents': destination/'analysis/spatial/spatial_latent.parquet'}}
        for section, fields in expected.items():
            for key, path in fields.items():
                if Path(candidate[section][key]).resolve() != path:
                    raise ValueError(f'Ordinary CLI routing mismatch: {section}.{key}')


def validate_selected(model: Path, name: str, checkpoint: str = 'best') -> dict:
    """Reject a different fit, epoch, split, history or downstream scientific protocol."""
    import pandas as pd
    best = torch.load(model/'best.pt', map_location='cpu', weights_only=False)
    latest = torch.load(model/'latest.pt', map_location='cpu', weights_only=False)
    if checkpoint not in ('best', 'latest'):
        raise ValueError('Checkpoint must be best or latest.')
    selected = best if checkpoint == 'best' else latest
    c = selected['config']
    weights = tuple(c['loss']['batch_alignment'][n]['weight'] for n in ('nhsic', 'anova', 'soft_nn'))
    if weights != RUNS[name] or best['epoch'] != 24 or latest['epoch'] != 34:
        raise ValueError(f'Unexpected selected checkpoint: {name}')
    if c['model']['latent_dim'] != 512 or c['loss']['he_weight'] != 0:
        raise ValueError('Expected a 512-dimensional no-H&E model.')
    if len(selected['split_indices']['validation']) != 166598 or len(pd.read_csv(model/'history.csv')) != 35:
        raise ValueError('Expected complete validation identities and 35-epoch history.')
    archives = {p.name for p in (model/'archives').glob('epoch_*.pt')}
    if archives != {f'epoch_{e:04d}.pt' for e in range(5, 36, 5)}:
        raise ValueError('Incomplete five-epoch checkpoint archives.')
    if c['analysis']['max_samples'] is not None or c['analysis']['density_mc_samples'] != 1000:
        raise ValueError('Unexpected analysis sampling protocol.')
    if c['analysis']['split'] != 'validation' or len(c['analysis']['exceedance_thresholds']) != 3:
        raise ValueError('Missing frozen validation/CD8 threshold protocol.')
    assert_equal(best['split_indices'], latest['split_indices'], 'split_indices')
    assert_equal(best['target_columns'], latest['target_columns'], 'target_columns')
    assert_equal(best.get('target_transform'), latest.get('target_transform'), 'target_transform')
    assert_equal(best.get('data_identity'), latest.get('data_identity'), 'data_identity')
    if latest.get('best_epoch', 25) != 25:
        raise ValueError('Latest checkpoint lost the supervised-best epoch.')
    return selected


def verify_sources(manifest: dict) -> None:
    """Hash the entire original experiment and root config, including paused sweep evidence."""
    if downstream.inventory(Path(manifest['source']), hashes=True) != manifest['source_files']:
        raise ValueError('Original experiment changed.')
    if downstream.digest(Path(manifest['root_config'])) != manifest['root_config_sha256']:
        raise ValueError('Root configuration changed.')
    for directory, files in manifest.get('preserved_results', {}).items():
        if downstream.inventory(Path(directory), hashes=True) != files:
            raise ValueError(f'Previous promotion results changed: {directory}')
    for path, sha256 in manifest.get('preserved_manifests', {}).items():
        if downstream.digest(Path(path)) != sha256:
            raise ValueError(f'Previous promotion manifest changed: {path}')


def publish_model(source: Path, destination: Path, expected: dict, checkpoint: str = 'best') -> dict:
    """Atomically publish a checked model copy, or reuse an identical prior promotion."""
    provenance = destination/'promotion/provenance.json'
    if destination.exists():
        if not provenance.is_file():
            raise FileExistsError(f'Conflicting promotion destination: {destination}')
        receipt = json.loads(provenance.read_text())
        if (receipt['source'] != str(source) or receipt['original_files'] != expected
                or receipt.get('checkpoint', 'best') != checkpoint):
            raise FileExistsError(f'Promotion source conflict: {destination}')
        if downstream.inventory(destination/'model', hashes=True) != receipt['promoted_files']:
            raise ValueError(f'Promoted model changed: {destination}')
        return receipt
    # The staging name is invocation-owned; only an interrupted, unpublished copy is retried.
    staging = destination.with_name('.'+destination.name+'.promoting')
    if staging.exists():
        owner = staging/'owner.json'
        if not owner.is_file() or json.loads(owner.read_text()) != {'source': str(source)}:
            raise FileExistsError(f'Unowned promotion staging directory: {staging}')
        shutil.rmtree(staging)
    staging.mkdir()
    downstream.save_json(staging/'owner.json', {'source': str(source)})
    shutil.copytree(source/'model', staging/'model', copy_function=shutil.copy2)
    changes = {}
    for path in sorted((staging/'model').rglob('*.pt')):
        old = torch.load(path, map_location='cpu', weights_only=False)
        relocated = copy.deepcopy(old)
        relocated['config'], edits = rebase_config(old['config'], source, destination, checkpoint)
        torch.save(relocated, path)
        restored = torch.load(path, map_location='cpu', weights_only=False)
        assert_equal(relocated, restored)
        restored['config'] = old['config']
        assert_equal(old, restored)
        changes[str(path.relative_to(staging/'model'))] = edits
    resolved = staging/'model/resolved_config.yaml'
    config, edits = rebase_config(yaml.safe_load(resolved.read_text()), source, destination, checkpoint)
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    changes['resolved_config.yaml'] = edits
    best = torch.load(staging/'model/best.pt', map_location='cpu', weights_only=False)
    selected = torch.load(staging/f'model/{checkpoint}.pt', map_location='cpu', weights_only=False)
    validate_routing(load_config(resolved), selected, destination, checkpoint)
    promoted = downstream.inventory(staging/'model', hashes=True)
    if expected.keys() != promoted.keys():
        raise ValueError('Model copy omitted files.')
    for name in expected.keys() - changes.keys():
        if expected[name] != promoted[name]:
            raise ValueError(f'Non-configuration artifact changed: {name}')
    receipt = dict(source=str(source), destination=str(destination), original_files=expected,
                   promoted_files=promoted, configuration_changes=changes,
                   checkpoint=checkpoint, evaluated_epoch=selected['epoch']+1,
                   best_epoch=best['epoch']+1, scientific_state_exact=True,
                   created_utc=datetime.now(timezone.utc).isoformat())
    (staging/'promotion').mkdir()
    downstream.save_json(staging/'promotion/provenance.json', receipt)
    (staging/'owner.json').unlink()
    staging.rename(destination)
    return receipt


def prepare(source: Path = SOURCE, results: Path = RESULTS, checkpoint: str = 'best') -> Path:
    """Prepare an explicit winner-first manifest and preserve complete source evidence."""
    source, results = source.resolve(), results.resolve()
    if checkpoint not in ('best', 'latest'):
        raise ValueError('Checkpoint must be best or latest.')
    suffix = '__latest_epoch35' if checkpoint == 'latest' else ''
    combination = results/COMBINATION
    combination.mkdir(parents=True, exist_ok=True)
    batch = combination/('_promotion_alignment_latest_epoch35' if checkpoint == 'latest' else '_promotion_alignment')
    batch.mkdir(exist_ok=True)
    with (batch/'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = batch/'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest['source'] != str(source) or manifest.get('checkpoint', 'best') != checkpoint:
                raise ValueError('Promotion source or checkpoint changed.')
            verify_sources(manifest)
        else:
            for name in RUNS:
                validate_selected(source/'runs'/COMBINATION/name/'model', name, checkpoint)
                destination = combination/(name+suffix)
                if destination.exists():
                    raise FileExistsError(f'Conflicting destination before promotion: {destination}')
            if shutil.disk_usage(combination).free < 50*1024**3:
                raise OSError('Need at least 50 GiB free for downstream outputs.')
            root_config = Path('dann/config.yaml').resolve()
            preserved = {str(combination/name): downstream.inventory(combination/name, hashes=True)
                         for name in RUNS if checkpoint == 'latest' and (combination/name).exists()}
            old_manifest = combination/'_promotion_alignment/manifest.json'
            preserved_manifests = ({str(old_manifest): downstream.digest(old_manifest)}
                                  if checkpoint == 'latest' and old_manifest.is_file() else {})
            manifest = dict(version=2, checkpoint=checkpoint,
                preserved_results=preserved, preserved_manifests=preserved_manifests,
                source=str(source), source_files=downstream.inventory(source, hashes=True),
                root_config=str(root_config), root_config_sha256=downstream.digest(root_config),
                runs=[], inputs={}, archives=[], archived=True, status='preparing')
            downstream.save_json(path, manifest)
        for name in RUNS:
            original = source/'runs'/COMBINATION/name
            destination = combination/(name+suffix)
            expected = {k.split('/model/', 1)[1]: v for k, v in manifest['source_files'].items()
                        if k.startswith(f'runs/{COMBINATION}/{name}/model/')}
            receipt = publish_model(original, destination, expected, checkpoint)
            evidence = batch/'batch_records'/name
            evidence.mkdir(parents=True, exist_ok=True)
            config_path = evidence/'config.yaml'
            if not config_path.exists():
                shutil.copy2(destination/'model/resolved_config.yaml', config_path)
            existing = next((run for run in manifest['runs'] if run['name'] == name), None)
            if existing and downstream.digest(config_path) != existing['config_sha256']:
                raise ValueError(f'Promoted runtime configuration changed: {name}')
            config = load_config(config_path)
            if name not in {run['name'] for run in manifest['runs']}:
                manifest['runs'].append(dict(name=name, directory=str(destination), config=str(config_path),
                    checkpoint=checkpoint, evaluated_epoch=receipt.get('evaluated_epoch', receipt['best_epoch']), best_epoch=receipt['best_epoch'],
                    config_sha256=downstream.digest(config_path), model=receipt['promoted_files'], stages={}))
                for key in ('path', 'context_path'):
                    input_path = Path(config['data'][key]).resolve()
                    stat = input_path.stat()
                    manifest['inputs'][str(input_path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
                downstream.save_json(path, manifest)
        if [run['name'] for run in manifest['runs']] != list(RUNS):
            raise ValueError('Promotion manifest must contain exactly the two selected runs in order.')
        for run in manifest['runs']:
            if run.get('checkpoint', 'best') != checkpoint:
                raise ValueError('Run checkpoint differs from promotion batch.')
            validate_routing(load_config(Path(run['config'])),
                validate_selected(Path(run['directory'])/'model', run['name'], checkpoint),
                Path(run['directory']), checkpoint)
        if manifest['status'] == 'preparing':
            manifest['status'] = 'prepared'
            downstream.save_json(path, manifest)
        verify_sources(manifest)
    return batch


def bounded_checks(batch: Path) -> None:
    """Run isolated capped real-data analysis and inference before the full stages."""
    manifest = json.loads((batch/'manifest.json').read_text())
    for run in manifest['runs']:
        directory = Path(run['directory'])/'verification/bounded'
        directory.mkdir(parents=True, exist_ok=True)
        config = load_config(Path(run['config']))
        config.pop('results', None)
        config['analysis'].update(output_dir=str(directory/'analysis'), max_samples=256,
                                  latent_interactive=False)
        config['analysis']['peaks']['enabled'] = False
        config['spatial'] = dict(output=str(directory/'analysis/spatial/spatial_predictions.parquet'),
                                latents=str(directory/'analysis/spatial/spatial_latent.parquet'))
        config_path = directory/'config.yaml'
        text = yaml.safe_dump(config, sort_keys=False)
        if config_path.exists() and config_path.read_text() != text:
            raise ValueError('Bounded check configuration changed.')
        if not config_path.exists():
            config_path.write_text(text)
        bounded = dict(name=run['name']+'/bounded', directory=str(directory), config=str(config_path),
                       checkpoint=run.get('checkpoint', 'best'),
                       stages=run.setdefault('bounded_stages', {}))
        checkpoint = str(downstream.checkpoint_path(run))
        commands = [('analyze', ['dann.analyze', '--config', str(config_path), '--checkpoint', checkpoint,
                                '--output-dir', str(directory/'analysis')]),
                    ('spatial', ['dann.spatial', '--config', str(config_path), '--checkpoint', checkpoint,
                                 '--max-rows', '256', '--overwrite'])]
        invalid = False
        for stage, arguments in commands:
            previous = bounded['stages'].get(stage, {})
            support_ready = stage != 'spatial' or (directory/'analysis/spatial/spatial_predictions.support.parquet').is_file()
            if not invalid and support_ready and downstream.same_checkpoint(run, previous) and previous.get('state') == 'completed' and downstream.artifacts_valid(
                    directory/'analysis', previous.get('artifacts', {})):
                continue
            invalid = True
            if not downstream.execute(batch, manifest, bounded, stage, arguments):
                raise RuntimeError(f'Bounded check failed: {run["name"]} {stage}')


def write_reports(batch: Path, manifest: dict, reports: dict) -> None:
    """Publish local result indexes only after every linked principal output exists."""
    principal = {
        'Validation latent UMAP': 'analysis/latent/sample_umap.html',
        'Latent feature correlations': 'analysis/latent/feature_correlations.html',
        'Latent feature UMAP': 'analysis/latent/feature_umap.html',
        'Latent–IHC scatter': 'analysis/latent/latent_ihc_scatter.html',
        'Latent prediction scatter': 'analysis/latent/prediction_scatter.html',
        'Peak similarity viewer': 'analysis/peaks/similarity/global/full_similarity.html',
        'Observed CD8 and families: tissue': 'analysis/peaks/spatial/cd8_comparison/cd8_families_tissue.html',
        'Observed CD8 and families: lesions': 'analysis/peaks/spatial/cd8_comparison/cd8_families_lesion.html',
        'Calibration metrics': 'analysis/calibration/calibration_metrics.csv',
        'Calibration by slide': 'analysis/calibration/calibration_metrics_by_batch.csv',
        'Presence calibration': 'analysis/calibration/calibration_hurdle_reliability.png',
        'Predictive interval coverage': 'analysis/calibration/calibration_interval_coverage.png',
        'Predictive distribution calibration': 'analysis/calibration/calibration_pit.png',
        'CD8 predictive-distribution UMAP': 'analysis/umap/diagnostics/latent_umap_ziln_diagnostics_Density_CD8.png',
        'Complete training curves': 'analysis/loss_curves.png',
        'Tissue predictions': 'analysis/spatial/spatial_predictions.parquet',
        'Tissue latents': 'analysis/spatial/spatial_latent.parquet',
        'Tissue support': 'analysis/spatial/spatial_predictions.support.parquet',
        'Inference provenance': 'analysis/spatial/spatial_predictions.provenance.json',
        'Latent variance summary': 'analysis/latent_variance/latent_variance_summary.csv',
        'Latent variance figure': 'analysis/latent_variance/explained_variance.png',
        'Artifact audit': 'analysis/artifact_audit.json',
        'Promotion provenance': 'promotion/provenance.json',
    }
    indexes = []
    style = '<style>body{font:16px system-ui;max-width:1000px;margin:3em auto;line-height:1.6}a{color:#165aa7}li{margin:.25em 0}</style>'
    for run in manifest['runs']:
        directory = Path(run['directory'])
        report = reports[run['name']]
        links = dict(principal)
        for path in sorted((directory/'analysis/spatial/heatmaps').rglob('*.png')):
            links[path.stem] = str(path.relative_to(directory))
        for label, relative in links.items():
            if not (directory/relative).is_file():
                raise FileNotFoundError(f'Missing principal output: {relative}')
        body = ''.join(f'<li><a href="{html.escape(relative, quote=True)}">{html.escape(label)}</a></li>'
                       for label, relative in links.items())
        title = html.escape(directory.name)
        page = f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>{style}</head><body><h1>{title}</h1>'
        page += (f'<p>Evaluated checkpoint: {run.get("checkpoint", "best")}. '
                 f'Evaluated epoch: {report["evaluated_epoch"]}. '
                 f'Supervised-best checkpoint: epoch 25. Complete training history: 35 epochs. '
                 f'Validation: {report["validation_rows"]:,} rows. Tissue: {report["tissue_rows"]:,} rows '
                 f'across {len(report["slides"])} slides. Latent width: {report["latent_width"]}.</p>'
                 '<p>Frozen preprocessing and CD8 thresholds; 1,000 Monte Carlo draws. H&amp;E supervision disabled. '
                 'Pointwise support denotes the observed center pixel only. Original experiment and paused sweep preserved.</p>')
        page += f'<ul>{body}</ul></body></html>'
        (directory/'index.html').write_text(page)
        downstream.validate_html_links(directory/'index.html')
        indexes.append(f'<li><a href="{html.escape(os.path.relpath(directory/"index.html", batch), quote=True)}">{title}</a></li>')
    (batch/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Promoted MLP results</title>'
                                  +style+'</head><body><h1>Promoted MLP results</h1><ul>'+''.join(indexes)+'</ul></body></html>')
    downstream.validate_html_links(batch/'index.html')


def audit(batch: Path) -> dict:
    """Verify promotion preservation plus every completed downstream artifact contract."""
    manifest = json.loads((batch/'manifest.json').read_text())
    verify_sources(manifest)
    reports = {}
    for run in manifest['runs']:
        directory = Path(run['directory'])
        selected = run.get('checkpoint', 'best')
        if selected != manifest.get('checkpoint', 'best'):
            raise ValueError('Run checkpoint differs from promotion batch.')
        checkpoint = validate_selected(directory/'model', run['name'], selected)
        if (directory/'analysis/he').exists() or list((directory/'analysis').rglob('he_predictions.parquet')):
            raise ValueError('Unexpected H&E performance artifacts for a disabled head.')
        if downstream.inventory(directory/'model', hashes=True) != run['model']:
            raise ValueError('Promoted training artifacts changed.')
        if downstream.digest(Path(run['config'])) != run['config_sha256']:
            raise ValueError('Promoted runtime configuration changed.')
        validate_routing(load_config(Path(run['config'])), checkpoint, directory, selected)
        if not (directory/'analysis/spatial/spatial_predictions.support.parquet').is_file():
            raise ValueError('Promotion requires explicit per-row support information.')
        reports[run['name']] = downstream.audit_outputs(Path(run['config']), write_report=False)
    result = dict(status='passed', original_experiment_unchanged=True, root_config_unchanged=True,
                  previous_results_unchanged=True, checkpoint=manifest.get('checkpoint', 'best'),
                  selected_epochs={name:report['evaluated_epoch'] for name, report in reports.items()},
                  supervised_best_epochs={name:25 for name in RUNS}, runs=reports)
    write_reports(batch, manifest, reports)
    downstream.save_json(batch/'promotion_audit.json', result)
    return result


def main() -> int:
    """Prepare, run/resume, or audit exactly the two authorized promotions."""
    def terminate(signum, frame):
        raise KeyboardInterrupt('Promotion termination requested; preserve completed stages.')
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--results-root', type=Path, default=RESULTS)
    parser.add_argument('--checkpoint', choices=('best', 'latest'), default='best')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', '--resume', action='store_true', dest='run')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if not any((args.prepare, args.run, args.audit)):
        parser.error('Choose --prepare, --run/--resume, or --audit.')
    batch = prepare(args.source, args.results_root, args.checkpoint)
    print(f'PROMOTION {batch}', flush=True)
    # Minimal containers can keep verification-only Chromium dependencies local.
    libraries = batch/'verification/browser_runtime/root/usr/lib/x86_64-linux-gnu'
    if libraries.is_dir():
        os.environ['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None,
            (str(libraries), os.environ.get('LD_LIBRARY_PATH', ''))))
    fonts = batch/'verification/browser_runtime/fonts.conf'
    if fonts.is_file():
        os.environ['FONTCONFIG_FILE'] = str(fonts)
    browsers = batch/'verification/browser_runtime/ms-playwright'
    if browsers.is_dir():
        os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(browsers)
    else:
        installed_cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))
        os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(installed_cache/'ms-playwright'))
    cache = batch/'verification/cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['XDG_CACHE_HOME'] = str(cache)
    if args.run:
        with (batch/'promotion.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            bounded_checks(batch)
            result = downstream.orchestrate(batch)
            if result['status'] != 'completed':
                return 1
            audit(batch)
    elif args.audit:
        audit(batch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

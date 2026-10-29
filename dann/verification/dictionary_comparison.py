"""Restartable matched full dictionary fits using the production training loop.

python -m dann.verification.dictionary_comparison --output NEW_DIR --prepare
python -m dann.verification.dictionary_comparison --output NEW_DIR --run
Stages are isolated processes, sequential on the GPU. The immutable manifest and
configurations are checked before every stage; completion records are separate.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import yaml

from dann.analysis_execution import digest_file, fingerprint
from dann.checkpoints import architecture_contract, checkpoint_config, validate_checkpoint
from dann.config import load_config, resolve_execution, seed_everything
from dann.data_loader import create_data_bundle, stratified_cap_indices
from dann.initialization import initialize_supervised_heads
from dann.model import AdversarialLatentFusion
from dann.spatial import _file_identity
from dann.targets import target_contract
from dann.tiles import row_identity
from dann.train import train_model, move_batch_to_device
from dann.training_artifacts import atomic_json
from dann.verification.stabilization import stage_probe

BASELINE = Path('data/PDAC/Results/dann/deep_sets__agg-mlp__bio-mlp__disc-mlp__he-mlp/he10_biology2_discriminator1_non_centered/model/best.pt')
ARMS = ('baseline', 'l1', 'l2')


def parameter_hash(model: torch.nn.Module) -> str:
    """Fingerprint both registration order and every parameter value."""
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def immutable_json(path: Path, value: dict) -> None:
    """Create immutable experiment metadata; never overwrite an existing manifest."""
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def close(bundle) -> None:
    """Release backed handles after a metadata preflight."""
    for dataset in bundle.datasets.values():
        dataset.close()


def prepare(root: Path, source: Path, baseline_path: Path) -> None:
    """Freeze configurations, splits, initialization and source identities."""
    root.mkdir(parents=True, exist_ok=False)
    for child in ('configs', 'stages', 'logs', 'diagnostics'):
        (root/child).mkdir()
    baseline_path = baseline_path.resolve()
    baseline = torch.load(baseline_path, map_location='cpu', weights_only=False)
    source_config = load_config(source)
    assert architecture_contract(source_config) == architecture_contract(checkpoint_config(baseline)), 'Current architecture differs from the intended baseline'
    assert source_config['loss'] == baseline['config']['loss']
    assert target_contract(source_config) == baseline['target_transform']
    assert baseline['epoch'] + 1 == 116
    (root/'source_config.yaml').write_text(source.read_text())
    # Metadata preflight is backed; resident spectra are only allocated for fitting.
    c = copy.deepcopy(source_config)
    c['performance'] = {}
    bundle = create_data_bundle(c)
    try:
        assert row_identity(Path(c['data']['path']), c['data']) == baseline['data_identity']
        for name, rows in bundle.split_indices.items():
            np.testing.assert_array_equal(rows, baseline['split_indices'][name])
        assert len(bundle.split_indices['train']) == 1528567
        assert len(bundle.split_indices['validation']) == 169861
        np.savez_compressed(root/'splits.npz', **bundle.split_indices)
        train = bundle.split_indices['train']
        probe_rows = stratified_cap_indices(train, bundle.metadata.batch_codes, 65536, 20260719)
        audit_rows = stratified_cap_indices(bundle.split_indices['validation'], bundle.metadata.batch_codes, 2048, 20260719)
        np.save(root/'probe_train_rows.npy', probe_rows)
        np.save(root/'audit_rows.npy', audit_rows)
        j = c['data']['target_columns'].index('Density_CD8')
        y = bundle.metadata.targets[train, j]
        thresholds = np.quantile(y[bundle.metadata.target_valid_mask[train, j] & (y > 0)], [.25, .5, .75]).tolist()
        s = c['data']['target_columns'].index('Density_Stroma')
        stroma = bundle.metadata.targets[train, s]
        quartiles = np.quantile(stroma[bundle.metadata.target_valid_mask[train, s]], [.25, .5, .75]).tolist()
        runs = {}
        initial_hashes, head_hashes = [], []
        raw = yaml.safe_load(source.read_text())
        for arm in ARMS:
            candidate = copy.deepcopy(raw)
            candidate['results'].update(root=str(root/'runs'), run_name=arm)
            candidate['training'].update(seed=20260719, epochs=300, resume_checkpoint=None,
                grl_horizon_updates=300 * int(np.ceil(len(train)/candidate['training']['pixel']['batch_size'])))
            candidate['analysis']['exceedance_thresholds'] = thresholds
            if arm != 'baseline':
                candidate['model']['spectral_encoder']['deep_sets'].update(
                    pooling='intensity_weighted_dictionary', normalization=arm, pool_centering='dictionary_mean')
            path = root/'configs'/f'{arm}.yaml'
            path.write_text(yaml.safe_dump(candidate, sort_keys=False))
            resolved = load_config(path)
            seed_everything(20260719)
            model = AdversarialLatentFusion.from_config(resolved, len(bundle.metadata.batch_names), 4)
            initial_hashes.append(parameter_hash(model))
            stats = initialize_supervised_heads(model, bundle, resolved)
            head_hashes.append(parameter_hash(model))
            assert stats == baseline['head_initialization'], 'Head initialization statistics differ from baseline'
            directory = Path(resolved['results']['run_dir'])
            directory.mkdir(parents=True)
            immutable_json(directory/'expected_initialization.json', dict(
                initial_parameters_sha256=initial_hashes[-1], initialized_heads_sha256=head_hashes[-1], statistics=stats))
            (directory/'resolved_config.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False))
            runs[arm] = dict(config=str(path), config_sha256=digest_file(path), directory=str(directory),
                checkpoint=str(baseline_path if arm == 'baseline' else directory/'model/best.pt'),
                architecture=architecture_contract(resolved))
        assert len(set(initial_hashes)) == len(set(head_hashes)) == 1
        inputs = {key: _file_identity(Path(c['data'][key])) for key in ('path', 'context_path')}
        manifest = dict(version=1, created_utc=datetime.now(timezone.utc).isoformat(), runs=runs,
            baseline=dict(checkpoint=str(baseline_path), sha256=digest_file(baseline_path), selected_epoch=116, stopped_epoch=136),
            inputs=inputs, data_identity=baseline['data_identity'], seed=20260719,
            splits={k:dict(rows=len(v), sha256=fingerprint(v)) for k,v in bundle.split_indices.items()},
            target_transform=target_contract(c), cd8_thresholds=thresholds, stroma_quartiles=quartiles,
            probe_training_rows=65536, initial_parameters_sha256=initial_hashes[0], initialized_heads_sha256=head_hashes[0],
            bootstrap=dict(unit='patient', paired=True, replicates=1000, seed=20260719, includes_training_seed_uncertainty=False),
            software=dict(python=sys.version, torch=str(torch.__version__), gpu=torch.cuda.get_device_name()),
            sources={str(p):digest_file(p) for p in sorted(Path('dann').rglob('*.py'))},
            claims_source='markdowns/2026-10-28-deep-sets-pooling-flaws-and-alternatives.md')
        immutable_json(root/'manifest.json', manifest)
    finally:
        close(bundle)
    print(f'Prepared {root}', flush=True)


def manifest_for(root: Path) -> dict:
    """Read the manifest only after verifying immutable configs and inputs."""
    manifest = json.loads((root/'manifest.json').read_text())
    for run in manifest['runs'].values():
        if digest_file(Path(run['config'])) != run['config_sha256']:
            raise ValueError('Immutable experiment configuration changed')
    if digest_file(Path(manifest['baseline']['checkpoint'])) != manifest['baseline']['sha256']:
        raise ValueError('Historical baseline checkpoint changed')
    c = load_config(Path(manifest['runs']['l1']['config']))
    for key, identity in manifest['inputs'].items():
        if _file_identity(Path(c['data'][key])) != identity:
            raise ValueError(f'Experiment input changed: {key}')
    return manifest


def training_observer(root: Path, arm: str, manifest: dict):
    """Build a read-only initial/early-update observer for the production loop."""
    records = []
    audit = None
    directory = root/'diagnostics'/arm
    directory.mkdir(exist_ok=True)
    def observe(event, model, bundle, c, record):
        nonlocal audit
        if event == 'parameters_initialized':
            assert parameter_hash(model) == manifest['initial_parameters_sha256']
            for key, rows in bundle.split_indices.items():
                assert fingerprint(rows) == manifest['splits'][key]['sha256']
            return
        if event == 'heads_initialized':
            assert parameter_hash(model) == manifest['initialized_heads_sha256']
            atomic_json(directory/'initialization.json', dict(parameters_sha256=parameter_hash(model), **record))
            # A weights-only diagnostic snapshot, never used as a resume checkpoint.
            torch.save(model.state_dict(), directory/'initial_parameters.pt')
        if audit is None:
            from dann.data_loader import SparseAnnDataDataset, sparse_collate
            ds = SparseAnnDataDataset(Path(c['data']['path']), np.load(root/'audit_rows.npy'), bundle.metadata,
                **{k:c['data'][k] for k in ('matrix_key', 'intensity_transform', 'intensity_clip_max', 'nonzero_threshold')},
                resident=getattr(bundle.datasets['train'], 'resident', None))
            try:
                audit = move_batch_to_device(sparse_collate(ds.__getitems__(list(range(len(ds))))), next(model.parameters()).device)
            finally:
                ds.close()
        records.append(dict(event=event, update=record.get('optimizer_updates', 0), **stage_probe(model, audit)))
        atomic_json(directory/'early_diagnostics.json', records)
        print(f'{arm} diagnostic {event} update {record.get("optimizer_updates", 0)}', flush=True)
    return observe


def validate_fit(root: Path, arm: str, manifest: dict) -> dict:
    """Require the configured stopping condition and exact scientific contracts."""
    run = manifest['runs'][arm]
    checkpoint = torch.load(run['checkpoint'], map_location='cpu', weights_only=False)
    c = load_config(Path(run['config']))
    validate_checkpoint(checkpoint, c)
    for name, rows in checkpoint['split_indices'].items():
        assert fingerprint(rows) == manifest['splits'][name]['sha256']
    assert checkpoint['data_identity'] == manifest['data_identity']
    assert checkpoint['target_transform'] == manifest['target_transform']
    if arm != 'baseline':
        latest = torch.load(Path(run['checkpoint']).with_name('latest.pt'), map_location='cpu', weights_only=False)
        stopped = latest['early_stopping_state']['stale_epochs'] >= c['training']['early_stopping_patience']
        assert latest['epoch'] + 1 == 300 or stopped, 'Training has not reached its configured stopping condition'
        assert latest['training_progress']['supervised_rows'] == (latest['epoch'] + 1)*1528567
        assert latest['optimization']['grl_horizon_updates'] == c['training']['grl_horizon_updates']
        summaries = sorted(Path(run['checkpoint']).parent.glob('sampling/epoch_*_summary.json'))
        assert len(summaries) == latest['epoch'] + 1
        matched_orders = []
        for summary_path in summaries:
            summary = json.loads(summary_path.read_text())
            assert summary['complete'] and summary['processed_rows'] == 1528567
            assert summary['batches'] == 747
            reference = Path(manifest['baseline']['checkpoint']).parent/'sampling'/summary_path.name
            if reference.exists():
                historical = json.loads(reference.read_text())
                matched_orders.append(all(summary[key] == historical[key] for key in
                    ('row_order_sha256', 'boundaries_sha256')))
        atomic_json(root/'diagnostics'/arm/'training_coverage.json', dict(
            completed_epochs=len(summaries), rows_per_epoch=1528567, complete=True,
            historical_order_comparable_epochs=len(matched_orders),
            historical_row_and_batch_orders_match=all(matched_orders)))
    for value in checkpoint['model_state'].values():
        assert torch.isfinite(value).all()
    return dict(checkpoint=run['checkpoint'], checkpoint_sha256=digest_file(Path(run['checkpoint'])), selected_epoch=checkpoint['epoch']+1)


def smoke(root: Path, arm: str, c: dict) -> dict:
    """Exercise all downstream stages on small, isolated real-data subsets."""
    import anndata as ad
    from dann.config import apply_smoke_overrides
    from dann.workflow import run_post_training_workflow, _validate_exports
    directory = root/'smoke_data'
    directory.mkdir(exist_ok=True)
    # The same source subsets are used for both arms; source files stay untouched.
    for key, count in [('path', 2048), ('context_path', 512)]:
        destination = directory/f'{key}.h5ad'
        if not destination.exists():
            data = ad.read_h5ad(c['data'][key], backed='r')
            rows = stratified_cap_indices(np.arange(data.n_obs), data.obs['batch'].astype('category').cat.codes.to_numpy(), count, 20260719)
            data[np.sort(rows)].to_memory().write_h5ad(destination)
            data.file.close()
        c['data'][key] = str(destination)
    c = apply_smoke_overrides(c)
    c['training']['grl_horizon_updates'] = 300*747
    c['analysis'].update(figure_dpi=70, family_count=4, heatmap_top_peaks=64, density_mc_samples=32)
    c['analysis']['execution'].update(render_workers=2)
    c['analysis']['spatial_heatmaps'].update(workers=2)
    c['performance'].update(async_artifacts=False)
    path = train_model(c)
    gc.collect(); torch.cuda.empty_cache()
    c['post_training']['enabled'] = True
    run_post_training_workflow(c, path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    _validate_exports(c, checkpoint, path, Path(c['data']['context_path']), predictions=True, latents=True)
    return dict(checkpoint=str(path), rows=512, all_downstream_stages=True)


def downstream(root: Path, arm: str, c: dict, manifest: dict, stage: str) -> dict:
    """Run one production stage or validate reusable historical artifacts."""
    from dann.workflow import _validate_exports, _workflow_config, run_post_training_workflow
    path = Path(manifest['runs'][arm]['checkpoint'])
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    # Baseline outputs are reused only after provenance validation. Missing work
    # goes to this experiment's paths, never to the historical run directory.
    c['analysis']['output_dir'] = str(Path(manifest['runs'][arm]['directory'])/'analysis')
    effective = _workflow_config(c, checkpoint, path)
    effective['analysis']['output_dir'] = c['analysis']['output_dir']
    effective['spatial'] = dict(output=str(Path(c['analysis']['output_dir'])/'spatial/spatial_predictions.parquet'),
                               latents=str(Path(c['analysis']['output_dir'])/'spatial/spatial_latent.parquet'))
    effective['latent_variance']['output_dir'] = str(Path(c['analysis']['output_dir'])/'latent_variance')
    if arm == 'baseline':
        original = checkpoint_config(checkpoint)
        if stage == 'inference':
            _validate_exports(original, checkpoint, path, Path(c['data']['context_path']), predictions=True, latents=True)
            dest = Path(effective['spatial']['output']).parent
            dest.mkdir(parents=True, exist_ok=True)
            for key in ('output', 'latents'):
                link = Path(effective['spatial'][key])
                if not link.exists():
                    link.symlink_to(Path(original['spatial'][key]))
            # Validation uses resolved identities, so the copied provenance remains valid.
            provenance = Path(original['spatial']['output']).with_suffix('.provenance.json')
            link = Path(effective['spatial']['output']).with_suffix('.provenance.json')
            if not link.exists():
                link.symlink_to(provenance)
            return dict(reused=True, provenance=str(provenance))
        if stage == 'analysis':
            original_dir = Path(original['analysis']['output_dir'])
            validate_analysis(original_dir, checkpoint, path)
            return dict(reused=True, path=str(original_dir), checkpoint_sha256=digest_file(path))
        if stage == 'spatial_heatmaps':
            import struct
            directory = Path(original['analysis']['output_dir'])/'spatial/heatmaps'
            report = json.loads((directory/'heatmap_run.json').read_text())
            assert report['sources']['input'] == _file_identity(Path(c['data']['context_path']))
            assert report['sources']['predictions'] == _file_identity(Path(original['spatial']['output']))
            assert report['settings'] == c['analysis']['spatial_heatmaps']
            assert report['densities'] == c['data']['target_columns']
            assert report['dpi'] == c['analysis']['figure_dpi']
            for output in report['outputs']:
                image = Path(output['path'])
                assert image.stat().st_size == output['bytes']
                with image.open('rb') as handle:
                    header = handle.read(24)
                assert header[:8] == b'\x89PNG\r\n\x1a\n'
                assert list(struct.unpack('>II', header[16:24])) == output['size']
            link = Path(effective['analysis']['output_dir'])/'spatial/heatmaps'
            if not link.exists():
                link.symlink_to(directory, target_is_directory=True)
            return dict(reused=True, path=str(directory), images=len(report['outputs']))
    effective['post_training'].update(enabled=True, overwrite_inference=True,
        **{f'run_{s}':s == stage for s in ('analysis','inference','spatial_heatmaps','latent_variance')})
    outputs = run_post_training_workflow(effective, path)
    if stage == 'inference':
        _validate_exports(effective, checkpoint, path, Path(c['data']['context_path']), predictions=True, latents=True)
    if stage == 'analysis':
        validate_analysis(Path(effective['analysis']['output_dir']), checkpoint, path)
    return {k:str(v) for k,v in outputs.items()}


def validate_analysis(directory: Path, checkpoint: dict, path: Path) -> None:
    """Verify cached numerical payloads, full split coverage and required renders."""
    report = json.loads((directory/'analysis_execution.json').read_text())
    assert report['source_identity']['checkpoint'] == digest_file(path)
    assert report['source_identity']['identity'] == checkpoint['data_identity']
    current = _file_identity(Path(checkpoint['config']['data']['path']))
    assert report['source_identity']['source'][:3] == [current['path'], current['bytes'], current['mtime_ns']]
    assert report['output_rows']['validation'] == len(checkpoint['split_indices']['validation'])
    cache = directory/'.analysis-cache/extracted-validation'/report['cache_entries']['extracted-validation']
    saved = json.loads((cache/'manifest.json').read_text())
    for name, info in saved['arrays'].items():
        assert digest_file(cache/info['file']) == info['sha256']
        array = np.load(cache/info['file'], mmap_mode='r')
        if name in ('latent', 'pi', 'mu', 'sigma', 'he_squared_error'):
            assert np.isfinite(array).all(), f'Nonfinite analysis output: {name}'
        if name == 'row_ids':
            np.testing.assert_array_equal(np.sort(array), np.sort(checkpoint['split_indices']['validation']))
    assert all((directory/name).is_file() for name in report['artifacts'])


def execute_stage(root: Path, arm: str, stage: str) -> None:
    """Record completion only after validation; preserve restartable failures."""
    manifest = manifest_for(root)
    c = load_config(Path(manifest['runs'][arm]['config']))
    stamp = root/'stages'/f'{arm}-{stage}.json'
    identity = dict(config_sha256=manifest['runs'][arm]['config_sha256'], stage=stage, arm=arm)
    if stage not in ('smoke','train'):
        identity['checkpoint_sha256'] = digest_file(Path(manifest['runs'][arm]['checkpoint']))
    if stamp.exists():
        saved = json.loads(stamp.read_text())
        if saved.get('identity') == identity and saved['status'] == 'complete':
            if stage == 'train':
                result = validate_fit(root, arm, manifest)
                assert result == saved['result'], 'Completed checkpoint identity changed'
            elif stage == 'diagnostics':
                result = saved['result']
                assert Path(result['metrics']).is_file()
                assert json.loads(Path(result['metrics']).read_text())['identity']['checkpoint_sha256'] == identity['checkpoint_sha256']
            elif stage in ('analysis', 'inference'):
                # Reuse validation is read-only; never rerun a complete export.
                from dann.workflow import _validate_exports
                checkpoint = torch.load(manifest['runs'][arm]['checkpoint'], map_location='cpu', weights_only=False)
                if stage == 'analysis':
                    directory = Path(saved['result'].get('path', saved['result'].get('analysis', c['analysis']['output_dir'])))
                    validate_analysis(directory, checkpoint, Path(manifest['runs'][arm]['checkpoint']))
                else:
                    _validate_exports(c, checkpoint, Path(manifest['runs'][arm]['checkpoint']),
                                      Path(c['data']['context_path']), predictions=True, latents=True)
            print(f'Reusing completed {arm} {stage}', flush=True)
            return
    start = time.monotonic()
    atomic_json(stamp, dict(identity=identity, status='running', started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        if stage == 'smoke':
            result = smoke(root, arm, c)
        elif stage == 'train':
            latest = Path(c['training']['output_dir'])/'latest.pt'
            if latest.exists():
                c['training']['resume_checkpoint'] = str(latest)
            train_model(c, observer=training_observer(root, arm, manifest))
            result = validate_fit(root, arm, manifest)
        elif stage == 'diagnostics':
            validate_fit(root, arm, manifest)
            from dann.verification.pooling_diagnostics import run_diagnostics
            result = run_diagnostics(root, arm, manifest)
        else:
            validate_fit(root, arm, manifest)
            result = downstream(root, arm, c, manifest, stage)
        atomic_json(stamp, dict(identity=identity, status='complete', seconds=time.monotonic()-start, result=result,
            sources={str(p):digest_file(p) for p in sorted(Path('dann').rglob('*.py'))}))
    except BaseException as error:
        atomic_json(stamp, dict(identity=identity, status='failed', seconds=time.monotonic()-start,
                               error=repr(error), traceback=traceback.format_exc()))
        raise


def run_all(root: Path) -> None:
    """Execute isolated stages sequentially, then assemble the comparison report."""
    # Training remains sequential; diagnostics and exports use the selected weights.
    stages = [(a, 'smoke') for a in ('l1','l2')] + [(a, 'train') for a in ('l1','l2')]
    stages += [(a, s) for a in ARMS for s in ('analysis','inference','spatial_heatmaps','latent_variance','diagnostics')]
    for arm, stage in stages:
        with (root/'logs'/f'{arm}-{stage}.log').open('a') as log:
            print(f'Starting {arm} {stage}', flush=True)
            subprocess.run([sys.executable, '-m', 'dann.verification.dictionary_comparison', '--output', str(root), '--arm', arm, '--stage', stage],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    from dann.verification.pooling_diagnostics import compare
    compare(root)


def main() -> None:
    """Run the comparison command-line interface."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--stage', choices=('smoke','train','analysis','inference','spatial_heatmaps','latent_variance','diagnostics'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.output.resolve()
    if args.prepare:
        prepare(root, args.source, args.baseline)
    if args.stage:
        execute_stage(root, args.arm, args.stage)
    if args.run:
        run_all(root)


if __name__ == '__main__':
    main()

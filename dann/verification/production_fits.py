"""Sequential, isolated production fits with a validation-only downstream gate.

Prepare and execute: python -m dann.verification.production_fits --output NEW_DIR
The manifest, baseline and sampler hashes are persisted before any training.
Existing output directories are never reused; failed fits are never retried.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml

from dann.checkpoints import (architecture_contract, sampling_contract, validate_checkpoint,
                              validate_sampling_contract, validate_training_data_contract)
from dann.config import load_config
from dann.data_loader import create_data_bundle
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.targets import checkpoint_contract, target_contract
from dann.tiles import row_identity
from dann.train import run_epoch
from dann.verification.fit_diagnosis import constant_parameters

ARCHIVE = Path('data/PDAC/Results/dann/diagnostics/2026-10-01-batch-execution-1000')
DOWNSTREAM = ('dann.analyze', 'dann.spatial', 'dann.spatial_heatmaps', 'dann.latent_variance')


def save_json(path: Path, value: Any) -> None:
    """Atomically persist strict JSON, including failures and in-progress stages."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(rows: Any) -> str:
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()


def close_bundle(bundle: Any) -> None:
    for loader in bundle.loaders.values():
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
    for dataset in bundle.datasets.values():
        dataset.close()


def assess(metrics: dict, baseline: dict) -> dict:
    """Apply all three prespecified validation conditions, rejecting nonfinite scores."""
    keys = ('cd8_loss', 'biology_loss', 'mean_r2_cd8')
    finite = all(math.isfinite(metrics[k]) for k in keys)
    checks = dict(finite=finite,
                  cd8=finite and metrics['cd8_loss'] <= baseline['cd8_loss'] - .001,
                  biology=finite and metrics['biology_loss'] < baseline['biology_loss'],
                  positive_cd8_logit_r2=finite and metrics['mean_r2_cd8'] > 0)
    return dict(passed=all(checks.values()), checks=checks,
                cd8_improvement=baseline['cd8_loss'] - metrics['cd8_loss'] if finite else None,
                biology_improvement=baseline['biology_loss'] - metrics['biology_loss'] if finite else None)


def fit_baseline(bundle: Any, config: dict) -> dict:
    """Fit train-only hurdle MLEs and score all validation labels with production loss."""
    rows = bundle.split_indices
    y = torch.from_numpy(bundle.metadata.targets[rows['train']])
    mask = torch.from_numpy(bundle.metadata.target_valid_mask[rows['train']])
    params = {k: v[0].clone() for k, v in constant_parameters(
        y, mask, config['loss']['logit_epsilon'], config['model']['sigma_min']).items()}
    y = torch.from_numpy(bundle.metadata.targets[rows['validation']])
    mask = torch.from_numpy(bundle.metadata.target_valid_mask[rows['validation']])
    out = {k: v.expand_as(y) for k, v in params.items()}
    loss = ZILNLoss.from_config(config)
    biology = loss(*(out[k] for k in ('pi_logits', 'mu', 'sigma')), y, mask)
    j = config['data']['target_columns'].index('Density_CD8')
    cd8 = ZILNLoss([1.], loss.logit_epsilon, 'mean', loss.include_normal_constant)(
        *(out[k][:, j:j+1] for k in ('pi_logits', 'mu', 'sigma')), y[:, j:j+1], mask[:, j:j+1])
    z = torch.logit(y[mask[:, j] & (y[:, j] > 0), j].double().clamp(
        loss.logit_epsilon, 1-loss.logit_epsilon))
    r2 = 1 - ((z-params['mu'][j])**2).sum() / ((z-z.mean())**2).sum()
    validation = dict(cd8_loss=float(cd8.total), biology_loss=float(biology.total), mean_r2_cd8=float(r2))
    return dict(fit_split='train', score_split='validation', validation=validation,
                parameters={k: v.tolist() for k, v in params.items()},
                target_transform=target_contract(config), loss=config['loss'],
                thresholds=dict(cd8_loss_max=validation['cd8_loss']-.001,
                                biology_loss_strict_max=validation['biology_loss'],
                                positive_cd8_logit_r2_strict_min=0.0))


def sampler_evidence(bundle: Any) -> dict:
    """Hash a full first epoch and verify exact coverage and the retained partial batch."""
    requests = list(bundle.loaders['train'].sampler)
    rows = np.concatenate([r.row_ids for r in requests])
    boundaries = np.cumsum([0] + [len(r.row_ids) for r in requests])
    np.testing.assert_array_equal(np.sort(rows), np.sort(bundle.split_indices['train']))
    return dict(row_order_sha256=digest(rows), boundaries_sha256=digest(boundaries),
                tile_order_sha256=digest([t for r in requests for t in r.tile_ids]),
                batches=len(requests), final_batch_rows=len(requests[-1].row_ids),
                processed_rows=len(rows))


def gpu_snapshot() -> dict:
    def query(arguments):
        return subprocess.check_output(['nvidia-smi', *arguments], text=True).strip()
    return dict(device=query(['--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader']),
                compute_pids=query(['--query-compute-apps=pid', '--format=csv,noheader']))


def prepare(source: Path, output: Path, archive: Path) -> dict:
    """Resolve isolated configs and establish immutable pre-training evidence."""
    output.mkdir(parents=True, exist_ok=False)
    raw = yaml.safe_load(source.read_text())
    (output / 'source_config.yaml').write_text(source.read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + os.urandom(3).hex()
    runs = []
    for name, aggregation, halo in [('A', 'cnn', 6), ('B', 'mlp', 3)]:
        candidate = copy.deepcopy(raw)
        candidate['model']['aggregation']['type'] = aggregation
        candidate['results']['run_name'] = f'production-{stamp}-{name}'
        path = output / f'{name}.yaml'
        path.write_text(yaml.safe_dump(candidate, sort_keys=False))
        config = load_config(path)
        t = config['training']
        assert config['execution']['halo'] == halo
        assert config['model']['heads']['biology']['type'] == 'cnn'
        assert config['model']['heads']['discriminator']['type'] == 'mlp'
        assert (t['seed'], t['epochs'], t['early_stopping_patience'], t['early_stopping_min_delta']) == (20260719, 300, 5, .001)
        assert (t['core_size'], t['supervised_rows_per_batch'], t['sampling_strategy']) == (8, 2048, 'proportional_slide_tiles')
        assert t['resume_checkpoint'] is None and not config['results'].get('smoke', False)
        assert all(config['data'].get(f'max_{s}_samples') is None for s in ('train', 'validation', 'test'))
        assert config['analysis']['density_mc_samples'] == 1000 and config['analysis']['max_samples'] is None
        directory = Path(config['results']['run_dir'])
        directory.mkdir(parents=True, exist_ok=False)
        for key in ('path', 'context_path'):
            assert Path(config['data'][key]).is_file()
        bundle = create_data_bundle(config)
        try:
            identity = row_identity(Path(config['data']['path']), config['data'])
            assert identity == json.loads((archive / 'data_identity.json').read_text())
            with np.load(archive / 'splits.npz') as saved:
                for key, values in bundle.split_indices.items():
                    np.testing.assert_array_equal(values, saved[key])
            sampling = sampler_evidence(bundle)
            from dann.verification.batch_execution_diagnosis import Plan
            reference = Plan.load(archive / 'plan_20260719_Q.npz')
            batches = sampling['batches']
            assert sampling['row_order_sha256'] == digest(reference.rows[:len(bundle.split_indices['train'])])
            assert sampling['boundaries_sha256'] == digest(reference.boundaries[:batches+1])
            splits = {k: dict(rows=len(v), sha256=digest(v)) for k, v in bundle.split_indices.items()}
            if name == 'A':
                baseline = fit_baseline(bundle, config)
                archived = json.loads((archive / 'constant_baseline.json').read_text())
                for key in baseline['parameters']:
                    np.testing.assert_allclose(baseline['parameters'][key], archived['parameters'][key], atol=2e-6, rtol=1e-6)
                for current, old in [('cd8_loss', 'cd8'), ('biology_loss', 'biology')]:
                    np.testing.assert_allclose(baseline['validation'][current], archived['validation'][old], atol=2e-6, rtol=1e-6)
                baseline['archive_comparison'] = dict(path=str(archive.resolve()), data_identity_matches=True,
                    splits_match=True, parameters_and_losses_match=True, archived_validation=archived['validation'])
                save_json(output / 'constant_baseline.json', baseline)
                save_json(output / 'data_identity.json', identity)
                np.savez_compressed(output / 'splits.npz', **bundle.split_indices)
            else:
                assert splits == runs[0]['splits'] and sampling == runs[0]['sampling']
            runs.append(dict(name=name, config=str(path.resolve()), directory=str(directory.resolve()),
                             architecture=architecture_contract(config), splits=splits, sampling=sampling,
                             sampling_contract=sampling_contract(config, bundle.split_indices['train'])))
        finally:
            close_bundle(bundle)
    assert torch.cuda.is_available()
    snapshot = gpu_snapshot()
    manifest = dict(created_utc=stamp, runs=runs, baseline=str(output / 'constant_baseline.json'),
                    gpu=snapshot, disk_free_bytes=shutil.disk_usage(output).free,
                    python=sys.executable, torch=torch.__version__, status='prepared')
    save_json(output / 'manifest.json', manifest)
    return manifest


class CoverageLoader:
    """Track actual complete validation coverage without changing spatial batches."""
    def __init__(self, loader):
        self.loader, self.rows = loader, []
        self.dataset, self.sampler = loader.dataset, loader.sampler

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            self.rows.extend(batch['row_ids'].tolist())
            yield batch


def evaluate(config_path: Path, evidence: Path) -> None:
    """Independently score best.pt on every saved validation row with full halos."""
    config = load_config(config_path)
    path = Path(config['analysis']['checkpoint'])
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    validate_training_data_contract(checkpoint, config)
    validate_sampling_contract(checkpoint, config)
    assert checkpoint_contract(checkpoint) == target_contract(config)
    with np.load(evidence / 'splits.npz') as saved:
        for key, rows in checkpoint['split_indices'].items():
            np.testing.assert_array_equal(rows, saved[key])
    bundle = create_data_bundle(config, checkpoint['split_indices'], checkpoint['data_identity'])
    try:
        model = AdversarialLatentFusion.from_config(config, len(bundle.metadata.batch_names), len(config['data']['target_columns']))
        validate_checkpoint(checkpoint, config, model, optimizer=True)
        model.load_state_dict(checkpoint['model_state'])
        device = torch.device(config['training']['device'])
        model.to(device)
        loader = CoverageLoader(bundle.loaders['validation'])
        t, loss = config['training'], config['loss']
        torch.cuda.reset_peak_memory_stats()
        metrics = run_epoch(model, loader, ZILNLoss.from_config(config).to(device), device,
            loss['biology_weight'], loss['batch_weight'],
            bundle.batch_class_weights.to(device) if loss['balance_batch_classes'] else None,
            None, checkpoint['epoch'], t['epochs'], t['grl_schedule'], t['grl_max_lambda'],
            t['grl_gamma'], None, target_columns=config['data']['target_columns'])
        np.testing.assert_array_equal(np.sort(loader.rows), np.sort(bundle.split_indices['validation']))
        assert all(math.isfinite(v) for v in metrics.values())
        history = pd.read_csv(Path(t['output_dir']) / 'history.csv')
        best_row = history.loc[history.epoch == checkpoint['epoch'] + 1].iloc[0]
        for key in ('cd8_loss', 'biology_loss', 'mean_r2_cd8'):
            np.testing.assert_allclose(metrics[key], best_row['validation_' + key], atol=1e-5, rtol=1e-5)
        baseline = json.loads((evidence / 'constant_baseline.json').read_text())['validation']
        result = dict(checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            best_epoch=checkpoint['epoch']+1, stopping_epoch=int(history.epoch.iloc[-1]),
            validation=metrics, assessment=assess(metrics, baseline),
            validation_rows=len(loader.rows), validation_coverage_sha256=digest(np.sort(loader.rows)),
            full_spatial_execution=True, halo=config['execution']['halo'],
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        save_json(Path(config['results']['run_dir']) / 'assessment.json', result)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        close_bundle(bundle)


def verify_outputs(config_path: Path) -> dict:
    """Check completed exports, all configured heatmaps and finite variance tables."""
    from dann.spatial import read_axis_names, validate_output, validate_latent_output
    config = load_config(config_path)
    expected = len(read_axis_names(Path(config['data']['context_path']), 'obs'))
    input_path = Path(config['data']['context_path'])
    validate_output(Path(config['spatial']['output']), input_path, config['data']['target_columns'], expected)
    validate_latent_output(Path(config['spatial']['latents']), input_path, config['model']['latent_dim'], expected)
    heatmaps = Path(config['analysis']['output_dir']) / 'spatial/heatmaps'
    images = [heatmaps / f'spatial_heatmap_{prefix}{target}.png'
              for target in config['data']['target_columns'] for prefix in ('', 'diagnostics_')]
    assert all(path.is_file() and path.stat().st_size > 0 for path in images)
    variance_dir = Path(config['latent_variance']['output_dir'])
    for filename in ('latent_variance_summary.csv', 'pca_variance.csv'):
        values = pd.read_csv(variance_dir / filename).select_dtypes(include='number')
        assert np.isfinite(values.to_numpy()).all()
    summary = pd.read_csv(variance_dir / 'latent_variance_summary.csv').iloc[0].to_dict()
    assert int(summary['n_observations']) == expected
    return dict(spatial_prediction_rows=expected, spatial_latent_rows=expected,
                heatmaps=[str(p) for p in images], variance=summary)


def execute_stage(run: dict, stage: str, command: list[str]) -> bool:
    """Run one subprocess, persisting logs, sampled GPU use and time/RSS accounting."""
    directory = Path(run['directory'])
    status_path = directory / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else dict(stages={})
    if stage in status['stages']:
        raise RuntimeError(f'Refusing to repeat stage {stage} in {directory}')
    while gpu_snapshot()['compute_pids']:
        status['waiting_for_gpu'] = gpu_snapshot()
        save_json(status_path, status)
        time.sleep(30)
    status.pop('waiting_for_gpu', None)
    entry = dict(command=command, started_utc=datetime.now(timezone.utc).isoformat(), state='running')
    status['stages'][stage] = entry
    save_json(status_path, status)
    start = time.monotonic()
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg')
    with (directory / f'{stage}.log').open('w') as log, (directory / f'{stage}.gpu.jsonl').open('w') as gpu_log:
        process = subprocess.Popen([sys.executable, '-m', 'dann.verification.production_fits',
                                    '--worker', str(directory / f'{stage}.resources.json'), *command[2:]],
                                   stdout=log, stderr=subprocess.STDOUT, env=env)
        entry['pid'] = process.pid
        save_json(status_path, status)
        while True:
            snapshot = dict(seconds=time.monotonic()-start, **gpu_snapshot(), disk_free_bytes=shutil.disk_usage(directory).free)
            gpu_log.write(json.dumps(snapshot) + '\n')
            gpu_log.flush()
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                continue
    entry.update(state='completed' if code == 0 else 'failed', returncode=code, seconds=time.monotonic()-start)
    save_json(status_path, status)
    return code == 0


def orchestrate(manifest: dict, evidence: Path, runner: Callable = execute_stage,
                verifier: Callable = verify_outputs) -> dict:
    """Attempt A then B once; route only successful validation fits downstream."""
    results = {}
    for run in manifest['runs']:
        command = [sys.executable, '-m', 'dann.train', '--config', run['config']]
        results[run['name']] = dict(training='completed' if runner(run, 'train', command) else 'failed')
        save_json(evidence / 'results.json', results)
    for run in manifest['runs']:
        result = results[run['name']]
        if result['training'] != 'completed':
            result['downstream'] = 'skipped_training_failure'
            continue
        command = [sys.executable, '-m', __name__ if __name__ != '__main__' else 'dann.verification.production_fits',
                   '--evaluate', '--config', run['config'], '--output', str(evidence)]
        if not runner(run, 'validation', command):
            result['downstream'] = 'skipped_validation_error'
        else:
            assessment = json.loads((Path(run['directory']) / 'assessment.json').read_text())
            result.update(assessment=assessment)
            result['downstream'] = 'skipped_failed_criteria'
            if assessment['assessment']['passed']:
                result['downstream'] = 'completed'
                for module in DOWNSTREAM:
                    if not runner(run, module.split('.')[-1], [sys.executable, '-m', module, '--config', run['config']]):
                        result['downstream'] = 'failed_' + module
                        break
                if result['downstream'] == 'completed':
                    try:
                        result['verification'] = verifier(Path(run['config']))
                    except Exception:
                        result.update(downstream='failed_verification', error=traceback.format_exc())
        save_json(evidence / 'results.json', results)
    save_json(evidence / 'results.json', results)
    return results


def resource_worker(arguments: list[str]) -> None:
    """Execute an unchanged module CLI and account for this process and its workers."""
    import resource
    import runpy
    destination, module, *module_args = arguments
    sys.argv = [module, *module_args]
    started = time.monotonic()
    try:
        runpy.run_module(module, run_name='__main__')
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        summary = dict(seconds=time.monotonic()-started, max_rss_kib=usage.ru_maxrss,
                       children_max_rss_kib=children.ru_maxrss,
                       user_seconds=usage.ru_utime+children.ru_utime,
                       system_seconds=usage.ru_stime+children.ru_stime)
        if torch.cuda.is_initialized():
            summary.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        save_json(Path(destination), summary)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        resource_worker(sys.argv[2:])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.evaluate:
        evaluate(args.config, args.output)
        return 0
    manifest = prepare(args.config, args.output.resolve(), args.archive)
    print('Preflight complete; baseline and immutable configurations saved.', flush=True)
    orchestrate(manifest, args.output.resolve())
    manifest['status'] = 'finished'
    save_json(args.output / 'manifest.json', manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

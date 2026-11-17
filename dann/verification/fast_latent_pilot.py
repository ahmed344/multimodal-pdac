"""Benchmark real spatial batches and run three complete epochs of the fast model.

python -m dann.verification.fast_latent_pilot --output NEW_REPORT_DIRECTORY
The configured run directory must also be new. Scheduling keeps the configured
full-fit horizon; the pilot stops at epoch three and saves resumable checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from dann.batch_alignment import NAMES
from dann.calibration import compute_target_metrics
from dann.checkpoints import checkpoint_config, validate_checkpoint
from dann.conditional_alignment import BiologyConditioner
from dann.config import load_config, resolve_device, resolve_execution
from dann.data_loader import create_data_bundle
from dann.execution import evaluation_groups
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.progress import EpochProgress
from dann.train import run_epoch, train_model
from dann.training_artifacts import atomic_json
from dann.verification.execution_benchmark import Requests, train_loader
from dann.ziln import exceedance_probability


def portable(value):
    """Represent unavailable diagnostics as null in portable JSON reports."""
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def require_finite_gradients(model) -> float:
    """Validate present gradients and return the whole-model postclip norm."""
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise FloatingPointError('Missing or nonfinite pilot gradients.')
    return float(torch.linalg.vector_norm(torch.stack([g.double().norm() for g in gradients])))


def benchmark(model, bundle, config) -> dict:
    """Measure median/largest epoch-zero peak footprints with full monitoring.

    Each arm uses one warmup and three measured updates, including AdamW,
    clipping, all enabled objectives, and gradient monitoring on every update.
    The pilot model, sampler, and optimizer are not mutated.
    """
    device = next(model.parameters()).device
    requests = list(bundle.loaders['train'].sampler)
    tiles = bundle.datasets['train']
    counts = np.diff(tiles.resident.indptr)
    tile_peaks = np.array([counts[g['context_rows']].sum() for g in tiles.context_rows])
    footprints = np.array([tile_peaks[list(r.tile_ids)].sum() for r in requests])
    order = np.argsort(footprints)
    conditioner = BiologyConditioner.restore(config['alignment_conditioner'], config, bundle.split_indices['train'])
    result = dict(epoch_updates=len(requests), epoch_peak_visits=int(footprints.sum()), arms={})
    for name, position in [('representative', order[len(order) // 2]), ('largest', order[-1])]:
        candidate = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(config['training']['beta1'], config['training']['beta2']))
        loader = train_loader(tiles, config, Requests([requests[position]] * 4))
        trials = []

        def observe(record):
            trials.append(dict(seconds=record['seconds'], gradient_norm=require_finite_gradients(candidate)))

        metrics = run_epoch(candidate, loader, ZILNLoss.from_config(config).to(device), device,
            biology_weight=config['loss']['biology_weight'], batch_weight=config['loss']['batch_weight'],
            he_weight=config['loss']['he_weight'], class_weights=bundle.batch_class_weights.to(device),
            optimizer=optimizer, epoch_index=2, total_epochs=config['training']['epochs'],
            grl_schedule=config['training']['grl_schedule'], grl_max_lambda=config['training']['grl_max_lambda'],
            grl_gamma=config['training']['grl_gamma'], gradient_clip_norm=config['training']['gradient_clip_norm'],
            target_columns=config['data']['target_columns'],
            cd8_index=config['data']['target_columns'].index('Density_CD8'),
            monitoring=dict(enabled=True, gradient_updates_per_epoch=None),
            batch_alignment=config['loss']['batch_alignment'], alignment_conditioner=conditioner,
            alignment_seed=config['training']['seed'], update_callback=observe)
        result['arms'][name] = dict(request_index=int(position), peak_visits=int(footprints[position]),
            tiles=len(requests[position].tile_ids), warmup=trials[0], trials=trials[1:],
            median_seconds=float(np.median([r['seconds'] for r in trials[1:]])),
            metrics=metrics)
        print(json.dumps(portable({'benchmark': name, **{k: v for k, v in result['arms'][name].items() if k != 'metrics'}})), flush=True)
        del candidate, optimizer, loader
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return result


def evaluate_export(checkpoint: Path, output: Path) -> dict:
    """Score all held-out CD8 distributions and export row-aligned 256D latents."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint_config(saved)
    device = resolve_device(config['training']['device'])
    bundle = create_data_bundle(config, saved['split_indices'], saved['data_identity'])
    model = AdversarialLatentFusion.from_config(config, len(saved['batch_names']), len(config['data']['target_columns'])).to(device).eval()
    validate_checkpoint(saved, config, model)
    model.load_state_dict(saved['model_state'])
    count, width = len(bundle.split_indices['validation']), config['model']['latent_dim']
    latents = np.lib.format.open_memmap(output / 'validation_latent.npy', mode='w+', dtype=np.float32, shape=(count, width))
    parts, position = [], 0
    progress = EpochProgress(saved['epoch'] + 1, config['training']['epochs'], config['training']['progress_interval_seconds'])
    index = config['data']['target_columns'].index('Density_CD8')
    try:
        with torch.inference_mode():
            for _, batch, values in evaluation_groups(model, bundle.loaders['validation'], device, progress):
                n = len(batch['row_ids'])
                latents[position:position+n] = values['latent'].cpu().numpy()
                parts.append(torch.stack([batch['row_ids'].double(), batch['targets'][:, index].double(),
                    batch['target_valid_mask'][:, index].double(), values['pi_logits'][:, index].sigmoid().double(),
                    values['mu'][:, index].double(), values['sigma'][:, index].double()], 1).cpu().numpy())
                position += n
                progress.update(position)
        progress.finish_phase()
        latents.flush()
        array = np.concatenate(parts)
        expected = np.sort(bundle.split_indices['validation'])
        if position != count or not np.array_equal(np.sort(array[:, 0].astype(np.int64)), expected):
            raise ValueError('Held-out export coverage or row identity mismatch.')
        np.save(output / 'validation_row_ids.npy', array[:, 0].astype(np.int64))
        truth, valid, pi, mu, sigma = array[:, 1:].T
        truth = np.where(valid, truth, np.nan)
        metrics = compute_target_metrics(truth, pi, mu, sigma, config['loss']['logit_epsilon'])
        exceedance = {}
        for threshold in (.01, .05, .1):
            probability = exceedance_probability(pi, mu, sigma, threshold)
            keep = valid.astype(bool)
            exceedance[str(threshold)] = dict(brier=float(np.mean((probability[keep] - (truth[keep] > threshold))**2)),
                observed_rate=float(np.mean(truth[keep] > threshold)), predicted_rate=float(np.mean(probability[keep])))
        metadata = dict(checkpoint=str(checkpoint), epoch=saved['epoch']+1, split='validation',
            latent_dim=width, shape=[count, width], dtype='float32', latent='exported aggregation before heads',
            rows='validation_row_ids.npy', values='validation_latent.npy')
        atomic_json(output / 'latent_metadata.json', metadata)
        return dict(metadata=metadata, cd8=metrics, exceedance=exceedance)
    finally:
        del latents
        for dataset in bundle.datasets.values():
            dataset.close()


def run(config_path: Path, output: Path) -> dict:
    """Run a fresh full-coverage pilot and retain all benchmark/checkpoint evidence."""
    config = load_config(config_path)
    if config['training'].get('resume_checkpoint') or any(config['data'].get(f'max_{s}_samples') is not None for s in ('train', 'validation')):
        raise ValueError('This pilot requires fresh full-coverage training and validation.')
    config['training']['archive_checkpoint_every'] = 1
    resolve_execution(config)
    model_dir = Path(config['training']['output_dir'])
    if model_dir.exists():
        raise FileExistsError(f'Preserve existing results; use a new run directory: {model_dir}')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(config=str(config_path), model_dir=str(model_dir), experimental=True)

    def observer(event, model, bundle, resolved, record):
        if event == 'heads_initialized':
            report.update(parameters=sum(p.numel() for p in model.parameters()), halo=resolved['execution']['halo'],
                          split_sizes={k: len(v) for k, v in bundle.split_indices.items()},
                          benchmark=benchmark(model, bundle, resolved))
            atomic_json(output / 'benchmark.json', portable(report))
        elif event == 'update':
            report.setdefault('gradient_checks', []).append(dict(update=record['optimizer_updates'],
                norm=require_finite_gradients(model)))

    started = perf_counter()
    best = train_model(config, observer=observer, observer_updates=(1,), stop_after_epoch=3)
    report['training_and_benchmark_seconds'] = perf_counter() - started
    history = pd.read_csv(model_dir / 'history.csv')
    if len(history) != 3:
        raise ValueError('Pilot must complete three validation epochs.')
    for row in history.to_dict('records'):
        if row['train_gradient_preclip_norm_count'] != row['train_optimizer_updates']:
            raise FloatingPointError('Some optimizer updates had nonfinite gradients.')
        for split in ('train', 'validation'):
            required = [f'{split}_total_loss'] + [f'{split}_alignment_{n}' for n in NAMES if config['loss']['batch_alignment'][n]['weight'] > 0]
            if not all(math.isfinite(row[k]) for k in required):
                raise FloatingPointError('Pilot has invalid weighted objectives.')
        if not (model_dir / f"archives/epoch_{int(row['epoch']):04d}.pt").exists():
            raise FileNotFoundError('Missing completed epoch checkpoint.')
    if not np.allclose(history['train_cca_effective_weight'], [0., .05, .1]):
        raise ValueError('Pilot did not cover the expected H&E ramp.')
    report.update(best_checkpoint=str(best), history=history.to_dict('records'))
    atomic_json(output / 'pilot.json', portable(report))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    report['held_out'] = evaluate_export(model_dir / 'latest.pt', output)
    atomic_json(output / 'pilot.json', portable(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == '__main__':
    main()

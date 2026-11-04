"""Benchmark shared H&E startup and matched, bounded pixel training on real data.

Never modifies datasets or production outputs. Selection uses the smallest worker
count within five percent of the fastest median across three repeated trials.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

from dann.config import load_config, seed_everything
from dann.data_loader import create_data_bundle
from dann.execution import ResidentSpectra
from dann.he import HETargetReader
from dann.initialization import initialize_supervised_heads
from dann.losses import ZILNLoss
from dann.lr_scheduler import UpdateWarmup
from dann.model import AdversarialLatentFusion
from dann.monitoring import freeze_monitoring
from dann.train import run_epoch
from dann.verification.pixel_loader_benchmark import GPUUsage, freeze_requests, loader_for


def select_workers(records: list[dict], key: str) -> int:
    """Prefer fewer workers when elapsed medians are within five percent."""
    medians = {w: float(np.median([r[key] for r in records if r['workers'] == w]))
               for w in sorted({r['workers'] for r in records})}
    return min(w for w, seconds in medians.items() if seconds <= min(medians.values()) * 1.05)


def benchmark(config_path: Path, output: Path, steps: int = 16,
              startup_report: Path | None = None) -> dict:
    """Measure actual startup, train-only means, CPU preparation and CUDA updates."""
    if output.exists():
        raise FileExistsError('Benchmark output must be a new diagnostic directory.')
    # Set before CUDA initialization so the exact audit also works via the CLI
    # without requiring a caller to supply this environment variable.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    c = load_config(config_path)
    if c['execution']['mode'] != 'pixel' or steps < 1:
        raise ValueError('This matched benchmark requires pixel mode and positive steps.')
    c['performance'].update(ram_cache=True, he_ram_cache=True)
    c = freeze_monitoring(c)
    output.mkdir(parents=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    report = dict(startup=[], training=[], original_loader=[], steps=steps, repetitions=3)
    def save():
        report['peak_host_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        (output / 'benchmark.json').write_text(json.dumps(report, indent=2))
    seed_everything(c['training']['seed'], False)
    started = perf_counter()
    bundle = create_data_bundle(c)
    report['data_preparation_seconds'] = perf_counter() - started
    reader = bundle.he_reader
    if reader is None:
        raise ValueError('Benchmark requires enabled H&E supervision.')
    report.update(he_cache_seconds=reader.startup_seconds, he_cache_bytes=reader.nbytes,
                  training_rows=len(bundle.split_indices['train']))
    reference_mean = reader.mean(bundle.split_indices['train'])
    if startup_report is not None:
        previous = json.loads(startup_report.read_text())
        saved_config = yaml.safe_load((startup_report.parent / 'config.yaml').read_text())
        if saved_config != c or len(previous.get('startup', [])) != 15:
            raise ValueError('Reused startup measurements require identical config and all 15 trials.')
        report['startup'] = previous['startup']
        report['startup_report'] = str(startup_report.resolve())
    # Every candidate loads the same CSR and reduces the complete training subset.
    for trial in range(0 if startup_report is not None else 3):
        for workers in ((1, 2, 4, 8, 16) if trial % 2 == 0 else (16, 8, 4, 2, 1)):
            started = perf_counter()
            cache = ResidentSpectra(Path(c['data']['path']), c['data'],
                                    int(c['performance']['host_budget_gib'] * 2**30), workers=workers)
            spectra_seconds = perf_counter() - started
            started = perf_counter()
            mean = reader.mean(bundle.split_indices['train'], workers=workers,
                               budget_bytes=int(c['performance']['host_budget_gib'] * 2**30))
            mean_seconds = perf_counter() - started
            np.testing.assert_array_equal(mean, reference_mean)
            for name in ('indptr', 'indices', 'values'):
                np.testing.assert_array_equal(getattr(cache, name), getattr(bundle.datasets['train'].resident, name))
            report['startup'].append(dict(trial=trial, workers=workers, spectra_seconds=spectra_seconds,
                mean_seconds=mean_seconds, seconds=spectra_seconds + mean_seconds))
            print('Startup benchmark:', report['startup'][-1], flush=True)
            del cache
            gc.collect()
            save()
    requests = freeze_requests(bundle, warmup=2, steps=steps, validation_steps=2)
    np.savez(output / 'requests.npz', train=np.stack(requests['train']),
             validation=np.stack(requests['validation']))
    device = torch.device('cuda')
    model = AdversarialLatentFusion.from_config(c, len(bundle.metadata.batch_names),
                                              len(c['data']['target_columns'])).to(device)
    started = perf_counter()
    initialize_supervised_heads(model, bundle, c)
    report['head_initialization_seconds'] = perf_counter() - started
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    weights = bundle.batch_class_weights.to(device) if c['loss']['balance_batch_classes'] else None
    loss = ZILNLoss.from_config(c).to(device)
    from dann.batch_alignment import alignment_settings
    alignment = alignment_settings(c)
    dataset = bundle.datasets['train']
    backed = HETargetReader(reader.path, reader.key, reader.rows, reader.width)
    states = None

    def run(settings, rows, optimizer, validation=False):
        loader = loader_for(bundle.datasets['validation'] if validation else dataset,
                            rows, settings, scalar=False, validation=validation)
        training = settings['training']
        validation_rows = None
        if validation:
            from dann.batch_alignment import sample_rows
            source_rows = bundle.datasets['validation'].indices[np.concatenate(rows)]
            validation_rows = sample_rows(source_rows, bundle.metadata.batch_codes[source_rows],
                                          alignment['validation_per_slide'], training['seed'])
        warmup = UpdateWarmup(optimizer, settings, len(bundle.loaders['train'])) if optimizer is not None else None
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        with GPUUsage(device) as usage:
            metrics = run_epoch(model, loader, loss, device,
                settings['loss']['biology_weight'], settings['loss']['batch_weight'], weights,
                optimizer, 0, training['epochs'], training['grl_schedule'], training['grl_max_lambda'],
                training['grl_gamma'], training['gradient_clip_norm'],
                target_columns=settings['data']['target_columns'], he_weight=settings['loss']['he_weight'],
                monitoring=settings['monitoring'], warmup=warmup, batch_alignment=alignment,
                alignment_seed=training['seed'], validation_alignment_rows=validation_rows)
        metrics['mean_gpu_utilization'] = float(np.mean([x['utilization'] for x in usage.samples])) if usage.samples else None
        return metrics

    def reset(deterministic=False):
        seed_everything(c['training']['seed'], deterministic)
        model.load_state_dict(initial)
        return torch.optim.AdamW(model.parameters(), lr=c['training']['learning_rate'],
            weight_decay=c['training']['weight_decay'],
            betas=(c['training']['beta1'], c['training']['beta2']))

    # Separate exact equivalence from default-profile timings: CUDA scatter
    # reductions need not be bitwise repeatable in nondeterministic training.
    audit_state = None
    for source in (backed, reader):
        dataset.he_reader = source
        audit_config = copy.deepcopy(c)
        audit_config['performance'].update(preparation_workers=2 if source is backed else 16,
                                          prepared_batches=2 if source is backed else 32)
        optimizer = reset(deterministic=True)
        run(audit_config, requests['train_warmup'], optimizer)
        current = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if audit_state is None:
            audit_state = current
        else:
            for key in audit_state:
                torch.testing.assert_close(current[key], audit_state[key], rtol=0, atol=0)
    report['deterministic_equivalence'] = dict(updates=2, parameter_equality='exact', workers=[2, 16])
    print('Deterministic cached/backed CUDA updates match exactly.', flush=True)
    save()
    for trial in range(3):
        candidates = [('original', 2, 2), *[('cached', w, 2*w) for w in (2, 4, 8, 16)]]
        if trial % 2:
            candidates.reverse()
        for kind, workers, queue in candidates:
            settings = copy.deepcopy(c)
            settings['performance'].update(preparation_workers=workers, prepared_batches=queue,
                                           he_ram_cache=kind == 'cached')
            dataset.he_reader = reader if kind == 'cached' else backed
            optimizer = reset()
            run(settings, requests['train_warmup'], optimizer)
            optimizer = reset()
            metrics = run(settings, requests['train'], optimizer)
            record = dict(trial=trial, workers=workers, prepared_batches=queue, **metrics)
            report['training' if kind == 'cached' else 'original_loader'].append(record)
            current = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if states is None:
                states = current
            else:
                record['max_abs_parameter_difference'] = max(
                    float((current[key] - states[key]).abs().max()) for key in states)
            print(f'Train benchmark: {kind}, workers={workers}, trial={trial}, '
                  f"seconds={metrics['seconds']:.3f}, rows/s={metrics['rows_per_second']:.1f}, "
                  f"wait={metrics['loader_wait_seconds']:.3f}", flush=True)
            save()
    dataset.he_reader = reader
    report['selected_startup_workers'] = select_workers(report['startup'], 'seconds')
    report['selected_preparation_workers'] = select_workers(report['training'], 'seconds')
    selected = report['selected_preparation_workers']
    c['performance'].update(preparation_workers=selected, prepared_batches=2*selected)
    save()
    report['validation'] = run(c, requests['validation'], None, validation=True)
    save()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--startup-report', type=Path,
                        help='Reuse completed startup trials from an identical configuration.')
    args = parser.parse_args()
    benchmark(args.config, args.output, args.steps, args.startup_report)


if __name__ == '__main__':
    main()

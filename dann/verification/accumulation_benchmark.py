"""Isolated full-epoch accumulation timings using the production training loop."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import resource
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import yaml

from dann.config import resolve_execution, seed_everything
from dann.data_loader import create_data_bundle
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.targets import CD8
from dann.train import TrainingProgress, run_epoch
from dann.training_artifacts import atomic_json

STEPS = (1, 2, 4, 8)
ROWS = 32768
GPU_BUDGET_ERROR = 'Measured execution exceeded VRAM budget.'


def benchmark_config(base: dict, steps: int) -> dict:
    """Change only physical accumulation and disable unrelated output workflows."""
    if isinstance(steps, bool) or steps not in STEPS:
        raise ValueError(f'Accumulation steps must be one of {STEPS}.')
    config = copy.deepcopy(base)
    spatial = config['training']['spatial']
    if spatial['core_size'] != 32 or spatial['supervised_rows_per_batch'] != ROWS:
        raise ValueError('Use the core-32, 32,768-row pilot candidate checkpoint.')
    if spatial['sampling_strategy'] != 'proportional_slide_tiles':
        raise ValueError('Benchmark requires proportional_slide_tiles.')
    for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
        if (group['type'] != 'gatv2' or group['gatv2']['neighbor_radius'] != 1.5
                or group['gatv2']['attention_dropout'] != .1):
            raise ValueError('Benchmark requires all-GATv2, radius 1.5, attention dropout 0.10.')
    spatial['microbatch_rows'] = None if steps == 1 else ROWS // steps
    config['training'].update(early_stopping_enabled=False, resume_checkpoint=None)
    config['performance']['async_artifacts'] = False
    config['post_training'] = {key: False for key in config['post_training']}
    return resolve_execution(config)


def state_hash(state: dict) -> str:
    """Hash ordered model tensors without allocating a second model."""
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def reset_training(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   source: dict, seed: int, epoch: int) -> TrainingProgress:
    """Undo warmup, including AdamW moments, gradients, and all random streams."""
    model.load_state_dict(source['model_state'])
    optimizer.load_state_dict(copy.deepcopy(source['optimizer_state']))
    optimizer.zero_grad(set_to_none=True)
    seed_everything(seed, source['config']['training']['deterministic_algorithms'])
    return TrainingProgress(epoch=epoch)


class AuditedLoader:
    """Check delivered supervision against each expected logical sampler request.

    Only CPU row IDs are inspected. No per-update device synchronization or file
    writing is added. A successful full epoch must also match optimizer counters;
    prefetched rows alone never establish completion.
    """
    def __init__(self, source: Any, expected: list) -> None:
        self.source = source
        self.expected = expected
        self.records: list[dict] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def __len__(self) -> int:
        return len(self.source)

    def __iter__(self):
        self.records = []
        position, group_index = 0, 0
        digest = hashlib.sha256()
        physical = 0
        for batch in self.source:
            request = self.expected[group_index]
            rows = batch['row_ids'].numpy().astype('<i8', copy=False)
            expected = np.asarray(request.row_ids[position:position + len(rows)], dtype='<i8')
            if not np.array_equal(rows, expected):
                raise ValueError('Delivered microbatch rows differ from the logical request.')
            group = batch.get('accumulation', {})
            if group.get('index', 0) != physical:
                raise ValueError('Microbatch indices are not consecutive.')
            digest.update(rows.tobytes())
            position += len(rows)
            physical += 1
            last = group.get('index', 0) + 1 == group.get('count', 1)
            if last:
                if position != len(request.row_ids):
                    raise ValueError('Logical group lost or duplicated supervised rows.')
                self.records.append(dict(
                    rows=position, microbatches=physical,
                    row_ids_sha256=digest.hexdigest(), tiles=len(request.tile_ids)))
                position, physical = 0, 0
                group_index += 1
                digest = hashlib.sha256()
            yield batch


def verify_epoch(records: list[dict], expected: list, selected_rows: np.ndarray,
                 progress: TrainingProgress, optimizer_steps: int, epoch: int) -> dict:
    """Require complete sampler coverage and completed optimizer updates."""
    rows = np.concatenate([np.asarray(request.row_ids, dtype='<i8') for request in expected])
    if not np.array_equal(np.sort(rows), np.sort(selected_rows)):
        raise ValueError('Sampler did not cover every training row exactly once.')
    if (len(records) != len(expected) or optimizer_steps != len(expected)
            or progress.optimizer_updates != len(expected)
            or progress.supervised_rows != len(rows)
            or progress.epoch != epoch + 1 or progress.update_in_epoch != 0):
        raise ValueError('Epoch records and completed optimizer counters disagree.')
    for record, request in zip(records, expected, strict=True):
        expected_rows = np.asarray(request.row_ids, dtype='<i8')
        if (record['rows'] != len(expected_rows)
                or record['row_ids_sha256'] != hashlib.sha256(expected_rows.tobytes()).hexdigest()):
            raise ValueError('Consumed row hashes differ from the planned epoch.')
    return dict(passed=True, rows=len(rows), optimizer_updates=len(expected),
                physical_microbatches=sum(r['microbatches'] for r in records),
                row_order_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
                final_partial_rows=len(expected[-1].row_ids))


def failure_kind(error: Exception) -> str:
    """Separate actual allocation failures, configured budgets, and other errors."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return 'cuda_oom'
    if isinstance(error, MemoryError):
        return 'memory_budget' if 'budget' in str(error).lower() else 'host_oom'
    if isinstance(error, FloatingPointError):
        return 'nonfinite'
    return 'error'


def memory() -> dict:
    """Read this isolated process's CUDA and host high-water marks."""
    return dict(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)


def execute_epoch(arguments: dict, **kwargs) -> dict:
    """Retain hardware-fit evidence even when the completed pass exceeds 24 GiB."""
    try:
        return run_epoch(**arguments, **kwargs)
    except MemoryError as error:
        # This production check runs only after the pass has completed. Continue
        # measuring the requested batch without changing any execution controls.
        if str(error) != GPU_BUDGET_ERROR:
            raise
        return dict(configured_gpu_budget_exceeded=True)


def worker(args: argparse.Namespace) -> dict:
    """Measure one configuration/repetition in a process with a fresh allocator."""
    started = perf_counter()
    directory = args.output
    result = dict(steps=args.worker_steps, repetition=args.repetition,
                  sampler_epoch=args.repetition, status='starting')
    bundle = None
    stage = 'setup'
    try:
        source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if (source['training_progress']['optimizer_updates'] != 0
                or source['optimizer_state']['state']):
            raise ValueError('The benchmark must start from update_0000.pt, before training.')
        config = benchmark_config(source['config'], args.worker_steps)
        (directory / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        training, loss = config['training'], config['loss']
        seed = training['seed'] + args.repetition
        bundle = create_data_bundle(config, source['split_indices'], source['data_identity'],
                                    requested_splits=('train',))
        loader = bundle.loaders['train']
        loader.sampler.set_epoch(args.repetition)
        expected = list(loader.sampler.logical_requests())
        audited = AuditedLoader(loader, expected)
        model = AdversarialLatentFusion.from_config(
            config, len(bundle.metadata.batch_names), len(config['data']['target_columns'])).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
            weight_decay=training['weight_decay'], betas=(training['beta1'], training['beta2']))
        progress = reset_training(model, optimizer, source, seed, args.repetition)
        initial_hash = state_hash(model.state_dict())
        if initial_hash != state_hash(source['model_state']):
            raise ValueError('Initial model differs from the saved initial weights.')
        arguments = dict(model=model, ziln_loss=ZILNLoss.from_config(config).cuda(),
            device=torch.device('cuda'), biology_weight=loss['biology_weight'],
            batch_weight=loss['batch_weight'], he_weight=loss['he_weight'],
            class_weights=bundle.batch_class_weights.cuda() if loss['balance_batch_classes'] else None,
            optimizer=optimizer, epoch_index=args.repetition, total_epochs=training['epochs'],
            grl_schedule=training['grl_schedule'], grl_max_lambda=training['grl_max_lambda'],
            grl_gamma=training['grl_gamma'], grl_horizon_updates=training['grl_horizon_updates'],
            gradient_clip_norm=training['gradient_clip_norm'],
            cd8_index=config['data']['target_columns'].index(CD8),
            target_columns=config['data']['target_columns'], deadline=args.deadline)
        result.update(initial_state_sha256=initial_hash, microbatch_rows=ROWS // args.worker_steps,
            configured_gpu_budget_gib=config['performance']['vram_budget_gib'],
            configured_host_budget_gib=config['performance']['host_budget_gib'],
            gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
            preparation_seconds=perf_counter() - started)
        stage = 'warmup'
        print(f'Warmup: steps={args.worker_steps}, repetition={args.repetition}', flush=True)
        warm_started = perf_counter()
        execute_epoch(arguments, loader=loader, progress_state=progress, max_updates=args.warmup)
        torch.cuda.synchronize()
        result.update(warmup_seconds=perf_counter() - warm_started, warmup_memory=memory())
        if progress.optimizer_updates != args.warmup:
            raise TimeoutError('Deadline interrupted warmup.')
        progress = reset_training(model, optimizer, source, seed, args.repetition)
        result['reset_verified'] = (
            state_hash(model.state_dict()) == initial_hash and not optimizer.state
            and all(p.grad is None for p in model.parameters()))
        if not result['reset_verified']:
            raise ValueError('Warmup reset failed.')
        completed_steps = []
        hook = optimizer.register_step_post_hook(lambda *_: completed_steps.append(True))
        stage = 'timed_epoch'
        print(f'Timed epoch: steps={args.worker_steps}, repetition={args.repetition}', flush=True)
        torch.cuda.synchronize()
        timed_started = perf_counter()
        metrics = execute_epoch(arguments, loader=audited, progress_state=progress)
        torch.cuda.synchronize()
        seconds = perf_counter() - timed_started
        measured_memory = memory()
        hook.remove()
        result.update(epoch_seconds=seconds, metrics=metrics, **measured_memory,
                      progress=asdict(progress), completed_optimizer_steps=len(completed_steps))
        if progress.optimizer_updates != len(expected):
            raise TimeoutError('Deadline interrupted the measured epoch.')
        stage = 'verification'
        result['coverage'] = verify_epoch(audited.records, expected, bundle.split_indices['train'],
                                         progress, len(completed_steps), args.repetition)
        finite = [torch.isfinite(p).all() for p in model.parameters()]
        finite += [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
        if not torch.stack(finite).all().item():
            raise FloatingPointError('Nonfinite parameters or final gradients.')
        if not all(int(s['step'].item()) == len(expected) for s in optimizer.state.values()):
            raise ValueError('AdamW parameter step counters differ from logical updates.')
        gpu_peak = max(measured_memory['peak_allocated_gib'],
                       result['warmup_memory']['peak_allocated_gib'])
        host_peak = max(measured_memory['peak_host_gib'], result['warmup_memory']['peak_host_gib'])
        within_budget = (gpu_peak <= result['configured_gpu_budget_gib']
                         and host_peak <= result['configured_host_budget_gib'])
        result.update(status='passed' if within_budget else 'budget_exceeded',
            hardware_fit=True, within_budget=within_budget,
            seconds_per_update=seconds / len(expected),
            rows_per_second=len(bundle.split_indices['train']) / seconds)
        atomic_json(directory / 'updates.json', audited.records)
    except Exception as error:
        result.update(status='timeout' if isinstance(error, TimeoutError) else failure_kind(error),
                      error=str(error), stage=stage, traceback=traceback.format_exc())
        if torch.cuda.is_initialized():
            result.update(memory())
    finally:
        if bundle is not None:
            for dataset in bundle.datasets.values():
                dataset.close()
        result['process_seconds'] = perf_counter() - started
        atomic_json(directory / 'result.json', result)
    return result


def summarize(records: list[dict], repeats: int) -> dict:
    """Rank complete, budget-compliant repetitions, preferring memory within 3%."""
    summaries = []
    for steps in STEPS:
        trials = [r for r in records if r['steps'] == steps]
        complete = [r for r in trials if r['status'] in ('passed', 'budget_exceeded')]
        summary = dict(steps=steps, microbatch_rows=ROWS // steps,
                       completed_repetitions=len(complete),
                       statuses=[r['status'] for r in trials],
                       eligible=len(trials) == repeats and all(r['status'] == 'passed' for r in trials))
        if complete:
            times = [r['epoch_seconds'] for r in complete]
            summary.update(median_epoch_seconds=float(np.median(times)),
                min_epoch_seconds=min(times), max_epoch_seconds=max(times),
                median_seconds_per_update=float(np.median([r['seconds_per_update'] for r in complete])),
                median_rows_per_second=float(np.median([r['rows_per_second'] for r in complete])),
                peak_allocated_gib=max(max(r['peak_allocated_gib'], r['warmup_memory']['peak_allocated_gib'])
                                       for r in complete),
                peak_reserved_gib=max(max(r['peak_reserved_gib'], r['warmup_memory']['peak_reserved_gib'])
                                      for r in complete),
                peak_host_gib=max(r['peak_host_gib'] for r in complete))
        summaries.append(summary)
    # Mismatched rows or weights invalidate the comparison, even if fast.
    for repetition in range(repeats):
        complete = [r for r in records if r['repetition'] == repetition
                    and r['status'] in ('passed', 'budget_exceeded')]
        if (len({r['initial_state_sha256'] for r in complete}) > 1
                or len({r['coverage']['row_order_sha256'] for r in complete}) > 1):
            raise ValueError('Repetitions used different starting weights or row sequences.')
    eligible = [s for s in summaries if s['eligible']]
    recommended = None
    if eligible:
        fastest = min(s['median_epoch_seconds'] for s in eligible)
        near = [s for s in eligible if s['median_epoch_seconds'] <= fastest * 1.03]
        recommended = min(near, key=lambda s: (s['peak_allocated_gib'], s['median_epoch_seconds']))['steps']
    baseline = next((s for s in summaries if s['steps'] == 8 and s['eligible']), None)
    for summary in summaries:
        if baseline is not None and summary['eligible']:
            summary['speedup_over_8'] = baseline['median_epoch_seconds'] / summary['median_epoch_seconds']
    return dict(settings=summaries, recommended_steps=recommended,
                minimum_passing_steps=min((s['steps'] for s in eligible), default=None))


def write_report(root: Path, report: dict) -> None:
    """Publish measurements and a plain-language recommendation after each trial."""
    atomic_json(root / 'results.json', report)
    fields = ['steps', 'microbatch_rows', 'completed_repetitions', 'eligible',
              'median_epoch_seconds', 'median_seconds_per_update', 'median_rows_per_second',
              'peak_allocated_gib', 'peak_reserved_gib', 'peak_host_gib', 'speedup_over_8']
    with (root / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['summary']['settings'])
    lines = ['Gradient accumulation benchmark: 32,768 supervised rows per update',
             'Core 32; radius 1.5; attention dropout 0.10; identical saved initial weights and split.',
             'Timing: production full epochs, including preparation/transfer and the final partial update.',
             f"{report['warmup']} warmup updates are excluded; weights, AdamW, gradients, counters and RNG reset afterward.",
             'Startup and report writing are excluded. No per-update CUDA synchronization is added.',
             'GPU peaks cover warmup and measured training; host RSS includes process preparation.',
             'Configured allocation budgets are recorded per trial and differ from physical GPU capacity.',
             'Reserved GPU memory includes live allocations and PyTorch allocator cache.', '',
             'Steps  Physical rows  Median epoch(s)  Seconds/update  Allocated(GiB)  Reserved(GiB)  Host(GiB)  Passed']
    for setting in report['summary']['settings']:
        passed = setting['statuses'].count('passed')
        if 'median_epoch_seconds' in setting:
            lines.append(
                f"{setting['steps']:>5}  {setting['microbatch_rows']:>13,}  "
                f"{setting['median_epoch_seconds']:>15.2f}  "
                f"{setting['median_seconds_per_update']:>14.3f}  "
                f"{setting['peak_allocated_gib']:>14.2f}  "
                f"{setting['peak_reserved_gib']:>13.2f}  "
                f"{setting['peak_host_gib']:>9.2f}  {passed}/{report['repeats']}")
            lines.append(
                f"       Epoch range: {setting['min_epoch_seconds']:.2f}–"
                f"{setting['max_epoch_seconds']:.2f}s; median rows/s: "
                f"{setting['median_rows_per_second']:,.0f}; statuses: {setting['statuses']}")
        else:
            lines.append(f"{setting['steps']:>5}  No completed measurements; statuses: {setting['statuses']}")
    lines += ['', f"Recommended accumulation steps: {report['summary']['recommended_steps']}",
              f"Minimum passing steps: {report['summary']['minimum_passing_steps']}",
              'All requested repetitions must pass to qualify; within 3% of fastest, prefer lower allocated memory.',
              f"Elapsed minutes: {report['elapsed_seconds'] / 60:.2f}",
              'Seconds/update averages include the partial final update, identical across settings.',
              'These are memory/throughput results at radius 1.5, not convergence or biological-quality evidence.']
    if report['summary']['minimum_passing_steps'] == 1:
        lines += ['', 'The full 32,768-row physical batch completed every repetition within budget.',
                  'Gradient accumulation was not required for these tested epochs and hardware.']
    eligible = {s['steps']: s for s in report['summary']['settings'] if s['eligible']}
    if 1 in eligible and 8 in eligible:
        saving = 100 * (1 - eligible[1]['median_epoch_seconds'] / eligible[8]['median_epoch_seconds'])
        lines.append(f'One step reduced epoch time by {saving:.1f}% relative to eight steps.')
    if 1 in eligible and 2 in eligible:
        slowdown = 100 * (eligible[2]['median_epoch_seconds'] / eligible[1]['median_epoch_seconds'] - 1)
        memory_saving = 100 * (1 - eligible[2]['peak_allocated_gib'] / eligible[1]['peak_allocated_gib'])
        lines.append(f'Two steps took {slowdown:.1f}% longer than one and reduced allocated memory by {memory_saving:.1f}%.')
    (root / 'report.txt').write_text('\n'.join(lines) + '\n')


def run(args: argparse.Namespace) -> dict:
    """Run successive isolated trials within one bounded wall-clock budget."""
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    deadline = started + args.minutes * 60
    report = dict(checkpoint=str(args.checkpoint.resolve()), warmup=args.warmup,
                  repeats=args.repeats, budget_minutes=args.minutes, trials=[])
    failed = set()
    for repetition in range(args.repeats):
        order = STEPS[repetition % len(STEPS):] + STEPS[:repetition % len(STEPS)]
        for steps in order:
            if steps in failed or perf_counter() >= deadline:
                continue
            directory = args.output / f'steps_{steps}_rep_{repetition}'
            directory.mkdir()
            command = [sys.executable, '-u', '-m', 'dann.verification.accumulation_benchmark',
                       '--checkpoint', str(args.checkpoint),
                       '--output', str(directory), '--worker-steps', str(steps),
                       '--repetition', str(repetition), '--warmup', str(args.warmup),
                       '--deadline', str(deadline)]
            print(f'Starting accumulation={steps}, repetition={repetition + 1}/{args.repeats}', flush=True)
            with (directory / 'worker.log').open('w') as log:
                try:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=max(deadline - perf_counter(), .01))
                    path = directory / 'result.json'
                    record = (json.loads(path.read_text()) if path.exists() else
                              dict(steps=steps, repetition=repetition, status='process_error',
                                   returncode=process.returncode))
                except subprocess.TimeoutExpired:
                    record = dict(steps=steps, repetition=repetition, status='timeout',
                                  error='Parent terminated trial at the experiment deadline.')
                    atomic_json(directory / 'result.json', record)
            report['trials'].append(record)
            if record['status'] not in ('passed', 'budget_exceeded'):
                failed.add(steps)
            report.update(elapsed_seconds=perf_counter() - started,
                          summary=summarize(report['trials'], args.repeats))
            write_report(args.output, report)
            print(json.dumps({k: record[k] for k in
                ('steps', 'repetition', 'status', 'epoch_seconds', 'peak_allocated_gib', 'error')
                if k in record}), flush=True)
    report.update(elapsed_seconds=perf_counter() - started,
                  stopping_reason='budget' if perf_counter() >= deadline else 'completed',
                  summary=summarize(report['trials'], args.repeats))
    write_report(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Candidate update_0000.pt.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minutes', type=float, default=30.)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--worker-steps', type=int, choices=STEPS, help=argparse.SUPPRESS)
    parser.add_argument('--repetition', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--deadline', type=float, default=float('inf'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.minutes <= 0 or args.repeats < 1 or args.warmup < 1:
        parser.error('minutes, repeats, and warmup must be positive')
    if args.worker_steps is not None:
        worker(args)
    else:
        run(args)


if __name__ == '__main__':
    main()

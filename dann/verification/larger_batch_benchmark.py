"""Repeated joint batch sweeps against the resident production profile.

Use execution_benchmark --larger-batches to run this harness. Reports are resumable;
production YAML and checkpoints are never written by the benchmark itself.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import itertools
import json
import resource
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from dann.config import load_config, resolve_execution, seed_everything
from dann.data_loader import create_data_bundle
from dann.execution import (PreparedLoader, build_validation_features, device_batches,
                            evaluation_groups, pooled_validation_loader, validation_execution)
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.sampling import ProportionalSlideTileSampler
from dann.tiles import tile_collate
from dann.train import grl_strength
from dann.verification.execution_benchmark import Requests, epoch, train_loader

DEVICE = torch.device('cuda')
FAILURES = (torch.cuda.OutOfMemoryError, MemoryError, FloatingPointError)
TIMING_KEYS = {'seconds', 'observed_rows', 'rows_per_second', 'loader_wait_seconds',
               'peak_vram_gib', 'peak_reserved_vram_gib', 'peak_host_gib', 'prepared_batch_mib'}


def changed(config: dict, **controls: int) -> dict:
    """Resolve both YAML and runtime fields so recorded candidates are reproducible."""
    result = copy.deepcopy(config)
    for name, value in controls.items():
        if name == 'rows':
            result['training']['spatial']['supervised_rows_per_batch'] = value
        elif name in ('peaks', 'gat'):
            result['model'][{'peaks': 'spectral_peak_budget', 'gat': 'gatv2_tile_budget'}[name]] = value
        else:
            result['performance'][name] = value
    return resolve_execution(result)


def controls(config: dict) -> dict:
    """Return the independent axes in compact form for progress messages."""
    return dict(rows=config['training']['supervised_rows_per_batch'],
                peaks=config['model']['spectral_peak_budget'], gat=config['model']['gatv2_tile_budget'],
                **{k: config['performance'][k] for k in
                   ('validation_tiles', 'validation_gat_tiles', 'validation_peak_budget', 'feature_cache_rows')})


def summarize(trials: list[dict]) -> dict:
    """Aggregate repetitions without disguising their timing variability."""
    seconds = [r['seconds'] for r in trials]
    result = dict(seconds=float(np.median(seconds)), minimum_seconds=min(seconds), maximum_seconds=max(seconds),
                  peak_vram_gib=max(r['peak_vram_gib'] for r in trials),
                  peak_reserved_vram_gib=max(r['peak_reserved_vram_gib'] for r in trials),
                  peak_host_gib=max(r['peak_host_gib'] for r in trials))
    if 'rows_per_second' in trials[0]:
        result['rows_per_second'] = float(np.median([r['rows_per_second'] for r in trials]))
    return result


def choose(records: list[dict], baseline: dict | None = None) -> dict:
    """Prefer lower memory within 3% of the fastest, requiring 3% over baseline."""
    passing = [r for r in records if 'rejected' not in r]
    if baseline is not None:
        passing = [r for r in passing if r['summary']['seconds'] <= baseline['summary']['seconds'] * .97]
        if not passing:
            return baseline
    fastest = min(r['summary']['seconds'] for r in passing)
    near = [r for r in passing if r['summary']['seconds'] <= fastest * 1.03]
    return min(near, key=lambda r: (r['summary']['peak_vram_gib'], r['summary']['seconds']))


def cleanup() -> None:
    """Release rejected-trial tensors before another candidate starts."""
    gc.collect()
    torch.cuda.empty_cache()


def measurements(seconds: float, config: dict, loader=None) -> dict:
    """Capture and enforce memory budgets, including prefetched allocations."""
    result = dict(seconds=seconds, peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_vram_gib=torch.cuda.max_memory_reserved()/2**30,
                  peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    if loader is not None:
        result['loader_wait_seconds'] = loader.wait_seconds
    if result['peak_vram_gib'] > config['performance']['vram_budget_gib']:
        raise MemoryError(f'VRAM budget exceeded: {result}')
    if result['peak_host_gib'] > config['performance']['host_budget_gib']:
        raise MemoryError(f'Host budget exceeded: {result}')
    return result


class Experiment:
    """Shared immutable inputs and append-only candidate evidence for one sweep."""
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if args.output.exists() and not args.resume:
            raise FileExistsError('Report exists; choose a new output or use --resume.')
        self.report = json.loads(args.output.read_text()) if args.resume else {}
        self.base = self.report.get('baseline', load_config(args.config))
        if self.base['execution']['mode'] != 'spatial' or self.base['training']['sampling_strategy'] != 'proportional_slide_tiles':
            raise ValueError('This experiment requires proportional spatial sampling.')
        if not all(self.base['performance'][k] for k in ('ram_cache', 'cache_geometry', 'validation_feature_cache')):
            raise ValueError('The baseline must enable resident inputs, geometry, and validation features.')
        torch.set_num_threads(self.base['performance']['torch_threads'] or 4)
        seed_everything(self.base['training']['seed'])
        started = perf_counter()
        self.bundle = create_data_bundle(self.base)
        self.train = self.bundle.datasets['train']
        self.validation = self.bundle.datasets['validation']
        self.weights = self.bundle.batch_class_weights.to(DEVICE) if self.base['loss']['balance_batch_classes'] else None
        model = AdversarialLatentFusion.from_config(self.base, len(self.bundle.metadata.batch_names),
                                                   len(self.base['data']['target_columns']))
        self.initial = copy.deepcopy(model.state_dict())
        self.requests: dict[int, list] = {}
        counts = np.diff(self.train.resident.indptr)
        self.tile_peaks = [int(counts[g['context_rows']].sum()) for g in self.train.context_rows]
        identity = {name: hashlib.sha256(dataset.indices.tobytes()).hexdigest()
                    for name, dataset in self.bundle.datasets.items()}
        if self.report and self.report['split_hashes'] != identity:
            raise ValueError('The resumed report belongs to different data splits.')
        self.report.setdefault('baseline', self.base)
        self.report.setdefault('split_hashes', identity)
        self.report.setdefault('protocol', dict(warmup=args.warmup, steps=args.steps, repeats=args.repeats))
        if self.report['protocol'] != dict(warmup=args.warmup, steps=args.steps, repeats=args.repeats):
            raise ValueError('Do not mix repetition protocols in one report.')
        self.report.setdefault('records', [])
        self.report.setdefault('sessions', []).append(dict(startup_seconds=perf_counter()-started,
            gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
            resident_gib=self.train.resident.nbytes/2**30, geometry_gib=self.train.geometry_bytes/2**30,
            source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        self.save()
        print(json.dumps({'startup': self.report['sessions'][-1]}), flush=True)

    def save(self) -> None:
        """Atomically preserve completed trials, without overwriting earlier reports."""
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.report, indent=2))
        temporary.replace(self.args.output)

    def model(self, config: dict, trial: int = 0) -> torch.nn.Module:
        """Recreate the same initial weights and paired random seed for every trial."""
        seed_everything(self.base['training']['seed'] + trial)
        model = AdversarialLatentFusion.from_config(config, len(self.bundle.metadata.batch_names),
                                                   len(config['data']['target_columns'])).to(DEVICE)
        model.load_state_dict(self.initial)
        return model

    def optimizer(self, model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
        """Use the production optimizer without batch-size learning-rate scaling."""
        training = config['training']
        return torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
            weight_decay=training['weight_decay'], betas=(training['beta1'], training['beta2']))

    def val_loader(self, config: dict) -> PreparedLoader:
        """Keep historical reduction groups while changing physical batches."""
        return PreparedLoader(DataLoader(self.validation, batch_size=config['performance']['validation_tiles'],
            collate_fn=tile_collate), config['performance'], config['model'], validation=True)

    def record(self, stage: str, config: dict, measure: Callable, warmup: bool = False) -> dict:
        """Resume completed candidates; persist each repetition and each rejection."""
        key = json.dumps([stage, controls(config)], sort_keys=True)
        record = next((r for r in self.report['records'] if r['key'] == key), None)
        if record is None:
            record = dict(key=key, stage=stage, controls=controls(config), config=config, trials=[])
            self.report['records'].append(record)
        if 'summary' in record or 'rejected' in record:
            return record
        print(json.dumps({'starting': stage, **controls(config)}), flush=True)
        try:
            if warmup and not record['trials']:
                measure(config, -1)
                cleanup()
            for trial in range(len(record['trials']), self.args.repeats):
                cleanup()
                result = measure(config, trial)
                record['trials'].append(result)
                self.save()
                print(json.dumps({'stage': stage, 'trial': trial, **controls(config),
                                  **{k: result[k] for k in ('seconds', 'peak_vram_gib')}}), flush=True)
            record['summary'] = summarize(record['trials'])
        except FAILURES as error:
            record['rejected'] = f'{type(error).__name__}: {error}'
            print(json.dumps({'rejected': stage, 'reason': record['rejected'], **controls(config)}), flush=True)
        finally:
            cleanup()
        self.save()
        return record

    def requests_for(self, rows: int) -> list:
        """Cache the epoch's sampler order across repeated execution-only candidates."""
        if rows not in self.requests:
            self.requests[rows] = list(ProportionalSlideTileSampler(self.train,
                self.base['training']['seed'], rows))
        return self.requests[rows]

    def training_trial(self, config: dict, trial: int) -> dict:
        """Time a continuous training window, then stress the largest complete request."""
        requests = self.requests_for(config['training']['supervised_rows_per_batch'])
        n = self.args.warmup + self.args.steps
        if len(requests) < n:
            raise MemoryError('Insufficient full-epoch requests for the requested timing window.')
        largest = max(requests, key=lambda r: sum(self.tile_peaks[i] for i in r.tile_ids))
        loader = train_loader(self.train, config, Requests(requests[:n] + [largest]))
        model = self.model(config, trial).train()
        optimizer = self.optimizer(model, config)
        objective = ZILNLoss.from_config(config).to(DEVICE)
        finite = torch.ones((), device=DEVICE, dtype=torch.bool)
        rows = 0
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        iterator = device_batches(loader, DEVICE)
        try:
            for step, (_, batch) in enumerate(iterator):
                optimizer.zero_grad(set_to_none=True)
                progress = step / max(config['training']['epochs'] * len(requests) - 1, 1)
                strength = grl_strength(progress, config['training']['grl_schedule'],
                    config['training']['grl_max_lambda'], config['training']['grl_gamma'])
                out = model(batch, grl_strength=strength)
                bio = objective(out['pi_logits'], out['mu'], out['sigma'], batch['targets'],
                                batch['target_valid_mask'], validate_targets=False)
                loss = config['loss']['biology_weight'] * bio.total + config['loss']['batch_weight'] * torch.nn.functional.cross_entropy(out['batch_logits'], batch['batches'], weight=self.weights)
                finite &= torch.isfinite(loss.detach()) & torch.stack([torch.isfinite(v).all() for v in out.values()]).all()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'] or float('inf'))
                finite &= torch.isfinite(norm)
                optimizer.step()
                if step >= self.args.warmup and step < n:
                    rows += len(batch['targets'])
                if step in (self.args.warmup-1, n-1, n):
                    torch.cuda.synchronize()
                    if not finite.item():
                        raise FloatingPointError('Nonfinite output, objective, or gradient.')
                    measurements(0., config)
                    if step == self.args.warmup-1:
                        started = perf_counter()
                    elif step == n-1:
                        elapsed = perf_counter() - started
                        started = perf_counter()
                    else:
                        largest_seconds = perf_counter() - started
        finally:
            iterator.close()
        result = measurements(elapsed, config, loader)
        result.update(rows_per_second=rows/elapsed, observed_rows=rows,
                      largest_seconds=largest_seconds, largest_tiles=len(largest.tile_ids),
                      maximum_epoch_tiles=max(len(r.tile_ids) for r in requests))
        # Training windows contain different row counts; compare seconds per supervised row.
        result['window_seconds'] = result['seconds']
        result['seconds'] = 1. / result['rows_per_second']
        return result

    def training_sweep(self) -> None:
        """Search interacting controls, deduplicating equivalent GAT partitions."""
        for rows, peaks in itertools.product((2048, 4096, 8192), (524288, 1048576, 2097152)):
            maximum = max(len(r.tile_ids) for r in self.requests_for(rows))
            seen = set()
            for gat in (64, 128, 256):
                if min(gat, maximum) in seen:
                    continue
                seen.add(min(gat, maximum))
                self.record('training', changed(self.base, rows=rows, peaks=peaks, gat=gat), self.training_trial)
        self.extend('training', self.training_trial, {'rows': 8192, 'peaks': 2097152, 'gat': 256})

    def extend(self, stage: str, measure: Callable, boundaries: dict, warmup: bool = False) -> None:
        """Double a winning boundary until memory rejection or two sub-3% gains."""
        for axis, boundary in boundaries.items():
            passing = [r for r in self.report['records'] if r['stage'] == stage and 'summary' in r]
            best = min(passing, key=lambda r: r['summary']['seconds'])
            if best['controls'][axis] < boundary:
                continue
            current, small = best, 0
            while small < 2:
                candidate = changed(current['config'], **{axis: current['controls'][axis] * 2})
                if axis == 'gat':
                    maximum = max(len(r.tile_ids) for r in self.requests_for(candidate['training']['supervised_rows_per_batch']))
                    if current['controls']['gat'] >= maximum:
                        break
                if axis == 'validation_gat_tiles' and current['controls'][axis] >= current['controls']['validation_tiles']:
                    break
                record = self.record(stage, candidate, measure, warmup)
                if 'rejected' in record:
                    break
                gain = 1. - record['summary']['seconds']/current['summary']['seconds']
                small = small + 1 if gain < .03 else 0
                current = record

    def cache_trial(self, config: dict, trial: int) -> dict:
        """Time all unique-context encoding, comparing features after the timed region."""
        model = self.model(config).eval()
        loader = self.val_loader(config)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), validation_execution(model, config['performance']):
            started = perf_counter()
            lookup, features = build_validation_features(model, loader, DEVICE)
            torch.cuda.synchronize()
            result = measurements(perf_counter()-started, config)
            if not torch.isfinite(features).all():
                raise FloatingPointError('Nonfinite cached features.')
            np.testing.assert_array_equal(lookup, self.reference_lookup)
            torch.testing.assert_close(features.cpu(), self.reference_features, rtol=5e-5, atol=5e-5)
        result['context_rows'] = len(features)
        return result

    def spatial_trial(self, config: dict, trial: int) -> dict:
        """Measure every validation row with a shared frozen cache, excluding its build."""
        model = self.model(config).eval()
        loader = self.val_loader(config)
        lookup, features = self.reference_lookup, self.spatial_features
        finite = torch.ones((), dtype=torch.bool, device=DEVICE)
        rows = 0
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        with torch.inference_mode(), validation_execution(model, config['performance']), pooled_validation_loader(loader, config['performance']):
            iterator = device_batches(loader, DEVICE)
            try:
                for cpu, batch in iterator:
                    ids = lookup[cpu['context_row_ids'].numpy()]
                    if np.any(ids < 0):
                        raise AssertionError('Missing context features.')
                    batch['pooled_features'] = features[torch.as_tensor(ids, device=DEVICE)]
                    outputs = model(batch, grl_strength=0.)
                    finite &= torch.stack([torch.isfinite(v).all() for v in outputs.values()]).all()
                    rows += len(batch['targets'])
            finally:
                iterator.close()
        torch.cuda.synchronize()
        result = measurements(perf_counter()-started, config, loader)
        if not finite.item():
            raise FloatingPointError('Nonfinite validation outputs.')
        if rows != len(self.validation.indices):
            raise AssertionError('Incomplete validation pass.')
        result['observed_rows'] = rows
        return result

    def predictions(self, config: dict) -> dict:
        """Collect ordered full-split outputs outside benchmark timing."""
        model = self.model(config).eval()
        result: dict[str, list] = {}
        with torch.inference_mode():
            for _, batch, outputs in evaluation_groups(model, self.val_loader(config), DEVICE):
                for key, value in dict(row_ids=batch['row_ids'], **outputs).items():
                    result.setdefault(key, []).append(value.cpu())
        return {key: torch.cat(values) for key, values in result.items()}

    def validation_trial(self, config: dict, trial: int) -> dict:
        """Include feature construction and reporting in complete validation timing."""
        model = self.model(config).eval()
        result = epoch(model, self.val_loader(config), config, class_weights=self.weights)
        measurements(result['seconds'], config)
        for key, value in self.reference_metrics.items():
            if key not in TIMING_KEYS and not np.isclose(result[key], value, rtol=5e-5, atol=5e-5, equal_nan=True):
                raise AssertionError(f'Validation metric changed: {key}: {result[key]} versus {value}')
        return result

    def validation_sweep(self) -> None:
        """Screen the two validation stages, then test their best combinations end to end."""
        model = self.model(self.base).eval()
        with torch.inference_mode(), validation_execution(model, self.base['performance']):
            self.reference_lookup, features = build_validation_features(model, self.val_loader(self.base), DEVICE)
            self.reference_features = features.cpu()
        del model, features
        cleanup()
        for rows, peaks in itertools.product((1024, 2048, 4096, 8192), (262144, 524288, 1048576, 2097152)):
            self.record('validation_cache', changed(self.base, feature_cache_rows=rows, validation_peak_budget=peaks), self.cache_trial, True)
        self.extend('validation_cache', self.cache_trial, {'feature_cache_rows': 8192, 'validation_peak_budget': 2097152}, True)
        self.spatial_features = self.reference_features.to(DEVICE)
        for physical, gat in itertools.product((64, 128, 256, 512), (32, 64, 128, 256)):
            if gat <= physical:
                self.record('validation_spatial', changed(self.base, validation_tiles=physical, validation_gat_tiles=gat), self.spatial_trial, True)
        self.extend('validation_spatial', self.spatial_trial, {'validation_tiles': 512, 'validation_gat_tiles': 256}, True)
        del self.spatial_features, self.reference_features
        cleanup()
        self.reference_metrics = epoch(self.model(self.base), self.val_loader(self.base), self.base, class_weights=self.weights)
        reference = self.predictions(self.base)
        best = {}
        for stage in ('validation_cache', 'validation_spatial'):
            best[stage] = sorted((r for r in self.report['records'] if r['stage'] == stage and 'summary' in r),
                                 key=lambda r: r['summary']['seconds'])[:2]
        candidates = [self.base]
        for cache, spatial in itertools.product(best['validation_cache'], best['validation_spatial']):
            candidates.append(changed(self.base, **{k: cache['controls'][k] for k in ('feature_cache_rows', 'validation_peak_budget')},
                **{k: spatial['controls'][k] for k in ('validation_tiles', 'validation_gat_tiles')}))
        for config in candidates:
            predictions = self.predictions(config)
            for key in reference:
                torch.testing.assert_close(predictions[key], reference[key], rtol=5e-5, atol=5e-5)
            del predictions
            self.record('validation_full', config, self.validation_trial, True)
        records = [r for r in self.report['records'] if r['stage'] == 'validation_full']
        baseline = next(r for r in records if r['controls'] == controls(self.base))
        winner = choose(records, baseline)
        self.report['validation_winner'] = winner['config']
        self.save()

    def full_epoch_trial(self, config: dict, trial: int) -> dict:
        """Train and validate all rows, resetting model and optimizer for paired trials."""
        model = self.model(config, trial)
        optimizer = self.optimizer(model, config)
        # Gradient hooks retain finite checking without synchronizing each parameter.
        finite = torch.ones((), dtype=torch.bool, device=DEVICE)
        def check_gradient(gradient):
            finite.logical_and_(torch.isfinite(gradient).all())
            return gradient
        handles = [parameter.register_hook(check_gradient) for parameter in model.parameters()]
        completed = 0
        progress_started = perf_counter()
        def report_progress(optimizer, args, kwargs):
            nonlocal completed
            completed += 1
            budget = config['training']['supervised_rows_per_batch']
            seen = min(completed * budget, len(self.train.indices))
            if completed % 50 == 0 or seen == len(self.train.indices):
                print(json.dumps({'epoch_progress': trial, 'training_rows': seen,
                    'total_training_rows': len(self.train.indices),
                    'elapsed_seconds': perf_counter() - progress_started,
                    **controls(config)}), flush=True)
        progress_handle = optimizer.register_step_post_hook(report_progress)
        try:
            train = epoch(model, train_loader(self.train, config), config, optimizer, self.weights)
            if not finite.item():
                raise FloatingPointError('Nonfinite training gradients.')
        finally:
            progress_handle.remove()
            for handle in handles:
                handle.remove()
        print(json.dumps({'training_complete': trial, 'seconds': train['seconds'],
                          **controls(config)}), flush=True)
        validation = epoch(model, self.val_loader(config), config, class_weights=self.weights)
        if train['observed_rows'] != len(self.train.indices) or validation['observed_rows'] != len(self.validation.indices):
            raise AssertionError('Incomplete matched epoch.')
        result = dict(train=train, validation=validation, seconds=train['seconds']+validation['seconds'])
        for key in ('peak_vram_gib', 'peak_reserved_vram_gib', 'peak_host_gib'):
            result[key] = max(train[key], validation[key])
        measurements(validation['seconds'], config)
        return result

    def full_epochs(self) -> None:
        """Confirm two screening finalists against today's accelerated baseline."""
        training = sorted((r for r in self.report['records'] if r['stage'] == 'training' and 'summary' in r),
                          key=lambda r: r['summary']['seconds'])[:2]
        if len(training) < 2 or 'validation_winner' not in self.report:
            raise ValueError('Complete training and validation sweeps before full epochs.')
        configs = [self.base]
        for record in training:
            candidate = copy.deepcopy(record['config'])
            candidate['performance'] = copy.deepcopy(self.report['validation_winner']['performance'])
            configs.append(resolve_execution(candidate))
        # Interleave independent paired repetitions rather than running one profile first.
        original_repeats = self.args.repeats
        for trial in range(original_repeats):
            self.args.repeats = trial + 1
            for config in configs[trial % len(configs):] + configs[:trial % len(configs)]:
                key = json.dumps(['full_epoch', controls(config)], sort_keys=True)
                existing = next((r for r in self.report['records'] if r['key'] == key), None)
                if existing and len(existing['trials']) < self.args.repeats:
                    existing.pop('summary', None)
                self.record('full_epoch', config, self.full_epoch_trial)
        self.args.repeats = original_repeats
        records = [r for r in self.report['records'] if r['stage'] == 'full_epoch']
        baseline = next(r for r in records if r['controls'] == controls(self.base))
        if 'rejected' in baseline:
            raise RuntimeError('The baseline failed; no candidate can be selected.')
        winner = choose(records, baseline)
        self.report['selected_configuration'] = winner['config']
        self.report['speedup'] = baseline['summary']['seconds']/winner['summary']['seconds']
        self.save()
        print(json.dumps({'selected': controls(winner['config']), 'speedup': self.report['speedup']}), flush=True)


def main(argv: list[str] | None = None) -> None:
    """Run or resume selected stages without touching production defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stage', choices=('all', 'training', 'validation', 'epochs'), default='all')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.steps < 1 or args.repeats < 1:
        parser.error('warmup, steps, and repeats must be positive')
    experiment = Experiment(args)
    if args.stage in ('all', 'training'):
        experiment.training_sweep()
    if args.stage in ('all', 'validation'):
        experiment.validation_sweep()
    if args.stage in ('all', 'epochs'):
        experiment.full_epochs()


if __name__ == '__main__':
    main()

"""Measure execution candidates on real footprints, then matched complete capped epochs.

Run with --full-epochs for uncapped epoch comparisons. No checkpoints or production
configuration are modified. All candidates retain FP32 and the configured geometry.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from dann.config import load_config, resolve_execution, seed_everything
from dann.data_loader import create_data_bundle
from dann.execution import PreparedLoader, device_batches, tensor_bytes
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.sampling import ProportionalSlideTileSampler, SpatialBatchDataset
from dann.tiles import SpatialTileDataset, tile_collate
from dann.train import run_epoch


class TimedLoader:
    """Measure blocking loader time on the unprepared reference path."""
    def __init__(self, source, settings):
        self.source, self.settings = source, settings
        self.dataset, self.sampler = source.dataset, source.sampler
        self.wait_seconds, self.max_batch_bytes = 0., 0
    def __len__(self):
        return len(self.source)
    def __iter__(self):
        self.wait_seconds = 0.
        iterator = iter(self.source)
        while True:
            started = perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            self.wait_seconds += perf_counter() - started
            self.max_batch_bytes = max(self.max_batch_bytes, tensor_bytes(batch))
            yield batch


class Requests:
    """Replay fixed representative and worst-case sampler requests."""
    def __init__(self, requests):
        self.requests = requests
    def __len__(self):
        return len(self.requests)
    def __iter__(self):
        return iter(self.requests)


def train_loader(tiles, config, requests=None):
    sampler = requests or ProportionalSlideTileSampler(tiles, config['training']['seed'],
        config['training']['supervised_rows_per_batch'])
    source = DataLoader(SpatialBatchDataset(tiles), batch_size=None, sampler=sampler,
        num_workers=0)
    return PreparedLoader(source, config['performance'], config['model'])


def measure_training(tiles, config, initial, warmup, steps):
    sampler = ProportionalSlideTileSampler(tiles, config['training']['seed'],
        config['training']['supervised_rows_per_batch'])
    requests = list(sampler)
    # Largest total stored-entry footprint, not just the largest supervision count.
    counts = np.diff(tiles.resident.indptr)
    tile_peaks = [int(counts[g['context_rows']].sum()) for g in tiles.context_rows]
    largest = max(requests, key=lambda r: sum(tile_peaks[i] for i in r.tile_ids))
    selected = requests[:warmup + steps] + [largest]
    loader = train_loader(tiles, config, Requests(selected))
    model = AdversarialLatentFusion.from_config(config, len(tiles.metadata.batch_names), len(config['data']['target_columns'])).cuda()
    model.load_state_dict(initial)
    loss = ZILNLoss.from_config(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(config['training']['beta1'], config['training']['beta2']))
    from dann.data_loader import compute_batch_class_weights
    weights = compute_batch_class_weights(tiles.metadata.batch_codes, tiles.indices).cuda() if config['loss']['balance_batch_classes'] else None
    model.train()
    torch.cuda.reset_peak_memory_stats()
    elapsed, rows, largest_seconds = 0., 0, None
    started = perf_counter()
    for step, (cpu, batch) in enumerate(device_batches(loader, torch.device('cuda'))):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch, grl_strength=.1)
        biology = loss(outputs['pi_logits'], outputs['mu'], outputs['sigma'],
                       batch['targets'], batch['target_valid_mask'], validate_targets=False)
        total = config['loss']['biology_weight'] * biology.total + config['loss']['batch_weight'] * torch.nn.functional.cross_entropy(outputs['batch_logits'], batch['batches'], weight=weights)
        if not torch.isfinite(total):
            raise FloatingPointError('Non-finite benchmark objective.')
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'])
        optimizer.step()
        torch.cuda.synchronize()
        duration = perf_counter() - started
        if step >= warmup and step < warmup + steps:
            elapsed += duration
            rows += len(batch['targets'])
        if step == len(selected)-1:
            largest_seconds = duration
        started = perf_counter()
    result = dict(rows_per_second=rows/elapsed, seconds=elapsed, largest_seconds=largest_seconds,
                  largest_tiles=len(largest.tile_ids), loader_wait_seconds=loader.wait_seconds,
                  peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_vram_gib=torch.cuda.max_memory_reserved()/2**30,
                  peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    if result['peak_vram_gib'] > config['performance']['vram_budget_gib']:
        raise MemoryError(f'24 GiB budget exceeded: {result}')
    return result


def epoch(model, loader, config, optimizer=None, class_weights=None):
    device = torch.device('cuda')
    started = perf_counter()
    result = run_epoch(model, loader, ZILNLoss.from_config(config).to(device), device,
        config['loss']['biology_weight'], config['loss']['batch_weight'], class_weights, optimizer,
        0, config['training']['epochs'], config['training']['grl_schedule'],
        config['training']['grl_max_lambda'], config['training']['grl_gamma'],
        config['training']['gradient_clip_norm'], target_columns=config['data']['target_columns'])
    torch.cuda.synchronize()
    result.setdefault('seconds', perf_counter()-started)
    return result


def main():
    if '--larger-batches' in sys.argv:
        from dann.verification.larger_batch_benchmark import main as larger_main
        larger_main([arg for arg in sys.argv[1:] if arg != '--larger-batches'])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--epoch-rows', type=int, default=8192)
    parser.add_argument('--epoch-validation-tiles', type=int, default=128)
    parser.add_argument('--full-epochs', action='store_true')
    parser.add_argument('--skip-sweep', action='store_true')
    parser.add_argument('--selection-from', type=Path, help='Reuse a previous sweep for full matched epochs.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_config(args.config)
    config['performance'].update(ram_cache=True, cache_geometry=True,
        preparation_workers=2, prepared_batches=2, transfer_prefetch=True, torch_threads=4)
    config = resolve_execution(config)
    seed_everything(config['training']['seed'])
    started = perf_counter()
    bundle = create_data_bundle(config)
    tiles = bundle.datasets['train']
    report = dict(startup_seconds=perf_counter()-started,
                  resident_gib=tiles.resident.nbytes/2**30,
                  geometry_gib=tiles.geometry_bytes/2**30,
                  gpu=torch.cuda.get_device_name(), config=config, candidates=[], validation=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    print(json.dumps({k: report[k] for k in ('startup_seconds', 'resident_gib', 'geometry_gib', 'gpu')}), flush=True)
    initial_model = AdversarialLatentFusion.from_config(config, len(bundle.metadata.batch_names), len(config['data']['target_columns']))
    initial = copy.deepcopy(initial_model.state_dict())
    del initial_model
    best = copy.deepcopy(config)
    previous = json.loads(args.selection_from.read_text()) if args.selection_from else None
    if previous:
        best['model'].update(previous['selected_execution']['model'])
        best['training']['supervised_rows_per_batch'] = previous['selected_execution']['training_rows']
    if not args.skip_sweep and previous is None:
        # Coordinate sweep: tune execution axes independently, then retain the fastest passing candidate.
        axes = [('rows', [1024, 2048, 4096, 8192]), ('peaks', [65536, 131072, 262144, 524288]),
                ('gat', [8, 16, 32, 64]), ('spectral_checkpointing', [True, False]),
                ('gatv2_checkpointing', [True, False])]
        for axis, values in axes:
            passing = []
            for value in values:
                candidate = copy.deepcopy(best)
                if axis == 'rows':
                    candidate['training']['supervised_rows_per_batch'] = value
                elif axis == 'peaks':
                    candidate['model']['spectral_peak_budget'] = value
                elif axis == 'gat':
                    candidate['model']['gatv2_tile_budget'] = value
                else:
                    candidate['model'][axis] = value
                record = dict(axis=axis, value=value)
                try:
                    record.update(measure_training(tiles, candidate, initial, args.warmup, args.steps))
                    passing.append((record['rows_per_second'], candidate))
                except (torch.cuda.OutOfMemoryError, MemoryError) as error:
                    record['rejected'] = str(error)
                report['candidates'].append(record)
                print(json.dumps(record), flush=True)
                save()
                gc.collect(); torch.cuda.empty_cache()
            if passing:
                best = max(passing, key=lambda x: x[0])[1]
    report['selected_execution'] = dict(model=best['model'], training_rows=best['training']['supervised_rows_per_batch'])
    # Validation comparisons include unique-context feature construction on every pass.
    validation = bundle.datasets['validation']
    if not args.full_epochs:
        selected = np.array([r for _, pairs in validation.tiles[:args.epoch_validation_tiles] for r, _ in pairs])
        validation = SpatialTileDataset(config, selected, bundle.metadata, geometry=tiles.geometry,
            resident=tiles.resident, spatial_cache=tiles.spatial_cache)
    model = AdversarialLatentFusion.from_config(best, len(bundle.metadata.batch_names), len(config['data']['target_columns'])).cuda()
    model.load_state_dict(initial)
    validation_reference = None
    if previous:
        report['validation'] = previous['validation']
        report['validation_sweep_source'] = str(args.selection_from)
    class_weights = bundle.batch_class_weights.cuda() if config['loss']['balance_batch_classes'] else None
    for cached in (() if previous else (False, True)):
        for physical in (8, 32, 64, 128):
            candidate = copy.deepcopy(best)
            candidate['performance'].update(validation_tiles=physical,
                validation_feature_cache=cached, validation_peak_budget=262144, validation_gat_tiles=32)
            loader = PreparedLoader(DataLoader(validation, batch_size=physical, collate_fn=tile_collate),
                candidate['performance'], candidate['model'], validation=True)
            try:
                record = epoch(model, loader, candidate, class_weights=class_weights)
                metrics = {k: v for k, v in record.items() if k.endswith(('loss', 'accuracy')) or k.startswith(('hurdle_acc', 'mean_r2'))}
                if validation_reference is None:
                    validation_reference = metrics
                for key, value in metrics.items():
                    if not np.isclose(value, validation_reference[key], rtol=5e-5, atol=5e-5, equal_nan=True):
                        raise AssertionError(f'Validation metric changed: {key}: {value} vs {validation_reference[key]}')
                record.update(tiles=physical, feature_cache=cached)
                report['validation'].append(record)
                print(json.dumps(record), flush=True); save()
            except (torch.cuda.OutOfMemoryError, MemoryError) as error:
                report['validation'].append(dict(tiles=physical, feature_cache=cached, rejected=str(error)))
                save()
            gc.collect(); torch.cuda.empty_cache()
    winner = min((r for r in report['validation'] if 'rejected' not in r), key=lambda r: r['seconds'])
    best['performance'].update(validation_tiles=winner['tiles'], validation_feature_cache=winner['feature_cache'],
                               validation_peak_budget=262144, validation_gat_tiles=32)
    report['selected_configuration'] = copy.deepcopy(best)
    del model
    gc.collect(); torch.cuda.empty_cache()
    # Same initial state, row budgets, splits, optimizer, and epoch-based GRL in both paths.
    # Optimizer row-budget sweep is reported separately because it changes the trajectory.
    best['training']['supervised_rows_per_batch'] = config['training']['supervised_rows_per_batch']
    requests = list(ProportionalSlideTileSampler(tiles, config['training']['seed'], config['training']['supervised_rows_per_batch']))
    selected = tiles.indices if args.full_epochs else np.array([r for req in requests for r in req.row_ids][:args.epoch_rows])
    report['matched_epochs'] = dict(training_rows=len(selected), validation_rows=len(validation.indices), full=args.full_epochs)
    save()
    states = []
    for accelerated in (False, True):
        candidate = copy.deepcopy(best if accelerated else config)
        if not accelerated:
            candidate['performance'].update(ram_cache=False, cache_geometry=False, preparation_workers=0,
                transfer_prefetch=False, validation_tiles=None, validation_feature_cache=False)
        datasets = {name: SpatialTileDataset(candidate, rows, bundle.metadata, geometry=tiles.geometry,
            resident=tiles.resident if accelerated else None,
            spatial_cache=tiles.spatial_cache if accelerated else None)
            for name, rows in [('train', selected), ('validation', validation.indices)]}
        if accelerated:
            training_loader = train_loader(datasets['train'], candidate)
            validation_loader = PreparedLoader(DataLoader(datasets['validation'],
                batch_size=candidate['performance']['validation_tiles'], collate_fn=tile_collate),
                candidate['performance'], candidate['model'], validation=True)
        else:
            sampler = ProportionalSlideTileSampler(datasets['train'], candidate['training']['seed'],
                candidate['training']['supervised_rows_per_batch'])
            training_loader = DataLoader(SpatialBatchDataset(datasets['train']), batch_size=None,
                sampler=sampler, num_workers=candidate['training']['num_workers'], pin_memory=True,
                prefetch_factor=candidate['training']['prefetch_factor'] if candidate['training']['num_workers'] else None)
            validation_loader = DataLoader(datasets['validation'], batch_size=candidate['training']['tiles_per_batch'],
                collate_fn=tile_collate, num_workers=candidate['training']['num_workers'], pin_memory=True,
                prefetch_factor=candidate['training']['prefetch_factor'] if candidate['training']['num_workers'] else None)
        seed_everything(config['training']['seed'])
        model = AdversarialLatentFusion.from_config(candidate, len(bundle.metadata.batch_names), len(config['data']['target_columns'])).cuda()
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'], betas=(config['training']['beta1'], config['training']['beta2']))
        if not accelerated:
            training_loader = TimedLoader(training_loader, candidate['performance'])
            validation_loader = TimedLoader(validation_loader, candidate['performance'])
        print(f'Starting {"accelerated" if accelerated else "backed"} full training pass ({len(selected)} rows)', flush=True)
        train = epoch(model, training_loader, candidate, optimizer, class_weights)
        print(f'Training complete in {train["seconds"]:.2f}s; starting validation ({len(validation.indices)} rows)', flush=True)
        val = epoch(model, validation_loader, candidate, class_weights=class_weights)
        label = 'accelerated' if accelerated else 'backed'
        report['matched_epochs'][label] = dict(train=train, validation=val, total_seconds=train['seconds']+val['seconds'])
        print(json.dumps({label: report['matched_epochs'][label]}), flush=True); save()
        states.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
        del optimizer, model
        for dataset in datasets.values():
            dataset.close()
        gc.collect(); torch.cuda.empty_cache()
    report['matched_epochs']['max_parameter_delta'] = max(float((states[0][k]-states[1][k]).abs().max()) for k in states[0])
    save()


if __name__ == '__main__':
    main()

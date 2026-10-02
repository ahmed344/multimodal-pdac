"""Bounded, read-only real-data parity audit of the activated production sampler.

Writes isolated artifacts; never evaluates test labels or runs a full fit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import signal
import subprocess
import time
import traceback

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dann.config import load_config, seed_everything
from dann.data_loader import create_data_bundle
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.sampling import ProportionalSlideTileSampler, SamplingEpochAudit, SpatialBatchDataset
from dann.tiles import SpatialTileDataset
from dann.train import run_epoch
from dann.verification.batch_execution_diagnosis import Plan, PlannedDataset


class LimitedLoader:
    """Bound consumption while retaining the production GRL epoch length."""
    def __init__(self, loader, limit):
        self.loader, self.limit = loader, limit
        self.sampler, self.dataset = loader.sampler, loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return itertools.islice(self.loader, self.limit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path(
        'data/PDAC/Results/dann/diagnostics/2026-10-01-batch-execution-1000'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(passed=False, budget_seconds=1200, checks={}, test_rows_scored=0)
    def save():
        report['elapsed_seconds'] = time.monotonic() - start
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    def record(name, value):
        report['checks'][name] = value
        save()
        print(name + ': complete', flush=True)
    def deadline(*_):
        raise TimeoutError('20-minute real-data audit deadline reached.')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(1200)
    bundle = None
    try:
        torch.set_num_threads(4)
        config = load_config(Path('dann/config.yaml'))
        (args.output / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        with np.load(args.archive / 'splits.npz') as source:
            splits = {k: source[k] for k in source.files}
        identity = json.loads((args.archive / 'data_identity.json').read_text())
        bundle = create_data_bundle(config, splits, identity)
        assert (bundle.metadata.num_observations, bundle.metadata.num_peaks,
                len(bundle.metadata.batch_names)) == (1698428, 5808, 43)
        assert [len(bundle.split_indices[k]) for k in ('train', 'validation', 'test')] == [1358726, 169824, 169878]
        for k in splits:
            np.testing.assert_array_equal(bundle.split_indices[k], splits[k])
        tiles = bundle.datasets['train']
        assert tiles.core_size == 8 and tiles.halo == 6 and tiles.size == 20
        record('data_contract', dict(rows=1698428, features=5808, slides=43,
            splits={k: len(v) for k, v in splits.items()}, identity=identity['row_digest'],
            halo=tiles.halo, footprint=tiles.size))
        plans, summaries = {}, []
        largest = (-1, None, None, None)
        with h5py.File(config['data']['path'], 'r') as handle:
            counts = np.diff(handle['X/indptr'][:])
        # Full original footprints, including repeated context in different tiles.
        tile_peaks = []
        for (slide, tx, ty), _ in tiles.tiles:
            x0, y0 = tx * 8 - tiles.halo, ty * 8 - tiles.halo
            tile_peaks.append(sum(int(counts[r]) for y in range(y0, y0 + tiles.size)
                for x in range(x0, x0 + tiles.size)
                if (r := tiles.lookup.get((slide, x, y))) is not None))
        for seed in (20260719, 20260720, 20260721):
            reference = Plan.load(args.archive / f'plan_{seed}_Q.npz')
            plans[seed] = reference
            sampler = ProportionalSlideTileSampler(tiles, seed, 2048)
            for epoch in range(2):
                sampler.set_epoch(epoch)
                audit = SamplingEpochAudit(epoch, len(tiles.indices))
                seen = []
                for step, request in enumerate(sampler):
                    index = int(reference.passes[epoch]) + step
                    np.testing.assert_array_equal(request.row_ids, reference.batch(index))
                    np.testing.assert_array_equal(request.tile_ids, reference.tile_order[
                        reference.tile_boundaries[index]:reference.tile_boundaries[index + 1]])
                    seen.extend(request.row_ids)
                    audit.update(dict(row_ids=torch.tensor(request.row_ids),
                        batches=torch.from_numpy(bundle.metadata.batch_codes[list(request.row_ids)]),
                        occupancy=torch.empty((len(request.tile_ids), 0))))
                    peaks = sum(tile_peaks[t] for t in request.tile_ids)
                    if peaks > largest[0]:
                        largest = (peaks, seed, epoch, request)
                np.testing.assert_array_equal(np.sort(seen), np.sort(tiles.indices))
                summaries.append(dict(seed=seed, archive_plan_sha256=reference.sha256, **audit.summary()))
        record('six_complete_passes', summaries)
        dataset = SpatialBatchDataset(tiles)
        sampler = bundle.loaders['train'].sampler
        requests = list(sampler)
        split_index = next(i + 1 for i, (a, b) in enumerate(zip(requests, requests[1:]))
                           if set(a.tile_ids) & set(b.tile_ids))
        sampler.set_epoch(1)
        next_request = next(iter(sampler))
        reference = PlannedDataset(tiles, plans[20260719], True)
        checked = []
        for name, index, request in [('first', 0, requests[0]),
                ('partial_tile', split_index, requests[split_index]),
                ('final_partial', len(requests) - 1, requests[-1]),
                ('next_epoch', len(requests), next_request)]:
            actual, expected = dataset[request], reference[index]
            assert actual.keys() == expected.keys()
            for k in actual:
                torch.testing.assert_close(actual[k], expected[k], atol=0, rtol=0, equal_nan=True)
            checked.append(dict(name=name, rows=len(request.row_ids), tiles=len(request.tile_ids),
                                active_peaks=len(actual['peak_indices'])))
            del actual, expected
        record('real_batch_field_parity', checked)
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if processes:
            raise RuntimeError('GPU busy with another compute process; no process interrupted: ' + processes)
        assert torch.cuda.is_available()
        device = torch.device('cuda')
        training = config['training']
        loss = ZILNLoss.from_config(config).to(device)
        weights = bundle.batch_class_weights.to(device) if config['loss']['balance_batch_classes'] else None
        def fresh():
            seed_everything(training['seed'], training['deterministic_algorithms'])
            model = AdversarialLatentFusion.from_config(config, 43, 4).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
                weight_decay=training['weight_decay'], betas=(training['beta1'], training['beta2']))
            return model, opt
        def execute(model, loader, optimizer, directory):
            return run_epoch(model, loader, loss, device, config['loss']['biology_weight'],
                config['loss']['batch_weight'], weights, optimizer, 0, training['epochs'],
                training['grl_schedule'], training['grl_max_lambda'], training['grl_gamma'],
                training['gradient_clip_norm'], target_columns=config['data']['target_columns'],
                sampling_audit_dir=directory)
        model, opt = fresh()
        torch.cuda.reset_peak_memory_stats()
        before = time.monotonic()
        peak_loader = DataLoader(dataset, batch_size=None, sampler=[largest[3]], num_workers=0)
        metrics = execute(model, peak_loader, opt, args.output / 'preflight_sampling')
        torch.cuda.synchronize()
        record('largest_context_preflight', dict(active_peaks=largest[0], seed=largest[1],
            epoch=largest[2], tiles=len(largest[3].tile_ids), supervised_rows=len(largest[3].row_ids),
            seconds=time.monotonic() - before, allocated_bytes=torch.cuda.max_memory_allocated(),
            reserved_bytes=torch.cuda.max_memory_reserved(), metrics=metrics))
        del model, opt
        torch.cuda.empty_cache()
        model, opt = fresh()
        torch.cuda.reset_peak_memory_stats()
        before = time.monotonic()
        metrics = execute(model, LimitedLoader(bundle.loaders['train'], 10), opt, args.output / 'training_sampling')
        torch.cuda.synchronize()
        actual_sampling = json.loads((args.output / 'training_sampling/epoch_0000_summary.json').read_text())
        reference = plans[20260719]
        assert actual_sampling['batches'] == 10 and actual_sampling['processed_rows'] == 20480
        assert actual_sampling['row_order_sha256'] == hashlib.sha256(
            reference.rows[:20480].astype('<i8').tobytes()).hexdigest()
        assert actual_sampling['boundaries_sha256'] == hashlib.sha256(
            reference.boundaries[:11].astype('<i8').tobytes()).hexdigest()
        record('ten_production_updates', dict(metrics=metrics, seconds=time.monotonic() - before,
            allocated_bytes=torch.cuda.max_memory_allocated(), reserved_bytes=torch.cuda.max_memory_reserved(),
            loader_workers=training['num_workers'], loader_prefetch=training['prefetch_factor'],
            full_epoch_length=len(bundle.loaders['train'])))
        # Contiguous original validation tiles keep this bounded and exercise full footprints.
        validation_rows = np.array([r for _, selected in bundle.datasets['validation'].tiles[:8]
                                   for r, _ in selected], dtype=np.int64)
        validation = SpatialTileDataset(config, validation_rows, bundle.metadata, geometry=tiles.geometry)
        from dann.tiles import tile_collate
        validation_loader = DataLoader(validation, batch_size=training['tiles_per_batch'], collate_fn=tile_collate)
        seen = torch.cat([b['row_ids'] for b in validation_loader]).numpy()
        np.testing.assert_array_equal(np.sort(seen), np.sort(validation_rows))
        assert len(np.unique(seen)) == len(validation_rows)
        metrics = execute(model, validation_loader, None, None)
        record('bounded_spatial_validation', dict(rows=len(seen), tiles=len(validation),
            row_sha256=hashlib.sha256(seen.astype('<i8').tobytes()).hexdigest(), exact_coverage=True, metrics=metrics))
        validation.close()
        report['passed'] = True
    except Exception as error:
        report['failure'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        signal.alarm(0)
        save()
        if bundle is not None:
            for loader in bundle.loaders.values():
                if loader._iterator is not None:
                    loader._iterator._shutdown_workers()
            for dataset in bundle.datasets.values():
                dataset.close()
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Reproducible spatial inference stage screening and complete export comparisons.

Run only on an idle GPU. Artifacts are written below --output, never production paths.
Each candidate receives one warmup and three timed repetitions. Complete exports
retain checkpoint sampling counts and seeds unless explicitly overridden.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import platform
import json
from pathlib import Path
import statistics
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from dann.config import load_config, spatial_execution_settings
from dann.execution import (PreparedLoader, ResidentSpectra, build_validation_features,
                            device_batches, validation_execution)
from dann.spatial import load_checkpoint_model, run_inference
from dann.spatial_execution import InferenceProgress, density_blocks, inference_performance
from dann.tiles import SpatialTileDataset, tile_collate


def timed(function, device, repetitions=3):
    """Warm up, synchronize, and record every repetition, retaining variability."""
    times, memory = [], []
    for i in range(repetitions+1):
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        function()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        seconds = perf_counter()-start
        if not i:
            warmup = seconds
        if i:
            times.append(seconds)
            memory.append(torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0.)
    return dict(seconds=times, warmup_seconds=warmup, median_seconds=statistics.median(times),
                spread_seconds=max(times)-min(times), peak_allocated_gib=max(memory))


def compare_exports(left: Path, right: Path) -> dict:
    """Compare complete artifacts with bounded reads; identity/support are exact."""
    report = {}
    for name in ('predictions.parquet', 'latent.parquet', 'predictions.support.parquet'):
        a, b = pq.ParquetFile(left/name), pq.ParquetFile(right/name)
        if not a.schema_arrow.equals(b.schema_arrow) or a.metadata.num_rows != b.metadata.num_rows:
            raise AssertionError(f'Schema/row mismatch: {name}')
        maxima = {column: 0. for column in a.schema_arrow.names if column not in ('row_position', 'obs_name')}
        failures = {key: 0 for key in maxima}
        examples = {key: [] for key in maxima}
        count = 0
        for aa, bb in zip(a.iter_batches(batch_size=8192), b.iter_batches(batch_size=8192), strict=True):
            for key in aa.schema.names:
                x, y = aa.column(key).to_numpy(zero_copy_only=False), bb.column(key).to_numpy(zero_copy_only=False)
                if key in ('row_position', 'obs_name') or 'support' in key:
                    np.testing.assert_array_equal(x, y)
                else:
                    delta = np.abs(x-y)
                    bad = np.flatnonzero(~np.isfinite(x) | ~np.isfinite(y) | (delta > 5e-5 + 5e-5*np.abs(y)))
                    failures[key] += len(bad)
                    examples[key].extend((count+bad[:max(0, 10-len(examples[key]))]).tolist())
                    maxima[key] = max(maxima[key], float(np.max(delta)))
            count += len(aa)
        report[name] = dict(rows=count, max_absolute_difference=maxima,
                            outside_tolerance=failures, example_rows={k: v for k, v in examples.items() if v},
                            within_tolerance=not any(failures.values()))
    return report


def screen(config, args):
    """Screen physical batches, spectral limits, cache chunks, queues and density."""
    model, saved, targets, device, _ = load_checkpoint_model(args.checkpoint, args.device)
    torch.set_num_threads(config['performance']['torch_threads'] or 4)
    settings = spatial_execution_settings(config['analysis'].get('spatial_execution'))
    performance = inference_performance(saved, {'performance': config['performance']}, settings)
    saved['performance'] = performance
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Benchmark directory is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path('dann').rglob('*.py')}
    manifest = dict(config=config, checkpoint=str(args.checkpoint.resolve()),
                    input=str(args.input.resolve()), rows=args.rows,
                    repetitions=args.repetitions, python=platform.python_version(),
                    torch=torch.__version__, numpy=np.__version__, source_sha256=source_hashes,
                    gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    cuda=torch.version.cuda)
    for name, path in [('checkpoint_identity', args.checkpoint), ('source_identity', args.input)]:
        stat = path.stat()
        manifest[name] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    records = []
    def record(stage, workload, candidate, fn):
        try:
            result = timed(fn, device, args.repetitions)
        except (MemoryError, torch.OutOfMemoryError) as error:
            result = dict(rejected=str(error))
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        records.append(dict(stage=stage, workload=workload, candidate=candidate, **result))
        (output/'screen.json').write_text(json.dumps(records, indent=2))
        print(records[-1], flush=True)
    density_parameters = None
    with InferenceProgress() as progress, torch.inference_mode():
        resident = ResidentSpectra(args.input, saved['data'], int(performance['host_budget_gib']*2**30))
        for rows in args.rows:
            if rows > len(resident.indptr)-1:
                raise ValueError('Screen workload exceeds source rows.')
            first = 0
            if rows == max(args.rows) and len(args.rows) > 1:
                starts = np.arange(0, len(resident.indptr)-rows, max(1, rows//4))
                masses = resident.indptr[starts+rows] - resident.indptr[starts]
                first = int(starts[np.argmax(masses)])
            manifest.setdefault('workloads', {})[str(rows)] = dict(
                first_row=first, stop_row=first+rows,
                stored_entries=int(resident.indptr[first+rows]-resident.indptr[first]))
            (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
            dataset = SpatialTileDataset(saved, np.arange(first, first+rows), input_path=args.input, resident=resident)
            parameters = np.zeros((rows, 1+3*len(targets)+2), dtype=np.float32)
            loader = PreparedLoader(DataLoader(dataset, batch_size=settings['tiles_per_batch'], collate_fn=tile_collate),
                                    performance, saved['model'], validation=True)
            loader.inference = True
            with validation_execution(model, performance):
                for peaks in (262144, 524288, 1048576, 2097152):
                    for chunk in (1024, 2048, 4096):
                        candidate = dict(spectral_peak_budget=peaks, feature_cache_rows=chunk)
                        def encode():
                            loader.settings = {**performance, 'validation_peak_budget': peaks, 'feature_cache_rows': chunk}
                            with validation_execution(model, loader.settings):
                                build_validation_features(model, loader, device, progress)
                        record('feature encoding', rows, candidate, encode)
                loader.settings = performance
                cache = build_validation_features(model, loader, device, progress)
                lookup, features = cache
                for tiles in (8, 32, 64, 128, 256):
                    for gat in (32, 64, 128):
                        if gat > tiles and gat != 32:
                            continue
                        candidate = dict(tiles_per_batch=tiles, gat_tiles=gat)
                        loader = PreparedLoader(DataLoader(dataset, batch_size=tiles, collate_fn=tile_collate),
                                                {**performance, 'validation_gat_tiles': gat}, saved['model'], validation=True)
                        loader.pooled_only = True
                        def predict():
                            with validation_execution(model, loader.settings):
                                for cpu, batch in device_batches(loader, device):
                                    batch['pooled_features'] = features[torch.as_tensor(lookup[cpu['context_row_ids'].numpy()], device=device)]
                                    outputs = model(batch, grl_strength=0.)
                                    packed = torch.cat([outputs[k] for k in ('latent', 'pi', 'mu', 'sigma')], dim=1).cpu().numpy()
                                    parameters[cpu['row_ids'].numpy()-first, 1:1+3*len(targets)] = packed[:, model.latent_dim:]
                        record('spatial prediction', rows, candidate, predict)
                for workers in (2, 4, 8):
                    loader = PreparedLoader(DataLoader(dataset, batch_size=64, collate_fn=tile_collate),
                                            {**performance, 'preparation_workers': workers, 'prepared_batches': 2*workers},
                                            saved['model'], validation=True)
                    loader.pooled_only = True
                    record('preparation', rows, dict(preparation_workers=workers, prepared_batches=2*workers), predict)
                del features, lookup, cache, loader
            if rows == max(args.rows):
                density_parameters = parameters
            dataset.close()
            del dataset
            gc.collect()
        del resident
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        # Frozen FP32 parameters from real output; all candidates receive identical input.
        if args.parameters is not None:
            table = pq.read_table(args.parameters).slice(0, max(args.rows))
            means = np.column_stack([table[f'mu_{t}'].to_numpy() for t in targets]) if f'mu_{targets[0]}' in table.column_names else np.column_stack([table[f'predicted_{t}'].to_numpy() for t in targets])
            pi = 1-np.column_stack([table[f'prob_of_presence_{t}'].to_numpy() for t in targets])
            sigma = np.column_stack([table[f'sigma_{t}'].to_numpy() for t in targets])
            values = np.concatenate([means[:, :1], pi, means, sigma, means[:, :2]], axis=1).astype('float32')
        else:
            values = density_parameters
        path = output/'density_parameters.bin'
        values.tofile(path)
        for rows in args.rows:
            for workers in (0, 2, 4, 8, 16):
                record('density sampling', rows, dict(density_workers=workers),
                       lambda: list(density_blocks(path, (rows, values.shape[1]), 1, len(targets),
                                                  settings['output_rows'], workers, 1000, 20260719, performance)))
    return records



def density_steady(config, args):
    """Measure steady sampling with enough tasks to occupy all 16 workers.

    Process startup is captured by the warmup. Complete exports separately measure
    startup, staging and writing. All workers read the same frozen FP32 file.
    """
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from dann.spatial_execution import _init_density, _density_range
    path = args.parameters
    if path is None:
        raise ValueError('--parameters must name the frozen density_parameters.bin file.')
    targets = len(config['data']['target_columns'])
    columns = 1+3*targets+2
    rows = path.stat().st_size//(columns*4)
    block = min(config['analysis']['spatial_execution']['output_rows'], max(1, rows//32))
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for workers in (0, 2, 4, 8, 16):
        def serial():
            list(density_blocks(path, (rows, columns), 1, targets, block, 0, 1000, 20260719))
        if workers:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=_init_density,
                                     initargs=(str(path), (rows, columns), 1, targets, 1000, 20260719)) as pool:
                def parallel():
                    ranges = iter((start, min(start+block, rows)) for start in range(0, rows, block))
                    pending = deque()
                    for _ in range(2*workers):
                        pair = next(ranges, None)
                        if pair is not None:
                            pending.append(pool.submit(_density_range, *pair))
                    while pending:
                        pending.popleft().result()
                        pair = next(ranges, None)
                        if pair is not None:
                            pending.append(pool.submit(_density_range, *pair))
                result = timed(parallel, torch.device('cpu'), args.repetitions)
        else:
            result = timed(serial, torch.device('cpu'), args.repetitions)
        records.append(dict(workers=workers, rows=rows, block_rows=block, **result))
        print(records[-1], flush=True)
        (args.output/'density_steady.json').write_text(json.dumps(records, indent=2))
    return records


def density_exact(config, args):
    """Verify bitwise float64 means across all screened worker counts and blocks."""
    if args.parameters is None:
        raise ValueError('--parameters must name a frozen density_parameters.bin file.')
    targets = len(config['data']['target_columns'])
    columns = 1+3*targets+2
    rows = args.parameters.stat().st_size//(columns*4)
    args.output.mkdir(parents=True, exist_ok=False)
    records, reference = [], None
    for workers, block in ((0, 4096), (2, 16384), (4, 8192), (8, 2048), (16, 4096)):
        digest, completed = hashlib.sha256(), 0
        for start, means in density_blocks(args.parameters, (rows, columns), 1, targets,
                                           block, workers, 1000, 20260719):
            if start != completed or means.dtype != np.float64:
                raise AssertionError('Density block identity or dtype changed.')
            digest.update(means.tobytes())
            completed += len(means)
        value = digest.hexdigest()
        if completed != rows or (reference is not None and value != reference):
            raise AssertionError('Density means differ across worker counts or block sizes.')
        reference = value
        records.append(dict(workers=workers, block_rows=block, rows=rows, sha256=value))
        print(records[-1], flush=True)
    report = dict(samples=1000, seed=20260719, float64_bitwise_identical=True,
                  parameters_sha256=hashlib.sha256(args.parameters.read_bytes()).hexdigest(), trials=records)
    (args.output/'density_equivalence.json').write_text(json.dumps(report, indent=2))
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['screen', 'density-steady', 'density-exact', 'export', 'compare'])
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--rows', type=int, nargs='+', default=[16384, 131072])
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--parameters', type=Path)
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--left', type=Path)
    parser.add_argument('--right', type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    args.input = args.input or Path(config['data']['context_path'])
    args.checkpoint = args.checkpoint or Path(config['analysis']['checkpoint'])
    if args.mode == 'screen':
        screen(config, args)
    elif args.mode == 'density-exact':
        density_exact(config, args)
    elif args.mode == 'density-steady':
        density_steady(config, args)
    elif args.mode == 'export':
        run_inference(args.input, args.checkpoint, args.output/'predictions.parquet',
                      latent_output_path=args.output/'latent.parquet', device_override=args.device,
                      execution_overrides={'spatial_execution': {'enabled': False} if args.legacy else config['analysis']['spatial_execution'],
                                           'performance': config['performance']})
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        comparison = compare_exports(args.left, args.right)
        (args.output/'comparison.json').write_text(json.dumps(comparison, indent=2))
        if not all(artifact['within_tolerance'] for artifact in comparison.values()):
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Matched capped sync/async GPU runs; writes only to a new isolated result root."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from dann.config import apply_smoke_overrides, load_config, resolve_execution
from dann.train import train_model
from dann.training_artifacts import atomic_json


def compare_runs(sync_path: Path, async_path: Path, *, exact: bool = False) -> dict:
    """Compare numerical state/history while excluding execution-time diagnostics."""
    differences = {}
    for name in ('best.pt', 'latest.pt'):
        left = torch.load(sync_path / name, map_location='cpu', weights_only=False)
        right = torch.load(async_path / name, map_location='cpu', weights_only=False)
        for key in ('epoch', 'best_epoch', 'selection_metric', 'selection_start_epoch',
                    'early_stopping_state', 'architecture', 'sampler_state'):
            assert left[key] == right[key], (name, key)
        for branch in ('model_state', 'optimizer_state'):
            maximum = 0.

            def compare(a, b):
                nonlocal maximum
                if isinstance(a, torch.Tensor):
                    torch.testing.assert_close(a, b, rtol=0. if exact else 5e-5, atol=0. if exact else 1e-5)
                    if a.numel():
                        maximum = max(maximum, float((a.double() - b.double()).abs().max()))
                elif isinstance(a, dict):
                    assert a.keys() == b.keys()
                    for key in a:
                        compare(a[key], b[key])
                elif isinstance(a, (tuple, list)):
                    assert len(a) == len(b)
                    for aa, bb in zip(a, b):
                        compare(aa, bb)
                else:
                    assert a == b

            compare(left[branch], right[branch])
            differences[f'{name}_{branch}_max_abs_difference'] = maximum
    a, b = (pd.read_csv(path / 'history.csv') for path in (sync_path, async_path))
    keys = [key for key in a if any(token in key for token in ('loss', 'objective', 'r2', 'accuracy', 'hurdle_acc'))]
    np.testing.assert_allclose(a[keys], b[keys], rtol=0. if exact else 5e-5,
                               atol=0. if exact else 1e-5, equal_nan=True)
    assert a.epoch.tolist() == b.epoch.tolist()
    differences['history_max_abs_difference'] = float(np.nanmax(np.abs(a[keys].to_numpy() - b[keys].to_numpy())))
    for split in ('train', 'validation'):
        ihc = b[f'{split}_biology_weighted_loss']
        detailed = sum(b[f'{split}_{branch}_weighted_loss_{signal}']
                       for signal in ('cd8', 'tumor', 'collagen', 'stroma')
                       for branch in ('hurdle', 'positive'))
        np.testing.assert_allclose(ihc, detailed, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(b[f'{split}_supervised_objective'],
                                   ihc + b.get(f'{split}_he_weighted_loss', 0.), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(b[f'{split}_total_loss'],
                                   b[f'{split}_supervised_objective'] + b[f'{split}_batch_weighted_loss'], rtol=1e-6, atol=1e-6)
    for path in (sync_path / 'sampling').glob('*'):
        assert path.read_bytes() == (async_path / 'sampling' / path.name).read_bytes()
    return differences


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--architectures', nargs='+', choices=['mlp', 'cnn', 'gatv2'], default=['mlp', 'cnn', 'gatv2'])
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic GPU kernels and require exact numerical agreement.')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    base = load_config(args.config)
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        base['training']['deterministic_algorithms'] = True
    base['training'].update(smoke_epochs=3, smoke_train_samples=128,
                            smoke_validation_samples=64, smoke_test_samples=32)
    base['results']['root'] = str(args.root.resolve())
    results = {}
    for kind in args.architectures:
        paths, measured = {}, {}
        for mode in ('sync', 'async'):
            config = copy.deepcopy(base)
            for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
                group['type'] = kind
            config['results']['run_name'] = mode
            config['performance']['async_artifacts'] = mode == 'async'
            config = apply_smoke_overrides(resolve_execution(config))
            print(f'Starting {kind} {mode}', flush=True)
            started = perf_counter()
            best = train_model(config)
            paths[mode] = best.parent
            timings = json.loads((best.parent / 'artifact_timings.json').read_text())
            timings['wall_seconds_including_data_setup'] = perf_counter() - started
            timings['mean_boundary_seconds'] = float(np.mean([row['boundary_seconds'] for row in timings['epochs']]))
            measured[mode] = timings
            gc.collect()
            torch.cuda.empty_cache()
        results[kind] = dict(paths={key: str(value) for key, value in paths.items()},
                             timings=measured, deterministic=args.deterministic,
                             comparison=compare_runs(paths['sync'], paths['async'], exact=args.deterministic))
        atomic_json(args.root / 'benchmark.json', results)
        print(json.dumps({kind: results[kind]}), flush=True)


if __name__ == '__main__':
    main()

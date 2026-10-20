"""Paired scalar/batched pixel loading on frozen row requests, including GPU work.

Outputs belong in an isolated diagnostic directory. No production configuration,
checkpoint, dataset, allocator setting, or OS file cache is modified. RAM uses
the configured preparation threads; backed loading uses zero preparation threads
as required by the existing execution settings validator. Both implementations
within each storage comparison use identical settings and persistent transfers.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import threading
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dann.config import load_config, performance_settings, seed_everything
from dann.data_loader import DataBundle, SparseAnnDataDataset, create_data_bundle, sparse_collate
from dann.execution import PreparedLoader
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.train import run_epoch


class ScalarPixels(Dataset):
    """Expose only the original scalar interface to the unchanged prepared loader."""
    def __init__(self, source: SparseAnnDataDataset) -> None:
        self.source = source
        self.resident, self.indices, self.metadata = source.resident, source.indices, source.metadata

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, item: int) -> dict:
        return self.source[item]


class MeasuredPreparation(PreparedLoader):
    """Record aggregate preparation wall time (threads may overlap)."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.preparation_seconds = 0.

    def prepare(self, request: list[int] | np.ndarray) -> dict:
        started = perf_counter()
        batch = super().prepare(request)
        with self._metrics_lock:
            self.preparation_seconds += perf_counter() - started
        return batch


class GPUUsage:
    """Sample device utilization independently of model/loader timing."""
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, name='pixel-benchmark-gpu', daemon=True)

    def _sample(self) -> None:
        while not self.stop.is_set():
            try:
                value = subprocess.check_output([
                    'nvidia-smi', '-i', str(self.device.index or 0),
                    '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                    text=True, timeout=5).strip().split(',')
                self.samples.append(dict(time=perf_counter(), utilization=float(value[0]),
                                         memory_mib=float(value[1])))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(.5)

    def __enter__(self) -> GPUUsage:
        if self.device.type == 'cuda':
            self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join()


def freeze_requests(bundle: DataBundle, warmup: int, steps: int,
                    validation_steps: int) -> dict[str, list[np.ndarray]]:
    """Freeze dataset-local row indices, including full epochs and stress cases."""
    requests = {}
    dataset = bundle.datasets['train']
    # Match PreparedLoader/DataLoader's base-seed draw before sampler iteration.
    source = getattr(bundle.loaders['train'], 'source', bundle.loaders['train'])
    torch.empty((), dtype=torch.int64).random_(generator=source.generator)
    for split, count in [('train', steps), ('validation', validation_steps)]:
        full = [np.asarray(items, dtype=np.int64) for items in bundle.loaders[split].batch_sampler]
        if len(full) < warmup + count:
            raise ValueError(f'{split} needs at least {warmup + count} batches; observed {len(full)}.')
        requests[f'{split}_full'] = full
        requests[f'{split}_warmup'] = full[:warmup]
        requests[split] = full[warmup:warmup + count]
    if dataset.resident is not None:
        counts = np.diff(dataset.resident.indptr)
    else:
        with h5py.File(dataset.path, 'r') as handle:
            counts = np.diff(np.asarray(handle[f'{dataset.group_path}/indptr'], dtype=np.int64))
    full = requests['train_full']
    largest = max(full, key=lambda items: int(counts[dataset.indices[items]].sum()))
    requests['stress'] = [largest, full[-1]]
    return requests


def save_requests(directory: Path, requests: dict, splits: dict) -> None:
    """Persist replayable requests and their mapping to source rows."""
    arrays = {f'split_{key}': value for key, value in splits.items()}
    for key, batches in requests.items():
        arrays[f'{key}_items'] = np.concatenate(batches)
        arrays[f'{key}_offsets'] = np.concatenate(([0], np.cumsum([len(b) for b in batches])))
    np.savez(directory / 'row_requests.npz', **arrays)


def loader_for(dataset: SparseAnnDataDataset, requests: list[np.ndarray], config: dict,
               scalar: bool, validation: bool = False) -> MeasuredPreparation:
    """Keep sampler boundaries, settings and transfer behavior equal within pairs."""
    source = DataLoader(ScalarPixels(dataset) if scalar else dataset,
        batch_sampler=requests, collate_fn=sparse_collate,
        generator=torch.Generator().manual_seed(config['training']['seed']),
        # PreparedLoader owns preparation; no process workers are started.
        num_workers=0)
    return MeasuredPreparation(source, performance_settings(config), config['model'], validation)


def measure(model: AdversarialLatentFusion, loader: MeasuredPreparation, config: dict,
            weights: torch.Tensor | None, device: torch.device,
            optimizer: torch.optim.Optimizer | None = None) -> dict:
    """Run the production loss/metric/transfer path with sampled utilization."""
    with GPUUsage(device) as usage:
        result = run_epoch(model, loader, ZILNLoss.from_config(config).to(device), device,
            config['loss']['biology_weight'], config['loss']['batch_weight'], weights, optimizer,
            0, config['training']['epochs'], config['training']['grl_schedule'],
            config['training']['grl_max_lambda'], config['training']['grl_gamma'],
            config['training']['gradient_clip_norm'], target_columns=config['data']['target_columns'],
            he_weight=config['loss'].get('he_weight', 1.))
    result.update(preparation_seconds=loader.preparation_seconds,
                  loader_wait_fraction=result['loader_wait_seconds'] / result['seconds'],
                  gpu_samples=usage.samples,
                  mean_gpu_utilization=(float(np.mean([s['utilization'] for s in usage.samples]))
                                        if usage.samples else None))
    return result


def check_fields(dataset: SparseAnnDataDataset, requests: list[np.ndarray]) -> list[dict]:
    """Check all tensor fields on representative, largest, and partial requests."""
    checked = []
    for request in requests:
        expected = sparse_collate([dataset[int(i)] for i in request])
        actual = dataset.get_batch(request)
        assert expected.keys() == actual.keys()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)
        checked.append(dict(rows=len(request), peaks=int(actual['peak_counts'].sum())))
    dataset.close()
    return checked


def compare_runs(left: dict, right: dict, left_state: dict, right_state: dict) -> dict:
    """Report end-to-end numerical differences and paired throughput ratios."""
    max_state_error = max(float((left_state[k] - right_state[k]).abs().max()) for k in left_state)
    close = True
    for key in left_state:
        try:
            torch.testing.assert_close(left_state[key], right_state[key], rtol=1e-4, atol=1e-5)
        except AssertionError:
            close = False
    result = dict(state_close=close, max_state_absolute_error=max_state_error)
    for phase in ('train', 'validation'):
        result[phase] = dict(speedup=left[phase]['seconds'] / right[phase]['seconds'],
            optimized_wait_fraction=right[phase]['loader_wait_fraction'],
            loss_absolute_difference=abs(left[phase]['total_loss'] - right[phase]['total_loss']))
    return result


def run_pair(bundle: DataBundle, config: dict, requests: dict, initial: dict,
             directory: Path, device: torch.device, trial: int, full: bool = False,
             deterministic: bool = False) -> dict:
    """Alternate implementation order; reset model, optimizer and RNG for each run."""
    records, states = {}, {}
    weights = bundle.batch_class_weights.to(device) if config['loss']['balance_batch_classes'] else None
    for kind in (('scalar', 'batched') if trial % 2 == 0 else ('batched', 'scalar')):
        seed_everything(config['training']['seed'], deterministic=deterministic)
        torch.use_deterministic_algorithms(deterministic)
        model = AdversarialLatentFusion.from_config(config, len(bundle.metadata.batch_names),
            len(config['data']['target_columns'])).to(device)
        model.load_state_dict(initial)
        training = config['training']
        def optimizer_for():
            return torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
                weight_decay=training['weight_decay'], betas=(training['beta1'], training['beta2']))
        optimizer = optimizer_for()
        def phase(split, indices, opt):
            loader = loader_for(bundle.datasets[split], indices, config, kind == 'scalar', split == 'validation')
            return measure(model, loader, config, weights, device, opt)
        # Warm both training and validation kernels/readers, then restore initial weights/RNG.
        phase('train', requests['train_warmup'], optimizer)
        phase('validation', requests['validation_warmup'], None)
        model.load_state_dict(initial)
        optimizer = optimizer_for()
        seed_everything(training['seed'], deterministic=deterministic)
        torch.use_deterministic_algorithms(deterministic)
        records[kind] = {}
        for split in ('train', 'validation'):
            print(f'{directory.name}: {kind} {split} starting', flush=True)
            result = phase(split, requests[f'{split}_full' if full else split],
                           optimizer if split == 'train' else None)
            records[kind][split] = result
            print(f'{directory.name}: {kind} {split}: {result["seconds"]:.2f}s, '
                  f'{result["rows_per_second"]:.0f} rows/s, wait {result["loader_wait_fraction"]:.1%}', flush=True)
            (directory / f'{kind}.json').write_text(json.dumps(records[kind], indent=2))
        states[kind] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(states[kind], directory / f'{kind}_final_state.pt')
        # Exercise large sparse footprint and actual final partial batch through training.
        records[kind]['stress'] = phase('train', requests['stress'], optimizer)
        (directory / f'{kind}.json').write_text(json.dumps(records[kind], indent=2))
        for dataset in bundle.datasets.values():
            dataset.close()
        del optimizer, model
    comparison = compare_runs(records['scalar'], records['batched'], states['scalar'], states['batched'])
    (directory / 'comparison.json').write_text(json.dumps(comparison, indent=2))
    if deterministic and (comparison['max_state_absolute_error'] != 0
            or any(comparison[phase]['loss_absolute_difference'] != 0
                   for phase in ('train', 'validation'))):
        raise AssertionError('Deterministic scalar/batched replay changed model state or losses.')
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--storage', nargs='+', choices=['resident', 'backed'], default=['resident', 'backed'])
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--validation-steps', type=int, default=24)
    parser.add_argument('--full-epoch', action='store_true', help='Also compare one complete resident train/validation epoch.')
    parser.add_argument('--deterministic', action='store_true',
        help='Additional numerical audit; CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8.')
    args = parser.parse_args()
    if min(args.trials, args.warmup, args.steps, args.validation_steps) < 1:
        parser.error('Trial and batch counts must be positive.')
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    if config['execution']['mode'] != 'pixel':
        parser.error('Use an all-MLP pixel configuration.')
    device = torch.device(args.device)
    if device.type == 'cuda':
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if busy:
            raise RuntimeError(f'GPU has active compute processes ({busy}); none were interrupted.')
    files = [str(p) for p in Path('dann').rglob('*.py')] + [str(args.config)]
    for filename in files:
        source = Path(filename)
        relative = source if not source.is_absolute() else Path('config.yaml')
        destination = args.output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    config=config, torch_version=torch.__version__, numpy_version=np.__version__,
                    h5py_version=h5py.__version__,
                    source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
                    cuda_device=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    summary = {}
    initial, splits, requests = None, None, None
    for storage in args.storage:
        current = copy.deepcopy(config)
        if storage == 'backed':
            current['performance'].update(ram_cache=False, preparation_workers=0,
                                          cache_geometry=False, validation_feature_cache=False)
        else:
            current['performance']['ram_cache'] = True
        bundle = create_data_bundle(current, saved_splits=splits, requested_splits=['train', 'validation'])
        if requests is None:
            requests = freeze_requests(bundle, args.warmup, args.steps, args.validation_steps)
            splits = bundle.split_indices
            save_requests(args.output, requests, splits)
            seed_everything(config['training']['seed'])
            model = AdversarialLatentFusion.from_config(config, len(bundle.metadata.batch_names),
                len(config['data']['target_columns']))
            initial = copy.deepcopy(model.state_dict())
            torch.save(initial, args.output / 'initial_state.pt')
            del model
        checked = check_fields(bundle.datasets['train'], [requests['train'][0], *requests['stress']])
        check_fields(bundle.datasets['validation'], [requests['validation'][0], requests['validation_full'][-1]])
        summary[storage] = dict(settings=performance_settings(current), field_checks=checked, trials=[])
        for trial in range(args.trials):
            directory = args.output / f'{storage}_{trial}'
            directory.mkdir()
            result = run_pair(bundle, current, requests, initial, directory, device, trial,
                              deterministic=args.deterministic)
            summary[storage]['trials'].append(result)
            (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        if args.full_epoch and storage == 'resident':
            directory = args.output / 'resident_full_epoch'
            directory.mkdir()
            summary[storage]['full_epoch'] = run_pair(bundle, current, requests, initial, directory,
                device, 0, full=True, deterministic=args.deterministic)
        for dataset in bundle.datasets.values():
            dataset.close()
        del dataset, bundle
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

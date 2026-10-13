"""Invocation-local inference resources and bounded, deterministic export workers."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, Iterator, Mapping

import h5py
import numpy as np
import psutil
import torch


class InferenceProgress:
    """Report phase progress every five seconds, including during blocking work."""

    def __init__(self) -> None:
        self.started = self.phase_started = perf_counter()
        self.phase = 'startup'
        self.completed = self.total = 0
        self.unit = 'rows'
        self.durations: dict[str, float] = {}
        self.peak_host_bytes = 0
        self.details: dict[str, Any] = {}
        self._lock = Lock()
        self._stop = Event()
        self._thread = Thread(target=self._run, daemon=True, name='dann-progress')

    def __enter__(self):
        print('Inference phase: startup', flush=True)
        self._thread.start()
        return self

    def memory(self) -> int:
        """Resident memory of the parent and all living export/loader workers."""
        process = psutil.Process()
        processes = [process, *process.children(recursive=True)]
        total = 0
        for child in processes:
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        self.peak_host_bytes = max(self.peak_host_bytes, total)
        return total

    def set_phase(self, phase: str, total: int = 0, unit: str = 'rows') -> None:
        with self._lock:
            now = perf_counter()
            self.durations[self.phase] = self.durations.get(self.phase, 0.) + now - self.phase_started
            self.phase, self.phase_started = phase, now
            self.completed, self.total, self.unit = 0, total, unit
        print(f'Inference phase: {phase}', flush=True)

    def update(self, completed: int) -> None:
        self.completed = completed

    def _run(self) -> None:
        while not self._stop.wait(5.):
            with self._lock:
                elapsed = perf_counter() - self.phase_started
                print(f'{self.phase}: {self.completed:,}/{self.total:,} {self.unit}; '
                      f'{self.completed / max(elapsed, .001):,.1f} {self.unit}/s; '
                      f'host including workers {self.memory() / 2**30:.2f} GiB', flush=True)

    def snapshot(self, device: torch.device) -> dict[str, Any]:
        durations = dict(self.durations)
        durations[self.phase] = durations.get(self.phase, 0.) + perf_counter() - self.phase_started
        self.memory()
        return dict(self.details, phase_seconds=durations, total_seconds=perf_counter() - self.started,
                    peak_host_gib=self.peak_host_bytes / 2**30,
                    peak_allocated_vram_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0.,
                    peak_reserved_vram_gib=torch.cuda.max_memory_reserved(device) / 2**30 if device.type == 'cuda' else 0.)

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()


def inference_performance(config: Mapping[str, Any], overrides: Mapping[str, Any], settings: Mapping[str, Any]) -> dict:
    """Resolve runtime controls without inheriting saved training annotations."""
    from dann.config import PERFORMANCE_DEFAULTS, performance_settings
    shared = {**config.get('performance', {}), **overrides.get('performance', {})}
    # Training/validation annotations neither select nor constrain inference.
    for key in PERFORMANCE_DEFAULTS:
        if key.startswith('validation_'):
            shared[key] = PERFORMANCE_DEFAULTS[key]
    shared['feature_cache_rows'] = settings['feature_cache_rows']
    performance = performance_settings({'performance': shared})
    performance.update(validation_tiles=settings['tiles_per_batch'],
                       validation_peak_budget=settings['spectral_peak_budget'],
                       validation_gat_tiles=settings['gat_tiles'],
                       validation_feature_cache=settings['feature_cache'],
                       feature_cache_rows=settings['feature_cache_rows'])
    if settings['feature_cache'] and not (performance['ram_cache'] and performance['cache_geometry']):
        raise ValueError('Spatial feature_cache requires performance.ram_cache and cache_geometry.')
    return performance


def preflight(path: Path, config: Mapping[str, Any], rows: int, performance: Mapping[str, Any]) -> dict:
    """Conservatively reserve CSR construction, geometry, queues, and staging space."""
    from dann.config import execution_settings
    from dann.spatial import matrix_group_path
    execution = execution_settings(config)
    size = config['training'].get('core_size', 32) + 2 * execution['halo']
    core = config['training'].get('core_size', 32)
    with h5py.File(path, 'r') as handle:
        group = handle[matrix_group_path(config['data']['matrix_key'])]
        if group.attrs.get('encoding-type') != 'csr_matrix':
            raise TypeError('Context MSI must be CSR encoded.')
        csr = group['data'].size * 12 + group['indptr'].size * 8
        source_rows = len(group['indptr']) - 1
    # Full-grid keys/dictionaries, construction scratch and mmap resident pages.
    scratch = csr * (3 if config['data']['nonzero_threshold'] > 0 else 2) if performance['ram_cache'] else 0
    metadata = source_rows * 600
    staging = rows * (config['model']['latent_dim'] + 3 * len(config['data']['target_columns']) + 2) * 4
    required = psutil.Process().memory_info().rss + scratch + metadata + staging
    budget = performance['host_budget_gib'] * 2**30
    if required > budget:
        raise MemoryError(f'Inference host budget: estimated construction {required / 2**30:.2f} GiB '
                          f'(CSR scratch {scratch / 2**30:.2f}, metadata {metadata / 2**30:.2f}, '
                          f'staging {staging / 2**30:.2f}) > {budget / 2**30:.2f} GiB.')
    return dict(resident_estimated_bytes=csr, construction_estimated_bytes=required,
                footprint_size=size, core_size=core)



def check_feature_budget(model: torch.nn.Module, rows: int, device: torch.device,
                         settings: Mapping[str, Any]) -> dict[str, int]:
    """Reject an explicit cache that cannot fit before reading resident spectra."""
    encoder = model.encoder
    element_size = encoder.embedding.weight.element_size()
    required = rows * encoder.peak_mlp[-1].out_features * element_size
    hidden = max(m.out_features for m in encoder.peak_mlp if isinstance(m, torch.nn.Linear))
    peaks = settings.get('validation_peak_budget') or model.peak_budget
    # Inference retains weighted embeddings plus the current MLP input/output;
    # reserve a further hidden-width temporary for normalization/activation.
    workspace = peaks * (2 * encoder.embedding.embedding_dim + 3 * hidden) * element_size
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device)
        if allocated + required + workspace > settings['vram_budget_gib'] * 2**30:
            raise MemoryError(f'Inference feature budget: cache {required / 2**30:.2f} GiB + '
                              f'spectral workspace {workspace / 2**30:.2f} GiB + '
                              f'allocated {allocated / 2**30:.2f} GiB exceeds '
                              f"VRAM budget {settings['vram_budget_gib']:.2f} GiB.")
    else:
        check_host_budget(settings, required + workspace)
    return dict(feature_bytes=required, spectral_workspace_bytes=workspace)

def check_host_budget(performance: Mapping[str, Any], extra_bytes: int = 0, *, workers: bool = False) -> None:
    """Include actual Python geometry/metadata and additional pending allocations."""
    process = psutil.Process()
    used = process.memory_info().rss + extra_bytes
    if workers:
        for child in process.children(recursive=True):
            try:
                used += child.memory_info().rss
            except psutil.Error:
                pass
    if used > performance['host_budget_gib'] * 2**30:
        raise MemoryError(f'Inference host budget exceeded: {used / 2**30:.2f} GiB required, '
                          f"{performance['host_budget_gib']:.2f} GiB allowed.")


def release_preparation(loader: Any) -> None:
    """Drop invocation-owned caches and stop readers before spawning export workers."""
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None and hasattr(iterator, '_shutdown_workers'):
        iterator._shutdown_workers()
    dataset = loader.dataset
    dataset.close()
    for name in ('prepared_items', 'context_rows', 'spatial_cache', 'lookup', 'geometry', 'keys', 'tiles'):
        if hasattr(dataset, name):
            setattr(dataset, name, None)
    dataset.resident = None


def sample_rows(pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                start: int, samples: int, seed: int) -> np.ndarray:
    """Preserve the historical per-row stream, draw order and float64 accumulation."""
    from dann.ziln import sampled_density_mean
    return np.concatenate([sampled_density_mean(
        pi[i:i+1], mu[i:i+1], sigma[i:i+1], num_samples=samples,
        rng=np.random.default_rng([seed, start + i])) for i in range(len(pi))])


# Worker state contains only a read-only prediction mmap and scalar metadata.
_DENSITY_MATRIX = None
_DENSITY_OPTIONS = None


def _init_density(path: str, shape: tuple[int, int], width: int, targets: int,
                  samples: int, seed: int) -> None:
    global _DENSITY_MATRIX, _DENSITY_OPTIONS
    torch.set_num_threads(1)
    _DENSITY_MATRIX = np.memmap(path, mode='r', dtype=np.float32, shape=shape)
    _DENSITY_OPTIONS = width, targets, samples, seed


def _density_range(start: int, stop: int) -> tuple[int, np.ndarray]:
    width, targets, samples, seed = _DENSITY_OPTIONS
    values = _DENSITY_MATRIX[start:stop, width:width+3*targets]
    return start, sample_rows(*np.split(values, 3, axis=1), start, samples, seed)


def density_blocks(path: Path, shape: tuple[int, int], width: int, targets: int,
                   block_rows: int, workers: int, samples: int, seed: int,
                   performance: Mapping[str, Any] | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Bound spawn tasks to 2*workers and consume results in original row order."""
    ranges = iter((start, min(start+block_rows, shape[0])) for start in range(0, shape[0], block_rows))
    if not workers:
        matrix = np.memmap(path, mode='r', dtype=np.float32, shape=shape)
        try:
            for start, stop in ranges:
                yield start, sample_rows(*np.split(matrix[start:stop, width:width+3*targets], 3, axis=1),
                                        start, samples, seed)
        finally:
            matrix._mmap.close()
        return
    if performance is not None:
        # Spawn imports torch/scipy, never source spectra; reserve 1 GiB per worker,
        # prediction mmap pages, and the bounded queue of float64 density results.
        check_host_budget(performance, workers * 2**30 + np.prod(shape) * 4
                          + 2 * workers * block_rows * targets * 8)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_density,
                             initargs=(str(path), shape, width, targets, samples, seed)) as pool:
        pending = deque()
        try:
            for _ in range(2 * workers):
                pair = next(ranges, None)
                if pair is not None:
                    pending.append(pool.submit(_density_range, *pair))
            while pending:
                result = pending.popleft().result()
                if performance is not None:
                    check_host_budget(performance, workers=True)
                pair = next(ranges, None)
                if pair is not None:
                    pending.append(pool.submit(_density_range, *pair))
                yield result
        finally:
            for future in pending:
                future.cancel()

"""Bounded, ordered preparation and evaluation-only execution caches.

Resident arrays are shared by preparation threads, never replicated in processes.
All caches are execution state and deliberately absent from model state dictionaries.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from time import perf_counter
from threading import Lock
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import h5py
import numpy as np
import torch


class ResidentSpectra:
    """Read and transform CSR storage once without sorting or dropping stored zeros."""

    def __init__(self, path: Path, data: Mapping[str, Any], budget_bytes: int) -> None:
        from dann.data_loader import _matrix_group_path
        started = perf_counter()
        self.source_identity = self.identity(path, data)
        with h5py.File(path, 'r') as handle:
            group = handle[_matrix_group_path(data['matrix_key'])]
            size = group['data'].size * 12 + group['indptr'].size * 8
            construction_factor = 3 if data['nonzero_threshold'] > 0 else 2
            if size * construction_factor > budget_bytes:
                raise MemoryError('Resident CSR construction exceeds host budget (including scratch).')
            self.indptr = np.asarray(group['indptr'], dtype=np.int64)
            self.indices = np.asarray(group['indices'], dtype=np.int64)
            self.values = np.asarray(group['data'], dtype=np.float32)
            self.num_peaks = int(group.attrs['shape'][1])
        if (self.indptr[0] != 0 or self.indptr[-1] != len(self.values)
                or np.any(np.diff(self.indptr) < 0) or len(self.indices) != len(self.values)):
            raise ValueError('Invalid CSR row pointers.')
        threshold = data['nonzero_threshold']
        if threshold > 0:
            keep = self.values > threshold
            cumulative = np.concatenate(([0], np.cumsum(keep, dtype=np.int64)))
            self.indptr = cumulative[self.indptr]
            self.indices, self.values = self.indices[keep], self.values[keep]
        # Bound temporary validation/transform arrays independently of source size.
        for start in range(0, len(self.values), 4_000_000):
            values = self.values[start:start + 4_000_000]
            peaks = self.indices[start:start + 4_000_000]
            if np.any(peaks < 0) or np.any(peaks >= self.num_peaks):
                raise ValueError('CSR peak index out of range.')
            if data['intensity_transform'] == 'log1p':
                np.log1p(values, out=values)
            elif data['intensity_transform'] == 'sqrt':
                np.sqrt(values, out=values)
            if data.get('intensity_clip_max') is not None:
                np.minimum(values, data['intensity_clip_max'], out=values)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Invalid transformed MSI intensities in resident cache.')
        for array in (self.indptr, self.indices, self.values):
            array.flags.writeable = False
        self.nbytes = sum(a.nbytes for a in (self.indptr, self.indices, self.values))
        self.startup_seconds = perf_counter() - started

    @staticmethod
    def identity(path: Path, data: Mapping[str, Any]) -> tuple:
        """Bind a cache to its ordered source and frozen preprocessing settings."""
        path = Path(path)
        stat = path.stat()
        return (str(path.resolve()), stat.st_size, stat.st_mtime_ns,
                data['matrix_key'], data['intensity_transform'],
                data.get('intensity_clip_max'), data['nonzero_threshold'])

    def gather(self, rows: Sequence[int] | np.ndarray) -> dict[str, torch.Tensor]:
        """Gather a whole ordered CSR row batch, retaining duplicates and empty rows."""
        rows = np.asarray(rows, dtype=np.int64)
        lengths = self.indptr[rows + 1] - self.indptr[rows]
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        membership = np.repeat(np.arange(len(rows)), lengths)
        entries = np.arange(offsets[-1]) + np.repeat(self.indptr[rows] - offsets[:-1], lengths)
        return dict(peak_indices=torch.from_numpy(self.indices[entries]),
                    intensities=torch.from_numpy(self.values[entries]),
                    sample_indices=torch.from_numpy(membership),
                    peak_counts=torch.from_numpy(lengths), row_ids=torch.from_numpy(rows.copy()))


def spectral_chunks(counts: Sequence[int], budget: int) -> tuple[tuple[int, int, int, int], ...]:
    """Compute complete-spectrum boundaries on CPU before a CUDA transfer."""
    chunks, first, offset, mass = [], 0, 0, 0
    for row, count in enumerate(counts):
        count = int(count)
        if mass + count > budget and row > first:
            chunks.append((first, row, offset, offset + mass))
            offset += mass
            first, mass = row, 0
        mass += count
    chunks.append((first, len(counts), offset, offset + mass))
    return tuple(chunks)


def tree_map(value: Any, function: Callable[[torch.Tensor], Any]) -> Any:
    """Apply a tensor operation to nested execution metadata."""
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {k: tree_map(v, function) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree_map(v, function) for v in value)
    return value


def tensor_bytes(value: Any) -> int:
    """Count tensor payload bytes, excluding Python containers."""
    sizes = []
    tree_map(value, lambda t: sizes.append(t.numel() * t.element_size()))
    return sum(sizes)


def pack_graphs(graphs: Sequence[tuple[torch.Tensor, ...]], area: int) -> tuple[torch.Tensor, ...]:
    """Join independent cached tile graphs without changing any edge or attribute."""
    positions, edges, features, offset = [], [], [], 0
    for i, (p, e, f) in enumerate(graphs):
        positions.append(p + i * area)
        edges.append(e + offset)
        features.append(f)
        offset += len(p)
    return torch.cat(positions), torch.cat(edges, 1), torch.cat(features)


class PreparedLoader:
    """Prepare a bounded number of batches on shared-memory threads, in sampler order."""

    def __init__(self, loader: Any, settings: Mapping[str, Any],
                 model_config: Mapping[str, Any], validation: bool = False) -> None:
        self.source = loader
        self.dataset, self.sampler = loader.dataset, loader.sampler
        self.settings, self.validation = settings, validation
        self.peak_budget = (settings['validation_peak_budget'] if validation else None) or model_config.get('spectral_peak_budget', 65536)
        self.tile_budget = (settings['validation_gat_tiles'] if validation else None) or model_config.get('gatv2_tile_budget', 8)
        self._iteration_created = False
        self._metrics_lock = Lock()
        self.pooled_only = False
        self.inference = False
        self.wait_seconds = 0.
        self.max_batch_bytes = 0

    def __getattr__(self, name: str) -> Any:
        """Preserve loader inspection and worker-cleanup APIs for existing callers."""
        return getattr(self.source, name)

    def __len__(self) -> int:
        return len(self.source)

    def prepare(self, request: Any) -> dict[str, Any]:
        """Gather one complete batch and prepare execution metadata on CPU."""
        if self.source.batch_sampler is None:
            batch = self.dataset[request]
        else:
            items = (self.dataset.__getitems__(request) if hasattr(self.dataset, '__getitems__')
                     else [self.dataset[i] for i in request])
            if self.pooled_only:
                from dann.tiles import tile_collate
                batch = tile_collate(items, include_spectra=False)
            else:
                batch = self.source.collate_fn(items)
        if 'peak_counts' in batch:
            batch['spectral_chunks'] = spectral_chunks(batch['peak_counts'].tolist(), self.peak_budget)
        if '_tile_graphs' in batch:
            graphs = batch.pop('_tile_graphs')
            area = batch['occupancy'].shape[-1] * batch['occupancy'].shape[-2]
            batch['graph_tile_budget'] = self.tile_budget
            batch['prepared_graphs'] = {
                radius: [pack_graphs([g[radius] for g in graphs[start:start+self.tile_budget]], area)
                         for start in range(0, len(graphs), self.tile_budget)]
                for radius in graphs[0]}
        size = tensor_bytes(batch)
        with self._metrics_lock:
            self.max_batch_bytes = max(self.max_batch_bytes, size)
        tiles = getattr(self.dataset, 'tiles', None)
        resident = getattr(self.dataset, 'resident', None) or getattr(tiles, 'resident', None)
        resident_bytes = resident.nbytes if resident is not None else 0
        geometry_bytes = getattr(self.dataset, 'geometry_bytes', 0) or getattr(tiles, 'geometry_bytes', 0)
        # CPU prepared, pinned staging, current and transfer batches can coexist.
        required = resident_bytes + geometry_bytes + size * (3 * self.settings['prepared_batches'] + 6)
        if required > self.settings['host_budget_gib'] * 2**30:
            raise MemoryError('Resident cache plus bounded preparation exceeds host budget.')
        if self.inference:
            from dann.spatial_execution import check_host_budget
            check_host_budget(self.settings, size * (3 * self.settings['prepared_batches'] + 6))
        return batch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self.wait_seconds = 0.
        # Match DataLoader's base-seed draw before iterating its sampler.
        if not self._iteration_created or not self.source.persistent_workers:
            torch.empty((), dtype=torch.int64).random_(generator=self.source.generator)
        self._iteration_created = True
        requests = iter(self.source.batch_sampler if self.source.batch_sampler is not None else self.sampler)
        workers = self.settings['preparation_workers']
        if not workers:
            for request in requests:
                started = perf_counter()
                batch = self.prepare(request)
                self.wait_seconds += perf_counter() - started
                yield batch
            return
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='dann-prepare') as pool:
            pending = deque()
            for _ in range(self.settings['prepared_batches']):
                try:
                    pending.append(pool.submit(self.prepare, next(requests)))
                except StopIteration:
                    break
            while pending:
                started = perf_counter()
                batch = pending.popleft().result()
                self.wait_seconds += perf_counter() - started
                try:
                    pending.append(pool.submit(self.prepare, next(requests)))
                except StopIteration:
                    pass
                yield batch


def device_batches(loader: Any, device: torch.device) -> Iterator[tuple[dict, dict]]:
    """Overlap one pinned batch transfer with compute; retain sources until DMA completes."""
    settings = getattr(loader, 'settings', {})
    enabled = device.type == 'cuda' and settings.get('transfer_prefetch', False)
    if not enabled:
        for cpu in loader:
            yield cpu, tree_map(cpu, lambda t: t.to(device, non_blocking=True))
        return
    stream = torch.cuda.Stream(device=device)
    iterator = iter(loader)

    def stage(cpu):
        pinned = tree_map(cpu, lambda t: t if t.is_pinned() else t.pin_memory())
        with torch.cuda.stream(stream):
            batch = tree_map(pinned, lambda t: t.to(device, non_blocking=True))
            event = torch.cuda.Event()
            event.record(stream)
        return pinned, batch, event

    pending = None
    try:
        try:
            pending = stage(next(iterator))
        except StopIteration:
            return
        while pending is not None:
            cpu, batch, event = pending
            current = torch.cuda.current_stream(device)
            current.wait_event(event)
            tree_map(batch, lambda t: t.record_stream(current))
            try:
                pending = stage(next(iterator))
            except StopIteration:
                pending = None
            yield cpu, batch
            # References to pinned inputs survive the consuming step and DMA event.
            event.synchronize()
    finally:
        stream.synchronize()
        if hasattr(iterator, 'close'):
            iterator.close()


@contextmanager
def validation_execution(model: torch.nn.Module, settings: Mapping[str, Any]) -> Iterator[None]:
    """Temporarily override only execution budgets, restoring them even on failure."""
    from dann.model_components.layers_gatv2 import GATv2Network
    if not hasattr(model, 'peak_budget'):
        yield
        return
    old_peak = model.peak_budget
    encoder = getattr(model, 'encoder', None)
    old_inner = getattr(encoder, 'inference_peak_chunk_size', None)
    networks = [(m, m.tile_budget) for m in model.modules() if isinstance(m, GATv2Network)]
    try:
        model.peak_budget = settings.get('validation_peak_budget') or old_peak
        # Deep Sets also chunks inside forward() when gradients are disabled.
        # Move that cap with an explicit validation budget so it can take effect.
        if old_inner is not None and settings.get('validation_peak_budget') is not None:
            encoder.inference_peak_chunk_size = model.peak_budget
        for m, budget in networks:
            m.tile_budget = settings.get('validation_gat_tiles') or budget
        yield
    finally:
        model.peak_budget = old_peak
        if old_inner is not None:
            encoder.inference_peak_chunk_size = old_inner
        for m, budget in networks:
            m.tile_budget = budget


def build_validation_features(model: torch.nn.Module, loader: PreparedLoader,
                              device: torch.device, progress=None, *,
                              context_datasets=None) -> tuple[np.ndarray, torch.Tensor] | None:
    """Encode every required context row once; return a pass-local GPU/CPU cache."""
    if model.training or torch.is_grad_enabled():
        raise ValueError('Feature caches are evaluation-only and require disabled gradients.')
    dataset = loader.dataset
    if not hasattr(dataset, 'context_rows'):
        raise ValueError('Validation feature caching requires prepared spatial geometry.')
    if not dataset.context_rows:
        return None
    required_rows = np.zeros(len(dataset.keys), dtype=bool)
    for context in (context_datasets if context_datasets is not None else [dataset]):
        if context.geometry[-1] != dataset.geometry[-1]:
            raise ValueError('Feature cache context sources differ.')
        if context.resident.source_identity != dataset.resident.source_identity:
            raise ValueError('Feature cache context preprocessing differs.')
        for geometry in context.context_rows:
            required_rows[geometry['context_rows']] = True
    rows = np.flatnonzero(required_rows)
    del required_rows
    if progress is not None:
        progress.set_phase('feature encoding', len(rows))
    width = model.encoder.peak_mlp[-1].out_features
    element_size = model.encoder.embedding.weight.element_size()
    required = len(rows) * width * element_size
    settings = loader.settings
    if device.type == 'cuda' and torch.cuda.memory_allocated(device) + required > settings['vram_budget_gib'] * 2**30:
        raise MemoryError(f'Pooled feature cache requires {required / 2**30:.2f} GiB in addition to '
                          f'current allocations; VRAM budget {settings["vram_budget_gib"]:.2f} GiB.')
    if progress is not None:
        from dann.spatial_execution import check_host_budget, check_feature_budget
        check_feature_budget(model, len(rows), device, settings)
        check_host_budget(settings, len(dataset.keys) * 16 + (required if device.type == 'cpu' else 0))
        progress.details['feature_cache_rows'] = len(rows)
        progress.details['feature_cache_bytes'] = required
    values = torch.empty((len(rows), width), device=device, dtype=model.encoder.embedding.weight.dtype)
    lookup = np.full(len(dataset.keys), -1, dtype=np.int64)
    lookup[rows] = np.arange(len(rows))
    for start in range(0, len(rows), settings['feature_cache_rows']):
        selected = rows[start:start + settings['feature_cache_rows']]
        cpu = dataset.resident.gather(selected)
        chunks = spectral_chunks(cpu['peak_counts'].tolist(), model.peak_budget)
        batch = tree_map(cpu, lambda t: t.to(device))
        pooled = model.encoder.microbatched(*[batch[k] for k in
            ('peak_indices', 'intensities', 'sample_indices', 'peak_counts')],
            peak_budget=model.peak_budget, checkpointing=False, chunks=chunks)
        values[start:start + len(selected)] = pooled
        if progress is not None:
            progress.update(start + len(selected))
    return lookup, values


@contextmanager
def pooled_validation_loader(loader: Any, settings: Mapping[str, Any]) -> Iterator[None]:
    """Skip repeated spectra gathering and transfer after building pooled features."""
    old = getattr(loader, 'pooled_only', False)
    try:
        if settings.get('validation_feature_cache'):
            loader.pooled_only = True
        yield
    finally:
        if hasattr(loader, 'pooled_only'):
            loader.pooled_only = old


def evaluation_groups(model: torch.nn.Module, loader: Any,
                      device: torch.device) -> Iterator[tuple[dict, dict, dict]]:
    """Execute physical tile batches, then restore historical loss-reduction groups."""
    settings = getattr(loader, 'settings', {})
    cache_settings = settings if getattr(model, 'spatial', False) else {}
    with validation_execution(model, settings), pooled_validation_loader(loader, cache_settings):
        cache = build_validation_features(model, loader, device) if settings.get('validation_feature_cache') and model.spatial else None
        pending_id, pending_batch, pending_outputs = None, [], []

        def finish():
            return ({k: torch.cat([b[k] for b in pending_batch]) for k in pending_batch[0]},
                    {k: torch.cat([o[k] for o in pending_outputs]) for k in pending_outputs[0]})

        for cpu, batch in device_batches(loader, device):
            if cache is not None:
                lookup, values = cache
                ids = lookup[cpu['context_row_ids'].numpy()]
                if np.any(ids < 0):
                    raise ValueError('Context row missing from this validation pass.')
                batch['pooled_features'] = values[torch.as_tensor(ids, device=device)]
            outputs = model(batch, grl_strength=0.)
            if 'reduction_groups' not in cpu:
                yield cpu, batch, outputs
                continue
            groups = cpu['reduction_groups'].numpy()
            boundaries = np.r_[0, np.flatnonzero(np.diff(groups)) + 1, len(groups)]
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                group = int(groups[start])
                if pending_id is not None and group != pending_id:
                    b, o = finish()
                    yield {}, b, o
                    pending_batch, pending_outputs = [], []
                pending_id = group
                pending_batch.append({k: batch[k][start:stop] for k in
                    ('targets', 'target_valid_mask', 'batches', 'row_ids', 'he_targets') if k in batch})
                pending_outputs.append({k: v[start:stop] for k, v in outputs.items()})
        if pending_batch:
            b, o = finish()
            yield {}, b, o

"""Raw H&E targets, exact MSE statistics, and bounded feature-vector exports."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
from uuid import uuid4

import h5py
import numpy as np
import torch


def he_settings(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the enabled architecture, preserving legacy opt-out behavior."""
    from dann.config import component_settings
    return component_settings(config['model'])['heads'].get('he')


class HETargetReader:
    """Read selected obsm rows from disk or a shared immutable float32 cache.

    Backed reads own their handles; resident reads need no HDF5 calls. Values are
    validated only when selected, so held-out nonfinite rows cannot affect fitting.
    """
    def __init__(self, path: Path, key: str, rows: int, width: int) -> None:
        self.path, self.key, self.width = Path(path), key, width
        self.rows = rows
        self.values: np.ndarray | None = None
        self.startup_seconds = 0.
        with h5py.File(self.path, 'r') as handle:
            name = f'obsm/{key}'
            if name not in handle:
                raise ValueError(f'Missing H&E targets: {name} in {path}.')
            array = handle[name]
            if not isinstance(array, h5py.Dataset) or array.shape != (rows, width):
                raise ValueError(f'H&E targets {name} must have shape ({rows}, {width}).')
            if array.dtype.kind not in 'fiu':
                raise TypeError(f'H&E targets {name} must have a real numeric dtype.')
            self.chunk_rows = array.chunks[0] if array.chunks else None

    @property
    def nbytes(self) -> int:
        """Bytes retained by this shared reader, excluding temporary batches."""
        return self.values.nbytes if self.values is not None else 0

    def cache_in_memory(self, budget_bytes: int) -> None:
        """Read once into preallocated RAM, rejecting oversized allocations first."""
        from dann.execution import check_startup_budget
        if self.values is not None:
            return
        required = self.rows * self.width * np.dtype(np.float32).itemsize
        check_startup_budget(budget_bytes, required + min(required, 64 * 2**20), 'H&E cache')
        started = perf_counter()
        values = np.empty((self.rows, self.width), dtype=np.float32)
        # HDF5 converts directly into the final array; no second matrix or IPC copy.
        if self.rows:
            with h5py.File(self.path, 'r') as handle:
                handle[f'obsm/{self.key}'].read_direct(values)
        values.flags.writeable = False
        self.values = values
        self.startup_seconds = perf_counter() - started

    def _rows(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= self.rows):
            raise ValueError('H&E row positions must be one-dimensional and within the source.')
        return rows

    @staticmethod
    def _validate(values: np.ndarray, rows: np.ndarray) -> None:
        invalid = rows[~np.isfinite(values).all(axis=1)]
        if len(invalid):
            raise ValueError(f'Invalid H&E supervised source row positions: {np.unique(invalid).tolist()} (nonfinite float32 values).')

    def read(self, rows: np.ndarray) -> np.ndarray:
        """Read arbitrary row order and explicitly report nonfinite supervised rows."""
        rows = self._rows(rows)
        if not len(rows):
            return np.empty((0, self.width), dtype=np.float32)
        if self.values is not None:
            values = self.values[rows]
            self._validate(values, rows)
            return values
        unique, inverse = np.unique(rows, return_inverse=True)
        with h5py.File(self.path, 'r') as handle:
            with np.errstate(over='ignore', invalid='ignore'):
                values = np.asarray(handle[f'obsm/{self.key}'][unique], dtype=np.float32)
        self._validate(values, unique)
        return values[inverse]

    def mean(self, rows: np.ndarray, chunk_rows: int = 1024, *, workers: int = 1,
             budget_bytes: int | None = None) -> np.ndarray:
        """Reduce selected rows in storage order with deterministic float64 sums.

        Backed input visits each relevant row block once using one handle. Cached
        input uses independent NumPy reductions, without changing sampler order.
        Duplicate row requests retain their multiplicity; empty input returns zero.
        """
        from dann.execution import check_startup_budget
        if chunk_rows <= 0 or workers <= 0:
            raise ValueError('H&E reduction chunk_rows and workers must be positive.')
        rows = self._rows(rows)
        total = np.zeros(self.width, dtype=np.float64)
        if not len(rows):
            return total
        block_rows = min(max(chunk_rows, self.chunk_rows or 1),
                         max(1, 64 * 2**20 // (self.width * 4)))
        workers = min(workers, len(rows), len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1)
        if budget_bytes is not None:
            # One storage slab plus a selected-row buffer per task, and sort/index scratch.
            scratch = 3 * rows.nbytes + (workers if self.values is not None else 1) * (
                block_rows * self.width * 9 + self.width * 8)
            check_startup_budget(budget_bytes, scratch, 'H&E mean reduction')
        ordered = np.sort(rows)
        blocks = ordered // block_rows
        boundaries = np.r_[0, np.flatnonzero(np.diff(blocks)) + 1, len(ordered)]
        spans = list(zip(boundaries[:-1], boundaries[1:]))

        def reduce_block(source, span):
            left, right = span
            start = int(ordered[left] // block_rows * block_rows)
            stop = min(start + block_rows, self.rows)
            # Select before validating: held-out NaNs never enter fitted statistics.
            slab = np.asarray(source[start:stop], dtype=np.float32)
            partial = np.zeros(self.width, dtype=np.float64)
            # Also bound requests containing many duplicates within one storage block.
            for first in range(left, right, block_rows):
                selected = ordered[first:min(first + block_rows, right)]
                values = slab[selected - start]
                self._validate(values, selected)
                partial += values.sum(axis=0, dtype=np.float64)
            return partial

        if self.values is None:
            with h5py.File(self.path, 'r') as handle, np.errstate(over='ignore', invalid='ignore'):
                source = handle[f'obsm/{self.key}']
                for span in spans:
                    total += reduce_block(source, span)
        elif workers == 1:
            for span in spans:
                total += reduce_block(self.values, span)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='dann-he-mean') as pool:
                for partial in pool.map(lambda span: reduce_block(self.values, span), spans):
                    total += partial
        return total / len(rows)


def he_mse(prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Ordinary MSE over supervised core rows and every stored feature dimension."""
    if prediction.shape != targets.shape or prediction.ndim != 2 or not targets.numel():
        raise ValueError('H&E prediction and targets must be nonempty matching matrices.')
    if not torch.isfinite(targets).all():
        raise ValueError('Nonfinite H&E supervised targets.')
    return (prediction - targets).square().mean()


class HEWriter:
    """Stream ordered float32 feature lists with source identities, then validate coverage."""
    def __init__(self, output: Path, source: Path, width: int, expected_rows: np.ndarray) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from dann.tiles import _axis_names
        from dann.preprocessing import artifact_preprocessing, parquet_preprocessing
        self.output, self.width = Path(output), width
        self.expected = np.asarray(expected_rows, dtype=np.int64)
        if len(np.unique(self.expected)) != len(self.expected):
            raise ValueError('Duplicate expected H&E export rows.')
        with h5py.File(source, 'r') as handle:
            self.names = _axis_names(handle['obs'])
        if np.any(self.expected < 0) or np.any(self.expected >= len(self.names)):
            raise ValueError('H&E export rows outside source.')
        self.seen = np.zeros(len(self.names), dtype=bool)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.output.with_name(f'.{self.output.name}.{uuid4().hex}.tmp')
        self.schema = pa.schema([('row_position', pa.int64()), ('obs_name', pa.string()),
                                 ('he_prediction', pa.list_(pa.float32(), width))],
                                metadata={b'target_convention': b'raw_stored_values',
                                          b'feature_order': b'stored_obsm_column_order',
                                          **parquet_preprocessing(artifact_preprocessing(source))})
        self.writer = pq.ParquetWriter(self.temporary, self.schema, compression='zstd')

    def write(self, rows: np.ndarray, prediction: np.ndarray) -> None:
        """Append a batch, rejecting duplicate identities or invalid feature vectors."""
        import pyarrow as pa
        rows = np.asarray(rows, dtype=np.int64)
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != (len(rows), self.width) or not np.isfinite(prediction).all():
            raise ValueError('Invalid H&E export feature vectors.')
        if (np.any(rows < 0) or np.any(rows >= len(self.seen)) or len(np.unique(rows)) != len(rows)
                or np.any(self.seen[rows])):
            raise ValueError('Duplicate or invalid H&E export rows.')
        self.writer.write_table(pa.Table.from_arrays([
            pa.array(rows), pa.array(self.names[rows].astype(str)),
            pa.FixedSizeListArray.from_arrays(pa.array(prediction.ravel()), self.width)], schema=self.schema))
        self.seen[rows] = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        import pyarrow.parquet as pq
        try:
            self.writer.close()
            if exc_type is None:
                if not np.array_equal(np.flatnonzero(self.seen), np.sort(self.expected)):
                    raise ValueError('Incomplete H&E export row coverage.')
                parquet = pq.ParquetFile(self.temporary)
                if parquet.metadata.num_rows != len(self.expected) or not parquet.schema_arrow.equals(self.schema):
                    raise ValueError('Invalid H&E Parquet schema or row count.')
                self.seen[:] = False
                for batch in parquet.iter_batches(batch_size=1024):
                    rows = batch.column('row_position').to_numpy()
                    names = batch.column('obs_name').to_numpy(zero_copy_only=False)
                    values = batch.column('he_prediction').flatten().to_numpy()
                    if (np.any(rows < 0) or np.any(rows >= len(self.seen))
                            or len(np.unique(rows)) != len(rows) or np.any(self.seen[rows])
                            or not np.array_equal(names, self.names[rows].astype(str))
                            or not np.isfinite(values).all()):
                        raise ValueError('Invalid H&E exported identities or values.')
                    self.seen[rows] = True
                if not np.array_equal(np.flatnonzero(self.seen), np.sort(self.expected)):
                    raise ValueError('Incomplete H&E exported row coverage.')
                os.replace(self.temporary, self.output)
        finally:
            self.temporary.unlink(missing_ok=True)


def analysis_he_statistics(predictions: Mapping, batch: Mapping) -> dict[str, np.ndarray]:
    """Keep per-row sums/counts, avoiding retention of large feature predictions."""
    if 'cca_msi' in predictions:
        # Small paired projections remain row-aligned in the analysis cache.
        return {key: predictions[key].detach().cpu().numpy() for key in ('cca_msi', 'cca_he')}
    if 'he_prediction' not in predictions:
        return {}
    error = (predictions['he_prediction'].double() - batch['he_targets'].double()).square()
    return dict(he_squared_error=error.sum(1).cpu().numpy(),
                he_element_count=np.full(len(error), error.shape[1], dtype=np.int64))


def report_he(extracted: Mapping, directory: Path, split: str, weight: float) -> None:
    """Write exact aggregate analysis MSE even when feature export is disabled."""
    import json
    if 'cca_msi' in extracted:
        from dann.deep_cca import SlideCovariance, CCASummary
        statistics = SlideCovariance(extracted['cca_msi'].shape[1])
        for start in range(0, len(extracted['cca_msi']), 8192):
            take = slice(start, start + 8192)
            statistics.update(*(torch.as_tensor(extracted[k][take]) for k in ('cca_msi', 'cca_he', 'batches')))
        summary = CCASummary(weight)
        summary.add(statistics.finish())
        metrics = summary.finish(strict=False)
        metrics = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in metrics.items()}
        output = Path(directory) / 'he'
        output.mkdir(parents=True, exist_ok=True)
        (output / f'{split}_cca_metrics.json').write_text(json.dumps(metrics, indent=2))
        return
    if 'he_squared_error' not in extracted:
        return
    count = int(extracted['he_element_count'].sum())
    total = float(extracted['he_squared_error'].sum())
    mse = total / count
    directory = Path(directory) / 'he'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{split}_metrics.json').write_text(json.dumps(dict(
        squared_error_sum=total, element_count=count, mse=mse, weighted_loss=weight*mse,
        target_convention='raw_stored_values'), indent=2))


def extract_with_he(model, loader, device, config: Mapping, directory: Path,
                    split: str, extractor=None) -> dict:
    """Run bounded extraction with optional H&E streaming and unconditional metrics."""
    from contextlib import nullcontext
    from dann.analyze import extract_latent_predictions
    writer = nullcontext(None)
    if getattr(model, 'he_cca', None) is not None and config['analysis'].get('export_he_predictions', False):
        raise ValueError('Raw H&E reconstruction export is unavailable with deep_cca.')
    if config['analysis'].get('export_he_predictions', False) and getattr(model, 'he_output_dim', 0):
        writer = HEWriter(Path(directory) / 'he' / f'{split}_predictions.parquet',
                          Path(config['data']['path']), model.he_output_dim, loader.dataset.indices)
    with writer as stream:
        values = (extractor or extract_latent_predictions)(model, loader, device,
                                                               **({"he_writer": stream} if stream is not None else {}))
    report_he(values, directory, split, config['loss'].get('he_weight', 1.0))
    return values

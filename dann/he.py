"""Raw H&E targets, exact MSE statistics, and bounded feature-vector exports."""
from __future__ import annotations

import os
from pathlib import Path
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
    """Read selected obsm rows with invocation-local handles, safe across workers/threads.

    No HDF5 handle or target matrix is retained in the dataset or geometry cache.
    Stored feature order and values are preserved (float32 model precision).
    """
    def __init__(self, path: Path, key: str, rows: int, width: int) -> None:
        self.path, self.key, self.width = Path(path), key, width
        with h5py.File(self.path, 'r') as handle:
            name = f'obsm/{key}'
            if name not in handle:
                raise ValueError(f'Missing H&E targets: {name} in {path}.')
            array = handle[name]
            if not isinstance(array, h5py.Dataset) or array.shape != (rows, width):
                raise ValueError(f'H&E targets {name} must have shape ({rows}, {width}).')
            if array.dtype.kind not in 'fiu':
                raise TypeError(f'H&E targets {name} must have a real numeric dtype.')

    def read(self, rows: np.ndarray) -> np.ndarray:
        """Read arbitrary row order and explicitly report nonfinite supervised rows."""
        rows = np.asarray(rows, dtype=np.int64)
        unique, inverse = np.unique(rows, return_inverse=True)
        with h5py.File(self.path, 'r') as handle:
            with np.errstate(over='ignore', invalid='ignore'):
                values = np.asarray(handle[f'obsm/{self.key}'][unique], dtype=np.float32)
        invalid = unique[~np.isfinite(values).all(axis=1)]
        if len(invalid):
            raise ValueError(f'Invalid H&E supervised source row positions: {invalid.tolist()} (nonfinite float32 values).')
        return values[inverse]


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
                                          b'feature_order': b'stored_obsm_column_order'})
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
    if 'he_prediction' not in predictions:
        return {}
    error = (predictions['he_prediction'].double() - batch['he_targets'].double()).square()
    return dict(he_squared_error=error.sum(1).cpu().numpy(),
                he_element_count=np.full(len(error), error.shape[1], dtype=np.int64))


def report_he(extracted: Mapping, directory: Path, split: str, weight: float) -> None:
    """Write exact aggregate analysis MSE even when feature export is disabled."""
    import json
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
    if config['analysis'].get('export_he_predictions', False) and getattr(model, 'he_output_dim', 0):
        writer = HEWriter(Path(directory) / 'he' / f'{split}_predictions.parquet',
                          Path(config['data']['path']), model.he_output_dim, loader.dataset.indices)
    with writer as stream:
        values = (extractor or extract_latent_predictions)(model, loader, device,
                                                               **({"he_writer": stream} if stream is not None else {}))
    report_he(values, directory, split, config['loss'].get('he_weight', 1.0))
    return values

"""Epoch-local proportional slide queues and immutable lazy spatial batches."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from dann.tiles import SpatialTileDataset, tile_collate

ALGORITHM_VERSION = 1
SUPERVISION_FIELDS = ('core_positions', 'row_ids', 'support', 'latent_support',
                      'targets', 'raw_targets', 'target_valid_mask', 'batches')


@dataclass(frozen=True)
class SpatialBatchRequest:
    """Ordered supervision and first-appearance tile IDs, sent from the main process."""
    row_ids: tuple[int, ...]
    tile_ids: tuple[int, ...]


class ProportionalSlideTileSampler(Sampler[SpatialBatchRequest]):
    """One complete pass, seeded independently of model and worker random streams.

    Tile sizes count selected rows, never valid/positive labels. A split tile
    retains its original identity and complete context on both sides of a boundary.
    Only tile ordering and the current request are materialized for an epoch.
    """
    def __init__(self, tiles: SpatialTileDataset, seed: int, row_budget: int) -> None:
        if isinstance(row_budget, bool) or not isinstance(row_budget, int) or row_budget <= 0:
            raise ValueError('row_budget must be a positive integer.')
        self.tiles = tiles
        self.seed = seed
        self.row_budget = row_budget
        self.epoch = 0
        self.queues: dict[str, list[int]] = {}
        self.lengths = np.array([len(rows) for _, rows in tiles.tiles], dtype=np.int64)
        for i, (key, _) in enumerate(tiles.tiles):
            self.queues.setdefault(key[0], []).append(i)
        self.num_rows = int(self.lengths.sum())

    def __len__(self) -> int:
        return (self.num_rows + self.row_budget - 1) // self.row_budget

    def set_epoch(self, epoch: int) -> None:
        """Select a zero-based epoch before creating the DataLoader iterator."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError('epoch must be a nonnegative integer.')
        self.epoch = epoch

    def __iter__(self) -> Iterator[SpatialBatchRequest]:
        rng = np.random.default_rng(self.seed + 50000 + self.epoch)
        ordered = []
        for slide in sorted(self.queues):
            queue = rng.permutation(self.queues[slide])
            sizes = self.lengths[queue]
            positions = (sizes.cumsum() - sizes / 2) / sizes.sum()
            ordered.extend(zip(positions.tolist(), rng.random(len(queue)).tolist(), queue.tolist()))
        ordered.sort()
        rows, tile_ids = [], []
        for _, _, tile_id in ordered:
            selected = self.tiles.tiles[tile_id][1]
            offset = 0
            while offset < len(selected):
                take = min(self.row_budget - len(rows), len(selected) - offset)
                tile_ids.append(tile_id)
                rows.extend(int(r) for r, _ in selected[offset:offset + take])
                offset += take
                if len(rows) == self.row_budget:
                    yield SpatialBatchRequest(tuple(rows), tuple(tile_ids))
                    rows, tile_ids = [], []
        if rows:
            yield SpatialBatchRequest(tuple(rows), tuple(tile_ids))


class SpatialBatchDataset(Dataset):
    """Read original footprints lazily; subset only supervision after collation."""
    def __init__(self, tiles: SpatialTileDataset) -> None:
        self.tiles = tiles
        self.indices = tiles.indices
        self.metadata = tiles.metadata

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, request: SpatialBatchRequest) -> dict[str, torch.Tensor]:
        batch = tile_collate([self.tiles[i] for i in request.tile_ids])
        lookup = {int(row): i for i, row in enumerate(batch['row_ids'].tolist())}
        selection = torch.tensor([lookup[row] for row in request.row_ids], dtype=torch.long)
        for key in SUPERVISION_FIELDS:
            if key in batch:
                batch[key] = batch[key][selection]
        return batch

    def close(self) -> None:
        """Release the process-local sparse reader."""
        self.tiles.close()


class SamplingEpochAudit:
    """Hash actual consumed rows/boundaries and summarize observed batch composition."""
    def __init__(self, epoch: int, total_rows: int) -> None:
        self.epoch, self.total_rows = epoch, total_rows
        self.rows_hash = hashlib.sha256()
        self.boundaries_hash = hashlib.sha256(np.asarray([0], dtype='<i8').tobytes())
        self.processed_rows = 0
        self.compositions: list[dict] = []

    def update(self, batch: dict[str, torch.Tensor]) -> dict:
        """Record one successfully executed optimizer batch (context is not exposure)."""
        rows = batch['row_ids'].numpy().astype('<i8', copy=False)
        self.rows_hash.update(rows.tobytes())
        self.processed_rows += len(rows)
        self.boundaries_hash.update(np.asarray([self.processed_rows], dtype='<i8').tobytes())
        _, counts = np.unique(batch['batches'].numpy(), return_counts=True)
        fractions = counts / counts.sum()
        result = dict(epoch=self.epoch, step=len(self.compositions), supervised_rows=len(rows),
                      unique_tiles=int(batch['occupancy'].shape[0]), distinct_slides=len(counts),
                      slide_entropy=float(-(fractions * np.log(fractions)).sum()),
                      largest_slide_fraction=float(fractions.max()), processed_rows=self.processed_rows,
                      epoch_position=self.processed_rows / max(self.total_rows, 1))
        self.compositions.append(result)
        return result

    def summary(self) -> dict:
        """Return independently reproducible row/boundary digests and epoch statistics."""
        return dict(epoch=self.epoch, processed_rows=self.processed_rows, batches=len(self.compositions),
                    complete=self.processed_rows == self.total_rows,
                    row_order_sha256=self.rows_hash.hexdigest(),
                    boundaries_sha256=self.boundaries_hash.hexdigest(),
                    composition={key: dict(min=min(values), mean=float(np.mean(values)), max=max(values))
                        for key in ('supervised_rows', 'unique_tiles', 'distinct_slides',
                                    'slide_entropy', 'largest_slide_fraction')
                        if (values := [r[key] for r in self.compositions])})

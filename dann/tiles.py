"""Sparse spatial tiles with explicit context, core supervision, and grid identities."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from dann.data_loader import AnnDataMetadata
    from dann.model import AdversarialLatentFusion
import hashlib
import h5py
import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def open_backed_anndata(path: Path) -> ad.AnnData:
    """Open a backed AnnData file without warning about duplicate observation names.

    Grid coordinates identify rows. Duplicate ``obs_names`` are left unchanged so
    saved row fingerprints stay stable.

    Args:
        path (Path): ``.h5ad`` path.

    Returns:
        ad.AnnData: Backed AnnData object. The caller closes ``adata.file``.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Observation names are not unique\.",
            category=UserWarning,
        )
        return ad.read_h5ad(path, backed="r")


def read_grid(path: Path, data: Mapping[str, Any]) -> tuple[pd.MultiIndex, np.ndarray]:
    """Read unique integer (slide, x, y) keys; observation names are not identities."""
    adata = open_backed_anndata(path)
    try:
        columns = [data['batch_column'], data['x_column'], data['y_column']]
        obs = adata.obs[columns]
        if obs.isna().any().any():
            raise ValueError("Grid columns cannot contain missing values.")
        coordinates = obs[columns[1:]].to_numpy(dtype=float)
        if not np.isfinite(coordinates).all() or not np.equal(coordinates, np.floor(coordinates)).all():
            raise ValueError("Spatial coordinates must be finite integers.")
        keys = pd.MultiIndex.from_arrays([obs[columns[0]].astype(str),
                                         coordinates[:, 0].astype(np.int64),
                                         coordinates[:, 1].astype(np.int64)], names=columns)
        if not keys.is_unique:
            raise ValueError("Duplicate (batch, x, y) grid positions.")
        return keys, np.asarray(adata.var_names, dtype=str)
    finally:
        adata.file.close()


def row_identity(path: Path, data: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint ordered rows and features, guarding saved split positions."""
    adata = open_backed_anndata(path)
    try:
        columns = [data['batch_column']]
        columns += [data[k] for k in ('x_column', 'y_column') if data.get(k) in adata.obs]
        frame = adata.obs[columns].astype(str)
        digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
        return {'row_digest': digest, 'rows': adata.n_obs,
                'feature_order': list(map(str, adata.var_names))}
    finally:
        adata.file.close()


class SpatialTileDataset(Dataset):
    """Gather full MSI context while exposing labels only for selected core rows."""
    def __init__(self, config: Mapping[str, Any], selected_rows: np.ndarray,
                 metadata: AnnDataMetadata | None = None, input_path: Path | None = None,
                 geometry: tuple | None = None, resident=None, spatial_cache=None) -> None:
        from dann.config import execution_settings
        from dann.spatial import SparseInferenceDataset
        self.data = config['data']
        self.resident = resident
        self.performance = config.get('performance', {}) if metadata is not None or resident is not None else {}
        self.spatial_cache = spatial_cache if spatial_cache is not None else {}
        self.geometry_bytes = 0
        from dann.config import component_settings
        components = component_settings(config['model'])
        self.graph_radii = {g['gatv2']['neighbor_radius'] for g in
            (components['aggregation'], *components['heads'].values()) if g['type'] == 'gatv2'}
        self.reduction_tiles = config['training'].get('tiles_per_batch', 8)
        if metadata is not None:
            self.path = Path(self.data['path'])
            if input_path is not None and Path(input_path).resolve() != self.path.resolve():
                raise ValueError("Supervised spectra and context must come from data.path.")
        else:
            if input_path is None:
                raise ValueError("Inference requires an explicit input path.")
            self.path = Path(input_path)
        if resident is not None and resident.source_identity != resident.identity(self.path, self.data):
            raise ValueError('Resident spectra source or preprocessing differs from tile input.')
        stat = self.path.stat()
        source_identity = (str(self.path.resolve()), stat.st_size, stat.st_mtime_ns,
                           self.data['matrix_key'], self.data['batch_column'],
                           self.data['x_column'], self.data['y_column'])
        if geometry is not None and (len(geometry) != 5 or geometry[4] != source_identity):
            raise ValueError("Cached geometry source identity differs from the input file.")
        if geometry is None:
            self.keys, features = read_grid(self.path, self.data)
            from dann.data_loader import _matrix_group_path
            with h5py.File(self.path, "r") as handle:
                matrix = handle[_matrix_group_path(self.data["matrix_key"])]
                if matrix.attrs.get("encoding-type") != "csr_matrix":
                    raise TypeError("Context MSI must be CSR encoded.")
                if tuple(matrix.attrs["shape"]) != (len(self.keys), len(features)):
                    raise ValueError("Context MSI matrix and grid/feature metadata differ.")
            if len(features) != config["model"]["num_peaks"]:
                raise ValueError("Context feature width differs from configured num_peaks.")
        else:
            self.keys, features = geometry[0], geometry[1]
        self.indices = np.asarray(selected_rows, dtype=np.int64)
        self.metadata = metadata
        self.core_size = int(config['training'].get('core_size', 32))
        self.halo = execution_settings(config)['halo']
        self.latent_radius = execution_settings(config)['latent_radius']
        self.biology_radius = execution_settings(config)['biology_radius']
        self.size = self.core_size + 2 * self.halo
        if len(features) != config['model']['num_peaks']:
            raise ValueError("Context feature width differs from configured num_peaks.")
        if metadata is not None and metadata.num_observations != len(self.keys):
            raise ValueError("Supervised metadata rows differ from labeled grid.")
        if self.indices.ndim != 1 or np.any(self.indices < 0) or np.any(self.indices >= len(self.keys)):
            raise ValueError("Selected rows are outside the source grid.")
        selected_context = self.indices
        self.lookup = ({key: row for row, key in enumerate(self.keys)} if geometry is None else geometry[2])
        self.geometry = (self.keys, features, self.lookup, None, source_identity)
        groups = {}
        for output_row, context_row in zip(self.indices, selected_context):
            slide, x, y = self.keys[context_row]
            tile = (slide, x // self.core_size, y // self.core_size)
            groups.setdefault(tile, []).append((int(output_row), int(context_row)))
        self.tiles = list(groups.items())
        self.spectra = SparseInferenceDataset(self.path, len(self.keys), self.data['matrix_key'],
            self.data['intensity_transform'], self.data.get('intensity_clip_max'),
            self.data['nonzero_threshold'])
        if self.performance.get('cache_geometry'):
            self.context_rows = [self._prepare_geometry(i) for i in range(len(self.tiles))]
            if self.resident is not None:
                self.prepared_items = [self._prepared_item(i) for i in range(len(self.tiles))]
            from dann.execution import tensor_bytes
            self.geometry_bytes = sum(
                sum(v.nbytes for v in g.values() if isinstance(v, np.ndarray))
                + tensor_bytes(g.get('graphs', {})) for g in self.spatial_cache.values())
            if self.geometry_bytes + (resident.nbytes if resident is not None else 0) > self.performance.get('host_budget_gib', 128) * 2**30:
                raise MemoryError('Spatial geometry plus spectra exceed host budget.')

    def _prepare_geometry(self, item: int) -> dict[str, Any]:
        """Cache immutable full footprints and graphs shared by all label splits."""
        tile_key, _ = self.tiles[item]
        key = (self.geometry[-1], self.core_size, self.halo, tuple(sorted(self.graph_radii)), tile_key)
        if key in self.spatial_cache:
            return self.spatial_cache[key]
        slide, tile_x, tile_y = tile_key
        x0, y0 = tile_x * self.core_size - self.halo, tile_y * self.core_size - self.halo
        rows, positions = [], []
        mask = np.zeros((self.size, self.size), dtype=np.float32)
        for y in range(self.size):
            for x in range(self.size):
                row = self.lookup.get((slide, x0+x, y0+y))
                if row is not None:
                    rows.append(row)
                    positions.append(y*self.size+x)
                    mask[y, x] = 1
        from dann.model_components.layers_gatv2 import grid_graph
        result = dict(context_rows=np.asarray(rows, dtype=np.int64),
                      spatial_positions=np.asarray(positions, dtype=np.int64), occupancy=mask,
                      graphs={radius: grid_graph(torch.from_numpy(mask)[None, None], radius)
                              for radius in self.graph_radii})
        for value in result.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        self.spatial_cache[key] = result
        return result

    def _prepared_item(self, item: int) -> dict[str, Any]:
        geometry = self.context_rows[item] if hasattr(self, 'context_rows') else self._prepare_geometry(item)
        (slide, tile_x, tile_y), selected = self.tiles[item]
        x0, y0 = tile_x * self.core_size - self.halo, tile_y * self.core_size - self.halo
        rows, core, support, latent_support = [], [], [], []
        mask = geometry['occupancy']
        for output_row, context_row in selected:
            _, x, y = self.keys[context_row]
            lx, ly = x-x0, y-y0
            rows.append(output_row)
            core.append(ly*self.size+lx)
            for radius, result in ((self.biology_radius, support), (self.latent_radius, latent_support)):
                result.append(mask[ly-radius:ly+radius+1, lx-radius:lx+radius+1].sum() / (2*radius+1)**2)
        result = dict(spatial_positions=geometry['spatial_positions'], occupancy=mask,
                      context_row_ids=geometry['context_rows'], core_positions=core,
                      row_ids=np.asarray(rows), support=np.asarray(support, dtype=np.float32),
                      latent_support=np.asarray(latent_support, dtype=np.float32),
                      reduction_groups=np.full(len(rows), item // self.reduction_tiles, dtype=np.int64))
        if self.performance.get('cache_geometry'):
            result['_graphs'] = geometry['graphs']
        if self.resident is not None:
            result['_resident'] = self.resident
        else:
            result['samples'] = [self.spectra[row] for row in geometry['context_rows']]
        if self.metadata is not None:
            m = self.metadata
            result.update(targets=m.targets[rows], raw_targets=m.raw_targets[rows],
                          target_valid_mask=m.target_valid_mask[rows], batches=m.batch_codes[rows])
        return result

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, item: int) -> dict[str, Any]:
        if hasattr(self, 'prepared_items'):
            return self.prepared_items[item]
        if self.resident is not None or self.performance.get('cache_geometry'):
            return self._prepared_item(item)
        (slide, tile_x, tile_y), selected = self.tiles[item]
        x0 = tile_x * self.core_size - self.halo
        y0 = tile_y * self.core_size - self.halo
        samples, positions = [], []
        mask = np.zeros((self.size, self.size), dtype=np.float32)
        for y in range(self.size):
            for x in range(self.size):
                row = self.lookup.get((slide, x0+x, y0+y))
                if row is not None:
                    samples.append(self.spectra[row])
                    positions.append(y*self.size+x)
                    mask[y, x] = 1
        rows, core, support, latent_support = [], [], [], []
        for output_row, context_row in selected:
            _, x, y = self.keys[context_row]
            lx, ly = x-x0, y-y0
            rows.append(output_row)
            core.append(ly*self.size+lx)
            for radius, result in ((self.biology_radius, support), (self.latent_radius, latent_support)):
                result.append(mask[ly-radius:ly+radius+1, lx-radius:lx+radius+1].sum() / (2*radius+1)**2)
        result = dict(samples=samples, spatial_positions=positions, core_positions=core,
                      occupancy=mask, row_ids=np.asarray(rows), support=np.asarray(support, dtype=np.float32),
                      latent_support=np.asarray(latent_support, dtype=np.float32))
        if self.metadata is not None:
            m = self.metadata
            result.update(targets=m.targets[rows], raw_targets=m.raw_targets[rows],
                          target_valid_mask=m.target_valid_mask[rows], batches=m.batch_codes[rows])
        return result

    def close(self) -> None:
        """Release the process-local sparse reader."""
        self.spectra.close()


def tile_collate(tiles: Sequence[Mapping[str, Any]], include_spectra: bool = True) -> dict[str, torch.Tensor]:
    """Pack sparse spectra and map indices without densifying the spectral axis."""
    from dann.spatial import sparse_inference_collate
    if not include_spectra:
        batch = {}
    elif '_resident' in tiles[0]:
        batch = tiles[0]['_resident'].gather(np.concatenate([t['context_row_ids'] for t in tiles]))
    else:
        batch = sparse_inference_collate([sample for tile in tiles for sample in tile['samples']])
    if '_graphs' in tiles[0]:
        batch['_tile_graphs'] = [t['_graphs'] for t in tiles]
    if 'context_row_ids' in tiles[0]:
        batch['context_row_ids'] = torch.from_numpy(np.concatenate([t['context_row_ids'] for t in tiles]))
    area = tiles[0]['occupancy'].size
    for key in ('spatial_positions', 'core_positions'):
        batch[key] = torch.from_numpy(np.concatenate([
            np.asarray(tile[key], dtype=np.int64) + i*area for i, tile in enumerate(tiles)]))
    batch['occupancy'] = torch.from_numpy(np.stack([t['occupancy'] for t in tiles])[:, None])
    for key in ('row_ids', 'support', 'latent_support', 'targets', 'raw_targets', 'target_valid_mask', 'batches', 'reduction_groups'):
        if key in tiles[0]:
            batch[key] = torch.from_numpy(np.concatenate([t[key] for t in tiles]))
    return batch


def ordered_spatial_predictions(
    model: AdversarialLatentFusion, loader, selected_rows: int, device: torch.device,
    directory: Path, output_batch_size: int = 4096,
) -> Iterator[dict[str, torch.Tensor]]:
    """Stage tile-order results on disk and stream bounded, original-row-order blocks."""
    import tempfile
    from dann.train import move_batch_to_device
    width = model.latent_dim
    targets = model.num_targets
    with tempfile.TemporaryDirectory(prefix='.dann-stage-', dir=directory) as temporary:
        matrix = np.memmap(Path(temporary) / 'predictions.bin', mode='w+', dtype=np.float32,
                           shape=(selected_rows, width + 3*targets + 2))
        seen = np.memmap(Path(temporary) / 'seen.bin', mode='w+', dtype=np.uint8, shape=(selected_rows,))
        seen[:] = 0
        try:
            with torch.inference_mode():
                for cpu in loader:
                    rows = cpu['row_ids'].numpy()
                    if np.any(seen[rows]) or len(np.unique(rows)) != len(rows):
                        raise ValueError('Duplicate output core rows.')
                    outputs = model(move_batch_to_device(cpu, device), grl_strength=0.)
                    arrays = [outputs[k].cpu().numpy() for k in ('latent', 'pi', 'mu', 'sigma')]
                    matrix[rows] = np.concatenate([*arrays, cpu['support'].numpy()[:, None],
                                                    cpu['latent_support'].numpy()[:, None]], axis=1)
                    seen[rows] = 1
            for start in range(0, selected_rows, output_batch_size):
                stop = min(start+output_batch_size, selected_rows)
                if not np.all(seen[start:stop]):
                    raise ValueError('Incomplete spatial prediction export.')
                values = np.array(matrix[start:stop])
                boundaries = [0, width, width+targets, width+2*targets, width+3*targets]
                result = {key: torch.from_numpy(values[:, a:b]) for key, a, b in
                          zip(('latent', 'pi', 'mu', 'sigma'), boundaries[:-1], boundaries[1:])}
                result.update(row_ids=torch.arange(start, stop), support=torch.from_numpy(values[:, -2]),
                              latent_support=torch.from_numpy(values[:, -1]))
                yield result
        finally:
            matrix._mmap.close()
            seen._mmap.close()

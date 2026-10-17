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
    from anndata.io import read_elem
    with h5py.File(path, 'r') as handle:
        columns = [data['batch_column'], data['x_column'], data['y_column']]
        obs = pd.DataFrame({column: read_elem(handle['obs'][column]) for column in columns})
        features = _axis_names(handle['var'])
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
    return keys, features


def _axis_names(group: h5py.Group) -> np.ndarray:
    """Decode only an axis index, including categorical and nullable encodings."""
    from anndata.io import read_elem
    key = group.attrs.get('_index', '_index')
    if isinstance(key, bytes):
        key = key.decode()
    values = read_elem(group[key])
    if pd.isna(values).any():
        raise ValueError('Axis names contain missing values.')
    return np.asarray(values, dtype=str)


def row_identity(path: Path, data: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint the original ordered names, grid columns, and feature index."""
    from anndata.io import read_elem
    with h5py.File(path, 'r') as handle:
        obs = handle['obs']
        columns = [data['batch_column']]
        columns += [data[k] for k in ('x_column', 'y_column') if data.get(k) in obs]
        frame = pd.DataFrame({column: read_elem(obs[column]) for column in columns},
                             index=pd.Index(_axis_names(obs)))
        # AnnData preserves the stored index name except for its reserved _index key.
        key = obs.attrs.get('_index', '_index')
        if isinstance(key, bytes):
            key = key.decode()
        frame.index.name = None if key == '_index' else key
        digest = hashlib.sha256(pd.util.hash_pandas_object(frame.astype(str), index=True).values.tobytes()).hexdigest()
        return {'row_digest': digest, 'rows': len(frame),
                'feature_order': list(_axis_names(handle['var']))}


class SpatialTileDataset(Dataset):
    """Gather full MSI context while exposing labels only for selected core rows."""
    def __init__(self, config: Mapping[str, Any], selected_rows: np.ndarray,
                 metadata: AnnDataMetadata | None = None, input_path: Path | None = None,
                 geometry: tuple | None = None, resident=None, spatial_cache=None, progress=None) -> None:
        from dann.config import execution_settings
        from dann.spatial import SparseInferenceDataset
        self.data = config['data']
        self.resident = resident
        self.performance = config.get('performance', {}) if metadata is not None or resident is not None or spatial_cache is not None else {}
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
        from dann.he import HETargetReader, he_settings
        he = he_settings(config)
        self.he_reader = (HETargetReader(self.path, self.data.get('he_features_key', 'hes_features'),
                                        len(self.keys), he['output_dim']) if metadata is not None and he else None)
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
            if progress is not None:
                progress.set_phase(getattr(progress, 'geometry_phase', 'startup'),
                                   len(self.tiles) * (2 if self.resident is not None else 1), 'tile preparation steps')
            self.context_rows = []
            for i in range(len(self.tiles)):
                self.context_rows.append(self._prepare_geometry(i))
                if progress is not None:
                    progress.update(i + 1)
                if progress is not None and i % 256 == 0:
                    from dann.spatial_execution import check_host_budget
                    check_host_budget(self.performance)
            if self.resident is not None:
                self.prepared_items = []
                for i in range(len(self.tiles)):
                    self.prepared_items.append(self._prepared_item(i))
                    if progress is not None:
                        progress.update(len(self.tiles) + i + 1)
                    if progress is not None and i % 256 == 0:
                        from dann.spatial_execution import check_host_budget
                        check_host_budget(self.performance)
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
            if self.he_reader is not None:
                result["_he_reader"] = self.he_reader
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
            if self.he_reader is not None:
                result["_he_reader"] = self.he_reader
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
    if '_he_reader' in tiles[0]:
        batch['he_targets'] = torch.from_numpy(tiles[0]['_he_reader'].read(batch['row_ids'].numpy()))
    return batch


def ordered_spatial_predictions(
    model: AdversarialLatentFusion, loader, selected_rows: int, device: torch.device,
    directory: Path, output_batch_size: int = 4096, *, execution=None,
    density_samples: int = 1000, density_seed: int = 20260719, progress=None, predict_he: bool = False,
) -> Iterator[dict[str, torch.Tensor]]:
    """Stage tile results, release preparation, then stream ordered density blocks."""
    import tempfile
    from dann.execution import (build_validation_features, device_batches,
                                pooled_validation_loader, validation_execution)
    from dann.spatial_execution import density_blocks, release_preparation
    settings = getattr(loader, 'settings', {}) if execution else {}
    width, targets = model.latent_dim, model.num_targets
    he_width = getattr(model, "he_output_dim", 0) if predict_he else 0
    with tempfile.TemporaryDirectory(prefix='.dann-stage-', dir=directory) as temporary:
        path = Path(temporary) / 'predictions.bin'
        matrix = np.memmap(path, mode='w+', dtype=np.float32,
                           shape=(selected_rows, width + 3*targets + he_width + 2))
        seen = np.memmap(Path(temporary) / 'seen.bin', mode='w+', dtype=np.uint8, shape=(selected_rows,))
        seen[:] = 0
        cache = None
        density = None
        try:
            with torch.inference_mode(), validation_execution(model, settings), pooled_validation_loader(loader, settings):
                if settings.get('validation_feature_cache'):
                    cache = build_validation_features(model, loader, device, progress=progress)
                if progress is not None:
                    progress.set_phase('spatial prediction', selected_rows)
                completed = 0
                for cpu, batch in device_batches(loader, device):
                    rows = cpu['row_ids'].numpy()
                    if np.any(rows < 0) or np.any(rows >= selected_rows):
                        raise ValueError('Output core rows outside selected range.')
                    if np.any(seen[rows]) or len(np.unique(rows)) != len(rows):
                        raise ValueError('Duplicate output core rows.')
                    if cache is not None:
                        lookup, features = cache
                        ids = lookup[cpu['context_row_ids'].numpy()]
                        if np.any(ids < 0):
                            raise ValueError('Context row missing from this inference pass.')
                        batch['pooled_features'] = features[torch.as_tensor(ids, device=device)]
                    outputs = model(batch, grl_strength=0.,
                                    **({'predict_he': predict_he} if getattr(model, 'he_output_dim', 0) else {}))
                    # A single synchronized device-to-host copy for all model outputs.
                    packed = torch.cat([outputs[k] for k in ('latent', 'pi', 'mu', 'sigma', *(['he_prediction'] if he_width else []))], dim=1).cpu().numpy()
                    matrix[rows, :-2] = packed
                    matrix[rows, -2] = cpu['support'].numpy()
                    matrix[rows, -1] = cpu['latent_support'].numpy()
                    seen[rows] = 1
                    completed += len(rows)
                    if progress is not None:
                        progress.update(completed)
                    if settings and device.type == 'cuda' and torch.cuda.max_memory_allocated(device) > settings['vram_budget_gib'] * 2**30:
                        raise MemoryError('Spatial inference exceeded allocated VRAM budget.')
                if completed:
                    del cpu, batch, outputs, packed
                if cache is not None:
                    del lookup, features
                cache = None
            for start in range(0, selected_rows, output_batch_size):
                if not np.all(seen[start:start+output_batch_size]):
                    raise ValueError('Incomplete spatial prediction export.')
            matrix.flush()
            if progress is not None:
                progress.details.update(preparation_wait_seconds=getattr(loader, 'wait_seconds', 0.),
                                        max_prepared_batch_bytes=getattr(loader, 'max_batch_bytes', 0))
            if execution:
                release_preparation(loader)
                import gc
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            if progress is not None:
                progress.set_phase('density sampling/writing', selected_rows)
            if execution:
                density = density_blocks(path, matrix.shape, width, targets, output_batch_size,
                                         execution['density_workers'], density_samples, density_seed, settings)
            for start in range(0, selected_rows, output_batch_size):
                stop = min(start+output_batch_size, selected_rows)
                values = np.array(matrix[start:stop])
                boundaries = [0, width, width+targets, width+2*targets, width+3*targets]
                result = {key: torch.from_numpy(values[:, a:b]) for key, a, b in
                          zip(('latent', 'pi', 'mu', 'sigma'), boundaries[:-1], boundaries[1:])}
                if he_width:
                    result["he_prediction"] = torch.from_numpy(values[:, width+3*targets:-2])
                result.update(row_ids=torch.arange(start, stop), support=torch.from_numpy(values[:, -2]),
                              latent_support=torch.from_numpy(values[:, -1]))
                if density is not None:
                    density_start, means = next(density)
                    if density_start != start:
                        raise RuntimeError('Density workers changed row order.')
                    result['density_mean'] = torch.from_numpy(means)
                yield result
        finally:
            if density is not None:
                density.close()
            matrix._mmap.close()
            seen._mmap.close()

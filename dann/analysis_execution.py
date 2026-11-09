"""Bounded checkpoint analysis with validated numerical caches and shared rendering.

Persistent caches contain only arrays and JSON. Spectra, models, CUDA tensors and
HDF5 handles never enter a rendering worker or survive an analysis invocation.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import contextmanager
import copy
import gc
import hashlib
import importlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping
import uuid

import numpy as np
import torch

from dann.spatial_execution import InferenceProgress, check_host_budget


def digest_file(path: Path) -> str:
    """Hash files in bounded chunks, including array payloads before cache reuse."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_snapshot(path: Path) -> tuple[dict, str]:
    """Load and hash the same open checkpoint, rejecting concurrent writes."""
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        checkpoint = torch.load(handle, map_location='cpu', weights_only=False)
        handle.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
        if any(getattr(before, key) != getattr(after, key) for key in ('st_size', 'st_mtime_ns', 'st_ctime_ns')):
            raise ValueError('Checkpoint changed while being read; retry with a stable checkpoint.')
    return checkpoint, digest.hexdigest()


def fingerprint(value: Any) -> str:
    """Hash JSON metadata or a numerical array without dtype/shape ambiguity."""
    if isinstance(value, np.ndarray):
        digest = hashlib.sha256(str((value.dtype.str, value.shape)).encode())
        digest.update(memoryview(np.ascontiguousarray(value)).cast('B'))
        return digest.hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                     separators=(',', ':')).encode()).hexdigest()


class StageCache:
    """Publish a checksummed manifest last; incomplete generations are never read."""

    def __init__(self, root: Path, policy: str, progress: InferenceProgress):
        self.root, self.policy, self.progress = root, policy, progress

    def get(self, stage: str, identity: Any):
        key = fingerprint(identity)
        directory = self.root / stage / key
        if self.policy != 'reuse':
            return None
        try:
            manifest = json.loads((directory / 'manifest.json').read_text())
            if manifest['key'] != key or manifest['version'] != 1:
                raise ValueError('Cache identity/version mismatch.')
            if manifest.get('metadata_sha256') != fingerprint(manifest.get('metadata', {})):
                raise ValueError('Cache metadata checksum mismatch.')
            arrays = {}
            for name, info in manifest['arrays'].items():
                if Path(info['file']).name != info['file']:
                    raise ValueError('Invalid cache filename.')
                path = directory / info['file']
                if digest_file(path) != info['sha256']:
                    raise ValueError('Cache checksum mismatch.')
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                if list(array.shape) != info['shape'] or array.dtype.str != info['dtype']:
                    raise ValueError('Cache array shape/dtype mismatch.')
                arrays[name] = array
            self.progress.details.setdefault('cache_hits', []).append(stage)
            self.progress.details.setdefault('cache_entries', {})[stage] = key
            return arrays, manifest.get('metadata', {})
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError, OSError) as error:
            print(f'Ignoring invalid {stage} cache: {error}', flush=True)
            return None

    def put(self, stage: str, identity: Any, arrays: Mapping[str, np.ndarray], metadata=None):
        if self.policy == 'off':
            return
        key = fingerprint(identity)
        directory = self.root / stage / key
        directory.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        manifest = dict(version=1, key=key, arrays={}, metadata=metadata or {})
        manifest['metadata_sha256'] = fingerprint(manifest['metadata'])
        owned = []
        try:
            for name, array in arrays.items():
                path = directory / f'{generation}-{name}.npy'
                owned.append(path)
                np.save(path, array, allow_pickle=False)
                manifest['arrays'][name] = dict(file=path.name, sha256=digest_file(path),
                    dtype=array.dtype.str, shape=list(array.shape))
            temporary = directory / f'{generation}.json'
            owned.append(temporary)
            temporary.write_text(json.dumps(manifest, indent=2))
            os.replace(temporary, directory / 'manifest.json')
            self.progress.details.setdefault('cache_entries', {})[stage] = key
        except BaseException:
            for path in owned:
                path.unlink(missing_ok=True)
            raise


def sampled_density_mean(pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray, *,
                         num_samples: int, rng: np.random.Generator, workers: int = 1,
                         progress: InferenceProgress | None = None) -> np.ndarray:
    """Parallelize arithmetic while preserving every legacy draw and reduction."""
    from dann import ziln
    from scipy.special import expit
    if not workers:
        return ziln.sampled_density_mean(pi, mu, sigma, num_samples=num_samples, rng=rng)
    if isinstance(num_samples, bool) or not isinstance(num_samples, (int, np.integer)) or num_samples <= 0:
        raise ValueError('num_samples must be a positive integer.')
    zero, location, scale = np.broadcast_arrays(ziln._validate_probability(pi, 'pi'),
        ziln._as_float_array(mu, 'mu'), ziln._validate_sigma(sigma))
    shape = location.shape
    zero, location, scale = zero.ravel(), location.ravel(), scale.ravel()
    result = np.zeros(location.size, dtype=np.float64)

    def calculate(start, stop, buffers):
        total = np.zeros(stop-start, dtype=np.float64)
        for draws, uniforms in buffers:
            with np.errstate(over='ignore'):
                draws *= scale[start:stop, None]
                draws += location[start:stop, None]
            expit(draws, out=draws)
            draws *= uniforms >= zero[start:stop, None]
            total += draws.sum(axis=1, dtype=np.float64)
        return start, stop, total

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='analysis-density') as pool:
        pending = deque()

        def consume():
            start, stop, values = pending.popleft().result()
            result[start:stop] = values
            if progress is not None:
                progress.update(stop)

        for start in range(0, result.size, 1024):
            stop = min(start+1024, result.size)
            buffers = []
            for draw_start in range(0, num_samples, 128):
                block_shape = (stop-start, min(128, num_samples-draw_start))
                # Only this producer touches rng, in the legacy call order.
                buffers.append((rng.standard_normal(block_shape), rng.random(block_shape)))
            pending.append(pool.submit(calculate, start, stop, buffers))
            if len(pending) >= 2*workers:
                consume()
        while pending:
            consume()
    return (result / num_samples).reshape(shape)


def extract_prepared(model: torch.nn.Module, loader: Any, device: torch.device,
                     cache: tuple[np.ndarray, torch.Tensor] | None,
                     progress: InferenceProgress, *, he_writer=None, collect_he: bool = True) -> dict[str, np.ndarray]:
    """Pack model outputs once per physical batch; keep labels/identities on CPU."""
    from dann.execution import device_batches, pooled_validation_loader
    from dann.targets import observed_targets
    if model.training or torch.is_grad_enabled():
        raise ValueError('Analysis extraction requires eval and inference mode.')
    expected = np.asarray([row for _, selected in loader.dataset.tiles for row, _ in selected], dtype=np.int64)
    arrays = {}
    offset = 0
    progress.set_phase('spatial prediction', len(expected))
    with pooled_validation_loader(loader, {'validation_feature_cache': cache is not None}):
        for cpu, batch in device_batches(loader, device):
            if cache is not None:
                lookup, features = cache
                ids = lookup[cpu['context_row_ids'].numpy()]
                if np.any(ids < 0):
                    raise ValueError('Missing analysis context in feature cache.')
                batch['pooled_features'] = features[torch.as_tensor(ids, device=device)]
            predictions = model(batch, grl_strength=0., **({'predict_he': False} if not collect_he and he_writer is None else {}))
            names = ('latent', 'pi', 'mu', 'sigma')
            widths = [predictions[name].shape[1] for name in names]
            packed = torch.cat([predictions[name] for name in names], dim=1).cpu().numpy()
            values = dict(zip(names, np.split(packed, np.cumsum(widths)[:-1], axis=1)))
            from dann.he import analysis_he_statistics
            if collect_he:
                values.update(analysis_he_statistics(predictions, batch))
            if he_writer is not None:
                he_writer.write(cpu['row_ids'].numpy(), predictions['he_prediction'].cpu().numpy())
            values.update({name: cpu[name].numpy() for name in
                ('targets', 'raw_targets', 'target_valid_mask', 'batches', 'row_ids', 'support', 'latent_support')
                if name in cpu})
            count = len(packed)
            if not np.array_equal(values['row_ids'], expected[offset:offset+count]):
                raise ValueError('Analysis output order differs from saved tile order.')
            for name, value in values.items():
                if name not in arrays:
                    arrays[name] = np.empty((len(expected), *value.shape[1:]), dtype=value.dtype)
                arrays[name][offset:offset+count] = value
            offset += count
            progress.update(offset)
            if device.type == 'cuda' and torch.cuda.max_memory_allocated(device) > loader.settings['vram_budget_gib'] * 2**30:
                raise MemoryError('Analysis exceeded allocated VRAM budget.')
    if offset != len(expected) or len(np.unique(expected)) != len(expected):
        raise ValueError('Missing or duplicate analysis output rows.')
    if not arrays:
        raise ValueError('Analysis split contains no rows.')
    arrays['targets'] = observed_targets(arrays)
    return arrays


_WORKER_LIMIT = None
_WORKER_RC = None


def _init_renderer(rc=None, colormaps=None):
    global _WORKER_LIMIT, _WORKER_RC
    from threadpoolctl import threadpool_limits
    _WORKER_LIMIT = threadpool_limits(limits=1)
    torch.set_num_threads(1)
    _WORKER_RC = rc
    # Calibration helpers can run before analyze.py imports seaborn in a worker.
    import seaborn  # noqa: F401 -- registers the same named palettes as the parent
    import matplotlib
    from matplotlib.colors import ListedColormap
    if isinstance(colormaps, str):
        colormaps = json.loads(Path(colormaps).read_text())
    for name, (colors, bad, under, over) in (colormaps or {}).items():
        if name not in matplotlib.colormaps:
            cmap = ListedColormap(colors, name=name)
            cmap.set_bad(bad)
            cmap.set_under(under)
            cmap.set_over(over)
            matplotlib.colormaps.register(cmap)


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {'__analysis_array__'}:
            return np.load(value['__analysis_array__'], mmap_mode='r', allow_pickle=False)
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_decode(item) for item in value)
    return value


def _render_task(module, function, args, kwargs):
    """Import only trusted, explicitly queued plotting functions in a spawn worker."""
    function = getattr(importlib.import_module(module), function)
    if _WORKER_RC is not None:
        import matplotlib
        matplotlib.rcParams.update(_WORKER_RC)
    function(*_decode(args), **_decode(kwargs))


@contextmanager
def worker_environment():
    names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: '1' for name in names})
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class RenderQueue:
    """Queue small descriptors and share immutable arrays through local mmap files."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tasks = []
        self.arrays = {}
        self.tables = []
        self.row_counts = {}
        self.canvas_bytes = 0

    def _encode(self, value):
        if isinstance(value, np.ndarray):
            known = self.arrays.get(id(value))
            if known is not None and known[0] is value:
                return {'__analysis_array__': str(known[1])}
            path = self.directory / f'{fingerprint(value)}.npy'
            if not path.exists():
                np.save(path, value, allow_pickle=False)
            self.arrays[id(value)] = (value, path)
            return {'__analysis_array__': str(path)}
        if isinstance(value, dict):
            return {key: self._encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(self._encode(item) for item in value)
        return value

    def submit(self, function, *args, **kwargs):
        # Agg canvases grow quadratically with DPI. Reserve four canvas-sized
        # buffers for draw/encode temporaries, including unusually large figures.
        configs = [value for value in (*args, *kwargs.values())
                   if isinstance(value, Mapping) and 'figure_dpi' in value]
        if function.__name__.startswith('plot_'):
            dpi = max([int(value['figure_dpi']) for value in configs] or [300])
            targets = 4
            if function.__name__ in ('plot_latent_umap_densities', 'plot_ziln_density_scatter'):
                targets = len(args[2])
            elif function.__name__ in ('plot_pit', 'plot_hurdle_reliability'):
                targets = len(args[1])
            area = max(36*12, 24*6*((targets+1)//2), 5*targets*9)
            if function.__name__ == 'plot_loss_curves':
                import pandas as pd
                from dann.objective_curves import objective_layout, weighted_history, active_weighted_history
                from dann.curve_layout import figure_size
                config = kwargs.get('config')
                history = active_weighted_history(weighted_history(pd.read_csv(args[0]), config), config)
                rows = objective_layout(history, config, (kwargs.get('selection') or {}).get('selection_metric'))
                sizes = [figure_size(kind, len(rows[index])) for index, kind, flag in
                         ((0, 'loss', 'overall'), (1, 'ihc', 'ihc')) if kwargs.get(flag, True)]
                area = max((w*h for w, h in sizes), default=0)
                dpi = int(args[2])
            self.canvas_bytes = max(self.canvas_bytes, area*dpi*dpi*4)
        self.tasks.append((function.__module__, function.__name__, self._encode(args), self._encode(kwargs)))

    def table(self, kind, extracted, coordinates, targets, batches, split, epsilon, path,
              *, block_rows=16384):
        """Write bounded row parts; scatter parts follow target-major legacy order."""
        needed = {'row_ids', 'batches', 'latent'} if kind == 'latent' else {
            'row_ids', 'batches', 'targets', 'raw_targets', 'target_valid_mask',
            'pi', 'mu', 'sigma', 'density_mean', 'target_transform'}
        selected = {key: value for key, value in extracted.items() if key in needed}
        parts = []
        columns = range(len(targets)) if kind == 'scatter' else [None]
        for column in columns:
            for start in range(0, max(1, len(extracted['row_ids'])), block_rows):
                part = self.directory / f'{path.name}-{len(parts):05d}.csv'
                self.submit(_write_csv_part, kind, selected, coordinates, targets, batches, split,
                            epsilon, start, start+block_rows, column, not parts, part)
                parts.append(part)
        self.tables.append((path, parts))
        from dann.targets import observed_targets
        self.row_counts[path] = (int((observed_targets(extracted) > 0).sum()) if kind == 'scatter'
                                 else len(extracted['row_ids']))

    def assemble(self, progress):
        if self.tables:
            progress.set_phase('ordered CSV assembly', len(self.tables), 'tables')
        for index, (path, parts) in enumerate(self.tables):
            with path.open('wb') as destination:
                for part in parts:
                    with part.open('rb') as source:
                        shutil.copyfileobj(source, destination, length=8*1024**2)
            progress.update(index+1)

    def run(self, workers, performance, progress):
        progress.set_phase('figure and table rendering', len(self.tasks), 'tasks')
        if not workers:
            for index, task in enumerate(self.tasks):
                _render_task(*task)
                progress.update(index+1)
            self.assemble(progress)
            return
        # Imports, Agg canvases and temporary plotting arrays dominate worker RAM.
        largest = max((value.nbytes for value, _ in self.arrays.values()), default=0)
        per_worker = max(2 * 2**30, 2**30 + 4*self.canvas_bytes + 3*largest)
        import psutil
        available = min(performance['host_budget_gib'] * 2**30 - psutil.Process().memory_info().rss,
                        psutil.virtual_memory().available)
        requested = workers
        workers = min(workers, max(1, len(self.tasks)), max(1, int(available // per_worker)))
        progress.details.update(requested_render_workers=requested, effective_render_workers=workers,
                                render_worker_budget_bytes=per_worker, render_canvas_bytes=self.canvas_bytes)
        check_host_budget(performance, workers * per_worker)
        import matplotlib
        colormaps = {name: (cmap(np.arange(cmap.N)).tolist(), cmap.get_bad().tolist(),
                           cmap.get_under().tolist(), cmap.get_over().tolist())
                     for name in matplotlib.colormaps for cmap in [matplotlib.colormaps[name]]}
        palettes = self.directory / 'palettes.json'
        palettes.write_text(json.dumps(colormaps))
        with worker_environment(), ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn'), initializer=_init_renderer,
                initargs=(dict(matplotlib.rcParams), str(palettes))) as pool:
            pending, tasks, completed = set(), iter(self.tasks), 0
            try:
                for _ in range(2*workers):
                    task = next(tasks, None)
                    if task is not None:
                        pending.add(pool.submit(_render_task, *task))
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                        completed += 1
                        progress.update(completed)
                        check_host_budget(performance, workers=True)
                        task = next(tasks, None)
                        if task is not None:
                            pending.add(pool.submit(_render_task, *task))
            finally:
                for future in pending:
                    future.cancel()
        self.assemble(progress)


def _write_csv_part(kind, extracted, coordinates, targets, batches, split, epsilon,
                    start, stop, column, header, path):
    """Use existing frame builders, preserving column precision and row ordering."""
    from dann import analyze as a
    subset = {key: value[start:stop] if isinstance(value, np.ndarray) else value
              for key, value in extracted.items()}
    if kind == 'latent':
        frame = a.build_latent_embeddings_frame(subset, batches, split)
    elif kind == 'umap':
        frame = a.build_latent_umap_frame(subset, coordinates[start:stop], targets, batches, split)
    elif kind == 'scatter':
        for key in ('targets', 'raw_targets', 'target_valid_mask', 'pi', 'mu', 'sigma', 'density_mean'):
            if key in subset:
                subset[key] = subset[key][:, column:column+1]
        frame = a.build_ziln_scatter_frame(subset, [targets[column]], batches, split, epsilon)
    else:
        raise ValueError(f'Unknown analysis table kind: {kind}')
    frame.to_csv(path, index=False, header=header, float_format=None if kind == 'latent' else '%.17g')


def validate_outputs(directory: Path, *, expected_rows: Mapping[Path, int] | None = None) -> list[Path]:
    """Fully parse tables/arrays and verify image payloads before publishing."""
    import pandas as pd
    from PIL import Image
    files = sorted(path for path in directory.rglob('*') if path.is_file())
    if not files:
        raise ValueError('Analysis produced no outputs.')
    if expected_rows and set(expected_rows) - set(files):
        raise ValueError('Missing required analysis table outputs.')
    for path in files:
        if path.stat().st_size == 0:
            raise ValueError(f'Empty analysis output: {path}')
        if path.suffix == '.png':
            from dann.heatmap_execution import validate_png
            validate_png(path)
        elif path.suffix == '.webp':
            with Image.open(path) as image:
                image.load()
        elif path.suffix == '.csv':
            rows = sum(len(frame) for frame in pd.read_csv(path, chunksize=16384))
            if expected_rows and path in expected_rows and rows != expected_rows[path]:
                raise ValueError(f'Incomplete analysis table {path.name}: {rows} rows, expected {expected_rows[path]}.')
        elif path.suffix == '.npy':
            np.load(path, mmap_mode='r', allow_pickle=False)
        elif path.suffix == '.json':
            json.loads(path.read_text())
    return files


from dann.analysis_outputs import write_artifacts


def run_optimized_analysis(config: Mapping[str, Any], checkpoint_override: Path | None,
                           settings: Mapping[str, Any], output_override: Path | None = None) -> Path:
    """Execute validated numerical stages, then publish complete analysis artifacts."""
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from dann import analyze as a
    from dann.checkpoints import validate_checkpoint, validate_training_data_contract
    from dann.config import density_sampling_settings, execution_settings, resolve_device, seed_everything
    from dann.data_loader import create_data_bundle, load_anndata_metadata
    from dann.execution import PreparedLoader, build_validation_features, validation_execution
    from dann.spatial_execution import inference_performance, preflight, release_preparation
    from dann.targets import checkpoint_analysis_config, target_contract
    from dann.tiles import row_identity, tile_collate

    from dann.analysis_outputs import analysis_needs
    if not analysis_needs(config['analysis'])['inference']:
        return a.run_analysis(config, checkpoint_override, execution_overrides=settings, output_override=output_override)
    with InferenceProgress('Analysis') as progress:
        progress.geometry_phase = 'input and geometry preparation'
        needs = analysis_needs(config['analysis'])
        config = a._analysis_config(config)
        seed_everything(int(config['training']['seed']))
        device = resolve_device(str(config['training']['device']))
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        checkpoint_path = Path(checkpoint_override or config['analysis'].get('checkpoint')
                               or Path(config['training']['output_dir']) / 'best.pt')
        checkpoint, checkpoint_digest = load_checkpoint_snapshot(checkpoint_path)
        config = checkpoint_analysis_config(config, checkpoint)
        validate_training_data_contract(checkpoint, config)
        validate_checkpoint(checkpoint, config)
        output = Path(output_override or config['analysis']['output_dir']).resolve()
        output.mkdir(parents=True, exist_ok=True)
        splits = list(dict.fromkeys(([config['analysis']['split']] if needs['main'] else []) + ([config['analysis']['ziln_scatter_split']] if needs['scatter'] else [])))
        fast = settings['enabled'] and execution_settings(config)['mode'] == 'spatial' and 'train' not in splits
        performance = inference_performance(config, {'performance': settings['performance']},
                                          {**settings, 'feature_cache': settings['feature_cache'] and fast})
        progress.details.update(execution=settings, performance=performance, cache_hits=[],
                                feature_cache_active=settings['feature_cache'] and fast)
        data = config['data']
        source = Path(data['path'])
        from dann.preprocessing import artifact_preprocessing
        preprocessing = artifact_preprocessing(source, data)
        stat = source.stat()
        identity = row_identity(source, data)
        if checkpoint.get('data_identity') is not None and checkpoint['data_identity'] != identity:
            raise ValueError('Checkpoint row identities or feature order differ from input data.')
        metadata = load_anndata_metadata(source, data['target_columns'], data['batch_column'],
                                        data['matrix_key'], data.get('cd8_normalization'))
        if metadata.batch_names != tuple(checkpoint['batch_names']):
            raise ValueError('Analysis batch order differs from checkpoint.')
        if metadata.num_peaks != config['model']['num_peaks']:
            raise ValueError('Analysis feature width differs from checkpoint.')
        saved_splits = checkpoint.get('split_indices')
        if saved_splits is not None:
            if set(saved_splits) != {'train', 'validation', 'test'}:
                raise ValueError('Checkpoint must contain all three saved splits.')
            combined = np.concatenate([np.asarray(rows, dtype=np.int64) for rows in saved_splits.values()])
            if (combined < 0).any() or (combined >= metadata.num_observations).any() or len(np.unique(combined)) != len(combined):
                raise ValueError('Invalid or overlapping saved split identities.')
        versions = {name: importlib.metadata.version(name) for name in
                    ('numpy', 'torch', 'scipy', 'umap-learn', 'scikit-learn', 'anndata',
                     'numba', 'pynndescent', 'llvmlite')}
        common = dict(version=1, checkpoint=checkpoint_digest, source=(str(source.resolve()),
                      stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns), identity=identity,
                      versions=versions, preprocessing=preprocessing)
        progress.details.update(source_identity=common, checkpoint=str(checkpoint_path),
                                checkpoint_epoch=int(checkpoint['epoch'])+1)
        cache = StageCache(output / '.analysis-cache', settings['cache_policy'], progress)
        keys, extracted_by_split = {}, {}
        for split in splits:
            keys[split] = dict(common, split=split, data=data, model=config['model'],
                loss=config['loss'], training=config['training'],
                max_samples=config['analysis']['max_samples'],
                saved_rows={name: fingerprint(np.asarray(rows)) for name, rows in checkpoint.get('split_indices', {}).items()},
                performance=performance if fast else config.get('performance', {}),
                physical_batch=config['analysis']['batch_size'], fast=fast,
                export_he_predictions=config['analysis'].get('export_he_predictions', False),
                **({} if needs['features']['he_metrics']['enabled'] else {'collect_he_metrics': False}),
                device=str(device), cuda=torch.version.cuda)
            # Training samplers may intentionally repeat rows. Leave their existing
            # extraction behavior alone; numerical caches are for held-out analysis.
            hit = cache.get(f'extracted-{split}', keys[split]) if split != 'train' and not config['analysis'].get('export_he_predictions', False) else None
            if hit is not None:
                extracted_by_split[split] = hit[0]
        missing = [split for split in splits if split not in extracted_by_split]
        model = bundle = feature_cache = None
        loaders = {}
        try:
            if missing:
                model, epoch = a.load_model(checkpoint_path, config,
                    SimpleNamespace(metadata=metadata), device,
                    checkpoint=checkpoint)
            if missing:
                progress.set_phase('input and geometry preparation')
                runtime = copy.deepcopy(config)
                if fast:
                    runtime['performance'] = performance
                progress.details['memory_preflight'] = preflight(source, runtime,
                    sum(len(checkpoint.get('split_indices', {}).get(split, [])) for split in missing), performance,
                    predict_he=needs['features']['he_metrics']['enabled'] or config['analysis'].get('export_he_predictions', False))
                bundle = create_data_bundle(runtime, checkpoint.get('split_indices'),
                    checkpoint.get('data_identity'), requested_splits=missing, progress=progress)
                if fast:
                    for split in missing:
                        loader = DataLoader(bundle.datasets[split], batch_size=settings['tiles_per_batch'],
                                            shuffle=False, collate_fn=tile_collate)
                        loaders[split] = PreparedLoader(loader, performance, config['model'], validation=True)
                        loaders[split].inference = True
                    with torch.inference_mode(), validation_execution(model, performance):
                        if settings['feature_cache']:
                            feature_cache = build_validation_features(model, loaders[missing[0]], device,
                                progress, context_datasets=[bundle.datasets[split] for split in missing])
                        for split in missing:
                            from functools import partial
                            from dann.he import extract_with_he
                            extracted_by_split[split] = extract_with_he(model, loaders[split], device, config,
                                output, split, partial(extract_prepared, cache=feature_cache, progress=progress), report_metrics=False)
                            progress.details.setdefault('preparation_wait_seconds', {})[split] = loaders[split].wait_seconds
                            progress.details.setdefault('max_prepared_batch_bytes', {})[split] = loaders[split].max_batch_bytes
                else:
                    for split in missing:
                        progress.set_phase(f'legacy {split} extraction')
                        from dann.he import extract_with_he
                        extracted_by_split[split] = extract_with_he(model, bundle.loaders[split], device, config, output, split, report_metrics=False)
                for split in missing:
                    if split != 'train':
                        cache.put(f'extracted-{split}', keys[split], extracted_by_split[split])
        finally:
            feature_cache = None
            if bundle is not None:
                for loader in bundle.loaders.values():
                    release_preparation(loader)
            model = bundle = None
            loaders.clear()
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        samples, seed = density_sampling_settings(config['analysis'])
        progress.details['density_sampling'] = dict(samples=samples, seed=seed, split_order=splits)
        progress.details['output_rows'] = {split: len(values['row_ids'])
                                           for split, values in extracted_by_split.items()}
        rng = np.random.default_rng(seed)
        for split in splits:
            extracted = extracted_by_split[split]
            extracted['target_transform'] = target_contract(config)
            if not ((split == config['analysis']['split'] and needs['density_main']) or
                    (split == config['analysis']['ziln_scatter_split'] and needs['density_scatter'])):
                continue
            sample_key = dict(version=1, parameters=[fingerprint(extracted[name]) for name in ('pi', 'mu', 'sigma')],
                              samples=samples, rng_state=copy.deepcopy(rng.bit_generator.state), numpy=versions['numpy'], preprocessing=preprocessing)
            hit = cache.get(f'density-{split}', sample_key)
            progress.set_phase(f'{split} density sampling', extracted['pi'].size, 'parameters')
            if hit is None:
                check_host_budget(performance, (2*settings['sampling_workers']+1)*1024*samples*16)
                means = sampled_density_mean(extracted['pi'], extracted['mu'], extracted['sigma'],
                    num_samples=samples, rng=rng, workers=settings['sampling_workers'], progress=progress)
                cache.put(f'density-{split}', sample_key, {'means': means},
                          {'rng_state': rng.bit_generator.state})
            else:
                means = hit[0]['means']
                rng.bit_generator.state = hit[1]['rng_state']
            extracted['density_mean'] = means
            extracted['target_transform'] = target_contract(config)

        extracted = extracted_by_split[splits[0]]
        sample = extracted
        coordinates = pca_coordinates = variance = pca_metadata = None
        provenance = {}
        if needs['samples']:
            from dann.analysis_latents import select_sample_rows, cached_pca, variance_from_pca, fit_pca_umap
            sample, slides, provenance = select_sample_rows(extracted, config, metadata.batch_names)
            provenance.update(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=checkpoint_digest,
                checkpoint_epoch=int(checkpoint['epoch'])+1, row_ids_sha256=fingerprint(sample['row_ids']),
                source_identity=common, latent_source='inferred or validated extraction cache')
            pca = None
            if len(sample['latent']) >= 2 and any(needs['features'][n]['enabled'] for n in ('variance', 'pca_umap')):
                pca = cached_pca(sample['latent'], cache)
            if needs['features']['variance']['enabled'] and pca is not None:
                variance = variance_from_pca(sample['latent'], slides, pca)
            if needs['features']['pca_umap']['enabled']:
                pca_coordinates, pca_metadata = fit_pca_umap(sample['latent'], pca, cache)
                progress.details['pca_umap'] = pca_metadata
            if needs['features']['direct_umap']['enabled'] and len(sample['latent']):
                umap_key = dict(version=2, latent=fingerprint(sample['latent']), versions=versions,
                    settings={key: config['analysis'][key] for key in
                        ('umap_neighbors', 'umap_min_dist', 'umap_metric', 'umap_random_state')})
                progress.details['umap_settings'] = umap_key['settings']
                hit = cache.get('umap', umap_key)
                progress.set_phase('seeded UMAP', len(sample['latent']))
                if hit is None:
                    coordinates = a.fit_latent_umap(sample['latent'], config['analysis'])
                    cache.put('umap', umap_key, {'coordinates': coordinates})
                else:
                    coordinates = hit[0]['coordinates']
                progress.update(len(coordinates))

        with tempfile.TemporaryDirectory(prefix='.analysis-work-', dir=output) as temporary:
            work = Path(temporary)
            staged = work / 'outputs'
            staged.mkdir()
            queue = RenderQueue(work / 'arrays')
            progress.set_phase('tables and figure preparation')
            write_artifacts(config, checkpoint_path, int(checkpoint['epoch'])+1, metadata,
                extracted, extracted_by_split.get(config['analysis']['ziln_scatter_split']), coordinates,
                None, None, staged, queue.submit, checkpoint=checkpoint, table_writer=queue.table,
                sample_data=sample, pca_coordinates=pca_coordinates, variance=variance,
                sample_provenance=provenance, pca_metadata=pca_metadata, cache=cache)
            queue.run(settings['render_workers'], performance, progress)
            from dann.analysis_outputs import write_latent_manifest
            write_latent_manifest(staged, output, sample, cache)
            from dann.preprocessing import analysis_provenance
            analysis_provenance(staged, data, write=True)
            progress.set_phase('output validation')
            final_stat = source.stat()
            if any(getattr(stat, key) != getattr(final_stat, key) for key in ('st_size', 'st_mtime_ns', 'st_ctime_ns')):
                raise ValueError('Analysis source changed during processing; outputs were not published.')
            files = validate_outputs(staged, expected_rows=queue.row_counts)
            from dann.artifact_layout import analysis_relative_path
            for path in files:
                destination = output / analysis_relative_path(path.relative_to(staged))
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination)
            progress.details['artifacts'] = [str(analysis_relative_path(path.relative_to(staged))) for path in files]
            if config['analysis'].get('export_he_predictions', False):
                progress.details['artifacts'] += [f'he/{split}_predictions.parquet' for split in missing
                    if (output / 'he' / f'{split}_predictions.parquet').is_file()]
            metrics = work / 'analysis_execution.json'
            metrics.write_text(json.dumps(progress.snapshot(device), indent=2))
            os.replace(metrics, output / 'analysis_execution.json')
        return output

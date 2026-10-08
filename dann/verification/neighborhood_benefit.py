"""Frozen-checkpoint validation of local context, with bounded GPU allocation.

Run with --cnn-checkpoint PATH --mlp-checkpoint PATH --output NEW_DIRECTORY.
Only validation labels are scored. Other labeled-grid spectra provide context;
no tissue-wide predictions, training, or test-label evaluation are performed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
from scipy.special import expit, roots_hermitenorm
import torch

from dann.checkpoints import architecture_contract
from dann.data_loader import AnnDataMetadata, load_anndata_metadata
from dann.model import AdversarialLatentFusion, transform_biology
from dann.spatial import SparseInferenceDataset, load_checkpoint_model, sparse_inference_collate
from dann.targets import CD8, checkpoint_contract
from dann.tiles import SpatialTileDataset, read_grid, row_identity, tile_collate
from dann.train import move_batch_to_device

SEEDS = tuple(range(20261003, 20261008))
RADIUS = 3
GIB = 1024 ** 3
PARAMETERS = ('pi_logits', 'mu', 'sigma')


def save_json(path: Path, value: object) -> None:
    """Atomically persist audit metadata, retaining completed chunk artifacts."""
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=convert) + '\n')
    temporary.replace(path)


def file_hash(path: Path) -> str:
    """Hash a checkpoint or artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    """Fingerprint ordered split identities without changing their representation."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def numpy_safe_globals() -> list:
    """Allow only standard NumPy array reconstruction in restricted torch loads."""
    return [np.dtype, np.ndarray, np._core.multiarray._reconstruct, np.dtypes.Int64DType]


def read_checkpoint(path: Path) -> dict:
    """Read tensor/dictionary/NumPy checkpoint data with the restricted unpickler."""
    with torch.serialization.safe_globals(numpy_safe_globals()):
        return torch.load(path, map_location='cpu', weights_only=True)


def load_frozen_model(path: Path) -> tuple[AdversarialLatentFusion, dict, tuple[str, ...], int]:
    """Use the inference-compatible loader, enforcing its restricted unpickler.

    Legacy training provenance remains absent; it is never synthesized to pass
    the newer training loader's requirements.
    """
    previous = os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD')
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
    try:
        with torch.serialization.safe_globals(numpy_safe_globals()):
            model, config, targets, _, epoch = load_checkpoint_model(path, 'cpu')
    finally:
        if previous is None:
            os.environ.pop('TORCH_FORCE_WEIGHTS_ONLY_LOAD', None)
        else:
            os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = previous
    return model.float().eval(), config, targets, epoch


def training_running() -> bool:
    """Observe training processes without signaling or modifying them."""
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = proc.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b'dann.train' in args and b'-m' in args:
            return True
    return False


class ResourceGuard:
    """Pause below free-memory thresholds and wait out training after an OOM."""
    def __init__(self, path: Path, device: torch.device,
                 free_bytes: Callable | None = None, sleep: Callable = time.sleep,
                 training: Callable = training_running):
        self.path, self.device = path, device
        self.free_bytes = free_bytes or (lambda: torch.cuda.mem_get_info(device)[0])
        self.sleep, self.training = sleep, training
        self.last_print = 0.

    def log(self, event: str, **values) -> None:
        record = dict(time_utc=pd.Timestamp.now(tz='UTC').isoformat(), event=event, **values)
        with self.path.open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        if event != 'chunk' or time.monotonic() - self.last_print > 30:
            print(json.dumps(record), flush=True)
            self.last_print = time.monotonic()

    def wait(self, initial: bool = False) -> None:
        threshold = (16 if initial else 12) * GIB
        while True:
            free = self.free_bytes()
            if free >= threshold:
                self.log('chunk', free_gib=free / GIB)
                return
            self.log('memory_pause', free_gib=free / GIB, required_gib=threshold / GIB)
            self.sleep(30)

    def run(self, function: Callable):
        while True:
            self.wait()
            try:
                return function()
            except torch.cuda.OutOfMemoryError:
                pass
            # Leave the exception scope so its traceback releases GPU temporaries.
            self.log('analysis_oom', action='preserve completed chunks; wait for training exit')
            gc.collect()
            torch.cuda.empty_cache()
            while self.training():
                self.log('oom_wait_training')
                self.sleep(30)
            self.wait(initial=True)
            # Retry once after training exits; a second OOM is a runner failure.
            return function()


def neighbor_rows(keys: pd.MultiIndex, rows: np.ndarray, radius: int = RADIUS) -> np.ndarray:
    """Look up square patches in y-major order using immutable grid identities."""
    slide = np.asarray(keys.get_level_values(0))[rows]
    x = np.asarray(keys.get_level_values(1))[rows]
    y = np.asarray(keys.get_level_values(2))[rows]
    offsets = [(dx, dy) for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)]
    result = np.empty((len(rows), len(offsets)), dtype=np.int32)
    for j, (dx, dy) in enumerate(offsets):
        result[:, j] = keys.get_indexer(pd.MultiIndex.from_arrays([slide, x + dx, y + dy]))
    if not np.array_equal(result[:, len(offsets) // 2], rows):
        raise ValueError('Patch centers do not match selected rows.')
    return result


def donor_pools(keys: pd.MultiIndex) -> dict[str, np.ndarray]:
    """Find centers with complete 7x7 support, using geometry alone."""
    slides = np.asarray(keys.get_level_values(0))
    x, y = (np.asarray(keys.get_level_values(i), dtype=np.int64) for i in (1, 2))
    pools = {}
    for slide in np.unique(slides):
        rows = np.flatnonzero(slides == slide)
        xx, yy = x[rows] - x[rows].min(), y[rows] - y[rows].min()
        if (xx.max() + 1) * (yy.max() + 1) > 100_000_000:
            raise ValueError('Slide coordinate range is too large for the occupancy grid.')
        mask = np.zeros((yy.max() + 1, xx.max() + 1), dtype=bool)
        mask[yy, xx] = True
        complete = binary_erosion(mask, structure=np.ones((7, 7), dtype=bool))
        pools[str(slide)] = rows[complete[yy, xx]]
        if not len(pools[str(slide)]):
            raise ValueError(f'No fully supported donors on slide {slide}.')
    return pools


def select_donors(keys: pd.MultiIndex, rows: np.ndarray, pools: dict,
                  rng: np.random.Generator) -> np.ndarray:
    """Sample same-slide donors whose complete square cannot overlap recipients."""
    slides = np.asarray(keys.get_level_values(0))
    coordinates = np.array([keys.get_level_values(1), keys.get_level_values(2)]).T
    donors = np.empty(len(rows), dtype=np.int64)
    for slide in np.unique(slides[rows]):
        positions = np.flatnonzero(slides[rows] == slide)
        pool = pools[str(slide)]
        pending = positions
        for _ in range(100):
            sampled = rng.choice(pool, len(pending))
            separate = (np.abs(coordinates[sampled] - coordinates[rows[pending]]) > 2 * RADIUS).any(1)
            donors[pending[separate]] = sampled[separate]
            pending = pending[~separate]
            if not len(pending):
                break
        for i in pending:
            eligible = pool[(np.abs(coordinates[pool] - coordinates[rows[i]]) > 2 * RADIUS).any(1)]
            if not len(eligible):
                raise ValueError(f'No separated donor for row {rows[i]}.')
            donors[i] = rng.choice(eligible)
    return donors


def perturb_sources(original: np.ndarray, condition: str, rng: np.random.Generator,
                    donor_patches: np.ndarray | None = None) -> np.ndarray:
    """Return feature-source identities; preserve every center and occupancy bit."""
    result = original.copy()
    center = original.shape[1] // 2
    if condition == 'distant':
        if donor_patches is None or donor_patches.shape != original.shape or (donor_patches < 0).any():
            raise ValueError('Distant neighborhoods require complete, aligned donor patches.')
        result[:] = donor_patches
    elif condition == 'rearranged':
        for i, patch in enumerate(original):
            occupied = np.flatnonzero((patch >= 0) & (np.arange(len(patch)) != center))
            result[i, occupied] = rng.permutation(patch[occupied])
    elif condition == 'replicated':
        result[:] = original[:, center, None]
    elif condition != 'original':
        raise ValueError(condition)
    result[original < 0] = -1
    result[:, center] = original[:, center]
    if not np.array_equal(result < 0, original < 0):
        raise AssertionError('Occupancy changed.')
    return result


def patch_predictions(model: AdversarialLatentFusion, features: np.ndarray, sources: np.ndarray,
                      device: torch.device) -> np.ndarray:
    """Apply the frozen biology head to independent, explicitly masked patches."""
    side = int(np.sqrt(sources.shape[1]))
    mask = sources >= 0
    values = np.asarray(features[np.maximum(sources, 0)]).copy()
    values[~mask] = 0
    grid = torch.from_numpy(values.reshape(len(sources), side, side, -1)).movedim(-1, 1).to(device)
    occupancy = torch.from_numpy(mask.reshape(-1, 1, side, side).astype(np.float32)).to(device)
    with torch.inference_mode():
        if model.biology_predictor.radius:
            raw = model.biology_predictor(grid, occupancy)[:, :, side // 2, side // 2]
            prediction = transform_biology(raw, model.num_targets, model.sigma_min)
        else:
            prediction = model.biology_predictor(grid[:, :, side // 2, side // 2])
    return np.stack([prediction[k].cpu().numpy() for k in PARAMETERS], axis=-1)


def quadrature_mean(prediction: np.ndarray, tolerance: float = 1e-6) -> tuple[np.ndarray, dict]:
    """Converge unconditional logistic-normal means with 64--512 Gaussian nodes."""
    logits, mu, sigma = np.moveaxis(np.asarray(prediction, dtype=np.float64), -1, 0)
    if not np.isfinite(prediction).all() or (sigma <= 0).any():
        raise ValueError('Non-finite parameters or non-positive sigma.')
    def evaluate(nodes):
        abscissas, weights = roots_hermitenorm(nodes)
        result = np.empty(mu.size, dtype=np.float64)
        for start in range(0, mu.size, 2048):
            stop = min(start + 2048, mu.size)
            values = expit(mu.ravel()[start:stop, None] + sigma.ravel()[start:stop, None] * abscissas)
            result[start:stop] = values @ (weights / np.sqrt(2 * np.pi))
        return result.reshape(mu.shape) * expit(-logits)
    previous = evaluate(64)
    comparisons = {}
    for nodes in (128, 256, 512):
        current = evaluate(nodes)
        error = float(np.max(np.abs(current - previous)))
        comparisons[str(nodes)] = error
        if error <= tolerance:
            return current, dict(nodes=nodes, converged=True, maximum_absolute_difference=error,
                                 comparisons=comparisons)
        previous = current
    return current, dict(nodes=512, converged=False, maximum_absolute_difference=error,
                         comparisons=comparisons)


def elementwise_metrics(prediction: np.ndarray, density: np.ndarray, targets: np.ndarray,
                        valid: np.ndarray, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Compute per-row sufficient statistics with the production float32 loss."""
    logits, mu, sigma = (torch.from_numpy(np.array(prediction[..., i], dtype=np.float32)) for i in range(3))
    y = torch.from_numpy(np.where(valid, targets, 0).astype(np.float32))
    z = torch.logit(y.clamp(config['loss']['logit_epsilon'], 1 - config['loss']['logit_epsilon']))
    positive = (targets > 0) & valid
    nll = torch.nn.functional.binary_cross_entropy_with_logits(logits, (y == 0).float(), reduction='none')
    gaussian = .5 * ((z - mu) / sigma).square() + sigma.log()
    if config['loss']['include_normal_constant']:
        gaussian += .5 * np.log(2 * np.pi)
    nll += torch.where(torch.from_numpy(positive), gaussian, 0)
    observed = np.where(valid, targets, 0)
    return dict(loss=np.where(valid, nll.numpy(), 0).astype(np.float64),
                brier=np.where(valid, (expit(prediction[..., 0]) - (observed == 0)) ** 2, 0).astype(np.float64),
                absolute_error=np.where(valid, np.abs(density - observed), 0),
                squared_error=np.where(valid, (density - observed) ** 2, 0),
                positive_sse=np.where(positive, (z.numpy() - mu.numpy()) ** 2, 0).astype(np.float64))


def summarize(metrics: Mapping[str, np.ndarray], targets: np.ndarray, valid: np.ndarray,
              selected: np.ndarray, weights: np.ndarray, epsilon: float) -> dict:
    """Reduce metrics by valid counts; positive R² uses only positive labels."""
    count = valid[selected].sum(0)
    positive = (targets[selected] > 0) & valid[selected]
    z = torch.logit(torch.from_numpy(targets[selected]).clamp(epsilon, 1 - epsilon)).numpy().astype(float)
    pc = positive.sum(0)
    zsum = np.where(positive, z, 0).sum(0)
    tss = np.where(positive, z * z, 0).sum(0) - zsum ** 2 / np.maximum(pc, 1)
    result = {k: np.divide(v[selected].sum(0), count, out=np.full(len(count), np.nan), where=count > 0)
              for k, v in metrics.items() if k != 'positive_sse'}
    result['rmse'] = np.sqrt(result.pop('squared_error'))
    result['positive_logit_r2'] = 1 - np.divide(metrics['positive_sse'][selected].sum(0), tss,
        out=np.full(len(count), np.nan), where=(pc >= 2) & (tss > 1e-12))
    mass = np.dot(count, weights)
    result['weighted_biology_loss'] = float(np.dot(metrics['loss'][selected].sum(0), weights) / mass) if mass else np.nan
    result['valid_count'] = count
    return result


def bootstrap_delta(delta: np.ndarray, valid: np.ndarray, slide_codes: np.ndarray,
                    weights: np.ndarray, resamples: int = 2000) -> dict:
    """Paired slide-cluster bootstrap, preserving pixel-weighted point estimates."""
    slides = np.unique(slide_codes)
    sums = np.array([np.where(valid[slide_codes == s], delta[slide_codes == s], 0).sum(0) for s in slides])
    counts = np.array([valid[slide_codes == s].sum(0) for s in slides])
    rng = np.random.default_rng(SEEDS[0])
    draws = rng.integers(len(slides), size=(resamples, len(slides)))
    total, mass = sums[draws].sum(1), counts[draws].sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        boot = total / mass
        overall = sums.sum(0) / counts.sum(0)
        weighted_boot = (total @ weights) / (mass @ weights)
    return dict(delta=overall, lower=np.nanquantile(boot, .025, axis=0),
                upper=np.nanquantile(boot, .975, axis=0),
                weighted_delta=float((sums.sum(0) @ weights) / (counts.sum(0) @ weights)),
                weighted_lower=float(np.nanquantile(weighted_boot, .025)),
                weighted_upper=float(np.nanquantile(weighted_boot, .975)),
                resamples=resamples, slides=len(slides))



def bootstrap_metric_changes(baseline: Mapping[str, np.ndarray], perturbed: Mapping[str, np.ndarray],
                             targets: np.ndarray, valid: np.ndarray, slide_codes: np.ndarray,
                             epsilon: float, resamples: int = 2000) -> dict:
    """Bootstrap paired changes in Brier, MAE, RMSE and positive-logit R².

    RMSE and R² are recomputed from sufficient statistics within each slide
    resample, rather than averaging per-slide scores or prediction parameters.
    """
    slides = np.unique(slide_codes)
    def aggregate(values):
        return np.array([values[slide_codes == slide].sum(0) for slide in slides])
    counts = aggregate(valid)
    positive = valid & (targets > 0)
    z = torch.logit(torch.from_numpy(np.where(valid, targets, 0).astype(np.float32)).clamp(epsilon, 1-epsilon)).numpy().astype(float)
    pcs = aggregate(positive)
    zs = aggregate(np.where(positive, z, 0))
    zss = aggregate(np.where(positive, z*z, 0))
    draws = np.random.default_rng(SEEDS[0]).integers(len(slides), size=(resamples, len(slides)))
    mass = counts[draws].sum(1)
    positive_mass = pcs[draws].sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        tss = zss[draws].sum(1) - zs[draws].sum(1)**2 / positive_mass
        total_tss = zss.sum(0) - zs.sum(0)**2 / pcs.sum(0)
        output = {}
        for key, label in [('brier', 'brier'), ('absolute_error', 'mae'),
                           ('squared_error', 'rmse'), ('positive_sse', 'positive_logit_r2')]:
            a, b = aggregate(baseline[key]), aggregate(perturbed[key])
            if key == 'squared_error':
                values = np.sqrt(b[draws].sum(1)/mass) - np.sqrt(a[draws].sum(1)/mass)
                estimate = np.sqrt(b.sum(0)/counts.sum(0)) - np.sqrt(a.sum(0)/counts.sum(0))
            elif key == 'positive_sse':
                values = np.where((positive_mass >= 2) & (tss > 1e-12),
                                  (a[draws].sum(1) - b[draws].sum(1))/tss, np.nan)
                estimate = np.where((pcs.sum(0) >= 2) & (total_tss > 1e-12),
                                    (a.sum(0)-b.sum(0))/total_tss, np.nan)
            else:
                values = (b[draws].sum(1)-a[draws].sum(1))/mass
                estimate = (b.sum(0)-a.sum(0))/counts.sum(0)
            output[label] = dict(delta=estimate, lower=np.nanquantile(values, .025, axis=0),
                                 upper=np.nanquantile(values, .975, axis=0))
    return output


def adjacent_pairs(keys: pd.MultiIndex, rows: np.ndarray) -> np.ndarray:
    """Unique horizontal/vertical adjacent pairs with both endpoints in validation."""
    local = keys[rows]
    pairs = []
    for dx, dy in ((1, 0), (0, 1)):
        shifted = pd.MultiIndex.from_arrays([local.get_level_values(0),
            local.get_level_values(1) + dx, local.get_level_values(2) + dy])
        other = local.get_indexer(shifted)
        left = np.flatnonzero(other >= 0)
        pairs.extend(zip(left, other[left]))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def create_array(path: Path, shape: tuple[int, ...], dtype: Any = np.float32) -> np.memmap:
    """Create a self-describing disk-backed NumPy array."""
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)


def cache_features(model: AdversarialLatentFusion, config: Mapping[str, Any], keys: pd.MultiIndex,
                   directory: Path, guard: ResourceGuard, device: torch.device, batch_size: int) -> np.memmap:
    """Cache each model's own pointwise encoder features; never fit preprocessing."""
    data = config['data']
    cache = create_array(directory / 'aggregation_features.npy', (len(keys), model.latent_dim))
    dataset = SparseInferenceDataset(Path(data['path']), len(keys), data['matrix_key'],
        data['intensity_transform'], data.get('intensity_clip_max'), data['nonzero_threshold'])
    try:
        for start in range(0, len(keys), batch_size):
            stop = min(start + batch_size, len(keys))
            cpu = sparse_inference_collate([dataset[i] for i in range(start, stop)])
            def encode():
                with torch.inference_mode():
                    batch = move_batch_to_device(cpu, device)
                    args = [batch[k] for k in ('peak_indices', 'intensities', 'sample_indices', 'peak_counts')]
                    pooled = model.encoder.microbatched(*args, peak_budget=model.peak_budget, checkpointing=False)
                    return model.encoder.aggregation_mlp(pooled).cpu().numpy()
            cache[start:stop] = guard.run(encode)
            if start % (batch_size * 32) == 0:
                cache.flush()
                save_json(directory / 'progress.json', dict(stage='features', completed=stop, total=len(keys)))
    finally:
        dataset.close()
        cache.flush()
    return cache


def normal_predictions(model: AdversarialLatentFusion, config: Mapping[str, Any],
                       metadata: AnnDataMetadata, rows: np.ndarray, directory: Path,
                       guard: ResourceGuard, device: torch.device, batch_size: int) -> np.memmap:
    """Recompute ordinary forward-path validation predictions in original row order."""
    result = create_array(directory / 'normal_predictions.npy', (len(rows), model.num_targets, 3))
    positions = np.full(metadata.num_observations, -1, dtype=np.int64)
    positions[rows] = np.arange(len(rows))
    seen = np.zeros(len(rows), bool)
    if model.spatial:
        dataset = SpatialTileDataset(config, rows, metadata)
        try:
            for index in range(len(dataset)):
                cpu = tile_collate([dataset[index]])
                ids = positions[cpu['row_ids'].numpy()]
                def forward():
                    with torch.inference_mode():
                        output = model(move_batch_to_device(cpu, device), grl_strength=0.)
                    return np.stack([output[k].cpu().numpy() for k in PARAMETERS], -1)
                result[ids] = guard.run(forward)
                seen[ids] = True
                if index % 128 == 0:
                    result.flush()
                    save_json(directory / 'progress.json', dict(stage='normal_tiles', completed=index+1, total=len(dataset)))
        finally:
            dataset.close()
    else:
        data = config['data']
        dataset = SparseInferenceDataset(Path(data['path']), metadata.num_observations, data['matrix_key'],
            data['intensity_transform'], data.get('intensity_clip_max'), data['nonzero_threshold'])
        try:
            for start in range(0, len(rows), batch_size):
                stop = min(start + batch_size, len(rows))
                cpu = sparse_inference_collate([dataset[int(row)] for row in rows[start:stop]])
                def forward():
                    with torch.inference_mode():
                        output = model(move_batch_to_device(cpu, device), grl_strength=0.)
                    return np.stack([output[k].cpu().numpy() for k in PARAMETERS], -1)
                result[start:stop] = guard.run(forward)
                seen[start:stop] = True
        finally:
            dataset.close()
    result.flush()
    if not seen.all():
        raise AssertionError('Normal validation did not cover every row.')
    return result


def run_patches(model: AdversarialLatentFusion, features: np.ndarray, sources: np.ndarray,
                path: Path, guard: ResourceGuard, device: torch.device, batch_size: int) -> np.memmap:
    """Persist central predictions in bounded independent-patch batches."""
    result = create_array(path, (len(sources), model.num_targets, 3))
    for start in range(0, len(sources), batch_size):
        stop = min(start + batch_size, len(sources))
        result[start:stop] = guard.run(lambda: patch_predictions(model, features, sources[start:stop], device))
        if start % (batch_size * 32) == 0:
            result.flush()
            save_json(path.parent / 'progress.json', dict(stage=path.stem, completed=stop, total=len(sources)))
    result.flush()
    return result


def preflight(paths: Mapping[str, Path], output: Path, expected_rows: int) -> tuple[dict, dict, np.ndarray]:
    """Require aligned frozen identities, transformations, saved splits and scores."""
    payloads, configs, manifests = {}, {}, {}
    for name, path in paths.items():
        checkpoint = read_checkpoint(path)
        model, config, targets, epoch = load_frozen_model(path)
        del model
        if config['loss']['ziln_reduction'] != 'mean':
            raise ValueError('This analysis requires mean-reduced training losses.')
        identity = row_identity(Path(config['data']['path']), config['data'])
        if not checkpoint.get('data_identity') or identity != checkpoint['data_identity']:
            raise ValueError(f'{name}: saved input identity does not match.')
        if checkpoint_contract(checkpoint)['target_columns'] != list(targets):
            raise ValueError('Target transformation order differs.')
        architecture = architecture_contract(config)
        if architecture['components']['aggregation']['type'] != 'mlp':
            raise ValueError('Only pointwise aggregation checkpoints are supported.')
        if architecture['components']['biology']['type'] != ('cnn' if name == 'cnn' else 'mlp'):
            raise ValueError('Incorrect biology-head type.')
        if name == 'cnn' and architecture['components']['biology']['cnn']['depth'] != RADIUS:
            raise ValueError('CNN biology must have a 7x7 receptive field.')
        splits = {k: np.asarray(v, dtype=np.int64) for k, v in checkpoint['split_indices'].items()}
        combined = np.concatenate(list(splits.values()))
        if (combined < 0).any() or (combined >= identity['rows']).any() or len(np.unique(combined)) != len(combined):
            raise ValueError('Invalid or overlapping saved splits.')
        if len(splits['validation']) != expected_rows:
            raise ValueError(f'Expected {expected_rows} validation rows.')
        history = pd.read_csv(path.parent / 'history.csv')
        saved = history.loc[history.epoch == epoch]
        if len(saved) != 1:
            raise ValueError('Saved checkpoint epoch absent from history.')
        manifests[name] = dict(path=str(path), sha256=file_hash(path), epoch=epoch,
            architecture=architecture, config=config, saved_config=checkpoint['config'], target_transform=checkpoint_contract(checkpoint),
            data_identity=identity, split_hashes={k: array_hash(v) for k, v in splits.items()},
            split_counts={k: len(v) for k, v in splits.items()},
            training_data_contract=checkpoint.get('training_data_contract'),
            training_provenance_status='saved' if 'training_data_contract' in checkpoint else 'unknown_legacy_provenance',
            sampling_contract=checkpoint.get('sampling_contract'),
            saved_validation=saved.iloc[0].to_dict())
        payloads[name], configs[name] = checkpoint, config
    a, b = payloads.values()
    if a['data_identity'] != b['data_identity'] or checkpoint_contract(a) != checkpoint_contract(b):
        raise ValueError('Model data/target identities differ.')
    for split in a['split_indices']:
        if not np.array_equal(a['split_indices'][split], b['split_indices'][split]):
            raise ValueError(f'Model {split} rows differ.')
    if configs['cnn']['loss'] != configs['mlp']['loss']:
        raise ValueError('Loss settings differ; paired loss would be ambiguous.')
    save_json(output / 'checkpoint_manifests.json', manifests)
    for name, payload in payloads.items():
        np.savez(output / f'{name}_splits.npz', **payload['split_indices'])
    return configs, manifests, np.asarray(a['split_indices']['validation'], dtype=np.int64)


def build_conditions(keys: pd.MultiIndex, rows: np.ndarray, output: Path) -> tuple[list, pd.DataFrame]:
    """Save reproducible feature-source identities and common spatial groups."""
    original = neighbor_rows(keys, rows)
    pools = donor_pools(keys)
    save_json(output / 'donor_support.json', {s: len(v) for s, v in pools.items()})
    conditions = [('original', None), ('replicated', None)] + [
        (condition, seed) for condition in ('distant', 'rearranged') for seed in SEEDS]
    paths = []
    for condition, seed in conditions:
        name = condition if seed is None else f'{condition}_{seed}'
        rng = np.random.default_rng(seed if seed is not None else SEEDS[0])
        donors = select_donors(keys, rows, pools, rng) if condition == 'distant' else None
        patches = neighbor_rows(keys, donors) if donors is not None else None
        sources = perturb_sources(original, condition, rng, patches)
        path = output / f'{name}_sources.npy'
        np.save(path, sources)
        if donors is not None:
            np.save(output / f'{name}_donor_centers.npy', donors)
        paths.append((condition, seed, name, path))
    frame = keys[rows].to_frame(index=False)
    frame.insert(0, 'row_id', rows)
    frame['occupied_positions'] = (original >= 0).sum(1)
    frame['group'] = np.where((original >= 0).all(1), 'interior', 'boundary')
    frame.to_csv(output / 'validation_rows.csv', index=False)
    return paths, frame


def evaluate_model(name: str, path: Path, config: Mapping[str, Any], manifest: Mapping[str, Any],
                   metadata: AnnDataMetadata, keys: pd.MultiIndex, rows: np.ndarray, conditions: list,
                   output: Path, guard: ResourceGuard, device: torch.device, args: argparse.Namespace) -> None:
    """Verify a frozen model and persist every validation intervention."""
    directory = output / name
    directory.mkdir()
    model, _, targets, _ = load_frozen_model(path)
    guard.wait(initial=True)
    model = guard.run(lambda: model.to(device))
    # Bound spectral temporaries without changing any learned parameters.
    model.peak_budget = 8192
    model.encoder.inference_peak_chunk_size = 8192
    model.checkpointing = False
    features = cache_features(model, config, keys, directory, guard, device, args.encoder_batch_size)
    normal = normal_predictions(model, config, metadata, rows, directory, guard, device, args.encoder_batch_size)
    original = None
    for condition, seed, label, sources_path in conditions:
        sources = np.load(sources_path, mmap_mode='r')
        prediction = run_patches(model, features, sources, directory / f'{label}_predictions.npy',
                                 guard, device, args.patch_batch_size)
        if condition == 'original':
            difference = np.abs(np.asarray(prediction) - normal)
            check = dict(maximum_absolute_difference=float(difference.max()),
                         rows=len(rows), atol=2e-5, rtol=2e-5)
            save_json(directory / 'patch_tile_equivalence.json', check)
            np.testing.assert_allclose(prediction, normal, atol=2e-5, rtol=2e-5)
            original = prediction
            density, _ = quadrature_mean(prediction)
            metrics = elementwise_metrics(prediction, density, metadata.targets[rows], metadata.target_valid_mask[rows], config)
            summary = summarize(metrics, metadata.targets[rows], metadata.target_valid_mask[rows],
                np.ones(len(rows), bool), np.asarray(config['loss']['target_weights']), config['loss']['logit_epsilon'])
            saved = manifest['saved_validation']
            checks = {'cd8_loss': (summary['loss'][list(targets).index(CD8)], saved['validation_cd8_loss']),
                      'biology_loss': (summary['weighted_biology_loss'], saved['validation_biology_loss'])}
            save_json(directory / 'saved_score_agreement.json', {k: dict(recomputed=float(v[0]), saved=v[1], difference=float(v[0]-v[1])) for k, v in checks.items()})
            for value, reference in checks.values():
                if abs(value - reference) > 2e-5:
                    raise ValueError(f'{name}: unmodified validation does not reproduce saved loss.')
        elif name == 'mlp':
            if not np.array_equal(prediction, original):
                raise AssertionError(f'MLP negative control changed for {label}.')
    save_json(directory / 'completion.json', dict(conditions=len(conditions), rows=len(rows), mlp_invariant=name == 'mlp'))
    del model, features, normal, original
    gc.collect()
    torch.cuda.empty_cache()


def analyze_results(configs: Mapping[str, Any], manifests: Mapping[str, Any],
                    metadata: AnnDataMetadata, keys: pd.MultiIndex, rows: np.ndarray,
                    conditions: list, frame: pd.DataFrame, output: Path) -> None:
    """Export paired effects, slide/boundary metrics, contrasts, plots and report."""
    targets = configs['cnn']['data']['target_columns']
    y, valid = metadata.targets[rows], metadata.target_valid_mask[rows]
    weights = np.asarray(configs['cnn']['loss']['target_weights'])
    epsilon = configs['cnn']['loss']['logit_epsilon']
    slide_labels, slide_codes = np.unique(frame.iloc[:, 1].to_numpy(), return_inverse=True)
    groups = {'all': np.ones(len(rows), bool), 'interior': frame.group.to_numpy() == 'interior',
              'boundary': frame.group.to_numpy() == 'boundary'}
    metric_rows, effect_rows, consistency, changes, quadrature, contrast_rows = [], [], [], [], {}, []
    metric_effect_rows = []
    pairs = adjacent_pairs(keys, rows)
    np.save(output / 'adjacent_validation_pairs.npy', pairs)
    original_density = {}
    for model_name, config in configs.items():
        directory = output / model_name
        averaged, baseline = {}, None
        baseline_parameters = None
        for condition, seed, label, _ in conditions:
            prediction = np.load(directory / f'{label}_predictions.npy', mmap_mode='r')
            density, convergence = quadrature_mean(prediction)
            quadrature[f'{model_name}/{label}'] = convergence
            np.save(directory / f'{label}_density_mean.npy', density)
            metric = elementwise_metrics(prediction, density, y, valid, config)
            if baseline is None:
                baseline, baseline_parameters = metric, prediction
                original_density[model_name] = density
            else:
                difference = np.abs(prediction - baseline_parameters)
                density_difference = np.abs(density - original_density[model_name])
                for j, target in enumerate(targets):
                    changes.append(dict(model=model_name, condition=condition, seed=seed, target=target,
                        mean_abs_pi_logits_change=float(difference[:, j, 0].mean()),
                        mean_abs_mu_change=float(difference[:, j, 1].mean()),
                        mean_abs_sigma_change=float(difference[:, j, 2].mean()),
                        mean_abs_density_change=float(density_difference[:, j].mean()),
                        max_abs_density_change=float(density_difference[:, j].max())))
            if condition not in averaged:
                averaged[condition] = {k: v.copy() for k, v in metric.items()}
            else:
                for k in metric:
                    averaged[condition][k] += metric[k]
            if seed is not None:
                delta = bootstrap_delta(metric['loss'] - baseline['loss'], valid, slide_codes, weights)
                consistency.append(dict(model=model_name, condition=condition, seed=seed,
                    cd8_delta=float(delta['delta'][targets.index(CD8)]),
                    weighted_delta=delta['weighted_delta']))
        for condition, metric in averaged.items():
            if condition in ('distant', 'rearranged'):
                metric = {k: v / len(SEEDS) for k, v in metric.items()}
            for group, selection in groups.items():
                for slide in [None, *range(len(slide_labels))]:
                    mask = selection if slide is None else selection & (slide_codes == slide)
                    if not mask.any():
                        continue
                    summary = summarize(metric, y, valid, mask, weights, epsilon)
                    for j, target in enumerate(targets):
                        metric_rows.append(dict(model=model_name, condition=condition, group=group,
                            slide='ALL' if slide is None else slide_labels[slide], target=target,
                            rows=int(mask.sum()), **{k: float(v[j]) for k, v in summary.items() if k != 'weighted_biology_loss'},
                            weighted_biology_loss=summary['weighted_biology_loss']))
                if condition != 'original' and selection.any():
                    # Repetitions are averaged at the row-loss level before slide resampling.
                    delta = bootstrap_delta((metric['loss'] - baseline['loss'])[selection], valid[selection],
                                             slide_codes[selection], weights)
                    additional = bootstrap_metric_changes(
                        {k: v[selection] for k, v in baseline.items()},
                        {k: v[selection] for k, v in metric.items()}, y[selection], valid[selection],
                        slide_codes[selection], epsilon)
                    for metric_name, interval in additional.items():
                        for j, target in enumerate(targets):
                            metric_effect_rows.append(dict(model=model_name, condition=condition, group=group,
                                target=target, metric=metric_name, delta=float(interval['delta'][j]),
                                lower=float(interval['lower'][j]), upper=float(interval['upper'][j]), resamples=2000))
                    for j, target in enumerate(targets):
                        effect_rows.append(dict(model=model_name, condition=condition, group=group, target=target,
                            delta=float(delta['delta'][j]), lower=float(delta['lower'][j]), upper=float(delta['upper'][j]),
                            weighted_delta=delta['weighted_delta'], weighted_lower=delta['weighted_lower'],
                            weighted_upper=delta['weighted_upper'], resamples=2000, slides=delta['slides']))
        for j, target in enumerate(targets):
            eligible = valid[pairs[:, 0], j] & valid[pairs[:, 1], j]
            selected_pairs = pairs[eligible]
            if not len(selected_pairs):
                continue
            a, b = selected_pairs.T
            observed = y[b, j] - y[a, j]
            predicted = original_density[model_name][b, j] - original_density[model_name][a, j]
            threshold = np.quantile(np.abs(observed), .75)
            for group, selection in groups.items():
                both = selection[a] & selection[b]
                for contrast_group, high in [('all', np.ones(len(a), bool)), ('highest_observed_quartile', np.abs(observed) >= threshold)]:
                    for slide in [None, *range(len(slide_labels))]:
                        keep = both & high
                        if slide is not None:
                            keep &= slide_codes[a] == slide
                        if not keep.any():
                            continue
                        o, p = observed[keep], predicted[keep]
                        observed_mean = float(np.abs(o).mean())
                        predicted_mean = float(np.abs(p).mean())
                        contrast_rows.append(dict(model=model_name, target=target, group=group,
                            slide='ALL' if slide is None else slide_labels[slide], contrast_group=contrast_group,
                            pairs=int(keep.sum()), observed_abs_contrast=observed_mean,
                            predicted_abs_contrast=predicted_mean,
                            attenuation_ratio=predicted_mean / observed_mean if observed_mean else np.nan,
                            signed_contrast_rmse=float(np.sqrt(np.mean((p-o)**2))),
                            signed_contrast_correlation=float(np.corrcoef(o, p)[0, 1]) if len(o)>1 and o.std()>0 and p.std()>0 else np.nan,
                            high_quartile_threshold=float(threshold)))
    table = pd.DataFrame(metric_rows)
    table.to_csv(output / 'metrics.csv', index=False)
    table[(table.slide == 'ALL') & (table.group == 'all')].to_csv(output / 'comparison.csv', index=False)
    table[table.slide != 'ALL'].to_csv(output / 'per_slide.csv', index=False)
    table[(table.slide == 'ALL') & (table.group != 'all')].to_csv(output / 'boundary_metrics.csv', index=False)
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(output / 'paired_effects.csv', index=False)
    pd.DataFrame(metric_effect_rows).to_csv(output / 'paired_metric_effects.csv', index=False)
    pd.DataFrame(consistency).to_csv(output / 'repetition_consistency.csv', index=False)
    pd.DataFrame(changes).to_csv(output / 'prediction_changes.csv', index=False)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(output / 'adjacent_contrasts.csv', index=False)
    save_json(output / 'quadrature_convergence.json', quadrature)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, group in zip(axes, ('all', 'interior', 'boundary')):
        values = effects[(effects.model == 'cnn') & (effects.target == CD8) & (effects.group == group)]
        for i, record in enumerate(values.itertuples()):
            ax.plot([record.lower, record.upper], [i, i], color='tab:blue')
            ax.plot(record.delta, i, 'o', color='tab:blue')
        ax.set_yticks(range(len(values)), values.condition)
        ax.axvline(0, color='black', linewidth=.8)
        ax.axvline(.001, color='gray', linestyle=':', linewidth=.8)
        ax.set(title=f'CNN CD8: {group}', xlabel='Perturbed minus original loss (95% slide interval)')
    fig.tight_layout()
    fig.savefig(output / 'effect_sizes.png', dpi=180)
    plt.close(fig)
    if len(contrasts):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, contrast_group in zip(axes, ('all', 'highest_observed_quartile')):
            selected = contrasts[(contrasts.target == CD8) & (contrasts.group == 'all') &
                                 (contrasts.slide == 'ALL') & (contrasts.contrast_group == contrast_group)]
            if len(selected):
                ax.bar(['Observed', *selected.model.str.upper()],
                       [selected.observed_abs_contrast.iloc[0], *selected.predicted_abs_contrast],
                       color=['gray', 'tab:blue', 'tab:orange'])
            ax.set(title=contrast_group.replace('_', ' '), ylabel='Mean absolute adjacent CD8 contrast')
        fig.tight_layout()
        fig.savefig(output / 'adjacent_contrasts.png', dpi=180)
        plt.close(fig)
    primary = effects[(effects.model == 'cnn') & (effects.target == CD8) & (effects.group == 'all') & (effects.condition == 'distant')].iloc[0]
    change = pd.DataFrame(changes)
    dependency = change[(change.model == 'cnn') & (change.condition == 'distant') & (change.target == CD8)]
    unresolved = [k for k, v in quadrature.items() if not v['converged']]
    support = 'Evidence of useful local context' if primary.lower > 0 else 'No positive-confidence-interval evidence of useful local context'
    size = 'smaller than 0.001' if abs(primary.delta) < .001 else 'at least 0.001 in absolute magnitude'
    repeat_table = pd.DataFrame(consistency)
    primary_repeats = repeat_table[(repeat_table.model == 'cnn') & (repeat_table.condition == 'distant')].cd8_delta
    normalization = configs['cnn']['data'].get('cd8_normalization', {})
    cd8_scale = ('CD8 is the checkpoint’s density divided by non-tumor fraction; the saved validity exclusions are preserved.'
                 if normalization.get('enabled', False) else 'CD8 uses the checkpoint’s raw density scale.')
    report = [
        '# Frozen biology neighborhood evaluation', '',
        f"CNN epoch {manifests['cnn']['epoch']}; MLP epoch {manifests['mlp']['epoch']}; {len(rows):,} identical validation rows across {len(slide_labels)} slides.",
        cd8_scale + f' Valid CD8 labels: {int(valid[:, targets.index(CD8)].sum()):,}.',
        'Both unmodified losses reproduce their saved history; every CNN patch agrees with the normal tiled forward path within the recorded float32 tolerances.', '',
        f"**Predictive benefit:** {support}. Primary CD8 loss increase after distant-neighborhood replacement: {primary.delta:.8f}, 95% paired slide-bootstrap interval [{primary.lower:.8f}, {primary.upper:.8f}]. The effect is {size}.",
        f"Repetition consistency: {int((primary_repeats > 0).sum())}/5 distant-neighborhood repetitions increase CD8 loss; individual changes range from {primary_repeats.min():.8f} to {primary_repeats.max():.8f}.",
        f"**Neighbor dependence:** mean absolute CD8 density prediction change under distant replacement, averaged over seeds: {dependency.mean_abs_density_change.mean():.8f}. MLP predictions are exactly invariant under every perturbation.", '',
        '## Original predictions', '',
        '| Model | CD8 loss | Weighted biology loss | CD8 density RMSE |',
        '|---|---:|---:|---:|']
    for name in ('cnn', 'mlp'):
        row = table[(table.model == name) & (table.condition == 'original') & (table.group == 'all') & (table.slide == 'ALL') & (table.target == CD8)].iloc[0]
        report.append(f'| {name} | {row.loss:.8f} | {row.weighted_biology_loss:.8f} | {row.rmse:.8f} |')
    report += ['', '## Neighbor interventions', '',
               '| Intervention | CD8 loss change | 95% slide interval | Magnitude |',
               '|---|---:|---:|---|']
    for effect in effects[(effects.model == 'cnn') & (effects.target == CD8) & (effects.group == 'all')].itertuples():
        magnitude = 'below 0.001' if abs(effect.delta) < .001 else 'at least 0.001'
        if effect.lower > 0 and abs(effect.delta) < .001:
            magnitude += '; statistically detectable but small'
        report.append(f'| {effect.condition} | {effect.delta:.8f} | [{effect.lower:.8f}, {effect.upper:.8f}] | {magnitude} |')
    report += ['', '**Possible oversmoothing:** adjacent-pixel contrasts use only edges with both endpoints in validation and valid labels. Boundary and interior groups and high observed-contrast quartiles are identical for both models; ties at the quartile threshold are included. Ratios below one describe attenuation, which alone does not prove harmful smoothing.']
    if len(contrasts):
        for name in ('cnn', 'mlp'):
            chosen = contrasts[(contrasts.model == name) & (contrasts.target == CD8) & (contrasts.group == 'all') & (contrasts.slide == 'ALL') & (contrasts.contrast_group == 'highest_observed_quartile')]
            if len(chosen):
                r = chosen.iloc[0]
                report.append(f"{name.upper()}: high-contrast CD8 edges {int(r.pairs):,}; observed mean absolute contrast {r.observed_abs_contrast:.6f}, predicted {r.predicted_abs_contrast:.6f}, ratio {r.attenuation_ratio:.4f}, signed contrast RMSE {r.signed_contrast_rmse:.6f}.")
    report += ['', '## Interpretation and audit', '',
        'Positive paired deltas mean the original neighbors have lower loss. The 2,000-resample bootstrap samples slides as clusters, retaining pixel-weighted losses and the target-validity denominators. Randomized losses and other additive metrics are averaged over the five seeds before aggregation; repetition_consistency.csv retains individual repetition effects. R² and RMSE use averaged squared errors, not averaged prediction parameters. paired_effects.csv contains target and weighted biology loss intervals; paired_metric_effects.csv contains Brier, MAE, RMSE and positive-logit R² change intervals. All changes are perturbed minus original; a positive R² change indicates improvement.',
        f"CNN learning rate: {configs['cnn']['training']['learning_rate']}; MLP learning rate: {configs['mlp']['training']['learning_rate']}. CNN sampling: {configs['cnn']['training'].get('sampling_strategy', 'legacy pixel sampling')}; MLP sampling: {configs['mlp']['training'].get('sampling_strategy', 'legacy pixel sampling')}. Different learned encoders, sampling and learning rates confound the direct architecture comparison. Full saved configurations and sampling contracts are in checkpoint_manifests.json; missing MLP training provenance remains explicitly unknown.",
        'Gaussian quadrature compares 64 and 128 nodes, escalating through 256 and 512 until maximum absolute change is at most 1e-6. ' + (f'UNRESOLVED numerical discrepancies: {unresolved}. Density-based differences for these conditions must not be treated as model effects.' if unresolved else 'All conditions converged within 1e-6.'),
        'These perturbations describe these frozen models. They do not estimate the performance of a separately retrained neighbor-free model or generalization to unseen slides. No test-label scoring, tissue-wide prediction, architecture changes, or retraining was performed. Spectra from the labeled source grid, including other split rows, supply the same spatial context as ordinary validation.',
        'Resource policy: float32, eval/inference mode, two CPU threads, no loader workers, a 3 GiB PyTorch allocator cap, initial free-memory threshold 16 GiB, between-chunk threshold 12 GiB. The cap applies to PyTorch allocations; CUDA context/library memory is outside that allocator. Resource events and pauses are logged in resources.jsonl.',
        '', '## Artifacts', '',
        'validation_rows.csv maps every prediction row to source row and grid identity. *_sources.npy contains the exact feature-source row at each y-major 7×7 location (-1 means unoccupied); donor centers are saved separately. Model directories contain frozen aggregation_features.npy, normal_predictions.npy, each condition’s predictions.npy (last axis: pi_logits, mu, sigma), deterministic density_mean.npy, and equivalence/score checks. Original labels and validity are stored in validation_targets.npz. comparison.csv, per_slide.csv, boundary_metrics.csv, paired_effects.csv, prediction_changes.csv, adjacent_contrasts.csv, and effect_sizes.png contain the results.']
    (output / 'report.md').write_text('\n'.join(report) + '\n')
    save_json(output / 'completion.json', dict(status='completed', validation_rows=len(rows), slides=len(slide_labels),
        quadrature_converged=not unresolved, primary_cd8_delta=float(primary.delta),
        primary_interval=[float(primary.lower), float(primary.upper)]))



def summarize_existing(output: Path) -> None:
    """Rebuild CPU summaries from completed, identity-checked inference artifacts."""
    manifests = json.loads((output / 'checkpoint_manifests.json').read_text())
    configs = {name: manifest['config'] for name, manifest in manifests.items()}
    for name, manifest in manifests.items():
        completion = json.loads((output / name / 'completion.json').read_text())
        if completion['conditions'] != 12:
            raise ValueError('Incomplete inference conditions.')
        path = Path(manifest['path'])
        if file_hash(path) != manifest['sha256']:
            raise ValueError('Checkpoint changed since inference.')
        if row_identity(Path(configs[name]['data']['path']), configs[name]['data']) != manifest['data_identity']:
            raise ValueError('Input identity changed since inference.')
    config = configs['cnn']
    data = config['data']
    with np.load(output / 'cnn_splits.npz') as splits:
        rows = splits['validation']
    keys, _ = read_grid(Path(data['path']), data)
    metadata = load_anndata_metadata(Path(data['path']), data['target_columns'],
        data['batch_column'], data['matrix_key'], data['cd8_normalization'])
    with np.load(output / 'validation_targets.npz') as saved:
        np.testing.assert_array_equal(saved['targets'], metadata.targets[rows])
        np.testing.assert_array_equal(saved['valid'], metadata.target_valid_mask[rows])
    frame = pd.read_csv(output / 'validation_rows.csv', dtype={data['batch_column']: str})
    np.testing.assert_array_equal(frame.row_id, rows)
    conditions = [('original', None), ('replicated', None)] + [
        (condition, seed) for condition in ('distant', 'rearranged') for seed in SEEDS]
    conditions = [(condition, seed, condition if seed is None else f'{condition}_{seed}', None)
                  for condition, seed in conditions]
    (output / 'reporting_source.py').write_bytes(Path(__file__).read_bytes())
    save_json(output / 'reporting_provenance.json', dict(source_sha256=file_hash(Path(__file__)),
        inference_source_sha256=file_hash(output / 'neighborhood_benefit.py'),
        method='CPU summaries from completed frozen prediction arrays',
        checkpoints={name: manifest['sha256'] for name, manifest in manifests.items()}))
    analyze_results(configs, manifests, metadata, keys, rows, conditions, frame, output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cnn-checkpoint', type=Path)
    parser.add_argument('--mlp-checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summarize-only', action='store_true', help='Rebuild CPU reports from completed inference in --output.')
    parser.add_argument('--encoder-batch-size', type=int, default=256)
    parser.add_argument('--patch-batch-size', type=int, default=128)
    parser.add_argument('--expected-validation-rows', type=int, default=169824)
    args = parser.parse_args(argv)
    if min(args.encoder_batch_size, args.patch_batch_size, args.expected_validation_rows) <= 0:
        parser.error('Batch sizes and expected validation count must be positive.')
    output = args.output.resolve()
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if args.summarize_only:
        summarize_existing(output)
        return
    if args.cnn_checkpoint is None or args.mlp_checkpoint is None:
        parser.error('Both checkpoint paths are required for inference.')
    output.mkdir(parents=True, exist_ok=False)
    paths = dict(cnn=args.cnn_checkpoint.resolve(), mlp=args.mlp_checkpoint.resolve())
    save_json(output / 'invocation.json', dict(arguments=vars(args), seeds=SEEDS,
        allocator_cap_gib=3, cpu_threads=2, workers=0,
        source_sha256=file_hash(Path(__file__))))
    # Retain an exact source snapshot alongside the scientific results.
    (output / 'neighborhood_benefit.py').write_bytes(Path(__file__).read_bytes())
    try:
        configs, manifests, rows = preflight(paths, output, args.expected_validation_rows)
        config = configs['cnn']
        keys, _ = read_grid(Path(config['data']['path']), config['data'])
        metadata = load_anndata_metadata(Path(config['data']['path']), config['data']['target_columns'],
            config['data']['batch_column'], config['data']['matrix_key'], config['data']['cd8_normalization'])
        for name in paths:
            if tuple(manifests[name]['config']['data']['target_columns']) != tuple(config['data']['target_columns']):
                raise ValueError('Target order mismatch.')
            if tuple(read_checkpoint(paths[name])['batch_names']) != metadata.batch_names:
                raise ValueError('Saved batch order differs from input data.')
        np.savez(output / 'validation_targets.npz', targets=metadata.targets[rows],
                 raw_targets=metadata.raw_targets[rows], valid=metadata.target_valid_mask[rows])
        conditions, frame = build_conditions(keys, rows, output)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for this concurrent analysis.')
        device = torch.device('cuda:0')
        guard = ResourceGuard(output / 'resources.jsonl', device)
        guard.wait(initial=True)
        torch.cuda.set_per_process_memory_fraction(3 * GIB / torch.cuda.get_device_properties(device).total_memory, device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        guard.log('start', allocator_cap_gib=3, cpu_threads=2, loader_workers=0)
        for name, path in paths.items():
            evaluate_model(name, path, configs[name], manifests[name], metadata, keys, rows, conditions,
                           output, guard, device, args)
        analyze_results(configs, manifests, metadata, keys, rows, conditions, frame, output)
        for name, path in paths.items():
            if file_hash(path) != manifests[name]['sha256']:
                raise ValueError(f'{name} checkpoint changed during evaluation.')
        guard.log('completed', maximum_allocator_gib=torch.cuda.max_memory_allocated(device) / GIB)
    except BaseException:
        save_json(output / 'failure.json', dict(traceback=traceback.format_exc(), status='incomplete; artifacts preserved'))
        raise


if __name__ == '__main__':
    main()

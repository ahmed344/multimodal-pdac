"""Measured peak interpretation with frozen checkpoint semantics and no inference.

Sparse duplicates are summed into one measured peak before positivity, thresholding,
transformation, and detection counting. No pixels-by-all-peaks dense array is built.
"""
from __future__ import annotations

import gc
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from anndata.io import read_elem
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy.stats import rankdata
import torch

from dann.analysis_execution import StageCache, RenderQueue, fingerprint, load_checkpoint_snapshot, validate_outputs
from dann.config import peak_analysis_settings
from dann.data_loader import _matrix_group_path
from dann.execution import ResidentSpectra
from dann.heatmap_execution import source_identity
from dann.spatial_execution import InferenceProgress, check_host_budget
from dann.targets import checkpoint_analysis_config, prepare_observed_targets, target_contract, target_label
from dann.tiles import _axis_names, row_identity


def read_peak_metadata(path: Path, data: Mapping[str, Any], lesion_column: str,
                       *, spatial: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Read only identity, lesion, coordinate, and target columns from AnnData."""
    required = [data['batch_column'], lesion_column]
    if spatial:
        required += [data['x_column'], data['y_column']]
    with h5py.File(path, 'r') as handle:
        missing = set(required) - set(handle['obs'])
        if missing:
            raise ValueError(f'Missing peak metadata: {sorted(missing)}')
        frame = pd.DataFrame({key: read_elem(handle['obs'][key]) for key in required})
        frame.insert(0, 'obs_name', _axis_names(handle['obs']))
        frame.insert(0, 'row_position', np.arange(len(frame), dtype=np.int64))
        for column in data['target_columns']:
            frame[column] = read_elem(handle['obs'][column]) if column in handle['obs'] else np.nan
        features = _axis_names(handle['var'])
        group = handle[_matrix_group_path(data['matrix_key'])]
        if group.attrs.get('encoding-type') != 'csr_matrix' or tuple(group.attrs['shape']) != (len(frame), len(features)):
            raise ValueError('Peak source must be CSR with aligned metadata.')
    if frame[data['batch_column']].isna().any():
        raise ValueError('Slide identities must not be missing.')
    lesion = frame[lesion_column]
    if not pd.api.types.is_bool_dtype(lesion.dtype) or lesion.isna().any():
        raise ValueError('Lesion annotations must be nonmissing booleans.')
    if spatial:
        xy = frame[[data['x_column'], data['y_column']]].to_numpy()
        if not np.isfinite(xy).all() or not np.equal(xy, np.round(xy)).all():
            raise ValueError('Spatial coordinates must be finite integers.')
        if frame.duplicated([data['batch_column'], data['x_column'], data['y_column']]).any():
            raise ValueError('Duplicate spatial grid positions.')
    return frame, features


def canonical_block(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Coalesce measured intensities and remove stored zeros without densifying."""
    matrix = matrix.astype(np.float64, copy=True)
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError('Measured MSI intensities must be finite and nonnegative.')
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def resident_matrix(path: Path, data: Mapping[str, Any], performance: Mapping[str, Any]) -> sparse.csr_matrix:
    """Reuse the resident CSR reader with raw values and preflight construction RAM."""
    with h5py.File(path, 'r') as handle:
        group = handle[_matrix_group_path(data['matrix_key'])]
        required = group['data'].size * 12 + group['indptr'].size * 8
    check_host_budget(performance, 2 * required)
    resident = ResidentSpectra(path, {**data, 'intensity_transform': 'none',
        'intensity_clip_max': None, 'nonzero_threshold': 0.0}, int(performance['host_budget_gib'] * 2**30))
    return sparse.csr_matrix((resident.values, resident.indices, resident.indptr),
                            shape=(len(resident.indptr)-1, resident.num_peaks), copy=False)


def peak_statistics(matrix: sparse.csr_matrix, slides: np.ndarray, lesion: np.ndarray,
                    block_rows: int, progress=None) -> dict[str, np.ndarray]:
    """Count positive distinct peaks and raw positive sums on all lesion rows."""
    names, codes = np.unique(np.asarray(slides, dtype=str), return_inverse=True)
    counts = np.zeros((len(names), matrix.shape[1]), dtype=np.int64)
    sums = np.zeros_like(counts, dtype=np.float64)
    rows = np.flatnonzero(lesion)
    denominators = np.bincount(codes[rows], minlength=len(names))
    for start in range(0, len(rows), block_rows):
        selected = rows[start:start+block_rows]
        block = canonical_block(matrix[selected])
        members = np.repeat(codes[selected], np.diff(block.indptr))
        indices = members * matrix.shape[1] + block.indices
        counts += np.bincount(indices, minlength=counts.size).reshape(counts.shape)
        sums += np.bincount(indices, weights=block.data, minlength=sums.size).reshape(sums.shape)
        if progress:
            progress.update(start + len(selected))
    # Preserve the original top-peak selection: stored occurrences in all labeled rows.
    frequency = np.zeros(matrix.shape[1], dtype=np.int64)
    for start in range(0, matrix.nnz, 5_000_000):
        frequency += np.bincount(matrix.indices[start:start+5_000_000], minlength=matrix.shape[1])
    return dict(slides=names, counts=counts, sums=sums, rows=denominators, frequency=frequency)


def peak_correlations(matrix: sparse.csr_matrix, slides: np.ndarray, rows: np.ndarray,
                      targets: np.ndarray, min_observations: int = 20,
                      block_peaks: int = 64, progress=None) -> dict[str, np.ndarray]:
    """Per-slide Spearman correlations, with absent intensities represented as zero."""
    names = np.unique(np.asarray(slides, dtype=str))
    result = np.full((len(names), matrix.shape[1], targets.shape[1]), np.nan)
    nobs = np.zeros((len(names), targets.shape[1]), dtype=np.int64)
    for slide_index, name in enumerate(names):
        selected = rows[np.asarray(slides, dtype=str)[rows] == name]
        measured = canonical_block(matrix[selected]).tocsc()
        for target in range(targets.shape[1]):
            valid = np.isfinite(targets[selected, target])
            nobs[slide_index, target] = valid.sum()
            y = targets[selected, target][valid]
            if len(y) < min_observations or np.ptp(y) == 0:
                continue
            # SciPy preserves FP32 input precision in rankdata; use FP64 for
            # both rank norms so results agree with its full Spearman reference.
            ranks_y = rankdata(np.asarray(y, dtype=np.float64))
            ranks_y -= ranks_y.mean()
            norm_y = np.linalg.norm(ranks_y)
            subset = measured[valid]
            for start in range(0, matrix.shape[1], block_peaks):
                # A bounded peak block, never a full pixels-by-peaks dense matrix.
                x = subset[:, start:start+block_peaks].toarray()
                ranks_x = rankdata(x, axis=0)
                ranks_x -= ranks_x.mean(axis=0)
                denominator = np.linalg.norm(ranks_x, axis=0) * norm_y
                result[slide_index, start:start+x.shape[1], target] = np.divide(
                    ranks_y @ ranks_x, denominator, out=np.full(x.shape[1], np.nan), where=denominator > 0)
        if progress:
            progress.update(slide_index + 1)
    valid_count = np.isfinite(result).sum(axis=0)
    mean = np.divide(np.nansum(result, axis=0), valid_count,
                     out=np.full(result.shape[1:], np.nan), where=valid_count > 0)
    return dict(slides=names, correlations=result, valid_observations=nobs,
                mean=mean, contributing_slides=valid_count)


def family_correlations(matrix: sparse.csr_matrix, families: np.ndarray, slides: np.ndarray,
                        rows: np.ndarray, targets: np.ndarray, data: Mapping[str, Any],
                        block_rows: int = 4096, min_observations: int = 20,
                        progress=None) -> dict[str, np.ndarray]:
    """Associate mean transformed family signal on saved test-lesion rows only."""
    signals = family_spatial_scores(matrix[rows], families, data, block_rows)['signal']
    # Retain slides with no selected rows in the exported schema.
    selected_slides = np.asarray(slides, dtype=str)[rows]
    result = peak_correlations(sparse.csr_matrix(signals), selected_slides,
                              np.arange(len(rows)), targets[rows], min_observations,
                              max(1, len(np.unique(families))), progress)
    names = np.unique(np.asarray(slides, dtype=str))
    indices = np.searchsorted(names, result['slides'])
    correlations = np.full((len(names), len(np.unique(families)), targets.shape[1]), np.nan)
    nobs = np.zeros((len(names), targets.shape[1]), dtype=np.int64)
    correlations[indices] = result['correlations']
    nobs[indices] = result['valid_observations']
    return {**result, 'slides': names, 'correlations': correlations, 'valid_observations': nobs}


def family_hierarchy(weights: np.ndarray, count: int) -> dict[str, np.ndarray]:
    """Average-linkage cosine hierarchy with one reusable maxclust family cut."""
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2 or len(weights) == 0 or not np.isfinite(weights).all():
        raise ValueError('Invalid peak embedding matrix.')
    norms = np.linalg.norm(weights, axis=1)
    normalized = weights / np.maximum(norms[:, None], np.finfo(np.float32).eps)
    similarity = np.clip(normalized @ normalized.T, -1., 1.)
    if len(weights) == 1:
        linkage, labels, order = np.empty((0, 4)), np.ones(1, dtype=np.int32), np.zeros(1, dtype=np.int64)
    else:
        linkage = hierarchy.linkage(distance.squareform(1 - similarity, checks=False), method='average')
        labels = hierarchy.fcluster(linkage, t=count, criterion='maxclust').astype(np.int32)
        order = hierarchy.leaves_list(linkage).astype(np.int64)
    return dict(similarity=similarity, families=labels, order=order, linkage=linkage, norms=norms)


def family_similarity(similarity: np.ndarray, families: np.ndarray) -> np.ndarray:
    """Mean pairwise cosine similarity, excluding self pairs on the diagonal."""
    unique = np.unique(families)
    result = np.full((len(unique), len(unique)), np.nan)
    for i, left in enumerate(unique):
        rows = np.flatnonzero(families == left)
        for j, right in enumerate(unique):
            columns = np.flatnonzero(families == right)
            block = similarity[np.ix_(rows, columns)].astype(np.float64)
            if i == j:
                if len(rows) > 1:
                    result[i, j] = (block.sum() - np.trace(block)) / (len(rows) * (len(rows) - 1))
            else:
                result[i, j] = block.mean()
    return result


def family_spatial_scores(matrix: sparse.csr_matrix, families: np.ndarray, data: Mapping[str, Any],
                          block_rows: int, progress=None) -> dict[str, np.ndarray]:
    """Compute distinct detection percentage and transformed mean over all members."""
    unique, codes = np.unique(families, return_inverse=True)
    sizes = np.bincount(codes)
    mapping = sparse.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)),
                               shape=(len(codes), len(unique)))
    signal = np.empty((matrix.shape[0], len(unique)), dtype=np.float32)
    detection = np.empty_like(signal)
    for start in range(0, matrix.shape[0], block_rows):
        end = min(start+block_rows, matrix.shape[0])
        block = canonical_block(matrix[start:end])
        block.data[block.data <= max(0., float(data['nonzero_threshold']))] = 0
        block.eliminate_zeros()
        detected = block.copy()
        detected.data[:] = 1
        detection[start:end] = (detected @ mapping).toarray() * (100. / sizes)
        # Match checkpoint FP32 intensity transforms; accumulate family sums in FP64.
        values = block.data.astype(np.float32)
        if data['intensity_transform'] == 'log1p':
            values = np.log1p(values)
        elif data['intensity_transform'] == 'sqrt':
            values = np.sqrt(values)
        elif data['intensity_transform'] != 'none':
            raise ValueError('Unsupported checkpoint intensity transform.')
        if data.get('intensity_clip_max') is not None:
            values = np.minimum(values, float(data['intensity_clip_max']))
        block.data = values.astype(np.float64)
        signal[start:end] = (block @ mapping).toarray() / sizes
        if progress:
            progress.update(end)
    return dict(signal=signal, detection=detection)


def signal_limits(signal: np.ndarray, percentile: float) -> np.ndarray:
    """One positive-score percentile per family across all tissue rows."""
    return np.array([np.percentile(column[column > 0], percentile) if np.any(column > 0) else 1.
                     for column in signal.T], dtype=np.float64)


def spatial_geometry(frame: pd.DataFrame, data: Mapping[str, Any], lesion_column: str) -> list[dict]:
    """Map physical integer coordinates to grids without compressing empty rows/columns."""
    slides = []
    for name, rows in frame.groupby(data['batch_column'], sort=False, observed=True).indices.items():
        part = frame.iloc[rows]
        x = part[data['x_column']].to_numpy(dtype=np.int64)
        y = part[data['y_column']].to_numpy(dtype=np.int64)
        x, y = x-x.min(), y-y.min()
        mask = part[lesion_column].to_numpy(dtype=bool)
        crop = [int(y[mask].min()), int(y[mask].max())+1, int(x[mask].min()), int(x[mask].max())+1] if mask.any() else None
        slides.append(dict(name=str(name), rows=rows, x=x, y=y,
                           shape=(int(y.max())+1, int(x.max())+1), lesion=mask, crop=crop))
    return slides


def spatial_grid(values: np.ndarray, slide: dict, region: str) -> np.ndarray:
    """Preserve valid zeros, tissue gaps, and lesion holes using the same row scores."""
    grid = np.full(slide['shape'], np.nan, dtype=np.float32)
    grid[slide['y'], slide['x']] = values[slide['rows']]
    if region == 'tissue':
        return grid
    if slide['crop'] is None:
        return np.full((1, 1), np.nan, dtype=np.float32)
    grid[slide['y'][~slide['lesion']], slide['x'][~slide['lesion']]] = np.nan
    y0, y1, x0, x1 = slide['crop']
    return grid[y0:y1, x0:x1]


def _cached(cache, name, key, compute):
    hit = cache.get(name, key)
    if hit is not None:
        return hit[0]
    arrays = compute()
    cache.put(name, key, arrays)
    return arrays


def _write_spatial_table(path, frame, scores, families, data, lesion_column, metadata):
    columns = ['row_position', 'obs_name', data['batch_column'], data['x_column'], data['y_column'], lesion_column]
    with pq.ParquetWriter(path, pa.Table.from_pandas(_spatial_frame(frame.iloc[:0][columns], scores, families, 0, 0),
            preserve_index=False).schema.with_metadata({b'dann_peak_analysis': json.dumps(metadata).encode()})) as writer:
        for start in range(0, len(frame), 16384):
            end = min(start+16384, len(frame))
            writer.write_table(pa.Table.from_pandas(_spatial_frame(frame.iloc[start:end][columns], scores, families, start, end), preserve_index=False))


def _spatial_frame(frame, scores, families, start, end):
    result = frame.copy()
    for i, family in enumerate(families):
        result[f'family_{family:02d}_mean_signal'] = scores['signal'][start:end, i]
        result[f'family_{family:02d}_detection_percent'] = scores['detection'][start:end, i]
    return result


def validate_peak_outputs(directory: Path, weights: np.ndarray, families: np.ndarray,
                          spatial_rows: int | None, scores=None) -> list[Path]:
    """Check complete artifacts and exact FP32/row-aligned numerical export contracts."""
    files = validate_outputs(directory)
    exported = pd.read_csv(directory/'peak_embeddings.csv', float_precision='round_trip')
    if not np.array_equal(exported.filter(regex='^embedding_').to_numpy(dtype=np.float32), weights):
        raise ValueError('Peak embedding CSV failed exact FP32 round-trip.')
    for name in ('peak_annotations.csv', 'family_membership.csv', 'peak_umap.csv'):
        table = pd.read_csv(directory/name)
        if not np.array_equal(table['family'].to_numpy(), families) or not np.array_equal(table['peak_index'], np.arange(len(families))):
            raise ValueError(f'Inconsistent peak ordering/families: {name}')
    if spatial_rows is not None:
        parquet = pq.ParquetFile(directory/'family_spatial_activity.parquet')
        if parquet.metadata.num_rows != spatial_rows:
            raise ValueError('Incomplete spatial activity export.')
        offset = 0
        for batch in parquet.iter_batches(batch_size=16384):
            frame = batch.to_pandas()
            end = offset + len(frame)
            if not np.array_equal(frame.row_position, np.arange(offset, end)):
                raise ValueError('Spatial row order changed.')
            for i, family in enumerate(np.unique(families)):
                for key, suffix in (('signal', 'mean_signal'), ('detection', 'detection_percent')):
                    values = frame[f'family_{family:02d}_{suffix}'].to_numpy()
                    if not np.isfinite(values).all() or np.any(values < 0) or (key == 'detection' and np.any(values > 100)):
                        raise ValueError('Invalid spatial score.')
                    if scores is not None and not np.array_equal(values, scores[key][offset:end, i]):
                        raise ValueError('Spatial score round-trip failed.')
            offset = end
    for path in directory.glob('*.html'):
        text = path.read_text()
        if ('Plotly.newPlot' not in text and 'peak-similarity-viewer' not in text and 'cd8-comparison-viewer' not in text) or '<html>' not in text:
            raise ValueError(f'Invalid interactive artifact: {path}')
    return files


def run_peak_analysis(config: Mapping[str, Any], checkpoint_override: Path | None,
                      execution: Mapping[str, Any], output_override: Path | None = None,
                      spatial_input: Path | None = None) -> Path:
    """Cache numerical stages, render immutable data, validate, and publish artifacts."""
    from dann.checkpoints import validate_training_data_contract
    from dann import peak_plots as plots
    from threadpoolctl import threadpool_limits
    settings = peak_analysis_settings(config['analysis'].get('peaks'))
    checkpoint_path = Path(checkpoint_override or config['analysis'].get('checkpoint') or Path(config['training']['output_dir'])/'best.pt')
    performance = {'host_budget_gib': execution['performance'].get('host_budget_gib', 192.)}
    with InferenceProgress('Peaks') as progress, threadpool_limits(limits=4):
        checkpoint, digest = load_checkpoint_snapshot(checkpoint_path)
        config = checkpoint_analysis_config(config, checkpoint)
        validate_training_data_contract(checkpoint, config)
        data, analysis = config['data'], config['analysis']
        if analysis['clustering_linkage'] != 'average':
            raise ValueError('Peak analysis requires average linkage.')
        output = Path(output_override or analysis['output_dir']).resolve()
        output.mkdir(parents=True, exist_ok=True)
        destination = output/'peaks'
        source = Path(data['path'])
        sources = {'labeled': source_identity(source)}
        progress.set_phase('metadata and checkpoint validation')
        frame, features = read_peak_metadata(source, data, settings['lesion_column'])
        identity = row_identity(source, data)
        if checkpoint.get('data_identity', identity) != identity:
            raise ValueError('Checkpoint row/feature identity mismatch.')
        weights = checkpoint['model_state']['encoder.embedding.weight'].detach().cpu().numpy().astype(np.float32, copy=True)
        if len(weights) != len(features):
            raise ValueError('Checkpoint embedding/feature width mismatch.')
        rows = np.asarray(checkpoint['split_indices']['test'], dtype=np.int64)
        if (rows < 0).any() or (rows >= len(frame)).any() or len(np.unique(rows)) != len(rows):
            raise ValueError('Invalid saved test rows.')
        lesion = frame[settings['lesion_column']].to_numpy(dtype=bool)
        rows = rows[lesion[rows]]
        targets, valid = prepare_observed_targets(frame[data['target_columns']].to_numpy(), data['target_columns'], data.get('cd8_normalization'))
        targets = np.where(valid, targets, np.nan)
        del checkpoint
        cache = StageCache(output/'.analysis-cache', execution['cache_policy'], progress)
        versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'umap-learn', 'numba', 'pynndescent')}
        common = dict(version=1, checkpoint=digest, sources=sources.copy(), identity=identity,
                      versions=versions, matrix=data['matrix_key'])
        progress.details.update(settings=settings, execution=execution, checkpoint=str(checkpoint_path),
                                source_identity=common, cache_hits=[], test_lesion_rows=len(rows),
                                target_transform=target_contract(config),
                                preprocessing={key: data[key] for key in
                                    ('matrix_key', 'nonzero_threshold', 'intensity_transform', 'intensity_clip_max')})
        check_host_budget(performance, 64*len(weights)**2)
        family_key = dict(version=1, checkpoint=digest, count=analysis['family_count'], linkage='average', versions=versions)
        progress.set_phase('embedding hierarchy')
        tree = _cached(cache, 'peak-hierarchy', family_key, lambda: family_hierarchy(weights, analysis['family_count']))
        families, order = tree['families'], tree['order']
        unique = np.unique(families)
        stats_key = dict(common, statistics_version=2, lesion_column=settings['lesion_column'], lesion=fingerprint(lesion))
        corr_key = dict(stats_key, correlation_version=2, rows=fingerprint(rows), target_transform=target_contract(config),
                        minimum=settings['min_observations'], targets=fingerprint(targets))
        family_corr_key = dict(corr_key, family_correlation_version=1, families=fingerprint(families),
            preprocessing={k: data[k] for k in ('nonzero_threshold', 'intensity_transform', 'intensity_clip_max')})
        family_corr_hit = cache.get('family-correlations', family_corr_key)
        stats_hit, corr_hit = cache.get('peak-statistics', stats_key), cache.get('peak-correlations', corr_key)
        matrix = None
        if stats_hit is None or corr_hit is None or family_corr_hit is None:
            progress.set_phase('resident labeled spectra')
            matrix = resident_matrix(source, data, performance)
            check_host_budget(performance, settings['block_rows']*len(features)*32)
        if stats_hit is None:
            progress.set_phase('lesion peak statistics', int(lesion.sum()))
            stats = peak_statistics(matrix, frame[data['batch_column']].to_numpy(), lesion, settings['block_rows'], progress)
            cache.put('peak-statistics', stats_key, stats)
        else:
            stats = stats_hit[0]
        if corr_hit is None:
            progress.set_phase('held-out slide associations', len(stats['slides']), 'slides')
            correlations = peak_correlations(matrix, frame[data['batch_column']].to_numpy(), rows, targets,
                settings['min_observations'], settings['correlation_block_peaks'], progress)
            cache.put('peak-correlations', corr_key, correlations)
        else:
            correlations = corr_hit[0]
        if family_corr_hit is None:
            progress.set_phase('held-out family signal associations')
            family_associations = family_correlations(matrix, families, frame[data['batch_column']].to_numpy(),
                rows, targets, data, settings['block_rows'], settings['min_observations'], progress)
            cache.put('family-correlations', family_corr_key, family_associations)
        else:
            family_associations = family_corr_hit[0]
        del matrix, targets
        gc.collect()
        progress.set_phase('peak UMAP')
        umap_key = dict(family_key, settings={k: v for k, v in settings.items() if k.startswith('umap_')})
        def fit():
            import umap
            if len(weights) < 4:
                return {'coordinates': np.column_stack((np.arange(len(weights)), np.zeros(len(weights)))).astype(np.float32)}
            return {'coordinates': umap.UMAP(n_neighbors=min(settings['umap_neighbors'], len(weights)-1),
                min_dist=settings['umap_min_dist'], metric=settings['umap_metric'],
                random_state=settings['umap_random_state'], n_jobs=1).fit_transform(weights)}
        coordinates = _cached(cache, 'peak-umap', umap_key, fit)['coordinates']
        counts, sums = stats['counts'].sum(axis=0), stats['sums'].sum(axis=0)
        prevalence = 100.*counts/stats['rows'].sum() if stats['rows'].sum() else np.full(len(counts), np.nan)
        mean = np.divide(sums, counts, out=np.full(len(counts), np.nan), where=counts > 0)
        annotations = pd.DataFrame(dict(peak_index=np.arange(len(weights)), mz=features.astype(float),
            family=families, lesion_detection_count=counts, lesion_intensity_sum=sums,
            lesion_prevalence_percent=prevalence, mean_detected_intensity=mean, embedding_norm=tree['norms'], occurrence_count=stats['frequency']))
        annotations['family_color'] = [plots.family_colors(len(unique))[f-1] for f in families]
        for i, column in enumerate(data['target_columns']):
            annotations[f'spearman_{column}'] = correlations['mean'][:, i]
            annotations[f'contributing_slides_{column}'] = correlations['contributing_slides'][:, i]
            annotations[f'label_{column}'] = target_label(column, target_contract(config))
        slide_prevalence = np.divide(100.*stats['counts'], stats['rows'][:, None],
            out=np.full(stats['counts'].shape, np.nan), where=stats['rows'][:, None] > 0)
        scores = tissue = None
        with tempfile.TemporaryDirectory(prefix='.peaks-work-', dir=output) as temporary:
            work = Path(temporary)
            staged = work/'outputs'
            staged.mkdir()
            queue = RenderQueue(work/'arrays')
            comparison_queue = RenderQueue(work/'comparison-arrays')
            progress.set_phase('peak tables and plot preparation')
            embedding_frame = pd.DataFrame(weights, columns=[f'embedding_{i}' for i in range(weights.shape[1])])
            embedding_frame.insert(0, 'mz', features.astype(float))
            embedding_frame.insert(0, 'peak_index', np.arange(len(weights)))
            # Preserve FP32 embeddings exactly and retain the FP64 numeric m/z identity.
            embedding_frame.to_csv(staged/'peak_embeddings.csv', index=False, float_format='%.17g')
            annotations.to_csv(staged/'peak_annotations.csv', index=False)
            annotations[['peak_index', 'mz', 'family', 'family_color']].to_csv(staged/'family_membership.csv', index=False)
            pd.DataFrame(dict(family=unique, size=[int((families == f).sum()) for f in unique],
                color=plots.family_colors(len(unique)))).to_csv(staged/'family_sizes.csv', index=False)
            annotations[['peak_index', 'mz', 'family']].assign(umap_1=coordinates[:, 0], umap_2=coordinates[:, 1]).to_csv(staged/'peak_umap.csv', index=False)
            for key in ('linkage', 'order'):
                np.save(staged/f'peak_{key}.npy', tree[key])
            pd.DataFrame(family_similarity(tree['similarity'], families), index=unique, columns=unique).to_csv(staged/'family_similarity.csv', index_label='family')
            pd.concat([pd.DataFrame(dict(slide=name, peak_index=np.arange(len(weights)),
                lesion_rows=stats['rows'][i], detection_count=stats['counts'][i], intensity_sum=stats['sums'][i],
                prevalence_percent=slide_prevalence[i], mean_detected_intensity=np.divide(stats['sums'][i], stats['counts'][i],
                    out=np.full(len(weights), np.nan), where=stats['counts'][i] > 0))) for i, name in enumerate(stats['slides'])]).to_csv(staged/'peak_slide_statistics.csv', index=False)
            pd.concat([pd.DataFrame(dict(slide=name, peak_index=np.arange(len(weights)), target=column,
                target_label=target_label(column, target_contract(config)), spearman=correlations['correlations'][s, :, t],
                valid_observations=correlations['valid_observations'][s, t]))
                for s, name in enumerate(correlations['slides']) for t, column in enumerate(data['target_columns'])]).to_csv(staged/'peak_slide_ihc_associations.csv', index=False)
            pd.concat([pd.DataFrame(dict(family=unique, target=column,
                target_label=target_label(column, target_contract(config)), spearman=family_associations['mean'][:, t],
                contributing_slides=family_associations['contributing_slides'][:, t]))
                for t, column in enumerate(data['target_columns'])]).to_csv(staged/'family_ihc_associations.csv', index=False)
            pd.concat([pd.DataFrame(dict(slide=name, family=unique, target=column,
                target_label=target_label(column, target_contract(config)), spearman=family_associations['correlations'][s, :, t],
                valid_observations=family_associations['valid_observations'][s, t]))
                for s, name in enumerate(family_associations['slides']) for t, column in enumerate(data['target_columns'])
                ]).to_csv(staged/'family_slide_ihc_associations.csv', index=False)
            queue.submit(plots.plot_family_correlations, families, correlations['mean'], family_associations['mean'],
                [target_label(c, target_contract(config)) for c in data['target_columns']],
                staged/'peak_family_ihc_correlations.png', analysis)
            plots.queue_peak_plots(queue, staged, annotations, coordinates, tree, weights, slide_prevalence,
                                   stats['slides'], analysis, target_contract(config))
            if settings['spatial']:
                tissue_path = Path(spatial_input or data['context_path'])
                sources['tissue'] = source_identity(tissue_path)
                progress.set_phase('tissue metadata')
                tissue, tissue_features = read_peak_metadata(tissue_path, data, settings['lesion_column'], spatial=True)
                if not np.array_equal(features, tissue_features):
                    raise ValueError('Tissue feature identities/order differ from checkpoint.')
                spatial_key = dict(common, tissue=sources['tissue'], family_key=family_key,
                    families=fingerprint(families), features=fingerprint(tissue_features),
                    preprocessing={k: data[k] for k in ('nonzero_threshold', 'intensity_transform', 'intensity_clip_max')})
                hit = cache.get('peak-spatial', spatial_key)
                if hit is None:
                    progress.set_phase('resident tissue spectra')
                    matrix = resident_matrix(tissue_path, data, performance)
                    check_host_budget(performance, len(tissue)*len(unique)*8 + settings['block_rows']*len(features)*64)
                    progress.set_phase('measured family scores', len(tissue))
                    scores = family_spatial_scores(matrix, families, data, settings['block_rows'], progress)
                    del matrix
                    gc.collect()
                    cache.put('peak-spatial', spatial_key, scores)
                else:
                    scores = hit[0]
                limits = signal_limits(scores['signal'], settings['signal_percentile'])
                spatial_metadata = dict(family_ids=unique.tolist(), signal_vmin=0., signal_vmax=limits.tolist(),
                    detection_limits=[0., 100.], signal_percentile=settings['signal_percentile'], signal_cmap=settings['sequential_cmap'],
                    detection_cmap=settings['detection_cmap'], sources=sources,
                    preprocessing=spatial_key['preprocessing'], family_key=family_key)
                (staged/'family_spatial_settings.json').write_text(json.dumps(spatial_metadata, indent=2))
                progress.set_phase('spatial parquet and geometry', len(tissue))
                _write_spatial_table(staged/'family_spatial_activity.parquet', tissue, scores, unique, data,
                                     settings['lesion_column'], spatial_metadata)
                slides = spatial_geometry(tissue, data, settings['lesion_column'])
                plots.queue_spatial_plots(queue, staged, tissue_path, slides, scores, limits, unique,
                    analysis, settings, performance)
                transformed, mask = prepare_observed_targets(tissue[data['target_columns']].to_numpy(),
                    data['target_columns'], data.get('cd8_normalization'))
                cd8_index = data['target_columns'].index('Density_CD8')
                cd8 = np.where(mask[:, cd8_index], transformed[:, cd8_index], np.nan)
                plots.queue_cd8_comparisons(comparison_queue, staged, slides, cd8,
                    target_label('Density_CD8', target_contract(config)), scores, limits, unique, analysis, settings)
                del transformed, mask
            progress.details.update(requested_family_count=analysis['family_count'], actual_family_count=len(unique),
                embedding_shape=list(weights.shape), sources=sources, excluded_from_log_plots=int(np.sum(~np.isfinite(mean) | (mean <= 0))),
                family_colors=plots.family_colors(len(unique)), effective_render_workers=execution['render_workers'])
            # Plotting uses only immutable mmap arrays; all numerical pools are finished.
            expected_plots = {arg for _, _, args, _ in [*queue.tasks, *comparison_queue.tasks] for arg in args
                              if isinstance(arg, Path) and arg.parent == staged}
            if not execution['render_workers']:
                largest = max((array.nbytes for array, _ in queue.arrays.values()), default=0)
                check_host_budget(performance, max(2*2**30, 2**30+4*queue.canvas_bytes+3*largest))
            queue.run(execution['render_workers'], performance, progress)
            if comparison_queue.tasks:
                render_details = {key: value for key, value in progress.details.items() if 'render' in key}
                if not execution['render_workers']:
                    largest = max((array.nbytes for array, _ in comparison_queue.arrays.values()), default=0)
                    check_host_budget(performance, 2**30+4*comparison_queue.canvas_bytes+3*largest)
                comparison_queue.run(execution['render_workers'], performance, progress)
                progress.details['comparison_render_workers'] = progress.details.get('effective_render_workers', 0)
                progress.details['comparison_canvas_bytes'] = comparison_queue.canvas_bytes
                progress.details.update(render_details)
            missing = [str(path) for path in expected_plots if not path.is_file()]
            if missing:
                raise ValueError(f'Missing requested peak plots: {missing}')
            progress.set_phase('peak output validation')
            for name, original in sources.items():
                if source_identity(Path(original['path'])) != original:
                    raise ValueError(f'{name} source changed during analysis.')
            files = validate_peak_outputs(staged, weights, families, len(tissue) if tissue is not None else None, scores)
            progress.details['spatial_png_count'] = len(list(staged.glob('family_*_spatial.png')))
            progress.details['comparison_png_count'] = len(list(staged.glob('cd8_families_*.png')))
            progress.details['png_count'] = len(list(staged.glob('*.png')))
            progress.details['html_count'] = len(list(staged.glob('*.html')))
            from dann.artifact_layout import peak_relative_path, remove_owned
            relative_paths = {path: (peak_relative_path(path.name) if path.parent == staged else
                Path('spatial/cd8_comparison')/path.relative_to(staged)) for path in files}
            progress.details['artifacts'] = [str(relative_paths[path]) for path in files]
            (staged/'peak_run.json').write_text(json.dumps(progress.snapshot(torch.device('cpu')), indent=2))
            destination.mkdir(exist_ok=True)
            previous_report = destination/'peak_run.json'
            previous_artifacts = set()
            if previous_report.exists():
                previous_artifacts = set(json.loads(previous_report.read_text()).get('artifacts', []))
            for path, relative in relative_paths.items():
                target = destination/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, target)
            remove_owned(destination, previous_artifacts - {str(p) for p in relative_paths.values()})
            os.replace(staged/'peak_run.json', previous_report)
            for report_name in ('peak_audit.json', 'peak_browser_audit.json'):
                old_report = destination/report_name
                if old_report.exists():
                    reports = destination/'reports'
                    reports.mkdir(exist_ok=True)
                    os.replace(old_report, reports/report_name)
        return destination

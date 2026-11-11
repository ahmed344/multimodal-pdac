"""Controlled latent discovery and CD8-guided views of one frozen checkpoint.

Run ``python -m dann.latent_geometry --config dann/config.yaml --checkpoint ...``.
Existing analysis products are references and are never rewritten by this CLI.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
import torch
from anndata.io import read_elem

from dann.analysis_execution import StageCache, digest_file, fingerprint, load_checkpoint_snapshot
from dann.config import latent_geometry_settings
from dann import geometry as g
from dann.spatial_execution import InferenceProgress


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def source_observations(config: Mapping[str, Any]) -> pd.DataFrame:
    """Read only observation metadata, preserving source-row and target identities."""
    from dann.targets import prepare_targets
    data, analysis = config['data'], config['analysis']
    settings = latent_geometry_settings(analysis.get('latent_geometry'))
    columns = data['target_columns']
    names = dict(slide=analysis.get('slide_column', data['batch_column']),
                 batch=data['batch_column'], lesion=analysis.get('lesion_column', 'Lesion'),
                 x=data['x_column'], y=data['y_column'])
    with h5py.File(data['path'], 'r') as handle:
        obs = handle['obs']
        needed = set(names.values()) | set(columns) | set(settings['technical_columns'])
        missing = needed - set(obs)
        if missing:
            raise ValueError(f'Missing geometry observation columns: {sorted(missing)}')
        values = {key: np.asarray(read_elem(obs[key])) for key in needed}
    frame = pd.DataFrame({key: values[name] for key, name in names.items()})
    for key in ('slide', 'batch'):
        if pd.isna(frame[key]).any() or frame[key].astype(str).str.strip().eq('').any():
            raise ValueError(f'Missing {key} labels.')
        frame[key] = frame[key].astype(str)
    if frame.lesion.isna().any() or not frame.lesion.isin([False, True, 0, 1]).all():
        raise ValueError('Lesion annotations must be nonmissing booleans.')
    frame['lesion'] = frame.lesion.astype(bool)
    if not np.isfinite(frame[['x', 'y']].to_numpy(float)).all():
        raise ValueError('Spatial diagnostics require finite x/y coordinates.')
    raw = np.column_stack([values[name] for name in columns])
    targets, valid, reasons = prepare_targets(raw, columns, data.get('cd8_normalization'))
    for j, name in enumerate(columns):
        frame[name], frame['raw_'+name], frame['valid_'+name] = targets[:, j], raw[:, j], valid[:, j]
    for name in settings['technical_columns']:
        reserved = {'row_id', 'split', 'eligible', 'cd8_exclusion_reason', 'excluded_non_lesion',
                    'excluded_invalid_cd8', 'cd8_presence'}
        if name in frame or name in reserved or name.startswith('cd8_exceedance_'):
            raise ValueError(f'Technical column collides with analysis metadata: {name}')
        frame[name] = np.asarray(values[name], dtype=float)
    frame['row_id'] = np.arange(len(frame), dtype=np.int64)
    frame['cd8_exclusion_reason'] = np.asarray(['valid', 'insufficient_non_tumor_area', 'normalized_ratio_above_one'])[reasons]
    return frame


def validate_splits(splits: Mapping[str, Any], count: int) -> dict[str, np.ndarray]:
    """Require the checkpoint's original, disjoint study splits; never resplit."""
    if set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('Geometry experiments require all three saved checkpoint splits.')
    result = {key: np.asarray(value) for key, value in splits.items()}
    if any(v.ndim != 1 or v.dtype.kind not in 'iu' for v in result.values()):
        raise ValueError('Saved split identities must be integer vectors.')
    combined = np.concatenate(list(result.values()))
    if len(np.unique(combined)) != len(combined) or np.any((combined < 0) | (combined >= count)):
        raise ValueError('Invalid or overlapping checkpoint splits.')
    if any(len(result[key]) < 4 for key in ('train', 'validation')):
        raise ValueError('Training and validation require at least four saved rows.')
    return result


def eligible_samples(extracted: Mapping[str, np.ndarray], observations: pd.DataFrame,
                     split: str, batch_names: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Apply the common eligibility mask after inference and audit separate reasons."""
    rows = np.asarray(extracted['row_ids'])
    if rows.dtype.kind not in 'iu' or rows.ndim != 1 or len(np.unique(rows)) != len(rows):
        raise ValueError('Inference must return unique integer source rows.')
    if np.any((rows < 0) | (rows >= len(observations))):
        raise ValueError('Inference rows outside source observations.')
    frame = observations.iloc[rows].reset_index(drop=True).copy()
    if not np.array_equal(frame.batch.to_numpy(), np.asarray(batch_names)[extracted['batches']]):
        raise ValueError('Inference batch identities disagree with observation source.')
    for key in ('latent', 'pi', 'mu', 'sigma'):
        if len(extracted[key]) != len(rows) or not np.isfinite(extracted[key]).all():
            raise ValueError(f'Nonfinite or misaligned inference {key}.')
    frame['split'] = split
    frame['excluded_non_lesion'] = ~frame.lesion
    frame['excluded_invalid_cd8'] = ~frame.valid_Density_CD8
    frame['eligible'] = ~(frame.excluded_non_lesion | frame.excluded_invalid_cd8)
    audit = frame[['row_id', 'split', 'slide', 'eligible', 'excluded_non_lesion',
                   'excluded_invalid_cd8', 'cd8_exclusion_reason']].copy()
    keep = frame.eligible.to_numpy()
    order = np.argsort(rows[keep])
    selected = {key: value[keep][order] for key, value in extracted.items()
                if isinstance(value, np.ndarray) and value.ndim and len(value) == len(rows)}
    return selected, frame.loc[keep].iloc[order].reset_index(drop=True), audit


def extract_checkpoint(config: dict, checkpoint: dict, checkpoint_path: Path, checkpoint_hash: str,
                       output: Path, progress: InferenceProgress) -> tuple[dict, dict, dict, dict]:
    """Use sequential evaluation loaders for both splits, retaining spatial context."""
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from dann.analyze import extract_latent_predictions, load_model
    from dann.checkpoints import validate_checkpoint, validate_training_data_contract
    from dann.config import execution_settings, resolve_device
    from dann.data_loader import create_data_bundle, load_anndata_metadata, sparse_collate, stratified_cap_indices
    from dann.tiles import row_identity, tile_collate
    from dann.spatial_execution import release_preparation

    settings = latent_geometry_settings(config['analysis'].get('latent_geometry'))
    if not {'Density_CD8', 'Density_Tumor'}.issubset(config['data']['target_columns']):
        raise ValueError('Geometry experiments require CD8 and tumor targets.')
    validate_checkpoint(checkpoint, config)
    validate_training_data_contract(checkpoint, config)
    obs = source_observations(config)
    splits = validate_splits(checkpoint.get('split_indices', {}), len(obs))
    identity = row_identity(Path(config['data']['path']), config['data'])
    if checkpoint.get('data_identity') != identity:
        raise ValueError('Geometry experiments require a matching saved data identity.')
    metadata = load_anndata_metadata(Path(config['data']['path']), config['data']['target_columns'],
        config['data']['batch_column'], config['data']['matrix_key'], config['data'].get('cd8_normalization'))
    if tuple(checkpoint['batch_names']) != metadata.batch_names:
        raise ValueError('Source batch order differs from checkpoint.')
    train_obs = obs.iloc[splits['train']]
    threshold_info = g.training_thresholds(train_obs.loc[train_obs.lesion & train_obs.valid_Density_CD8, 'Density_CD8'].to_numpy(),
                                          settings['threshold_quantiles'], config['loss']['logit_epsilon'])
    threshold_info['source'] = 'all eligible rows in the saved training split, before inference caps'
    population_eligibility = {}
    for split, rows in splits.items():
        part = obs.iloc[rows]
        population_eligibility[split] = dict(rows=len(rows), eligible=int((part.lesion & part.valid_Density_CD8).sum()),
            non_lesion=int((~part.lesion).sum()), invalid_cd8=int((~part.valid_Density_CD8).sum()),
            cd8_reasons={str(k): int(v) for k, v in part.cd8_exclusion_reason.value_counts().items()})
    sources = {}
    for key in ('path', 'context_path'):
        if config['data'].get(key):
            path = Path(config['data'][key]).resolve()
            stat = path.stat()
            sources[key] = [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'scipy', 'scikit-learn', 'umap-learn')}
    source_code = fingerprint({str(p.relative_to(Path(__file__).parent)): digest_file(p)
                               for p in sorted(Path(__file__).parent.rglob('*.py'))})
    common = dict(checkpoint_sha256=checkpoint_hash, sources=sources, identity=fingerprint(identity),
        source_code=source_code, versions=versions,
        config={key: config.get(key) for key in ('data', 'model', 'loss', 'training', 'performance')},
        inference_batch_size=config['analysis']['batch_size'])
    cache = StageCache(output.parent/'.latent-geometry-cache', 'reuse', progress)
    arrays, frames, audits, chosen, keys = {}, {}, [], {}, {}
    runtime = copy.deepcopy(config)
    for name in ('train', 'validation', 'test'):
        runtime['data'][f'max_{name}_samples'] = None
    for index, split in enumerate(('train', 'validation')):
        cap = settings[f'max_{split}_samples']
        chosen[split] = np.sort(stratified_cap_indices(splits[split], metadata.batch_codes, cap, settings['seeds'][0]+index))
        keys[split] = dict(common, split=split, rows=fingerprint(chosen[split]))
        hit = cache.get('geometry-inference-'+split, keys[split])
        if hit is not None:
            arrays[split] = hit[0]
    missing = [name for name in chosen if name not in arrays]
    if missing:
        selected_splits = {**splits, **chosen}
        device = resolve_device(config['training']['device'])
        # The datasets keep all context rows; only supervised output rows are capped.
        bundle = create_data_bundle(runtime, selected_splits, identity, requested_splits=missing, progress=progress)
        try:
            model, _ = load_model(checkpoint_path, config, SimpleNamespace(metadata=metadata), device, checkpoint=checkpoint)
            spatial = execution_settings(config)['mode'] == 'spatial'
            for split in missing:
                progress.set_phase(f'geometry {split} inference')
                loader = DataLoader(bundle.datasets[split], shuffle=False, num_workers=0,
                    batch_size=int(config['training']['validation_batch_size'] if spatial else config['analysis']['batch_size']),
                    collate_fn=tile_collate if spatial else sparse_collate)
                arrays[split] = extract_latent_predictions(model, loader, device, collect_he=False)
                if not np.array_equal(np.sort(arrays[split]['row_ids']), chosen[split]):
                    raise ValueError('Inference omitted or duplicated selected saved-split rows.')
                cache.put('geometry-inference-'+split, keys[split], arrays[split])
            del model
        finally:
            for loader in bundle.loaders.values():
                release_preparation(loader)
            for dataset in bundle.datasets.values():
                if hasattr(dataset, 'close'):
                    dataset.close()
            del bundle
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    for split in chosen:
        if not np.array_equal(np.sort(arrays[split]['row_ids']), chosen[split]):
            raise ValueError('Cached inference row identities differ from requested split.')
        arrays[split], frames[split], audit = eligible_samples(arrays[split], obs, split, list(metadata.batch_names))
        audits.append(audit)
    pd.concat(audits).to_csv(output/'eligibility.csv', index=False)
    provenance = dict(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=checkpoint_hash,
        checkpoint_epoch=int(checkpoint['epoch'])+1, sources=sources, source_identity=fingerprint(identity),
        source_code_sha256=source_code, versions=versions,
        saved_split_counts={k: len(v) for k, v in splits.items()},
        population_eligibility=population_eligibility,
        saved_split_hashes={k: fingerprint(np.sort(v)) for k, v in splits.items()},
        inference_counts={k: len(v) for k, v in chosen.items()},
        eligible_counts={k: len(v) for k, v in frames.items()},
        eligible_row_hashes={k: fingerprint(v.row_id.to_numpy()) for k, v in frames.items()},
        validation_scope='all saved validation rows' if settings['max_validation_samples'] is None else 'capped study validation sample')
    return arrays, frames, threshold_info, provenance


def render_layout(coordinates: np.ndarray, frame: pd.DataFrame, probabilities: np.ndarray,
                  output: Path, title: str, cap: int, seed: int, thresholds: list[float]) -> None:
    """Save static panels and an offline color-selectable viewer with row identities."""
    from matplotlib import pyplot as plt
    import plotly.graph_objects as go
    from dann.html_display import write_plotly
    selected = np.sort(np.random.default_rng(seed).choice(len(frame), min(cap, len(frame)), replace=False))
    xy, part = coordinates[selected], frame.iloc[selected]
    slides, names = pd.factorize(part.slide, sort=True)
    colors = {'Slide (category code)': slides, 'Observed CD8': part.Density_CD8.to_numpy(),
              'Predicted CD8 presence': probabilities[selected, 0]}
    for j, threshold in enumerate(thresholds, 1):
        colors[f'Predicted P(CD8 > {threshold:.8g}) · t{j}'] = probabilities[selected, j]
    for name in ('Density_Tumor', 'Density_Collagen', 'Density_Stroma'):
        if name in part:
            colors[name] = part[name].to_numpy()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    try:
        for axis, (name, values) in zip(axes, colors.items()):
            scatter = axis.scatter(xy[:, 0], xy[:, 1], c=values, s=2, cmap='turbo', rasterized=True)
            axis.set_title(name, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(scatter, ax=axis, shrink=.7)
        fig.suptitle(title, fontsize=10)
        fig.savefig(output/'layout.png', dpi=150)
    finally:
        plt.close(fig)
    figure = go.Figure(go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode='markers',
        text=[f'Row {r} · {s}' for r, s in zip(part.row_id, part.slide)],
        hovertemplate='%{text}<extra></extra>', marker=dict(size=3, color=slides, colorscale='Turbo', showscale=True)))
    buttons = [dict(label=name, method='restyle', args=[{'marker.color': [values.tolist()],
               'marker.colorbar.title.text': name}]) for name, values in colors.items()]
    figure.update_layout(title=title+f' · displaying {len(selected):,}/{len(frame):,} held-out rows',
                         template='plotly_white', height=800, updatemenus=[dict(buttons=buttons)])
    write_plotly(figure, output/'layout.html')
    _write_json(output/'slide_colors.json', dict(enumerate(names.tolist())))


def run_experiments(arrays: Mapping[str, dict], frames: Mapping[str, pd.DataFrame],
                    threshold_info: dict, settings: dict, output: Path,
                    target_columns: list[str], epsilon: float) -> dict:
    """Fit training transforms once, compare fixed rows, and publish separate views."""
    train_frame, valid_frame = frames['train'], frames['validation']
    train, valid = arrays['train']['latent'], arrays['validation']['latent']
    if min(len(train), len(valid)) < 4:
        raise ValueError('At least four eligible rows per split are required.')
    if set(train_frame.row_id) & set(valid_frame.row_id):
        raise ValueError('Training and validation row identities overlap.')
    fitted = g.fit_preprocessing(train)
    np.savez(output/'preprocessing.npz', **fitted, training_row_ids=train_frame.row_id.to_numpy())
    _write_json(output/'thresholds.json', threshold_info)
    thresholds = threshold_info['thresholds']
    index = target_columns.index('Density_CD8')
    probabilities = {split: g.probability_block(data['pi'][:, index], data['mu'][:, index], data['sigma'][:, index], thresholds, epsilon)
                     for split, data in arrays.items()}
    for split, frame in frames.items():
        table = frame.copy()
        for j in range(probabilities[split].shape[1]):
            table['cd8_presence' if j == 0 else f'cd8_exceedance_{j}'] = probabilities[split][:, j]
        table.to_csv(output/f'{split}_observations.csv', index=False)
    calibration, reliability = g.calibration_tables(valid_frame, probabilities['validation'], thresholds)
    calibration.to_csv(output/'calibration.csv', index=False)
    reliability.to_csv(output/'reliability.csv', index=False)
    from dann.calibration import compute_target_metrics
    metrics = compute_target_metrics(valid_frame.Density_CD8.to_numpy(), arrays['validation']['pi'][:, index],
                                     arrays['validation']['mu'][:, index], arrays['validation']['sigma'][:, index], epsilon)
    _write_json(output/'distribution_metrics.json', {k: v if np.isfinite(v) else None for k, v in metrics.items()})
    for split in frames:
        g.association_diagnostics(arrays[split]['latent'], frames[split], fitted,
            list(dict.fromkeys(target_columns+settings['technical_columns']))).to_csv(output/f'{split}_associations.csv', index=False)
    # Full eligible slide summaries precede diagnostic downsampling. Components are
    # graph components, never promoted to biological states or state proportions.
    valid_frame.groupby('slide')[target_columns].agg(['count', 'mean', 'median']).to_csv(output/'slide_observed_summary.csv')
    seed = settings['seeds'][0]
    panels = g.sample_panels(valid_frame, settings, seed)
    train_rows = np.sort(np.random.default_rng(seed).choice(len(train), min(len(train), settings['diagnostic_max_samples']), replace=False))
    np.savez(output/'diagnostic_rows.npz', **{k: valid_frame.row_id.to_numpy()[v] for k, v in panels.items()},
             training_reference=train_frame.row_id.to_numpy()[train_rows])
    tumor_edges = np.unique(np.quantile(train_frame.Density_Tumor, [.25, .5, .75]))
    _write_json(output/'tumor_strata.json', dict(training_quartile_edges=tumor_edges.tolist()))
    baseline = {}
    for panel, rows in panels.items():
        if len(rows) >= 2:
            for k in settings['n_neighbors']:
                baseline[panel, k] = g.neighbors(valid[rows], k, 'euclidean')[0]
    report, probe_tables, slide_tables, stability, skipped = [], [], [], [], []
    selected_geometry = None
    probability_scale = g.distance_scale(probabilities['train'], settings['distance_pairs'], seed)
    layouts = {}

    def evaluate(name, training, validation, metric, info, view, weight=None, reference=None):
        folder = output/name
        folder.mkdir()
        _write_json(folder/'settings.json', dict(info, view=view, cd8_weight=weight,
            interpretation='IHC-supervised latent; no additional CD8 guidance' if view == 'discovery' else 'predicted CD8 guidance; smooth CD8 is intended, not validation',
            layout_fit='training only; validation uses UMAP.transform'))
        np.savez(folder/'representation.npz', row_ids=valid_frame.row_id.to_numpy(), values=validation)
        for panel, rows in panels.items():
            if len(rows) < 2:
                skipped.append(dict(arm=name, panel=panel, reason='fewer than two sampled rows'))
                continue
            for k in settings['n_neighbors']:
                diagnostic, per_row, ids = g.graph_diagnostics(validation[rows], valid_frame.iloc[rows], k, metric,
                                                               target_columns, baseline[panel, k])
                report.append(dict(arm=name, view=view, metric=metric, panel=panel, requested_neighbors=k, **diagnostic))
                by_slide = per_row.groupby('slide')[['same_slide', 'chance', 'excess', 'enrichment']].mean().reset_index()
                by_slide['arm'], by_slide['panel'], by_slide['neighbors'] = name, panel, k
                slide_tables.append(by_slide)
                np.savez_compressed(folder/f'neighbors_{panel}_{k}.npz', row_ids=valid_frame.row_id.to_numpy()[rows], neighbors=ids,
                                    components=per_row.component.to_numpy())
        rows = panels['representative']
        for k in settings['n_neighbors']:
            probes = g.heldout_probes(training[train_rows], validation[rows], train_frame.iloc[train_rows],
                valid_frame.iloc[rows], target_columns, thresholds, k, metric, tumor_edges)
            probes['arm'], probes['requested_neighbors'] = name, k
            probe_tables.append(probes)
            first_neighbors = None
            for layout_seed in settings['seeds'] if settings['layouts'] else []:
                print(f'Geometry: {name}, neighbors={k}, seed={layout_seed}', flush=True)
                key = (name, k, layout_seed)
                if reference is not None and (reference, k, layout_seed) in layouts:
                    coordinates = layouts[reference, k, layout_seed]
                else:
                    from umap import UMAP
                    model = UMAP(n_neighbors=min(k, len(training)-1), min_dist=settings['min_dist'],
                        metric=metric, random_state=layout_seed, transform_seed=layout_seed, n_jobs=1,
                        init='spectral')
                    model.fit(training)
                    coordinates = model.transform(validation)
                if name == settings['guided_geometry']:
                    layouts[key] = coordinates
                if not np.isfinite(coordinates).all():
                    raise ValueError('Nonfinite UMAP layout.')
                layout_folder = folder/f'umap_k{k}_seed{layout_seed}'
                layout_folder.mkdir()
                _write_json(layout_folder/'settings.json', dict(requested_neighbors=k,
                    effective_neighbors=min(k, len(training)-1), min_dist=settings['min_dist'], metric=metric,
                    random_state=layout_seed, transform_seed=layout_seed, n_jobs=1, init='spectral',
                    spread=1., n_components=2, fit_split='train', fit_rows=len(training), transform_rows=len(validation),
                    reused_from=reference if reference is not None else None))
                pd.DataFrame(dict(row_id=valid_frame.row_id, umap_1=coordinates[:, 0], umap_2=coordinates[:, 1])).to_csv(layout_folder/'coordinates.csv', index=False)
                render_layout(coordinates, valid_frame, probabilities['validation'], layout_folder,
                    f'{"Discovery" if view == "discovery" else "CD8-guided"}: {name} · k={k} · seed={layout_seed}',
                    settings['plot_max_samples'], seed, thresholds)
                ids, _ = g.neighbors(coordinates[rows], k, 'euclidean')
                higher_ids, _ = g.neighbors(validation[rows], k, metric)
                stability.append(dict(arm=name, neighbors=k, seed=layout_seed,
                    layout_neighbor_preservation=g.neighbor_overlap(ids, higher_ids),
                    layout_overlap_with_first_seed=g.neighbor_overlap(ids, first_neighbors) if first_neighbors is not None else None))
                if first_neighbors is None:
                    first_neighbors = ids

    for name in settings['geometries']:
        print(f'Geometry: diagnostics for {name}', flush=True)
        try:
            a, metric, info = g.transform_geometry(train, fitted, name, settings)
            b, _, _ = g.transform_geometry(valid, fitted, name, settings)
        except ValueError as error:
            skipped.append(dict(arm=name, reason=str(error)))
            continue
        evaluate(name, a, b, metric, info, 'discovery')
        if name == settings['guided_geometry']:
            selected_geometry = a, b, info
    if selected_geometry is not None:
        a, b, info = selected_geometry
        latent_scale = g.distance_scale(a, settings['distance_pairs'], seed)
        for weight in settings['cd8_weights']:
            name = 'cd8_weight_'+str(weight).replace('.', 'p')
            try:
                train_guided = g.guided_geometry(a, probabilities['train'], weight, latent_scale, probability_scale)
                valid_guided = g.guided_geometry(b, probabilities['validation'], weight, latent_scale, probability_scale)
            except ValueError as error:
                skipped.append(dict(arm=name, reason=str(error)))
                continue
            evaluate(name, train_guided, valid_guided, 'euclidean', dict(info,
                latent_distance_scale=latent_scale, probability_distance_scale=probability_scale,
                scale_fit='RMS seeded distinct training-row pair distances',
                normalized_squared_distance='(1-w)*d_latent^2/s_latent^2 + w*d_probability^2/s_probability^2'),
                'cd8_guided', weight, settings['guided_geometry'] if weight == 0 else None)
    else:
        skipped.append(dict(arm='cd8_guided', reason='selected discovery geometry unavailable'))
    pd.DataFrame(report).to_csv(output/'neighborhood_diagnostics.csv', index=False)
    if probe_tables:
        pd.concat(probe_tables).to_csv(output/'heldout_probes.csv', index=False)
        pd.concat(slide_tables).to_csv(output/'slide_neighborhood_diagnostics.csv', index=False)
    pd.DataFrame(stability).to_csv(output/'layout_stability.csv', index=False)
    _write_json(output/'skipped.json', skipped)
    summary = dict(completed_arms=sorted(set(r['arm'] for r in report)), skipped=skipped,
        diagnostic_panel_counts={k: len(v) for k, v in panels.items()},
        training_reference_rows=len(train_rows),
        boundaries=['Discovery starts from an IHC-supervised representation.',
            'CD8-guided geometry summarizes checkpoint predictions; it does not validate biology.',
            'Calibration is evaluated once and is unchanged by geometry or layout.',
            'Within-slide study validation retains spatial dependence; it is not independent-patient validation.',
            'Mixing, layout smoothness and connected components do not define biological states.',
            'Sampling and resampling diagnostics do not estimate full-data state proportions.',
            'MNN augmentation and alternative layouts require review of these diagnostics first.'])
    _write_json(output/'summary.json', summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run new analysis artifacts without modifying checkpoints or reference outputs."""
    from dann.checkpoints import checkpoint_config
    from dann.config import load_config, seed_everything
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, help='New output directory; existing nonempty directories are rejected.')
    parser.add_argument('--smoke-test', action='store_true', help='Cap inference to 256 training / 192 validation rows and diagnostic panels to 160.')
    parser.add_argument('--diagnostics-only', action='store_true', help='Skip UMAP fits and rendering.')
    args = parser.parse_args(argv)
    runtime = load_config(args.config)
    path = args.checkpoint or Path(runtime['analysis'].get('checkpoint')
                                   or Path(runtime['training']['output_dir'])/'best.pt')
    checkpoint, digest = load_checkpoint_snapshot(path)
    config = checkpoint_config(checkpoint)
    config['analysis'] = copy.deepcopy(runtime['analysis'])
    # Only source locations and execution preferences come from the working YAML.
    # Architecture, targets, preprocessing, split and training contracts stay frozen.
    for key in ('path', 'context_path'):
        if key in runtime['data']:
            config['data'][key] = runtime['data'][key]
    config['training']['device'] = runtime['training']['device']
    settings = latent_geometry_settings(config['analysis'].get('latent_geometry'))
    if args.smoke_test:
        settings.update(max_train_samples=256, max_validation_samples=192, diagnostic_max_samples=160,
                        plot_max_samples=192, resamples=1, distance_pairs=1000, seeds=settings['seeds'][:1])
        # A capped read should not populate full-spectrum/H&E RAM caches.
        config.setdefault('performance', {}).update(ram_cache=False, he_ram_cache=False, torch_threads=4,
            preparation_workers=0, validation_feature_cache=False, cache_geometry=False)
    if args.diagnostics_only:
        settings['layouts'] = False
    config['analysis']['latent_geometry'] = settings
    seed_everything(settings['seeds'][0])
    output = args.output or path.parent.parent/'analysis'/'latent_geometry'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Use a new output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output/'settings.json', settings)
    with InferenceProgress('Latent geometry') as progress:
        arrays, frames, thresholds, provenance = extract_checkpoint(config, checkpoint, path, digest, output, progress)
        provenance['smoke_test'] = args.smoke_test
        _write_json(output/'provenance.json', provenance)
        run_experiments(arrays, frames, thresholds, settings, output, config['data']['target_columns'], config['loss']['logit_epsilon'])
    print(f'Geometry experiments saved to {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

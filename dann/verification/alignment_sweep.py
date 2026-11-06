"""Restartable, sequential alignment study with matched slide-mixing selection.

python -m dann.verification.alignment_sweep --output NEW_DIR --prepare
python -m dann.verification.alignment_sweep --output NEW_DIR --run
The runner freezes its Python sources and uses that snapshot in subprocesses.
Study recommendations never replace supervised-selected best.pt checkpoints.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import yaml

from dann.analysis_execution import digest_file, fingerprint
from dann.batch_alignment import NAMES, alignment_settings, evaluate_alignment
from dann.checkpoints import architecture_contract, checkpoint_config, validate_checkpoint
from dann.config import load_config, resolve_execution, seed_everything
from dann.data_loader import create_data_bundle, stratified_cap_indices
from dann.initialization import initialize_supervised_heads
from dann.model import AdversarialLatentFusion
from dann.spatial import _file_identity
from dann.targets import target_contract
from dann.tiles import row_identity
from dann.train import train_model
from dann.training_artifacts import atomic_json
from dann.verification.alignment_study_metrics import (
    LIMITS, SEED, SEEDS, balanced_rows, choose, eligibility, geometry, improves, variance_spectrum,
)
from dann.verification.dictionary_comparison import close, immutable_json, parameter_hash
from dann.verification.pooling_diagnostics import json_clean, save_json

BASELINE = Path('data/PDAC/Results/dann/deep_sets__agg-mlp__bio-mlp__disc-none__he-mlp/'
                'he10_biology2_nhsic0p1_512_occurrence_mlp_count_zero_input_cleaned/model/best.pt')
INITIAL_ARMS = dict(control=(0., 0., 0.), nhsic1=(1., 0., 0.),
                    nhsic10=(10., 0., 0.), nhsic100=(100., 0., 0.))


def make_config(source: dict, root: Path, arm: str, weights: tuple) -> dict:
    """Change alignment weights and output controls only; keep scientific settings."""
    c = copy.deepcopy(source)
    for key in ('monitoring_definitions', 'validation_alignment_sample'):
        c.pop(key, None)
    c['results'].update(root=str(root/'runs'), run_name=arm, smoke=False)
    c['training'].update(archive_checkpoint_every=5, resume_checkpoint=None)
    c['post_training']['enabled'] = False
    for name, weight in zip(NAMES, weights):
        c['loss']['batch_alignment'][name] = dict(enabled=True, weight=weight)
    return resolve_execution(c)


def register_arm(root: Path, arm: str, weights: tuple) -> dict:
    """Add an immutable adaptive arm after its base decision has been recorded."""
    source = yaml.safe_load((root/'source_config.yaml').read_text())
    c = make_config(source, root, arm, weights)
    path = root/'configs'/f'{arm}.yaml'
    text = yaml.safe_dump(c, sort_keys=False)
    record_path = root/'arms'/f'{arm}.json'
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Previously registered arm changed: {arm}')
    else:
        with path.open('x') as handle:
            handle.write(text)
    record = dict(arm=arm, weights=list(weights), config=str(path), config_sha256=digest_file(path),
                  directory=c['results']['run_dir'])
    if record_path.exists():
        if json.loads(record_path.read_text()) != record:
            raise ValueError('Arm identity changed.')
    else:
        immutable_json(record_path, record)
    return record


def prepare(root: Path, source: Path, baseline_path: Path) -> None:
    """Freeze data, rows, thresholds, initialization, sources and comparison rules."""
    c = load_config(source)
    baseline_path = baseline_path.resolve()
    baseline = torch.load(baseline_path, map_location='cpu', weights_only=False)
    saved = checkpoint_config(baseline)
    if architecture_contract(c) != architecture_contract(saved) or c['data'] != saved['data']:
        raise ValueError('Source architecture/preprocessing/data settings differ from reference.')
    if c['loss'] != saved['loss'] or target_contract(c) != baseline['target_transform']:
        raise ValueError('Source loss or target contract differs from reference.')
    if (len(baseline['batch_names']) != 40 or c['model']['latent_dim'] != 512 or
            c['training']['seed'] != SEED or c['training']['epochs'] != 300 or
            c['training']['early_stopping_metric'] != 'supervised_objective' or
            c['model']['heads']['discriminator']['enabled'] or
            c['loss']['biology_weight'] != 2 or c['loss']['he_weight'] != 10 or
            c['data']['train_fraction'] != .9 or c['data']['validation_fraction'] != .1):
        raise ValueError('Study requires the specified 40-slide 512-MLP reference settings.')
    ignored = {'output_dir', 'resume_checkpoint', 'archive_checkpoint_every'}
    if {k:v for k,v in c['training'].items() if k not in ignored} != {
            k:v for k,v in saved['training'].items() if k not in ignored}:
        raise ValueError('Training settings differ from reference.')
    root.mkdir(parents=True, exist_ok=False)
    for child in ('configs', 'arms', 'stages', 'logs', 'diagnostics', 'decisions', 'source'):
        (root/child).mkdir()
    c['post_training']['enabled'] = False
    # Head initialization uses the same cached reduction path as production.
    bundle = create_data_bundle(c)
    try:
        if row_identity(Path(c['data']['path']), c['data']) != baseline['data_identity']:
            raise ValueError('Reference row/feature identity changed.')
        for name, rows in bundle.split_indices.items():
            np.testing.assert_array_equal(rows, baseline['split_indices'][name])
        if tuple(bundle.metadata.batch_names) != tuple(baseline['batch_names']):
            raise ValueError('Slide order changed.')
        np.savez_compressed(root/'splits.npz', **bundle.split_indices)
        tr, va = (bundle.split_indices[k] for k in ('train', 'validation'))
        m = bundle.metadata
        if list(c['data']['target_columns']) != ['Density_CD8','Density_Tumor','Density_Collagen','Density_Stroma']:
            raise ValueError('Existing fixed probe protocol requires its original target order.')
        cd8 = m.targets[tr, 0]
        thresholds = np.quantile(cd8[m.target_valid_mask[tr, 0] & (cd8 > 0)], [.25,.5,.75]).tolist()
        if len(set(thresholds)) != 3:
            raise ValueError('Training-positive CD8 quartiles are not distinct.')
        quartiles = np.quantile(m.targets[tr, 3][m.target_valid_mask[tr, 3]], [.25,.5,.75]).tolist()
        probe = stratified_cap_indices(tr, m.batch_codes, 65536, SEED)
        panel = balanced_rows(va, m.batch_codes[va], 512, SEED, required=64)
        samples = {str(seed):balanced_rows(panel, m.batch_codes[panel], 64, seed, required=64) for seed in SEEDS}
        np.save(root/'probe_train_rows.npy', probe)
        np.save(root/'panel_rows.npy', panel)
        np.savez(root/'samples.npz', **samples)
        pd.DataFrame(dict(row_id=va, slide=m.batch_codes[va],
                         stroma_quartile=np.where(m.target_valid_mask[va,3], np.digitize(m.targets[va,3],quartiles)+1, 0))).to_csv(root/'validation_rows.csv', index=False)
        c['analysis']['exceedance_thresholds'] = thresholds
        (root/'source_config.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
        seed_everything(SEED, c['training']['deterministic_algorithms'])
        model = AdversarialLatentFusion.from_config(c, 40, 4)
        initial = parameter_hash(model)
        stats = initialize_supervised_heads(model, bundle, c)
        if stats != baseline['head_initialization']:
            raise ValueError('Training-only head initialization differs from reference.')
        initialized = parameter_hash(model)
        del model
        source_root = Path(__file__).resolve().parents[2]
        sources = {}
        for path in sorted((source_root/'dann').rglob('*.py')):
            relative = path.relative_to(source_root)
            destination = root/'source'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            sources[str(relative)] = digest_file(path)
        frozen_files = ['splits.npz','probe_train_rows.npy','panel_rows.npy','samples.npz',
                        'validation_rows.csv','source_config.yaml']
        manifest = dict(version=1, created_utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
            baseline=dict(checkpoint=str(baseline_path), sha256=digest_file(baseline_path), epoch=baseline['epoch']+1),
            inputs={k:_file_identity(Path(c['data'][k])) for k in ('path','context_path')},
            data_identity=baseline['data_identity'], batch_names=list(baseline['batch_names']),
            splits={k:dict(rows=len(v),sha256=fingerprint(v)) for k,v in bundle.split_indices.items()},
            target_transform=baseline['target_transform'], cd8_thresholds=thresholds, stroma_quartiles=quartiles,
            initial_parameters_sha256=initial, initialized_heads_sha256=initialized,
            head_initialization=stats, sources=sources,
            files={f:digest_file(root/f) for f in frozen_files}, limits=LIMITS,
            protocol=dict(diagnostic_seeds=SEEDS, panel_per_slide=len(panel)//40, neighbors=30,
                samples='five independent balanced 64/slide draws from the fixed random validation panel',
                chance='same-slide count excluding self / (sample size - 1); excess = observed - chance',
                stroma='training quartile cut points; neighbors recomputed within each sampled quartile',
                umap=dict(metric='cosine', n_neighbors=30, min_dist=.3, random_state=SEED),
                collapse='raw and normalized variance > 1e-8, numerical ranks >= 2 (eigenvalue > max(1e-12, largest*1e-8)), all norms > 1e-8',
                loss_limit='candidate <= reference + 0.10 * abs(reference), including negative losses',
                selection='eligible and both mixing excesses improve; latent ties within 0.01 use UMAP then CD8 loss',
                uncertainty='single training seed; diagnostic samples are not training replications; validation-selected study',
                transfer='H&E-mediated transfer is a separate assessment'),
            software=dict(python=sys.version,torch=str(torch.__version__),gpu=torch.cuda.get_device_name()))
        immutable_json(root/'manifest.json', json_clean(manifest))
        for arm, weights in INITIAL_ARMS.items():
            register_arm(root, arm, weights)
    finally:
        close(bundle)
    print(f'Prepared immutable experiment at {root}', flush=True)


def manifest_for(root: Path) -> dict:
    """Fail closed on changed source snapshots, input files or frozen identities."""
    m = json.loads((root/'manifest.json').read_text())
    for relative, digest in m['sources'].items():
        if digest_file(root/'source'/relative) != digest:
            raise ValueError(f'Frozen source changed: {relative}')
    for relative, digest in m['files'].items():
        if digest_file(root/relative) != digest:
            raise ValueError(f'Frozen study file changed: {relative}')
    if digest_file(Path(m['baseline']['checkpoint'])) != m['baseline']['sha256']:
        raise ValueError('Reference checkpoint changed.')
    c = yaml.safe_load((root/'source_config.yaml').read_text())
    for key, identity in m['inputs'].items():
        if _file_identity(Path(c['data'][key])) != identity:
            raise ValueError(f'Study input changed: {key}')
    for path in (root/'arms').glob('*.json'):
        record = json.loads(path.read_text())
        if digest_file(Path(record['config'])) != record['config_sha256']:
            raise ValueError(f'Arm configuration changed: {path.stem}')
    return m


def config_for(root: Path, arm: str) -> dict:
    """Load only registered configurations; reference uses a separate output tree."""
    if arm == 'reference':
        return make_config(yaml.safe_load((root/'source_config.yaml').read_text()), root, arm, (.1,0.,0.))
    return load_config(root/'configs'/f'{arm}.yaml')


def checkpoint_paths(root: Path, arm: str, m: dict) -> list[Path]:
    """Include every archive and both live selections, deduplicating exact epochs."""
    if arm == 'reference':
        return [Path(m['baseline']['checkpoint'])]
    directory = Path(config_for(root, arm)['training']['output_dir'])
    paths = [directory/'best.pt', directory/'latest.pt', *sorted((directory/'archives').glob('epoch_*.pt'))]
    seen, result = {}, []
    for path in paths:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        epoch = checkpoint['epoch'] + 1
        state_hash = fingerprint({k:fingerprint(v.numpy()) for k,v in checkpoint['model_state'].items()})
        if epoch in seen and seen[epoch] != state_hash:
            raise ValueError(f'Different model states claim epoch {epoch}.')
        if epoch not in seen:
            result.append(path)
        seen[epoch] = state_hash
    return sorted(result, key=lambda p: torch.load(p, map_location='cpu', weights_only=False)['epoch'])


def validate_fit(root: Path, arm: str, m: dict) -> dict:
    """Require completed stopping, full coverage, finite weights and saved RNG."""
    c = config_for(root, arm)
    directory = Path(c['training']['output_dir'])
    latest = torch.load(directory/'latest.pt', map_location='cpu', weights_only=False)
    if not (latest['epoch']+1 == c['training']['epochs'] or
            latest['early_stopping_state']['stale_epochs'] >= c['training']['early_stopping_patience']):
        raise ValueError('Training has not reached the configured stopping condition.')
    for path in checkpoint_paths(root, arm, m):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        validate_checkpoint(checkpoint, c)
        validate_identity(checkpoint, m)
        if not checkpoint.get('training_rng_state'):
            raise ValueError('Study checkpoints require RNG state for restart.')
        if not all(torch.isfinite(v).all() for v in checkpoint['model_state'].values()):
            raise ValueError('Nonfinite checkpoint weights.')
    expected = set(range(5, latest['epoch']+2, 5))
    actual = {int(p.stem.split('_')[-1]) for p in (directory/'archives').glob('epoch_*.pt')}
    if actual != expected:
        raise ValueError('Missing or unexpected archived epochs.')
    coverage = []
    for epoch in range(latest['epoch']+1):
        path = directory/'sampling'/f'epoch_{epoch:04d}_summary.json'
        record = json.loads(path.read_text())
        if not record['complete'] or record['processed_rows'] != m['splits']['train']['rows']:
            raise ValueError('Training did not cover every frozen training row.')
        reference = Path(m['baseline']['checkpoint']).parent/'sampling'/path.name
        if reference.exists():
            historical = json.loads(reference.read_text())
            if any(record[k] != historical[k] for k in ('row_order_sha256','boundaries_sha256')):
                raise ValueError('Training sample order differs from reference.')
            coverage.append(epoch+1)
    best = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
    return dict(stopped_epoch=latest['epoch']+1, supervised_selected_epoch=best['epoch']+1,
                compared_reference_sampling_epochs=coverage,
                files=[str(directory/'best.pt'),str(directory/'latest.pt'),
                       *map(str, sorted((directory/'archives').glob('epoch_*.pt')))])


def validate_identity(checkpoint: dict, m: dict) -> None:
    """Verify exact split, feature/row order, target transform and slide ordering."""
    if checkpoint['data_identity'] != m['data_identity'] or checkpoint['target_transform'] != m['target_transform']:
        raise ValueError('Checkpoint source/target identity mismatch.')
    if list(checkpoint['batch_names']) != m['batch_names']:
        raise ValueError('Checkpoint slide order mismatch.')
    if set(checkpoint['split_indices']) != set(m['splits']):
        raise ValueError('Checkpoint split names mismatch.')
    for key, rows in checkpoint['split_indices'].items():
        if fingerprint(rows) != m['splits'][key]['sha256']:
            raise ValueError('Checkpoint split identity mismatch.')


def training_observer(root: Path, arm: str, m: dict):
    """Assert initialization and split contracts without consuming training RNG."""
    def observe(event, model, bundle, config, record):
        if event == 'parameters_initialized':
            if parameter_hash(model) != m['initial_parameters_sha256']:
                raise ValueError('Initial parameters differ between arms.')
            for key, rows in bundle.split_indices.items():
                if fingerprint(rows) != m['splits'][key]['sha256']:
                    raise ValueError('Fresh split differs from reference.')
        elif event == 'heads_initialized':
            if parameter_hash(model) != m['initialized_heads_sha256'] or record['head_initialization'] != m['head_initialization']:
                raise ValueError('Initialized supervised heads differ between arms.')
            atomic_json(root/'diagnostics'/f'{arm}_initialization.json', dict(verified=True, **record))
    return observe


def restore_row_order(arrays: dict, rows: np.ndarray) -> dict:
    """Reject duplicate, missing or extra core identities and restore frozen order."""
    rows = np.asarray(rows)
    actual = np.asarray(arrays['rows'])
    if len(np.unique(rows)) != len(rows) or len(np.unique(actual)) != len(actual):
        raise ValueError('Duplicate requested or extracted row identities.')
    order = pd.Index(actual).get_indexer(rows)
    if len(actual) != len(rows) or np.any(order < 0):
        raise ValueError('Missing or unexpected extracted row identities.')
    if any(len(value) != len(actual) for value in arrays.values()):
        raise ValueError('Extracted outputs do not align with core identities.')
    return {key:value[order] for key,value in arrays.items()}


def extract_rows(model, c: dict, bundle, rows: np.ndarray, *, he: bool = False) -> tuple[dict, dict]:
    """Extract pixel or spatial core outputs with lesion context and frozen order.

    H&E targets are never read when the head is disabled. Spatial context uses
    the normal supervised tile dataset; only requested core rows emit outputs.
    """
    from torch.utils.data import DataLoader
    from dann.config import execution_settings
    from dann.data_loader import SparseAnnDataDataset, sparse_collate
    from dann.execution import PreparedLoader
    from dann.he import he_settings
    from dann.tiles import SpatialTileDataset, tile_collate
    from dann.train import move_batch_to_device, _update_he_statistics
    enabled = bool(he_settings(c))
    collect_he = he and enabled
    source = next(iter(bundle.datasets.values()))
    resident = getattr(source, 'resident', None)
    reader = bundle.he_reader if collect_he else None
    if execution_settings(c)['mode'] == 'spatial':
        ds = SpatialTileDataset(c, rows, bundle.metadata, resident=resident,
            geometry=getattr(source, 'geometry', None), he_reader=reader)
        collate = tile_collate
        batch_size = c['performance'].get('validation_tiles') or c['training']['tiles_per_batch']
    else:
        ds = SparseAnnDataDataset(Path(c['data']['path']), rows, bundle.metadata,
            **{k:c['data'][k] for k in ('matrix_key','intensity_transform','intensity_clip_max','nonzero_threshold')},
            resident=resident, he_reader=reader)
        collate, batch_size = sparse_collate, 2048
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                        generator=torch.Generator().manual_seed(SEED))
    loader = PreparedLoader(loader, c['performance'], c['model'], validation=True)
    collected = {k:[] for k in ('rows','latent','pi_logits','mu','sigma')}
    device = next(model.parameters()).device
    statistics = {k:torch.zeros((),dtype=torch.float64,device=device) for k in ('he_sum','he_count','he_rows')}
    try:
        with torch.inference_mode():
            for cpu in loader:
                batch = move_batch_to_device(cpu, device)
                output = model(batch, grl_strength=0., predict_he=collect_he)
                if not all(torch.isfinite(v).all() for v in output.values()):
                    raise FloatingPointError('Nonfinite evaluation outputs.')
                collected['rows'].append(batch['row_ids'].cpu().numpy())
                for key in ('latent','pi_logits','mu','sigma'):
                    collected[key].append(output[key].cpu().numpy())
                if collect_he:
                    _update_he_statistics(statistics, output['he_prediction'], batch['he_targets'])
    finally:
        ds.close()
    arrays = restore_row_order({k:np.concatenate(v) for k,v in collected.items()}, rows)
    he_metrics = dict(available=False, reason='head_disabled' if not enabled else 'not_requested')
    if collect_he:
        he_metrics = dict(available=True, mse=float(statistics['he_sum']/statistics['he_count']),
                          r2=float(1-statistics['he_sum']/statistics['he_target_m2'].sum()))
    return arrays, he_metrics


def fit_umap(latent: np.ndarray) -> np.ndarray:
    """Use the unchanged study UMAP settings with a deterministic single worker."""
    from umap import UMAP
    return UMAP(n_neighbors=30, min_dist=.3, metric='cosine', random_state=SEED,
                n_jobs=1).fit_transform(latent)


def plot_umap(coords: np.ndarray, rows: np.ndarray, data: dict, metadata, m: dict,
              destination: Path, title: str) -> None:
    """Fixed slide colors and common probability scales for every checkpoint."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from dann.verification.pooling_diagnostics import probability_arrays
    order = pd.Index(data['rows']).get_indexer(rows)
    if (order < 0).any():
        raise ValueError('UMAP rows missing from extracted validation outputs.')
    selected = {k:v[order] for k,v in data.items()}
    presence, exceeds = probability_arrays(selected, m['cd8_thresholds'], 1e-5)
    slide_colors = [*plt.get_cmap('tab20').colors, *plt.get_cmap('tab20b').colors]
    y = metadata.targets[rows].astype(float)
    y[~metadata.target_valid_mask[rows]] = np.nan
    panels = [('Slide', metadata.batch_codes[rows], ListedColormap(slide_colors), -.5,39.5)]
    panels += [(f'Observed {name}',y[:,i],'viridis',0,1) for i,name in enumerate(('CD8','tumor','collagen','stroma'))]
    panels += [('Observed CD8 presence',np.where(np.isfinite(y[:,0]),(y[:,0]>0).astype(float),np.nan),'viridis',0,1),
               ('Predicted CD8 presence',presence,'viridis',0,1)]
    panels += [(f'P(CD8 > {t:.4g})',exceeds[:,i],'viridis',0,1) for i,t in enumerate(m['cd8_thresholds'])]
    fig, axes = plt.subplots(2,5,figsize=(22,9),layout='constrained')
    for ax,(name,values,cmap,low,high) in zip(axes.flat,panels):
        points = ax.scatter(coords[:,0],coords[:,1],c=values,s=.5,rasterized=True,
                            cmap=cmap,vmin=low,vmax=high,alpha=.65,linewidths=0)
        ax.set(title=name,xticks=[],yticks=[])
        if name != 'Slide':
            fig.colorbar(points,ax=ax,shrink=.65)
    fig.suptitle(title)
    fig.savefig(destination,dpi=160)
    plt.close(fig)
    colors = {name:matplotlib.colors.to_hex(slide_colors[i]) for i,name in enumerate(m['batch_names'])}
    atomic_json(destination.with_suffix('.colors.json'),colors)


def mixing_diagnostics(data: dict, metadata, root: Path, m: dict, out: Path) -> dict:
    """Score five matched draws and recompute neighbors within stromal strata."""
    panel = np.load(root/'panel_rows.npy')
    positions = pd.Index(data['rows']).get_indexer(panel)
    if np.any(positions < 0):
        raise ValueError('Evaluation did not cover the frozen panel.')
    latent = data['latent'][positions]
    coords = fit_umap(latent)
    if not np.isfinite(coords).all():
        raise FloatingPointError('Nonfinite UMAP coordinates.')
    np.savez_compressed(out/'panel.npz', rows=panel, umap=coords)
    plot_umap(coords,panel,data,metadata,m,out/'panel.png',out.name)
    summaries, per_slide, diagnostics = [], [], []
    settings = alignment_settings(yaml.safe_load((root/'source_config.yaml').read_text()))
    with np.load(root/'samples.npz') as samples:
        for seed in SEEDS:
            rows = samples[str(seed)]
            index = pd.Index(panel).get_indexer(rows)
            labels = metadata.batch_codes[rows]
            quartiles = np.digitize(metadata.targets[rows,3],m['stroma_quartiles'])
            strata = {'all':np.ones(len(rows),bool), **{f'stroma_q{q+1}':(quartiles==q)&metadata.target_valid_mask[rows,3] for q in range(4)}}
            for stratum, keep in strata.items():
                for space, values, metric in [('latent',latent[index],'cosine'),('umap',coords[index],'euclidean')]:
                    try:
                        summary, neighbors = geometry(values[keep],labels[keep],metric=metric)
                    except ValueError as error:
                        if stratum == 'all':
                            raise
                        summaries.append(dict(seed=seed,stratum=stratum,space=space,reason=str(error)))
                        continue
                    summaries.append(dict(seed=seed,stratum=stratum,space=space,**summary))
                    for slide, frame in neighbors.groupby('slide'):
                        per_slide.append(dict(seed=seed,stratum=stratum,space=space,slide=int(slide),
                            slide_name=m['batch_names'][int(slide)],rows=len(frame),
                            **{k:float(frame[k].mean()) for k in ('same_slide','chance','excess')}))
            with torch.inference_mode():
                measurements = evaluate_alignment(torch.as_tensor(latent[index],device='cuda'),
                    torch.as_tensor(labels,device='cuda'), NAMES, settings['bandwidths'], settings['temperature'])
            for name, measurement in measurements.items():
                diagnostics.append(dict(seed=seed,objective=name,score=float(measurement.score) if measurement.score is not None else None,
                    reason=measurement.reason,**{k:float(v) for k,v in measurement.diagnostics.items()}))
    frame = pd.DataFrame(summaries)
    frame.to_csv(out/'mixing_samples.csv',index=False)
    pd.DataFrame(per_slide).to_csv(out/'mixing_per_slide.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'alignment_diagnostics.csv',index=False)
    result = {}
    for space in ('latent','umap'):
        selected = frame[(frame.space==space)&(frame.stratum=='all')]
        result[space] = {k:float(selected[k].mean()) for k in ('same_slide','chance','excess')}
        result[space]['sample_sd'] = float(selected.excess.std(ddof=1))
    return result


def evaluate_checkpoint(root: Path, arm: str, path: Path, m: dict, c: dict, bundle) -> dict:
    """Evaluate a checkpoint fully; cache only compact, checksummed evidence."""
    from scipy.special import ndtr
    from scipy.stats import kstest
    from dann.verification.pooling_diagnostics import distribution_metrics, fit_probes, probe_metrics
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    validate_identity(checkpoint,m)
    identifier = f'{arm}_epoch{checkpoint["epoch"]+1:04d}'
    out = root/'diagnostics'/identifier
    out.mkdir(exist_ok=True)
    identity = dict(checkpoint_sha256=digest_file(path),manifest_sha256=digest_file(root/'manifest.json'))
    stamp = out/'completion.json'
    if reusable(stamp,identity):
        return json.loads((out/'metrics.json').read_text())
    print(f'Evaluating {identifier}: {path}',flush=True)
    model = AdversarialLatentFusion.from_config(checkpoint_config(checkpoint),len(m['batch_names']),len(c['data']['target_columns'])).cuda().eval()
    model.load_state_dict(checkpoint['model_state'])
    valid, he = extract_rows(model,c,bundle,checkpoint['split_indices']['validation'],he=True)
    record = dict(id=identifier,arm=arm,epoch=checkpoint['epoch']+1,checkpoint=str(path),identity=identity,
                  supervised_selected=path.name=='best.pt',spectrum=variance_spectrum(valid['latent']),he=he)
    if not record['spectrum']['noncollapsed']:
        record.update(invalid_reason='Numerically collapsed latent representation',mixing={},probes={},distributions={})
    else:
        train, _ = extract_rows(model,c,bundle,np.load(root/'probe_train_rows.npy'))
        metadata = bundle.metadata
        rows = valid['rows']; y, mask = metadata.targets[rows],metadata.target_valid_mask[rows]
        epsilon = c['loss']['logit_epsilon']
        distributions = distribution_metrics(valid,y,mask,m['cd8_thresholds'],epsilon)
        for j,name in enumerate(('cd8','tumor','collagen','stroma')):
            positive = mask[:,j] & (y[:,j]>0)
            values = np.clip(y[positive,j].astype(float),epsilon,1-epsilon)
            pit = ndtr((np.log(values/(1-values))-valid['mu'][positive,j])/valid['sigma'][positive,j])
            distributions[name]['positive_pit_ks'] = float(kstest(pit,'uniform').statistic) if len(pit) else None
        record['distributions'] = distributions
        predictions = fit_probes(train,valid,metadata,m['cd8_thresholds'],epsilon,out,persist_arrays=False)
        record['probes'] = probe_metrics(predictions,y,mask,metadata.batch_codes[rows],m['cd8_thresholds'],np.ones(len(rows),bool))
        quartiles = np.digitize(y[:,3],m['stroma_quartiles'])
        record['stratified_probes'] = {f'stroma_q{q+1}':probe_metrics(predictions,y,mask,metadata.batch_codes[rows],
            m['cd8_thresholds'],(quartiles==q)&mask[:,3]) for q in range(4)}
        record['mixing'] = mixing_diagnostics(valid,metadata,root,m,out)
    history_path = path.parent/'history.csv' if path.parent.name != 'archives' else path.parent.parent/'history.csv'
    history = pd.read_csv(history_path)
    row = history.loc[history.epoch==record['epoch']]
    record['training_history'] = row.to_dict('records')
    columns = [k for k in history if k=='epoch' or any(word in k for word in ('gradient','alignment','nhsic','anova','soft_nn'))]
    row[columns].to_csv(out/'gradient_and_alignment_history.csv',index=False)
    save_json(out/'metrics.json',record)
    files = [str(p) for p in out.iterdir() if p.is_file() and p != stamp]
    complete(stamp,identity,dict(files=files),time.monotonic())
    del model,valid
    torch.cuda.empty_cache()
    return json.loads((out/'metrics.json').read_text())


def reusable(stamp: Path, identity: dict) -> bool:
    """Completion means identity and every recorded output checksum still match."""
    if not stamp.exists():
        return False
    saved = json.loads(stamp.read_text())
    if saved.get('status') != 'complete':
        return False
    if saved['identity'] != identity:
        raise ValueError(f'Completed stage identity changed: {stamp}')
    for path, digest in saved['outputs'].items():
        if not Path(path).is_file() or digest_file(Path(path)) != digest:
            raise ValueError(f'Completed stage artifact changed: {path}')
    return True


def complete(stamp: Path, identity: dict, result: dict, start: float) -> None:
    """Publish completion last, only after hashing all declared output artifacts."""
    atomic_json(stamp,dict(status='complete',identity=identity,result=result,
                          outputs={p:digest_file(Path(p)) for p in result.get('files',[])},
                          seconds=time.monotonic()-start,completed_utc=datetime.now(timezone.utc).isoformat()))



def smoke_inputs(root: Path, c: dict, *, include_he: bool = True, lesion_only: bool = False,
                 spatial_regions: bool = False) -> None:
    """Copy bounded real arrays without mislabeling a subset as a full cohort.

    Schema-3 cohort manifests describe exact original rows and cannot accompany
    a slice. These explicitly marked fixtures retain frozen linear intensities,
    target columns and raw H&E vectors, with source identities saved separately.
    No preprocessing is fitted, and no fixture participates in study selection.
    """
    import anndata as ad
    from anndata.io import read_elem
    import h5py
    from scipy.sparse import csr_matrix
    from dann.training_artifacts import atomic_output
    directory = root/'smoke_data'
    directory.mkdir(exist_ok=True)
    provenance = dict(fixture_only=True, preprocessing_refitted=False,
                      original_preprocessing_fingerprint=c['data'].get('preprocessing_fingerprint'), inputs={})
    slides = None
    for key, count in ([('path',2048)] if lesion_only else [('path',2048),('context_path',512)]):
        source = Path(c['data'][key])
        destination = directory/f'{key}.h5ad'
        with h5py.File(source,'r') as handle:
            obs = read_elem(handle['obs'])
            labels = pd.Categorical(obs[c['data']['batch_column']]).codes
            if slides is None:
                slides = sorted(obs[c['data']['batch_column']].astype(str).unique())[:2]
            eligible = np.flatnonzero(obs[c['data']['batch_column']].astype(str).isin(slides).to_numpy())
            if spatial_regions:
                # Compact real regions retain neighboring MSI context in bounded
                # CNN/GATv2 fixtures; selection never depends on target labels.
                regions = []
                for slide in slides:
                    indices_for_slide = eligible[obs.iloc[eligible][c['data']['batch_column']].astype(str).to_numpy() == slide]
                    xy = obs.iloc[indices_for_slide][[c['data']['x_column'],c['data']['y_column']]].to_numpy(float)
                    center = xy[np.argmin(np.square(xy-np.median(xy,axis=0)).sum(axis=1))]
                    order = np.argsort(np.square(xy-center).sum(axis=1),kind='stable')
                    regions.append(indices_for_slide[order[:count//len(slides)]])
                rows = np.sort(np.concatenate(regions))
            else:
                rows = np.sort(stratified_cap_indices(eligible,labels,count,SEED))
            var = read_elem(handle['var'])
            group = handle['X']
            pointers = group['indptr'][:]
            lengths = pointers[rows+1]-pointers[rows]
            outptr = np.r_[0,np.cumsum(lengths)].astype(np.int64)
            values = np.empty(int(outptr[-1]),dtype=group['data'].dtype)
            indices = np.empty(int(outptr[-1]),dtype=group['indices'].dtype)
            for i,row in enumerate(rows):
                left,right = pointers[row:row+2]
                values[outptr[i]:outptr[i+1]] = group['data'][left:right]
                indices[outptr[i]:outptr[i+1]] = group['indices'][left:right]
            matrix = csr_matrix((values,indices,outptr),shape=(len(rows),len(var)))
            he_key = c['data']['he_features_key']
            obsm = {}
            if include_he:
                obsm[he_key] = np.asarray(handle[f'obsm/{he_key}'][rows],dtype=np.float32)
            uns = {'alignment_smoke_fixture': True}
            if 'obsm/spatial' in handle:
                obsm['spatial'] = np.asarray(handle['obsm/spatial'][rows])
            if not lesion_only and 'uns/spatial' in handle:
                uns['spatial'] = {slide:read_elem(handle['uns/spatial'][slide]) for slide in slides}
            fixture = ad.AnnData(X=matrix,obs=obs.iloc[rows].copy(),var=var.copy(),obsm=obsm,uns=uns)
            with atomic_output(destination) as temporary:
                fixture.write_h5ad(temporary)
            np.save(directory/f'{key}_source_rows.npy',rows)
            provenance['inputs'][key] = dict(source=_file_identity(source),rows=len(rows),
                rows_sha256=fingerprint(rows),fixture_sha256=digest_file(destination))
    provenance['slides'] = slides
    atomic_json(directory/'provenance.json',provenance)
    # The fixture has separate provenance, not the original exact-cohort claim.
    c['data']['preprocessing_fingerprint'] = None


def execute_stage(root: Path, arm: str, stage: str) -> None:
    """Restart one stage safely, preserving logs, best.pt and previous results."""
    m = manifest_for(root)
    c = config_for(root,arm)
    identity = dict(manifest_sha256=digest_file(root/'manifest.json'),arm=arm,stage=stage,
                    config=fingerprint(c))
    if stage in ('evaluate','full_umap'):
        paths = checkpoint_paths(root,arm,m)
        identity['checkpoints'] = {str(p):digest_file(p) for p in paths}
    if stage == 'full_umap':
        selection = json.loads((root/'decisions'/'selected_epochs.json').read_text())[arm]
        identity['selected'] = selection['id']
    stamp = root/'stages'/f'{arm}-{stage}.json'
    if reusable(stamp,identity):
        print(f'Reusing {arm} {stage}',flush=True)
        return
    start = time.monotonic()
    atomic_json(stamp,dict(status='running',identity=identity,started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        if stage == 'smoke':
            from dann.verification.alignment_refinement import bounded_smoke
            # Train and validate in isolation; never invoke the downstream helper.
            c['loss']['batch_alignment'].update({name:dict(enabled=True,weight=weight) for name,weight in zip(NAMES,(1.,.1,1.))})
            result = bounded_smoke(root,c)
        elif stage == 'train':
            latest = Path(c['training']['output_dir'])/'latest.pt'
            if latest.exists():
                saved = torch.load(latest,map_location='cpu',weights_only=False)
                validate_identity(saved,m)
                if not saved.get('training_rng_state'):
                    raise ValueError('Interrupted study fit lacks exact RNG restart state.')
                c['training']['resume_checkpoint'] = str(latest)
            train_model(c,observer=training_observer(root,arm,m))
            result = validate_fit(root,arm,m)
        elif stage == 'evaluate':
            if arm != 'reference':
                validate_fit(root,arm,m)
            saved = torch.load(paths[0],map_location='cpu',weights_only=False)
            bundle = create_data_bundle(c,saved['split_indices'],saved['data_identity'],requested_splits=('validation',))
            try:
                records = [evaluate_checkpoint(root,arm,p,m,c,bundle) for p in paths]
            finally:
                close(bundle)
            destination = root/'diagnostics'/f'{arm}_checkpoints.json'
            atomic_json(destination,records)
            result = dict(files=[str(destination),*[str(root/'diagnostics'/r['id']/'completion.json') for r in records]],checkpoints=len(records))
        elif stage == 'full_umap':
            path = Path(selection['checkpoint'])
            checkpoint = torch.load(path,map_location='cpu',weights_only=False)
            model = AdversarialLatentFusion.from_config(checkpoint_config(checkpoint),len(m['batch_names']),len(c['data']['target_columns'])).cuda().eval()
            model.load_state_dict(checkpoint['model_state'])
            bundle = create_data_bundle(c,checkpoint['split_indices'],checkpoint['data_identity'],requested_splits=('validation',))
            out = root/'diagnostics'/selection['id']/'full_validation'
            out.mkdir(exist_ok=True)
            try:
                data,_ = extract_rows(model,c,bundle,checkpoint['split_indices']['validation'])
                coords = fit_umap(data['latent'])
                np.savez_compressed(out/'umap.npz',rows=data['rows'],umap=coords)
                plot_umap(coords,data['rows'],data,bundle.metadata,m,out/'umap.png',selection['id']+' full validation')
                atomic_json(out/'provenance.json',dict(checkpoint_sha256=digest_file(path),rows=len(data['rows']),
                    rows_sha256=fingerprint(data['rows']),settings=m['protocol']['umap']))
            finally:
                close(bundle)
            result = dict(files=[str(p) for p in out.iterdir() if p.is_file()])
        else:
            raise ValueError(f'Unknown stage: {stage}')
        complete(stamp,identity,result,start)
    except BaseException as error:
        if stage == 'train' and (isinstance(error, FloatingPointError) or
                isinstance(error, ValueError) and str(error).startswith('Enabled alignment objective ')):
            # This is a scientific exclusion, not an infrastructure retry. Keep
            # all partial checkpoints and explicitly exclude the incomplete fit.
            exclusion = root/'diagnostics'/f'{arm}_exclusion.json'
            atomic_json(exclusion,dict(arm=arm,reason=str(error),traceback=traceback.format_exc(),
                                       partial_checkpoints_preserved=True))
            empty = root/'diagnostics'/f'{arm}_checkpoints.json'
            atomic_json(empty,[])
            complete(stamp,identity,dict(excluded=True,reason=str(error),files=[str(exclusion),str(empty)]),start)
            print(f'Excluded numerically invalid fit {arm}: {error}',flush=True)
            return
        atomic_json(stamp,dict(status='failed',identity=identity,seconds=time.monotonic()-start,
                              error=repr(error),traceback=traceback.format_exc()))
        raise


def records_for(root: Path, arm: str) -> list[dict]:
    """Read completed checkpoint metrics; absent arms remain visibly pending."""
    path = root/'diagnostics'/f'{arm}_checkpoints.json'
    return json.loads(path.read_text()) if path.exists() else []


def decision(root: Path, name: str, value: dict) -> None:
    """Freeze adaptive decisions; a rerun cannot silently choose a new branch."""
    path = root/'decisions'/f'{name}.json'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Previously recorded adaptive decision changed: {name}')
    else:
        immutable_json(path,value)


def run_subprocess(root: Path, arm: str, stage: str) -> None:
    """One GPU subprocess at a time, always importing the frozen source snapshot."""
    with (root/'logs'/f'{arm}-{stage}.log').open('a') as log:
        print(f'Starting {arm} {stage}; log: {root / "logs" / (arm+"-"+stage+".log")}',flush=True)
        env = dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4',
                   MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
        subprocess.run([sys.executable,'-m','dann.verification.alignment_sweep','--output',str(root),
                        '--arm',arm,'--stage',stage],cwd=root/'source',env=env,
                        stdout=log,stderr=subprocess.STDOUT,check=True)


def recommendation_workflow(root: Path, selected: dict) -> None:
    """Full tissue inference and spatial analysis in the dedicated study tree."""
    from dann.workflow import run_post_training_workflow, _validate_exports
    m = manifest_for(root)
    path = Path(selected['checkpoint'])
    c = config_for(root,selected['arm'])
    directory = root/'recommendation'
    c['analysis'].update(output_dir=str(directory/'analysis'),checkpoint=str(path),max_samples=None)
    c['spatial'] = dict(output=str(directory/'analysis/spatial/spatial_predictions.parquet'),
                        latents=str(directory/'analysis/spatial/spatial_latent.parquet'))
    c['latent_variance']['output_dir'] = str(directory/'analysis/latent_variance')
    c['post_training'].update(enabled=True,run_analysis=True,run_inference=True,
                             run_spatial_heatmaps=True,run_latent_variance=True,overwrite_inference=True)
    stamp = root/'stages'/'recommendation-workflow.json'
    identity = dict(checkpoint_sha256=digest_file(path),manifest_sha256=digest_file(root/'manifest.json'),config=fingerprint(c))
    if reusable(stamp,identity):
        return
    start = time.monotonic()
    atomic_json(stamp,dict(status='running',identity=identity))
    try:
        checkpoint = torch.load(path,map_location='cpu',weights_only=False)
        runtime = copy.deepcopy(c)
        # A failed later render must not force another full tissue export. Only
        # reuse a complete, provenance-validated pair; partial exports are rebuilt
        # in this experiment's dedicated recommendation tree.
        try:
            _validate_exports(c,checkpoint,path,Path(c['data']['context_path']),predictions=True,latents=True)
        except ValueError:
            pass
        else:
            runtime['post_training']['run_inference'] = False
        run_post_training_workflow(runtime,path)
        _validate_exports(c,checkpoint,path,Path(c['data']['context_path']),predictions=True,latents=True)
        complete(stamp,identity,dict(files=[str(p) for p in directory.rglob('*') if p.is_file()]),start)
    except BaseException as error:
        atomic_json(stamp,dict(status='failed',identity=identity,error=repr(error),traceback=traceback.format_exc()))
        raise


def report(root: Path, *, finished: bool | None = None) -> None:
    """Build a self-contained HTML comparison with auditable tables and panels."""
    import html
    if finished is None:
        status = root/'status.json'
        finished = bool(json.loads(status.read_text()).get('complete', False)) if status.exists() else False
    m = manifest_for(root)
    arms = ['reference',*[p.stem for p in sorted((root/'arms').glob('*.json'))]]
    records = [r for arm in arms for r in records_for(root,arm)]
    reference = next(iter(records_for(root,'reference')),None)
    matrix, primary, calibration, probes, limits, history, mixing, per_slide, alignments = [], [], [], [], [], [], [], [], []
    for arm in arms:
        config = config_for(root,arm)
        statuses = {p.stem.split('-',1)[-1]:json.loads(p.read_text()).get('status') for p in (root/'stages').glob(f'{arm}-*.json')}
        matrix.append(dict(arm=arm,**{name:config['loss']['batch_alignment'][name]['weight'] for name in NAMES},
                           **statuses,checkpoints=len(records_for(root,arm))))
        exclusion = root/'diagnostics'/f'{arm}_exclusion.json'
        if exclusion.exists():
            matrix[-1]['exclusion'] = json.loads(exclusion.read_text())['reason']
    for r in records:
        if reference is not None:
            check = eligibility(r,reference)
            limits.extend(dict(id=r['id'],**v) for v in check['checks'])
        else:
            check = dict(eligible=False)
        primary.append(dict(id=r['id'],epoch=r['epoch'],arm=r['arm'],supervised_selected=r['supervised_selected'],
            eligible=check['eligible'],**{f'{s}_{k}':v for s,values in r['mixing'].items() for k,v in values.items()},
            **{f'{n}_participation_ratio':r['spectrum'].get(n,{}).get('participation_ratio') for n in ('raw','normalized')},
            cd8_loss=r['distributions'].get('cd8',{}).get('loss'),**{f'he_{k}':v for k,v in r['he'].items()}))
        for name, scores in r['distributions'].items():
            calibration.append(dict(id=r['id'],target=name,loss=scores['loss'],presence_brier=scores['presence']['brier'],
                pit_ks=scores['positive_pit_ks'],**{f'coverage_{level}':v['coverage'] for level,v in scores['intervals'].items()},
                **{f'exceedance_{threshold}_brier':v['brier'] for threshold,v in scores.get('exceedance',{}).items()}))
        probes.extend(dict(id=r['id'],probe=name,**scores) for name,scores in r['probes'].items())
        history.extend(dict(id=r['id'],**row) for row in r['training_history'])
        for filename, target in [('mixing_samples.csv',mixing),('mixing_per_slide.csv',per_slide),('alignment_diagnostics.csv',alignments)]:
            path = root/'diagnostics'/r['id']/filename
            if path.exists():
                target.extend(dict(id=r['id'],**row) for row in pd.read_csv(path).to_dict('records'))
    tables = dict(run_matrix=matrix,checkpoint_comparison=primary,calibration=calibration,biological_probes=probes,
                  eligibility_limits=limits,training_history=history,mixing_samples=mixing,mixing_per_slide=per_slide,alignment_diagnostics=alignments)
    for name,values in tables.items():
        pd.DataFrame(values).to_csv(root/f'{name}.csv',index=False)
    final_path = root/'decisions'/'recommendation.json'
    final = json.loads(final_path.read_text()) if final_path.exists() else None
    text = 'Study in progress; no final recommendation.'
    if final:
        text = ('Recommended checkpoint: '+final['selected']['checkpoint']) if final['selected'] else 'No candidate met the eligibility and mixing improvement requirements. No winner is declared.'
    selected_path = root/'decisions'/'selected_epochs.json'
    selections = json.loads(selected_path.read_text()) if selected_path.exists() else {}
    panels = []
    for arm, selected in selections.items():
        identifier = selected['id']
        relative = f'diagnostics/{identifier}'
        panels.append(f'<h3>{html.escape(identifier)}</h3><img src="{relative}/panel.png" width="100%">'
                      f'<p><a href="{relative}/full_validation/umap.png">Full-validation UMAP</a> · '
                      f'<a href="{relative}/metrics.json">All metrics and variance spectra</a></p>')
    links = ' · '.join(f'<a href="{name}.csv">{name.replace("_"," ")}</a>' for name in tables)
    def table(rows):
        return pd.DataFrame(rows).to_html(index=False,float_format=lambda x:f'{x:.5g}',escape=True)
    body = f'''<!doctype html><meta charset="utf-8"><title>DANN alignment sweep</title>
<style>body{{font:15px system-ui;margin:30px;max-width:1700px}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:5px;border:1px solid #ddd}}.scroll{{overflow:auto}}img{{max-width:1600px}}</style>
<h1>DANN alignment sweep</h1><p>{html.escape(text)}</p><p>Workflow complete: {finished}. Reference epoch {m['baseline']['epoch']}.
Single training seed {SEED}; five diagnostic draws measure sampling stability, not independent training replication.
Validation rows are used for study selection; these are not unbiased test-set estimates. H&amp;E transfer remains a separate assessment.</p>
<p>{links} · <a href="manifest.json">Frozen protocol and identities</a> · <a href="decisions/selected_epochs.json">Selected epochs</a></p>
<h2>Run matrix</h2>{table(matrix)}<h2>Checkpoint mixing and eligibility</h2><div class="scroll">{table(primary)}</div>
<p>Same-slide excess is observed minus leave-one-out chance. Latent distances use cosine; UMAP distances use Euclidean.
Both mixing measures must improve. Latent excess ties within 0.01 use UMAP excess, then CD8 loss.
All biological and CD8 limits are inclusive; relative loss deterioration uses 10% of the absolute reference loss.
Raw and normalized variance must exceed 1e-8, with at least two numerically resolved covariance directions.
Participation ratios describe concentration of variance and are reported without an additional eligibility cutoff.</p>
<h2>Calibration and other IHC</h2><div class="scroll">{table(calibration)}</div>
<h2>Training-fitted biological probes</h2>{table(probes)}<h2>Matched UMAP panels</h2>{''.join(panels)}
<p>All slide colors and row identities are fixed. CD8 exceedance thresholds and stromal quartiles are fitted on training rows only.
Full per-slide and within-quartile mixing, weighted gradient contributions/conflicts, spectra, reliability bins and interval widths are linked above.</p>'''
    (root/'report.html').write_text(body)
    atomic_json(root/'status.json',dict(complete=finished,recommendation=final,arms=matrix,updated_utc=datetime.now(timezone.utc).isoformat()))


def run_all(root: Path) -> None:
    """Run control/sweep, choose a base, test additions, then conditionally combine."""
    manifest_for(root)
    # flock is released on process exit; no stale-lock recovery is needed.
    with (root/'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another sweep runner owns this experiment.') from error
        run_subprocess(root,'control','smoke')
        run_subprocess(root,'reference','evaluate')
        reference = records_for(root,'reference')[0]
        if not eligibility(reference,reference)['eligible']:
            raise ValueError('Reference cannot establish finite eligibility limits.')
        report(root)
        for arm in INITIAL_ARMS:
            run_subprocess(root,arm,'train')
            if not (root/'diagnostics'/f'{arm}_exclusion.json').exists():
                run_subprocess(root,arm,'evaluate')
            report(root)
        base = choose([r for arm in ('nhsic1','nhsic10','nhsic100') for r in records_for(root,arm)],reference) or reference
        weight = .1 if base['arm']=='reference' else config_for(root,base['arm'])['loss']['batch_alignment']['nhsic']['weight']
        decision(root,'nhsic_base',dict(selected=base,weight=weight,retained_reference=base['arm']=='reference'))
        additions = dict(anova0p1=(weight,.1,0.),anova1=(weight,1.,0.),soft_nn1=(weight,0.,1.),soft_nn10=(weight,0.,10.))
        for arm, weights in additions.items():
            register_arm(root,arm,weights)
            run_subprocess(root,arm,'train')
            if not (root/'diagnostics'/f'{arm}_exclusion.json').exists():
                run_subprocess(root,arm,'evaluate')
            report(root)
        anova = choose([r for arm in ('anova0p1','anova1') for r in records_for(root,arm)],reference,base=base)
        soft = choose([r for arm in ('soft_nn1','soft_nn10') for r in records_for(root,arm)],reference,base=base)
        decision(root,'additions',dict(base=base['id'],anova=anova,soft_nn=soft,run_combined=bool(anova and soft)))
        if anova and soft:
            weights = (weight,config_for(root,anova['arm'])['loss']['batch_alignment']['anova']['weight'],
                       config_for(root,soft['arm'])['loss']['batch_alignment']['soft_nn']['weight'])
            register_arm(root,'combined',weights)
            run_subprocess(root,'combined','train')
            if not (root/'diagnostics'/'combined_exclusion.json').exists():
                run_subprocess(root,'combined','evaluate')
        arms = ['reference',*INITIAL_ARMS,*additions,*(['combined'] if anova and soft else [])]
        selected_epochs = {}
        candidates = []
        for arm in arms:
            records = records_for(root,arm)
            if not records:
                continue
            helpful = choose(records,reference,base=base if arm in (*additions,'combined') else None)
            candidates.extend([r for r in records if arm not in (*additions,'combined') or improves(r,base)])
            selected_epochs[arm] = helpful or next(r for r in records if r['supervised_selected'])
        recommended = choose(candidates,reference)
        decision(root,'selected_epochs',selected_epochs)
        decision(root,'recommendation',dict(selected=recommended,
            reason='Eligible improvement in both latent and UMAP excess' if recommended else 'No eligible improvement in both spaces',
            supervised_checkpoints_preserved=True))
        report(root)
        for arm in selected_epochs:
            run_subprocess(root,arm,'full_umap')
        if recommended:
            with (root/'logs'/'recommendation-workflow.log').open('a') as log:
                subprocess.run([sys.executable,'-m','dann.verification.alignment_sweep','--output',str(root),'--recommendation-workflow'],
                    cwd=root/'source',stdout=log,stderr=subprocess.STDOUT,check=True,
                    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4',MPLBACKEND='Agg'))
        report(root,finished=True)


def main() -> None:
    """Prepare once; execute only the frozen runner for all subsequent stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('dann/config.yaml'))
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--arm')
    parser.add_argument('--stage',choices=('smoke','train','evaluate','full_umap'))
    parser.add_argument('--recommendation-workflow',action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    torch.set_num_threads(4)
    if args.prepare:
        prepare(root,args.source,args.baseline)
    if args.run or args.stage or args.report or args.recommendation_workflow:
        snapshot = root/'source/dann/verification/alignment_sweep.py'
        if Path(__file__).resolve() != snapshot:
            command = [sys.executable,'-m','dann.verification.alignment_sweep','--output',str(root)]
            if args.run:
                command.append('--run')
            if args.report:
                command.append('--report')
            if args.stage:
                command += ['--arm',args.arm,'--stage',args.stage]
            if args.recommendation_workflow:
                command.append('--recommendation-workflow')
            subprocess.run(command,cwd=root/'source',check=True)
            return
    if args.stage:
        if not args.arm:
            parser.error('--stage requires --arm')
        execute_stage(root,args.arm,args.stage)
    if args.run:
        run_all(root)
    if args.report:
        report(root)
    if args.recommendation_workflow:
        selected = json.loads((root/'decisions/recommendation.json').read_text())['selected']
        if selected:
            recommendation_workflow(root,selected)


if __name__ == '__main__':
    main()

"""Compare archived aggregation exports without applying current data configuration.

Example: python -m dann.verification.batch_alignment_comparison ANALYSIS_A
ANALYSIS_B --output NEW_DIRECTORY --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dann.batch_alignment import BANDWIDTHS, NAMES, alignment_contract, evaluate_alignment, sample_rows

SEEDS = (20260719, 20260720, 20260721, 20260722, 20260723)


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_archive(directory: Path):
    """Verify recorded checkpoint/source identity and checksum of archived exports."""
    execution = json.loads((directory / 'analysis_execution.json').read_text())
    source = execution['source_identity']
    if not isinstance(execution.get('checkpoint_epoch'), int) or len(source.get('checkpoint', '')) != 64:
        raise ValueError('Archive lacks checkpoint epoch or fingerprint provenance.')
    path = directory / 'tables' / 'latent_embeddings.csv'
    record = json.loads((directory / 'latent' / 'latent_run.json').read_text())
    hashes = [v for k, v in record['source_hashes'].items() if Path(k).name == path.name]
    digest = file_hash(path)
    if hashes != [digest]:
        raise ValueError('Archived latent export checksum does not match recorded provenance.')
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    latent_columns = [c for c in columns if c.startswith('latent_')]
    if latent_columns != [f'latent_{i}' for i in range(len(latent_columns))] or not latent_columns:
        raise ValueError('Latent columns must retain contiguous exported order.')
    # A metadata pass avoids materializing the entire 512-wide export in memory.
    table = pd.read_csv(path, usecols=['row_id', 'batch', 'batch_id', 'split']).set_index('row_id')
    if not table.index.is_unique or table.isna().any().any():
        raise ValueError('Archive row IDs must be unique and metadata complete.')
    if table.groupby('batch').batch_id.nunique().max() != 1 or table.groupby('batch_id').batch.nunique().max() != 1:
        raise ValueError('Inconsistent slide names and codes.')
    provenance = dict(directory=str(directory.resolve()), checkpoint_path=execution['checkpoint'],
                      epoch=execution['checkpoint_epoch'], checkpoint_fingerprint=source['checkpoint'],
                      source_identity=source['identity'], latent_sha256=digest,
                      latent_columns=latent_columns, rows=len(table),
                      verification='saved checkpoint fingerprint and epoch; saved export SHA256; no loading through current config')
    return table, provenance


def matched_metadata(first, second, slides=43, per_slide=64):
    """Join on row identity, reject label changes, and require all requested slides."""
    common = first.index.intersection(second.index).sort_values()
    a, b = first.loc[common], second.loc[common]
    if not a[['batch', 'batch_id', 'split']].equals(b[['batch', 'batch_id', 'split']]):
        raise ValueError('Slide labels or split assignments disagree for shared row IDs.')
    counts = a.groupby('batch').size()
    if len(counts) != slides or (counts < per_slide).any():
        raise ValueError(f'Require {per_slide} common observations in each of {slides} slides; found {counts.to_dict()}.')
    return a


def read_selected(directory, row_ids, columns):
    parts = []
    for chunk in pd.read_csv(directory / 'tables' / 'latent_embeddings.csv', chunksize=8192):
        parts.append(chunk.loc[chunk.row_id.isin(row_ids), ['row_id', *columns]])
    return pd.concat(parts).set_index('row_id').loc[row_ids, columns]


def geometry(latent, labels):
    """Exact cosine 30-neighbor fractions and off-diagonal distance distributions."""
    raw = latent.double()
    x = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)
    distance = (1 - x @ x.T).clamp(0, 2)
    same = labels[:, None] == labels[None, :]
    off = ~torch.eye(len(x), dtype=torch.bool, device=x.device)
    rows = []
    for name, mask in [('all', off), ('same_slide', off & same), ('different_slide', off & ~same)]:
        values = distance[mask].cpu().numpy()
        quantiles = np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1])
        rows.append(dict(pairs=name, count=len(values), mean=float(values.mean()), std=float(values.std()),
                         **{f'q{q}': float(v) for q, v in zip(['0','01','05','25','50','75','95','99','100'], quantiles)}))
    # Stable full sort supplies exact distances and deterministic identity-order tie breaks.
    neighbor = distance.masked_fill(~off, torch.inf).argsort(dim=1, stable=True)[:, :30]
    fractions = (labels[neighbor] == labels[:, None]).double().mean(1)
    centered = raw - raw.mean(0)
    covariance = centered.T @ centered / (len(raw)-1)
    variance = covariance.trace()
    normalized = x - x.mean(0)
    c = normalized.T @ normalized / (len(x)-1)
    summary = dict(exact_cosine_30_same_slide=float(fractions.mean()),
                   exact_cosine_30_chance=float((same.sum(1)-1).double().mean() / (len(x)-1)),
                   latent_total_variance=float(variance), latent_participation_ratio=float(variance.square()/covariance.square().sum()),
                   normalized_sample_variance=float(c.trace()), normalized_participation_ratio=float(c.trace().square()/c.square().sum()))
    return summary, rows, fractions.cpu().numpy()


def context(directory, epoch):
    """Copy existing likelihood/calibration and whole-tissue variance context."""
    result = {}
    for name, relative in [('calibration', 'calibration/calibration_metrics.csv'),
                           ('whole_tissue_variance', 'latent_variance/latent_variance_summary.csv')]:
        path = directory / relative
        if path.exists():
            frame = pd.read_csv(path)
            if 'target' in frame:
                frame = frame[frame.target == 'Density_CD8']
            result[name] = dict(path=str(path.resolve()), sha256=file_hash(path), records=frame.to_dict('records'))
    history = directory.parent / 'model' / 'history.csv'
    if history.exists():
        frame = pd.read_csv(history)
        frame = frame.loc[frame.epoch == epoch]
        columns = [c for c in frame if c == 'epoch' or c.startswith('validation_') and any(
            key in c for key in ('cd8', 'supervised_objective', 'he_r2', 'batch_loss', 'batch_accuracy', 'latent_'))]
        result['history'] = dict(path=str(history.resolve()), sha256=file_hash(history), records=frame[columns].to_dict('records'))
    return result


def compare(first: Path, second: Path, output: Path, device='cpu') -> dict:
    """Run five matched 43-by-64 comparisons and save manifests and diagnostics."""
    if output.exists():
        raise FileExistsError('Comparison output must be a fresh directory.')
    tables, provenance = zip(*(read_archive(p) for p in (first, second)))
    if provenance[0]['source_identity'] != provenance[1]['source_identity']:
        raise ValueError('Archived source identity differs; integer row IDs cannot safely be joined.')
    if provenance[0]['latent_columns'] != provenance[1]['latent_columns']:
        raise ValueError('Archived latent-column order differs.')
    if provenance[0]['checkpoint_fingerprint'] == provenance[1]['checkpoint_fingerprint']:
        raise ValueError('Comparison requires distinct archived checkpoints.')
    matched = matched_metadata(*tables)
    samples = {seed: sample_rows(matched.index.to_numpy(), matched.batch.to_numpy(), 64, seed) for seed in SEEDS}
    union = np.unique(np.concatenate(list(samples.values())))
    exports = [read_selected(p, union, provenance[0]['latent_columns']) for p in (first, second)]
    output.mkdir(parents=True)
    manifest = pd.concat([matched.loc[ids].reset_index().assign(seed=seed, observation_weight=1/2752)
                          for seed, ids in samples.items()], ignore_index=True)
    manifest.to_csv(output/'sample_manifest.csv', index=False)
    (output/'provenance.json').write_text(json.dumps(dict(archives=provenance, seeds=SEEDS, observations_per_slide=64,
        slides=43, common_rows=len(matched), sampled_rows=2752, estimator=alignment_contract({'loss': {'batch_alignment': {n: dict(enabled=True, weight=0.) for n in NAMES}}}),
        source_population='archived validation exports, all pixels, no tissue conditioning',
        biological_context=[context(p, v['epoch']) for p, v in zip((first, second), provenance)]), indent=2))
    results, distances, neighbors = [], [], []
    with torch.inference_mode():
        for seed, ids in samples.items():
            labels = torch.tensor(pd.Categorical(matched.loc[ids].batch).codes.copy(), dtype=torch.long, device=device)
            for export, info in zip(exports, provenance):
                epoch = info['epoch']
                latent = torch.tensor(export.loc[ids].to_numpy(), device=device)
                geo, dist, fractions = geometry(latent, labels)
                distances.extend(dict(seed=seed, epoch=epoch, **r) for r in dist)
                neighbors.extend(dict(seed=seed, epoch=epoch, row_id=int(i), same_slide_fraction=float(v)) for i, v in zip(ids, fractions))
                variants = {'default': BANDWIDTHS, **{f'bandwidth_{b:g}': (b,) for b in BANDWIDTHS},
                            **{f'mixture_x{scale:g}': tuple(b*scale for b in BANDWIDTHS) for scale in (.5, 1., 2.)}}
                for variant, bandwidths in variants.items():
                    measurements = evaluate_alignment(latent, labels, NAMES if variant == 'default' else ('nhsic',), bandwidths)
                    for name, m in measurements.items():
                        row = dict(seed=seed, epoch=epoch, variant=variant, objective=name,
                                   score=float(m.score) if m.score is not None else None, reason=m.reason,
                                   **{k: float(v) for k, v in m.diagnostics.items()}, **geo)
                        results.append(row)
                print(f'Compared seed {seed}, epoch {epoch}', flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output/'objectives.csv', index=False)
    pd.DataFrame(distances).to_csv(output/'cosine_distances.csv', index=False)
    pd.DataFrame(neighbors).to_csv(output/'exact_cosine_30_neighbors.csv', index=False)
    summary = frame.groupby(['epoch','objective','variant']).score.agg(['mean','std','min','max']).reset_index()
    summary.to_csv(output/'summary.csv', index=False)
    ranking = []
    epochs = [p['epoch'] for p in provenance]
    for (objective, variant), group in frame.groupby(['objective', 'variant']):
        pivot = group.pivot(index='seed', columns='epoch', values='score')
        delta = pivot[epochs[1]] - pivot[epochs[0]]
        ranking.append(dict(objective=objective, variant=variant, first_epoch=epochs[0], second_epoch=epochs[1],
                            second_minus_first_mean=float(delta.mean()), first_lower_count=int((delta>0).sum()),
                            second_lower_count=int((delta<0).sum()), unavailable_count=int(delta.isna().sum())))
    (output/'rankings.json').write_text(json.dumps(ranking, indent=2))
    (output/'results.json').write_text(frame.to_json(orient='records', indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout='constrained')
    for ax, name in zip(axes, NAMES):
        subset = frame[(frame.objective == name) & (frame.variant == 'default')]
        for seed, group in subset.groupby('seed'):
            ax.plot(group.epoch, group.score, '.-', alpha=.7)
        ax.set(title=name, xlabel='Checkpoint epoch', xticks=epochs)
    fig.savefig(output/'objectives.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.5), layout='constrained')
    hsic = summary[summary.objective == 'nhsic']
    variants = hsic.variant.unique()
    for epoch in epochs:
        rows = hsic[hsic.epoch == epoch].set_index('variant').loc[variants]
        ax.errorbar(np.arange(len(variants)), rows['mean'], yerr=rows['std'], marker='.', label=f'Epoch {epoch}')
    ax.set(xticks=np.arange(len(variants)), xticklabels=variants, ylabel='Normalized HSIC')
    ax.tick_params(axis='x', rotation=25); ax.legend()
    fig.savefig(output/'bandwidth_sensitivity.png', dpi=160); plt.close(fig)
    return dict(output=str(output), rankings=ranking)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path); parser.add_argument('second', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu'); parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    compare(args.first, args.second, args.output, args.device)


if __name__ == '__main__':
    main()

"""Compare matched peak geometry and held-out family biology without retraining.

python -m dann.verification.peak_comparison --comparison COMPARISON_DIRECTORY
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path

from anndata.io import read_elem
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
import torch

from dann.analysis_execution import digest_file, fingerprint
from dann.config import load_config
from dann.peak_analysis import (
    canonical_block, family_hierarchy, family_similarity, read_peak_metadata, resident_matrix,
)
from dann.targets import prepare_observed_targets
from dann.verification.dictionary_comparison import manifest_for
from dann.verification.pooling_diagnostics import covariance_metrics, save_json, variance_component_counts

SEED = 20260719
TARGETS = ('Density_CD8', 'Density_Tumor', 'Density_Collagen', 'Density_Stroma')
ARMS = ('baseline', 'l1', 'l2')


def family_match(left: np.ndarray, right: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Match arbitrary family IDs by maximum member overlap; retain match quality."""
    left_ids, a = np.unique(left, return_inverse=True)
    right_ids, b = np.unique(right, return_inverse=True)
    overlap = np.zeros((len(left_ids), len(right_ids)), dtype=np.int64)
    np.add.at(overlap, (a, b), 1)
    i, j = linear_sum_assignment(-overlap)
    left_size, right_size = overlap.sum(1), overlap.sum(0)
    table = pd.DataFrame(dict(left_family=left_ids[i], right_family=right_ids[j],
        overlap=overlap[i, j], left_size=left_size[i], right_size=right_size[j],
        jaccard=overlap[i, j] / (left_size[i] + right_size[j] - overlap[i, j])))
    return dict(adjusted_rand=adjusted_rand_score(left, right),
        adjusted_mutual_information=adjusted_mutual_info_score(left, right),
        matched_peak_fraction=overlap[i, j].sum() / len(left),
        mean_matched_jaccard=table.jaccard.mean()), table


def nearest_indices(similarity: np.ndarray, k: int) -> np.ndarray:
    """Find cosine neighbors, explicitly excluding each peak itself."""
    work = similarity.copy()
    np.fill_diagonal(work, -np.inf)
    selected = np.argpartition(work, -k, axis=1)[:, -k:]
    scores = np.take_along_axis(work, selected, axis=1)
    return np.take_along_axis(selected, np.argsort(-scores, axis=1), axis=1)


def rank_correlations(x: np.ndarray, y: np.ndarray, controls: np.ndarray | None = None) -> np.ndarray:
    """Columnwise Spearman or partial Spearman after rank-linear residualization."""
    x = rankdata(np.asarray(x, dtype=np.float64), axis=0)
    y = rankdata(np.asarray(y, dtype=np.float64))
    x -= x.mean(0)
    y -= y.mean()
    if controls is not None:
        c = rankdata(np.asarray(controls, dtype=np.float64), axis=0)
        c -= c.mean(0)
        x -= c @ np.linalg.lstsq(c, x, rcond=None)[0]
        y -= c @ np.linalg.lstsq(c, y, rcond=None)[0]
    denominator = np.linalg.norm(x, axis=0) * np.linalg.norm(y)
    return np.divide(y @ x, denominator, out=np.full(x.shape[1], np.nan), where=denominator > 1e-12)


def mean_finite(values: np.ndarray, axis=0) -> np.ndarray:
    """Mean finite observations, leaving empty cells explicitly undefined."""
    count = np.isfinite(values).sum(axis=axis)
    return np.divide(np.nansum(values, axis=axis), count,
                     out=np.full(np.shape(count), np.nan), where=count > 0)


def neighbor_coherence(profile: np.ndarray, neighbors: np.ndarray) -> tuple[float, float]:
    """Measure same-target association agreement, not a predictive validation score."""
    predicted = mean_finite(profile[neighbors], axis=1)
    keep = np.isfinite(profile) & np.isfinite(predicted)
    error = predicted[keep] - profile[keep]
    variance = np.sum((profile[keep] - profile[keep].mean()) ** 2)
    r2 = 1 - np.sum(error ** 2) / variance if variance > 0 else np.nan
    return float(np.mean(np.abs(error))), float(r2)


def matched_permutations(stats: pd.DataFrame, repetitions: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle peak identities within training prevalence/intensity quintile bins."""
    bins = np.zeros(len(stats), dtype=int)
    for key in ('prevalence', 'positive_mean'):
        values = stats[key].to_numpy()
        bins = bins * 5 + np.searchsorted(np.quantile(values, [.2, .4, .6, .8]), values, side='right')
    rng = np.random.default_rng(SEED)
    permutations = np.tile(np.arange(len(stats)), (repetitions, 1))
    for indices in (np.flatnonzero(bins == code) for code in np.unique(bins)):
        for row in permutations:
            row[indices] = rng.permutation(indices)
    return permutations, bins


def geometry(weights: np.ndarray, labels: np.ndarray, pairs: np.ndarray, k: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Separate centered rank, cosine structure, family separation and hubness."""
    x = np.asarray(weights, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    normalized = x / np.maximum(norms[:, None], 1e-30)
    similarity = np.clip((normalized @ normalized.T).astype(np.float32), -1, 1)
    neighbors = nearest_indices(similarity, k)
    incoming = np.bincount(neighbors.ravel(), minlength=len(x))
    sorted_incoming = np.sort(incoming)
    gini = np.sum((2*np.arange(1, len(x)+1)-len(x)-1)*sorted_incoming)/(len(x)*sorted_incoming.sum())
    covariance = covariance_metrics(x, {})
    sampled = similarity[pairs[:, 0], pairs[:, 1]]
    within = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    distances = np.clip(1-similarity, 0, 2)
    np.fill_diagonal(distances, 0)
    sizes = np.unique(labels, return_counts=True)[1]
    result = dict(peaks=len(x), dimensions=x.shape[1], families=len(sizes),
        norm_mean=norms.mean(), norm_cv=norms.std()/max(norms.mean(), 1e-30),
        norm_min=norms.min(), norm_max=norms.max(), zero_norm_peaks=int((norms == 0).sum()),
        centered_participation_ratio=covariance['participation_ratio'],
        effective_rank_fraction=covariance['participation_ratio']/x.shape[1],
        offset_energy_fraction=covariance['shared_offset_energy_fraction'],
        **variance_component_counts(covariance['eigenvalues']),
        mean_pair_cosine=float(sampled.mean()), pair_cosine_std=float(sampled.std()),
        within_family_cosine=float(sampled[within].mean()), between_family_cosine=float(sampled[~within].mean()),
        silhouette_cosine=float(silhouette_score(distances, labels, metric='precomputed')),
        hubness_gini=float(gini), max_neighbor_indegree=int(incoming.max()),
        largest_family_fraction=float(sizes.max()/len(x)),
        effective_family_count=float(np.exp(-np.sum(sizes/len(x)*np.log(sizes/len(x))))))
    return result, neighbors, sampled, family_similarity(similarity, labels)


def measured_family_signals(matrix: sparse.csr_matrix, labels: dict, data: dict) -> tuple[dict, np.ndarray]:
    """Transform measured spectra once and compute bounded family sums for all cuts."""
    matrix = canonical_block(matrix)
    matrix.data[matrix.data <= max(0, data['nonzero_threshold'])] = 0
    matrix.eliminate_zeros()
    values = matrix.data.astype(np.float32)
    if data['intensity_transform'] == 'log1p':
        values = np.log1p(values)
    elif data['intensity_transform'] == 'sqrt':
        values = np.sqrt(values)
    elif data['intensity_transform'] != 'none':
        raise ValueError('Unsupported spectral transformation.')
    if data['intensity_clip_max'] is not None:
        values = np.minimum(values, data['intensity_clip_max'])
    matrix.data = values.astype(np.float64)
    controls = np.column_stack((np.asarray(matrix.sum(1)).ravel(), np.diff(matrix.indptr)))
    mappings, sizes, boundaries = [], [], [0]
    for family in labels.values():
        _, codes = np.unique(family, return_inverse=True)
        size = np.bincount(codes)
        sizes.extend(size)
        boundaries.append(boundaries[-1]+len(size))
        mappings.append(sparse.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)),
                                         shape=(len(codes), len(size))))
    mapping = sparse.hstack(mappings, format='csr')
    signals = {key:np.empty((matrix.shape[0], boundaries[i+1]-boundaries[i]), dtype=np.float32)
               for i, key in enumerate(labels)}
    for start in range(0, matrix.shape[0], 4096):
        stop = min(start+4096, matrix.shape[0])
        block = (matrix[start:stop] @ mapping).toarray()/np.asarray(sizes)
        for i, key in enumerate(labels):
            signals[key][start:stop] = block[:, boundaries[i]:boundaries[i+1]]
    return signals, controls


def family_associations(signals: np.ndarray, targets: np.ndarray, slides: np.ndarray,
                        controls: np.ndarray, minimum: int = 20) -> dict:
    """Use identical validation rows for within-slide raw and partial associations."""
    names = np.unique(np.asarray(slides, dtype=str))
    shape = (len(names), signals.shape[1], targets.shape[1])
    raw, partial = np.full(shape, np.nan), np.full(shape, np.nan)
    signal_corr, partial_signal_corr = [], []
    for i, slide in enumerate(names):
        selected = slides == slide
        x = signals[selected]
        c = controls[selected]
        ranks = rankdata(x.astype(float), axis=0)
        ranks -= ranks.mean(0)
        nuisance = rankdata(c.astype(float), axis=0)
        nuisance -= nuisance.mean(0)
        residual = ranks - nuisance @ np.linalg.lstsq(nuisance, ranks, rcond=None)[0]
        for values, collection in [(ranks, signal_corr), (residual, partial_signal_corr)]:
            norms = np.linalg.norm(values, axis=0)
            denom = norms[:, None]*norms[None, :]
            collection.append(np.divide(values.T@values, denom, out=np.full(denom.shape, np.nan), where=denom > 1e-12))
        for j in range(targets.shape[1]):
            y = targets[selected, j]
            keep = np.isfinite(y)
            if keep.sum() >= minimum and np.ptp(y[keep]) > 0:
                raw[i, :, j] = rank_correlations(x[keep], y[keep])
                partial[i, :, j] = rank_correlations(x[keep], y[keep], c[keep])
    return dict(slides=names, raw=raw, partial=partial,
        signal_correlation=mean_finite(np.asarray(signal_corr)),
        partial_signal_correlation=mean_finite(np.asarray(partial_signal_corr)))


def family_scores(values: np.ndarray, sizes: np.ndarray) -> dict:
    """Report family-balanced and member-weighted association strength separately."""
    means = mean_finite(values)
    sign = np.sign(means)
    agreement = mean_finite(np.where(np.isfinite(values), np.sign(values)==sign, np.nan))
    return dict(mean_abs_slide_mean_rho=mean_finite(np.abs(means)),
        size_weighted_mean_abs_rho=np.nansum(np.abs(means)*sizes[:, None], axis=0)/np.sum(np.isfinite(means)*sizes[:, None], axis=0),
        mean_sign_agreement=mean_finite(agreement), max_abs_slide_mean_rho=np.nanmax(np.abs(means), axis=0))


def patient_bootstrap(associations: dict, labels: dict, slide_patients: np.ndarray) -> pd.DataFrame:
    """Paired patient intervals for family strength, conditional on the fitted cuts."""
    patients, codes = np.unique(slide_patients, return_inverse=True)
    rng = np.random.default_rng(SEED)
    draws = np.stack([np.bincount(rng.integers(len(patients), size=len(patients)), minlength=len(patients)) for _ in range(1000)])
    estimates, records = {}, []
    for key, item in associations.items():
        sizes = np.unique(labels[key], return_counts=True)[1]
        for kind in ('raw', 'partial'):
            values = item[kind]
            sums = np.stack([np.nansum(values[codes == i], axis=0) for i in range(len(patients))])
            counts = np.stack([np.isfinite(values[codes == i]).sum(0) for i in range(len(patients))])
            total = np.einsum('bp,pft->bft', draws, sums)
            count = np.einsum('bp,pft->bft', draws, counts)
            means = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
            points = family_scores(values, sizes)
            for metric, weights in [('mean_abs_slide_mean_rho', np.ones(len(sizes))), ('size_weighted_mean_abs_rho', sizes)]:
                boot = np.nansum(np.abs(means)*weights[None, :, None], axis=1)/np.sum(np.isfinite(means)*weights[None, :, None], axis=1)
                estimates[key, kind, metric] = (points[metric], boot)
                for j, target in enumerate(TARGETS):
                    lo, hi = np.nanquantile(boot[:, j], [.025, .975])
                    records.append(dict(representation=key, reference='absolute', association=kind, metric=metric,
                        target=target, estimate=points[metric][j], lower=lo, upper=hi, patients=len(patients)))
    comparisons = [('l1_raw','baseline_raw'), ('l2_raw','baseline_raw'), ('l2_raw','l1_raw'),
                   ('l1_effective','baseline_response_mid'), ('l2_effective','baseline_response_mid'),
                   ('l2_effective','l1_effective'),
                   *[(f'{arm}_effective', f'baseline_response_{level}')
                     for arm in ('l1','l2') for level in ('low','high')]]
    for key, reference in comparisons:
        for kind in ('raw', 'partial'):
            for metric in ('mean_abs_slide_mean_rho', 'size_weighted_mean_abs_rho'):
                point, boot = estimates[key, kind, metric]
                base, base_boot = estimates[reference, kind, metric]
                for j, target in enumerate(TARGETS):
                    lo, hi = np.nanquantile(boot[:, j]-base_boot[:, j], [.025, .975])
                    records.append(dict(representation=key, reference=reference, association=kind, metric=metric,
                        target=target, estimate=point[j]-base[j], lower=lo, upper=hi, patients=len(patients)))
    return pd.DataFrame(records)


def family_profile_r2(profile: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of between-peak association variance explained by a family cut."""
    keep = np.isfinite(profile)
    _, codes = np.unique(labels[keep], return_inverse=True)
    y = profile[keep]
    means = np.bincount(codes, weights=y)/np.bincount(codes)
    denominator = np.sum((y-y.mean())**2)
    return float(1-np.sum((y-means[codes])**2)/denominator) if denominator > 0 else np.nan


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Compare centered geometry across widths without arbitrary axis alignment."""
    a = left.astype(float)-left.mean(0, dtype=float)
    b = right.astype(float)-right.mean(0, dtype=float)
    denominator = np.sqrt(np.square(a.T@a).sum()*np.square(b.T@b).sum())
    return float(np.square(a.T@b).sum()/denominator) if denominator else np.nan


def neighbor_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Mean peak-wise neighbor-set overlap."""
    return float(np.mean([len(set(a)&set(b))/len(set(a)|set(b)) for a,b in zip(left,right)]))


def run(comparison: Path) -> Path:
    """Validate saved artifacts, evaluate peak vectors, and write a linked report."""
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        return _run(comparison.resolve())


def verify_vector_exports(root: Path, manifest: dict, vectors: dict, initial: dict) -> dict:
    """Recompute raw and transformed vectors from frozen checkpoint weights on CPU."""
    from dann.checkpoints import checkpoint_config
    from dann.model import AdversarialLatentFusion
    initial_state = torch.load(root/'diagnostics/l1/initial_parameters.pt', map_location='cpu', weights_only=True)
    quantiles = json.loads((root/'diagnostics/baseline/metrics.json').read_text())['training_intensity_quantiles']
    levels = dict(low=quantiles[0], mid=(quantiles[0]+quantiles[2])/2, high=quantiles[2])
    records = {}
    for arm in ARMS:
        path = Path(manifest['runs'][arm]['checkpoint'])
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint_config(checkpoint)
        with torch.random.fork_rng(devices=[]):
            model = AdversarialLatentFusion.from_config(config, len(checkpoint['batch_names']), 4).eval()
        for phase, state, expected in [('selected', checkpoint['model_state'], vectors), ('initial', initial_state, initial)]:
            model.load_state_dict(state)
            np.testing.assert_array_equal(expected[f'{arm}_raw'], model.encoder.embedding.weight.detach().numpy())
            with torch.no_grad():
                if arm == 'baseline':
                    enc = model.encoder
                    reference = enc.peak_mlp(enc.embedding.weight.new_zeros((1, enc.embedding_dim)))
                    transformed = {f'baseline_response_{level}':enc.peak_mlp(enc.embedding.weight*value)-reference
                                   for level, value in levels.items()}
                else:
                    transformed = {f'{arm}_effective':model.encoder.effective_dictionary()}
            for key, value in transformed.items():
                array = value.numpy()
                np.testing.assert_allclose(expected[key], array, atol=2e-5, rtol=2e-5)
                records[f'{phase}_{key}'] = dict(max_absolute_difference=float(np.max(np.abs(expected[key]-array))),
                    rows=len(array), checkpoint_sha256=digest_file(path) if phase=='selected' else digest_file(root/'diagnostics/l1/initial_parameters.pt'))
    return dict(atol=2e-5, rtol=2e-5, actual_transformed_intensity_levels=levels, checks=records)


def _run(root: Path) -> Path:
    manifest = manifest_for(root)
    out = root/'peak_comparison'
    out.mkdir(exist_ok=True)
    (out/'arrays').mkdir(exist_ok=True)
    source_hashes, peak_dirs, frames, runs = {}, {}, {}, {}
    for arm in ARMS:
        directory = Path(manifest['runs'][arm]['checkpoint']).parent.parent/'analysis/peaks'
        peak_dirs[arm] = directory
        runs[arm] = json.loads((directory/'peak_run.json').read_text())
        assert runs[arm]['source_identity']['checkpoint'] == digest_file(Path(manifest['runs'][arm]['checkpoint']))
        assert runs[arm]['correlation_split'] == 'validation'
        assert runs[arm]['actual_family_count'] == 32
        for name in ('peak_embeddings', 'peak_annotations', 'family_membership', 'family_slide_ihc_associations',
                     'peak_slide_ihc_associations', 'peak_slide_statistics'):
            path = directory/'tables'/f'{name}.csv'
            source_hashes[str(path)] = digest_file(path)
        frames[arm] = pd.read_csv(directory/'tables/peak_annotations.csv')
        np.testing.assert_array_equal(frames[arm].peak_index, np.arange(5808))
        np.testing.assert_array_equal(frames[arm].mz, frames['baseline'].mz)
        for name in ('peak_slide_ihc_associations','peak_slide_statistics'):
            assert source_hashes[str(directory/'tables'/f'{name}.csv')] == source_hashes[str(peak_dirs['baseline']/'tables'/f'{name}.csv')]
        assert runs[arm]['source_identity']['identity'] == manifest['data_identity']
        assert runs[arm]['preprocessing'] == runs['baseline']['preprocessing']
        assert runs[arm]['target_transform'] == runs['baseline']['target_transform']
    vectors, initial, labels, descriptions = {}, {}, {}, {}
    initial_path = root/'diagnostics/l1/initial_parameters.pt'
    initial_raw = torch.load(initial_path, map_location='cpu', weights_only=True)['encoder.embedding.weight'].numpy()
    source_hashes[str(initial_path)] = digest_file(initial_path)
    for arm in ARMS:
        table = pd.read_csv(peak_dirs[arm]/'tables/peak_embeddings.csv', float_precision='round_trip')
        key = f'{arm}_raw'
        vectors[key] = table.filter(regex='^embedding_').to_numpy(dtype=np.float32)
        initial[key] = initial_raw
        labels[key] = frames[arm].family.to_numpy()
        descriptions[key] = 'Raw learned identity embedding, 128 dimensions; original 32-family cut.'
        checkpoint = torch.load(manifest['runs'][arm]['checkpoint'], map_location='cpu', weights_only=False)
        np.testing.assert_array_equal(vectors[key], checkpoint['model_state']['encoder.embedding.weight'].numpy())
    paths = sorted((root/'diagnostics/baseline').glob('selected_conditioned_response_*.npy'),
                   key=lambda p:float(p.stem.split('_')[-1]))
    for level, path in zip(('low','mid','high'), paths):
        key = f'baseline_response_{level}'
        vectors[key] = np.load(path)
        initial_file = path.with_name(path.name.replace('selected_', 'initial_reconstructed_'))
        initial[key] = np.load(initial_file)
        descriptions[key] = f'Historical zero-input-centered peak response at transformed intensity {path.stem.split("_")[-1]}; not an intensity-independent dictionary.'
        for p in (path, initial_file): source_hashes[str(p)] = digest_file(p)
    for arm in ('l1','l2'):
        key = f'{arm}_effective'
        path = root/'diagnostics'/arm/'selected_effective_dictionary.npy'
        init_path = path.with_name('initial_effective_dictionary.npy')
        vectors[key], initial[key] = np.load(path), np.load(init_path)
        descriptions[key] = 'Mean-centered transformed dictionary actually used in sparse pooling, 256 dimensions.'
        for p in (path, init_path): source_hashes[str(p)] = digest_file(p)
    assert len(vectors) == 8
    save_json(out/'vector_provenance.json', verify_vector_exports(root, manifest, vectors, initial))
    c = load_config(Path(manifest['runs']['baseline']['config']))
    checkpoint = torch.load(manifest['runs']['baseline']['checkpoint'], map_location='cpu', weights_only=False)
    frame, features = read_peak_metadata(Path(c['data']['path']), c['data'], 'Lesion')
    rows = np.asarray(checkpoint['split_indices']['validation'])
    rows = rows[frame.Lesion.to_numpy()[rows]]
    assert len(rows) == 169861 == runs['baseline']['correlation_lesion_rows']
    np.testing.assert_array_equal(features.astype(float), frames['baseline'].mz)
    targets, valid = prepare_observed_targets(frame[list(TARGETS)].to_numpy(), list(TARGETS), c['data'].get('cd8_normalization'))
    targets = np.where(valid[rows], targets[rows], np.nan)
    slides = frame[c['data']['batch_column']].astype(str).to_numpy()[rows]
    with h5py.File(c['data']['path']) as handle:
        patients = np.asarray(read_elem(handle['obs/patient']), dtype=str)[rows]
    slide_names = np.unique(slides)
    slide_patients = []
    for slide in slide_names:
        members = np.unique(patients[slides == slide])
        assert len(members) == 1
        slide_patients.append(members[0])
    print('Loading common measured spectra and training-only peak covariates', flush=True)
    matrix = resident_matrix(Path(c['data']['path']), c['data'], {'host_budget_gib':128.})
    train_rows = np.load(root/'probe_train_rows.npy')
    train = canonical_block(matrix[train_rows])
    counts = np.bincount(train.indices, minlength=len(features))
    sums = np.bincount(train.indices, weights=train.data, minlength=len(features))
    training_stats = pd.DataFrame(dict(peak_index=np.arange(len(features)), mz=features.astype(float),
        prevalence=counts/len(train_rows), positive_mean=np.divide(sums, counts, out=np.zeros_like(sums), where=counts>0)))
    del train
    matrix = matrix[rows].copy()
    training_stats.to_csv(out/'training_peak_covariates.csv', index=False)
    permutations, bins = matched_permutations(training_stats)
    rng = np.random.default_rng(SEED)
    pairs = rng.integers(len(features), size=(240000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]][:200000]
    np.save(out/'arrays/common_peak_pairs.npy', pairs)
    np.save(out/'arrays/matched_permutations.npy', permutations)
    protocol = dict(version=1, seed=SEED, neighbors=15, families=32, family_method='Average-linkage cosine, original cuts for raw embeddings and fresh cuts for transformed responses.',
        validation_rows=len(rows), validation_rows_sha256=fingerprint(rows), training_covariate_rows=len(train_rows),
        training_covariate_rows_sha256=fingerprint(train_rows), peaks=len(features), slides=len(slide_names), patients=len(np.unique(patients)),
        pair_samples=len(pairs), permutations=100, permutation_control='Training prevalence and positive-intensity quintile combinations; identities shuffled within bins.',
        permutation_bin_counts=np.bincount(bins).tolist(), minimum_profile_contributing_slides=20,
        bootstrap=dict(unit='patient', replicates=1000, seed=SEED, training_seed_uncertainty=False),
        partial_correlations='Within-slide Spearman residualized against ranks of transformed total intensity and positive distinct peak count. This may remove biological abundance information; it is a sensitivity control.',
        family_signal='Mean transformed measured intensity over all family members, with absent peaks zero; identical preprocessing across models.',
        descriptions=descriptions, sources=source_hashes)
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Peak comparison inputs or protocol changed.'
    else:
        with protocol_path.open('x') as handle:json.dump(protocol, handle, indent=2)
    neighbors, pair_values, family_cosines, geometry_rows = {}, {}, {}, []
    profiles = frames['baseline'][[f'spearman_{t}' for t in TARGETS]].to_numpy()
    for j, target in enumerate(TARGETS):
        profiles[frames['baseline'][f'contributing_slides_{target}'].to_numpy()<20, j] = np.nan
    coherence_rows, family_profile_rows = [], []
    for key, weights in vectors.items():
        print('Geometry and matched controls:', key, flush=True)
        if key not in labels:
            labels[key] = family_hierarchy(weights, 32)['families']
        result, neighbors[key], pair_values[key], family_cosines[key] = geometry(weights, labels[key], pairs, 15)
        norm = np.linalg.norm(weights, axis=1)
        result.update(representation=key, norm_prevalence_spearman=spearmanr(norm, training_stats.prevalence).statistic,
            norm_intensity_spearman=spearmanr(norm, training_stats.positive_mean).statistic,
            norm_mz_spearman=spearmanr(norm, training_stats.mz).statistic,
            initialization_linear_cka=linear_cka(weights, initial[key]))
        normalized = initial[key]/np.maximum(np.linalg.norm(initial[key], axis=1, keepdims=True), 1e-30)
        initial_neighbors = nearest_indices(normalized@normalized.T, 15)
        result['initialization_neighbor_jaccard'] = neighbor_jaccard(neighbors[key], initial_neighbors)
        geometry_rows.append(result)
        np.savez_compressed(out/'arrays'/f'{key}.npz', vectors=weights, families=labels[key],
            neighbors=neighbors[key], sampled_cosines=pair_values[key], family_cosines=family_cosines[key])
        for j, target in enumerate(TARGETS):
            profile = profiles[:, j]
            mae, score = neighbor_coherence(profile, neighbors[key])
            family_score = family_profile_r2(profile, labels[key])
            null = np.array([neighbor_coherence(profile[p], neighbors[key]) for p in permutations])
            null_family = np.array([family_profile_r2(profile[p], labels[key]) for p in permutations])
            coherence_rows.append(dict(representation=key, target=target, eligible_peaks=int(np.isfinite(profile).sum()),
                neighbor_profile_mae=mae, neighbor_profile_r2=score, null_mean_r2=np.nanmean(null[:, 1]),
                null_r2_05=np.nanquantile(null[:, 1], .05), null_r2_95=np.nanquantile(null[:, 1], .95),
                excess_r2_over_matched_null=score-np.nanmean(null[:, 1])))
            family_profile_rows.append(dict(representation=key, target=target, family_profile_r2=family_score,
                null_mean_r2=np.nanmean(null_family), null_r2_05=np.nanquantile(null_family, .05),
                null_r2_95=np.nanquantile(null_family, .95), excess_r2_over_matched_null=family_score-np.nanmean(null_family)))
    mz = features.astype(float)
    mz_neighbors = nearest_indices(-np.abs(mz[:, None]-mz[None, :]), 15)
    for j, target in enumerate(TARGETS):
        mae, score = neighbor_coherence(profiles[:, j], mz_neighbors)
        coherence_rows.append(dict(representation='mz_neighbors_control', target=target,
            eligible_peaks=int(np.isfinite(profiles[:, j]).sum()), neighbor_profile_mae=mae, neighbor_profile_r2=score))
    pair_rows, matching = [], []
    keys = list(vectors)
    for i, left in enumerate(keys):
        for right in keys[i+1:]:
            measures, match = family_match(labels[left], labels[right])
            pair_rows.append(dict(left=left, right=right, **measures,
                pair_cosine_spearman=spearmanr(pair_values[left], pair_values[right]).statistic,
                neighbor_jaccard=neighbor_jaccard(neighbors[left], neighbors[right]),
                linear_cka=linear_cka(vectors[left], vectors[right])))
            matching.append(match.assign(left=left, right=right))
    print('Computing common-row measured family signals', flush=True)
    signals, controls = measured_family_signals(matrix, labels, c['data'])
    del matrix
    association_arrays, family_rows, summaries = {}, [], []
    for key, signal in signals.items():
        print('Held-out family and abundance-controlled correlations:', key, flush=True)
        association_arrays[key] = family_associations(signal, targets, slides, controls)
        item = association_arrays[key]
        np.testing.assert_array_equal(item['slides'], slide_names)
        unique, sizes = np.unique(labels[key], return_counts=True)
        if key.endswith('_raw'):
            arm = key.removesuffix('_raw')
            expected = pd.read_csv(peak_dirs[arm]/'tables/family_slide_ihc_associations.csv')
            for j, target in enumerate(TARGETS):
                reference = expected[expected.target==target].pivot(index='slide', columns='family', values='spearman').loc[slide_names, unique].to_numpy()
                np.testing.assert_allclose(item['raw'][:, :, j], reference, rtol=0, atol=1e-10, equal_nan=True)
        np.savez_compressed(out/'arrays'/f'{key}_family_associations.npz', **item)
        pd.DataFrame(item['signal_correlation'], index=unique, columns=unique).to_csv(out/f'{key}_family_signal_correlations.csv', index_label='family')
        pd.DataFrame(item['partial_signal_correlation'], index=unique, columns=unique).to_csv(out/f'{key}_family_signal_partial_correlations.csv', index_label='family')
        off_diagonal = ~np.eye(len(unique), dtype=bool)
        for kind in ('raw', 'partial'):
            scores = family_scores(item[kind], sizes)
            means = mean_finite(item[kind])
            for j, target in enumerate(TARGETS):
                summaries.append(dict(representation=key, association=kind, target=target,
                    **{metric:values[j] for metric,values in scores.items()},
                    mean_abs_family_signal_correlation=float(np.nanmean(np.abs(item['signal_correlation' if kind=='raw' else 'partial_signal_correlation'][off_diagonal])))))
                for f, family in enumerate(unique):
                    family_rows.append(dict(representation=key, association=kind, family=family, size=sizes[f],
                        target=target, mean_slide_spearman=means[f, j], contributing_slides=int(np.isfinite(item[kind][:, f, j]).sum()),
                        sign_agreement=float(np.mean(np.sign(item[kind][:, f, j][np.isfinite(item[kind][:, f, j])])==np.sign(means[f, j])))))
    tables = dict(geometry=pd.DataFrame(geometry_rows), neighborhood_coherence=pd.DataFrame(coherence_rows),
        family_profile_coherence=pd.DataFrame(family_profile_rows), cross_representation=pd.DataFrame(pair_rows),
        family_matches=pd.concat(matching, ignore_index=True), family_associations=pd.DataFrame(family_rows),
        family_summary=pd.DataFrame(summaries),
        paired_patient_intervals=patient_bootstrap(association_arrays, labels, np.asarray(slide_patients)))
    for name, table in tables.items():table.to_csv(out/f'{name}.csv', index=False)
    pd.DataFrame(dict(peak_index=np.arange(len(features)), mz=features.astype(float), **labels)).to_csv(out/'all_family_memberships.csv', index=False)
    render(out, tables, family_cosines, labels)
    save_json(out/'completion.json', dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
        source=str(Path(__file__).resolve()), source_sha256=digest_file(Path(__file__)),
        protocol_sha256=digest_file(protocol_path), raw_family_correlations_reproduce_saved_exports=True,
        tables={name:digest_file(out/f'{name}.csv') for name in tables}, report_sha256=digest_file(out/'report.html')))
    print('Completed peak comparison:', out/'report.html', flush=True)
    return out


def render(out: Path, tables: dict, family_cosines: dict, labels: dict) -> None:
    """Render common-scale comparisons and an offline report with explicit limits."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = out/'figures'
    folder.mkdir(exist_ok=True)
    short = dict(baseline_raw='Baseline raw', l1_raw='L1 raw', l2_raw='L2 raw',
        baseline_response_low='Baseline low I', baseline_response_mid='Baseline mid I',
        baseline_response_high='Baseline high I', l1_effective='L1 dictionary', l2_effective='L2 dictionary')
    keys = list(short)
    names = [short[k] for k in keys]
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder/f'{name}.png', dpi=180)
        fig.savefig(folder/f'{name}.pdf')
        plt.close(fig)
    def alignment(base, key):
        if base == key:return np.arange(32)
        _, match = family_match(labels[base], labels[key])
        return match.sort_values('left_family').right_family.to_numpy()-1
    geometry_frame = tables['geometry'].set_index('representation').loc[keys]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, metric, label in zip(axes.ravel(),
            ['effective_rank_fraction','silhouette_cosine','largest_family_fraction','hubness_gini'],
            ['Centered participation ratio / dimension','Cosine silhouette of 32-family cut','Fraction in largest family','Neighbor indegree Gini (hubness)']):
        ax.bar(np.arange(len(keys)), geometry_frame[metric])
        ax.set(xticks=np.arange(len(keys)), xticklabels=names, ylabel=label)
        ax.tick_params(axis='x', labelrotation=45)
    save(fig, 'vector_geometry')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    raw = ['baseline_raw','l1_raw','l2_raw']
    limit = max(np.max(np.abs(family_cosines[k])) for k in raw)
    for ax, key in zip(axes, raw):
        order = alignment('baseline_raw', key)
        display = ax.imshow(family_cosines[key][np.ix_(order, order)], vmin=-limit, vmax=limit, cmap='RdBu_r')
        fig.colorbar(display, ax=ax, fraction=.046, pad=.04)
        ax.set(title=short[key], xlabel='Matched family position', ylabel='Matched family position')
    fig.suptitle('Raw family mean cosine; overlap alignment does not imply identical membership')
    save(fig, 'raw_family_cosines')
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    families = tables['family_associations']
    groups = [raw, ['baseline_response_mid','l1_effective','l2_effective']]
    selected = families[(families.association=='raw')&families.representation.isin(sum(groups, []))]
    limit = np.nanmax(np.abs(selected.mean_slide_spearman))
    for row, group in enumerate(groups):
        for ax, key in zip(axes[row], group):
            values = selected[selected.representation==key].pivot(index='family', columns='target', values='mean_slide_spearman').reindex(columns=TARGETS).to_numpy()
            order = alignment(group[0], key)
            display = ax.imshow(values[order], vmin=-limit, vmax=limit, cmap='RdBu_r', aspect='auto')
            fig.colorbar(display, ax=ax, fraction=.046, pad=.04)
            ax.set(title=short[key], xticks=np.arange(4), xticklabels=['CD8','Tumor','Collagen','Stroma'], ylabel='Matched family position')
    fig.suptitle('Measured family signal vs IHC: mean within-slide validation Spearman; each row aligned separately')
    save(fig, 'family_ihc_correlations')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    summary = tables['family_summary']
    for ax, target in zip(axes.ravel(), TARGETS):
        for offset, kind in [(-.18,'raw'),(.18,'partial')]:
            values = summary[(summary.target==target)&(summary.association==kind)].set_index('representation').loc[keys]
            ax.bar(np.arange(len(keys))+offset, values.size_weighted_mean_abs_rho, width=.36, label=kind)
        ax.set(title=target, xticks=np.arange(len(keys)), xticklabels=names,
               ylabel='Member-weighted |mean slide Spearman|')
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
    save(fig, 'family_association_controls')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, table, metric, title in [
        (axes[0], tables['neighborhood_coherence'], 'excess_r2_over_matched_null', 'Neighbor association coherence above matched null'),
        (axes[1], tables['family_profile_coherence'], 'excess_r2_over_matched_null', 'Family association coherence above matched null')]:
        values = table[table.representation.isin(keys)].pivot(index='representation', columns='target', values=metric).reindex(index=keys, columns=TARGETS).to_numpy()
        limit = np.nanmax(np.abs(values))
        ax.imshow(values, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        for i in range(len(keys)):
            for j in range(4):ax.text(j, i, f'{values[i,j]:.2f}', ha='center', va='center', fontsize=8)
        ax.set(title=title, yticks=np.arange(len(keys)), yticklabels=names, xticks=np.arange(4), xticklabels=['CD8','Tumor','Collagen','Stroma'])
    save(fig, 'biological_coherence')
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    cross = tables['cross_representation']
    for ax, metric, title in zip(axes, ['pair_cosine_spearman','neighbor_jaccard'], ['Similarity-matrix Spearman','15-neighbor Jaccard']):
        values = np.eye(len(keys))
        for row in cross.itertuples():
            i, j = keys.index(row.left), keys.index(row.right)
            values[i,j] = values[j,i] = getattr(row, metric)
        ax.imshow(values, vmin=0, vmax=1, cmap='viridis')
        for i in range(len(keys)):
            for j in range(len(keys)):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if values[i,j]<.5 else 'black')
        ax.set(title=title, xticks=np.arange(len(keys)), xticklabels=names, yticks=np.arange(len(keys)), yticklabels=names)
        ax.tick_params(axis='x', labelrotation=60)
    save(fig, 'cross_model_similarity')
    top = families.loc[families.groupby(['representation','association','target']).mean_slide_spearman.apply(lambda x:x.abs().nlargest(3).index).explode().astype(int)]
    top.to_csv(out/'top_families_exploratory.csv', index=False)
    intervals = tables['paired_patient_intervals']
    findings = []
    for right in ('l1_raw', 'l2_raw'):
        row = cross[(cross.left=='baseline_raw')&(cross.right==right)].iloc[0]
        findings.append(f'{short[right]} versus Baseline raw: cosine-matrix Spearman {row.pair_cosine_spearman:.3f}, '
            f'15-neighbor Jaccard {row.neighbor_jaccard:.1%}, family adjusted Rand {row.adjusted_rand:.3f}, '
            f'and only {row.matched_peak_fraction:.1%} of peaks in their matched family. Family IDs are not interchangeable.')
    findings.append(f'All eight 32-family cuts have negative cosine silhouette scores '
        f'({geometry_frame.silhouette_cosine.min():.3f} to {geometry_frame.silhouette_cosine.max():.3f}). '
        'This supports treating these cuts as exploratory summaries of overlapping continuous geometry, not sharply separated biological classes.')
    for arm in ('l1','l2'):
        for suffix, reference in [('raw','baseline_raw'),('effective','baseline_response_mid')]:
            key = f'{arm}_{suffix}'
            row = intervals[(intervals.representation==key)&(intervals.reference==reference)&
                (intervals.association=='partial')&(intervals.metric=='size_weighted_mean_abs_rho')&(intervals.target=='Density_Stroma')].iloc[0]
            verdict = 'higher' if row.lower>0 else 'lower' if row.upper<0 else 'unresolved'
            findings.append(f'{short[key]} versus {short[reference]}: abundance-controlled, member-weighted stromal family association '
                f'difference {row.estimate:+.4f}, 95% paired patient interval [{row.lower:+.4f}, {row.upper:+.4f}] ({verdict}).')
    coherence = tables['neighborhood_coherence']
    for target in ('Density_CD8', 'Density_Stroma'):
        records = coherence[coherence.target==target].set_index('representation')
        findings.append(f'{target} neighbor-profile reconstruction R²: '
            + ', '.join(f'{short[key]} {records.loc[key,"neighbor_profile_r2"]:.3f}'
                        for key in ('baseline_response_mid','l1_effective','l2_effective'))
            + f'; the unlearned m/z-neighbor control gives {records.loc["mz_neighbors_control","neighbor_profile_r2"]:.3f}. '
            'These descriptive point differences have no training-seed confidence interval.')
    findings.append('The learned neighbors outperform the m/z-neighbor control for CD8 association coherence in this experiment, '
        'but m/z neighbors are stronger for stroma, collagen and tumor. Positive lift over prevalence/intensity-matched shuffles '
        'therefore does not establish superiority to every relevant control.')
    for key in ('baseline_response_mid','l1_effective','l2_effective'):
        values = tables['family_summary'][(tables['family_summary'].representation==key)&(tables['family_summary'].target=='Density_Stroma')].set_index('association')
        findings.append(f'{short[key]} mean absolute family-to-family signal correlation falls from '
            f'{values.loc["raw","mean_abs_family_signal_correlation"]:.3f} to '
            f'{values.loc["partial","mean_abs_family_signal_correlation"]:.3f} after abundance adjustment. '
            'A substantial part of family co-movement is associated with shared spectral abundance.')
    findings.append('Raw identity embeddings and transformed pooling vectors answer different questions. The historical encoder has no single intensity-independent dictionary; its three response levels span the training 1–99% intensity range.')
    findings.append('Initialization comparisons use the saved common seed-20260719 initialization of the new arms. For the historical baseline this is a reconstructed reference, not its observed original initialization or training trajectory.')
    findings.append('Families are learned geometric groups, not identified molecules or cell types. Correlations and neighbor-profile reconstruction are descriptive internal-validation diagnostics. Low-intensity peaks are not assumed to be noise; controlling abundance can remove real biological information.')
    findings.append('Family-balanced summaries give each family equal weight; member-weighted summaries account for unequal family sizes. Top-family rankings and the many comparisons are exploratory; intervals are not multiplicity-adjusted. Patient intervals condition on one training seed and fixed family cuts; they do not cover seed or clustering-selection uncertainty. All three historical intensity levels are included in the interval table.')
    findings.append('Constant family signals have undefined correlations, retained as missing rather than zero. Summaries use finite contributing slides, with counts provided for every family and target.')
    findings.append('Neighbor/family coherence nulls shuffle IHC-association profiles within training prevalence/intensity bins. They are matched reference distributions, not independent molecular-validation tests. The m/z-neighbor control is provided separately.')
    save_json(out/'summary.json', dict(findings=findings, protocol='protocol.json', tables=list(tables),
        scope='Same 5,808 peaks and 169,861 validation lesion rows; 43 slides nested in 20 patients; no retraining.'))
    def table(frame):return frame.to_html(index=False, float_format=lambda x:f'{x:.4g}', border=0, escape=True)
    useful = geometry_frame.reset_index()[['representation','dimensions','centered_participation_ratio','pcs_90','pcs_99',
        'silhouette_cosine','largest_family_fraction','effective_family_count','hubness_gini','norm_prevalence_spearman',
        'initialization_linear_cka','initialization_neighbor_jaccard']]
    figures = ''.join(f'<figure><img src="figures/{p.name}" alt="{p.stem}"><figcaption><a href="figures/{p.stem}.pdf">PDF</a></figcaption></figure>' for p in sorted(folder.glob('*.png')))
    links = ''.join(f'<li><a href="{name}.csv">{name.replace("_"," ")}</a></li>' for name in tables)
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Peak-vector and family comparison</title>
<style>body{{font:16px system-ui;max-width:1400px;margin:2em auto;padding:0 1em;color:#17212b}}table{{border-collapse:collapse;display:block;overflow-x:auto}}td,th{{padding:.45em;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}a{{color:#145a96}}</style>
<h1>Peak vectors, similarities and family biology</h1><p><a href="../report.html">Full encoder comparison</a> · <a href="protocol.json">Frozen protocol and input hashes</a> · <a href="vector_provenance.json">Checkpoint vector reproduction checks</a></p>
<p>Common vocabulary: 5,808 peaks. Family signals use all 169,861 validation lesion rows, with correlations computed within each of 43 slides. Patient resampling uses 20 clusters. Models and training preprocessing are unchanged.</p>
{''.join(f'<p>{html.escape(text)}</p>' for text in findings)}
<h2>Geometry and family structure</h2><p>Silhouette, rank, norms, and hubness are geometric descriptions, not stand-alone biological-quality scores. Raw vectors have 128 coordinates; pooling responses have 256.</p>{table(useful)}
<h2>Family associations</h2><p>Raw is within-slide Spearman. Partial removes rank-linear effects of total transformed intensity and positive peak count. Mean absolute slide-mean correlation retains both positive and negative associations; signed family results are in the linked tables.</p>{table(tables['family_summary'])}
<h2>Neighborhood biological coherence</h2><p>R² reconstructs a peak's held-out mean IHC association from its 15 vector neighbors. It is a descriptive association-profile score, not a new IHC prediction model. Profiles require at least 20 contributing slides.</p>{table(tables['neighborhood_coherence'])}
<h2>Paired patient differences</h2><p>Intervals are 95% percentiles from 1,000 paired patient resamples. Differences are representation minus reference. Cross-model family numbers are matched by overlap only for displays; low overlap means the labels should not be treated as equivalent.</p>{table(intervals[intervals.reference!='absolute'])}
{figures}<h2>Artifacts</h2><ul>{links}<li><a href="all_family_memberships.csv">All family memberships</a></li><li><a href="training_peak_covariates.csv">Training peak covariates</a></li><li><a href="top_families_exploratory.csv">Exploratory top family associations</a></li><li><a href="arrays">Complete neighbor indices and numerical arrays</a></li></ul></html>'''
    (out/'report.html').write_text(document)


def main() -> None:
    """Run the common-peak comparison from an existing completed experiment."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    args = parser.parse_args()
    run(args.comparison)


if __name__ == '__main__':
    main()

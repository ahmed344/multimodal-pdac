"""Frozen observed-biology neighborhoods and local conditional alignment estimates.

These are local, biased finite-sample diagnostics, not an exact continuous
conditional-independence test. No learned prediction or CCA projection enters
biological weights. Pairwise matrices are shared; local HSIC uses matrix moments
rather than allocating an anchor-by-row-by-row tensor.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from dann.batch_alignment import Measurement
    from dann.data_loader import DataBundle


FEATURES = ('Density_CD8', 'Density_Tumor', 'Density_Collagen', 'Density_Stroma')
DEFAULTS = dict(enabled=False, alpha=.1, ihc_bandwidth=1., ihc_min_similarity=.05,
                minimum_candidates=2, minimum_ess=2., reference_per_slide=1024,
                he_pairs=16384, anchors_per_slide=2, reference_seed=0, anchor_seed=0)


def conditioning_settings(raw: Mapping) -> dict:
    """Validate optional biological geometry and deterministic sampling controls."""
    if raw is None or not isinstance(raw, Mapping) or raw.keys() - DEFAULTS.keys():
        raise ValueError('Invalid batch_alignment.conditioning settings.')
    settings = {**DEFAULTS, **raw}
    if not isinstance(settings['enabled'], bool):
        raise ValueError('conditioning.enabled must be boolean.')
    for key in ('alpha', 'ihc_bandwidth', 'ihc_min_similarity', 'minimum_ess'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'conditioning.{key} must be finite numeric.')
    if not 0 <= settings['alpha'] <= 1:
        raise ValueError('conditioning.alpha must be in [0, 1].')
    if settings['ihc_bandwidth'] <= 0 or not 0 < settings['ihc_min_similarity'] <= 1 or settings['minimum_ess'] < 2:
        raise ValueError('Conditioning requires positive bandwidth, threshold in (0, 1], and ESS >= 2.')
    for key, minimum, maximum in [('reference_per_slide', 2, 1024), ('he_pairs', 1, 16384),
                                  ('anchors_per_slide', 1, 2), ('minimum_candidates', 2, None),
                                  ('reference_seed', 0, None), ('anchor_seed', 0, None)]:
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum and value > maximum):
            raise ValueError(f'Invalid conditioning.{key}.')
    return settings


def conditioning_contract(config: Mapping) -> dict | None:
    """Freeze preprocessing semantics independently of model heads."""
    from dann.batch_alignment import alignment_settings, NAMES
    alignment = alignment_settings(config)
    s = alignment['conditioning']
    if not s['enabled'] or not any(alignment[n]['enabled'] for n in NAMES):
        return None
    columns = list(config['data']['target_columns'])
    if not set(FEATURES) <= set(columns):
        raise ValueError('Conditioned alignment requires observed CD8, tumor, collagen and stroma.')
    epsilon = config['loss']['logit_epsilon']
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not 0 < epsilon < .5:
        raise ValueError('Conditioning requires logit_epsilon in (0, .5).')
    from dann.targets import target_contract
    return dict(version=1, settings=s, target_transform=target_contract(config),
                feature_order=list(FEATURES), logit_epsilon=epsilon,
                he_key=config['data'].get('he_features_key', 'hes_features') if s['alpha'] else None,
                he_feature_order='stored_obsm_column_order' if s['alpha'] else None,
                training_seed=int(config['training']['seed']),
                ihc_kernel='exp(-sum_squared_scaled_differences/(2*bandwidth**2)); CD8 groups disjoint',
                he_kernel='exp(-cosine_distance/median_distinct_pair_distance); constant geometry=1')


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _rows_digest(rows) -> str:
    return hashlib.sha256(np.sort(np.asarray(rows, dtype='<i8')).tobytes()).hexdigest()


def eligible_targets(targets: np.ndarray, valid: np.ndarray | None, columns: Sequence[str]) -> np.ndarray:
    """Restrict alignment only; use the already normalized targets and validity."""
    indices = [list(columns).index(n) for n in FEATURES]
    values = np.asarray(targets)[:, indices]
    valid = np.ones_like(values, dtype=bool) if valid is None else np.asarray(valid)[:, indices]
    return (valid & np.isfinite(values) & (values >= 0) & (values <= 1)).all(1)


def _reference(config, bundle):
    contract = conditioning_contract(config)
    rows = bundle.split_indices['train']
    valid = eligible_targets(bundle.metadata.targets, bundle.metadata.target_valid_mask,
                             config['data']['target_columns'])
    seed = np.random.SeedSequence([contract['training_seed'], contract['settings']['reference_seed'], 1949])
    rng = np.random.default_rng(seed)
    eligible = rows[valid[rows]]
    chosen = []
    for slide in np.unique(bundle.metadata.batch_codes[eligible]):
        candidates = np.sort(eligible[bundle.metadata.batch_codes[eligible] == slide])
        chosen.extend(rng.choice(candidates, min(len(candidates), contract['settings']['reference_per_slide']), replace=False).tolist())
    chosen = np.asarray(sorted(chosen), dtype=np.int64)
    return chosen, int((~valid[rows]).sum())


def _reference_values(config, bundle, rows):
    columns = config['data']['target_columns']
    ihc = bundle.metadata.targets[rows][:, [columns.index(n) for n in FEATURES]].astype(np.float64)
    he = None
    if conditioning_contract(config)['settings']['alpha']:
        if bundle.he_reader is None:
            raise ValueError('H&E conditioning requires fixed UNI2-h embeddings.')
        he = bundle.he_reader.read(rows).astype(np.float64)
        norms = np.linalg.norm(he, axis=1)
        if not np.isfinite(he).all() or np.any(norms <= 0):
            raise ValueError('H&E conditioning rejects nonfinite or zero-norm vectors.')
    digest = hashlib.sha256(ihc.astype('<f8').tobytes())
    if he is not None:
        digest.update(he.astype('<f8').tobytes())
        he /= np.linalg.norm(he, axis=1, keepdims=True)
    return ihc, he, digest.hexdigest()


@dataclass
class ConditioningInput:
    """Detached, row-aligned frozen biological features for one estimator call."""
    ihc: torch.Tensor
    positive: torch.Tensor
    valid: torch.Tensor
    row_ids: torch.Tensor
    he: torch.Tensor | None
    settings: dict
    he_bandwidth: float | None

    def select(self, index) -> 'ConditioningInput':
        return ConditioningInput(*(x[index].detach() for x in (self.ihc, self.positive, self.valid, self.row_ids)),
                                 None if self.he is None else self.he[index].detach(), self.settings, self.he_bandwidth)

    @classmethod
    def concatenate(cls, parts) -> 'ConditioningInput':
        first = parts[0]
        if any(p.settings != first.settings or p.he_bandwidth != first.he_bandwidth for p in parts):
            raise ValueError('Inconsistent frozen conditioning inputs.')
        return cls(*(torch.cat([getattr(p, key).detach() for p in parts])
                     for key in ('ihc', 'positive', 'valid', 'row_ids')),
                   None if first.he is None else torch.cat([p.he.detach() for p in parts]),
                   first.settings, first.he_bandwidth)

    @torch.no_grad()
    def kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return IHC and combined similarity with no autograd edges."""
        x = self.ihc.detach().double()
        # cdist avoids cancellation of differences in uncentered scaled densities.
        ihc = torch.exp(-torch.cdist(x, x, compute_mode='donot_use_mm_for_euclid_dist').square() / (2 * self.settings['ihc_bandwidth'] ** 2))
        ihc *= self.positive[:, None] == self.positive[None, :]
        ihc *= self.valid[:, None] & self.valid[None, :]
        bio = ihc
        if self.settings['alpha']:
            if self.he is None:
                raise ValueError('Missing H&E conditioning features.')
            if self.he_bandwidth is None:
                raise ValueError('Missing fitted H&E bandwidth.')
            he = (torch.ones_like(ihc) if self.he_bandwidth == 0 else
                  torch.exp(-(1 - self.he.detach().double() @ self.he.detach().double().T).clamp(0, 2) / self.he_bandwidth))
            bio = ihc * ((1 - self.settings['alpha']) + self.settings['alpha'] * he)
        return ihc, bio


class BiologyConditioner:
    """Portable training-only preprocessing, protected by content/provenance hashes."""
    def __init__(self, state: dict) -> None:
        self.state = copy.deepcopy(state)

    @classmethod
    def fit(cls, config: Mapping, bundle: DataBundle) -> 'BiologyConditioner':
        """Fit exactly once from a deterministic eligible training reference."""
        contract = conditioning_contract(config)
        if contract is None:
            raise ValueError('Cannot fit disabled conditioning.')
        rows, excluded = _reference(config, bundle)
        if len(rows) < 2:
            raise ValueError('Conditioning requires at least two eligible training reference rows.')
        ihc, he, source_hash = _reference_values(config, bundle, rows)
        positive = ihc[:, 0] > 0
        eps = contract['logit_epsilon']
        cd8 = np.clip(ihc[positive, 0], eps, 1 - eps)
        values = [ihc[:, i] for i in range(1, 4)] + [np.log(cd8 / (1 - cd8))]
        scales, omitted, methods = [], [], []
        for name, value in zip([*FEATURES[1:], 'positive_logit_CD8'], values):
            scale = float(np.subtract(*np.percentile(value, [75, 25]))) if len(value) else 0.
            method = 'iqr'
            if scale == 0:
                scale, method = (float(np.std(value)) if len(value) else 0.), 'std'
            if scale == 0:
                omitted.append(name)
                method = 'omitted_constant'
            scales.append(scale)
            methods.append(method)
        he_bandwidth, pair_count = None, 0
        if he is not None:
            n = len(he)
            pair_count = min(contract['settings']['he_pairs'], n * (n - 1) // 2)
            rng = np.random.default_rng(np.random.SeedSequence([contract['training_seed'], contract['settings']['reference_seed'], 1951]))
            # Unique unordered pairs without constructing the quadratic pair list.
            pair_ids = rng.choice(n * (n - 1) // 2, pair_count, replace=False)
            starts = np.arange(n, dtype=np.int64) * (2 * n - np.arange(n) - 1) // 2
            left = np.searchsorted(starts, pair_ids, side='right') - 1
            right = left + 1 + pair_ids - starts[left]
            distances = np.clip(1 - (he[left] * he[right]).sum(1), 0, 2)
            constant = bool(np.max(np.linalg.norm(he - he[:1], axis=1)) <= 1e-12)
            he_bandwidth = 0. if constant else max(float(np.median(distances)), 1e-12)
        state = dict(version=1, contract=contract, scales=scales, scale_methods=methods,
                     omitted_dimensions=omitted, reference_rows=rows.tolist(),
                     training_rows_sha256=_rows_digest(bundle.split_indices['train']),
                     reference_values_sha256=source_hash, fit_split='train',
                     excluded_training_rows=excluded, reference_count=len(rows),
                     he_width=None if he is None else he.shape[1], he_bandwidth=he_bandwidth,
                     he_pair_count=pair_count, he_neutral=he_bandwidth == 0,
                     source=str(config['data']['path']))
        state['sha256'] = _digest(state)
        return cls(state)

    @classmethod
    def restore(cls, state: dict, config: Mapping, training_rows: np.ndarray | None = None,
                bundle: DataBundle | None = None) -> 'BiologyConditioner':
        """Validate stored statistics and optionally recheck reference data, never refit."""
        if not isinstance(state, dict):
            raise ValueError('Missing fitted biological conditioning preprocessing.')
        state = copy.deepcopy(state)
        digest = state.pop('sha256', None)
        if digest != _digest(state) or state.get('version') != 1 or state.get('contract') != conditioning_contract(config):
            raise ValueError('Biological conditioning preprocessing contract/hash mismatch.')
        rows = np.asarray(state['reference_rows'])
        if (state['fit_split'] != 'train' or len(rows) != state['reference_count'] or len(rows) < 2
                or len(np.unique(rows)) != len(rows) or np.any(np.diff(rows) <= 0)
                or len(state['scales']) != 4 or any(not math.isfinite(x) or x < 0 for x in state['scales'])):
            raise ValueError('Invalid biological conditioning statistics or reference identities.')
        if training_rows is not None and (state['training_rows_sha256'] != _rows_digest(training_rows)
                                         or not np.isin(rows, training_rows).all()):
            raise ValueError('Biological conditioning training provenance mismatch.')
        if bundle is not None:
            expected, excluded = _reference(config, bundle)
            if not np.array_equal(expected, rows) or excluded != state['excluded_training_rows']:
                raise ValueError('Biological conditioning reference sample changed.')
            _, _, values_hash = _reference_values(config, bundle, rows)
            if values_hash != state['reference_values_sha256']:
                raise ValueError('Biological conditioning reference data changed.')
        state['sha256'] = digest
        return cls(state)

    @torch.no_grad()
    def transform(self, targets: torch.Tensor, valid: torch.Tensor | None, row_ids: torch.Tensor,
                  he: torch.Tensor | None = None) -> ConditioningInput:
        """Apply frozen scales to already normalized observations, detaching inputs."""
        contract = self.state['contract']
        columns = contract['target_transform']['target_columns']
        index = [columns.index(n) for n in FEATURES]
        x = targets.detach().double()[:, index]
        eligible = torch.isfinite(x).all(1) & ((x >= 0) & (x <= 1)).all(1)
        if valid is not None:
            eligible &= valid.detach()[:, index].all(1)
        x = torch.nan_to_num(x).clamp(0, 1)
        positive = x[:, 0] > 0
        cd8 = x[:, 0].clamp(contract['logit_epsilon'], 1 - contract['logit_epsilon'])
        features = torch.cat((x[:, 1:], torch.where(positive, torch.logit(cd8), 0.)[:, None]), 1)
        for i, scale in enumerate(self.state['scales']):
            features[:, i] = features[:, i] / scale if scale else 0.
        he_features = None
        if contract['settings']['alpha']:
            if he is None or he.ndim != 2 or he.shape != (len(x), self.state['he_width']):
                raise ValueError('Missing or incompatible fixed H&E conditioning features.')
            he_features = he.detach().double()
            norms = torch.linalg.vector_norm(he_features, dim=1)
            if not torch.isfinite(he_features).all() or torch.any(norms <= 0):
                raise ValueError('H&E conditioning rejects nonfinite or zero-norm vectors.')
            he_features = he_features / norms[:, None]
        return ConditioningInput(features, positive, eligible, row_ids.detach(), he_features,
                                 contract['settings'], self.state['he_bandwidth'])


def evaluate_conditional(latent: torch.Tensor, labels: torch.Tensor, names: Sequence[str],
                         bandwidths: Sequence[float], temperature: float,
                         conditioning: ConditioningInput) -> dict[str, Measurement]:
    """Evaluate local estimates, reducing valid anchors equally within/across slides."""
    from dann.batch_alignment import Measurement, NORM_FLOOR, DENOMINATOR_FLOOR
    c = conditioning
    if len(c.ihc) != len(latent) or c.row_ids.shape != labels.shape:
        raise ValueError('Conditioning features must align with latent row identities.')
    if len(torch.unique(c.row_ids)) != len(labels):
        raise ValueError('Conditioned alignment requires unique row identities.')
    keep = c.valid.detach()
    x, labels, c = latent[keep].double(), labels[keep], c.select(keep)
    unique, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    coverage = dict(input_rows=len(latent), sample_rows=len(x), sample_slides=len(unique),
                    excluded_invalid_rows=int((~keep).sum()), row_coverage=len(x)/max(len(latent), 1))
    def invalid(reason):
        return {n: Measurement(None, dict(coverage), reason) for n in names}
    if len(unique) < 2:
        return invalid('insufficient_slides')
    norms = torch.linalg.vector_norm(x, dim=1)
    if not torch.isfinite(x).all():
        return invalid('nonfinite_latents')
    if torch.any(norms <= NORM_FLOOR):
        return invalid('near_zero_latent_norm')
    x = x / norms[:, None]
    with torch.no_grad():
        ihc, bio = c.kernels()
        raw = bio * (ihc >= c.settings['ihc_min_similarity'])
        onehot = torch.nn.functional.one_hot(inverse, len(unique)).to(x)
        mass, square = raw @ onehot, raw.square() @ onehot
        candidates = (raw > 0).to(x) @ onehot
        ess = mass.square() / square.clamp_min(1e-300)
        participating = ((candidates >= c.settings['minimum_candidates']) &
                         (ess >= c.settings['minimum_ess'] - 1e-10))
        slide_count = participating.sum(1)
        supported = participating.gather(1, inverse[:, None]).squeeze(1) & (slide_count >= 2)
        weights = raw * (participating / mass.clamp_min(1e-300))[:, inverse]
        weights /= slide_count.clamp_min(1)[:, None]
        weights[~supported] = 0
        anchors = torch.nonzero(supported, as_tuple=True)[0]
        coverage.update(supported_anchors=len(anchors), unsupported_anchors=len(x)-len(anchors),
                        anchor_coverage=len(anchors)/max(len(x), 1),
                        supported_slides=len(torch.unique(labels[anchors])))
        if not len(anchors):
            return invalid('no_cross_slide_biological_support')
        coverage.update(neighborhood_slides_mean=float(slide_count[anchors].double().mean()),
                        neighborhood_ess_min=float(ess[anchors][participating[anchors]].min()),
                        neighborhood_ess_mean=float(ess[anchors][participating[anchors]].mean()),
                        effective_rows=float((1 / weights[anchors].square().sum(1)).mean()))
        # Anchor RNG is independent of optimizer, model, reference and row-order RNGs.
        rng = np.random.default_rng(c.settings['anchor_seed'])
        canonical = anchors.cpu().numpy()
        canonical = canonical[np.argsort(c.row_ids[anchors].cpu().numpy())]
        shuffled = rng.permutation(canonical)
        anchor_labels = labels.detach().cpu().numpy()
        chosen = []
        for slide in unique.cpu().tolist():
            chosen.extend(shuffled[anchor_labels[shuffled] == slide][:c.settings['anchors_per_slide']].tolist())
        selected = torch.as_tensor(chosen, device=x.device)
    similarity = (x @ x.T).clamp(-1, 1)
    same = (labels[:, None] == labels[None, :]).to(x)
    scores, diagnostics, reasons = {}, {}, {}
    w = weights[selected]
    # Moment identity: <Kc,Lc>_w = E[KL] - 2 E[(Kw)(Lw)] + E[K]E[L].
    def moment(a, b, aw, bw, am, bm):
        return ((w @ (a * b)) * w).sum(1) - 2 * (w * aw * bw).sum(1) + am * bm
    if 'nhsic' in names:
        kernel = sum(torch.exp(-(1 - similarity) / b) for b in bandwidths) / len(bandwidths)
        # Subtract a common centering first: subsequent local centering is
        # unchanged, while moment subtraction no longer cancels O(1) terms for
        # nearly constant latent kernels. This remains one shared N-by-N matrix.
        mean_kernel = kernel.mean(1)
        kernel = kernel - mean_kernel[:, None] - mean_kernel[None, :] + mean_kernel.mean()
        mean_same = same.mean(1)
        label_kernel = same - mean_same[:, None] - mean_same[None, :] + mean_same.mean()
        kw, lw = w @ kernel, w @ label_kernel
        km, lm = (w * kw).sum(1), (w * lw).sum(1)
        num = moment(kernel, label_kernel, kw, lw, km, lm)
        kn = moment(kernel, kernel, kw, kw, km, km).clamp_min(0)
        ln = moment(label_kernel, label_kernel, lw, lw, lm, lm).clamp_min(0)
        denom = (kn * ln).clamp_min(DENOMINATOR_FLOOR**2).sqrt()
        good = (kn > NORM_FLOOR**2) & (ln > NORM_FLOOR**2) & (denom > DENOMINATOR_FLOOR)
        # A variance check catches cancellation in constant latent kernels.
        mean = w @ x
        variance = (w @ x.square().sum(1)) - mean.square().sum(1)
        good &= variance > NORM_FLOOR
        scores['nhsic'] = (selected, num / denom, good)
        diagnostics['nhsic'] = dict(numerator=num, denominator=denom,
                                    latent_kernel_norm=kn.sqrt(), label_kernel_norm=ln.sqrt())
        reasons['nhsic'] = 'degenerate_kernel_norm'
    if 'anova' in names:
        mean = w @ x
        variance = (w @ x.square().sum(1)) - mean.square().sum(1)
        # Sum of squared weighted slide means / slide masses, minus overall mean².
        between = torch.zeros_like(variance)
        for slide in range(len(unique)):
            sw = w * (inverse == slide)
            between += (sw @ x).square().sum(1) / sw.sum(1).clamp_min(DENOMINATOR_FLOOR)
        between = (between - mean.square().sum(1)).clamp_min(0)
        scores['anova'] = (selected, between / variance.clamp_min(DENOMINATOR_FLOOR), variance > NORM_FLOOR)
        diagnostics['anova'] = dict(numerator=between, denominator=variance)
        reasons['anova'] = 'degenerate_latent_variance'
    if 'soft_nn' in names:
        # Joint pair measure: equal-slide supported anchors times conditional
        # candidate weights. Remove self before normalizing either pair or chance.
        sw = weights[anchors].clone()
        sw[torch.arange(len(anchors), device=x.device), anchors] = 0
        anchor_counts = torch.bincount(inverse[anchors], minlength=len(unique))
        aw = 1 / (coverage['supported_slides'] * anchor_counts[inverse[anchors]].to(x))
        pair = aw[:, None] * sw
        pair /= pair.sum()
        sim = similarity[anchors]
        center = (pair * sim).sum()
        variance = (pair * (sim - center).square()).sum()
        std = variance.clamp_min(DENOMINATOR_FLOOR).sqrt()
        logits = (sim - center) / (std * temperature) + sw.clamp_min(1e-300).log()
        probability = logits.masked_fill(sw == 0, -torch.inf).softmax(1)
        same_probability = (probability * same[anchors]).sum(1)
        chance = (sw * same[anchors]).sum(1) / sw.sum(1)
        scores['soft_nn'] = (anchors, (same_probability - chance).clamp_min(0).square(),
                             torch.ones(len(anchors), dtype=torch.bool, device=x.device) & (variance > NORM_FLOOR))
        diagnostics['soft_nn'] = dict(similarity_std=std.expand(len(anchors)),
                                      same_slide_probability=same_probability, chance_probability=chance)
        reasons['soft_nn'] = 'degenerate_similarity_variance'
    result = {}
    for name in names:
        ids, value, good = scores[name]
        good = good & torch.isfinite(value)
        def reduce(values):
            return torch.stack([values[good & (labels[ids] == slide)].mean() for slide in torch.unique(labels[ids[good]])]).mean()
        d = dict(coverage, evaluated_anchors=len(ids), valid_anchors=int(good.sum()),
                 degenerate_anchors=int((~good).sum()), valid_anchor_slides=len(torch.unique(labels[ids[good]])))
        if good.any():
            d.update({k: reduce(v).detach() for k, v in diagnostics[name].items()})
            result[name] = Measurement(reduce(value), d)
        else:
            result[name] = Measurement(None, d, reasons[name])
    return result

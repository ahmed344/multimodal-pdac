"""Independent dense NumPy references for frozen local conditional estimators."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.special import softmax

from dann.batch_alignment import (NAMES, BANDWIDTHS, ValidationSample, AlignmentSummary,
                                 alignment_settings, alignment_contract, evaluate_alignment, sample_rows)
from dann.conditional_alignment import (BiologyConditioner, ConditioningInput, FEATURES,
                                       conditioning_settings, conditioning_contract, eligible_targets)
from dann.tests.test_batch_alignment import reference, alignment_epoch_fixture
from dann.tests.test_components import spatial_config


def config(alpha=.1):
    return dict(data=dict(target_columns=list(FEATURES), path='synthetic', he_features_key='uni2'),
                training=dict(seed=19), loss=dict(logit_epsilon=1e-5,
                batch_alignment=dict(train_per_slide=128, conditioning=dict(enabled=True, alpha=alpha),
                                     **{n: dict(enabled=True, weight=.1) for n in NAMES})))


def fixture(alpha=.1, n=40):
    rng = np.random.default_rng(41)
    c = config(alpha)
    targets = rng.uniform(.1, .5, (n, 4))
    targets[::7, 0] = 0
    he = rng.normal(size=(n, 6))
    bundle = SimpleNamespace(split_indices=dict(train=np.arange(n-4), validation=np.arange(n-4, n)),
        metadata=SimpleNamespace(targets=targets, target_valid_mask=np.ones_like(targets, dtype=bool),
                                 batch_codes=np.arange(n) % 3),
        he_reader=SimpleNamespace(read=lambda rows: he[rows]))
    conditioner = BiologyConditioner.fit(c, bundle)
    features = conditioner.transform(torch.tensor(targets), torch.ones((n, 4), dtype=torch.bool),
                                     torch.arange(n), torch.tensor(he))
    return c, bundle, conditioner, features, he


def numpy_reference(x, labels, c):
    """Explicit local centering/centroids; no production kernels or moment identities."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    biology = c.ihc.numpy()
    ihc = np.exp(-np.sum((biology[:, None] - biology[None, :])**2, axis=2) / (2*c.settings['ihc_bandwidth']**2))
    ihc *= c.positive.numpy()[:, None] == c.positive.numpy()[None, :]
    he = np.ones_like(ihc) if not c.settings['alpha'] or c.he_bandwidth == 0 else np.exp(-np.clip(1-c.he.numpy() @ c.he.numpy().T, 0, 2)/c.he_bandwidth)
    raw = ihc * ((1-c.settings['alpha']) + c.settings['alpha']*he) * (ihc >= c.settings['ihc_min_similarity'])
    slides = np.unique(labels)
    weights, supported = np.zeros_like(raw), []
    for anchor, r in enumerate(raw):
        participating = []
        for slide in slides:
            v = r[labels == slide]
            if np.count_nonzero(v) >= c.settings['minimum_candidates'] and v.sum()**2 / max(np.sum(v*v), 1e-300) >= c.settings['minimum_ess'] - 1e-10:
                participating.append(slide)
        if labels[anchor] not in participating or len(participating) < 2:
            continue
        supported.append(anchor)
        for slide in participating:
            mask = labels == slide
            weights[anchor, mask] = r[mask] / (len(participating) * r[mask].sum())
    rng = np.random.default_rng(c.settings['anchor_seed'])
    canonical = np.asarray(supported)[np.argsort(c.row_ids.numpy()[supported])]
    shuffled = rng.permutation(canonical)
    selected = []
    for slide in slides:
        selected.extend(shuffled[labels[shuffled] == slide][:c.settings['anchors_per_slide']].tolist())
    dot = np.clip(x @ x.T, -1, 1)
    kernel = np.mean([np.exp(-(1-dot)/b) for b in BANDWIDTHS], axis=0)
    same = (labels[:, None] == labels).astype(float)
    scores = {n: [] for n in NAMES}
    for i in selected:
        w = weights[i]
        h = np.eye(len(x)) - np.ones((len(x), 1)) @ w[None, :]
        a, b = h @ kernel @ h.T, h @ same @ h.T
        pair = w[:, None] * w[None, :]
        scores['nhsic'].append(np.sum(pair*a*b)/np.sqrt(np.sum(pair*a*a)*np.sum(pair*b*b)))
        mean = w @ x
        between = 0
        for slide in slides:
            mask = labels == slide
            mass = w[mask].sum()
            if mass:
                centroid = w[mask] @ x[mask] / mass
                between += mass * np.sum((centroid-mean)**2)
        scores['anova'].append(between / np.sum(w[:, None]*(x-mean)**2))
    off = weights[supported].copy()
    off[np.arange(len(supported)), supported] = 0
    aw = np.array([1/(len(np.unique(labels[supported])) * np.count_nonzero(labels[supported] == labels[i])) for i in supported])
    pair = aw[:, None]*off
    pair /= pair.sum()
    sim = dot[supported]
    mean = np.sum(pair * sim)
    std = np.sqrt(np.sum(pair*(sim-mean)**2))
    with np.errstate(divide='ignore'):
        probs = softmax((sim-mean)/(.1*std)+np.log(off), axis=1)
    chance = np.sum(off*same[supported], axis=1)/off.sum(1)
    scores['soft_nn'] = np.maximum(np.sum(probs*same[supported], axis=1)-chance, 0)**2
    def reduce(values, anchors):
        return np.mean([np.mean(np.asarray(values)[labels[anchors] == slide]) for slide in np.unique(labels[anchors])])
    return {n: reduce(v, supported if n == 'soft_nn' else selected) for n, v in scores.items()}, reduce(chance, supported)


@pytest.mark.parametrize('alpha', [0., .1, 1.])
def test_independent_references_self_exclusion_local_centering_and_permutations(alpha):
    _, _, _, c, _ = fixture(alpha)
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(3), [8, 13, 19])
    x = rng.normal(size=(40, 5))
    expected, chance = numpy_reference(x, labels, c)
    result = evaluate_alignment(torch.tensor(x), torch.tensor(labels), conditioning=c)
    for n in NAMES:
        assert result[n].score.item() == pytest.approx(expected[n], abs=1e-12)
    assert result['soft_nn'].diagnostics['chance_probability'].item() == pytest.approx(chance)
    order = rng.permutation(40)
    changed = evaluate_alignment(torch.tensor(x[order]), torch.tensor(labels[order]), conditioning=c.select(order))
    relabeled = evaluate_alignment(torch.tensor(x), torch.tensor(np.array([90, -2, 3])[labels]), conditioning=c)
    for n in NAMES:
        torch.testing.assert_close(result[n].score, changed[n].score, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(result[n].score, relabeled[n].score, atol=1e-12, rtol=1e-12)


def constant_features(n, alpha=0):
    return ConditioningInput(torch.zeros(n, 4, dtype=torch.double), torch.ones(n, dtype=torch.bool),
        torch.ones(n, dtype=torch.bool), torch.arange(n), torch.ones(n, 2, dtype=torch.double)/math.sqrt(2) if alpha else None,
        conditioning_settings(dict(enabled=True, alpha=alpha)), 0 if alpha else None)


def test_constant_biology_matches_unconditional_with_unequal_slides():
    rng = np.random.default_rng(14)
    x = rng.normal(size=(19, 5))
    labels = np.repeat(np.arange(3), [3, 7, 9])
    expected, _ = reference(x, labels)
    for alpha in (0, .1, 1):
        result = evaluate_alignment(torch.tensor(x), torch.tensor(labels), conditioning=constant_features(19, alpha))
        for n in NAMES:
            assert result[n].score.item() == pytest.approx(expected[n], abs=1e-12)


@pytest.mark.parametrize('name', NAMES)
def test_latent_gradients_and_detached_biology(name):
    _, _, _, c, _ = fixture(0, 16)
    c.ihc.requires_grad_()
    x = torch.randn(16, 3, generator=torch.Generator().manual_seed(8), dtype=torch.double, requires_grad=True)
    labels = torch.arange(16) % 2
    fn = lambda value: evaluate_alignment(value, labels, (name,), conditioning=c)[name].score
    assert torch.autograd.gradcheck(fn, (x,), atol=1e-5)
    fn(x).backward()
    assert c.ihc.grad is None
    assert torch.isfinite(x.grad).all() and x.grad.norm() > 0


@pytest.mark.parametrize('bad', [None, [], {'unknown':1}, {'enabled':1}, {'alpha':True}, {'alpha':-.1},
    {'alpha':1.1}, {'alpha':math.nan}, {'alpha':math.inf}, {'alpha':'0.1'}, {'ihc_bandwidth':0},
    {'minimum_ess':1}, {'minimum_candidates':1}, {'reference_per_slide':1025}, {'he_pairs':16385},
    {'anchors_per_slide':3}, {'anchor_seed':-1}, {'ihc_min_similarity':0}])
def test_conditioning_schema(bad):
    with pytest.raises(ValueError):
        alignment_settings({'loss':{'batch_alignment':{'conditioning':bad}}})


def test_frozen_fit_training_only_iqr_std_constant_and_restore():
    c, bundle, fitted, _, he = fixture()
    state = fitted.state
    restored = BiologyConditioner.restore(state, c, bundle.split_indices['train'], bundle)
    assert restored.state == state
    bundle.metadata.targets[-4:] = 1e10
    he[-4:] = np.nan
    assert BiologyConditioner.fit(c, bundle).state == state  # held-out values never enter fit
    tampered = copy.deepcopy(state); tampered['scales'][0] *= 2
    with pytest.raises(ValueError, match='hash'):
        BiologyConditioner.restore(tampered, c)
    with pytest.raises(ValueError, match='provenance'):
        BiologyConditioner.restore(state, c, np.arange(40))
    bundle.metadata.targets[0, 1] += .01
    with pytest.raises(ValueError, match='data changed'):
        BiologyConditioner.restore(state, c, bundle.split_indices['train'], bundle)
    bundle.metadata.targets[:] = 0
    bundle.metadata.targets[0, 1] = .5
    he[:] = 1
    fitted = BiologyConditioner.fit(c, bundle)
    assert fitted.state['scale_methods'] == ['std', 'omitted_constant', 'omitted_constant', 'omitted_constant']
    assert fitted.state['he_neutral'] and fitted.state['he_bandwidth'] == 0


def test_exact_ihc_only_no_he_requirement_and_hurdle_groups():
    c, bundle, fitted, features, _ = fixture(0)
    bundle.he_reader = None
    assert BiologyConditioner.fit(c, bundle).state == fitted.state
    values = torch.tensor(bundle.metadata.targets)
    bad_he = torch.full((len(values), 7), float('nan'), requires_grad=True)
    actual = fitted.transform(values, None, torch.arange(len(values)), bad_he)
    a, b = actual.kernels()
    assert torch.equal(a, b)
    assert torch.equal(a, features.kernels()[0])
    assert torch.count_nonzero(a[actual.positive][:, ~actual.positive]) == 0
    assert actual.he is None and fitted.state['he_pair_count'] == 0


@pytest.mark.parametrize('bad', [0., float('nan'), float('inf')])
def test_reject_invalid_he_when_active(bad):
    c, bundle, fitted, _, he = fixture()
    he[0] = bad
    with pytest.raises(ValueError, match='H&E'):
        BiologyConditioner.fit(c, bundle)
    with pytest.raises(ValueError, match='H&E'):
        fitted.transform(torch.tensor(bundle.metadata.targets), None, torch.arange(40), torch.tensor(he))


def test_invalid_targets_alignment_only_and_support_failure():
    _, bundle, fitted, _, he = fixture(0)
    targets = torch.tensor(bundle.metadata.targets)
    valid = torch.ones_like(targets, dtype=torch.bool)
    targets[0, 1] = math.nan
    targets[1, 2] = 1.1
    valid[2, 0] = False
    features = fitted.transform(targets, valid, torch.arange(40))
    assert not features.valid[:3].any() and features.valid[3:].all()
    np.testing.assert_array_equal(features.valid.numpy(), eligible_targets(targets.numpy(), valid.numpy(), FEATURES))
    labels = torch.arange(40)%2
    features.positive = labels == 0  # No cross-slide biology support.
    result = evaluate_alignment(torch.randn(40, 5), labels, conditioning=features)
    assert all(m.score is None and m.reason == 'no_cross_slide_biological_support' for m in result.values())
    summary = AlignmentSummary(alignment_settings(config(0))); summary.add(result)
    with pytest.raises(ValueError, match='no valid evaluations'):
        summary.finish()


def test_rare_population_ess_and_degeneracy_unavailable():
    features = constant_features(10)
    features.positive[:2] = False
    labels = torch.arange(10)%2  # one absent candidate on each slide: unsupported
    result = evaluate_alignment(torch.randn(10, 5), labels, conditioning=features)
    assert result['soft_nn'].diagnostics['unsupported_anchors'] == 2
    assert result['soft_nn'].diagnostics['supported_anchors'] == 8
    result = evaluate_alignment(torch.ones(10, 5), labels, conditioning=features)
    assert all(m.score is None for m in result.values())
    features = constant_features(4)
    features.ihc[0, 0] = 1  # two candidates with unequal weights have ESS < 2
    result = evaluate_alignment(torch.randn(4, 5), torch.arange(4)%2, conditioning=features)
    assert all(m.score is None for m in result.values())


def test_composition_shift_preserved_but_matched_technical_shift_penalized():
    # Biology differs in proportions across slides, but both groups have support.
    labels = torch.tensor([0]*40 + [1]*40)
    group = torch.tensor([0]*32+[1]*8+[0]*8+[1]*32)
    features = constant_features(80)
    features.positive = group.bool()
    gen = torch.Generator().manual_seed(17)
    templates = .05*torch.randn(2, 8, 5, generator=gen, dtype=torch.double)
    noise = torch.cat((templates[0].repeat(4, 1), templates[1], templates[0], templates[1].repeat(4, 1)))
    x = torch.nn.functional.one_hot(group, 5).double() + noise
    uncond = evaluate_alignment(x, labels)
    conditional = evaluate_alignment(x, labels, conditioning=features)
    shifted = x + 1.5*torch.nn.functional.one_hot(labels+2, 5).double()
    technical = evaluate_alignment(shifted, labels, conditioning=features)
    for name in NAMES:
        assert conditional[name].score < uncond[name].score
        assert technical[name].score > conditional[name].score


def test_validation_collection_detaches_and_preserves_order():
    _, _, _, features, _ = fixture()
    labels = torch.arange(40)%3
    x = torch.randn(40, 5, requires_grad=True)
    selected = sample_rows(np.arange(40), labels.numpy(), 9, 12)
    results = []
    for size, order in [(1, torch.arange(40)), (7, torch.arange(40).flip(0)), (40, torch.arange(40))]:
        collector = ValidationSample(selected)
        for ids in order.split(size):
            collector.add(ids, labels[ids], x[ids], features.select(ids))
        xx, bb = collector.tensors()
        biology = collector.conditioning()
        assert not xx.requires_grad and not biology.ihc.requires_grad
        assert biology.row_ids.tolist() == selected.tolist()
        results.append(evaluate_alignment(xx, bb, conditioning=biology))
    for n in NAMES:
        for result in results[1:]:
            torch.testing.assert_close(results[0][n].score, result[n].score)


def test_conditioned_epoch_and_disabled_bypasses_conditioner():
    from dann.train import run_epoch
    from dann.tests.test_objectives import Batches
    model, batch, kwargs = alignment_epoch_fixture()
    # All positive and close: sufficiently broad support on all three slides.
    batch['targets'][:] = .2
    kwargs['batch_alignment'].update(train_per_slide=128, conditioning=dict(enabled=True, alpha=0))
    c = config(0)
    bundle = SimpleNamespace(split_indices=dict(train=np.arange(12)), metadata=SimpleNamespace(
        targets=batch['targets'].numpy(), target_valid_mask=None, batch_codes=batch['batches'].numpy()), he_reader=None)
    fitted = BiologyConditioner.fit(c, bundle)
    kwargs['alignment_conditioner'] = fitted
    kwargs['validation_alignment_rows'] = batch['row_ids'].tolist()
    results=[]
    for size, order in [(1, torch.arange(12)), (5, torch.arange(12).flip(0)), (12, torch.arange(12))]:
        kwargs['loader'] = Batches({k:v[ids] for k,v in batch.items()} for ids in order.split(size))
        results.append(run_epoch(model=model, optimizer=None, **kwargs))
    for key in ('alignment_nhsic', 'alignment_anova', 'total_loss'):
        assert results[0][key] == pytest.approx(results[1][key], abs=1e-10)
        assert results[0][key] == pytest.approx(results[2][key], abs=1e-10)
    train_metrics = run_epoch(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=.01), **kwargs)
    assert train_metrics['gradient_overall_nhsic_norm_mean'] > 0
    kwargs['batch_alignment']['conditioning']['enabled'] = False
    kwargs['alignment_conditioner'] = SimpleNamespace(transform=lambda *a: pytest.fail('disabled conditioner called'))
    disabled = run_epoch(model=model, optimizer=None, **kwargs)
    kwargs['batch_alignment'].pop('conditioning')
    legacy = run_epoch(model=model, optimizer=None, **kwargs)
    assert disabled['alignment_nhsic'] == legacy['alignment_nhsic']
    assert alignment_settings({'loss':{'batch_alignment':kwargs['batch_alignment']}})['train_per_slide'] == 128


def test_contract_preserves_legacy_and_rejects_setting_changes():
    from dann.train import resume_training_config
    old = {'loss': {'batch_alignment': {}}}
    assert alignment_contract(old)['version'] == 1
    assert 'conditioning' not in alignment_contract(old)['settings']
    c = config()
    for key, value in [('alpha',0), ('enabled',False), ('anchors_per_slide',1), ('reference_seed',3)]:
        changed = copy.deepcopy(c)
        changed['loss']['batch_alignment']['conditioning'][key] = value
        with pytest.raises(ValueError, match='fresh training'):
            resume_training_config(c, changed)
    changed = copy.deepcopy(c); changed['loss']['batch_alignment']['train_per_slide'] = 64
    with pytest.raises(ValueError, match='fresh training'):
        resume_training_config(c, changed)


def test_he_loaded_without_auxiliary_and_not_required_at_alpha_zero(spatial_config):
    import h5py
    from dann.data_loader import create_data_bundle
    from dann.config import resolve_execution
    c = spatial_config
    c['loss']['batch_alignment'] = config(0)['loss']['batch_alignment']
    bundle = create_data_bundle(resolve_execution(c))
    assert bundle.he_reader is None
    for d in bundle.datasets.values(): d.close()
    c['loss']['batch_alignment']['conditioning']['alpha'] = .1
    with h5py.File(c['data']['path'], 'a') as f:
        f.require_group('obsm').create_dataset('hes_features', data=np.ones((bundle.metadata.num_observations, 6)))
    bundle = create_data_bundle(resolve_execution(c))
    assert bundle.he_reader.width == 6
    assert 'he_targets' in next(iter(bundle.loaders['train']))
    for d in bundle.datasets.values(): d.close()


def test_conditioned_checkpoint_resume_no_refit_and_tamper_detection(spatial_config, monkeypatch):
    from dann import train
    from dann.config import resolve_execution
    from dann.checkpoints import checkpoint_config
    c = spatial_config
    c['model']['heads']['discriminator']['enabled'] = False
    c['model']['aggregation']['type'] = 'mlp'
    c['model']['heads']['biology']['type'] = 'mlp'
    c['loss']['batch_alignment'] = config(0)['loss']['batch_alignment']
    c['loss']['batch_alignment']['soft_nn']['weight'] = 0.  # Tiny fixture latents can have degenerate cosine variance.
    c['training'].update(epochs=1, early_stopping_enabled=False)
    c['training']['pixel']['batch_size'] = 64
    # Small fixture needs broad neighborhoods; still exercises nonuniform weights.
    c['loss']['batch_alignment']['conditioning']['ihc_bandwidth'] = 3.
    resolve_execution(c)
    monkeypatch.setattr(train, 'refresh_loss_curves', lambda *a, **k: None)
    path = train.train_model(c)
    first = torch.load(path, weights_only=False)
    assert first['alignment_conditioner']['fit_split'] == 'train'
    selected = first['validation_alignment_sample']['row_ids']
    assert len(selected) == first['validation_alignment_sample']['eligible_rows']
    assert first['validation_alignment_sample']['excluded_invalid_rows'] > 0
    monkeypatch.setattr(BiologyConditioner, 'fit', lambda *a, **k: pytest.fail('refit on resume'))
    latest = path.with_name('latest.pt')
    c['training'].update(epochs=2, resume_checkpoint=str(latest))
    train.train_model(c)
    second = torch.load(latest, weights_only=False)
    assert first['alignment_conditioner'] == second['alignment_conditioner']
    assert first['validation_alignment_sample'] == second['validation_alignment_sample']
    assert second['epoch'] == 1
    second['alignment_conditioner']['scales'][0] *= 2
    with pytest.raises(ValueError, match='hash'):
        checkpoint_config(second)
    del second['alignment_conditioner']
    with pytest.raises(ValueError, match='Missing fitted'):
        checkpoint_config(second)


def test_reference_cap_validity_before_sampling_and_reproducibility():
    c, bundle, _, _, _ = fixture(0, 200)
    c['loss']['batch_alignment']['conditioning']['reference_per_slide'] = 8
    bundle.metadata.target_valid_mask[::2, 0] = False
    fitted = BiologyConditioner.fit(c, bundle)
    refs = np.asarray(fitted.state['reference_rows'])
    assert len(refs) == 24 and (refs % 2 == 1).all()
    assert fitted.state['excluded_training_rows'] == 98
    bundle.split_indices['train'] = bundle.split_indices['train'][::-1]
    assert BiologyConditioner.fit(c, bundle).state == fitted.state


def test_distinct_he_reference_pairs_exhaustive_median():
    c, bundle, _, _, he = fixture(1, 12)
    fitted = BiologyConditioner.fit(c, bundle)
    rows = np.array(fitted.state['reference_rows'])
    normalized = he[rows]/np.linalg.norm(he[rows], axis=1, keepdims=True)
    expected = np.median([max(0, 1 - normalized[i] @ normalized[j]) for i in range(len(rows)) for j in range(i+1, len(rows))])
    assert fitted.state['he_pair_count'] == len(rows)*(len(rows)-1)//2
    assert fitted.state['he_bandwidth'] == pytest.approx(expected, abs=1e-15)


def test_disabled_never_fits_or_requires_biological_targets(spatial_config, monkeypatch):
    from dann import train
    from dann.config import resolve_execution
    c = spatial_config
    c['model']['heads']['discriminator']['enabled'] = False
    c['model']['aggregation']['type'] = 'mlp'
    c['model']['heads']['biology']['type'] = 'mlp'
    c['loss']['batch_alignment'] = config()['loss']['batch_alignment']
    c['loss']['batch_alignment']['conditioning']['enabled'] = False
    c['loss']['batch_alignment']['soft_nn']['weight'] = 0.  # Tiny fixture latents can have degenerate cosine variance.
    c['training'].update(epochs=1, early_stopping_enabled=False)
    c['training']['pixel']['batch_size'] = 64
    resolve_execution(c)
    monkeypatch.setattr(BiologyConditioner, 'fit', lambda *a, **k: pytest.fail('disabled fit'))
    monkeypatch.setattr(train, 'refresh_loss_curves', lambda *a, **k: None)
    path = train.train_model(c)
    saved = torch.load(path, weights_only=False)
    assert saved['alignment_conditioner'] is None
    assert saved['objective_contract']['settings']['train_per_slide'] == 128
    c['data']['target_columns'] = ['arbitrary']
    assert conditioning_contract(c) is None


def test_nearly_constant_latent_kernel_moments_remain_accurate():
    _, _, _, features, _ = fixture(0)
    rng = np.random.default_rng(53)
    labels = np.arange(40)%3
    x = np.ones((40, 6)) + 1e-3*rng.normal(size=(40, 6))
    expected = numpy_reference(x, labels, features)[0]['nhsic']
    result = evaluate_alignment(torch.tensor(x), torch.tensor(labels), ('nhsic',), conditioning=features)['nhsic']
    assert result.score.item() == pytest.approx(expected, abs=1e-10)


def test_preprocessing_changes_require_fresh_training():
    from dann.train import resume_training_config
    c = config()
    changed = copy.deepcopy(c)
    changed['loss']['logit_epsilon'] *= 10
    with pytest.raises(ValueError, match='fresh training'):
        resume_training_config(c, changed)


def test_empty_validation_sample_reports_unavailable():
    collector = ValidationSample([])
    c = constant_features(4)
    c.valid[:] = False
    collector.add(torch.arange(4), torch.arange(4)%2, torch.ones(4, 3), c)
    x, labels = collector.tensors()
    result = evaluate_alignment(x, labels, conditioning=collector.conditioning())
    assert all(m.score is None and m.reason == 'insufficient_slides' for m in result.values())
    assert collector.coverage()['sampling_invalid_rows'] == 4


def test_ihc_distances_do_not_cancel_large_common_scaled_offsets():
    features = constant_features(40)
    features.ihc = torch.randint(0, 5, (40, 4), generator=torch.Generator().manual_seed(3)).double()
    expected = features.kernels()[0]
    features.ihc += 2**32
    torch.testing.assert_close(features.kernels()[0], expected, atol=0, rtol=0)

"""Dense numerical references and end-to-end contracts for dictionary pooling."""
import copy

import numpy as np
import pytest
import torch

from dann.checkpoints import architecture_contract, checkpoint_config, validate_checkpoint
from dann.config import component_settings, resolve_execution
from dann.data_loader import create_data_bundle
from dann.execution import evaluation_groups
from dann.model import AdversarialLatentFusion
from dann.model_components.deep_sets import DeepSetsEncoder
from dann.tests.test_components import spatial_config
from dann.tests.test_execution import enable
from dann.tests.test_he import he_config
from dann.train import resume_training_config, train_model


def encoder(norm='l1', dropout=0., centering='dictionary_mean', epsilon=1e-8):
    return DeepSetsEncoder(5, 4, [4, 4], 6, dropout=dropout, peak_mlp_residual=True,
                           pooling='intensity_weighted_dictionary', normalization=norm,
                           pool_centering=centering, normalization_epsilon=epsilon)


def spectra():
    # Repeated (row, peak), explicit zeros, and an empty row.
    return (torch.tensor([0, 0, 2, 1, 4, 3]),
            torch.tensor([.2, .3, 0., 1., 2., .4], dtype=torch.double),
            torch.tensor([0, 0, 0, 1, 1, 3]), torch.tensor([3, 2, 0, 1]))


def dense_reference(model, ids, values, rows, counts):
    matrix = values.new_zeros((len(counts), model.num_peaks)).index_put(
        (rows, ids), values, accumulate=True)
    dictionary = model.peak_mlp(model.embedding.weight)
    if model.pool_centering == 'dictionary_mean':
        dictionary = dictionary - dictionary.mean(0)
    norm = (matrix.square().sum(1).clamp_min(model.normalization_epsilon**2).sqrt()
            if model.normalization == 'l2' else matrix.sum(1).clamp_min(model.normalization_epsilon))
    return matrix @ dictionary / norm[:, None]


@pytest.mark.parametrize('norm', ['l1', 'l2'])
@pytest.mark.parametrize('centering', ['none', 'dictionary_mean'])
@pytest.mark.parametrize('epsilon', [1e-8, 2.])
def test_dense_values_and_gradients(norm, centering, epsilon):
    a = encoder(norm, centering=centering, epsilon=epsilon).double()
    b = copy.deepcopy(a)
    ids, v, rows, counts = spectra()
    x, y = v.clone().requires_grad_(), v.clone().requires_grad_()
    actual, expected = a(ids, x, rows, counts), dense_reference(b, ids, y, rows, counts)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward(); (expected * weight).sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize('norm', ['l1', 'l2'])
def test_invariance_duplicates_zeros_empty_continuity_and_scaling(norm):
    model = encoder(norm).double().eval()
    ids, values, rows, counts = spectra()
    expected = model(ids, values, rows, counts)
    order = torch.tensor([5, 2, 0, 3, 1, 4])
    torch.testing.assert_close(model(ids[order], values[order], rows[order], counts), expected)
    keep = torch.tensor([0, 3, 4, 5])
    joined = torch.tensor([.5, 1., 2., .4], dtype=torch.double)
    torch.testing.assert_close(model(ids[keep], joined, rows[keep], torch.tensor([1, 2, 0, 1])), expected)
    torch.testing.assert_close(model(ids, values * 73, rows, counts), expected)
    zeros = torch.zeros(4, dtype=torch.double, requires_grad=True)
    empty = model(ids[:4], zeros, rows[:4], counts)
    assert torch.equal(empty, torch.zeros_like(empty))
    empty.sum().backward()
    assert torch.isfinite(zeros.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad()
    empty = model(ids[:0], values[:0], rows[:0], torch.zeros(3, dtype=torch.long))
    empty.sum().backward()
    assert torch.equal(empty, torch.zeros_like(empty))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    changes = []
    for delta in [1e-3, 1e-5, 1e-7]:
        changed = values.clone(); changed[2] = delta
        changes.append((model(ids, changed, rows, counts) - expected).norm().item())
    assert changes[2] < changes[1] < changes[0]
    # Below the floor, response tends to zero rather than staying scale invariant.
    tiny = model(ids[:1], values[:1] * 1e-12, rows[:1], torch.tensor([1]))
    assert tiny.norm() < expected[0].norm() * 1e-3


@pytest.mark.parametrize('norm', ['l1', 'l2'])
@pytest.mark.parametrize('dropout', [0., .4])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_direct_chunks_checkpoint_rng_gradients_and_no_stale_dictionary(norm, dropout, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    a = encoder(norm, dropout).double().to(device).train()
    args = tuple(value.to(device) for value in spectra())
    results = []
    for chunked, checkpointed in [(False, False), (True, False), (True, True)]:
        model = copy.deepcopy(a)
        calls = []
        hook = model.peak_mlp.register_forward_hook(lambda m, i, o: calls.append(len(i[0])))
        torch.manual_seed(9)
        out = model.microbatched(*args, peak_budget=2, checkpointing=checkpointed) if chunked else model(*args)
        assert calls == [5]  # Once per invocation, never once per chunk/occurrence.
        out.square().sum().backward()
        results.append((out, [p.grad for p in model.parameters()], torch.get_rng_state(),
                        torch.cuda.get_rng_state() if device == 'cuda' else None))
        hook.remove()
    for actual in results[1:]:
        torch.testing.assert_close(actual[0], results[0][0])
        for p, q in zip(actual[1], results[0][1]):
            torch.testing.assert_close(p, q, atol=1e-12, rtol=1e-10)
        assert torch.equal(actual[2], results[0][2])
        if device == 'cuda':
            assert torch.equal(actual[3], results[0][3])
    a.eval()
    with torch.no_grad():
        before = a(*args)
        a.embedding.weight[0].add_(.2)
        assert not torch.allclose(before, a(*args))


@pytest.mark.parametrize('version', range(2, 9))
def test_checkpoint_versions_and_seeded_parameter_order(he_config, version):
    c = he_config
    for group in [c['model']['aggregation'], *c['model']['heads'].values()]:
        group['type'] = 'mlp'
    resolve_execution(c)
    contract = architecture_contract(c)
    contract['version'] = version
    if version < 8:
        for key in ('pooling', 'normalization', 'normalization_epsilon'):
            contract['components']['spectral_encoder']['deep_sets'].pop(key)
            c['model']['spectral_encoder']['deep_sets'].pop(key, None)
    if version < 7:
        for key in ('peak_mlp_residual', 'pool_centering'):
            contract['components']['spectral_encoder']['deep_sets'].pop(key)
        for name, group in contract['components'].items():
            if name != 'spectral_encoder':
                group['mlp'].pop('residual')
    payload = dict(config=c, architecture=contract)
    restored = checkpoint_config(payload)
    validate_checkpoint(payload, restored)
    models = []
    for norm in ('count', 'l1', 'l2'):
        candidate = copy.deepcopy(restored)
        if norm != 'count':
            candidate['model']['spectral_encoder']['deep_sets'].update(
                pooling='intensity_weighted_dictionary', normalization=norm, pool_centering='dictionary_mean')
        torch.manual_seed(20260719)
        models.append(AdversarialLatentFusion.from_config(candidate, 2, 4))
        if norm != 'count':
            with pytest.raises(ValueError, match='fresh training'):
                validate_checkpoint(payload, candidate)
            with pytest.raises(ValueError, match='fresh training'):
                resume_training_config(restored, candidate)
            new_payload = dict(config=candidate, architecture=architecture_contract(candidate))
            validate_checkpoint(new_payload, checkpoint_config(new_payload), models[-1])
    for model in models[1:]:
        assert list(model.state_dict()) == list(models[0].state_dict())
        for p, q in zip(model.parameters(), models[0].parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)


@pytest.mark.parametrize('kwargs', [dict(pooling='bad'), dict(normalization='l1'),
    dict(pooling='intensity_weighted_dictionary'), dict(pool_centering='dictionary_mean'),
    dict(normalization_epsilon=0), dict(normalization_epsilon=float('nan'))])
def test_invalid_combinations(he_config, kwargs):
    he_config['model']['spectral_encoder']['deep_sets'].update(kwargs)
    with pytest.raises(ValueError):
        component_settings(he_config['model'])


@pytest.mark.parametrize('norm', ['l1', 'l2'])
@pytest.mark.parametrize('kind', ['cnn', 'gatv2'])
def test_cached_evaluation_equivalence(he_config, norm, kind):
    c = he_config
    c['model']['spectral_encoder']['deep_sets'].update(
        pooling='intensity_weighted_dictionary', normalization=norm, pool_centering='dictionary_mean')
    c['model']['aggregation']['type'] = kind
    resolve_execution(c)
    direct = create_data_bundle(c)
    cached = create_data_bundle(enable(copy.deepcopy(c), validation_tiles=3,
        validation_feature_cache=True, validation_peak_budget=17, validation_gat_tiles=3))
    model = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    outputs = []
    try:
        with torch.no_grad():
            for bundle in (direct, cached):
                values = {}
                for _, batch, out in evaluation_groups(model, bundle.loaders['validation'], torch.device('cpu')):
                    values.update({int(row): out['latent'][i] for i, row in enumerate(batch['row_ids'])})
                outputs.append(values)
        assert outputs[0].keys() == outputs[1].keys()
        for row in outputs[0]:
            torch.testing.assert_close(outputs[0][row], outputs[1][row], atol=2e-6, rtol=1e-5)
    finally:
        for bundle in (direct, cached):
            for dataset in bundle.datasets.values():
                dataset.close()


def test_observation_preserves_training_rng_parameters_and_modes(he_config, tmp_path, monkeypatch):
    import random
    import dann.train as training
    c=he_config
    for group in [c['model']['aggregation'],*c['model']['heads'].values()]:group['type']='mlp'
    c['model']['dropout']=.2
    c['model']['spectral_encoder']['deep_sets'].update(pooling='intensity_weighted_dictionary',
        normalization='l1',pool_centering='dictionary_mean')
    c['training'].update(epochs=1,early_stopping_enabled=False)
    c['monitoring']['enabled']=False
    c['performance']={}
    monkeypatch.setattr(training,'refresh_loss_curves',lambda *a,**kw: None)
    seen=[]
    def observe(event,model,bundle,config,record):
        assert not model.training and not torch.is_grad_enabled()
        seen.append(event)
        random.random();np.random.random(17);torch.rand(19)
        model.train()
    weights=[]
    for name,observer in [('unobserved',None),('observed',observe)]:
        c['results']['run_name']=name
        resolve_execution(c)
        path=train_model(c,observer=observer,observer_updates=(1,2))
        weights.append(torch.load(path,map_location='cpu',weights_only=False)['model_state'])
    assert seen[:2]==['parameters_initialized','heads_initialized']
    assert 'update' in seen
    for name in weights[0]:
        torch.testing.assert_close(weights[0][name],weights[1][name],atol=0,rtol=0)


def test_l1_l2_differ_only_by_row_norm_at_fixed_dictionary():
    l1 = encoder('l1').double().eval()
    l2 = copy.deepcopy(l1)
    l2.normalization = 'l2'
    ids, values, rows, counts = spectra()
    dense = values.new_zeros((len(counts), l1.num_peaks)).index_put((rows, ids), values, accumulate=True)
    ratio = dense.sum(1).clamp_min(l1.normalization_epsilon) / dense.square().sum(1).clamp_min(l1.normalization_epsilon**2).sqrt()
    torch.testing.assert_close(l2(ids, values, rows, counts), l1(ids, values, rows, counts) * ratio[:, None])


def test_version_eight_requires_explicit_pooling_metadata(he_config):
    contract = architecture_contract(he_config)
    contract['components']['spectral_encoder']['deep_sets'].pop('normalization')
    with pytest.raises(ValueError, match='Version 8.*explicit pooling'):
        checkpoint_config(dict(config=he_config, architecture=contract))

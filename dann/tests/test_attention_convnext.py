"""Numerical references and execution contracts for attention + ConvNeXt."""
import copy

import h5py
import numpy as np
import pytest
import torch

from dann.checkpoints import architecture_contract, checkpoint_config, validate_checkpoint
from dann.config import component_settings, execution_settings, resolve_execution
from dann.data_loader import create_data_bundle
from dann.execution import evaluation_groups
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.model_components.aggregation_convnext import ConvNeXtAggregation, convnext_reach
from dann.model_components.deep_sets import DeepSetsEncoder
from dann.tests.test_components import spatial_config, predict_tiles
from dann.tests.test_execution import enable
from dann.tests.test_he import he_config
from dann.train import load_training_checkpoint, save_checkpoint, resume_training_config


def attention_encoder(dropout=0.):
    return DeepSetsEncoder(5, 4, [7], 6, dropout=dropout, pooling='attention',
        normalization='softmax', attention=dict(num_queries=2, token_dim=5, key_dim=3, value_dim=4))


def spectra(dtype=torch.double, device='cpu'):
    # Duplicate peak, explicit zero, an empty row, and a zero-only row.
    return (torch.tensor([0, 0, 2, 1, 4, 3, 2], device=device),
            torch.tensor([.2, .3, 0., 1., 2., .4, 0.], dtype=dtype, device=device),
            torch.tensor([0, 0, 0, 1, 1, 3, 4], device=device),
            torch.tensor([3, 2, 0, 1, 1], device=device))


def dense_attention_reference(model, ids, intensity, rows, counts):
    dense = intensity.new_zeros((len(counts), model.num_peaks)).index_put(
        (rows, ids), intensity, accumulate=True)
    a = model.attention_pool
    outputs = []
    for values in dense:
        selected = (values > 0).nonzero().flatten()
        if not len(selected):
            outputs.append(values.new_zeros(model.output_dim))
            continue
        x = values[selected]
        token = model.peak_mlp(torch.cat((model.embedding(selected), x.log1p()[:, None]), dim=1))
        scores = a.queries @ a.keys(token).T / a.settings['key_dim'] ** .5 + x.log()[None]
        outputs.append(a.output((scores.softmax(-1) @ a.values(token)).flatten()))
    return torch.stack(outputs)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_attention_matches_dense_outputs_and_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(9)
    model = attention_encoder().double().to(device)
    reference = copy.deepcopy(model)
    ids, x, rows, counts = spectra(device=device)
    x, y = x.requires_grad_(), x.detach().clone().requires_grad_()
    actual = model(ids, x, rows, counts)
    expected = dense_attention_reference(reference, ids, y, rows, counts)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-10)
    for p, q in zip(model.parameters(), reference.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, atol=1e-12, rtol=1e-10)


def test_attention_permutation_duplicates_zeros_and_bounded_inference():
    model = attention_encoder().double().eval()
    args = spectra()
    ids, x, rows, counts = args
    with torch.no_grad():
        expected = model(*args)
        order = torch.tensor([6, 3, 2, 0, 5, 1, 4])
        torch.testing.assert_close(model(ids[order], x[order], rows[order], counts), expected)
        torch.testing.assert_close(model(torch.tensor([0, 1, 4, 3]),
            x.new_tensor([.5, 1., 2., .4]), torch.tensor([0, 1, 1, 3]),
            torch.tensor([1, 2, 0, 1, 0])), expected)
        calls = []
        hook = model.peak_mlp.register_forward_hook(lambda m, i, o: calls.append(len(i[0])))
        model.inference_peak_chunk_size = 1
        actual = model(*args)
        hook.remove()
        assert max(calls) == 1
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        assert torch.equal(actual[[2, 4]], torch.zeros_like(actual[[2, 4]]))
        # Changing intensity must change the representation, even for one peak.
        changed = x.clone(); changed[5] *= 3
        assert not torch.allclose(model(ids, changed, rows, counts)[3], expected[3])


@pytest.mark.parametrize('empty', [True, False])
def test_attention_zero_spectra_have_finite_zero_gradients(empty):
    model = attention_encoder().double()
    ids, x, rows, counts = spectra()
    if empty:
        ids, x, rows, counts = ids[:0], x[:0], rows[:0], torch.zeros(5, dtype=torch.long)
    x = torch.zeros_like(x, requires_grad=True)
    result = model(ids, x, rows, counts)
    assert torch.equal(result, torch.zeros_like(result))
    result.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_attention_extreme_positive_intensities_remain_finite():
    model = attention_encoder().eval()
    ids = torch.tensor([0, 1, 2, 3])
    values = torch.tensor([1e-30, 1e30, 1e-20, 1e20], requires_grad=True)
    rows = torch.tensor([0, 0, 1, 1])
    result = model(ids, values, rows, torch.tensor([2, 2]))
    assert torch.isfinite(result).all()
    result.square().sum().backward()
    assert torch.isfinite(values.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('dropout', [0., .3])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_attention_checkpointed_microbatches_preserve_values_gradients_rng(dropout, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    base = attention_encoder(dropout).double().to(device)
    args = spectra(device=device)
    results = []
    for checkpointed in (False, True):
        model = copy.deepcopy(base).train()
        torch.manual_seed(87)
        result = model.microbatched(*args, peak_budget=3, checkpointing=checkpointed)
        result.square().sum().backward()
        results.append((result, [p.grad for p in model.parameters()], torch.get_rng_state(),
                        torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state()))
    torch.testing.assert_close(results[0][0], results[1][0])
    for p, q in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(p, q)
    assert torch.equal(results[0][2], results[1][2])
    assert torch.equal(results[0][3], results[1][3])
    if dropout == 0:
        torch.testing.assert_close(base(*args), results[0][0])


def configure(config, *, attention=True, biology='mlp'):
    config['model']['aggregation'].update(type='convnext', convnext=dict(
        channels=[4, 8, 12], depths=[1, 1, 1], kernel_size=3, expansion=2,
        pyramid_dim=5, layer_scale_init=.5, dropout=0.))
    config['model']['heads']['biology']['type'] = biology
    config['model']['heads']['discriminator']['type'] = 'mlp'
    if attention:
        config['model']['spectral_encoder']['deep_sets'].update(
            pooling='attention', normalization='softmax', pool_centering='none',
            attention=dict(num_queries=2, token_dim=5, key_dim=3, value_dim=4))
    config['model']['spectral_peak_budget'] = 128
    return resolve_execution(config)


def test_masked_pyramid_ignores_holes_and_pools_only_occupied_children():
    model = ConvNeXtAggregation(3, 6, channels=[4, 8, 12], depths=[1, 1, 1],
                               kernel_size=3, dropout=0., layer_scale_init=.5).double().eval()
    x = torch.randn(2, 3, 17, 15, dtype=torch.double, requires_grad=True)
    mask = (torch.rand(2, 1, 17, 15) > .3).double()
    expected = model(x, mask)
    torch.testing.assert_close(model(x + (1-mask) * 10000, mask), expected)
    assert torch.count_nonzero(expected * (1-mask)) == 0
    expected.square().sum().backward()
    assert torch.count_nonzero(x.grad * (1-mask)) == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    empty = model(x.detach(), torch.zeros_like(mask))
    assert torch.equal(empty, torch.zeros_like(empty))
    pooled, valid = model.pool(torch.tensor([[[[2., 900.], [4., 900.]]]]),
                               torch.tensor([[[[1., 0.], [1., 0.]]]]))
    assert pooled.item() == 3 and valid.item() == 1


@pytest.mark.parametrize('biology', ['mlp', 'cnn', 'gatv2'])
def test_pyramid_tiling_matches_large_reference_with_negative_origins(spatial_config, biology):
    c = configure(spatial_config, biology=biology)
    # Test odd cores, changing pooling phase, holes, and negative global coordinates.
    for path in (c['data']['path'], c['data']['context_path']):
        with h5py.File(path, 'r+') as handle:
            handle['obs/x'][:] -= 5
            handle['obs/y'][:] -= 3
    model = AdversarialLatentFusion.from_config(c, 2, 4).double().eval()
    rows = np.arange(156)
    c['training']['core_size'] = 32
    reference = predict_tiles(c, model, rows)
    for size in (3, 5, 8):
        c['training']['core_size'] = size
        actual = predict_tiles(c, model, rows)
        for row in rows:
            for name in reference[row]:
                torch.testing.assert_close(actual[row][name], reference[row][name], atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize('attention', [False, True])
def test_combined_optimizer_checkpoint_and_cached_evaluation(spatial_config, tmp_path, attention):
    c = configure(spatial_config, attention=attention)
    bundle = create_data_bundle(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    try:
        batch = next(iter(bundle.loaders['train']))
        outputs = model(batch)
        assert outputs['latent'].shape == (len(batch['row_ids']), c['model']['latent_dim'])
        loss = ZILNLoss.from_config(c)(outputs['pi_logits'], outputs['mu'], outputs['sigma'],
                                      batch['targets'], batch['target_valid_mask']).total
        loss.backward()
        for component in (model.encoder.embedding, model.encoder.peak_mlp,
                          model.encoder.aggregation_mlp, model.biology_predictor):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())
        if attention:
            assert model.encoder.attention_pool.queries.grad.abs().sum() > 0
        optimizer.step()
        model.eval()
        expected = model(batch)
        path = tmp_path/'checkpoint.pt'
        save_checkpoint(path, model, optimizer, 0, c, bundle, 1.)
        restored = AdversarialLatentFusion.from_config(c, 2, 4).eval()
        load_training_checkpoint(path, restored, torch.optim.AdamW(restored.parameters()),
                                 torch.device('cpu'), c)
        for key, value in restored(batch).items():
            torch.testing.assert_close(value, expected[key])
        cached = create_data_bundle(enable(copy.deepcopy(c), validation_tiles=3,
            validation_feature_cache=True, validation_peak_budget=11))
        try:
            results = []
            with torch.no_grad():
                for data in (bundle, cached):
                    values = {}
                    for _, b, out in evaluation_groups(model, data.loaders['validation'], torch.device('cpu')):
                        values.update({int(row): out['latent'][i] for i, row in enumerate(b['row_ids'])})
                    results.append(values)
            assert results[0].keys() == results[1].keys()
            for row in results[0]:
                torch.testing.assert_close(results[0][row], results[1][row], atol=2e-6, rtol=1e-5)
        finally:
            for dataset in cached.datasets.values():
                dataset.close()
    finally:
        for dataset in bundle.datasets.values():
            dataset.close()


@pytest.mark.parametrize('mutation', ['query_count', 'pyramid_depth', 'latent_dim', 'semantics', 'version'])
def test_new_architecture_contract_rejects_changes(spatial_config, mutation):
    c = configure(spatial_config)
    payload = dict(config=c, architecture=architecture_contract(c))
    validate_checkpoint(payload, checkpoint_config(payload), AdversarialLatentFusion.from_config(c, 2, 4))
    bad = copy.deepcopy(payload)
    if mutation == 'query_count':
        bad['config']['model']['spectral_encoder']['deep_sets']['attention']['num_queries'] += 1
    elif mutation == 'pyramid_depth':
        bad['config']['model']['aggregation']['convnext']['depths'][1] += 1
    elif mutation == 'latent_dim':
        bad['config']['model']['latent_dim'] = 256
    elif mutation == 'semantics':
        bad['architecture']['components']['aggregation']['pyramid_semantics']['downsampling'] = 'unmasked'
    else:
        bad['architecture']['version'] = 10
    with pytest.raises(ValueError):
        checkpoint_config(bad)
    if mutation in ('query_count', 'pyramid_depth', 'latent_dim'):
        with pytest.raises(ValueError, match='fresh training'):
            resume_training_config(c, bad['config'])


def test_version10_remains_loadable_without_new_architecture(spatial_config):
    c = spatial_config
    payload = dict(config=c, architecture=architecture_contract(c))
    payload['architecture']['version'] = 10
    validate_checkpoint(payload, checkpoint_config(payload), AdversarialLatentFusion.from_config(c, 2, 4))


@pytest.mark.parametrize('kind', ['mlp', 'cnn', 'gatv2'])
def test_attention_can_use_existing_aggregators(spatial_config, kind):
    c = configure(spatial_config)
    c['model']['aggregation']['type'] = kind
    resolve_execution(c)
    bundle = create_data_bundle(c)
    try:
        model = AdversarialLatentFusion.from_config(c, 2, 4)
        result = model(next(iter(bundle.loaders['train'])))
        assert torch.isfinite(result['latent']).all()
        result['mu'].square().sum().backward()
        assert model.encoder.attention_pool.queries.grad.abs().sum() > 0
    finally:
        for dataset in bundle.datasets.values():
            dataset.close()


def test_combined_cca_gradients_and_msi_only_inference(he_config):
    from dann.deep_cca import evaluate_cca
    from dann.initialization import initialize_supervised_heads
    c = configure(he_config)
    c['model']['heads']['he'].update(objective='deep_cca', type='mlp',
        deep_cca=dict(dimension=2, msi_hidden_dim=7, he_hidden_dim=8,
                      biology_only_epochs=0, ramp_epochs=0))
    c['training']['spatial'].update(supervised_rows_per_batch=128, sampling_strategy='proportional_slide_tiles')
    resolve_execution(c)
    bundle = create_data_bundle(c)
    try:
        model = AdversarialLatentFusion.from_config(c, 2, 4)
        initialize_supervised_heads(model, bundle, c)
        batch = next(iter(bundle.loaders['train']))
        output = model(batch)
        loss = evaluate_cca(output['cca_msi'], output['cca_he'], batch['batches']).loss
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        for module in (model.encoder.attention_pool, model.encoder.aggregation_mlp,
                       model.he_cca.msi, model.he_cca.he):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        model.eval()
        expected = model(batch, predict_he=False)
        actual = model({k: v for k, v in batch.items() if k != 'he_targets'})
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key])
    finally:
        for dataset in bundle.datasets.values():
            dataset.close()


@pytest.mark.parametrize('settings', [dict(channels=[]), dict(depths=[1]), dict(kernel_size=4),
    dict(expansion=True), dict(dropout=float('nan')), dict(layer_scale_init=-1), dict(unknown=1)])
def test_invalid_convnext_settings(spatial_config, settings):
    c = configure(spatial_config)
    c['model']['aggregation']['convnext'].update(settings)
    with pytest.raises(ValueError):
        component_settings(c['model'])


@pytest.mark.parametrize('settings', [dict(num_queries=0), dict(token_dim=True), dict(key_dim=-1),
                                    dict(value_dim=1.5), dict(unknown=1)])
def test_invalid_attention_settings(spatial_config, settings):
    c = configure(spatial_config)
    c['model']['spectral_encoder']['deep_sets']['attention'].update(settings)
    with pytest.raises(ValueError):
        component_settings(c['model'])


def test_default_receptive_field_and_aggregation_only_selector(spatial_config):
    assert convnext_reach({}) == 24
    c = configure(spatial_config)
    assert execution_settings(c)['grid_alignment'] == 4
    assert execution_settings(c)['halo'] == 10
    c['model']['heads']['biology']['type'] = 'convnext'
    with pytest.raises(ValueError, match='component type'):
        component_settings(c['model'])

"""Production residual numerics, configuration, and historical checkpoint contracts."""
import copy

import pytest
import torch

from dann.analyze import load_model
from dann.checkpoints import architecture_contract, checkpoint_config, validate_checkpoint
from dann.config import component_settings, execution_settings, resolve_execution
from dann.data_loader import create_data_bundle
from dann.model import AdversarialLatentFusion
from dann.model_components.biology_cnn import CNNBiology
from dann.model_components.layers import SpatialNetwork
from dann.spatial import load_checkpoint_model
from dann.targets import checkpoint_analysis_config
from dann.tests.test_components import spatial_config
from dann.train import load_training_checkpoint, save_checkpoint
from dann.verification.biology_variants import ResidualBiology, make_variant


@pytest.mark.parametrize('setting', [None, True, False])
def test_residual_configuration_default_and_geometry(spatial_config, setting):
    config = spatial_config
    cnn = config['model']['heads']['biology']['cnn']
    cnn.pop('residual')
    expected = True if setting is None else setting
    if setting is not None:
        cnn['residual'] = setting
    geometry = execution_settings(config)
    assert component_settings(config['model'])['heads']['biology']['cnn']['residual'] is expected
    resolve_execution(config)
    assert cnn['residual'] is expected
    model = AdversarialLatentFusion.from_config(config, 2, 4)
    assert model.biology_predictor.residual is expected
    cnn['residual'] = not expected
    assert execution_settings(config) == geometry
    assert model.halo == geometry['halo']


@pytest.mark.parametrize('value', [0, 1, 'true', 'false', None, [], {}])
def test_residual_requires_boolean(spatial_config, value):
    spatial_config['model']['heads']['biology']['cnn']['residual'] = value
    with pytest.raises(ValueError, match='residual must be boolean'):
        resolve_execution(spatial_config)
    with pytest.raises(ValueError, match='residual must be boolean'):
        CNNBiology(3, 2, residual=value)


@pytest.mark.parametrize('residual', [False, True])
@pytest.mark.parametrize('dropout', [0., .3])
def test_residual_forward_gradients_initialization_and_rng(residual, dropout):
    torch.manual_seed(123)
    original = SpatialNetwork(4, 3, channels=5, depth=3, dropout=dropout).double()
    initial_rng = torch.get_rng_state().clone()
    torch.manual_seed(123)
    production = CNNBiology(4, 3, channels=5, depth=3, dropout=dropout,
                            residual=residual).double()
    assert torch.equal(initial_rng, torch.get_rng_state())
    assert list(original.state_dict()) == list(production.state_dict())
    assert list(dict(original.named_parameters())) == list(dict(production.named_parameters()))
    for name, value in original.state_dict().items():
        assert torch.equal(value, production.state_dict()[name])
    reference = ResidualBiology(original) if residual else original
    mask = torch.ones(2, 1, 9, 9, dtype=torch.float64)
    mask[:, :, 3, 4] = 0
    x = torch.randn(2, 4, 9, 9, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    torch.manual_seed(41)
    a = reference(x, mask)
    a.square().sum().backward()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(41)
    b = production(y, mask)
    b.square().sum().backward()
    assert torch.equal(expected_rng, torch.get_rng_state())
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for left, right in zip(reference.parameters(), production.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize('residual', [False, True])
def test_masking_and_receptive_field(residual):
    torch.manual_seed(33)
    model = CNNBiology(4, 3, channels=5, depth=2, dropout=0., residual=residual).double()
    x = torch.randn(1, 4, 11, 11, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 1, 11, 11, dtype=torch.float64)
    mask[:, :, 4, 5] = 0
    output = model(x, mask)
    assert torch.count_nonzero(output[:, :, 4, 5]) == 0
    gradient = torch.autograd.grad(output[0, 0, 5, 5], x)[0]
    assert torch.count_nonzero(gradient[:, :, 4, 5]) == 0
    outside = gradient.clone()
    outside[:, :, 3:8, 3:8] = 0
    assert torch.count_nonzero(outside) == 0
    assert gradient[:, :, 5, 5].norm() > 0
    assert gradient[:, :, 3, 3].norm() > 0
    assert model.radius == 2
    changed = x.detach().clone()
    changed[:, :, 4, 5] = 1e6
    torch.testing.assert_close(model(changed, mask), output, rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['enabled', 'disabled', 'version2', 'unversioned', 'omitted_default'])
def test_checkpoint_predictions_resume_and_analysis(spatial_config, tmp_path, kind):
    config = spatial_config
    cnn = config['model']['heads']['biology']['cnn']
    enabled = kind in ('enabled', 'omitted_default')
    cnn['residual'] = enabled
    if kind == 'omitted_default':
        del cnn['residual']
    bundle = create_data_bundle(config)
    try:
        model = AdversarialLatentFusion.from_config(config, 2, 4)
        optimizer = torch.optim.AdamW(model.parameters())
        batch = next(iter(bundle.loaders['train']))
        model(batch)['mu'].square().sum().backward()
        optimizer.step()
        model.eval()
        expected = model(batch)
        path = tmp_path / 'checkpoint.pt'
        save_checkpoint(path, model, optimizer, 0, config, bundle, 1.)
        payload = torch.load(path, weights_only=False)
        assert payload['architecture']['version'] == 3
        assert payload['config']['model']['heads']['biology']['cnn']['residual'] is enabled
        if kind in ('version2', 'unversioned'):
            del payload['config']['model']['heads']['biology']['cnn']['residual']
            del payload['architecture']['components']['biology']['cnn']['residual']
            payload['architecture']['version'] = 2
            if kind == 'unversioned':
                del payload['architecture']
            torch.save(payload, path)
        loaded, frozen, *_ = load_checkpoint_model(path, 'cpu')
        assert loaded.biology_predictor.residual is enabled
        assert frozen['model']['heads']['biology']['cnn']['residual'] is enabled
        for name, value in loaded(batch).items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        analysis = checkpoint_analysis_config(config, payload)
        assert analysis['model']['heads']['biology']['cnn']['residual'] is enabled
        analyzed, _ = load_model(path, analysis, bundle, torch.device('cpu'))
        for name, value in analyzed(batch).items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        fresh = AdversarialLatentFusion.from_config(config, 2, 4)
        resumed_optimizer = torch.optim.AdamW(fresh.parameters())
        assert load_training_checkpoint(path, fresh, resumed_optimizer, torch.device('cpu'), config) == (1, 1.)
        assert len(resumed_optimizer.state) == len(optimizer.state)
        for original, resumed in zip(optimizer.state.values(), resumed_optimizer.state.values()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(original[key], resumed[key], rtol=0, atol=0)
        incompatible = copy.deepcopy(config)
        incompatible['model']['heads']['biology']['cnn']['residual'] = not enabled
        with pytest.raises(ValueError, match='incompatible'):
            load_training_checkpoint(path, fresh, resumed_optimizer, torch.device('cpu'), incompatible)
        if kind in ('version2', 'unversioned'):
            assert 'residual' not in payload['config']['model']['heads']['biology']['cnn']
    finally:
        for dataset in bundle.datasets.values():
            dataset.close()


@pytest.mark.parametrize('change', ['flag', 'missing_config', 'missing_metadata', 'nonboolean', 'version'])
def test_reject_inconsistent_checkpoint_metadata(spatial_config, change):
    payload = dict(config=copy.deepcopy(spatial_config), architecture=architecture_contract(spatial_config))
    config_cnn = payload['config']['model']['heads']['biology']['cnn']
    metadata = payload['architecture']['components']['biology']['cnn']
    if change == 'flag':
        config_cnn['residual'] = False
    elif change == 'missing_config':
        del config_cnn['residual']
    elif change == 'missing_metadata':
        del metadata['residual']
    elif change == 'nonboolean':
        metadata['residual'] = 1
    else:
        payload['architecture']['version'] = 99
    with pytest.raises(ValueError, match='Checkpoint|checkpoint'):
        checkpoint_config(payload)


def test_runtime_model_flag_must_match_checkpoint(spatial_config):
    payload = dict(config=spatial_config, architecture=architecture_contract(spatial_config))
    model = AdversarialLatentFusion.from_config(spatial_config, 2, 4)
    model.biology_predictor.residual = False
    with pytest.raises(ValueError, match='residual setting is incompatible'):
        validate_checkpoint(payload, spatial_config, model)


def test_historical_diagnostic_baseline_stays_plain(spatial_config):
    spatial_config['model']['aggregation']['type'] = 'mlp'
    spatial_config['model']['heads']['biology']['cnn']['depth'] = 3
    baseline, _ = make_variant(spatial_config, 2, 19, 'C')
    assert not baseline.biology_predictor.residual
    assert spatial_config['model']['heads']['biology']['cnn']['residual'] is True

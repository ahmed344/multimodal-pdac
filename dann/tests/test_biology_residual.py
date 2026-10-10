"""Production residual numerics, configuration, and historical checkpoint contracts."""
import copy
import itertools

import pytest
import torch

from dann.analyze import load_model
from dann.checkpoints import architecture_contract, checkpoint_config, validate_checkpoint
from dann.config import component_settings, execution_settings, resolve_execution
from dann.data_loader import create_data_bundle
from dann.model import AdversarialLatentFusion
from dann.model_components.biology_cnn import CNNBiology
from dann.model_components.aggregation_cnn import CNNAggregation
from dann.model_components.discriminator_cnn import CNNDiscriminator
from dann.model_components.layers import SpatialNetwork
from dann.spatial import load_checkpoint_model
from dann.targets import checkpoint_analysis_config
from dann.tests.test_components import spatial_config
from dann.train import load_training_checkpoint, resume_training_config, save_checkpoint
from dann.verification.biology_variants import ResidualBiology, make_variant


COMPONENTS = {'aggregation': CNNAggregation, 'biology': CNNBiology,
              'discriminator': CNNDiscriminator}


def component_group(config, name):
    return config['model']['aggregation'] if name == 'aggregation' else config['model']['heads'][name]


def model_component(model, name):
    return {'aggregation': model.encoder.aggregation_mlp, 'biology': model.biology_predictor,
            'discriminator': model.batch_discriminator}[name]


class PlainCNNReference(SpatialNetwork):
    """Frozen original forward calculation, independent of residual branching."""

    def forward(self, value, mask):
        value = self.hidden(self.input(value * mask), self.norms[0], mask)
        for convolution, norm in zip(self.convolutions, self.norms[1:]):
            value = self.hidden(convolution(value), norm, mask)
        return self.output(value) * mask


@pytest.mark.parametrize('name', COMPONENTS)
@pytest.mark.parametrize('setting', [None, True, False])
def test_residual_configuration_default_and_geometry(spatial_config, name, setting):
    config = spatial_config
    group = component_group(config, name)
    group['type'] = 'cnn'
    cnn = group['cnn']
    cnn.pop('residual')
    expected = True if setting is None else setting
    if setting is not None:
        cnn['residual'] = setting
    geometry = execution_settings(config)
    settings = component_settings(config['model'])
    resolved = settings['aggregation'] if name == 'aggregation' else settings['heads'][name]
    assert resolved['cnn']['residual'] is expected
    resolve_execution(config)
    assert cnn['residual'] is expected
    model = AdversarialLatentFusion.from_config(config, 2, 4)
    assert model_component(model, name).residual is expected
    cnn['residual'] = not expected
    assert execution_settings(config) == geometry
    assert model.halo == geometry['halo']


@pytest.mark.parametrize('name', COMPONENTS)
@pytest.mark.parametrize('value', [0, 1, 'true', 'false', None, [], {}])
def test_residual_requires_boolean(spatial_config, name, value):
    component_group(spatial_config, name)['cnn']['residual'] = value
    with pytest.raises(ValueError, match='residual must be boolean'):
        resolve_execution(spatial_config)
    with pytest.raises(ValueError, match='residual must be boolean'):
        COMPONENTS[name](3, 2, residual=value)


def test_constructor_defaults():
    assert not SpatialNetwork(3, 2).residual
    for constructor in COMPONENTS.values():
        assert constructor(3, 2).residual


@pytest.mark.parametrize('residual', [False, True])
@pytest.mark.parametrize('dropout', [0., .3])
@pytest.mark.parametrize('constructor', COMPONENTS.values(), ids=COMPONENTS)
def test_residual_forward_gradients_initialization_and_rng(residual, dropout, constructor):
    torch.manual_seed(123)
    original = PlainCNNReference(4, 3, channels=5, depth=3, dropout=dropout).double()
    initial_rng = torch.get_rng_state().clone()
    torch.manual_seed(123)
    production = constructor(4, 3, channels=5, depth=3, dropout=dropout,
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
@pytest.mark.parametrize('constructor', COMPONENTS.values(), ids=COMPONENTS)
def test_masking_and_receptive_field(residual, constructor):
    torch.manual_seed(33)
    model = constructor(4, 3, channels=5, depth=2, dropout=0., residual=residual).double()
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


@pytest.mark.parametrize('kind,flags', [
    *[('version4', flags) for flags in itertools.product((False, True), repeat=3)],
    ('version3', (False, True, False)), ('version3', (False, False, False)),
    ('version2', (False, False, False)), ('unversioned', (False, False, False)),
    ('omitted_default', (True, True, True)),
])
def test_checkpoint_predictions_resume_and_analysis(spatial_config, tmp_path, kind, flags):
    config = spatial_config
    for name, enabled in zip(COMPONENTS, flags):
        group = component_group(config, name)
        group['type'] = 'cnn'
        group['cnn']['residual'] = enabled
        if kind == 'omitted_default':
            del group['cnn']['residual']
    resolve_execution(config)
    bundle = create_data_bundle(config)
    try:
        if kind == 'omitted_default':
            for name in COMPONENTS:
                del component_group(config, name)['cnn']['residual']
        model = AdversarialLatentFusion.from_config(config, 2, 4)
        optimizer = torch.optim.AdamW(model.parameters())
        batch = next(iter(bundle.loaders['train']))
        output = model(batch)
        (output['mu'].square().sum() + output['batch_logits'].square().sum()).backward()
        optimizer.step()
        model.eval()
        expected = model(batch)
        path = tmp_path / 'checkpoint.pt'
        save_checkpoint(path, model, optimizer, 0, config, bundle, 1.)
        payload = torch.load(path, weights_only=False)
        assert payload['architecture']['version'] == 5
        payload['architecture']['version'] = 4  # Exercise the historical v4 contract.
        for name, enabled in zip(COMPONENTS, flags):
            assert component_group(payload['config'], name)['cnn']['residual'] is enabled
        legacy = kind in ('version2', 'version3', 'unversioned')
        if legacy:
            for name in COMPONENTS:
                if kind == 'version3' and name == 'biology':
                    continue
                del component_group(payload['config'], name)['cnn']['residual']
                del payload['architecture']['components'][name]['cnn']['residual']
            payload['architecture']['version'] = 3 if kind == 'version3' else 2
            if kind == 'unversioned':
                del payload['architecture']
            torch.save(payload, path)
        loaded, frozen, *_ = load_checkpoint_model(path, 'cpu')
        for name, enabled in zip(COMPONENTS, flags):
            assert model_component(loaded, name).residual is enabled
            assert component_group(frozen, name)['cnn']['residual'] is enabled
        for name, value in loaded(batch).items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        analysis = checkpoint_analysis_config(config, payload)
        for name, enabled in zip(COMPONENTS, flags):
            assert component_group(analysis, name)['cnn']['residual'] is enabled
        analyzed, _ = load_model(path, analysis, bundle, torch.device('cpu'))
        for name, value in analyzed(batch).items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        requested = copy.deepcopy(config)
        for name, enabled in zip(COMPONENTS, flags):
            component_group(requested, name)['cnn']['residual'] = not enabled
        resumed_config = resume_training_config(checkpoint_config(payload), requested)
        fresh = AdversarialLatentFusion.from_config(resumed_config, 2, 4)
        resumed_optimizer = torch.optim.AdamW(fresh.parameters())
        assert load_training_checkpoint(path, fresh, resumed_optimizer, torch.device('cpu'), resumed_config) == (1, 1.)
        fresh.eval()
        for name, value in fresh(batch).items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        assert len(resumed_optimizer.state) == len(optimizer.state)
        for original, resumed in zip(optimizer.state.values(), resumed_optimizer.state.values()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(original[key], resumed[key], rtol=0, atol=0)
        for name, enabled in zip(COMPONENTS, flags):
            incompatible = copy.deepcopy(resumed_config)
            component_group(incompatible, name)['cnn']['residual'] = not enabled
            with pytest.raises(ValueError, match=f'{name.capitalize()} CNN residual setting differs'):
                load_training_checkpoint(path, fresh, resumed_optimizer, torch.device('cpu'), incompatible)
            if legacy and (kind != 'version3' or name != 'biology'):
                assert 'residual' not in component_group(payload['config'], name)['cnn']
    finally:
        for dataset in bundle.datasets.values():
            dataset.close()


@pytest.mark.parametrize('name', COMPONENTS)
@pytest.mark.parametrize('change', ['flag', 'missing_config', 'missing_metadata', 'nonboolean', 'nonboolean_config', 'version'])
def test_reject_inconsistent_checkpoint_metadata(spatial_config, name, change):
    component_group(spatial_config, name)['type'] = 'cnn'
    payload = dict(config=copy.deepcopy(spatial_config), architecture=architecture_contract(spatial_config))
    config_cnn = component_group(payload['config'], name)['cnn']
    metadata = payload['architecture']['components'][name]['cnn']
    if change == 'flag':
        config_cnn['residual'] = False
    elif change == 'missing_config':
        del config_cnn['residual']
    elif change == 'missing_metadata':
        del metadata['residual']
    elif change == 'nonboolean':
        metadata['residual'] = 1
    elif change == 'nonboolean_config':
        config_cnn['residual'] = 1
    else:
        payload['architecture']['version'] = 99
    with pytest.raises(ValueError, match='Checkpoint|checkpoint'):
        checkpoint_config(payload)


@pytest.mark.parametrize('name', COMPONENTS)
def test_runtime_model_flag_must_match_checkpoint(spatial_config, name):
    component_group(spatial_config, name)['type'] = 'cnn'
    payload = dict(config=spatial_config, architecture=architecture_contract(spatial_config))
    model = AdversarialLatentFusion.from_config(spatial_config, 2, 4)
    model_component(model, name).residual = False
    with pytest.raises(ValueError, match='residual setting is incompatible'):
        validate_checkpoint(payload, spatial_config, model)


@pytest.mark.parametrize('missing', ['config', 'metadata'])
def test_version3_still_requires_explicit_biology_flag(spatial_config, missing):
    config = copy.deepcopy(spatial_config)
    config['model']['aggregation']['cnn']['residual'] = False
    payload = dict(config=config, architecture=architecture_contract(config))
    payload['architecture']['version'] = 3
    del config['model']['aggregation']['cnn']['residual']
    del payload['architecture']['components']['aggregation']['cnn']['residual']
    biology = (config['model']['heads']['biology'] if missing == 'config'
               else payload['architecture']['components']['biology'])
    del biology['cnn']['residual']
    with pytest.raises(ValueError, match='Checkpoint.*biology CNN residual setting'):
        checkpoint_config(payload)


def test_historical_diagnostic_baseline_stays_plain(spatial_config):
    spatial_config['model']['aggregation']['type'] = 'mlp'
    spatial_config['model']['heads']['biology']['cnn']['depth'] = 3
    baseline, _ = make_variant(spatial_config, 2, 19, 'C')
    assert not baseline.biology_predictor.residual
    assert spatial_config['model']['heads']['biology']['cnn']['residual'] is True

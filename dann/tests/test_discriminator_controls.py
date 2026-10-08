"""Numerical isolation and artifact contracts for matched discriminator controls."""
import copy
import json

import numpy as np
import torch
import yaml

from dann.config import resolve_execution
from dann.data_loader import create_data_bundle
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.tests.test_components import spatial_config
from dann.tiles import SpatialTileDataset, tile_collate
from dann.train import save_checkpoint
from dann.verification.discriminator_controls import (
    evaluate, gradient_audit, main, make_control, objectives,
)
from dann.verification.epoch_comparison import flattened_tile_collate


def setup_control(config):
    for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
        group['type'] = 'mlp'
    config['model']['spectral_peak_budget'] = 4096
    resolve_execution(config)
    bundle = create_data_bundle(config)
    geometry = copy.deepcopy(config)
    geometry['model']['heads']['discriminator']['type'] = 'cnn'
    geometry['training']['core_size'] = config['training']['spatial']['core_size']
    tiles = SpatialTileDataset(geometry, bundle.split_indices['train'], bundle.metadata)
    selected = [tiles[i] for i in range(min(2, len(tiles)))]
    torch.manual_seed(41)
    reference = AdversarialLatentFusion.from_config(config, 2, 4)
    tiles.close()
    return bundle, reference, flattened_tile_collate(selected), tile_collate(selected)


def test_matched_initialization_core_inputs_and_pointwise_gradients(spatial_config):
    c = spatial_config
    bundle, reference, flat, tiled = setup_control(c)
    a = make_control(reference, c, False).double().eval()
    b = make_control(reference, c, True).double().eval()
    assert not b.batch_discriminator.residual
    assert c['model']['heads']['discriminator']['cnn']['residual'] is True
    reference.double().eval()
    flat = {k: v.double() if v.is_floating_point() else v for k, v in flat.items()}
    tiled = {k: v.double() if v.is_floating_point() else v for k, v in tiled.items()}
    for key in ('targets', 'target_valid_mask', 'batches', 'row_ids'):
        torch.testing.assert_close(flat[key], tiled[key])
    for key, value in reference.state_dict().items():
        if not key.startswith('batch_discriminator.'):
            torch.testing.assert_close(a.state_dict()[key], value, atol=0, rtol=0)
            torch.testing.assert_close(b.state_dict()[key], value, atol=0, rtol=0)
    loss = ZILNLoss.from_config(c).double()
    outputs = [model(batch) for model, batch in ((reference, flat), (a, tiled), (b, tiled))]
    for out in outputs[1:]:
        for key in ('latent', 'mu', 'sigma', 'pi_logits'):
            torch.testing.assert_close(out[key], outputs[0][key], atol=1e-9, rtol=1e-7)
    for model, batch in ((reference, flat), (a, tiled), (b, tiled)):
        bio, disc = objectives(model, batch, loss, 5., 0., .2, None)
        assert disc is None
        bio.backward()
        assert all(p.grad is None for p in model.batch_discriminator.parameters())
    for component in ('encoder', 'biology_predictor'):
        for p, q, r in zip(getattr(reference, component).parameters(), getattr(a, component).parameters(),
                           getattr(b, component).parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-8, rtol=1e-6)
            torch.testing.assert_close(p.grad, r.grad, atol=1e-8, rtol=1e-6)
    for dataset in bundle.datasets.values():
        dataset.close()


def test_zero_grl_retains_discriminator_gradient_and_audit_is_read_only(spatial_config):
    c = spatial_config
    bundle, reference, _, tiled = setup_control(c)
    model = make_control(reference, c, True).train()
    loss = ZILNLoss.from_config(c)
    _, disc = objectives(model, tiled, loss, 5., 1., 0., None)
    disc.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.biology_predictor.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.batch_discriminator.parameters())
    state = copy.deepcopy(model.state_dict())
    grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    result = gradient_audit(model, tiled, loss, c, 1., 0., None)
    assert result['encoder']['discriminator_l2'] == 0
    assert result['discriminator']['discriminator_l2'] > 0
    assert result['clip_factor'] <= result['biology_only_clip_factor']
    assert model.training and torch.equal(rng, torch.get_rng_state())
    for key, value in state.items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    for p, grad in zip(model.parameters(), grads):
        if grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)
    for dataset in bundle.datasets.values():
        dataset.close()


def test_full_validation_fast_path_matches_spatial_biology(spatial_config):
    c = spatial_config
    bundle, reference, _, _ = setup_control(c)
    model = make_control(reference, c, True).eval()
    scored, predictions = evaluate(model, bundle.loaders['validation'], bundle.metadata,
        ZILNLoss.from_config(c), torch.device('cpu'), bundle.split_indices['validation'])
    geometry = copy.deepcopy(c)
    geometry['model']['heads']['discriminator']['type'] = 'cnn'
    geometry['training']['core_size'] = 4
    tiles = SpatialTileDataset(geometry, bundle.split_indices['validation'], bundle.metadata)
    grid = tile_collate([tiles[i] for i in range(len(tiles))])
    with torch.no_grad():
        expected = model(grid)
    order = {row: i for i, row in enumerate(grid['row_ids'].tolist())}
    take = [order[row] for row in predictions['row_ids']]
    for key in ('pi_logits', 'mu', 'sigma'):
        torch.testing.assert_close(predictions[key], expected[key][take], atol=2e-6, rtol=2e-5)
    assert np.isfinite(scored['biology'])
    tiles.close()
    for dataset in bundle.datasets.values():
        dataset.close()


def test_runner_matches_order_preserves_schedule_and_saves_controls(spatial_config, tmp_path, monkeypatch):
    c = spatial_config
    bundle, reference, _, _ = setup_control(c)
    c['training']['epochs'] = 300
    source = tmp_path/'source.pt'
    save_checkpoint(source, reference, torch.optim.AdamW(reference.parameters()), 0, c, bundle, 1.)
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(c))
    from dann.verification import discriminator_controls as controls
    original = controls.evaluate
    def trigger_followups(*args, **kwargs):
        result, predictions = original(*args, **kwargs)
        result['positive_cd8_r2'] = -1.
        return result, predictions
    monkeypatch.setattr(controls, 'evaluate', trigger_followups)
    root = tmp_path/'diagnostic'
    main(['--config', str(config), '--reference-checkpoint', str(source), '--output', str(root),
          '--updates', '2', '--evaluate-every', '1'])
    completion = json.loads((root/'completion.json').read_text())
    assert completion['conditional_followups'] and len(completion['controls']) == 7
    results = json.loads((root/'results.json').read_text())
    assert len({results[k]['order_sha256'] for k in list(results)[:4]}) == 1
    for name in completion['controls']:
        history = json.loads((root/name/'history.json').read_text())
        assert [h['update'] for h in history] == [0, 1, 2]
        saved = torch.load(root/name/'final.pt', weights_only=False)
        assert saved['settings']['schedule_epochs'] == 300
        assert saved['settings']['config']['model']['heads']['discriminator']['cnn']['residual'] is False
        pred = torch.load(root/name/'validation_2.pt', weights_only=False)
        np.testing.assert_array_equal(pred['row_ids'], bundle.split_indices['validation'])
        assert not np.isin(pred['row_ids'], bundle.split_indices['test']).any()
    initial = [torch.load(root/name/'initial_state.pt', weights_only=True) for name in list(results)[:4]]
    for key in initial[0]:
        if key.startswith(('encoder.', 'biology_predictor.')):
            for state in initial[1:]:
                torch.testing.assert_close(initial[0][key], state[key], atol=0, rtol=0)
    for key in initial[1]:
        torch.testing.assert_close(initial[1][key], initial[2][key], atol=0, rtol=0)
        torch.testing.assert_close(initial[1][key], initial[3][key], atol=0, rtol=0)
    training = json.loads((root/'D_cnn_disc_ramp'/'training.json').read_text())
    settings = json.loads((root/'D_cnn_disc_ramp'/'settings.json').read_text())
    from dann.train import grl_strength
    assert training[-1]['grl_strength'] == grl_strength(
        1/(300*settings['steps_per_epoch']-1), c['training']['grl_schedule'],
        c['training']['grl_max_lambda'], c['training']['grl_gamma'])
    assert (root/'reference.pt').read_bytes() == source.read_bytes()

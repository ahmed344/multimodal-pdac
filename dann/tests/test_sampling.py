"""Production sampler parity, worker lifecycle, numerical and resume contracts."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dann.checkpoints import sampling_contract, validate_sampling_contract, validate_checkpoint
from dann.config import apply_smoke_overrides, resolve_execution
from dann.data_loader import create_data_bundle
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.sampling import ProportionalSlideTileSampler, SpatialBatchDataset, SamplingEpochAudit
from dann.tests.test_components import spatial_config
from dann.train import run_epoch, save_checkpoint, load_training_checkpoint, grl_strength
from dann.verification.extended_batch_diagnosis import proportional_plan
from dann.verification.batch_execution_diagnosis import PlannedDataset


@pytest.mark.parametrize('seed', [0, 20260719, 20260720, 20260721])
@pytest.mark.parametrize('budget', [1, 9, 64, 1000])
def test_exact_reference_stream(seed, budget):
    tiles, row = [], 0
    for slide, sizes in [('z', [1]), ('b', [17, 2, 8]), ('a', [3, 16, 16, 16])]:
        for i, size in enumerate(sizes):
            tiles.append(((slide, i, 0), [(r, r) for r in range(row, row + size)]))
            row += size
    source = SimpleNamespace(tiles=tiles)
    reference = proportional_plan(source, np.arange(row), seed, 3, budget)
    sampler = ProportionalSlideTileSampler(source, seed, budget)
    rng = np.random.get_state()
    for epoch in range(3):
        sampler.set_epoch(epoch)
        requests = list(sampler)
        assert len(requests) == len(sampler) == (row + budget - 1) // budget
        for i, request in enumerate(requests, int(reference.passes[epoch])):
            np.testing.assert_array_equal(request.row_ids, reference.batch(i))
            np.testing.assert_array_equal(request.tile_ids, reference.tile_order[
                reference.tile_boundaries[i]:reference.tile_boundaries[i + 1]])
        assert sorted(r for request in requests for r in request.row_ids) == list(range(row))
        resumed = ProportionalSlideTileSampler(source, seed, budget)
        resumed.set_epoch(epoch)
        assert list(resumed) == requests
    assert all(np.array_equal(a, b) for a, b in zip(rng, np.random.get_state()))


def small_bundle(config, budget=7):
    config['training']['spatial'].update(sampling_strategy='proportional_slide_tiles',
                                        supervised_rows_per_batch=budget)
    resolve_execution(config)
    return create_data_bundle(config)


def test_worker_epoch_and_full_context_reference(spatial_config):
    bundle = small_bundle(spatial_config)
    tiles = bundle.datasets['train']
    dataset = SpatialBatchDataset(tiles)
    samplers = [ProportionalSlideTileSampler(tiles, 71, 7) for _ in range(2)]
    loaders = [DataLoader(dataset, batch_size=None, sampler=s, num_workers=w,
                          persistent_workers=bool(w)) for s, w in zip(samplers, (0, 2))]
    reference = proportional_plan(tiles, tiles.indices, 71, 2, 7)
    planned = PlannedDataset(tiles, reference, True)
    assert tiles.halo == 4  # architecture-derived, deliberately neither three nor six
    assert tiles.size == 12
    try:
        for epoch in range(2):
            for s in samplers:
                s.set_epoch(epoch)
            batches = [list(loader) for loader in loaders]
            for step, (a, b) in enumerate(zip(*batches)):
                expected = planned[int(reference.passes[epoch]) + step]
                for key in expected:
                    torch.testing.assert_close(a[key], expected[key], atol=0, rtol=0, equal_nan=True)
                    torch.testing.assert_close(a[key], b[key], atol=0, rtol=0, equal_nan=True)
                assert set(a['row_ids'].tolist()) <= set(tiles.indices)
        # The same split tile has the same complete footprint in adjacent updates.
        requests = list(samplers[0])
        assert any(set(a.tile_ids) & set(b.tile_ids) for a, b in zip(requests, requests[1:]))
    finally:
        for loader in loaders:
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()
        for d in bundle.datasets.values():
            d.close()


def test_float64_forward_gradients_clipped_adamw_and_excluded_labels(spatial_config):
    c = spatial_config
    bundle = small_bundle(c, 7)
    tiles = bundle.datasets['train']
    reference = proportional_plan(tiles, tiles.indices, c['training']['seed'], 1, 7)
    reference = PlannedDataset(tiles, reference, True)
    models = [AdversarialLatentFusion.from_config(c, 2, 4).double()]
    models.append(copy.deepcopy(models[0]))
    opts = [torch.optim.AdamW(m.parameters(), lr=.001) for m in models]
    loss = ZILNLoss.from_config(c).double()
    for i, actual in enumerate(bundle.loaders['train']):
        batches = [{k: v.double() if v.is_floating_point() else v for k, v in b.items()}
                   for b in (actual, reference[i])]
        outputs = []
        for model, opt, batch in zip(models, opts, batches):
            opt.zero_grad(set_to_none=True)
            out = model(batch, grl_strength=.17)
            bio = loss(out['pi_logits'], out['mu'], out['sigma'], batch['targets'], batch['target_valid_mask']).total
            changed = batch['targets'].clone()
            changed[~batch['target_valid_mask']] = 12345.
            torch.testing.assert_close(bio, loss(out['pi_logits'], out['mu'], out['sigma'],
                                                 changed, batch['target_valid_mask']).total, atol=0, rtol=0)
            total = 5 * bio + torch.nn.functional.cross_entropy(out['batch_logits'], batch['batches'])
            total.backward()
            outputs.append(out)
        for key in outputs[0]:
            torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=1e-9, rtol=1e-7)
        for p, q in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-8, rtol=1e-6)
        for m, opt in zip(models, opts):
            torch.nn.utils.clip_grad_norm_(m.parameters(), .5)
            opt.step()
        for p, q in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-6)
    for d in bundle.datasets.values():
        d.close()


@pytest.mark.parametrize('key,value', [('core_size', True), ('tiles_per_batch', 0),
    ('supervised_rows_per_batch', True), ('supervised_rows_per_batch', 0),
    ('supervised_rows_per_batch', 3.5), ('sampling_strategy', 'equal_slides')])
def test_invalid_settings(spatial_config, key, value):
    spatial_config['training']['spatial'][key] = value
    with pytest.raises(ValueError):
        resolve_execution(spatial_config)


def test_config_legacy_idempotence_smoke_and_evaluation(spatial_config):
    c = spatial_config
    bundle = small_bundle(c, 7)
    assert bundle.loaders['train'].batch_size is None
    assert len(bundle.loaders['train']) == (len(bundle.split_indices['train']) + 6) // 7
    assert bundle.loaders['validation'].batch_size == bundle.loaders['test'].batch_size == 2
    expected = copy.deepcopy(c)
    assert resolve_execution(c) == expected
    smoke = apply_smoke_overrides(c)
    assert smoke['training']['supervised_rows_per_batch'] == 7
    assert smoke['training']['tiles_per_batch'] == 2
    assert smoke['execution']['training_supervised_rows_per_batch'] == 7
    for location in (c['training'], c['training']['spatial']):
        location.pop('sampling_strategy', None)
        location.pop('supervised_rows_per_batch', None)
    resolve_execution(c)
    assert c['training']['sampling_strategy'] == 'shuffled_tiles'
    assert c['training']['batch_size'] == 2
    legacy = create_data_bundle(c, bundle.split_indices)
    assert legacy.loaders['train'].batch_size == 2
    for b in (bundle, legacy):
        for d in b.datasets.values():
            d.close()


def test_checkpoint_contract_and_inference_independence(spatial_config, tmp_path):
    c = spatial_config
    bundle = small_bundle(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, model, opt, 2, c, bundle, 1.)
    checkpoint = torch.load(path, weights_only=False)
    assert load_training_checkpoint(path, model, opt, torch.device('cpu'), c,
                                    bundle.split_indices['train'])[0] == 3
    for key, value in [('seed', 41), ('core_size', 8), ('supervised_rows_per_batch', 13),
                       ('sampling_strategy', 'shuffled_tiles')]:
        changed = copy.deepcopy(c)
        changed['training'][key] = value
        with pytest.raises(ValueError, match='sampling contract'):
            validate_sampling_contract(checkpoint, changed)
        validate_checkpoint(checkpoint, changed)  # inference ignores training sampling
    with pytest.raises(ValueError, match='sampling contract'):
        validate_sampling_contract(checkpoint, c, bundle.split_indices['train'][::-1])
    old = copy.deepcopy(checkpoint)
    del old['sampling_contract']
    del old['sampler_state']
    with pytest.raises(ValueError, match='sampling contract'):
        validate_sampling_contract(old, c)
    legacy = copy.deepcopy(c)
    legacy['training']['spatial']['sampling_strategy'] = 'shuffled_tiles'
    resolve_execution(legacy)
    old['config'] = legacy
    validate_sampling_contract(old, legacy)
    checkpoint['sampler_state']['next_epoch'] = 8
    with pytest.raises(ValueError, match='epoch state'):
        validate_sampling_contract(checkpoint, c)
    for d in bundle.datasets.values():
        d.close()


def test_epoch_grl_and_persisted_composition(spatial_config, tmp_path):
    c = spatial_config
    bundle = small_bundle(c)
    loader = bundle.loaders['train']
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    strengths = []
    original = model.forward
    def forward(batch, grl_strength=0.):
        strengths.append(grl_strength)
        return original(batch, grl_strength=grl_strength)
    model.forward = forward
    run_epoch(model, loader, ZILNLoss.from_config(c), torch.device('cpu'), 5., 1., None,
              torch.optim.AdamW(model.parameters(), lr=.001), 2, 5, 'dann', .25, 10., 5.,
              target_columns=c['data']['target_columns'], sampling_audit_dir=tmp_path)
    assert loader.sampler.epoch == 2
    np.testing.assert_array_equal(strengths, [grl_strength((2 * len(loader) + i) / (5 * len(loader) - 1),
                                  'dann', .25, 10.) for i in range(len(loader))])
    report = json.loads((tmp_path / 'epoch_0002_summary.json').read_text())
    requests = list(loader.sampler)
    rows = np.asarray([r for request in requests for r in request.row_ids], dtype='<i8')
    assert report['complete'] and report['processed_rows'] == len(rows)
    assert report['row_order_sha256'] == hashlib.sha256(rows.tobytes()).hexdigest()
    assert len((tmp_path / 'epoch_0002_batches.jsonl').read_text().splitlines()) == len(loader)
    for d in bundle.datasets.values():
        d.close()


def test_spatial_batch_units_never_inherit_pixel_settings(spatial_config):
    c = spatial_config
    c['training']['pixel'].update(core_size=99, tiles_per_batch=99,
        sampling_strategy='proportional_slide_tiles', supervised_rows_per_batch=99)
    for key in ('core_size', 'tiles_per_batch', 'sampling_strategy', 'supervised_rows_per_batch'):
        c['training']['spatial'].pop(key, None)
    resolve_execution(c)
    assert c['training']['sampling_strategy'] == 'shuffled_tiles'
    assert c['training']['core_size'] == 32
    assert c['training']['batch_size'] == c['training']['tiles_per_batch'] == 8
    c['training']['spatial']['sampling_strategy'] = 'proportional_slide_tiles'
    resolve_execution(c)
    assert c['training']['supervised_rows_per_batch'] == c['training']['batch_size'] == 2048
    assert c['execution']['evaluation_tiles_per_batch'] == 8


def test_selected_subset_keeps_native_gaps_and_original_footprint(spatial_config):
    from dann.sampling import SpatialBatchRequest
    from dann.tiles import tile_collate
    bundle = small_bundle(spatial_config)
    tiles = bundle.datasets['train']
    tile_id = next(i for i, (_, selected) in enumerate(tiles.tiles) if len(selected) > 1)
    full = tile_collate([tiles[tile_id]])
    # Deliberately reverse two rows and exclude the other labels in this tile.
    request = SpatialBatchRequest(tuple(reversed(full['row_ids'][:2].tolist())), (tile_id,))
    actual = SpatialBatchDataset(tiles)[request]
    assert actual['row_ids'].tolist() == list(request.row_ids)
    from dann.sampling import SUPERVISION_FIELDS
    for key in full:
        if key not in SUPERVISION_FIELDS:
            torch.testing.assert_close(actual[key], full[key], atol=0, rtol=0)
    slide, tx, ty = tiles.tiles[tile_id][0]
    x0, y0 = tx * tiles.core_size - tiles.halo, ty * tiles.core_size - tiles.halo
    expected = {(y - y0) * tiles.size + x - x0 for (s, x, y) in tiles.keys
                if s == slide and x0 <= x < x0 + tiles.size and y0 <= y < y0 + tiles.size}
    assert set(actual['spatial_positions'].tolist()) == expected
    assert int(actual['occupancy'].sum()) == len(expected) < tiles.size ** 2
    for d in bundle.datasets.values():
        d.close()

"""Logical optimizer batches preserve the objective, context, and resume position."""
import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
import yaml

from dann.config import resolve_execution
from dann.data_loader import create_data_bundle
from dann.execution import tree_map
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.tests.test_components import spatial_config
from dann.tests.test_he import he_config
from dann.train import TrainingProgress, run_epoch, checkpoint_payload


class DoubleLoader:
    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        return getattr(self.source, name)

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        for batch in self.source:
            yield tree_map(batch, lambda t: t.double() if t.is_floating_point() else t)


def settings(config, micro, prepared=False):
    c = copy.deepcopy(config)
    for group in [c['model']['aggregation'], *c['model']['heads'].values()]:
        group['type'] = 'gatv2'
        group['gatv2'].update(depth=1, dropout=0., attention_dropout=0., neighbor_radius=1.5)
    c['model'].update(spectral_checkpointing=False)
    c['training']['spatial'].update(supervised_rows_per_batch=13, microbatch_rows=micro)
    if prepared:
        c['performance'].update(ram_cache=True, cache_geometry=True, preparation_workers=2,
                                prepared_batches=2, validation_tiles=2)
    return resolve_execution(c)


def execute(model, bundle, config, optimizer, state, **kwargs):
    return run_epoch(model, DoubleLoader(bundle.loaders['train']), ZILNLoss.from_config(config).double(),
                     torch.device('cpu'), 5., 1., torch.tensor([.3, 1.7], dtype=torch.float64),
                     optimizer, state.epoch, 3, 'dann', .25, 10., .5,
                     target_columns=config['data']['target_columns'], he_weight=.7,
                     progress_state=state, grl_horizon_updates=500, **kwargs)


@pytest.mark.parametrize('reduction,prepared', [('mean', False), ('mean', True), ('sum', False)])
def test_monolithic_gradients_adamw_clipping_and_resume(he_config, monkeypatch, reduction, prepared, tmp_path):
    torch.manual_seed(17)
    configs = [settings(he_config, micro, prepared) for micro in (None, 5)]
    for c in configs:
        c['loss']['ziln_reduction'] = reduction
    bundles = [create_data_bundle(c) for c in configs]
    # One whole physical microbatch has no valid biology targets.
    masked = next(bundles[0].loaders['train'].sampler.logical_requests()).row_ids[:5]
    if not prepared:  # Prepared targets are frozen copies, never mutated during training.
        for bundle in bundles:
            bundle.metadata.target_valid_mask[list(masked)] = False
    models = [AdversarialLatentFusion.from_config(configs[0], 2, 4).double()]
    models.append(copy.deepcopy(models[0]))
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.0001) for m in models]
    states = [TrainingProgress(), TrainingProgress()]
    clips = []
    original = torch.nn.utils.clip_grad_norm_

    def capture(parameters, *args, **kwargs):
        params = list(parameters)
        clips.append([p.grad.clone() for p in params])
        return original(params, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', capture)
    strengths = []
    models[1].register_forward_pre_hook(lambda m, args, kw: strengths.append(kw['grl_strength']), with_kwargs=True)
    expected_updates = bundles[0].loaders['train'].sampler.logical_updates
    result = execute(models[0], bundles[0], configs[0], optimizers[0], states[0])
    reference_clips = clips[:]
    clips.clear()
    records = []
    execute(models[1], bundles[1], configs[1], optimizers[1], states[1], max_updates=2,
            update_callback=records.append)
    saved = checkpoint_payload({}, dict(model_state=models[1].state_dict(),
                                        optimizer_state=optimizers[1].state_dict()),
                               -1, 0., progress_state=states[1])
    torch.save(saved, tmp_path/'partial.pt')
    saved = torch.load(tmp_path/'partial.pt', weights_only=False)
    models[1] = AdversarialLatentFusion.from_config(configs[1], 2, 4).double()
    models[1].load_state_dict(saved['model_state'])
    models[1].register_forward_pre_hook(lambda m, args, kw: strengths.append(kw['grl_strength']), with_kwargs=True)
    optimizers[1] = torch.optim.AdamW(models[1].parameters())
    optimizers[1].load_state_dict(saved['optimizer_state'])
    states[1] = TrainingProgress.from_checkpoint(saved, expected_updates, len(bundles[1].split_indices['train']))
    assert states[1].optimizer_updates == states[1].update_in_epoch == 2
    execute(models[1], bundles[1], configs[1], optimizers[1], states[1], update_callback=records.append)
    assert len(clips) == len(reference_clips) == expected_updates
    assert asdict(states[0]) == asdict(states[1])
    assert states[1].epoch == 1 and states[1].update_in_epoch == 0
    assert states[1].supervised_rows == len(bundles[1].split_indices['train'])
    assert result['optimizer_updates'] == expected_updates
    for left, right in zip(reference_clips, clips):
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-6)
    for a, b in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-6)
    for optimizer in optimizers:
        assert all(s['step'] == expected_updates for s in optimizer.state.values())
    offset = 0
    for record in records:
        count = (record['supervised_rows'] + 4) // 5
        assert strengths[offset:offset+count] == [record['grl_strength']] * count
        offset += count
    assert offset == len(strengths)
    for b in bundles:
        for d in b.datasets.values():
            d.close()


def test_requests_coverage_context_validation_and_dropout(he_config):
    configs = [settings(he_config, micro, prepared=True) for micro in (None, 5)]
    bundles = [create_data_bundle(c) for c in configs]
    full, micro = [b.loaders['train'] for b in bundles]
    requests = list(micro.sampler)
    assert sorted(row for r in requests for row in r.row_ids) == sorted(bundles[1].split_indices['train'])
    assert len(requests[-1].row_ids) < 5  # unequal final microbatch
    for request in requests:
        actual = micro.dataset[request]
        for tile in request.tile_ids:
            original = micro.dataset.tiles[tile]
            assert set(original['context_row_ids']).issubset(set(actual['context_row_ids'].tolist()))
    model = AdversarialLatentFusion.from_config(configs[0], 2, 4).eval()
    with torch.no_grad():
        for a, b in zip(bundles[0].loaders['validation'], bundles[1].loaders['validation']):
            for key, value in model(a).items():
                torch.testing.assert_close(value, model(b)[key], atol=0, rtol=0)
    c = configs[1]
    for group in [c['model']['aggregation'], *c['model']['heads'].values()]:
        group['gatv2']['attention_dropout'] = .1
    model = AdversarialLatentFusion.from_config(c, 2, 4).double()
    metrics = execute(model, bundles[1], c, torch.optim.AdamW(model.parameters()), TrainingProgress(),
                      max_updates=1)
    assert np.isfinite(metrics['total_loss'])
    for b in bundles:
        for d in b.datasets.values():
            d.close()


@pytest.mark.parametrize('value', [0, -1, True, 2.5])
def test_invalid_microbatch_rows(spatial_config, value):
    spatial_config['training']['spatial']['microbatch_rows'] = value
    with pytest.raises(ValueError, match='microbatch_rows'):
        resolve_execution(spatial_config)


def test_pilot_contract_and_diagnostic_statistics(he_config, tmp_path):
    from dann.verification.core_batch_pilot import Diagnostics, pilot_config
    a, b = [pilot_config(he_config, name, tmp_path) for name in ('reference', 'candidate')]
    assert a['training']['batch_size'] == 2048 and b['training']['batch_size'] == 32768
    assert b['training']['microbatch_rows'] == 4096
    for config in (a, b):
        assert config['training']['learning_rate'] == .0001
        assert config['training']['grl_horizon_updates'] == 500
        assert config['performance']['validation_tiles'] == 8
        assert not config['post_training']['enabled']
    torch.manual_seed(1)
    first = AdversarialLatentFusion.from_config(a, 2, 4)
    torch.manual_seed(1)
    second = AdversarialLatentFusion.from_config(b, 2, 4)
    for p, q in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    diagnostic = Diagnostics(0)
    batch = dict(targets=torch.tensor([[0.], [.5], [0.], [.5]]),
                 target_valid_mask=torch.ones(4, 1, dtype=torch.bool),
                 row_ids=torch.tensor([0, 23, 46, 69]), batches=torch.tensor([0, 0, 1, 1]))
    output = dict(pi_logits=torch.zeros(4, 1), mu=torch.zeros(4, 1), sigma=torch.ones(4, 1),
                  latent=torch.tensor([[0., 0.], [2., 0.], [0., 2.], [2., 2.]]),
                  batch_logits=torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]))
    diagnostic(batch, output)
    result = diagnostic.finish()
    assert result['cd8']['presence_brier'] == .25
    assert result['cd8']['positive_pit_mean'] == .5
    assert result['latent']['between_slide_fraction'] == pytest.approx(.5)
    assert result['latent']['sampled_effective_rank'] == pytest.approx(2.)
    assert result['discriminator_balanced_accuracy'] == 1.


def test_pilot_replay_checks_consumed_rows_and_partial_epochs(he_config, tmp_path):
    from dann.sampling import SamplingEpochAudit
    from dann.verification.core_batch_pilot import audit_sampling
    config = settings(he_config, None)
    bundle = create_data_bundle(config)
    root = tmp_path/'audit'
    directory = root/'reference'
    directory.mkdir(parents=True)
    (root/'manifest.json').write_text(json.dumps(dict(arm_directories={'reference': str(directory)})))
    (directory/'resolved_config.yaml').write_text(yaml.safe_dump(config))
    np.savez(root/'splits.npz', **bundle.split_indices)
    records, rows = [], 0
    for epoch in range(2):
        bundle.loaders['train'].sampler.set_epoch(epoch)
        audit = SamplingEpochAudit(epoch, len(bundle.split_indices['train']))
        for batch in bundle.loaders['train']:
            record = audit.update(batch)
            rows += record['supervised_rows']
            record.update(optimizer_updates=len(records)+1, cumulative_supervised_rows=rows)
            records.append(record)
            if epoch == 1:
                break
    path = directory/'updates.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    result = audit_sampling(root)['reference']
    assert result['passed'] and result['supervised_rows'] == rows
    assert [e['complete'] for e in result['epochs']] == [True, False]
    records[-1]['row_ids_sha256'] = 'incorrect'
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError, match='consumed request differs'):
        audit_sampling(root)
    for dataset in bundle.datasets.values():
        dataset.close()

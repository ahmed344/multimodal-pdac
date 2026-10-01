"""Matched, update-capped discriminator controls; never modifies production runs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dann.config import load_config, resolve_device, seed_everything
from dann.data_loader import create_data_bundle, sparse_collate
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.model_components.discriminator_cnn import CNNDiscriminator
from dann.targets import checkpoint_contract, target_contract
from dann.tiles import SpatialTileDataset, tile_collate
from dann.train import grl_strength, move_batch_to_device
from dann.verification.epoch_comparison import extra_metrics, file_hash
from dann.verification.fit_diagnosis import (
    constant_parameters, metrics, optimizer_for, save_json, stage_audit,
)


def make_control(reference: AdversarialLatentFusion, config: dict,
                 cnn: bool, spatial: bool = True) -> AdversarialLatentFusion:
    """Copy shared weights exactly and construct the discriminator in an isolated RNG."""
    model = copy.deepcopy(reference)
    model.spatial = spatial
    if cnn:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['training']['seed'] + 10000)
            model.batch_discriminator = CNNDiscriminator(
                model.latent_dim, reference.batch_discriminator.network[-1].out_features,
                **config['model']['heads']['discriminator']['cnn'])
    model.halo = config['model']['heads']['discriminator']['cnn']['depth'] if spatial else 0
    return model


def objectives(model: AdversarialLatentFusion, batch: dict, loss: ZILNLoss,
               biology_weight: float, batch_weight: float,
               strength: float, class_weights: torch.Tensor | None) -> tuple:
    """Return loss terms; a disabled discriminator has no backward/AdamW path."""
    out = model(batch, grl_strength=strength)
    bio = biology_weight * loss(out['pi_logits'], out['mu'], out['sigma'],
                               batch['targets'], batch['target_valid_mask']).total
    disc = batch_weight * nn.functional.cross_entropy(
        out['batch_logits'], batch['batches'], weight=class_weights) if batch_weight else None
    return bio, disc


def _norm(values: list[torch.Tensor | None]) -> float:
    return math.sqrt(sum(float(x.detach().double().square().sum()) for x in values if x is not None))


def gradient_audit(model: AdversarialLatentFusion, batch: dict, loss: ZILNLoss,
                   config: dict, batch_weight: float, strength: float,
                   class_weights: torch.Tensor | None) -> dict[str, Any]:
    """Measure separate training gradients without touching .grad or training RNG."""
    was_training = model.training
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config['training']['seed'] + 20000)
            model.train()
            named = list(model.named_parameters())
            params = [p for _, p in named]
            bio, disc = objectives(model, batch, loss, config['loss']['biology_weight'],
                                   batch_weight, strength, class_weights)
            bg = torch.autograd.grad(bio, params, retain_graph=disc is not None, allow_unused=True)
            dg = (torch.autograd.grad(disc, params, allow_unused=True) if disc is not None
                  else (None,) * len(params))
            combined = [b if d is None else d if b is None else b+d for b, d in zip(bg, dg)]
            result = {}
            for name, prefix in [('encoder', 'encoder.'), ('biology', 'biology_predictor.'),
                                 ('discriminator', 'batch_discriminator.')]:
                indices = [i for i, (key, _) in enumerate(named) if key.startswith(prefix)]
                bn, dn = _norm([bg[i] for i in indices]), _norm([dg[i] for i in indices])
                dot = sum(float((bg[i].double()*dg[i].double()).sum()) for i in indices
                          if bg[i] is not None and dg[i] is not None)
                result[name] = dict(biology_l2=bn, discriminator_l2=dn,
                    combined_l2=_norm([combined[i] for i in indices]),
                    cosine=dot/(bn*dn) if bn*dn else None)
            norm, bio_norm = _norm(combined), _norm(list(bg))
            clip = config['training']['gradient_clip_norm']
            result.update(total_norm=norm, biology_only_norm=bio_norm,
                clip_factor=min(1., clip/(norm+1e-6)) if clip is not None else 1.,
                biology_only_clip_factor=min(1., clip/(bio_norm+1e-6)) if clip is not None else 1.,
                strength=strength, batch_weight=batch_weight)
            return result
    finally:
        model.train(was_training)


@torch.no_grad()
def evaluate(model: AdversarialLatentFusion, loader: DataLoader, metadata: Any,
             loss: ZILNLoss, device: torch.device, expected_rows: np.ndarray) -> tuple[dict, dict]:
    """Score all validation biology pointwise; both shared components are MLPs."""
    model.eval()
    predictions, identities = [], []
    for cpu in loader:
        batch = move_batch_to_device(cpu, device)
        latent = model.encoder(*(batch[k] for k in
            ('peak_indices', 'intensities', 'sample_indices', 'peak_counts')))
        out = model.biology_predictor(latent)
        predictions.append({k: out[k].cpu() for k in ('pi_logits', 'mu', 'sigma')})
        identities.append(cpu['row_ids'])
    ids = torch.cat(identities).numpy()
    np.testing.assert_array_equal(ids, expected_rows)
    out = {k: torch.cat([p[k] for p in predictions]) for k in predictions[0]}
    batch = dict(targets=torch.from_numpy(metadata.targets[ids]),
                 target_valid_mask=torch.from_numpy(metadata.target_valid_mask[ids]))
    cpu_loss = copy.deepcopy(loss).cpu()
    result = metrics(out, batch, cpu_loss)
    result.update(extra_metrics(out, batch['targets'], batch['target_valid_mask'], loss.logit_epsilon))
    return result, dict(row_ids=ids, **out)


def run_control(name: str, reference: AdversarialLatentFusion, config: dict, bundle: Any,
                tiles: SpatialTileDataset, root: Path, cnn: bool, batch_weight: float,
                ramp: bool, lr: float, updates: int, evaluate_every: int,
                spatial: bool = True) -> dict:
    """Fit one control with fixed sampler/RNG seeds and an unchanged GRL horizon."""
    directory = root/name
    directory.mkdir()
    t, seed = config['training'], config['training']['seed']
    device = resolve_device(t['device'])
    seed_everything(seed, t['deterministic_algorithms'])
    model = make_control(reference, config, cnn, spatial).to(device)
    optimizer = optimizer_for(model, config, lr)
    loss = ZILNLoss.from_config(config).to(device)
    mode = t['spatial' if spatial else 'pixel']
    workers = mode['num_workers']
    dataset = tiles if spatial else bundle.datasets['train']
    collate = tile_collate if spatial else sparse_collate
    size = mode['tiles_per_batch'] if spatial else mode['batch_size']
    kwargs = dict(num_workers=workers, pin_memory=t['pin_memory'], persistent_workers=workers > 0)
    if workers:
        kwargs['prefetch_factor'] = mode['prefetch_factor']
    loader = DataLoader(dataset, batch_size=size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed), **kwargs)
    # Evaluation uses its own generator, so iterator creation cannot advance dropout RNG.
    validation_loader = DataLoader(bundle.datasets['validation'], batch_size=t['pixel']['validation_batch_size'],
        shuffle=False, collate_fn=sparse_collate, generator=torch.Generator().manual_seed(seed+1), **kwargs)
    weights = bundle.batch_class_weights.to(device) if config['loss']['balance_batch_classes'] else None
    audit_cpu = collate([dataset[i] for i in range(min(size, len(dataset)))])
    dataset.close()
    torch.save(audit_cpu, directory/'audit_batch.pt')
    audit_batch = move_batch_to_device(audit_cpu, device)
    settings = dict(config=config, cnn_discriminator=cnn, spatial=spatial, learning_rate=lr,
        batch_weight=batch_weight, grl='configured_ramp' if ramp else 'zero', max_updates=updates,
        evaluate_every=evaluate_every, steps_per_epoch=len(loader), schedule_epochs=t['epochs'],
        seed=seed, dropout_seed='seed + update_index + 1', audit_split='train',
        validation_execution='equivalent_pointwise_biology', test_evaluation=False,
        train_rows=len(bundle.split_indices['train']), validation_rows=len(bundle.split_indices['validation']))
    save_json(directory/'settings.json', settings)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(initial, directory/'initial_state.pt')
    start = time.monotonic()
    history, training = [], []
    order_hash = hashlib.sha256()
    rows_seen = 0
    iterator = None
    for update in range(updates+1):
        strength = grl_strength(max(update-1, 0)/max(t['epochs']*len(loader)-1, 1),
            t['grl_schedule'], t['grl_max_lambda'], t['grl_gamma']) if ramp else 0.
        if update:
            if iterator is None:
                iterator = iter(loader)
            try:
                cpu = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                cpu = next(iterator)
            order_hash.update(cpu['row_ids'].numpy().tobytes())
            rows_seen += len(cpu['row_ids'])
            batch = move_batch_to_device(cpu, device)
            # Isolate stochastic layers in one control from the next update in another.
            torch.manual_seed(seed+update)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            bio, disc = objectives(model, batch, loss, config['loss']['biology_weight'],
                                   batch_weight, strength, weights)
            total = bio if disc is None else bio+disc
            if not torch.isfinite(total):
                raise FloatingPointError(f'{name}: nonfinite loss at update {update}')
            total.backward()
            clip = t['gradient_clip_norm']
            norm = float(nn.utils.clip_grad_norm_(model.parameters(), clip if clip is not None else float('inf'),
                                                  error_if_nonfinite=True))
            optimizer.step()
            training.append(dict(update=update, rows=len(cpu['row_ids']), slides=int(cpu['batches'].unique().numel()),
                biology=float(bio.detach())/config['loss']['biology_weight'],
                discriminator=float(disc.detach()) if disc is not None else None,
                gradient_norm=norm, clip_factor=min(1., clip/(norm+1e-6)) if clip is not None else 1.,
                grl_strength=strength, seconds=time.monotonic()-start))
            del batch, bio, disc, total
            if update % 25 == 0:
                print(json.dumps(dict(control=name, update=update, rows_seen=rows_seen,
                                      seconds=time.monotonic()-start)), flush=True)
        if update % evaluate_every == 0 or update == updates:
            optimizer.zero_grad(set_to_none=True)
            devices = [device.index or 0] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                scored, predictions = evaluate(model, validation_loader, bundle.metadata, loss,
                                               device, bundle.split_indices['validation'])
                audit = stage_audit(model, audit_batch, loss, None, seed)
                audit['gradients'] = gradient_audit(model, audit_batch, loss, config, batch_weight, strength, weights)
            row = dict(update=update, rows_seen=rows_seen, order_sha256=order_hash.hexdigest(),
                       validation=scored, audit=audit, seconds=time.monotonic()-start)
            history.append(row)
            save_json(directory/'history.json', history)
            save_json(directory/'training.json', training)
            torch.save(predictions, directory/f'validation_{update}.pt')
            print(json.dumps(dict(control=name, update=update, validation=scored, seconds=row['seconds'])), flush=True)
    torch.save(dict(format='diagnostic-discriminator-v1', model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(), settings=settings, updates=updates,
        split_indices=bundle.split_indices), directory/'final.pt')
    save_json(directory/'completion.json', dict(updates=updates, rows_seen=rows_seen,
                                               seconds=time.monotonic()-start))
    dataset.close()
    bundle.datasets['validation'].close()
    del iterator, loader, validation_loader, model, optimizer, audit_batch
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return history[-1]


def main(argv: list[str] | None = None) -> None:
    """Run four primary controls and the prespecified conditional follow-ups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--reference-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--evaluate-every', type=int, default=100)
    args = parser.parse_args(argv)
    if min(args.updates, args.evaluate_every) <= 0:
        parser.error('Update limits and evaluation intervals must be positive.')
    config = load_config(args.config)
    if config['execution']['mode'] != 'pixel':
        parser.error('The reference configuration must be all MLP.')
    if any(config['data'].get('max_'+s+'_samples') is not None for s in ('train', 'validation', 'test')):
        parser.error('Controls require uncapped saved splits.')
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    # The user's concurrent training may replace best.pt. Freeze one complete version.
    snapshot = root/'reference.pt'
    for _ in range(3):
        before = file_hash(args.reference_checkpoint)
        shutil.copyfile(args.reference_checkpoint, snapshot)
        if before == file_hash(snapshot) == file_hash(args.reference_checkpoint):
            break
    else:
        raise RuntimeError('Could not snapshot a stable reference checkpoint.')
    (root/'source_config.yaml').write_bytes(args.config.read_bytes())
    for source in Path('dann').rglob('*.py'):
        destination = root/'source'/source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    payload = torch.load(snapshot, map_location='cpu', weights_only=False)
    if target_contract(config) != checkpoint_contract(payload):
        raise ValueError('Reference target contract differs.')
    bundle = create_data_bundle(config, payload['split_indices'], payload['data_identity'])
    if tuple(payload['batch_names']) != bundle.metadata.batch_names:
        raise ValueError('Reference class ordering differs.')
    np.savez(root/'splits.npz', **bundle.split_indices)
    seed_everything(config['training']['seed'], config['training']['deterministic_algorithms'])
    reference = AdversarialLatentFusion.from_config(config, len(bundle.metadata.batch_names),
                                                   len(config['data']['target_columns']))
    geometry = copy.deepcopy(config)
    geometry['training']['core_size'] = config['training']['spatial']['core_size']
    geometry['model']['heads']['discriminator']['type'] = 'cnn'
    tiles = SpatialTileDataset(geometry, bundle.split_indices['train'], bundle.metadata)
    train, val = (bundle.split_indices[s] for s in ('train', 'validation'))
    m, loss = bundle.metadata, ZILNLoss.from_config(config)
    constant = constant_parameters(torch.from_numpy(m.targets[train]), torch.from_numpy(m.target_valid_mask[train]),
                                   loss.logit_epsilon, config['model']['sigma_min'])
    out = {k: v[:1].expand(len(val), -1) for k, v in constant.items()}
    batch = dict(targets=torch.from_numpy(m.targets[val]), target_valid_mask=torch.from_numpy(m.target_valid_mask[val]))
    baseline = metrics(out, batch, loss)
    baseline.update(extra_metrics(out, batch['targets'], batch['target_valid_mask'], loss.logit_epsilon))
    save_json(root/'constant_baseline.json', dict(validation=baseline, fit_split='train',
        parameters={k: v[0].tolist() for k, v in constant.items()}))
    del constant, out, payload
    save_json(root/'manifest.json', dict(reference_source=str(args.reference_checkpoint), reference_sha256=before,
        config=config, torch_version=torch.__version__, updates=args.updates,
        evaluate_every=args.evaluate_every, halo=tiles.halo,
        note='Reference weights are not loaded; concurrent source training is left untouched.'))
    results = {}
    for name, cnn, weight, ramp in [('A_spatial_mlp', False, 0., False),
        ('B_cnn_disc_disabled', True, 0., False), ('C_cnn_disc_zero_grl', True, config['loss']['batch_weight'], False),
        ('D_cnn_disc_ramp', True, config['loss']['batch_weight'], True)]:
        results[name] = run_control(name, reference, config, bundle, tiles, root, cnn, weight, ramp,
                                   .001, args.updates, args.evaluate_every)
        save_json(root/'results.json', results)
    followup = all(results[name]['validation']['positive_cd8_r2'] is not None and
                   results[name]['validation']['positive_cd8_r2'] <= 0
                   for name in ('A_spatial_mlp', 'B_cnn_disc_disabled'))
    if followup:
        for name, cnn, lr, spatial in [('A_low_lr', False, .0001, True),
                                      ('B_low_lr', True, .0001, True), ('E_pixel_mlp', False, .001, False)]:
            results[name] = run_control(name, reference, config, bundle, tiles, root, cnn, 0., False,
                                       lr, args.updates, args.evaluate_every, spatial)
            save_json(root/'results.json', results)
    assert file_hash(snapshot) == before
    for dataset in bundle.datasets.values():
        dataset.close()
    tiles.close()
    save_json(root/'completion.json', dict(controls=list(results), conditional_followups=followup,
        reference_snapshot_unchanged=True, updates_per_control=args.updates))


if __name__ == '__main__':
    main()

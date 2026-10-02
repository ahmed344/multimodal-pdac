"""Frozen, deadline-bounded batch/execution experiments. Never changes production fits.

Stages are explicitly selected after inspecting the preceding evidence. All runs use
persisted complete-pass plans; interrupted attempts are immutable and never resumed
as if complete. --resume skips only hash-verified, contract-matching completions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dann.config import load_config, resolve_device, seed_everything
from dann.data_loader import create_data_bundle, sparse_collate
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.model_components.deep_sets import DeepSetsEncoder
from dann.targets import checkpoint_contract, target_contract
from dann.tiles import SpatialTileDataset, tile_collate
from dann.train import grl_strength, move_batch_to_device
from dann.verification.epoch_comparison import extra_metrics, file_hash, flattened_tile_collate
from dann.verification.fit_diagnosis import (
    altered_batches, groups, metrics, optimizer_for, save_json, variation,
)

PREVIOUS = Path('data/PDAC/Results/dann/diagnostics/2026-09-30-discriminator-controls')
REFERENCE_HASH = '375e2d64c6f96b689053fa1babdd8654e0b4de0711698941659536b5596d7d26'
SEEDS = [20260719, 20260720, 20260721]
SPECTRAL_KEYS = ('peak_indices', 'intensities', 'sample_indices', 'peak_counts')
LABEL_KEYS = ('row_ids', 'targets', 'raw_targets', 'target_valid_mask', 'batches', 'support', 'latent_support')
MODES = ('X0', 'X1', 'X2', 'X3', 'X4')


def digest(value: Any) -> str:
    """Hash a JSON-compatible scientific contract deterministically."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass
class Plan:
    """Complete-pass row stream, optimizer boundaries, and native tile stream."""
    rows: np.ndarray
    boundaries: np.ndarray
    passes: np.ndarray
    tile_order: np.ndarray
    tile_boundaries: np.ndarray

    @property
    def sha256(self) -> str:
        h = hashlib.sha256()
        for value in (self.rows, self.boundaries, self.passes, self.tile_order, self.tile_boundaries):
            array = np.asarray(value, dtype='<i8')
            h.update(np.asarray(array.shape, dtype='<i8').tobytes())
            h.update(array.tobytes())
        return h.hexdigest()

    def save(self, path: Path) -> None:
        """Save without replacing earlier evidence."""
        if path.exists():
            if self.load(path).sha256 != self.sha256:
                raise ValueError(f'Existing plan differs: {path}')
            return
        np.savez_compressed(path, **vars(self))

    @classmethod
    def load(cls, path: Path) -> Plan:
        with np.load(path) as data:
            return cls(**{k: data[k] for k in data.files})

    def batch(self, index: int) -> np.ndarray:
        return self.rows[self.boundaries[index]:self.boundaries[index+1]]


def permutations(n: int, seed: int, passes: int = 2):
    """Match the archived persistent DataLoader/RandomSampler RNG consumption."""
    g = torch.Generator().manual_seed(seed)
    torch.empty((), dtype=torch.int64).random_(generator=g)  # iterator base seed
    for _ in range(passes):
        yield torch.randperm(n, generator=g).numpy()
        torch.randperm(n, generator=g)  # RandomSampler's exhausted remainder


def make_plans(tiles: SpatialTileDataset, train: np.ndarray, codes: np.ndarray,
               seed: int, passes: int = 2, small: int = 2048, tiles_per_batch: int = 8) -> dict[str, Plan]:
    """Construct all composition schedules without labels or spectral reads."""
    names = ('F-native', 'F-small', 'F-dispersed', 'G-native', 'G-small')
    records = {name: dict(rows=[], boundaries=[0], passes=[0], tile_order=[], tile_boundaries=[0]) for name in names}
    global_orders = list(permutations(len(train), seed, passes))
    for pass_index, order in enumerate(permutations(len(tiles), seed, passes)):
        native_batches = [order[i:i+tiles_per_batch] for i in range(0, len(order), tiles_per_batch)]
        chunks = [np.asarray([r for t in batch for r, _ in tiles.tiles[t][1]], dtype=np.int64) for batch in native_batches]
        native = np.concatenate(chunks)
        sizes = [len(x) for x in chunks]
        dispersed = native.copy()
        rng = np.random.default_rng(seed + 30000 + pass_index)
        for slide in np.unique(codes[train]):
            positions = np.flatnonzero(codes[native] == slide)
            dispersed[positions] = rng.permutation(train[codes[train] == slide])
        global_rows = train[global_orders[pass_index]]
        for name in names:
            rec = records[name]
            rows = global_rows if name.startswith('G') else dispersed if name == 'F-dispersed' else native
            lengths = ([min(small, len(rows)-i) for i in range(0, len(rows), small)]
                       if name.endswith('small') else sizes)
            rec['rows'].append(rows)
            rec['boundaries'].extend((rec['boundaries'][-1] + np.cumsum(lengths)).tolist())
            rec['passes'].append(len(rec['boundaries'])-1)
            rec['tile_order'].extend(order.tolist())
            rec['tile_boundaries'].extend((rec['tile_boundaries'][-1] + np.cumsum([len(b) for b in native_batches])).tolist())
    result = {name: Plan(**{k: np.concatenate(v) if k == 'rows' else np.asarray(v, dtype=np.int64)
                            for k, v in rec.items()}) for name, rec in records.items()}
    for plan in result.values():
        verify_plan(plan, train)
    return result


def round_plan(tiles: SpatialTileDataset, native: Plan, train: np.ndarray, seed: int,
               small: int | None = None) -> Plan:
    """Shuffle per-slide 8x8 queues; take one tile per active slide per random round.

    Row repartitioning carries partial-tile supervision forward without dropping
    context. Exhausted slides disappear, so small slides are never oversampled.
    """
    rows, boundaries, passes, tile_order, tile_boundaries = [], [0], [0], [], [0]
    row_tile = {r: i for i, (_, selected) in enumerate(tiles.tiles) for r, _ in selected}
    for pass_index, (start, stop) in enumerate(zip(native.passes[:-1], native.passes[1:])):
        rng = np.random.default_rng(seed + 40000 + pass_index)
        queues = {}
        for i, (key, _) in enumerate(tiles.tiles):
            queues.setdefault(key[0], []).append(i)
        queues = {slide: list(rng.permutation(queue)) for slide, queue in queues.items()}
        stream = []
        while queues:
            for slide in rng.permutation(sorted(queues)):
                tile = queues[slide].pop()
                stream.extend(r for r, _ in tiles.tiles[tile][1])
                if not queues[slide]:
                    del queues[slide]
        stream = np.asarray(stream, dtype=np.int64)
        sizes = (np.diff(native.boundaries[start:stop+1]) if small is None else
                 np.array([min(small, len(stream)-i) for i in range(0, len(stream), small)]))
        cursor = 0
        for size in sizes:
            selected = stream[cursor:cursor+size]
            ordered_tiles = list(dict.fromkeys(row_tile[int(r)] for r in selected))
            tile_order.extend(ordered_tiles)
            tile_boundaries.append(len(tile_order))
            cursor += size
            boundaries.append(boundaries[-1]+int(size))
        assert cursor == len(stream)
        rows.append(stream); passes.append(len(boundaries)-1)
    plan = Plan(np.concatenate(rows), np.array(boundaries), np.array(passes),
                np.array(tile_order), np.array(tile_boundaries))
    verify_plan(plan, train)
    return plan


def verify_plan(plan: Plan, train: np.ndarray) -> None:
    """Reject missing, duplicated, out-of-split, or empty optimizer batches."""
    assert plan.boundaries[0] == 0 and plan.boundaries[-1] == len(plan.rows)
    assert np.all(np.diff(plan.boundaries) > 0)
    for start, end in zip(plan.passes[:-1], plan.passes[1:]):
        rows = plan.rows[plan.boundaries[start]:plan.boundaries[end]]
        np.testing.assert_array_equal(np.sort(rows), np.sort(train))


def select_labels(batch: dict, indices: torch.Tensor) -> dict:
    """Select supervision only, preserving every occupied context spectrum."""
    result = dict(batch)
    for key in (*LABEL_KEYS, 'core_positions'):
        if key in result:
            result[key] = result[key][indices]
    return result


def core_flat(batch: dict) -> dict:
    """Gather complete core spectra in exact supervision order, including empties."""
    lookup = torch.full((batch['occupancy'].numel(),), -1, dtype=torch.long,
                        device=batch['peak_counts'].device)
    lookup[batch['spatial_positions']] = torch.arange(len(batch['peak_counts']), device=lookup.device)
    core = lookup[batch['core_positions']]
    if (core < 0).any():
        raise ValueError('Supervised core missing from occupied context.')
    counts = batch['peak_counts']
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    take = torch.cat([torch.arange(offsets[i], offsets[i+1], device=core.device) for i in core])
    selected_counts = counts[core]
    return {**{k: batch[k] for k in LABEL_KEYS if k in batch},
            'peak_indices': batch['peak_indices'][take], 'intensities': batch['intensities'][take],
            'peak_counts': selected_counts,
            'sample_indices': torch.repeat_interleave(torch.arange(len(core), device=core.device), selected_counts)}


class ExecutionModel(AdversarialLatentFusion):
    """Diagnostic execution ladder with unchanged production parameter names."""
    mode: str = 'X0'

    def forward(self, batch, grl_strength=0.):
        if self.mode in ('X0', 'X4'):
            return super().forward(batch, grl_strength)
        args = [batch[k] for k in SPECTRAL_KEYS]
        pooled = self.encoder.microbatched(*args, peak_budget=self.peak_budget,
                                          checkpointing=self.mode != 'X1')
        if self.mode == 'X3':
            lookup = torch.full((batch['occupancy'].numel(),), -1, device=pooled.device, dtype=torch.long)
            lookup[batch['spatial_positions']] = torch.arange(len(pooled), device=pooled.device)
            pooled = pooled[lookup[batch['core_positions']]]
        latent = self.encoder.aggregation_mlp(pooled)
        return dict(latent=latent, **self.biology_predictor(latent),
                    batch_logits=self.batch_discriminator(self.gradient_reversal(latent, grl_strength)))


def adapt(model: AdversarialLatentFusion, mode: str, dropout: str = 'unchanged') -> ExecutionModel:
    """Copy parameters and change only an explicitly named execution/dropout factor."""
    if mode not in MODES or dropout not in ('unchanged', 'off', 'spectral-off', 'aggregation-off'):
        raise ValueError('Unsupported execution or dropout setting.')
    if mode != 'X4' and any(c.radius for c in (model.encoder.aggregation_mlp, model.biology_predictor,
                                               model.batch_discriminator)):
        raise ValueError('CNN components require X4 and full spatial validation.')
    result = copy.deepcopy(model)
    result.__class__ = ExecutionModel
    result.mode, result.spatial = mode, mode == 'X4'
    module = {'off': result, 'spectral-off': result.encoder.peak_mlp,
              'aggregation-off': result.encoder.aggregation_mlp}.get(dropout)
    if module is not None:
        for layer in module.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = 0.
    return result


def objective(model: AdversarialLatentFusion, batch: dict, loss: ZILNLoss, config: dict,
              strength: float = 0., adversarial: bool = False,
              weights: torch.Tensor | None = None) -> tuple:
    """Compute biology and an optional genuinely active discriminator objective."""
    out = model(batch, grl_strength=strength)
    bio = loss(*(out[k] for k in ('pi_logits', 'mu', 'sigma')), batch['targets'], batch['target_valid_mask']).total
    total = config['loss']['biology_weight'] * bio
    if adversarial:
        total = total + config['loss']['batch_weight'] * nn.functional.cross_entropy(out['batch_logits'], batch['batches'], weight=weights)
    return out, bio, total


def equivalence(reference: AdversarialLatentFusion, flat: dict, tiled: dict, config: dict,
                modes: tuple[str, ...] = MODES, checkpoint_dropout: bool = False) -> dict:
    """Assert strict float64 outputs, gradients, RNG, and a clipped AdamW step."""
    models, outputs, gradients, states, rngs, shapes = [], [], [], [], [], []
    device = next(reference.parameters()).device
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        for mode in modes:
            model = adapt(reference, mode, 'unchanged' if checkpoint_dropout else 'off').double().train()
            batch = tiled if mode in ('X3', 'X4') else flat
            batch = {k: v.double() if v.is_floating_point() else v for k, v in batch.items()}
            loss = ZILNLoss.from_config(config).to(device).double()
            opt = optimizer_for(model, config, .001)
            calls = []
            handles = [module.register_forward_pre_hook(lambda m, a, name=name: calls.append((name, list(a[0].shape))))
                       for name, module in model.named_modules() if isinstance(module, nn.Dropout)]
            torch.manual_seed(801)
            out, bio, total = objective(model, batch, loss, config)
            total.backward()
            if not checkpoint_dropout:
                with torch.no_grad():
                    args = [batch[k] for k in SPECTRAL_KEYS]
                    pooled = (DeepSetsEncoder.forward(model.encoder, *args) if mode=='X0' else
                              model.encoder.microbatched(*args, peak_budget=model.peak_budget, checkpointing=False))
                    if mode in ('X3','X4'):
                        lookup = torch.full((batch['occupancy'].numel(),), -1, device=device, dtype=torch.long)
                        lookup[batch['spatial_positions']] = torch.arange(len(pooled), device=device)
                        pooled = pooled[lookup[batch['core_positions']]]
                    out['pooled'] = pooled
            for handle in handles:
                handle.remove()
            gradients.append({k: None if p.grad is None else p.grad.clone() for k, p in model.named_parameters()})
            rngs.append((torch.get_rng_state(), torch.cuda.get_rng_state(device) if devices else None))
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'])
            opt.step()
            outputs.append({**{k: v.detach() for k, v in out.items() if k != 'batch_logits'}, 'loss': bio.detach()})
            states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            shapes.append(calls)
            models.append(model)
        errors = dict(forward=0., gradient=0., update=0.)
        for index in range(1, len(models)):
            for key in outputs[0]:
                torch.testing.assert_close(outputs[0][key], outputs[index][key], atol=1e-9, rtol=1e-7)
                errors['forward'] = max(errors['forward'], float((outputs[0][key]-outputs[index][key]).abs().max()))
            for key, grad in gradients[0].items():
                other = gradients[index][key]
                assert (grad is None) == (other is None), key
                if grad is not None:
                    torch.testing.assert_close(grad, other, atol=1e-8, rtol=1e-6)
                    errors['gradient'] = max(errors['gradient'], float((grad-other).abs().max()))
            for key in states[0]:
                torch.testing.assert_close(states[0][key], states[index][key], atol=1e-7, rtol=1e-6)
                errors['update'] = max(errors['update'], float((states[0][key]-states[index][key]).abs().max()))
            if checkpoint_dropout:
                assert torch.equal(rngs[0][0], rngs[index][0])
                if devices:
                    assert torch.equal(rngs[0][1], rngs[index][1])
    return dict(passed=True, modes=list(modes), checkpoint_dropout=checkpoint_dropout, errors=errors,
                dropout_calls=dict(zip(modes, shapes)))


class PlannedDataset(Dataset):
    """Lazy whole-batch reads, allowing partial tile supervision with complete halos."""
    def __init__(self, tiles, plan, spatial):
        self.tiles, self.plan, self.spatial = tiles, plan, spatial
        self.row_tile = {r: i for i, (_, selected) in enumerate(tiles.tiles) for r, _ in selected}

    def __len__(self):
        return len(self.plan.boundaries)-1

    def __getitem__(self, index):
        rows = self.plan.batch(index)
        m = self.tiles.metadata
        if not self.spatial:
            return sparse_collate([dict(self.tiles.spectra[int(r)], row_id=int(r),
                targets=m.targets[r], raw_targets=m.raw_targets[r], target_valid_mask=m.target_valid_mask[r],
                batch=m.batch_codes[r]) for r in rows])
        tile_ids = list(dict.fromkeys(self.row_tile[int(r)] for r in rows))
        batch = tile_collate([self.tiles[t] for t in tile_ids])
        lookup = {int(r): i for i, r in enumerate(batch['row_ids'])}
        return select_labels(batch, torch.tensor([lookup[int(r)] for r in rows]))


def loader_for(dataset: Dataset, seed: int, workers: int = 4) -> DataLoader:
    """Read persisted batches with a loader RNG isolated from model randomness."""
    return DataLoader(dataset, batch_size=None, shuffle=False, num_workers=workers,
        persistent_workers=workers > 0, pin_memory=True,
        generator=torch.Generator().manual_seed(seed), **({'prefetch_factor': 2} if workers else {}))


def audit(model: ExecutionModel, batch: dict, loss: ZILNLoss, config: dict, seed: int) -> dict:
    """Fixed training-only sensitivity and component gradients; preserve RNG/.grad/mode."""
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == 'cuda' else []
    was_training = model.training
    result = {}
    try:
        with torch.random.fork_rng(devices=devices):
            for training in (False, True):
                model.train(training)
                # Keep checkpoint recomputation available for eval-mode gradients.
                # Set only this parent's flag: child Dropout modules stay in eval
                # mode, so the deterministic function and its gradients are unchanged.
                if not training:
                    model.encoder.training = True
                torch.manual_seed(seed + 20000)
                out, bio, total = objective(model, batch, loss, config)
                params = [p for p in model.parameters()]
                grads = torch.autograd.grad(total, params, allow_unused=True)
                by_id = {id(p): g for p, g in zip(params, grads)}
                norms = {name: math.sqrt(sum(float(by_id[id(p)].detach().double().square().sum())
                         for p in ps if by_id[id(p)] is not None)) for name, ps in groups(model).items()}
                with torch.no_grad():
                    pooled = model.encoder.microbatched(*(batch[k] for k in SPECTRAL_KEYS),
                                                        peak_budget=model.peak_budget, checkpointing=False)
                    if 'occupancy' in batch:
                        lookup = torch.full((batch['occupancy'].numel(),), -1, device=device, dtype=torch.long)
                        lookup[batch['spatial_positions']] = torch.arange(len(pooled), device=device)
                        pooled = pooled[lookup[batch['core_positions']]]
                    torch.manual_seed(seed + 20000)
                    changed = model(altered_batches(batch, seed)['shuffled_spectra'])
                    scored = metrics(changed, batch, loss)
                result['training' if training else 'eval'] = dict(gradient_norms=norms,
                    variation=dict(pooled=variation(pooled), **{k: variation(out[k]) for k in ('latent', 'mu', 'sigma', 'pi_logits')}),
                    biology=float(bio.detach()), shuffled_metrics=scored,
                    shuffled_mu_rms=float((changed['mu'][:, 0]-out['mu'][:, 0]).detach().square().mean().sqrt()))
    finally:
        model.train(was_training)
    return result


@torch.no_grad()
def evaluate(model: ExecutionModel, loader: DataLoader, metadata: Any, loss: ZILNLoss,
             device: torch.device, expected_rows: np.ndarray, spatial: bool = False) -> tuple[dict, dict]:
    """Exact coverage/order validation; CNN biology/aggregation forbids the shortcut."""
    if not spatial and (model.encoder.aggregation_mlp.radius or model.biology_predictor.radius):
        raise ValueError('CNN biology/aggregation requires full spatial validation.')
    model.eval()
    predictions, identities = [], []
    for cpu in loader:
        batch = move_batch_to_device(cpu, device)
        if spatial:
            out = model(batch)
        else:
            latent = model.encoder(*(batch[k] for k in SPECTRAL_KEYS))
            out = model.biology_predictor(latent)
        predictions.append({k: out[k].cpu() for k in ('pi_logits', 'mu', 'sigma')})
        identities.append(cpu['row_ids'])
    ids = torch.cat(identities).numpy()
    assert len(ids) == len(expected_rows) and len(np.unique(ids)) == len(ids)
    order = np.argsort(ids)
    np.testing.assert_array_equal(ids[order], np.sort(expected_rows))
    positions = np.searchsorted(ids[order], expected_rows)
    take = order[positions]
    out = {k: torch.cat([p[k] for p in predictions])[take] for k in predictions[0]}
    batch = dict(targets=torch.from_numpy(metadata.targets[expected_rows]),
                 target_valid_mask=torch.from_numpy(metadata.target_valid_mask[expected_rows]))
    cpu_loss = copy.deepcopy(loss).cpu()
    result = metrics(out, batch, cpu_loss)
    result.update(extra_metrics(out, batch['targets'], batch['target_valid_mask'], loss.logit_epsilon))
    result['targets'] = {}
    for j, name in enumerate(('CD8', 'tumor', 'collagen', 'stroma')):
        one = {k: v[:, j:j+1] for k, v in out.items()}
        y, mask = batch['targets'][:, j:j+1], batch['target_valid_mask'][:, j:j+1]
        result['targets'][name] = dict(loss=float(ZILNLoss([1.], loss.logit_epsilon, 'mean', loss.include_normal_constant)(
            *(one[k] for k in ('pi_logits', 'mu', 'sigma')), y, mask).total),
            **extra_metrics(one, y, mask, loss.logit_epsilon))
    return result, dict(row_ids=expected_rows, **out)


class Deadline:
    """One persistent wall-clock deadline, including evaluation/audits and restarts."""
    def __init__(self, path: Path, minutes: float):
        self.path, self.minutes = path, minutes
        self.record = json.loads(path.read_text()) if path.exists() else None
        if self.record and self.record['budget_minutes'] != minutes:
            raise ValueError('Cannot change a started budget.')

    def start(self) -> None:
        """Persist the start once; restarts never extend the deadline."""
        if self.record is None:
            now = time.time()
            self.record = dict(started=now, deadline=now + self.minutes*60, budget_minutes=self.minutes)
            save_json(self.path, self.record)

    def remaining(self, reserve: float = 0.) -> float:
        """Return available wall seconds after the requested reserve."""
        if self.record is None:
            return self.minutes*60-reserve
        return self.record['deadline']-time.time()-reserve

    def check(self, reserve: float = 300.) -> None:
        """Stop before spending the reserved save/report interval."""
        if self.remaining(reserve) <= 0:
            raise TimeoutError('Persistent experiment deadline/reserve reached.')


def completed_matches(directory: Path, contract: dict) -> bool:
    """Skip only completed, intact runs with exactly the requested contract."""
    completion = directory/'completion.json'
    if not completion.exists():
        return False
    data = json.loads(completion.read_text())
    if data['contract'] != contract:
        raise ValueError(f'Resume contract mismatch: {directory}')
    for name, expected in data['hashes'].items():
        if file_hash(directory/name) != expected:
            raise ValueError(f'Corrupt evidence: {directory/name}')
    return True


def fit(directory: Path, reference: AdversarialLatentFusion, config: dict, bundle: Any,
        tiles: SpatialTileDataset, plan: Plan, mode: str, dropout: str, updates: int,
        deadline: Deadline, audit_cpu: dict, adversarial: bool = False,
        resume: bool = False, reserve: float = 300.) -> dict:
    """Fit one immutable diagnostic attempt, saving exact state on failure or deadline."""
    device = resolve_device(config['training']['device'])
    seed = config['training']['seed']
    contract = dict(mode=mode, dropout=dropout, plan=plan.sha256, updates=updates, seed=seed,
        adversarial=adversarial, config=digest(config), initialization=file_hash(directory.parent/f'initial_{seed}.pt'),
        implementation=file_hash(Path(__file__)), learning_rate=.001, grl_horizon=300*312-1,
        reference_sha256=REFERENCE_HASH)
    if directory.exists():
        if resume and completed_matches(directory, contract):
            return json.loads((directory/'history.json').read_text())[-1]
        raise FileExistsError(f'Incomplete or unmatched evidence is immutable: {directory}')
    directory.mkdir()
    save_json(directory/'settings.json', contract)
    model = adapt(reference, mode, dropout).to(device)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory/'initial_state.pt')
    optimizer = optimizer_for(model, config, .001)
    loss = ZILNLoss.from_config(config).to(device)
    spatial = mode in ('X3', 'X4')
    workers = config['training']['spatial']['num_workers']
    dataset = PlannedDataset(tiles, plan, spatial)
    loader = loader_for(dataset, seed+1, workers)
    val_spatial = bool(model.encoder.aggregation_mlp.radius or model.biology_predictor.radius)
    val_tiles = None
    if val_spatial:
        val_tiles = SpatialTileDataset(config, bundle.split_indices['validation'], bundle.metadata, geometry=tiles.geometry)
        validation = DataLoader(val_tiles, batch_size=8, collate_fn=tile_collate, shuffle=False,
            num_workers=workers, persistent_workers=workers>0, generator=torch.Generator().manual_seed(seed+2))
    else:
        validation = DataLoader(bundle.datasets['validation'], batch_size=4096, collate_fn=sparse_collate,
            shuffle=False, num_workers=workers, persistent_workers=workers>0,
            generator=torch.Generator().manual_seed(seed+2))
    batch_audit = move_batch_to_device(audit_cpu if spatial else core_flat(audit_cpu), device)
    weights = bundle.batch_class_weights.to(device) if config['loss']['balance_batch_classes'] else None
    history, audits, training = [], [], []
    started, rows_seen, completed = time.monotonic(), 0, 0
    iterator = None
    status, failure = 'incomplete', None
    try:
        for update in range(updates+1):
            deadline.check(reserve)
            if update:
                if iterator is None:
                    iterator = iter(loader)
                start = time.monotonic()
                cpu = next(iterator)
                np.testing.assert_array_equal(cpu['row_ids'].numpy(), plan.batch(update-1))
                batch = move_batch_to_device(cpu, device)
                torch.manual_seed(seed+update)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                strength = grl_strength((update-1)/(300*312-1), config['training']['grl_schedule'],
                    config['training']['grl_max_lambda'], config['training']['grl_gamma']) if adversarial else 0.
                _, bio, total = objective(model, batch, loss, config, strength, adversarial, weights)
                if not torch.isfinite(total):
                    raise FloatingPointError(f'Nonfinite loss at update {update}')
                total.backward()
                norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))
                optimizer.step()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                rows_seen += len(cpu['row_ids']); completed = update
                counts = torch.bincount(cpu['batches'], minlength=len(bundle.metadata.batch_names)).numpy()
                proportions = counts/counts.sum()
                valid = cpu['target_valid_mask']; positive = (cpu['targets'] > 0) & valid
                row = dict(update=update, rows=len(cpu['row_ids']), peaks=len(cpu['peak_indices']),
                    context_appearances=len(cpu['peak_counts']),
                    positive_cd8_fraction=float(positive[:, 0].sum()/valid[:, 0].sum()) if valid[:, 0].any() else None,
                    slides=int((counts>0).sum()), slide_proportions=proportions.tolist(),
                    slide_entropy=float(-(proportions[proportions>0]*np.log(proportions[proportions>0])).sum()),
                    valid=valid.sum(0).tolist(), positive=positive.sum(0).tolist(),
                    zero=((cpu['targets']==0)&valid).sum(0).tolist(),
                    valid_weight_mass=float((valid*torch.tensor(config['loss']['target_weights'])).sum()),
                    biology=float(bio.detach()), gradient_norm=norm, clipping_factor=min(1., 5./(norm+1e-6)),
                    rows_seen=rows_seen, pass_index=int(np.searchsorted(plan.passes[1:], update-1, side='right')),
                    plan_position=update, step_seconds=time.monotonic()-start,
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                    grl_strength=strength)
                training.append(row)
                with (directory/'updates.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                del batch, bio, total
            devices = [device.index or 0] if device.type=='cuda' else []
            if update in (0, 1, 5, 10, 25, 50) or update % 100 == 0:
                deadline.check(reserve)
                audits.append(dict(update=update, rows_seen=rows_seen,
                                   **audit(model, batch_audit, loss, config, seed)))
                save_json(directory/'audits.json', audits)
            if update % 100 == 0 or update in plan.passes or update == updates:
                deadline.check(reserve)
                with torch.random.fork_rng(devices=devices):
                    scored, predictions = evaluate(model, validation, bundle.metadata, loss, device,
                                                   bundle.split_indices['validation'], val_spatial)
                history.append(dict(update=update, rows_seen=rows_seen, validation=scored,
                                    seconds=time.monotonic()-started))
                save_json(directory/'history.json', history)
                torch.save(predictions, directory/f'validation_{update}.pt')
                print(json.dumps(dict(run=directory.name, **history[-1])), flush=True)
            if update and update % 25 == 0:
                print(json.dumps(dict(run=directory.name, update=update, seconds=time.monotonic()-started)), flush=True)
        status = 'complete'
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        save_json(directory/'failure.json', failure)
        print(json.dumps(dict(run=directory.name, failure=failure)), flush=True)
    finally:
        torch.save(dict(format='batch-execution-diagnostic-v1', model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(), contract=contract, completed_update=completed,
            plan_position=completed, processed_rows=rows_seen, status=status,
            split_indices=bundle.split_indices, split_sha256=file_hash(directory.parent/'splits.npz') if (directory.parent/'splits.npz').exists() else None),
            directory/'final.pt')
        save_json(directory/'training.json', training)
        save_json(directory/'status.json', dict(status=status, completed_update=completed,
                                                seconds=time.monotonic()-started, failure=failure))
        if status == 'complete':
            save_json(directory/'completion.json', dict(contract=contract, hashes={p.name:file_hash(p)
                for p in directory.iterdir() if p.is_file() and p.name != 'completion.json'}))
        del iterator, loader, validation, model, optimizer, batch_audit
        tiles.close(); bundle.datasets['validation'].close()
        if val_tiles is not None:
            val_tiles.close()
        if device.type=='cuda':
            torch.cuda.empty_cache()
    if failure:
        raise RuntimeError(f'{directory.name}: {failure["type"]}: {failure["message"]}')
    return history[-1]


def geometry_config(config: dict, placement: str = 'none', core_size: int = 32) -> dict:
    """Derive diagnostic tile geometry while keeping production configuration intact."""
    result = copy.deepcopy(config)
    result['training']['core_size'] = core_size
    if placement == 'none':
        result['model']['heads']['discriminator']['type'] = 'cnn'
    return result


def freeze(root: Path, previous: Path, config: dict, seeds: list[int], budget: float) -> None:
    """Freeze provenance before any GPU work; reject mismatched resumed inputs."""
    sources = {str(p): file_hash(p) for p in Path('dann').rglob('*.py')}
    contract = dict(previous=str(previous.resolve()), reference_sha256=file_hash(previous/'reference.pt'),
                    source_config_sha256=file_hash(previous/'source_config.yaml'), seeds=seeds,
                    config=config, sources=sources, budget_minutes=budget)
    if contract['reference_sha256'] != REFERENCE_HASH:
        raise ValueError('Archived reference SHA256 differs.')
    if (root/'manifest.json').exists():
        if json.loads((root/'manifest.json').read_text())['contract'] != contract:
            raise ValueError('Root resume contract differs.')
        return
    for source in sources:
        dest = root/'source'/source
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    shutil.copyfile(previous/'source_config.yaml', root/'source_config.yaml')
    shutil.copyfile(previous/'constant_baseline.json', root/'constant_baseline.json')
    save_json(root/'manifest.json', dict(contract=contract, environment=dict(
        torch=torch.__version__, python=platform.python_version(), cuda=torch.version.cuda,
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        deterministic_algorithms=config['training']['deterministic_algorithms'],
        allocator=os.environ.get('PYTORCH_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF')),
        threads=torch.get_num_threads()), constraints=dict(test_evaluation=False, preprocessing_refit=False,
        tissue_export=False, optimizer_lr=.001, dropout_seed='seed + update', grl_progress='(update - 1) / (300 * 312 - 1)')))


def prepare_audit(previous: Path, tiles: SpatialTileDataset, train: np.ndarray) -> dict:
    """512 archived training core IDs, complete context, unchanged ordered labels."""
    archived = torch.load(previous/'A_spatial_mlp/audit_batch.pt', weights_only=False)
    selected = archived['row_ids'][torch.linspace(0, len(archived['row_ids'])-1, 512).long()].numpy()
    assert np.isin(selected, train).all() and len(np.unique(selected)) == len(selected)
    plan = Plan(selected, np.array([0, len(selected)]), np.array([0, 1]), np.array([], dtype='int64'), np.array([0]))
    return PlannedDataset(tiles, plan, True)[0]


def preflight(root: Path, reference: AdversarialLatentFusion, config: dict,
              tiles: SpatialTileDataset, plans: dict[str, Plan], audit_cpu: dict,
              deadline: Deadline) -> dict:
    """Audit each seed and preflight each flat schedule without blocking smaller arms.

    Peak loads differ between seeds and samplers. A successful first-seed check
    must never be treated as a memory guarantee for a later seed's larger batch.
    """
    seed = config['training']['seed']
    path = root/f'preflight_{seed}.json'
    contract = dict(seed=seed, plans={name: plan.sha256 for name, plan in plans.items()})
    if path.exists():
        result = json.loads(path.read_text())
        if result.get('contract') != contract:
            raise ValueError('Cached preflight sampling contract differs.')
        if not result['equivalence']['passed'] or not result['checkpoint']['passed']:
            raise ValueError('Prior equivalence audit failed.')
        return result
    deadline.start()
    deadline.check(300.)
    device = resolve_device(config['training']['device'])
    reference = copy.deepcopy(reference).to(device)
    one = tile_collate([tiles[0]])
    one = select_labels(one, torch.arange(min(8, len(one['row_ids']))))
    flat = move_batch_to_device(core_flat(one), device)
    tiled = move_batch_to_device(one, device)
    try:
        result = dict(contract=contract, equivalence=equivalence(reference, flat, tiled, config),
                      checkpoint=equivalence(reference, flat, tiled, config, ('X1', 'X2'), True))
    except Exception as exc:
        save_json(root/f'preflight_failure_{seed}.json', dict(type=type(exc).__name__,
            message=str(exc), traceback=traceback.format_exc()))
        raise
    del flat, tiled
    import h5py
    with h5py.File(config['data']['path'], 'r') as handle:
        counts = np.diff(handle['X/indptr'][:])
    result['flat_feasibility'] = {}
    for name, plan in plans.items():
        deadline.check(300.)
        mass = np.add.reduceat(counts[plan.rows], plan.boundaries[:-1])
        index = int(mass.argmax())
        cpu = PlannedDataset(tiles, plan, False)[index]
        trial = dict(plan=name, batch=index, rows=len(cpu['row_ids']), peaks=int(mass[index]))
        model = adapt(reference, 'X0').train()
        optimizer = optimizer_for(model, config, .001)
        batch = move_batch_to_device(cpu, device)
        loss = ZILNLoss.from_config(config).to(device)
        out = bio = total = None
        try:
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            torch.manual_seed(seed+1)
            out, bio, total = objective(model, batch, loss, config)
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite monolithic preflight loss.')
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            trial.update(passed=True, gpu_peak_bytes=(torch.cuda.max_memory_allocated(device)
                                                     if device.type == 'cuda' else 0))
        except torch.cuda.OutOfMemoryError as exc:
            trial.update(passed=False, failure=str(exc), traceback=traceback.format_exc(),
                         blocked_modes=['X0'])
        except Exception as exc:
            save_json(root/f'preflight_failure_{seed}.json', dict(plan=name,
                type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
            raise
        finally:
            del model, optimizer, batch, loss, out, bio, total
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        result['flat_feasibility'][name] = trial
    result['largest_flat'] = max(result['flat_feasibility'].values(), key=lambda x: x['peaks'])
    save_json(path, result)
    if not (root/'preflight.json').exists():
        save_json(root/'preflight.json', result)
    tiles.close()
    return result


def report(root: Path) -> None:
    """Build per-seed comparisons and curves from complete artifacts only."""
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    baseline = json.loads((root/'constant_baseline.json').read_text())['validation']
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for directory in sorted(root.iterdir()):
        if not (directory/'history.json').exists():
            continue
        settings = json.loads((directory/'settings.json').read_text())
        history = json.loads((directory/'history.json').read_text())
        completed = (directory/'completion.json').exists()
        training = json.loads((directory/'training.json').read_text()) if (directory/'training.json').exists() else []
        for item in history:
            val = item['validation']
            rows.append(dict(run=directory.name, seed=settings['seed'], mode=settings['mode'], dropout=settings['dropout'],
                completed=completed, update=item['update'], rows_seen=item['rows_seen'], cd8=val['cd8'],
                biology=val['biology'], positive_cd8_r2=val['positive_cd8_r2'],
                passed=val['cd8']<=baseline['cd8']-.001 and val['biology']<baseline['biology'] and
                       val['positive_cd8_r2'] is not None and val['positive_cd8_r2']>0,
                mu_std=val['cd8_parameter_std']['mu'], seconds=item['seconds']))
        plotted = [h for h in history if h['update'] > 0]
        for ax, x in zip(axes[0], ('update', 'rows_seen')):
            ax.plot([h[x] for h in plotted], [h['validation']['cd8'] for h in plotted], label=directory.name)
            ax.set(xlabel=x, ylabel='Validation CD8 loss')
            ax.axhline(baseline['cd8'], color='black', ls=':')
        axes[1, 0].plot([h['update'] for h in history], [h['validation']['cd8_parameter_std']['mu'] for h in history])
        if training:
            axes[1, 1].plot([t['update'] for t in training], [t['slides'] for t in training], alpha=.4)
    axes[1, 0].set(xlabel='Updates', ylabel='CD8 mu standard deviation')
    axes[1, 1].set(xlabel='Updates', ylabel='Slides per update')
    if rows:
        axes[0, 0].legend(fontsize=5, ncol=2)
        pd.DataFrame(rows).to_csv(root/'comparison.csv', index=False)
    fig.tight_layout(); fig.savefig(root/'learning_curves.png', dpi=180); plt.close(fig)


def validate_selection(stage: str, controls: list[str], dropout: str,
                       placement: str, sampler: str) -> None:
    """Reject scientifically incompatible stage, execution, and correction choices."""
    allowed = dict(A=set(), B={'P', 'F', 'S'},
        C1={'F-native', 'F-small', 'F-dispersed', 'G-native', 'G-small'},
        C2=set(MODES), D={'S'})
    if stage not in allowed or not set(controls).issubset(allowed[stage]):
        raise ValueError('Controls are incompatible with the selected stage.')
    if placement != 'none' and stage != 'D':
        raise ValueError('CNN confirmation belongs to Stage D.')
    if sampler != 'native' and stage != 'D':
        raise ValueError('Correction samplers belong to Stage D.')
    permitted_dropout = ({'unchanged', 'off'} if stage == 'C2' else
                         {'unchanged', 'spectral-off', 'aggregation-off'} if stage == 'D' else {'unchanged'})
    if dropout not in permitted_dropout:
        raise ValueError('Dropout change is incompatible with this stage.')
    if sampler != 'native' and dropout != 'unchanged':
        raise ValueError('Do not combine sampler and dropout corrections.')


def main(argv: list[str] | None = None) -> None:
    """Execute an explicitly selected, frozen diagnostic stage within one deadline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-run', type=Path, default=PREVIOUS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-minutes', type=float, default=120.)
    parser.add_argument('--updates', type=int, help='Explicit update cap; default preserves the previous stage protocol.')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--stage', choices=['A', 'B', 'C1', 'C2', 'D', 'report'], default='B')
    parser.add_argument('--controls', nargs='+')
    parser.add_argument('--dropout', choices=['unchanged', 'off', 'spectral-off', 'aggregation-off'], default='unchanged')
    parser.add_argument('--placement', choices=['none', 'discriminator', 'aggregation', 'biology'], default='none')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sampler', choices=['native', 'small', 'rounds', 'rounds-small'], default='native')
    parser.add_argument('--repeat', default='')
    parser.add_argument('--estimate-seconds', type=float, default=420., help='Conservative per-run estimate for group admission.')
    args = parser.parse_args(argv)
    root = args.output
    if args.updates is not None and args.updates <= 0:
        parser.error('Updates must be positive.')
    if args.stage == 'report':
        report(root); return
    if not math.isfinite(args.budget_minutes) or args.budget_minutes <= 0:
        parser.error('Budget must be finite and positive.')
    if not math.isfinite(args.estimate_seconds) or args.estimate_seconds <= 0:
        parser.error('Runtime estimate must be finite and positive.')
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(SEEDS):
        parser.error('Use unique prescribed seeds; use --repeat for fresh repeat artifacts.')
    selected = args.controls or dict(A=[], B=['P','F','S'],
        C1=['F-native','F-small','F-dispersed','G-native','G-small'],
        C2=['X1','X2','X3'], D=['S'])[args.stage]
    try:
        validate_selection(args.stage, selected, args.dropout, args.placement, args.sampler)
    except ValueError as exc:
        parser.error(str(exc))
    if root.exists() and not args.resume:
        parser.error('Output exists; choose a new directory or explicit --resume.')
    root.mkdir(parents=True, exist_ok=args.resume)
    config = load_config(args.previous_run/'source_config.yaml')
    freeze(root, args.previous_run, config, SEEDS, args.budget_minutes)
    payload = torch.load(args.previous_run/'reference.pt', map_location='cpu', weights_only=False)
    if target_contract(config) != checkpoint_contract(payload):
        raise ValueError('Target contract differs.')
    bundle = create_data_bundle(config, payload['split_indices'], payload['data_identity'])
    assert tuple(payload['batch_names']) == bundle.metadata.batch_names
    assert bundle.metadata.num_observations == 1698428
    assert config['model']['num_peaks'] == 5808 and len(bundle.metadata.batch_names) == 43
    assert [len(bundle.split_indices[s]) for s in ('train', 'validation', 'test')] == [1358726, 169824, 169878]
    if not (root/'splits.npz').exists():
        np.savez(root/'splits.npz', **bundle.split_indices)
        save_json(root/'data_identity.json', payload['data_identity'])
    del payload
    deadline = Deadline(root/'deadline.json', args.budget_minutes)
    tiles = SpatialTileDataset(geometry_config(config), bundle.split_indices['train'], bundle.metadata)
    audit_cpu = prepare_audit(args.previous_run, tiles, bundle.split_indices['train'])
    tiles.close()
    if not (root/'audit_batch.pt').exists():
        torch.save(audit_cpu, root/'audit_batch.pt')
    definitions = dict(P=('G-small','X0'), F=('F-native','X0'), S=('F-native','X4'),
        X0=('F-native','X0'), X1=('F-native','X1'), X2=('F-native','X2'), X3=('F-native','X3'), X4=('F-native','X4'),
        **{name:(name,'X0') for name in ('F-native','F-small','F-dispersed','G-native','G-small')})
    defaults = dict(A=[], B=['P','F','S'], C1=['F-native','F-small','F-dispersed','G-native','G-small'],
                    C2=['X1','X2','X3'], D=['S'])
    controls = args.controls or defaults[args.stage]
    if any(name not in definitions for name in controls):
        parser.error('Unknown control.')
    if args.placement != 'none' and (args.stage != 'D' or any(definitions[n][1]!='X4' for n in controls)):
        parser.error('CNN confirmations require Stage D / X4.')
    if args.sampler != 'native' and args.stage != 'D':
        parser.error('Correction samplers belong to Stage D.')
    if args.stage != 'D' and args.dropout not in ('unchanged','off'):
        parser.error('Targeted dropout correction belongs to Stage D.')
    for seed in args.seeds:
        if seed not in SEEDS:
            parser.error('Only the three prescribed seeds are allowed.')
        config['training']['seed'] = seed
        seed_everything(seed, config['training']['deterministic_algorithms'])
        reference = AdversarialLatentFusion.from_config(config, 43, 4)
        initial_path = root/f'initial_{seed}.pt'
        if seed == SEEDS[0]:
            reference.load_state_dict(torch.load(args.previous_run/'A_spatial_mlp/initial_state.pt', weights_only=True))
        if initial_path.exists():
            reference.load_state_dict(torch.load(initial_path, weights_only=True))
        else:
            torch.save(reference.state_dict(), initial_path)
        passes = max(2, math.ceil((args.updates or 500) / min(math.ceil(len(tiles)/8), math.ceil(len(bundle.split_indices['train'])/2048))))
        plans = make_plans(tiles, bundle.split_indices['train'], bundle.metadata.batch_codes, seed, passes=passes)
        for name, plan in plans.items():
            plan.save(root/f'plan_{seed}_{name}.npz')
        checks = dict(seed=seed, hashes={k:p.sha256 for k,p in plans.items()},
            native_rows_500=int(plans['F-native'].boundaries[500]),
            native_order_500=hashlib.sha256(plans['F-native'].rows[:plans['F-native'].boundaries[500]].tobytes()).hexdigest())
        if seed == SEEDS[0]:
            archived = json.loads((args.previous_run/'A_spatial_mlp/history.json').read_text())[-1]
            assert checks['native_order_500'] == archived['order_sha256'], 'Native order does not reproduce archive.'
            assert checks['native_rows_500'] == archived['rows_seen']
        save_json(root/f'matching_{seed}.json', checks)
        checks = preflight(root, reference, config, tiles, plans, audit_cpu, deadline)
        if args.stage == 'A':
            break
        config_run = copy.deepcopy(config)
        run_tiles, run_audit = tiles, audit_cpu
        if args.placement != 'none':
            group = config_run['model']['aggregation'] if args.placement=='aggregation' else config_run['model']['heads'][args.placement]
            group['type'] = 'cnn'
            config_run['training']['core_size'] = 32
            from dann.model_components.aggregation_cnn import CNNAggregation
            from dann.model_components.biology_cnn import CNNBiology
            from dann.model_components.discriminator_cnn import CNNDiscriminator
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+10000)
                if args.placement=='aggregation':
                    reference.encoder.aggregation_mlp = CNNAggregation(256,256,**group['cnn'])
                elif args.placement=='biology':
                    reference.biology_predictor = CNNBiology(256,12,**group['cnn'])
                else:
                    reference.batch_discriminator = CNNDiscriminator(256,43,**group['cnn'])
            reference.halo = reference.encoder.aggregation_mlp.radius+max(reference.biology_predictor.radius,reference.batch_discriminator.radius)
            run_tiles = SpatialTileDataset(config_run, bundle.split_indices['train'], bundle.metadata, geometry=tiles.geometry)
            run_audit = prepare_audit(args.previous_run, run_tiles, bundle.split_indices['train'])
            run_tiles.close()
        correction_plan = None
        if args.sampler.startswith('rounds'):
            geometry = geometry_config(config_run, args.placement, core_size=8)
            run_tiles = SpatialTileDataset(geometry, bundle.split_indices['train'], bundle.metadata, geometry=tiles.geometry)
            correction_plan = round_plan(run_tiles, plans['F-native'], bundle.split_indices['train'], seed,
                                         2048 if args.sampler=='rounds-small' else None)
            correction_plan.save(root/f'plan_{seed}_{args.sampler}.npz')
            # Audit stays on the same native 512 row IDs and complete native context.
        elif args.sampler=='small':
            correction_plan = plans['F-small']
        tag = args.repeat + ('' if args.sampler=='native' else '_' + args.sampler)
        reserve = 300. if args.stage=='D' else 2100.
        pending = [n for n in controls if not (root/f'{args.stage}_{n}_{seed}_{args.placement}_{args.dropout}{tag}'/'completion.json').exists()]
        if deadline.remaining(reserve) < args.estimate_seconds*len(pending)*1.2:
            save_json(root/f'budget_skip_{args.stage}_{seed}{args.repeat}.json', dict(controls=pending,
                available=deadline.remaining(reserve), required=args.estimate_seconds*len(pending)*1.2))
            break
        for name in controls:
            plan_name, mode = definitions[name]
            if mode=='X0' and not checks['flat_feasibility'][plan_name]['passed']:
                save_json(root/f'blocked_{args.stage}_{name}_{seed}.json', dict(reason='This seed/schedule monolithic preflight OOMed; no silent microbatch substitution.'))
                continue
            plan = correction_plan if correction_plan is not None else plans[plan_name]
            updates = args.updates if args.updates is not None else len(plan.boundaries)-1 if args.stage=='C1' else 500
            fit(root/f'{args.stage}_{name}_{seed}_{args.placement}_{args.dropout}{tag}', reference,
                config_run, bundle, run_tiles, plan, mode, args.dropout, updates, deadline, run_audit,
                adversarial=args.placement!='none', resume=args.resume, reserve=reserve)
            report(root)
    for dataset in bundle.datasets.values():
        dataset.close()
    tiles.close()
    report(root)


if __name__ == '__main__':
    main()

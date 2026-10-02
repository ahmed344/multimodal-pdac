"""Full-epoch controls for input source, tile batching, and spatial execution.

Each invocation creates a new directory and runs one explicitly selected control.
It never evaluates test rows, exports tissue, or changes production configuration.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from dann.config import load_config, seed_everything, resolve_device
from dann.data_loader import create_data_bundle, sparse_collate, DataBundle
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion, transform_biology
from dann.model_components.layers import SpatialNetwork
from dann.tiles import SpatialTileDataset, tile_collate
from dann.train import run_epoch, save_checkpoint, move_batch_to_device
from dann.verification.fit_diagnosis import (
    constant_parameters, metrics, stage_audit, gradient_stats, optimizer_for, save_json,
)


def flattened_tile_collate(tiles: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    """Flatten exactly the supervised core rows, in tile_collate's order."""
    samples = []
    for tile in tiles:
        lookup = dict(zip(tile['spatial_positions'], tile['samples']))
        for i, position in enumerate(tile['core_positions']):
            samples.append(dict(lookup[position], row_id=tile['row_ids'][i],
                targets=tile['targets'][i], raw_targets=tile['raw_targets'][i],
                target_valid_mask=tile['target_valid_mask'][i], batch=tile['batches'][i]))
    return sparse_collate(samples)


class FlatTileDataset(SpatialTileDataset):
    """Reuse spatial tile membership while reading only supervised core spectra."""

    def __getitem__(self, item: int) -> dict[str, Any]:
        _, selected = self.tiles[item]
        rows = np.asarray([row for row, _ in selected], dtype=np.int64)
        positions = np.arange(len(rows), dtype=np.int64)
        m = self.metadata
        return dict(samples=[self.spectra[context_row] for _, context_row in selected],
                    spatial_positions=positions, core_positions=positions, row_ids=rows,
                    targets=m.targets[rows], raw_targets=m.raw_targets[rows],
                    target_valid_mask=m.target_valid_mask[rows], batches=m.batch_codes[rows])


class ConvertedNetwork(nn.Module):
    """Copy an MLP to convolutions, retaining its exact hidden layer ordering."""

    def __init__(self, network: nn.Sequential, kernel_size: int = 1) -> None:
        super().__init__()
        layers = []
        for layer in network:
            if isinstance(layer, nn.Linear):
                conv = nn.Conv2d(layer.in_features, layer.out_features, kernel_size,
                                 padding=kernel_size // 2).to(layer.weight)
                with torch.no_grad():
                    conv.weight.zero_()
                    conv.weight[:, :, kernel_size // 2, kernel_size // 2].copy_(layer.weight)
                    conv.bias.copy_(layer.bias)
                layers.append(conv)
            else:
                layers.append(copy.deepcopy(layer))
        self.layers = nn.ModuleList(layers)
        self.radius = sum(isinstance(x, nn.Conv2d) for x in layers) * (kernel_size // 2)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = value * mask
        for layer in self.layers:
            value = (layer(value.movedim(1, -1)).movedim(-1, 1)
                     if isinstance(layer, nn.LayerNorm) else layer(value))
            value = value * mask
        return value


class SpatialControl(AdversarialLatentFusion):
    """Diagnostic-only spatial adapter; production model and checkpoint API stay intact."""

    @staticmethod
    def _on_map(component, value, mask):
        if isinstance(component, (ConvertedNetwork, SpatialNetwork)):
            return component(value, mask)
        return AdversarialLatentFusion._on_map(component, value, mask)

    def forward(self, batch, grl_strength=1.):
        grid = self._latent_map(batch)
        mask = batch['occupancy']
        raw = self._on_map(self.biology_predictor, grid, mask)
        logits = self._on_map(self.batch_discriminator,
                             self.gradient_reversal(grid, grl_strength), mask)
        return dict(latent=self._select(grid, batch),
                    **transform_biology(self._select(raw, batch), self.num_targets, self.sigma_min),
                    batch_logits=self._select(logits, batch))


def spatial_control(reference: AdversarialLatentFusion, config: Mapping[str, Any],
                    mixing: str = 'none', production: str = 'none') -> SpatialControl:
    """Convert a copy; 3x3 neighbors begin at zero and centers copy the MLP."""
    model = copy.deepcopy(reference)
    model.__class__ = SpatialControl
    model.spatial = True
    model.encoder.aggregation_mlp = ConvertedNetwork(reference.encoder.aggregation_mlp,
                                                    3 if mixing == 'aggregation' else 1)
    model.biology_predictor = ConvertedNetwork(reference.biology_predictor.network,
                                               3 if mixing == 'biology' else 1)
    model.batch_discriminator = ConvertedNetwork(reference.batch_discriminator.network)
    if production == 'aggregation':
        model.encoder.aggregation_mlp = SpatialNetwork(
            config['model']['spectral_encoder']['deep_sets']['peak_output_dim'],
            model.latent_dim, **config['model']['aggregation']['cnn'])
    elif production == 'biology':
        model.biology_predictor = SpatialNetwork(model.latent_dim, model.num_targets * 3,
                                                **{k: v for k, v in config['model']['heads']['biology']['cnn'].items()
                                                   if k != 'residual'})
    model.halo = model.encoder.aggregation_mlp.radius + max(
        model.biology_predictor.radius, model.batch_discriminator.radius)
    return model


def equivalence_audit(reference: AdversarialLatentFusion, spatial: SpatialControl,
                      flat: Mapping[str, torch.Tensor], tiled: Mapping[str, torch.Tensor],
                      config: Mapping[str, Any],
                      class_weights: torch.Tensor | None = None) -> dict[str, Any]:
    """Assert forward, all parameter gradients, and a complete AdamW update match."""
    a, b = copy.deepcopy(reference).train(), copy.deepcopy(spatial).train()
    for model in (a, b):
        for layer in model.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = 0.
    # Float64 removes near-zero AdamW gradient sign sensitivity in the audit.
    a.double(); b.double()
    flat = {k: v.double() if v.is_floating_point() else v for k, v in flat.items()}
    tiled = {k: v.double() if v.is_floating_point() else v for k, v in tiled.items()}
    loss = ZILNLoss.from_config(config).to(next(a.parameters()).device)
    optimizers = [optimizer_for(m, config, .01) for m in (a, b)]
    if class_weights is not None:
        class_weights = class_weights.to(device=next(a.parameters()).device, dtype=torch.float64)
    outputs = []
    for model, batch in ((a, flat), (b, tiled)):
        out = model(batch, grl_strength=.1)
        objective = config['loss']['biology_weight'] * loss(
            out['pi_logits'], out['mu'], out['sigma'], batch['targets'], batch['target_valid_mask']).total
        objective = objective + config['loss']['batch_weight'] * nn.functional.cross_entropy(
            out['batch_logits'], batch['batches'], weight=class_weights)
        objective.backward()
        outputs.append(out)
    forward_error = max(float((outputs[0][k]-outputs[1][k]).detach().abs().max()) for k in outputs[0])
    for k in outputs[0]:
        torch.testing.assert_close(outputs[0][k], outputs[1][k], atol=1e-9, rtol=1e-7)
    def paired():
        left, right = list(a.parameters()), list(b.parameters())
        assert len(left) == len(right)
        for p, q in zip(left, right):
            yield p, q
    gradient_error = 0.
    for p, q in paired():
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            gradient_error = max(gradient_error, float((p.grad-q.grad.reshape_as(p)).abs().max()))
            torch.testing.assert_close(p.grad, q.grad.reshape_as(p), atol=1e-8, rtol=1e-6)
    for model, opt in zip((a, b), optimizers):
        if config['training']['gradient_clip_norm'] is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'])
        opt.step()
    update_error = 0.
    for p, q in paired():
        update_error = max(update_error, float((p-q.reshape_as(p)).detach().abs().max()))
        torch.testing.assert_close(p, q.reshape_as(p), atol=1e-7, rtol=1e-6)
    return dict(passed=True, dtype='float64', dropout=False, forward_max_abs=forward_error,
                gradient_max_abs=gradient_error, updated_parameter_max_abs=update_error)


def extra_metrics(outputs: Mapping[str, torch.Tensor], targets: torch.Tensor,
                  mask: torch.Tensor, epsilon: float) -> dict[str, float | None]:
    """Full-validation positive CD8 correlation/R² and structural-zero discrimination."""
    from sklearn.metrics import roc_auc_score
    y = targets[:, 0].double(); valid = mask[:, 0]
    positive = valid & (y > 0)
    z = torch.logit(y[positive].clamp(epsilon, 1-epsilon))
    mu = outputs['mu'][positive, 0].double()
    correlation = (float(torch.corrcoef(torch.stack((z, mu)))[0, 1])
                   if len(z) > 1 and mu.std() > 1e-12 and z.std() > 0 else None)
    truth = (y[valid] == 0).numpy()
    logits = outputs['pi_logits'][valid, 0].double().numpy()
    # Use one elementwise implementation, avoiding vector/tail sigmoid rounding
    # that can give a mathematically constant baseline a spurious AUC ranking.
    from scipy.special import expit
    p = expit(logits)
    return dict(positive_cd8_correlation=correlation,
        positive_cd8_r2=float(1-(mu-z).square().sum()/(z-z.mean()).square().sum()) if len(z)>1 and z.max() > z.min() else None,
        cd8_zero_auc=float(roc_auc_score(truth, p)) if len(np.unique(truth)) == 2 else None,
        cd8_zero_brier=float(np.mean((p-truth)**2)))


def file_hash(path: str | Path) -> str:
    """Stream SHA256 without loading checkpoints or exports into memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    """Run one full-data control without changing the schedule horizon or source run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--reference-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-source', choices=['labeled', 'tissue'], required=True)
    parser.add_argument('--execution-path', choices=['pixel', 'flat-tiles', 'spatial'], required=True)
    parser.add_argument('--epoch-limit', type=int, default=3)
    parser.add_argument('--mixing', choices=['none', 'aggregation', 'biology'], default='none')
    parser.add_argument('--production', choices=['none', 'aggregation', 'biology'], default='none')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config['model']['heads']['biology']['cnn']['residual'] = False  # Historical plain-CNN controls.
    t = config['training']; seed = int(t['seed'])
    if not 1 <= args.epoch_limit <= t['epochs']:
        parser.error('epoch-limit must be positive and cannot exceed the unchanged schedule horizon')
    if args.input_source != 'labeled':
        parser.error('Tissue training is invalid: all supervised spectra/context must use the labeled file.')
    if (args.mixing != 'none' or args.production != 'none') and args.execution_path != 'spatial':
        parser.error('Mixing/production controls require spatial execution.')
    if args.mixing != 'none' and args.production != 'none':
        parser.error('Isolate mixing and production changes separately.')
    if config['execution']['mode'] != 'pixel' or t['learning_rate'] != .01:
        parser.error('Reference must be all-MLP at learning rate 0.01.')
    if any(config['data'].get('max_'+s+'_samples') is not None for s in ('train','validation','test')):
        parser.error('Full-epoch controls do not allow sample caps.')
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_hashes = {str(p): file_hash(p) for p in (args.config, args.reference_checkpoint)}
    (root/'source_config.yaml').write_bytes(args.config.read_bytes())
    (root/'runner.py').write_bytes(Path(__file__).read_bytes())
    for source in Path('dann').rglob('*.py'):
        destination = root/'source'/source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    reference = torch.load(args.reference_checkpoint, map_location='cpu', weights_only=False)
    if reference.get('data_identity') is None:
        raise ValueError('Reference requires saved row and feature identities.')
    seed_everything(seed, t['deterministic_algorithms'])
    bundle = create_data_bundle(config, reference['split_indices'], reference['data_identity'])
    if tuple(reference['batch_names']) != bundle.metadata.batch_names:
        raise ValueError('Reference batch class ordering differs.')
    from dann.targets import target_contract, checkpoint_contract
    if target_contract(config) != checkpoint_contract(reference):
        raise ValueError('Reference target contract differs.')
    m = bundle.metadata
    device = resolve_device(t['device'])
    model = AdversarialLatentFusion.from_config(config, len(m.batch_names), len(config['data']['target_columns']))
    initial = {k: v.clone() for k,v in model.state_dict().items()}
    torch.save(initial, root/'initial_mlp_state.pt')
    if args.execution_path == 'spatial':
        # Conversion must not advance dropout's training RNG stream.
        with torch.random.fork_rng():
            model = spatial_control(model, config, args.mixing, args.production)
    model.to(device)
    datasets = {k: bundle.datasets[k] for k in ('train','validation')}
    collator = sparse_collate
    if args.execution_path != 'pixel':
        geometry_config = copy.deepcopy(config)
        geometry_config['training']['core_size'] = t['spatial']['core_size']
        # The tile reader derives radii from component settings. Use geometry-only settings.
        if args.execution_path == 'spatial':
            for component, radius in ((geometry_config['model']['aggregation'], model.encoder.aggregation_mlp.radius),
                (geometry_config['model']['heads']['biology'], model.biology_predictor.radius)):
                if radius:
                    component['type'] = 'cnn'; component['cnn']['depth'] = radius
        datasets = {}
        geometry = None
        dataset_type = SpatialTileDataset if args.execution_path == 'spatial' else FlatTileDataset
        for s in ('train','validation'):
            datasets[s] = dataset_type(geometry_config, bundle.split_indices[s], m, geometry=geometry)
            geometry = datasets[s].geometry
        collator = tile_collate if args.execution_path == 'spatial' else flattened_tile_collate
    pixel = args.execution_path == 'pixel'
    workers = t['pixel' if pixel else 'spatial']['num_workers']
    sizes = (t['pixel']['batch_size'], t['pixel']['validation_batch_size']) if pixel else (t['spatial']['tiles_per_batch'],)*2
    kwargs = dict(num_workers=workers, persistent_workers=workers>0, pin_memory=t['pin_memory'], collate_fn=collator)
    if workers:
        kwargs['prefetch_factor'] = t['pixel' if pixel else 'spatial']['prefetch_factor']
    train_generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(datasets[s], batch_size=size, shuffle=s=='train',
        generator=train_generator if s=='train' else torch.Generator().manual_seed(seed+1), **kwargs)
        for s, size in zip(('train','validation'), sizes)}
    bundle = DataBundle(m, bundle.split_indices, datasets, loaders, bundle.batch_class_weights)
    settings = dict(input_source=args.input_source, execution_path=args.execution_path, mixing=args.mixing,
        production=args.production, epoch_limit=args.epoch_limit, schedule_epochs=t['epochs'],
        learning_rate=t['learning_rate'], core_size=t['spatial']['core_size'], tiles_per_batch=t['spatial']['tiles_per_batch'],
        dropout=config['model']['dropout'], test_evaluation=False, tissue_exports=False,
        training_rows=len(bundle.split_indices['train']), validation_rows=len(bundle.split_indices['validation']),
        steps_per_epoch=len(loaders['train']), seed=seed, config=config, source_hashes=source_hashes,
        torch_version=torch.__version__, device=str(device),
        cuda_allocator=os.environ.get('PYTORCH_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF')))
    from dann.checkpoints import training_data_contract
    settings['training_data_contract'] = training_data_contract(config, reference['data_identity'])
    save_json(root/'settings.json', settings)
    np.savez(root/'splits.npz', **bundle.split_indices)
    loss = ZILNLoss.from_config(config).to(device)
    # Baseline fitting sees training rows only; scoring sees validation rows only.
    rows = bundle.split_indices['train']
    constant = constant_parameters(torch.from_numpy(m.targets[rows]), torch.from_numpy(m.target_valid_mask[rows]),
                                   loss.logit_epsilon, config['model']['sigma_min'])
    params = {k: v[:1] for k,v in constant.items()}
    del constant
    rows = bundle.split_indices['validation']
    val_targets = torch.from_numpy(m.targets[rows]); val_mask = torch.from_numpy(m.target_valid_mask[rows])
    baseline_out = {k: v.expand(len(rows), -1) for k,v in params.items()}
    baseline = metrics(baseline_out, dict(targets=val_targets,target_valid_mask=val_mask), ZILNLoss.from_config(config))
    baseline.update(extra_metrics(baseline_out, val_targets, val_mask, loss.logit_epsilon))
    save_json(root/'constant_baseline.json', dict(parameters={k:v[0].tolist() for k,v in params.items()},
        validation=baseline, fit_split='train', evaluation_split='validation'))
    # Fixed first validation batch for stage diagnostics; never advances the training sampler.
    audit_cpu = collator([datasets['validation'][i] for i in range(min(sizes[1],len(datasets['validation'])))])
    datasets['validation'].close()  # Workers must open their own HDF5 handles.
    audit_batch = move_batch_to_device(audit_cpu, device)
    torch.save(audit_cpu, root/'audit_batch.pt')
    if args.execution_path == 'spatial' and args.mixing == args.production == 'none':
        tiles = [datasets['train'][0]]
        flat = move_batch_to_device(flattened_tile_collate(tiles), device)
        tiled = move_batch_to_device(tile_collate(tiles), device)
        datasets['train'].close()
        with torch.random.fork_rng():
            pointwise = AdversarialLatentFusion.from_config(config, len(m.batch_names), len(config['data']['target_columns'])).to(device)
            pointwise.load_state_dict(initial)
            save_json(root/'equivalence.json', equivalence_audit(pointwise, model, flat, tiled, config,
                bundle.batch_class_weights if config['loss']['balance_batch_classes'] else None))
        del pointwise, flat, tiled
    optimizer = optimizer_for(model, config, t['learning_rate'])
    common = dict(model=model, ziln_loss=loss, device=device, biology_weight=config['loss']['biology_weight'],
        batch_weight=config['loss']['batch_weight'], class_weights=bundle.batch_class_weights.to(device)
        if config['loss']['balance_batch_classes'] else None,
        total_epochs=t['epochs'], grl_schedule=t['grl_schedule'], grl_max_lambda=t['grl_max_lambda'],
        grl_gamma=t['grl_gamma'], gradient_clip_norm=t['gradient_clip_norm'],
        target_columns=config['data']['target_columns'])
    history=[]
    for epoch in range(args.epoch_limit+1):
        row = dict(epoch=epoch, updates=epoch*len(loaders['train']))
        if epoch:
            tick=time.monotonic()
            order_hash = hashlib.sha256()
            seen_rows = []
            def training_order(module, inputs):
                ids = inputs[0]['row_ids'].detach().cpu().numpy()
                order_hash.update(ids.tobytes())
                seen_rows.append(ids)
            handle = model.register_forward_pre_hook(training_order)
            try:
                row['train'] = run_epoch(loader=loaders['train'], optimizer=optimizer, epoch_index=epoch-1, **common)
            finally:
                handle.remove()
            np.testing.assert_array_equal(np.sort(np.concatenate(seen_rows)), np.sort(bundle.split_indices['train']))
            row['training_order_sha256'] = order_hash.hexdigest()
            row['train_seconds']=time.monotonic()-tick
        # Evaluation and diagnostics must not perturb the training dropout RNG.
        with torch.random.fork_rng():
            tick=time.monotonic(); predictions=[]; row_order=[]
            def collect(module, inputs, output):
                predictions.append({k:output[k].detach().cpu() for k in ('pi_logits','mu','sigma')})
                row_order.append(inputs[0]['row_ids'].cpu())
            handle=model.register_forward_hook(collect)
            try:
                row['validation']=run_epoch(loader=loaders['validation'], optimizer=None,
                                           epoch_index=max(epoch-1,0), **common)
            finally:
                handle.remove()
            out={k:torch.cat([p[k] for p in predictions]) for k in predictions[0]}
            ids=torch.cat(row_order).numpy()
            np.testing.assert_array_equal(np.sort(ids), np.sort(rows))
            row['validation'].update(extra_metrics(out,torch.from_numpy(m.targets[ids]),
                torch.from_numpy(m.target_valid_mask[ids]), loss.logit_epsilon))
            torch.save(dict(row_ids=ids,**out),root/f'validation_epoch_{epoch}.pt')
            row['validation_seconds']=time.monotonic()-tick
            row['audit']=stage_audit(model,audit_batch,loss,None,seed)
            model.zero_grad(set_to_none=True)
            out=model(audit_batch,grl_strength=0.)
            objective=config['loss']['biology_weight']*loss(out['pi_logits'],out['mu'],out['sigma'],
                audit_batch['targets'],audit_batch['target_valid_mask']).total
            objective.backward()
            row['audit']['biology_gradients']=gradient_stats(model)
            model.zero_grad(set_to_none=True)
        row['elapsed_seconds']=time.monotonic()-started
        history.append(row)
        save_json(root/'history.json',history)
        if epoch:
            if args.execution_path != 'spatial':
                save_checkpoint(root/f'epoch_{epoch}.pt',model,optimizer,epoch-1,config,bundle,
                                min(h['validation']['cd8_loss'] for h in history[1:]))
            else:
                torch.save(dict(format='diagnostic-spatial-v1',
                    training_data_contract=settings['training_data_contract'], model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),epoch=epoch,settings=settings,
                    split_indices=bundle.split_indices),root/f'epoch_{epoch}.pt')
        print(json.dumps({k:v for k,v in row.items() if k!='audit'}),flush=True)
    for dataset in datasets.values():
        dataset.close()
    assert all(file_hash(p)==h for p,h in source_hashes.items())
    final = history[-1]['validation']
    progression = dict(
        cd8_improves_constant=final['cd8_loss'] <= baseline['cd8'] - t['early_stopping_min_delta'],
        biology_below_constant=final['biology_loss'] < baseline['biology'],
        positive_cd8_r2=final['positive_cd8_r2'] is not None and final['positive_cd8_r2'] > 0)
    save_json(root/'decision.json', dict(passed=all(progression.values()), criteria=progression,
        min_delta=t['early_stopping_min_delta'], epoch=args.epoch_limit))
    save_json(root/'completion.json' ,dict(completed_epochs=args.epoch_limit, source_files_unchanged=True,
        total_seconds=time.monotonic()-started, updates=args.epoch_limit*len(loaders['train'])))


if __name__ == '__main__':
    main()

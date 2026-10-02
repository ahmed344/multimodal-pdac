"""Training-only, time-capped fit diagnosis; never writes to the source run.

Run: python -m dann.verification.fit_diagnosis --output <new-directory>
Four 8x8 diagnostic cores retain the architecture's full six-pixel halo.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import h5py
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from dann.config import seed_everything
from dann.data_loader import load_anndata_metadata, SparseAnnDataDataset, sparse_collate
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.model_components.deep_sets import DeepSetsEncoder
from dann.tiles import SpatialTileDataset, read_grid, tile_collate, row_identity
from dann.train import grl_strength, move_batch_to_device


def save_json(path: Path, value: object) -> None:
    """Write portable numeric evidence, failing on nonfinite values."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def constant_parameters(targets: torch.Tensor, mask: torch.Tensor,
                        epsilon: float, sigma_min: float) -> dict[str, torch.Tensor]:
    """Fit independent marginal MLEs using only supplied valid training labels."""
    params = {k: torch.zeros_like(targets) for k in ('pi_logits', 'mu', 'sigma')}
    for j in range(targets.shape[1]):
        y = targets[mask[:, j], j]
        if not len(y):
            raise ValueError('Constant baseline requires valid labels for every target.')
        p = (y == 0).float().mean().clamp(1e-7, 1-1e-7)
        z = torch.logit(y[y > 0].clamp(epsilon, 1-epsilon))
        if not len(z):
            raise ValueError('Constant baseline requires positive labels for every target.')
        params['pi_logits'][:, j] = torch.logit(p)
        params['mu'][:, j] = z.mean()
        params['sigma'][:, j] = z.std(unbiased=False).clamp_min(sigma_min)
    return params


def select_rows(keys, train_rows, targets, mask, seed: int, core_size: int = 8):
    """Select two tiles on each of two slides, eight zeros and positives per tile."""
    frame = keys.to_frame(index=False).iloc[train_rows].copy()
    frame['row_id'] = train_rows
    frame['tx'] = frame.iloc[:, 1].to_numpy() // core_size
    frame['ty'] = frame.iloc[:, 2].to_numpy() // core_size
    frame['zero'] = (targets[train_rows, 0] == 0) & mask[train_rows, 0]
    frame['positive'] = (targets[train_rows, 0] > 0) & mask[train_rows, 0]
    counts = frame.groupby([frame.columns[0], 'tx', 'ty'], observed=True)[['zero', 'positive']].sum()
    eligible = counts[(counts.zero >= 8) & (counts.positive >= 8)]
    slides = [s for s, g in eligible.groupby(level=0) if len(g) >= 2]
    rng = np.random.default_rng(seed)
    if len(slides) < 2:
        raise ValueError('Cannot construct the prespecified balanced subset.')
    selected = []
    for slide in rng.choice(sorted(slides), 2, replace=False):
        tiles = eligible.loc[slide].index.tolist()
        for i in rng.choice(len(tiles), 2, replace=False):
            tx, ty = tiles[i]
            group = frame[(frame.iloc[:, 0] == slide) & (frame.tx == tx) & (frame.ty == ty)]
            for column in ('zero', 'positive'):
                selected.extend(rng.choice(group.loc[group[column], 'row_id'], 8, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def metrics(out, batch, loss, baseline=None):
    """Score exact configured weighted loss, CD8 mean loss, and positive correlation."""
    y, m = batch['targets'], batch['target_valid_mask']
    biology = loss(*(out[k] for k in ('pi_logits', 'mu', 'sigma')), y, m)
    cd8loss = ZILNLoss([1.], loss.logit_epsilon, 'mean', loss.include_normal_constant).to(y.device)
    cd8 = cd8loss(*(out[k][:, :1] for k in ('pi_logits', 'mu', 'sigma')), y[:, :1], m[:, :1])
    pos = (y[:, 0] > 0) & m[:, 0]
    mu = out['mu'][pos, 0].detach().double()
    z = torch.logit(y[pos, 0].double().clamp(loss.logit_epsilon, 1-loss.logit_epsilon))
    corr = float(torch.corrcoef(torch.stack([mu, z]))[0, 1]) if mu.std() > 1e-12 else None
    result = dict(biology=float(biology.total.detach()), cd8=float(cd8.total.detach()),
                  positive_cd8_logit_correlation=corr,
                  cd8_parameter_std={k: float(out[k][:, 0].detach().std(unbiased=False))
                                     for k in ('pi_logits', 'mu', 'sigma')})
    if baseline:
        result.update(cd8_improvement=baseline['cd8']-result['cd8'],
                      biology_improvement=baseline['biology']-result['biology'])
        result['passed'] = bool(result['cd8_improvement'] >= .05 and
                                result['biology_improvement'] >= .05 and corr is not None and corr > .8)
    return result


def variation(value):
    """Measure between-row variation relative to overall feature magnitude."""
    x = value.detach().float()
    rms = float(x.square().mean().sqrt())
    centered = float((x-x.mean(0)).square().mean().sqrt())
    return dict(shape=list(x.shape), rms=rms, centered_rms=centered,
                centered_over_rms=centered/max(rms, 1e-30))


def groups(model):
    return {'embedding': list(model.encoder.embedding.parameters()),
            'peak_mlp': list(model.encoder.peak_mlp.parameters()),
            'aggregation': list(model.encoder.aggregation_mlp.parameters()),
            'biology': list(model.biology_predictor.parameters()),
            'discriminator': list(model.batch_discriminator.parameters())}


def gradient_stats(model):
    result = {}
    for name, params in groups(model).items():
        grad = sum(float(p.grad.detach().square().sum()) for p in params if p.grad is not None)**.5
        norm = sum(float(p.detach().square().sum()) for p in params)**.5
        result[name] = dict(gradient_l2=grad, parameter_l2=norm,
                            gradient_over_parameter=grad/max(norm, 1e-30))
    return result


def update_stats(model, before):
    result = {}
    for name, params in groups(model).items():
        delta = sum(float((p.detach()-old).square().sum()) for p, old in zip(params, before[name]))**.5
        norm = sum(float(old.square().sum()) for old in before[name])**.5
        result[name] = dict(update_l2=delta, relative_update=delta/max(norm, 1e-30))
    return result


def altered_batches(batch, seed):
    """Zero amplitudes, remove all peaks, or permute complete spectra over occupied sites."""
    zero = dict(batch, intensities=torch.zeros_like(batch['intensities']))
    empty = dict(batch, peak_indices=batch['peak_indices'][:0], intensities=batch['intensities'][:0],
                 sample_indices=batch['sample_indices'][:0], peak_counts=torch.zeros_like(batch['peak_counts']))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(batch['peak_counts']))
    counts = batch['peak_counts'].cpu().numpy()
    offsets = np.r_[0, np.cumsum(counts)]
    indices = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in permutation])
    take = torch.as_tensor(indices, device=batch['peak_indices'].device)
    shuffled_counts = batch['peak_counts'][torch.as_tensor(permutation, device=take.device)]
    shuffled = dict(batch, peak_indices=batch['peak_indices'][take], intensities=batch['intensities'][take],
                    peak_counts=shuffled_counts,
                    sample_indices=torch.repeat_interleave(torch.arange(len(counts), device=take.device), shuffled_counts))
    return {'zero_amplitudes': zero, 'empty_spectra': empty, 'shuffled_spectra': shuffled}


@torch.no_grad()
def stage_audit(model, batch, loss, baseline, seed):
    model.eval()
    out = model(batch, grl_strength=0.)
    pooled = model.encoder.microbatched(*(batch[k] for k in
        ('peak_indices', 'intensities', 'sample_indices', 'peak_counts')),
        peak_budget=model.peak_budget, checkpointing=False)
    if model.spatial:
        lookup = torch.full((batch['occupancy'].numel(),), -1, device=pooled.device, dtype=torch.long)
        lookup[batch['spatial_positions']] = torch.arange(len(pooled), device=pooled.device)
        pooled = pooled[lookup[batch['core_positions']]]
    result = {'metrics': metrics(out, batch, loss, baseline),
              'variation': {'pooled_core': variation(pooled), **{k: variation(out[k]) for k in
                            ('latent', 'pi_logits', 'mu', 'sigma')}}, 'sensitivity': {}}
    for name, altered in altered_batches(batch, seed).items():
        changed = model(altered, grl_strength=0.)
        result['sensitivity'][name] = {'metrics': metrics(changed, batch, loss, baseline),
            'cd8_rms_change': {k: float((changed[k][:, 0]-out[k][:, 0]).square().mean().sqrt())
                               for k in ('pi_logits', 'mu', 'sigma')}}
    return result


def optimizer_for(model, config, lr):
    t = config['training']
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(t['beta1'], t['beta2']), weight_decay=t['weight_decay'])


def make_model(config, num_batches, state=None):
    seed_everything(config['training']['seed'])
    model = AdversarialLatentFusion.from_config(config, num_batches, 4).cuda()
    if state is not None:
        model.load_state_dict(state)
    return model


def run_fit(name, config, batch, num_batches, lr, adversarial, baseline, root, deadline):
    """Run at most 500 fixed full-subset updates with time reserved for final evaluation."""
    directory = root/name
    directory.mkdir()
    save_json(directory/'settings.json', dict(config=config, learning_rate=lr, adversarial=adversarial,
               max_updates=500, max_seconds=300, full_subset_batch=True))
    start = time.monotonic()
    end = min(deadline, start+300)
    model = make_model(config, num_batches)
    optimizer = optimizer_for(model, config, lr)
    loss = ZILNLoss.from_config(config).cuda()
    counts = torch.bincount(batch['batches'], minlength=num_batches).float()
    weights = torch.zeros_like(counts)
    weights[counts > 0] = 1/counts[counts > 0]
    weights[counts > 0] /= weights[counts > 0].mean()
    history = []
    step_times = []
    best = None
    for step in range(501):
        # Conservative estimate includes final scoring and saving overhead.
        reserve = max(8., 3*max(step_times[-5:], default=2.))
        final = step == 500 or time.monotonic()+reserve >= end
        if step % 10 == 0 or final:
            model.eval()
            with torch.no_grad():
                out = model(batch, grl_strength=0.)
                scored = metrics(out, batch, loss, baseline)
            scored.update(step=step, seconds=time.monotonic()-start)
            history.append(scored)
            save_json(directory/'history.json', history)
            if best is None or scored['biology'] < best['biology']:
                best = dict(scored)
            print(name, json.dumps(scored), flush=True)
            if scored['passed'] or final:
                break
        before_time = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        strength = grl_strength(step/499, config['training']['grl_schedule'],
                               config['training']['grl_max_lambda'], config['training']['grl_gamma']) if adversarial else 0.
        out = model(batch, grl_strength=strength)
        objective = config['loss']['biology_weight']*loss(*(out[k] for k in ('pi_logits', 'mu', 'sigma')),
                                   batch['targets'], batch['target_valid_mask']).total
        if adversarial:
            objective = objective + config['loss']['batch_weight']*F.cross_entropy(
                out['batch_logits'], batch['batches'], weight=weights)
        if not torch.isfinite(objective):
            raise FloatingPointError(f'{name} nonfinite objective at update {step}')
        objective.backward()
        if step in (0, 9, 49, 99, 249, 499):
            save_json(directory/f'gradients_{step+1:03d}.json', gradient_stats(model))
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'])
        before = {k: [p.detach().clone() for p in ps] for k, ps in groups(model).items()} if step == 0 else None
        optimizer.step()
        if before is not None:
            save_json(directory/'first_update.json', update_stats(model, before))
        torch.cuda.synchronize()
        step_times.append(time.monotonic()-before_time)
    model.eval()
    with torch.no_grad():
        out = model(batch, grl_strength=0.)
    np.savez(directory/'predictions.npz', row_ids=batch['row_ids'].cpu().numpy(),
             **{k: out[k].cpu().numpy() for k in ('pi', 'mu', 'sigma')})
    from dann.checkpoints import training_data_contract
    torch.save({'model_state': model.cpu().state_dict(), 'config': config, 'diagnostic_only': True,
                'training_data_contract': training_data_contract(config)}, directory/'last.pt')
    result = dict(name=name, final=history[-1], best_biology=best, updates=step,
                  seconds=time.monotonic()-start, mean_update_seconds=float(np.mean(step_times)) if step_times else 0.)
    save_json(directory/'summary.json', result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('data/PDAC/Results/dann/deep_sets__agg-cnn__bio-cnn__disc-mlp/fit/model/best.pt'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.monotonic()
    source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    from dann.checkpoints import checkpoint_config, validate_training_data_contract
    config = checkpoint_config(checkpoint)
    validate_training_data_contract(checkpoint, config)
    config['training']['core_size'] = 8
    seed = config['training']['seed']
    data = config['data']
    # Verify saved split positions still refer to the same ordered rows and features.
    assert row_identity(Path(data['path']), data) == checkpoint['data_identity']
    metadata = load_anndata_metadata(Path(data['path']), data['target_columns'], data['batch_column'],
                                    data['matrix_key'], data['cd8_normalization'])
    keys, _ = read_grid(Path(data['path']), data)
    selected = select_rows(keys, checkpoint['split_indices']['train'], metadata.targets,
                           metadata.target_valid_mask, seed)
    assert len(selected) == len(np.unique(selected)) == 64
    assert np.isin(selected, checkpoint['split_indices']['train']).all()
    for split in ('validation', 'test'):
        assert not np.isin(selected, checkpoint['split_indices'][split]).any()
    tile_data = SpatialTileDataset(config, selected, metadata)
    assert len(tile_data) == 4
    tiles = [tile_data[i] for i in range(4)]
    cpu = tile_collate(tiles)
    selected = cpu['row_ids'].numpy()
    identities = keys.to_frame(index=False).iloc[selected].copy()
    identities.insert(0, 'row_id', selected)
    identities['context_row_id'] = selected
    for j, target in enumerate(data['target_columns']):
        identities[target+'_raw'] = metadata.raw_targets[selected, j]
        identities[target] = metadata.targets[selected, j]
        identities[target+'_valid'] = metadata.target_valid_mask[selected, j]
    identities.to_csv(root/'selected_rows.csv', index=False)
    # Independently verify coordinate -> grid index -> sparse spectrum and all labels.
    pixel_data = SparseAnnDataDataset(Path(data['path']), selected, metadata, data['matrix_key'],
        data['intensity_transform'], data.get('intensity_clip_max'), data['nonzero_threshold'])
    pixel_cpu = sparse_collate([pixel_data[i] for i in range(64)])
    checked = 0
    discrepancies = []
    context_pixel_samples = []
    for ti, tile in enumerate(tiles):
        pos_to_sample = dict(zip(tile['spatial_positions'], tile['samples']))
        for row, pos in zip(tile['row_ids'], tile['core_positions']):
            label = pixel_data[int(np.flatnonzero(selected == row)[0])]
            context = pos_to_sample[pos]
            same_peaks = np.array_equal(label['peak_indices'], context['peak_indices'])
            same_values = same_peaks and np.array_equal(label['intensities'], context['intensities'])
            discrepancies.append(dict(row_id=int(row), same_peaks=same_peaks, same_intensities=same_values,
                transformed_intensity_rmse=float(np.sqrt(np.mean((label['intensities']-context['intensities'])**2)))
                    if same_peaks and len(label['intensities']) else None))
            context_pixel_samples.append(dict(label, peak_indices=context['peak_indices'],
                                               intensities=context['intensities']))
            # Check the context reader independently against raw CSR disk entries.
            context_row = int(row)
            from dann.data_loader import _matrix_group_path
            with h5py.File(data['path'], 'r') as handle:
                matrix = handle[_matrix_group_path(data['matrix_key'])]
                a, b = matrix['indptr'][context_row:context_row+2]
                peaks = np.asarray(matrix['indices'][a:b], dtype=np.int64)
                values = np.asarray(matrix['data'][a:b], dtype=np.float32)
            if data['nonzero_threshold'] > 0:
                keep = values > data['nonzero_threshold']
                peaks, values = peaks[keep], values[keep]
            if data['intensity_transform'] == 'log1p':
                values = np.log1p(values)
            elif data['intensity_transform'] == 'sqrt':
                values = np.sqrt(values)
            if data.get('intensity_clip_max') is not None:
                values = np.minimum(values, data['intensity_clip_max'])
            np.testing.assert_array_equal(peaks, context['peak_indices'])
            np.testing.assert_array_equal(values, context['intensities'])
            slide, tx, ty = tile_data.tiles[ti][0]
            x = tx*8-tile_data.halo+pos%tile_data.size
            y = ty*8-tile_data.halo+pos//tile_data.size
            assert keys[row] == (slide, x, y)
            checked += 1
    for key in ('targets', 'raw_targets', 'target_valid_mask', 'row_ids', 'batches'):
        torch.testing.assert_close(cpu[key], pixel_cpu[key], equal_nan=True)
    save_json(root/'labeled_vs_context_spectra.json', discrepancies)
    labeled_pixel_cpu = pixel_cpu
    pixel_cpu = sparse_collate(context_pixel_samples)
    save_json(root/'alignment.json', dict(checked_rows=checked, unique_slides=2, occupied_tiles=4,
        core_size=8, halo=tile_data.halo, context_pixels=len(cpu['peak_counts']),
        active_peaks=len(cpu['peak_indices']), all_targets_and_masks_match=True,
        context_reader_matches_raw_csr=True, saved_row_identity_verified=True,
        labeled_context_identical_rows=sum(r['same_intensities'] for r in discrepancies),
        mlp_uses_same_context_spectra_as_cnn=True,
        valid_counts=cpu['target_valid_mask'].sum(0).tolist(),
        cd8_zero_count=int((cpu['targets'][:, 0] == 0).sum()),
        intensity_quantiles=np.quantile(cpu['intensities'].numpy(), [0,.01,.5,.99,1]).tolist()))
    tile_data.close()
    pixel_data.close()
    torch.save({'spatial': cpu, 'pixel': pixel_cpu, 'labeled_pixel': labeled_pixel_cpu}, root/'batches.pt')
    save_json(root/'manifest.json', dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=source_hash,
        checkpoint_epoch=checkpoint['epoch']+1, seed=seed, preparation_seconds=time.monotonic()-started,
        config=config, gpu_budget_seconds=1800, audit_budget_seconds=300,
        fit_budget_seconds=900, followup_budget_seconds=600,
        notes='Only saved training rows supervise. Diagnostic core size 8; complete halo retained.'))
    # The GPU clock begins only after CPU input preparation.
    gpu_start = time.monotonic()
    deadline = gpu_start+1800
    batch = move_batch_to_device(cpu, torch.device('cuda'))
    pixel_batch = move_batch_to_device(pixel_cpu, torch.device('cuda'))
    loss = ZILNLoss.from_config(config).cuda()
    constant = constant_parameters(batch['targets'], batch['target_valid_mask'],
                                   loss.logit_epsilon, config['model']['sigma_min'])
    baseline = metrics(constant, batch, loss)
    save_json(root/'constant_baseline.json', dict(metrics=baseline,
        parameters={k: v[0].tolist() for k,v in constant.items()}, fit_rows=selected.tolist()))
    audit = {}
    for name, state in [('fresh', None), ('fitted', checkpoint['model_state'])]:
        model = make_model(config, len(metadata.batch_names), state)
        audit[name] = stage_audit(model, batch, loss, baseline, seed)
        model.eval()  # isolate biology gradients and deterministic one-step updates
        optimizer = optimizer_for(model, config, .001)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch, grl_strength=0.)
        objective = config['loss']['biology_weight']*loss(*(out[k] for k in ('pi_logits','mu','sigma')),
                                      batch['targets'], batch['target_valid_mask']).total
        objective.backward()
        audit[name]['biology_gradients_before_clip'] = gradient_stats(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'])
        audit[name]['biology_gradients_after_clip'] = gradient_stats(model)
        before = {k: [p.detach().clone() for p in ps] for k, ps in groups(model).items()}
        optimizer.step()
        audit[name]['one_step_update'] = update_stats(model, before)
        del model, optimizer, before, out, objective
        torch.cuda.empty_cache()
        audit['seconds'] = time.monotonic()-gpu_start
        save_json(root/'audit.json', audit)
        if audit['seconds'] > 290:
            raise TimeoutError('Audit exhausted its five-minute allocation.')
    print('AUDIT', json.dumps(audit), flush=True)
    no_dropout = copy.deepcopy(config)
    no_dropout['model']['dropout'] = 0.
    no_dropout['model']['aggregation']['cnn']['dropout'] = 0.
    for head in no_dropout['model']['heads'].values():
        head['cnn']['dropout'] = 0.
    mlp = copy.deepcopy(no_dropout)
    mlp['model']['aggregation']['type'] = 'mlp'
    for head in mlp['model']['heads'].values():
        head['type'] = 'mlp'
    results = []
    for name, cfg, data_batch, lr in [('cnn_lr_1e-3', no_dropout, batch, .001),
        ('cnn_lr_1e-4', no_dropout, batch, .0001), ('mlp_lr_1e-4', mlp, pixel_batch, .0001)]:
        results.append(run_fit(name, cfg, data_batch, len(metadata.batch_names), lr, False,
                               baseline, root, deadline))
    passed = [r for r in results[:2] if r['final']['passed']]
    if passed:
        lr = .001 if passed[0]['name'] == 'cnn_lr_1e-3' else .0001
        results.append(run_fit('cnn_dropout_restored', config, batch, len(metadata.batch_names),
                               lr, False, baseline, root, deadline))
        results.append(run_fit('cnn_adversarial_restored', no_dropout, batch, len(metadata.batch_names),
                               lr, True, baseline, root, deadline))
        followup = 'Dropout and adversarial effects measured separately from fresh initialization.'
    else:
        followup = 'No CNN gate pass; inspect final stage sensitivity without broader fits.'
        follow_start = time.monotonic()
        follow = {}
        for name in ('cnn_lr_1e-3', 'cnn_lr_1e-4', 'mlp_lr_1e-4'):
            saved = torch.load(root/name/'last.pt', map_location='cpu', weights_only=False)
            model = make_model(saved['config'], len(metadata.batch_names), saved['model_state'])
            follow[name] = stage_audit(model, pixel_batch if name.startswith('mlp') else batch, loss, baseline, seed)
            del model
            torch.cuda.empty_cache()
        follow['seconds'] = time.monotonic()-follow_start
        save_json(root/'followup_audit.json', follow)
    final_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    assert final_hash == source_hash
    summary = dict(results=results, followup=followup, gpu_experiment_seconds=time.monotonic()-gpu_start,
                   total_seconds=time.monotonic()-started, source_checkpoint_unchanged=True,
                   gpu=torch.cuda.get_device_name())
    assert summary['gpu_experiment_seconds'] <= 1800
    save_json(root/'summary.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

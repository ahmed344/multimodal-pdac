"""Read-only layer and residual follow-up for a completed fit_diagnosis run."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from dann.losses import ZILNLoss
from dann.train import move_batch_to_device
from dann.verification.fit_diagnosis import make_model, save_json, variation


def profile_layers(model, batch):
    """Record between-core-row variation and reverse-mode biology sensitivity."""
    model.eval()
    records = {}
    handles = []
    for prefix, module in [('aggregation', model.encoder.aggregation_mlp),
                           ('biology', model.biology_predictor)]:
        for name, child in module.named_modules():
            if not isinstance(child, (nn.Conv2d, nn.LayerNorm)):
                continue
            def hook(layer, inputs, output, key=prefix+'.'+name):
                values = output if isinstance(layer, nn.LayerNorm) else output.movedim(1, -1)
                values = values.reshape(-1, values.shape[-1])[batch['core_positions']]
                records[key] = variation(values)
            handles.append(child.register_forward_hook(hook))
    with torch.no_grad():
        model(batch, grl_strength=0.)
    for handle in handles:
        handle.remove()
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    manifest = json.loads((root/'manifest.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    elapsed = summary['gpu_experiment_seconds']
    if elapsed >= 1200:
        raise RuntimeError('Insufficient remaining budget for the follow-up allocation.')
    checkpoint = torch.load(manifest['checkpoint'], map_location='cpu', weights_only=False)
    from dann.checkpoints import validate_training_data_contract
    validate_training_data_contract(checkpoint, checkpoint['config'])
    cached = torch.load(root/'batches.pt', map_location='cpu', weights_only=False)
    torch.set_num_threads(4)
    start = time.monotonic()
    batch = move_batch_to_device(cached['spatial'], torch.device('cuda'))
    pixel = move_batch_to_device(cached['pixel'], torch.device('cuda'))
    labeled = move_batch_to_device(cached['labeled_pixel'], torch.device('cuda'))
    results = {}
    cases = [('fresh', checkpoint['config'], None), ('fitted', checkpoint['config'], checkpoint['model_state'])]
    for name in ('cnn_lr_1e-3', 'cnn_lr_1e-4', 'mlp_lr_1e-4'):
        saved = torch.load(root/name/'last.pt', map_location='cpu', weights_only=False)
        cases.append((name, saved['config'], saved['model_state']))
    residuals = []
    for name, config, state in cases:
        if time.monotonic()-start > 580:
            raise TimeoutError('Layer follow-up exhausted its ten-minute allocation.')
        model = make_model(config, len(checkpoint['batch_names']), state)
        data = batch if model.spatial else pixel
        loss = ZILNLoss.from_config(config).cuda()
        model.eval()
        result = {}
        if model.spatial:
            result['layers'] = profile_layers(model, batch)
        model.zero_grad(set_to_none=True)
        # Capture biology gradients with respect to the actual pooled representation.
        original = model.encoder.microbatched
        pooled = original(*(data[k] for k in ('peak_indices','intensities','sample_indices','peak_counts')),
                           peak_budget=model.peak_budget, checkpointing=False).detach().requires_grad_(True)
        if model.spatial:
            model.encoder.microbatched = lambda *args, **kwargs: pooled
            out = model(data, grl_strength=0.)
        else:
            latent = model.encoder.aggregation_mlp(pooled)
            out = model.biology_predictor(latent)
        objective = config['loss']['biology_weight']*loss(*(out[k] for k in ('pi_logits','mu','sigma')),
                                      data['targets'], data['target_valid_mask']).total
        objective.backward()
        result['gradient_at_pooled_l2'] = float(pooled.grad.norm())
        result['gradient_at_pooled_rms'] = float(pooled.grad.square().mean().sqrt())
        model.encoder.microbatched = original
        mask = ((data['targets'][:, 0] > 0) & data['target_valid_mask'][:, 0]).cpu().numpy()
        values = torch.logit(data['targets'][:, 0].clamp(loss.logit_epsilon, 1-loss.logit_epsilon)).cpu().numpy()
        mu, sigma = (out[k][:, 0].detach().cpu().numpy() for k in ('mu','sigma'))
        errors = values[mask]-mu[mask]
        result['positive_cd8'] = dict(count=int(mask.sum()), logit_rmse=float(np.sqrt(np.mean(errors**2))),
            sigma_quantiles=np.quantile(sigma[mask], [0,.25,.5,.75,1]).tolist(),
            abs_error_sigma_correlation=float(np.corrcoef(np.abs(errors), sigma[mask])[0,1]))
        for i in np.flatnonzero(mask):
            residuals.append(dict(model=name, row_id=int(data['row_ids'][i]), observed_logit=float(values[i]),
                                  mu=float(mu[i]), sigma=float(sigma[i]), residual=float(values[i]-mu[i])))
        if name == 'mlp_lr_1e-4':
            with torch.no_grad():
                other = model(labeled, grl_strength=0.)
                result['labeled_file_input_change'] = dict(
                    cd8_mu_rms_change=float((other['mu'][:,0]-out['mu'][:,0]).square().mean().sqrt()),
                    weighted_biology=float(loss(*(other[k] for k in ('pi_logits','mu','sigma')),
                                         data['targets'], data['target_valid_mask']).total))
        results[name] = result
        del model, pooled, out, objective
        torch.cuda.empty_cache()
    duration = time.monotonic()-start
    results['timing'] = dict(seconds=duration, prior_gpu_experiment_seconds=elapsed,
                            total_gpu_experiment_seconds=elapsed+duration)
    assert duration < 600 and elapsed+duration < 1800
    save_json(root/'layer_followup.json', results)
    pd.DataFrame(residuals).to_csv(root/'positive_cd8_residuals.csv', index=False)
    print(json.dumps(results['timing']), flush=True)


if __name__ == '__main__':
    main()

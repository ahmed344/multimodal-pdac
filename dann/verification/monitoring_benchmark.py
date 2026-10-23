"""Isolated synthetic timing/memory comparison of disabled, five, and all updates.

Uses the production DANN model and run_epoch, including validation monitoring.
Artifacts stay in the requested output directory; no study data/config is changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from dann.config import load_config
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.monitoring import freeze_monitoring
from dann.train import run_epoch


class SyntheticBatches(list):
    """Deterministic in-memory sparse batches with ordinary logical updates."""
    sampler = None


def run_arm(arm: str, device: str, updates: int, epochs: int, output: Path) -> dict:
    """Warm up, reset model/RNG/optimizer, then time complete train/validation epochs."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(157)
    config = load_config(Path('dann/config.yaml'))
    config['model'].update(num_peaks=256, latent_dim=128, dropout=.1)
    config['model']['spectral_encoder']['deep_sets'].update(embedding_dim=64, peak_hidden_dims=[128], peak_output_dim=128)
    for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
        group['type'] = 'mlp'
        group['mlp']['hidden_dims'] = [128]
    config['model']['heads']['he'].update(enabled=True, output_dim=64)
    config['monitoring'] = dict(enabled=arm != 'disabled', gradient_updates_per_epoch=None if arm == 'all' else 5)
    batches = SyntheticBatches()
    rows, peaks = 128, 16
    for index in range(updates):
        targets = torch.rand(rows, 4)
        targets[torch.rand(rows, 4) < .35] = 0
        batches.append(dict(peak_indices=torch.randint(256, (rows*peaks,)), intensities=torch.rand(rows*peaks),
                            sample_indices=torch.arange(rows).repeat_interleave(peaks), peak_counts=torch.full((rows,), peaks),
                            targets=targets, target_valid_mask=torch.rand(rows, 4) > .05,
                            batches=torch.arange(rows) % 3, row_ids=torch.arange(rows)+index*rows,
                            he_targets=torch.randn(rows, 64)))
    validation = SyntheticBatches(batches[:4])
    config = freeze_monitoring(config)
    model = AdversarialLatentFusion.from_config(config, 3, 4).to(device)
    loss = ZILNLoss.from_config(config).to(device)
    original = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    kwargs = dict(model=model, ziln_loss=loss, device=torch.device(device), biology_weight=5., batch_weight=.3,
                  class_weights=None, total_epochs=epochs, grl_schedule='constant', grl_max_lambda=.3,
                  grl_gamma=10., target_columns=config['data']['target_columns'],
                  monitoring=config['monitoring'])
    run_epoch(loader=SyntheticBatches(batches[:2]), optimizer=optimizer, epoch_index=0, gradient_clip_norm=1., **kwargs)
    run_epoch(loader=validation, optimizer=None, epoch_index=0, gradient_clip_norm=None, **kwargs)
    model.load_state_dict(original)
    del original
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    torch.manual_seed(239)
    records, seconds, allocated, reserved = [], [], [], []
    for epoch in range(epochs):
        if device == 'cuda':
            torch.cuda.synchronize()
        started = perf_counter()
        training = run_epoch(loader=batches, optimizer=optimizer, epoch_index=epoch, gradient_clip_norm=1., **kwargs)
        if device == 'cuda':
            allocated.append(torch.cuda.max_memory_allocated())
            reserved.append(torch.cuda.max_memory_reserved())
        held_out = run_epoch(loader=validation, optimizer=None, epoch_index=epoch, gradient_clip_norm=None, **kwargs)
        if device == 'cuda':
            torch.cuda.synchronize()
            allocated.append(torch.cuda.max_memory_allocated())
            reserved.append(torch.cuda.max_memory_reserved())
        seconds.append(perf_counter()-started)
        records.append(dict(epoch=epoch+1, **{f'train_{k}': v for k, v in training.items()},
                            **{f'validation_{k}': v for k, v in held_out.items()}))
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        digest.update(value.detach().cpu().numpy().tobytes())
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output/f'{arm}_history.csv', index=False)
    report = dict(monitoring_definitions=config.get('monitoring_definitions'), arm=arm, device=device, hardware=torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
                  model='Deep Sets + MLP aggregation/heads, dropout 0.1; synthetic sparse inputs',
                  parameters=sum(p.numel() for p in model.parameters()), shared_parameters=sum(p.numel() for p in model.encoder.parameters()),
                  updates_per_epoch=updates, training_rows=rows*updates, validation_rows=rows*len(validation),
                  seconds=seconds, median_seconds=float(np.median(seconds)),
                  peak_allocated_mib=max(allocated, default=0)/2**20, peak_reserved_mib=max(reserved, default=0)/2**20,
                  process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  model_sha256=digest.hexdigest())
    (output/f'{arm}.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--arm', choices=('disabled', 'five', 'all'))
    args = parser.parse_args()
    if args.updates < 1 or args.epochs < 1:
        parser.error('updates and epochs must be positive')
    if args.arm:
        run_arm(args.arm, args.device, args.updates, args.epochs, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for arm in ('disabled', 'five', 'all'):
        subprocess.run([sys.executable, '-m', 'dann.verification.monitoring_benchmark', '--arm', arm,
                        '--output', str(args.output), '--device', args.device, '--updates', str(args.updates),
                        '--epochs', str(args.epochs)], check=True)
        reports.append(json.loads((args.output/f'{arm}.json').read_text()))
    reference = reports[0]
    for report in reports:
        report['runtime_overhead_percent'] = 100*(report['median_seconds']/reference['median_seconds']-1)
        report['allocated_overhead_mib'] = report['peak_allocated_mib']-reference['peak_allocated_mib']
        report['rss_overhead_mib'] = report['process_peak_rss_mib']-reference['process_peak_rss_mib']
    result = dict(arms=reports, identical_final_parameters=len({r['model_sha256'] for r in reports}) == 1,
                  scope='Isolated deterministic synthetic benchmark; overhead depends on architecture, effective batch size, and epoch length.')
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

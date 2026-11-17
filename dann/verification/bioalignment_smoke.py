"""One isolated 2,048-row optimizer update with conditional alignment and validation.

Run with: python -m dann.verification.bioalignment_smoke --output NEW_DIRECTORY
The output must be new. This never launches a full training comparison.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
from time import perf_counter

import pandas as pd
import torch

from dann.batch_alignment import NAMES
from dann.checkpoints import checkpoint_config
from dann.conditional_alignment import BiologyConditioner
from dann.config import apply_smoke_overrides, load_config, resolve_execution
from dann.train import train_model


def run(config_path: Path, output: Path) -> dict:
    """Run one capped epoch, assert finite gradients/losses, and record budgets."""
    output.mkdir(parents=True, exist_ok=False)
    config = apply_smoke_overrides(load_config(config_path))
    if config['execution']['mode'] != 'pixel' or config['training']['batch_size'] != 2048:
        raise ValueError('This check requires the configured pixel optimizer batch of 2048.')
    config['data'].update(max_train_samples=2048, max_validation_samples=512, max_test_samples=0)
    config['training'].update(epochs=1, resume_checkpoint=None)
    config['results'].update(root=str(output.resolve()), run_name='bioalign_a0p1_n128')
    config['post_training']['enabled'] = False
    resolve_execution(config)
    gradients = []
    def observe(event, model, bundle, resolved, record):
        if event != 'update':
            return
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        finite = bool(grads) and all(torch.isfinite(g).all().item() for g in grads)
        if not finite:
            raise FloatingPointError('Nonfinite or missing gradients in capped conditional smoke.')
        norm = torch.linalg.vector_norm(torch.stack([g.double().norm() for g in grads])).item()
        gradients.append(dict(update=record['optimizer_updates'], rows=record['cumulative_supervised_rows'],
                              finite=finite, norm=norm))
    started = perf_counter()
    checkpoint_path = train_model(config, observer=observe, observer_updates=(1,))
    elapsed = perf_counter() - started
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_config(saved)
    restored = BiologyConditioner.restore(saved['alignment_conditioner'], saved['config'], saved['split_indices']['train'])
    history = pd.read_csv(checkpoint_path.parent / 'history.csv').iloc[-1].to_dict()
    for split in ('train', 'validation'):
        for name in NAMES:
            if not math.isfinite(history[f'{split}_alignment_{name}']):
                raise FloatingPointError(f'Unavailable {split} {name} in capped smoke.')
        if not math.isfinite(history[f'{split}_total_loss']):
            raise FloatingPointError('Nonfinite total loss in capped smoke.')
    if len(gradients) != 1 or gradients[0]['rows'] != 2048:
        raise ValueError('Expected exactly one complete 2048-row optimizer update.')
    peak_host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    peak_vram = max(history.get(f'{s}_peak_vram_gib', 0.) for s in ('train', 'validation'))
    budgets = {k: config['performance'][k] for k in ('host_budget_gib', 'vram_budget_gib')}
    within = peak_host <= budgets['host_budget_gib'] and peak_vram <= budgets['vram_budget_gib']
    report = dict(checkpoint=str(checkpoint_path), optimizer_batch=2048, training_updates=1,
                  elapsed_seconds=elapsed, runtime_budget_seconds=None,
                  peak_host_gib=peak_host, peak_vram_gib=peak_vram, budgets=budgets,
                  within_memory_budgets=within, gradients=gradients,
                  restored_preprocessing_sha256=restored.state['sha256'],
                  conditioning={k: v for k, v in restored.state.items() if k not in ('reference_rows', 'contract')},
                  metrics={k: v for k, v in history.items() if 'alignment' in k or k.endswith(('total_loss', 'seconds', 'peak_vram_gib', 'peak_reserved_vram_gib'))})
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if not within:
        raise MemoryError('Capped conditional smoke exceeded a configured memory budget.')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config, args.output)
    print(json.dumps({k: v for k, v in report.items() if k not in ('metrics', 'conditioning')}, indent=2))


if __name__ == '__main__':
    main()

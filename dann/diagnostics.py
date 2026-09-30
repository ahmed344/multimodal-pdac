"""Held-out density errors and local contrasts, including neighborhood support."""
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd


def write_density_diagnostics(extracted: Mapping[str, np.ndarray], config: Mapping[str, Any],
                              output_dir: Path, split: str) -> None:
    """Report density errors by boundary support and observed adjacent-pixel contrasts."""
    targets = extracted['targets']
    predicted = extracted['density_mean']
    support = extracted.get('support')
    groups = {'all': np.ones(len(targets), dtype=bool)}
    if support is not None:
        groups.update(boundary=support < 1-1e-6, interior=support >= 1-1e-6)
        pd.DataFrame({'row_id': extracted['row_ids'], 'biology_support': support,
                      'latent_support': extracted['latent_support']}).to_csv(
                          output_dir / f'{split}_neighborhood_support.csv', index=False)
    errors = []
    for name, group in groups.items():
        for column, target in enumerate(config['data']['target_columns']):
            valid = group & np.isfinite(targets[:, column])
            difference = predicted[valid, column] - targets[valid, column]
            errors.append({'split': split, 'region': name, 'target': target, 'count': int(valid.sum()),
                'mae': np.mean(np.abs(difference)) if valid.any() else np.nan,
                'rmse': np.sqrt(np.mean(difference**2)) if valid.any() else np.nan,
                'bias': np.mean(difference) if valid.any() else np.nan,
                'zero_brier': np.mean((extracted['pi'][valid, column] - (targets[valid, column] == 0))**2)
                              if valid.any() else np.nan})
    pd.DataFrame(errors).to_csv(output_dir / f'{split}_density_errors.csv', index=False)
    if support is None:
        return
    from dann.tiles import read_grid
    keys, _ = read_grid(Path(config['data']['path']), config['data'])
    keys = keys[extracted['row_ids']]
    lookup = {key: i for i, key in enumerate(keys)}
    pairs = [(i, lookup[neighbor]) for i, (slide, x, y) in enumerate(keys)
             for neighbor in ((slide, x+1, y), (slide, x, y+1)) if neighbor in lookup]
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    contrasts = []
    for column, target in enumerate(config['data']['target_columns']):
        usable = np.isfinite(targets[pairs, column]).all(axis=1)
        selected = pairs[usable]
        observed_delta = targets[selected[:, 1], column] - targets[selected[:, 0], column]
        predicted_delta = predicted[selected[:, 1], column] - predicted[selected[:, 0], column]
        contrasts.append({'split': split, 'target': target, 'adjacent_pairs': len(selected),
            'observed_mean_absolute_contrast': np.mean(abs(observed_delta)) if len(selected) else np.nan,
            'predicted_mean_absolute_contrast': np.mean(abs(predicted_delta)) if len(selected) else np.nan,
            'contrast_mae': np.mean(abs(predicted_delta-observed_delta)) if len(selected) else np.nan})
    pd.DataFrame(contrasts).to_csv(output_dir / f'{split}_spatial_contrasts.csv', index=False)

"""Fresh-run supervised head initialization from selected training marginals."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from dann.config import head_initialization_settings
from dann.he import HETargetReader


def final_projection(head: nn.Module) -> nn.Module:
    """Return the final linear/pointwise projection for all supported heads."""
    return head.output if hasattr(head, 'output') else head.network[-1]


def biology_marginals(targets: np.ndarray, valid: np.ndarray, names: list[str],
                      epsilon: float, min_sigma: float, sigma_min: float) -> tuple[np.ndarray, list[dict]]:
    """Compute ordered hurdle/logit-normal marginals using only supplied valid rows."""
    biases, records = [], []
    for index, name in enumerate(names):
        y = np.asarray(targets[:, index][valid[:, index]], dtype=np.float64)
        positive = y[y > 0]
        fallbacks = []
        if not len(y):
            zero_fraction = .5
            fallbacks.append('no_valid_observations: zero_probability=0.5')
        else:
            zero_fraction = float(np.mean(y == 0))
        probability = float(np.clip(zero_fraction, epsilon, 1-epsilon))
        if len(positive):
            p = np.clip(positive, epsilon, 1-epsilon)
            logits = np.log(p) - np.log1p(-p)
            mu, sigma = float(logits.mean()), max(float(logits.std(ddof=0)), min_sigma)
        else:
            mu, sigma = 0., 1.
            fallbacks.append('no_positive_observations: mu=0, sigma=1')
        scale = sigma - sigma_min
        if scale <= 0:
            raise ValueError('Initialized positive sigma must exceed model.sigma_min.')
        raw_sigma = scale + np.log(-np.expm1(-scale))
        pi_logit = np.log(probability) - np.log1p(-probability)
        biases.extend([pi_logit, mu, raw_sigma])
        records.append(dict(target=name, valid_count=len(y), positive_count=len(positive),
                            zero_count=int(np.sum(y == 0)), zero_fraction=zero_fraction,
                            zero_probability=probability, pi_logit=float(pi_logit), mu=mu,
                            sigma=sigma, raw_sigma=float(raw_sigma), fallbacks=fallbacks))
    return np.asarray(biases, dtype=np.float32), records


def initialize_supervised_heads(model: nn.Module, bundle: Any,
                                config: Mapping[str, Any], chunk_rows: int = 1024) -> dict:
    """Initialize fresh heads; H&E means stream only selected training source rows."""
    settings = head_initialization_settings(config['training'])
    report = dict(settings, fit_split='train')
    if not settings['enabled']:
        return report
    rows = bundle.split_indices['train']
    metadata = bundle.metadata
    bias, report['biology'] = biology_marginals(
        metadata.targets[rows], metadata.target_valid_mask[rows], config['data']['target_columns'],
        config['loss']['logit_epsilon'], settings['min_positive_sigma'], config['model']['sigma_min'])
    projections = [(final_projection(model.biology_predictor), bias)]
    if model.he_predictor is not None:
        reader = HETargetReader(Path(config['data']['path']), config['data'].get('he_features_key', 'hes_features'),
                                metadata.num_observations, model.he_output_dim)
        total = np.zeros(model.he_output_dim, dtype=np.float64)
        for start in range(0, len(rows), chunk_rows):
            total += reader.read(rows[start:start+chunk_rows]).sum(axis=0, dtype=np.float64)
        mean = total / max(len(rows), 1)
        report['he'] = dict(count=len(rows), mean=mean.tolist(),
                            fallbacks=[] if len(rows) else ['no_training_rows: mean=0'])
        projections.append((final_projection(model.he_predictor), mean))
    with torch.no_grad():
        for projection, values in projections:
            projection.weight.mul_(settings['output_weight_scale'])
            projection.bias.copy_(torch.as_tensor(values, device=projection.bias.device,
                                                   dtype=projection.bias.dtype))
    return report

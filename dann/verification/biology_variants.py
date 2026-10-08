"""Diagnostic-only biology adapters; production selectors/checkpoints stay unchanged."""
from __future__ import annotations

import copy
import hashlib
import math
from contextlib import contextmanager
import random
from typing import Iterator, Mapping

import numpy as np
import torch
from torch import nn

from dann.model import AdversarialLatentFusion, BiologyPredictor, transform_biology
from dann.verification.fit_diagnosis import variation

VARIANTS = ('M', 'C', 'P', 'shallow', 'residual', 'pointwise_init',
            'no_norm', 'no_dropout', 'low_lr')
RADII = dict(M=0, C=3, P=0, shallow=1, residual=3, pointwise_init=3,
             no_norm=3, no_dropout=3, low_lr=3)


class DiagnosticFusion(AdversarialLatentFusion):
    """Always use tiled execution, independently of the head's actual radius."""
    def biology_map(self, grid, mask):
        if isinstance(self.biology_predictor, BiologyPredictor):
            raw = self.biology_predictor.network(grid.movedim(1, -1))
            return raw.movedim(-1, 1) * mask
        return self.biology_predictor(grid, mask)

    def forward(self, batch, grl_strength=0.):
        grid = self._latent_map(batch)
        mask = batch['occupancy']
        raw = self._select(self.biology_map(grid, mask), batch)
        logits = self._on_map(self.batch_discriminator,
                             self.gradient_reversal(grid, grl_strength), mask)
        return dict(latent=self._select(grid, batch),
                    **transform_biology(raw, self.num_targets, self.sigma_min),
                    batch_logits=self._select(logits, batch))


class ResidualBiology(nn.Module):
    """Production blocks with normalized same-width residual additions."""
    def __init__(self, head):
        super().__init__()
        for name in ('input', 'convolutions', 'norms', 'activation', 'dropout', 'output'):
            setattr(self, name, getattr(head, name))
        self.radius = head.radius

    def hidden(self, value, norm, mask):
        return self.dropout(self.activation(norm(value.movedim(1, -1)).movedim(-1, 1))) * mask

    def forward(self, value, mask):
        value = self.hidden(self.input(value * mask), self.norms[0], mask)
        for conv, norm in zip(self.convolutions, self.norms[1:]):
            value = (value + self.hidden(conv(value), norm, mask)) / math.sqrt(2)
            value = value * mask
        return self.output(value) * mask


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, types, shapes and exact CPU bytes."""
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def make_variant(config: dict, num_batches: int, seed: int, variant: str,
                 adversarial: bool = False) -> tuple[DiagnosticFusion, dict]:
    """Recreate production C then copy shared/compatible tensors into each variant.

    Shape-changing tensors use standard PyTorch fan-in initialization at seed+1000.
    P and pointwise_init use exactly the same initial pointwise function. Construction
    preserves the caller's random streams; the fit seeds its dropout stream separately.
    """
    if variant not in VARIANTS:
        raise ValueError(variant)
    config = copy.deepcopy(config)
    for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
        group['cnn']['residual'] = False  # Historical plain-CNN controls.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = AdversarialLatentFusion.from_config(config, num_batches,
                                                    len(config['data']['target_columns']))
        baseline_hash = state_hash(model.state_dict())
        model.__class__ = DiagnosticFusion
        model.spatial, model.halo = True, 3
        head = model.biology_predictor
        torch.random.default_generator.manual_seed(seed + 1000)
        if variant == 'M':
            m = config['model']
            replacement = BiologyPredictor(m['latent_dim'], m['heads']['biology']['mlp']['hidden_dims'],
                model.num_targets, model.sigma_min, m['activation'], m['dropout'], m['use_layer_norm'])
            # Copy only semantically corresponding tensors with compatible shapes.
            old_layers = [head.input, *head.convolutions, head.output]
            new_layers = [layer for layer in replacement.network if isinstance(layer, nn.Linear)]
            for old, new in zip(old_layers, new_layers):
                if old.kernel_size == (1, 1) and old.weight.shape[:2] == new.weight.shape:
                    new.weight.data.copy_(old.weight[:, :, 0, 0])
                    new.bias.data.copy_(old.bias)
            model.biology_predictor = replacement
        elif variant in ('P', 'pointwise_init'):
            pointwise = nn.ModuleList()
            for old in head.convolutions:
                layer = nn.Conv2d(old.in_channels, old.out_channels, 1)
                layer.bias.data.copy_(old.bias)
                pointwise.append(layer)
            if variant == 'P':
                head.convolutions = pointwise
            else:
                for old, new in zip(head.convolutions, pointwise):
                    old.weight.data.zero_()
                    old.weight.data[:, :, 1:2, 1:2].copy_(new.weight)
                    old.bias.data.copy_(new.bias)
        elif variant == 'shallow':
            head.convolutions = nn.ModuleList([head.convolutions[0]])
            head.norms = nn.ModuleList(list(head.norms[:2]))
        elif variant == 'residual':
            model.biology_predictor = ResidualBiology(head)
        elif variant == 'no_norm':
            head.norms = nn.ModuleList(nn.Identity() for _ in head.norms)
        elif variant == 'no_dropout':
            head.dropout.p = 0.
        model.biology_predictor.radius = RADII[variant]
        for p in model.batch_discriminator.parameters():
            p.requires_grad_(adversarial)
        metadata = dict(variant=variant, seed=seed, shape_initialization_seed=seed+1000,
            cpu_dropout_seed=seed, cuda_dropout_seed=seed+1000,
            production_initialization_sha256=baseline_hash,
            initial_sha256=state_hash(model.state_dict()),
            shared_sha256=state_hash({k: v for k, v in model.state_dict().items()
                                     if not k.startswith('biology_predictor.')}),
            supplied_halo=3, receptive_radius=RADII[variant],
            receptive_field=2*RADII[variant]+1, adversarial=adversarial,
            biology_learning_rate=.0001 if variant == 'low_lr' else .001,
            shared_learning_rate=.001)
    return model, metadata


def optimizer_for(model: DiagnosticFusion, config: dict, variant: str) -> torch.optim.AdamW:
    """Retain AdamW settings, changing only the specified biology learning rate."""
    t = config['training']
    groups = [dict(params=[p for n, p in model.named_parameters()
                           if p.requires_grad and not n.startswith('biology_predictor.')], lr=.001),
              dict(params=list(model.biology_predictor.parameters()),
                   lr=.0001 if variant == 'low_lr' else .001)]
    return torch.optim.AdamW(groups, betas=(t['beta1'], t['beta2']), weight_decay=t['weight_decay'])


@contextmanager
def preserve_state(model: nn.Module) -> Iterator[None]:
    """Audits preserve RNG streams, module modes and existing parameter gradients."""
    python, numpy = random.getstate(), np.random.get_state()
    modes = {m: m.training for m in model.modules()}
    grads = {p: p.grad for p in model.parameters()}
    with torch.random.fork_rng():
        try:
            yield
        finally:
            random.setstate(python)
            np.random.set_state(numpy)
            for m, mode in modes.items():
                m.training = mode
            for p, grad in grads.items():
                p.grad = grad


def audit(model: DiagnosticFusion, batch: dict[str, torch.Tensor], loss: nn.Module,
          biology_weight: float) -> dict:
    """Training-only block variation, gradients and local input sensitivity, no steps."""
    records, handles, calls = {}, [], {}
    with preserve_state(model):
        model.eval()
        model.zero_grad(set_to_none=True)
        def capture(name):
            def hook(module, inputs, output):
                index = calls.get(name, 0)
                calls[name] = index + 1
                for side, tensor in (('input', inputs[0]), ('output', output)):
                    if tensor.ndim == 4:
                        channels_first = isinstance(module, nn.Conv2d) or (
                            isinstance(module, nn.Dropout) and not isinstance(model.biology_predictor, BiologyPredictor))
                        values = tensor.movedim(1, -1) if channels_first else tensor
                        values = values.reshape(-1, values.shape[-1])[batch['core_positions']]
                    else:
                        values = tensor.reshape(-1, tensor.shape[-1])
                    records[f'{name}/{index}/{side}'] = variation(values)
            return hook
        for name, module in model.biology_predictor.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.LayerNorm, nn.Dropout)):
                handles.append(module.register_forward_hook(capture(name)))
        try:
            out = model(batch, grl_strength=0.)
            objective = biology_weight * loss(*(out[k] for k in ('pi_logits', 'mu', 'sigma')),
                                               batch['targets'], batch['target_valid_mask']).total
            objective.backward()
        finally:
            for handle in handles:
                handle.remove()
        gradients = {name: float(p.grad.norm()) if p.grad is not None else 0.
                     for name, p in model.named_parameters()}
        output_variation = {k: variation(out[k]) for k in ('latent', 'pi', 'mu', 'sigma')}
        # Independent gradients for one selected output at a time avoid cancellation
        # between adjacent cores. These derivatives are with respect to latent features.
        with torch.no_grad():
            grid = model._latent_map(batch).detach()
        grid.requires_grad_(True)
        raw = model._select(model.biology_map(grid, batch['occupancy']), batch)
        predicted = transform_biology(raw, model.num_targets, model.sigma_min)
        sensitivity = {}
        positions = batch['core_positions']
        n, c, h, w = grid.shape
        for key in ('pi', 'mu', 'sigma'):
            center, neighbor, outside, missing = [], [], [], []
            for i in np.linspace(0, len(positions)-1, min(8, len(positions))).astype(int):
                grad = torch.autograd.grad(predicted[key][i, 0], grid, retain_graph=True)[0]
                tile, local = divmod(int(positions[i]), h*w)
                y, x = divmod(local, w)
                center.append(float(grad[tile, :, y, x].norm()))
                local_grad = grad[tile].clone()
                local_grad[:, y, x] = 0
                neighbor.append(float(local_grad.norm()))
                distant = grad.clone()
                distant[tile, :, max(0,y-3):y+4, max(0,x-3):x+4] = 0
                outside.append(float(distant.norm()))
                missing.append(float((grad * (1-batch['occupancy'])).norm()))
            sensitivity[key] = dict(center_l2=float(np.mean(center)), neighbor_l2=float(np.mean(neighbor)),
                                    outside_halo_max=max(outside), missing_tissue_max=max(missing))
        weights = {}
        for name, layer in model.biology_predictor.named_modules():
            if isinstance(layer, nn.Conv2d):
                weight = layer.weight.detach()
                middle = weight.shape[-1] // 2
                surrounding = weight.clone()
                surrounding[:, :, middle, middle] = 0
                weights[name] = dict(center_l2=float(weight[:, :, middle, middle].norm()),
                                     surrounding_l2=float(surrounding.norm()))
        return dict(layers=records, variation=output_variation, gradients=gradients,
                    sensitivity=sensitivity, convolution_weights=weights)

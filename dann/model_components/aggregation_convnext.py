"""Masked ConvNeXt hierarchy with a top-down feature pyramid at pixel resolution."""
from __future__ import annotations

import math
from numbers import Real
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


CONVNEXT_DEFAULTS = dict(channels=[64, 128, 256], depths=[1, 1, 1], kernel_size=7,
                        expansion=4, pyramid_dim=128, layer_scale_init=1e-6, dropout=.1)
CONVNEXT_SEMANTICS = dict(
    version=1, normalization='per_pixel_channel_layernorm',
    downsampling='occupied_mean_2x2_stride2', mask='any_occupied_child',
    upsampling='nearest_repeat2_crop', pyramid='top_down_normalized_sum',
    local_skip='pointwise_spectral_projection', origin='global_grid_stride_aligned',
)


def convnext_settings(settings: Mapping | None = None) -> dict:
    """Validate hierarchy widths, depths, and ConvNeXt block numerics."""
    if settings is not None and not isinstance(settings, Mapping):
        raise ValueError('aggregation.convnext must be a mapping.')
    result = {**CONVNEXT_DEFAULTS, **(settings or {})}
    if result.keys() != CONVNEXT_DEFAULTS.keys():
        raise ValueError('Unknown aggregation.convnext settings.')
    for key in ('channels', 'depths'):
        value = result[key]
        if (not isinstance(value, (list, tuple)) or not value
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in value)):
            raise ValueError(f'ConvNeXt {key} must contain positive integers.')
        result[key] = list(value)
    if len(result['channels']) != len(result['depths']):
        raise ValueError('ConvNeXt channels and depths must have the same length.')
    for key in ('kernel_size', 'expansion', 'pyramid_dim'):
        v = result[key]
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise ValueError(f'ConvNeXt {key} must be a positive integer.')
    if result['kernel_size'] < 3 or result['kernel_size'] % 2 != 1:
        raise ValueError('ConvNeXt kernel_size must be odd and >= 3.')
    for key in ('layer_scale_init', 'dropout'):
        v = result[key]
        if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v < 0:
            raise ValueError(f'Invalid ConvNeXt {key}.')
    if result['dropout'] >= 1:
        raise ValueError('ConvNeXt dropout must be in [0, 1).')
    return result


def convnext_reach(settings: Mapping) -> int:
    """Bound every phase's input support, including pooled cell extent."""
    s = convnext_settings(settings)
    return ((s['kernel_size'] // 2) * sum(d * 2 ** i for i, d in enumerate(s['depths']))
            + 2 ** (len(s['depths']) - 1) - 1)


class ChannelNorm(nn.LayerNorm):
    """Normalize channels independently at each pixel, including tiny grids."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(value.movedim(1, -1)).movedim(-1, 1)


class MaskedConvNeXtBlock(nn.Module):
    """Depthwise spatial mixing and expanded channel MLP with a scaled residual."""

    def __init__(self, channels: int, settings: Mapping) -> None:
        super().__init__()
        k = settings['kernel_size']
        self.depthwise = nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels)
        self.norm = ChannelNorm(channels)
        self.expand = nn.Conv2d(channels, channels * settings['expansion'], 1)
        self.contract = nn.Conv2d(channels * settings['expansion'], channels, 1)
        self.scale = nn.Parameter(torch.full((channels,), settings['layer_scale_init']))
        self.dropout = nn.Dropout(settings['dropout'])

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(self.depthwise(value * mask))
        hidden = self.contract(F.gelu(self.expand(hidden)))
        return (value + self.dropout(hidden * self.scale[None, :, None, None])) * mask


class ConvNeXtAggregation(nn.Module):
    """Export fused local/contextual latents on the original occupied grid.

    Input origins must be aligned to ``alignment`` in global grid coordinates.
    SpatialTileDataset supplies aligned origins and the complete derived halo.
    Missing tissue is zeroed after every spatial block; pooling averages only
    occupied children. The local skip preserves a direct spectral pathway.
    """

    def __init__(self, input_dim: int, output_dim: int, **settings) -> None:
        super().__init__()
        self.settings = convnext_settings(settings)
        s = self.settings
        self.radius = convnext_reach(s)
        self.alignment = 2 ** (len(s['channels']) - 1)
        self.input = nn.Conv2d(input_dim, s['channels'][0], 1)
        self.stages = nn.ModuleList(nn.ModuleList(MaskedConvNeXtBlock(c, s) for _ in range(d))
                                   for c, d in zip(s['channels'], s['depths']))
        self.downsamples = nn.ModuleList(nn.Sequential(ChannelNorm(a), nn.Conv2d(a, b, 1))
                                        for a, b in zip(s['channels'], s['channels'][1:]))
        self.laterals = nn.ModuleList(nn.Conv2d(c, s['pyramid_dim'], 1) for c in s['channels'])
        self.output = nn.Sequential(ChannelNorm(s['pyramid_dim']), nn.GELU(),
                                    nn.Conv2d(s['pyramid_dim'], output_dim, 1))
        self.local = nn.Conv2d(input_dim, output_dim, 1)

    @staticmethod
    def pool(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool valid children only, padding odd trailing dimensions as absent."""
        padding = (0, value.shape[-1] % 2, 0, value.shape[-2] % 2)
        count = F.avg_pool2d(F.pad(mask, padding), 2, 2) * 4
        total = F.avg_pool2d(F.pad(value * mask, padding), 2, 2) * 4
        occupied = (count > 0).to(mask.dtype)
        return total / count.clamp_min(1), occupied

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return masked, full-resolution latents for arbitrary tile sizes."""
        hidden = self.input(value * mask) * mask
        features, masks = [], []
        for i, stage in enumerate(self.stages):
            if i:
                hidden, mask_i = self.pool(hidden, masks[-1])
                hidden = self.downsamples[i - 1](hidden) * mask_i
            else:
                mask_i = mask
            for block in stage:
                hidden = block(hidden, mask_i)
            features.append(self.laterals[i](hidden) * mask_i)
            masks.append(mask_i)
        pyramid = features[-1]
        for i in range(len(features) - 2, -1, -1):
            up = pyramid.repeat_interleave(2, -2).repeat_interleave(2, -1)
            up = up[..., :features[i].shape[-2], :features[i].shape[-1]]
            pyramid = (features[i] + up) / math.sqrt(2) * masks[i]
        return (self.output(pyramid) + self.local(value * mask)) / math.sqrt(2) * mask

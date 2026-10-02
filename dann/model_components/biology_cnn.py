"""Spatial biology component."""
import math

import torch

from .layers import SpatialNetwork


class CNNBiology(SpatialNetwork):
    """Local spatial biology with optional residual blocks and a fixed radius."""

    def __init__(self, input_dim: int, output_dim: int, channels: int = 128,
                 depth: int = 3, dropout: float = 0.10, residual: bool = True) -> None:
        if not isinstance(residual, bool):
            raise ValueError("Biology CNN residual must be boolean.")
        super().__init__(input_dim, output_dim, channels, depth, dropout)
        self.residual = residual

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply masked convolutions with normalized same-width residual additions."""
        if not self.residual:
            return super().forward(value, mask)
        value = self.hidden(self.input(value * mask), self.norms[0], mask)
        for convolution, norm in zip(self.convolutions, self.norms[1:]):
            value = (value + self.hidden(convolution(value), norm, mask)) / math.sqrt(2)
            value = value * mask
        return self.output(value) * mask

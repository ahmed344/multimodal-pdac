"""Pointwise biology component."""
from typing import Any, Sequence
import torch
from torch import nn
from .layers import build_mlp


class MLPBiology(nn.Module):
    """Independent per-pixel biology with the original MLP layer ordering."""
    radius = 0

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int,
                 **common: Any) -> None:
        super().__init__()
        self.network = build_mlp(input_dim, hidden_dims, output_dim, **common)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return raw component outputs without changing spatial neighborhoods."""
        return self.network(latent)

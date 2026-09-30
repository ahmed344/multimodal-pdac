"""Original post-pooling aggregation MLP."""
from typing import Any, Sequence
import torch
from torch import nn
from .layers import build_mlp


class MLPAggregation(nn.Sequential):
    """Independent per-pixel aggregation with the original MLP layer ordering."""
    radius = 0

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int,
                 **common: Any) -> None:
        super().__init__(*build_mlp(input_dim, hidden_dims, output_dim, **common))

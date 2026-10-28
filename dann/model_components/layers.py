"""Shared pointwise and local spatial layers."""
import math
from typing import Sequence
import torch
from torch import nn

def build_activation(name: str) -> nn.Module:
    """Construct an activation module by configuration name.

    Args:
        name (str): Activation name.

    Returns:
        nn.Module: Stateless activation module.
    """

    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation {name!r}; choose from {sorted(activations)}.")
    return activations[name]()


class NormalizedResidual(nn.Sequential):
    """Equal-width hidden update with normalized identity addition."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value + super().forward(value)) / math.sqrt(2)


def without_dropout(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    """Evaluate an MLP reference with gradients, without RNG draws or mode changes."""
    if isinstance(module, nn.Dropout):
        return value
    if isinstance(module, nn.Sequential):
        update = value
        for child in module:
            update = without_dropout(child, update)
        return (value + update) / math.sqrt(2) if isinstance(module, NormalizedResidual) else update
    return module(value)


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str,
    dropout: float,
    use_layer_norm: bool,
    residual: bool = False,
) -> nn.Sequential:
    """Build an MLP with normalized, activated hidden layers.

    Args:
        input_dim (int): Input feature width.
        hidden_dims (Sequence[int]): Hidden layer widths.
        output_dim (int): Linear output width.
        activation (str): Hidden activation name.
        dropout (float): Hidden dropout probability.
        use_layer_norm (bool): Add LayerNorm after each hidden linear layer.
        residual (bool): Normalize identity additions around equal-width hidden blocks.

    Returns:
        nn.Sequential: Configured feed-forward network.
    """

    if not isinstance(residual, bool):
        raise ValueError("MLP residual must be boolean.")
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        start = len(layers)
        layers.append(nn.Linear(previous, int(width)))
        if use_layer_norm:
            layers.append(nn.LayerNorm(int(width)))
        layers.append(build_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        if residual and previous == int(width):
            block = NormalizedResidual(*layers[start:])
            layers[start:] = [block]
        previous = int(width)
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)



class SpatialNetwork(nn.Module):
    """Local convolutions with channel-only normalization and tissue masking."""

    def __init__(self, input_dim: int, output_dim: int, channels: int = 128,
                 depth: int = 3, dropout: float = 0.10, residual: bool = False) -> None:
        super().__init__()
        if not isinstance(residual, bool):
            raise ValueError("CNN residual must be boolean.")
        self.residual = residual
        self.radius = depth
        self.input = nn.Conv2d(input_dim, channels, 1)
        self.convolutions = nn.ModuleList(nn.Conv2d(channels, channels, 3, padding=1)
                                         for _ in range(depth))
        self.norms = nn.ModuleList(nn.LayerNorm(channels) for _ in range(depth + 1))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv2d(channels, output_dim, 1)

    def hidden(self, value, norm, mask):
        value = norm(value.movedim(1, -1)).movedim(-1, 1)
        return self.dropout(self.activation(value)) * mask

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply masked hidden blocks with optional normalized residual additions."""
        value = self.hidden(self.input(value * mask), self.norms[0], mask)
        for convolution, norm in zip(self.convolutions, self.norms[1:]):
            hidden = self.hidden(convolution(value), norm, mask)
            value = ((value + hidden) / math.sqrt(2)) * mask if self.residual else hidden
        return self.output(value) * mask

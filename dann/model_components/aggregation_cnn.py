"""Spatial aggregation component."""
from .layers import SpatialNetwork


class CNNAggregation(SpatialNetwork):
    """Local spatial aggregation with optional residual blocks and a fixed radius."""
    def __init__(self, input_dim: int, output_dim: int, channels: int = 128,
                 depth: int = 3, dropout: float = 0.10, residual: bool = True) -> None:
        if not isinstance(residual, bool):
            raise ValueError("Aggregation CNN residual must be boolean.")
        super().__init__(input_dim, output_dim, channels, depth, dropout, residual)

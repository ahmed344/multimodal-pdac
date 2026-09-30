"""Spatial aggregation component."""
from .layers import SpatialNetwork


class CNNAggregation(SpatialNetwork):
    """Local spatial aggregation with a finite, depth-defined radius."""
    def __init__(self, input_dim: int, output_dim: int, channels: int = 128,
                 depth: int = 3, dropout: float = 0.10) -> None:
        super().__init__(input_dim, output_dim, channels, depth, dropout)

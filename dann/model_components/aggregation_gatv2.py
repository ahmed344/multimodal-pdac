"""Independent GATv2 aggregation component."""
from .layers_gatv2 import GATv2Network


class AggregationGATv2(GATv2Network):
    """Map occupied spatial features to latent_dim channels.

    Channels is the total concatenated width; neighbor_radius is a per-layer
    Euclidean grid cutoff. Residual and both dropout settings are independent
    of the other model components.
    """

    def __init__(self, input_dim: int, output_dim: int, channels: int = 128,
                 depth: int = 3, heads: int = 4, neighbor_radius: float = 1.5,
                 residual: bool = True, dropout: float = 0.10,
                 attention_dropout: float = 0.0) -> None:
        super().__init__(input_dim, output_dim, channels=channels, depth=depth,
                         heads=heads, neighbor_radius=neighbor_radius, residual=residual,
                         dropout=dropout, attention_dropout=attention_dropout)

"""Local occupied-grid graphs and node-wise GATv2 blocks (lazy PyG dependency)."""
from functools import lru_cache
import math
from numbers import Real
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

GATV2_DEFAULTS = dict(channels=128, depth=3, heads=4, neighbor_radius=1.5,
                      residual=True, dropout=0.10, attention_dropout=0.0)
GRAPH_SEMANTICS = dict(
    version=1, neighborhood='euclidean_grid_radius_inclusive',
    nodes='occupied_pixels_including_empty_spectra', edges='all_eligible_directed_endpoints_per_tile',
    edge_features='source_minus_receiver_xy_and_distance_divided_by_radius',
    self_edges='exactly_one_explicit_zero_features', heads='concatenated',
    residual='identity_plus_update_divided_by_sqrt_2', normalization='per_node_layer_norm',
    share_weights=False, negative_slope=0.2, bias=True, internal_residual=False)


def gatv2_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve defaults and validate graph settings without importing PyG."""
    result = {**GATV2_DEFAULTS, **settings}
    if set(result) != set(GATV2_DEFAULTS):
        raise ValueError('Unknown GATv2 settings.')
    for key in ('channels', 'depth', 'heads'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'GATv2 {key} must be a positive integer.')
    if result['channels'] % result['heads']:
        raise ValueError('GATv2 channels must be divisible by heads.')
    if not isinstance(result['residual'], bool):
        raise ValueError('GATv2 residual must be boolean.')
    for key in ('neighbor_radius', 'dropout', 'attention_dropout'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f'GATv2 {key} must be a finite number.')
        if (value < 1 if key == 'neighbor_radius' else not 0 <= value < 1):
            raise ValueError(f'Invalid GATv2 {key}: radius must be >= 1; dropout must be in [0, 1).')
    return result


@lru_cache(maxsize=128)
def grid_offsets(neighbor_radius: float) -> tuple[tuple[int, int], ...]:
    """Return all (dx, dy) integer offsets inside the inclusive Euclidean cutoff."""
    radius = gatv2_settings({'neighbor_radius': neighbor_radius})['neighbor_radius']
    limit = math.floor(radius)
    return tuple((dx, dy) for dy in range(-limit, limit + 1)
                 for dx in range(-limit, limit + 1) if math.hypot(dx, dy) <= radius)


def graph_reach(depth: int, neighbor_radius: float) -> int:
    """Enclosing square reach using exactly the graph's allowed offsets."""
    return depth * max(max(abs(dx), abs(dy)) for dx, dy in grid_offsets(neighbor_radius))


def grid_graph(mask: torch.Tensor, neighbor_radius: float,
               dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather node positions and directed edges without dense adjacency or pairs.

    Mask is [tiles, 1, height, width]. Edges are source to receiver; attributes
    are relative source x/y and distance, normalized by the per-layer radius.
    Only occupancy controls connectivity; each tile has an independent graph.
    """
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError('GATv2 mask must have shape [tiles, 1, height, width].')
    occupied = mask[:, 0].bool()
    positions = occupied.reshape(-1).nonzero().flatten()
    lookup = torch.full(occupied.shape, -1, dtype=torch.long, device=mask.device)
    lookup.reshape(-1)[positions] = torch.arange(positions.numel(), device=mask.device)
    h, w = occupied.shape[-2:]
    edges, attributes = [], []
    for dx, dy in grid_offsets(neighbor_radius):
        if abs(dx) >= w or abs(dy) >= h:
            continue
        x0, x1 = max(0, -dx), min(w, w-dx)
        y0, y1 = max(0, -dy), min(h, h-dy)
        receiver = lookup[:, y0:y1, x0:x1].reshape(-1)
        source = lookup[:, y0+dy:y1+dy, x0+dx:x1+dx].reshape(-1)
        valid = (receiver >= 0) & (source >= 0)
        edge = torch.stack((source[valid], receiver[valid]))
        edges.append(edge)
        attributes.append(torch.tensor([dx/neighbor_radius, dy/neighbor_radius,
                                        math.hypot(dx, dy)/neighbor_radius],
                                       device=mask.device, dtype=dtype).expand(edge.shape[1], 3))
    return positions, torch.cat(edges, dim=1), torch.cat(attributes)


class GATv2Network(nn.Module):
    """Grid adapter with pointwise projections and local normalized attention skips."""

    def __init__(self, input_dim: int, output_dim: int, **settings: Any) -> None:
        super().__init__()
        self.settings = gatv2_settings(settings)
        self.tile_budget = 8  # Execution-only bound on independent graphs per attention call.
        try:
            from torch_geometric.nn import GATv2Conv
        except ImportError as error:
            raise ImportError('GATv2 requires PyTorch Geometric: pip install torch-geometric==2.8.0.post1') from error
        s = self.settings
        self.residual = s['residual']
        self.radius = graph_reach(s['depth'], s['neighbor_radius'])
        self.input = nn.Linear(input_dim, s['channels'])
        self.attentions = nn.ModuleList(GATv2Conv(
            s['channels'], s['channels']//s['heads'], heads=s['heads'], concat=True,
            edge_dim=3, share_weights=False, negative_slope=0.2, bias=True,
            add_self_loops=False, residual=False, dropout=s['attention_dropout'])
            for _ in range(s['depth']))
        self.norms = nn.ModuleList(nn.LayerNorm(s['channels']) for _ in range(s['depth']+1))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(s['dropout'])
        self.output = nn.Linear(s['channels'], output_dim)

    def _attention_update(self, hidden: torch.Tensor, edges: torch.Tensor,
                          features: torch.Tensor, attention: nn.Module,
                          norm: nn.Module) -> torch.Tensor:
        return self.dropout(self.activation(norm(attention(hidden, edges, features))))

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return masked spatial outputs, bounding independent graph work per call.

        Tile grouping retains every node/edge and the full optimizer batch. With
        dropout enabled it changes random draw ordering, not the dropout law.
        """
        if value.shape[0] <= self.tile_budget:
            return self._forward_tiles(value, mask)
        return torch.cat([self._forward_tiles(value[start:start+self.tile_budget],
                                              mask[start:start+self.tile_budget])
                          for start in range(0, value.shape[0], self.tile_budget)], dim=0)

    def _forward_tiles(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Build each tile graph once and reuse it across attention layers."""
        positions, edges, features = grid_graph(mask, self.settings['neighbor_radius'], value.dtype)
        n, _, h, w = value.shape
        nodes = value.movedim(1, -1).reshape(n*h*w, -1)[positions]
        hidden = self.dropout(self.activation(self.norms[0](self.input(nodes))))
        for attention, norm in zip(self.attentions, self.norms[1:]):
            if self.training and torch.is_grad_enabled():
                # Edge-wise attention activations dominate memory on complete tile
                # batches. Recompute them in backward, preserving dropout RNG.
                update = checkpoint(self._attention_update, hidden, edges, features,
                                    attention, norm, use_reentrant=False, preserve_rng_state=True)
            else:
                update = self._attention_update(hidden, edges, features, attention, norm)
            hidden = (hidden + update) / math.sqrt(2) if self.residual else update
        output = self.output(hidden)
        flat = output.new_zeros((n*h*w, output.shape[-1])).index_copy(0, positions, output)
        return flat.reshape(n, h, w, -1).movedim(-1, 1)

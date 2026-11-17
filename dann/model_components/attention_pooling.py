"""Learned-query pooling over sparse spectra, with bounded inference workspace."""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn


ATTENTION_DEFAULTS = dict(num_queries=8, token_dim=128, key_dim=32, value_dim=64)
ATTENTION_SEMANTICS = dict(
    version=1, tokens='peak_embedding_concat_log1p_transformed_intensity',
    scores='scaled_query_key_plus_log_transformed_intensity',
    duplicates='sum_transformed_intensities', zeros='exclude_nonpositive_entries',
    pooling='per_spectrum_per_query_softmax', empty='exact_zero',
    output='concatenate_query_values_then_linear',
)


def attention_settings(settings: Mapping | None = None) -> dict:
    """Validate the dimensions of sparse learned-query attention pooling."""
    if settings is not None and not isinstance(settings, Mapping):
        raise ValueError('deep_sets.attention must be a mapping.')
    result = {**ATTENTION_DEFAULTS, **(settings or {})}
    if result.keys() != ATTENTION_DEFAULTS.keys():
        raise ValueError('Unknown deep_sets.attention settings.')
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in result.values()):
        raise ValueError('Attention dimensions must be positive integers.')
    return result


class SparseAttentionPooling(nn.Module):
    """Pool a variable number of peaks into several learned spectral summaries.

    Intensity is both an explicit token input and a softmax prior. No dense
    pixel-by-vocabulary matrix or pairwise peak attention matrix is constructed.
    Streaming softmax merges complete chunk statistics during inference.
    """

    def __init__(self, output_dim: int, settings: Mapping) -> None:
        super().__init__()
        self.settings = attention_settings(settings)
        s = self.settings
        self.queries = nn.Parameter(torch.empty(s['num_queries'], s['key_dim']))
        nn.init.normal_(self.queries, std=s['key_dim'] ** -.5)
        self.keys = nn.Linear(s['token_dim'], s['key_dim'], bias=False)
        self.values = nn.Linear(s['token_dim'], s['value_dim'])
        self.output = nn.Linear(s['num_queries'] * s['value_dim'], output_dim)

    def forward(self, peak_indices: torch.Tensor, intensities: torch.Tensor,
                sample_indices: torch.Tensor, peak_counts: torch.Tensor,
                embedding: nn.Embedding, peak_mlp: nn.Module,
                chunk_size: int) -> torch.Tensor:
        """Return ordered pixel features; duplicate entries denote one peak."""
        rows = peak_counts.numel()
        matrix = torch.sparse_coo_tensor(
            torch.stack((sample_indices, peak_indices)), intensities,
            (rows, embedding.num_embeddings)).coalesce()
        positive = matrix.values() > 0
        samples, peaks = matrix.indices()[:, positive]
        intensity = matrix.values()[positive]
        queries, width = self.settings['num_queries'], self.settings['value_dim']
        maximum = intensity.new_full((rows, queries), -torch.inf)
        denominator = intensity.new_zeros((rows, queries))
        numerator = intensity.new_zeros((rows, queries, width))
        # Include one empty chunk to keep the zero-spectrum path differentiable.
        for start in range(0, max(len(peaks), 1), chunk_size):
            p, r, x = peaks[start:start + chunk_size], samples[start:start + chunk_size], intensity[start:start + chunk_size]
            token = peak_mlp(torch.cat((embedding(p), x.log1p().unsqueeze(-1)), dim=1))
            score = self.keys(token) @ self.queries.T / math.sqrt(self.settings['key_dim'])
            score = score + x.log().unsqueeze(-1)
            chunk_max = score.new_full((rows, queries), -torch.inf)
            chunk_max.scatter_reduce_(0, r[:, None].expand(-1, queries), score.detach(),
                                      reduce='amax', include_self=True)
            weight = (score - chunk_max[r]).exp()
            chunk_denominator = weight.new_zeros((rows, queries)).index_add(0, r, weight)
            value = self.values(token)
            # Process queries separately: never materialize [nnz, queries, width].
            chunk_numerator = torch.stack([
                value.new_zeros((rows, width)).index_add(0, r, weight[:, q:q+1] * value)
                for q in range(queries)], dim=1)
            merged_max = torch.maximum(maximum, chunk_max)
            shift = torch.where(torch.isfinite(merged_max), merged_max, 0.)
            old_scale, new_scale = (maximum - shift).exp(), (chunk_max - shift).exp()
            numerator = numerator * old_scale[..., None] + chunk_numerator * new_scale[..., None]
            denominator = denominator * old_scale + chunk_denominator * new_scale
            maximum = merged_max
        pooled = numerator / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)[..., None]
        output = self.output(pooled.flatten(1))
        return torch.where((denominator.sum(1) > 0)[:, None], output, torch.zeros_like(output))

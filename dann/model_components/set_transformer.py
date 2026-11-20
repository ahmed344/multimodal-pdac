"""Set Transformer for unordered sets of detected MSI peaks (Lee et al., 2019)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


# Bound one attention call to 64 MiB of float32 scores, even on the math backend.
MAX_ATTENTION_SCORES = 2 ** 24


def _rff(width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))


class MAB(nn.Module):
    """Multihead attention block with paper scaling and two post-residual norms."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.scale = width ** -0.5
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.attention_norm = nn.LayerNorm(width)
        self.rff = _rff(width)
        self.output_norm = nn.LayerNorm(width)

    def _heads(self, rows: torch.Tensor) -> torch.Tensor:
        return rows.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def forward(self, a: torch.Tensor, b: torch.Tensor, query_mask: torch.Tensor | None = None,
                key_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Attend from each valid row of A to the valid rows of B in the same pixel."""
        q, k, v = self._heads(self.query(a)), self._heads(self.key(b)), self._heads(self.value(b))
        mask = None if key_mask is None else key_mask[:, None, None, :]
        query_limit = max(1, MAX_ATTENTION_SCORES // (a.size(0) * self.heads * b.size(1)))
        # Each slice sees every key/value. Only the query axis can be tiled here.
        if q.size(2) <= query_limit:
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        else:
            pieces = []
            for part in q.split(query_limit, dim=2):
                if torch.is_grad_enabled():
                    # Recompute each slice during backward on backends that save dense scores.
                    piece = checkpoint(F.scaled_dot_product_attention, part, k, v, attn_mask=mask,
                                       scale=self.scale, use_reentrant=False)
                else:
                    piece = F.scaled_dot_product_attention(part, k, v, attn_mask=mask, scale=self.scale)
                pieces.append(piece)
            attended = torch.cat(pieces, dim=2)
        h = self.attention_norm(a + self.output(attended.transpose(1, 2).flatten(2)))
        result = self.output_norm(h + self.rff(h))
        return result if query_mask is None else result.masked_fill(~query_mask.unsqueeze(-1), 0)


class SAB(nn.Module):
    """Bidirectional self-attention over the valid rows of one set."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.mab = MAB(width, heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.mab(x, x, mask, mask)


class ISAB(nn.Module):
    """Induce summaries from peaks, then contextualize each peak with those summaries."""

    def __init__(self, width: int, heads: int, points: int) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(points, width))
        nn.init.xavier_uniform_(self.inducing)
        self.to_inducing = MAB(width, heads)
        self.to_peaks = MAB(width, heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.to_inducing(self.inducing.expand(x.size(0), -1, -1), x, key_mask=mask)
        return self.to_peaks(x, h, query_mask=mask)


class SetTransformerEncoder(nn.Module):
    """Adapt sparse MSI rows to sets, then flatten the decoded seeds in learned seed order."""

    def __init__(self, num_peaks: int, *, block_type: str = "sab", embedding_dim: int = 8, d_model: int = 32,
                 num_blocks: int = 2, num_heads: int = 4, num_seeds: int = 4, num_inducing_points: int = 16,
                 peak_budget: int = 65536) -> None:
        super().__init__()
        if block_type not in ("sab", "isab"):
            raise ValueError("Set Transformer block_type must be sab or isab.")
        self.num_peaks = num_peaks
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_seeds = num_seeds
        self.block_type = block_type
        self.num_inducing_points = num_inducing_points if block_type == 'isab' else 0
        self.output_dim = num_seeds * d_model
        self.peak_budget = peak_budget
        self.embedding = nn.Embedding(num_peaks, embedding_dim)
        self.input_projection = nn.Linear(embedding_dim + 1, d_model)
        self.blocks = nn.ModuleList(
            SAB(d_model, num_heads) if block_type == 'sab' else ISAB(d_model, num_heads, num_inducing_points)
            for _ in range(num_blocks))
        self.rff_in = _rff(d_model)
        self.seeds = nn.Parameter(torch.empty(num_seeds, d_model))
        nn.init.xavier_uniform_(self.seeds)
        self.pma = MAB(d_model, num_heads)
        self.decoder_sab = SAB(d_model, num_heads)
        self.rff_out = _rff(d_model)

    def _encode_group(self, ids: torch.Tensor, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # MSI adapter: the loader already transformed intensities. Append them once.
        rows = self.input_projection(torch.cat([self.embedding(ids), values.unsqueeze(-1)], dim=-1))
        for block in self.blocks:
            rows = block(rows, valid)
        summaries = self.pma(self.seeds.expand(rows.size(0), -1, -1), self.rff_in(rows), key_mask=valid)
        return self.rff_out(self.decoder_sab(summaries)).flatten(1)

    def _group_size(self, length: int, peak_budget: int) -> int:
        # The largest single MAB determines peak workspace, including the decoder.
        context = length if self.block_type == 'sab' else self.num_inducing_points
        scores = self.num_heads * max(length * context, self.num_seeds * length, self.num_seeds ** 2)
        return max(1, min(peak_budget // length, MAX_ATTENTION_SCORES // scores))

    def workspace_bytes(self, spectra: int, peak_budget: int, element_size: int) -> int:
        """Bound inference attention workspace for one physical cache batch."""
        peak_rows = min(spectra * self.num_peaks, max(self.num_peaks, peak_budget))
        # Tiny spectra may have more inducing rows or seed rows than peak rows.
        aux_rows = min(spectra, self._group_size(1, peak_budget)) * max(self.num_inducing_points, self.num_seeds)
        token_values = 12 * max(peak_rows, aux_rows) * self.d_model
        context = self.num_peaks if self.block_type == 'sab' else self.num_inducing_points
        max_scores = self.num_heads * max(peak_rows * context, peak_rows * self.num_seeds,
                                          spectra * self.num_seeds ** 2)
        return (token_values + 3 * min(MAX_ATTENTION_SCORES, max_scores)) * element_size

    def forward(self, peak_indices: torch.Tensor, intensities: torch.Tensor,
                sample_indices: torch.Tensor, peak_counts: torch.Tensor) -> torch.Tensor:
        """Return one pooled spectral vector per input pixel, including empty pixels."""
        return self.microbatched(peak_indices, intensities, sample_indices, peak_counts,
                                 peak_budget=self.peak_budget, checkpointing=False)

    def microbatched(self, peak_indices: torch.Tensor, intensities: torch.Tensor,
                     sample_indices: torch.Tensor, peak_counts: torch.Tensor, *,
                     peak_budget: int, checkpointing: bool, chunks=None) -> torch.Tensor:
        """Batch complete coalesced spectra by length and bound their padded attention work."""
        batch_size = peak_counts.numel()
        sparse = torch.sparse_coo_tensor(
            torch.stack([sample_indices, peak_indices]), intensities,
            (batch_size, self.num_peaks), check_invariants=True).coalesce()
        positive = sparse.values() > 0
        members, bins = sparse.indices()[:, positive]
        values = sparse.values()[positive]
        lengths = torch.bincount(members, minlength=batch_size)
        starts = lengths.cumsum(0) - lengths
        # Prepared chunk offsets count raw stored entries. Re-plan using actual positive
        # sets so duplicate/zero entries and arbitrary input ordering are safe.
        counts = lengths.tolist()
        order = sorted((r for r, n in enumerate(counts) if n), key=counts.__getitem__, reverse=True)
        if not order:
            return self.embedding.weight.new_zeros(batch_size, self.output_dim) + self.embedding.weight[0, 0] * 0
        ordered_rows = torch.tensor(order, device=bins.device)
        outputs = []
        first = 0
        while first < len(order):
            width = counts[order[first]]
            rows = ordered_rows[first:first + self._group_size(width, peak_budget)]
            col = torch.arange(width, device=bins.device)
            valid = col < lengths[rows, None]
            offsets = torch.where(valid, starts[rows, None] + col, 0)
            args = bins[offsets], values[offsets], valid
            if checkpointing and self.training and torch.is_grad_enabled():
                pooled = checkpoint(self._encode_group, *args, use_reentrant=False)
            else:
                pooled = self._encode_group(*args)
            outputs.append(pooled)
            first += len(rows)
        pooled = torch.cat(outputs)
        return pooled.new_zeros(batch_size, self.output_dim).index_copy(0, ordered_rows, pooled)

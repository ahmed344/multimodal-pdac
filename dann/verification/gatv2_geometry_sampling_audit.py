"""Audit GATv2 geometry and exact production sampling using CPU metadata only.

Run from the repository root. This script never constructs a model, reads spectra,
loads checkpoints, or changes training settings. Outputs must use a new directory.
"""
from __future__ import annotations

import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS'):
        os.environ[_name] = '1'
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/pdac_gatv2_audit_matplotlib')

import argparse
import copy
import csv
from contextlib import ExitStack
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import shlex
import shutil
import sys
import time

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'dann/sampling.py').is_file())
sys.path.insert(0, str(REPO))

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from anndata.io import read_elem

from dann.config import execution_settings
from dann.data_loader import stratified_split_indices, stratified_cap_indices
from dann.model_components.layers_gatv2 import grid_offsets, graph_reach, grid_graph
from dann.sampling import ProportionalSlideTileSampler, SamplingEpochAudit
from dann.tiles import SpatialTileDataset

SETTINGS = [(r, d) for r in (1.5, 2.5) for d in (2, 3)]
CORE_SIZES = (8, 16, 32)
DEFAULT_RUN = REPO / 'data/PDAC/Results/dann/deep_sets__agg-gatv2__bio-gatv2__disc-gatv2__he-gatv2/core32_attention0_radius15/model'


def sha256(path: Path) -> str:
    """Hash a small provenance file."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def memory_gib() -> float:
    """Return peak process RSS on Linux and enforce the audit memory target."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    if value > 8:
        raise MemoryError(f'Audit exceeded 8 GiB working-memory target: {value:.2f} GiB')
    return value


def graph_counts(masks: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Count occupied nodes and directed edges per independent tile, including self."""
    masks = masks.astype(bool, copy=False)
    n, height, width = masks.shape
    nodes = masks.sum(axis=(1, 2), dtype=np.int64)
    edges = np.zeros(n, dtype=np.int64)
    for dx, dy in grid_offsets(radius):
        if abs(dx) >= width or abs(dy) >= height:
            continue
        x0, x1 = max(0, -dx), min(width, width - dx)
        y0, y1 = max(0, -dy), min(height, height - dy)
        edges += (masks[:, y0:y1, x0:x1]
                  & masks[:, y0+dy:y1+dy, x0+dx:x1+dx]).sum(axis=(1, 2), dtype=np.int64)
    return nodes, edges


def reachable(radius: float, layers: int) -> set[tuple[int, int]]:
    """Enumerate exact possible support on an infinitely occupied integer grid."""
    sites = {(0, 0)}
    for _ in range(layers):
        sites = {(x+dx, y+dy) for x, y in sites for dx, dy in grid_offsets(radius)}
    return sites


def check_numerics() -> dict:
    """Validate counting independently against production graphs and closed forms."""
    rng = np.random.default_rng(90210)
    checks = 0
    for radius in (1.5, 2.5):
        for height, width in ((1, 1), (1, 5), (2, 3), (9, 9), (16, 20)):
            masks = np.stack([np.ones((height, width), bool),
                              np.zeros((height, width), bool),
                              rng.random((height, width)) < .6])
            nodes, edges = graph_counts(masks, radius)
            for i, mask in enumerate(masks):
                positions, actual_edges, _ = grid_graph(torch.from_numpy(mask)[None, None], radius)
                assert nodes[i] == positions.numel()
                assert edges[i] == actual_edges.shape[1]
                assert int((actual_edges[0] == actual_edges[1]).sum()) == nodes[i]
                checks += 1
            formula = sum(max(0, height-abs(dy))*max(0, width-abs(dx))
                          for dx, dy in grid_offsets(radius))
            assert edges[0] == formula
    assert len(grid_offsets(1.5)) == 9 and len(grid_offsets(2.5)) == 21
    assert [(len(reachable(r, d)), len(reachable(r, 2*d))) for r, d in SETTINGS] == [
        (25, 81), (49, 169), (69, 249), (145, 541)]
    return dict(production_graph_comparisons=checks, closed_form_edge_checks=10,
                exact_support_checks=4, passed=True)


def variant(base: dict, core: int, radius: float, depth: int, budget: int | None = None) -> dict:
    """Modify an in-memory configuration only, keeping every other setting fixed."""
    config = copy.deepcopy(base)
    for group in (config['model']['aggregation'], *config['model']['heads'].values()):
        assert group['type'] == 'gatv2'
        group['gatv2'].update(neighbor_radius=radius, depth=depth)
    config['training']['core_size'] = core
    config['training']['spatial']['core_size'] = core
    if budget is not None:
        config['training']['supervised_rows_per_batch'] = budget
        config['training']['spatial']['supervised_rows_per_batch'] = budget
    config['execution'] = execution_settings(config)
    # No resident cache, graph cache, data loader, or target reader is instantiated.
    config['performance'] = {}
    return config


def tile_costs(tiles: SpatialTileDataset, codes: np.ndarray, names: list[str],
               x: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray, int]:
    """Stream small occupancy-mask batches; never materialize graph adjacency."""
    name_codes = {name: i for i, name in enumerate(names)}
    tile_codes = np.array([name_codes[key[0]] for key, _ in tiles.tiles], dtype=np.int64)
    tile_x = np.array([key[1] for key, _ in tiles.tiles], dtype=np.int64)
    tile_y = np.array([key[2] for key, _ in tiles.tiles], dtype=np.int64)
    output = {s: (np.zeros(len(tiles), np.int64), np.zeros(len(tiles), np.int64)) for s in SETTINGS}
    actual_checks = 0
    for slide_code in range(len(names)):
        rows = np.flatnonzero(codes == slide_code)
        ids = np.flatnonzero(tile_codes == slide_code)
        xmin, xmax = int(x[rows].min()), int(x[rows].max())
        ymin, ymax = int(y[rows].min()), int(y[rows].max())
        pad = tiles.core_size + max(2 * graph_reach(d, r) for r, d in SETTINGS)
        grid = np.zeros((ymax-ymin+1+2*pad, xmax-xmin+1+2*pad), bool)
        grid[y[rows]-ymin+pad, x[rows]-xmin+pad] = True
        for radius, depth in SETTINGS:
            halo = 2 * graph_reach(depth, radius)
            size = tiles.core_size + 2 * halo
            offsets = np.arange(size)
            for start in range(0, len(ids), 256):
                selected = ids[start:start+256]
                xs = tile_x[selected, None]*tiles.core_size-halo-xmin+pad+offsets
                ys = tile_y[selected, None]*tiles.core_size-halo-ymin+pad+offsets
                masks = grid[ys[:, :, None], xs[:, None, :]]
                nn, ee = graph_counts(masks, radius)
                output[(radius, depth)][0][selected] = nn
                output[(radius, depth)][1][selected] = ee
                if start == 0:
                    positions, edges, _ = grid_graph(torch.from_numpy(masks[:1])[:, None], radius)
                    assert positions.numel() == nn[0] and edges.shape[1] == ee[0]
                    # Independently verify construction against production coordinate lookup.
                    tid = int(selected[0])
                    sx = tile_x[tid]*tiles.core_size-halo
                    sy = tile_y[tid]*tiles.core_size-halo
                    reference = np.array([[(names[slide_code], sx+dx, sy+dy) in tiles.lookup
                                           for dx in range(size)] for dy in range(size)])
                    np.testing.assert_array_equal(masks[0], reference)
                    actual_checks += 1
        memory_gib()
    return output, tile_codes, actual_checks


def describe(series: pd.Series, prefix: str) -> dict:
    """Compact comparable statistics, with quantiles over batches."""
    return {f'{prefix}_{key}': float(value) for key, value in dict(
        min=series.min(), p05=series.quantile(.05), mean=series.mean(),
        median=series.median(), p95=series.quantile(.95), max=series.max()).items()}


DEFAULT_PREVIOUS = REPO / 'data/PDAC/Results/dann/gatv2_geometry_sampling_audit_20261021/results'
SOURCE_FILES = ('dann/sampling.py', 'dann/tiles.py', 'dann/data_loader.py',
                'dann/model_components/layers_gatv2.py', 'dann/config.py')
CONFIG_KEYS = ['core_size', 'neighbor_radius', 'depth', 'supervised_rows_per_batch']
RELATIVE_METRICS = {
    'relative_node_work_per_supervised_row': 'nodes_per_supervised_row',
    'relative_attention_work_per_supervised_row': 'attention_edge_layers_per_supervised_row',
    'relative_node_work_per_full_step': 'full_batch_nodes_mean',
    'relative_attention_work_per_full_step': 'full_batch_attention_edge_layers_mean',
}


class CsvStream:
    """Write detailed records immediately, retaining no table-sized Python lists."""

    def __init__(self, handle) -> None:
        self.handle = handle
        self.writer = None
        self.rows = 0

    def write(self, row: dict) -> None:
        """Initialize a fixed schema on the first record and append one row."""
        if self.writer is None:
            self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.rows += 1


def expected_boundaries(rows: int, budget: int) -> np.ndarray:
    """Calculate optimizer boundaries independently of the sampler, including zero."""
    if rows < 1 or budget < 1:
        raise ValueError('Rows and budget must be positive.')
    return np.append(np.arange(0, rows, budget, dtype='<i8'), rows).astype('<i8')


def normalize_comparison(comparison: pd.DataFrame, reference: dict) -> tuple[pd.DataFrame, dict]:
    """Normalize each workload measure to the same explicitly selected configuration."""
    selected = np.ones(len(comparison), dtype=bool)
    for key in CONFIG_KEYS:
        selected &= comparison[key].to_numpy() == reference[key]
    if selected.sum() != 1:
        raise ValueError('Reference must identify exactly one comparison configuration.')
    baseline = comparison.loc[selected].iloc[0]
    result, denominators = comparison.copy(), {}
    for relative, absolute in RELATIVE_METRICS.items():
        denominator = float(baseline[absolute])
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError(f'Invalid reference workload: {absolute}.')
        result[relative] = result[absolute] / denominator
        denominators[absolute] = denominator
        assert float(result.loc[selected, relative].iloc[0]) == 1.0
    return result, denominators


def summarize_slides(counts: np.ndarray, sizes: np.ndarray, names: list[str],
                     training_counts: np.ndarray, core: int, budget: int) -> list[dict]:
    """Include absent slides as zero shares and separate full and partial batches."""
    summaries = []
    for kind, mask in [('full', sizes == budget), ('partial', sizes < budget)]:
        if not mask.any():
            continue
        selected, rows = counts[mask], sizes[mask]
        fractions = selected / rows[:, None]
        for code, name in enumerate(names):
            summaries.append(dict(
                core_size=core, supervised_rows_per_batch=budget, batch_kind=kind,
                slide=name, batches=len(rows), included_batches=int((selected[:, code] > 0).sum()),
                inclusion_frequency=float((selected[:, code] > 0).mean()),
                supervised_rows_total=int(selected[:, code].sum()),
                row_fraction_within_kind=float(selected[:, code].sum() / rows.sum()),
                training_rows=int(training_counts[code]),
                training_fraction=float(training_counts[code] / training_counts.sum()),
                **describe(pd.Series(fractions[:, code]), 'batch_fraction')))
    return summaries


def replay(tiles: SpatialTileDataset, costs: dict, codes: np.ndarray, names: list[str],
           train_mask: np.ndarray, validation_mask: np.ndarray, seed: int, budget: int,
           epochs: int, previous_hashes: dict, streams: dict) -> tuple[list, list, list, list]:
    """Replay one core/budget pair and validate coverage, order, boundaries and repeats."""
    core = tiles.core_size
    sampler = ProportionalSlideTileSampler(tiles, seed, budget)
    nrows = int(train_mask.sum())
    boundaries = expected_boundaries(nrows, budget)
    boundary_hash = hashlib.sha256(boundaries.tobytes()).hexdigest()
    steps = len(boundaries) - 1
    expected_counts = np.bincount(codes[train_mask], minlength=len(names))
    expected_fraction = expected_counts / nrows
    counts_table = np.zeros((steps * epochs, len(names)), np.int64)
    sizes = np.zeros(steps * epochs, np.int64)
    row_tiles = np.full(len(codes), -1, np.int64)
    for tile_id, (_, selected) in enumerate(tiles.tiles):
        row_tiles[[row for row, _ in selected]] = tile_id
    assert (row_tiles[train_mask] >= 0).all()
    records, coverage = [], []
    totals = {(r, d, kind): dict(batches=0, supervised_rows=0, tile_graphs=0,
                                occupied_node_visits=0, directed_edge_visits=0,
                                attention_edge_layer_visits=0)
              for r, d in SETTINGS for kind in ('full', 'partial')}
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        visited = np.zeros(len(codes), np.int32)
        tile_visits = np.zeros(len(tiles), np.int64)
        audit = SamplingEpochAudit(epoch, nrows)
        epoch_counts = np.zeros(len(names), np.int64)
        last_tile, split_boundaries = None, 0
        for step, request in enumerate(sampler.logical_requests()):
            rows = np.asarray(request.row_ids, dtype=np.int64)
            ids = np.asarray(request.tile_ids, dtype=np.int64)
            assert step < steps
            assert len(rows) == boundaries[step + 1] - boundaries[step]
            assert audit.processed_rows == boundaries[step]
            assert train_mask[rows].all() and not validation_mask[rows].any()
            np.testing.assert_array_equal(ids, list(dict.fromkeys(row_tiles[rows].tolist())))
            assert len(ids) == len(np.unique(ids))
            split_boundaries += int(last_tile == int(ids[0]))
            last_tile = int(ids[-1])
            np.add.at(visited, rows, 1)
            tile_visits[ids] += 1
            counts = np.bincount(codes[rows], minlength=len(names))
            epoch_counts += counts
            counts_table[epoch * steps + step] = counts
            sizes[epoch * steps + step] = len(rows)
            fractions = counts / len(rows)
            record = audit.update(dict(row_ids=torch.from_numpy(rows),
                batches=torch.from_numpy(codes[rows]), unique_tiles=len(ids)))
            kind = 'full' if len(rows) == budget else 'partial'
            common = dict(core_size=core, supervised_rows_per_batch=budget,
                          epoch=epoch, step=step, batch_kind=kind)
            batch = dict(**common, **{k: v for k, v in record.items() if k not in common},
                         effective_slides=math.exp(record['slide_entropy']),
                         inverse_simpson_slides=float(1 / np.square(fractions).sum()),
                         slide_total_variation=float(np.abs(fractions - expected_fraction).sum() / 2))
            streams['batch_composition'].write(batch)
            records.append(batch)
            for code, name in enumerate(names):
                streams['batch_slide_counts'].write(dict(
                    **common, slide=name, supervised_rows=int(counts[code]),
                    batch_fraction=float(fractions[code]), included=bool(counts[code]),
                    training_rows=int(expected_counts[code]), training_fraction=float(expected_fraction[code])))
            for radius, depth in SETTINGS:
                nn, ee = costs[(radius, depth)]
                nodes, edges = int(nn[ids].sum()), int(ee[ids].sum())
                work = dict(supervised_rows=len(rows), tile_graphs=len(ids),
                            occupied_node_visits=nodes, directed_edge_visits=edges,
                            attention_edge_layer_visits=edges * depth * 4)
                total = totals[(radius, depth, kind)]
                total['batches'] += 1
                for key, value in work.items():
                    total[key] += value
                streams['batch_workload'].write(dict(
                    **common, neighbor_radius=radius, depth=depth, **work,
                    nodes_per_supervised_row=nodes / len(rows),
                    attention_edge_layers_per_supervised_row=edges * depth * 4 / len(rows)))
        np.testing.assert_array_equal(visited, train_mask.astype(np.int32))
        np.testing.assert_array_equal(epoch_counts, expected_counts)
        assert (tile_visits > 0).all()
        assert int(tile_visits.sum()) == len(tiles) + split_boundaries
        summary = audit.summary()
        assert summary['complete'] and summary['batches'] == steps
        assert summary['boundaries_sha256'] == boundary_hash
        previous = previous_hashes[(core, epoch)]
        assert summary['row_order_sha256'] == previous['row_order_sha256'], (core, budget, epoch)
        coverage.append(dict(core_size=core, supervised_rows_per_batch=budget, **summary,
            exact_duplicate_free_coverage=True, validation_supervised_rows=int(visited[validation_mask].sum()),
            previous_audit_row_order_sha256=previous['row_order_sha256'], previous_row_order_match=True,
            expected_boundaries_sha256=boundary_hash, independent_boundaries_match=True,
            full_batches=nrows // budget, final_partial_rows=nrows % budget,
            unique_training_tiles=len(tiles), tile_visits=int(tile_visits.sum()),
            repeated_tile_visits=split_boundaries, repeated_tile_accounting_passed=True))
        for stream in streams.values():
            stream.handle.flush()
        print(f'Core {core}, budget {budget}, epoch {epoch}: {steps} steps; coverage, '
              f'previous row order, boundaries and repeats passed; RSS {memory_gib():.2f} GiB', flush=True)
    slide_summary = summarize_slides(counts_table, sizes, names, expected_counts, core, budget)
    work_summary = []
    for (radius, depth, kind), total in totals.items():
        if not total['batches']:
            continue
        nn, ee = costs[(radius, depth)]
        work_summary.append(dict(
            core_size=core, supervised_rows_per_batch=budget, neighbor_radius=radius,
            depth=depth, batch_kind=kind, **total,
            nodes_per_supervised_row=total['occupied_node_visits'] / total['supervised_rows'],
            attention_edge_layers_per_supervised_row=total['attention_edge_layer_visits'] / total['supervised_rows'],
            nodes_per_step_mean=total['occupied_node_visits'] / total['batches'],
            edges_per_step_mean=total['directed_edge_visits'] / total['batches'],
            attention_edge_layers_per_step_mean=total['attention_edge_layer_visits'] / total['batches']))
    return records, slide_summary, work_summary, coverage


def render(output: Path, batches: pd.DataFrame, slides: pd.DataFrame,
           comparison: pd.DataFrame, budgets: list[int], reference: dict, epochs: int) -> None:
    """Plot budgets side by side with shared scales and full batches only."""
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})

    def save(fig, name: str) -> None:
        for extension in ('png', 'pdf'):
            fig.savefig(output / f'{name}.{extension}')
        plt.close(fig)

    full = batches[batches.batch_kind == 'full']
    specs = [('distinct_slides', 'Slides per full step', 1),
             ('effective_slides', 'Effective slides: exp(entropy)', 1),
             ('largest_slide_fraction', 'Largest slide contribution (%)', 100),
             ('unique_tiles', 'Tile graphs per full step', 1)]
    fig, axes = plt.subplots(4, len(budgets), figsize=(5 * len(budgets), 13),
                             sharey='row', squeeze=False, constrained_layout=True)
    for col, budget in enumerate(budgets):
        group = full[full.supervised_rows_per_batch == budget]
        for row, (metric, title, scale) in enumerate(specs):
            ax = axes[row, col]
            arrays = [group.loc[group.core_size == core, metric].to_numpy() * scale for core in CORE_SIZES]
            ax.boxplot(arrays, tick_labels=CORE_SIZES, showmeans=True,
                       meanprops={'marker': 'o', 'markerfacecolor': '#d65b32', 'markeredgecolor': '#d65b32'})
            ax.set(xlabel='Core side length (grid pixels)')
            if col == 0:
                ax.set_ylabel(title)
            if row == 0:
                ax.set_title(f'{budget:,} supervised rows')
            ax.grid(axis='y', alpha=.2)
    fig.suptitle(f'Supervised slide composition: {epochs} replayed epochs, full steps only\n'
                 'The four radius/depth variants share each core/budget composition', fontsize=14)
    save(fig, 'slide_mix')

    full_slides = slides[slides.batch_kind == 'full']
    expected = (full_slides.drop_duplicates('slide').set_index('slide').training_fraction
                .sort_values(ascending=False) * 100)
    order = expected.index.tolist()
    labels = [f'{name} ({p:.2f}%)' for name, p in expected.items()]
    fig, axes = plt.subplots(1, len(budgets), figsize=(5 * len(budgets) + 2, 14),
                             sharey=True, squeeze=False, constrained_layout=True)
    for ax, budget in zip(axes[0], budgets):
        group = full_slides[full_slides.supervised_rows_per_batch == budget]
        values = group.pivot(index='slide', columns='core_size', values='inclusion_frequency').loc[order, list(CORE_SIZES)] * 100
        img = ax.imshow(values, aspect='auto', cmap='YlGnBu', vmin=0, vmax=100)
        ax.set_xticks(range(len(CORE_SIZES)), CORE_SIZES)
        ax.set_yticks(range(len(order)), labels, fontsize=8)
        ax.set(title=f'{budget:,} supervised rows', xlabel='Core side length')
        for i in range(len(order)):
            for j in range(len(CORE_SIZES)):
                value = values.iloc[i, j]
                ax.text(j, i, f'{value:.0f}', ha='center', va='center', fontsize=8,
                        color='white' if value > 60 else 'black')
    fig.colorbar(img, ax=list(axes[0]), shrink=.45, label='Full batches containing slide (%)')
    fig.suptitle('Per-slide inclusion in full optimizer batches\nParentheses show overall training row share', fontsize=14)
    save(fig, 'per_slide_inclusion')

    fig, axes = plt.subplots(1, len(budgets), figsize=(5 * len(budgets) + 2, 14),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    positions = np.arange(len(order))
    for ax, budget in zip(axes[0], budgets):
        group = full_slides[full_slides.supervised_rows_per_batch == budget]
        values = group.pivot(index='slide', columns='core_size', values='batch_fraction_mean').loc[order, list(CORE_SIZES)] * 100
        ax.scatter(expected, positions, marker='|', s=150, color='black', label='Training share', zorder=5)
        for shift, core, color in zip((-.2, 0, .2), CORE_SIZES, ('#2674a5', '#e78a28', '#8d65b0')):
            ax.scatter(values[core], positions + shift, s=15, color=color, label=f'Core {core}')
        ax.set_ylim(len(order) - .5, -.5)
        ax.set_yticks(positions, labels, fontsize=8)
        ax.set(title=f'{budget:,} supervised rows', xlabel='Mean share of rows in full batches (%)')
        ax.grid(axis='x', alpha=.2)
    axes[0, -1].legend(loc='lower right')
    fig.suptitle('Per-slide supervised row proportions\nAbsent slides contribute zero to the mean; final partial batches excluded', fontsize=14)
    save(fig, 'per_slide_row_proportions')

    labels = [f'r={r}, depth={d}' for r, d in SETTINGS]
    titles = ['Node visits / supervised row', 'Attention edge-layer visits / supervised row',
              'Node visits / full step', 'Attention edge-layer visits / full step']
    fig, axes = plt.subplots(4, len(budgets), figsize=(5 * len(budgets) + 1, 14),
                             squeeze=False, constrained_layout=True)
    for row, (metric, title) in enumerate(zip(RELATIVE_METRICS, titles)):
        vmax = float(comparison[metric].max())
        for col, budget in enumerate(budgets):
            ax = axes[row, col]
            group = comparison[comparison.supervised_rows_per_batch == budget]
            values = np.array([[group.loc[(group.core_size == core) & (group.neighbor_radius == r)
                               & (group.depth == d), metric].iloc[0] for core in CORE_SIZES] for r, d in SETTINGS])
            img = ax.imshow(values, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='auto')
            ax.set_xticks(range(len(CORE_SIZES)), CORE_SIZES)
            ax.set_yticks(range(len(SETTINGS)), labels if col == 0 else [''] * len(SETTINGS))
            ax.set(xlabel='Core side length', title=f'{budget:,} rows\n{title}')
            for i in range(len(SETTINGS)):
                for j in range(len(CORE_SIZES)):
                    ax.text(j, i, f'{values[i, j]:.2f}×', ha='center', va='center', fontsize=13,
                            color='white' if values[i, j] > vmax * .62 else 'black')
        fig.colorbar(img, ax=list(axes[row]), shrink=.8, label='Relative workload')
    fig.suptitle('Measured forward geometry workload (includes repeated tile processing)\n'
                 f'Common reference = core {reference["core_size"]}, radius {reference["neighbor_radius"]}, '
                 f'depth {reference["depth"]}, {reference["supervised_rows_per_batch"]:,} rows: 1.00×\n'
                 'Per-row totals include partial batches; full-step means exclude them', fontsize=13)
    save(fig, 'configuration_costs')


def build_comparison(geometry: pd.DataFrame, work: pd.DataFrame,
                     batch_summary: pd.DataFrame, epochs: int) -> pd.DataFrame:
    """Reduce streamed workloads to one row per setting, retaining partial-step costs."""
    rows = []
    for (core, radius, depth, budget), group in work.groupby(CONFIG_KEYS):
        full = group[group.batch_kind == 'full'].iloc[0]
        partial = group[group.batch_kind == 'partial']
        g = geometry[(geometry.core_size == core) & (geometry.neighbor_radius == radius)
                     & (geometry.depth == depth)].iloc[0]
        row = g.to_dict()
        nrows = int(group.supervised_rows.sum())
        row.update(supervised_rows_per_batch=int(budget), epochs=epochs, supervised_rows_total=nrows,
                   full_batches_per_epoch=int(full.batches) // epochs,
                   partial_batches_per_epoch=int(partial.batches.sum()) // epochs,
                   final_partial_rows=int(partial.supervised_rows.sum()) // epochs,
                   total_steps_per_epoch=int(group.batches.sum()) // epochs,
                   nodes_per_supervised_row=float(group.occupied_node_visits.sum() / nrows),
                   edges_per_supervised_row=float(group.directed_edge_visits.sum() / nrows),
                   attention_edge_layers_per_supervised_row=float(group.attention_edge_layer_visits.sum() / nrows),
                   full_batch_nodes_mean=float(full.nodes_per_step_mean),
                   full_batch_edges_mean=float(full.edges_per_step_mean),
                   full_batch_attention_edge_layers_mean=float(full.attention_edge_layers_per_step_mean),
                   full_batch_nodes_per_supervised_row=float(full.nodes_per_supervised_row),
                   full_batch_attention_edge_layers_per_supervised_row=float(full.attention_edge_layers_per_supervised_row),
                   partial_batch_nodes_mean=float(partial.nodes_per_step_mean.mean()) if len(partial) else 0.,
                   partial_batch_attention_edge_layers_mean=float(partial.attention_edge_layers_per_step_mean.mean()) if len(partial) else 0.,
                   repeated_tile_processing_factor=float(group.tile_graphs.sum() / (g.training_tiles * epochs)),
                   repeated_tile_node_factor=float(group.occupied_node_visits.sum() / (g.unique_tile_nodes_sum * epochs)))
        mix = batch_summary[(batch_summary.core_size == core) & (batch_summary.supervised_rows_per_batch == budget)
                            & (batch_summary.batch_kind == 'full')].iloc[0]
        for metric in ('distinct_slides', 'effective_slides', 'inverse_simpson_slides',
                       'unique_tiles', 'largest_slide_fraction', 'slide_total_variation'):
            row.update({f'{metric}_{stat}': float(mix[f'{metric}_{stat}']) for stat in ('min', 'mean', 'max')})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['supervised_rows_per_batch', 'neighbor_radius', 'depth', 'core_size'])


def run_audit(args: argparse.Namespace, tests: dict, started: float) -> dict:
    """Run the bounded metadata audit using the previous run's frozen configuration."""
    output = args.output
    previous = json.loads((args.previous_audit / 'manifest.json').read_text())
    assert previous['status'] == 'passed'
    assert 0 < args.epochs <= previous['epochs']
    reference = dict(core_size=args.reference_core, neighbor_radius=args.reference_radius,
                     depth=args.reference_depth, supervised_rows_per_batch=args.reference_budget)
    prior_records = json.loads((args.previous_audit / 'coverage_and_hash_checks.json').read_text())
    previous_hashes = {(v['core_size'], v['epoch']): v for v in prior_records}
    prior_files = [p for p in args.previous_audit.rglob('*') if p.is_file()]
    if (args.previous_audit.parent / 'audit.py').is_file():
        prior_files.append(args.previous_audit.parent / 'audit.py')
    protected_paths = prior_files + [REPO / p for p in SOURCE_FILES] + [
        args.run_dir / 'resolved_config.yaml', args.run_dir / 'data_schema.json']
    protected_hashes = {str(p.resolve()): sha256(p) for p in protected_paths}
    # The live config is not an audit input and may be edited concurrently by the user.
    live_config_before = sha256(REPO / 'dann/config.yaml')
    assert sha256(args.run_dir / 'resolved_config.yaml') == previous['config_sha256']
    base = yaml.safe_load((args.run_dir / 'resolved_config.yaml').read_text())
    assert base == yaml.safe_load((args.previous_audit / 'source_config.yaml').read_text())
    schema = json.loads((args.run_dir / 'data_schema.json').read_text())
    assert base['training']['sampling_strategy'] == 'proportional_slide_tiles'
    assert base['training']['microbatch_rows'] is None
    assert base['model']['heads']['he']['enabled']
    assert len(base['model']['heads']) == 3
    assert all(g['type'] == 'gatv2' for g in (base['model']['aggregation'], *base['model']['heads'].values()))
    shutil.copyfile(args.run_dir / 'resolved_config.yaml', output / 'source_config.yaml')
    shutil.copyfile(args.run_dir / 'data_schema.json', output / 'source_data_schema.json')
    shutil.copyfile(Path(__file__), output / 'audit.py')
    for filename in SOURCE_FILES:
        destination = output / 'source' / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / filename, destination)
    command = [sys.executable, str((output / 'audit.py').resolve()), '--output', 'NEW_RESULTS_DIRECTORY',
               '--run-dir', str(args.run_dir.resolve()), '--previous-audit', str(args.previous_audit.resolve()),
               '--epochs', str(args.epochs), '--batch-budgets', *map(str, args.batch_budgets),
               '--reference-core', str(args.reference_core), '--reference-radius', str(args.reference_radius),
               '--reference-depth', str(args.reference_depth), '--reference-budget', str(args.reference_budget)]
    (output / 'reproduce.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(REPO))
                                         + '\n' + shlex.join(command) + '\n')
    data = base['data']
    with h5py.File(data['path'], 'r') as handle:
        categorical = pd.Series(read_elem(handle['obs'][data['batch_column']])).astype('category')
        names = [str(value) for value in categorical.cat.categories]
        codes = categorical.cat.codes.to_numpy(dtype=np.int64, copy=True)
        x = np.asarray(read_elem(handle['obs'][data['x_column']]))
        y = np.asarray(read_elem(handle['obs'][data['y_column']]))
    assert names == schema['batch_names'] and codes.min() >= 0
    assert len(names) == previous['slides'] == 43 and len(codes) == previous['grid_rows']
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert np.equal(x, np.floor(x)).all() and np.equal(y, np.floor(y)).all()
    x, y = x.astype(np.int64), y.astype(np.int64)
    seed = int(base['training']['seed'])
    assert seed == previous['seed']
    splits = stratified_split_indices(codes, data['train_fraction'], data['validation_fraction'], data['test_fraction'], seed)
    splits = {name: stratified_cap_indices(rows, codes, data.get(f'max_{name}_samples'), seed + offset + 1)
              for offset, (name, rows) in enumerate(splits.items())}
    split_sizes = {name: len(rows) for name, rows in splits.items()}
    assert split_sizes == schema['split_sizes'] == previous['split_sizes']
    train = splits['train']
    assert max(args.batch_budgets) <= len(train), 'Each budget must have at least one full batch.'
    train_mask, validation_mask = np.zeros(len(codes), bool), np.zeros(len(codes), bool)
    train_mask[train], validation_mask[splits['validation']] = True, True
    assert train_mask.sum() == len(train) and not (train_mask & validation_mask).any()
    np.savez_compressed(output / 'splits.npz', **splits)
    plan = pd.DataFrame([dict(supervised_rows_per_batch=b, full_batches_per_epoch=len(train) // b,
                             final_partial_rows=len(train) % b, total_steps_per_epoch=math.ceil(len(train) / b))
                         for b in args.batch_budgets])
    if len(train) == 1528567:
        expected_plan = {2048: (746, 759, 747), 4096: (373, 759, 374), 8192: (186, 4855, 187)}
        for row in plan.itertuples(index=False):
            if row.supervised_rows_per_batch in expected_plan:
                assert tuple(row[1:]) == expected_plan[row.supervised_rows_per_batch]
    plan.to_csv(output / 'batch_plan.csv', index=False)
    geometry_records, batch_records, slide_records, work_records, coverage_records = [], [], [], [], []
    shared_geometry, actual_graph_checks = None, 0
    with ExitStack() as stack:
        streams = {name: CsvStream(stack.enter_context((output / f'{name}.csv').open('w', newline='')))
                   for name in ('batch_composition', 'batch_slide_counts', 'batch_workload', 'tile_geometry')}
        for core in CORE_SIZES:
            print(f'Core {core}: counting geometry once for all budgets; RSS {memory_gib():.2f} GiB', flush=True)
            config = variant(base, core, 1.5, 3)
            tiles = SpatialTileDataset(config, train, input_path=Path(data['path']), geometry=shared_geometry)
            try:
                shared_geometry = tiles.geometry
                costs, tile_codes, checks = tile_costs(tiles, codes, names, x, y)
                actual_graph_checks += checks
                for radius, depth in SETTINGS:
                    reach = graph_reach(depth, radius)
                    halo = 2 * reach
                    execution = execution_settings(variant(base, core, radius, depth))
                    assert execution['halo'] == halo and execution['latent_radius'] == reach
                    nn, ee = costs[(radius, depth)]
                    geometry_records.append(dict(
                        core_size=core, neighbor_radius=radius, depth=depth,
                        direct_neighbors=len(grid_offsets(radius)) - 1, latent_radius=reach,
                        latent_window_side=2 * reach + 1, biology_window_side=4 * reach + 1, halo=halo,
                        input_side=core + 2 * halo, dense_area_to_core_ratio=((core + 2 * halo) / core)**2,
                        maximum_latent_sites=len(reachable(radius, depth)), maximum_biology_sites=len(reachable(radius, 2 * depth)),
                        training_tiles=len(tiles), selected_core_rows_mean=len(train) / len(tiles),
                        occupied_nodes_per_tile_mean=float(nn.mean()), edges_per_tile_mean=float(ee.mean()),
                        unique_tile_nodes_sum=int(nn.sum()), unique_tile_edges_sum=int(ee.sum())))
                    for tid, ((slide, tx, ty), selected) in enumerate(tiles.tiles):
                        streams['tile_geometry'].write(dict(core_size=core, neighbor_radius=radius, depth=depth,
                            tile_id=tid, slide=slide, tile_x=tx, tile_y=ty, selected_core_rows=len(selected),
                            occupied_nodes=int(nn[tid]), directed_edges=int(ee[tid])))
                for budget in args.batch_budgets:
                    batches, slides, work, coverage = replay(tiles, costs, codes, names, train_mask, validation_mask,
                        seed, budget, args.epochs, previous_hashes, streams)
                    batch_records.extend(batches)
                    slide_records.extend(slides)
                    work_records.extend(work)
                    coverage_records.extend(coverage)
                    (output / 'coverage_and_hash_checks.json').write_text(json.dumps(coverage_records, indent=2))
            finally:
                tiles.close()
            del tiles, costs
            gc.collect()
        streamed_rows = {name: stream.rows for name, stream in streams.items()}
    batches, slides, work = pd.DataFrame(batch_records), pd.DataFrame(slide_records), pd.DataFrame(work_records)
    geometry = pd.DataFrame(geometry_records)
    prior_geometry = pd.read_csv(args.previous_audit / 'geometry.csv')
    pd.testing.assert_frame_equal(geometry, prior_geometry, check_dtype=False, rtol=1e-12, atol=1e-9)
    summary = []
    for (core, budget, kind), group in batches.groupby(['core_size', 'supervised_rows_per_batch', 'batch_kind']):
        row = dict(core_size=core, supervised_rows_per_batch=budget, batch_kind=kind, batches=len(group))
        for metric in ('supervised_rows', 'distinct_slides', 'effective_slides', 'inverse_simpson_slides',
                       'unique_tiles', 'largest_slide_fraction', 'slide_total_variation'):
            row.update(describe(group[metric], metric))
        summary.append(row)
    batch_summary = pd.DataFrame(summary)
    comparison, denominators = normalize_comparison(build_comparison(geometry, work, batch_summary, args.epochs), reference)
    assert len(comparison) == len(CORE_SIZES) * len(SETTINGS) * len(args.batch_budgets)
    assert not comparison.duplicated(CONFIG_KEYS).any()
    expected_batches = int(plan.total_steps_per_epoch.sum()) * len(CORE_SIZES) * args.epochs
    assert streamed_rows['batch_composition'] == len(batches) == expected_batches
    assert streamed_rows['batch_slide_counts'] == expected_batches * len(names)
    assert streamed_rows['batch_workload'] == expected_batches * len(SETTINGS)
    receptive = geometry[geometry.core_size == CORE_SIZES[0]][[
        'neighbor_radius', 'depth', 'direct_neighbors', 'latent_radius', 'latent_window_side',
        'biology_window_side', 'halo', 'maximum_latent_sites', 'maximum_biology_sites']]
    footprints = geometry[['core_size', 'neighbor_radius', 'depth', 'halo', 'input_side', 'dense_area_to_core_ratio']]
    for name, frame in [('comparison', comparison), ('geometry', geometry), ('receptive_fields', receptive),
                        ('input_footprints', footprints), ('batch_summary', batch_summary),
                        ('per_slide_summary', slides), ('workload_summary', work)]:
        frame.to_csv(output / f'{name}.csv', index=False)
    render(output, batches, slides, comparison, args.batch_budgets, reference, args.epochs)
    changed_inputs = [p for p, digest in protected_hashes.items() if sha256(Path(p)) != digest]
    assert not changed_inputs, f'Frozen audit inputs changed during execution: {changed_inputs}'
    manifest = dict(
        status='passed', run_dir=str(args.run_dir.resolve()), previous_audit=str(args.previous_audit.resolve()),
        data_path=data['path'], grid_rows=len(codes), slides=len(names), split_sizes=split_sizes, seed=seed,
        epochs=args.epochs, batch_budgets=args.batch_budgets, core_sizes=CORE_SIZES,
        radii=[1.5, 2.5], depths=[2, 3], configurations=len(comparison),
        components=['aggregation', 'biology', 'he', 'discriminator'], latent_dim=base['model']['latent_dim'],
        reference=reference, reference_denominators=denominators, reference_normalization_passed=True,
        numerical_checks=tests, actual_tissue_graph_and_mask_checks=actual_graph_checks,
        geometry_matches_previous=True, geometry_computations=len(geometry),
        geometry_reused_across_budgets=True, replayed_core_budget_combinations=len(CORE_SIZES) * len(args.batch_budgets),
        matched_previous_epoch_row_hashes=len(coverage_records), independent_boundary_checks=len(coverage_records),
        duplicate_free_coverage_checks=len(coverage_records), validation_supervised_rows=0,
        total_replayed_batches=len(batches), streamed_table_rows=streamed_rows,
        cpu_threads=1, cpu_affinity=sorted(os.sched_getaffinity(0)), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        peak_rss_gib=memory_gib(), memory_target_gib=8, elapsed_seconds=time.monotonic() - started,
        source_hashes={p: sha256(REPO / p) for p in SOURCE_FILES},
        sources_match_previous={p: sha256(REPO / p) == previous['source_hashes'][p] for p in SOURCE_FILES},
        script_sha256=sha256(Path(__file__)), config_sha256=sha256(output / 'source_config.yaml'),
        split_hashes={k: hashlib.sha256(v.astype('<i8').tobytes()).hexdigest() for k, v in splits.items()},
        protected_files_sha256=protected_hashes, protected_files_unchanged=True,
        live_config_observation=dict(path=str(REPO / 'dann/config.yaml'), used_as_input=False,
                                     sha256_before=live_config_before,
                                     sha256_after=sha256(REPO / 'dann/config.yaml')),
        artifact_sha256={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()
                         and p.name not in ('manifest.json', 'run.log')},
        definitions={
            'nodes': 'Occupied pixel visits in independent tile footprints, including halos; overlap and split-tile repeats count again.',
            'edges': 'Directed edges within each tile, including one self-edge per occupied node.',
            'attention_edge_layer_visits': 'Edges times depth times four GATv2 components. Four fixed attention heads are not an additional multiplier.',
            'per_supervised_row': 'Total workload / all supervised rows across complete epochs, including final partial batches.',
            'per_full_step': 'Arithmetic mean workload over full optimizer batches only; final partial batches excluded.',
            'relative_workload': 'Every ratio uses its corresponding metric at the one common reference configuration.',
            'effective_slides': 'exp(-sum(p*log(p))); inverse_simpson_slides is additionally 1/sum(p**2).',
            'slide_total_variation': 'Half the L1 distance from overall training slide proportions.',
            'per_slide_inclusion_frequency': 'Fraction of full or partial batches containing at least one supervised row from the slide.',
            'per_slide_batch_fraction_mean': 'Mean within-batch row proportion, counting absence as zero; full and partial batches separated.',
            'maximum_sites': 'Exact reachable sites on an infinitely occupied grid. Missing tissue can reduce support.',
            'repeat_factor': 'Total tile or node visits / one visit to each training tile per epoch.'},
        limitations=[
            'No model training or inference: latent quality and CD8 presence/density-exceedance calibration were not measured.',
            'Forward geometry counts are workload proxies, not measured runtime, FLOPs, backward work, or peak GPU memory.',
            'Geometry uses the same labeled-grid input as the previous production audit; coordinates are grid units.',
            'Validation rows may occur as unsupervised input context but never as supervised rows.',
            'Invalid-target rows remain in sampling; slide row shares are not target-validity or loss-weight shares.',
            'Batch budget is the complete optimizer batch, without microbatch subdivision.',
            'All 90/10 split sizes, seed, slide identities, and concatenated row-order hashes match the prior audit.'])
    return manifest


def main() -> None:
    """Parse the audit matrix and create a new, reproducible result directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--previous-audit', type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-budgets', type=int, nargs='+', default=[2048, 4096, 8192])
    parser.add_argument('--reference-core', type=int, choices=CORE_SIZES, default=8)
    parser.add_argument('--reference-radius', type=float, choices=[1.5, 2.5], default=1.5)
    parser.add_argument('--reference-depth', type=int, choices=[2, 3], default=3)
    parser.add_argument('--reference-budget', type=int, default=2048)
    parser.add_argument('--self-test-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or any(b < 1 for b in args.batch_budgets):
        parser.error('Epochs and budgets must be positive.')
    if len(set(args.batch_budgets)) != len(args.batch_budgets):
        parser.error('Batch budgets must be distinct.')
    if args.reference_budget not in args.batch_budgets:
        parser.error('Reference budget must be included in --batch-budgets.')
    if not args.self_test_only and args.output is None:
        parser.error('--output is required.')
    args.batch_budgets.sort()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    started = time.monotonic()
    tests = check_numerics()
    print('Numerical checks:', tests, flush=True)
    if args.self_test_only:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        manifest = run_audit(args, tests, started)
    except Exception as error:
        (args.output / 'manifest.json').write_text(json.dumps(dict(
            status='failed', error_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic() - started), indent=2))
        raise
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('COMPLETE', json.dumps({k: manifest[k] for k in (
        'status', 'configurations', 'elapsed_seconds', 'peak_rss_gib', 'matched_previous_epoch_row_hashes')}), flush=True)


if __name__ == '__main__':
    main()


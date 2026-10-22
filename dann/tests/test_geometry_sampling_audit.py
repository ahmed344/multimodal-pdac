"""Boundary, repeat-count and reference contracts for the metadata-only audit."""
import hashlib
import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dann.sampling import ProportionalSlideTileSampler
from dann.verification.gatv2_geometry_sampling_audit import (
    CsvStream, RELATIVE_METRICS, SETTINGS, expected_boundaries,
    normalize_comparison, replay, summarize_slides,
)


@pytest.mark.parametrize('budget,full,tail,steps', [
    (2048, 746, 759, 747), (4096, 373, 759, 374), (8192, 186, 4855, 187),
])
def test_study_boundaries(budget, full, tail, steps):
    boundaries = expected_boundaries(1528567, budget)
    assert boundaries[0] == 0 and boundaries[-1] == 1528567
    np.testing.assert_array_equal(np.diff(boundaries), [budget] * full + [tail])
    assert len(boundaries) == steps + 1


def test_divisible_and_smaller_than_budget_boundaries():
    np.testing.assert_array_equal(expected_boundaries(12, 6), [0, 6, 12])
    np.testing.assert_array_equal(expected_boundaries(5, 6), [0, 5])


class TinyTiles(SimpleNamespace):
    def __len__(self):
        return len(self.tiles)


@pytest.mark.parametrize('budget', [3, 6, 12])
def test_replay_coverage_hashes_and_repeated_tile_work(budget):
    # Includes a tile larger than the smallest budget, plus two validation rows.
    tiles = TinyTiles(core_size=8, tiles=[
        (('a', 0, 0), [(r, r) for r in range(5)]),
        (('a', 1, 0), [(r, r) for r in range(5, 9)]),
        (('b', 0, 0), [(r, r) for r in range(9, 11)]),
        (('b', 1, 0), [(r, r) for r in range(11, 14)]),
    ])
    codes = np.array([0] * 9 + [1] * 5 + [0, 1])
    train = np.arange(16) < 14
    full_order = np.asarray(next(iter(ProportionalSlideTileSampler(tiles, 17, 100))).row_ids, dtype='<i8')
    previous = {(8, 0): dict(row_order_sha256=hashlib.sha256(full_order.tobytes()).hexdigest())}
    costs = {setting: (np.array([10, 20, 30, 40]), np.array([30, 60, 90, 120])) for setting in SETTINGS}
    streams = {name: CsvStream(io.StringIO()) for name in ('batch_composition', 'batch_slide_counts', 'batch_workload')}
    batches, slides, work, coverage = replay(tiles, costs, codes, ['a', 'b'], train, ~train,
                                            17, budget, 1, previous, streams)
    row_tiles = np.array([0] * 5 + [1] * 4 + [2] * 2 + [3] * 3)
    expected_visits, expected_nodes = 0, 0
    for start in range(0, 14, budget):
        ids = np.unique(row_tiles[full_order[start:start + budget]])
        expected_visits += len(ids)
        expected_nodes += costs[SETTINGS[0]][0][ids].sum()
    assert coverage[0]['tile_visits'] == expected_visits
    assert coverage[0]['repeated_tile_visits'] == expected_visits - 4
    assert coverage[0]['validation_supervised_rows'] == 0
    assert coverage[0]['previous_row_order_match']
    assert coverage[0]['independent_boundaries_match']
    for radius, depth in SETTINGS:
        selected = [v for v in work if v['neighbor_radius'] == radius and v['depth'] == depth]
        assert sum(v['occupied_node_visits'] for v in selected) == expected_nodes
        assert sum(v['attention_edge_layer_visits'] for v in selected) == expected_nodes * 3 * depth * 4
        assert sum(v['supervised_rows'] for v in selected) == 14
    assert streams['batch_slide_counts'].rows == len(batches) * 2
    assert sum(v['supervised_rows_total'] for v in slides) == 14
    previous[(8, 0)]['row_order_sha256'] = 'incorrect'
    with pytest.raises(AssertionError):
        replay(tiles, costs, codes, ['a', 'b'], train, ~train, 17, budget, 1, previous, streams)


def test_common_reference_distinguishes_efficiency_and_step_work():
    reference = dict(core_size=8, neighbor_radius=1.5, depth=3, supervised_rows_per_batch=2048)
    comparison = pd.DataFrame([
        dict(**reference, nodes_per_supervised_row=10., attention_edge_layers_per_supervised_row=100.,
             full_batch_nodes_mean=20480., full_batch_attention_edge_layers_mean=204800.),
        dict(**{**reference, 'supervised_rows_per_batch': 4096},
             nodes_per_supervised_row=8., attention_edge_layers_per_supervised_row=80.,
             full_batch_nodes_mean=32768., full_batch_attention_edge_layers_mean=327680.),
    ])
    result, _ = normalize_comparison(comparison, reference)
    assert all(result.loc[0, column] == 1. for column in RELATIVE_METRICS)
    assert result.loc[1, 'relative_node_work_per_supervised_row'] == .8
    assert result.loc[1, 'relative_attention_work_per_supervised_row'] == .8
    assert result.loc[1, 'relative_node_work_per_full_step'] == 1.6
    assert result.loc[1, 'relative_attention_work_per_full_step'] == 1.6
    with pytest.raises(ValueError, match='exactly one'):
        normalize_comparison(comparison, {**reference, 'core_size': 16})


def test_absent_slide_shares_are_zero_and_partial_steps_are_separate():
    rows = summarize_slides(np.array([[4, 0], [2, 2], [0, 1]]), np.array([4, 4, 1]),
                            ['a', 'b'], np.array([6, 3]), 8, 4)
    full_b = next(r for r in rows if r['batch_kind'] == 'full' and r['slide'] == 'b')
    partial_b = next(r for r in rows if r['batch_kind'] == 'partial' and r['slide'] == 'b')
    assert full_b['inclusion_frequency'] == .5
    assert full_b['batch_fraction_mean'] == .25
    assert partial_b['inclusion_frequency'] == 1.
    assert partial_b['batch_fraction_mean'] == 1.

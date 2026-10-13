"""Audit Monte Carlo discontinuities between FP32 spatial exports.

Presence is exported as rounded float32 (1 - pi), so its exact original pi may
lie anywhere in the inverse rounding interval. This audit replays the unchanged
row RNG at both interval endpoints, then bounds changed hurdle draws and the
continuous sigmoid effect of mu/sigma perturbations. It never edits exports or
changes the sampler.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dann.spatial_execution import sample_rows


def pi_interval(presence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert round-to-nearest float32 subtraction, retaining both tie endpoints."""
    value = np.asarray(presence, dtype=np.float32)
    previous = np.nextafter(value, np.float32(-np.inf)).astype(np.float64)
    following = np.nextafter(value, np.float32(np.inf)).astype(np.float64)
    value = value.astype(np.float64)
    return np.clip(1-(value+following)/2, 0, 1), np.clip(1-(value+previous)/2, 0, 1)


def audit(left: Path, right: Path, samples: int = 1000, seed: int = 20260719) -> dict:
    """Explain every density exception without relaxing parameter/latent checks."""
    reference, candidate = pq.ParquetFile(left), pq.ParquetFile(right)
    targets = [name[3:] for name in reference.schema_arrow.names if name.startswith('mu_')]
    columns = ['row_position', *[f'{prefix}_{target}' for prefix in
               ('mu', 'sigma', 'prob_of_presence', 'density_mean') for target in targets]]
    report = dict(samples=samples, seed=seed, rows=0, density_values=0, exception_rows=0,
                  exceptions_by_target={t: 0 for t in targets}, max_density_difference=0.,
                  max_possible_hurdle_switches=0, max_continuous_difference_bound=0.,
                  unexplained_rows=[], rtol=5e-5, atol=5e-5)
    for a, b in zip(reference.iter_batches(batch_size=65536, columns=columns),
                    candidate.iter_batches(batch_size=65536, columns=columns), strict=True):
        ids = a.column('row_position').to_numpy()
        np.testing.assert_array_equal(ids, b.column('row_position').to_numpy())
        def parameters(batch):
            return {prefix: np.column_stack([batch.column(f'{prefix}_{t}').to_numpy() for t in targets])
                    for prefix in ('mu', 'sigma', 'prob_of_presence', 'density_mean')}
        aa, bb = parameters(a), parameters(b)
        for key in ('mu', 'sigma', 'prob_of_presence'):
            np.testing.assert_allclose(aa[key], bb[key], rtol=5e-5, atol=5e-5)
        delta = np.abs(aa['density_mean'].astype(np.float64)-bb['density_mean'])
        bad = delta > 5e-5 + 5e-5*np.abs(bb['density_mean'])
        report['rows'] += len(ids)
        report['density_values'] += bad.size
        report['max_density_difference'] = max(report['max_density_difference'], float(delta.max()))
        for j, target in enumerate(targets):
            report['exceptions_by_target'][target] += int(bad[:, j].sum())
        for i in np.flatnonzero(bad.any(axis=1)):
            row = int(ids[i])
            report['exception_rows'] += 1
            intervals = []
            explained = True
            for values in (aa, bb):
                lo, hi = pi_interval(values['prob_of_presence'][i:i+1])
                intervals.append((lo, hi))
                mu, sigma = values['mu'][i:i+1], values['sigma'][i:i+1]
                lower = sample_rows(hi, mu, sigma, row, samples, seed).astype(np.float32)
                upper = sample_rows(lo, mu, sigma, row, samples, seed).astype(np.float32)
                actual = values['density_mean'][i:i+1]
                explained &= bool(np.all(actual >= np.nextafter(lower, np.float32(-np.inf)))
                                  and np.all(actual <= np.nextafter(upper, np.float32(np.inf))))
            lo = np.minimum(intervals[0][0], intervals[1][0]).ravel()
            hi = np.maximum(intervals[0][1], intervals[1][1]).ravel()
            rng = np.random.default_rng([seed, row])
            switches = np.zeros(len(targets), dtype=np.int64)
            absolute_normals = np.zeros(len(targets), dtype=np.float64)
            for start in range(0, samples, 128):
                shape = (len(targets), min(128, samples-start))
                normals = rng.standard_normal(shape)
                uniform = rng.random(shape)
                switches += ((uniform >= lo[:, None]) & (uniform < hi[:, None])).sum(axis=1)
                absolute_normals += np.abs(normals).sum(axis=1)
            continuous = .25*(np.abs(aa['mu'][i].astype(np.float64)-bb['mu'][i])
                              + absolute_normals/samples * np.abs(aa['sigma'][i].astype(np.float64)-bb['sigma'][i]))
            bound = switches/samples + continuous + 2*np.finfo(np.float32).eps
            explained &= bool(np.all(delta[i] <= bound))
            report['max_possible_hurdle_switches'] = max(report['max_possible_hurdle_switches'], int(switches.max()))
            report['max_continuous_difference_bound'] = max(report['max_continuous_difference_bound'], float(continuous.max()))
            if not explained:
                report['unexplained_rows'].append(row)
    report['all_explained'] = not report['unexplained_rows']
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260719)
    args = parser.parse_args(argv)
    result = audit(args.left, args.right, args.samples, args.seed)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['all_explained'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

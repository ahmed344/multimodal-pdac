"""Independent bounded real-data checks for exported peak interpretation artifacts.

Run after ``python -m dann.analyze --peaks-only``. The audit scans four selected
raw peak columns across all labeled rows, and independently evaluates sampled
tissue spectra without using the production aggregation functions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from anndata.io import read_elem
import h5py
import numpy as np
import pandas as pd
from dann.artifact_layout import peak_file
import pyarrow.parquet as pq
from scipy import sparse
from scipy.stats import spearmanr
import torch

from dann.config import load_config
from dann.data_loader import _matrix_group_path
from dann.targets import checkpoint_analysis_config, prepare_observed_targets


def audit(config: dict, output: Path, checkpoint_path: Path) -> dict:
    """Compare exports to independent intensity, correlation, and spatial references."""
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint_analysis_config(config, checkpoint)
    data = config['data']
    annotations = pd.read_csv(peak_file(output, 'peak_annotations.csv'))
    peaks = np.unique([0, len(annotations)-1, int(annotations.lesion_detection_count.idxmax()),
                       int(annotations.lesion_detection_count.idxmin())])
    families = annotations.family.to_numpy()
    family_ids = np.unique(families)[np.unique([0, len(np.unique(families))//2, len(np.unique(families))-1])]
    with h5py.File(data['path'], 'r') as handle:
        lesion = np.asarray(read_elem(handle['obs/Lesion']), dtype=bool)
        slides = np.asarray(read_elem(handle[f"obs/{data['batch_column']}"]), dtype=str)
        raw = np.column_stack([read_elem(handle[f'obs/{column}']) for column in data['target_columns']])
        observed, mask = prepare_observed_targets(raw, data['target_columns'], data.get('cd8_normalization'))
        observed = np.where(mask, observed, np.nan)
        group = handle[_matrix_group_path(data['matrix_key'])]
        pointers = group['indptr'][:]
        # Four columns for reference calculations, never all peaks.
        selected = np.zeros((len(lesion), len(peaks)), dtype=np.float64)
        family_selected = np.zeros((len(lesion), len(family_ids)), dtype=np.float32)
        for start in range(0, len(lesion), 8192):
            end = min(start+8192, len(lesion))
            lo, hi = pointers[start], pointers[end]
            block = sparse.csr_matrix((np.asarray(group['data'][lo:hi], dtype=np.float64),
                group['indices'][lo:hi], pointers[start:end+1]-lo), shape=(end-start, len(annotations)))
            selected[start:end] = block[:, peaks].toarray()
            block.sum_duplicates()
            block.data[block.data <= max(0., data['nonzero_threshold'])] = 0
            values = block.data.astype(np.float32)
            if data['intensity_transform'] != 'none':
                values = getattr(np, data['intensity_transform'])(values)
            if data.get('intensity_clip_max') is not None:
                values = np.minimum(values, data['intensity_clip_max'])
            block.data = values.astype(np.float64)
            for j, family in enumerate(family_ids):
                members = families == family
                family_selected[start:end, j] = np.asarray(block[:, members].sum(axis=1)).ravel()/members.sum()
    counts = (selected[lesion] > 0).sum(axis=0)
    sums = np.where(selected[lesion] > 0, selected[lesion], 0).sum(axis=0)
    np.testing.assert_array_equal(annotations.lesion_detection_count.to_numpy()[peaks], counts)
    np.testing.assert_allclose(annotations.lesion_intensity_sum.to_numpy()[peaks], sums, rtol=1e-12)
    np.testing.assert_allclose(annotations.lesion_prevalence_percent.to_numpy()[peaks], 100*counts/lesion.sum())
    test = np.asarray(checkpoint['split_indices']['test'])
    test = test[lesion[test]]
    correlations = pd.read_csv(peak_file(output, 'peak_slide_ihc_associations.csv'))
    reference = []
    for slide in np.unique(slides):
        for t, column in enumerate(data['target_columns']):
            rows = test[(slides[test] == slide) & np.isfinite(observed[test, t])]
            for j, peak in enumerate(peaks):
                expected = np.nan
                if len(rows) >= config['analysis']['peaks'].get('min_observations', 20) and np.ptp(selected[rows, j]) and np.ptp(observed[rows, t]):
                    expected = spearmanr(selected[rows, j], observed[rows, t]).statistic
                record = correlations[(correlations.slide == slide) & (correlations.target == column) & (correlations.peak_index == peak)]
                np.testing.assert_allclose(record.spearman.iloc[0], expected, rtol=1e-12, atol=1e-12, equal_nan=True,
                    err_msg=f'{slide}, {column}, peak={peak}, rows={len(rows)}, raw dtype={raw.dtype}, selected dtype={selected.dtype}')
                reference.append(expected)
    family_export = pd.read_csv(peak_file(output, 'family_slide_ihc_associations.csv'))
    family_aggregate = pd.read_csv(peak_file(output, 'family_ihc_associations.csv'))
    checked_families = 0
    for t, column in enumerate(data['target_columns']):
        for j, family in enumerate(family_ids):
            valid_values = []
            for slide in np.unique(slides):
                rows = test[(slides[test] == slide) & np.isfinite(observed[test, t])]
                expected = np.nan
                if len(rows) >= config['analysis']['peaks'].get('min_observations', 20) and np.ptp(family_selected[rows, j]) and np.ptp(observed[rows, t]):
                    expected = spearmanr(family_selected[rows, j].astype(float), observed[rows, t].astype(float)).statistic
                    valid_values.append(expected)
                record = family_export[(family_export.slide == slide) & (family_export.target == column) & (family_export.family == family)]
                np.testing.assert_allclose(record.spearman.iloc[0], expected, rtol=1e-12, atol=1e-12, equal_nan=True)
                assert record.valid_observations.iloc[0] == len(rows)
                checked_families += 1
            record = family_aggregate[(family_aggregate.target == column) & (family_aggregate.family == family)]
            np.testing.assert_allclose(record.spearman.iloc[0], np.mean(valid_values) if valid_values else np.nan, atol=1e-12, equal_nan=True)
            assert record.contributing_slides.iloc[0] == len(valid_values)
    unique = np.unique(families)
    columns = [f'family_{f:02d}_{suffix}' for f in unique for suffix in ('mean_signal', 'detection_percent')]
    table = pq.read_table(peak_file(output, 'family_spatial_activity.parquet'), columns=['row_position', *columns]).to_pandas()
    rng = np.random.default_rng(20260719)
    sampled = np.unique(np.r_[np.arange(30), rng.integers(0, len(table), 100)])
    with h5py.File(data['context_path'], 'r') as handle:
        group = handle[_matrix_group_path(data['matrix_key'])]
        for row in sampled:
            lo, hi = group['indptr'][row:row+2]
            values = {}
            for peak, value in zip(group['indices'][lo:hi], group['data'][lo:hi]):
                values[int(peak)] = values.get(int(peak), 0.)+float(value)
            for family in unique:
                members = np.flatnonzero(families == family)
                raw_values = np.array([values.get(int(p), 0.) for p in members], dtype=np.float32)
                detected = raw_values > max(0., data['nonzero_threshold'])
                values_t = np.where(detected, raw_values, 0.)
                if data['intensity_transform'] != 'none':
                    values_t = getattr(np, data['intensity_transform'])(values_t)
                if data.get('intensity_clip_max') is not None:
                    values_t = np.minimum(values_t, data['intensity_clip_max'])
                np.testing.assert_allclose(table[f'family_{family:02d}_mean_signal'].iloc[row], values_t.sum(dtype=np.float64)/len(members), rtol=1e-6)
                np.testing.assert_allclose(table[f'family_{family:02d}_detection_percent'].iloc[row], 100*detected.mean(), rtol=1e-6)
    limits = json.loads((peak_file(output, 'family_spatial_settings.json')).read_text())
    for i, family in enumerate(unique):
        values = table[f'family_{family:02d}_mean_signal'].to_numpy()
        positive = values[values > 0]
        expected = np.percentile(positive, limits['signal_percentile']) if len(positive) else 1.
        np.testing.assert_equal(limits['signal_vmax'][i], expected)
    run = json.loads((peak_file(output, 'peak_run.json')).read_text())
    from dann.heatmap_execution import validate_png
    assert run['spatial_png_count'] == len(unique)
    assert run['png_count'] == 14+2*len(unique)
    assert run['comparison_png_count'] == 2
    comparison = json.loads(peak_file(output, 'cd8_family_comparison_settings.json').read_text())
    assert comparison['cmap'] == 'turbo'
    np.testing.assert_array_equal(comparison['family_vmax'], limits['signal_vmax'])
    with h5py.File(data['context_path'], 'r') as handle:
        raw_cd8_targets = np.column_stack([read_elem(handle[f'obs/{column}'])
            if column in handle['obs'] else np.full(len(table), np.nan) for column in data['target_columns']])
    transformed, valid = prepare_observed_targets(raw_cd8_targets, data['target_columns'], data.get('cd8_normalization'))
    index = data['target_columns'].index('Density_CD8')
    cd8 = transformed[:, index][valid[:, index]]
    positive_cd8 = cd8[cd8 > 0]
    expected_cd8_limit = np.percentile(positive_cd8, comparison['percentile']) if len(positive_cd8) else 1.
    np.testing.assert_equal(comparison['cd8_vmax'], expected_cd8_limit)
    del raw_cd8_targets, transformed, valid
    for path in output.rglob('cd8_families_*.png'):
        validate_png(path)
        from PIL import Image
        with Image.open(path) as image:
            image.load()
            assert max(image.size) <= 8000 and image.width*image.height <= 64_000_000
    assert run['html_count'] == 6+len(unique)
    for path in output.rglob('family_*_spatial.png'):
        validate_png(path, [36*config['analysis']['figure_dpi'], 4*len(np.unique(slides))*config['analysis']['figure_dpi']])
    display = json.loads((peak_file(output, 'ihc_display_settings.json')).read_text())
    for column in data['target_columns']:
        values = annotations[f'spearman_{column}'].to_numpy()
        finite = np.abs(values[np.isfinite(values)])
        expected = np.percentile(finite, display[column]['percentile']) if len(finite) else 0.
        if expected <= 0:
            expected = float(finite.max()) if len(finite) and finite.max() else 1.
        np.testing.assert_allclose(display[column]['limit'], expected, rtol=1e-12)
        assert display[column]['clipped_count'] == int((finite > expected).sum())
    result = dict(status='passed', checked_peaks=peaks.tolist(), checked_correlations=len(reference),
        checked_family_correlations=checked_families, sampled_tissue_rows=len(sampled), tissue_rows=len(table), actual_family_count=len(unique),
        spatial_png_count=run['spatial_png_count'], comparison_png_count=run['comparison_png_count'], png_count=run['png_count'], html_count=run['html_count'],
        run_seconds=run['total_seconds'], peak_host_gib=run['peak_host_gib'], audit_seconds=time.perf_counter()-started)
    (output/'reports').mkdir(exist_ok=True)
    (output/'reports'/'peak_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main() -> None:
    """Audit the configured checkpoint and output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('dann/config.yaml'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(audit(config, args.output_dir or Path(config['analysis']['output_dir'])/'peaks',
                           Path(config['analysis']['checkpoint'])), indent=2))


if __name__ == '__main__':
    main()

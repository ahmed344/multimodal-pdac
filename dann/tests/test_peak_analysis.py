"""Reference mathematics, row contracts, offline rendering, and failure isolation."""
import json
from pathlib import Path

from anndata.io import write_elem
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq
import pytest
import torch
from scipy import sparse
from scipy.stats import spearmanr

from dann.artifact_layout import peak_file
from dann.analysis_execution import RenderQueue
from dann.config import peak_analysis_settings, analysis_execution_settings
from dann.peak_analysis import (
    canonical_block, family_hierarchy, family_similarity, family_spatial_scores,
    peak_correlations, peak_statistics, read_peak_metadata, run_peak_analysis,
    signal_limits, spatial_geometry, spatial_grid,
)
from dann.peak_plots import plot_family_spatial, restricted_linkage
from dann.spatial_execution import InferenceProgress
from dann.tests.test_components import spatial_config
from dann.tests.test_spatial_execution import checkpoint


def measured():
    # Unsorted duplicates, a stored zero, and an empty row.
    return sparse.csr_matrix((np.array([2., 1., 3., 0., 4., 1., 2.], dtype='float32'),
        np.array([1, 0, 1, 2, 2, 0, 2]), np.array([0, 4, 4, 5, 7])), shape=(4, 4))


@pytest.mark.parametrize('block', [1, 3, 100])
def test_statistics_and_spatial_reference(block):
    matrix = measured()
    dense = matrix.toarray()
    labels = np.array([1, 1, 2, 3])
    lesion = np.array([True, True, False, True])
    slides = np.array(['a', 'a', 'b', 'b'])
    stats = peak_statistics(matrix, slides, lesion, block)
    for i, name in enumerate(stats['slides']):
        values = dense[(slides == name) & lesion]
        np.testing.assert_array_equal(stats['counts'][i], (values > 0).sum(axis=0))
        np.testing.assert_array_equal(stats['sums'][i], values.sum(axis=0))
    for transform in ('none', 'log1p', 'sqrt'):
        data = dict(nonzero_threshold=1., intensity_transform=transform, intensity_clip_max=1.5)
        actual = family_spatial_scores(matrix, labels, data, block)
        detected = dense > 1
        values = np.where(detected, dense, 0)
        if transform != 'none':
            values = getattr(np, transform)(values)
        values = np.minimum(values, 1.5)
        for i, family in enumerate(np.unique(labels)):
            np.testing.assert_allclose(actual['signal'][:, i], values[:, labels == family].mean(axis=1))
            np.testing.assert_allclose(actual['detection'][:, i], 100*detected[:, labels == family].mean(axis=1))
        assert not actual['signal'][1].any()
        assert not actual['detection'][1].any()
        assert signal_limits(actual['signal'], 99)[-1] == 1
    assert matrix.nnz == 7  # Input residency remains immutable.


def test_spearman_reference_missing_constant_and_slide_weights():
    rng = np.random.default_rng(41)
    dense = rng.integers(0, 5, (81, 5)).astype(float)
    dense[:, -1] = 0
    targets = rng.random((81, 3)).astype('float32')
    targets[:, 1] = 1
    targets[:30, 2] = np.nan
    slides = np.array(['a']*30+['b']*50+['c'])
    rows = np.arange(1, 81)
    result = peak_correlations(sparse.csr_matrix(dense), slides, rows, targets, 20, 2)
    for s, slide in enumerate(result['slides']):
        for t in range(3):
            chosen = rows[(slides[rows] == slide) & np.isfinite(targets[rows, t])]
            for p in range(5):
                expected = spearmanr(dense[chosen, p], targets[chosen, t]).statistic if len(chosen) >= 20 and np.ptp(dense[chosen, p]) and np.ptp(targets[chosen, t]) else np.nan
                np.testing.assert_allclose(result['correlations'][s, p, t], expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    expected = (result['correlations'][0, :, 0]+result['correlations'][1, :, 0])/2
    np.testing.assert_allclose(result['mean'][:, 0], expected, equal_nan=True)
    assert np.isnan(result['mean'][:, 1]).all()
    assert result['contributing_slides'][0, 2] == 1


def test_hierarchy_and_singleton_summary():
    from dann.analyze import embedding_cosine_families
    from scipy.cluster import hierarchy
    weights = np.random.default_rng(3).normal(size=(40, 8)).astype('float32')
    actual = family_hierarchy(weights, 7)
    for name, expected in zip(('similarity', 'families', 'order'), embedding_cosine_families(weights, 7, 'average')):
        np.testing.assert_array_equal(actual[name], expected)
    selected = actual['order'][::3]
    pruned = restricted_linkage(actual['linkage'], selected)
    np.testing.assert_array_equal(hierarchy.leaves_list(pruned), np.arange(len(selected)))
    similarity = weights[:3] @ weights[:3].T
    summary = family_similarity(similarity, np.array([1, 1, 2]))
    np.testing.assert_allclose(summary[0, 0], similarity[0, 1], rtol=1e-6)
    assert np.isnan(summary[1, 1])
    assert family_hierarchy(weights[:1], 32)['families'].tolist() == [1]


def test_spatial_holes_zeros_and_shared_scores():
    frame = pd.DataFrame(dict(batch=['a']*4+['b'], x=[0, 2, 2, 0, 5], y=[0, 0, 2, 2, 8],
                              Lesion=[True, False, True, True, False]))
    data = dict(batch_column='batch', x_column='x', y_column='y')
    slides = spatial_geometry(frame, data, 'Lesion')
    values = np.array([0, 10, 20, 30, 0], dtype='float32')
    tissue, lesion = (spatial_grid(values, slides[0], region) for region in ('tissue', 'lesion'))
    assert tissue.shape == (3, 3)
    assert np.isnan(tissue[1]).all() and np.isnan(lesion[0, 2])
    assert tissue[0, 0] == lesion[0, 0] == 0
    np.testing.assert_array_equal(tissue[np.isfinite(lesion)], lesion[np.isfinite(lesion)])
    assert np.isnan(spatial_grid(values, slides[1], 'lesion')).all()


@pytest.mark.parametrize('change', [{'block_rows': 0}, {'min_observations': True}, {'signal_percentile': 0},
    {'enabled': 'yes'}, {'unknown': 1}, {'umap_metric': 'euclidean'}, {'slides_per_page': -1}, {'ihc_gamma': 0}, {'ihc_gamma': 2}, {'ihc_percentile': 101},
    {'ihc_percentile': 0}, {'sequential_cmap': 'missing'}, {'detection_cmap': None}, {'ihc_cmap': 2}])
def test_peak_config_validation(change):
    with pytest.raises(ValueError):
        peak_analysis_settings(change)
    assert peak_analysis_settings()['enabled'] is False


def peak_fixture(config):
    config['analysis'].update(peaks=dict(enabled=True, spatial=True, block_rows=11), figure_dpi=20, family_count=2)
    for path in (config['data']['path'], config['data']['context_path']):
        with h5py.File(path, 'r+') as handle:
            batches = handle['obs/batch/codes'][:]
            n = len(batches)
            write_elem(handle['obs'], 'Lesion', batches == 0)
            coords = np.column_stack((handle['obs/x'][:], handle['obs/y'][:])).astype(float)
            write_elem(handle.require_group('obsm'), 'spatial', coords)
            spatial = {name: {'images': {'HES': np.zeros((10, 10, 3), dtype='uint8')},
                'scalefactors': {'tissue_HES_scalef': 1.}} for name in ('a', 'b')}
            write_elem(handle.require_group('uns'), 'spatial', spatial)
    return config


def test_pipeline_cache_exports_and_failure_cleanup(spatial_config, tmp_path, monkeypatch):
    from dann import peak_analysis as peaks
    import umap
    config = peak_fixture(spatial_config)
    config['analysis']['split'] = 'validation'
    saved = checkpoint(config, tmp_path)
    payload = torch.load(saved, weights_only=False)
    frame, _ = read_peak_metadata(Path(config['data']['path']), config['data'], 'Lesion')
    from dann.targets import prepare_observed_targets
    targets, valid = prepare_observed_targets(frame[config['data']['target_columns']].to_numpy(),
        config['data']['target_columns'], config['data']['cd8_normalization'])
    expected_targets = np.where(valid, targets, np.nan)
    assert (~valid).any()
    calls = []
    def capture(function, row_position, target_position):
        def wrapped(*args, **kwargs):
            calls.append((function.__name__, args[row_position].copy(), args[target_position].copy()))
            return function(*args, **kwargs)
        return wrapped
    monkeypatch.setattr(peaks, 'peak_correlations', capture(peaks.peak_correlations, 2, 3))
    monkeypatch.setattr(peaks, 'family_correlations', capture(peaks.family_correlations, 3, 4))
    resident_matrix = peaks.resident_matrix
    monkeypatch.setattr(umap.UMAP, 'fit_transform', lambda self, w: np.column_stack((w[:, 0], w[:, 1])))
    # Exercise real exports and numerical stages; rendering equivalence is tested separately.
    def minimal(queue, output, *args):
        (output/'rendered.txt').write_text('complete')
    monkeypatch.setattr('dann.peak_plots.queue_peak_plots', minimal)
    monkeypatch.setattr('dann.peak_plots.queue_spatial_plots', minimal)
    execution = analysis_execution_settings({'render_workers': 0})
    out = run_peak_analysis(config, saved, execution, tmp_path/'out')
    report = json.loads((out/'peak_run.json').read_text())
    assert report['requested_family_count'] == report['actual_family_count'] == 2
    assert report['embedding_shape'] == [5, 4]
    assert report['correlation_split'] == 'validation'
    rows = payload['split_indices']['validation']
    expected_rows = rows[frame.Lesion.to_numpy()[rows]]
    assert 0 < report['correlation_lesion_rows'] == len(expected_rows) < len(rows)
    for name in ('peak_correlations', 'family_correlations'):
        _, actual_rows, actual_targets = next(call for call in calls if call[0] == name)
        np.testing.assert_array_equal(actual_rows, expected_rows)
        np.testing.assert_array_equal(actual_targets, expected_targets)
        assert not np.isin(actual_rows, payload['split_indices']['train']).any()
        assert not np.isin(actual_rows, payload['split_indices']['test']).any()
    table = pq.read_table(peak_file(out, 'family_spatial_activity.parquet')).to_pandas()
    assert len(table) == 156
    assert not table.filter(like='mean_signal').iloc[0].any()
    original = (peak_file(out, 'peak_embeddings.csv')).read_bytes()
    # A successful rerun removes obsolete owned plots but preserves unrelated files.
    (out/'old_owned.png').write_text('previous artifact')
    (out/'user_note.txt').write_text('preserve')
    report['artifacts'].append('old_owned.png')
    (out/'peak_run.json').write_text(json.dumps(report))
    def forbidden(*a, **k):
        raise AssertionError('numerical stage should be cached')
    monkeypatch.setattr(peaks, 'resident_matrix', forbidden)
    config['analysis']['peaks'].update(ihc_percentile=95., ihc_gamma=.75, sequential_cmap='plasma', slides_per_page=1)
    config['analysis']['heatmap_top_peaks'] = 3
    run_peak_analysis(config, saved, execution, tmp_path/'out')
    assert (peak_file(out, 'peak_embeddings.csv')).read_bytes() == original
    assert not (out/'old_owned.png').exists()
    assert (out/'user_note.txt').read_text() == 'preserve'
    report = json.loads((out/'peak_run.json').read_text())
    assert set(report['cache_hits']) == {'peak-hierarchy', 'peak-statistics', 'peak-correlations', 'peak-umap', 'peak-spatial', 'family-correlations'}
    # Changing the selected split invalidates both correlation stages only.
    config['analysis']['split'] = 'test'
    calls.clear()
    monkeypatch.setattr(peaks, 'resident_matrix', resident_matrix)
    run_peak_analysis(config, saved, execution, tmp_path/'out')
    report = json.loads((out/'peak_run.json').read_text())
    assert report['correlation_split'] == 'test'
    rows = payload['split_indices']['test']
    expected_rows = rows[frame.Lesion.to_numpy()[rows]]
    assert report['correlation_lesion_rows'] == len(expected_rows)
    for name in ('peak_correlations', 'family_correlations'):
        _, actual_rows, _ = next(call for call in calls if call[0] == name)
        np.testing.assert_array_equal(actual_rows, expected_rows)
    assert set(report['cache_hits']) == {'peak-hierarchy', 'peak-statistics', 'peak-umap', 'peak-spatial'}
    monkeypatch.setattr(peaks, 'resident_matrix', forbidden)
    monkeypatch.setattr(peaks, 'validate_peak_outputs', lambda *a: (_ for _ in ()).throw(ValueError('incomplete')))
    with pytest.raises(ValueError, match='incomplete'):
        run_peak_analysis(config, saved, execution, tmp_path/'out')
    assert (peak_file(out, 'peak_embeddings.csv')).read_bytes() == original
    assert not list((tmp_path/'out').glob('.peaks-work-*'))
    with h5py.File(config['data']['context_path'], 'r+') as handle:
        handle['X/data'][0] += 1
    with pytest.raises(AssertionError, match='cached'):
        run_peak_analysis(config, saved, execution, tmp_path/'out')


def test_serial_parallel_spatial_render_and_failure(tmp_path):
    frame = pd.DataFrame(dict(batch=['a', 'a'], x=[0, 2], y=[0, 2], Lesion=[True, False]))
    slide = spatial_geometry(frame, dict(batch_column='batch', x_column='x', y_column='y'), 'Lesion')[0]
    slide.update(image=np.zeros((4, 4, 3), dtype='uint8'), scale=1., tissue_hes=np.array([[0, 0], [3, 3]]),
                 lesion_hes=np.array([[0, 0], [1, 1]]))
    values = np.array([0., 2.], dtype='float32')
    empty = {**slide, 'crop': None, 'lesion': np.zeros(2, dtype=bool), 'lesion_hes': None}
    for workers in (0, 2):
        queue = RenderQueue(tmp_path/f'arrays-{workers}')
        for region in ('tissue', 'lesion'):
            queue.submit(plot_family_spatial, [slide, empty], values, values, 1, 1.5, region,
                         tmp_path/f'{workers}-{region}.png', {'figure_dpi': 20}, 99.)
        queue.run(workers, {'host_budget_gib': 192}, InferenceProgress())
    for region in ('tissue', 'lesion'):
        np.testing.assert_array_equal(np.asarray(Image.open(tmp_path/f'0-{region}.png')),
                                      np.asarray(Image.open(tmp_path/f'2-{region}.png')))
    queue = RenderQueue(tmp_path/'bad')
    queue.submit(plot_family_spatial, [slide], values, values, 1, 1.5, 'tissue',
                 tmp_path/'missing'/'bad.png', {'figure_dpi': 20}, 99.)
    with pytest.raises(FileNotFoundError):
        queue.run(2, {'host_budget_gib': 192}, InferenceProgress())


def test_metadata_feature_validation_and_missing_targets(spatial_config, tmp_path):
    config = peak_fixture(spatial_config)
    with h5py.File(config['data']['context_path'], 'r+') as handle:
        del handle['obs/Density_CD8']
    frame, features = read_peak_metadata(Path(config['data']['context_path']), config['data'], 'Lesion', spatial=True)
    assert frame.Density_CD8.isna().all()
    assert features.tolist() == ['100', '101', '102', '103', '104']
    saved = checkpoint(config, tmp_path)
    with h5py.File(config['data']['context_path'], 'r+') as handle:
        key = handle['var'].attrs['_index']
        del handle[f'var/{key}']
        write_elem(handle['var'], key, np.array(['999', '101', '102', '103', '104'], dtype=object))
    with pytest.raises(ValueError, match='feature identities/order'):
        run_peak_analysis(config, saved, analysis_execution_settings({'render_workers': 0}), tmp_path/'out')
    assert not list((tmp_path/'out').glob('.peaks-work-*'))


def test_all_peak_views_offline_and_complete(tmp_path):
    from dann.peak_plots import queue_peak_plots
    from dann.analysis_execution import validate_outputs
    weights = np.random.default_rng(2).normal(size=(5, 4)).astype('float32')
    tree = family_hierarchy(weights, 2)
    frame = pd.DataFrame(dict(peak_index=np.arange(5), mz=np.arange(5)+100.,
        family=tree['families'], occurrence_count=[3, 4, 0, 1, 2],
        lesion_prevalence_percent=[30, 40, 0, 10, 20], mean_detected_intensity=[1, 2, np.nan, 4, 5]))
    columns = ['Density_CD8', 'Density_Tumor', 'Density_Stroma', 'Density_Collagen']
    for column in columns:
        frame[f'spearman_{column}'] = [-1, -.5, np.nan, .5, 1]
        frame[f'label_{column}'] = column
    queue = RenderQueue(tmp_path/'arrays')
    output = tmp_path/'output'
    output.mkdir()
    queue_peak_plots(queue, output, frame, weights[:, :2], tree, weights,
        np.array([[30, 40, 0, 10, 20]]), np.array(['slide']),
        dict(figure_dpi=20, heatmap_top_peaks=4), dict(target_columns=columns))
    queue.run(0, {'host_budget_gib': 192}, InferenceProgress())
    validate_outputs(output)
    assert len(list(output.glob('*.html'))) == 6
    assert len(list(output.glob('*.png'))) == 13
    html = (output/'peak_umap.html').read_text()
    assert 'Plotly.newPlot' in html and 'slide: prevalence (%)' in html
    assert pd.read_csv(output/'top_similarity_membership.csv').peak_index.tolist() == tree['order'][tree['order'] != 2].tolist()


def test_worker_failure_preserves_published_artifacts_and_cleans_staging(spatial_config, tmp_path, monkeypatch):
    import umap
    from dann.peak_plots import plot_family_summary
    config = peak_fixture(spatial_config)
    saved = checkpoint(config, tmp_path)
    monkeypatch.setattr(umap.UMAP, 'fit_transform', lambda self, w: w[:, :2])
    def failing_plot(queue, output, *args):
        queue.submit(plot_family_summary, np.eye(2), np.array([1, 2]),
                     output/'absent_directory'/'broken.png', {'figure_dpi': 20})
    monkeypatch.setattr('dann.peak_plots.queue_peak_plots', failing_plot)
    monkeypatch.setattr('dann.peak_plots.queue_spatial_plots', lambda *args: None)
    output = tmp_path/'published'
    (output/'peaks').mkdir(parents=True)
    (output/'peaks'/'sentinel.txt').write_text('prior outputs')
    with pytest.raises(FileNotFoundError):
        run_peak_analysis(config, saved, analysis_execution_settings({'render_workers': 2}), output)
    assert (output/'peaks'/'sentinel.txt').read_text() == 'prior outputs'
    assert not (output/'peaks'/'peak_run.json').exists()
    assert not list(output.glob('.peaks-work-*'))


@pytest.mark.parametrize('empty', [False, True])
def test_family_signal_correlations_reference(empty):
    from dann.peak_analysis import family_correlations
    rng = np.random.default_rng(102)
    raw = rng.integers(0, 15, (100, 5)).astype('float32')
    raw[:, -1] = 0
    families = np.array([1, 1, 2, 3, 4])
    slides = np.array(['a']*30+['b']*50+['empty']*20)
    rows = np.array([], dtype=int) if empty else np.arange(2, 80)
    targets = rng.random((100, 3)).astype('float32')
    targets[:, 2] = np.nan
    targets[:30, 1] = 0
    data = dict(nonzero_threshold=3., intensity_transform='sqrt', intensity_clip_max=3.5)
    actual = family_correlations(sparse.csr_matrix(raw), families, slides, rows, targets, data, 9)
    transformed = np.minimum(np.sqrt(np.where(raw > 3, raw, 0)), 3.5)
    signal = np.column_stack([transformed[:, families == f].mean(axis=1, dtype=np.float64).astype('float32')
                              for f in np.unique(families)])
    for s, name in enumerate(actual['slides']):
        for t in range(targets.shape[1]):
            selected = rows[(slides[rows] == name) & np.isfinite(targets[rows, t])]
            for f in range(4):
                expected = np.nan
                if len(selected) >= 20 and np.ptp(signal[selected, f]) and np.ptp(targets[selected, t]):
                    expected = spearmanr(signal[selected, f].astype(float), targets[selected, t].astype(float)).statistic
                np.testing.assert_allclose(actual['correlations'][s, f, t], expected, atol=1e-12, equal_nan=True)
    assert np.isnan(actual['mean'][-1]).all()  # Empty-signal family.
    assert (actual['contributing_slides'][:, 2] == 0).all()
    if empty:
        assert np.isnan(actual['mean']).all()
    else:
        expected = (actual['correlations'][0, :, 0]+actual['correlations'][1, :, 0])/2
        np.testing.assert_allclose(actual['mean'][:, 0], expected, equal_nan=True)


def test_ihc_shared_nonlinear_scales_and_original_values():
    from dann.peak_plots import association_display, _color_options, _plotly_colors
    settings = peak_analysis_settings()
    original = np.r_[np.linspace(-.2, .2, 101), 1., np.nan]
    display = association_display(original, settings)
    expected = np.percentile(np.abs(original[np.isfinite(original)]), 99)
    assert display['limit'] == expected and display['clipped_count'] == 1
    variant = dict(kind='association', label='target', values=original, **display)
    norm = _color_options(variant, 2)['norm']
    values, options, colorbar = _plotly_colors(variant, 2)
    np.testing.assert_allclose(norm(original[:-1]), (values[:-1]+1)/2)
    np.testing.assert_allclose(norm.inverse(norm(np.array([-.1, 0, .1]))), [-.1, 0, .1])
    assert norm(.01) > .5 + .01/(2*expected)
    assert options['cmin'] == -1 and options['cmax'] == 1
    assert 'tails clipped' in colorbar['title']
    assert original[-2] == 1 and np.isnan(original[-1])
    assert association_display(np.array([0., np.nan]), settings)['limit'] == 1
    assert association_display(np.array([np.nan]), settings)['limit'] == 1
    assert association_display(original*.1, settings)['limit'] == pytest.approx(expected*.1)


def test_spatial_canvas_layout_and_shared_limits(tmp_path, monkeypatch):
    from dann import peak_plots
    frame = pd.DataFrame(dict(batch=['a', 'a', 'b'], x=[0, 2, 0], y=[0, 2, 0], Lesion=[True, False, False]))
    slides = spatial_geometry(frame, dict(batch_column='batch', x_column='x', y_column='y'), 'Lesion')
    for slide in slides:
        slide.update(image=np.zeros((4, 4, 3)), scale=1., tissue_hes=np.array([[0, 0], [3, 3]]),
                     lesion_hes=np.array([[0, 0], [1, 1]]) if slide['crop'] else None)
    captured = []
    monkeypatch.setattr(peak_plots, '_save', lambda figure, *args: captured.append(figure))
    plot_family_spatial(slides, np.array([0., 2., 0.]), np.array([0., 100., 0.]), 1, 1.5, 'all',
                        tmp_path/'spatial.png', {'figure_dpi': 300}, 99.)
    figure = captured[0]
    np.testing.assert_array_equal(figure.get_size_inches(), [36, 8])
    panels = figure.axes[:12]
    for row in range(2):
        for offset in (0, 3):
            assert panels[row*6+offset].images[0].get_clim() == (0, 1.5)
            detection = panels[row*6+offset+1].images[0]
            assert detection.get_clim() == (0, 100) and detection.get_cmap().name == 'turbo'
    assert panels[9].texts[0].get_text() == 'No lesion'
    assert panels[0].images[0].get_array()[0, 0] == 0
    assert panels[0].images[0].get_array().mask[1, 1]
    peak_plots.plt.close(figure)


def test_packed_triangle_preserves_every_fp32_cell():
    from dann.peak_viewer import pack_upper
    rng = np.random.default_rng(47)
    matrix = rng.random((12, 12), dtype=np.float32)
    matrix = np.triu(matrix)+np.triu(matrix, 1).T
    order = rng.permutation(len(matrix))
    packed = pack_upper(matrix, order)
    assert packed.nbytes == 12*13//2*4
    for row in range(12):
        for col in range(12):
            a, b = sorted((row, col))
            index = a*12-a*(a-1)//2+b-a
            assert packed[index] == matrix[order[row], order[col]]


def test_render_workers_reduce_for_actual_canvas_budget(tmp_path):
    from dann.peak_plots import plot_family_summary
    queue = RenderQueue(tmp_path/'budget-arrays')
    queue.submit(plot_family_summary, np.eye(2), np.array([1, 2]), tmp_path/'budget.png', {'figure_dpi': 20})
    queue.canvas_bytes = 2**30
    progress = InferenceProgress()
    queue.run(16, {'host_budget_gib': 8}, progress)
    assert progress.details['requested_render_workers'] == 16
    assert progress.details['effective_render_workers'] == 1
    assert progress.details['render_worker_budget_bytes'] >= 5*2**30



def test_cd8_comparison_shared_scales_holes_and_layout(tmp_path, monkeypatch):
    from dann import peak_plots
    frame = pd.DataFrame(dict(batch=['a', 'a', 'b'], x=[0, 2, 0], y=[0, 2, 0], Lesion=[True, False, False]))
    slides = spatial_geometry(frame, dict(batch_column='batch', x_column='x', y_column='y'), 'Lesion')
    cd8 = np.array([0., .4, np.nan])
    signal = np.array([[0., 1.], [2., 0.], [0., 0.]])
    captured = []
    monkeypatch.setattr(peak_plots, '_save', lambda figure, *args: captured.append(figure))
    for region in ('tissue', 'lesion'):
        peak_plots.plot_cd8_family_comparison(slides, cd8, signal, np.array([1, 2]), np.array([1.5, 1.]),
            .3, 'Normalized CD8', region, tmp_path/f'{region}.png', {'figure_dpi': 20})
    for figure in captured:
        np.testing.assert_array_equal(figure.get_size_inches(), [12, 8])
        assert len(figure.axes) == 9  # Six panels plus one colorbar per column.
        for row in range(2):
            for col, limit in enumerate([.3, 1.5, 1.]):
                image = figure.axes[row*3+col].images[0]
                assert image.get_clim() == (0, limit)
                assert image.get_cmap().name == 'turbo'
        assert figure.axes[0].images[0].get_array()[0, 0] == 0
    assert captured[0].axes[0].images[0].get_array().mask[1, 1]
    assert captured[1].axes[3].texts[0].get_text() == 'No lesion'
    for figure in captured:
        peak_plots.plt.close(figure)


def test_artifact_organization_and_safe_manifest_cleanup(tmp_path):
    from dann.artifact_layout import organize_analysis, peak_relative_path, remove_owned
    (tmp_path/'umap').mkdir()
    (tmp_path/'latent_embeddings.csv').write_text('data')
    (tmp_path/'umap/latent_umap_ziln_Density_CD8.png').write_text('generated')
    (tmp_path/'notes.txt').write_text('keep')
    organize_analysis(tmp_path)
    assert (tmp_path/'tables/latent_embeddings.csv').read_text() == 'data'
    assert (tmp_path/'umap/parameters/latent_umap_ziln_Density_CD8.png').exists()
    assert not (tmp_path/'latent_embeddings.csv').exists()
    assert peak_relative_path('cd8_families_tissue.png') == Path('spatial/cd8_comparison/cd8_families_tissue.png')
    outside = tmp_path.parent/'outside.txt'; outside.write_text('keep')
    remove_owned(tmp_path, {'umap/parameters/latent_umap_ziln_Density_CD8.png', '../outside.txt'})
    assert outside.exists() and (tmp_path/'notes.txt').exists()
    assert not (tmp_path/'umap/parameters').exists()

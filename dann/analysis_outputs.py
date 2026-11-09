"""Feature-driven exports and render requests for every analysis execution mode."""
from pathlib import Path
from typing import Any, Mapping, Callable
import json
import numpy as np
import pandas as pd

from dann.config import analysis_feature_settings


def analysis_needs(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve shared numerical dependencies independently of table publication."""
    f = analysis_feature_settings(analysis)
    sample = any(f[n]['enabled'] for n in ('latent_exports', 'variance', 'direct_umap', 'pca_umap', 'latent_interpretation'))
    diagnostic_density = f['density_diagnostics']['errors'] or f['density_diagnostics']['contrasts']
    density_main = diagnostic_density or any(
        any(f[n][k] for k in ('tables', 'interactive', 'density_means', 'parameters'))
        for n in ('direct_umap', 'pca_umap')) or f['latent_interpretation']['ihc_scatter']
    density_scatter = diagnostic_density or f['prediction_scatter']['tables']
    return dict(features=f, samples=sample, density=density_main or density_scatter,
        density_main=density_main, density_scatter=density_scatter,
        main=sample or any(f[n]['enabled'] for n in ('calibration', 'density_diagnostics', 'he_metrics')),
        inference=any(v['enabled'] for k, v in f.items() if k != 'loss_curves'),
        scatter=any(f[n]['enabled'] for n in ('prediction_scatter', 'calibration', 'density_diagnostics')))


def write_loss_artifacts(config: Mapping[str, Any], checkpoint_path: Path,
                         checkpoint: Mapping[str, Any], output: Path, render: Callable) -> None:
    """Use the selected checkpoint's contracts and shared monitoring layout."""
    from dann.analysis_execution import load_checkpoint_snapshot
    from dann.objective_curves import plot_loss_curves
    from dann.selection import checkpoint_selection
    switches = analysis_feature_settings(config['analysis'])['loss_curves']
    if not switches['enabled']:
        return
    model_dir = Path(config['training']['output_dir'])
    best = model_dir / 'best.pt'
    curve_checkpoint = checkpoint
    if checkpoint_path.resolve() != best.resolve() and best.is_file():
        curve_checkpoint, _ = load_checkpoint_snapshot(best)
    selection = checkpoint_selection(curve_checkpoint)
    best_epoch = int(curve_checkpoint['epoch']) + 1 if best.is_file() else selection['best_epoch']
    render(plot_loss_curves, model_dir / 'history.csv', output / 'loss_curves.png',
        int(config['analysis']['figure_dpi']), best_epoch=best_epoch,
        config=curve_checkpoint['config'], selection=selection,
        overall=switches['overall'], ihc=switches['ihc'])


def write_artifacts(config, checkpoint_path, epoch, metadata, extracted, scatter_data,
                    coordinates, peptides, activity, output, render, *, checkpoint,
                    table_writer=None, sample_data=None, pca_coordinates=None,
                    variance=None, sample_provenance=None, pca_metadata=None, cache=None) -> None:
    """Publish only requested artifacts; viewers consume frames without CSV round trips."""
    from dann import analyze as a
    from dann.targets import target_contract, target_label
    from dann.latent_analysis import sample_viewers, prediction_viewer, interpret_features
    analysis = config['analysis']
    f = analysis_feature_settings(analysis)
    split, scatter_split = analysis['split'], analysis['ziln_scatter_split']
    targets, batches = config['data']['target_columns'], metadata.batch_names
    labels = [target_label(c, target_contract(config)) for c in targets]
    epsilon = float(config['loss']['logit_epsilon'])
    sample = extracted if sample_data is None else sample_data
    (output / 'target_transform.json').write_text(json.dumps(target_contract(config), indent=2))

    def table(kind, values, coords, selected_split, path):
        if table_writer is not None:
            table_writer(kind, values, coords, targets, batches, selected_split, epsilon, path)
        else:
            if kind == 'latent':
                frame = a.build_latent_embeddings_frame(values, batches, selected_split)
            elif kind == 'umap':
                frame = a.build_latent_umap_frame(values, coords, targets, batches, selected_split)
            else:
                frame = a.build_ziln_scatter_frame(values, targets, batches, selected_split, epsilon)
            frame.to_csv(path, index=False, float_format='%.17g')

    if f['latent_exports']['tables']:
        table('latent', sample, None, split, output / 'latent_embeddings.csv')
    if f['variance']['enabled']:
        folder = output / 'latent_variance'; folder.mkdir()
        if variance is None:
            report = dict(sample_provenance or {}, skipped='fewer than two lesion observations')
        else:
            summary, spectrum = variance
            report = {**summary, **(sample_provenance or {})}
            if f['variance']['tables']:
                pd.DataFrame([report]).to_csv(folder / 'latent_variance_summary.csv', index=False)
                spectrum.to_csv(folder / 'pca_variance.csv', index=False)
            if f['variance']['explained_variance']:
                from dann.analysis_latents import plot_explained_variance
                render(plot_explained_variance, spectrum, report, folder / 'explained_variance.png', int(analysis['figure_dpi']))
        (folder / 'provenance.json').write_text(json.dumps(report, indent=2))
    for name, directory, coords in (('direct_umap', 'umap', coordinates), ('pca_umap', 'umap_pca', pca_coordinates)):
        flags = f[name]
        if not flags['enabled']:
            continue
        folder = output / directory; folder.mkdir()
        report = {**(sample_provenance or {}), **(pca_metadata or {} if name == 'pca_umap' else {
            key: analysis[key] for key in ('umap_neighbors', 'umap_min_dist', 'umap_metric', 'umap_random_state')})}
        if name == 'direct_umap':
            report['effective_n_neighbors'] = min(int(analysis['umap_neighbors']), max(len(sample['row_ids']) - 1, 2))
        if coords is None:
            report.setdefault('skipped', 'no selected observations')
        (folder / 'settings.json').write_text(json.dumps(report, indent=2))
        if coords is None:
            continue
        if flags['tables']:
            table('umap', sample, coords, split, folder / 'latent_umap.csv')
        if flags['batch']:
            render(a.plot_latent_umap_batch, coords, sample['batches'], batches, folder / 'latent_umap_batch.png', analysis)
        for flag, suffix, values, kwargs in (
            ('densities', 'densities', sample['targets'], {}),
            ('logit_densities', 'densities_logit', sample['targets'], {'logit_epsilon': epsilon}),
            ('density_means', 'density_means', sample.get('density_mean'), {'logit_epsilon': epsilon, 'sampled_means': True})):
            if flags[flag]:
                render(a.plot_latent_umap_densities, coords, values, labels, folder / f'latent_umap_{suffix}.png', analysis, **kwargs)
        for index, column in enumerate(targets):
            args = (coords, sample['targets'][:, index], sample['pi'][:, index], sample['mu'][:, index],
                    sample['sigma'][:, index], sample['batches'], batches, labels[index])
            if flags['parameters']:
                render(a.plot_latent_umap_ziln_per_target, *args, folder / f'latent_umap_ziln_{column}.png',
                       analysis, epsilon, sample['density_mean'][:, index])
            if flags['diagnostics']:
                render(a.plot_latent_umap_ziln_diagnostics_per_target, *args,
                       folder / f'latent_umap_ziln_diagnostics_{column}.png', analysis, epsilon)
        if flags['interactive']:
            frame = a.build_latent_umap_frame(sample, coords, targets, batches, split)
            latent = pd.DataFrame(sample['latent'], columns=[f'latent_{i}' for i in range(sample['latent'].shape[1])])
            viewer_folder = output / 'latent' if name == 'direct_umap' else folder
            viewer_folder.mkdir(exist_ok=True)
            sample_viewers(viewer_folder, latent, frame, targets, epsilon, modes=('umap',))
    if f['prediction_scatter']['enabled']:
        flags = f['prediction_scatter']
        if flags['tables']:
            table('scatter', scatter_data, None, scatter_split, output / 'ziln_density_scatter.csv')
        if flags['plot']:
            render(a.plot_ziln_density_scatter, scatter_data['mu'], scatter_data['targets'], labels, epsilon,
                   output / 'ziln_density_scatter.png', analysis)
        if flags['interactive']:
            folder = output / 'latent'; folder.mkdir(exist_ok=True)
            frame = a.build_ziln_scatter_frame(scatter_data, targets, batches, scatter_split, epsilon)
            if len(frame):
                prediction_viewer(frame, folder / 'prediction_scatter.html')
    if f['latent_interpretation']['enabled'] and len(sample['row_ids']):
        interpret_features(config, sample, batches, output / 'latent', cache, coordinates)
    if f['calibration']['enabled']:
        from dann.calibration import run_calibration
        run_calibration(extracted, targets, batches, split, output / 'calibration', analysis, epsilon, renderer=render)
        if scatter_split != split:
            run_calibration(scatter_data, targets, batches, scatter_split, output / 'calibration' / scatter_split, analysis, epsilon, renderer=render)
    if f['density_diagnostics']['enabled']:
        from dann.diagnostics import write_density_diagnostics
        grid_keys = None
        if f['density_diagnostics']['contrasts'] and 'support' in extracted:
            from dann.tiles import read_grid
            grid_keys = read_grid(Path(config['data']['path']), config['data'])[0]
        write_density_diagnostics(extracted, config, output, split, grid_keys=grid_keys)
        if scatter_split != split:
            write_density_diagnostics(scatter_data, config, output, scatter_split, grid_keys=grid_keys)
    if f['he_metrics']['enabled']:
        from dann.he import report_he
        report_he(extracted, output, split, config['loss'].get('he_weight', 1.))
        if scatter_split != split and scatter_data is not None:
            report_he(scatter_data, output, scatter_split, config['loss'].get('he_weight', 1.))
    write_loss_artifacts(config, checkpoint_path, checkpoint, output, render)


def write_latent_manifest(staged: Path, output: Path, sample, cache) -> None:
    """Record only newly rendered latent files and available source-table checksums."""
    from dann.analysis_execution import digest_file
    from dann.artifact_layout import analysis_relative_path
    folder = staged / 'latent'
    if not folder.is_dir():
        return
    sources = [staged / p for p in ('latent_embeddings.csv', 'umap/latent_umap.csv', 'ziln_density_scatter.csv')]
    report = dict(sample_rows=len(sample['row_ids']), feature_rows=len(sample['row_ids']),
        features=sample['latent'].shape[1],
        source_hashes={str(output / analysis_relative_path(p.relative_to(staged))): digest_file(p)
                       for p in sources if p.is_file()},
        artifacts=[str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()],
        cache_hits=cache.progress.details.get('cache_hits', []))
    (folder / 'latent_run.json').write_text(json.dumps(report, indent=2))

"""Offline latent interpretation from saved analysis exports; no model inference.

Run ``python -m dann.latent_analysis --config dann/config.yaml`` after analysis.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
from pathlib import Path
import tempfile
import torch
import time

import h5py
from anndata.io import read_elem
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from dann.analysis_execution import StageCache, digest_file, fingerprint
from dann.artifact_layout import analysis_file
from dann.html_display import format_number, formatter_script, write_plotly
from dann.spatial_execution import InferenceProgress


def slide_correlations(values, targets, slides, min_observations=20):
    """Equal-slide Spearman means, retaining undefined entries and support counts."""
    x, y = np.asarray(values, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    names = np.unique(slides)
    pairs = np.full((len(names), x.shape[1], x.shape[1]), np.nan)
    associations = np.full((len(names), x.shape[1], y.shape[1]), np.nan)
    observations = np.zeros(associations.shape, dtype=np.int64)
    for s, name in enumerate(names):
        a, b = x[slides == name], y[slides == name]
        if len(a) >= min_observations and np.isfinite(a).all():
            ranks = rankdata(a, axis=0); ranks -= ranks.mean(axis=0)
            norms = np.linalg.norm(ranks, axis=0)
            denominator = norms[:, None]*norms[None, :]
            pairs[s] = np.divide(ranks.T @ ranks, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)
        else:
            for i in range(x.shape[1]):
                for j in range(i, x.shape[1]):
                    valid = np.isfinite(a[:, i]) & np.isfinite(a[:, j])
                    if valid.sum() >= min_observations:
                        pairs[s, i, j] = pairs[s, j, i] = _spearman(a[valid, i], a[valid, j])
        for t in range(y.shape[1]):
            valid_y = np.isfinite(b[:, t])
            for i in range(x.shape[1]):
                valid = valid_y & np.isfinite(a[:, i])
                observations[s, i, t] = valid.sum()
                if valid.sum() >= min_observations:
                    associations[s, i, t] = _spearman(a[valid, i], b[valid, t])
    def aggregate(array):
        count = np.isfinite(array).sum(axis=0)
        return np.divide(np.nansum(array, axis=0), count, out=np.full(count.shape, np.nan), where=count > 0), count
    correlation, counts = aggregate(pairs)
    association, target_counts = aggregate(associations)
    return dict(correlation=np.clip(correlation, -1, 1), contributing=counts,
                association=association, target_counts=target_counts, per_slide=pairs,
                target_per_slide=associations, observations=observations, slides=names.astype(str))


def _spearman(x, y):
    rx, ry = rankdata(np.asarray(x, dtype=np.float64)), rankdata(np.asarray(y, dtype=np.float64))
    rx -= rx.mean(); ry -= ry.mean()
    denominator = np.linalg.norm(rx)*np.linalg.norm(ry)
    return float(np.clip(rx @ ry / denominator, -1, 1)) if denominator else np.nan


def feature_layout(correlation, settings, *, umap=True):
    """Seeded UMAP and average-linkage ordering without inventing missing scores."""
    n = len(correlation)
    distance = 1 - np.nan_to_num(correlation, nan=0.)
    distance = np.clip((distance+distance.T)/2, 0, 2)
    np.fill_diagonal(distance, 0)
    linkage = hierarchy.linkage(squareform(distance, checks=False), method='average') if n > 1 else np.empty((0, 4))
    order = hierarchy.leaves_list(linkage).astype(np.int64) if n > 1 else np.arange(n)
    eligible = np.flatnonzero(np.isfinite(np.diag(correlation)))
    coordinates = np.full((n, 2), np.nan)
    if not umap:
        pass
    elif len(eligible) <= 2:
        coordinates[eligible] = np.column_stack((np.arange(len(eligible)), np.zeros(len(eligible))))
    else:
        from umap import UMAP
        coordinates[eligible] = UMAP(n_neighbors=min(settings['umap_neighbors'], len(eligible)-1),
            min_dist=settings['umap_min_dist'], random_state=settings['umap_random_state'],
            metric='precomputed', init='random', n_jobs=1).fit_transform(distance[np.ix_(eligible, eligible)])
    return dict(coordinates=coordinates, order=order, linkage=linkage)


def _json(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')


def write_asset(path, values):
    """Store exact exported FP64 values once; load a column through a local script."""
    packed = base64.b64encode(np.asarray(values, dtype='<f8').tobytes()).decode('ascii')
    path.write_text('window.latentAsset('+_json(path.stem)+','+_json(packed)+');')


def sample_viewers(output, latent, frame, targets, epsilon, *, modes=('umap', 'scatter')):
    """Shared on-demand columns keep 256 selectors out of the initial HTML payload."""
    from plotly.offline import get_plotlyjs
    from matplotlib import colormaps
    from matplotlib.colors import to_hex
    assets = output/'assets'; assets.mkdir(exist_ok=True)
    columns = {}
    for index, name in enumerate(latent.columns):
        key = f'feature_{index}'
        write_asset(assets/(key+'.js'), latent[name].to_numpy())
        columns[name] = dict(key=key, signed=True)
    for name in frame.columns:
        if name in ('row_id', 'batch_id', 'umap_1', 'umap_2') or any(name == t or name.endswith('_'+t) for t in targets) and pd.api.types.is_numeric_dtype(frame[name]):
            if name.startswith('valid_'):
                continue
            key = f'column_{len(columns)}'
            values = frame[name].to_numpy(dtype=float)
            write_asset(assets/(key+'.js'), values)
            columns[name] = dict(key=key, signed=name.startswith(('mu_', 'umap_')))
    for target in targets:
        for source in (target, 'density_mean_'+target):
            values = frame[source].to_numpy(dtype=float)
            clipped = np.clip(values, epsilon, 1-epsilon)
            logits = np.where(values > 0, np.log(clipped/(1-clipped)), np.nan)
            key = f'column_{len(columns)}'; write_asset(assets/(key+'.js'), logits)
            columns['logit_'+source] = dict(key=key, signed=True)
        values = 1-frame[f'pi_{target}'].to_numpy()
        key = f'column_{len(columns)}'; write_asset(assets/(key+'.js'), values)
        columns['1-pi_'+target] = dict(key=key, signed=False)
    from dann.peak_plots import family_colors
    batches = frame[['batch_id', 'batch']].drop_duplicates().sort_values('batch_id')
    palettes = {name: [[float(x), to_hex(colormaps[name](x))] for x in np.linspace(0, 1, 257)] for name in ('turbo','RdBu_r')}
    metadata = dict(palettes=palettes, columns=columns, features=latent.columns.tolist(), targets=targets,
        batches=batches.to_dict('records'), batch_colors=family_colors(int(batches.batch_id.max())+1), split=str(frame['split'].iloc[0]), count=len(frame), epsilon=epsilon)
    (assets/'plotly.js').write_text(get_plotlyjs())
    (assets/'display.js').write_text(formatter_script())
    (assets/'sample_viewer.js').write_text(Path(__file__).with_name('latent_sample_viewer.js').read_text())
    for mode, filename, title in [('umap','sample_umap.html','Sample latent UMAP'),('scatter','latent_ihc_scatter.html','Latent feature versus observed IHC')]:
        if mode not in modes:
            continue
        html = '''<!doctype html><html><head><meta charset="utf-8"><title>TITLE</title>
<style>body{font:14px system-ui;margin:20px;color:#222}.controls{display:flex;gap:14px;flex-wrap:wrap}select{max-width:300px;font:inherit}#plot{height:80vh;min-height:650px}#status{margin:12px 0}</style></head><body>
<h1>TITLE</h1><div class="controls"><label id="feature-label">Feature <select id="feature"></select></label><label id="target-label">Target <select id="target"></select></label><label id="scale-label">Scale <select id="scale"><option value="density">Density</option><option value="logit">Positive-density logit</option></select></label><label>Color <select id="color"></select></label><label>Slide <select id="slide"><option value="all">All slides</option></select></label></div>
<p id="status">Loading saved data…</p><div id="plot"></div><script src="assets/plotly.js"></script><script src="assets/display.js"></script><script>window.latentMetadata=METADATA;window.latentMode=MODE;</script><script src="assets/sample_viewer.js"></script></body></html>'''
        (output/filename).write_text(html.replace('TITLE',title).replace('METADATA',_json(metadata)).replace('MODE',_json(mode)))


def prediction_viewer(frame, path):
    """Interactive version of the existing validation positive-branch scatter."""
    import plotly.graph_objects as go
    figure = go.Figure(); buttons = []
    targets = list(frame.target.unique())
    for i, target in enumerate(targets):
        part = frame[frame.target == target]
        custom = part[['row_id', 'batch_id', 'true_logit', 'predicted_mu']].to_numpy()
        figure.add_trace(go.Scattergl(x=part.true_logit, y=part.predicted_mu, mode='markers',
            marker=dict(size=3, color=part.predicted_sigma, colorscale='Turbo', showscale=True,
                        colorbar=dict(title='Sigma', tickformat='.8g')), customdata=custom,
            text=[f'Row {int(r)}<br>Batch {int(b)}<br>Observed logit {format_number(x)}<br>Predicted mu {format_number(y)}<br>Sigma {format_number(z)}' for (r,b,x,y),z in zip(custom,part.predicted_sigma)],
            hovertemplate='%{text}<extra></extra>', visible=i==0, name=target))
        buttons.append(dict(label=target,method='update',args=[{'visible':[i==j for j in range(len(targets))]}, {'title':target+' · '+str(part.split.iloc[0])}]))
    figure.update_layout(title='Prediction scatter · '+str(frame.split.iloc[0]),template='plotly_white',height=850,
        xaxis_title='Observed positive-density logit',yaxis_title='Predicted mu',updatemenus=[dict(buttons=buttons)])
    write_plotly(figure,path)


def render_features(output, numerical, layout, latent, targets, slides, settings, *, flags=None):
    flags = flags or dict(tables=True, correlations=True, feature_umap=True)
    import plotly.graph_objects as go
    from dann.peak_plots import association_display, _plotly_colors, plot_interactive_heatmap
    ids = np.arange(latent.shape[1]); values = latent.to_numpy()
    annotations = [dict(label='Mean latent value',kind='signed',values=np.nanmean(values,axis=0),cmap='RdBu_r'),
                   dict(label='Standard deviation',kind='continuous',values=np.nanstd(values,axis=0),cmap='turbo')]
    displays = {}
    for t, target in enumerate(targets):
        display = association_display(numerical['association'][:,t],settings)
        displays[target] = display
        annotations.append(dict(label=target,kind='association',values=numerical['association'][:,t],**display))
    if flags['correlations']:
        plot_interactive_heatmap(numerical['correlation'],layout['order'],ids,annotations,output/'feature_correlations.html',
            'Latent feature correlations · equal-weight within-slide Spearman',linkage=layout['linkage'],
            identity_label='Feature',identity_kind='integer',matrix_label='Spearman',contributing=numerical['contributing'])
    if flags['feature_umap']:
        for slide in np.unique(slides):
            annotations.append(dict(label=f'{slide}: mean latent value',kind='signed',values=np.nanmean(values[slides==slide],axis=0),cmap='RdBu_r'))
        figure=go.Figure(); buttons=[]
        valid=np.isfinite(layout['coordinates']).all(axis=1)
        for i, annotation in enumerate(annotations):
            colors,options,colorbar=_plotly_colors(annotation,1)
            figure.add_trace(go.Scatter(x=layout['coordinates'][valid,0],y=layout['coordinates'][valid,1],mode='markers',
                customdata=np.column_stack((ids[valid],annotation['values'][valid])),
                text=[f'Feature {j}<br>{annotation["label"]}: {format_number(annotation["values"][j])}' for j in ids[valid]],
                hovertemplate='%{text}<extra></extra>',marker=dict(size=9,color=colors[valid],showscale=True,colorbar=colorbar,**options),visible=i==0))
            buttons.append(dict(label=annotation['label'],method='update',args=[{'visible':[i==j for j in range(len(annotations))]}]))
        figure.update_layout(title=f'Latent feature UMAP · unavailable features: {ids[~valid].tolist()}',template='plotly_white',height=850,
            xaxis_title='UMAP 1',yaxis_title='UMAP 2',updatemenus=[dict(buttons=buttons)])
        write_plotly(figure,output/'feature_umap.html')
    if flags['tables']:
        (output/'tables').mkdir()
        np.save(output/'tables/feature_correlations.npy',numerical['correlation'])
        np.save(output/'tables/contributing_slides.npy',numerical['contributing'])
        np.save(output/'tables/feature_linkage.npy',layout['linkage'])
        np.save(output/'tables/feature_order.npy',layout['order'])
        np.savez(output/'tables/per_slide_correlations.npz',correlation=numerical['per_slide'],slides=numerical['slides'])
        pd.DataFrame(dict(feature=ids,umap_1=layout['coordinates'][:,0],umap_2=layout['coordinates'][:,1],
            mean=np.nanmean(values,axis=0),std=np.nanstd(values,axis=0))).to_csv(output/'tables/feature_statistics.csv',index=False)
        pd.concat([pd.DataFrame(dict(feature=ids,target=target,spearman=numerical['association'][:,t],contributing_slides=numerical['target_counts'][:,t])) for t,target in enumerate(targets)]).to_csv(output/'tables/ihc_associations.csv',index=False)
        records = [pd.DataFrame(dict(feature=ids,slide=slide,target=target,spearman=numerical['target_per_slide'][s,:,t],valid_observations=numerical['observations'][s,:,t])) for s,slide in enumerate(numerical['slides']) for t,target in enumerate(targets)]
        (pd.concat(records) if records else pd.DataFrame(columns=['feature','slide','target','spearman','valid_observations'])).to_csv(output/'tables/slide_ihc_associations.csv',index=False)
    (output/'ihc_display_settings.json').write_text(json.dumps(displays,indent=2))


def align_exports(latent, frame):
    """Require a one-to-one sample identity join, never rely on CSV row order."""
    if not latent.row_id.is_unique or not frame.row_id.is_unique or set(latent.row_id)!=set(frame.row_id):
        raise ValueError('Latent and UMAP exports must have identical unique row IDs.')
    frame=frame.set_index('row_id').loc[latent.row_id].reset_index()
    for key in ('row_id','split','batch_id','batch'):
        if not np.array_equal(latent[key].to_numpy(),frame[key].to_numpy()):
            raise ValueError(f'Latent/UMAP identity mismatch: {key}')
    columns=[f'latent_{i}' for i in range(sum(c.startswith('latent_') for c in latent.columns))]
    if not columns or set(columns)!={c for c in latent.columns if c.startswith('latent_')}:
        raise ValueError('Latent columns must be contiguous and numerically ordered.')
    return latent[columns],frame


def interpret_features(config, sample, batch_names, output, cache, coordinates=None):
    """Compute only requested interpretation views using shared sample arrays."""
    from dann.config import analysis_feature_settings, peak_analysis_settings
    from dann import analyze as a
    features = analysis_feature_settings(config['analysis'])
    flags = features['latent_interpretation']
    if not flags['enabled'] or not len(sample['row_ids']):
        return
    output.mkdir(parents=True, exist_ok=True)
    targets = config['data']['target_columns']
    settings = peak_analysis_settings(config['analysis'].get('peaks'))
    latent = pd.DataFrame(sample['latent'], columns=[f'latent_{i}' for i in range(sample['latent'].shape[1])])
    slides = np.asarray(batch_names)[sample['batches']]
    if any(flags[k] for k in ('tables', 'correlations', 'feature_umap')):
        feature_sample = sample
        scope = config['analysis'].get('sample_scope', 'lesion' if any(features[n]['enabled'] for n in ('variance', 'pca_umap')) else 'split')
        if scope != 'lesion':
            # Historical direct UMAPs used the whole split, while feature
            # correlations have always been restricted to lesion observations.
            from dann.analysis_latents import select_sample_rows
            lesion_config = {**config, 'analysis': {**config['analysis'], 'sample_scope': 'lesion'}}
            feature_sample, slides, _ = select_sample_rows(sample, lesion_config, batch_names)
        feature_latent = pd.DataFrame(feature_sample['latent'], columns=latent.columns)
        key = dict(version=2, latent=fingerprint(feature_sample['latent']), targets=fingerprint(feature_sample['targets']),
                   slides=fingerprint(slides), minimum=settings['min_observations'])
        hit = cache.get('latent-feature-associations', key) if cache else None
        if hit is None:
            numerical = slide_correlations(feature_sample['latent'], feature_sample['targets'], slides, settings['min_observations'])
            if cache: cache.put('latent-feature-associations', key, numerical)
        else:
            numerical = hit[0]
        layout_key = dict(version=2, correlation=fingerprint(numerical['correlation']),
            umap=flags['feature_umap'], settings={k:settings[k] for k in ('umap_neighbors','umap_min_dist','umap_random_state')})
        hit = cache.get('latent-feature-umap', layout_key) if cache else None
        if hit is None:
            layout = feature_layout(numerical['correlation'], settings, umap=flags['feature_umap'])
            if cache: cache.put('latent-feature-umap', layout_key, layout)
        else:
            layout = hit[0]
        render_features(output, numerical, layout, feature_latent, targets, slides, settings, flags=flags)
    if flags['ihc_scatter']:
        if coordinates is None:
            coordinates = np.full((len(latent), 2), np.nan)
        frame = a.build_latent_umap_frame(sample, coordinates, targets, batch_names, config['analysis']['split'])
        sample_viewers(output, latent, frame, targets, float(config['loss']['logit_epsilon']), modes=('scatter',))


def run_latent_analysis(config, output_override=None):
    """Regenerate enabled offline viewers from saved arrays, without inference."""
    from dann.config import analysis_feature_settings
    from dann.calibration import load_extracted_from_frame
    from dann.analysis_latents import select_sample_rows
    import copy
    config = copy.deepcopy(config)
    if 'features' not in config['analysis'] and 'latent_interactive' not in config['analysis']:
        config['analysis']['latent_interactive'] = True  # Historical explicit standalone invocation.
    flags = analysis_feature_settings(config['analysis'])
    root = Path(output_override or config['analysis']['output_dir'])
    if not any(flags[n]['interactive'] for n in ('direct_umap', 'pca_umap', 'prediction_scatter')) and not flags['latent_interpretation']['enabled']:
        return root / 'latent'
    from dann.preprocessing import analysis_provenance
    analysis_provenance(root, config['data'])
    root.mkdir(parents=True, exist_ok=True)
    progress = InferenceProgress('Latent viewers')
    cache = StageCache(root / '.analysis-cache', config['analysis'].get('execution', {}).get('cache_policy', 'reuse'), progress)
    source_paths = set()

    def read_table(relative):
        path = analysis_file(root, relative)
        source_paths.add(path)
        return pd.read_csv(path, float_precision='round_trip')

    with tempfile.TemporaryDirectory(prefix='.latent-work-', dir=root) as temporary:
        staged = Path(temporary)
        for name, directory in (('direct_umap', 'umap'), ('pca_umap', 'umap_pca')):
            if not flags[name]['interactive'] and not (name == 'direct_umap' and flags['latent_interpretation']['enabled']):
                continue
            full = read_table('latent_embeddings.csv')
            frame = read_table(directory + '/latent_umap.csv')
            # Legacy latent tables may include nonlesion rows absent from new UMAPs.
            full = full[full.row_id.isin(frame.row_id)]
            latent, frame = align_exports(full, frame)
            sample, batch_names = load_extracted_from_frame(frame, config['data']['target_columns'])
            sample['latent'] = latent.to_numpy()
            sample['row_ids'] = frame.row_id.to_numpy(dtype=np.int64)
            sample['density_mean'] = frame[['density_mean_' + t for t in config['data']['target_columns']]].to_numpy()
            config['analysis'].setdefault('split', str(frame['split'].iloc[0]))
            if set(frame['split']) != {config['analysis']['split']}:
                raise ValueError('Saved viewer rows do not match analysis.split.')
            sample, _, _ = select_sample_rows(sample, config, batch_names)
            frame = frame.set_index('row_id').loc[sample['row_ids']].reset_index()
            latent = pd.DataFrame(sample['latent'], columns=latent.columns)
            destination = staged / ('latent' if name == 'direct_umap' else directory)
            destination.mkdir(exist_ok=True)
            if flags[name]['interactive']:
                sample_viewers(destination, latent, frame, config['data']['target_columns'], float(config['loss']['logit_epsilon']), modes=('umap',))
            if name == 'direct_umap' and flags['latent_interpretation']['enabled']:
                interpret_features(config, sample, batch_names, destination, cache, frame[['umap_1','umap_2']].to_numpy())
        if flags['prediction_scatter']['interactive']:
            destination = staged / 'latent'; destination.mkdir(exist_ok=True)
            prediction_viewer(read_table('ziln_density_scatter.csv'), destination / 'prediction_scatter.html')
        files = [p for p in staged.rglob('*') if p.is_file()]
        for path in files:
            destination = root / path.relative_to(staged)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, destination)
        (root / 'latent').mkdir(exist_ok=True)
        report = dict(progress.snapshot(torch.device('cpu')),
            source_hashes={str(path): digest_file(path) for path in sorted(source_paths)},
            artifacts=[str(p.relative_to(staged / 'latent')) for p in files if p.is_relative_to(staged / 'latent')])
        (root / 'latent/latent_run.json').write_text(json.dumps(report, indent=2))
    return root / 'latent'


def main():
    from dann.config import load_config
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('dann/config.yaml'))
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    print(run_latent_analysis(load_config(args.config),args.output_dir))


if __name__=='__main__':
    main()

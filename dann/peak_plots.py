"""Static and offline interactive views of a single cached peak hierarchy."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from anndata.io import read_elem
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, to_hex, FuncNorm
import numpy as np
from scipy.cluster import hierarchy

from dann.spatial_execution import check_host_budget
from dann.html_display import format_number, write_plotly


def family_colors(count: int) -> list[str]:
    """Stable categorical palette shared by tables, HTML, and every PNG."""
    pool = [*plt.cm.tab20.colors, *plt.cm.tab20b.colors, *plt.cm.tab20c.colors]
    return [to_hex(pool[i] if i < len(pool) else plt.cm.hsv(i/count)) for i in range(count)]


def _save(figure, path, config):
    try:
        figure.savefig(path, dpi=int(config['figure_dpi']), pil_kwargs={'compress_level': 1})
    finally:
        plt.close(figure)


def association_display(values: np.ndarray, settings: Mapping[str, Any]) -> dict:
    """Freeze a target's symmetric display limits without modifying correlations."""
    finite = np.abs(np.asarray(values)[np.isfinite(values)])
    limit = float(np.percentile(finite, settings['ihc_percentile'])) if finite.size else 0.
    if limit <= 0:
        limit = float(finite.max()) if finite.size and finite.max() > 0 else 1.
    return dict(limit=limit, gamma=settings['ihc_gamma'], cmap=settings['ihc_cmap'],
                percentile=settings['ihc_percentile'], clipped_count=int(np.sum(finite > limit)))


def signed_power(values, gamma):
    """Signed power transform; the exported values remain on the original scale."""
    return np.sign(values) * np.abs(values)**gamma


def _color_options(variant, count):
    name = variant['kind']
    if name == 'family':
        return dict(cmap=ListedColormap(family_colors(count)), vmin=.5, vmax=count+.5)
    if name == 'association':
        limit, gamma = variant['limit'], variant['gamma']
        norm = FuncNorm((lambda x: signed_power(x, gamma), lambda x: signed_power(x, 1/gamma)),
                        vmin=-limit, vmax=limit, clip=True)
        return dict(cmap=variant['cmap'], norm=norm)
    if name == 'prevalence':
        return dict(cmap=variant.get('cmap', 'turbo'), vmin=0., vmax=100.)
    return dict(cmap=variant.get('cmap', 'turbo'))


def _plotly_colors(variant, count):
    """Plotly colors and ticks share the Matplotlib normalization exactly."""
    values = np.asarray(variant['values'])
    kind = variant['kind']
    cmap = matplotlib.colormaps[variant.get('cmap', 'turbo')]
    colors = [[float(i), to_hex(cmap(i))] for i in np.linspace(0, 1, 257)]
    options = dict(colorscale=colors)
    colorbar = dict(title=variant['label'])
    if kind == 'association':
        limit, gamma = variant['limit'], variant['gamma']
        values = signed_power(np.clip(values/limit, -1, 1), gamma)
        ticks = np.linspace(-limit, limit, 5)
        options.update(cmin=-1, cmax=1)
        colorbar.update(tickvals=signed_power(ticks/limit, gamma).tolist(),
                        ticktext=[format_number(v) for v in ticks],
                        title=variant['label']+' (tails clipped)')
    elif kind == 'signed':
        finite = np.abs(values[np.isfinite(values)])
        limit = float(np.percentile(finite, 99)) if len(finite) else 1.
        options.update(cmin=-(limit or 1.), cmax=limit or 1.)
    elif kind == 'prevalence':
        options.update(cmin=0, cmax=100)
    elif kind == 'family':
        colors = family_colors(count)
        options.update(cmin=.5, cmax=count+.5,
            colorscale=[[edge, color] for i, color in enumerate(colors) for edge in (i/count, (i+1)/count)])
        colorbar.update(tickvals=list(range(1, count+1)))
    return values, options, colorbar


def plot_peak_scatter(x: np.ndarray, y: np.ndarray, variants: list[dict], path: Path, config: Mapping[str, Any],
                      family_count: int, *, logarithmic: bool = False, columns: int = 3) -> None:
    """Render one panel per annotation, retaining all points in UMAP views."""
    cols = columns
    rows = (len(variants)+cols-1)//cols
    figure, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False, layout='constrained')
    mask = np.isfinite(x) & np.isfinite(y)
    if logarithmic:
        mask &= y > 0
    for axis, variant in zip(axes.flat, variants):
        values = np.asarray(variant['values'])
        # Unavailable annotations remain visible as neutral points.
        axis.scatter(x[mask], y[mask], s=3, c='#cccccc', linewidths=0, rasterized=True)
        colored = mask & np.isfinite(values)
        options = _color_options(variant, family_count)
        points = axis.scatter(x[colored], y[colored], c=values[colored], s=5, linewidths=0,
                              rasterized=True, **options)
        ticks = np.arange(1, family_count+1) if variant['kind'] == 'family' else None
        figure.colorbar(points, ax=axis, ticks=np.linspace(-variant['limit'], variant['limit'], 5)
                        if variant['kind'] == 'association' else ticks, label=variant['label'],
                        extend='both' if variant['kind'] == 'association' else 'neither')
        axis.set_title(variant['label'])
        if logarithmic:
            axis.set(xlabel='Lesion prevalence (%)', ylabel='Mean detected intensity', yscale='log', xlim=(0, 100))
        else:
            axis.set(xticks=[], yticks=[], xlabel='UMAP 1', ylabel='UMAP 2')
    for axis in list(axes.flat)[len(variants):]:
        axis.set_visible(False)
    if logarithmic:
        figure.suptitle(f'One point per peak; {int((~mask).sum())} never-detected/nonpositive peaks excluded')
    _save(figure, path, config)


def plot_interactive_scatter(x: np.ndarray, y: np.ndarray, mz: np.ndarray, families: np.ndarray,
                             variants: list[dict], path: Path, *, logarithmic: bool = False) -> None:
    """Offline Plotly selector for family, continuous annotations, and each slide."""
    import plotly.graph_objects as go
    count = int(np.max(families))
    mask = np.isfinite(x) & np.isfinite(y)
    if logarithmic:
        mask &= y > 0
    custom = np.column_stack((np.arange(len(mz)), mz, families))[mask]
    figure = go.Figure()
    buttons = []
    for i, variant in enumerate(variants):
        values, options, colorbar = _plotly_colors(variant, count)
        marker = dict(color=values[mask], size=5, showscale=True, colorbar=colorbar, **options)
        raw_custom = np.column_stack((custom, np.asarray(variant['values'])[mask]))
        figure.add_trace(go.Scattergl(x=x[mask], y=y[mask], mode='markers', marker=marker,
            customdata=raw_custom, visible=i == 0, name=variant['label'],
            text=[f'Peak {int(p)}<br>m/z {format_number(m, "mz")}<br>Family {int(f)}<br>x {format_number(a)}<br>y {format_number(b)}<br>Annotation {format_number(v, "integer" if variant["kind"] == "family" else "mz" if variant["label"] == "m/z" else "float")}' for (p, m, f, v), a, b in zip(raw_custom, x[mask], y[mask])],
            hovertemplate='%{text}<extra></extra>'))
        buttons.append(dict(label=variant['label'], method='update',
            args=[{'visible': [i == j for j in range(len(variants))]}, {'title': variant['label']}]))
    figure.update_layout(title=variants[0]['label'], template='plotly_white',
        updatemenus=[dict(buttons=buttons, x=0, y=1.16)], height=850,
        xaxis_title='Lesion prevalence (%)' if logarithmic else 'UMAP 1',
        yaxis_title='Mean detected intensity' if logarithmic else 'UMAP 2')
    if logarithmic:
        figure.update_yaxes(type='log')
        figure.add_annotation(text=f'{int((~mask).sum())} peaks excluded from log plot', xref='paper', yref='paper', x=1, y=1.12, showarrow=False)
    write_plotly(figure, path)


def restricted_linkage(linkage: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Prune unselected leaves from the original tree, preserving merge distances."""
    if len(selected) < 2:
        return np.empty((0, 4))
    n = len(linkage)+1
    nodes = {int(peak): (i, 1) for i, peak in enumerate(selected)}
    result = []
    for i, (left, right, height, _) in enumerate(linkage):
        a, b = nodes.get(int(left)), nodes.get(int(right))
        if a is not None and b is not None:
            result.append([a[0], b[0], height, a[1]+b[1]])
            nodes[n+i] = (len(selected)+len(result)-1, a[1]+b[1])
        elif a is not None or b is not None:
            nodes[n+i] = a if a is not None else b
    return np.asarray(result, dtype=np.float64)


def plot_peak_heatmap(matrix: np.ndarray, order: np.ndarray, mz: np.ndarray, families: np.ndarray,
                      annotations: list[dict], path: Path, config: Mapping[str, Any],
                      *, title: str, kind: str = 'similarity', column_labels: list[str] | None = None,
                      linkage: np.ndarray | None = None, mz_ticks: bool = False) -> None:
    """Show all matrix cells and annotation strips in the saved hierarchical order."""
    order = np.asarray(order, dtype=np.int64)
    block = matrix[np.ix_(order, order)] if kind == 'similarity' else matrix[order]
    count = int(np.max(families))
    nstrips = len(annotations)
    figure = plt.figure(figsize=(18, 18) if linkage is not None else (16, 12), layout='constrained')
    widths = ([1.8] if linkage is not None else []) + [.24]*nstrips + [10, .35]
    grid = figure.add_gridspec(1, len(widths), width_ratios=widths)
    offset = 0
    if linkage is not None:
        axis = figure.add_subplot(grid[0, 0])
        pruned = restricted_linkage(linkage, order)
        if len(pruned):
            hierarchy.dendrogram(pruned, orientation='left', no_labels=True, color_threshold=0,
                                 above_threshold_color='#444444', ax=axis)
            axis.invert_yaxis()
        axis.axis('off')
        offset = 1
    for i, annotation in enumerate(annotations):
        axis = figure.add_subplot(grid[0, offset+i])
        values = np.asarray(annotation['values'])[order]
        if annotation['kind'] == 'intensity':
            values = np.log10(np.where(values > 0, values, np.nan))
        axis.imshow(values[:, None], aspect='auto', interpolation='nearest', **_color_options(annotation, count))
        axis.set(xticks=[], yticks=[])
        axis.set_title(annotation['label'] + (' (log10)' if annotation['kind'] == 'intensity' else ' (tails clipped)' if annotation['kind'] == 'association' else ''), rotation=90, fontsize=8, pad=8)
    axis = figure.add_subplot(grid[0, offset+nstrips])
    options = dict(cmap='coolwarm', vmin=-1, vmax=1) if kind == 'similarity' else dict(cmap='RdBu_r') if kind == 'embedding' else dict(cmap=config.get('peaks', {}).get('sequential_cmap', 'turbo'), vmin=0, vmax=100)
    image = axis.imshow(block, aspect='auto', interpolation='nearest', **options)
    figure.colorbar(image, cax=figure.add_subplot(grid[0, -1]), label={'similarity': 'Cosine similarity', 'embedding': 'Embedding coordinate', 'prevalence': 'Detection (%)'}[kind])
    ticks = np.arange(0, len(order), max(1, int(np.ceil(len(order)/40)))) if mz_ticks else []
    axis.set_yticks(ticks, [f'{mz[order[i]]:.4f}' for i in ticks], fontsize=6)
    axis.yaxis.tick_right()
    if kind == 'similarity':
        axis.set_xticks(ticks, [f'{mz[order[i]]:.4f}' for i in ticks], rotation=90, fontsize=6)
    elif column_labels is not None:
        axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=90, fontsize=6)
    boundaries = np.flatnonzero(np.diff(families[order]))+.5
    for boundary in boundaries:
        axis.axhline(boundary, color='black', linewidth=.35)
        if kind == 'similarity':
            axis.axvline(boundary, color='black', linewidth=.35)
    axis.set_title(title, pad=12)
    axis.set_ylabel('m/z (saved hierarchy order)' if mz_ticks else 'Peaks (saved hierarchy order)')
    _save(figure, path, config)


def plot_family_summary(matrix: np.ndarray, families: np.ndarray, path: Path, config: Mapping[str, Any]) -> None:
    """Display the complete family-by-family mean pairwise similarity matrix."""
    figure, axis = plt.subplots(figsize=(12, 10), layout='constrained')
    image = axis.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axis.set(xticks=np.arange(len(families)), yticks=np.arange(len(families)),
             xticklabels=families, yticklabels=families, xlabel='Family', ylabel='Family',
             title='Mean pairwise cosine similarity (diagonal excludes self pairs)')
    for labels in (axis.get_xticklabels(), axis.get_yticklabels()):
        for label, color in zip(labels, family_colors(len(families))):
            label.set_color(color)
    figure.colorbar(image, ax=axis)
    _save(figure, path, config)


def plot_interactive_heatmap(matrix: np.ndarray, order: np.ndarray, mz: np.ndarray,
                             annotations: list[dict], path: Path, title: str,
                             linkage: np.ndarray | None = None, *, identity_label: str = 'm/z',
                             identity_kind: str = 'mz', matrix_label: str = 'Cosine',
                             contributing: np.ndarray | None = None) -> None:
    """Export exact pairs with numeric row axes shared by strips and dendrogram."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    order = np.asarray(order, dtype=np.int64)
    n = len(order)
    offset = int(linkage is not None)
    columns = len(annotations)+1+offset
    figure = make_subplots(rows=1, cols=columns,
        column_widths=([.12] if offset else [])+[.025]*len(annotations)+[.8], shared_yaxes=True,
        horizontal_spacing=.005)
    if linkage is not None and n > 1:
        dendrogram = hierarchy.dendrogram(restricted_linkage(linkage, order), no_plot=True)
        xx, yy = [], []
        for ys, xs in zip(dendrogram['icoord'], dendrogram['dcoord']):
            xx.extend([*xs, None]); yy.extend([*((np.asarray(ys)-5)/10).tolist(), None])
        figure.add_trace(go.Scatter(x=xx, y=yy, mode='lines', line=dict(color='#444', width=1),
                                   hoverinfo='skip', showlegend=False), row=1, col=1)
        figure.update_xaxes(autorange='reversed', row=1, col=1)
    for col, annotation in enumerate(annotations, 1+offset):
        raw = np.asarray(annotation['values'])[order]
        local = {**annotation, 'values': raw}
        if annotation['kind'] == 'intensity':
            local['values'] = np.log10(np.where(raw > 0, raw, np.nan))
        values, options, _ = _plotly_colors(local, max(1, int(np.nanmax(annotation['values'])))
                                             if annotation['kind'] == 'family' else 1)
        options = {k.replace('cmin', 'zmin').replace('cmax', 'zmax'): v for k, v in options.items()}
        figure.add_trace(go.Heatmap(z=values[:, None], y=np.arange(n), x=[0],
            customdata=np.column_stack((mz[order], raw))[:, None, :], showscale=False,
            text=[[f'{identity_label} {format_number(m, identity_kind)}<br>Value {format_number(v, "integer" if annotation["kind"] == "family" else "float")}'] for m, v in zip(mz[order], raw)],
            hovertemplate='%{text}<extra></extra>', **options), row=1, col=col)
        suffix = ' (tails clipped)' if annotation['kind'] == 'association' else ' (log10)' if annotation['kind'] == 'intensity' else ''
        domain = figure.layout[f'xaxis{col if col > 1 else ""}'].domain
        figure.add_annotation(x=sum(domain)/2, y=1.01, xref='paper', yref='paper',
            text=annotation['label']+suffix, textangle=-90, xanchor='center', yanchor='bottom', showarrow=False)
    custom = np.empty((n, n, 2))
    custom[:, :, 0], custom[:, :, 1] = mz[order, None], mz[order][None, :]
    figure.add_trace(go.Heatmap(z=matrix[np.ix_(order, order)], x=np.arange(n), y=np.arange(n), customdata=custom,
        zmin=-1, zmax=1, colorscale='RdBu_r', colorbar=dict(title=matrix_label),
        text=[[f'Row {identity_label} {format_number(mz[r], identity_kind)}<br>Column {identity_label} {format_number(mz[c], identity_kind)}<br>{matrix_label} {format_number(matrix[r, c])}' + (f'<br>Contributing slides {int(contributing[r, c])}' if contributing is not None else '') for c in order] for r in order],
        hovertemplate='%{text}<extra></extra>'), row=1, col=columns)
    figure.update_yaxes(range=[n-.5, -.5], showticklabels=False)
    figure.update_xaxes(showticklabels=False)
    header = max(300, max((len(a['label'])+16)*8 for a in annotations)+40)
    figure.update_layout(title=dict(text=title, y=.99), height=header+800, margin=dict(t=header), template='plotly_white')
    write_plotly(figure, path)


def plot_family_correlations(families: np.ndarray, peaks: np.ndarray, signals: np.ndarray,
                             targets: list[str], path: Path, config: Mapping[str, Any]) -> None:
    """Compare individual peak associations to associations of mean family signal."""
    unique = np.unique(families)
    figure, axes = plt.subplots(2, 2, figsize=(24, 14), layout='constrained', squeeze=False)
    rng = np.random.default_rng(20260719)
    jitter = rng.uniform(-.3, .3, len(families))
    for t, (axis, label) in enumerate(zip(axes.flat, targets)):
        for i, (family, color) in enumerate(zip(unique, family_colors(len(unique)))):
            selected = families == family
            axis.scatter(i+jitter[selected], peaks[selected, t], s=7, alpha=.35, color=color, linewidths=0)
            axis.scatter(i, signals[i, t], s=90, marker='D', color=color, edgecolors='black', linewidths=1.2)
        axis.axhline(0, color='#777', linewidth=.7)
        values = np.r_[peaks[:, t], signals[:, t], 0.]
        values = values[np.isfinite(values)]
        padding = max(.03, float(np.ptp(values))*.06)
        axis.set(ylim=(values.min()-padding, values.max()+padding), title=label,
                 xticks=np.arange(len(unique)), xticklabels=unique, xlabel='Family', ylabel='Spearman correlation')
        for tick, color in zip(axis.get_xticklabels(), family_colors(len(unique))):
            tick.set_color(color)
    for axis in list(axes.flat)[len(targets):]:
        axis.set_visible(False)
    figure.suptitle('Held-out lesion associations: dots = peaks; outlined diamonds = mean family signal')
    _save(figure, path, config)


def queue_peak_plots(queue: Any, output: Path, frame: Any, coordinates: np.ndarray,
                     tree: Mapping[str, np.ndarray], weights: np.ndarray, prevalence: np.ndarray,
                     slides: np.ndarray, analysis: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    """Queue every requested view using one set of colors, annotations, and coordinates."""
    from dann.peak_analysis import family_similarity
    from dann.config import peak_analysis_settings
    from dann.peak_viewer import plot_full_similarity
    settings = peak_analysis_settings(analysis.get("peaks"))
    count = len(np.unique(tree['families']))
    variants = [dict(label='Family', kind='family', values=tree['families']),
                dict(label='m/z', kind='continuous', values=frame.mz.to_numpy()),
                dict(label='Lesion prevalence (%)', kind='prevalence', values=frame.lesion_prevalence_percent.to_numpy()),
                dict(label='Mean detected intensity', kind='intensity', values=frame.mean_detected_intensity.to_numpy()),
                dict(label='Embedding norm', kind='continuous', values=tree['norms'])]
    associations = [dict(label=f"{frame[f'label_{target}'].iloc[0]} Spearman (held-out)", kind='association',
                         values=frame[f'spearman_{target}'].to_numpy()) for target in contract['target_columns']]
    for variant in variants:
        if variant['kind'] != 'family':
            variant['cmap'] = settings['sequential_cmap']
    for variant in associations:
        variant.update(association_display(variant['values'], settings))
    (output/'ihc_display_settings.json').write_text(json.dumps({target: {k: v for k, v in variant.items()
        if k != 'values'} for target, variant in zip(contract['target_columns'], associations)}, indent=2))
    per_slide = [dict(label=f'{name}: prevalence (%)', kind='prevalence', cmap=settings['sequential_cmap'], values=prevalence[i]) for i, name in enumerate(slides)]
    x, y = coordinates.T
    for logarithmic, prefix in ((False, 'peak_umap'), (True, 'prevalence_intensity')):
        if logarithmic:
            x, y = frame.lesion_prevalence_percent.to_numpy(), frame.mean_detected_intensity.to_numpy()
        for name, panels, columns in (('annotations', variants, 3), ('ihc', associations, 2), ('slides', per_slide, 6)):
            queue.submit(plot_peak_scatter, x, y, panels, output/f'{prefix}_{name}.png', analysis, count,
                         logarithmic=logarithmic, columns=columns)
        queue.canvas_bytes = max(queue.canvas_bytes, 36*5*((len(per_slide)+5)//6)*int(analysis['figure_dpi'])**2*4)
        queue.submit(plot_interactive_scatter, x, y, frame.mz.to_numpy(), tree['families'],
                     variants+associations+per_slide, output/f'{prefix}.html', logarithmic=logarithmic)
    annotations = [variants[0], *variants[2:], *associations]
    common = (frame.mz.to_numpy(), tree['families'], annotations)
    queue.submit(plot_peak_heatmap, weights, tree['order'], *common, output/'embedding_matrix.png', analysis,
                 title='All peak embeddings', kind='embedding')
    queue.submit(plot_peak_heatmap, tree['similarity'], tree['order'], *common,
                 output/'full_similarity.png', analysis, title='All peak-pair cosine similarities', linkage=tree['linkage'])
    queue.submit(plot_full_similarity, tree['similarity'], tree['order'], frame.mz.to_numpy(), annotations,
                 tree['linkage'], output/'full_similarity.html')
    top_count = min(int(analysis['heatmap_top_peaks']), len(frame))
    active = np.argpartition(frame.occurrence_count.to_numpy(), -top_count)[-top_count:]
    top = tree['order'][np.isin(tree['order'], active)]
    frame.iloc[top][['peak_index', 'mz', 'family', 'occurrence_count']].to_csv(output/'top_similarity_membership.csv', index=False)
    queue.submit(plot_peak_heatmap, tree['similarity'], top, *common, output/f'top_{top_count}_similarity.png', analysis,
                 title=f'{top_count} most frequent peaks (all labeled rows)', linkage=tree['linkage'])
    queue.submit(plot_interactive_heatmap, tree['similarity'], top, frame.mz.to_numpy(), annotations,
                 output/f'top_{top_count}_similarity.html', f'{top_count} most frequent peaks', linkage=tree['linkage'])
    queue.submit(plot_peak_heatmap, prevalence.T, tree['order'], *common, output/'batch_prevalence.png', analysis,
                 title='Peak prevalence by slide (lesion pixels)', kind='prevalence', column_labels=slides.tolist())
    unique = np.unique(tree['families'])
    queue.submit(plot_family_summary, family_similarity(tree['similarity'], tree['families']), unique,
                 output/'family_similarity.png', analysis)
    for family in unique:
        members = tree['order'][tree['families'][tree['order']] == family]
        queue.submit(plot_peak_heatmap, tree['similarity'], members, *common,
                     output/f'family_{family:02d}_similarity.png', analysis,
                     title=f'Family {family}: all {len(members)} peaks', mz_ticks=True)
        queue.submit(plot_interactive_heatmap, tree['similarity'], members, frame.mz.to_numpy(), annotations,
                     output/f'family_{family:02d}_similarity.html', f'Family {family}: all {len(members)} peaks')


def plot_family_spatial(slides: list[dict], signal: np.ndarray, detection: np.ndarray, family: int,
                        limit: float, region: str, path: Path, config: Mapping[str, Any], percentile: float) -> None:
    """Render every slide in six tissue/lesion columns with shared physical scales.

    The legacy region argument is accepted but all regions are always displayed.
    """
    from dann.peak_analysis import spatial_grid
    from dann.config import peak_analysis_settings
    settings = peak_analysis_settings(config.get('peaks'))
    figure, axes = plt.subplots(len(slides), 6, figsize=(36, 4*len(slides)),
                                squeeze=False, layout='constrained')
    for row, slide in enumerate(slides):
        for offset, region in ((0, 'tissue'), (3, 'lesion')):
            empty = region == 'lesion' and slide['crop'] is None
            for col, (values, maximum, label) in enumerate(((signal, limit, 'Mean signal'), (detection, 100., 'Detection (%)'))):
                grid = spatial_grid(values, slide, region)
                image = axes[row, offset+col].imshow(grid, cmap=settings['sequential_cmap' if col == 0 else 'detection_cmap'],
                    vmin=0, vmax=maximum, interpolation='nearest')
                figure.colorbar(image, ax=axes[row, offset+col], extend='max' if col == 0 else 'neither', label=label)
            if empty:
                for axis in axes[row, offset:offset+3]:
                    axis.text(.5, .5, 'No lesion', ha='center', va='center', transform=axis.transAxes)
            else:
                bounds = np.asarray(slide[f'{region}_hes']) * slide['scale']
                axis = axes[row, offset+2]
                axis.imshow(slide['image'])
                axis.set_xlim(bounds[1, 0], bounds[0, 0])
                axis.set_ylim(bounds[0, 1], bounds[1, 1])
        axes[row, 0].set_ylabel(slide['name'])
        for axis in axes[row]:
            axis.set(xticks=[], yticks=[], facecolor='#dddddd')
    for axis, title in zip(axes[0], ['Tissue signal', 'Tissue detection', 'Tissue HES', 'Lesion signal', 'Lesion detection', 'Lesion HES']):
        axis.set_title(title)
    figure.suptitle(f'Family {family:02d} — signal saturated above tissue P{percentile:g}={limit:.4g}')
    _save(figure, path, config)


def queue_spatial_plots(queue: Any, output: Path, source: Path, slides: list[dict],
                        scores: Mapping[str, np.ndarray], limits: np.ndarray, families: np.ndarray,
                        analysis: Mapping[str, Any], settings: Mapping[str, Any],
                        performance: Mapping[str, Any]) -> None:
    """Share immutable HES/geometry arrays among all family/page render tasks."""
    with h5py.File(source, 'r') as handle:
        image_bytes = sum(handle[f"uns/spatial/{slide['name']}/images/HES"].size *
                          handle[f"uns/spatial/{slide['name']}/images/HES"].dtype.itemsize for slide in slides)
        grid_bytes = max(np.prod(slide['shape']) for slide in slides)*4*2
        check_host_budget(performance, int(image_bytes + 4*grid_bytes))
        coordinates = np.asarray(read_elem(handle['obsm/spatial']))
        if coordinates.shape != (scores['signal'].shape[0], 2) or not np.isfinite(coordinates).all():
            raise ValueError('HES spatial coordinates must be finite and row aligned.')
        for slide in slides:
            library = handle[f"uns/spatial/{slide['name']}"]
            slide['image'] = np.asarray(read_elem(library['images/HES']))
            scales = read_elem(library['scalefactors'])
            slide['scale'] = float(scales['tissue_HES_scalef'])
            xy = coordinates[slide['rows']]
            slide['tissue_hes'] = np.vstack((xy.min(axis=0), xy.max(axis=0)))
            selected = xy[slide['lesion']]
            slide['lesion_hes'] = np.vstack((selected.min(axis=0), selected.max(axis=0))) if len(selected) else None
    for i, family in enumerate(families):
        signal, detection = scores['signal'][:, i], scores['detection'][:, i]
        queue.submit(plot_family_spatial, slides, signal, detection, int(family), float(limits[i]), 'all',
                     output/f'family_{family:02d}_spatial.png', analysis, settings['signal_percentile'])
    # Reserve the actual tall canvas plus draw/encoding buffers through RenderQueue.
    canvas = 36 * 4 * len(slides) * int(analysis['figure_dpi'])**2 * 4
    queue.canvas_bytes = max(queue.canvas_bytes, int(4*grid_bytes), int(canvas))


def plot_cd8_family_comparison(slides: list[dict], cd8: np.ndarray, signal: np.ndarray,
                               families: np.ndarray, limits: np.ndarray, cd8_limit: float,
                               cd8_label: str, region: str, path: Path,
                               config: Mapping[str, Any]) -> None:
    """Compare observed CD8 and all family signals with shared column scales."""
    from dann.peak_analysis import spatial_grid
    columns = len(families)+1
    scale = min(1., 8000/(4*max(columns, len(slides))*int(config['figure_dpi'])))
    figure, axes = plt.subplots(len(slides), columns, figsize=(4*columns*scale, 4*len(slides)*scale), squeeze=False)
    # Fixed margins avoid a global layout solver on more than 1,400 panels.
    figure.subplots_adjust(left=.012, right=.998, bottom=.02, top=.985, wspace=.09, hspace=.09)
    titles = [cd8_label, *[f'Family {family:02d}' for family in families]]
    for col, (title, limit) in enumerate(zip(titles, np.r_[cd8_limit, limits])):
        values = cd8 if col == 0 else signal[:, col-1]
        for row, slide in enumerate(slides):
            axis = axes[row, col]
            grid = spatial_grid(values, slide, region)
            image = axis.imshow(grid, cmap='turbo', vmin=0, vmax=limit, interpolation='nearest')
            axis.set(xticks=[], yticks=[], facecolor='#dddddd')
            if col == 0:
                axis.set_ylabel(slide['name'], fontsize=max(2, 11*scale))
            if region == 'lesion' and slide['crop'] is None:
                axis.text(.5, .5, 'No lesion', ha='center', va='center', transform=axis.transAxes)
            elif col == 0 and not np.isfinite(grid).any():
                axis.text(.5, .5, 'No observed CD8', ha='center', va='center', transform=axis.transAxes)
        axes[0, col].set_title(title, fontsize=max(2, 13*scale))
        # One colorbar per column; tissue and lesion use the identical limits.
        bounds = axes[-1, col].get_subplotspec().get_position(figure)
        color_axis = figure.add_axes([bounds.x0, .009, bounds.width, .0025])
        colorbar = figure.colorbar(image, cax=color_axis, orientation='horizontal', extend='max')
        colorbar.ax.tick_params(labelsize=max(2, 10*scale))
        colorbar.set_label('Observed density' if col == 0 else 'Mean signal', size=max(2, 10*scale))
    figure.suptitle(f'Observed CD8 and mean family signal — {region}; common column scales across slides and regions', fontsize=max(3, 18*scale), y=.998)
    _save(figure, path, config)
    from dann.cd8_viewer import write_cd8_viewer
    write_cd8_viewer(slides, cd8, signal, families, limits, cd8_limit, cd8_label, region, path.with_suffix('.html'))


def queue_cd8_comparisons(queue: Any, output: Path, slides: list[dict], cd8: np.ndarray,
                          cd8_label: str, scores: Mapping[str, np.ndarray], limits: np.ndarray,
                          families: np.ndarray, analysis: Mapping[str, Any], settings: Mapping[str, Any]) -> None:
    """Queue the two wide canvases separately from smaller peak plots."""
    from dann.peak_analysis import signal_limits
    geometry = [{key: value for key, value in slide.items()
                 if key in ('name', 'rows', 'x', 'y', 'shape', 'lesion', 'crop')} for slide in slides]
    cd8_limit = float(signal_limits(cd8[:, None], settings['signal_percentile'])[0])
    for region in ('tissue', 'lesion'):
        queue.submit(plot_cd8_family_comparison, geometry, cd8, scores['signal'], families, limits,
                     cd8_limit, cd8_label, region, output/f'cd8_families_{region}.png', analysis)
    scale = min(1., 8000/(4*max(len(families)+1, len(slides))*int(analysis['figure_dpi'])))
    queue.canvas_bytes = int(4*(len(families)+1)*4*len(slides)*int(analysis['figure_dpi'])**2*4*scale**2)
    metadata = dict(cd8_source='observed; checkpoint target normalization and masks', cd8_label=cd8_label,
        cmap='turbo', scale='linear', vmin=0., cd8_vmax=cd8_limit, family_vmax=limits.tolist(),
        percentile=settings['signal_percentile'], family_ids=families.tolist(),
        slides=[s['name'] for s in slides], width_inches=4*(len(families)+1)*scale, height_inches=4*len(slides)*scale,
        max_edge_pixels=8000, detail_format='lossless native-grid WebP',
        dpi=int(analysis['figure_dpi']))
    (output/'cd8_family_comparison_settings.json').write_text(json.dumps(metadata, indent=2))

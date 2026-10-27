"""Weighted objective curves below their fit metrics, including old histories."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from dann.curve_layout import FIGURE_ROWS, HEADER_HEIGHT, FOOTER_HEIGHT, figure_size
from dann.selection import selection_metric


def weighted_history(history: pd.DataFrame, config: Mapping[str, Any] | None) -> pd.DataFrame:
    """Recover only aggregate contributions available from a run's saved weights.

    Per-target likelihoods cannot be recovered from historical fit metrics. Missing
    entries in existing columns stay missing, including after a resumed run.
    """
    history = history.copy()
    if config is None:
        return history
    weights = config["loss"]
    he_enabled = config.get("model", {}).get("heads", {}).get("he", {}).get("enabled", False)
    for split in ("train", "validation"):
        for branch, weight in (("biology", "biology_weight"), ("hurdle", "biology_weight"),
                              ("positive", "biology_weight"), ("batch", "batch_weight"),
                              ("he", "he_weight")):
            source = f"{split}_{branch}_loss" if branch != "he" else f"{split}_he_mse"
            destination = f"{split}_{branch}_weighted_loss"
            if destination not in history and source in history and weight in weights:
                history[destination] = history[source] * float(weights[weight])
        supervised = f"{split}_supervised_objective"
        ihc, he = f"{split}_biology_weighted_loss", f"{split}_he_weighted_loss"
        if supervised not in history and ihc in history:
            if not he_enabled:
                history[supervised] = history[ihc]
            elif he in history:
                history[supervised] = history[ihc] + history[he]
    return history


# Target colors are stable across figures; branch and adversarial colors differ.
COLORS = dict(ihc='C0', he='C2', batch='C5', cd8='C0', tumor='C1', collagen='C2',
              stroma='C4', hurdle='#333333', positive='C3')
LABELS = dict(ihc='IHC', he='H&E', batch='Adversarial', cd8='CD8', tumor='Tumor',
              collagen='Collagen', stroma='Stroma', hurdle='Hurdle', positive='Positive')
TARGET_ORDER = ('cd8', 'tumor', 'collagen', 'stroma')


def plot_loss_curves(history_path: Path, output_path: Path, figure_dpi: int,
                     best_epoch: int | float | None = None, *,
                     config: Mapping[str, Any] | None = None,
                     selection: Mapping[str, Any] | None = None) -> None:
    """Read history once and regenerate both overall and IHC figures."""
    plot_objective_history(pd.read_csv(history_path), output_path, figure_dpi,
                           best_epoch=best_epoch, config=config, selection=selection)


def _values(history, key):
    return np.asarray(history.get(key, pd.Series(np.nan, index=history.index)), dtype=float)


def _fit(key, title, color='C0'):
    return _lines(title, [(f'train_{key}', 'Train', color, '-'),
                          (f'validation_{key}', 'Validation', color, '--')])


def _lines(title, series, **kwargs):
    return dict(kind='lines', title=title, series=series, **kwargs)


def _gradient_panels(prefix, names, title):
    from itertools import combinations
    pairs = list(combinations(names, 2))
    series = lambda metric: [(f'train_{prefix}_{n}_{metric}_mean', LABELS[n], COLORS[n], '-') for n in names]
    return [
        _lines(f'{title}: weighted gradient magnitude', series('norm'), log=True),
        dict(kind='shares', title=f'{title}: relative magnitude (%)', series=series('share')),
        dict(kind='heatmap', title=f'{title}: pairwise cosine (−1 … +1)',
             series=[(f'train_{prefix}_{a}__{b}_cosine_mean', f'{LABELS[a]} / {LABELS[b]}') for a, b in pairs]),
        dict(kind='heatmap', title=f'{title}: conflict frequency (0 … 1)', conflict=True,
             series=[(f'train_{prefix}_{a}__{b}_conflict_mean', f'{LABELS[a]} / {LABELS[b]}') for a, b in pairs]),
        _lines(f'{title}: retained magnitude', [(f'train_{prefix}_retained_mean', '‖sum‖ / Σ‖component‖', '#333333', '-')]),
        _lines(f'{title}: signed contributions', series('signed'), zero=True),
    ]


def _render(history, rows, output_path, dpi, kind, title, best_epoch, metric, components):
    """Use the same isolated style in training, background workers, and analysis."""
    defaults = {key: value for key, value in mpl.rcParamsDefault.items() if key != 'backend'}
    defaults.update({'axes.titlesize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
                     'legend.fontsize': 12, 'figure.titlesize': 18})
    with mpl.rc_context(defaults):
        _render_snapshot(history, rows, output_path, dpi, kind, title, best_epoch, metric, components)


def _render_snapshot(history, rows, output_path, dpi, kind, title, best_epoch, metric, components):
    """Render explicit rows and column spans, preserving gaps and real zeros."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    width, height = figure_size(kind)
    figure = plt.figure(figsize=(width, height))
    grid = figure.add_gridspec(FIGURE_ROWS[kind], 4, left=.075, right=.985,
                              bottom=FOOTER_HEIGHT/height, top=1-HEADER_HEIGHT/height,
                              hspace=.65, wspace=.48)
    figure.suptitle(title, y=1-.18/height)
    figure.text(.5, 1-.65/height,
                'Gradients summarize sampled training updates. Magnitude shares compare weighted gradients; retained magnitude measures cancellation.',
                ha='center', va='top', fontsize=12)
    figure.text(.5, 1-.98/height,
                f'Signed contributions may be negative or >1. Selection metric: {metric or "unavailable"}. Missing observations remain unavailable.',
                ha='center', va='top', fontsize=12)
    styles = [Line2D([], [], color='#333333', linestyle=style, label=label)
              for style, label in [('-', 'Train'), ('--', 'Validation')]]
    if best_epoch is not None:
        styles.append(Line2D([], [], color='red', linestyle='--',
                             label=f'Selected checkpoint: epoch {int(best_epoch)}'))
    figure.legend(handles=styles, loc='upper center', bbox_to_anchor=(.5, 1-1.32/height),
                  ncol=len(styles), frameon=False)
    figure.legend(handles=[Line2D([], [], color=COLORS[n], linewidth=3, label=LABELS[n]) for n in components],
                  loc='upper center', bbox_to_anchor=(.5, 1-1.78/height), ncol=len(components), frameon=False)
    epochs = _values(history, 'epoch')
    for row_index, panels in enumerate(rows):
        column = 0
        for panel in panels:
            span = panel.get('span', 1) if panel is not None else 1
            axis = figure.add_subplot(grid[row_index, column:column+span])
            column += span
            if panel is None:
                axis.set_visible(False)
                continue
            kind = panel['kind']
            available = False
            if kind in {'lines', 'shares'}:
                data = [_values(history, key) for key, *_ in panel['series']]
                if kind == 'shares':
                    # Mask incomplete comparisons instead of inventing zero shares.
                    if data:
                        data = np.asarray(data) * 100
                        data[:, ~np.isfinite(data).all(axis=0)] = np.nan
                        available = bool(np.isfinite(data).any())
                        axis.stackplot(epochs, data, colors=[s[2] for s in panel['series']], alpha=.8)
                    axis.set_ylim(0, 100)
                    axis.yaxis.set_major_formatter(PercentFormatter())
                else:
                    zeros = False
                    for values, (_, label, color, style) in zip(data, panel['series']):
                        available |= bool(np.isfinite(values).any())
                        if panel.get('log'):
                            zeros |= bool((values == 0).any())
                            values = np.where(values > 0, values, np.nan)
                        axis.plot(epochs, values, label=label, color=color, linestyle=style,
                                  marker='.', markersize=3)
                    if panel.get('log'):
                        axis.set_yscale('log')
                        if not any(np.any(v > 0) for v in data):
                            axis.set_ylim(.1, 1.)
                        if zeros:
                            axis.text(.02, .02, 'Zeros omitted on log scale', transform=axis.transAxes,
                                      fontsize=12, color='dimgray', va='bottom')
                    if panel.get('zero'):
                        axis.axhline(0, color='gray', linewidth=.7)
                if panel.get('legend'):
                    axis.legend(loc='best')
            elif kind == 'heatmap':
                data = np.asarray([_values(history, key) for key, _ in panel['series']])
                available = bool(data.size and np.isfinite(data).any())
                if data.size:
                    edges = (np.r_[epochs[0]-.5, epochs[0]+.5] if len(epochs) == 1 else
                             np.r_[epochs[0]-(epochs[1]-epochs[0])/2, (epochs[:-1]+epochs[1:])/2,
                                   epochs[-1]+(epochs[-1]-epochs[-2])/2])
                    axis.pcolormesh(edges, np.arange(len(data)+1)-.5, np.ma.masked_invalid(data),
                                    cmap=panel.get('cmap', 'Reds' if panel.get('conflict') else 'RdBu'),
                                    vmin=0 if panel.get('conflict') else -1, vmax=1, shading='flat')
                axis.set_yticks(np.arange(len(panel['series'])), [label for _, label in panel['series']])
            if not available:
                axis.text(.5, .5, 'Unavailable in saved history', transform=axis.transAxes,
                          ha='center', va='center', fontsize=12, color='dimgray')
            if best_epoch is not None:
                axis.axvline(float(best_epoch), color='red', linestyle='--', linewidth=1, zorder=5)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
            if len(epochs) == 1:
                selected = float(best_epoch) if best_epoch is not None else epochs[0]
                axis.set_xlim(min(epochs[0], selected)-.5, max(epochs[0], selected)+.5)
            axis.set_title(fill(panel['title'], width=34 * span), pad=10)
            axis.grid(alpha=.2)
    try:
        figure.savefig(output_path, dpi=dpi)
    finally:
        plt.close(figure)


def plot_objective_history(history: pd.DataFrame, output_path: Path, figure_dpi: int,
                           best_epoch: int | float | None = None, *,
                           config: Mapping[str, Any] | None = None,
                           selection: Mapping[str, Any] | None = None) -> None:
    """Render both figures from the same immutable snapshot, including old histories.

    Used by synchronous training, the asynchronous artifact worker, and analysis.
    The sibling IHC image is atomically replaced, including when the caller already
    wraps the overall destination in an atomic temporary filename.
    """
    from dann.training_artifacts import atomic_output
    history = weighted_history(history, config)
    if selection is not None:
        best_epoch = selection.get('best_epoch') if best_epoch is None else best_epoch
        metric = selection.get('selection_metric')
    else:
        metric = selection_metric(config) if config is not None else None
        if best_epoch is None and 'selected_epoch' in history:
            selected = history['selected_epoch'].dropna()
            best_epoch = float(selected.iloc[-1]) if len(selected) else None
    he = (config.get('model', {}).get('heads', {}).get('he', {}).get('enabled', False) if config is not None
          else any('he_weighted_loss' in c or 'he_mse' in c or 'he_r2' in c for c in history))
    overall = ['ihc', *(['he'] if he else []), 'batch']
    gradients = _gradient_panels('gradient_overall', overall, 'Objectives')
    gradients[2]['cmap'] = 'coolwarm'
    rows = [
        [_fit('hurdle_acc', 'IHC balanced hurdle accuracy (macro average)', COLORS['hurdle']),
         _fit('mean_r2', 'IHC positive-logit R² (macro average)', COLORS['positive']),
         _fit('he_r2', 'H&E R² (variance-weighted features)', COLORS['he']) if he else None,
         _fit('batch_accuracy', 'Discriminator accuracy', COLORS['batch'])],
        [_fit('hurdle_weighted_loss', 'IHC hurdle objective', COLORS['hurdle']),
         _fit('positive_weighted_loss', 'IHC positive objective (location and scale)', COLORS['positive']),
         _fit('he_weighted_loss', 'H&E objective', COLORS['he']) if he else None,
         _fit('batch_weighted_loss', 'Batch discriminator objective', COLORS['batch'])],
        [_fit('total_loss', 'Total objective'),
         _fit('supervised_objective', 'Supervised objective' + (' — early stopping' if metric == 'supervised_objective' else '')),
         _lines('Actual GRL strength', [('train_gradient_grl_strength_mean', 'All updates', COLORS['batch'], '-'),
                                      ('train_gradient_sampled_grl_strength_mean', 'Sampled updates', 'C4', '--')], legend=True),
         _fit('batch_aware_objective', 'Batch-aware objective' + (' — early stopping' if metric == 'batch_aware_objective' else ''))],
        [*gradients[:3],
         _lines('Validation latent participation ratio', [('validation_latent_participation_ratio', 'Participation ratio', 'C0', '--')])],
        [*gradients[4:], gradients[3],
         _lines('Validation latent total variance', [('validation_latent_total_variance', 'Total variance', 'C0', '--')])],
        [_lines('Whole-model pre-clipping norm (all updates)', [('train_gradient_preclip_norm_mean', 'Mean norm', '#333333', '-')], log=True),
         _lines('Clipping activity (all updates)', [('train_gradient_clip_multiplier_mean', 'Mean multiplier', '#333333', '-'),
                                                  ('train_gradient_clipped_mean', 'Fraction clipped', 'C3', '-')], legend=True)],
    ]
    ihc_rows = []
    for key, title in [('hurdle_acc', 'balanced hurdle accuracy'), ('hurdle_weighted_loss', 'hurdle objective'),
                       ('mean_r2', 'positive-logit R²'), ('positive_weighted_loss', 'positive objective (location and scale)')]:
        ihc_rows.append([_fit(f'{key}_{name}', f'{LABELS[name]} {title}', COLORS[name]) for name in TARGET_ORDER])
    gradients = _gradient_panels('gradient_ihc', TARGET_ORDER, 'IHC targets')
    ihc_rows += [gradients[:4], [dict(panel, span=2) for panel in gradients[4:]]]
    branches = {name: _gradient_panels(f'gradient_{name}', ['hurdle', 'positive'], LABELS[name]) for name in TARGET_ORDER}
    ihc_rows += [
        [branches[name][0] for name in TARGET_ORDER],
        [branches[name][1] for name in TARGET_ORDER],
        [_lines(f'{LABELS[name]}: hurdle / positive cosine',
                [(f'train_gradient_{name}_hurdle__positive_cosine_mean', 'Cosine', COLORS[name], '-')], zero=True)
         for name in TARGET_ORDER],
        [dict(branches[name][4], series=[(f'train_gradient_{name}_retained_mean', 'Retained magnitude', COLORS[name], '-')])
         for name in TARGET_ORDER],
    ]
    output_path = Path(output_path)
    with atomic_output(output_path.with_name('ihc_curves.png')) as temporary:
        _render(history, ihc_rows, temporary, figure_dpi, 'ihc',
                'IHC target fit, weighted objectives, and gradient composition', best_epoch, metric,
                [*TARGET_ORDER, 'hurdle', 'positive'])
    _render(history, rows, output_path, figure_dpi, 'loss',
            'Overall objectives, shared gradients, and latent diversity', best_epoch, metric,
            [*overall, 'hurdle', 'positive'])

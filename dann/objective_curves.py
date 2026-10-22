"""Weighted objective curves below their fit metrics, including old histories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

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


# Shared hues across objective, target, and branch panels. Line style denotes split.
COLORS = dict(ihc='C0', he='C2', batch='C3', cd8='C0', tumor='C1', collagen='C2',
              stroma='C4', hurdle='C0', positive='C1')
LABELS = dict(ihc='IHC', he='H&E', batch='Adversarial', cd8='CD8', tumor='Tumor',
              collagen='Collagen', stroma='Stroma', hurdle='Hurdle', positive='Positive')


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
    return dict(kind='lines', title=title, series=[
        (f'train_{key}', 'Train', color, '-'), (f'validation_{key}', 'Validation', color, '--')])


def _lines(title, series, **kwargs):
    return dict(kind='lines', title=title, series=series, **kwargs)


def _gradient_panels(prefix, names, title):
    from itertools import combinations
    pairs = list(combinations(names, 2))
    series = lambda metric: [(f'train_{prefix}_{n}_{metric}_mean', LABELS.get(n, n), COLORS.get(n, 'C0'), '-') for n in names]
    return [
        _lines(f'{title}: weighted gradient magnitude', series('norm'), log=True),
        dict(kind='shares', title=f'{title}: relative magnitude (%)', series=series('share')),
        dict(kind='heatmap', title=f'{title}: pairwise cosine (−1 … +1)',
             series=[(f'train_{prefix}_{a}__{b}_cosine_mean', f'{LABELS.get(a,a)} / {LABELS.get(b,b)}') for a, b in pairs]),
        dict(kind='heatmap', title=f'{title}: conflict frequency (0 … 1)', conflict=True,
             series=[(f'train_{prefix}_{a}__{b}_conflict_mean', f'{LABELS.get(a,a)} / {LABELS.get(b,b)}') for a, b in pairs]),
        _lines(f'{title}: retained magnitude', [(f'train_{prefix}_retained_mean', '‖sum‖ / Σ‖component‖', 'C0', '-')]),
        _lines(f'{title}: signed contribution', series('signed'), zero=True),
    ]


def _render(history, panels, output_path, dpi, title, best_epoch):
    """Use the same isolated style in training, background workers, and analysis."""
    defaults = {key: value for key, value in mpl.rcParamsDefault.items() if key != 'backend'}
    with mpl.rc_context(defaults):
        _render_snapshot(history, panels, output_path, dpi, title, best_epoch)


def _render_snapshot(history, panels, output_path, dpi, title, best_epoch):
    """Render one four-column figure, preserving missing values and real zeros."""
    from matplotlib.ticker import PercentFormatter
    panels = list(panels) + [None] * (-len(panels) % 4)
    rows = len(panels) // 4
    height = rows * 3.5 + 1.4
    figure, axes = plt.subplots(rows, 4, figsize=(20, height), squeeze=False)
    figure.subplots_adjust(left=.065, right=.985, bottom=.5/height, top=1-1.6/height,
                           hspace=.48, wspace=.32)
    figure.suptitle(title, fontsize=12, y=1-.12/height)
    epochs = _values(history, 'epoch')
    for axis, panel in zip(axes.flat, panels):
        if panel is None:
            axis.set_visible(False)
            continue
        kind = panel['kind']
        available = False
        if kind in {'lines', 'shares'}:
            data = [_values(history, key) for key, *_ in panel['series']]
            if kind == 'shares':
                available = bool(data) and bool(np.isfinite(data).any())
                # Mask incomplete comparisons instead of inventing zero shares.
                if data:
                    data = np.asarray(data) * 100
                    data[:, ~np.isfinite(data).all(axis=0)] = np.nan
                    axis.stackplot(epochs, data, labels=[s[1] for s in panel['series']],
                                   colors=[s[2] for s in panel['series']], alpha=.8)
                axis.set_ylim(0, 100)
                axis.yaxis.set_major_formatter(PercentFormatter())
            else:
                for values, (key, label, color, style) in zip(data, panel['series']):
                    available |= bool(np.isfinite(values).any())
                    if panel.get('log'):
                        # Zero remains in history; annotate its presence on the logarithmic plot.
                        if (values == 0).any():
                            label += ' (zeros omitted)'
                        values = np.where(values > 0, values, np.nan)
                    axis.plot(epochs, values, label=label, color=color, linestyle=style,
                              marker='.', markersize=3)
                if panel.get('log') and any(np.any(v > 0) for v in data):
                    axis.set_yscale('log')
                if panel.get('zero'):
                    axis.axhline(0, color='gray', linewidth=.7)
                for level in panel.get('references', []):
                    axis.axhline(level, color='gray', linestyle=':', linewidth=.7)
            if panel['series']:
                axis.legend(fontsize=8, loc='best')
        elif kind == 'heatmap':
            data = np.asarray([_values(history, key) for key, _ in panel['series']])
            available = bool(data.size and np.isfinite(data).any())
            if data.size:
                edges = (np.r_[epochs[0]-.5, epochs[0]+.5] if len(epochs) == 1 else
                         np.r_[epochs[0]-(epochs[1]-epochs[0])/2, (epochs[:-1]+epochs[1:])/2,
                               epochs[-1]+(epochs[-1]-epochs[-2])/2])
                axis.pcolormesh(edges, np.arange(len(data)+1)-.5, np.ma.masked_invalid(data),
                                cmap='Reds' if panel.get('conflict') else 'RdBu',
                                vmin=0 if panel.get('conflict') else -1, vmax=1, shading='flat')
                axis.set_yticks(np.arange(len(data)), [label for _, label in panel['series']], fontsize=8)
        else:  # Reliability uses the latest saved epoch, never an earlier nonmissing epoch.
            axis.plot([0, 1], [0, 1], color='gray', linestyle=':')
            if len(history):
                event = panel['event']
                prediction = [_values(history, f'validation_cd8_{event}_bin{b}_predicted')[-1] for b in range(10)]
                observed = [_values(history, f'validation_cd8_{event}_bin{b}_observed')[-1] for b in range(10)]
                counts = np.asarray([_values(history, f'validation_cd8_{event}_bin{b}_count')[-1] for b in range(10)])
                available = bool(np.isfinite(prediction).any())
                axis.plot(prediction, observed, color=panel['color'], linewidth=1)
                axis.scatter(prediction, observed, s=10 + 60*np.nan_to_num(counts)/max(np.nan_to_num(counts).max(), 1),
                             color=panel['color'])
            axis.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted probability', ylabel='Observed frequency')
        if not available:
            axis.text(.5, .5, 'Unavailable in saved history', transform=axis.transAxes,
                      ha='center', va='center', fontsize=9, color='dimgray')
        if kind != 'reliability':
            if best_epoch is not None:
                axis.axvline(float(best_epoch), color='red', linestyle='--', linewidth=1, zorder=5)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
            axis.set_xlabel('Epoch')
            if len(epochs) == 1:
                selected = float(best_epoch) if best_epoch is not None else epochs[0]
                axis.set_xlim(min(epochs[0], selected)-.5, max(epochs[0], selected)+.5)
        axis.set_title(panel['title'], fontsize=10)
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
    from dann.monitoring import coverage_key
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
    columns = config['data']['target_columns'] if config is not None else ['Density_CD8', 'Density_Tumor', 'Density_Collagen', 'Density_Stroma']
    names = [c.removeprefix('Density_').lower() for c in columns]
    order = [n for n in ('cd8', 'tumor', 'collagen', 'stroma') if n in names]
    order += [n for n in names if n not in order]
    panels = [
        _fit('total_loss', 'Total objective'),
        _fit('batch_accuracy', 'Discriminator accuracy', COLORS['batch']),
        _fit('he_r2', 'H&E R² (variance-weighted features)', COLORS['he']) if he else None,
        None,
        _fit('supervised_objective', 'Supervised objective' + (' — early stopping' if metric == 'supervised_objective' else '')),
        _fit('batch_weighted_loss', 'Batch discriminator objective', COLORS['batch']),
        _fit('he_weighted_loss', 'H&E objective', COLORS['he']) if he else None,
        None,
        _fit('hurdle_acc', 'IHC balanced hurdle accuracy (macro average)'),
        _fit('mean_r2', 'IHC positive-logit R² (macro average)'),
        _fit('hurdle_weighted_loss', 'IHC hurdle objective'),
        _fit('positive_weighted_loss', 'IHC positive objective (location and scale)'),
    ]
    overall = ['ihc', *(['he'] if he else []), 'batch']
    panels += _gradient_panels('gradient_overall', overall, 'Objectives')
    panels += [
        _lines('Whole-model pre-clipping norm (all updates)', [('train_gradient_preclip_norm_mean', 'Mean norm', 'C0', '-')], log=True),
        _lines('Clipping activity (all updates)', [('train_gradient_clip_multiplier_mean', 'Mean multiplier', 'C0', '-'),
                                                 ('train_gradient_clipped_mean', 'Fraction clipped', 'C3', '-')]),
        _lines('Actual GRL strength', [('train_gradient_grl_strength_mean', 'All updates', 'C3', '-'),
                                       ('train_gradient_sampled_grl_strength_mean', 'Sampled updates', 'C4', '--')]),
        _lines('Validation latent total variance', [('validation_latent_total_variance', 'Total variance', 'C0', '-')]),
        _lines('Validation latent participation ratio', [('validation_latent_participation_ratio', 'Participation ratio', 'C0', '-')]),
        _lines('Validation latent observations', [('validation_latent_count', 'Core pixels', 'C0', '-')]),
    ]
    definitions = (config or {}).get('monitoring_definitions', {})
    levels = definitions.get('coverage_levels')
    if levels is None:
        # History alone is sufficient even if the analysis config later changes.
        levels = [float(c.removeprefix('validation_cd8_coverage_').replace('p', '.'))/100
                  for c in history if c.startswith('validation_cd8_coverage_')]
        levels = levels or [.5, .9]
    events = ['presence', 'exceedance_q50', 'exceedance_q75', 'exceedance_q90']
    labels = ['Presence']
    for i, q in enumerate((50, 75, 90)):
        stored = _values(history, f'validation_cd8_exceedance_q{q}_cutoff')
        cutoff = stored[np.isfinite(stored)][-1] if np.isfinite(stored).any() else definitions.get('cutoffs', [None]*3)[i]
        labels.append(f'Exploratory train q{q}: > {cutoff:.5g}' if cutoff is not None else f'Exploratory train q{q}: unavailable')
    panels += [
        _lines('Validation CD8 presence Brier score', [('validation_cd8_presence_brier', 'Presence', 'C0', '-')]),
        _lines('CD8 unconditional exceedance Brier scores', [(f'validation_cd8_{e}_brier', label, f'C{i}', '-')
                                                            for i, (e, label) in enumerate(zip(events[1:], labels[1:]), 1)]),
        _lines('CD8 positive-branch interval coverage', [(f'validation_cd8_coverage_{coverage_key(level)}', f'{level:.0%} interval', f'C{i}', '-')
                                                       for i, level in enumerate(levels)], references=levels),
        _lines('Validation CD8 observations', [('validation_cd8_presence_count', 'Valid CD8', 'C0', '-'),
                                               ('validation_cd8_positive_count', 'Positive CD8', 'C1', '-')]),
    ]
    panels += [dict(kind='reliability', event=event, title=f'Latest-epoch reliability\n{label}', color=f'C{i}')
               for i, (event, label) in enumerate(zip(events, labels))]
    ihc_panels = []
    for name in order:
        label, color = LABELS.get(name, name), COLORS.get(name, 'C0')
        ihc_panels += [
            _fit(f'hurdle_acc_{name}', f'{label} balanced hurdle accuracy', color),
            _fit(f'mean_r2_{name}', f'{label} positive-logit R²', color),
            _fit(f'hurdle_weighted_loss_{name}', f'{label} hurdle objective', color),
            _fit(f'positive_weighted_loss_{name}', f'{label} positive objective (location and scale)', color),
        ]
    ihc_panels += _gradient_panels('gradient_ihc', order, 'IHC targets') + [None, None]
    for name in order:
        prefix, label = f'gradient_{name}', LABELS.get(name, name)
        branches = _gradient_panels(prefix, ['hurdle', 'positive'], label)
        ihc_panels += [branches[0], branches[1],
                       _lines(f'{label}: hurdle / positive cosine', [(f'train_{prefix}_hurdle__positive_cosine_mean', 'Cosine', COLORS.get(name, 'C0'), '-')], zero=True),
                       branches[4]]
    selection_text = f'Selected checkpoint: epoch {int(best_epoch)} (red dashed)' if best_epoch is not None else 'Selected checkpoint: unavailable'
    header = ('Gradient statistics estimate sampled training updates; not percentages of AdamW parameter updates.\n'
              'Magnitude shares compare weighted gradients; retained magnitude measures cancellation; signed contributions may be negative or >1.\n'
              f'{selection_text}; selection metric: {metric or "unavailable"}. Missing observations remain unavailable.')
    output_path = Path(output_path)
    with atomic_output(output_path.with_name('ihc_curves.png')) as temporary:
        _render(history, ihc_panels, temporary, figure_dpi,
                'IHC target fit, weighted objectives, and gradient composition\n' + header, best_epoch)
    _render(history, panels, output_path, figure_dpi,
            'Overall objectives, shared gradients, latent diversity, and CD8 probability quality\n' + header, best_epoch)

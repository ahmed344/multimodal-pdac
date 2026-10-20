"""Weighted objective curves below their fit metrics, including old histories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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


def plot_loss_curves(
    history_path: Path,
    output_path: Path,
    figure_dpi: int,
    best_epoch: int | float | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    selection: Mapping[str, Any] | None = None,
) -> None:
    """Render four columns with fit metrics above their weighted objectives.

    ``config`` supplies resolved weights and target order. ``selection`` supplies
    the actual selected epoch, metric, and first eligible epoch. No minimum-loss
    epoch is inferred: older runs without selection metadata have no marker.
    """
    history = weighted_history(pd.read_csv(history_path), config)
    if selection is not None:
        if best_epoch is None:
            best_epoch = selection.get("best_epoch")
        metric = selection.get("selection_metric")
    else:
        metric = selection_metric(config) if config is not None else None
        if best_epoch is None and "selected_epoch" in history:
            selected = history["selected_epoch"].dropna()
            best_epoch = float(selected.iloc[-1]) if len(selected) else None
    he_enabled = (config.get("model", {}).get("heads", {}).get("he", {}).get("enabled", False)
                  if config is not None else any("he_weighted_loss" in c or "he_mse" in c or "he_r2" in c for c in history))
    columns = (config["data"]["target_columns"] if config is not None else
               ["Density_CD8", "Density_Tumor", "Density_Collagen", "Density_Stroma"])
    names = {name.removeprefix("Density_").lower(): name.removeprefix("Density_") for name in columns}
    order = [name for name in ("cd8", "tumor", "collagen", "stroma") if name in names]
    order += [name for name in names if name not in order]
    panels = [
        ("total_loss", "Total objective"),
        ("batch_accuracy", "Discriminator accuracy"),
        ("he_r2", "H&E R² (variance-weighted features)") if he_enabled else None,
        None,
        ("supervised_objective", "Supervised objective" + (" — early stopping" if metric == "supervised_objective" else "")),
        ("batch_weighted_loss", "Batch discriminator objective"),
        ("he_weighted_loss", "H&E objective") if he_enabled else None,
        None,
    ]
    metric_objective_pairs = []
    for suffix, label in [("", "IHC"), *[(f"_{name}", names[name]) for name in order]]:
        macro = " (macro average)" if not suffix else ""
        metric_objective_pairs.extend([
            ((f"hurdle_acc{suffix}", f"{label} balanced hurdle accuracy{macro}"),
             (f"hurdle_weighted_loss{suffix}", f"{label} hurdle objective")),
            ((f"mean_r2{suffix}", f"{label} positive-logit R²{macro}"),
             (f"positive_weighted_loss{suffix}", f"{label} positive objective\nLikelihood: location and scale")),
        ])
    # Each block has four metrics followed by their objectives in the same columns.
    for start in range(0, len(metric_objective_pairs), 4):
        block = metric_objective_pairs[start:start + 4]
        block += [(None, None)] * (4 - len(block))
        panels.extend(metric_panel for metric_panel, _ in block)
        panels.extend(objective_panel for _, objective_panel in block)
    rows = len(panels) // 4
    height = rows * 4.0 + 1.4
    figure, axes = plt.subplots(rows, 4, figsize=(20, height), sharex=True)
    # Reserve a fixed-height header for formulas and one shared legend.
    figure.subplots_adjust(left=.05, right=.99, bottom=.45 / height,
                           top=1 - 1.9 / height, hspace=.65, wspace=.25)
    he_term = " + H&E" if he_enabled else ""
    selection_text = {"supervised_objective": "supervised objective", "cd8_loss": "mean valid CD8 loss"}.get(metric, "unavailable")
    selected_text = f"epoch {int(best_epoch)}" if best_epoch is not None else "unavailable"
    boundary = selection.get("selection_start_epoch", 1) if selection else 1
    figure.suptitle(
        "Training objectives and fit metrics\n"
        f"Total = Supervised + Batch;   Supervised = IHC{he_term}\n"
        "IHC = aggregate hurdle + aggregate positive = Σ signals (hurdle + positive)\n"
        "Objectives include configured target/global weights; breakdowns are parts of IHC.\n"
        f"Selection: {selection_text}; selected checkpoint: {selected_text}; eligible from epoch {boundary}",
        fontsize=13, y=1 - .06 / height,
    )
    for axis, panel in zip(axes.flat, panels, strict=True):
        if panel is None:
            axis.set_visible(False)
            continue
        key, title = panel
        available = False
        for split, label, color in (("train", "Train", "C0"), ("validation", "Validation", "C1")):
            values = history.get(f"{split}_{key}", pd.Series(np.nan, index=history.index))
            axis.plot(history["epoch"], values, label=label, color=color,
                      marker=".", markersize=4, zorder=3)
            available |= bool(np.isfinite(values).any())
        if best_epoch is not None:
            axis.axvline(float(best_epoch), color="red", linestyle="--", linewidth=1,
                         label=f"Selected epoch {int(best_epoch)}", zorder=1)
        if not available:
            axis.text(.5, .5, "Unavailable in saved history", transform=axis.transAxes,
                      ha="center", va="center", color="dimgray")
        axis.set_title(title, fontsize=13)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        if len(history) == 1:
            epoch = float(history["epoch"].iloc[0])
            selected = float(best_epoch) if best_epoch is not None else epoch
            axis.set_xlim(min(epoch, selected) - .5, max(epoch, selected) + .5)
        axis.tick_params(labelbottom=True, labelsize=12)
        axis.grid(alpha=.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(handles),
                  bbox_to_anchor=(.5, 1 - 1.3 / height), borderaxespad=0, fontsize=11)
    figure.savefig(output_path, dpi=figure_dpi)
    plt.close(figure)

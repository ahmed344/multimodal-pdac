"""Checkpoint-selection semantics shared by training and reporting."""

from __future__ import annotations

import math
from typing import Any, Mapping


def selection_metric(config: Mapping[str, Any]) -> str:
    """Resolve the configured metric, retaining CD8 selection for old YAML files."""
    metric = config.get("training", {}).get("early_stopping_metric", "cd8_loss")
    if metric not in ("cd8_loss", "supervised_objective"):
        raise ValueError("training.early_stopping_metric must be supervised_objective or cd8_loss.")
    return metric


def checkpoint_selection(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Read explicit selection metadata or the recognized legacy CD8 contract.

    Epochs in selection metadata are one-based; the checkpoint's model epoch is
    zero-based. Unknown historical selection rules are not inferred from losses.
    """
    metric = checkpoint.get("selection_metric")
    if metric == "mean_valid_cd8_ziln":
        metric = "cd8_loss"
    score = checkpoint.get("best_selection_score", math.inf)
    if metric == "cd8_loss" and "best_selection_score" not in checkpoint:
        score = checkpoint.get("best_validation_cd8", checkpoint.get("best_validation_biology", math.inf))
    return dict(
        selection_metric=metric,
        best_selection_score=float(score),
        best_epoch=checkpoint.get("best_epoch"),
        selection_start_epoch=int(checkpoint.get("selection_start_epoch", 1)),
        early_stopping_state=checkpoint.get("early_stopping_state"),
    )

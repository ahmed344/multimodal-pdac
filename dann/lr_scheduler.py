"""Validation-driven learning rates and their production resume contract."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dann.config import lr_scheduler_settings


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: Mapping[str, Any],
) -> ReduceLROnPlateau | None:
    """Build the scheduler using the resolved mode's original starting LR."""
    training = config["training"]
    settings = lr_scheduler_settings(training)
    if not settings["enabled"]:
        return None
    return ReduceLROnPlateau(
        optimizer, mode="min", threshold_mode="abs",
        min_lr=float(training["learning_rate"]) * settings["min_lr_factor"],
        **{key: settings[key] for key in ("factor", "patience", "cooldown", "threshold", "eps")},
    )


def step_lr_scheduler(
    scheduler: ReduceLROnPlateau | None, validation_metrics: Mapping[str, float],
) -> None:
    """Step once per completed validation epoch, independently of selection."""
    if scheduler is None:
        return
    value = validation_metrics["supervised_objective"]
    if not math.isfinite(value):
        raise ValueError("LR scheduling requires a finite validation supervised_objective.")
    scheduler.step(value)


def validate_lr_scheduler_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Refuse production continuation when enabled scheduling has lost its state."""
    enabled = lr_scheduler_settings(checkpoint["config"]["training"])["enabled"]
    if not enabled:
        return
    state = checkpoint.get("lr_scheduler_state")
    if not state:
        raise ValueError("Cannot resume: saved configuration enables LR scheduling but "
                         "lr_scheduler_state is missing.")
    required = {"best", "num_bad_epochs", "cooldown_counter", "last_epoch", "min_lrs", "_last_lr",
                "mode", "threshold_mode", "factor", "patience", "cooldown", "threshold", "eps"}
    if not isinstance(state, Mapping) or not required <= state.keys():
        raise ValueError("Cannot resume: lr_scheduler_state is incomplete.")

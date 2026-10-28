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


class UpdateWarmup:
    """Linear LR ramp advanced once after each complete logical optimizer update.

    The first of N updates uses start_factor and update N uses the base LR (N=1
    uses start_factor, with the base rate installed for the following update).
    Once finished, this controller never overwrites plateau scheduler changes.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, config: Mapping[str, Any],
                 updates_per_epoch: int) -> None:
        from dann.config import warmup_settings
        self.optimizer = optimizer
        self.settings = warmup_settings(config['training'])
        self.horizon = self.settings['epochs'] * updates_per_epoch
        self.base_lrs = [float(group['lr']) for group in optimizer.param_groups]
        self.completed_updates = 0
        self._apply()

    @property
    def active(self) -> bool:
        """Whether the next optimizer update belongs to warmup."""
        return self.completed_updates < self.horizon

    def _apply(self) -> None:
        if self.active:
            fraction = self.completed_updates / max(self.horizon - 1, 1)
            factor = self.settings['start_factor'] + (1-self.settings['start_factor']) * fraction
            for group, base in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = base * factor

    def step(self) -> None:
        """Install the next-update rate without changing a post-warmup plateau LR."""
        was_active = self.active
        self.completed_updates += 1
        if was_active and not self.active:
            for group, base in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = base
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        """Serialize the resolved horizon, base rates, and exact update position."""
        return dict(horizon=self.horizon, base_lrs=list(self.base_lrs),
                    completed_updates=self.completed_updates, settings=dict(self.settings))

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore progress after optimizer restoration, checking the saved LR."""
        if (state['horizon'] != self.horizon or state['settings'] != self.settings
                or state['base_lrs'] != self.base_lrs):
            raise ValueError('Cannot resume: inconsistent warmup horizon, settings, or base rates.')
        self.completed_updates = state['completed_updates']
        if self.active:
            actual = [group['lr'] for group in self.optimizer.param_groups]
            self._apply()
            if actual != [group['lr'] for group in self.optimizer.param_groups]:
                raise ValueError('Cannot resume: optimizer LR disagrees with warmup progress.')


def validate_warmup_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Reject lost or inconsistent enabled-warmup state before resuming training."""
    from dann.config import warmup_settings
    settings = warmup_settings(checkpoint['config']['training'])
    if not settings['epochs']:
        return
    state = checkpoint.get('warmup_state')
    required = {'horizon', 'base_lrs', 'completed_updates', 'settings'}
    if not isinstance(state, Mapping) or not required <= state.keys():
        raise ValueError('Cannot resume: warmup_state is missing or incomplete.')
    updates = checkpoint.get('optimization', {}).get('logical_updates_per_epoch')
    if (not isinstance(updates, int) or updates < 1 or state['settings'] != settings
            or state['horizon'] != updates * settings['epochs']):
        raise ValueError('Cannot resume: inconsistent warmup horizon/settings.')
    progress = state['completed_updates']
    expected = checkpoint.get('training_progress', {}).get('optimizer_updates',
                                                         (checkpoint['epoch'] + 1) * updates)
    if isinstance(progress, bool) or not isinstance(progress, int) or progress < 0 or progress != expected:
        raise ValueError('Cannot resume: inconsistent warmup progress.')
    groups = checkpoint['optimizer_state']['param_groups']
    if (len(state['base_lrs']) != len(groups) or any(not math.isfinite(x) or x <= 0 for x in state['base_lrs'])
            or any(x != checkpoint['config']['training']['learning_rate'] for x in state['base_lrs'])):
        raise ValueError('Cannot resume: inconsistent warmup base rates.')
    if progress < state['horizon']:
        factor = settings['start_factor'] + (1-settings['start_factor']) * progress / max(state['horizon']-1, 1)
        if any(not math.isclose(group['lr'], base * factor, rel_tol=1e-14, abs_tol=0.)
               for group, base in zip(groups, state['base_lrs'])):
            raise ValueError('Cannot resume: inconsistent warmup optimizer LR.')

"""YAML-driven training entry point for adversarial latent fusion."""

from __future__ import annotations

import argparse
import copy
import json
import math
import resource
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from dann.config import apply_smoke_overrides, load_config, resolve_device, seed_everything
from dann.data_loader import DataBundle, create_data_bundle
from dann.sampling import SamplingEpochAudit
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.checkpoints import (architecture_contract, checkpoint_config, validate_checkpoint, training_data_contract,
                              validate_training_data_contract, sampling_contract, validate_sampling_contract)
from dann.tiles import row_identity
from dann.targets import CD8, checkpoint_contract, target_contract
from dann.selection import checkpoint_selection, selection_metric


@dataclass
class EarlyStopping:
    """Track validation improvement and stopping patience."""

    patience: int
    min_delta: float
    best: float = math.inf
    stale_epochs: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        """Update state with one validation metric.

        Args:
            value (float): Current validation selection score.

        Returns:
            tuple[bool, bool]: Whether improved and whether training should stop.
        """

        improved = value < self.best - self.min_delta
        if improved:
            self.best = value
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return improved, self.stale_epochs >= self.patience


_RESUME_OVERRIDES = (
    "early_stopping_patience",
    "early_stopping_min_delta",
    "epochs",
    "num_workers",
    "prefetch_factor",
    "resume_checkpoint",
)


def resume_training_config(saved_config: Mapping[str, Any], requested_config: Mapping[str, Any]) -> dict[str, Any]:
    """Continue from saved training values, applying a short set of current overrides.

    Args:
        saved_config (Mapping[str, Any]): Configuration stored in the checkpoint.
        requested_config (Mapping[str, Any]): Current resolved configuration.

    Returns:
        dict[str, Any]: Checkpoint configuration with epochs, workers, prefetch,
        early-stopping metric, patience, minimum improvement, and the resume path replaced
        by the requested values, plus current post-training switches. The
        checkpoint learning rate stays in place.
    """

    has_he = any(c.get('model', {}).get('heads', {}).get('he', {}).get('enabled', False)
                 for c in (saved_config, requested_config))
    if has_he and architecture_contract(saved_config)['components'].get('he') != architecture_contract(requested_config)['components'].get('he'):
        raise ValueError('Incompatible H&E settings on resume; adding or changing the head requires fresh training.')
    config = copy.deepcopy(dict(saved_config))
    training = config.setdefault("training", {})
    requested = requested_config["training"]
    for key in _RESUME_OVERRIDES:
        training[key] = requested[key]
    training["early_stopping_metric"] = selection_metric(requested_config)
    from dann.config import performance_settings, post_training_settings
    config['performance'] = performance_settings(requested_config)
    config['post_training'] = post_training_settings(requested_config.get('post_training'))
    for key in ('spectral_peak_budget', 'spectral_checkpointing', 'gatv2_tile_budget',
                'gatv2_checkpointing', 'inference_peak_chunk_size'):
        if key in requested_config.get('model', {}):
            config['model'][key] = requested_config['model'][key]
    mode = config.get("execution", {}).get("mode")
    if mode not in {"pixel", "spatial"}:
        from dann.config import execution_settings
        mode = execution_settings(config)["mode"]
    block = training.get(mode)
    if isinstance(block, dict):
        for key in ("num_workers", "prefetch_factor"):
            block[key] = requested[key]
    execution = config.get("execution")
    if isinstance(execution, dict):
        for key in ("num_workers", "prefetch_factor"):
            if key in execution:
                execution[key] = requested[key]
    # Refresh execution metadata without changing saved sampling fields.
    performance = config['performance']
    if isinstance(execution, dict):
        prepared = any(performance[k] for k in ('ram_cache', 'cache_geometry', 'transfer_prefetch',
                       'validation_tiles', 'validation_peak_budget', 'validation_gat_tiles'))
        execution.update(loader_backend='shared_threads' if prepared else 'pytorch_workers',
                         preparation_workers=performance['preparation_workers'] if prepared else training['num_workers'],
                         physical_validation_batch_size=(performance['validation_tiles'] or training.get('validation_batch_size'))
                             if mode == 'spatial' else training.get('validation_batch_size'))
    return config


def _config_values_equal(left: Any, right: Any) -> bool:
    """Compare configuration leaves, treating close numbers as equal.

    Args:
        left (Any): Saved or requested value.
        right (Any): Value from the other configuration.

    Returns:
        bool: Whether the two leaves represent the same setting.
    """

    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        left_value, right_value = float(left), float(right)
        if math.isnan(left_value) and math.isnan(right_value):
            return True
        return math.isclose(left_value, right_value, rel_tol=0, abs_tol=1e-12)
    return left == right


def ignored_resume_settings(requested_config: Mapping[str, Any], effective_config: Mapping[str, Any]) -> list[str]:
    """List requested settings that resume does not apply.

    Args:
        requested_config (Mapping[str, Any]): Configuration loaded from the current YAML.
        effective_config (Mapping[str, Any]): Configuration that training will use.

    Returns:
        list[str]: Sorted dotted paths whose requested values differ from the effective configuration.
    """

    def walk(left: Any, right: Any, prefix: str) -> list[str]:
        """Collect dotted paths that differ between two configuration values.

        Args:
            left (Any): Requested configuration node.
            right (Any): Effective configuration node.
            prefix (str): Dotted path of this node.

        Returns:
            list[str]: Sorted differing paths at or below this node.
        """

        if isinstance(left, Mapping) and isinstance(right, Mapping):
            paths: list[str] = []
            for key in sorted(set(left) | set(right), key=str):
                path = f"{prefix}.{key}" if prefix else str(key)
                if key not in left or key not in right:
                    paths.append(path)
                else:
                    paths.extend(walk(left[key], right[key], path))
            return paths
        if _config_values_equal(left, right):
            return []
        return [prefix]

    return walk(requested_config, effective_config, "")


def _history_row_count(path: Path) -> int:
    """Count data rows in a history CSV.

    Args:
        path (Path): History file path.

    Returns:
        int: Number of non-empty data rows, or zero when the file is absent.
    """

    if not path.is_file():
        return 0
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return max(len(lines) - 1, 0)


def load_resume_history(
    path: Path, completed_epoch: int
) -> tuple[list[dict[str, float]], list[str], dict[float, str]]:
    """Keep completed rows from an existing training history.

    Args:
        path (Path): ``history.csv`` path. A missing file yields an empty history.
        completed_epoch (int): Last finished one-based epoch. Later rows are excluded
            because those epochs will be trained again.

    Returns:
        tuple[list[dict[str, float]], list[str], dict[float, str]]: Kept rows in epoch
        order, their column order, and the original text of each kept row.
    """

    if completed_epoch <= 0 or not path.is_file():
        return [], [], {}
    text = path.read_text(encoding="utf-8").splitlines()
    if not text or "epoch" not in text[0].split(","):
        raise ValueError("history.csv is missing the epoch column.")
    columns = text[0].split(",")
    epoch_index = columns.index("epoch")
    preserved: dict[float, str] = {}
    for line in text[1:]:
        if not line.strip():
            continue
        preserved[float(line.split(",")[epoch_index])] = line
    frame = pd.read_csv(path)
    frame = frame.loc[frame["epoch"] <= completed_epoch]
    frame = frame.drop_duplicates("epoch", keep="last").sort_values("epoch")
    rows = [
        {column: float(record[column]) for column in columns}
        for record in frame.to_dict(orient="records")
    ]
    preserved = {float(row["epoch"]): preserved[float(row["epoch"])] for row in rows}
    return rows, columns, preserved


def restore_resume_stopping(
    stopper: EarlyStopping,
    history: Sequence[Mapping[str, float]],
    use_checkpoint_best: bool,
) -> float | None:
    """Replay validation CD8 loss so the early-stopping streak continues.

    Args:
        stopper (EarlyStopping): Stopper initialized with the checkpoint selection score.
        history (Sequence[Mapping[str, float]]): Completed history rows to replay.
        use_checkpoint_best (bool): Whether the checkpoint score remains the selection
            threshold. Legacy checkpoints leave the stopper unchanged.

    Returns:
        float | None: One-based epoch of the last CD8 improvement, or ``None`` when
        the streak is not restored.
    """

    if not use_checkpoint_best or not history:
        return None
    if any("validation_cd8_loss" not in row for row in history):
        raise ValueError("history.csv is missing validation_cd8_loss; cannot restore early stopping.")
    replay = EarlyStopping(patience=stopper.patience, min_delta=stopper.min_delta)
    best_epoch = None
    for row in history:
        improved, _ = replay.update(float(row["validation_cd8_loss"]))
        if improved:
            best_epoch = float(row["epoch"])
    stopper.stale_epochs = replay.stale_epochs
    return best_epoch


def restore_selection_state(
    stopper: EarlyStopping,
    checkpoint: Mapping[str, Any],
    metric: str,
    history: Sequence[Mapping[str, float]],
) -> tuple[float | None, int]:
    """Restore compatible stopping progress or start selection after the resume boundary."""
    saved = checkpoint_selection(checkpoint)
    if saved["selection_metric"] != metric:
        stopper.best, stopper.stale_epochs = math.inf, 0
        return None, int(checkpoint["epoch"]) + 2
    stopper.best = saved["best_selection_score"]
    state = saved["early_stopping_state"]
    if state is not None:
        stopper.stale_epochs = int(state["stale_epochs"])
        return saved["best_epoch"], saved["selection_start_epoch"]
    # Recognized old checkpoints have no explicit patience state. Replay only
    # their CD8 history; new checkpoints never depend on an external CSV.
    previous_delta = checkpoint.get("config", {}).get("training", {}).get("early_stopping_min_delta", stopper.min_delta)
    replay = EarlyStopping(stopper.patience, float(previous_delta), best=stopper.best)
    best_epoch = restore_resume_stopping(replay, history, metric == "cd8_loss")
    stopper.stale_epochs = replay.stale_epochs
    return best_epoch, saved["selection_start_epoch"]


def write_training_history(
    path: Path,
    rows: Sequence[Mapping[str, float]],
    columns: Sequence[str],
    preserved_lines: Mapping[float, str] | None = None,
) -> list[str]:
    """Write the full training history, preserving existing row text.

    Args:
        path (Path): Destination ``history.csv`` path.
        rows (Sequence[Mapping[str, float]]): Completed and new epoch rows.
        columns (Sequence[str]): Existing column order. New keys are appended.
        preserved_lines (Mapping[float, str] | None): Original CSV lines keyed by
            one-based epoch. Matching lines are copied unchanged.

    Returns:
        list[str]: Column order written to disk.
    """

    ordered = list(columns)
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(str(key))
    preserved = {} if preserved_lines is None else preserved_lines
    lines = [",".join(ordered)]
    for row in rows:
        epoch = float(row["epoch"])
        saved = preserved.get(epoch)
        if saved is not None and saved.count(",") == len(ordered) - 1:
            lines.append(saved)
            continue
        formatted = pd.DataFrame([{key: row.get(key, math.nan) for key in ordered}], columns=ordered)
        lines.append(formatted.to_csv(index=False, header=False).rstrip("\n"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ordered


def refresh_loss_curves(
    history_path: Path,
    output_path: Path,
    figure_dpi: int,
    best_epoch: float | None,
    config: Mapping[str, Any] | None = None,
    selection: Mapping[str, Any] | None = None,
) -> None:
    """Redraw train and validation curves from the complete history.

    Args:
        history_path (Path): Training history CSV.
        output_path (Path): Destination ``loss_curves.png`` path.
        figure_dpi (int): Saved figure resolution.
        best_epoch (float | None): One-based epoch of the selected checkpoint.
        config (Mapping[str, Any] | None): Saved weights and target ordering.
        selection (Mapping[str, Any] | None): Selection metric, epoch, and boundary.

    Returns:
        None: Figure is saved to disk.
    """

    from dann.analyze import plot_loss_curves
    plot_loss_curves(history_path, output_path, figure_dpi, best_epoch=best_epoch,
                     config=config, selection=selection)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse training command-line arguments.

    Args:
        argv (Sequence[str] | None): Optional explicit argument sequence.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Commented YAML configuration path.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use configured real-data smoke caps and one short epoch.",
    )
    return parser.parse_args(argv)


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move a collated sparse batch to the training device.

    Args:
        batch (Mapping[str, torch.Tensor]): CPU collator output.
        device (torch.device): Destination device.

    Returns:
        dict[str, torch.Tensor]: Device-resident tensor dictionary.
    """

    from dann.execution import tree_map
    return tree_map(batch, lambda value: value.to(device, non_blocking=True))


def grl_strength(
    progress: float,
    schedule: str,
    maximum: float,
    gamma: float,
) -> float:
    """Calculate gradient reversal strength at normalized training progress.

    Args:
        progress (float): Fraction of optimizer steps completed in ``[0, 1]``.
        schedule (str): ``constant``, ``linear``, or ``dann``.
        maximum (float): Maximum reversal strength.
        gamma (float): Logistic steepness for the DANN schedule.

    Returns:
        float: Current nonnegative gradient reversal multiplier.
    """

    progress = min(max(float(progress), 0.0), 1.0)
    if schedule == "constant":
        scale = 1.0
    elif schedule == "linear":
        scale = progress
    elif schedule == "dann":
        scale = 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
    else:
        raise ValueError(f"Unknown GRL schedule: {schedule!r}")
    return float(maximum * scale)


def _metric_suffix(target_name: str) -> str:
    """Create a lowercase history suffix from one target column name.

    Args:
        target_name (str): Configured target column, optionally prefixed by ``Density_``.

    Returns:
        str: Lowercase name with a leading ``Density_`` prefix removed.
    """

    prefix = "Density_"
    short_name = target_name[len(prefix) :] if target_name.startswith(prefix) else target_name
    return short_name.lower()


def _target_display_name(target_name: str) -> str:
    """Create the bracket label for one target column.

    Args:
        target_name (str): Configured target column, optionally prefixed by ``Density_``.

    Returns:
        str: Display name with a leading ``Density_`` prefix removed.
    """

    prefix = "Density_"
    if target_name.startswith(prefix):
        return target_name[len(prefix) :]
    return target_name


def _empty_metrics(num_targets: int) -> dict[str, Any]:
    """Create metric accumulators for one epoch.

    Args:
        num_targets (int): Number of ordered biology targets.

    Returns:
        dict[str, Any]: Zero-initialized loss sums and per-target fit counts.
    """

    if num_targets <= 0:
        raise ValueError("num_targets must be positive.")
    return {
        "cd8_sum": 0.0,
        "cd8_count": 0.0,
        "samples": 0.0,
        "biology_mass": 0.0,
        "total": 0.0,
        "biology": 0.0,
        "hurdle": 0.0,
        "positive": 0.0,
        "hurdle_per_target": np.zeros(num_targets, dtype=np.float64),
        "positive_per_target": np.zeros(num_targets, dtype=np.float64),
        "batch": 0.0,
        "batch_correct": 0.0,
        "zero_correct": np.zeros(num_targets, dtype=np.float64),
        "positive_correct": np.zeros(num_targets, dtype=np.float64),
        "zero_count": np.zeros(num_targets, dtype=np.float64),
        "positive_count": np.zeros(num_targets, dtype=np.float64),
        "logit_sum": np.zeros(num_targets, dtype=np.float64),
        "logit_sum_squares": np.zeros(num_targets, dtype=np.float64),
        "logit_residual_sum_squares": np.zeros(num_targets, dtype=np.float64),
    }


def _update_fit_statistics(
    metrics: dict[str, Any],
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    logit_epsilon: float,
) -> None:
    """Accumulate hurdle decisions and positive-logit regression sums.

    Args:
        metrics (dict[str, Any]): Mutable epoch accumulators.
        outputs (Mapping[str, torch.Tensor]): Model outputs for one batch.
        batch (Mapping[str, torch.Tensor]): Device-resident target tensors.
        logit_epsilon (float): Clamp used before positive-value logits.

    Returns:
        None: ``metrics`` is updated in place.
    """

    targets = batch["targets"]
    valid = batch.get("target_valid_mask")
    if valid is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    observed = valid & torch.isfinite(targets)
    zero_mask = observed & (targets == 0.0)
    positive_mask = observed & (targets > 0.0)
    predicted_zero = outputs["pi_logits"] >= 0.0
    safe_targets = targets.clamp(logit_epsilon, 1.0 - logit_epsilon)
    logits = torch.logit(safe_targets)
    residual = outputs["mu"] - logits
    updates = {
        "zero_correct": (predicted_zero & zero_mask).sum(dim=0),
        "positive_correct": ((~predicted_zero) & positive_mask).sum(dim=0),
        "zero_count": zero_mask.sum(dim=0),
        "positive_count": positive_mask.sum(dim=0),
        "logit_sum": torch.where(positive_mask, logits, torch.zeros_like(logits)).sum(dim=0),
        "logit_sum_squares": torch.where(
            positive_mask, logits.square(), torch.zeros_like(logits)
        ).sum(dim=0),
        "logit_residual_sum_squares": torch.where(
            positive_mask, residual.square(), torch.zeros_like(residual)
        ).sum(dim=0),
    }
    for key, value in updates.items():
        accumulator = metrics[key]
        if isinstance(accumulator, torch.Tensor):
            accumulator += value.detach().to(dtype=torch.float64)
            continue
        if not isinstance(accumulator, np.ndarray):
            raise TypeError(f"Metric accumulator {key!r} must be an array.")
        accumulator += value.detach().to(dtype=torch.float64, device="cpu").numpy()


def _finalize_metrics(
    metrics: Mapping[str, Any],
    target_columns: Sequence[str],
) -> dict[str, float]:
    """Convert sample-weighted accumulators to epoch means.

    Args:
        metrics (Mapping[str, Any]): Raw epoch sums and per-target counts.
        target_columns (Sequence[str]): Ordered target names for per-target keys.

    Returns:
        dict[str, float]: Mean losses, batch accuracy, hurdle accuracy, and mean R².
    """

    samples = max(float(metrics["samples"]), 1.0)
    biology_mass = max(float(metrics["biology_mass"]), 1e-12)
    count_keys = (
        "zero_correct",
        "positive_correct",
        "zero_count",
        "positive_count",
        "logit_sum",
        "logit_sum_squares",
        "logit_residual_sum_squares",
    )
    arrays = {key: np.asarray(metrics[key], dtype=np.float64) for key in count_keys}
    target_count = len(target_columns)
    if target_count == 0 or any(value.shape != (target_count,) for value in arrays.values()):
        raise ValueError("Target names must match the metric accumulator dimensions.")
    suffixes = [_metric_suffix(name) for name in target_columns]
    if len(suffixes) != len(set(suffixes)):
        raise ValueError("Target metric suffixes must be unique.")
    result = {
        "cd8_loss": metrics["cd8_sum"] / metrics["cd8_count"] if metrics["cd8_count"] else math.nan,
        "total_loss": metrics["total"] / samples,
        "biology_loss": metrics["biology"] / biology_mass,
        "hurdle_loss": metrics["hurdle"] / biology_mass,
        "positive_loss": metrics["positive"] / biology_mass,
        "batch_loss": metrics["batch"] / samples,
        "batch_accuracy": metrics["batch_correct"] / samples,
    }
    balanced_accuracies: list[float] = []
    mean_r2_values: list[float] = []
    for index, suffix in enumerate(suffixes):
        recalls: list[float] = []
        if arrays["zero_count"][index] > 0.0:
            recalls.append(arrays["zero_correct"][index] / arrays["zero_count"][index])
        if arrays["positive_count"][index] > 0.0:
            recalls.append(arrays["positive_correct"][index] / arrays["positive_count"][index])
        balanced_accuracy = float(np.mean(recalls)) if recalls else math.nan
        positive_count = arrays["positive_count"][index]
        logit_sum = arrays["logit_sum"][index]
        total_sum_squares = (
            arrays["logit_sum_squares"][index] - logit_sum * logit_sum / positive_count
            if positive_count > 0.0
            else 0.0
        )
        mean_r2 = (
            1.0 - arrays["logit_residual_sum_squares"][index] / total_sum_squares
            if positive_count >= 2.0 and total_sum_squares > np.finfo(np.float64).eps
            else math.nan
        )
        result[f"hurdle_acc_{suffix}"] = balanced_accuracy
        result[f"mean_r2_{suffix}"] = float(mean_r2)
        for branch in ("hurdle", "positive"):
            result[f"{branch}_loss_{suffix}"] = float(metrics[f"{branch}_per_target"][index]) / biology_mass
        balanced_accuracies.append(balanced_accuracy)
        mean_r2_values.append(float(mean_r2))
    finite_balanced = [value for value in balanced_accuracies if math.isfinite(value)]
    finite_r2 = [value for value in mean_r2_values if math.isfinite(value)]
    result["hurdle_acc"] = float(np.mean(finite_balanced)) if finite_balanced else math.nan
    result["mean_r2"] = float(np.mean(finite_r2)) if finite_r2 else math.nan
    return result


def _format_pair(train_value: float, validation_value: float, decimals: int) -> str:
    """Format one train, validation metric pair.

    Args:
        train_value (float): Training metric.
        validation_value (float): Validation metric.
        decimals (int): Digits after the decimal point.

    Returns:
        str: ``"{train}, {validation}"``, with non-finite values written as ``nan``.
    """

    rendered = []
    for value in (train_value, validation_value):
        rendered.append(f"{value:.{decimals}f}" if math.isfinite(value) else "nan")
    return f"{rendered[0]}, {rendered[1]}"


def _format_epoch_summary(
    epoch: int,
    epochs: int,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    target_columns: Sequence[str],
    metric: str = "cd8_loss",
) -> str:
    """Build the epoch log with one aligned line per IHC target.

    Args:
        epoch (int): Completed one-based epoch number.
        epochs (int): Configured maximum epoch count.
        train_metrics (Mapping[str, float]): Finalized training metrics.
        validation_metrics (Mapping[str, float]): Finalized validation metrics.
        target_columns (Sequence[str]): Ordered target column names.
        metric (str): Validation score used for early stopping and checkpoint selection.

    Returns:
        str: Loss line, macro fit line, and one ``[Name]`` line per target.
    """

    loss_parts = (
        ("total objective", "total_loss", 6),
        (f"stopping objective ({metric})", metric, 6),
        ("IHC objective", "biology_weighted_loss", 6),
        ("batch objective", "batch_weighted_loss", 6),
        ("hurdle objective", "hurdle_weighted_loss", 6),
        ("positive objective (location + scale)", "positive_weighted_loss", 6),
    )
    if "he_mse" in train_metrics:
        loss_parts += (("H&E objective", "he_weighted_loss", 6),)
    fit_parts = (
        ("batch_acc", "batch_accuracy", 4),
        ("macro hurdle_acc", "hurdle_acc", 4),
        ("macro mean_r2", "mean_r2", 4),
    )
    if "he_r2" in train_metrics:
        fit_parts += (("H&E R²", "he_r2", 4),)
    loss_text = " | ".join(
        f"{label} = {_format_pair(train_metrics[key], validation_metrics[key], decimals)}"
        for label, key, decimals in loss_parts
    )
    fit_text = " | ".join(
        f"{label} = {_format_pair(train_metrics[key], validation_metrics[key], decimals)}"
        for label, key, decimals in fit_parts
    )
    displays = [_target_display_name(name) for name in target_columns]
    width = max(len(f"[{name}]") for name in displays)
    lines = [
        f"epoch={epoch}/{epochs} | {loss_text}",
        f"  {fit_text}",
    ]
    for name, column in zip(displays, target_columns, strict=True):
        suffix = _metric_suffix(column)
        label = f"[{name}]".ljust(width)
        lines.append(
            f"  {label} hurdle_acc = "
            f"{_format_pair(train_metrics[f'hurdle_acc_{suffix}'], validation_metrics[f'hurdle_acc_{suffix}'], 4)}"
            f" | mean_r2 = "
            f"{_format_pair(train_metrics[f'mean_r2_{suffix}'], validation_metrics[f'mean_r2_{suffix}'], 4)}"
        )
    return "\n".join(lines)


@torch.no_grad()
def _update_he_statistics(
    metrics: dict[str, torch.Tensor],
    prediction: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Accumulate H&E SSE and merge float64 per-feature centered moments.

    Center features separately so R² compares with each feature's epoch mean,
    not a pooled scalar mean. Parallel variance merging preserves differences
    between batch means without retaining predictions or all target vectors.
    """
    values = targets.double()
    rows = values.shape[0]
    mean = values.mean(dim=0)
    centered_sum = (values - mean).square().sum(dim=0)
    if "he_target_mean" not in metrics:
        metrics["he_target_mean"] = torch.zeros_like(mean)
        metrics["he_target_m2"] = torch.zeros_like(mean)
    previous_rows = metrics["he_rows"]
    combined_rows = previous_rows + rows
    difference = mean - metrics["he_target_mean"]
    metrics["he_target_m2"] += centered_sum + difference.square() * (previous_rows * rows / combined_rows)
    metrics["he_target_mean"] += difference * (rows / combined_rows)
    metrics["he_rows"] += rows
    metrics["he_sum"] += (prediction.double() - values).square().sum()
    metrics["he_count"] += values.numel()


def run_epoch(
    model: AdversarialLatentFusion,
    loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    ziln_loss: ZILNLoss,
    device: torch.device,
    biology_weight: float,
    batch_weight: float,
    class_weights: torch.Tensor | None,
    optimizer: torch.optim.Optimizer | None,
    epoch_index: int,
    total_epochs: int,
    grl_schedule: str,
    grl_max_lambda: float,
    grl_gamma: float,
    gradient_clip_norm: float | None,
    cd8_index: int = 0,
    target_columns: Sequence[str] | None = None,
    sampling_audit_dir: Path | None = None,
    he_weight: float = 1.0,
) -> dict[str, float]:
    """Run one training or validation epoch and report loss components.

    Args:
        model (AdversarialLatentFusion): Model being optimized or evaluated.
        loader (torch.utils.data.DataLoader): Sparse batch loader.
        ziln_loss (ZILNLoss): Biology objective.
        device (torch.device): Compute device.
        biology_weight (float): Biology objective multiplier.
        batch_weight (float): Discriminator objective multiplier.
        he_weight (float): Raw H&E feature MSE multiplier when the head is enabled.
        class_weights (torch.Tensor | None): Optional batch class weights.
        optimizer (torch.optim.Optimizer | None): Optimizer, or ``None`` to validate.
        epoch_index (int): Zero-based epoch index.
        total_epochs (int): Configured total epoch count.
        grl_schedule (str): Gradient reversal schedule name.
        grl_max_lambda (float): Maximum reversal strength.
        grl_gamma (float): Logistic schedule steepness.
        gradient_clip_norm (float | None): Optional global clipping threshold.
        cd8_index (int): Column of the CD8 target inside ``target_columns``.
        target_columns (Sequence[str] | None): Ordered target names. Defaults to
            ``target_{index}`` suffixes when omitted.
        sampling_audit_dir (Path | None): Optional spatial batch logs and epoch hashes.

    Returns:
        dict[str, float]: Mean losses, batch accuracy, hurdle accuracy, and mean R².
    """

    training = optimizer is not None
    dataset = getattr(loader, 'dataset', None)
    resident = getattr(dataset, 'resident', None) or getattr(getattr(dataset, 'tiles', None), 'resident', None)
    validate_targets = resident is None
    model.train(training)
    num_targets = int(ziln_loss.target_weights.numel())
    if target_columns is None:
        target_columns = [f"target_{index}" for index in range(num_targets)]
    elif len(target_columns) != num_targets:
        raise ValueError(
            f"Expected {num_targets} target columns, received {len(target_columns)}."
        )
    started = perf_counter()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    metrics = {key: torch.as_tensor(value, dtype=torch.float64, device=device)
               for key, value in _empty_metrics(num_targets).items()}
    metrics.update(he_sum=torch.zeros((), dtype=torch.float64, device=device),
                   he_count=torch.zeros((), dtype=torch.float64, device=device),
                   he_rows=torch.zeros((), dtype=torch.float64, device=device))
    cd8_loss = ZILNLoss([1.0], ziln_loss.logit_epsilon, "sum",
                        ziln_loss.include_normal_constant).to(device)
    steps_per_epoch = max(len(loader), 1)
    # Production retains the epoch-based GRL horizon, unlike diagnostic fixed-update runs.
    if training and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch_index)
    audit = (SamplingEpochAudit(epoch_index, len(loader.dataset.indices))
             if training and sampling_audit_dir is not None else None)
    if audit is not None:
        sampling_audit_dir.mkdir(parents=True, exist_ok=True)
        (sampling_audit_dir / f"epoch_{epoch_index:04d}_batches.jsonl").write_text("")
    from dann.execution import device_batches, evaluation_groups
    batches = ((cpu, batch, None) for cpu, batch in device_batches(loader, device)) if training else evaluation_groups(model, loader, device)
    with (torch.enable_grad() if training else torch.inference_mode()):
        for step, (cpu_batch, batch, outputs) in enumerate(batches):
            progress = (epoch_index * steps_per_epoch + step) / max(
                total_epochs * steps_per_epoch - 1, 1
            )
            strength = grl_strength(
                progress, grl_schedule, grl_max_lambda, grl_gamma
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            if outputs is None:
                outputs = model(batch, grl_strength=strength)
            biology = ziln_loss(
                outputs["pi_logits"],
                outputs["mu"],
                outputs["sigma"],
                batch["targets"],
                batch.get("target_valid_mask"),
                validate_targets=validate_targets,
            )
            with torch.no_grad():
                selected = slice(cd8_index, cd8_index + 1)
                mask = batch.get("target_valid_mask", torch.ones_like(batch["targets"], dtype=torch.bool))[:, selected]
                cd8 = cd8_loss(outputs["pi_logits"][:, selected], outputs["mu"][:, selected],
                               outputs["sigma"][:, selected], batch["targets"][:, selected], mask, validate_targets=validate_targets)
                metrics["cd8_sum"] += cd8.total.detach()
                metrics["cd8_count"] += mask.sum()
                _update_fit_statistics(
                    metrics, outputs, batch, ziln_loss.logit_epsilon
                )
            discriminator = F.cross_entropy(
                outputs["batch_logits"], batch["batches"], weight=class_weights
            )
            he_loss = biology.total.new_zeros(())
            if 'he_prediction' in outputs:
                from dann.he import he_mse
                he_loss = he_mse(outputs['he_prediction'], batch['he_targets'])
                _update_he_statistics(metrics, outputs['he_prediction'], batch['he_targets'])
            total = biology_weight * biology.total + batch_weight * discriminator + he_weight * he_loss
            finite_outputs = torch.stack([torch.isfinite(value).all() for value in outputs.values()]).all()
            if not (torch.isfinite(total) & finite_outputs):
                raise FloatingPointError(
                    f"Non-finite objective or model output at epoch {epoch_index + 1}, step {step + 1}."
                )
            if training:
                total.backward()
                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
                if audit is not None and "occupancy" in cpu_batch:
                    audit.update(cpu_batch)

            batch_size = int(batch["targets"].shape[0])
            metrics["samples"] += batch_size
            metrics["total"] += total.detach() * batch_size
            mass = (
                biology.valid_weight_mass.detach()
                if ziln_loss.reduction == "mean" else batch_size
            )
            metrics["biology_mass"] += mass
            metrics["biology"] += biology.total.detach() * mass
            metrics["hurdle"] += biology.hurdle.detach() * mass
            metrics["positive"] += biology.positive.detach() * mass
            metrics["hurdle_per_target"] += biology.hurdle_per_target * mass
            metrics["positive_per_target"] += biology.positive_per_target * mass
            metrics["batch"] += discriminator.detach() * batch_size
            predictions = outputs["batch_logits"].argmax(dim=1)
            metrics["batch_correct"] += (predictions == batch["batches"]).sum().detach()
    # One reporting transfer for all scalar and per-target sufficient statistics.
    packed = torch.cat([v.reshape(-1) for v in metrics.values()]).cpu().numpy()
    offset = 0
    for key, value in list(metrics.items()):
        size = value.numel()
        metrics[key] = packed[offset:offset+size].reshape(tuple(value.shape)) if value.ndim else float(packed[offset])
        offset += size
    finalized = _finalize_metrics(metrics, target_columns)
    if getattr(model, 'he_predictor', None) is not None:
        finalized['he_mse'] = metrics['he_sum'] / metrics['he_count'] if metrics['he_count'] else 0.0
        finalized['he_weighted_loss'] = he_weight * finalized['he_mse']
        total_variance = float(np.sum(metrics.get('he_target_m2', 0.0)))
        finalized['he_r2'] = (1.0 - metrics['he_sum'] / total_variance
                              if metrics['he_rows'] >= 2 and total_variance > 0 else math.nan)
    for branch in ("biology", "hurdle", "positive"):
        finalized[f"{branch}_weighted_loss"] = biology_weight * finalized[f"{branch}_loss"]
    for column in target_columns:
        suffix = _metric_suffix(column)
        for branch in ("hurdle", "positive"):
            finalized[f"{branch}_weighted_loss_{suffix}"] = biology_weight * finalized[f"{branch}_loss_{suffix}"]
    finalized["batch_weighted_loss"] = batch_weight * finalized["batch_loss"]
    finalized["supervised_objective"] = finalized["biology_weighted_loss"] + finalized.get("he_weighted_loss", 0.0)
    finalized["total_loss"] = finalized["supervised_objective"] + finalized["batch_weighted_loss"]
    if audit is not None and audit.compositions:
        with (sampling_audit_dir / f"epoch_{epoch_index:04d}_batches.jsonl").open("w") as handle:
            handle.writelines(json.dumps(row) + "\n" for row in audit.compositions)
        with (sampling_audit_dir / f"epoch_{epoch_index:04d}_summary.json").open("w") as handle:
            json.dump(audit.summary(), handle, indent=2)
    if hasattr(loader, 'settings'):
        elapsed = perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
        if peak > loader.settings['vram_budget_gib'] * 2**30:
            raise MemoryError('Measured execution exceeded VRAM budget.')
        finalized.update(seconds=elapsed, observed_rows=metrics['samples'], rows_per_second=metrics['samples'] / max(elapsed, 1e-9),
                         loader_wait_seconds=loader.wait_seconds,
                         peak_vram_gib=peak / 2**30,
                         peak_reserved_vram_gib=(torch.cuda.max_memory_reserved(device) / 2**30 if device.type == 'cuda' else 0),
                         peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                         prepared_batch_mib=loader.max_batch_bytes / 2**20)
    return finalized


def save_checkpoint(
    path: Path,
    model: AdversarialLatentFusion,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Mapping[str, Any],
    bundle: DataBundle,
    best_selection_score: float,
    *,
    best_epoch: float | None = None,
    selection_start_epoch: int = 1,
    stopper: EarlyStopping | None = None,
) -> None:
    """Persist model, optimizer, schema, split, and configuration state.

    Args:
        path (Path): Destination checkpoint path.
        model (AdversarialLatentFusion): Trained model.
        optimizer (torch.optim.Optimizer): Optimizer state.
        epoch (int): Last completed zero-based epoch.
        config (Mapping[str, Any]): Effective configuration.
        bundle (DataBundle): Data metadata and selected split indices.
        best_selection_score (float): Best validation score in selection-metric units.
        best_epoch (float | None): Selected one-based epoch, if known.
        selection_start_epoch (int): First one-based epoch eligible for selection.
        stopper (EarlyStopping | None): Explicit patience progress for training resumes.

    Returns:
        None: Checkpoint is written atomically through a temporary file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    identity = row_identity(Path(config["data"]["path"]), config["data"])
    saved_config = copy.deepcopy(dict(config))
    architecture = architecture_contract(config)
    if "spectral_encoder" in saved_config["model"]:
        from dann.config import component_settings
        settings = component_settings(config["model"])
        for name, group in [("aggregation", saved_config["model"]["aggregation"]),
                            *saved_config["model"]["heads"].items()]:
            if name == "he" and name not in settings["heads"]:
                continue
            resolved = settings["aggregation"] if name == "aggregation" else settings["heads"][name]
            group.setdefault("cnn", {})["residual"] = resolved["cnn"]["residual"]
            group["gatv2"] = resolved["gatv2"]
    # Distribution metadata is informational, never an architecture compatibility key.
    from importlib.metadata import PackageNotFoundError, version
    try:
        pyg_version = version("torch-geometric")
    except PackageNotFoundError:
        pyg_version = None
    validate_checkpoint({"config": saved_config, "architecture": architecture}, config, model)
    payload = {
        "software_versions": {"torch": str(torch.__version__), "torch_geometric": pyg_version},
        "sampling_contract": sampling_contract(config, bundle.split_indices["train"]),
        "sampler_state": {"completed_epoch": int(epoch), "sampler_epoch": int(epoch),
                          "next_epoch": int(epoch) + 1},
        "architecture": architecture,
        "optimizer_parameter_names": list(dict(model.named_parameters())),
        "data_identity": identity,
        "training_data_contract": training_data_contract(config, identity),
        "selection_metric": selection_metric(config),
        "best_selection_score": float(best_selection_score),
        "best_epoch": best_epoch,
        "selection_start_epoch": int(selection_start_epoch),
        "early_stopping_state": (dict(stale_epochs=stopper.stale_epochs,
            patience=stopper.patience, min_delta=stopper.min_delta,
            stopped=stopper.stale_epochs >= stopper.patience) if stopper is not None else None),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "config": saved_config,
        "batch_names": bundle.metadata.batch_names,
        "target_columns": tuple(config["data"]["target_columns"]),
        "target_transform": target_contract(config),
        "split_indices": bundle.split_indices,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_checkpoint(
    path: Path,
    model: AdversarialLatentFusion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Mapping[str, Any],
    selected_rows=None,
) -> tuple[int, float]:
    """Restore model and optimizer state for resumed training.

    Args:
        path (Path): Existing checkpoint path.
        model (AdversarialLatentFusion): Model receiving saved parameters.
        optimizer (torch.optim.Optimizer): Optimizer receiving saved state.
        device (torch.device): Tensor map location.
        config (Mapping[str, Any]): Requested training configuration to validate.
        selected_rows: Actual selected training rows; defaults to saved rows for callers
            that do not rebuild the training bundle.

    Returns:
        tuple[int, float]: Next epoch index and compatible prior best selection score.
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint_contract(checkpoint) != target_contract(config):
        raise ValueError("Resume target transformation or target order is incompatible with checkpoint.")
    validate_training_data_contract(checkpoint, config)
    validate_checkpoint(checkpoint, config, model, optimizer=True)
    validate_sampling_contract(checkpoint, config, selected_rows)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    saved = checkpoint_selection(checkpoint)
    return int(checkpoint["epoch"]) + 1, (
        saved["best_selection_score"] if saved["selection_metric"] == selection_metric(config) else math.inf
    )


def summarize_target_validity(bundle: DataBundle, config: Mapping[str, Any]) -> dict[str, Any]:
    """Report exclusions and require usable normalized-CD8 supervision."""
    contract = target_contract(config)
    if not contract["cd8_normalization"]["enabled"]:
        return {}
    cd8 = contract["target_columns"].index(CD8)
    metadata = bundle.metadata
    summary = {}
    for name, indices in bundle.split_indices.items():
        valid = metadata.target_valid_mask[indices, cd8]
        reasons = metadata.exclusion_reasons[indices]
        summary[name] = {
            "rows": int(indices.size), "valid_cd8": int(valid.sum()),
            "positive_cd8": int(((metadata.targets[indices, cd8] > 0) & valid).sum()),
            "excluded_low_non_tumor": int((reasons == 1).sum()),
            "excluded_ratio_above_one": int((reasons == 2).sum()),
        }
    print("CD8 target validity: " + json.dumps(summary), flush=True)
    if summary["train"]["valid_cd8"] == 0:
        raise ValueError("Training has no valid CD8 observations after normalization.")
    if summary["train"]["positive_cd8"] == 0:
        raise ValueError("Training has no valid positive CD8 observations after normalization.")
    return summary


def train_model(config: Mapping[str, Any]) -> Path:
    """Train adversarial latent fusion and save best/latest checkpoints.

    Args:
        config (Mapping[str, Any]): Effective DANN configuration.

    Returns:
        Path: Best validation checkpoint path.
    """

    requested = config
    training = config["training"]
    resume = training.get("resume_checkpoint")
    resume_payload: dict[str, Any] = {}
    if resume:
        if config.get("results", {}).get("smoke"):
            raise ValueError("Cannot resume a checkpoint during a smoke test.")
        resume_payload = torch.load(resume, map_location="cpu", weights_only=False)
        config = resume_training_config(checkpoint_config(resume_payload), requested)
        ignored = ignored_resume_settings(requested, config)
        if ignored:
            print("Resume kept checkpoint values for: " + ", ".join(ignored), flush=True)
        training = config["training"]
        if (checkpoint_selection(resume_payload)["selection_metric"] != selection_metric(config)
                and int(training["epochs"]) <= int(resume_payload["epoch"]) + 1):
            raise ValueError("Changing the selection metric requires at least one additional epoch; increase training.epochs.")
    metric = selection_metric(config)
    seed_everything(
        int(training["seed"]), bool(training["deterministic_algorithms"])
    )
    device = resolve_device(str(training["device"]))
    if resume_payload:
        validate_training_data_contract(resume_payload, config)
    preparation_started = perf_counter()
    bundle = create_data_bundle(config, resume_payload.get("split_indices"), resume_payload.get("data_identity"))
    if resume_payload and tuple(resume_payload["batch_names"]) != bundle.metadata.batch_names:
        raise ValueError("Resume batch order is incompatible.")
    preparation_seconds = perf_counter() - preparation_started
    target_summary = summarize_target_validity(bundle, config)
    resident = getattr(bundle.datasets['train'], 'resident', None)
    if resident is not None:
        print(json.dumps(dict(data_preparation_seconds=preparation_seconds,
                              resident_spectra_gib=resident.nbytes / 2**30,
                              performance=config.get('performance', {}))), flush=True)
    model = AdversarialLatentFusion.from_config(
        config,
        num_batches=len(bundle.metadata.batch_names),
        num_targets=len(config["data"]["target_columns"]),
    ).to(device)
    ziln_loss = ZILNLoss.from_config(config).to(device)
    class_weights = (
        bundle.batch_class_weights.to(device)
        if config["loss"]["balance_batch_classes"]
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(float(training["beta1"]), float(training["beta2"])),
    )
    start_epoch = 0
    prior_best = math.inf
    resume = training.get("resume_checkpoint")
    if resume:
        start_epoch, prior_best = load_training_checkpoint(
            Path(resume), model, optimizer, device, config, bundle.split_indices["train"]
        )
    stopper = EarlyStopping(
        patience=int(training["early_stopping_patience"]),
        min_delta=float(training["early_stopping_min_delta"]),
        best=prior_best,
    )
    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "execution_startup.json").open('w') as handle:
        json.dump(dict(preparation_seconds=preparation_seconds,
                       resident_spectra_bytes=resident.nbytes if resident is not None else 0,
                       performance=config.get('performance', {})), handle, indent=2)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
    with (output_dir / "data_schema.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "shape": [
                    bundle.metadata.num_observations,
                    bundle.metadata.num_peaks,
                ],
                "batch_names": bundle.metadata.batch_names,
                "target_columns": config["data"]["target_columns"],
                "target_transform": target_contract(config),
                "target_validity": target_summary,
                "split_sizes": {
                    name: int(indices.size)
                    for name, indices in bundle.split_indices.items()
                },
            },
            handle,
            indent=2,
        )

    history: list[dict[str, float]] = []
    history_columns: list[str] = []
    preserved_lines: dict[float, str] = {}
    best_epoch: float | None = None
    selection_start_epoch = 1
    history_path = output_dir / "history.csv"
    try:
        if resume:
            history, history_columns, preserved_lines = load_resume_history(history_path, start_epoch)
            best_epoch, selection_start_epoch = restore_selection_state(stopper, resume_payload, metric, history)
            if checkpoint_selection(resume_payload)["selection_metric"] != metric:
                print(f"Selection changed to {metric}; score and patience reset at epoch {selection_start_epoch}.", flush=True)
            if history and len(preserved_lines) != _history_row_count(history_path):
                history_columns = write_training_history(
                    history_path, history, history_columns, preserved_lines)
            if history:
                refresh_loss_curves(
                    history_path, output_dir / "loss_curves.png",
                    int(config["analysis"]["figure_dpi"]), best_epoch, config,
                    dict(selection_metric=metric, best_epoch=best_epoch, selection_start_epoch=selection_start_epoch))
        if resume and stopper.stale_epochs >= stopper.patience:
            print(f"Early stopping remains in effect after epoch {start_epoch}.", flush=True)
            return output_dir / "best.pt"
        epochs = int(training["epochs"])
        clip = training.get("gradient_clip_norm")
        gradient_clip_norm = None if clip is None else float(clip)
        target_columns = list(config["data"]["target_columns"])
        for epoch in range(start_epoch, epochs):
            train_metrics = run_epoch(
                model=model,
                loader=bundle.loaders["train"],
                ziln_loss=ziln_loss,
                device=device,
                biology_weight=float(config["loss"]["biology_weight"]),
                batch_weight=float(config["loss"]["batch_weight"]),
                he_weight=float(config["loss"].get("he_weight", 1.0)),
                class_weights=class_weights,
                optimizer=optimizer,
                epoch_index=epoch,
                total_epochs=epochs,
                grl_schedule=training["grl_schedule"],
                grl_max_lambda=float(training["grl_max_lambda"]),
                grl_gamma=float(training["grl_gamma"]),
                gradient_clip_norm=gradient_clip_norm,
                cd8_index=target_columns.index(CD8),
                target_columns=target_columns,
                sampling_audit_dir=output_dir / "sampling",
            )
            validation_metrics = run_epoch(
                model=model,
                loader=bundle.loaders["validation"],
                ziln_loss=ziln_loss,
                device=device,
                biology_weight=float(config["loss"]["biology_weight"]),
                batch_weight=float(config["loss"]["batch_weight"]),
                he_weight=float(config["loss"].get("he_weight", 1.0)),
                class_weights=class_weights,
                optimizer=None,
                epoch_index=epoch,
                total_epochs=epochs,
                grl_schedule=training["grl_schedule"],
                grl_max_lambda=float(training["grl_max_lambda"]),
                grl_gamma=float(training["grl_gamma"]),
                gradient_clip_norm=None,
                cd8_index=target_columns.index(CD8),
                target_columns=target_columns,
            )
            row: dict[str, float] = {"epoch": float(epoch + 1)}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update(
                {f"validation_{key}": value for key, value in validation_metrics.items()}
            )
            if not math.isfinite(validation_metrics[metric]):
                raise ValueError(f"Validation requires a finite {metric} for checkpoint selection.")
            improved, should_stop = stopper.update(validation_metrics[metric])
            if improved:
                best_epoch = float(epoch + 1)
            row.update(selected_epoch=best_epoch, selection_start_epoch=float(selection_start_epoch))
            history.append(row)
            history_columns = write_training_history(
                history_path, history, history_columns, preserved_lines)
            refresh_loss_curves(
                history_path, output_dir / "loss_curves.png",
                int(config["analysis"]["figure_dpi"]), best_epoch, config,
                dict(selection_metric=metric, best_epoch=best_epoch, selection_start_epoch=selection_start_epoch))
            if improved:
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    config,
                    bundle,
                    stopper.best,
                    best_epoch=best_epoch, selection_start_epoch=selection_start_epoch, stopper=stopper,
                )
            if (epoch + 1) % int(training["checkpoint_every"]) == 0 or should_stop or epoch + 1 == epochs:
                save_checkpoint(
                    output_dir / "latest.pt",
                    model,
                    optimizer,
                    epoch,
                    config,
                    bundle,
                    stopper.best,
                    best_epoch=best_epoch, selection_start_epoch=selection_start_epoch, stopper=stopper,
                )
            print(
                _format_epoch_summary(
                    epoch + 1,
                    epochs,
                    train_metrics,
                    validation_metrics,
                    target_columns,
                    metric,
                ),
                flush=True,
            )
            if should_stop:
                print(f"Early stopping after epoch {epoch + 1}.", flush=True)
                break
    finally:
        for dataset in bundle.datasets.values():
            dataset.close()
    return output_dir / "best.pt"


def main(argv: Sequence[str] | None = None) -> int:
    """Load configuration and run training.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        int: Process exit status, zero on success.
    """

    args = parse_args(argv)
    config = load_config(args.config)
    if args.smoke_test:
        config = apply_smoke_overrides(config)
    best_path = train_model(config)
    print(f"Best checkpoint: {best_path}", flush=True)
    # The training frame and datasets have been released before downstream work.
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from dann.workflow import PostTrainingError, run_post_training_workflow
    try:
        run_post_training_workflow(config, best_path)
    except PostTrainingError as error:
        print(str(error), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

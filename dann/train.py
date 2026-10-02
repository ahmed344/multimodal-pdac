"""YAML-driven training entry point for adversarial latent fusion."""

from __future__ import annotations

import argparse
import json
import math
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
from dann.checkpoints import (architecture_contract, validate_checkpoint, training_data_contract,
                              validate_training_data_contract, sampling_contract, validate_sampling_contract)
from dann.tiles import row_identity
from dann.targets import CD8, checkpoint_contract, target_contract


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
            value (float): Current validation biology loss.

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

    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


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
) -> str:
    """Build the epoch log with one aligned line per IHC target.

    Args:
        epoch (int): Completed one-based epoch number.
        epochs (int): Configured maximum epoch count.
        train_metrics (Mapping[str, float]): Finalized training metrics.
        validation_metrics (Mapping[str, float]): Finalized validation metrics.
        target_columns (Sequence[str]): Ordered target column names.

    Returns:
        str: Loss line, macro fit line, and one ``[Name]`` line per target.
    """

    loss_parts = (
        ("biology_loss", "biology_loss", 6),
        ("batch_loss", "batch_loss", 6),
        ("hurdle_loss", "hurdle_loss", 6),
        ("mean_loss", "positive_loss", 6),
        ("cd8_loss", "cd8_loss", 6),
    )
    fit_parts = (
        ("batch_acc", "batch_accuracy", 4),
        ("hurdle_acc", "hurdle_acc", 4),
        ("mean_r2", "mean_r2", 4),
    )
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
) -> dict[str, float]:
    """Run one training or validation epoch and report loss components.

    Args:
        model (AdversarialLatentFusion): Model being optimized or evaluated.
        loader (torch.utils.data.DataLoader): Sparse batch loader.
        ziln_loss (ZILNLoss): Biology objective.
        device (torch.device): Compute device.
        biology_weight (float): Biology objective multiplier.
        batch_weight (float): Discriminator objective multiplier.
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
    model.train(training)
    num_targets = int(ziln_loss.target_weights.numel())
    if target_columns is None:
        target_columns = [f"target_{index}" for index in range(num_targets)]
    elif len(target_columns) != num_targets:
        raise ValueError(
            f"Expected {num_targets} target columns, received {len(target_columns)}."
        )
    metrics = _empty_metrics(num_targets)
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
    with torch.set_grad_enabled(training):
        for step, cpu_batch in enumerate(loader):
            batch = move_batch_to_device(cpu_batch, device)
            progress = (epoch_index * steps_per_epoch + step) / max(
                total_epochs * steps_per_epoch - 1, 1
            )
            strength = grl_strength(
                progress, grl_schedule, grl_max_lambda, grl_gamma
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(batch, grl_strength=strength)
            biology = ziln_loss(
                outputs["pi_logits"],
                outputs["mu"],
                outputs["sigma"],
                batch["targets"],
                batch.get("target_valid_mask"),
            )
            with torch.no_grad():
                selected = slice(cd8_index, cd8_index + 1)
                mask = batch.get("target_valid_mask", torch.ones_like(batch["targets"], dtype=torch.bool))[:, selected]
                cd8 = cd8_loss(outputs["pi_logits"][:, selected], outputs["mu"][:, selected],
                               outputs["sigma"][:, selected], batch["targets"][:, selected], mask)
                metrics["cd8_sum"] += float(cd8.total)
                metrics["cd8_count"] += int(mask.sum())
                _update_fit_statistics(
                    metrics, outputs, batch, ziln_loss.logit_epsilon
                )
            discriminator = F.cross_entropy(
                outputs["batch_logits"], batch["batches"], weight=class_weights
            )
            total = biology_weight * biology.total + batch_weight * discriminator
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite objective at epoch {epoch_index + 1}, step {step + 1}."
                )
            if training:
                total.backward()
                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
                if audit is not None and "occupancy" in cpu_batch:
                    composition = audit.update(cpu_batch)
                    with (sampling_audit_dir / f"epoch_{epoch_index:04d}_batches.jsonl").open("a") as handle:
                        handle.write(json.dumps(composition) + "\n")

            batch_size = int(batch["targets"].shape[0])
            metrics["samples"] += batch_size
            metrics["total"] += float(total.detach()) * batch_size
            mass = (
                float(biology.valid_weight_mass.detach())
                if ziln_loss.reduction == "mean" else batch_size
            )
            metrics["biology_mass"] += mass
            metrics["biology"] += float(biology.total.detach()) * mass
            metrics["hurdle"] += float(biology.hurdle.detach()) * mass
            metrics["positive"] += float(biology.positive.detach()) * mass
            metrics["batch"] += float(discriminator.detach()) * batch_size
            predictions = outputs["batch_logits"].argmax(dim=1)
            metrics["batch_correct"] += float(
                (predictions == batch["batches"]).sum().detach()
            )
    finalized = _finalize_metrics(metrics, target_columns)
    finalized["total_loss"] = (
        biology_weight * finalized["biology_loss"] + batch_weight * finalized["batch_loss"]
    )
    if audit is not None and audit.compositions:
        with (sampling_audit_dir / f"epoch_{epoch_index:04d}_summary.json").open("w") as handle:
            json.dump(audit.summary(), handle, indent=2)
    return finalized


def save_checkpoint(
    path: Path,
    model: AdversarialLatentFusion,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Mapping[str, Any],
    bundle: DataBundle,
    best_validation_biology: float,
) -> None:
    """Persist model, optimizer, schema, split, and configuration state.

    Args:
        path (Path): Destination checkpoint path.
        model (AdversarialLatentFusion): Trained model.
        optimizer (torch.optim.Optimizer): Optimizer state.
        epoch (int): Last completed zero-based epoch.
        config (Mapping[str, Any]): Effective configuration.
        bundle (DataBundle): Data metadata and selected split indices.
        best_validation_biology (float): Best validation biology loss.

    Returns:
        None: Checkpoint is written atomically through a temporary file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    identity = row_identity(Path(config["data"]["path"]), config["data"])
    payload = {
        "sampling_contract": sampling_contract(config, bundle.split_indices["train"]),
        "sampler_state": {"completed_epoch": int(epoch), "sampler_epoch": int(epoch),
                          "next_epoch": int(epoch) + 1},
        "architecture": architecture_contract(config),
        "optimizer_parameter_names": list(dict(model.named_parameters())),
        "data_identity": identity,
        "training_data_contract": training_data_contract(config, identity),
        "selection_metric": "mean_valid_cd8_ziln",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "config": dict(config),
        "batch_names": bundle.metadata.batch_names,
        "target_columns": tuple(config["data"]["target_columns"]),
        "target_transform": target_contract(config),
        "split_indices": bundle.split_indices,
        "best_validation_biology": float(best_validation_biology),
        "best_validation_cd8": float(best_validation_biology),
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
        tuple[int, float]: Next epoch index and prior best validation biology loss.
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint_contract(checkpoint) != target_contract(config):
        raise ValueError("Resume target transformation or target order is incompatible with checkpoint.")
    validate_training_data_contract(checkpoint, config)
    validate_checkpoint(checkpoint, config, model, optimizer=True)
    validate_sampling_contract(checkpoint, config, selected_rows)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint["epoch"]) + 1, float(
        checkpoint.get("best_validation_biology", math.inf) if checkpoint.get("selection_metric") == "mean_valid_cd8_ziln" else math.inf
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

    training = config["training"]
    seed_everything(
        int(training["seed"]), bool(training["deterministic_algorithms"])
    )
    device = resolve_device(str(training["device"]))
    resume_payload = (torch.load(training["resume_checkpoint"], map_location="cpu", weights_only=False)
                      if training.get("resume_checkpoint") else {})
    if resume_payload:
        validate_training_data_contract(resume_payload, config)
    bundle = create_data_bundle(config, resume_payload.get("split_indices"), resume_payload.get("data_identity"))
    if resume_payload and tuple(resume_payload["batch_names"]) != bundle.metadata.batch_names:
        raise ValueError("Resume batch order is incompatible.")
    target_summary = summarize_target_validity(bundle, config)
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
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        if not math.isfinite(validation_metrics["cd8_loss"]):
            raise ValueError("Validation requires valid CD8 observations for checkpoint selection.")
        improved, should_stop = stopper.update(validation_metrics["cd8_loss"])
        if improved:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                config,
                bundle,
                stopper.best,
            )
        if (epoch + 1) % int(training["checkpoint_every"]) == 0:
            save_checkpoint(
                output_dir / "latest.pt",
                model,
                optimizer,
                epoch,
                config,
                bundle,
                stopper.best,
            )
        print(
            _format_epoch_summary(
                epoch + 1,
                epochs,
                train_metrics,
                validation_metrics,
                target_columns,
            ),
            flush=True,
        )
        if should_stop:
            print(f"Early stopping after epoch {epoch + 1}.", flush=True)
            break
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""YAML-driven training entry point for adversarial latent fusion."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from dann.config import apply_smoke_overrides, load_config, resolve_device, seed_everything
from dann.data_loader import DataBundle, create_data_bundle
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion


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


def _empty_metrics() -> dict[str, float]:
    """Create metric accumulators for one epoch.

    Args:
        None.

    Returns:
        dict[str, float]: Zero-initialized sums and counts.
    """

    return {
        "samples": 0.0,
        "total": 0.0,
        "biology": 0.0,
        "hurdle": 0.0,
        "positive": 0.0,
        "batch": 0.0,
        "batch_correct": 0.0,
    }


def _finalize_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Convert sample-weighted accumulators to epoch means.

    Args:
        metrics (Mapping[str, float]): Raw epoch sums.

    Returns:
        dict[str, float]: Mean losses and batch accuracy.
    """

    samples = max(float(metrics["samples"]), 1.0)
    return {
        "total_loss": metrics["total"] / samples,
        "biology_loss": metrics["biology"] / samples,
        "hurdle_loss": metrics["hurdle"] / samples,
        "positive_loss": metrics["positive"] / samples,
        "batch_loss": metrics["batch"] / samples,
        "batch_accuracy": metrics["batch_correct"] / samples,
    }


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

    Returns:
        dict[str, float]: Mean total/component losses and batch accuracy.
    """

    training = optimizer is not None
    model.train(training)
    metrics = _empty_metrics()
    steps_per_epoch = max(len(loader), 1)
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

            batch_size = int(batch["targets"].shape[0])
            metrics["samples"] += batch_size
            metrics["total"] += float(total.detach()) * batch_size
            metrics["biology"] += float(biology.total.detach()) * batch_size
            metrics["hurdle"] += float(biology.hurdle.detach()) * batch_size
            metrics["positive"] += float(biology.positive.detach()) * batch_size
            metrics["batch"] += float(discriminator.detach()) * batch_size
            predictions = outputs["batch_logits"].argmax(dim=1)
            metrics["batch_correct"] += float(
                (predictions == batch["batches"]).sum().detach()
            )
    return _finalize_metrics(metrics)


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
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "config": dict(config),
        "batch_names": bundle.metadata.batch_names,
        "target_columns": tuple(config["data"]["target_columns"]),
        "split_indices": bundle.split_indices,
        "best_validation_biology": float(best_validation_biology),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_checkpoint(
    path: Path,
    model: AdversarialLatentFusion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    """Restore model and optimizer state for resumed training.

    Args:
        path (Path): Existing checkpoint path.
        model (AdversarialLatentFusion): Model receiving saved parameters.
        optimizer (torch.optim.Optimizer): Optimizer receiving saved state.
        device (torch.device): Tensor map location.

    Returns:
        tuple[int, float]: Next epoch index and prior best validation biology loss.
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint["epoch"]) + 1, float(
        checkpoint.get("best_validation_biology", math.inf)
    )


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
    bundle = create_data_bundle(config)
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
            Path(resume), model, optimizer, device
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
        )
        row: dict[str, float] = {"epoch": float(epoch + 1)}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {f"validation_{key}": value for key, value in validation_metrics.items()}
        )
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        improved, should_stop = stopper.update(validation_metrics["biology_loss"])
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
            f"epoch={epoch + 1}/{epochs} "
            f"train_bio={train_metrics['biology_loss']:.6f} "
            f"validation_bio={validation_metrics['biology_loss']:.6f} "
            f"validation_batch_acc={validation_metrics['batch_accuracy']:.4f}",
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

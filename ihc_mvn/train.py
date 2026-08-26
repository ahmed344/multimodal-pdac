"""Train the standalone sparse IHC multivariate-normal hurdle model."""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .config import apply_smoke_overrides, load_config, resolve_device, seed_everything
from .data_loader import DataBundle, create_data_bundle
from .losses import MVNHurdleLoss
from .model import IHCMultivariateModel


@dataclass
class EarlyStopping:
    """Track validation-loss improvements and stopping patience."""

    patience: int
    min_delta: float
    best: float = math.inf
    stale_epochs: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        """Update early-stopping state from one validation loss.

        Args:
            value (float): Current finite validation total loss.

        Returns:
            tuple[bool, bool]: Improvement flag and stopping flag.
        """

        if not math.isfinite(value):
            raise FloatingPointError("Validation loss is non-finite.")
        improved = value < self.best - self.min_delta
        if improved:
            self.best = value
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return improved, self.stale_epochs >= self.patience


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse standalone training command-line arguments.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        argparse.Namespace: Parsed configuration path and smoke-test flag.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="IHC-MVN YAML configuration path.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Apply configured sample caps and train for one epoch.",
    )
    return parser.parse_args(argv)


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move every collated batch tensor to a runtime device.

    Args:
        batch (Mapping[str, torch.Tensor]): CPU batch from ``sparse_collate``.
        device (torch.device): Destination training or evaluation device.

    Returns:
        dict[str, torch.Tensor]: Device-resident batch with unchanged keys.
    """

    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def resolved_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fill optional model, loss, and optimization settings with defaults.

    Args:
        config (Mapping[str, Any]): Validated base IHC-MVN configuration.

    Returns:
        dict[str, Any]: Deep-copied effective configuration.
    """

    resolved = copy.deepcopy(dict(config))
    training = resolved.setdefault("training", {})
    training.setdefault("epochs", 100)
    training.setdefault("learning_rate", 1.0e-3)
    training.setdefault("weight_decay", 1.0e-4)
    training.setdefault("beta1", 0.9)
    training.setdefault("beta2", 0.999)
    training.setdefault("gradient_clip_norm", 5.0)
    training.setdefault("early_stopping_patience", 15)
    training.setdefault("early_stopping_min_delta", 0.0)
    training.setdefault("resume_checkpoint", None)
    resolved.setdefault("model", {})
    resolved.setdefault("loss", {})
    resolved["loss"].setdefault("prior_weight", 1.0)
    return resolved


def build_model(config: Mapping[str, Any], bundle: DataBundle) -> IHCMultivariateModel:
    """Instantiate a model with data-derived feature and category counts.

    Args:
        config (Mapping[str, Any]): Effective complete configuration.
        bundle (DataBundle): Prepared training data and fitted mappings.

    Returns:
        IHCMultivariateModel: Newly initialized standalone model.
    """

    model_config = copy.deepcopy(dict(config))
    model_config.setdefault("model", {})["num_peaks"] = bundle.metadata.num_features
    return IHCMultivariateModel.from_config(
        model_config,
        num_patients=len(bundle.metadata.patient_mapping.names),
        num_slides=len(bundle.metadata.slide_mapping.names),
        num_targets=len(bundle.target_standardizer.target_names),
    )


def _empty_metrics() -> dict[str, float]:
    """Create sample-weighted epoch metric accumulators.

    Args:
        None.

    Returns:
        dict[str, float]: Zero-initialized loss sums and observation counts.
    """

    return {
        "samples": 0.0,
        "total": 0.0,
        "hurdle": 0.0,
        "positive": 0.0,
        "prior": 0.0,
        "positive_pixels": 0.0,
        "positive_targets": 0.0,
    }


def _finalize_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Convert raw sample-weighted accumulators to epoch metrics.

    Args:
        metrics (Mapping[str, float]): Completed epoch accumulators.

    Returns:
        dict[str, float]: Mean loss components and positive observation counts.
    """

    samples = float(metrics["samples"])
    if samples <= 0.0:
        raise ValueError("Cannot finalize metrics for an empty loader.")
    return {
        "total_loss": float(metrics["total"]) / samples,
        "hurdle_loss": float(metrics["hurdle"]) / samples,
        "positive_loss": float(metrics["positive"]) / samples,
        "prior_loss": float(metrics["prior"]) / samples,
        "samples": samples,
        "positive_pixels": float(metrics["positive_pixels"]),
        "positive_targets": float(metrics["positive_targets"]),
    }


def validate_finite_outputs(predictions: Mapping[str, torch.Tensor]) -> None:
    """Reject non-finite floating-point model outputs.

    Args:
        predictions (Mapping[str, torch.Tensor]): Model prediction mapping.

    Returns:
        None: Validation succeeds or raises ``FloatingPointError``.
    """

    for key, value in predictions.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"Model output {key!r} contains non-finite values.")


def run_epoch(
    model: IHCMultivariateModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    criterion: MVNHurdleLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> dict[str, float]:
    """Run one train or evaluation epoch with the exact hurdle/MVN inputs.

    Args:
        model (IHCMultivariateModel): Model to optimize or evaluate.
        loader (DataLoader[dict[str, torch.Tensor]]): Sparse labeled batch loader.
        criterion (MVNHurdleLoss): Hurdle and masked-Gaussian objective.
        device (torch.device): Runtime tensor device.
        optimizer (torch.optim.Optimizer | None): Optimizer for training, or ``None``.
        gradient_clip_norm (float | None): Optional positive global gradient norm cap.

    Returns:
        dict[str, float]: Sample-weighted loss metrics and positive counts.
    """

    training = optimizer is not None
    if gradient_clip_norm is not None and gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive when provided.")
    model.train(training)
    metrics = _empty_metrics()
    with torch.set_grad_enabled(training):
        for step, cpu_batch in enumerate(loader, start=1):
            batch = move_batch_to_device(cpu_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(batch)
            validate_finite_outputs(predictions)
            loss = criterion(
                predictions,
                targets=batch["target_densities"],
                standardized_targets=batch["targets"],
                prior_nll=model.prior_nll(),
            )
            components = (loss.total, loss.hurdle, loss.positive, loss.prior)
            if not all(torch.isfinite(component).all() for component in components):
                raise FloatingPointError(
                    f"Non-finite objective component at loader step {step}."
                )
            if training:
                loss.total.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(
                            f"Non-finite gradient at loader step {step}."
                        )
                if gradient_clip_norm is not None:
                    gradient_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError(
                            f"Non-finite gradient norm at loader step {step}."
                        )
                optimizer.step()
                for parameter in model.parameters():
                    if not torch.isfinite(parameter).all():
                        raise FloatingPointError(
                            f"Non-finite parameter after loader step {step}."
                        )

            batch_size = int(batch["target_densities"].shape[0])
            metrics["samples"] += batch_size
            metrics["total"] += float(loss.total.detach()) * batch_size
            metrics["hurdle"] += float(loss.hurdle.detach()) * batch_size
            metrics["positive"] += float(loss.positive.detach()) * batch_size
            metrics["prior"] += float(loss.prior.detach()) * batch_size
            metrics["positive_pixels"] += int(loss.positive_pixel_count.detach())
            metrics["positive_targets"] += int(loss.positive_target_count.detach())
    return _finalize_metrics(metrics)


def checkpoint_payload(
    model: IHCMultivariateModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Mapping[str, Any],
    bundle: DataBundle,
    best_validation: float,
    stale_epochs: int,
) -> dict[str, Any]:
    """Build a complete resumable checkpoint payload.

    Args:
        model (IHCMultivariateModel): Current trained model.
        optimizer (torch.optim.Optimizer): Current AdamW optimizer.
        epoch (int): Last completed zero-based epoch.
        config (Mapping[str, Any]): Effective configuration.
        bundle (DataBundle): Data splits, schemas, mappings, and fitted transforms.
        best_validation (float): Best validation total loss observed so far.
        stale_epochs (int): Current early-stopping stale-epoch count.

    Returns:
        dict[str, Any]: Serializable model, optimizer, schema, and preprocessing state.
    """

    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "config": copy.deepcopy(dict(config)),
        "split_indices": {
            name: np.asarray(indices, dtype=np.int64)
            for name, indices in bundle.split_indices.items()
        },
        "feature_names": tuple(bundle.metadata.feature_names),
        "feature_order": tuple(bundle.metadata.feature_names),
        "scaler_state": bundle.scaler.to_state(),
        "target_standardizer_state": bundle.target_standardizer.to_state(),
        "slide_mapping_state": bundle.metadata.slide_mapping.to_state(),
        "patient_mapping_state": bundle.metadata.patient_mapping.to_state(),
        "target_columns": tuple(bundle.target_standardizer.target_names),
        "best_validation": float(best_validation),
        "best_validation_loss": float(best_validation),
        "early_stopping_stale_epochs": int(stale_epochs),
        "num_features": int(bundle.metadata.num_features),
        "num_patients": len(bundle.metadata.patient_mapping.names),
        "num_slides": len(bundle.metadata.slide_mapping.names),
    }


def atomic_save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically save a PyTorch checkpoint in its destination directory.

    Args:
        path (Path): Final checkpoint path.
        payload (Mapping[str, Any]): Complete serializable checkpoint mapping.

    Returns:
        None: The destination is atomically replaced after a successful write.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def validate_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    bundle: DataBundle,
) -> None:
    """Validate resumed split, feature, target, and mapping state.

    Args:
        checkpoint (Mapping[str, Any]): Loaded training checkpoint.
        bundle (DataBundle): Newly prepared deterministic data bundle.

    Returns:
        None: Exact schema agreement is confirmed or an exception is raised.
    """

    required = {
        "model_state",
        "optimizer_state",
        "epoch",
        "split_indices",
        "feature_names",
        "scaler_state",
        "target_standardizer_state",
        "slide_mapping_state",
        "patient_mapping_state",
        "target_columns",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Resume checkpoint is missing keys: {sorted(missing)}")
    if tuple(checkpoint["feature_names"]) != bundle.metadata.feature_names:
        raise ValueError("Resume checkpoint feature names or order do not match.")
    if tuple(checkpoint["target_columns"]) != bundle.target_standardizer.target_names:
        raise ValueError("Resume checkpoint target columns or order do not match.")
    for name, indices in bundle.split_indices.items():
        stored = np.asarray(checkpoint["split_indices"][name], dtype=np.int64)
        if not np.array_equal(stored, indices):
            raise ValueError(f"Resume checkpoint {name!r} split indices do not match.")
    expected_states = {
        "scaler_state": bundle.scaler.to_state(),
        "target_standardizer_state": bundle.target_standardizer.to_state(),
        "slide_mapping_state": bundle.metadata.slide_mapping.to_state(),
        "patient_mapping_state": bundle.metadata.patient_mapping.to_state(),
    }
    for key, expected in expected_states.items():
        if checkpoint[key] != expected:
            raise ValueError(f"Resume checkpoint {key} does not match prepared data.")


def load_training_checkpoint(
    path: Path,
    model: IHCMultivariateModel,
    optimizer: torch.optim.Optimizer,
    bundle: DataBundle,
    device: torch.device,
) -> tuple[int, float, int]:
    """Restore and validate a complete training checkpoint.

    Args:
        path (Path): Existing checkpoint path.
        model (IHCMultivariateModel): Model receiving parameter state.
        optimizer (torch.optim.Optimizer): Optimizer receiving AdamW state.
        bundle (DataBundle): Current deterministic data preparation.
        device (torch.device): Runtime map location.

    Returns:
        tuple[int, float, int]: Next epoch, best validation loss, and stale epochs.
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Training checkpoint root must be a mapping.")
    validate_resume_checkpoint(checkpoint, bundle)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(
            checkpoint.get(
                "best_validation",
                checkpoint.get("best_validation_loss", math.inf),
            )
        ),
        int(checkpoint.get("early_stopping_stale_epochs", 0)),
    )


def _load_history(path: Path, start_epoch: int) -> list[dict[str, float]]:
    """Load history rows compatible with a resumed epoch index.

    Args:
        path (Path): Existing or future history CSV path.
        start_epoch (int): Zero-based epoch index at which training resumes.

    Returns:
        list[dict[str, float]]: Existing rows ending at ``start_epoch``.
    """

    if start_epoch == 0 or not path.exists():
        return []
    frame = pd.read_csv(path)
    rows = frame.to_dict(orient="records")
    if len(rows) != start_epoch:
        raise ValueError(
            f"History has {len(rows)} rows but resume expects {start_epoch}."
        )
    return [
        {str(key): float(value) for key, value in row.items()}
        for row in rows
    ]


def _write_history(path: Path, rows: Sequence[Mapping[str, float]]) -> None:
    """Atomically write training history as CSV.

    Args:
        path (Path): Final history CSV path.
        rows (Sequence[Mapping[str, float]]): Ordered epoch metric rows.

    Returns:
        None: The CSV is atomically replaced.
    """

    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def train_model(config: Mapping[str, Any]) -> Path:
    """Train, validate, test, and checkpoint the standalone IHC-MVN model.

    Args:
        config (Mapping[str, Any]): Validated and optionally smoke-adjusted config.

    Returns:
        Path: Atomic best-validation checkpoint path.
    """

    resolved = resolved_training_config(config)
    training = resolved["training"]
    seed_everything(
        int(training["seed"]),
        bool(training.get("deterministic_algorithms", False)),
    )
    device = resolve_device(str(training.get("device", "auto")))
    bundle = create_data_bundle(resolved)
    resolved["model"]["num_peaks"] = bundle.metadata.num_features
    model = build_model(resolved, bundle).to(device)
    criterion = MVNHurdleLoss(
        training_sample_count=len(bundle.datasets["train"]),
        prior_weight=float(resolved["loss"].get("prior_weight", 1.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1.0e-3)),
        weight_decay=float(training.get("weight_decay", 1.0e-4)),
        betas=(
            float(training.get("beta1", 0.9)),
            float(training.get("beta2", 0.999)),
        ),
    )

    output_dir = Path(
        training.get("output_dir", resolved["output"]["directory"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)

    start_epoch = 0
    best_validation = math.inf
    stale_epochs = 0
    resume = training.get("resume_checkpoint")
    if resume:
        start_epoch, best_validation, stale_epochs = load_training_checkpoint(
            Path(str(resume)), model, optimizer, bundle, device
        )
    stopper = EarlyStopping(
        patience=int(training.get("early_stopping_patience", 15)),
        min_delta=float(training.get("early_stopping_min_delta", 0.0)),
        best=best_validation,
        stale_epochs=stale_epochs,
    )
    if stopper.patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")

    history_path = output_dir / "history.csv"
    history = _load_history(history_path, start_epoch)
    epochs = int(training.get("epochs", 100))
    if epochs <= 0:
        raise ValueError("training.epochs must be positive.")
    clip_value = training.get("gradient_clip_norm", 5.0)
    gradient_clip_norm = None if clip_value is None else float(clip_value)
    try:
        for epoch in range(start_epoch, epochs):
            train_metrics = run_epoch(
                model,
                bundle.loaders["train"],
                criterion,
                device,
                optimizer=optimizer,
                gradient_clip_norm=gradient_clip_norm,
            )
            validation_metrics = run_epoch(
                model,
                bundle.loaders["validation"],
                criterion,
                device,
            )
            improved, should_stop = stopper.update(
                validation_metrics["total_loss"]
            )
            row: dict[str, float] = {"epoch": float(epoch + 1)}
            row.update(
                {f"train_{key}": value for key, value in train_metrics.items()}
            )
            row.update(
                {
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                }
            )
            history.append(row)
            _write_history(history_path, history)

            payload = checkpoint_payload(
                model,
                optimizer,
                epoch,
                resolved,
                bundle,
                stopper.best,
                stopper.stale_epochs,
            )
            if improved:
                atomic_save_checkpoint(output_dir / "best.pt", payload)
            atomic_save_checkpoint(output_dir / "latest.pt", payload)
            print(
                f"epoch={epoch + 1}/{epochs} "
                f"train={train_metrics['total_loss']:.6f} "
                f"validation={validation_metrics['total_loss']:.6f}",
                flush=True,
            )
            if should_stop:
                print(f"Early stopping after epoch {epoch + 1}.", flush=True)
                break

        best_path = output_dir / "best.pt"
        if not best_path.exists():
            raise RuntimeError("Training completed without producing best.pt.")
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state"])
        test_metrics = run_epoch(
            model,
            bundle.loaders["test"],
            criterion,
            device,
        )
        if history:
            history[-1].update(
                {f"test_{key}": value for key, value in test_metrics.items()}
            )
            _write_history(history_path, history)
        print(
            f"test={test_metrics['total_loss']:.6f} "
            f"best_validation={stopper.best:.6f}",
            flush=True,
        )
        return best_path
    finally:
        for dataset in bundle.datasets.values():
            dataset.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Load configuration and execute standalone training.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        int: Zero process status after successful training.
    """

    args = parse_args(argv)
    config = load_config(args.config)
    if args.smoke_test:
        config = apply_smoke_overrides(config)
        config["training"]["epochs"] = 1
    best_path = train_model(config)
    print(f"Best checkpoint: {best_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Configuration loading and runtime helpers for the DANN pipeline."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate a YAML configuration.

    Args:
        path (Path): YAML configuration path.

    Returns:
        dict[str, Any]: Mutable nested configuration dictionary.
    """

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required_sections = {"data", "model", "loss", "training", "analysis"}
    missing = required_sections.difference(config)
    if missing:
        raise KeyError(f"Missing configuration sections: {sorted(missing)}")
    fractions = [
        float(config["data"][f"{name}_fraction"])
        for name in ("train", "validation", "test")
    ]
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Data split fractions must sum to one, observed {fractions}.")
    if any(value < 0.0 for value in fractions):
        raise ValueError("Data split fractions cannot be negative.")
    return config


def apply_smoke_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create a configuration capped for a fast real-data smoke run.

    Args:
        config (Mapping[str, Any]): Complete user configuration.

    Returns:
        dict[str, Any]: Deep copy containing smoke-test overrides.
    """

    updated = copy.deepcopy(dict(config))
    data = updated["data"]
    training = updated["training"]
    data["max_train_samples"] = int(training["smoke_train_samples"])
    data["max_validation_samples"] = int(training["smoke_validation_samples"])
    data["max_test_samples"] = int(training["smoke_test_samples"])
    training["epochs"] = int(training["smoke_epochs"])
    training["num_workers"] = int(training["smoke_num_workers"])
    training["early_stopping_patience"] = max(int(training["epochs"]), 1)
    updated["analysis"]["max_samples"] = int(training["smoke_test_samples"])
    updated["analysis"]["num_workers"] = int(training["smoke_num_workers"])
    updated["analysis"]["activity_max_nonzeros"] = 5_000_000
    output_dir = Path(training["output_dir"]) / "smoke"
    training["output_dir"] = str(output_dir)
    updated["analysis"]["output_dir"] = str(output_dir / "analysis")
    updated["analysis"]["checkpoint"] = str(output_dir / "best.pt")
    return updated


def resolve_device(requested: str) -> torch.device:
    """Resolve a configured runtime device.

    Args:
        requested (str): Device string or ``"auto"``.

    Returns:
        torch.device: Available PyTorch device.
    """

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested but CUDA is unavailable.")
    return device


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    Args:
        seed (int): Reproducibility seed.
        deterministic (bool): Whether to request deterministic PyTorch algorithms.

    Returns:
        None: Global random state is updated in place.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

"""Configuration loading and runtime helpers for the DANN pipeline."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


def density_sampling_settings(analysis: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve and validate Monte Carlo settings, including legacy defaults."""
    count = analysis.get("density_mc_samples", 1000)
    seed = analysis.get("density_mc_seed", 20260719)
    for name, value, minimum in (
        ("density_mc_samples", count, 1), ("density_mc_seed", seed, 0)
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"analysis.{name} must be an integer >= {minimum}.")
    return count, seed


def latent_variance_settings(settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve standalone latent-variance defaults and validate their types."""
    root = "/workspaces/multimodal-pdac/data/PDAC"
    defaults = {
        "input": f"{root}/Raw/adata_assembled_tissue.h5ad",
        "latents": f"{root}/Results/dann/analysis/spatial/spatial_latent.parquet",
        "output_dir": f"{root}/Results/dann/latent_variance",
        "slide_column": "batch",
        "chunk_size": 8192,
        "figure_dpi": 300,
    }
    if settings is not None and not isinstance(settings, Mapping):
        raise ValueError("latent_variance must be a mapping.")
    resolved = {**defaults, **(settings or {})}
    for key in ("input", "latents", "output_dir", "slide_column"):
        if not isinstance(resolved[key], str) or not resolved[key].strip():
            raise ValueError(f"latent_variance.{key} must be a nonempty string.")
    for key in ("chunk_size", "figure_dpi"):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"latent_variance.{key} must be a positive integer.")
    return resolved


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
    max_samples = config["analysis"].get("max_samples")
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples <= 0
    ):
        raise ValueError("analysis.max_samples must be null or a positive integer.")
    count, seed = density_sampling_settings(config["analysis"])
    config["analysis"].update(density_mc_samples=count, density_mc_seed=seed)
    config["latent_variance"] = latent_variance_settings(config.get("latent_variance"))
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

"""Validated settings for MSI QC; all intensities are in the selected layer's units."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load configuration, rejecting ambiguous thresholds and invalid budgets."""
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("QC configuration must be a mapping.")
    for key in ("input", "output", "layer"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{key} must be a nonempty string.")
    for key in ("block_rows", "spatial_pairs", "shuffles", "min_spatial_detections",
                "sample_values_per_slide", "example_slides"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer.")
    thresholds = np.asarray(config.get("thresholds"), dtype=float)
    if (thresholds.ndim != 1 or len(thresholds) < 2 or not np.isfinite(thresholds).all()
            or thresholds[0] != 0 or np.any(np.diff(thresholds) <= 0)):
        raise ValueError("thresholds must start at zero and increase strictly.")
    for key in ("retained_signal_min", "retained_measurements_min", "weak_fraction_screen"):
        value = float(config[key])
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{key} must be in (0, 1].")
    for key in ("spatial_support_excess", "spatial_weak_excess", "mad_multiplier"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite.")
    if config["spatial_weak_excess"] >= config["spatial_support_excess"]:
        raise ValueError("The weak spatial screen must be below the support screen.")
    if config["weak_intensity_screen"] not in config["thresholds"]:
        raise ValueError("weak_intensity_screen must be one of the tested thresholds.")
    prevalence = np.asarray(config["prevalence_screens"], dtype=float)
    if (prevalence.ndim != 1 or not len(prevalence) or not np.isfinite(prevalence).all()
            or np.any(prevalence <= 0) or np.any(prevalence >= 1)
            or np.any(np.diff(prevalence) <= 0)):
        raise ValueError("prevalence_screens must increase within (0, 1).")
    return config

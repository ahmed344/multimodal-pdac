"""Configuration validation for independent, optional preprocessing stages."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load all settings, rejecting invalid budgets and ambiguous stage toggles."""
    config = yaml.safe_load(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate numerical settings before opening inputs or creating artifacts."""
    if not isinstance(config, dict):
        raise ValueError("Preprocessing settings must be a mapping.")
    for key in ("lesion_input", "tissue_input", "output", "dann_config", "patient_column"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{key} must be a nonempty string.")
    for group in ("stages", "selection", "integration", "local"):
        if not isinstance(config.get(group), dict):
            raise ValueError(f"Missing settings: {group}")
    for key in ("selection", "integration", "local"):
        if type(config["stages"].get(key)) is not bool:
            raise ValueError(f"stages.{key} must be boolean.")
    workers = config.get("application_workers", 4)
    if type(workers) is not int or workers < 1:
        raise ValueError("application_workers must be a positive integer.")
    integers = [(config, k, 1) for k in ("block_rows", "feature_block", "report_peaks")]
    integers += [(config, "seed", 0)]
    integers += [(config["selection"], k, 1) for k in
                 ("min_spatial_detections", "spatial_pairs", "shuffles")]
    integers += [(config["selection"], "patch_pixels", 4),
                 (config["integration"], "min_anchors", 1),
                 (config["local"], "draws", 2), (config["local"], "min_neighbors", 1)]
    for group, key, lower in integers:
        if type(group.get(key)) is not int or group[key] < lower:
            raise ValueError(f"{key} must be an integer >= {lower}.")
    if config["local"]["min_neighbors"] > 4:
        raise ValueError("min_neighbors must allow assessment of one-step neighborhoods (<=4).")
    positive = [(config, k) for k in ("memory_gib", "disk_reserve_gib")]
    positive += [(config["selection"], k) for k in
                 ("weak_intensity_screen", "spatial_weak_excess", "spatial_support_excess")]
    positive += [(config["local"], "weak_boundary"), (config["integration"], "max_correction")]
    for group, key in positive:
        if not np.isfinite(group.get(key, np.nan)) or group[key] <= 0:
            raise ValueError(f"{key} must be positive and finite.")
    for group, key in [(config["selection"], "weak_fraction_screen"),
                       *[(config["integration"], k) for k in
                         ("prevalence", "slide_fraction", "support_fraction")],
                       (config["local"], "envelope")]:
        if not np.isfinite(group.get(key, np.nan)) or not 0 < group[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1].")
    if config["selection"]["spatial_weak_excess"] >= config["selection"]["spatial_support_excess"]:
        raise ValueError("Weak excess must be smaller than supported excess.")
    if config["integration"]["max_correction"] < 1:
        raise ValueError("max_correction must be >= 1.")
    cuts = np.asarray(config["local"].get("sensitivity"), dtype=float)
    if (cuts.ndim != 1 or len(cuts) == 0 or not np.isfinite(cuts).all()
            or np.any(cuts <= 0) or np.any(np.diff(cuts) <= 0)):
        raise ValueError("sensitivity must be strictly increasing positive finite values.")

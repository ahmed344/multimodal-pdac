"""Shared, versioned CD8 target preparation and observation semantics."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import numpy as np

CD8 = "Density_CD8"
TUMOR = "Density_Tumor"


def normalization_settings(settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate normalization settings; missing settings preserve legacy targets."""
    if settings is not None and not isinstance(settings, Mapping):
        raise ValueError("data.cd8_normalization must be a mapping.")
    settings = dict(settings or {})
    if set(settings) - {"enabled", "min_non_tumor_fraction"}:
        raise ValueError("Unknown data.cd8_normalization setting.")
    enabled = settings.get("enabled", False)
    minimum = settings.get("min_non_tumor_fraction", 0.01)
    if not isinstance(enabled, bool):
        raise ValueError("cd8_normalization.enabled must be boolean.")
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not 0 < minimum <= 1:
        raise ValueError("min_non_tumor_fraction must be finite and in (0, 1].")
    return {"enabled": enabled, "min_non_tumor_fraction": float(minimum)}


def target_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the serializable target transformation contract."""
    columns = list(config["data"]["target_columns"])
    settings = normalization_settings(config["data"].get("cd8_normalization"))
    if len(set(columns)) != len(columns):
        raise ValueError("target_columns must be unique.")
    if settings["enabled"] and not {CD8, TUMOR}.issubset(columns):
        raise ValueError("CD8 normalization requires Density_CD8 and Density_Tumor targets.")
    return {"version": 1, "target_columns": columns, "cd8_normalization": settings}


def checkpoint_contract(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Read checkpoint semantics, treating pre-contract checkpoints as raw CD8."""
    columns = list(checkpoint["target_columns"])
    contract = checkpoint.get("target_transform")
    if contract is None:
        return target_contract({"data": {"target_columns": columns}})
    if contract.get("version") != 1:
        raise ValueError("Unsupported checkpoint target transformation version.")
    resolved = target_contract({"data": contract})
    if resolved["target_columns"] != columns:
        raise ValueError("Checkpoint target order disagrees with target transformation.")
    if resolved != target_contract(checkpoint["config"]):
        raise ValueError("Checkpoint configuration disagrees with target transformation.")
    return resolved


def checkpoint_analysis_config(config: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Use frozen model/preprocessing/target semantics with runtime analysis settings."""
    resolved = copy.deepcopy(dict(config))
    from dann.checkpoints import checkpoint_config
    saved = checkpoint_config(checkpoint)
    resolved["model"] = copy.deepcopy(saved["model"])
    for key in ("matrix_key", "batch_column", "intensity_transform", "intensity_clip_max", "nonzero_threshold", "x_column", "y_column"):
        if key in saved["data"]:
            resolved["data"][key] = saved["data"][key]
    resolved["loss"]["logit_epsilon"] = saved["loss"]["logit_epsilon"]
    contract = checkpoint_contract(checkpoint)
    resolved["data"].update({key: contract[key] for key in ("target_columns", "cd8_normalization")})
    for key in ("core_size", "tiles_per_batch", "batch_size", "validation_batch_size"):
        if key in saved.get("training", {}):
            resolved["training"][key] = saved["training"][key]
    if "results" in resolved:
        from dann.config import resolve_execution
        if "results" in saved:
            resolved["results"] = copy.deepcopy(saved["results"])
        for key in ("pixel", "spatial"):
            if key in saved.get("training", {}):
                resolved["training"][key] = copy.deepcopy(saved["training"][key])
        resolve_execution(resolved)
        resolved["training"]["num_workers"] = int(resolved["analysis"]["num_workers"])
    return resolved


def prepare_targets(raw: np.ndarray, columns: Sequence[str], settings: Mapping[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite training targets, validity, and CD8 exclusion reason per row.

    Reasons are 0 (valid), 1 (insufficient non-tumor area), and 2 (ratio > 1).
    Division uses float64 before conversion to model precision. Threshold
    comparisons use the source precision so a stored threshold boundary is kept.
    """
    source = np.asarray(raw)
    values = np.asarray(source, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError("Target matrix shape does not match target_columns.")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Raw density targets must be finite and inside [0, 1].")
    contract = target_contract({"data": {"target_columns": columns, "cd8_normalization": settings}})
    settings = contract["cd8_normalization"]
    targets = values.copy()
    valid = np.ones(values.shape, dtype=bool)
    reasons = np.zeros(values.shape[0], dtype=np.uint8)
    if settings["enabled"]:
        cd8, tumor = columns.index(CD8), columns.index(TUMOR)
        threshold = 1.0 - settings["min_non_tumor_fraction"]
        if np.issubdtype(source.dtype, np.floating):
            threshold = float(np.asarray(threshold, dtype=source.dtype))
        eligible = (values[:, tumor] <= threshold) & (values[:, tumor] < 1.0)
        reasons[~eligible] = 1
        ratio = np.zeros(values.shape[0], dtype=np.float64)
        ratio[eligible] = values[eligible, cd8] / (1.0 - values[eligible, tumor])
        reasons[eligible & (ratio > 1.0)] = 2
        valid[:, cd8] = reasons == 0
        targets[:, cd8] = np.where(valid[:, cd8], ratio, 0.0)
    return targets.astype(np.float32), valid, reasons


def prepare_observed_targets(
    raw: np.ndarray,
    columns: Sequence[str],
    settings: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare plotting observations, retaining missing tissue annotations.

    Unlike training inputs, tissue-wide observations can be unavailable. NaNs
    are masked per target; normalized CD8 additionally requires observed tumor.
    Infinite or out-of-range observed values still fail normal validation.
    """
    raw = np.asarray(raw)
    missing = np.isnan(raw)
    targets, valid, _ = prepare_targets(np.where(missing, 0.0, raw), columns, settings)
    valid &= ~missing
    if normalization_settings(settings)["enabled"]:
        valid[:, columns.index(CD8)] &= ~missing[:, columns.index(TUMOR)]
    return np.where(valid, targets, np.nan), valid


def observed_targets(extracted: Mapping[str, Any]) -> np.ndarray:
    """Represent unavailable labels as missing for all downstream consumers."""
    targets = np.asarray(extracted["targets"])
    valid = np.asarray(extracted.get("target_valid_mask", np.isfinite(targets)), dtype=bool)
    return np.where(valid, targets, np.nan)


def target_label(column: str, contract: Mapping[str, Any] | None) -> str:
    """Return a human-readable target label without changing machine column names."""
    if column == CD8 and contract and contract["cd8_normalization"]["enabled"]:
        return "Normalized CD8 (non-tumor area)"
    return column

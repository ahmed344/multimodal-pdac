"""Manual cohort decisions and exact retained-to-source observation mappings."""
from __future__ import annotations

import hashlib
from pathlib import Path

from anndata.io import read_elem
import h5py
import numpy as np
import pandas as pd

from .io import axes


def rows_digest(rows: np.ndarray) -> str:
    """Hash ordered original row positions in a platform-independent encoding."""
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def decide_cohort(lesion: pd.DataFrame, tissue: pd.DataFrame, config: dict) -> tuple[dict, dict]:
    """Validate manual exclusions against both sources before any fitting."""
    batch = config["coordinate_columns"]["batch_column"]
    patient = config["patient_column"]
    exclusions = config.get("exclude_slides", {})
    frames = {"lesion": lesion, "tissue": tissue}
    labels = {name: frame[batch].astype(str).to_numpy() for name, frame in frames.items()}
    for name, values in labels.items():
        missing = set(exclusions) - set(values)
        if missing:
            raise ValueError(f"Excluded slide IDs absent from {name}: {sorted(missing)}")
    rows = {name: np.flatnonzero(~np.isin(values, list(exclusions))).astype(np.int64)
            for name, values in labels.items()}
    if any(not len(value) for value in rows.values()):
        raise ValueError("No slides remain after exclusion.")
    retained = sorted(set(labels["lesion"][rows["lesion"]]))
    if set(retained) != set(labels["tissue"][rows["tissue"]]):
        raise ValueError("Retained lesion and tissue slides differ.")
    records = []
    for slide in sorted(set(labels["lesion"]) | set(labels["tissue"])):
        patients = set()
        counts = {}
        for name, frame in frames.items():
            mask = labels[name] == slide
            counts[f"{name}_rows"] = int(mask.sum())
            patients.update(frame.loc[mask, patient].astype(str))
        if len(patients) != 1:
            raise ValueError(f"Lesion/tissue patient identities disagree for {slide}.")
        records.append(dict(slide=slide, patient=patients.pop(), included=slide not in exclusions,
                            reason=exclusions.get(slide, ""), **counts))
    cohort = dict(retained_slides=retained, decisions=records,
                  source_rows={name: dict(count=len(value), sha256=rows_digest(value))
                               for name, value in rows.items()})
    return cohort, rows


def source_rows(manifest: dict, state: dict, kind: str, n: int) -> np.ndarray:
    """Read the frozen row mapping, preserving the schema 1/2 identity contract."""
    rows = state[f"{kind}_source_rows"] if manifest["schema"] >= 3 else np.arange(n, dtype=np.int64)
    if manifest["schema"] >= 3:
        contract = manifest["cohort"]["source_rows"][kind]
        if (len(rows) != contract["count"] or rows_digest(rows) != contract["sha256"]
                or np.any(rows < 0) or np.any(rows >= n) or np.any(np.diff(rows) <= 0)):
            raise ValueError(f"Invalid frozen {kind} source-row mapping.")
    return rows


def aligned_export_rows(path: str | Path, manifest: dict, state: dict, kind: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Verify exported keys against original positions before comparing intensities."""
    config, data = manifest["config"], manifest["coordinate_columns"]
    original, _ = axes(config[f"{kind}_input"], data, config["patient_column"])
    exported, _ = axes(path, data, config["patient_column"])
    rows = source_rows(manifest, state, kind, len(original))
    if not np.array_equal(original.iloc[rows].astype(str).to_numpy(), exported.astype(str).to_numpy()):
        raise ValueError(f"Exported {kind} keys disagree with source-row mapping.")
    with h5py.File(path, "r") as handle:
        if manifest["schema"] >= 3 and not np.array_equal(read_elem(handle["obs/msi_source_row"]), rows):
            raise ValueError(f"Exported {kind} source-row mapping differs from fit.")
    return exported, rows

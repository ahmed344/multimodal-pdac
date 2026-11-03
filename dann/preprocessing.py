"""Validate frozen MSI preprocessing without loading its large parameter arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from anndata.io import read_elem
import h5py
import numpy as np


def validate_preprocessing(path: Path, data: Mapping[str, Any], *, training_rows=None) -> dict | None:
    """Check linear units, fingerprint and (when supplied) the uncapped fitting split."""
    expected = data.get("preprocessing_fingerprint")
    with h5py.File(path, "r") as handle:
        if "uns/msi_preprocess" not in handle:
            if expected:
                raise ValueError("Configured preprocessing fingerprint is missing from the input file.")
            return None
        group = handle["uns/msi_preprocess"]
        actual = str(read_elem(group["fingerprint"]))
        scale = str(read_elem(group["intensity_scale"]))
        manifest = json.loads(str(read_elem(group["manifest_json"])))
    if actual != expected:
        raise ValueError("MSI preprocessing fingerprint mismatch; use matching inputs and fresh training.")
    if scale != "linear" or data["matrix_key"] != "X" or data["intensity_transform"] != "log1p":
        raise ValueError("Preprocessed MSI requires linear X and exactly one DANN log1p transform.")
    if training_rows is not None:
        digest = hashlib.sha256(np.sort(training_rows).astype("<i8").tobytes()).hexdigest()
        if digest != manifest["split"]["training_rows_sha256"]:
            raise ValueError("DANN training split differs from the frozen MSI preprocessing fit; refit preprocessing.")
    return manifest

"""Validate frozen MSI preprocessing without loading its large parameter arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from anndata.io import read_elem
import h5py
import numpy as np
import pandas as pd


def _validate_cohort(handle: h5py.File, manifest: dict) -> None:
    """Validate observed domains and ordered source positions for schema 3."""
    if manifest.get("fitting", {}).get("scope") != "retained_lesion":
        raise ValueError("Unsupported MSI preprocessing fitting scope.")
    cohort = manifest["cohort"]
    kind = str(read_elem(handle["uns/msi_preprocess/dataset_kind"]))
    if kind not in ("lesion", "tissue"):
        raise ValueError("Invalid preprocessing dataset kind.")
    rows = np.asarray(read_elem(handle["obs/msi_source_row"]))
    contract = cohort["source_rows"][kind]
    if (rows.dtype.kind not in "iu" or len(rows) != contract["count"]
            or np.any(rows < 0) or np.any(np.diff(rows) <= 0)
            or hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest() != contract["sha256"]):
        raise ValueError("MSI preprocessing source-row mapping mismatch.")
    data = manifest["coordinate_columns"]
    labels = np.asarray(read_elem(handle["obs"][data["batch_column"]]), dtype=str)
    patients = np.asarray(read_elem(handle["obs"][manifest["config"]["patient_column"]]), dtype=str)
    names, counts = np.unique(labels, return_counts=True)
    decisions = {d["slide"]: d for d in cohort["decisions"] if d["included"]}
    if set(names) != set(cohort["retained_slides"]) or set(names) != set(decisions):
        raise ValueError("Observed slides disagree with the retained preprocessing cohort.")
    if len(labels) != len(rows) or tuple(handle["X"].attrs["shape"])[0] != len(rows):
        raise ValueError("Preprocessing observation count mismatch.")
    for name, count in zip(names, counts):
        if count != decisions[name][f"{kind}_rows"] or set(patients[labels == name]) != {decisions[name]["patient"]}:
            raise ValueError("Observed slide counts/patients disagree with preprocessing cohort.")
    if kind == "lesion":
        keys = pd.DataFrame({data[key]: read_elem(handle["obs"][data[key]])
                             for key in ("batch_column", "x_column", "y_column")})
        key_digest = hashlib.sha256(pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype="<u8").tobytes()).hexdigest()
        if key_digest != manifest["fitting"]["keys_sha256"]:
            raise ValueError("Retained lesion pixel identities differ from the fitting scope.")


def validate_preprocessing(path: Path, data: Mapping[str, Any], *, training_rows=None) -> dict | None:
    """Check linear units, fingerprint and (when supplied) the declared fitting scope."""
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
        if manifest.get("schema") == 3:
            _validate_cohort(handle, manifest)
    if manifest.get("fingerprint", actual) != actual:
        raise ValueError("MSI preprocessing manifest fingerprint mismatch.")
    if actual != expected:
        raise ValueError("MSI preprocessing fingerprint mismatch; use matching inputs and fresh training.")
    if scale != "linear" or data["matrix_key"] != "X" or data["intensity_transform"] != "log1p":
        raise ValueError("Preprocessed MSI requires linear X and exactly one DANN log1p transform.")
    schema = manifest.get("schema", 1)
    if schema not in (1, 2, 3):
        raise ValueError("Unsupported MSI preprocessing schema.")
    if schema == 2 and manifest.get("fitting", {}).get("scope") != "all_lesion":
        raise ValueError("Unsupported MSI preprocessing fitting scope.")
    if training_rows is not None and schema == 1:
        digest = hashlib.sha256(np.sort(training_rows).astype("<i8").tobytes()).hexdigest()
        if digest != manifest["split"]["training_rows_sha256"]:
            raise ValueError("DANN training split differs from the frozen MSI preprocessing fit; refit preprocessing.")
    return manifest


def artifact_preprocessing(path: Path, data: Mapping[str, Any] | None = None) -> dict:
    """Return validated provenance for artifacts; legacy unprocessed inputs stay empty."""
    if data is None:
        with h5py.File(path, "r") as handle:
            if "uns/msi_preprocess" not in handle:
                return {}
            expected = str(read_elem(handle["uns/msi_preprocess/fingerprint"]))
        data = dict(preprocessing_fingerprint=expected, matrix_key="X", intensity_transform="log1p")
    manifest = validate_preprocessing(path, data)
    if manifest is None:
        return {}
    return dict(preprocessing_fingerprint=manifest["fingerprint"], preprocessing_schema=manifest["schema"],
                retained_slides=manifest.get("cohort", {}).get("retained_slides", []))


def parquet_preprocessing(provenance: dict) -> dict[bytes, bytes]:
    """Encode cohort provenance in a Parquet schema."""
    return {b"dann_preprocessing": json.dumps(provenance, sort_keys=True).encode()} if provenance else {}


def validate_artifact_preprocessing(path: Path, metadata: Mapping | None) -> dict:
    """Reject stale prediction/latent artifacts even when names and widths match."""
    expected = artifact_preprocessing(path)
    actual = dict(metadata or {})
    if b"dann_preprocessing" in actual:
        actual = json.loads(actual[b"dann_preprocessing"])
    if max(expected.get("preprocessing_schema", 0), actual.get("preprocessing_schema", 0)) >= 3:
        if actual != expected:
            raise ValueError("Artifact preprocessing/cohort mismatch; regenerate from matching inputs.")
    return expected


ANALYSIS_TABLES = ("latent_embeddings.csv", "umap/latent_umap.csv", "ziln_density_scatter.csv")


def analysis_provenance(root: Path, data: Mapping[str, Any], *, write: bool = False) -> dict:
    """Bind saved CSV consumers to preprocessing and the exact exported tables."""
    from dann.artifact_layout import analysis_file
    provenance = artifact_preprocessing(Path(data["path"]), data)
    path = root / "preprocessing.json"
    if provenance.get("preprocessing_schema", 0) < 3:
        if write:
            path.unlink(missing_ok=True)
        elif path.exists() and json.loads(path.read_text()).get("preprocessing", {}).get("preprocessing_schema", 0) >= 3:
            raise ValueError("Saved analysis preprocessing/cohort mismatch with legacy input.")
        return provenance
    hashes = {}
    for name in ANALYSIS_TABLES:
        value = hashlib.sha256()
        table = root / name if write and (root / name).exists() else analysis_file(root, name)
        with table.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        hashes[name] = value.hexdigest()
    expected = dict(preprocessing=provenance, tables=hashes)
    if write:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(expected, indent=2))
        temporary.replace(path)
    elif not path.exists() or json.loads(path.read_text()) != expected:
        raise ValueError("Saved analysis preprocessing/cohort or table checksum mismatch.")
    return provenance

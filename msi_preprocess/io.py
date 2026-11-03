"""Bounded sparse H5AD reading/writing and immutable input identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from anndata.io import read_elem, write_elem
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .qc import read_csr_range


def identity(path: str | Path) -> dict:
    """Identify an immutable source without rereading tens of gigabytes to hash it."""
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def json_write(path: Path, value: Any) -> None:
    """Write strict, human-readable JSON atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value: Any) -> str:
    """Hash canonical settings or metadata."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def axes(path: str | Path, data: dict, patient_column: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Read only coordinate/identity metadata, never IHC or H&E labels."""
    names = [data["batch_column"], data["x_column"], data["y_column"], patient_column]
    with h5py.File(path, "r") as h:
        obs = pd.DataFrame({key: read_elem(h["obs"][key]) for key in names})
        features = np.asarray(read_elem(h["var"][h["var"].attrs["_index"]]))
        if pd.isna(features).any():
            raise ValueError("Missing feature identities.")
        features = features.astype(str)
        group = h["layers/counts"]
        if group.attrs.get("encoding-type") != "csr_matrix":
            raise ValueError("layers/counts must be CSR encoded.")
        if tuple(group.attrs["shape"]) != (len(obs), len(features)):
            raise ValueError("Counts shape disagrees with metadata.")
        if (len(group["indptr"]) != len(obs) + 1 or group["indptr"][0] != 0
                or group["indptr"][-1] != len(group["data"])
                or len(group["indices"]) != len(group["data"])):
            raise ValueError("Invalid CSR dimensions.")
    if obs.isna().any().any() or len(np.unique(features)) != len(features):
        raise ValueError("Missing metadata or duplicate feature names.")
    xy = obs[[data["x_column"], data["y_column"]]].to_numpy(dtype=float)
    if not np.isfinite(xy).all() or not np.equal(xy, np.round(xy)).all():
        raise ValueError("Grid coordinates must be finite integers.")
    if obs.duplicated(names[:3]).any():
        raise ValueError("Duplicate (batch, x, y) keys.")
    if (obs.groupby(names[0], observed=True)[patient_column].nunique() != 1).any():
        raise ValueError("Each slide must belong to exactly one patient.")
    return obs, features


def match_rows(lesion: pd.DataFrame, tissue: pd.DataFrame, data: dict) -> np.ndarray:
    """Map lesion to tissue by unique (batch,x,y), preserving lesion order."""
    keys = [data[k] for k in ("batch_column", "x_column", "y_column")]
    result = pd.MultiIndex.from_frame(tissue[keys]).get_indexer(pd.MultiIndex.from_frame(lesion[keys]))
    if np.any(result < 0):
        raise ValueError("Every lesion row must match a whole-tissue (batch, x, y) key.")
    return result


def read_rows(group: h5py.Group, rows: np.ndarray, block_rows: int) -> sparse.csr_matrix:
    """Read arbitrary rows in caller order using coalesced bounded spans."""
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return sparse.csr_matrix((0, int(group.attrs["shape"][1])), dtype=float)
    order = np.argsort(rows)
    sorted_rows = rows[order]
    parts = []
    # Merge nearby requests to avoid one HDF5 call per randomly withheld row.
    start = 0
    while start < len(rows):
        end = int(np.searchsorted(sorted_rows, sorted_rows[start] + block_rows))
        lo, hi = int(sorted_rows[start]), int(sorted_rows[end - 1]) + 1
        parts.append(read_csr_range(group, lo, hi)[sorted_rows[start:end] - lo])
        start = end
    result = sparse.vstack(parts, format="csr")
    return result[np.argsort(order)]


def check_memory(group: h5py.Group, rows: np.ndarray, config: dict) -> None:
    """Conservatively bound slide CSR copies, feature blocks and graph scratch."""
    ptr = group["indptr"][:]
    nnz = int(np.sum(ptr[rows + 1] - ptr[rows]))
    workers = config.get("application_workers", 4)
    required = nnz * 64 + len(rows) * (workers * config["feature_block"] * 8 * 16 + 600)
    if required > config["memory_gib"] * 2**30:
        raise MemoryError(f"Slide needs approximately {required / 2**30:.2f} GiB; raise memory_gib explicitly.")


def check_disk(paths: list[str | Path], output: Path, reserve_gib: float) -> dict:
    """Check an uncompressed upper bound for two retained CSR matrices + metadata.

    CSR feature indices use int32 and row pointers int64, even when the source
    uses int64 feature indices. No compression savings are assumed.
    """
    required = 0
    for path in paths:
        with h5py.File(path, "r") as h:
            matrix = h["layers/counts"]
            n = int(matrix.attrs["shape"][0])
            nnz = len(matrix["data"])
            excluded = sum(h[key].id.get_storage_size() for root in ("X", "layers/counts")
                           for key in (f"{root}/data", f"{root}/indices", f"{root}/indptr"))
            required += max(0, Path(path).stat().st_size - excluded)
            required += 2 * (nnz * (max(4, matrix["data"].dtype.itemsize) + 4) + (n + 1) * 8)
    required = int(required * 1.03) + int(reserve_gib * 2**30)
    free = shutil.disk_usage(output).free
    if free < required:
        raise OSError(f"Insufficient disk: require {required / 2**30:.1f} GiB, available {free / 2**30:.1f} GiB.")
    return dict(required_bytes=required, available_bytes=free)


class CSRWriter:
    """Append CSR row blocks without retaining a full matrix in memory."""

    def __init__(self, parent: h5py.Group, key: str, shape: tuple, dtype: np.dtype):
        self.group = parent.create_group(key)
        self.group.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": shape})
        self.data = self.group.create_dataset("data", shape=(0,), maxshape=(None,), dtype=dtype, chunks=(262144,))
        self.indices = self.group.create_dataset("indices", shape=(0,), maxshape=(None,), dtype="int32", chunks=(262144,))
        self.ptr = self.group.create_dataset("indptr", shape=(shape[0] + 1,), dtype="int64")
        self.ptr[0] = 0
        self.rows = self.nnz = 0

    def append(self, matrix: sparse.csr_matrix) -> None:
        """Append a canonical block in the final observation order."""
        end = int(self.nnz) + int(matrix.nnz)
        self.data.resize(end, axis=0)
        self.indices.resize(end, axis=0)
        self.data[self.nnz:end] = matrix.data
        self.indices[self.nnz:end] = matrix.indices
        # A small block normally has int32 indptr, but its global offset can
        # exceed 2**31. Promote BEFORE addition (NumPy 2 rejects that overflow).
        self.ptr[self.rows + 1:self.rows + matrix.shape[0] + 1] = matrix.indptr[1:].astype(np.int64) + self.nnz
        self.rows += matrix.shape[0]
        self.nnz = end


def copy_metadata(source: h5py.File, destination: h5py.File, retained: np.ndarray,
                  provenance: dict) -> None:
    """Copy observation metadata unchanged and subset feature metadata in order."""
    destination.attrs.update(source.attrs)
    for key in source:
        if key not in {"X", "layers", "var", "varm", "varp", "raw"}:
            source.copy(key, destination)
    if "raw" in source or any(key != "counts" for key in source["layers"]):
        raise ValueError("Unexpected raw/extra layers: define their feature-subsetting semantics before export.")
    var = read_elem(source["var"]).iloc[retained].copy()
    var["msi_original_feature_index"] = retained
    write_elem(destination, "var", var)
    for key in ("varm", "varp"):
        items = read_elem(source[key]) if key in source else {}
        subset = {}
        for name, value in items.items():
            if isinstance(value, pd.DataFrame):
                subset[name] = value.iloc[retained].iloc[:, retained] if key == "varp" else value.iloc[retained]
            else:
                subset[name] = value[retained][:, retained] if key == "varp" else value[retained]
        write_elem(destination, key, subset)
    if "uns" not in destination:
        write_elem(destination, "uns", {})
    if "msi_preprocess" in destination["uns"]:
        del destination["uns/msi_preprocess"]
    write_elem(destination["uns"], "msi_preprocess", provenance)
    write_elem(destination, "layers", {})

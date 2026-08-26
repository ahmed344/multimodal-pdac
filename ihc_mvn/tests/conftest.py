"""Shared tiny synthetic AnnData fixtures for standalone IHC-MVN tests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ihc_mvn.targets import TARGET_COLUMNS


def write_csr_anndata(
    path: Path,
    counts: np.ndarray,
    batches: Sequence[str],
    patients: Sequence[str],
    *,
    feature_names: Sequence[str] | None = None,
    densities: np.ndarray | None = None,
) -> Path:
    """Write a minimal CSR AnnData file with the production metadata schema.

    Args:
        path (Path): Destination ``.h5ad`` path.
        counts (np.ndarray): Nonnegative count matrix with shape ``[N, P]``.
        batches (Sequence[str]): Slide name for every observation.
        patients (Sequence[str]): Patient name for every observation.
        feature_names (Sequence[str] | None): Optional ordered feature names.
        densities (np.ndarray | None): Optional four-column ordered densities.

    Returns:
        Path: The written AnnData path.
    """

    values = np.asarray(counts, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("counts must be a two-dimensional array.")
    if len(batches) != values.shape[0] or len(patients) != values.shape[0]:
        raise ValueError("Batch and patient metadata must match count rows.")
    names = (
        [f"peak_{index}" for index in range(values.shape[1])]
        if feature_names is None
        else [str(name) for name in feature_names]
    )
    if len(names) != values.shape[1]:
        raise ValueError("Feature names must match count columns.")

    obs = pd.DataFrame(
        {
            "batch": pd.Categorical([str(value) for value in batches]),
            "patient": pd.Categorical([str(value) for value in patients]),
        },
        index=[f"pixel_{index}" for index in range(values.shape[0])],
    )
    if densities is not None:
        target_values = np.asarray(densities, dtype=np.float32)
        if target_values.shape != (values.shape[0], len(TARGET_COLUMNS)):
            raise ValueError("densities must have shape [N, 4].")
        for target_index, target_name in enumerate(TARGET_COLUMNS):
            obs[target_name] = target_values[:, target_index]

    matrix = sparse.csr_matrix(values)
    adata = ad.AnnData(
        X=sparse.csr_matrix(values.shape, dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(names, name="feature")),
    )
    adata.layers["counts"] = matrix
    adata.write_h5ad(path)
    return path


@pytest.fixture
def tiny_h5ad(tmp_path: Path) -> Path:
    """Create a labeled two-slide CSR AnnData file.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        Path: Path to the synthetic ``.h5ad`` file.
    """

    row_count = 20
    counts = np.zeros((row_count, 3), dtype=np.float32)
    counts[:, 0] = np.arange(row_count) % 4
    counts[::2, 1] = 2.0
    counts[1::3, 2] = 5.0
    batches = ["slide_a"] * 10 + ["slide_b"] * 10
    patients = ["patient_a"] * 10 + ["patient_b"] * 10
    base = np.linspace(0.0, 0.8, row_count, dtype=np.float32)
    densities = np.column_stack(
        (
            base,
            np.roll(base, 1),
            np.roll(base, 2),
            np.roll(base, 3),
        )
    )
    densities[0, :] = 0.01
    densities[10, :] = 0.02
    return write_csr_anndata(
        tmp_path / "tiny.h5ad",
        counts,
        batches,
        patients,
        densities=densities,
    )

"""Streaming linear-value verification and sampled lesion/DANN-loader checks."""
from __future__ import annotations

import argparse
from pathlib import Path

from anndata.io import read_elem
import h5py
import numpy as np
import yaml

from .fit import load_fit
from .io import axes, json_write, match_rows, read_rows
from .qc import read_csr_range


def verify_linear_values(handle: h5py.File, batch: np.ndarray, state: dict, config: dict) -> int:
    """Stream all tissue nonzeros, checking exact scaling and allowed removals."""
    slide_index = {str(slide): i for i, slide in enumerate(state["slides"])}
    p = int(handle["X"].attrs["shape"][1])
    checked = 0
    for start in range(0, len(batch), config["block_rows"]):
        stop = min(start + config["block_rows"], len(batch))
        counts = read_csr_range(handle["layers/counts"], start, stop)
        actual = read_csr_range(handle["X"], start, stop)
        factors = np.array([state["multipliers"][slide_index[str(slide)]] for slide in batch[start:stop]])
        expected = counts.data / np.repeat(factors, np.diff(counts.indptr))
        count_keys = np.repeat(np.arange(stop-start, dtype=np.int64), np.diff(counts.indptr)) * p + counts.indices
        actual_keys = np.repeat(np.arange(stop-start, dtype=np.int64), np.diff(actual.indptr)) * p + actual.indices
        positions = np.searchsorted(count_keys, actual_keys)
        if np.any(positions >= len(count_keys)) or not np.array_equal(count_keys[positions], actual_keys):
            raise ValueError("Preprocessing introduced nonzero measurements.")
        np.testing.assert_array_equal(actual.data, expected[positions].astype(handle["X/data"].dtype).astype(float))
        removed = np.ones(counts.nnz, dtype=bool)
        removed[positions] = False
        if (not config["stages"]["local"] and removed.any()) or np.any(expected[removed] > config["local"]["weak_boundary"]):
            raise ValueError("Export removed a measurement outside the allowed weak boundary.")
        checked += actual.nnz
    return checked


def verify(fitted: str | Path, rows_per_slide: int = 128) -> dict:
    """Verify all tissue linear values and reproducible lesion/tissue samples.

    All matched original counts were checked during application. This separate
    read-back check reports full linear-value coverage separately from sampled
    lesion/tissue agreement, rather than claiming a second exhaustive row match.
    """
    if rows_per_slide < 1:
        raise ValueError("rows_per_slide must be positive.")
    manifest, state = load_fit(fitted)
    output = Path(fitted).resolve().parent
    config = manifest["config"]
    data = manifest["coordinate_columns"]
    tissue_obs, tissue_features = axes(output / "tissue.h5ad", data, config["patient_column"])
    lesion_obs, lesion_features = axes(output / "lesion.h5ad", data, config["patient_column"])
    expected_features = state["features"][state["retained"]]
    np.testing.assert_array_equal(tissue_features, expected_features)
    np.testing.assert_array_equal(lesion_features, expected_features)
    mapping = match_rows(lesion_obs, tissue_obs, data)
    batch = tissue_obs[data["batch_column"]].astype(str).to_numpy()
    lesion_batch = lesion_obs[data["batch_column"]].astype(str).to_numpy()
    lesion_samples = []
    for i, slide in enumerate(state["slides"]):
        rng = np.random.default_rng(config["seed"] + i)
        rows = np.flatnonzero(lesion_batch == slide)
        lesion_samples.extend(rng.choice(rows, min(len(rows), rows_per_slide), replace=False))
    lesion_samples = np.sort(lesion_samples)
    with h5py.File(output / "tissue.h5ad", "r") as tissue, h5py.File(output / "lesion.h5ad", "r") as lesion:
        for handle, n in ((tissue, len(tissue_obs)), (lesion, len(lesion_obs))):
            for key in ("X", "layers/counts"):
                group = handle[key]
                if (group.attrs["encoding-type"] != "csr_matrix"
                        or tuple(group.attrs["shape"]) != (n, len(expected_features))
                        or np.any(np.diff(group["indptr"][:]) < 0)
                        or group["indptr"][-1] != len(group["data"])):
                    raise ValueError("Invalid exported CSR structure.")
            if str(read_elem(handle["uns/msi_preprocess/fingerprint"])) != manifest["fingerprint"]:
                raise ValueError("Export preprocessing fingerprint differs from fitted state.")
            if str(read_elem(handle["uns/msi_preprocess/intensity_scale"])) != "linear":
                raise ValueError("Exported values must remain linear.")
        linear_values = verify_linear_values(tissue, batch, state, config)
        for start in range(0, len(lesion_samples), rows_per_slide):
            rows = lesion_samples[start:start+rows_per_slide]
            for key in ("X", "layers/counts"):
                left = read_rows(lesion[key], rows, config["block_rows"])
                right = read_rows(tissue[key], mapping[rows], config["block_rows"])
                if (left != right).nnz:
                    raise ValueError("Written lesion and tissue values disagree.")
    # Use the actual DANN sparse loader without building the resident full-data cache.
    from dann.data_loader import SparseAnnDataDataset, load_anndata_metadata
    from dann.preprocessing import validate_preprocessing
    dann = yaml.safe_load(Path(config["dann_config"]).read_text())
    settings = dict(dann["data"], path=str(output / "lesion.h5ad"), matrix_key="X",
                    intensity_transform="log1p", preprocessing_fingerprint=manifest["fingerprint"])
    validate_preprocessing(output / "lesion.h5ad", settings, training_rows=state["training_rows"])
    metadata = load_anndata_metadata(output / "lesion.h5ad", settings["target_columns"], settings["batch_column"],
                                    "X", settings.get("cd8_normalization"))
    dataset = SparseAnnDataDataset(output / "lesion.h5ad", lesion_samples, metadata,
                                   matrix_key="X", intensity_transform="log1p")
    try:
        with h5py.File(output / "lesion.h5ad", "r") as handle:
            for k in range(min(32, len(lesion_samples))):
                row = read_rows(handle["X"], lesion_samples[k:k+1], config["block_rows"])
                sample = dataset[k]
                np.testing.assert_array_equal(sample["peak_indices"], row.indices)
                np.testing.assert_array_equal(sample["intensities"], np.log1p(row.data.astype(np.float32)))
    finally:
        dataset.close()
    result = dict(fingerprint=manifest["fingerprint"], schema_checked="all_rows",
                  linear_value_rows_checked=len(tissue_obs), linear_positive_values_checked=linear_values,
                  lesion_value_rows_checked=len(lesion_samples),
                  all_linear_values=True, all_sparse_zeros=True, sampled_lesion_tissue_agreement=True,
                  dann_log1p_exactly_once=True)
    json_write(output / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", default="data/PDAC/Processed/msi_preprocess/fit.json")
    parser.add_argument("--rows-per-slide", type=int, default=128)
    args = parser.parse_args()
    print(verify(args.fit, args.rows_per_slide))


if __name__ == "__main__":
    main()

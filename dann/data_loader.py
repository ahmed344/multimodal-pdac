"""Sparse, backed AnnData loading for adversarial latent fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class AnnDataMetadata:
    """In-memory metadata accompanying a backed sparse MSI matrix."""

    targets: np.ndarray
    batch_codes: np.ndarray
    batch_names: tuple[str, ...]
    mz_values: np.ndarray
    num_observations: int
    num_peaks: int


@dataclass(frozen=True)
class DataBundle:
    """Datasets, loaders, and shared metadata for all data splits."""

    metadata: AnnDataMetadata
    split_indices: dict[str, np.ndarray]
    datasets: dict[str, "SparseAnnDataDataset"]
    loaders: dict[str, DataLoader[dict[str, torch.Tensor]]]
    batch_class_weights: torch.Tensor


def _matrix_group_path(matrix_key: str) -> str:
    """Translate a user-facing AnnData matrix key to an HDF5 group path.

    Args:
        matrix_key (str): ``"X"``, a layer name, or ``"layers/<name>"``.

    Returns:
        str: HDF5 group path storing a CSR matrix.
    """

    if matrix_key == "X":
        return "X"
    if matrix_key.startswith("layers/"):
        return matrix_key
    return f"layers/{matrix_key}"


def load_anndata_metadata(
    path: Path,
    target_columns: Sequence[str],
    batch_column: str,
    matrix_key: str = "X",
) -> AnnDataMetadata:
    """Read labels and feature metadata while leaving MSI values on disk.

    Args:
        path (Path): Input ``.h5ad`` file.
        target_columns (Sequence[str]): Ordered density observation columns.
        batch_column (str): Categorical batch observation column.
        matrix_key (str): Sparse matrix source key.

    Returns:
        AnnDataMetadata: Validated labels, batches, m/z values, and dimensions.
    """

    adata = ad.read_h5ad(path, backed="r")
    required = [*target_columns, batch_column]
    missing = [column for column in required if column not in adata.obs]
    if missing:
        adata.file.close()
        raise KeyError(f"Missing required observation columns: {missing}")

    targets = adata.obs[list(target_columns)].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(targets).all():
        adata.file.close()
        raise ValueError("Density targets contain non-finite values.")
    if np.any((targets < 0.0) | (targets > 1.0)):
        adata.file.close()
        raise ValueError("Density targets must lie inside [0, 1].")

    batch_series = adata.obs[batch_column].astype("category")
    if batch_series.isna().any():
        adata.file.close()
        raise ValueError(f"Batch column {batch_column!r} contains missing values.")
    batch_codes = batch_series.cat.codes.to_numpy(dtype=np.int64, copy=True)
    batch_names = tuple(str(value) for value in batch_series.cat.categories)
    try:
        mz_values = np.asarray(adata.var_names, dtype=np.float64)
    except ValueError as error:
        adata.file.close()
        raise ValueError("AnnData var_names must contain numeric m/z values.") from error

    group_path = _matrix_group_path(matrix_key)
    with h5py.File(path, "r") as handle:
        if group_path not in handle:
            adata.file.close()
            raise KeyError(f"Sparse matrix group {group_path!r} is absent from {path}.")
        group = handle[group_path]
        encoding = group.attrs.get("encoding-type", "")
        if encoding != "csr_matrix":
            adata.file.close()
            raise TypeError(
                f"{group_path!r} must be CSR encoded, observed {encoding!r}."
            )
        shape = tuple(int(value) for value in group.attrs["shape"])
    adata.file.close()
    if shape != (targets.shape[0], mz_values.size):
        raise ValueError(
            f"Matrix shape {shape} disagrees with metadata "
            f"{(targets.shape[0], mz_values.size)}."
        )
    return AnnDataMetadata(
        targets=targets,
        batch_codes=batch_codes,
        batch_names=batch_names,
        mz_values=mz_values,
        num_observations=shape[0],
        num_peaks=shape[1],
    )


def stratified_split_indices(
    batch_codes: np.ndarray,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Create deterministic within-batch train, validation, and test splits.

    Args:
        batch_codes (np.ndarray): Integer batch label for every observation.
        train_fraction (float): Training fraction.
        validation_fraction (float): Validation fraction.
        test_fraction (float): Test fraction.
        seed (int): Random seed.

    Returns:
        dict[str, np.ndarray]: Global row indices keyed by split name.
    """

    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction], dtype=np.float64
    )
    if np.any(fractions < 0.0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError(f"Split fractions must be nonnegative and sum to one: {fractions}.")
    rng = np.random.default_rng(seed)
    split_parts: dict[str, list[np.ndarray]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for batch in np.unique(batch_codes):
        batch_indices = np.flatnonzero(batch_codes == batch)
        rng.shuffle(batch_indices)
        n_batch = batch_indices.size
        n_train = int(np.floor(n_batch * train_fraction))
        n_validation = int(np.floor(n_batch * validation_fraction))
        boundaries = (n_train, n_train + n_validation)
        split_parts["train"].append(batch_indices[: boundaries[0]])
        split_parts["validation"].append(batch_indices[boundaries[0] : boundaries[1]])
        split_parts["test"].append(batch_indices[boundaries[1] :])

    splits: dict[str, np.ndarray] = {}
    for name, parts in split_parts.items():
        combined = np.concatenate(parts).astype(np.int64, copy=False)
        rng.shuffle(combined)
        splits[name] = combined
    return splits


def stratified_cap_indices(
    indices: np.ndarray,
    batch_codes: np.ndarray,
    maximum: int | None,
    seed: int,
) -> np.ndarray:
    """Cap a split while approximately retaining its batch composition.

    Args:
        indices (np.ndarray): Candidate global observation indices.
        batch_codes (np.ndarray): Batch labels for the complete dataset.
        maximum (int | None): Maximum retained rows, or ``None`` for all rows.
        seed (int): Random seed.

    Returns:
        np.ndarray: Possibly subsampled and shuffled global indices.
    """

    if maximum is None or maximum <= 0 or indices.size <= maximum:
        return indices.copy()
    rng = np.random.default_rng(seed)
    labels = batch_codes[indices]
    unique, counts = np.unique(labels, return_counts=True)
    quotas = np.floor(counts / counts.sum() * maximum).astype(int)
    quotas = np.minimum(quotas, counts)
    quotas[(quotas == 0) & (counts > 0)] = 1
    while quotas.sum() > maximum:
        reducible = np.flatnonzero(quotas > 1)
        quotas[reducible[np.argmax(quotas[reducible])]] -= 1
    remainders = counts / counts.sum() * maximum - quotas
    while quotas.sum() < maximum:
        candidates = np.flatnonzero(quotas < counts)
        selected = candidates[np.argmax(remainders[candidates])]
        quotas[selected] += 1
        remainders[selected] = -np.inf
    retained = []
    for batch, quota in zip(unique, quotas, strict=True):
        batch_indices = indices[labels == batch]
        retained.append(rng.choice(batch_indices, int(quota), replace=False))
    result = np.concatenate(retained).astype(np.int64, copy=False)
    rng.shuffle(result)
    return result


class SparseAnnDataDataset(Dataset[dict[str, Any]]):
    """Map-style dataset that reads individual CSR rows from HDF5 on demand."""

    def __init__(
        self,
        path: Path,
        indices: np.ndarray,
        metadata: AnnDataMetadata,
        matrix_key: str = "X",
        intensity_transform: str = "none",
        intensity_clip_max: float | None = None,
        nonzero_threshold: float = 0.0,
    ) -> None:
        """Initialize a lazy backed sparse dataset.

        Args:
            path (Path): Input AnnData file.
            indices (np.ndarray): Global row indices exposed by this dataset.
            metadata (AnnDataMetadata): Shared in-memory metadata.
            matrix_key (str): Sparse AnnData matrix source.
            intensity_transform (str): ``none``, ``log1p``, or ``sqrt``.
            intensity_clip_max (float | None): Optional post-transform upper clip.
            nonzero_threshold (float): Exclude values at or below this level.

        Returns:
            None: Dataset state is initialized.
        """

        if intensity_transform not in {"none", "log1p", "sqrt"}:
            raise ValueError(f"Unsupported intensity transform: {intensity_transform}")
        self.path = Path(path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.metadata = metadata
        self.group_path = _matrix_group_path(matrix_key)
        self.intensity_transform = intensity_transform
        self.intensity_clip_max = intensity_clip_max
        self.nonzero_threshold = float(nonzero_threshold)
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        """Return the number of selected observations.

        Args:
            None.

        Returns:
            int: Dataset length.
        """

        return int(self.indices.size)

    def _ensure_handle(self) -> h5py.File:
        """Open a process-local read-only HDF5 handle when needed.

        Args:
            None.

        Returns:
            h5py.File: Open HDF5 file handle.
        """

        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def _transform_values(self, values: np.ndarray) -> np.ndarray:
        """Apply configured transformations to nonzero peak intensities.

        Args:
            values (np.ndarray): Raw sparse row values.

        Returns:
            np.ndarray: Transformed float32 values.
        """

        values = values.astype(np.float32, copy=False)
        if self.intensity_transform == "log1p":
            values = np.log1p(values)
        elif self.intensity_transform == "sqrt":
            values = np.sqrt(values)
        if self.intensity_clip_max is not None:
            values = np.minimum(values, float(self.intensity_clip_max))
        return values.astype(np.float32, copy=False)

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Read one sparse MSI pixel and its labels.

        Args:
            item (int): Local dataset index.

        Returns:
            dict[str, Any]: Peak indices, intensities, targets, batch, and row ID.
        """

        row = int(self.indices[item])
        group = self._ensure_handle()[self.group_path]
        start, stop = np.asarray(group["indptr"][row : row + 2], dtype=np.int64)
        peak_indices = np.asarray(group["indices"][start:stop], dtype=np.int64)
        intensities = np.asarray(group["data"][start:stop], dtype=np.float32)
        if self.nonzero_threshold > 0.0:
            keep = intensities > self.nonzero_threshold
            peak_indices = peak_indices[keep]
            intensities = intensities[keep]
        intensities = self._transform_values(intensities)
        if not np.isfinite(intensities).all() or np.any(intensities < 0.0):
            raise ValueError(f"Invalid MSI intensities encountered in row {row}.")
        return {
            "peak_indices": peak_indices,
            "intensities": intensities,
            "targets": self.metadata.targets[row],
            "batch": int(self.metadata.batch_codes[row]),
            "row_id": row,
        }

    def __getstate__(self) -> dict[str, Any]:
        """Prepare pickle state without sharing an HDF5 handle across workers.

        Args:
            None.

        Returns:
            dict[str, Any]: Serializable instance state.
        """

        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self) -> None:
        """Close this process's HDF5 file handle.

        Args:
            None.

        Returns:
            None: The file handle is closed if it was open.
        """

        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        """Release the process-local file handle during garbage collection.

        Args:
            None.

        Returns:
            None: Resources are released best-effort.
        """

        self.close()


def sparse_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate variable-length sparse rows without dense feature expansion.

    Args:
        samples (Sequence[Mapping[str, Any]]): Sparse row dictionaries.

    Returns:
        dict[str, torch.Tensor]: Concatenated peaks with sample membership and labels.
    """

    if not samples:
        raise ValueError("Cannot collate an empty sample sequence.")
    lengths = np.asarray([len(sample["peak_indices"]) for sample in samples], dtype=np.int64)
    peak_indices = np.concatenate(
        [np.asarray(sample["peak_indices"], dtype=np.int64) for sample in samples]
    )
    intensities = np.concatenate(
        [np.asarray(sample["intensities"], dtype=np.float32) for sample in samples]
    )
    sample_indices = np.repeat(np.arange(len(samples), dtype=np.int64), lengths)
    targets = np.stack([sample["targets"] for sample in samples]).astype(
        np.float32, copy=False
    )
    batches = np.asarray([sample["batch"] for sample in samples], dtype=np.int64)
    row_ids = np.asarray([sample["row_id"] for sample in samples], dtype=np.int64)
    return {
        "peak_indices": torch.from_numpy(peak_indices),
        "intensities": torch.from_numpy(intensities),
        "sample_indices": torch.from_numpy(sample_indices),
        "peak_counts": torch.from_numpy(lengths),
        "targets": torch.from_numpy(targets),
        "batches": torch.from_numpy(batches),
        "row_ids": torch.from_numpy(row_ids),
    }


def compute_batch_class_weights(
    batch_codes: np.ndarray, train_indices: np.ndarray
) -> torch.Tensor:
    """Compute mean-one inverse-frequency discriminator weights.

    Args:
        batch_codes (np.ndarray): Batch code for every observation.
        train_indices (np.ndarray): Selected training row indices.

    Returns:
        torch.Tensor: Float class weight for each encoded batch.
    """

    num_classes = int(batch_codes.max()) + 1
    counts = np.bincount(batch_codes[train_indices], minlength=num_classes).astype(float)
    if np.any(counts == 0):
        raise ValueError("Every batch class must occur in the training split.")
    weights = counts.sum() / (num_classes * counts)
    weights /= weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def create_data_bundle(config: Mapping[str, Any]) -> DataBundle:
    """Build metadata, stratified datasets, and DataLoaders from configuration.

    Args:
        config (Mapping[str, Any]): Complete DANN configuration.

    Returns:
        DataBundle: All sparse data objects needed by training and analysis.
    """

    data_config = config["data"]
    training_config = config["training"]
    path = Path(data_config["path"])
    metadata = load_anndata_metadata(
        path,
        data_config["target_columns"],
        data_config["batch_column"],
        data_config["matrix_key"],
    )
    if int(config["model"]["num_peaks"]) != metadata.num_peaks:
        raise ValueError(
            f"Configured num_peaks={config['model']['num_peaks']} but data has "
            f"{metadata.num_peaks}."
        )
    seed = int(training_config["seed"])
    split_indices = stratified_split_indices(
        metadata.batch_codes,
        float(data_config["train_fraction"]),
        float(data_config["validation_fraction"]),
        float(data_config["test_fraction"]),
        seed,
    )
    cap_keys = {
        "train": "max_train_samples",
        "validation": "max_validation_samples",
        "test": "max_test_samples",
    }
    for offset, (name, cap_key) in enumerate(cap_keys.items()):
        split_indices[name] = stratified_cap_indices(
            split_indices[name],
            metadata.batch_codes,
            data_config.get(cap_key),
            seed + offset + 1,
        )

    dataset_kwargs = {
        "path": path,
        "metadata": metadata,
        "matrix_key": data_config["matrix_key"],
        "intensity_transform": data_config["intensity_transform"],
        "intensity_clip_max": data_config.get("intensity_clip_max"),
        "nonzero_threshold": data_config["nonzero_threshold"],
    }
    datasets = {
        name: SparseAnnDataDataset(indices=indices, **dataset_kwargs)
        for name, indices in split_indices.items()
    }
    num_workers = int(training_config["num_workers"])
    common_loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "collate_fn": sparse_collate,
        "pin_memory": bool(training_config["pin_memory"]),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = int(training_config["prefetch_factor"])
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=int(training_config["batch_size"]),
            shuffle=True,
            generator=generator,
            **common_loader_kwargs,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=int(training_config["validation_batch_size"]),
            shuffle=False,
            **common_loader_kwargs,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=int(config["analysis"]["batch_size"]),
            shuffle=False,
            **common_loader_kwargs,
        ),
    }
    return DataBundle(
        metadata=metadata,
        split_indices=split_indices,
        datasets=datasets,
        loaders=loaders,
        batch_class_weights=compute_batch_class_weights(
            metadata.batch_codes, split_indices["train"]
        ),
    )

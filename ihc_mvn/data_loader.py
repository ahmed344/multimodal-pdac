"""Standalone backed-CSR metadata, splitting, datasets, and data bundles."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .scaling import SparseFeatureScaler, fit_sparse_feature_scaler, matrix_group_path
from .targets import (
    TARGET_COLUMNS,
    TargetArrays,
    TargetStandardizer,
    build_target_arrays,
)


@dataclass(frozen=True)
class CategoryMapping:
    """Stable names and integer codes for one observation category."""

    column: str
    names: tuple[str, ...]
    name_to_code: dict[str, int]

    @classmethod
    def fit(cls, values: pd.Series, column: str) -> "CategoryMapping":
        """Fit a stable category mapping from an observation series.

        Args:
            values (pd.Series): Observation values to encode.
            column (str): Source observation column name.

        Returns:
            CategoryMapping: Fitted category names and integer mapping.
        """

        if values.isna().any():
            raise ValueError(f"Observation column {column!r} contains missing values.")
        categorical = values.astype("category")
        names = tuple(str(value) for value in categorical.cat.categories)
        if len(set(names)) != len(names):
            raise ValueError(
                f"String conversion creates duplicate categories in column {column!r}."
            )
        return cls(
            column=column,
            names=names,
            name_to_code={name: index for index, name in enumerate(names)},
        )

    @classmethod
    def from_names(cls, column: str, names: Sequence[str]) -> "CategoryMapping":
        """Construct a category mapping from stored ordered names.

        Args:
            column (str): Source observation column name.
            names (Sequence[str]): Ordered category names.

        Returns:
            CategoryMapping: Mapping assigning codes by sequence position.
        """

        normalized = tuple(str(value) for value in names)
        if len(set(normalized)) != len(normalized):
            raise ValueError("Category names must be unique.")
        return cls(
            column=column,
            names=normalized,
            name_to_code={name: index for index, name in enumerate(normalized)},
        )

    def encode(self, values: pd.Series, allow_unknown: bool = False) -> np.ndarray:
        """Encode values using this frozen category mapping.

        Args:
            values (pd.Series): Observation values to encode.
            allow_unknown (bool): Whether unseen names should receive code ``-1``.

        Returns:
            np.ndarray: Int64 category code for every input value.
        """

        if values.isna().any():
            raise ValueError(f"Observation column {self.column!r} contains missing values.")
        normalized = values.astype(str).to_numpy(copy=False)
        encoded = np.fromiter(
            (self.name_to_code.get(str(value), -1) for value in normalized),
            dtype=np.int64,
            count=len(normalized),
        )
        if not allow_unknown and np.any(encoded < 0):
            unknown = sorted(set(normalized[encoded < 0].tolist()))
            raise ValueError(
                f"Unknown categories in column {self.column!r}: {unknown[:10]}"
            )
        return encoded

    def to_state(self) -> dict[str, Any]:
        """Convert the category mapping to plain serialization state.

        Args:
            None.

        Returns:
            dict[str, Any]: Plain column, names, and name-to-code mapping.
        """

        return {
            "column": self.column,
            "names": list(self.names),
            "name_to_code": dict(self.name_to_code),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "CategoryMapping":
        """Restore a category mapping from plain serialization state.

        Args:
            state (Mapping[str, Any]): State returned by ``to_state``.

        Returns:
            CategoryMapping: Restored frozen category mapping.
        """

        mapping = cls.from_names(str(state["column"]), state["names"])
        stored = {str(key): int(value) for key, value in state["name_to_code"].items()}
        if stored != mapping.name_to_code:
            raise ValueError("Stored category names and name_to_code disagree.")
        return mapping


@dataclass(frozen=True)
class AnnDataMetadata:
    """In-memory metadata accompanying a backed sparse MSI matrix."""

    target_arrays: TargetArrays | None
    slide_codes: np.ndarray
    patient_codes: np.ndarray
    slide_mapping: CategoryMapping
    patient_mapping: CategoryMapping
    feature_names: tuple[str, ...]
    num_observations: int
    num_features: int


@dataclass(frozen=True)
class DataBundle:
    """Fitted preprocessing, metadata, datasets, and loaders for all splits."""

    metadata: AnnDataMetadata
    split_indices: dict[str, np.ndarray]
    datasets: dict[str, "SparseAnnDataDataset"]
    loaders: dict[str, DataLoader[dict[str, torch.Tensor]]]
    scaler: SparseFeatureScaler
    target_standardizer: TargetStandardizer


def _decode_attribute(value: Any) -> str:
    """Normalize an HDF5 text attribute to a Python string.

    Args:
        value (Any): HDF5 attribute value.

    Returns:
        str: Decoded attribute text.
    """

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_anndata_metadata(
    path: str | Path,
    target_columns: Sequence[str],
    batch_column: str,
    patient_column: str,
    matrix_key: str = "layers/counts",
    total_count: int = 36_100,
    haldane_correction: float = 0.5,
    slide_mapping: CategoryMapping | None = None,
    patient_mapping: CategoryMapping | None = None,
    require_targets: bool = True,
    load_targets: bool = True,
    allow_unknown_categories: bool = False,
) -> AnnDataMetadata:
    """Load labels and categories while keeping the sparse MSI matrix on disk.

    Args:
        path (str | Path): Input AnnData HDF5 path.
        target_columns (Sequence[str]): Ordered density target observation columns.
        batch_column (str): Slide/batch observation column.
        patient_column (str): Patient observation column.
        matrix_key (str): Backed CSR matrix key.
        total_count (int): Pixel count used to construct integer target counts.
        haldane_correction (float): Pseudocount used for target coordinates.
        slide_mapping (CategoryMapping | None): Optional frozen training slide mapping.
        patient_mapping (CategoryMapping | None): Optional frozen training patient mapping.
        require_targets (bool): Whether missing target columns are an error.
        load_targets (bool): Whether present target columns should be loaded and derived.
        allow_unknown_categories (bool): Whether unseen category names encode as ``-1``.

    Returns:
        AnnDataMetadata: Validated in-memory labels, mappings, and matrix dimensions.
    """

    file_path = Path(path)
    adata = ad.read_h5ad(file_path, backed="r")
    try:
        obs = adata.obs
        required_metadata = [batch_column, patient_column]
        missing_metadata = [name for name in required_metadata if name not in obs]
        if missing_metadata:
            raise KeyError(f"Missing required observation columns: {missing_metadata}")

        if require_targets and not load_targets:
            raise ValueError("require_targets=True is incompatible with load_targets=False.")
        target_names = tuple(str(value) for value in target_columns)
        present_targets = [name in obs for name in target_names]
        target_arrays = None
        if load_targets:
            if require_targets and not all(present_targets):
                missing_targets = [
                    name
                    for name, present in zip(
                        target_names, present_targets, strict=True
                    )
                    if not present
                ]
                raise KeyError(f"Missing required target columns: {missing_targets}")
            if any(present_targets) and not all(present_targets):
                raise KeyError("Target columns must be either all present or all absent.")
            if all(present_targets):
                densities = obs[list(target_names)].to_numpy(dtype=np.float32, copy=True)
                if np.isfinite(densities).all():
                    target_arrays = build_target_arrays(
                        densities,
                        total_count=total_count,
                        correction=haldane_correction,
                    )
                elif require_targets:
                    raise ValueError(
                        "Required training targets contain non-finite values."
                    )

        fitted_slide_mapping = slide_mapping or CategoryMapping.fit(
            obs[batch_column], batch_column
        )
        fitted_patient_mapping = patient_mapping or CategoryMapping.fit(
            obs[patient_column], patient_column
        )
        slide_codes = fitted_slide_mapping.encode(
            obs[batch_column],
            allow_unknown=allow_unknown_categories,
        )
        patient_codes = fitted_patient_mapping.encode(
            obs[patient_column],
            allow_unknown=allow_unknown_categories,
        )
        for slide_code in np.unique(slide_codes[slide_codes >= 0]):
            nested_patients = np.unique(patient_codes[slide_codes == slide_code])
            if nested_patients.size != 1 or nested_patients[0] < 0:
                slide_name = fitted_slide_mapping.names[int(slide_code)]
                raise ValueError(
                    f"Slide {slide_name!r} must be nested in exactly one known patient."
                )
        feature_names = tuple(str(value) for value in adata.var_names)
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("AnnData feature names must be unique.")

        group_path = matrix_group_path(matrix_key)
        with h5py.File(file_path, "r") as handle:
            if group_path not in handle:
                raise KeyError(
                    f"Sparse matrix group {group_path!r} is absent from {file_path}."
                )
            group = handle[group_path]
            encoding = _decode_attribute(group.attrs.get("encoding-type", ""))
            if encoding != "csr_matrix":
                raise TypeError(
                    f"{group_path!r} must be CSR encoded, observed {encoding!r}."
                )
            shape = tuple(int(value) for value in group.attrs["shape"])
        if shape != (obs.shape[0], len(feature_names)):
            raise ValueError(
                f"Matrix shape {shape} disagrees with metadata "
                f"{(obs.shape[0], len(feature_names))}."
            )
    finally:
        adata.file.close()

    return AnnDataMetadata(
        target_arrays=target_arrays,
        slide_codes=slide_codes,
        patient_codes=patient_codes,
        slide_mapping=fitted_slide_mapping,
        patient_mapping=fitted_patient_mapping,
        feature_names=feature_names,
        num_observations=shape[0],
        num_features=shape[1],
    )


def within_slide_split_indices(
    slide_codes: np.ndarray,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Create deterministic random train/validation/test splits within each slide.

    Args:
        slide_codes (np.ndarray): Nonnegative integer slide code for every row.
        train_fraction (float): Within-slide training fraction.
        validation_fraction (float): Within-slide validation fraction.
        test_fraction (float): Within-slide test fraction.
        seed (int): Random seed.

    Returns:
        dict[str, np.ndarray]: Shuffled global row indices keyed by split name.
    """

    codes = np.asarray(slide_codes, dtype=np.int64)
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction], dtype=np.float64
    )
    if codes.ndim != 1 or codes.size == 0 or np.any(codes < 0):
        raise ValueError("slide_codes must be a non-empty, nonnegative 1D array.")
    if np.any(fractions < 0.0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError(f"Split fractions must be nonnegative and sum to one: {fractions}.")

    rng = np.random.default_rng(seed)
    parts: dict[str, list[np.ndarray]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for slide_code in np.unique(codes):
        slide_rows = np.flatnonzero(codes == slide_code)
        rng.shuffle(slide_rows)
        train_count = int(np.floor(slide_rows.size * train_fraction))
        validation_count = int(np.floor(slide_rows.size * validation_fraction))
        validation_end = train_count + validation_count
        parts["train"].append(slide_rows[:train_count])
        parts["validation"].append(slide_rows[train_count:validation_end])
        parts["test"].append(slide_rows[validation_end:])

    splits: dict[str, np.ndarray] = {}
    for name, split_parts in parts.items():
        combined = np.concatenate(split_parts).astype(np.int64, copy=False)
        rng.shuffle(combined)
        splits[name] = combined
    return splits


def stratified_cap_indices(
    indices: np.ndarray,
    slide_codes: np.ndarray,
    maximum: int | None,
    seed: int,
) -> np.ndarray:
    """Cap selected rows while approximately preserving slide proportions.

    Args:
        indices (np.ndarray): Candidate global row indices.
        slide_codes (np.ndarray): Slide code for every complete-data row.
        maximum (int | None): Maximum retained rows, or no cap when null/non-positive.
        seed (int): Deterministic subsampling seed.

    Returns:
        np.ndarray: Possibly capped and deterministically shuffled global indices.
    """

    selected = np.asarray(indices, dtype=np.int64)
    codes = np.asarray(slide_codes, dtype=np.int64)
    if selected.ndim != 1 or codes.ndim != 1:
        raise ValueError("indices and slide_codes must be 1D arrays.")
    if np.any((selected < 0) | (selected >= codes.size)):
        raise IndexError("indices contain rows outside slide_codes.")
    if maximum is None or int(maximum) <= 0 or selected.size <= int(maximum):
        return selected.copy()

    cap = int(maximum)
    rng = np.random.default_rng(seed)
    labels = codes[selected]
    unique, counts = np.unique(labels, return_counts=True)
    ideal = counts.astype(np.float64) * cap / selected.size
    quotas = np.floor(ideal).astype(np.int64)
    remaining = cap - int(quotas.sum())
    order = np.argsort(-(ideal - quotas), kind="stable")
    for position in order:
        if remaining == 0:
            break
        if quotas[position] < counts[position]:
            quotas[position] += 1
            remaining -= 1
    if remaining != 0:
        raise RuntimeError("Unable to allocate stratified cap quotas.")

    retained: list[np.ndarray] = []
    for slide_code, quota in zip(unique, quotas, strict=True):
        if quota == 0:
            continue
        slide_rows = selected[labels == slide_code]
        retained.append(rng.choice(slide_rows, int(quota), replace=False))
    result = np.concatenate(retained).astype(np.int64, copy=False)
    rng.shuffle(result)
    return result


class SparseAnnDataDataset(Dataset[dict[str, Any]]):
    """Lazy map-style dataset reading and scaling individual backed CSR rows."""

    def __init__(
        self,
        path: str | Path,
        indices: np.ndarray,
        metadata: AnnDataMetadata,
        scaler: SparseFeatureScaler,
        target_standardizer: TargetStandardizer | None,
        matrix_key: str = "layers/counts",
    ) -> None:
        """Initialize a lazy sparse training or inference dataset.

        Args:
            path (str | Path): Input AnnData HDF5 path.
            indices (np.ndarray): Global rows exposed by this dataset.
            metadata (AnnDataMetadata): In-memory metadata and encoded categories.
            scaler (SparseFeatureScaler): Frozen training-fitted sparse scaler.
            target_standardizer (TargetStandardizer | None): Frozen target affine
                standardizer, or ``None`` for target-free inference.
            matrix_key (str): Backed CSR matrix key.

        Returns:
            None: Dataset state is initialized.
        """

        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError("indices must be a 1D array.")
        if np.any((selected < 0) | (selected >= metadata.num_observations)):
            raise IndexError("indices contain rows outside metadata.")
        if scaler.num_features != metadata.num_features:
            raise ValueError("Scaler feature count disagrees with dataset metadata.")
        if matrix_key != scaler.matrix_key:
            raise ValueError("Dataset matrix_key must match the fitted scaler matrix_key.")
        if metadata.target_arrays is not None and target_standardizer is None:
            raise ValueError("Labeled datasets require a fitted target_standardizer.")

        self.path = Path(path)
        self.indices = selected
        self.metadata = metadata
        self.scaler = scaler
        self.target_standardizer = target_standardizer
        self.group_path = matrix_group_path(matrix_key)
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
        """Open a process-local read-only HDF5 handle when first needed.

        Args:
            None.

        Returns:
            h5py.File: Open process-local HDF5 handle.
        """

        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Read and transform one sparse MSI row and its metadata.

        Args:
            item (int): Local dataset position.

        Returns:
            dict[str, Any]: Sparse features, category codes, row ID, and optional
                count-space and standardized targets.
        """

        row = int(self.indices[item])
        group = self._ensure_handle()[self.group_path]
        start, stop = np.asarray(group["indptr"][row : row + 2], dtype=np.int64)
        feature_indices = np.asarray(group["indices"][start:stop], dtype=np.int64)
        stored_values = np.asarray(group["data"][start:stop], dtype=np.float64)
        slide_code = int(self.metadata.slide_codes[row])
        transformed = self.scaler.transform_sparse(
            feature_indices,
            stored_values,
            slide_code,
        )
        sample: dict[str, Any] = {
            "feature_indices": feature_indices,
            "values": transformed,
            "slide_code": slide_code,
            "patient_code": int(self.metadata.patient_codes[row]),
            "row_id": row,
        }
        target_arrays = self.metadata.target_arrays
        if target_arrays is not None:
            if self.target_standardizer is None:
                raise RuntimeError("Labeled sample has no target standardizer.")
            coordinates = target_arrays.coordinates[row]
            sample.update(
                {
                    "target_densities": target_arrays.densities[row],
                    "target_counts": target_arrays.counts[row],
                    "extratumoral_count": target_arrays.extratumoral_counts[row],
                    "target_denominators": target_arrays.denominators[row],
                    "target_positive_mask": target_arrays.positive_mask[row],
                    "target_coordinates": coordinates.astype(np.float32),
                    "targets": self.target_standardizer.transform(coordinates),
                }
            )
        return sample

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state without a cross-process HDF5 handle.

        Args:
            None.

        Returns:
            dict[str, Any]: Serializable dataset instance state.
        """

        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self) -> None:
        """Close this process's HDF5 handle if it is open.

        Args:
            None.

        Returns:
            None: The local file handle is closed.
        """

        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        """Release the process-local HDF5 handle during garbage collection.

        Args:
            None.

        Returns:
            None: Resources are released best-effort.
        """

        self.close()


def sparse_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate variable-length sparse rows without dense feature expansion.

    Args:
        samples (Sequence[Mapping[str, Any]]): Sparse sample dictionaries.

    Returns:
        dict[str, torch.Tensor]: Concatenated sparse features, row membership,
            categories, and optional target tensors.
    """

    if not samples:
        raise ValueError("Cannot collate an empty sample sequence.")
    lengths = np.asarray(
        [len(sample["feature_indices"]) for sample in samples], dtype=np.int64
    )
    feature_indices = np.concatenate(
        [np.asarray(sample["feature_indices"], dtype=np.int64) for sample in samples]
    )
    values = np.concatenate(
        [np.asarray(sample["values"], dtype=np.float32) for sample in samples]
    )
    sample_indices = np.repeat(np.arange(len(samples), dtype=np.int64), lengths)
    result = {
        "feature_indices": torch.from_numpy(feature_indices),
        "values": torch.from_numpy(values),
        "sample_indices": torch.from_numpy(sample_indices),
        "feature_counts": torch.from_numpy(lengths),
        "slide_codes": torch.from_numpy(
            np.asarray([sample["slide_code"] for sample in samples], dtype=np.int64)
        ),
        "patient_codes": torch.from_numpy(
            np.asarray([sample["patient_code"] for sample in samples], dtype=np.int64)
        ),
        "row_ids": torch.from_numpy(
            np.asarray([sample["row_id"] for sample in samples], dtype=np.int64)
        ),
    }
    target_keys = (
        "target_densities",
        "target_counts",
        "target_denominators",
        "target_positive_mask",
        "target_coordinates",
        "targets",
    )
    has_targets = "targets" in samples[0]
    if any(("targets" in sample) != has_targets for sample in samples):
        raise ValueError("Cannot collate a mixture of labeled and unlabeled samples.")
    if has_targets:
        for key in target_keys:
            stacked = np.stack([np.asarray(sample[key]) for sample in samples])
            result[key] = torch.from_numpy(stacked)
        has_extratumoral = "extratumoral_count" in samples[0]
        if any(
            ("extratumoral_count" in sample) != has_extratumoral
            for sample in samples
        ):
            raise ValueError("extratumoral_count must be present in every sample or none.")
        if has_extratumoral:
            result["extratumoral_count"] = torch.from_numpy(
                np.asarray(
                    [sample["extratumoral_count"] for sample in samples],
                    dtype=np.int64,
                )
            )
    return result


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python random state inside a PyTorch DataLoader worker.

    Args:
        worker_id (int): Worker identifier supplied by PyTorch.

    Returns:
        None: Worker-local random state is updated.
    """

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_kwargs(training_config: Mapping[str, Any]) -> dict[str, Any]:
    """Build shared DataLoader keyword arguments.

    Args:
        training_config (Mapping[str, Any]): Training configuration section.

    Returns:
        dict[str, Any]: Shared deterministic sparse DataLoader options.
    """

    num_workers = int(training_config["num_workers"])
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "collate_fn": sparse_collate,
        "pin_memory": bool(training_config["pin_memory"]),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(training_config["prefetch_factor"])
    return kwargs


def create_data_bundle(config: Mapping[str, Any]) -> DataBundle:
    """Create fitted preprocessing and sparse datasets from configuration.

    Both the feature scaler and target standardizer are fitted only after split
    caps are applied, using the resulting actual selected training indices.

    Args:
        config (Mapping[str, Any]): Complete validated IHC-MVN configuration.

    Returns:
        DataBundle: Metadata, split rows, frozen fitted preprocessing, datasets,
            and deterministic DataLoaders.
    """

    data_config = config["data"]
    preprocessing_config = config["preprocessing"]
    target_config = config["targets"]
    training_config = config["training"]
    path = Path(data_config["path"])
    metadata = load_anndata_metadata(
        path=path,
        target_columns=data_config["target_columns"],
        batch_column=str(data_config["batch_column"]),
        patient_column=str(data_config["patient_column"]),
        matrix_key=str(data_config["matrix_key"]),
        total_count=int(target_config["total_count"]),
        haldane_correction=float(target_config["haldane_correction"]),
    )
    if metadata.target_arrays is None:
        raise RuntimeError("Training data must contain all target columns.")

    seed = int(training_config["seed"])
    split_indices = within_slide_split_indices(
        metadata.slide_codes,
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
    for offset, (split_name, cap_key) in enumerate(cap_keys.items(), start=1):
        split_indices[split_name] = stratified_cap_indices(
            split_indices[split_name],
            metadata.slide_codes,
            data_config.get(cap_key),
            seed + offset,
        )
        if split_indices[split_name].size == 0:
            raise ValueError(f"Selected {split_name!r} split is empty.")

    scaler = fit_sparse_feature_scaler(
        path=path,
        matrix_key=str(data_config["matrix_key"]),
        train_indices=split_indices["train"],
        slide_codes=metadata.slide_codes,
        slide_names=metadata.slide_mapping.names,
        feature_names=metadata.feature_names,
        scale_floor=float(preprocessing_config["scale_floor"]),
        row_chunk_size=int(preprocessing_config["fit_row_chunk_size"]),
    )
    target_standardizer = TargetStandardizer.fit(
        coordinates=metadata.target_arrays.coordinates,
        positive_mask=metadata.target_arrays.positive_mask,
        train_indices=split_indices["train"],
        target_names=data_config["target_columns"],
        standard_deviation_floor=float(
            target_config["standard_deviation_floor"]
        ),
    )

    datasets = {
        name: SparseAnnDataDataset(
            path=path,
            indices=indices,
            metadata=metadata,
            scaler=scaler,
            target_standardizer=target_standardizer,
            matrix_key=str(data_config["matrix_key"]),
        )
        for name, indices in split_indices.items()
    }
    common_loader_kwargs = _loader_kwargs(training_config)
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
            batch_size=int(training_config["test_batch_size"]),
            shuffle=False,
            **common_loader_kwargs,
        ),
    }
    return DataBundle(
        metadata=metadata,
        split_indices=split_indices,
        datasets=datasets,
        loaders=loaders,
        scaler=scaler,
        target_standardizer=target_standardizer,
    )


def create_inference_dataset(
    path: str | Path,
    scaler: SparseFeatureScaler,
    patient_mapping: CategoryMapping,
    batch_column: str = "batch",
    patient_column: str = "patient",
    target_columns: Sequence[str] = TARGET_COLUMNS,
    target_standardizer: TargetStandardizer | None = None,
) -> SparseAnnDataDataset:
    """Construct inference data from frozen scaler and category mappings.

    Unseen slides and patients are encoded as ``-1``. The scaler uses its global
    per-feature fallback for unseen slides and is never refitted.

    Args:
        path (str | Path): Inference AnnData HDF5 path.
        scaler (SparseFeatureScaler): Stored frozen training feature scaler.
        patient_mapping (CategoryMapping): Stored frozen training patient mapping.
        batch_column (str): Slide/batch observation column.
        patient_column (str): Patient observation column.
        target_columns (Sequence[str]): Optional ordered density target names.
        target_standardizer (TargetStandardizer | None): Stored frozen target
            standardizer when the inference file contains labels.

    Returns:
        SparseAnnDataDataset: Lazy full-row inference dataset using frozen state.
    """

    slide_mapping = CategoryMapping.from_names(batch_column, scaler.slide_names)
    metadata = load_anndata_metadata(
        path=path,
        target_columns=target_columns,
        batch_column=batch_column,
        patient_column=patient_column,
        matrix_key=scaler.matrix_key,
        slide_mapping=slide_mapping,
        patient_mapping=patient_mapping,
        require_targets=False,
        load_targets=target_standardizer is not None,
        allow_unknown_categories=True,
    )
    if metadata.feature_names != scaler.feature_names:
        raise ValueError(
            "Inference feature names or order do not match the frozen scaler schema."
        )
    if metadata.target_arrays is not None and target_standardizer is None:
        raise ValueError(
            "A target_standardizer is required when inference targets are present."
        )
    return SparseAnnDataDataset(
        path=path,
        indices=np.arange(metadata.num_observations, dtype=np.int64),
        metadata=metadata,
        scaler=scaler,
        target_standardizer=target_standardizer,
        matrix_key=scaler.matrix_key,
    )

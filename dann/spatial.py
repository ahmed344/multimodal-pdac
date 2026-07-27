"""Run trained DANN inference over every row of a tissue AnnData file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from dann.config import resolve_device, seed_everything
from dann.model import AdversarialLatentFusion


DEFAULT_INPUT = Path(
    "/workspaces/multimodal-pdac/data/PDAC/Raw/adata_assembled_tissue.h5ad"
)
DEFAULT_CHECKPOINT = Path(
    "/workspaces/multimodal-pdac/data/PDAC/Results/dann/model/best.pt"
)
DEFAULT_OUTPUT = Path(
    "/workspaces/multimodal-pdac/data/PDAC/Results/dann/analysis/spatial/"
    "spatial_inference.parquet"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse spatial inference command-line arguments.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        argparse.Namespace: Parsed spatial inference options.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        default=None,
        help="Runtime device override; defaults to the checkpoint training device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference rows per batch; defaults to checkpoint analysis.batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers; defaults to checkpoint analysis.num_workers.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional leading-row cap for smoke testing; default processes all rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only after successful inference.",
    )
    return parser.parse_args(argv)


def matrix_group_path(matrix_key: str) -> str:
    """Translate an AnnData matrix key into its HDF5 group path.

    Args:
        matrix_key (str): ``X``, a layer name, or ``layers/<name>``.

    Returns:
        str: HDF5 path containing the sparse matrix.
    """

    if matrix_key == "X":
        return "X"
    if matrix_key.startswith("layers/"):
        return matrix_key
    return f"layers/{matrix_key}"


def _axis_index_key(axis_group: h5py.Group) -> str:
    """Read and normalize an AnnData axis index key.

    Args:
        axis_group (h5py.Group): AnnData ``obs`` or ``var`` HDF5 group.

    Returns:
        str: Child key containing the axis index.
    """

    key = axis_group.attrs.get("_index", "_index")
    return key.decode("utf-8") if isinstance(key, bytes) else str(key)


def _string_values(index_object: h5py.Dataset | h5py.Group) -> h5py.Dataset:
    """Locate the string-value dataset for an AnnData index encoding.

    Args:
        index_object (h5py.Dataset | h5py.Group): Encoded AnnData index object.

    Returns:
        h5py.Dataset: Dataset containing index string values.
    """

    if isinstance(index_object, h5py.Dataset):
        return index_object
    if "values" not in index_object:
        raise TypeError("Only string or nullable-string AnnData indices are supported.")
    return index_object["values"]


def read_axis_names(path: Path, axis: str) -> np.ndarray:
    """Read an AnnData axis index as an in-memory string array.

    Args:
        path (Path): AnnData HDF5 path.
        axis (str): Axis group, either ``obs`` or ``var``.

    Returns:
        np.ndarray: String index values in stored order.
    """

    if axis not in {"obs", "var"}:
        raise ValueError(f"Unsupported AnnData axis: {axis!r}")
    with h5py.File(path, "r") as handle:
        group = handle[axis]
        index_object = group[_axis_index_key(group)]
        if isinstance(index_object, h5py.Group) and "mask" in index_object:
            if np.any(np.asarray(index_object["mask"][:], dtype=bool)):
                raise ValueError(f"{axis}_names contains missing values.")
        values = _string_values(index_object).asstr()[:]
    return np.asarray(values, dtype=str)


class ObservationNameReader:
    """Read contiguous observation-name slices without loading all names."""

    def __init__(self, path: Path) -> None:
        """Open the AnnData observation index for streaming reads.

        Args:
            path (Path): AnnData HDF5 path.

        Returns:
            None: Reader state is initialized.
        """

        self._handle = h5py.File(path, "r")
        obs = self._handle["obs"]
        self._index_object = obs[_axis_index_key(obs)]
        self._values = _string_values(self._index_object)

    def read(self, start: int, stop: int) -> np.ndarray:
        """Read one contiguous observation-name interval.

        Args:
            start (int): Inclusive zero-based row position.
            stop (int): Exclusive zero-based row position.

        Returns:
            np.ndarray: String observation names for ``[start, stop)``.
        """

        if isinstance(self._index_object, h5py.Group) and "mask" in self._index_object:
            mask = np.asarray(self._index_object["mask"][start:stop], dtype=bool)
            if np.any(mask):
                raise ValueError(
                    f"obs_names contains missing values in rows [{start}, {stop})."
                )
        return np.asarray(self._values.asstr()[start:stop], dtype=str)

    def close(self) -> None:
        """Close the underlying read-only HDF5 handle.

        Args:
            None.

        Returns:
            None: The handle is closed.
        """

        self._handle.close()

    def __enter__(self) -> "ObservationNameReader":
        """Enter a managed observation-name reader context.

        Args:
            None.

        Returns:
            ObservationNameReader: This open reader.
        """

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Close the reader when leaving a managed context.

        Args:
            exc_type (type[BaseException] | None): Raised exception type, if any.
            exc_value (BaseException | None): Raised exception value, if any.
            traceback (Any): Raised exception traceback, if any.

        Returns:
            None: The reader is closed.
        """

        self.close()


class SparseInferenceDataset(Dataset[dict[str, Any]]):
    """Read sequential sparse MSI rows without requiring target labels."""

    def __init__(
        self,
        path: Path,
        num_rows: int,
        matrix_key: str,
        intensity_transform: str,
        intensity_clip_max: float | None,
        nonzero_threshold: float,
    ) -> None:
        """Initialize a lazy read-only sparse inference dataset.

        Args:
            path (Path): Input AnnData path.
            num_rows (int): Number of leading rows exposed by the dataset.
            matrix_key (str): Sparse AnnData matrix source.
            intensity_transform (str): ``none``, ``log1p``, or ``sqrt``.
            intensity_clip_max (float | None): Optional post-transform upper clip.
            nonzero_threshold (float): Exclude values at or below this threshold.

        Returns:
            None: Dataset state is initialized.
        """

        if intensity_transform not in {"none", "log1p", "sqrt"}:
            raise ValueError(f"Unsupported intensity transform: {intensity_transform}")
        self.path = Path(path)
        self.num_rows = int(num_rows)
        self.group_path = matrix_group_path(matrix_key)
        self.intensity_transform = intensity_transform
        self.intensity_clip_max = intensity_clip_max
        self.nonzero_threshold = float(nonzero_threshold)
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        """Return the number of selected tissue observations.

        Args:
            None.

        Returns:
            int: Dataset length.
        """

        return self.num_rows

    def _ensure_handle(self) -> h5py.File:
        """Open a process-local HDF5 handle when first needed.

        Args:
            None.

        Returns:
            h5py.File: Open read-only AnnData handle.
        """

        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def _transform_values(self, values: np.ndarray) -> np.ndarray:
        """Apply checkpoint-matched preprocessing to sparse intensities.

        Args:
            values (np.ndarray): Raw nonzero sparse-row intensities.

        Returns:
            np.ndarray: Validated transformed float32 intensities.
        """

        values = values.astype(np.float32, copy=False)
        if self.intensity_transform == "log1p":
            values = np.log1p(values)
        elif self.intensity_transform == "sqrt":
            values = np.sqrt(values)
        if self.intensity_clip_max is not None:
            values = np.minimum(values, float(self.intensity_clip_max))
        values = values.astype(np.float32, copy=False)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("Encountered invalid transformed MSI intensities.")
        return values

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Read one sparse MSI row by its global row position.

        Args:
            item (int): Zero-based global tissue row position.

        Returns:
            dict[str, Any]: Peak indices, transformed intensities, and row position.
        """

        row = int(item)
        group = self._ensure_handle()[self.group_path]
        start, stop = np.asarray(group["indptr"][row : row + 2], dtype=np.int64)
        peak_indices = np.asarray(group["indices"][start:stop], dtype=np.int64)
        intensities = np.asarray(group["data"][start:stop], dtype=np.float32)
        if self.nonzero_threshold > 0.0:
            keep = intensities > self.nonzero_threshold
            peak_indices = peak_indices[keep]
            intensities = intensities[keep]
        return {
            "peak_indices": peak_indices,
            "intensities": self._transform_values(intensities),
            "row_id": row,
        }

    def __getstate__(self) -> dict[str, Any]:
        """Prepare pickle state without sharing an HDF5 handle.

        Args:
            None.

        Returns:
            dict[str, Any]: Serializable dataset state.
        """

        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self) -> None:
        """Close this process's HDF5 handle if it is open.

        Args:
            None.

        Returns:
            None: Open resources are released.
        """

        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        """Release the process-local HDF5 handle best-effort.

        Args:
            None.

        Returns:
            None: Open resources are released when possible.
        """

        self.close()


def sparse_inference_collate(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    """Collate variable-length inference rows without dense expansion.

    Args:
        samples (Sequence[Mapping[str, Any]]): Sparse row dictionaries.

    Returns:
        dict[str, torch.Tensor]: Concatenated sparse model inputs and row positions.
    """

    if not samples:
        raise ValueError("Cannot collate an empty inference batch.")
    lengths = np.asarray([len(sample["peak_indices"]) for sample in samples], dtype=np.int64)
    peak_indices = np.concatenate(
        [np.asarray(sample["peak_indices"], dtype=np.int64) for sample in samples]
    )
    intensities = np.concatenate(
        [np.asarray(sample["intensities"], dtype=np.float32) for sample in samples]
    )
    sample_indices = np.repeat(np.arange(len(samples), dtype=np.int64), lengths)
    row_ids = np.asarray([sample["row_id"] for sample in samples], dtype=np.int64)
    return {
        "peak_indices": torch.from_numpy(peak_indices),
        "intensities": torch.from_numpy(intensities),
        "sample_indices": torch.from_numpy(sample_indices),
        "peak_counts": torch.from_numpy(lengths),
        "row_ids": torch.from_numpy(row_ids),
    }


def validate_input_schema(
    input_path: Path,
    training_path: Path,
    matrix_key: str,
    target_columns: Sequence[str],
    num_peaks: int,
) -> int:
    """Validate tissue features and targets against checkpoint training data.

    Args:
        input_path (Path): Tissue AnnData inference path.
        training_path (Path): AnnData path recorded during model training.
        matrix_key (str): Sparse matrix key used during training.
        target_columns (Sequence[str]): Ordered checkpoint target names.
        num_peaks (int): Checkpoint model feature count.

    Returns:
        int: Number of tissue observations available for inference.
    """

    group_path = matrix_group_path(matrix_key)
    with h5py.File(input_path, "r") as handle:
        if group_path not in handle:
            raise KeyError(f"Sparse matrix group {group_path!r} is absent.")
        group = handle[group_path]
        encoding = group.attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        if encoding != "csr_matrix":
            raise TypeError(
                f"{group_path!r} must be CSR encoded, observed {encoding!r}."
            )
        shape = tuple(int(value) for value in group.attrs["shape"])
        missing_targets = [
            column for column in target_columns if column not in handle["obs"]
        ]
        if missing_targets:
            raise KeyError(f"Missing expected target columns: {missing_targets}")
    if shape[1] != int(num_peaks):
        raise ValueError(
            f"Tissue has {shape[1]} features but checkpoint expects {num_peaks}."
        )
    tissue_features = read_axis_names(input_path, "var")
    training_features = read_axis_names(training_path, "var")
    if not np.array_equal(tissue_features, training_features):
        raise ValueError(
            "Tissue var_names do not exactly match checkpoint training feature order."
        )
    return shape[0]


def prediction_column_names(target_columns: Sequence[str]) -> list[str]:
    """Build deterministic output names for every target triplet.

    Args:
        target_columns (Sequence[str]): Ordered density target names.

    Returns:
        list[str]: Per-target mean, presence-probability, and sigma columns.
    """

    names: list[str] = []
    for target in target_columns:
        names.extend(
            [
                f"mean_{target}",
                f"prob_of_presence_{target}",
                f"sigma_{target}",
            ]
        )
    return names


def output_schema(target_columns: Sequence[str]) -> pa.Schema:
    """Construct the stable Parquet output schema.

    Args:
        target_columns (Sequence[str]): Ordered density target names.

    Returns:
        pa.Schema: Row identity and float32 prediction fields.
    """

    fields = [
        pa.field("row_position", pa.int64(), nullable=False),
        pa.field("obs_name", pa.string(), nullable=False),
    ]
    fields.extend(
        pa.field(name, pa.float32(), nullable=False)
        for name in prediction_column_names(target_columns)
    )
    return pa.schema(fields)


def build_output_table(
    row_positions: np.ndarray,
    obs_names: np.ndarray,
    mu: np.ndarray,
    pi: np.ndarray,
    sigma: np.ndarray,
    target_columns: Sequence[str],
) -> pa.Table:
    """Build one typed output table from a model inference batch.

    Args:
        row_positions (np.ndarray): Global zero-based AnnData row positions.
        obs_names (np.ndarray): Original possibly duplicated observation names.
        mu (np.ndarray): Native ZILN means on the positive logit-density scale.
        pi (np.ndarray): Structural-zero probabilities.
        sigma (np.ndarray): Positive-branch ZILN standard deviations.
        target_columns (Sequence[str]): Ordered density target names.

    Returns:
        pa.Table: Typed Parquet-ready identity and prediction values.
    """

    expected_shape = (row_positions.size, len(target_columns))
    if mu.shape != expected_shape or pi.shape != expected_shape or sigma.shape != expected_shape:
        raise ValueError(
            f"Prediction arrays must all have shape {expected_shape}; observed "
            f"{mu.shape}, {pi.shape}, and {sigma.shape}."
        )
    if obs_names.size != row_positions.size:
        raise ValueError("Observation names and row positions must have equal length.")
    arrays: list[pa.Array] = [
        pa.array(np.asarray(row_positions, dtype=np.int64), type=pa.int64()),
        pa.array(np.asarray(obs_names, dtype=str), type=pa.string()),
    ]
    for index in range(len(target_columns)):
        arrays.extend(
            [
                pa.array(np.asarray(mu[:, index], dtype=np.float32), type=pa.float32()),
                pa.array(
                    np.asarray(1.0 - pi[:, index], dtype=np.float32),
                    type=pa.float32(),
                ),
                pa.array(
                    np.asarray(sigma[:, index], dtype=np.float32),
                    type=pa.float32(),
                ),
            ]
        )
    return pa.Table.from_arrays(arrays, schema=output_schema(target_columns))


def load_checkpoint_model(
    checkpoint_path: Path,
    device_override: str | None,
) -> tuple[
    AdversarialLatentFusion,
    dict[str, Any],
    tuple[str, ...],
    torch.device,
    int,
]:
    """Load a trained model using its immutable checkpoint configuration.

    Args:
        checkpoint_path (Path): Trained DANN checkpoint path.
        device_override (str | None): Optional runtime device override.

    Returns:
        tuple: Evaluation model, saved config, target names, device, and epoch.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model_state", "config", "batch_names", "target_columns", "epoch"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {sorted(missing)}")
    config = dict(checkpoint["config"])
    target_columns = tuple(str(value) for value in checkpoint["target_columns"])
    batch_names = tuple(str(value) for value in checkpoint["batch_names"])
    requested_device = (
        device_override
        if device_override is not None
        else str(config["training"]["device"])
    )
    device = resolve_device(requested_device)
    model = AdversarialLatentFusion.from_config(
        config,
        num_batches=len(batch_names),
        num_targets=len(target_columns),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config, target_columns, device, int(checkpoint["epoch"]) + 1


def create_inference_loader(
    dataset: SparseInferenceDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Create a deterministic sequential sparse inference loader.

    Args:
        dataset (SparseInferenceDataset): All-row sparse tissue dataset.
        batch_size (int): Number of tissue rows per inference batch.
        num_workers (int): Number of HDF5 reader processes.
        pin_memory (bool): Whether to pin collated CPU tensors.
        prefetch_factor (int): Batches prefetched by each worker.

    Returns:
        DataLoader[dict[str, torch.Tensor]]: Sequential inference loader.
    """

    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "collate_fn": sparse_inference_collate,
        "pin_memory": bool(pin_memory),
        "persistent_workers": int(num_workers) > 0,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def run_inference(
    input_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device_override: str | None = None,
    batch_size_override: int | None = None,
    num_workers_override: int | None = None,
    max_rows: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Run all-row sparse tissue inference and atomically write Parquet.

    Args:
        input_path (Path): Tissue AnnData inference path.
        checkpoint_path (Path): Trained DANN checkpoint path.
        output_path (Path): Destination Parquet path.
        device_override (str | None): Optional runtime device override.
        batch_size_override (int | None): Optional inference batch-size override.
        num_workers_override (int | None): Optional DataLoader-worker override.
        max_rows (int | None): Optional leading-row cap for smoke testing.
        overwrite (bool): Whether to replace an existing output after success.

    Returns:
        Path: Validated Parquet output path.
    """

    input_path = Path(input_path)
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    model, config, target_columns, device, checkpoint_epoch = load_checkpoint_model(
        checkpoint_path, device_override
    )
    data_config = config["data"]
    total_rows = validate_input_schema(
        input_path=input_path,
        training_path=Path(data_config["path"]),
        matrix_key=str(data_config["matrix_key"]),
        target_columns=target_columns,
        num_peaks=int(config["model"]["num_peaks"]),
    )
    if max_rows is None:
        selected_rows = total_rows
    else:
        if int(max_rows) <= 0:
            raise ValueError("max_rows must be positive when provided.")
        selected_rows = min(total_rows, int(max_rows))
    analysis_config = config["analysis"]
    training_config = config["training"]
    batch_size = int(batch_size_override or analysis_config["batch_size"])
    num_workers = int(
        analysis_config["num_workers"]
        if num_workers_override is None
        else num_workers_override
    )
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative.")
    seed_everything(int(training_config["seed"]))
    dataset = SparseInferenceDataset(
        path=input_path,
        num_rows=selected_rows,
        matrix_key=str(data_config["matrix_key"]),
        intensity_transform=str(data_config["intensity_transform"]),
        intensity_clip_max=data_config.get("intensity_clip_max"),
        nonzero_threshold=float(data_config["nonzero_threshold"]),
    )
    loader = create_inference_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=bool(training_config["pin_memory"]) and device.type == "cuda",
        prefetch_factor=int(training_config["prefetch_factor"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    schema = output_schema(target_columns)
    writer = pq.ParquetWriter(
        temporary_path,
        schema,
        compression="zstd",
        use_dictionary=["obs_name"],
    )
    processed = 0
    print(
        f"Inference checkpoint epoch {checkpoint_epoch} on {device}; "
        f"processing {selected_rows:,} rows.",
        flush=True,
    )
    try:
        with ObservationNameReader(input_path) as name_reader, torch.inference_mode():
            for batch_number, cpu_batch in enumerate(loader, start=1):
                row_positions = cpu_batch.pop("row_ids").numpy()
                expected = np.arange(
                    processed, processed + row_positions.size, dtype=np.int64
                )
                if not np.array_equal(row_positions, expected):
                    raise RuntimeError("Inference DataLoader changed tissue row order.")
                model_batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in cpu_batch.items()
                }
                latent = model.encode(model_batch)
                predictions = model.biology_predictor(latent)
                table = build_output_table(
                    row_positions=row_positions,
                    obs_names=name_reader.read(processed, processed + row_positions.size),
                    mu=predictions["mu"].cpu().numpy(),
                    pi=predictions["pi"].cpu().numpy(),
                    sigma=predictions["sigma"].cpu().numpy(),
                    target_columns=target_columns,
                )
                writer.write_table(table, row_group_size=row_positions.size)
                processed += row_positions.size
                if batch_number % 25 == 0 or processed == selected_rows:
                    print(
                        f"Processed {processed:,}/{selected_rows:,} rows "
                        f"({processed / selected_rows:.1%}).",
                        flush=True,
                    )
        if processed != selected_rows:
            raise RuntimeError(
                f"Inference wrote {processed} rows but expected {selected_rows}."
            )
        writer.close()
        os.replace(temporary_path, output_path)
    except BaseException:
        writer.close()
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    finally:
        dataset.close()
    validate_output(
        output_path=output_path,
        input_path=input_path,
        target_columns=target_columns,
        expected_rows=selected_rows,
    )
    return output_path


def validate_output(
    output_path: Path,
    input_path: Path,
    target_columns: Sequence[str],
    expected_rows: int,
    batch_size: int = 262_144,
) -> dict[str, Any]:
    """Validate identity, order, dtypes, and ranges in a Parquet export.

    Args:
        output_path (Path): Spatial inference Parquet path.
        input_path (Path): Source tissue AnnData path.
        target_columns (Sequence[str]): Ordered checkpoint targets.
        expected_rows (int): Required output row count.
        batch_size (int): Rows checked per validation batch.

    Returns:
        dict[str, Any]: Validated row count, columns, and value ranges.
    """

    parquet = pq.ParquetFile(output_path)
    expected_schema = output_schema(target_columns)
    if not parquet.schema_arrow.equals(expected_schema):
        raise TypeError(
            f"Unexpected Parquet schema:\n{parquet.schema_arrow}\n"
            f"Expected:\n{expected_schema}"
        )
    if parquet.metadata.num_rows != int(expected_rows):
        raise ValueError(
            f"Output has {parquet.metadata.num_rows} rows; expected {expected_rows}."
        )
    prediction_columns = prediction_column_names(target_columns)
    minima = {name: np.inf for name in prediction_columns}
    maxima = {name: -np.inf for name in prediction_columns}
    checked = 0
    with ObservationNameReader(input_path) as name_reader:
        for batch in parquet.iter_batches(batch_size=batch_size):
            size = batch.num_rows
            positions = batch.column("row_position").to_numpy(zero_copy_only=False)
            expected = np.arange(checked, checked + size, dtype=np.int64)
            if not np.array_equal(positions, expected):
                raise ValueError("Parquet row_position is not contiguous and ordered.")
            observed_names = np.asarray(batch.column("obs_name").to_pylist(), dtype=str)
            expected_names = name_reader.read(checked, checked + size)
            if not np.array_equal(observed_names, expected_names):
                raise ValueError("Parquet obs_name order differs from source AnnData.")
            for name in prediction_columns:
                values = batch.column(name).to_numpy(zero_copy_only=False)
                if not np.isfinite(values).all():
                    raise ValueError(f"Prediction column {name!r} contains non-finite values.")
                if name.startswith("prob_of_presence_") and np.any(
                    (values < 0.0) | (values > 1.0)
                ):
                    raise ValueError(f"Probability column {name!r} lies outside [0, 1].")
                if name.startswith("sigma_") and np.any(values <= 0.0):
                    raise ValueError(f"Sigma column {name!r} is not strictly positive.")
                minima[name] = min(minima[name], float(values.min()))
                maxima[name] = max(maxima[name], float(values.max()))
            checked += size
    if checked != expected_rows:
        raise ValueError(f"Validated {checked} rows; expected {expected_rows}.")
    print(
        f"Validated {checked:,} ordered rows and {len(prediction_columns)} "
        f"float32 prediction columns in {output_path}.",
        flush=True,
    )
    return {
        "rows": checked,
        "prediction_columns": prediction_columns,
        "minima": minima,
        "maxima": maxima,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Load options and execute full spatial inference.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        int: Process exit status, zero on success.
    """

    args = parse_args(argv)
    output = run_inference(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device_override=args.device,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
    )
    print(f"Spatial inference written to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

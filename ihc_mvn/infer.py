"""Stream standalone IHC-MVN predictions and latent vectors to Parquet."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import resolve_device, seed_everything
from .data_loader import (
    CategoryMapping,
    SparseAnnDataDataset,
    create_inference_dataset,
    seed_worker,
    sparse_collate,
)
from .losses import conditional_gaussian_cd8
from .model import IHCMultivariateModel
from .scaling import SparseFeatureScaler, matrix_group_path
from .targets import TargetStandardizer


DEFAULT_CHECKPOINT = Path(
    "/workspaces/multimodal-pdac/data/PDAC/Results/ihc_mvn/best.pt"
)
NORMAL_Q05 = -1.6448536269514722
NORMAL_Q95 = 1.6448536269514722


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse standalone inference command-line arguments.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        argparse.Namespace: Parsed inference overrides.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input H5AD; defaults to checkpoint data.inference_path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Standalone IHC-MVN checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Prediction Parquet; defaults below checkpoint output.directory.",
    )
    parser.add_argument(
        "--latent-output",
        type=Path,
        default=None,
        help="Latent Parquet; defaults beside the prediction output.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override; defaults to checkpoint training.device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size override.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Inference DataLoader worker-count override.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional positive leading-row cap.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing outputs after successful inference.",
    )
    return parser.parse_args(argv)


def _axis_index_key(axis_group: h5py.Group) -> str:
    """Read an AnnData axis index key.

    Args:
        axis_group (h5py.Group): AnnData ``obs`` or ``var`` group.

    Returns:
        str: Child key containing the encoded axis index.
    """

    value = axis_group.attrs.get("_index", "_index")
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _string_values(index_object: h5py.Dataset | h5py.Group) -> h5py.Dataset:
    """Locate the value dataset for an AnnData string index.

    Args:
        index_object (h5py.Dataset | h5py.Group): Encoded AnnData index.

    Returns:
        h5py.Dataset: Underlying string value dataset.
    """

    if isinstance(index_object, h5py.Dataset):
        return index_object
    if "values" not in index_object:
        raise TypeError("Only string or nullable-string AnnData indices are supported.")
    return index_object["values"]


def read_axis_names(path: Path, axis: str) -> tuple[str, ...]:
    """Read ordered AnnData observation or feature names.

    Args:
        path (Path): Input H5AD path.
        axis (str): Axis group name, either ``"obs"`` or ``"var"``.

    Returns:
        tuple[str, ...]: Ordered axis names, including duplicate observation names.
    """

    if axis not in {"obs", "var"}:
        raise ValueError("axis must be 'obs' or 'var'.")
    with h5py.File(path, "r") as handle:
        group = handle[axis]
        index_object = group[_axis_index_key(group)]
        if isinstance(index_object, h5py.Group) and "mask" in index_object:
            if np.any(np.asarray(index_object["mask"][:], dtype=bool)):
                raise ValueError(f"{axis}_names contains missing values.")
        values = _string_values(index_object).asstr()[:]
    return tuple(str(value) for value in values)


class ObservationNameReader:
    """Stream contiguous observation-name slices from AnnData."""

    def __init__(self, path: Path) -> None:
        """Open a read-only observation index.

        Args:
            path (Path): Input AnnData H5AD path.

        Returns:
            None: The streaming reader is initialized.
        """

        self._handle = h5py.File(path, "r")
        obs = self._handle["obs"]
        self._index_object = obs[_axis_index_key(obs)]
        self._values = _string_values(self._index_object)

    def read(self, start: int, stop: int) -> np.ndarray:
        """Read a contiguous observation-name interval.

        Args:
            start (int): Inclusive global row position.
            stop (int): Exclusive global row position.

        Returns:
            np.ndarray: String names in exact stored order; duplicates are retained.
        """

        if isinstance(self._index_object, h5py.Group) and "mask" in self._index_object:
            mask = np.asarray(self._index_object["mask"][start:stop], dtype=bool)
            if np.any(mask):
                raise ValueError(
                    f"obs_names contains missing values in rows [{start}, {stop})."
                )
        return np.asarray(self._values.asstr()[start:stop], dtype=str)

    def close(self) -> None:
        """Close the read-only HDF5 handle.

        Args:
            None.

        Returns:
            None: The file handle is closed.
        """

        self._handle.close()

    def __enter__(self) -> "ObservationNameReader":
        """Enter the observation-name reader context.

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
        """Close the reader when leaving its context.

        Args:
            exc_type (type[BaseException] | None): Raised exception type, if any.
            exc_value (BaseException | None): Raised exception value, if any.
            traceback (Any): Raised exception traceback, if any.

        Returns:
            None: The file handle is closed.
        """

        self.close()


def validate_input_schema(
    input_path: Path,
    feature_names: Sequence[str],
    matrix_key: str,
) -> int:
    """Validate exact checkpoint feature count, names, order, and CSR dimensions.

    Args:
        input_path (Path): Candidate inference H5AD.
        feature_names (Sequence[str]): Ordered checkpoint training feature names.
        matrix_key (str): Frozen scaler matrix key.

    Returns:
        int: Number of input observations.
    """

    expected = tuple(str(value) for value in feature_names)
    observed = read_axis_names(input_path, "var")
    if len(observed) != len(expected):
        raise ValueError(
            f"Input has {len(observed)} features but checkpoint expects {len(expected)}."
        )
    if observed != expected:
        mismatch = next(
            index
            for index, (left, right) in enumerate(zip(observed, expected, strict=True))
            if left != right
        )
        raise ValueError(
            "Input feature names or order differ from checkpoint at "
            f"position {mismatch}: observed {observed[mismatch]!r}, "
            f"expected {expected[mismatch]!r}."
        )
    with h5py.File(input_path, "r") as handle:
        group_path = matrix_group_path(matrix_key)
        if group_path not in handle:
            raise KeyError(f"Input lacks sparse matrix group {group_path!r}.")
        group = handle[group_path]
        encoding = group.attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        if str(encoding) != "csr_matrix":
            raise TypeError(f"{group_path!r} must be CSR encoded.")
        shape = tuple(int(value) for value in group.attrs["shape"])
    if len(shape) != 2 or shape[1] != len(expected):
        raise ValueError("Input sparse matrix dimensions disagree with feature schema.")
    return shape[0]


def validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Require all frozen model, mapping, feature, and preprocessing state.

    Args:
        checkpoint (Mapping[str, Any]): CPU-loaded checkpoint mapping.

    Returns:
        None: Required fields are confirmed or ``KeyError`` is raised.
    """

    required = {
        "model_state",
        "config",
        "epoch",
        "feature_names",
        "scaler_state",
        "target_standardizer_state",
        "slide_mapping_state",
        "patient_mapping_state",
        "target_columns",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {sorted(missing)}")


def load_checkpoint_model(
    checkpoint_path: Path,
    device_override: str | None = None,
) -> tuple[
    IHCMultivariateModel,
    dict[str, Any],
    SparseFeatureScaler,
    TargetStandardizer,
    CategoryMapping,
    CategoryMapping,
    tuple[str, ...],
    tuple[str, ...],
    torch.device,
]:
    """Restore the model and every frozen training transformation.

    Args:
        checkpoint_path (Path): Standalone IHC-MVN checkpoint path.
        device_override (str | None): Optional runtime device specification.

    Returns:
        tuple: Model, config, scaler, target standardizer, slide mapping,
            patient mapping, target names, feature names, and runtime device.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint root must be a mapping.")
    validate_checkpoint(checkpoint)
    config = dict(checkpoint["config"])
    scaler = SparseFeatureScaler.from_state(checkpoint["scaler_state"])
    target_standardizer = TargetStandardizer.from_state(
        checkpoint["target_standardizer_state"]
    )
    slide_mapping = CategoryMapping.from_state(checkpoint["slide_mapping_state"])
    patient_mapping = CategoryMapping.from_state(checkpoint["patient_mapping_state"])
    target_columns = tuple(str(value) for value in checkpoint["target_columns"])
    feature_names = tuple(str(value) for value in checkpoint["feature_names"])
    if scaler.num_features != len(feature_names):
        raise ValueError("Checkpoint scaler and feature-name count disagree.")
    if scaler.feature_names != feature_names:
        raise ValueError("Checkpoint scaler and feature-name order disagree.")
    if scaler.slide_names != slide_mapping.names:
        raise ValueError("Checkpoint scaler and slide mapping disagree.")
    if target_columns != target_standardizer.target_names:
        raise ValueError("Checkpoint target standardizer and target order disagree.")

    model_config = dict(config)
    model_values = dict(model_config.get("model", {}))
    model_values["num_peaks"] = len(feature_names)
    model_config["model"] = model_values
    model = IHCMultivariateModel.from_config(
        model_config,
        num_patients=len(patient_mapping.names),
        num_slides=len(slide_mapping.names),
        num_targets=len(target_columns),
    )
    model.load_state_dict(checkpoint["model_state"])
    requested_device = (
        device_override
        if device_override is not None
        else str(config.get("training", {}).get("device", "auto"))
    )
    device = resolve_device(requested_device)
    model.to(device)
    model.eval()
    return (
        model,
        config,
        scaler,
        target_standardizer,
        slide_mapping,
        patient_mapping,
        target_columns,
        feature_names,
        device,
    )


def create_inference_loader(
    dataset: SparseAnnDataDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Create a deterministic sequential sparse inference DataLoader.

    Args:
        dataset (SparseAnnDataDataset): Frozen-scaler all-row inference dataset.
        batch_size (int): Positive number of rows per batch.
        num_workers (int): Nonnegative number of reader workers.
        pin_memory (bool): Whether to pin collated CPU tensors.
        prefetch_factor (int): Positive worker prefetch count.

    Returns:
        DataLoader[dict[str, torch.Tensor]]: Sequential loader preserving row order.
    """

    if batch_size <= 0 or num_workers < 0 or prefetch_factor <= 0:
        raise ValueError(
            "batch_size and prefetch_factor must be positive; num_workers nonnegative."
        )
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "collate_fn": sparse_collate,
        "pin_memory": bool(pin_memory),
        "persistent_workers": int(num_workers) > 0,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def latent_column_names(latent_dim: int) -> list[str]:
    """Build ordered latent-vector column names.

    Args:
        latent_dim (int): Positive latent vector width.

    Returns:
        list[str]: Names ``latent_000`` through the final dimension.
    """

    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")
    return [f"latent_{index:03d}" for index in range(latent_dim)]


def correlation_column_names(target_columns: Sequence[str]) -> list[str]:
    """Build columns for the shared correlation off-diagonal entries.

    Args:
        target_columns (Sequence[str]): Ordered modeled target names.

    Returns:
        list[str]: Upper-triangle off-diagonal correlation column names.
    """

    targets = tuple(str(value) for value in target_columns)
    return [
        f"R_{targets[left]}_{targets[right]}"
        for left in range(len(targets))
        for right in range(left + 1, len(targets))
    ]


def random_effect_scale_column_names(
    target_columns: Sequence[str],
) -> list[str]:
    """Build ordered random-effect prior-scale column names.

    Args:
        target_columns (Sequence[str]): Ordered modeled target names.

    Returns:
        list[str]: Four prior-scale family columns per target.
    """

    families = (
        "patient_hurdle_prior_scale",
        "patient_mean_prior_scale",
        "slide_hurdle_prior_scale",
        "slide_mean_prior_scale",
    )
    return [
        f"{family}_{target}"
        for family in families
        for target in target_columns
    ]


def prediction_column_names(target_columns: Sequence[str]) -> list[str]:
    """Build the complete ordered prediction column list.

    Args:
        target_columns (Sequence[str]): Ordered modeled target names.

    Returns:
        list[str]: Per-target summaries, shared correlations, prior scales,
            and conditional CD8 diagnostics.
    """

    per_target_metrics = (
        "prob_presence",
        "standardized_mean_msi",
        "standardized_mean_patient",
        "standardized_mean_total",
        "haldane_mean_msi",
        "haldane_mean_patient",
        "haldane_mean_total",
        "marginal_standardized_sigma",
        "marginal_unstandardized_sigma",
        "positive_median_density",
        "q05_density",
        "q95_density",
    )
    names = [
        f"{metric}_{target}"
        for target in target_columns
        for metric in per_target_metrics
    ]
    names.extend(correlation_column_names(target_columns))
    names.extend(random_effect_scale_column_names(target_columns))
    names.extend(
        [
            "conditional_cd8_standardized_mean",
            "conditional_cd8_standardized_variance",
            "conditional_cd8_excess",
        ]
    )
    return names


def build_output_schema(target_columns: Sequence[str]) -> pa.Schema:
    """Build the prediction Parquet Arrow schema.

    Args:
        target_columns (Sequence[str]): Ordered modeled target names.

    Returns:
        pa.Schema: Row identity plus float32 prediction columns.
    """

    fields = [
        pa.field("row_position", pa.int64(), nullable=False),
        pa.field("obs_name", pa.string(), nullable=False),
    ]
    fields.extend(
        pa.field(name, pa.float32(), nullable=True)
        for name in prediction_column_names(target_columns)
    )
    return pa.schema(fields)


def output_schema(target_columns: Sequence[str]) -> pa.Schema:
    """Return the prediction Parquet Arrow schema.

    Args:
        target_columns (Sequence[str]): Ordered modeled target names.

    Returns:
        pa.Schema: Schema produced by ``build_output_schema``.
    """

    return build_output_schema(target_columns)


def build_latent_schema(latent_dim: int) -> pa.Schema:
    """Build the latent-vector Parquet Arrow schema.

    Args:
        latent_dim (int): Positive latent vector width.

    Returns:
        pa.Schema: Row identity plus ordered float32 latent dimensions.
    """

    fields = [
        pa.field("row_position", pa.int64(), nullable=False),
        pa.field("obs_name", pa.string(), nullable=False),
    ]
    fields.extend(
        pa.field(name, pa.float32(), nullable=False)
        for name in latent_column_names(latent_dim)
    )
    return pa.schema(fields)


def latent_output_schema(latent_dim: int) -> pa.Schema:
    """Return the latent-vector Parquet Arrow schema.

    Args:
        latent_dim (int): Positive latent vector width.

    Returns:
        pa.Schema: Schema produced by ``build_latent_schema``.
    """

    return build_latent_schema(latent_dim)


def random_effect_prior_scales(
    model: IHCMultivariateModel,
) -> dict[str, torch.Tensor]:
    """Extract learned positive random-effect prior scales.

    Args:
        model (IHCMultivariateModel): Loaded standalone model.

    Returns:
        dict[str, torch.Tensor]: Four standardized per-target scale vectors.
    """

    effects = model.random_intercepts
    return {
        "patient_hurdle_prior_scale": (
            F.softplus(effects.patient_hurdle_scale_raw) + effects.scale_floor
        ),
        "patient_mean_prior_scale": (
            F.softplus(effects.patient_mean_scale_raw) + effects.scale_floor
        ),
        "slide_hurdle_prior_scale": (
            F.softplus(effects.slide_hurdle_scale_raw) + effects.scale_floor
        ),
        "slide_mean_prior_scale": (
            F.softplus(effects.slide_mean_scale_raw) + effects.scale_floor
        ),
    }


def predictive_covariance(
    predictions: Mapping[str, torch.Tensor],
    patient_codes: torch.Tensor,
    slide_codes: torch.Tensor,
    prior_scales: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Add unknown-group random-mean prior variance to model covariance.

    Args:
        predictions (Mapping[str, torch.Tensor]): Model outputs with base covariance.
        patient_codes (torch.Tensor): Patient codes, where ``-1`` means unknown.
        slide_codes (torch.Tensor): Slide codes, where ``-1`` means unknown.
        prior_scales (Mapping[str, torch.Tensor]): Learned standardized RE scales.

    Returns:
        torch.Tensor: Predictive standardized covariance for each row.
    """

    covariance = predictions["covariance"].clone()
    diagonal_extra = torch.zeros_like(predictions["scales"])
    diagonal_extra = diagonal_extra + (patient_codes < 0).to(
        diagonal_extra.dtype
    ).unsqueeze(-1) * prior_scales["patient_mean_prior_scale"].square()
    diagonal_extra = diagonal_extra + (slide_codes < 0).to(
        diagonal_extra.dtype
    ).unsqueeze(-1) * prior_scales["slide_mean_prior_scale"].square()
    diagonal_indices = torch.arange(
        covariance.shape[-1], device=covariance.device
    )
    covariance[:, diagonal_indices, diagonal_indices] += diagonal_extra
    return covariance


def _sigmoid_numpy(values: np.ndarray) -> np.ndarray:
    """Evaluate the logistic sigmoid stably on NumPy values.

    Args:
        values (np.ndarray): Real-valued input array.

    Returns:
        np.ndarray: Logistic probabilities with matching shape.
    """

    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    nonnegative = array >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-array[nonnegative]))
    exponential = np.exp(array[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def build_prediction_arrays(
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    target_standardizer: TargetStandardizer,
    target_columns: Sequence[str],
    model: IHCMultivariateModel,
) -> dict[str, np.ndarray]:
    """Derive all exported prediction arrays from one model batch.

    Args:
        predictions (Mapping[str, torch.Tensor]): Hierarchical model outputs.
        batch (Mapping[str, torch.Tensor]): Device-resident inference batch.
        target_standardizer (TargetStandardizer): Frozen target affine transform.
        target_columns (Sequence[str]): Ordered modeled target names.
        model (IHCMultivariateModel): Model supplying learned RE prior scales.

    Returns:
        dict[str, np.ndarray]: Float32 arrays keyed by output column name.
    """

    targets = tuple(str(value) for value in target_columns)
    if targets != target_standardizer.target_names:
        raise ValueError("Target order differs from frozen target standardizer.")
    for key in (
        "hurdle_logits_total",
        "mean_msi",
        "mean_patient",
        "mean_total",
        "scales",
        "correlation",
        "covariance",
    ):
        if key not in predictions:
            raise KeyError(f"Model predictions lack required key {key!r}.")
        value = predictions[key]
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Prediction {key!r} contains non-finite values.")

    prior_scales = random_effect_prior_scales(model)
    covariance = predictive_covariance(
        predictions,
        batch["patient_codes"],
        batch["slide_codes"],
        prior_scales,
    )
    if not torch.isfinite(covariance).all():
        raise FloatingPointError("Predictive covariance contains non-finite values.")

    probabilities = torch.sigmoid(predictions["hurdle_logits_total"])
    standardized_means = {
        "standardized_mean_msi": predictions["mean_msi"],
        "standardized_mean_patient": predictions["mean_patient"],
        "standardized_mean_total": predictions["mean_total"],
    }
    means_np = {
        name: tensor.detach().cpu().numpy().astype(np.float64)
        for name, tensor in standardized_means.items()
    }
    target_means = target_standardizer.means.reshape(1, -1)
    target_scales = target_standardizer.scales.reshape(1, -1)
    haldane_means = {
        name.replace("standardized_", "haldane_"): values * target_scales
        + target_means
        for name, values in means_np.items()
    }
    predictive_sigma = torch.sqrt(
        covariance.diagonal(dim1=-2, dim2=-1)
    ).detach().cpu().numpy().astype(np.float64)
    unstandardized_sigma = predictive_sigma * target_scales
    total_haldane = haldane_means["haldane_mean_total"]

    coordinates = batch.get("targets")
    if coordinates is None:
        coordinates = torch.full_like(predictions["mean_total"], torch.nan)
    conditional = conditional_gaussian_cd8(
        coordinates=coordinates,
        means=predictions["mean_total"],
        covariance=covariance,
    )

    arrays: dict[str, np.ndarray] = {}
    probability_np = probabilities.detach().cpu().numpy()
    correlation_np = predictions["correlation"].detach().cpu().numpy()
    for target_index, target in enumerate(targets):
        arrays[f"prob_presence_{target}"] = probability_np[:, target_index]
        for name, values in means_np.items():
            arrays[f"{name}_{target}"] = values[:, target_index]
        for name, values in haldane_means.items():
            arrays[f"{name}_{target}"] = values[:, target_index]
        arrays[f"marginal_standardized_sigma_{target}"] = predictive_sigma[
            :, target_index
        ]
        arrays[f"marginal_unstandardized_sigma_{target}"] = (
            unstandardized_sigma[:, target_index]
        )
        arrays[f"positive_median_density_{target}"] = _sigmoid_numpy(
            total_haldane[:, target_index]
        )
        arrays[f"q05_density_{target}"] = _sigmoid_numpy(
            total_haldane[:, target_index]
            + NORMAL_Q05 * unstandardized_sigma[:, target_index]
        )
        arrays[f"q95_density_{target}"] = _sigmoid_numpy(
            total_haldane[:, target_index]
            + NORMAL_Q95 * unstandardized_sigma[:, target_index]
        )
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            arrays[f"R_{targets[left]}_{targets[right]}"] = np.full(
                probability_np.shape[0],
                correlation_np[left, right],
                dtype=np.float32,
            )
    for family, values in prior_scales.items():
        values_np = values.detach().cpu().numpy()
        for target_index, target in enumerate(targets):
            arrays[f"{family}_{target}"] = np.full(
                probability_np.shape[0],
                values_np[target_index],
                dtype=np.float32,
            )
    arrays["conditional_cd8_standardized_mean"] = (
        conditional.conditional_mean.detach().cpu().numpy()
    )
    arrays["conditional_cd8_standardized_variance"] = (
        conditional.conditional_variance.detach().cpu().numpy()
    )
    arrays["conditional_cd8_excess"] = conditional.excess.detach().cpu().numpy()
    return {
        name: np.asarray(values, dtype=np.float32)
        for name, values in arrays.items()
    }


def build_output_table(
    row_positions: np.ndarray,
    obs_names: np.ndarray,
    prediction_arrays: Mapping[str, np.ndarray],
    target_columns: Sequence[str],
) -> pa.Table:
    """Build one prediction Arrow table in global row order.

    Args:
        row_positions (np.ndarray): Global int64 row positions.
        obs_names (np.ndarray): Matching observation names; duplicates are allowed.
        prediction_arrays (Mapping[str, np.ndarray]): Derived float prediction vectors.
        target_columns (Sequence[str]): Ordered modeled target names.

    Returns:
        pa.Table: Batch table conforming exactly to the output schema.
    """

    rows = np.asarray(row_positions, dtype=np.int64)
    names = np.asarray(obs_names, dtype=str)
    if rows.ndim != 1 or names.ndim != 1 or rows.size != names.size:
        raise ValueError("row_positions and obs_names must be matching 1D arrays.")
    expected_columns = prediction_column_names(target_columns)
    if set(prediction_arrays) != set(expected_columns):
        missing = sorted(set(expected_columns).difference(prediction_arrays))
        extra = sorted(set(prediction_arrays).difference(expected_columns))
        raise ValueError(
            f"Prediction columns disagree with schema; missing={missing}, extra={extra}."
        )
    arrays: list[pa.Array] = [
        pa.array(rows, type=pa.int64()),
        pa.array(names, type=pa.string()),
    ]
    for column in expected_columns:
        values = np.asarray(prediction_arrays[column], dtype=np.float32)
        if values.shape != rows.shape:
            raise ValueError(f"Prediction column {column!r} has an invalid shape.")
        arrays.append(pa.array(values, type=pa.float32(), from_pandas=True))
    return pa.Table.from_arrays(arrays, schema=build_output_schema(target_columns))


def build_latent_output_table(
    row_positions: np.ndarray,
    obs_names: np.ndarray,
    latent: np.ndarray,
) -> pa.Table:
    """Build one ordered latent-vector Arrow table.

    Args:
        row_positions (np.ndarray): Global int64 row positions.
        obs_names (np.ndarray): Matching observation names; duplicates are allowed.
        latent (np.ndarray): Float latent matrix with shape ``[B, D]``.

    Returns:
        pa.Table: Batch table conforming exactly to the latent schema.
    """

    rows = np.asarray(row_positions, dtype=np.int64)
    names = np.asarray(obs_names, dtype=str)
    values = np.asarray(latent, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != rows.size or names.size != rows.size:
        raise ValueError("Latent rows, row positions, and names must have equal length.")
    arrays: list[pa.Array] = [
        pa.array(rows, type=pa.int64()),
        pa.array(names, type=pa.string()),
    ]
    arrays.extend(
        pa.array(values[:, index], type=pa.float32())
        for index in range(values.shape[1])
    )
    return pa.Table.from_arrays(arrays, schema=build_latent_schema(values.shape[1]))


def validate_output(
    output_path: Path,
    input_path: Path,
    target_columns: Sequence[str],
    expected_rows: int,
    batch_size: int = 65_536,
) -> dict[str, Any]:
    """Validate prediction schema, row order, names, finiteness, and quantiles.

    Args:
        output_path (Path): Prediction Parquet path.
        input_path (Path): Source H5AD used for observation-name comparison.
        target_columns (Sequence[str]): Ordered modeled target names.
        expected_rows (int): Expected output row count.
        batch_size (int): Positive validation scan batch size.

    Returns:
        dict[str, Any]: Validated row and prediction-column counts.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    parquet = pq.ParquetFile(output_path)
    expected_schema = build_output_schema(target_columns)
    if not parquet.schema_arrow.equals(expected_schema):
        raise ValueError("Prediction Parquet schema differs from expected schema.")
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"Prediction Parquet has {parquet.metadata.num_rows} rows; "
            f"expected {expected_rows}."
        )
    float_columns = prediction_column_names(target_columns)
    nullable = {"conditional_cd8_excess"}
    processed = 0
    with ObservationNameReader(input_path) as name_reader:
        for record_batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([record_batch])
            rows = table["row_position"].to_numpy()
            expected = np.arange(processed, processed + len(rows), dtype=np.int64)
            if not np.array_equal(rows, expected):
                raise ValueError("Prediction row positions are not sequential.")
            names = np.asarray(table["obs_name"].to_pylist(), dtype=str)
            if not np.array_equal(
                names, name_reader.read(processed, processed + len(rows))
            ):
                raise ValueError("Prediction observation names are out of order.")
            for column in float_columns:
                values = table[column].to_numpy(
                    zero_copy_only=False
                ).astype(np.float64)
                if column in nullable:
                    if np.isinf(values).any():
                        raise ValueError(f"Column {column!r} contains infinite values.")
                elif not np.isfinite(values).all():
                    raise ValueError(f"Column {column!r} contains non-finite values.")
            for target in target_columns:
                probability = table[f"prob_presence_{target}"].to_numpy()
                lower = table[f"q05_density_{target}"].to_numpy()
                median = table[f"positive_median_density_{target}"].to_numpy()
                upper = table[f"q95_density_{target}"].to_numpy()
                if np.any((probability < 0.0) | (probability > 1.0)):
                    raise ValueError(f"Presence probability for {target!r} is invalid.")
                if np.any(
                    (lower < 0.0)
                    | (lower > median)
                    | (median > upper)
                    | (upper > 1.0)
                ):
                    raise ValueError(f"Density quantiles for {target!r} are invalid.")
            processed += len(rows)
    if processed != expected_rows:
        raise ValueError("Prediction validation did not scan the expected row count.")
    return {"rows": processed, "prediction_columns": len(float_columns)}


def validate_latent_output(
    output_path: Path,
    input_path: Path,
    latent_dim: int,
    expected_rows: int,
    batch_size: int = 65_536,
) -> dict[str, Any]:
    """Validate latent schema, global row order, names, and finite values.

    Args:
        output_path (Path): Latent Parquet path.
        input_path (Path): Source H5AD used for observation-name comparison.
        latent_dim (int): Expected latent vector width.
        expected_rows (int): Expected output row count.
        batch_size (int): Positive validation scan batch size.

    Returns:
        dict[str, Any]: Validated row count and latent dimension.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    parquet = pq.ParquetFile(output_path)
    if not parquet.schema_arrow.equals(build_latent_schema(latent_dim)):
        raise ValueError("Latent Parquet schema differs from expected schema.")
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"Latent Parquet has {parquet.metadata.num_rows} rows; "
            f"expected {expected_rows}."
        )
    latent_columns = latent_column_names(latent_dim)
    processed = 0
    with ObservationNameReader(input_path) as name_reader:
        for record_batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([record_batch])
            rows = table["row_position"].to_numpy()
            expected = np.arange(processed, processed + len(rows), dtype=np.int64)
            if not np.array_equal(rows, expected):
                raise ValueError("Latent row positions are not sequential.")
            names = np.asarray(table["obs_name"].to_pylist(), dtype=str)
            if not np.array_equal(
                names, name_reader.read(processed, processed + len(rows))
            ):
                raise ValueError("Latent observation names are out of order.")
            values = np.column_stack(
                [table[column].to_numpy() for column in latent_columns]
            )
            if not np.isfinite(values).all():
                raise ValueError("Latent Parquet contains non-finite values.")
            processed += len(rows)
    if processed != expected_rows:
        raise ValueError("Latent validation did not scan the expected row count.")
    return {"rows": processed, "latent_dim": latent_dim}


def _default_paths(
    config: Mapping[str, Any],
    input_override: Path | None,
    output_override: Path | None,
    latent_override: Path | None,
) -> tuple[Path, Path, Path]:
    """Resolve input and output paths from checkpoint config and CLI overrides.

    Args:
        config (Mapping[str, Any]): Stored checkpoint configuration.
        input_override (Path | None): Optional CLI input path.
        output_override (Path | None): Optional CLI prediction path.
        latent_override (Path | None): Optional CLI latent path.

    Returns:
        tuple[Path, Path, Path]: Input, prediction output, and latent output paths.
    """

    data = config.get("data", {})
    output = config.get("output", {})
    if input_override is None:
        configured_input = data.get("inference_path", data.get("path"))
        if configured_input is None:
            raise KeyError("Checkpoint config has no data.inference_path or data.path.")
        input_path = Path(str(configured_input))
    else:
        input_path = Path(input_override)
    output_dir = Path(
        str(
            config.get("inference", {}).get(
                "output_dir",
                output.get("directory", Path.cwd()),
            )
        )
    )
    output_path = (
        Path(output_override)
        if output_override is not None
        else output_dir / "inference.parquet"
    )
    latent_path = (
        Path(latent_override)
        if latent_override is not None
        else output_dir / "inference_latent.parquet"
    )
    return input_path, output_path, latent_path


def run_inference(
    checkpoint_path: Path,
    input_path: Path | None = None,
    output_path: Path | None = None,
    latent_output_path: Path | None = None,
    device_override: str | None = None,
    batch_size_override: int | None = None,
    num_workers_override: int | None = None,
    max_rows: int | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Run ordered frozen-scaler inference and atomically stream two Parquets.

    Args:
        checkpoint_path (Path): Standalone IHC-MVN checkpoint.
        input_path (Path | None): H5AD override; configured tissue H5AD by default.
        output_path (Path | None): Prediction Parquet override.
        latent_output_path (Path | None): Latent Parquet override.
        device_override (str | None): Runtime device override.
        batch_size_override (int | None): Inference batch-size override.
        num_workers_override (int | None): DataLoader worker-count override.
        max_rows (int | None): Optional positive leading-row cap.
        overwrite (bool): Whether existing final outputs may be replaced.

    Returns:
        tuple[Path, Path]: Validated prediction and latent Parquet paths.
    """

    checkpoint_path = Path(checkpoint_path)
    (
        model,
        config,
        scaler,
        target_standardizer,
        slide_mapping,
        patient_mapping,
        target_columns,
        feature_names,
        device,
    ) = load_checkpoint_model(checkpoint_path, device_override)
    input_path, output_path, latent_output_path = _default_paths(
        config,
        input_path,
        output_path,
        latent_output_path,
    )
    if output_path.resolve() == latent_output_path.resolve():
        raise ValueError("Prediction and latent output paths must differ.")
    for candidate in (output_path, latent_output_path):
        if candidate.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {candidate}")

    total_rows = validate_input_schema(input_path, feature_names, scaler.matrix_key)
    if max_rows is not None and int(max_rows) <= 0:
        raise ValueError("max_rows must be positive when provided.")
    selected_rows = (
        total_rows if max_rows is None else min(total_rows, int(max_rows))
    )
    if selected_rows <= 0:
        raise ValueError("Inference input selection is empty.")
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    inference_config = config.get("inference", {})
    batch_size = int(
        batch_size_override
        if batch_size_override is not None
        else inference_config.get(
            "batch_size",
            training_config.get("test_batch_size", 4096),
        )
    )
    num_workers = int(
        num_workers_override
        if num_workers_override is not None
        else inference_config.get(
            "num_workers",
            training_config.get("num_workers", 0),
        )
    )
    prefetch_factor = int(training_config.get("prefetch_factor", 2))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative.")
    seed_everything(int(training_config.get("seed", 0)), deterministic=False)

    dataset = create_inference_dataset(
        path=input_path,
        scaler=scaler,
        patient_mapping=patient_mapping,
        batch_column=str(data_config.get("batch_column", slide_mapping.column)),
        patient_column=str(data_config.get("patient_column", patient_mapping.column)),
        target_columns=target_columns,
        target_standardizer=target_standardizer,
    )
    if dataset.metadata.slide_mapping.names != slide_mapping.names:
        raise ValueError("Frozen slide mapping was not preserved for inference.")
    dataset.indices = dataset.indices[:selected_rows]
    loader = create_inference_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=bool(training_config.get("pin_memory", False))
        and device.type == "cuda",
        prefetch_factor=prefetch_factor,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    latent_output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_latent = latent_output_path.with_suffix(
        latent_output_path.suffix + ".tmp"
    )
    for temporary in (temporary_output, temporary_latent):
        if temporary.exists():
            temporary.unlink()
    prediction_writer: pq.ParquetWriter | None = None
    latent_writer: pq.ParquetWriter | None = None
    processed = 0
    try:
        prediction_writer = pq.ParquetWriter(
            temporary_output,
            build_output_schema(target_columns),
            compression="zstd",
            use_dictionary=["obs_name"],
        )
        latent_writer = pq.ParquetWriter(
            temporary_latent,
            build_latent_schema(model.latent_dim),
            compression="zstd",
            use_dictionary=["obs_name"],
        )
        with ObservationNameReader(input_path) as name_reader, torch.inference_mode():
            for cpu_batch in loader:
                row_positions = cpu_batch["row_ids"].numpy()
                expected_positions = np.arange(
                    processed,
                    processed + row_positions.size,
                    dtype=np.int64,
                )
                if not np.array_equal(row_positions, expected_positions):
                    raise RuntimeError("Inference DataLoader changed global row order.")
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in cpu_batch.items()
                }
                predictions = model(batch)
                arrays = build_prediction_arrays(
                    predictions,
                    batch,
                    target_standardizer,
                    target_columns,
                    model,
                )
                obs_names = name_reader.read(
                    processed, processed + row_positions.size
                )
                prediction_writer.write_table(
                    build_output_table(
                        row_positions,
                        obs_names,
                        arrays,
                        target_columns,
                    ),
                    row_group_size=row_positions.size,
                )
                latent_writer.write_table(
                    build_latent_output_table(
                        row_positions,
                        obs_names,
                        predictions["latent"].detach().cpu().numpy(),
                    ),
                    row_group_size=row_positions.size,
                )
                processed += row_positions.size
                if processed == selected_rows or processed % (25 * batch_size) == 0:
                    print(
                        f"Processed {processed:,}/{selected_rows:,} rows.",
                        flush=True,
                    )
        if processed != selected_rows:
            raise RuntimeError(
                f"Inference processed {processed} rows but expected {selected_rows}."
            )
        prediction_writer.close()
        prediction_writer = None
        latent_writer.close()
        latent_writer = None
        validate_output(
            temporary_output,
            input_path,
            target_columns,
            selected_rows,
        )
        validate_latent_output(
            temporary_latent,
            input_path,
            model.latent_dim,
            selected_rows,
        )
        temporary_output.replace(output_path)
        temporary_latent.replace(latent_output_path)
    except BaseException:
        if prediction_writer is not None:
            prediction_writer.close()
        if latent_writer is not None:
            latent_writer.close()
        for temporary in (temporary_output, temporary_latent):
            if temporary.exists():
                temporary.unlink()
        raise
    finally:
        dataset.close()
    return output_path, latent_output_path


def main(argv: Sequence[str] | None = None) -> int:
    """Load a checkpoint and execute streaming standalone inference.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        int: Zero process status after successful inference.
    """

    args = parse_args(argv)
    output_path, latent_path = run_inference(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        output_path=args.output,
        latent_output_path=args.latent_output,
        device_override=args.device,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
    )
    print(f"Predictions: {output_path}", flush=True)
    print(f"Latents: {latent_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Configuration loading, validation, reproducibility, and device helpers."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

EXPECTED_TARGETS = [
    "Density_Tumor",
    "Density_Stroma",
    "Density_Collagen",
    "Density_CD8",
]


def _require_keys(section: Mapping[str, Any], keys: set[str], name: str) -> None:
    """Require configuration keys in one section.

    Args:
        section (Mapping[str, Any]): Configuration section to inspect.
        keys (set[str]): Required key names.
        name (str): Human-readable section name.

    Returns:
        None: The function returns after successful validation.
    """

    missing = keys.difference(section)
    if missing:
        raise KeyError(f"Missing keys in configuration section {name!r}: {sorted(missing)}")


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the complete standalone IHC-MVN configuration.

    Args:
        config (Mapping[str, Any]): Parsed configuration mapping.

    Returns:
        None: The function returns after successful validation.
    """

    required_sections = {
        "data",
        "preprocessing",
        "targets",
        "model",
        "loss",
        "training",
        "output",
        "inference",
    }
    missing_sections = required_sections.difference(config)
    if missing_sections:
        raise KeyError(f"Missing configuration sections: {sorted(missing_sections)}")

    data = config["data"]
    preprocessing = config["preprocessing"]
    targets = config["targets"]
    model = config["model"]
    loss = config["loss"]
    training = config["training"]
    output = config["output"]
    inference = config["inference"]
    for section_name, section in (
        ("data", data),
        ("preprocessing", preprocessing),
        ("targets", targets),
        ("model", model),
        ("loss", loss),
        ("training", training),
        ("output", output),
        ("inference", inference),
    ):
        if not isinstance(section, Mapping):
            raise TypeError(f"Configuration section {section_name!r} must be a mapping.")

    _require_keys(
        data,
        {
            "path",
            "inference_path",
            "matrix_key",
            "batch_column",
            "patient_column",
            "target_columns",
            "split_strategy",
            "train_fraction",
            "validation_fraction",
            "test_fraction",
        },
        "data",
    )
    _require_keys(
        preprocessing,
        {"transform", "center", "scale_floor", "fit_row_chunk_size"},
        "preprocessing",
    )
    _require_keys(
        targets,
        {"total_count", "haldane_correction", "standard_deviation_floor"},
        "targets",
    )
    _require_keys(
        model,
        {
            "num_peaks",
            "embedding_dim",
            "peak_hidden_dims",
            "peak_output_dim",
            "aggregation_hidden_dims",
            "latent_dim",
            "head_hidden_dims",
            "activation",
            "dropout",
            "use_layer_norm",
            "inference_peak_chunk_size",
            "sigma_min",
            "correlation_diagonal_floor",
            "random_effects",
        },
        "model",
    )
    _require_keys(loss, {"prior_weight"}, "loss")
    _require_keys(
        training,
        {
            "seed",
            "device",
            "deterministic_algorithms",
            "batch_size",
            "validation_batch_size",
            "test_batch_size",
            "num_workers",
            "prefetch_factor",
            "pin_memory",
            "smoke_train_samples",
            "smoke_validation_samples",
            "smoke_test_samples",
            "smoke_num_workers",
            "smoke_epochs",
            "epochs",
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "gradient_clip_norm",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "resume_checkpoint",
        },
        "training",
    )
    _require_keys(output, {"directory"}, "output")
    _require_keys(inference, {"batch_size", "num_workers", "output_dir"}, "inference")

    if list(data["target_columns"]) != EXPECTED_TARGETS:
        raise ValueError(
            "data.target_columns must be exactly "
            f"{EXPECTED_TARGETS!r} in this order."
        )
    if str(data["matrix_key"]) != "layers/counts":
        raise ValueError("data.matrix_key must be 'layers/counts'.")
    if str(data["batch_column"]) != "batch":
        raise ValueError("data.batch_column must be 'batch'.")
    if str(data["patient_column"]) != "patient":
        raise ValueError("data.patient_column must be 'patient'.")
    if str(data["split_strategy"]) != "random_within_slide":
        raise ValueError("Only data.split_strategy='random_within_slide' is supported.")

    fractions = np.asarray(
        [
            data["train_fraction"],
            data["validation_fraction"],
            data["test_fraction"],
        ],
        dtype=np.float64,
    )
    if np.any(fractions < 0.0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError(
            f"Split fractions must be nonnegative and sum to one, observed {fractions}."
        )
    if not np.allclose(fractions, np.asarray([0.8, 0.1, 0.1])):
        raise ValueError("The required split is random within-slide 80/10/10.")

    for path_key in ("path", "inference_path"):
        if not Path(str(data[path_key])).is_absolute():
            raise ValueError(f"data.{path_key} must be an absolute path.")
    if not Path(str(output["directory"])).is_absolute():
        raise ValueError("output.directory must be an absolute path.")

    if str(preprocessing["transform"]) != "log1p":
        raise ValueError("preprocessing.transform must be 'log1p'.")
    if bool(preprocessing["center"]):
        raise ValueError("preprocessing.center must be false.")
    if float(preprocessing["scale_floor"]) <= 0.0:
        raise ValueError("preprocessing.scale_floor must be positive.")
    if int(preprocessing["fit_row_chunk_size"]) <= 0:
        raise ValueError("preprocessing.fit_row_chunk_size must be positive.")

    if int(targets["total_count"]) != 36_100:
        raise ValueError("targets.total_count must be 36100.")
    if float(targets["haldane_correction"]) <= 0.0:
        raise ValueError("targets.haldane_correction must be positive.")
    if float(targets["standard_deviation_floor"]) <= 0.0:
        raise ValueError("targets.standard_deviation_floor must be positive.")

    positive_model_keys = (
        "num_peaks",
        "embedding_dim",
        "peak_output_dim",
        "latent_dim",
        "inference_peak_chunk_size",
    )
    for key in positive_model_keys:
        if int(model[key]) <= 0:
            raise ValueError(f"model.{key} must be positive.")
    for key in ("peak_hidden_dims", "aggregation_hidden_dims", "head_hidden_dims"):
        if any(int(width) <= 0 for width in model[key]):
            raise ValueError(f"Every model.{key} width must be positive.")
    if not 0.0 <= float(model["dropout"]) < 1.0:
        raise ValueError("model.dropout must lie in [0, 1).")
    if float(model["sigma_min"]) <= 0.0:
        raise ValueError("model.sigma_min must be positive.")
    random_effects = model["random_effects"]
    if not isinstance(random_effects, Mapping):
        raise TypeError("model.random_effects must be a mapping.")
    _require_keys(
        random_effects,
        {"initial_scale", "scale_floor", "scale_prior_std"},
        "model.random_effects",
    )
    if float(random_effects["initial_scale"]) <= float(random_effects["scale_floor"]):
        raise ValueError("random-effect initial_scale must exceed scale_floor.")
    if float(random_effects["scale_floor"]) <= 0.0:
        raise ValueError("random-effect scale_floor must be positive.")
    if float(random_effects["scale_prior_std"]) <= 0.0:
        raise ValueError("random-effect scale_prior_std must be positive.")
    if float(loss["prior_weight"]) < 0.0:
        raise ValueError("loss.prior_weight must be nonnegative.")

    for key in ("batch_size", "validation_batch_size", "test_batch_size"):
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive.")
    if int(training["num_workers"]) < 0:
        raise ValueError("training.num_workers cannot be negative.")
    if int(training["prefetch_factor"]) <= 0:
        raise ValueError("training.prefetch_factor must be positive.")
    if int(training["epochs"]) <= 0 or int(training["smoke_epochs"]) <= 0:
        raise ValueError("training epoch counts must be positive.")
    if float(training["learning_rate"]) <= 0.0:
        raise ValueError("training.learning_rate must be positive.")
    if float(training["weight_decay"]) < 0.0:
        raise ValueError("training.weight_decay must be nonnegative.")
    if int(training["early_stopping_patience"]) <= 0:
        raise ValueError("training.early_stopping_patience must be positive.")
    if int(inference["batch_size"]) <= 0 or int(inference["num_workers"]) < 0:
        raise ValueError(
            "inference.batch_size must be positive and num_workers nonnegative."
        )
    if not Path(str(inference["output_dir"])).is_absolute():
        raise ValueError("inference.output_dir must be an absolute path.")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Args:
        path (str | Path): YAML configuration path.

    Returns:
        dict[str, Any]: Mutable validated configuration dictionary.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError("The YAML root must be a mapping.")
    validate_config(loaded)
    return loaded


def apply_smoke_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied configuration with small real-data caps.

    Args:
        config (Mapping[str, Any]): Complete validated configuration.

    Returns:
        dict[str, Any]: Configuration containing smoke-run overrides.
    """

    validate_config(config)
    updated = copy.deepcopy(dict(config))
    data = updated["data"]
    training = updated["training"]
    data["max_train_samples"] = int(training["smoke_train_samples"])
    data["max_validation_samples"] = int(training["smoke_validation_samples"])
    data["max_test_samples"] = int(training["smoke_test_samples"])
    training["num_workers"] = int(training["smoke_num_workers"])
    training["epochs"] = int(training["smoke_epochs"])
    updated["output"]["directory"] = str(
        Path(updated["output"]["directory"]) / "smoke"
    )
    validate_config(updated)
    return updated


def resolve_device(requested: str) -> torch.device:
    """Resolve and validate a configured PyTorch runtime device.

    Args:
        requested (str): Device specification or ``"auto"``.

    Returns:
        torch.device: Available resolved PyTorch device.
    """

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested but CUDA is unavailable.")
    if device.type == "cuda" and device.index is not None:
        if device.index < 0 or device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)."
            )
    return device


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    Args:
        seed (int): Reproducibility seed.
        deterministic (bool): Whether to request deterministic PyTorch algorithms.

    Returns:
        None: Global random state is updated in place.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

"""Configuration loading and runtime helpers for the DANN pipeline."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from dann.model_components.layers_gatv2 import gatv2_settings, graph_reach
from dann.targets import normalization_settings, target_contract


def density_sampling_settings(analysis: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve and validate Monte Carlo settings, including legacy defaults."""
    count = analysis.get("density_mc_samples", 1000)
    seed = analysis.get("density_mc_seed", 20260719)
    for name, value, minimum in (
        ("density_mc_samples", count, 1), ("density_mc_seed", seed, 0)
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"analysis.{name} must be an integer >= {minimum}.")
    return count, seed


def latent_variance_settings(settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve standalone latent-variance defaults and validate their types."""
    root = "/workspaces/multimodal-pdac/data/PDAC"
    defaults = {
        "input": f"{root}/Raw/adata_assembled_tissue.h5ad",
        "latents": f"{root}/Results/dann/analysis/spatial/spatial_latent.parquet",
        "output_dir": f"{root}/Results/dann/latent_variance",
        "slide_column": "batch",
        "chunk_size": 8192,
        "figure_dpi": 300,
    }
    if settings is not None and not isinstance(settings, Mapping):
        raise ValueError("latent_variance must be a mapping.")
    resolved = {**defaults, **(settings or {})}
    for key in ("input", "latents", "output_dir", "slide_column"):
        if key == "input" and resolved[key] is None:
            continue  # Training configurations need no independent inference input.
        if not isinstance(resolved[key], str) or not resolved[key].strip():
            raise ValueError(f"latent_variance.{key} must be a nonempty string.")
    for key in ("chunk_size", "figure_dpi"):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"latent_variance.{key} must be a positive integer.")
    return resolved


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate a YAML configuration.

    Args:
        path (Path): YAML configuration path.

    Returns:
        dict[str, Any]: Mutable nested configuration dictionary.
    """

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required_sections = {"data", "model", "loss", "training", "analysis"}
    missing = required_sections.difference(config)
    if missing:
        raise KeyError(f"Missing configuration sections: {sorted(missing)}")
    fractions = [
        float(config["data"][f"{name}_fraction"])
        for name in ("train", "validation", "test")
    ]
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Data split fractions must sum to one, observed {fractions}.")
    if any(value < 0.0 for value in fractions):
        raise ValueError("Data split fractions cannot be negative.")
    max_samples = config["analysis"].get("max_samples")
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples <= 0
    ):
        raise ValueError("analysis.max_samples must be null or a positive integer.")
    count, seed = density_sampling_settings(config["analysis"])
    config["analysis"].update(density_mc_samples=count, density_mc_seed=seed)
    config["latent_variance"] = latent_variance_settings(config.get("latent_variance"))
    config["data"]["cd8_normalization"] = normalization_settings(config["data"].get("cd8_normalization"))
    target_contract(config)
    return resolve_execution(config)


def apply_smoke_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create a configuration capped for a fast real-data smoke run.

    Args:
        config (Mapping[str, Any]): Complete user configuration.

    Returns:
        dict[str, Any]: Deep copy containing smoke-test overrides.
    """

    updated = copy.deepcopy(dict(config))
    data = updated["data"]
    training = updated["training"]
    data["max_train_samples"] = int(training["smoke_train_samples"])
    data["max_validation_samples"] = int(training["smoke_validation_samples"])
    data["max_test_samples"] = int(training["smoke_test_samples"])
    training["epochs"] = int(training["smoke_epochs"])
    training["num_workers"] = int(training["smoke_num_workers"])
    training["early_stopping_patience"] = max(int(training["epochs"]), 1)
    updated["analysis"]["max_samples"] = int(training["smoke_test_samples"])
    updated["analysis"]["num_workers"] = int(training["smoke_num_workers"])
    updated["analysis"]["activity_max_nonzeros"] = 5_000_000
    if "results" in updated:
        updated["results"]["smoke"] = True
        mode = execution_settings(updated)["mode"]
        if mode in training:
            training[mode]["num_workers"] = training["smoke_num_workers"]
        if mode == "spatial":
            training["spatial"].update(core_size=training.get("smoke_core_size", 8),
                                      tiles_per_batch=training.get("smoke_tiles_per_batch", 2))
        resolve_execution(updated)
    else:
        output_dir = Path(training["output_dir"]) / "smoke"
        training["output_dir"] = str(output_dir)
        updated["analysis"]["output_dir"] = str(output_dir / "analysis")
        updated["analysis"]["checkpoint"] = str(output_dir / "best.pt")
    return updated


def resolve_device(requested: str) -> torch.device:
    """Resolve a configured runtime device.

    Args:
        requested (str): Device string or ``"auto"``.

    Returns:
        torch.device: Available PyTorch device.
    """

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested but CUDA is unavailable.")
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


def component_settings(model: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve flat legacy architecture as all-MLP, or validate explicit selectors."""
    if "spectral_encoder" not in model:
        settings = {
            "spectral_encoder": {"type": "deep_sets", "deep_sets": {
                key: model[key] for key in ("embedding_dim", "peak_hidden_dims", "peak_output_dim")}},
            "aggregation": {"type": "mlp", "mlp": {"hidden_dims": model["aggregation_hidden_dims"]}},
            "heads": {name: {"type": "mlp", "mlp": {"hidden_dims": model[f"{name}_hidden_dims"]}}
                      for name in ("biology", "discriminator")},
        }
    else:
        flat = {"embedding_dim", "peak_hidden_dims", "peak_output_dim", "aggregation_hidden_dims",
                "biology_hidden_dims", "discriminator_hidden_dims"}.intersection(model)
        if flat:
            raise ValueError(f"Cannot mix legacy flat and component settings: {sorted(flat)}")
        settings = copy.deepcopy({key: model[key] for key in ("spectral_encoder", "aggregation", "heads")})
    if settings["spectral_encoder"]["type"] != "deep_sets":
        raise ValueError("model.spectral_encoder.type must be deep_sets.")
    if set(settings["heads"]) != {"biology", "discriminator"}:
        raise ValueError("model.heads must contain exactly biology and discriminator.")
    spectral = settings["spectral_encoder"]["deep_sets"]
    positive = [model["latent_dim"], model["num_peaks"], spectral["embedding_dim"],
                spectral["peak_output_dim"], *spectral["peak_hidden_dims"],
                model.get("spectral_peak_budget", 65536), model.get("inference_peak_chunk_size", 262144),
                model.get("gatv2_tile_budget", 8)]
    for name, group in [("aggregation", settings["aggregation"]),
                        *((f"heads.{name}", group) for name, group in settings["heads"].items())]:
        if group["type"] not in {"mlp", "cnn", "gatv2"}:
            raise ValueError("Component type must be mlp or cnn or gatv2.")
        group.setdefault("mlp", {"hidden_dims": [512, 512]})
        group["cnn"] = {"channels": 128, "depth": 3, "dropout": .10,
                        "residual": True, **group.get("cnn", {})}
        group["gatv2"] = gatv2_settings(group.get("gatv2", {}))
        positive.extend([*group["mlp"]["hidden_dims"], group["cnn"]["channels"], group["cnn"]["depth"]])
        if not 0 <= group["cnn"]["dropout"] < 1:
            raise ValueError("CNN dropout must be in [0, 1).")
        if not isinstance(group["cnn"]["residual"], bool):
            raise ValueError(f"model.{name}.cnn.residual must be boolean.")
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in positive):
        raise ValueError("Model widths, depths, and peak budgets must be positive integers.")
    if not isinstance(model.get("spectral_checkpointing", True), bool):
        raise ValueError("model.spectral_checkpointing must be boolean.")
    if not 0 <= model["dropout"] < 1 or model["sigma_min"] <= 0:
        raise ValueError("Invalid model dropout or sigma_min.")
    return settings


def execution_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive execution mode and combined halo from the selected components."""
    settings = component_settings(config["model"])
    groups = [settings["aggregation"], settings["heads"]["biology"], settings["heads"]["discriminator"]]
    radii = []
    for group in groups:
        if group["type"] == "mlp":
            radii.append(0)
        elif group["type"] == "cnn":
            radii.append(group["cnn"]["depth"])
        elif group["type"] == "gatv2":
            radii.append(graph_reach(group["gatv2"]["depth"], group["gatv2"]["neighbor_radius"]))
    return {"mode": "spatial" if any(radii) else "pixel", "halo": radii[0] + max(radii[1:]),
            "latent_radius": radii[0], "biology_radius": radii[0] + radii[1],
            "combination": "deep_sets__agg-{}__bio-{}__disc-{}".format(*(g["type"] for g in groups))}


def resolve_execution(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve mode-specific settings and all run paths once, without overrides."""
    settings = component_settings(config["model"])
    if "spectral_encoder" in config["model"]:
        for name, group in [("aggregation", config["model"]["aggregation"]),
                            *config["model"]["heads"].items()]:
            resolved = settings["aggregation"] if name == "aggregation" else settings["heads"][name]
            group.setdefault("cnn", {})["residual"] = resolved["cnn"]["residual"]
            group["gatv2"] = resolved["gatv2"]
    execution = execution_settings(config)
    training = config["training"]
    mode = execution["mode"]
    spatial_batching = {key: training.get("spatial", training).get(key, default)
                       for key, default in (("core_size", 32), ("tiles_per_batch", 8),
                                            ("sampling_strategy", "shuffled_tiles"),
                                            ("supervised_rows_per_batch", 2048))}
    if "pixel" in training:
        training.update(training["pixel"])
    if mode in training:
        training.update(training[mode])
    if mode == "spatial":
        data = config["data"]
        for key in ("path", "x_column", "y_column", "batch_column"):
            if not isinstance(data.get(key), str) or not data[key]:
                raise ValueError(f"Spatial execution requires data.{key}.")
        training.update(spatial_batching)
        if not isinstance(training["sampling_strategy"], str) or training["sampling_strategy"] not in {"shuffled_tiles", "proportional_slide_tiles"}:
            raise ValueError("Unsupported training.spatial.sampling_strategy.")
        training["batch_size"] = (training["supervised_rows_per_batch"]
                                  if training["sampling_strategy"] == "proportional_slide_tiles"
                                  else training["tiles_per_batch"])
        training["validation_batch_size"] = training["tiles_per_batch"]
    for key in ("batch_size", "validation_batch_size", "prefetch_factor"):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer.")
    for key in ("core_size", "tiles_per_batch", "supervised_rows_per_batch") if mode == "spatial" else ():
        if not isinstance(training[key], int) or isinstance(training[key], bool) or training[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer.")
    if (isinstance(training["num_workers"], bool) or not isinstance(training["num_workers"], int)
            or training["num_workers"] < 0 or not np.isfinite(training["learning_rate"])
            or training["learning_rate"] <= 0):
        raise ValueError("Invalid worker count or learning rate.")
    config["execution"] = {**execution, **{key: training[key] for key in
        ("batch_size", "validation_batch_size", "num_workers", "prefetch_factor", "learning_rate")}}
    if mode == "spatial":
        config["execution"].update(
            core_size=training["core_size"], tiles_per_batch=training["tiles_per_batch"],
            sampling_strategy=training["sampling_strategy"],
            training_supervised_rows_per_batch=(training["supervised_rows_per_batch"]
                if training["sampling_strategy"] == "proportional_slide_tiles" else None),
            training_tiles_per_batch=(training["tiles_per_batch"]
                if training["sampling_strategy"] == "shuffled_tiles" else None),
            evaluation_tiles_per_batch=training["tiles_per_batch"])
    if "results" in config:
        result = config["results"]
        root = Path(result["root"]) / execution["combination"] / result.get("run_name", "fit")
        if result.get("smoke", False):
            root /= "smoke"
        training["output_dir"] = str(root / "model")
        config["analysis"]["output_dir"] = str(root / "analysis")
        config["analysis"]["checkpoint"] = str(root / "model" / "best.pt")
        config["spatial"] = {"output": str(root / "analysis/spatial/spatial_predictions.parquet"),
                             "latents": str(root / "analysis/spatial/spatial_latent.parquet")}
        config.setdefault("latent_variance", {}).update(
            input=config["data"].get("context_path"),
            latents=config["spatial"]["latents"], output_dir=str(root / "latent_variance"))
        result["run_dir"] = str(root)
    return config


def inference_input(config: Mapping[str, Any], explicit: Path | None = None) -> Path:
    """Resolve inference input independently of frozen training preprocessing."""
    value = explicit if explicit is not None else config['data'].get('context_path')
    if not value:
        raise ValueError("Inference requires --input or runtime data.context_path.")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Inference input does not exist: {path}")
    return path

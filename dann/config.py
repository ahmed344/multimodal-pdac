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
        "output_dir": f"{root}/Results/dann/analysis/latent_variance",
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


def post_training_settings(settings: Mapping[str, Any] | None = None) -> dict[str, bool]:
    """Resolve automatic workflow switches, rejecting typos and non-booleans."""
    defaults = dict(enabled=True, run_analysis=True, run_inference=True,
                    run_spatial_heatmaps=True, run_latent_variance=True,
                    overwrite_inference=True)
    if settings is not None and not isinstance(settings, Mapping):
        raise ValueError("post_training must be a mapping.")
    supplied = dict(settings or {})
    unknown = supplied.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown post_training settings: {sorted(unknown)}")
    resolved = {**defaults, **supplied}
    for key, value in resolved.items():
        if not isinstance(value, bool):
            raise ValueError(f"post_training.{key} must be a boolean.")
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
    if not isinstance(config['analysis'].get('latent_interactive', False), bool):
        raise ValueError('analysis.latent_interactive must be a boolean.')
    spatial_execution_settings(config["analysis"].get("spatial_execution"))
    analysis_execution_settings(config['analysis'].get('execution'))
    spatial_heatmap_settings(config["analysis"].get("spatial_heatmaps"))
    peak_analysis_settings(config["analysis"].get("peaks"))
    if "peaks" in config["analysis"]:
        count = config["analysis"].get("family_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("analysis.family_count must be a positive integer.")
        if config["analysis"].get("clustering_linkage") != "average":
            raise ValueError("Peak analysis requires average cosine linkage.")
    count, seed = density_sampling_settings(config["analysis"])
    config["analysis"].update(density_mc_samples=count, density_mc_seed=seed)
    if "post_training" in config and not isinstance(config["post_training"], Mapping):
        raise ValueError("post_training must be a mapping.")
    config["post_training"] = post_training_settings(config.get("post_training"))
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
    updated["post_training"] = post_training_settings(updated.get("post_training"))
    updated["post_training"]["enabled"] = False
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
    if not isinstance(model.get("gatv2_checkpointing", True), bool):
        raise ValueError("model.gatv2_checkpointing must be boolean.")
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
    config["performance"] = performance_settings(config)
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
    performance = config['performance']
    prepared = any(performance[k] for k in ('ram_cache', 'cache_geometry', 'transfer_prefetch',
                   'validation_tiles', 'validation_peak_budget', 'validation_gat_tiles'))
    execution.update(loader_backend='shared_threads' if prepared else 'pytorch_workers',
                     preparation_workers=performance['preparation_workers'] if prepared else training['num_workers'],
                     physical_validation_batch_size=(performance['validation_tiles'] or training['validation_batch_size'])
                         if mode == 'spatial' else training['validation_batch_size'])
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
            latents=config["spatial"]["latents"], output_dir=str(root / "analysis/latent_variance"))
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


PERFORMANCE_DEFAULTS = dict(
    ram_cache=False, cache_geometry=False, preparation_workers=0, prepared_batches=2, torch_threads=None,
    transfer_prefetch=False, validation_tiles=None, validation_peak_budget=None,
    validation_gat_tiles=None, validation_feature_cache=False, feature_cache_rows=1024,
    host_budget_gib=128., vram_budget_gib=24.,
)


def performance_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate execution-only controls, with backward-compatible disabled defaults."""
    supplied = config.get('performance', {})
    if not isinstance(supplied, Mapping):
        raise ValueError('performance must be a mapping.')
    settings = {**PERFORMANCE_DEFAULTS, **supplied}
    if set(settings) != set(PERFORMANCE_DEFAULTS):
        raise ValueError('Unknown performance setting.')
    for key in ('ram_cache', 'cache_geometry', 'transfer_prefetch', 'validation_feature_cache'):
        if not isinstance(settings[key], bool):
            raise ValueError(f'performance.{key} must be boolean.')
    for key in ('preparation_workers', 'prepared_batches', 'validation_tiles',
                'validation_peak_budget', 'validation_gat_tiles', 'feature_cache_rows', 'torch_threads'):
        value = settings[key]
        if value is None and (key.startswith('validation_') or key == 'torch_threads'):
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'preparation_workers' else 1):
            raise ValueError(f'Invalid performance.{key}.')
    for key in ('host_budget_gib', 'vram_budget_gib'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'performance.{key} must be finite and positive.')
    if settings['preparation_workers'] and not settings['ram_cache']:
        raise ValueError('Threaded preparation requires ram_cache (HDF5 handles are not shared).')
    if settings['validation_feature_cache'] and not (settings['ram_cache'] and settings['cache_geometry']):
        raise ValueError('Validation features require ram_cache and cache_geometry.')
    return settings


SPATIAL_EXECUTION_DEFAULTS = dict(
    enabled=False, tiles_per_batch=64, spectral_peak_budget=262144, gat_tiles=32,
    feature_cache=True, feature_cache_rows=1024, output_rows=16384, density_workers=16,
)


ANALYSIS_EXECUTION_DEFAULTS = dict(
    enabled=False, cache_policy='reuse', tiles_per_batch=64,
    spectral_peak_budget=262144, gat_tiles=32, feature_cache=True,
    feature_cache_rows=1024, sampling_workers=1, render_workers=16, performance={},
)


def analysis_execution_settings(supplied: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate standalone analysis controls without using training annotations."""
    if supplied is not None and not isinstance(supplied, Mapping):
        raise ValueError('analysis.execution must be a mapping.')
    settings = {**copy.deepcopy(ANALYSIS_EXECUTION_DEFAULTS), **(supplied or {})}
    if set(settings) != set(ANALYSIS_EXECUTION_DEFAULTS):
        raise ValueError('Unknown analysis.execution setting.')
    for key in ('enabled', 'feature_cache'):
        if not isinstance(settings[key], bool):
            raise ValueError(f'analysis.execution.{key} must be boolean.')
    for key in ('tiles_per_batch', 'spectral_peak_budget', 'gat_tiles', 'feature_cache_rows',
                'sampling_workers', 'render_workers'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key.endswith('workers') else 1):
            raise ValueError(f'Invalid analysis.execution.{key}.')
    if settings['cache_policy'] not in ('reuse', 'refresh', 'off'):
        raise ValueError('analysis.execution.cache_policy must be reuse, refresh, or off.')
    overrides = settings['performance']
    if not isinstance(overrides, Mapping) or set(overrides) - set(PERFORMANCE_DEFAULTS):
        raise ValueError('Invalid analysis.execution.performance overrides.')
    # Validate partial overrides in a compatible enabled profile; actual dependencies
    # are checked again after merging with current runtime performance settings.
    performance_settings({'performance': {**PERFORMANCE_DEFAULTS, 'ram_cache': True,
                                         'cache_geometry': True, **overrides}})
    return settings


def spatial_execution_settings(supplied: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate inference-only execution settings; absent sections stay legacy."""
    if supplied is not None and not isinstance(supplied, Mapping):
        raise ValueError('analysis.spatial_execution must be a mapping.')
    settings = {**SPATIAL_EXECUTION_DEFAULTS, **(supplied or {})}
    if set(settings) != set(SPATIAL_EXECUTION_DEFAULTS):
        raise ValueError('Unknown analysis.spatial_execution setting.')
    for key, value in settings.items():
        if key in ('enabled', 'feature_cache'):
            if not isinstance(value, bool):
                raise ValueError(f'analysis.spatial_execution.{key} must be boolean.')
        elif isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'density_workers' else 1):
            raise ValueError(f'Invalid analysis.spatial_execution.{key}.')
    return settings


SPATIAL_HEATMAP_DEFAULTS = dict(
    regions=["tissue", "lesion"], lesion_column="Lesion", workers=16,
    host_budget_gib=192.0, png_compress_level=1, cmap="turbo",
)


def spatial_heatmap_settings(supplied: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate standalone heatmap settings, independently of inference budgets."""
    if supplied is not None and not isinstance(supplied, Mapping):
        raise ValueError("analysis.spatial_heatmaps must be a mapping.")
    settings = {**copy.deepcopy(SPATIAL_HEATMAP_DEFAULTS), **(supplied or {})}
    if set(settings) != set(SPATIAL_HEATMAP_DEFAULTS):
        raise ValueError("Unknown analysis.spatial_heatmaps setting.")
    regions = settings["regions"]
    if (not isinstance(regions, (list, tuple)) or not regions
            or any(region not in ("tissue", "lesion") for region in regions)
            or len(set(regions)) != len(regions)):
        raise ValueError("spatial_heatmaps.regions must select tissue and/or lesion once each.")
    settings["regions"] = list(regions)
    if not isinstance(settings["lesion_column"], str) or not settings["lesion_column"].strip():
        raise ValueError("spatial_heatmaps.lesion_column must be a nonempty string.")
    for key, minimum, maximum in (("workers", 1, None), ("png_compress_level", 0, 9)):
        value = settings[key]
        if (isinstance(value, bool) or not isinstance(value, int) or value < minimum
                or (maximum is not None and value > maximum)):
            raise ValueError(f"Invalid spatial_heatmaps.{key}.")
    from matplotlib import colormaps
    if not isinstance(settings['cmap'], str) or settings['cmap'] not in colormaps:
        raise ValueError('spatial_heatmaps.cmap must name a registered colormap.')
    budget = settings["host_budget_gib"]
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not np.isfinite(budget) or budget <= 0:
        raise ValueError("spatial_heatmaps.host_budget_gib must be finite and positive.")
    return settings


PEAK_ANALYSIS_DEFAULTS = dict(
    enabled=False, spatial=True, lesion_column="Lesion", block_rows=4096,
    correlation_block_peaks=64, min_observations=20, umap_neighbors=30,
    umap_min_dist=0.30, umap_metric="cosine", umap_random_state=20260719,
    signal_percentile=99.0, slides_per_page=6,  # Accepted legacy field; no pagination.
    sequential_cmap="turbo", detection_cmap="turbo", ihc_cmap="RdBu_r",
    ihc_percentile=99.0, ihc_gamma=0.5,
)


def peak_analysis_settings(supplied: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate optional peak interpretation settings; old configurations opt out."""
    if supplied is not None and not isinstance(supplied, Mapping):
        raise ValueError("analysis.peaks must be a mapping.")
    settings = {**PEAK_ANALYSIS_DEFAULTS, **(supplied or {})}
    if set(settings) != set(PEAK_ANALYSIS_DEFAULTS):
        raise ValueError("Unknown analysis.peaks setting.")
    for key in ("enabled", "spatial"):
        if not isinstance(settings[key], bool):
            raise ValueError(f"analysis.peaks.{key} must be boolean.")
    for key in ("block_rows", "correlation_block_peaks", "min_observations", "umap_neighbors",
                "umap_random_state", "slides_per_page"):
        minimum = 0 if key == "umap_random_state" else 2 if key in ("min_observations", "umap_neighbors") else 1
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < minimum:
            raise ValueError(f"Invalid analysis.peaks.{key}.")
    for key, low, high in (("umap_min_dist", 0, 1), ("signal_percentile", 0, 100),
                           ("ihc_percentile", 0, 100), ("ihc_gamma", 0, 1)):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid analysis.peaks.{key}.")
    from matplotlib import colormaps
    for key in ('sequential_cmap', 'detection_cmap', 'ihc_cmap'):
        if not isinstance(settings[key], str) or settings[key] not in colormaps:
            raise ValueError(f'Invalid analysis.peaks.{key}.')
    if settings['ihc_percentile'] == 0 or settings['ihc_gamma'] == 0:
        raise ValueError('IHC percentile and gamma must be positive.')
    if settings['signal_percentile'] == 0 or settings['umap_metric'] != 'cosine':
        raise ValueError("Peak UMAP requires cosine distance and a positive signal percentile.")
    if not isinstance(settings['lesion_column'], str) or not settings['lesion_column'].strip():
        raise ValueError("analysis.peaks.lesion_column must be nonempty.")
    return settings

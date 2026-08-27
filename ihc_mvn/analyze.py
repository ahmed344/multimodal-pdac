"""Generate held-out, peptide, training, and MVN checkpoint diagnostics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
import yaml
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score

from .config import load_config, seed_everything
from .data_loader import create_inference_dataset
from .infer import (
    build_prediction_arrays,
    create_inference_loader,
    load_checkpoint_model,
)
from .scaling import matrix_group_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse checkpoint-analysis command-line arguments.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        argparse.Namespace: Parsed config and checkpoint paths.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="IHC-MVN YAML configuration path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint override; defaults to analysis.checkpoint or best.pt.",
    )
    return parser.parse_args(argv)


def _short_target(target: str) -> str:
    """Return a compact target display name.

    Args:
        target (str): Configured target column name.

    Returns:
        str: Name without the optional ``Density_`` prefix.
    """

    return str(target).removeprefix("Density_")


def _checkpoint_path(
    config: Mapping[str, Any], checkpoint_override: Path | None
) -> Path:
    """Resolve the checkpoint used for analysis.

    Args:
        config (Mapping[str, Any]): Complete pipeline configuration.
        checkpoint_override (Path | None): Optional explicit checkpoint.

    Returns:
        Path: Resolved checkpoint path.
    """

    configured = config["analysis"].get("checkpoint")
    if checkpoint_override is not None:
        return Path(checkpoint_override)
    if configured:
        return Path(str(configured))
    return Path(str(config["output"]["directory"])) / "best.pt"


def _select_rows(
    rows: np.ndarray,
    slide_codes: np.ndarray,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Deterministically cap rows while approximately preserving slides.

    Args:
        rows (np.ndarray): Global split row indices.
        slide_codes (np.ndarray): Slide code for every source observation.
        maximum (int): Positive maximum selected rows.
        seed (int): Random sampling seed.

    Returns:
        np.ndarray: Sorted selected global row indices.
    """

    values = np.asarray(rows, dtype=np.int64)
    if values.size <= maximum:
        return values.copy()
    generator = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    unique, counts = np.unique(slide_codes[values], return_counts=True)
    allocations = np.floor(maximum * counts / counts.sum()).astype(int)
    allocations = np.maximum(allocations, 1)
    while allocations.sum() > maximum:
        index = int(np.argmax(allocations))
        allocations[index] -= 1
    while allocations.sum() < maximum:
        candidates = np.flatnonzero(allocations < counts)
        allocations[candidates[0]] += 1
    for code, count in zip(unique, allocations, strict=True):
        candidates = values[slide_codes[values] == code]
        selected.append(generator.choice(candidates, size=int(count), replace=False))
    return np.sort(np.concatenate(selected))


def extract_analysis_data(
    config: Mapping[str, Any],
    checkpoint_path: Path,
) -> tuple[dict[str, np.ndarray], torch.nn.Module, Mapping[str, Any]]:
    """Extract frozen-checkpoint predictions on a configured held-out split.

    Args:
        config (Mapping[str, Any]): Complete current pipeline configuration.
        checkpoint_path (Path): Checkpoint containing split and fitted state.

    Returns:
        tuple[dict[str, np.ndarray], torch.nn.Module, Mapping[str, Any]]:
            Extracted arrays, restored model, and checkpoint mapping.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint root must be a mapping.")
    (
        model,
        stored_config,
        scaler,
        target_standardizer,
        slide_mapping,
        patient_mapping,
        target_columns,
        _,
        device,
    ) = load_checkpoint_model(checkpoint_path)
    analysis = config["analysis"]
    split = str(analysis["split"])
    if split not in checkpoint["split_indices"]:
        raise KeyError(f"Checkpoint has no split indices for {split!r}.")
    data_config = stored_config["data"]
    input_path = Path(str(data_config["path"]))
    dataset = create_inference_dataset(
        path=input_path,
        scaler=scaler,
        patient_mapping=patient_mapping,
        batch_column=str(data_config["batch_column"]),
        patient_column=str(data_config["patient_column"]),
        target_columns=target_columns,
        target_standardizer=target_standardizer,
    )
    if dataset.metadata.target_arrays is None:
        dataset.close()
        raise ValueError("Held-out analysis requires labeled training data.")
    rows = _select_rows(
        np.asarray(checkpoint["split_indices"][split], dtype=np.int64),
        dataset.metadata.slide_codes,
        int(analysis["max_samples"]),
        int(config["training"]["seed"]),
    )
    dataset.indices = rows
    loader = create_inference_loader(
        dataset,
        batch_size=int(analysis["batch_size"]),
        num_workers=int(analysis["num_workers"]),
        pin_memory=device.type == "cuda",
        prefetch_factor=int(config["training"].get("prefetch_factor", 2)),
    )
    collected: dict[str, list[np.ndarray]] = {
        "latent": [],
        "row_position": [],
        "slide_code": [],
        "patient_code": [],
        "target_densities": [],
        "targets": [],
        "target_positive_mask": [],
        "mahalanobis": [],
        "mahalanobis_df": [],
    }
    prediction_parts: dict[str, list[np.ndarray]] = {}
    try:
        with torch.inference_mode():
            for cpu_batch in loader:
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
                for key, value in arrays.items():
                    prediction_parts.setdefault(key, []).append(value)
                collected["latent"].append(
                    predictions["latent"].detach().cpu().numpy()
                )
                collected["row_position"].append(cpu_batch["row_ids"].numpy())
                collected["slide_code"].append(cpu_batch["slide_codes"].numpy())
                collected["patient_code"].append(cpu_batch["patient_codes"].numpy())
                collected["target_densities"].append(
                    cpu_batch["target_densities"].numpy()
                )
                collected["targets"].append(cpu_batch["targets"].numpy())
                mask = cpu_batch["target_positive_mask"].numpy().astype(bool)
                collected["target_positive_mask"].append(mask)
                means = predictions["mean_total"].detach().cpu().numpy()
                covariance = predictions["covariance"].detach().cpu().numpy()
                standardized = cpu_batch["targets"].numpy()
                distances = np.zeros(mask.shape[0], dtype=np.float64)
                degrees = mask.sum(axis=1).astype(np.int16)
                for row_index, observed in enumerate(mask):
                    if not observed.any():
                        continue
                    residual = (
                        standardized[row_index, observed]
                        - means[row_index, observed]
                    )
                    submatrix = covariance[row_index][np.ix_(observed, observed)]
                    distances[row_index] = residual @ np.linalg.solve(
                        submatrix, residual
                    )
                collected["mahalanobis"].append(distances)
                collected["mahalanobis_df"].append(degrees)
    finally:
        dataset.close()
    extracted = {
        key: np.concatenate(parts, axis=0) for key, parts in collected.items()
    }
    extracted.update(
        {
            key: np.concatenate(parts, axis=0)
            for key, parts in prediction_parts.items()
        }
    )
    extracted["slide_names"] = np.asarray(slide_mapping.names, dtype=object)
    extracted["patient_names"] = np.asarray(patient_mapping.names, dtype=object)
    extracted["target_columns"] = np.asarray(target_columns, dtype=object)
    return extracted, model, checkpoint


def fit_latent_umap(
    latent: np.ndarray, analysis_config: Mapping[str, Any]
) -> np.ndarray:
    """Fit a reproducible two-dimensional UMAP.

    Args:
        latent (np.ndarray): Latent matrix with shape ``[N, D]``.
        analysis_config (Mapping[str, Any]): UMAP configuration values.

    Returns:
        np.ndarray: UMAP coordinates with shape ``[N, 2]``.
    """

    values = np.asarray(latent, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("UMAP requires a two-dimensional matrix with at least 3 rows.")
    neighbors = min(int(analysis_config["umap_neighbors"]), values.shape[0] - 1)
    reducer = umap.UMAP(
        n_neighbors=max(neighbors, 2),
        min_dist=float(analysis_config["umap_min_dist"]),
        metric=str(analysis_config["umap_metric"]),
        random_state=int(analysis_config["umap_random_state"]),
    )
    return np.asarray(reducer.fit_transform(values), dtype=np.float32)


def build_analysis_frame(
    extracted: Mapping[str, np.ndarray],
    coordinates: np.ndarray,
    split: str,
) -> pd.DataFrame:
    """Build the complete held-out UMAP and prediction table.

    Args:
        extracted (Mapping[str, np.ndarray]): Extracted checkpoint arrays.
        coordinates (np.ndarray): UMAP coordinates with shape ``[N, 2]``.
        split (str): Source split label.

    Returns:
        pd.DataFrame: Row-aligned metadata, truth, predictions, and diagnostics.
    """

    row_ids = np.asarray(extracted["row_position"], dtype=np.int64)
    target_columns = [str(value) for value in extracted["target_columns"]]
    frame = pd.DataFrame(
        {
            "row_position": row_ids,
            "split": split,
            "slide": np.asarray(extracted["slide_names"], dtype=object)[
                np.asarray(extracted["slide_code"], dtype=np.int64)
            ],
            "patient": np.asarray(extracted["patient_names"], dtype=object)[
                np.asarray(extracted["patient_code"], dtype=np.int64)
            ],
            "umap_1": coordinates[:, 0],
            "umap_2": coordinates[:, 1],
            "mahalanobis": extracted["mahalanobis"],
            "mahalanobis_df": extracted["mahalanobis_df"],
        }
    )
    densities = np.asarray(extracted["target_densities"])
    standardized = np.asarray(extracted["targets"])
    for target_index, target in enumerate(target_columns):
        frame[f"observed_density_{target}"] = densities[:, target_index]
        frame[f"observed_standardized_{target}"] = standardized[:, target_index]
    ignored = {
        "latent",
        "row_position",
        "slide_code",
        "patient_code",
        "target_densities",
        "targets",
        "target_positive_mask",
        "mahalanobis",
        "mahalanobis_df",
        "slide_names",
        "patient_names",
        "target_columns",
    }
    for name, values in extracted.items():
        if name not in ignored and np.asarray(values).ndim == 1:
            frame[name] = values
    return frame


def _robust_limits(values: np.ndarray) -> tuple[float, float]:
    """Compute robust finite color limits.

    Args:
        values (np.ndarray): Numeric values, possibly containing NaNs.

    Returns:
        tuple[float, float]: Lower and upper color limits.
    """

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lower, upper = np.percentile(finite, [1.0, 99.0])
    if lower == upper:
        upper = lower + 1.0
    return float(lower), float(upper)


def _umap_scatter(
    axis: plt.Axes,
    coordinates: np.ndarray,
    values: np.ndarray,
    title: str,
    point_size: float,
    cmap: str,
    *,
    diverging: bool = False,
) -> None:
    """Draw one rasterized continuous UMAP panel.

    Args:
        axis (plt.Axes): Destination Matplotlib axis.
        coordinates (np.ndarray): Two-dimensional UMAP coordinates.
        values (np.ndarray): Continuous color values.
        title (str): Panel title.
        point_size (float): Marker size.
        cmap (str): Matplotlib colormap name.
        diverging (bool): Whether to center symmetric limits at zero.

    Returns:
        None: The axis is modified in place.
    """

    color_values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(color_values)
    if diverging:
        maximum = (
            float(np.percentile(np.abs(color_values[finite]), 99.0))
            if finite.any()
            else 1.0
        )
        vmin, vmax = -max(maximum, 1.0e-6), max(maximum, 1.0e-6)
    else:
        vmin, vmax = _robust_limits(color_values)
    image = axis.scatter(
        coordinates[finite, 0],
        coordinates[finite, 1],
        c=color_values[finite],
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        rasterized=True,
    )
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])


def plot_categorical_umaps(
    frame: pd.DataFrame,
    output_path: Path,
    point_size: float,
) -> None:
    """Plot latent UMAP colored by slide and patient.

    Args:
        frame (pd.DataFrame): Analysis frame containing UMAP/category columns.
        output_path (Path): Destination PNG path.
        point_size (float): Marker size.

    Returns:
        None: Figure is saved to disk.
    """

    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for axis, column in zip(axes, ("slide", "patient"), strict=True):
        codes, names = pd.factorize(frame[column], sort=True)
        image = axis.scatter(
            frame["umap_1"],
            frame["umap_2"],
            c=codes,
            cmap="tab20",
            s=point_size,
            linewidths=0,
            rasterized=True,
        )
        handles, _ = image.legend_elements(num=min(len(names), 20))
        axis.legend(
            handles,
            [str(value) for value in names[: len(handles)]],
            title=column.title(),
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=7,
        )
        axis.set_title(f"Latent UMAP by {column}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(output_path)
    plt.close(figure)


def plot_target_umaps(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
    output_dir: Path,
    config: Mapping[str, Any],
) -> None:
    """Write one MVN-aware UMAP figure per target.

    Args:
        frame (pd.DataFrame): Analysis table with UMAP and prediction columns.
        target_columns (Sequence[str]): Ordered modeled target names.
        output_dir (Path): UMAP output directory.
        config (Mapping[str, Any]): Analysis plot settings.

    Returns:
        None: Target figures are saved to disk.
    """

    point_size = float(config["point_size"])
    cmap = str(config["density_cmap"])
    for target in target_columns:
        truth = frame[f"observed_density_{target}"].to_numpy()
        observed_standardized = frame[f"observed_standardized_{target}"].to_numpy()
        predicted = frame[f"standardized_mean_total_{target}"].to_numpy()
        positive = truth > 0.0
        residual = np.full(truth.shape, np.nan)
        residual[positive] = predicted[positive] - observed_standardized[positive]
        width = (
            frame[f"q95_density_{target}"].to_numpy()
            - frame[f"q05_density_{target}"].to_numpy()
        )
        panels = (
            (truth, "Observed density", cmap, False),
            (
                frame[f"prob_presence_{target}"].to_numpy(),
                "Presence probability",
                "viridis",
                False,
            ),
            (predicted, "Fully adjusted mean", cmap, False),
            (
                frame[f"marginal_unstandardized_sigma_{target}"].to_numpy(),
                "Marginal uncertainty",
                "magma",
                False,
            ),
            (width, "90% density interval width", "magma", False),
            (residual, "Standardized residual", "coolwarm", True),
        )
        figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
        coordinates = frame[["umap_1", "umap_2"]].to_numpy()
        for axis, (values, title, color_map, diverging) in zip(
            axes.flat, panels, strict=True
        ):
            _umap_scatter(
                axis,
                coordinates,
                values,
                title,
                point_size,
                color_map,
                diverging=diverging,
            )
        figure.suptitle(f"{_short_target(target)} held-out MVN latent space")
        figure.savefig(
            output_dir / f"latent_umap_mvn_{target}.png",
            dpi=int(config["figure_dpi"]),
        )
        plt.close(figure)


def compute_peak_activity(
    path: Path,
    matrix_key: str,
    num_peaks: int,
    chunk_size: int,
    max_nonzeros: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Count peak occurrence and summed intensity directly from CSR storage.

    Args:
        path (Path): AnnData HDF5 path.
        matrix_key (str): AnnData sparse matrix key.
        num_peaks (int): Number of aligned feature bins.
        chunk_size (int): Sparse entries read per chunk.
        max_nonzeros (int | None): Optional diagnostic scan cap.

    Returns:
        tuple[np.ndarray, np.ndarray]: Per-peak counts and intensity sums.
    """

    counts = np.zeros(num_peaks, dtype=np.int64)
    intensity_sums = np.zeros(num_peaks, dtype=np.float64)
    with h5py.File(path, "r") as handle:
        group = handle[matrix_group_path(matrix_key)]
        total = int(group["data"].shape[0])
        stop = total if max_nonzeros is None else min(total, int(max_nonzeros))
        for start in range(0, stop, chunk_size):
            end = min(start + chunk_size, stop)
            indices = np.asarray(group["indices"][start:end], dtype=np.int64)
            values = np.asarray(group["data"][start:end], dtype=np.float64)
            counts += np.bincount(indices, minlength=num_peaks)
            intensity_sums += np.bincount(
                indices, weights=values, minlength=num_peaks
            )
    return counts, intensity_sums


def embedding_cosine_families(
    embedding_weights: np.ndarray,
    family_count: int,
    linkage_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster peak embeddings by cosine similarity.

    Args:
        embedding_weights (np.ndarray): Peak embedding matrix ``[P, E]``.
        family_count (int): Requested number of flat families.
        linkage_method (str): SciPy hierarchy linkage method.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Similarity matrix, one-based
            family labels, and hierarchy leaf order.
    """

    weights = np.asarray(embedding_weights, dtype=np.float32)
    if weights.ndim != 2 or weights.shape[0] < 2:
        raise ValueError("At least two peak embedding rows are required.")
    normalized = weights / np.maximum(
        np.linalg.norm(weights, axis=1, keepdims=True), np.finfo(np.float32).eps
    )
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    linkage = hierarchy.linkage(
        distance.squareform(1.0 - similarity, checks=False), method=linkage_method
    )
    families = hierarchy.fcluster(
        linkage, t=min(family_count, weights.shape[0]), criterion="maxclust"
    )
    return (
        similarity,
        families.astype(np.int32),
        hierarchy.leaves_list(linkage).astype(np.int64),
    )


def plot_similarity_heatmap(
    similarity: np.ndarray,
    activity: np.ndarray,
    order: np.ndarray,
    labels: Sequence[str],
    output_path: Path,
    top_peaks: int,
    dpi: int,
) -> np.ndarray:
    """Plot clustered similarity for the most active peaks.

    Args:
        similarity (np.ndarray): Full cosine similarity matrix.
        activity (np.ndarray): Per-peak occurrence counts.
        order (np.ndarray): Hierarchical leaf order.
        labels (Sequence[str]): Ordered feature labels.
        output_path (Path): Destination PNG path.
        top_peaks (int): Maximum displayed peaks.
        dpi (int): Saved figure resolution.

    Returns:
        np.ndarray: Ordered displayed peak indices.
    """

    count = min(int(top_peaks), activity.size)
    active = np.argpartition(activity, -count)[-count:]
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(order.size)
    selected = active[np.argsort(ranks[active])]
    tick_labels = [str(labels[index]) for index in selected] if count <= 40 else False
    figure, axis = plt.subplots(figsize=(12, 10), constrained_layout=True)
    sns.heatmap(
        similarity[np.ix_(selected, selected)],
        ax=axis,
        cmap="vlag",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar_kws={"label": "Cosine similarity"},
    )
    axis.set_title(f"Learned peptide similarity: {count} most active bins")
    axis.set_xlabel("Hierarchically ordered peaks")
    axis.set_ylabel("Peaks")
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return selected


def plot_training_curves(
    history_path: Path,
    output_path: Path,
    best_epoch: int,
    dpi: int,
) -> None:
    """Plot MVN losses and predictive metrics across training.

    Args:
        history_path (Path): Epoch history CSV.
        output_path (Path): Destination PNG path.
        best_epoch (int): One-based selected checkpoint epoch.
        dpi (int): Saved figure resolution.

    Returns:
        None: Figure is saved to disk.
    """

    history = pd.read_csv(history_path)
    panels = (
        ("total_loss", "Total objective"),
        ("hurdle_loss", "Hurdle loss"),
        ("positive_loss", "Masked MVN loss"),
        ("prior_loss", "Random-effect prior"),
        ("presence_balanced_accuracy_macro", "Presence balanced accuracy"),
        ("positive_r2_macro", "Positive-coordinate R²"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, panels, strict=True):
        for split in ("train", "validation"):
            column = f"{split}_{metric}"
            if column in history:
                axis.plot(history["epoch"], history[column], label=split.title())
        axis.axvline(
            best_epoch,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"Best ({best_epoch})",
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def export_random_effects(
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    output_dir: Path,
    dpi: int,
) -> pd.DataFrame:
    """Export scaled patient/slide random effects and heatmaps.

    Args:
        model (torch.nn.Module): Restored IHC multivariate model.
        checkpoint (Mapping[str, Any]): Checkpoint category/target state.
        output_dir (Path): Analysis output directory.
        dpi (int): Saved figure resolution.

    Returns:
        pd.DataFrame: Long-form random-effect table.
    """

    effects = model.random_intercepts
    targets = [str(value) for value in checkpoint["target_columns"]]
    groups = (
        (
            "patient",
            checkpoint["patient_mapping_state"]["names"],
            effects.patient_hurdle_raw.weight,
            effects.patient_hurdle_scale_raw,
            effects.patient_mean_raw.weight,
            effects.patient_mean_scale_raw,
        ),
        (
            "slide",
            checkpoint["slide_mapping_state"]["names"],
            effects.slide_hurdle_raw.weight,
            effects.slide_hurdle_scale_raw,
            effects.slide_mean_raw.weight,
            effects.slide_mean_scale_raw,
        ),
    )
    rows: list[dict[str, Any]] = []
    for group, names, hurdle_raw, hurdle_scale_raw, mean_raw, mean_scale_raw in groups:
        hurdle_scale = (
            torch.nn.functional.softplus(hurdle_scale_raw) + effects.scale_floor
        )
        mean_scale = torch.nn.functional.softplus(mean_scale_raw) + effects.scale_floor
        hurdle = (hurdle_raw * hurdle_scale).detach().cpu().numpy()
        means = (mean_raw * mean_scale).detach().cpu().numpy()
        hurdle_scale_values = hurdle_scale.detach().cpu().numpy()
        mean_scale_values = mean_scale.detach().cpu().numpy()
        figure, axes = plt.subplots(1, 2, figsize=(13, max(4, len(names) * 0.3)))
        for axis, matrix, branch in zip(
            axes, (hurdle, means), ("hurdle", "mean"), strict=True
        ):
            sns.heatmap(
                matrix,
                ax=axis,
                cmap="vlag",
                center=0.0,
                xticklabels=[_short_target(target) for target in targets],
                yticklabels=[str(name) for name in names],
                cbar_kws={"label": "Scaled random intercept"},
            )
            axis.set_title(f"{group.title()} {branch} effects")
        figure.tight_layout()
        figure.savefig(output_dir / f"random_effects_{group}.png", dpi=dpi)
        plt.close(figure)
        for level_index, level in enumerate(names):
            for target_index, target in enumerate(targets):
                rows.extend(
                    (
                        {
                            "group": group,
                            "level": str(level),
                            "branch": "hurdle",
                            "target": target,
                            "effect": float(hurdle[level_index, target_index]),
                            "prior_scale": float(hurdle_scale_values[target_index]),
                        },
                        {
                            "group": group,
                            "level": str(level),
                            "branch": "mean",
                            "target": target,
                            "effect": float(means[level_index, target_index]),
                            "prior_scale": float(mean_scale_values[target_index]),
                        },
                    )
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "random_effects.csv", index=False)
    return frame


def export_mvn_diagnostics(
    extracted: Mapping[str, np.ndarray],
    frame: pd.DataFrame,
    model: torch.nn.Module,
    output_dir: Path,
    config: Mapping[str, Any],
) -> None:
    """Write correlation, calibration, scatter, and Mahalanobis diagnostics.

    Args:
        extracted (Mapping[str, np.ndarray]): Extracted held-out arrays.
        frame (pd.DataFrame): Held-out analysis table.
        model (torch.nn.Module): Restored fitted model.
        output_dir (Path): Analysis output directory.
        config (Mapping[str, Any]): Analysis plot settings.

    Returns:
        None: CSV and PNG artifacts are written.
    """

    targets = [str(value) for value in extracted["target_columns"]]
    learned = model.R.detach().cpu().numpy()
    residuals = np.asarray(extracted["targets"]) - np.column_stack(
        [frame[f"standardized_mean_total_{target}"] for target in targets]
    )
    mask = np.asarray(extracted["target_positive_mask"], dtype=bool)
    empirical = np.eye(len(targets), dtype=np.float64)
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            common = mask[:, left] & mask[:, right]
            value = (
                np.corrcoef(residuals[common, left], residuals[common, right])[0, 1]
                if common.sum() >= 3
                else np.nan
            )
            empirical[left, right] = empirical[right, left] = value
    labels = [_short_target(target) for target in targets]
    pd.DataFrame(learned, index=targets, columns=targets).to_csv(
        output_dir / "learned_target_correlation.csv"
    )
    pd.DataFrame(empirical, index=targets, columns=targets).to_csv(
        output_dir / "empirical_residual_correlation.csv"
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, matrix, title in zip(
        axes,
        (learned, empirical),
        ("Learned MVN correlation", "Held-out residual correlation"),
        strict=True,
    ):
        sns.heatmap(
            matrix,
            ax=axis,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            cmap="vlag",
            annot=True,
            fmt=".2f",
            xticklabels=labels,
            yticklabels=labels,
        )
        axis.set_title(title)
    figure.savefig(
        output_dir / "target_correlation_heatmaps.png",
        dpi=int(config["figure_dpi"]),
    )
    plt.close(figure)

    calibration_rows: list[dict[str, Any]] = []
    figure, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    for target_index, (axis, target) in enumerate(
        zip(axes.flat, targets, strict=True)
    ):
        observed = np.asarray(extracted["target_densities"])[:, target_index]
        positive = observed > 0.0
        probability = frame[f"prob_presence_{target}"].to_numpy()
        brier = float(np.mean((probability - positive.astype(float)) ** 2))
        auc = (
            float(roc_auc_score(positive, probability))
            if np.unique(positive).size == 2
            else math.nan
        )
        lower = frame[f"q05_density_{target}"].to_numpy()
        upper = frame[f"q95_density_{target}"].to_numpy()
        coverage = (
            float(np.mean((observed[positive] >= lower[positive]) & (observed[positive] <= upper[positive])))
            if positive.any()
            else math.nan
        )
        calibration_rows.append(
            {
                "target": target,
                "presence_brier": brier,
                "presence_auc": auc,
                "positive_interval_90_coverage": coverage,
                "positive_count": int(positive.sum()),
            }
        )
        bins = np.linspace(0.0, 1.0, int(config["calibration_bins"]) + 1)
        bin_codes = np.clip(np.digitize(probability, bins) - 1, 0, len(bins) - 2)
        predicted, actual = [], []
        for bin_index in range(len(bins) - 1):
            selected = bin_codes == bin_index
            if selected.any():
                predicted.append(float(probability[selected].mean()))
                actual.append(float(positive[selected].mean()))
        axis.plot([0, 1], [0, 1], "k--")
        axis.plot(predicted, actual, marker="o")
        axis.set(
            title=f"{labels[target_index]} (Brier={brier:.3f}, AUC={auc:.3f})",
            xlabel="Predicted presence",
            ylabel="Observed frequency",
            xlim=(0, 1),
            ylim=(0, 1),
        )
    pd.DataFrame(calibration_rows).to_csv(
        output_dir / "calibration_metrics.csv", index=False
    )
    figure.savefig(
        output_dir / "hurdle_reliability.png", dpi=int(config["figure_dpi"])
    )
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    for target_index, (axis, target) in enumerate(
        zip(axes.flat, targets, strict=True)
    ):
        truth = np.asarray(extracted["targets"])[:, target_index]
        positive = np.asarray(extracted["target_positive_mask"])[:, target_index]
        predicted = frame[f"standardized_mean_total_{target}"].to_numpy()
        axis.scatter(
            truth[positive],
            predicted[positive],
            s=float(config["point_size"]),
            alpha=0.3,
            rasterized=True,
        )
        if positive.any():
            lower = min(truth[positive].min(), predicted[positive].min())
            upper = max(truth[positive].max(), predicted[positive].max())
            axis.plot([lower, upper], [lower, upper], "k--")
        axis.set(
            title=labels[target_index],
            xlabel="Observed standardized coordinate",
            ylabel="Predicted fully adjusted mean",
        )
    figure.savefig(
        output_dir / "positive_prediction_scatter.png",
        dpi=int(config["figure_dpi"]),
    )
    plt.close(figure)

    distances = np.asarray(extracted["mahalanobis"], dtype=np.float64)
    degrees = np.asarray(extracted["mahalanobis_df"], dtype=np.int64)
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    metric_rows = []
    for degree, axis in zip(range(1, 5), axes.flat, strict=True):
        selected = np.sort(distances[degrees == degree])
        probabilities = (np.arange(selected.size) + 0.5) / max(selected.size, 1)
        expected = stats.chi2.ppf(probabilities, degree) if selected.size else np.array([])
        axis.scatter(expected, selected, s=4)
        if selected.size:
            maximum = max(float(expected.max()), float(selected.max()))
            axis.plot([0, maximum], [0, maximum], "k--")
        axis.set(
            title=f"{degree} positive coordinate(s), n={selected.size}",
            xlabel=f"Theoretical chi-square({degree}) quantile",
            ylabel="Observed squared Mahalanobis distance",
        )
        metric_rows.append(
            {
                "degrees_of_freedom": degree,
                "rows": int(selected.size),
                "mean_mahalanobis": float(selected.mean()) if selected.size else math.nan,
                "expected_mean": float(degree),
            }
        )
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "mahalanobis_metrics.csv", index=False
    )
    figure.savefig(
        output_dir / "mahalanobis_qq.png", dpi=int(config["figure_dpi"])
    )
    plt.close(figure)


def run_analysis(
    config: Mapping[str, Any],
    checkpoint_override: Path | None = None,
) -> Path:
    """Run all held-out and checkpoint-level visualization exports.

    Args:
        config (Mapping[str, Any]): Complete validated configuration.
        checkpoint_override (Path | None): Optional checkpoint path override.

    Returns:
        Path: Analysis output directory.
    """

    seed_everything(int(config["training"]["seed"]), deterministic=False)
    checkpoint_path = _checkpoint_path(config, checkpoint_override)
    extracted, model, checkpoint = extract_analysis_data(config, checkpoint_path)
    output_dir = Path(str(config["analysis"]["output_dir"]))
    umap_dir = output_dir / "umap"
    output_dir.mkdir(parents=True, exist_ok=True)
    umap_dir.mkdir(parents=True, exist_ok=True)
    coordinates = fit_latent_umap(extracted["latent"], config["analysis"])
    frame = build_analysis_frame(extracted, coordinates, str(config["analysis"]["split"]))
    frame.to_csv(umap_dir / "latent_umap.csv", index=False)
    latent_frame = pd.DataFrame(
        extracted["latent"],
        columns=[f"latent_{index:03d}" for index in range(extracted["latent"].shape[1])],
    )
    latent_frame.insert(0, "row_position", extracted["row_position"])
    latent_frame.to_parquet(output_dir / "latent_embeddings.parquet", index=False)
    plot_categorical_umaps(
        frame,
        umap_dir / "latent_umap_groups.png",
        float(config["analysis"]["point_size"]),
    )
    plot_target_umaps(
        frame,
        [str(value) for value in extracted["target_columns"]],
        umap_dir,
        config["analysis"],
    )
    if "conditional_cd8_excess" in frame and np.isfinite(
        frame["conditional_cd8_excess"].to_numpy(dtype=np.float64)
    ).any():
        figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
        _umap_scatter(
            axis,
            frame[["umap_1", "umap_2"]].to_numpy(),
            frame["conditional_cd8_excess"].to_numpy(),
            "Conditional CD8 excess",
            float(config["analysis"]["point_size"]),
            "coolwarm",
            diverging=True,
        )
        figure.savefig(
            umap_dir / "latent_umap_conditional_cd8_excess.png",
            dpi=int(config["analysis"]["figure_dpi"]),
        )
        plt.close(figure)

    weights = model.encoder.embedding.weight.detach().cpu().numpy()
    similarity, families, order = embedding_cosine_families(
        weights,
        int(config["analysis"]["family_count"]),
        str(config["analysis"]["clustering_linkage"]),
    )
    np.save(output_dir / "embedding_cosine_similarity.npy", similarity)
    feature_names = [str(value) for value in checkpoint["feature_names"]]
    stored_config = checkpoint["config"]
    activity, intensity_sums = compute_peak_activity(
        Path(str(stored_config["data"]["path"])),
        str(stored_config["data"]["matrix_key"]),
        len(feature_names),
        int(config["analysis"]["activity_chunk_size"]),
        config["analysis"].get("activity_max_nonzeros"),
    )
    shown = plot_similarity_heatmap(
        similarity,
        activity,
        order,
        feature_names,
        output_dir / "peptide_similarity_heatmap.png",
        int(config["analysis"]["heatmap_top_peaks"]),
        int(config["analysis"]["figure_dpi"]),
    )
    pd.DataFrame(
        {
            "peak_index": np.arange(len(feature_names)),
            "feature_name": feature_names,
            "family": families,
            "occurrence_count": activity,
            "intensity_sum": intensity_sums,
            "embedding_norm": np.linalg.norm(weights, axis=1),
            "shown_in_heatmap": np.isin(np.arange(len(feature_names)), shown),
        }
    ).to_csv(output_dir / "peptide_families.csv", index=False)

    best_epoch = int(checkpoint["epoch"]) + 1
    plot_training_curves(
        Path(str(config["output"]["directory"])) / "history.csv",
        Path(str(config["output"]["directory"])) / "training_curves.png",
        best_epoch,
        int(config["analysis"]["figure_dpi"]),
    )
    export_random_effects(
        model, checkpoint, output_dir, int(config["analysis"]["figure_dpi"])
    )
    export_mvn_diagnostics(
        extracted, frame, model, output_dir, config["analysis"]
    )
    with (output_dir / "analysis_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": best_epoch,
                "split": str(config["analysis"]["split"]),
                "rows": len(frame),
            },
            handle,
            sort_keys=False,
        )
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Load configuration and run checkpoint analysis.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        int: Zero process status after successful analysis.
    """

    args = parse_args(argv)
    output_dir = run_analysis(load_config(args.config), args.checkpoint)
    print(f"Analysis written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

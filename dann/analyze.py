"""Post-training latent, ZILN, and peptide-embedding analysis."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.cluster import hierarchy
from scipy.spatial import distance

from dann.config import apply_smoke_overrides, load_config, resolve_device, seed_everything
from dann.data_loader import DataBundle, create_data_bundle
from dann.model import AdversarialLatentFusion
from dann.train import move_batch_to_device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse analysis command-line arguments.

    Args:
        argv (Sequence[str] | None): Optional explicit argument sequence.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Commented YAML configuration path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint override; defaults to analysis.checkpoint or training best.pt.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Apply the same small-data settings as training --smoke-test.",
    )
    return parser.parse_args(argv)


def _analysis_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Cap the selected split and align inference loader settings.

    Args:
        config (Mapping[str, Any]): Complete configuration.

    Returns:
        dict[str, Any]: Deep-copied inference configuration.
    """

    updated = copy.deepcopy(dict(config))
    split = str(updated["analysis"]["split"])
    scatter_split = str(updated["analysis"]["ziln_scatter_split"])
    valid_splits = {"train", "validation", "test"}
    if split not in valid_splits or scatter_split not in valid_splits:
        raise ValueError(
            f"Unsupported analysis splits: main={split!r}, scatter={scatter_split!r}"
        )
    for selected_split in {split, scatter_split}:
        updated["data"][f"max_{selected_split}_samples"] = int(
            updated["analysis"]["max_samples"]
        )
    updated["training"]["num_workers"] = int(updated["analysis"]["num_workers"])
    return updated


def load_model(
    checkpoint_path: Path,
    config: Mapping[str, Any],
    bundle: DataBundle,
    device: torch.device,
) -> tuple[AdversarialLatentFusion, int]:
    """Reconstruct a trained model from a checkpoint.

    Args:
        checkpoint_path (Path): Saved training checkpoint.
        config (Mapping[str, Any]): Effective model configuration.
        bundle (DataBundle): Dataset metadata defining class counts.
        device (torch.device): Inference device.

    Returns:
        tuple[AdversarialLatentFusion, int]: Evaluation-mode trained model and the
        checkpoint's one-based training epoch.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AdversarialLatentFusion.from_config(
        config,
        num_batches=len(bundle.metadata.batch_names),
        num_targets=len(config["data"]["target_columns"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, int(checkpoint["epoch"]) + 1


def extract_latent_predictions(
    model: AdversarialLatentFusion,
    loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Stream latent vectors and ZILN parameters from sparse MSI rows.

    Args:
        model (AdversarialLatentFusion): Trained evaluation-mode model.
        loader (torch.utils.data.DataLoader): Sparse inference loader.
        device (torch.device): Inference device.

    Returns:
        dict[str, np.ndarray]: Concatenated latents, parameters, labels, and row IDs.
    """

    outputs: dict[str, list[np.ndarray]] = {
        "latent": [],
        "pi": [],
        "mu": [],
        "sigma": [],
        "alpha": [],
        "targets": [],
        "batches": [],
        "row_ids": [],
    }
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_batch_to_device(cpu_batch, device)
            predictions = model(batch, grl_strength=0.0)
            for key in ("latent", "pi", "mu", "sigma", "alpha"):
                outputs[key].append(predictions[key].cpu().numpy())
            for key in ("targets", "batches", "row_ids"):
                outputs[key].append(batch[key].cpu().numpy())
    return {key: np.concatenate(parts, axis=0) for key, parts in outputs.items()}


def compute_peak_activity(
    path: Path,
    matrix_key: str,
    num_peaks: int,
    chunk_size: int,
    max_nonzeros: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Count peak occurrences and summed intensities directly from CSR storage.

    Args:
        path (Path): AnnData HDF5 path.
        matrix_key (str): ``X``, a layer name, or ``layers/<name>``.
        num_peaks (int): Number of aligned m/z bins.
        chunk_size (int): Number of sparse entries read per chunk.
        max_nonzeros (int | None): Optional scan cap for smoke diagnostics.

    Returns:
        tuple[np.ndarray, np.ndarray]: Occurrence counts and intensity sums per peak.
    """

    group_path = (
        "X"
        if matrix_key == "X"
        else matrix_key
        if matrix_key.startswith("layers/")
        else f"layers/{matrix_key}"
    )
    counts = np.zeros(num_peaks, dtype=np.int64)
    intensity_sums = np.zeros(num_peaks, dtype=np.float64)
    with h5py.File(path, "r") as handle:
        group = handle[group_path]
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
    """Compute all-pair cosine similarities and hierarchical peptide families.

    Args:
        embedding_weights (np.ndarray): Learned matrix ``[num_peaks, embedding_dim]``.
        family_count (int): Requested number of flat clusters.
        linkage_method (str): SciPy hierarchical linkage method.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Similarity matrix, one-based family
        labels, and hierarchical leaf order.
    """

    weights = np.asarray(embedding_weights, dtype=np.float32)
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    normalized = weights / np.maximum(norms, np.finfo(np.float32).eps)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    condensed = distance.squareform(1.0 - similarity, checks=False)
    linkage = hierarchy.linkage(condensed, method=linkage_method)
    labels = hierarchy.fcluster(linkage, t=family_count, criterion="maxclust")
    order = hierarchy.leaves_list(linkage)
    return similarity, labels.astype(np.int32), order.astype(np.int64)


def fit_latent_umap(latent: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    """Fit a two-dimensional UMAP embedding of latent vectors.

    Args:
        latent (np.ndarray): Extracted latent matrix with shape ``[n_samples, latent_dim]``.
        config (Mapping[str, Any]): Analysis settings including UMAP hyperparameters.

    Returns:
        np.ndarray: Two-dimensional UMAP coordinates with shape ``[n_samples, 2]``.
    """

    neighbors = min(int(config["umap_neighbors"]), max(latent.shape[0] - 1, 2))
    reducer = umap.UMAP(
        n_neighbors=neighbors,
        min_dist=float(config["umap_min_dist"]),
        metric=config["umap_metric"],
        random_state=int(config["umap_random_state"]),
    )
    return reducer.fit_transform(latent)


def _batch_name_series(
    batches: np.ndarray, batch_names: Sequence[str]
) -> pd.Series:
    """Map encoded batch IDs to display names.

    Args:
        batches (np.ndarray): Integer batch codes with shape ``[n_samples]``.
        batch_names (Sequence[str]): Display names indexed by batch code.

    Returns:
        pd.Series: Batch name per sample, aligned with ``batches``.
    """

    names = np.asarray(batch_names, dtype=object)
    codes = np.asarray(batches, dtype=np.int64)
    return pd.Series(names[codes], name="batch")


def build_latent_umap_frame(
    extracted: Mapping[str, np.ndarray],
    umap_coordinates: np.ndarray,
    target_columns: Sequence[str],
    batch_names: Sequence[str],
    split: str,
) -> pd.DataFrame:
    """Build a sample-aligned table of UMAP coordinates, labels, and ZILN outputs.

    Args:
        extracted (Mapping[str, np.ndarray]): Arrays from ``extract_latent_predictions``.
        umap_coordinates (np.ndarray): Two-dimensional UMAP coords ``[n_samples, 2]``.
        target_columns (Sequence[str]): Ordered IHC density column names.
        batch_names (Sequence[str]): Display names for encoded batch IDs.
        split (str): Name of the analysis split written into the table.

    Returns:
        pd.DataFrame: One row per sample with identity, batch, UMAP, targets, and ZILN.
    """

    frame = pd.DataFrame(
        {
            "row_id": np.asarray(extracted["row_ids"], dtype=np.int64),
            "split": split,
            "batch_id": np.asarray(extracted["batches"], dtype=np.int64),
            "batch": _batch_name_series(extracted["batches"], batch_names),
            "umap_1": umap_coordinates[:, 0],
            "umap_2": umap_coordinates[:, 1],
        }
    )
    targets = np.asarray(extracted["targets"])
    pi = np.asarray(extracted["pi"])
    mu = np.asarray(extracted["mu"])
    sigma = np.asarray(extracted["sigma"])
    alpha = np.asarray(extracted["alpha"])
    for index, column in enumerate(target_columns):
        frame[column] = targets[:, index]
        frame[f"pi_{column}"] = pi[:, index]
        frame[f"mu_{column}"] = mu[:, index]
        frame[f"sigma_{column}"] = sigma[:, index]
        frame[f"alpha_{column}"] = alpha[:, index]
    return frame


def build_latent_embeddings_frame(
    extracted: Mapping[str, np.ndarray],
    batch_names: Sequence[str],
    split: str,
) -> pd.DataFrame:
    """Build a sample-aligned table of full latent bottleneck vectors.

    Args:
        extracted (Mapping[str, np.ndarray]): Arrays from ``extract_latent_predictions``.
        batch_names (Sequence[str]): Display names for encoded batch IDs.
        split (str): Name of the analysis split written into the table.

    Returns:
        pd.DataFrame: One row per sample with identity, batch, and ``latent_*`` columns.
    """

    latent = np.asarray(extracted["latent"])
    frame = pd.DataFrame(
        {
            "row_id": np.asarray(extracted["row_ids"], dtype=np.int64),
            "split": split,
            "batch_id": np.asarray(extracted["batches"], dtype=np.int64),
            "batch": _batch_name_series(extracted["batches"], batch_names),
        }
    )
    for dim in range(latent.shape[1]):
        frame[f"latent_{dim}"] = latent[:, dim]
    return frame


def build_ziln_scatter_frame(
    extracted: Mapping[str, np.ndarray],
    target_columns: Sequence[str],
    batch_names: Sequence[str],
    split: str,
    epsilon: float,
) -> pd.DataFrame:
    """Build a long-format table of positive-density ZILN scatter points.

    Only rows with true density ``> 0`` are retained, matching the scatter plot.

    Args:
        extracted (Mapping[str, np.ndarray]): Arrays from ``extract_latent_predictions``.
        target_columns (Sequence[str]): Ordered IHC density column names.
        batch_names (Sequence[str]): Display names for encoded batch IDs.
        split (str): Name of the scatter split written into the table.
        epsilon (float): Boundary clamp applied before the true-value logit.

    Returns:
        pd.DataFrame: Long-format rows with true density, true logit, and predicted mu.
    """

    row_ids = np.asarray(extracted["row_ids"], dtype=np.int64)
    batch_ids = np.asarray(extracted["batches"], dtype=np.int64)
    batch_labels = _batch_name_series(batch_ids, batch_names).to_numpy()
    targets = np.asarray(extracted["targets"])
    mu = np.asarray(extracted["mu"])
    parts: list[pd.DataFrame] = []
    for index, column in enumerate(target_columns):
        positive = targets[:, index] > 0.0
        if not np.any(positive):
            continue
        truth = targets[positive, index]
        parts.append(
            pd.DataFrame(
                {
                    "row_id": row_ids[positive],
                    "split": split,
                    "batch_id": batch_ids[positive],
                    "batch": batch_labels[positive],
                    "target": column,
                    "true_density": truth,
                    "true_logit": _positive_logit_densities(truth, epsilon),
                    "predicted_mu": mu[positive, index],
                }
            )
        )
    if not parts:
        return pd.DataFrame(
            columns=[
                "row_id",
                "split",
                "batch_id",
                "batch",
                "target",
                "true_density",
                "true_logit",
                "predicted_mu",
            ]
        )
    return pd.concat(parts, ignore_index=True)


def _categorical_batch_colors(num_batches: int) -> np.ndarray:
    """Build high-contrast colors for many categorical batch labels.

    Stacks matplotlib ``tab20`` / ``tab20b`` / ``tab20c`` (60 colors). When
    ``num_batches`` exceeds that pool, appends HUSL samples for the remainder.

    Args:
        num_batches (int): Number of distinct batch categories to color.

    Returns:
        np.ndarray: RGBA colors with shape ``[num_batches, 4]``.
    """

    count = max(int(num_batches), 1)
    base = np.vstack(
        [
            np.asarray(plt.cm.tab20.colors, dtype=np.float64),
            np.asarray(plt.cm.tab20b.colors, dtype=np.float64),
            np.asarray(plt.cm.tab20c.colors, dtype=np.float64),
        ]
    )
    if count <= base.shape[0]:
        rgb = base[:count]
    else:
        extra = np.asarray(
            sns.color_palette("husl", n_colors=count - base.shape[0]),
            dtype=np.float64,
        )
        rgb = np.vstack([base, extra])
    alpha = np.ones((count, 1), dtype=np.float64)
    return np.concatenate([rgb, alpha], axis=1)


def _style_umap_axis(axis: Axes, title: str) -> None:
    """Apply shared UMAP panel styling: title only, no axes or spines.

        Args:
        axis (Axes): Matplotlib axes to style.
        title (str): Panel title text.

    Returns:
        None: ``axis`` is mutated in place.
    """

    axis.set_title(title, fontsize=14)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_latent_umap_batch(
    coordinates: np.ndarray,
    batches: np.ndarray,
    batch_names: Sequence[str],
    output_path: Path,
    config: Mapping[str, Any],
) -> None:
    """Plot latent UMAP colored by AnnData batch category names.

    Args:
        coordinates (np.ndarray): Precomputed UMAP coordinates.
        batches (np.ndarray): Encoded integer batch labels.
        batch_names (Sequence[str]): Display names for encoded batches.
        output_path (Path): Destination PNG path.
        config (Mapping[str, Any]): Analysis settings.

    Returns:
        None: Figure is saved to disk.
    """

    point_size = float(config["point_size"])
    num_batches = len(batch_names)
    colors = _categorical_batch_colors(num_batches)
    figure, axis = plt.subplots(figsize=(18, 9), constrained_layout=True)
    handles = []
    for batch_id, batch_name in enumerate(batch_names):
        mask = batches == batch_id
        if not np.any(mask):
            continue
        handle = axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            c=[colors[batch_id]],
            s=point_size,
            label=batch_name,
            rasterized=True,
        )
        handles.append(handle)
    _style_umap_axis(axis, f"Latent UMAP by batch ({num_batches} batches)")
    axis.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        markerscale=4.0,
        frameon=False,
        ncol=1,
    )
    figure.savefig(output_path, dpi=int(config["figure_dpi"]))
    plt.close(figure)


def _positive_logit_densities(values: np.ndarray, epsilon: float) -> np.ndarray:
    """Map positive densities onto the clamped logit scale used by ZILN.

    Args:
        values (np.ndarray): Strictly positive density values in ``(0, 1]``.
        epsilon (float): Boundary clamp applied before the logit transform.

    Returns:
        np.ndarray: Logit-transformed densities matching ``values`` shape.
    """

    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def _robust_color_limits(
    values: np.ndarray, lower_percentile: float, upper_percentile: float
) -> tuple[float, float]:
    """Compute outlier-resistant colormap bounds from value percentiles.

    Args:
        values (np.ndarray): Finite values used to color points.
        lower_percentile (float): Lower percentile in ``[0, 100]``.
        upper_percentile (float): Upper percentile in ``[0, 100]``.

    Returns:
        tuple[float, float]: Inclusive ``(vmin, vmax)`` color limits.
    """

    if values.size == 0:
        return 0.0, 1.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lower = float(np.percentile(finite, lower_percentile))
    upper = float(np.percentile(finite, upper_percentile))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        span = float(np.nanmax(finite) - np.nanmin(finite))
        center = float(np.nanmedian(finite))
        pad = max(abs(center) * 1e-3, span * 0.5, 1e-6)
        return center - pad, center + pad
    return lower, upper


def _symmetric_zero_color_limits(
    values: np.ndarray, lower_percentile: float, upper_percentile: float
) -> tuple[float, float]:
    """Return symmetric ``(-limit, limit)`` bounds so colormap center is 0.

    Args:
        values (np.ndarray): Finite values used to color points.
        lower_percentile (float): Lower percentile in ``[0, 100]``.
        upper_percentile (float): Upper percentile in ``[0, 100]``.

    Returns:
        tuple[float, float]: Inclusive ``(vmin, vmax)`` with ``vmax == -vmin``.
    """

    lower, upper = _robust_color_limits(values, lower_percentile, upper_percentile)
    limit = max(abs(lower), abs(upper), 1e-6)
    return -limit, limit


def plot_latent_umap_densities(
    coordinates: np.ndarray,
    targets: np.ndarray,
    target_columns: Sequence[str],
    output_path: Path,
    config: Mapping[str, Any],
    logit_epsilon: float | None = None,
) -> None:
    """Plot latent UMAP panels colored by each raw or logit IHC density target.

    Only points with density ``> 0`` are drawn so zeros do not saturate the cmap.
    When ``logit_epsilon`` is provided, positive densities are clamped and mapped
    with ``logit`` before coloring. Color limits use configured percentiles so
    extreme outliers do not dominate the colormap.

    Args:
        coordinates (np.ndarray): Precomputed UMAP coordinates.
        targets (np.ndarray): Raw density matrix with shape ``[n_samples, n_targets]``.
        target_columns (Sequence[str]): Ordered density column names matching ``targets``.
        output_path (Path): Destination PNG path.
        config (Mapping[str, Any]): Analysis settings.
        logit_epsilon (float | None): Optional clamp for logit coloring; ``None``
            keeps the raw density scale.

    Returns:
        None: Figure is saved to disk.
    """

    num_targets = len(target_columns)
    if targets.shape[1] != num_targets:
        raise ValueError(
            f"Target matrix width {targets.shape[1]} does not match "
            f"{num_targets} target columns."
        )
    ncols = min(2, num_targets)
    nrows = int(np.ceil(num_targets / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12 * ncols, 6 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    point_size = float(config["point_size"]) * 0.35
    cmap = str(config["density_cmap"])
    lower_percentile = float(config["density_vmin_percentile"])
    upper_percentile = float(config["density_vmax_percentile"])
    use_logit = logit_epsilon is not None
    for index, column in enumerate(target_columns):
        row, col = divmod(index, ncols)
        axis = axes[row][col]
        positive = targets[:, index] > 0.0
        values = targets[positive, index]
        if use_logit and values.size > 0:
            values = _positive_logit_densities(values, float(logit_epsilon))
        vmin, vmax = _robust_color_limits(values, lower_percentile, upper_percentile)
        scatter = axis.scatter(
            coordinates[positive, 0],
            coordinates[positive, 1],
            c=values,
            s=point_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        scale_label = f"logit({column})" if use_logit else column
        _style_umap_axis(axis, f"Latent UMAP by {scale_label}")
        if values.size > 0:
            figure.colorbar(scatter, ax=axis)
    for index in range(num_targets, nrows * ncols):
        row, col = divmod(index, ncols)
        axes[row][col].axis("off")
    figure.savefig(output_path, dpi=int(config["figure_dpi"]))
    plt.close(figure)


def plot_latent_umap_ziln_per_target(
    coordinates: np.ndarray,
    densities: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    alpha: np.ndarray,
    batches: np.ndarray,
    batch_names: Sequence[str],
    target_column: str,
    output_path: Path,
    config: Mapping[str, Any],
    logit_epsilon: float,
) -> None:
    """Plot a 2x3 latent UMAP for one target's density, parameters, and batches.

    All samples are drawn in every panel. For the logit panel, zero densities
    are shown in light gray and excluded from color-limit / colorbar scaling so
    they do not collapse the positive logit range. The existence panel colors by
    ``1 - pi`` (probability of a non-zero density). The categorical batch panel
    intentionally omits a legend because the standalone batch UMAP provides it.

    Args:
        coordinates (np.ndarray): Precomputed UMAP coordinates ``[n_samples, 2]``.
        densities (np.ndarray): True density values ``[n_samples]`` for ``target_column``.
        pi (np.ndarray): Predicted structural-zero probabilities ``[n_samples]``.
        mu (np.ndarray): Predicted logit-normal means ``[n_samples]``.
        sigma (np.ndarray): Predicted logit-normal standard deviations ``[n_samples]``.
        alpha (np.ndarray): Predicted skewness parameters ``[n_samples]``.
        batches (np.ndarray): Encoded integer batch labels ``[n_samples]``.
        batch_names (Sequence[str]): Display names for encoded batches.
        target_column (str): IHC density column name (e.g. ``Density_CD8``).
        output_path (Path): Destination PNG path.
        config (Mapping[str, Any]): Analysis plot settings.
        logit_epsilon (float): Clamp applied before the positive-density logit.

    Returns:
        None: Figure is saved to disk.
    """

    densities = np.asarray(densities, dtype=np.float64).reshape(-1)
    pi = np.asarray(pi, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    batches = np.asarray(batches, dtype=np.int64).reshape(-1)
    n_samples = coordinates.shape[0]
    for name, values in (
        ("densities", densities),
        ("pi", pi),
        ("mu", mu),
        ("sigma", sigma),
        ("alpha", alpha),
        ("batches", batches),
    ):
        if values.shape[0] != n_samples:
            raise ValueError(
                f"{name} length {values.shape[0]} does not match "
                f"{n_samples} UMAP coordinates."
            )

    ncols = 3
    nrows = 2
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12 * ncols, 6 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    point_size = float(config["point_size"]) * 0.35
    cmap = str(config["density_cmap"])
    lower_percentile = float(config["density_vmin_percentile"])
    upper_percentile = float(config["density_vmax_percentile"])
    zero_color = "#d0d0d0"

    logit_label = f"Logit_{target_column}"
    positive = densities > 0.0
    zero = ~positive
    axis = axes[0][0]
    if np.any(zero):
        axis.scatter(
            coordinates[zero, 0],
            coordinates[zero, 1],
            c=zero_color,
            s=point_size,
            rasterized=True,
        )
    positive_logits = (
        _positive_logit_densities(densities[positive], logit_epsilon)
        if np.any(positive)
        else np.asarray([], dtype=np.float64)
    )
    vmin, vmax = _robust_color_limits(
        positive_logits, lower_percentile, upper_percentile
    )
    scatter = None
    if np.any(positive):
        scatter = axis.scatter(
            coordinates[positive, 0],
            coordinates[positive, 1],
            c=positive_logits,
            s=point_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
    if scatter is not None:
        figure.colorbar(scatter, ax=axis)
    _style_umap_axis(axis, f"Latent UMAP by {logit_label}")

    parameter_panels = (
        (axes[0][1], 1.0 - pi, f"1-pi_{target_column}", cmap),
        (axes[0][2], mu, f"mu_{target_column}", cmap),
        (axes[1][0], sigma, f"sigma_{target_column}", cmap),
    )
    for axis, values, label, panel_cmap in parameter_panels:
        vmin, vmax = _robust_color_limits(values, lower_percentile, upper_percentile)
        scatter = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=values,
            s=point_size,
            cmap=panel_cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        _style_umap_axis(axis, f"Latent UMAP by {label}")
        figure.colorbar(scatter, ax=axis)

    alpha_axis = axes[1][1]
    alpha_label = f"alpha_{target_column}"
    vmin, vmax = _symmetric_zero_color_limits(
        alpha, lower_percentile, upper_percentile
    )
    scatter = alpha_axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=alpha,
        s=point_size,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    _style_umap_axis(alpha_axis, f"Latent UMAP by {alpha_label}")
    figure.colorbar(scatter, ax=alpha_axis)

    batch_axis = axes[1][2]
    batch_colors = _categorical_batch_colors(len(batch_names))
    for batch_id in range(len(batch_names)):
        mask = batches == batch_id
        if not np.any(mask):
            continue
        batch_axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            c=[batch_colors[batch_id]],
            s=point_size,
            rasterized=True,
        )
    _style_umap_axis(batch_axis, "Latent UMAP by batch")

    figure.savefig(output_path, dpi=int(config["figure_dpi"]))
    plt.close(figure)


def plot_similarity_heatmap(
    similarity: np.ndarray,
    activity: np.ndarray,
    hierarchy_order: np.ndarray,
    mz_values: np.ndarray,
    output_path: Path,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Plot clustered embedding cosine similarity for the most active peaks.

    Args:
        similarity (np.ndarray): Full all-peak cosine similarity matrix.
        activity (np.ndarray): Peak occurrence counts.
        hierarchy_order (np.ndarray): Global hierarchical leaf ordering.
        mz_values (np.ndarray): Numeric m/z values.
        output_path (Path): Destination PNG path.
        config (Mapping[str, Any]): Analysis settings.

    Returns:
        np.ndarray: Ordered peak indices displayed in the heatmap.
    """

    top_count = min(int(config["heatmap_top_peaks"]), activity.size)
    active = np.argpartition(activity, -top_count)[-top_count:]
    rank = np.empty(hierarchy_order.size, dtype=np.int64)
    rank[hierarchy_order] = np.arange(hierarchy_order.size)
    ordered = active[np.argsort(rank[active])]
    subset = similarity[np.ix_(ordered, ordered)]
    figure, axis = plt.subplots(figsize=(12, 10), constrained_layout=True)
    sns.heatmap(
        subset,
        ax=axis,
        cmap="vlag",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Cosine similarity"},
    )
    axis.set_title(f"Learned peptide similarity: {top_count} most active m/z bins")
    axis.set_xlabel(
        f"Hierarchically ordered peaks ({mz_values[ordered].min():.1f}–"
        f"{mz_values[ordered].max():.1f} m/z)"
    )
    axis.set_ylabel("Peaks")
    figure.savefig(output_path, dpi=int(config["figure_dpi"]))
    plt.close(figure)
    return ordered


def plot_loss_curves(
    history_path: Path,
    output_path: Path,
    figure_dpi: int,
    best_epoch: int | float | None = None,
) -> None:
    """Plot train/validation objective components and batch accuracy.

    A dashed red vertical line marks the best-model epoch selected by minimum
    validation biology loss when ``best_epoch`` is omitted.

    Args:
        history_path (Path): Training history CSV.
        output_path (Path): Destination PNG path.
        figure_dpi (int): Saved figure resolution.
        best_epoch (int | float | None): One-based epoch of the chosen best
            checkpoint. When ``None``, uses the epoch with the lowest
            ``validation_biology_loss`` in ``history_path``.

    Returns:
        None: Figure is saved to disk.
    """

    history = pd.read_csv(history_path)
    if best_epoch is None:
        best_epoch = float(
            history.loc[history["validation_biology_loss"].idxmin(), "epoch"]
        )
    else:
        best_epoch = float(best_epoch)
    panels = [
        ("total_loss", "Total objective"),
        ("biology_loss", "ZILN biology"),
        ("hurdle_loss", "Zero hurdle"),
        ("positive_loss", "Positive logit-normal"),
        ("batch_loss", "Batch discriminator"),
        ("batch_accuracy", "Batch accuracy"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, panels, strict=True):
        axis.plot(history["epoch"], history[f"train_{metric}"], label="Train")
        axis.plot(
            history["epoch"], history[f"validation_{metric}"], label="Validation"
        )
        axis.axvline(
            best_epoch,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Best model (epoch {int(best_epoch)})",
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output_path, dpi=figure_dpi)
    plt.close(figure)


def plot_ziln_density_scatter(
    predicted_mu: np.ndarray,
    true_targets: np.ndarray,
    target_columns: Sequence[str],
    epsilon: float,
    output_path: Path,
    config: Mapping[str, Any],
) -> None:
    """Plot predicted mu against positive true logit density for each target.

    Args:
        predicted_mu (np.ndarray): Predicted logit-normal means with shape
            ``[n_samples, n_targets]``.
        true_targets (np.ndarray): Raw bounded density targets with shape
            ``[n_samples, n_targets]``.
        target_columns (Sequence[str]): Ordered density column names matching
            ``predicted_mu`` and ``true_targets``.
        epsilon (float): Boundary clamp before true-value logits.
        output_path (Path): Destination PNG path.
        config (Mapping[str, Any]): Plot settings.

    Returns:
        None: Figure is saved to disk.
    """

    num_targets = len(target_columns)
    if predicted_mu.shape[1] != num_targets or true_targets.shape[1] != num_targets:
        raise ValueError(
            f"Prediction/target widths ({predicted_mu.shape[1]}, "
            f"{true_targets.shape[1]}) do not match {num_targets} target columns."
        )
    ncols = min(2, num_targets)
    nrows = int(np.ceil(num_targets / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12 * ncols, 6 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    point_size = float(config["point_size"])
    for index, column in enumerate(target_columns):
        row, col = divmod(index, ncols)
        axis = axes[row][col]
        positive = true_targets[:, index] > 0.0
        truth = _positive_logit_densities(true_targets[positive, index], epsilon)
        predictions = predicted_mu[positive, index]
        axis.scatter(
            truth,
            predictions,
            s=point_size,
            alpha=0.25,
            rasterized=True,
        )
        if truth.size > 0 and predictions.size > 0:
            lower = float(min(truth.min(), predictions.min()))
            upper = float(max(truth.max(), predictions.max()))
            axis.plot(
                [lower, upper],
                [lower, upper],
                linestyle="--",
                color="black",
                linewidth=1,
            )
        correlation = (
            np.corrcoef(truth, predictions)[0, 1] if truth.size > 1 else np.nan
        )
        axis.set(
            title=f"Positive {column} ZILN (Pearson r={correlation:.3f})",
            xlabel=f"True logit({column})",
            ylabel=f"Predicted mu ({column})",
        )
        axis.grid(alpha=0.25)
    for index in range(num_targets, nrows * ncols):
        row, col = divmod(index, ncols)
        axes[row][col].axis("off")
    figure.savefig(output_path, dpi=int(config["figure_dpi"]))
    plt.close(figure)


def run_analysis(config: Mapping[str, Any], checkpoint_override: Path | None) -> Path:
    """Run all checkpoint interpretation exports and required visualizations.

    Args:
        config (Mapping[str, Any]): Effective DANN configuration.
        checkpoint_override (Path | None): Optional CLI checkpoint path.

    Returns:
        Path: Analysis output directory.
    """

    config = _analysis_config(config)
    seed_everything(int(config["training"]["seed"]))
    device = resolve_device(str(config["training"]["device"]))
    bundle = create_data_bundle(config)
    configured_checkpoint = config["analysis"].get("checkpoint")
    checkpoint_path = checkpoint_override or (
        Path(configured_checkpoint)
        if configured_checkpoint
        else Path(config["training"]["output_dir"]) / "best.pt"
    )
    model, checkpoint_epoch = load_model(checkpoint_path, config, bundle, device)
    split = str(config["analysis"]["split"])
    extracted = extract_latent_predictions(model, bundle.loaders[split], device)
    output_dir = Path(config["analysis"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    umap_dir = output_dir / "umap"
    umap_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(config["training"]["output_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    target_columns = list(config["data"]["target_columns"])
    batch_names = bundle.metadata.batch_names
    umap_coordinates = fit_latent_umap(extracted["latent"], config["analysis"])
    build_latent_umap_frame(
        extracted,
        umap_coordinates,
        target_columns,
        batch_names,
        split,
    ).to_csv(umap_dir / "latent_umap.csv", index=False)
    build_latent_embeddings_frame(
        extracted,
        batch_names,
        split,
    ).to_csv(output_dir / "latent_embeddings.csv", index=False)
    plot_latent_umap_batch(
        umap_coordinates,
        extracted["batches"],
        batch_names,
        umap_dir / "latent_umap_batch.png",
        config["analysis"],
    )
    plot_latent_umap_densities(
        umap_coordinates,
        extracted["targets"],
        target_columns,
        umap_dir / "latent_umap_densities.png",
        config["analysis"],
    )
    logit_epsilon = float(config["loss"]["logit_epsilon"])
    plot_latent_umap_densities(
        umap_coordinates,
        extracted["targets"],
        target_columns,
        umap_dir / "latent_umap_densities_logit.png",
        config["analysis"],
        logit_epsilon=logit_epsilon,
    )
    for index, column in enumerate(target_columns):
        plot_latent_umap_ziln_per_target(
            umap_coordinates,
            extracted["targets"][:, index],
            extracted["pi"][:, index],
            extracted["mu"][:, index],
            extracted["sigma"][:, index],
            extracted["alpha"][:, index],
            extracted["batches"],
            batch_names,
            column,
            umap_dir / f"latent_umap_ziln_{column}.png",
            config["analysis"],
            logit_epsilon,
        )
    scatter_split = str(config["analysis"]["ziln_scatter_split"])
    scatter_data = (
        extracted
        if scatter_split == split
        else extract_latent_predictions(model, bundle.loaders[scatter_split], device)
    )
    build_ziln_scatter_frame(
        scatter_data,
        target_columns,
        batch_names,
        scatter_split,
        logit_epsilon,
    ).to_csv(output_dir / "ziln_density_scatter.csv", index=False)
    plot_ziln_density_scatter(
        scatter_data["mu"],
        scatter_data["targets"],
        target_columns,
        logit_epsilon,
        output_dir / "ziln_density_scatter.png",
        config["analysis"],
    )

    embedding_weights = model.encoder.embedding.weight.detach().cpu().numpy()
    similarity, families, hierarchy_order = embedding_cosine_families(
        embedding_weights,
        int(config["analysis"]["family_count"]),
        str(config["analysis"]["clustering_linkage"]),
    )
    np.save(output_dir / "embedding_cosine_similarity.npy", similarity)
    activity, intensity_sums = compute_peak_activity(
        Path(config["data"]["path"]),
        str(config["data"]["matrix_key"]),
        bundle.metadata.num_peaks,
        int(config["analysis"]["activity_chunk_size"]),
        config["analysis"].get("activity_max_nonzeros"),
    )
    displayed = plot_similarity_heatmap(
        similarity,
        activity,
        hierarchy_order,
        bundle.metadata.mz_values,
        output_dir / "peptide_similarity_heatmap.png",
        config["analysis"],
    )
    family_table = pd.DataFrame(
        {
            "peak_index": np.arange(bundle.metadata.num_peaks),
            "mz": bundle.metadata.mz_values,
            "family": families,
            "occurrence_count": activity,
            "intensity_sum": intensity_sums,
            "embedding_norm": np.linalg.norm(embedding_weights, axis=1),
            "shown_in_heatmap": np.isin(
                np.arange(bundle.metadata.num_peaks), displayed
            ),
        }
    )
    family_table.to_csv(output_dir / "peptide_families.csv", index=False)
    history_path = model_dir / "history.csv"
    best_checkpoint_path = model_dir / "best.pt"
    if checkpoint_path.resolve() == best_checkpoint_path.resolve():
        best_epoch = checkpoint_epoch
    elif best_checkpoint_path.is_file():
        best_payload = torch.load(
            best_checkpoint_path, map_location="cpu", weights_only=False
        )
        best_epoch = int(best_payload["epoch"]) + 1
    else:
        best_epoch = None
    plot_loss_curves(
        history_path,
        model_dir / "loss_curves.png",
        int(config["analysis"]["figure_dpi"]),
        best_epoch=best_epoch,
    )
    for dataset in bundle.datasets.values():
        dataset.close()
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Load configuration and execute post-training analysis.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        int: Process exit status, zero on success.
    """

    args = parse_args(argv)
    config = load_config(args.config)
    if args.smoke_test:
        config = apply_smoke_overrides(config)
    output_dir = run_analysis(config, args.checkpoint)
    print(f"Analysis written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

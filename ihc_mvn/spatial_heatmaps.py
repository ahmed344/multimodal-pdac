"""Plot MVN-aware spatial heatmaps from ordered tissue inference artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

from .config import load_config
from .targets import TARGET_COLUMNS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse spatial-visualization command-line arguments.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        argparse.Namespace: Parsed config and path overrides.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="IHC-MVN YAML configuration path.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Tissue H5AD override.")
    parser.add_argument(
        "--predictions", type=Path, default=None, help="Prediction Parquet override."
    )
    parser.add_argument(
        "--latent", type=Path, default=None, help="Latent Parquet override."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Figure directory override."
    )
    return parser.parse_args(argv)


def validate_ordered_frame(
    adata: ad.AnnData,
    frame: pd.DataFrame,
    artifact_name: str,
) -> None:
    """Validate an ordered inference table against an AnnData object.

    Args:
        adata (ad.AnnData): Source tissue observations.
        frame (pd.DataFrame): Prediction or latent table.
        artifact_name (str): Human-readable artifact label for errors.

    Returns:
        None: Exact row and observation-name alignment is confirmed.
    """

    required = {"row_position", "obs_name"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{artifact_name} lacks identity columns: {sorted(missing)}")
    if len(frame) != adata.n_obs:
        raise ValueError(
            f"{artifact_name} has {len(frame)} rows but AnnData has {adata.n_obs}."
        )
    expected = np.arange(adata.n_obs, dtype=np.int64)
    if not np.array_equal(frame["row_position"].to_numpy(dtype=np.int64), expected):
        raise ValueError(f"{artifact_name} row positions are not contiguous.")
    if not np.array_equal(
        frame["obs_name"].to_numpy(dtype=str),
        adata.obs_names.to_numpy(dtype=str),
    ):
        raise ValueError(f"{artifact_name} observation names do not match AnnData.")


def attach_predictions(
    adata: ad.AnnData,
    predictions: pd.DataFrame,
) -> list[str]:
    """Attach aligned numeric prediction columns to AnnData observations.

    Args:
        adata (ad.AnnData): Source tissue AnnData.
        predictions (pd.DataFrame): Ordered prediction table.

    Returns:
        list[str]: Prediction columns added to ``adata.obs``.
    """

    validate_ordered_frame(adata, predictions, "Prediction Parquet")
    columns = [
        name for name in predictions.columns if name not in {"row_position", "obs_name"}
    ]
    for column in columns:
        values = predictions[column].to_numpy()
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"Prediction column {column!r} must be numeric.")
        adata.obs[column] = values
    return columns


def attach_coordinates(
    adata: ad.AnnData,
    x_column: str,
    y_column: str,
) -> tuple[str, str]:
    """Ensure configured spatial coordinate columns exist.

    Args:
        adata (ad.AnnData): Tissue AnnData.
        x_column (str): Desired horizontal observation column.
        y_column (str): Desired vertical observation column.

    Returns:
        tuple[str, str]: Coordinate column names available in ``adata.obs``.
    """

    if x_column in adata.obs and y_column in adata.obs:
        return x_column, y_column
    if "spatial" not in adata.obsm:
        raise KeyError(
            f"AnnData lacks obs columns {x_column!r}/{y_column!r} and obsm['spatial']."
        )
    coordinates = np.asarray(adata.obsm["spatial"])
    if coordinates.ndim != 2 or coordinates.shape != (adata.n_obs, 2):
        raise ValueError("obsm['spatial'] must have shape [n_obs, 2].")
    adata.obs[x_column] = coordinates[:, 0]
    adata.obs[y_column] = coordinates[:, 1]
    return x_column, y_column


def add_spatial_derived_columns(
    adata: ad.AnnData,
    targets: Sequence[str],
) -> list[str]:
    """Create density summaries, uncertainty widths, and optional residuals.

    Args:
        adata (ad.AnnData): AnnData with attached inference columns.
        targets (Sequence[str]): Ordered modeled target names.

    Returns:
        list[str]: Names of newly created observation columns.
    """

    created: list[str] = []
    for target in targets:
        required = (
            f"haldane_mean_msi_{target}",
            f"haldane_mean_patient_{target}",
            f"haldane_mean_total_{target}",
            f"q05_density_{target}",
            f"q95_density_{target}",
            f"positive_median_density_{target}",
        )
        missing = [column for column in required if column not in adata.obs]
        if missing:
            raise KeyError(f"Missing prediction columns for {target!r}: {missing}")
        for layer in ("msi", "patient", "total"):
            column = f"positive_median_{layer}_{target}"
            adata.obs[column] = expit(
                adata.obs[f"haldane_mean_{layer}_{target}"].to_numpy(dtype=np.float64)
            ).astype(np.float32)
            created.append(column)
        width_column = f"interval_width_{target}"
        adata.obs[width_column] = (
            adata.obs[f"q95_density_{target}"].to_numpy(dtype=np.float64)
            - adata.obs[f"q05_density_{target}"].to_numpy(dtype=np.float64)
        ).astype(np.float32)
        created.append(width_column)
        if target in adata.obs:
            residual_column = f"density_residual_{target}"
            residual = (
                adata.obs[f"positive_median_density_{target}"].to_numpy(
                    dtype=np.float64
                )
                - adata.obs[target].to_numpy(dtype=np.float64)
            )
            residual[adata.obs[target].to_numpy(dtype=np.float64) <= 0.0] = np.nan
            adata.obs[residual_column] = residual.astype(np.float32)
            created.append(residual_column)
    return created


def _color_limits(
    values: np.ndarray,
    *,
    diverging: bool = False,
) -> tuple[float, float]:
    """Compute robust spatial color limits.

    Args:
        values (np.ndarray): Numeric values, possibly containing NaNs.
        diverging (bool): Whether limits should be symmetric around zero.

    Returns:
        tuple[float, float]: Lower and upper plotting limits.
    """

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0) if diverging else (0.0, 1.0)
    if diverging:
        maximum = max(float(np.percentile(np.abs(finite), 99.0)), 1.0e-8)
        return -maximum, maximum
    lower, upper = np.percentile(finite, [1.0, 99.0])
    if lower == upper:
        upper = lower + 1.0
    return float(lower), float(upper)


def _plot_spatial_values(
    axis: plt.Axes,
    observations: pd.DataFrame,
    x_column: str,
    y_column: str,
    value_column: str,
    cmap: str,
    *,
    diverging: bool = False,
) -> None:
    """Plot one coordinate-aligned spatial value panel.

    Args:
        axis (plt.Axes): Destination axis.
        observations (pd.DataFrame): One-slide observation rows.
        x_column (str): Horizontal coordinate column.
        y_column (str): Vertical coordinate column.
        value_column (str): Numeric color column.
        cmap (str): Matplotlib colormap name.
        diverging (bool): Whether to use symmetric zero-centered limits.

    Returns:
        None: The axis is modified in place.
    """

    values = observations[value_column].to_numpy(dtype=np.float64)
    vmin, vmax = _color_limits(values, diverging=diverging)
    image = axis.scatter(
        observations[x_column],
        observations[y_column],
        c=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=2,
        marker="s",
        linewidths=0,
        rasterized=True,
    )
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    axis.set_facecolor("lightgray")
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.set_xticks([])
    axis.set_yticks([])


def _hes_image(adata: ad.AnnData, batch: str) -> np.ndarray | None:
    """Find an HES image for one spatial library when available.

    Args:
        adata (ad.AnnData): Tissue AnnData with optional Scanpy spatial metadata.
        batch (str): Spatial library/batch identifier.

    Returns:
        np.ndarray | None: Image array or ``None`` when unavailable.
    """

    spatial = adata.uns.get("spatial")
    if not isinstance(spatial, Mapping):
        return None
    library = spatial.get(str(batch))
    if not isinstance(library, Mapping):
        return None
    images = library.get("images")
    if not isinstance(images, Mapping):
        return None
    for key in ("HES", "hes", "hires", "lowres"):
        if key in images:
            return np.asarray(images[key])
    return None


def plot_target_spatial_heatmap(
    adata: ad.AnnData,
    target: str,
    batches: Sequence[str],
    output_path: Path,
    config: Mapping[str, Any],
) -> Path:
    """Write hierarchical MVN prediction panels for one density target.

    Args:
        adata (ad.AnnData): Tissue AnnData with attached predictions.
        target (str): Density target name.
        batches (Sequence[str]): Ordered slide/batch names.
        output_path (Path): Destination PNG path.
        config (Mapping[str, Any]): Spatial visualization settings.

    Returns:
        Path: Written figure path.
    """

    batch_column = str(config["batch_column"])
    x_column = str(config["x_column"])
    y_column = str(config["y_column"])
    panels: list[tuple[str, str, str, bool]] = []
    if target in adata.obs:
        panels.append((target, "Observed density", str(config["cmap"]), False))
    panels.extend(
        (
            (
                f"prob_presence_{target}",
                "Presence probability",
                "viridis",
                False,
            ),
            (
                f"positive_median_msi_{target}",
                "MSI-only positive median",
                str(config["cmap"]),
                False,
            ),
            (
                f"positive_median_patient_{target}",
                "Patient-adjusted median",
                str(config["cmap"]),
                False,
            ),
            (
                f"positive_median_total_{target}",
                "Fully adjusted median",
                str(config["cmap"]),
                False,
            ),
            (
                f"interval_width_{target}",
                "90% interval width",
                "magma",
                False,
            ),
        )
    )
    if f"density_residual_{target}" in adata.obs:
        panels.append(
            (
                f"density_residual_{target}",
                "Median minus observed",
                str(config["residual_cmap"]),
                True,
            )
        )
    include_hes = bool(config["include_hes"]) and any(
        _hes_image(adata, str(batch)) is not None for batch in batches
    )
    columns = len(panels) + int(include_hes)
    figure, axes = plt.subplots(
        len(batches),
        columns,
        figsize=(4.2 * columns, 4 * len(batches)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, batch in enumerate(batches):
        observations = adata.obs.loc[adata.obs[batch_column].astype(str) == str(batch)]
        for column_index, (column, title, cmap, diverging) in enumerate(panels):
            _plot_spatial_values(
                axes[row_index, column_index],
                observations,
                x_column,
                y_column,
                column,
                cmap,
                diverging=diverging,
            )
            if row_index == 0:
                axes[row_index, column_index].set_title(title)
        axes[row_index, 0].set_ylabel(str(batch))
        if include_hes:
            image = _hes_image(adata, str(batch))
            hes_axis = axes[row_index, -1]
            if image is not None:
                hes_axis.imshow(image)
            hes_axis.axis("off")
            if row_index == 0:
                hes_axis.set_title("HES")
    figure.suptitle(f"{target.removeprefix('Density_')} spatial predictions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=int(config["figure_dpi"]))
    plt.close(figure)
    return output_path


def plot_conditional_cd8_heatmap(
    adata: ad.AnnData,
    batches: Sequence[str],
    output_path: Path,
    config: Mapping[str, Any],
) -> Path | None:
    """Plot conditional CD8 excess when finite diagnostics are available.

    Args:
        adata (ad.AnnData): Tissue AnnData with conditional diagnostic.
        batches (Sequence[str]): Ordered slide/batch names.
        output_path (Path): Destination PNG path.
        config (Mapping[str, Any]): Spatial visualization settings.

    Returns:
        Path | None: Written path, or ``None`` if no finite excess exists.
    """

    column = "conditional_cd8_excess"
    if column not in adata.obs or not np.isfinite(
        adata.obs[column].to_numpy(dtype=np.float64)
    ).any():
        return None
    figure, axes = plt.subplots(
        len(batches),
        1,
        figsize=(6, 4 * len(batches)),
        squeeze=False,
        constrained_layout=True,
    )
    batch_column = str(config["batch_column"])
    for row_index, batch in enumerate(batches):
        observations = adata.obs.loc[adata.obs[batch_column].astype(str) == str(batch)]
        _plot_spatial_values(
            axes[row_index, 0],
            observations,
            str(config["x_column"]),
            str(config["y_column"]),
            column,
            str(config["residual_cmap"]),
            diverging=True,
        )
        axes[row_index, 0].set_ylabel(str(batch))
    axes[0, 0].set_title("Conditional CD8 standardized excess")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=int(config["figure_dpi"]))
    plt.close(figure)
    return output_path


def run_spatial_heatmaps(
    config: Mapping[str, Any],
    input_path: Path | None = None,
    predictions_path: Path | None = None,
    latent_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Load ordered inference artifacts and write all spatial figures.

    Args:
        config (Mapping[str, Any]): Complete validated pipeline configuration.
        input_path (Path | None): Optional tissue H5AD override.
        predictions_path (Path | None): Optional prediction Parquet override.
        latent_path (Path | None): Optional latent Parquet override.
        output_dir (Path | None): Optional figure directory override.

    Returns:
        Path: Directory containing spatial figures.
    """

    inference_dir = Path(str(config["inference"]["output_dir"]))
    source = (
        Path(input_path)
        if input_path is not None
        else Path(str(config["data"]["inference_path"]))
    )
    prediction_file = (
        Path(predictions_path)
        if predictions_path is not None
        else inference_dir / "inference.parquet"
    )
    latent_file = (
        Path(latent_path)
        if latent_path is not None
        else inference_dir / "inference_latent.parquet"
    )
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path(str(config["spatial_visualization"]["output_dir"]))
    )
    adata = ad.read_h5ad(source)
    predictions = pd.read_parquet(prediction_file)
    latent = pd.read_parquet(latent_file)
    if len(predictions) != adata.n_obs:
        if len(predictions) <= 0 or len(predictions) > adata.n_obs:
            raise ValueError("Prediction rows cannot be aligned to the tissue AnnData.")
        if len(latent) != len(predictions):
            raise ValueError("Prediction and latent prefix lengths differ.")
        expected = np.arange(len(predictions), dtype=np.int64)
        if not np.array_equal(
            predictions["row_position"].to_numpy(dtype=np.int64), expected
        ):
            raise ValueError("Capped predictions must be a contiguous leading prefix.")
        adata = adata[: len(predictions)].copy()
    validate_ordered_frame(adata, latent, "Latent Parquet")
    attach_predictions(adata, predictions)
    settings = config["spatial_visualization"]
    attach_coordinates(
        adata, str(settings["x_column"]), str(settings["y_column"])
    )
    targets = tuple(str(value) for value in config["data"]["target_columns"])
    add_spatial_derived_columns(adata, targets)
    batch_column = str(settings["batch_column"])
    if batch_column not in adata.obs:
        raise KeyError(f"AnnData lacks configured batch column {batch_column!r}.")
    batches = [str(value) for value in pd.unique(adata.obs[batch_column].astype(str))]
    destination.mkdir(parents=True, exist_ok=True)
    for target in targets:
        plot_target_spatial_heatmap(
            adata,
            target,
            batches,
            destination / f"spatial_mvn_{target}.png",
            settings,
        )
    plot_conditional_cd8_heatmap(
        adata,
        batches,
        destination / "spatial_conditional_cd8_excess.png",
        settings,
    )
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    """Load configuration and generate spatial heatmaps.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        int: Zero process status after successful plotting.
    """

    args = parse_args(argv)
    output_dir = run_spatial_heatmaps(
        load_config(args.config),
        input_path=args.input,
        predictions_path=args.predictions,
        latent_path=args.latent,
        output_dir=args.output_dir,
    )
    print(f"Spatial heatmaps written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Plot spatial heatmaps from tissue AnnData and DANN spatial inference."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scanpy as sc
import seaborn as sns
from matplotlib import pyplot as plt

from dann.targets import prepare_observed_targets, target_label

warnings.filterwarnings("ignore")
sns.set_style("ticks", {"axes.grid": True})


DEFAULT_INPUT = Path(
    "/workspaces/multimodal-pdac/data/PDAC/Raw/adata_assembled_tissue.h5ad"
)
DEFAULT_PREDICTIONS = Path(
    "/workspaces/multimodal-pdac/data/PDAC/Results/dann/analysis/spatial/"
    "spatial_inference.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "/workspaces/multimodal-pdac/data/PDAC/Results/dann/analysis/spatial"
)
DEFAULT_DENSITIES = (
    "Density_CD8",
    "Density_Tumor",
    "Density_Stroma",
    "Density_Collagen",
)
IDENTITY_COLUMNS = ("row_position", "obs_name")


def _symmetric_zero_imshow_limits(heatmap_data: pd.DataFrame) -> tuple[float, float]:
    """Compute symmetric ``(-limit, limit)`` color limits centered at zero.

    Args:
        heatmap_data (pd.DataFrame): Pivoted spatial values (may contain NaNs).

    Returns:
        tuple[float, float]: Inclusive ``(vmin, vmax)`` with ``vmax == -vmin``.
    """

    values = np.asarray(heatmap_data.to_numpy(), dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    limit = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1e-6
    return -limit, limit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse spatial heatmap command-line arguments.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        argparse.Namespace: Parsed spatial heatmap options.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("dann/config.yaml"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--densities",
        nargs="+",
        default=list(DEFAULT_DENSITIES),
        help="Observed density columns to visualize.",
    )
    parser.add_argument(
        "--logit-epsilon",
        type=float,
        default=1e-4,
        help="Clamp applied before the observed-density logit transform.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved figure resolution.",
    )
    parser.add_argument(
        "--batch-column",
        default="batch",
        help="Observation column used to split heatmap rows.",
    )
    return parser.parse_args(argv)


def attach_predictions(adata: ad.AnnData, predictions: pd.DataFrame) -> list[str]:
    """Join ordered spatial inference columns onto AnnData observations.

    Args:
        adata (ad.AnnData): Tissue AnnData whose row order defines identity.
        predictions (pd.DataFrame): Spatial inference table with identity and
            prediction columns.

    Returns:
        list[str]: Prediction column names attached to ``adata.obs``.
    """

    if predictions["row_position"].dtype != np.int64:
        raise TypeError(
            f"row_position dtype is {predictions['row_position'].dtype}; "
            "expected int64."
        )
    if not np.array_equal(
        predictions["row_position"].to_numpy(),
        np.arange(adata.n_obs, dtype=np.int64),
    ):
        raise ValueError("Prediction row_position is not contiguous AnnData order.")
    if not np.array_equal(
        predictions["obs_name"].to_numpy(dtype=str),
        adata.obs_names.to_numpy(dtype=str),
    ):
        raise ValueError("Prediction obs_name order differs from source AnnData.")
    prediction_columns = predictions.columns.difference(
        list(IDENTITY_COLUMNS),
        sort=False,
    ).tolist()
    adata.obs[prediction_columns] = predictions[prediction_columns].to_numpy(
        dtype=np.float32,
        copy=False,
    )
    return prediction_columns


def add_logit_columns(
    adata: ad.AnnData,
    densities: Sequence[str],
    epsilon: float,
) -> list[str]:
    """Store clamped logit transforms of observed density columns.

    Args:
        adata (ad.AnnData): AnnData whose ``obs`` holds density columns.
        densities (Sequence[str]): Observed density column names.
        epsilon (float): Boundary clamp applied before the logit transform.

    Returns:
        list[str]: Created ``logit_*`` column names.
    """

    if not 0.0 < epsilon < 0.5:
        raise ValueError("logit_epsilon must lie strictly between zero and 0.5.")
    logit_columns: list[str] = []
    for density in densities:
        if density not in adata.obs.columns:
            raise KeyError(f"Density column {density!r} is missing from adata.obs.")
        values = np.asarray(adata.obs[density], dtype=np.float64)
        clipped = np.clip(values, epsilon, 1.0 - epsilon)
        column = f"logit_{density}"
        adata.obs[column] = np.log(clipped / (1.0 - clipped)).astype(np.float32)
        logit_columns.append(column)
    return logit_columns


def _positive_spatial_grid(
    frame: pd.DataFrame, density: str, column: str
) -> pd.DataFrame:
    """Mask zero observations while preserving the full batch coordinate grid."""
    values = frame[["x", "y", column]].copy()
    values[column] = values[column].where(frame[density] > 0.0)
    return values.pivot(index="y", columns="x", values=column)


def _plot_hes_context(
    adata: ad.AnnData, batch_mask: pd.Series, batch: str, axis: plt.Axes
) -> None:
    """Render HES with the full tissue extent using two transparent bounds markers.

    Scanpy normally constructs one invisible circle for each observation when
    displaying an image without a color variable. Only the coordinate extrema
    affect the limits, so two corners preserve the same crop and spot padding.
    """
    coordinates = np.asarray(adata.obsm["spatial"])[np.asarray(batch_mask)]
    bounds = ad.AnnData(obs=pd.DataFrame(index=["minimum", "maximum"]))
    bounds.obsm["spatial"] = np.vstack(
        (coordinates.min(axis=0), coordinates.max(axis=0))
    )
    bounds.uns["spatial"] = {batch: adata.uns["spatial"][batch]}
    sc.pl.spatial(
        bounds, library_id=batch, img_key="HES", frameon=False, show=False, ax=axis
    )


def plot_density_heatmap(
    adata: ad.AnnData,
    density: str,
    batches: Sequence[str],
    output_path: Path,
    *,
    batch_column: str = "batch",
    dpi: int = 300,
    logit_epsilon: float = 1e-4,
) -> Path:
    """Write one per-batch spatial heatmap figure for a density target.

    Args:
        adata (ad.AnnData): Tissue AnnData with densities, predictions, and spatial.
        density (str): Observed density column name.
        batches (Sequence[str]): Ordered batch labels to plot as figure rows.
        output_path (Path): Destination PNG path.
        batch_column (str): Observation column holding batch labels.
        dpi (int): Saved figure resolution.
        logit_epsilon (float): Same boundary clamp used for observed densities.

    Returns:
        Path: Written PNG path.
    """

    required = (
        density,
        f"logit_{density}",
        f"mu_{density}",
        f"prob_of_presence_{density}",
        f"sigma_{density}",
        f"density_mean_{density}",
        batch_column,
        "x",
        "y",
    )
    missing = [name for name in required if name not in adata.obs.columns]
    if f"density_mean_{density}" in missing:
        raise KeyError(
            f"Missing density_mean_{density}; rerun python -m dann.spatial "
            "with your input/checkpoint paths and --overwrite to regenerate predictions. "
            f"Missing columns: {missing}"
        )
    if missing:
        raise KeyError(f"Missing required obs columns for {density!r}: {missing}")
    if not batches:
        raise ValueError("At least one batch is required to plot heatmaps.")

    mean_column = f"density_mean_{density}"
    mean_logit_column = add_logit_columns(adata, [mean_column], logit_epsilon)[0]
    n_batches = len(batches)
    fig, axes = plt.subplots(
        figsize=(36, 4 * n_batches),
        nrows=n_batches,
        ncols=6,
        tight_layout=True,
        squeeze=False,
    )
    for i, batch in enumerate(batches):
        batch_mask = adata.obs[batch_column] == batch

        heatmap_data = _positive_spatial_grid(
            adata.obs.loc[batch_mask], density, f"logit_{density}"
        )
        im0 = axes[i, 0].imshow(heatmap_data, cmap="jet", origin="upper")
        fig.colorbar(im0, ax=axes[i, 0])
        axes[i, 0].set_ylabel(batch, fontsize=12)

        heatmap_data = adata.obs.loc[batch_mask].pivot(
            index="y",
            columns="x",
            values=f"mu_{density}",
        )
        im1 = axes[i, 1].imshow(heatmap_data, cmap="jet", origin="upper")
        fig.colorbar(im1, ax=axes[i, 1])

        heatmap_data = adata.obs.loc[batch_mask].pivot(
            index="y",
            columns="x",
            values=f"prob_of_presence_{density}",
        )
        im2 = axes[i, 2].imshow(heatmap_data, cmap="jet", origin="upper")
        fig.colorbar(im2, ax=axes[i, 2])

        heatmap_data = adata.obs.loc[batch_mask].pivot(
            index="y",
            columns="x",
            values=f"sigma_{density}",
        )
        im3 = axes[i, 3].imshow(heatmap_data, cmap="jet", origin="upper")
        fig.colorbar(im3, ax=axes[i, 3])

        heatmap_data = _positive_spatial_grid(
            adata.obs.loc[batch_mask], mean_column, mean_logit_column
        )
        im4 = axes[i, 4].imshow(
            heatmap_data, cmap="jet", origin="upper",
        )
        fig.colorbar(im4, ax=axes[i, 4])
        _plot_hes_context(adata, batch_mask, batch, axes[i, 5])

    axes[0, 0].set_title(f"logit_{density}", fontsize=12)
    axes[0, 1].set_title(f"mu_{density}", fontsize=12)
    axes[0, 2].set_title(f"prob_of_presence_{density}", fontsize=12)
    axes[0, 3].set_title(f"sigma_{density}", fontsize=12)
    axes[0, 4].set_title(f"logit(sampled mean density): {density}", fontsize=12)
    axes[0, 5].set_title("HES", fontsize=12)

    for ax in axes.flatten():
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("gray")

    for ax in axes[:, 5]:
        ax.invert_yaxis()
        ax.invert_xaxis()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    contract = adata.uns.get("dann_target_transform")
    if contract:
        label = target_label(density, contract)
        for axis in fig.axes:
            axis.set_title(axis.get_title().replace(density, label))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def add_residual_columns(
    adata: ad.AnnData,
    densities: Sequence[str],
) -> list[str]:
    """Store the logit-scale residual between predicted mean and observation.

    Only positive observations have a defined residual, so zeros are stored as
    ``NaN`` and drop out of both the pivot and the color scaling.

    Args:
        adata (ad.AnnData): AnnData holding densities, logits, and predictions.
        densities (Sequence[str]): Observed density column names.

    Returns:
        list[str]: Created ``residual_*`` column names.
    """

    residual_columns: list[str] = []
    for density in densities:
        for name in (density, f"logit_{density}", f"logit_mean_{density}"):
            if name not in adata.obs.columns:
                raise KeyError(f"Column {name!r} is missing from adata.obs.")
        observed = np.asarray(adata.obs[density], dtype=np.float64)
        residual = np.asarray(
            adata.obs[f"logit_mean_{density}"], dtype=np.float64
        ) - np.asarray(adata.obs[f"logit_{density}"], dtype=np.float64)
        residual[observed <= 0.0] = np.nan
        column = f"residual_{density}"
        adata.obs[column] = residual.astype(np.float32)
        residual_columns.append(column)
    return residual_columns


def add_interval_width_columns(
    adata: ad.AnnData,
    densities: Sequence[str],
) -> list[str]:
    """Store the width of the central 90% positive-branch interval.

    Args:
        adata (ad.AnnData): AnnData holding the spatial inference quantiles.
        densities (Sequence[str]): Observed density column names.

    Returns:
        list[str]: Created ``interval_width_*`` column names.
    """

    width_columns: list[str] = []
    for density in densities:
        for name in (f"logit_q05_{density}", f"logit_q95_{density}"):
            if name not in adata.obs.columns:
                raise KeyError(f"Column {name!r} is missing from adata.obs.")
        width = np.asarray(
            adata.obs[f"logit_q95_{density}"], dtype=np.float64
        ) - np.asarray(adata.obs[f"logit_q05_{density}"], dtype=np.float64)
        column = f"interval_width_{density}"
        adata.obs[column] = width.astype(np.float32)
        width_columns.append(column)
    return width_columns


def plot_diagnostics_density_heatmap(
    adata: ad.AnnData,
    density: str,
    batches: Sequence[str],
    output_path: Path,
    *,
    batch_column: str = "batch",
    dpi: int = 300,
) -> Path:
    """Write one per-batch heatmap of normal ZILN diagnostics for a target.

    The companion to ``plot_density_heatmap``, which shows the raw parameters.
    Here the parameters are converted into quantities that read as predictions:
    the normal mean ``E[Z | y > 0]``, the
    residual against observed truth, and the width of the central 90% interval.

    The numerical panels are masked to observed
    ``density > 0``. All summaries here are conditional on the positive branch,
    because ``logit(0)`` is undefined, so plotting them over pixels with no
    observed signal would be comparing against nothing.

    Args:
        adata (ad.AnnData): Tissue AnnData with densities, predictions, and spatial.
        density (str): Observed density column name.
        batches (Sequence[str]): Ordered batch labels to plot as figure rows.
        output_path (Path): Destination PNG path.
        batch_column (str): Observation column holding batch labels.
        dpi (int): Saved figure resolution.

    Returns:
        Path: Written PNG path.
    """

    required = (
        density,
        f"logit_{density}",
        f"logit_mean_{density}",
        f"residual_{density}",
        f"interval_width_{density}",
        batch_column,
        "x",
        "y",
    )
    missing = [name for name in required if name not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing required obs columns for {density!r}: {missing}")
    if not batches:
        raise ValueError("At least one batch is required to plot heatmaps.")

    n_batches = len(batches)
    fig, axes = plt.subplots(
        figsize=(30, 4 * n_batches),
        nrows=n_batches,
        ncols=5,
        tight_layout=True,
        squeeze=False,
    )
    sequential_columns = (
        (0, f"logit_{density}"),
        (1, f"logit_mean_{density}"),
        (3, f"interval_width_{density}"),
    )
    for i, batch in enumerate(batches):
        batch_mask = adata.obs[batch_column] == batch

        for column_index, column_name in sequential_columns:
            heatmap_data = _positive_spatial_grid(
                adata.obs.loc[batch_mask], density, column_name
            )
            image = axes[i, column_index].imshow(
                heatmap_data, cmap="jet", origin="upper"
            )
            fig.colorbar(image, ax=axes[i, column_index])
        axes[i, 0].set_ylabel(batch, fontsize=12)

        heatmap_data = _positive_spatial_grid(
            adata.obs.loc[batch_mask], density, f"residual_{density}"
        )
        vmin, vmax = _symmetric_zero_imshow_limits(heatmap_data)
        image = axes[i, 2].imshow(
            heatmap_data,
            cmap="coolwarm",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(image, ax=axes[i, 2])

        _plot_hes_context(adata, batch_mask, batch, axes[i, 4])

    axes[0, 0].set_title(f"logit_{density} (observed)", fontsize=12)
    axes[0, 1].set_title(f"logit_mean_{density}", fontsize=12)
    axes[0, 2].set_title(f"residual_{density}", fontsize=12)
    axes[0, 3].set_title(f"interval_width_{density}", fontsize=12)
    axes[0, 4].set_title("HES", fontsize=12)

    for ax in axes.flatten():
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("gray")

    for ax in axes[:, 4]:
        ax.invert_yaxis()
        ax.invert_xaxis()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    contract = adata.uns.get("dann_target_transform")
    if contract:
        label = target_label(density, contract)
        for axis in fig.axes:
            axis.set_title(axis.get_title().replace(density, label))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def run_spatial_heatmaps(
    input_path: Path,
    predictions_path: Path,
    output_dir: Path,
    densities: Sequence[str],
    logit_epsilon: float,
    dpi: int,
    batch_column: str = "batch",
) -> Path:
    """Load tissue predictions and write one heatmap PNG per density.

    Args:
        input_path (Path): Tissue AnnData path.
        predictions_path (Path): Spatial inference Parquet path.
        output_dir (Path): Directory for heatmap PNGs.
        densities (Sequence[str]): Observed density columns to visualize.
        logit_epsilon (float): Clamp applied before the logit transform.
        dpi (int): Saved figure resolution.
        batch_column (str): Observation column holding batch labels.

    Returns:
        Path: Output directory containing written PNG files.
    """

    if not input_path.is_file():
        raise FileNotFoundError(f"Tissue AnnData not found: {input_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions Parquet not found: {predictions_path}")

    print(f"Loading tissue AnnData from {input_path}", flush=True)
    adata = sc.read_h5ad(input_path)
    print(f"Loading predictions from {predictions_path}", flush=True)
    predictions = pd.read_parquet(predictions_path)
    prediction_columns = attach_predictions(adata, predictions)
    metadata = pq.read_schema(predictions_path).metadata or {}
    if b"dann_target_transform" in metadata:
        contract = json.loads(metadata[b"dann_target_transform"])
        if contract.get("version") != 1:
            raise ValueError("Unsupported spatial target transformation version.")
        columns = contract["target_columns"]
        raw = adata.obs[columns].to_numpy(copy=True)
        targets, valid = prepare_observed_targets(
            raw, columns, contract["cd8_normalization"]
        )
        for index, column in enumerate(columns):
            adata.obs[f"raw_{column}"] = raw[:, index]
            adata.obs[f"valid_{column}"] = valid[:, index]
            adata.obs[column] = np.where(valid[:, index], targets[:, index], np.nan)
        adata.uns["dann_target_transform"] = contract
    logit_columns = add_logit_columns(adata, densities, logit_epsilon)
    residual_columns = add_residual_columns(adata, densities)
    width_columns = add_interval_width_columns(adata, densities)
    batches = adata.obs[batch_column].unique().tolist()
    print(
        f"Attached {len(prediction_columns)} prediction columns, "
        f"{len(logit_columns)} logit columns, {len(residual_columns)} residual "
        f"columns, and {len(width_columns)} interval-width columns "
        f"for {len(batches)} batches.",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for density in densities:
        output_path = output_dir / f"spatial_heatmap_{density}.png"
        plot_density_heatmap(
            adata,
            density,
            batches,
            output_path,
            batch_column=batch_column,
            dpi=dpi,
            logit_epsilon=logit_epsilon,
        )
        print(f"Wrote {output_path}", flush=True)
        diagnostics_path = output_dir / f"spatial_heatmap_diagnostics_{density}.png"
        plot_diagnostics_density_heatmap(
            adata,
            density,
            batches,
            diagnostics_path,
            batch_column=batch_column,
            dpi=dpi,
        )
        print(f"Wrote {diagnostics_path}", flush=True)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Load options and write spatial heatmap figures.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        int: Process exit status, zero on success.
    """

    from dann.config import inference_input
    args = parse_args(argv)
    from dann.config import load_config
    config = load_config(args.config)
    output_dir = run_spatial_heatmaps(
        input_path=inference_input(config, args.input),
        predictions_path=args.predictions or Path(config["spatial"]["output"]),
        output_dir=args.output_dir or Path(config["analysis"]["output_dir"]) / "spatial/heatmaps",
        densities=args.densities,
        logit_epsilon=args.logit_epsilon,
        dpi=args.dpi,
        batch_column=args.batch_column,
    )
    print(f"Spatial heatmaps written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

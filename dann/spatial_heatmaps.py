"""Plot spatial heatmaps from tissue AnnData and DANN spatial inference."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import pyplot as plt

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse spatial heatmap command-line arguments.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        argparse.Namespace: Parsed spatial heatmap options.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def plot_density_heatmap(
    adata: ad.AnnData,
    density: str,
    batches: Sequence[str],
    output_path: Path,
    *,
    batch_column: str = "batch",
    dpi: int = 300,
) -> Path:
    """Write one per-batch spatial heatmap figure for a density target.

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
        f"mean_{density}",
        f"prob_of_presence_{density}",
        f"sigma_{density}",
        f"alpha_{density}",
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
        ncols=6,
        tight_layout=True,
        squeeze=False,
    )
    for i, batch in enumerate(batches):
        batch_mask = adata.obs[batch_column] == batch
        positive_mask = batch_mask & (adata.obs[density] > 0)

        heatmap_data = adata.obs.loc[positive_mask].pivot(
            index="y",
            columns="x",
            values=f"logit_{density}",
        )
        im0 = axes[i, 0].imshow(heatmap_data, cmap="jet", origin="upper")
        fig.colorbar(im0, ax=axes[i, 0])
        axes[i, 0].set_ylabel(batch, fontsize=12)

        heatmap_data = adata.obs.loc[batch_mask].pivot(
            index="y",
            columns="x",
            values=f"mean_{density}",
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

        heatmap_data = adata.obs.loc[batch_mask].pivot(
            index="y",
            columns="x",
            values=f"alpha_{density}",
        )
        im4 = axes[i, 4].imshow(heatmap_data, cmap="jet", origin="upper")
        fig.colorbar(im4, ax=axes[i, 4])

        sc.pl.spatial(
            adata[batch_mask],
            library_id=batch,
            img_key="HES",
            frameon=False,
            show=False,
            ax=axes[i, 5],
        )

    axes[0, 0].set_title(f"logit_{density}", fontsize=12)
    axes[0, 1].set_title(f"mean_{density}", fontsize=12)
    axes[0, 2].set_title(f"prob_of_presence_{density}", fontsize=12)
    axes[0, 3].set_title(f"sigma_{density}", fontsize=12)
    axes[0, 4].set_title(f"alpha_{density}", fontsize=12)
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
    logit_columns = add_logit_columns(adata, densities, logit_epsilon)
    batches = adata.obs[batch_column].unique().tolist()
    print(
        f"Attached {len(prediction_columns)} prediction columns and "
        f"{len(logit_columns)} logit columns for {len(batches)} batches.",
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
        )
        print(f"Wrote {output_path}", flush=True)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Load options and write spatial heatmap figures.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        int: Process exit status, zero on success.
    """

    args = parse_args(argv)
    output_dir = run_spatial_heatmaps(
        input_path=args.input,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        densities=args.densities,
        logit_epsilon=args.logit_epsilon,
        dpi=args.dpi,
        batch_column=args.batch_column,
    )
    print(f"Spatial heatmaps written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

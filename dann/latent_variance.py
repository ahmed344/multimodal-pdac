"""Measure latent variance from existing DANN spatial-inference exports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from anndata.io import read_elem
from matplotlib import pyplot as plt

from .config import latent_variance_settings, load_config


THRESHOLDS = ((0.9, "pcs_90"), (0.99, "pcs_99"), (0.999, "pcs_99_9"))


def read_slide_metadata(path: Path, slide_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Read observation names and slide labels without accessing the MSI matrix."""
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        if slide_column not in obs:
            raise ValueError(f"AnnData obs is missing slide column {slide_column!r}.")
        index_key = obs.attrs.get("_index", "_index")
        names = np.asarray(read_elem(obs[index_key]), dtype=str)
        labels = np.asarray(read_elem(obs[slide_column]))
    return names, labels


def latent_columns(schema: pa.Schema) -> list[str]:
    """Validate identity fields and return contiguous, numerically ordered latents."""
    names = schema.names
    if len(set(names)) != len(names):
        raise ValueError("Latent Parquet contains duplicate column names.")
    if "row_position" not in names or "obs_name" not in names:
        raise ValueError("Latent Parquet requires row_position and obs_name columns.")
    if schema.field("row_position").type != pa.int64():
        raise ValueError("row_position must have int64 dtype.")
    name_type = schema.field("obs_name").type
    if not (pa.types.is_string(name_type) or pa.types.is_large_string(name_type)):
        raise ValueError("obs_name must have string dtype.")
    columns = [name for name in names if name.startswith("latent_")]
    expected = [f"latent_{i}" for i in range(len(columns))]
    if not columns or set(columns) != set(expected):
        raise ValueError("Expected contiguous latent_0 through latent_<width-1> columns.")
    if any(not pa.types.is_floating(schema.field(name).type) for name in columns):
        raise ValueError("Latent columns must have floating-point dtype.")
    return expected


def summarize_variance(
    scatter: np.ndarray,
    count: int,
    mean: np.ndarray,
    slide_counts: np.ndarray,
    slide_means: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute covariance PCA and pixel-weighted between-slide variance fractions."""
    covariance = (scatter + scatter.T) / (2 * (count - 1))
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    tolerance = 100 * np.finfo(float).eps * max(1, len(mean)) * np.max(np.abs(eigenvalues))
    if np.any(eigenvalues < -tolerance):
        raise ValueError("Covariance has materially negative eigenvalues.")
    eigenvalues = np.maximum(eigenvalues, 0)
    total = float(eigenvalues.sum())
    summary: dict[str, Any] = {
        "latent_width": len(mean),
        "n_observations": count,
        "n_slides": len(slide_counts),
        "total_variance": total,
        "participation_ratio": None,
        **{name: None for _, name in THRESHOLDS},
        "variance_explained_by_slide_means": None,
    }
    fractions = np.full(len(mean), np.nan)
    cumulative = fractions.copy()
    if total > 0:
        fractions = eigenvalues / total
        cumulative = np.minimum(np.cumsum(fractions), 1.0)
        cumulative[-1] = 1.0
        summary["participation_ratio"] = float(1.0 / np.dot(fractions, fractions))
        for threshold, name in THRESHOLDS:
            # Count exact threshold ties despite floating-point summation error.
            roundoff = 8 * np.finfo(float).eps * len(mean)
            summary[name] = int(np.searchsorted(cumulative, threshold - roundoff) + 1)
        between = float(np.sum(slide_counts[:, None] * (slide_means - mean) ** 2))
        summary["variance_explained_by_slide_means"] = float(
            np.clip(between / np.trace(scatter), 0, 1)
        )
    spectrum = pd.DataFrame({
        "pc": np.arange(1, len(mean) + 1),
        "eigenvalue": eigenvalues,
        "explained_variance_fraction": fractions,
        "cumulative_variance_fraction": cumulative,
    })
    return summary, spectrum


def compute_latent_variance(
    latent_path: Path,
    input_path: Path,
    slide_column: str = "batch",
    chunk_size: int = 8192,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Stream exact covariance and slide means from an aligned inference prefix."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    for path in (latent_path, input_path):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Missing input: {path}. Run python -m dann.spatial first and provide "
                "its latent Parquet and matching AnnData."
            )
    parquet = pq.ParquetFile(latent_path)
    columns = latent_columns(parquet.schema_arrow)
    n_rows = parquet.metadata.num_rows
    if n_rows < 2:
        raise ValueError("At least two latent observations are required.")
    obs_names, labels = read_slide_metadata(input_path, slide_column)
    if n_rows > len(obs_names) or len(labels) != len(obs_names):
        raise ValueError("Latent row count is incompatible with AnnData observations.")
    labels = labels[:n_rows]
    if pd.isna(labels).any() or any(isinstance(x, str) and not x.strip() for x in pd.unique(labels)):
        raise ValueError("Slide labels cannot be missing or empty for analyzed rows.")
    codes, slide_names = pd.factorize(labels, sort=False)
    width = len(columns)
    mean = np.zeros(width, dtype=np.float64)
    scatter = np.zeros((width, width), dtype=np.float64)
    slide_counts = np.zeros(len(slide_names), dtype=np.int64)
    slide_means = np.zeros((len(slide_names), width), dtype=np.float64)
    count = 0
    origin = None
    for batch in parquet.iter_batches(
        batch_size=chunk_size, columns=["row_position", "obs_name", *columns]
    ):
        stop = count + batch.num_rows
        if any(column.null_count for column in batch.columns):
            raise ValueError("Latent Parquet contains null identities or latent values.")
        positions = batch.column("row_position").to_numpy()
        if not np.array_equal(positions, np.arange(count, stop)):
            raise ValueError("row_position must be a contiguous leading AnnData prefix.")
        names = np.asarray(batch.column("obs_name").to_pylist(), dtype=str)
        if not np.array_equal(names, obs_names[count:stop]):
            raise ValueError("obs_name order differs from the source AnnData.")
        values = np.column_stack([
            batch.column(name).to_numpy() for name in columns
        ]).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Latent values must be finite.")
        # A fixed translation preserves variance and avoids loss of precision
        # for large offsets; exactly constant dimensions remain exactly zero.
        if origin is None:
            origin = values[0].copy()
        values -= origin
        chunk_mean = values.mean(axis=0)
        centered = values - chunk_mean
        delta = chunk_mean - mean
        scatter += centered.T @ centered + np.outer(delta, delta) * (count * batch.num_rows / stop)
        mean += delta * (batch.num_rows / stop)
        chunk_codes = codes[count:stop]
        for slide in np.unique(chunk_codes):
            selected = values[chunk_codes == slide]
            added = len(selected)
            slide_counts[slide] += added
            slide_means[slide] += (
                selected.mean(axis=0) - slide_means[slide]
            ) * (added / slide_counts[slide])
        count = stop
        if count == n_rows or (count // chunk_size) % 100 == 0:
            print(f"Processed {count:,}/{n_rows:,} latent rows.", flush=True)
    summary, spectrum = summarize_variance(scatter, count, mean, slide_counts, slide_means)
    summary.update(
        source_latents=str(Path(latent_path).resolve()),
        source_anndata=str(Path(input_path).resolve()),
        slide_column=slide_column,
        source_n_observations=len(obs_names),
        inference_coverage_fraction=count / len(obs_names),
        is_prefix=count < len(obs_names),
    )
    return summary, spectrum


def plot_explained_variance(
    spectrum: pd.DataFrame, summary: Mapping[str, Any], path: Path, dpi: int = 300
) -> None:
    """Save individual and cumulative explained-variance panels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    try:
        axes[0].set(title="Variance per principal component", ylabel="Explained variance (%)")
        axes[1].set(title="Cumulative explained variance", ylabel="Cumulative variance (%)", ylim=(0, 102))
        for ax in axes:
            ax.set_xlabel("Principal component")
            ax.set_xlim(0.5, len(spectrum) + 0.5)
            ax.grid(alpha=0.25)
        if summary["total_variance"] == 0:
            for ax in axes:
                ax.text(0.5, 0.5, "Constant latents: variance fractions undefined",
                        ha="center", va="center", transform=ax.transAxes, wrap=True)
        else:
            axes[0].plot(spectrum["pc"], 100 * spectrum["explained_variance_fraction"])
            axes[1].plot(spectrum["pc"], 100 * spectrum["cumulative_variance_fraction"])
            for (threshold, key), color in zip(THRESHOLDS, ("C1", "C2", "C3")):
                pc = summary[key]
                axes[1].axhline(100 * threshold, color=color, linestyle=":", alpha=0.6)
                axes[1].axvline(pc, color=color, linestyle="--", alpha=0.6,
                               label=f"{100 * threshold:g}%: {pc} PCs")
            axes[1].legend(loc="lower right")
        fig.suptitle(f"Latent variance | {summary['n_observations']:,} pixels | {summary['latent_width']} dimensions")
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse configuration and optional standalone diagnostic overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("dann/config.yaml"))
    parser.add_argument("--latents", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--slide-column")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Compute and write diagnostics without invoking a model or inference."""
    args = parse_args(argv)
    settings = load_config(args.config)["latent_variance"]
    for key in ("latents", "input", "output_dir", "slide_column"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = str(value)
    settings = latent_variance_settings(settings)
    summary, spectrum = compute_latent_variance(
        Path(settings["latents"]), Path(settings["input"]),
        settings["slide_column"], settings["chunk_size"],
    )
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_dir / "latent_variance_summary.csv", index=False)
    spectrum.to_csv(output_dir / "pca_variance.csv", index=False)
    plot_explained_variance(spectrum, summary, output_dir / "explained_variance.png", settings["figure_dpi"])
    print(f"Latent variance diagnostics written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Distributional validation of the zero-inflated skew-logit-normal head.

Training loss decreasing says the optimizer worked; it says nothing about
whether the predicted distributions are *calibrated*. This module answers the
question the training loop cannot: given the four predicted parameters, do the
observed densities behave like draws from the predicted distributions?

Two branches are scored separately, because they are two different claims:

* **Positive branch** (zeros masked, logit scale). Accuracy of the location via
  correlation and error metrics, and calibration of the whole shape via the
  probability integral transform and interval coverage. Point accuracy is
  reported twice — once against the raw ``mu`` that the pipeline previously
  treated as a prediction, and once against the skew-corrected ``E[Z | y > 0]``
  — so the cost of the old shortcut is visible as a number.
* **Hurdle branch**. Calibration of ``pi`` against the observed rate of exact
  zeros, via Brier score, ROC AUC, and a reliability curve. This branch is the
  model *of* the zeros, so it is the one place zeros are deliberately not
  masked.

Everything is additionally broken out per batch, since batch invariance is the
entire purpose of the gradient reversal layer and a per-batch spread in these
metrics is the evidence for or against it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

from dann import ziln


DEFAULT_COVERAGE_LEVELS = (0.50, 0.90)
DEFAULT_PIT_BINS = 50
COVERAGE_SWEEP = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
RELIABILITY_BINS = 10


def _interval_bounds(level: float) -> tuple[float, float]:
    """Convert a central coverage level into its two quantile positions.

    Args:
        level (float): Central probability mass in ``(0, 1)``.

    Returns:
        tuple[float, float]: Lower and upper quantile positions.
    """

    if not 0.0 < float(level) < 1.0:
        raise ValueError("Coverage level must lie strictly between zero and one.")
    lower = (1.0 - float(level)) / 2.0
    return lower, 1.0 - lower


def compute_target_metrics(
    targets: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    alpha: np.ndarray,
    epsilon: float,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
) -> dict[str, float]:
    """Score one target's positive and hurdle branches.

    Args:
        targets (np.ndarray): Observed densities in ``[0, 1]``.
        pi (np.ndarray): Predicted structural-zero probabilities.
        mu (np.ndarray): Predicted positive-branch locations.
        sigma (np.ndarray): Predicted positive-branch scales.
        alpha (np.ndarray): Predicted positive-branch shapes.
        epsilon (float): Clamp applied before observed-value logits.
        coverage_levels (Sequence[float]): Central interval levels to score.

    Returns:
        dict[str, float]: Metric name to value, with ``NaN`` where undefined.
    """

    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    pi = np.asarray(pi, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    sizes = {targets.size, pi.size, mu.size, sigma.size, alpha.size}
    if len(sizes) != 1:
        raise ValueError(f"Metric inputs must share one length; observed {sizes}.")

    is_zero = targets == 0.0
    positive = ~is_zero
    metrics: dict[str, float] = {
        "n": float(targets.size),
        "n_positive": float(positive.sum()),
        "n_zero": float(is_zero.sum()),
        "observed_zero_rate": float(is_zero.mean()) if targets.size else np.nan,
        "mean_pi": float(pi.mean()) if pi.size else np.nan,
    }

    # Hurdle branch: pi is a probabilistic classifier for "this pixel is zero".
    if targets.size:
        metrics["brier"] = float(np.mean(np.square(pi - is_zero.astype(np.float64))))
    else:
        metrics["brier"] = np.nan
    if is_zero.any() and positive.any():
        metrics["auc"] = float(roc_auc_score(is_zero.astype(np.int64), pi))
    else:
        metrics["auc"] = np.nan

    if not positive.any():
        metrics.update(
            {
                "bias": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "pearson_r_mu": np.nan,
                "pearson_r_corrected": np.nan,
                "spearman_r_corrected": np.nan,
                "mean_skew_shift": np.nan,
                "pit_mean": np.nan,
                "pit_ks": np.nan,
            }
        )
        for level in coverage_levels:
            metrics[f"coverage_{int(round(float(level) * 100))}"] = np.nan
        return metrics

    truth = ziln.clipped_logit(targets[positive], epsilon)
    mu_positive = mu[positive]
    sigma_positive = sigma[positive]
    alpha_positive = alpha[positive]
    corrected = ziln.positive_logit_mean(mu_positive, sigma_positive, alpha_positive)
    residual = corrected - truth

    metrics["bias"] = float(residual.mean())
    metrics["mae"] = float(np.abs(residual).mean())
    metrics["rmse"] = float(np.sqrt(np.mean(np.square(residual))))
    metrics["mean_skew_shift"] = float(np.mean(corrected - mu_positive))
    metrics["pearson_r_mu"] = _safe_pearson(truth, mu_positive)
    metrics["pearson_r_corrected"] = _safe_pearson(truth, corrected)
    metrics["spearman_r_corrected"] = _safe_spearman(truth, corrected)

    pit = ziln.positive_logit_pit(
        mu_positive, sigma_positive, alpha_positive, targets[positive], epsilon
    )
    metrics["pit_mean"] = float(pit.mean())
    metrics["pit_ks"] = float(stats.kstest(pit, "uniform").statistic)

    for level in coverage_levels:
        lower_q, upper_q = _interval_bounds(float(level))
        lower = ziln.positive_logit_quantile(
            mu_positive, sigma_positive, alpha_positive, lower_q
        )
        upper = ziln.positive_logit_quantile(
            mu_positive, sigma_positive, alpha_positive, upper_q
        )
        inside = (truth >= lower) & (truth <= upper)
        metrics[f"coverage_{int(round(float(level) * 100))}"] = float(inside.mean())
    return metrics


def _safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Compute a Pearson correlation that tolerates degenerate inputs.

    Args:
        first (np.ndarray): First value vector.
        second (np.ndarray): Second value vector.

    Returns:
        float: Pearson correlation, or ``NaN`` when undefined.
    """

    if first.size < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Compute a Spearman correlation that tolerates degenerate inputs.

    Args:
        first (np.ndarray): First value vector.
        second (np.ndarray): Second value vector.

    Returns:
        float: Spearman correlation, or ``NaN`` when undefined.
    """

    if first.size < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return np.nan
    result = stats.spearmanr(first, second).statistic
    return float(result) if np.isfinite(result) else np.nan


def build_metrics_frame(
    extracted: Mapping[str, np.ndarray],
    target_columns: Sequence[str],
    split: str,
    epsilon: float,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
) -> pd.DataFrame:
    """Score every target over all samples in the split.

    Args:
        extracted (Mapping[str, np.ndarray]): Targets and ZILN parameter arrays.
        target_columns (Sequence[str]): Ordered density target names.
        split (str): Split name recorded in the table.
        epsilon (float): Clamp applied before observed-value logits.
        coverage_levels (Sequence[float]): Central interval levels to score.

    Returns:
        pd.DataFrame: One row per target.
    """

    rows: list[dict[str, Any]] = []
    for index, column in enumerate(target_columns):
        metrics = compute_target_metrics(
            np.asarray(extracted["targets"])[:, index],
            np.asarray(extracted["pi"])[:, index],
            np.asarray(extracted["mu"])[:, index],
            np.asarray(extracted["sigma"])[:, index],
            np.asarray(extracted["alpha"])[:, index],
            epsilon,
            coverage_levels,
        )
        rows.append({"split": split, "target": column, **metrics})
    return pd.DataFrame(rows)


def build_batch_metrics_frame(
    extracted: Mapping[str, np.ndarray],
    target_columns: Sequence[str],
    batch_names: Sequence[str],
    split: str,
    epsilon: float,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
) -> pd.DataFrame:
    """Score every target separately within each batch.

    A wide spread of these metrics across batches means the shared latent is
    still carrying batch identity into the biology head.

    Args:
        extracted (Mapping[str, np.ndarray]): Targets, parameters, and batch codes.
        target_columns (Sequence[str]): Ordered density target names.
        batch_names (Sequence[str]): Display names indexed by batch code.
        split (str): Split name recorded in the table.
        epsilon (float): Clamp applied before observed-value logits.
        coverage_levels (Sequence[float]): Central interval levels to score.

    Returns:
        pd.DataFrame: One row per ``(batch, target)`` pair.
    """

    batches = np.asarray(extracted["batches"], dtype=np.int64).reshape(-1)
    rows: list[dict[str, Any]] = []
    for batch_id in np.unique(batches):
        mask = batches == batch_id
        name = (
            str(batch_names[int(batch_id)])
            if int(batch_id) < len(batch_names)
            else str(batch_id)
        )
        for index, column in enumerate(target_columns):
            metrics = compute_target_metrics(
                np.asarray(extracted["targets"])[mask, index],
                np.asarray(extracted["pi"])[mask, index],
                np.asarray(extracted["mu"])[mask, index],
                np.asarray(extracted["sigma"])[mask, index],
                np.asarray(extracted["alpha"])[mask, index],
                epsilon,
                coverage_levels,
            )
            rows.append(
                {
                    "split": split,
                    "batch_id": int(batch_id),
                    "batch": name,
                    "target": column,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _grid(num_targets: int, ncols: int, width: float, height: float):
    """Create a figure and axes grid sized for a per-target panel set.

    Args:
        num_targets (int): Number of target panels required.
        ncols (int): Panels per row.
        width (float): Width in inches of one panel.
        height (float): Height in inches of one panel.

    Returns:
        tuple: Figure and a two-dimensional axes array.
    """

    ncols = max(1, min(int(ncols), int(num_targets)))
    nrows = int(np.ceil(num_targets / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width * ncols, height * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    return figure, axes


def _hide_unused(axes, num_targets: int) -> None:
    """Switch off axes beyond the number of plotted targets.

    Args:
        axes: Two-dimensional axes array.
        num_targets (int): Number of populated panels.

    Returns:
        None: Unused axes are hidden in place.
    """

    ncols = axes.shape[1]
    for index in range(num_targets, axes.shape[0] * ncols):
        row, col = divmod(index, ncols)
        axes[row][col].axis("off")


def _positive_arrays(
    extracted: Mapping[str, np.ndarray],
    index: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the masked positive-branch inputs for one target.

    Args:
        extracted (Mapping[str, np.ndarray]): Targets and parameter arrays.
        index (int): Target column position.
        epsilon (float): Clamp applied before observed-value logits.

    Returns:
        tuple: Observed logits and the matching mu, sigma, alpha, and pi arrays.
    """

    targets = np.asarray(extracted["targets"])[:, index]
    positive = targets > 0.0
    truth = ziln.clipped_logit(targets[positive], epsilon)
    return (
        truth,
        np.asarray(extracted["mu"])[positive, index],
        np.asarray(extracted["sigma"])[positive, index],
        np.asarray(extracted["alpha"])[positive, index],
        np.asarray(extracted["pi"])[positive, index],
    )


def plot_pit(
    extracted: Mapping[str, np.ndarray],
    target_columns: Sequence[str],
    epsilon: float,
    output_path: Path,
    config: Mapping[str, Any],
) -> None:
    """Plot PIT histograms and uniform QQ plots for the positive branch.

    Args:
        extracted (Mapping[str, np.ndarray]): Targets and parameter arrays.
        target_columns (Sequence[str]): Ordered density target names.
        epsilon (float): Clamp applied before observed-value logits.
        output_path (Path): Destination PNG path.
        config (Mapping[str, Any]): Plot settings.

    Returns:
        None: Figure is saved to disk.
    """

    bins = int(config.get("calibration_pit_bins", DEFAULT_PIT_BINS))
    num_targets = len(target_columns)
    figure, axes = plt.subplots(
        2,
        num_targets,
        figsize=(5 * num_targets, 9),
        constrained_layout=True,
        squeeze=False,
    )
    for index, column in enumerate(target_columns):
        truth, mu, sigma, alpha, _ = _positive_arrays(extracted, index, epsilon)
        if truth.size == 0:
            axes[0][index].axis("off")
            axes[1][index].axis("off")
            continue
        pit = ziln.positive_logit_cdf(mu, sigma, alpha, truth)

        histogram_axis = axes[0][index]
        histogram_axis.hist(
            pit, bins=bins, range=(0.0, 1.0), density=True, color="#4c72b0"
        )
        histogram_axis.axhline(1.0, linestyle="--", color="black", linewidth=1)
        ks = float(stats.kstest(pit, "uniform").statistic)
        histogram_axis.set(
            title=f"{column} PIT (KS={ks:.3f})",
            xlabel="PIT value",
            ylabel="Density",
            xlim=(0.0, 1.0),
        )
        histogram_axis.grid(alpha=0.25)

        qq_axis = axes[1][index]
        ordered = np.sort(pit)
        uniform = (np.arange(ordered.size) + 0.5) / ordered.size
        qq_axis.plot(uniform, ordered, color="#4c72b0", linewidth=1.5)
        qq_axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=1)
        qq_axis.set(
            title=f"{column} PIT QQ",
            xlabel="Uniform quantile",
            ylabel="Observed PIT quantile",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
        )
        qq_axis.grid(alpha=0.25)
    figure.savefig(output_path, dpi=int(config["figure_dpi"]))
    plt.close(figure)


def plot_corrected_scatter(
    extracted: Mapping[str, np.ndarray],
    target_columns: Sequence[str],
    epsilon: float,
    output_path: Path,
    config: Mapping[str, Any],
) -> None:
    """Scatter the skew-corrected logit mean against observed positive logits.

    This is the corrected counterpart of ``analyze.plot_ziln_density_scatter``,
    which plots the raw ``mu`` instead. Both correlations are annotated so the
    contribution of the skew correction is directly readable.

    Args:
        extracted (Mapping[str, np.ndarray]): Targets and parameter arrays.
        target_columns (Sequence[str]): Ordered density target names.
        epsilon (float): Clamp applied before observed-value logits.
        output_path (Path): Destination PNG path.
        config (Mapping[str, Any]): Plot settings.

    Returns:
        None: Figure is saved to disk.
    """

    num_targets = len(target_columns)
    figure, axes = _grid(num_targets, 2, 12.0, 6.0)
    point_size = float(config["point_size"])
    for index, column in enumerate(target_columns):
        row, col = divmod(index, axes.shape[1])
        axis = axes[row][col]
        truth, mu, sigma, alpha, _ = _positive_arrays(extracted, index, epsilon)
        if truth.size == 0:
            axis.axis("off")
            continue
        corrected = ziln.positive_logit_mean(mu, sigma, alpha)
        axis.scatter(truth, corrected, s=point_size, alpha=0.25, rasterized=True)
        lower = float(min(truth.min(), corrected.min()))
        upper = float(max(truth.max(), corrected.max()))
        axis.plot(
            [lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1
        )
        r_corrected = _safe_pearson(truth, corrected)
        r_mu = _safe_pearson(truth, mu)
        axis.set(
            title=(
                f"Positive {column}: corrected r={r_corrected:.3f} "
                f"(raw mu r={r_mu:.3f})"
            ),
            xlabel=f"True logit({column})",
            ylabel=f"Predicted E[Z|y>0] ({column})",
        )
        axis.grid(alpha=0.25)
    _hide_unused(axes, num_targets)
    figure.savefig(output_path, dpi=int(config["figure_dpi"]))
    plt.close(figure)


def plot_hurdle_reliability(
    extracted: Mapping[str, np.ndarray],
    target_columns: Sequence[str],
    output_path: Path,
    config: Mapping[str, Any],
) -> None:
    """Plot reliability curves for the structural-zero probability.

    Args:
        extracted (Mapping[str, np.ndarray]): Targets and parameter arrays.
        target_columns (Sequence[str]): Ordered density target names.
        output_path (Path): Destination PNG path.
        config (Mapping[str, Any]): Plot settings.

    Returns:
        None: Figure is saved to disk.
    """

    num_targets = len(target_columns)
    figure, axes = _grid(num_targets, 2, 7.0, 6.0)
    edges = np.linspace(0.0, 1.0, RELIABILITY_BINS + 1)
    for index, column in enumerate(target_columns):
        row, col = divmod(index, axes.shape[1])
        axis = axes[row][col]
        targets = np.asarray(extracted["targets"])[:, index]
        pi = np.asarray(extracted["pi"])[:, index]
        is_zero = (targets == 0.0).astype(np.float64)
        assignment = np.clip(np.digitize(pi, edges) - 1, 0, RELIABILITY_BINS - 1)
        centers: list[float] = []
        observed: list[float] = []
        counts: list[int] = []
        for bin_index in range(RELIABILITY_BINS):
            mask = assignment == bin_index
            if not np.any(mask):
                continue
            centers.append(float(pi[mask].mean()))
            observed.append(float(is_zero[mask].mean()))
            counts.append(int(mask.sum()))
        axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=1)
        if centers:
            sizes = np.asarray(counts, dtype=np.float64)
            sizes = 20.0 + 180.0 * sizes / sizes.max()
            axis.plot(centers, observed, color="#c44e52", linewidth=1.5, zorder=2)
            axis.scatter(
                centers, observed, s=sizes, color="#c44e52", zorder=3, alpha=0.8
            )
        brier = float(np.mean(np.square(pi - is_zero)))
        axis.set(
            title=f"{column} hurdle reliability (Brier={brier:.4f})",
            xlabel="Predicted pi",
            ylabel="Observed zero rate",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
        )
        axis.grid(alpha=0.25)
    _hide_unused(axes, num_targets)
    figure.savefig(output_path, dpi=int(config["figure_dpi"]))
    plt.close(figure)


def plot_interval_coverage(
    extracted: Mapping[str, np.ndarray],
    target_columns: Sequence[str],
    epsilon: float,
    output_path: Path,
    config: Mapping[str, Any],
) -> None:
    """Plot nominal against empirical central-interval coverage.

    A curve below the diagonal means the predictive intervals are too narrow and
    the uncertainty maps overstate confidence; above means they are too wide.

    Args:
        extracted (Mapping[str, np.ndarray]): Targets and parameter arrays.
        target_columns (Sequence[str]): Ordered density target names.
        epsilon (float): Clamp applied before observed-value logits.
        output_path (Path): Destination PNG path.
        config (Mapping[str, Any]): Plot settings.

    Returns:
        None: Figure is saved to disk.
    """

    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=1)
    for index, column in enumerate(target_columns):
        truth, mu, sigma, alpha, _ = _positive_arrays(extracted, index, epsilon)
        if truth.size == 0:
            continue
        empirical: list[float] = []
        for level in COVERAGE_SWEEP:
            lower_q, upper_q = _interval_bounds(level)
            lower = ziln.positive_logit_quantile(mu, sigma, alpha, lower_q)
            upper = ziln.positive_logit_quantile(mu, sigma, alpha, upper_q)
            empirical.append(float(((truth >= lower) & (truth <= upper)).mean()))
        axis.plot(COVERAGE_SWEEP, empirical, marker="o", linewidth=1.5, label=column)
    axis.set(
        title="Positive-branch predictive interval coverage",
        xlabel="Nominal coverage",
        ylabel="Empirical coverage",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.legend(loc="upper left", fontsize=9)
    axis.grid(alpha=0.25)
    figure.savefig(output_path, dpi=int(config["figure_dpi"]))
    plt.close(figure)


def run_calibration(
    extracted: Mapping[str, np.ndarray],
    target_columns: Sequence[str],
    batch_names: Sequence[str],
    split: str,
    output_dir: Path,
    config: Mapping[str, Any],
    epsilon: float,
) -> Path:
    """Write the complete calibration report for one split.

    Args:
        extracted (Mapping[str, np.ndarray]): Targets, parameters, and batch codes.
        target_columns (Sequence[str]): Ordered density target names.
        batch_names (Sequence[str]): Display names indexed by batch code.
        split (str): Split name recorded in the tables.
        output_dir (Path): Destination directory for tables and figures.
        config (Mapping[str, Any]): Analysis plot and calibration settings.
        epsilon (float): Clamp applied before observed-value logits.

    Returns:
        Path: Directory containing the written report.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_levels = tuple(
        float(value)
        for value in config.get("calibration_coverage_levels", DEFAULT_COVERAGE_LEVELS)
    )
    metrics = build_metrics_frame(
        extracted, target_columns, split, epsilon, coverage_levels
    )
    metrics.to_csv(output_dir / "calibration_metrics.csv", index=False)
    build_batch_metrics_frame(
        extracted, target_columns, batch_names, split, epsilon, coverage_levels
    ).to_csv(output_dir / "calibration_metrics_by_batch.csv", index=False)
    plot_pit(
        extracted,
        target_columns,
        epsilon,
        output_dir / "calibration_pit.png",
        config,
    )
    plot_corrected_scatter(
        extracted,
        target_columns,
        epsilon,
        output_dir / "calibration_scatter_corrected.png",
        config,
    )
    plot_hurdle_reliability(
        extracted,
        target_columns,
        output_dir / "calibration_hurdle_reliability.png",
        config,
    )
    plot_interval_coverage(
        extracted,
        target_columns,
        epsilon,
        output_dir / "calibration_interval_coverage.png",
        config,
    )
    print(metrics.to_string(index=False), flush=True)
    return output_dir


def load_extracted_from_frame(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Rebuild the arrays ``run_calibration`` needs from a saved wide table.

    Accepts the schema written by ``analyze.build_latent_umap_frame``, so the
    report can be regenerated from ``umap/latent_umap.csv`` without repeating
    model inference.

    Args:
        frame (pd.DataFrame): Wide table with per-target parameter columns.
        target_columns (Sequence[str]): Ordered density target names.

    Returns:
        tuple[dict[str, np.ndarray], list[str]]: Extracted arrays and batch names.
    """

    required = ["batch_id", "batch"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise KeyError(f"Table is missing required identity columns: {missing}")
    for column in target_columns:
        for prefix in ("", "pi_", "mu_", "sigma_", "alpha_"):
            name = f"{prefix}{column}"
            if name not in frame.columns:
                raise KeyError(f"Table is missing required column {name!r}.")
    batch_ids = frame["batch_id"].to_numpy(dtype=np.int64)
    pairs = (
        frame[["batch_id", "batch"]]
        .drop_duplicates()
        .sort_values("batch_id")
        .astype({"batch_id": int})
    )
    batch_names = ["" for _ in range(int(pairs["batch_id"].max()) + 1)]
    for batch_id, name in zip(pairs["batch_id"], pairs["batch"]):
        batch_names[int(batch_id)] = str(name)
    extracted: dict[str, np.ndarray] = {"batches": batch_ids}
    for key, prefix in (
        ("targets", ""),
        ("pi", "pi_"),
        ("mu", "mu_"),
        ("sigma", "sigma_"),
        ("alpha", "alpha_"),
    ):
        extracted[key] = np.column_stack(
            [
                frame[f"{prefix}{column}"].to_numpy(dtype=np.float64)
                for column in target_columns
            ]
        )
    return extracted, batch_names


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse standalone calibration command-line arguments.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        argparse.Namespace: Parsed calibration options.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("dann/config.yaml"),
        help="DANN configuration supplying targets, epsilon, and plot settings.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Saved wide table; defaults to analysis.output_dir/umap/latent_umap.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report directory; defaults to analysis.output_dir/calibration.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the calibration report from a saved table without model inference.

    Args:
        argv (Sequence[str] | None): Optional explicit command-line arguments.

    Returns:
        int: Process exit status, zero on success.
    """

    from dann.config import load_config

    args = parse_args(argv)
    config = load_config(args.config)
    analysis = config["analysis"]
    input_path = args.input or Path(analysis["output_dir"]) / "umap" / "latent_umap.csv"
    output_dir = args.output_dir or Path(analysis["output_dir"]) / "calibration"
    target_columns = list(config["data"]["target_columns"])
    frame = pd.read_csv(input_path)
    extracted, batch_names = load_extracted_from_frame(frame, target_columns)
    split = str(frame["split"].iloc[0]) if "split" in frame.columns else "unknown"
    print(
        f"Calibrating {len(frame):,} {split} rows from {input_path}.",
        flush=True,
    )
    run_calibration(
        extracted=extracted,
        target_columns=target_columns,
        batch_names=batch_names,
        split=split,
        output_dir=output_dir,
        config=analysis,
        epsilon=float(config["loss"]["logit_epsilon"]),
    )
    print(f"Calibration report written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fit real-line distributions to logit-transformed positive densities."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
from scipy import special, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

try:
    from .fit_density_distributions import (
        DENSITY_COLUMNS,
        adjust_positive_boundaries,
        load_observations,
        split_fitting_sample,
        stable_rng,
        validate_observations,
    )
except ImportError:
    from fit_density_distributions import (
        DENSITY_COLUMNS,
        adjust_positive_boundaries,
        load_observations,
        split_fitting_sample,
        stable_rng,
        validate_observations,
    )


MODEL_NAMES = (
    "normal",
    "logistic",
    "student_t",
    "generalized_normal",
    "skew_normal",
    "johnson_su",
)
EXPECTED_PARAMETER_COUNTS = {
    "normal": 2,
    "logistic": 2,
    "student_t": 3,
    "generalized_normal": 3,
    "skew_normal": 3,
    "johnson_su": 4,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options.

    Args:
        argv (Sequence[str] | None): Optional arguments, or ``None`` to use
            ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed command-line options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Fit real-line distributions to nonzero densities after a logit transform."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/workspaces/multimodal-pdac/data/PDAC/Raw/adata_assembled.h5ad"),
        help="Input AnnData .h5ad file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/workspaces/multimodal-pdac/data/PDAC/Results/distributions_tests/"
            "logit_positive"
        ),
        help="Directory for CSV tables, figures, and the summary.",
    )
    parser.add_argument("--batch-column", default="batch", help="Observation batch column.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of transformed positive values used for fitting.",
    )
    parser.add_argument(
        "--max-fit-observations",
        type=int,
        default=250_000,
        help="Maximum transformed values per density/scope; use 0 for no cap.",
    )
    parser.add_argument("--seed", type=int, default=20260721, help="Random seed.")
    return parser.parse_args(argv)


def transform_positive_values(values: np.ndarray) -> np.ndarray:
    """Boundary-correct and logit-transform nonzero density values.

    Args:
        values (np.ndarray): Original observations bounded by ``[0, 1]``.

    Returns:
        np.ndarray: Finite logit values for observations whose original value
            was greater than zero.
    """

    adjusted = adjust_positive_boundaries(np.asarray(values, dtype=float))
    positive = adjusted[values > 0.0]
    transformed = special.logit(positive)
    if not np.isfinite(transformed).all():
        raise ValueError("The logit transformation produced non-finite values.")
    return transformed


def describe_transformed_values(
    original_values: np.ndarray,
    transformed_values: np.ndarray,
    scope: str,
    group: str,
    density: str,
) -> dict[str, Any]:
    """Calculate descriptive statistics for one transformed positive group.

    Args:
        original_values (np.ndarray): Original bounded observations.
        transformed_values (np.ndarray): Logit-transformed positive observations.
        scope (str): Global or batch scope.
        group (str): Global label or batch name.
        density (str): Density column name.

    Returns:
        dict[str, Any]: One row of original and transformed summary statistics.
    """

    quantiles = np.quantile(
        transformed_values, [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]
    )
    return {
        "scope": scope,
        "group": group,
        "density": density,
        "n_total": int(original_values.size),
        "n_zero": int(np.count_nonzero(original_values == 0.0)),
        "n_positive": int(transformed_values.size),
        "p_zero": float(np.mean(original_values == 0.0)),
        "logit_mean": float(np.mean(transformed_values)),
        "logit_std": float(np.std(transformed_values, ddof=1)),
        "logit_skewness": float(stats.skew(transformed_values, bias=False)),
        "logit_excess_kurtosis": float(
            stats.kurtosis(transformed_values, fisher=True, bias=False)
        ),
        "logit_min": float(np.min(transformed_values)),
        "logit_q001": float(quantiles[0]),
        "logit_q01": float(quantiles[1]),
        "logit_q25": float(quantiles[2]),
        "logit_median": float(quantiles[3]),
        "logit_q75": float(quantiles[4]),
        "logit_q99": float(quantiles[5]),
        "logit_q999": float(quantiles[6]),
        "logit_max": float(np.max(transformed_values)),
    }


def fit_distribution(model: str, values: np.ndarray) -> dict[str, float]:
    """Fit a real-line distribution by maximum likelihood.

    Args:
        model (str): Candidate distribution name.
        values (np.ndarray): Finite logit-transformed training observations.

    Returns:
        dict[str, float]: Named maximum-likelihood parameters.
    """

    x = np.asarray(values, dtype=float)
    if x.size < 10:
        raise ValueError("At least 10 transformed observations are required.")
    if model == "normal":
        location, scale = stats.norm.fit(x)
        parameters = {"loc": float(location), "scale": float(scale)}
    elif model == "logistic":
        location, scale = stats.logistic.fit(x)
        parameters = {"loc": float(location), "scale": float(scale)}
    elif model == "student_t":
        degrees_freedom, location, scale = stats.t.fit(x)
        parameters = {
            "df": float(degrees_freedom),
            "loc": float(location),
            "scale": float(scale),
        }
    elif model == "generalized_normal":
        shape, location, scale = stats.gennorm.fit(x)
        parameters = {
            "shape": float(shape),
            "loc": float(location),
            "scale": float(scale),
        }
    elif model == "skew_normal":
        shape, location, scale = stats.skewnorm.fit(x)
        parameters = {
            "shape": float(shape),
            "loc": float(location),
            "scale": float(scale),
        }
    elif model == "johnson_su":
        shape_a, shape_b, location, scale = stats.johnsonsu.fit(x)
        parameters = {
            "a": float(shape_a),
            "b": float(shape_b),
            "loc": float(location),
            "scale": float(scale),
        }
    else:
        raise ValueError(f"Unknown model: {model}")
    if parameters["scale"] <= 0.0 or not all(
        np.isfinite(value) for value in parameters.values()
    ):
        raise ValueError(f"{model} produced invalid parameters: {parameters}")
    return parameters


def distribution_logpdf(
    model: str, values: np.ndarray, parameters: dict[str, float]
) -> np.ndarray:
    """Evaluate a fitted real-line log-density.

    Args:
        model (str): Candidate distribution name.
        values (np.ndarray): Real-valued observations.
        parameters (dict[str, float]): Named fitted parameters.

    Returns:
        np.ndarray: Log-density at each observation.
    """

    x = np.asarray(values, dtype=float)
    if model == "normal":
        return stats.norm.logpdf(x, loc=parameters["loc"], scale=parameters["scale"])
    if model == "logistic":
        return stats.logistic.logpdf(
            x, loc=parameters["loc"], scale=parameters["scale"]
        )
    if model == "student_t":
        return stats.t.logpdf(
            x,
            parameters["df"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )
    if model == "generalized_normal":
        return stats.gennorm.logpdf(
            x,
            parameters["shape"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )
    if model == "skew_normal":
        return stats.skewnorm.logpdf(
            x,
            parameters["shape"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )
    if model == "johnson_su":
        return stats.johnsonsu.logpdf(
            x,
            parameters["a"],
            parameters["b"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )
    raise ValueError(f"Unknown model: {model}")


def distribution_cdf(
    model: str, values: np.ndarray, parameters: dict[str, float]
) -> np.ndarray:
    """Evaluate a fitted real-line cumulative distribution.

    Args:
        model (str): Candidate distribution name.
        values (np.ndarray): Real-valued observations.
        parameters (dict[str, float]): Named fitted parameters.

    Returns:
        np.ndarray: Cumulative probabilities.
    """

    x = np.asarray(values, dtype=float)
    if model == "normal":
        return stats.norm.cdf(x, loc=parameters["loc"], scale=parameters["scale"])
    if model == "logistic":
        return stats.logistic.cdf(
            x, loc=parameters["loc"], scale=parameters["scale"]
        )
    if model == "student_t":
        return stats.t.cdf(
            x,
            parameters["df"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )
    if model == "generalized_normal":
        return stats.gennorm.cdf(
            x,
            parameters["shape"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )
    if model == "skew_normal":
        return stats.skewnorm.cdf(
            x,
            parameters["shape"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )
    if model == "johnson_su":
        return stats.johnsonsu.cdf(
            x,
            parameters["a"],
            parameters["b"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )
    raise ValueError(f"Unknown model: {model}")


def distribution_ppf(
    model: str, probabilities: np.ndarray, parameters: dict[str, float]
) -> np.ndarray:
    """Evaluate fitted real-line quantiles.

    Args:
        model (str): Candidate distribution name.
        probabilities (np.ndarray): Probabilities strictly between zero and one.
        parameters (dict[str, float]): Named fitted parameters.

    Returns:
        np.ndarray: Distribution quantiles.
    """

    p = np.asarray(probabilities, dtype=float)
    if model == "normal":
        return stats.norm.ppf(p, loc=parameters["loc"], scale=parameters["scale"])
    if model == "logistic":
        return stats.logistic.ppf(
            p, loc=parameters["loc"], scale=parameters["scale"]
        )
    if model == "student_t":
        return stats.t.ppf(
            p,
            parameters["df"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )
    if model == "generalized_normal":
        return stats.gennorm.ppf(
            p,
            parameters["shape"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )
    if model == "skew_normal":
        return stats.skewnorm.ppf(
            p,
            parameters["shape"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )
    if model == "johnson_su":
        return stats.johnsonsu.ppf(
            p,
            parameters["a"],
            parameters["b"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )
    raise ValueError(f"Unknown model: {model}")


def ks_distance(
    values: np.ndarray, model: str, parameters: dict[str, float]
) -> float:
    """Calculate the empirical-versus-fitted Kolmogorov-Smirnov distance.

    Args:
        values (np.ndarray): Held-out transformed observations.
        model (str): Candidate distribution name.
        parameters (dict[str, float]): Named fitted parameters.

    Returns:
        float: Maximum CDF distance.
    """

    ordered = np.sort(np.asarray(values, dtype=float))
    fitted = np.clip(distribution_cdf(model, ordered, parameters), 0.0, 1.0)
    lower = np.arange(ordered.size, dtype=float) / ordered.size
    upper = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    return float(max(np.max(fitted - lower), np.max(upper - fitted)))


def fit_candidate(
    train: np.ndarray,
    test: np.ndarray,
    model: str,
    scope: str,
    group: str,
    density: str,
    n_total: int,
    n_positive: int,
) -> dict[str, Any]:
    """Fit and score one transformed-value distribution.

    Args:
        train (np.ndarray): Transformed training values.
        test (np.ndarray): Transformed held-out values.
        model (str): Candidate distribution name.
        scope (str): Global or batch scope.
        group (str): Global label or batch name.
        density (str): Density column name.
        n_total (int): Number of original observations.
        n_positive (int): Number of original positive observations.

    Returns:
        dict[str, Any]: Fit parameters, convergence, and diagnostics.
    """

    base: dict[str, Any] = {
        "scope": scope,
        "group": group,
        "density": density,
        "model": model,
        "n_total": n_total,
        "n_positive": n_positive,
        "n_train": int(train.size),
        "n_test": int(test.size),
    }
    try:
        parameters = fit_distribution(model, train)
        train_logpdf = distribution_logpdf(model, train, parameters)
        test_logpdf = distribution_logpdf(model, test, parameters)
        if not np.isfinite(train_logpdf).all() or not np.isfinite(test_logpdf).all():
            raise ValueError("Non-finite fitted log-density.")
        train_log_likelihood = float(train_logpdf.sum())
        test_log_likelihood = float(test_logpdf.sum())
        parameter_count = len(parameters)
        base.update(
            {
                "converged": True,
                "error": "",
                "parameters": json.dumps(parameters, sort_keys=True),
                "parameter_count": parameter_count,
                "train_log_likelihood": train_log_likelihood,
                "test_log_likelihood": test_log_likelihood,
                "test_nll_per_observation": -test_log_likelihood / test.size,
                "aic": 2 * parameter_count - 2 * train_log_likelihood,
                "bic": parameter_count * math.log(train.size)
                - 2 * train_log_likelihood,
                "ks_distance": ks_distance(test, model, parameters),
            }
        )
    except Exception as error:  # Candidate failures must not abort other fits.
        base.update(
            {
                "converged": False,
                "error": f"{type(error).__name__}: {error}",
                "parameters": "{}",
                "parameter_count": 0,
                "train_log_likelihood": float("nan"),
                "test_log_likelihood": float("nan"),
                "test_nll_per_observation": float("nan"),
                "aic": float("nan"),
                "bic": float("nan"),
                "ks_distance": float("nan"),
            }
        )
    return base


def analyze_group(
    original_values: np.ndarray,
    scope: str,
    group: str,
    density: str,
    train_fraction: float,
    max_fit_observations: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Transform, describe, and fit every candidate for one group.

    Args:
        original_values (np.ndarray): Original bounded observations.
        scope (str): Global or batch scope.
        group (str): Global label or batch name.
        density (str): Density column name.
        train_fraction (float): Fraction used for fitting.
        max_fit_observations (int): Sample cap, or zero for no cap.
        seed (int): Base random seed.

    Returns:
        tuple[dict[str, Any], list[dict[str, Any]]]: Description and fit rows.
    """

    transformed = transform_positive_values(original_values)
    description = describe_transformed_values(
        original_values, transformed, scope, group, density
    )
    train, test = split_fitting_sample(
        transformed,
        train_fraction,
        max_fit_observations,
        stable_rng(seed, "logit-positive", scope, group, density),
    )
    fits = [
        fit_candidate(
            train,
            test,
            model,
            scope,
            group,
            density,
            int(original_values.size),
            int(transformed.size),
        )
        for model in MODEL_NAMES
    ]
    return description, fits


def select_winners(candidate_fits: pd.DataFrame) -> pd.DataFrame:
    """Select the lowest held-out NLL candidate for each group.

    Args:
        candidate_fits (pd.DataFrame): All transformed-value candidate fits.

    Returns:
        pd.DataFrame: One winning row per scope, group, and density.
    """

    successful = candidate_fits[
        candidate_fits["converged"]
        & np.isfinite(candidate_fits["test_nll_per_observation"])
    ]
    if successful.empty:
        raise RuntimeError("No transformed-value candidate converged.")
    winner_indices = successful.groupby(
        ["scope", "group", "density"], sort=False, observed=True
    )["test_nll_per_observation"].idxmin()
    winners = successful.loc[winner_indices].copy()
    winners["selection_criterion"] = "minimum held-out NLL per observation"
    return winners.sort_values(["scope", "group", "density"]).reset_index(drop=True)


def run_analysis(
    observations: pd.DataFrame,
    batch_column: str,
    density_columns: Sequence[str],
    train_fraction: float,
    max_fit_observations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run transformed-value fits globally and within every batch.

    Args:
        observations (pd.DataFrame): Validated batch and density observations.
        batch_column (str): Batch column name.
        density_columns (Sequence[str]): Density columns to analyze.
        train_fraction (float): Fraction used for fitting.
        max_fit_observations (int): Per-group sample cap, or zero for no cap.
        seed (int): Base random seed.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Descriptions, all fits,
            and selected winners.
    """

    descriptions: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = [("global", "all", observations)]
    groups.extend(
        ("batch", str(batch), frame)
        for batch, frame in observations.groupby(
            batch_column, observed=True, sort=True
        )
    )
    for scope, group, frame in groups:
        print(f"Analyzing logit positives for {scope}: {group}", flush=True)
        for density in density_columns:
            description, candidate_rows = analyze_group(
                frame[density].to_numpy(dtype=float),
                scope,
                group,
                density,
                train_fraction,
                max_fit_observations,
                seed,
            )
            descriptions.append(description)
            fits.extend(candidate_rows)
    description_frame = pd.DataFrame(descriptions)
    fit_frame = pd.DataFrame(fits)
    return description_frame, fit_frame, select_winners(fit_frame)


def plot_global_diagnostics(
    observations: pd.DataFrame,
    winners: pd.DataFrame,
    density_columns: Sequence[str],
    output_path: Path,
    seed: int,
) -> None:
    """Plot global logit histograms, fitted densities, and Q-Q diagnostics.

    Args:
        observations (pd.DataFrame): Full original observations.
        winners (pd.DataFrame): Selected transformed-value fits.
        density_columns (Sequence[str]): Density columns to plot.
        output_path (Path): Figure destination.
        seed (int): Display subsampling seed.

    Returns:
        None: Writes the diagnostic figure.
    """

    figure, axes = plt.subplots(
        len(density_columns),
        2,
        figsize=(13, 3.5 * len(density_columns)),
        constrained_layout=True,
    )
    global_winners = winners[winners["scope"] == "global"].set_index("density")
    probabilities = np.linspace(0.001, 0.999, 500)
    for row_index, density in enumerate(density_columns):
        transformed = transform_positive_values(
            observations[density].to_numpy(dtype=float)
        )
        if transformed.size > 250_000:
            rng = stable_rng(seed, "logit-positive-plot", density)
            display_values = transformed[
                rng.choice(transformed.size, 250_000, replace=False)
            ]
        else:
            display_values = transformed
        winner = global_winners.loc[density]
        model = str(winner["model"])
        parameters = json.loads(str(winner["parameters"]))
        lower, upper = np.quantile(display_values, [0.001, 0.999])
        grid = np.linspace(lower, upper, 1_000)
        pdf = np.exp(distribution_logpdf(model, grid, parameters))

        histogram_axis = axes[row_index, 0]
        histogram_axis.hist(
            display_values,
            bins=120,
            range=(lower, upper),
            density=True,
            alpha=0.45,
            color="tab:blue",
        )
        histogram_axis.plot(grid, pdf, color="tab:red", lw=2)
        histogram_axis.set(
            title=f"{density}: {model}",
            xlabel="logit(boundary-corrected positive density)",
            ylabel="Probability density",
            xlim=(lower, upper),
        )
        histogram_axis.grid(alpha=0.2)

        qq_axis = axes[row_index, 1]
        empirical_quantiles = np.quantile(display_values, probabilities)
        fitted_quantiles = distribution_ppf(model, probabilities, parameters)
        qq_axis.scatter(fitted_quantiles, empirical_quantiles, s=6, alpha=0.6)
        diagonal_min = float(min(fitted_quantiles.min(), empirical_quantiles.min()))
        diagonal_max = float(max(fitted_quantiles.max(), empirical_quantiles.max()))
        qq_axis.plot(
            [diagonal_min, diagonal_max],
            [diagonal_min, diagonal_max],
            color="tab:red",
            lw=1.5,
        )
        qq_axis.set(
            title=f"Q-Q; held-out KS={winner['ks_distance']:.4f}",
            xlabel="Fitted quantiles",
            ylabel="Observed quantiles",
        )
        qq_axis.grid(alpha=0.2)
    figure.suptitle("Global logit-positive distribution diagnostics", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_batch_heatmaps(winners: pd.DataFrame, output_path: Path) -> None:
    """Plot batch winners, held-out NLL, and KS distance.

    Args:
        winners (pd.DataFrame): Selected transformed-value fits.
        output_path (Path): Figure destination.

    Returns:
        None: Writes the batch heatmap figure.
    """

    batch_winners = winners[winners["scope"] == "batch"]
    model_matrix = batch_winners.pivot(
        index="group", columns="density", values="model"
    )
    nll_matrix = batch_winners.pivot(
        index="group", columns="density", values="test_nll_per_observation"
    ).reindex(model_matrix.index)
    ks_matrix = batch_winners.pivot(
        index="group", columns="density", values="ks_distance"
    ).reindex(model_matrix.index)
    model_codes = model_matrix.replace(
        {model: index for index, model in enumerate(MODEL_NAMES)}
    ).astype(float)
    model_cmap = ListedColormap(
        list(plt.get_cmap("tab10").colors[: len(MODEL_NAMES)])
    )
    model_boundaries = np.arange(-0.5, len(MODEL_NAMES) + 0.5, 1.0)
    model_norm = BoundaryNorm(model_boundaries, model_cmap.N)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, max(11, 0.27 * len(model_matrix))),
        constrained_layout=True,
    )
    model_image = axes[0].imshow(
        model_codes, aspect="auto", cmap=model_cmap, norm=model_norm
    )
    nll_image = axes[1].imshow(nll_matrix, aspect="auto", cmap="viridis")
    ks_image = axes[2].imshow(ks_matrix, aspect="auto", cmap="magma")
    for axis, title in zip(
        axes,
        ["Selected family", "Held-out NLL / observation", "Held-out KS distance"],
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xticks(np.arange(len(model_matrix.columns)))
        axis.set_xticklabels(
            [column.replace("Density_", "") for column in model_matrix.columns],
            rotation=45,
            ha="right",
        )
        axis.set_yticks(np.arange(len(model_matrix.index)))
        axis.set_yticklabels(model_matrix.index, fontsize=7)
    model_colorbar = figure.colorbar(
        model_image,
        ax=axes[0],
        ticks=np.arange(len(MODEL_NAMES)),
        boundaries=model_boundaries,
        shrink=0.7,
    )
    model_colorbar.ax.set_yticklabels(MODEL_NAMES)
    figure.colorbar(nll_image, ax=axes[1], shrink=0.7)
    figure.colorbar(ks_image, ax=axes[2], shrink=0.7)
    figure.suptitle("Batch logit-positive distribution results", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_summary(
    descriptions: pd.DataFrame,
    winners: pd.DataFrame,
    candidate_fits: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a concise Markdown summary of transformed-value fits.

    Args:
        descriptions (pd.DataFrame): Transformed descriptive statistics.
        winners (pd.DataFrame): Selected transformed-value fits.
        candidate_fits (pd.DataFrame): Every candidate fit.
        output_path (Path): Markdown destination.

    Returns:
        None: Writes the summary.
    """

    global_descriptions = descriptions[descriptions["scope"] == "global"].set_index(
        "density"
    )
    global_winners = winners[winners["scope"] == "global"].set_index("density")
    batch_winners = winners[winners["scope"] == "batch"]
    lines = [
        "# Logit-transformed positive-density fitting summary",
        "",
        "Zeros were excluded. Positive values received the same boundary correction "
        "used by the bounded analysis before applying the logit transform.",
        "",
        "## Global selections",
        "",
    ]
    for density in DENSITY_COLUMNS:
        description = global_descriptions.loc[density]
        winner = global_winners.loc[density]
        lines.append(
            f"- **{density}**: `{winner['model']}` "
            f"({int(winner['parameter_count'])} parameters); "
            f"logit skewness={description['logit_skewness']:.3f}, "
            f"excess kurtosis={description['logit_excess_kurtosis']:.3f}, "
            f"held-out NLL/observation={winner['test_nll_per_observation']:.5f}, "
            f"KS={winner['ks_distance']:.5f}."
        )
    lines.extend(["", "## Batch selections", ""])
    counts = (
        batch_winners.groupby(["density", "model"], observed=True)
        .size()
        .rename("batches")
        .reset_index()
    )
    for density in DENSITY_COLUMNS:
        density_counts = counts[counts["density"] == density].sort_values(
            "batches", ascending=False
        )
        rendered = ", ".join(
            f"`{row.model}`: {int(row.batches)}"
            for row in density_counts.itertuples(index=False)
        )
        lines.append(f"- **{density}**: {rendered}.")
    failed = candidate_fits[~candidate_fits["converged"]]
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Winners minimize held-out negative log likelihood per observation.",
            "- Beta was omitted because logit-transformed values have real-line support.",
            "- AIC, BIC, and held-out KS distance are available in `candidate_fits.csv`.",
            f"- Candidate fits that failed to converge: {len(failed)}.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_outputs(
    observations: pd.DataFrame,
    descriptions: pd.DataFrame,
    candidate_fits: pd.DataFrame,
    winners: pd.DataFrame,
    density_columns: Sequence[str],
    output_dir: Path,
    seed: int,
) -> None:
    """Save transformed fit tables, figures, and summary.

    Args:
        observations (pd.DataFrame): Full original observations.
        descriptions (pd.DataFrame): Transformed descriptive statistics.
        candidate_fits (pd.DataFrame): Every candidate fit.
        winners (pd.DataFrame): Selected transformed-value fits.
        density_columns (Sequence[str]): Density columns to plot.
        output_dir (Path): Artifact destination.
        seed (int): Plot subsampling seed.

    Returns:
        None: Writes all artifacts.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    descriptions.to_csv(
        output_dir / "transformed_descriptive_statistics.csv", index=False
    )
    candidate_fits.to_csv(output_dir / "candidate_fits.csv", index=False)
    winners.to_csv(output_dir / "selected_models.csv", index=False)
    plot_global_diagnostics(
        observations,
        winners,
        density_columns,
        output_dir / "global_logit_distribution_diagnostics.png",
        seed,
    )
    plot_batch_heatmaps(
        winners, output_dir / "batch_logit_distribution_heatmaps.png"
    )
    write_summary(
        descriptions, winners, candidate_fits, output_dir / "summary.md"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line transformed positive-density analysis.

    Args:
        argv (Sequence[str] | None): Optional command-line arguments.

    Returns:
        int: Process exit status, zero on success.
    """

    args = parse_args(argv)
    observations = load_observations(args.input, args.batch_column, DENSITY_COLUMNS)
    validate_observations(observations, args.batch_column, DENSITY_COLUMNS)
    descriptions, candidate_fits, winners = run_analysis(
        observations,
        args.batch_column,
        DENSITY_COLUMNS,
        args.train_fraction,
        args.max_fit_observations,
        args.seed,
    )
    save_outputs(
        observations,
        descriptions,
        candidate_fits,
        winners,
        DENSITY_COLUMNS,
        args.output_dir,
        args.seed,
    )
    print(f"Results written to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

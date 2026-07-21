#!/usr/bin/env python3
"""Fit bounded hurdle distributions to PDAC density observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


DENSITY_COLUMNS = (
    "Density_Collagen",
    "Density_CD8",
    "Density_Tumor",
    "Density_Stroma",
)
MODEL_NAMES = (
    "beta",
    "generalized_beta",
    "kumaraswamy",
    "logit_normal",
    "skew_logit_normal",
    "johnson_sb",
    "truncated_normal",
)
BOUNDARY_ALPHA = 0.5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options.

    Args:
        argv (Sequence[str] | None): Optional argument sequence. Uses ``sys.argv``
            when omitted.

    Returns:
        argparse.Namespace: Parsed command-line options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Fit bounded two-part hurdle distributions globally and by AnnData batch."
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
            "/workspaces/multimodal-pdac/data/PDAC/Results/distributions_tests"
        ),
        help="Directory for CSV tables, figures, and the summary.",
    )
    parser.add_argument("--batch-column", default="batch", help="Observation batch column.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of the fitting sample used for maximum-likelihood fitting.",
    )
    parser.add_argument(
        "--max-fit-observations",
        type=int,
        default=250_000,
        help=(
            "Maximum observations per density/scope used for train/test fitting. "
            "Use 0 for no cap."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260717, help="Random seed.")
    return parser.parse_args(argv)


def load_observations(
    input_path: Path, batch_column: str, density_columns: Sequence[str]
) -> pd.DataFrame:
    """Load only the observation columns required by the analysis.

    Args:
        input_path (Path): Path to the AnnData ``.h5ad`` file.
        batch_column (str): Name of the batch column in ``adata.obs``.
        density_columns (Sequence[str]): Density columns to load.

    Returns:
        pd.DataFrame: In-memory batch and density observations.
    """

    adata = ad.read_h5ad(input_path, backed="r")
    required = [batch_column, *density_columns]
    missing = [column for column in required if column not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing required observation columns: {missing}")
    observations = adata.obs[required].copy()
    if getattr(adata, "file", None) is not None:
        adata.file.close()
    return observations


def validate_observations(
    observations: pd.DataFrame, batch_column: str, density_columns: Sequence[str]
) -> None:
    """Validate batch labels and bounded density values.

    Args:
        observations (pd.DataFrame): Batch and density observations.
        batch_column (str): Name of the batch column.
        density_columns (Sequence[str]): Density columns to validate.

    Returns:
        None: Raises an exception when values are missing or outside ``[0, 1]``.
    """

    if observations[batch_column].isna().any():
        raise ValueError(f"{batch_column!r} contains missing values.")
    for column in density_columns:
        values = observations[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column!r} contains non-finite values.")
        if np.any((values < 0.0) | (values > 1.0)):
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            raise ValueError(
                f"{column!r} must be bounded by [0, 1], observed [{minimum}, {maximum}]."
            )


def stable_rng(seed: int, *labels: str) -> np.random.Generator:
    """Create a reproducible random generator for a named analysis group.

    Args:
        seed (int): User-supplied base seed.
        *labels (str): Stable labels identifying the group.

    Returns:
        np.random.Generator: Deterministically seeded NumPy generator.
    """

    digest = hashlib.sha256("|".join(labels).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng((seed + offset) % (2**63 - 1))


def split_fitting_sample(
    values: np.ndarray,
    train_fraction: float,
    max_observations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample observations if needed and make a random train/test split.

    Args:
        values (np.ndarray): One-dimensional bounded density observations.
        train_fraction (float): Fraction assigned to training.
        max_observations (int): Maximum sample size, or zero for no cap.
        rng (np.random.Generator): Reproducible random generator.

    Returns:
        tuple[np.ndarray, np.ndarray]: Training and test observations.
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    sample = np.asarray(values, dtype=float)
    if max_observations > 0 and sample.size > max_observations:
        sample = sample[rng.choice(sample.size, max_observations, replace=False)]
    permutation = rng.permutation(sample.size)
    train_size = min(max(int(sample.size * train_fraction), 1), sample.size - 1)
    return sample[permutation[:train_size]], sample[permutation[train_size:]]


def describe_values(
    values: np.ndarray, scope: str, group: str, density: str
) -> dict[str, Any]:
    """Calculate full-sample descriptive statistics.

    Args:
        values (np.ndarray): Bounded density values.
        scope (str): Analysis scope, either global or batch.
        group (str): Global label or batch name.
        density (str): Density column name.

    Returns:
        dict[str, Any]: One row of descriptive statistics.
    """

    values = np.asarray(values, dtype=float)
    positive = values[values > 0.0]
    interior = values[(values > 0.0) & (values < 1.0)]
    quantiles = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])
    positive_quantiles = np.quantile(positive, [0.01, 0.5, 0.99])
    return {
        "scope": scope,
        "group": group,
        "density": density,
        "n": int(values.size),
        "n_zero": int(np.count_nonzero(values == 0.0)),
        "n_one": int(np.count_nonzero(values == 1.0)),
        "n_interior": int(interior.size),
        "p_zero": float(np.mean(values == 0.0)),
        "p_exists": float(np.mean(values > 0.0)),
        "p_one": float(np.mean(values == 1.0)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "q01": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "max": float(np.max(values)),
        "positive_q01": float(positive_quantiles[0]),
        "positive_median": float(positive_quantiles[1]),
        "positive_q99": float(positive_quantiles[2]),
    }


def adjust_positive_boundaries(values: np.ndarray) -> np.ndarray:
    """Map positive proportions from ``(0, 1]`` into the open interval ``(0, 1)``.

    The Smithson-Verkuilen correction is applied only to positive observations,
    preserving exact zeros for the hurdle component while treating exact ones
    as boundary values of the continuous positive distribution.

    Args:
        values (np.ndarray): Bounded observations in ``[0, 1]``.

    Returns:
        np.ndarray: Copy with zeros unchanged and positive values in ``(0, 1)``.
    """

    adjusted = np.asarray(values, dtype=float).copy()
    positive_mask = adjusted > 0.0
    positive_count = int(np.count_nonzero(positive_mask))
    if positive_count == 0:
        return adjusted
    adjusted[positive_mask] = (
        adjusted[positive_mask] * (positive_count - 1.0) + 0.5
    ) / positive_count
    return adjusted


def fit_interior_distribution(model: str, values: np.ndarray) -> dict[str, float]:
    """Fit one continuous distribution to values strictly inside ``(0, 1)``.

    Args:
        model (str): Candidate model name.
        values (np.ndarray): Training observations strictly inside ``(0, 1)``.

    Returns:
        dict[str, float]: Named maximum-likelihood parameter estimates.
    """

    x = np.asarray(values, dtype=float)
    if x.size < 10:
        raise ValueError("At least 10 interior observations are required.")
    if np.any((x <= 0.0) | (x >= 1.0)):
        raise ValueError("Continuous fitting values must lie strictly inside (0, 1).")

    if model == "beta":
        alpha, beta, _, _ = stats.beta.fit(x, floc=0.0, fscale=1.0)
        return {"alpha": float(alpha), "beta": float(beta)}

    if model == "generalized_beta":
        beta_alpha, beta_beta, _, _ = stats.beta.fit(x, floc=0.0, fscale=1.0)

        def objective(log_parameters: np.ndarray) -> float:
            """Evaluate negative generalized-Beta log likelihood.

            Args:
                log_parameters (np.ndarray): Log shape parameters ``a``, ``p``,
                    and ``q``.

            Returns:
                float: Negative log likelihood.
            """

            a, p, q = np.exp(log_parameters)
            transformed = np.power(x, a)
            log_likelihood = (
                x.size * (math.log(a) - special.betaln(p, q))
                + (a * p - 1.0) * np.log(x).sum()
                + (q - 1.0) * np.log1p(-transformed).sum()
            )
            return float(-log_likelihood) if np.isfinite(log_likelihood) else np.inf

        initial = np.log([1.0, beta_alpha, beta_beta])
        bounds = [(-6.0, 6.0), (-10.0, 10.0), (-10.0, 10.0)]
        result = optimize.minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if not result.success:
            result = optimize.minimize(
                objective,
                result.x,
                method="Powell",
                bounds=bounds,
                options={"maxiter": 1_000},
            )
        if not result.success:
            raise RuntimeError(f"Generalized-Beta optimization failed: {result.message}")
        a, p, q = np.exp(result.x)
        return {"a": float(a), "p": float(p), "q": float(q)}

    if model == "kumaraswamy":
        beta_alpha, beta_beta, _, _ = stats.beta.fit(x, floc=0.0, fscale=1.0)

        def objective(log_parameters: np.ndarray) -> float:
            """Evaluate negative Kumaraswamy log likelihood.

            Args:
                log_parameters (np.ndarray): Log shape parameters ``a`` and ``b``.

            Returns:
                float: Negative log likelihood.
            """

            a, b = np.exp(log_parameters)
            log_one_minus_xa = np.log1p(-np.power(x, a))
            log_likelihood = (
                x.size * (math.log(a) + math.log(b))
                + (a - 1.0) * np.log(x).sum()
                + (b - 1.0) * log_one_minus_xa.sum()
            )
            return float(-log_likelihood) if np.isfinite(log_likelihood) else np.inf

        result = optimize.minimize(
            objective,
            np.log([beta_alpha, beta_beta]),
            method="L-BFGS-B",
            bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        )
        if not result.success:
            raise RuntimeError(f"Kumaraswamy optimization failed: {result.message}")
        a, b = np.exp(result.x)
        return {"a": float(a), "b": float(b)}

    if model == "logit_normal":
        transformed = special.logit(x)
        sigma = float(np.std(transformed, ddof=0))
        if not sigma > 0.0:
            raise ValueError("Logit-normal scale is not positive.")
        return {"mu": float(np.mean(transformed)), "sigma": sigma}

    if model == "skew_logit_normal":
        transformed = special.logit(x)
        shape, location, scale = stats.skewnorm.fit(transformed)
        if not scale > 0.0:
            raise ValueError("Skew-logit-normal scale is not positive.")
        return {
            "shape": float(shape),
            "loc": float(location),
            "scale": float(scale),
        }

    if model == "johnson_sb":
        shape_a, shape_b, _, _ = stats.johnsonsb.fit(x, floc=0.0, fscale=1.0)
        return {"a": float(shape_a), "b": float(shape_b)}

    if model == "truncated_normal":
        initial_scale = max(float(np.std(x, ddof=0)), 1e-4)

        def objective(parameters: np.ndarray) -> float:
            """Evaluate negative truncated-normal log likelihood.

            Args:
                parameters (np.ndarray): Underlying normal location and log scale.

            Returns:
                float: Negative log likelihood.
            """

            location, log_scale = parameters
            scale = math.exp(log_scale)
            lower = -location / scale
            upper = (1.0 - location) / scale
            log_pdf = stats.truncnorm.logpdf(
                x, lower, upper, loc=location, scale=scale
            )
            total = np.sum(log_pdf)
            return float(-total) if np.isfinite(total) else np.inf

        result = optimize.minimize(
            objective,
            np.array([float(np.mean(x)), math.log(initial_scale)]),
            method="L-BFGS-B",
            bounds=[(-5.0, 5.0), (-10.0, 3.0)],
        )
        if not result.success:
            result = optimize.minimize(
                objective,
                result.x,
                method="Powell",
                bounds=[(-5.0, 5.0), (-10.0, 3.0)],
                options={"maxiter": 1_000},
            )
        if not result.success:
            raise RuntimeError(f"Truncated-normal optimization failed: {result.message}")
        return {"loc": float(result.x[0]), "scale": float(math.exp(result.x[1]))}

    raise ValueError(f"Unknown model: {model}")


def interior_logpdf(
    model: str, values: np.ndarray, parameters: dict[str, float]
) -> np.ndarray:
    """Evaluate a fitted interior log-density.

    Args:
        model (str): Candidate model name.
        values (np.ndarray): Values strictly inside ``(0, 1)``.
        parameters (dict[str, float]): Named fitted parameters.

    Returns:
        np.ndarray: Log-density for each value.
    """

    x = np.asarray(values, dtype=float)
    if model == "beta":
        return stats.beta.logpdf(x, parameters["alpha"], parameters["beta"])
    if model == "generalized_beta":
        a, p, q = parameters["a"], parameters["p"], parameters["q"]
        return (
            math.log(a)
            - special.betaln(p, q)
            + (a * p - 1.0) * np.log(x)
            + (q - 1.0) * np.log1p(-np.power(x, a))
        )
    if model == "kumaraswamy":
        a, b = parameters["a"], parameters["b"]
        return (
            math.log(a)
            + math.log(b)
            + (a - 1.0) * np.log(x)
            + (b - 1.0) * np.log1p(-np.power(x, a))
        )
    if model == "logit_normal":
        transformed = special.logit(x)
        return (
            stats.norm.logpdf(
                transformed, loc=parameters["mu"], scale=parameters["sigma"]
            )
            - np.log(x)
            - np.log1p(-x)
        )
    if model == "skew_logit_normal":
        transformed = special.logit(x)
        return (
            stats.skewnorm.logpdf(
                transformed,
                parameters["shape"],
                loc=parameters["loc"],
                scale=parameters["scale"],
            )
            - np.log(x)
            - np.log1p(-x)
        )
    if model == "johnson_sb":
        return stats.johnsonsb.logpdf(x, parameters["a"], parameters["b"])
    if model == "truncated_normal":
        location, scale = parameters["loc"], parameters["scale"]
        return stats.truncnorm.logpdf(
            x,
            -location / scale,
            (1.0 - location) / scale,
            loc=location,
            scale=scale,
        )
    raise ValueError(f"Unknown model: {model}")


def interior_cdf(
    model: str, values: np.ndarray, parameters: dict[str, float]
) -> np.ndarray:
    """Evaluate a fitted interior cumulative distribution.

    Args:
        model (str): Candidate model name.
        values (np.ndarray): Values in the continuous support.
        parameters (dict[str, float]): Named fitted parameters.

    Returns:
        np.ndarray: Cumulative probability for each value.
    """

    x = np.asarray(values, dtype=float)
    if model == "beta":
        return stats.beta.cdf(x, parameters["alpha"], parameters["beta"])
    if model == "generalized_beta":
        return stats.beta.cdf(
            np.power(x, parameters["a"]), parameters["p"], parameters["q"]
        )
    if model == "kumaraswamy":
        return 1.0 - np.power(1.0 - np.power(x, parameters["a"]), parameters["b"])
    if model == "logit_normal":
        return stats.norm.cdf(
            special.logit(x), loc=parameters["mu"], scale=parameters["sigma"]
        )
    if model == "skew_logit_normal":
        return stats.skewnorm.cdf(
            special.logit(x),
            parameters["shape"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )
    if model == "johnson_sb":
        return stats.johnsonsb.cdf(x, parameters["a"], parameters["b"])
    if model == "truncated_normal":
        location, scale = parameters["loc"], parameters["scale"]
        return stats.truncnorm.cdf(
            x,
            -location / scale,
            (1.0 - location) / scale,
            loc=location,
            scale=scale,
        )
    raise ValueError(f"Unknown model: {model}")


def boundary_probabilities(values: np.ndarray) -> tuple[float, float]:
    """Estimate smoothed zero and positive hurdle probabilities.

    Args:
        values (np.ndarray): Training density observations.

    Returns:
        tuple[float, float]: Probabilities for zero and positive observations.
    """

    counts = np.array(
        [
            np.count_nonzero(values == 0.0),
            np.count_nonzero(values > 0.0),
        ],
        dtype=float,
    )
    probabilities = (counts + BOUNDARY_ALPHA) / (
        counts.sum() + BOUNDARY_ALPHA * counts.size
    )
    return tuple(float(value) for value in probabilities)


def mixture_log_likelihood(
    values: np.ndarray,
    model: str,
    parameters: dict[str, float],
    probabilities: tuple[float, float],
) -> float:
    """Evaluate the zero-hurdle continuous mixture likelihood.

    Args:
        values (np.ndarray): Bounded observations.
        model (str): Candidate interior model name.
        parameters (dict[str, float]): Fitted interior parameters.
        probabilities (tuple[float, float]): Zero and positive masses.

    Returns:
        float: Total log likelihood.
    """

    p_zero, p_positive = probabilities
    zero_count = np.count_nonzero(values == 0.0)
    positive = values[values > 0.0]
    continuous_log_pdf = interior_logpdf(model, positive, parameters)
    if not np.isfinite(continuous_log_pdf).all():
        return float("-inf")
    return float(
        zero_count * math.log(p_zero)
        + positive.size * math.log(p_positive)
        + continuous_log_pdf.sum()
    )


def ks_distance(
    values: np.ndarray, model: str, parameters: dict[str, float]
) -> float:
    """Calculate the one-sample Kolmogorov-Smirnov distance.

    Args:
        values (np.ndarray): Test observations strictly inside ``(0, 1)``.
        model (str): Candidate interior model name.
        parameters (dict[str, float]): Fitted interior parameters.

    Returns:
        float: Maximum empirical-versus-fitted CDF distance.
    """

    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 0:
        return float("nan")
    fitted = np.clip(interior_cdf(model, ordered, parameters), 0.0, 1.0)
    lower_empirical = np.arange(ordered.size, dtype=float) / ordered.size
    upper_empirical = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    return float(
        max(
            np.max(fitted - lower_empirical),
            np.max(upper_empirical - fitted),
        )
    )


def fit_candidate(
    train: np.ndarray,
    test: np.ndarray,
    model: str,
    scope: str,
    group: str,
    density: str,
    full_values: np.ndarray,
) -> dict[str, Any]:
    """Fit and score one hurdle-distribution candidate.

    Args:
        train (np.ndarray): Training observations.
        test (np.ndarray): Held-out test observations.
        model (str): Candidate interior model name.
        scope (str): Global or batch scope.
        group (str): Global label or batch name.
        density (str): Density column name.
        full_values (np.ndarray): Unsampled values used for empirical proportions.

    Returns:
        dict[str, Any]: Candidate parameters, diagnostics, and fit metrics.
    """

    train_positive = train[train > 0.0]
    test_positive = test[test > 0.0]
    base: dict[str, Any] = {
        "scope": scope,
        "group": group,
        "density": density,
        "model": model,
        "n_full": int(full_values.size),
        "n_train": int(train.size),
        "n_test": int(test.size),
        "n_train_positive": int(train_positive.size),
        "n_test_positive": int(test_positive.size),
        "p_zero": float(np.mean(full_values == 0.0)),
        "p_exists": float(np.mean(full_values > 0.0)),
        "p_one": float(np.mean(full_values == 1.0)),
    }
    try:
        parameters = fit_interior_distribution(model, train_positive)
        probabilities = boundary_probabilities(train)
        train_log_likelihood = mixture_log_likelihood(
            train, model, parameters, probabilities
        )
        test_log_likelihood = mixture_log_likelihood(
            test, model, parameters, probabilities
        )
        parameter_count = len(parameters) + 1
        base.update(
            {
                "converged": True,
                "error": "",
                "parameters": json.dumps(parameters, sort_keys=True),
                "fit_p_zero": probabilities[0],
                "fit_p_positive": probabilities[1],
                "train_log_likelihood": train_log_likelihood,
                "test_log_likelihood": test_log_likelihood,
                "test_nll_per_observation": -test_log_likelihood / test.size,
                "aic": 2 * parameter_count - 2 * train_log_likelihood,
                "bic": (
                    parameter_count * math.log(train.size)
                    - 2 * train_log_likelihood
                ),
                "ks_positive": ks_distance(test_positive, model, parameters),
                "parameter_count": parameter_count,
            }
        )
    except Exception as error:  # Candidate failures should not stop other fits.
        base.update(
            {
                "converged": False,
                "error": f"{type(error).__name__}: {error}",
                "parameters": "{}",
                "fit_p_zero": float("nan"),
                "fit_p_positive": float("nan"),
                "train_log_likelihood": float("nan"),
                "test_log_likelihood": float("nan"),
                "test_nll_per_observation": float("nan"),
                "aic": float("nan"),
                "bic": float("nan"),
                "ks_positive": float("nan"),
                "parameter_count": 0,
            }
        )
    return base


def analyze_group(
    values: np.ndarray,
    scope: str,
    group: str,
    density: str,
    train_fraction: float,
    max_fit_observations: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Describe a group and fit all candidate hurdle distributions.

    Args:
        values (np.ndarray): Full bounded density observations.
        scope (str): Global or batch scope.
        group (str): Global label or batch name.
        density (str): Density column name.
        train_fraction (float): Fraction used to fit parameters.
        max_fit_observations (int): Fitting sample cap, or zero for no cap.
        seed (int): Base random seed.

    Returns:
        tuple[dict[str, Any], list[dict[str, Any]]]: Description and candidate rows.
    """

    description = describe_values(values, scope, group, density)
    adjusted_values = adjust_positive_boundaries(values)
    train, test = split_fitting_sample(
        adjusted_values,
        train_fraction,
        max_fit_observations,
        stable_rng(seed, scope, group, density),
    )
    fits = [
        fit_candidate(train, test, model, scope, group, density, values)
        for model in MODEL_NAMES
    ]
    return description, fits


def select_winners(candidate_fits: pd.DataFrame) -> pd.DataFrame:
    """Select the lowest held-out negative-log-likelihood candidate per group.

    Args:
        candidate_fits (pd.DataFrame): All candidate fit rows.

    Returns:
        pd.DataFrame: Winning fit row for each scope, group, and density.
    """

    successful = candidate_fits[
        candidate_fits["converged"]
        & np.isfinite(candidate_fits["test_nll_per_observation"])
    ].copy()
    if successful.empty:
        raise RuntimeError("No distribution candidate converged.")
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
    """Run global and per-batch descriptions and candidate fits.

    Args:
        observations (pd.DataFrame): Validated batch and density data.
        batch_column (str): Name of the batch column.
        density_columns (Sequence[str]): Density columns to analyze.
        train_fraction (float): Fraction used to fit parameters.
        max_fit_observations (int): Fitting sample cap, or zero for no cap.
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
        print(f"Analyzing {scope}: {group}", flush=True)
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


def plot_global_fits(
    observations: pd.DataFrame,
    winners: pd.DataFrame,
    density_columns: Sequence[str],
    output_path: Path,
    seed: int,
) -> None:
    """Plot global positive-volume histograms and selected fitted densities.

    Args:
        observations (pd.DataFrame): Full density observations.
        winners (pd.DataFrame): Selected global and batch fits.
        density_columns (Sequence[str]): Density columns to plot.
        output_path (Path): PNG output path.
        seed (int): Base seed for display subsampling.

    Returns:
        None: Writes a figure to ``output_path``.
    """

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    global_winners = winners[winners["scope"] == "global"].set_index("density")
    for axis, density in zip(axes.flat, density_columns, strict=True):
        all_values = observations[density].to_numpy(dtype=float)
        adjusted_values = adjust_positive_boundaries(all_values)
        positive = adjusted_values[adjusted_values > 0.0]
        if positive.size > 250_000:
            rng = stable_rng(seed, "plot", density)
            positive = positive[rng.choice(positive.size, 250_000, replace=False)]
        winner = global_winners.loc[density]
        model = str(winner["model"])
        parameters = json.loads(str(winner["parameters"]))
        axis.hist(
            positive,
            bins=100,
            range=(0.0, 1.0),
            density=True,
            alpha=0.45,
            color="tab:blue",
            label="Observed interior density",
        )
        grid = np.linspace(1e-5, 1.0 - 1e-5, 1_000)
        pdf = np.exp(interior_logpdf(model, grid, parameters))
        finite_pdf = pdf[np.isfinite(pdf)]
        if finite_pdf.size:
            display_cap = np.quantile(
                np.concatenate([finite_pdf, np.array([0.0])]), 0.995
            )
            axis.plot(grid, np.minimum(pdf, display_cap), color="tab:red", lw=2)
        p_zero = float(winner["p_zero"])
        axis.set(
            title=f"{density}: {model}\nP(0)={p_zero:.3f}",
            xlabel="Density conditional on X > 0",
            ylabel="Probability density",
            xlim=(0.0, 1.0),
        )
        axis.grid(alpha=0.2)
    figure.suptitle("Global bounded hurdle-distribution fits", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_batch_heatmaps(
    descriptions: pd.DataFrame,
    winners: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot batch zero fractions, selected families, and held-out fit quality.

    Args:
        descriptions (pd.DataFrame): Global and batch descriptive rows.
        winners (pd.DataFrame): Selected global and batch fit rows.
        output_path (Path): PNG output path.

    Returns:
        None: Writes a figure to ``output_path``.
    """

    batch_descriptions = descriptions[descriptions["scope"] == "batch"]
    batch_winners = winners[winners["scope"] == "batch"]
    zero_matrix = batch_descriptions.pivot(
        index="group", columns="density", values="p_zero"
    )
    model_matrix = batch_winners.pivot(
        index="group", columns="density", values="model"
    ).reindex(zero_matrix.index)
    nll_matrix = batch_winners.pivot(
        index="group", columns="density", values="test_nll_per_observation"
    ).reindex(zero_matrix.index)
    model_codes = model_matrix.replace(
        {model: index for index, model in enumerate(MODEL_NAMES)}
    ).astype(float)
    model_cmap = ListedColormap(
        list(plt.get_cmap("tab10").colors[: len(MODEL_NAMES)])
    )
    model_norm = BoundaryNorm(
        np.arange(-0.5, len(MODEL_NAMES) + 0.5, 1.0), model_cmap.N
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, max(11, 0.27 * len(zero_matrix))),
        constrained_layout=True,
    )
    zero_image = axes[0].imshow(zero_matrix, aspect="auto", vmin=0.0, vmax=1.0)
    model_image = axes[1].imshow(
        model_codes,
        aspect="auto",
        cmap=model_cmap,
        norm=model_norm,
    )
    nll_image = axes[2].imshow(nll_matrix, aspect="auto", cmap="viridis")
    for axis, title in zip(
        axes,
        ["Observed P(X = 0)", "Selected conditional family", "Held-out NLL / observation"],
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xticks(np.arange(len(zero_matrix.columns)))
        axis.set_xticklabels(
            [column.replace("Density_", "") for column in zero_matrix.columns],
            rotation=45,
            ha="right",
        )
        axis.set_yticks(np.arange(len(zero_matrix.index)))
        axis.set_yticklabels(zero_matrix.index, fontsize=7)
    figure.colorbar(zero_image, ax=axes[0], shrink=0.7)
    model_colorbar = figure.colorbar(
        model_image,
        ax=axes[1],
        ticks=np.arange(len(MODEL_NAMES)),
        boundaries=np.arange(-0.5, len(MODEL_NAMES) + 0.5, 1.0),
        shrink=0.7,
    )
    model_colorbar.ax.set_yticklabels(MODEL_NAMES)
    figure.colorbar(nll_image, ax=axes[2], shrink=0.7)
    figure.suptitle("Batch-by-batch hurdle-distribution results", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_summary(
    descriptions: pd.DataFrame,
    winners: pd.DataFrame,
    candidate_fits: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a concise Markdown interpretation of the fit results.

    Args:
        descriptions (pd.DataFrame): Global and batch descriptive rows.
        winners (pd.DataFrame): Selected global and batch fits.
        candidate_fits (pd.DataFrame): Every candidate fit row.
        output_path (Path): Markdown output path.

    Returns:
        None: Writes the summary to ``output_path``.
    """

    global_descriptions = descriptions[descriptions["scope"] == "global"].set_index(
        "density"
    )
    global_winners = winners[winners["scope"] == "global"].set_index("density")
    batch_winners = winners[winners["scope"] == "batch"]
    lines = [
        "# Density distribution fitting summary",
        "",
        "Every candidate separates existence as an explicit point mass at zero. "
        "Rare exact ones receive the same boundary correction as the other positive "
        "values and are fitted by the continuous positive-density family.",
        "",
        "## Global selections",
        "",
    ]
    for density in DENSITY_COLUMNS:
        description = global_descriptions.loc[density]
        winner = global_winners.loc[density]
        lines.append(
            f"- **{density}**: `{winner['model']}`; "
            f"P(X=0)={description['p_zero']:.4f}, "
            f"P(X>0)={description['p_exists']:.4f}, "
            f"held-out NLL/observation={winner['test_nll_per_observation']:.5f}, "
            f"positive-component KS={winner['ks_positive']:.5f}."
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
            "- AIC, BIC, and KS distance are diagnostics in `candidate_fits.csv`.",
            f"- Candidate fits that failed to converge: {len(failed)}.",
            "- With very large spatial datasets, formal goodness-of-fit tests tend "
            "to reject harmless deviations; inspect effect-size diagnostics and plots.",
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
    """Save fit tables, figures, and a Markdown summary.

    Args:
        observations (pd.DataFrame): Full observations used for plotting.
        descriptions (pd.DataFrame): Descriptive statistic rows.
        candidate_fits (pd.DataFrame): Every candidate fit row.
        winners (pd.DataFrame): Selected winner rows.
        density_columns (Sequence[str]): Density columns to plot.
        output_dir (Path): Destination directory.
        seed (int): Plot subsampling seed.

    Returns:
        None: Writes all analysis artifacts.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    descriptions.to_csv(output_dir / "descriptive_statistics.csv", index=False)
    candidate_fits.to_csv(output_dir / "candidate_fits.csv", index=False)
    winners.to_csv(output_dir / "selected_models.csv", index=False)
    plot_global_fits(
        observations,
        winners,
        density_columns,
        output_dir / "global_distribution_fits.png",
        seed,
    )
    plot_batch_heatmaps(
        descriptions, winners, output_dir / "batch_distribution_heatmaps.png"
    )
    write_summary(
        descriptions,
        winners,
        candidate_fits,
        output_dir / "summary.md",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line density distribution analysis.

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

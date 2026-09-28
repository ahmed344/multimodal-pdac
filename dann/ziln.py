"""Summaries and density sampling for the structural-zero logit-normal hurdle.

The three parameters are the zero probability pi, logit mean mu, and logit
standard deviation sigma. Logit summaries are conditional on the positive
branch; exceedance probabilities and sampled density means include the hurdle.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats
from scipy.special import expit


__all__ = [
    "sampled_density_mean",
    "clipped_logit",
    "exceedance_probability",
    "hurdle_logit_quantile",
    "positive_logit_cdf",
    "positive_logit_mean",
    "positive_logit_pit",
    "positive_logit_quantile",
    "positive_logit_sd",
]



def _as_float_array(values: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    """Coerce an input to a finite float64 array.

    Args:
        values (np.ndarray | Sequence[float]): Raw parameter values.
        name (str): Parameter name used in error messages.

    Returns:
        np.ndarray: Finite float64 view of ``values``.
    """

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _validate_sigma(sigma: np.ndarray) -> np.ndarray:
    """Validate that a scale array is strictly positive.

    Args:
        sigma (np.ndarray): Positive-branch scale values.

    Returns:
        np.ndarray: Validated float64 scale array.
    """

    array = _as_float_array(sigma, "sigma")
    if np.any(array <= 0.0):
        raise ValueError("sigma must be strictly positive.")
    return array


def _validate_probability(values: np.ndarray, name: str) -> np.ndarray:
    """Validate that an array lies inside the closed unit interval.

    Args:
        values (np.ndarray): Probability values.
        name (str): Parameter name used in error messages.

    Returns:
        np.ndarray: Validated float64 probability array.
    """

    array = _as_float_array(values, name)
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must lie inside [0, 1].")
    return array


def clipped_logit(
    values: np.ndarray | Sequence[float],
    epsilon: float,
) -> np.ndarray:
    """Apply the boundary-clamped logit transform used across the pipeline.

    Args:
        values (np.ndarray | Sequence[float]): Bounded values in ``[0, 1]``.
        epsilon (float): Clamp applied before the transform.

    Returns:
        np.ndarray: Logit-transformed float64 values.
    """

    if not 0.0 < float(epsilon) < 0.5:
        raise ValueError("epsilon must lie strictly between zero and 0.5.")
    array = _as_float_array(values, "values")
    clipped = np.clip(array, float(epsilon), 1.0 - float(epsilon))
    return np.log(clipped / (1.0 - clipped))


def positive_logit_mean(
    mu: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return the normal positive-branch logit mean, broadcasting with sigma."""
    location, _ = np.broadcast_arrays(_as_float_array(mu, "mu"), _validate_sigma(sigma))
    return location


def positive_logit_sd(sigma: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return the normal positive-branch logit standard deviation."""
    return _validate_sigma(sigma)


def positive_logit_quantile(
    mu: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    quantile: float,
) -> np.ndarray:
    """Compute a positive-branch quantile on the logit scale.

    Args:
        mu (np.ndarray | Sequence[float]): Positive-branch location parameters.
        sigma (np.ndarray | Sequence[float]): Positive-branch scale parameters.
        quantile (float): Requested quantile in ``(0, 1)``.

    Returns:
        np.ndarray: Logit-scale quantile of ``Z`` given ``y > 0``.
    """

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must lie strictly between zero and one.")
    location = _as_float_array(mu, "mu")
    scale = _validate_sigma(sigma)
    return stats.norm.ppf(float(quantile), loc=location, scale=scale)


def positive_logit_cdf(
    mu: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    logit_values: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Evaluate the positive-branch CDF at logit-scale points.

    Args:
        mu (np.ndarray | Sequence[float]): Positive-branch location parameters.
        sigma (np.ndarray | Sequence[float]): Positive-branch scale parameters.
        logit_values (np.ndarray | Sequence[float]): Logit-scale evaluation points.

    Returns:
        np.ndarray: ``P(Z <= logit_values | y > 0)``.
    """

    location = _as_float_array(mu, "mu")
    scale = _validate_sigma(sigma)
    points = _as_float_array(logit_values, "logit_values")
    return stats.norm.cdf(points, loc=location, scale=scale)


def positive_logit_pit(
    mu: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    targets: np.ndarray | Sequence[float],
    epsilon: float,
) -> np.ndarray:
    """Compute the probability integral transform of positive observations.

    If the positive branch is correctly specified, the returned values are
    uniform on ``[0, 1]``. Deviations diagnose the failure mode directly: a
    U-shaped histogram means the predicted ``sigma`` is too small
    (overconfident), an inverted-U means it is too large, and a monotone slope
    means ``mu`` is biased.

    Callers must pass only observations with ``targets > 0``; zeros belong to
    the hurdle branch and have no logit-scale representation.

    Args:
        mu (np.ndarray | Sequence[float]): Positive-branch location parameters.
        sigma (np.ndarray | Sequence[float]): Positive-branch scale parameters.
        targets (np.ndarray | Sequence[float]): Observed positive densities.
        epsilon (float): Clamp applied before the observed-value logit.

    Returns:
        np.ndarray: PIT values in ``[0, 1]``.
    """

    observed = _as_float_array(targets, "targets")
    if np.any(observed <= 0.0):
        raise ValueError("positive_logit_pit requires strictly positive targets.")
    return positive_logit_cdf(mu, sigma, clipped_logit(observed, epsilon))


def hurdle_logit_quantile(
    pi: np.ndarray | Sequence[float],
    mu: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    quantile: float,
) -> np.ndarray:
    """Compute an unconditional quantile, accounting for the zero point mass.

    Below the zero mass the unconditional quantile is exactly zero, which has no
    logit-scale representation, so those entries are returned as ``NaN``. Note
    that the unconditional quantile is *not* ``(1 - pi)`` times the conditional
    one: the requested probability must be re-mapped through the hurdle first.

    Args:
        pi (np.ndarray | Sequence[float]): Structural-zero probabilities.
        mu (np.ndarray | Sequence[float]): Positive-branch location parameters.
        sigma (np.ndarray | Sequence[float]): Positive-branch scale parameters.
        quantile (float): Requested quantile in ``(0, 1)``.

    Returns:
        np.ndarray: Logit-scale quantile, ``NaN`` where the quantile is a zero.
    """

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must lie strictly between zero and one.")
    zero_probability = _validate_probability(pi, "pi")
    location = _as_float_array(mu, "mu")
    scale = _validate_sigma(sigma)
    location, scale, zero_probability = np.broadcast_arrays(
        location, scale, zero_probability
    )
    remaining = 1.0 - zero_probability
    result = np.full(zero_probability.shape, np.nan, dtype=np.float64)
    positive_branch = (remaining > 0.0) & (float(quantile) > zero_probability)
    if np.any(positive_branch):
        inner = (float(quantile) - zero_probability[positive_branch]) / remaining[
            positive_branch
        ]
        result[positive_branch] = stats.norm.ppf(
            inner,
            loc=location[positive_branch],
            scale=scale[positive_branch],
        )
    return result


def exceedance_probability(
    pi: np.ndarray | Sequence[float],
    mu: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    threshold: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute ``P(y > threshold)`` across both branches.

    This is the one summary that combines the hurdle and positive branches
    without needing an intractable integral, because the monotone ``expit``
    transform maps the density-scale threshold directly onto the logit scale.
    It is often a more interpretable spatial readout than any point estimate.

    Args:
        pi (np.ndarray | Sequence[float]): Structural-zero probabilities.
        mu (np.ndarray | Sequence[float]): Positive-branch location parameters.
        sigma (np.ndarray | Sequence[float]): Positive-branch scale parameters.
        threshold (float): Density threshold in ``(0, 1)``.
        epsilon (float): Clamp applied before the threshold logit.

    Returns:
        np.ndarray: Unconditional exceedance probabilities in ``[0, 1]``.
    """

    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must lie strictly between zero and one.")
    zero_probability = _validate_probability(pi, "pi")
    logit_threshold = float(clipped_logit(np.asarray([threshold]), epsilon)[0])
    survival = 1.0 - positive_logit_cdf(mu, sigma, logit_threshold)
    return (1.0 - zero_probability) * survival


def sampled_density_mean(
    pi: np.ndarray | Sequence[float],
    mu: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    *,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate the unconditional density mean using independent hurdle draws.

    Each draw is zero with probability ``pi`` and otherwise is the sigmoid of
    a normal draw with location ``mu`` and scale ``sigma``. Inputs broadcast;
    the returned float64 array has that broadcast shape. Sampling uses only
    the supplied generator and bounded blocks (at most 1024 cells x 128 draws),
    accumulating sums in float64. Repeatability requires the same seed,
    parameters, sample count, and processing order/batch boundaries.
    """
    if (
        isinstance(num_samples, bool)
        or not isinstance(num_samples, (int, np.integer))
        or num_samples <= 0
    ):
        raise ValueError("num_samples must be a positive integer.")
    zero, location, scale = np.broadcast_arrays(
        _validate_probability(pi, "pi"), _as_float_array(mu, "mu"),
        _validate_sigma(sigma),
    )
    output_shape = location.shape
    result = np.zeros(location.size, dtype=np.float64)
    zero, location, scale = zero.ravel(), location.ravel(), scale.ravel()
    for start in range(0, result.size, 1024):
        stop = min(start + 1024, result.size)
        for draw_start in range(0, num_samples, 128):
            shape = (stop - start, min(128, num_samples - draw_start))
            draws = rng.standard_normal(shape)
            with np.errstate(over="ignore"):
                draws *= scale[start:stop, None]
                draws += location[start:stop, None]
            expit(draws, out=draws)
            draws *= rng.random(shape) >= zero[start:stop, None]
            result[start:stop] += draws.sum(axis=1, dtype=np.float64)
    return (result / num_samples).reshape(output_shape)

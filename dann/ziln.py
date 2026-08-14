"""Derived summaries of the zero-inflated skew-logit-normal biology head.

The biology head emits four parameters per pixel and target, which together
define a distribution rather than a point prediction::

    y = 0                                          with probability pi
    y = expit(Z),  Z ~ SkewNormal(mu, sigma, alpha) with probability 1 - pi

``mu`` is the *location* of the positive branch on the logit scale. It is not a
mean: the skew-normal mean is shifted by ``sigma * delta * sqrt(2 / pi)``, it
lives on the logit scale rather than the density scale, and it carries no
information about ``pi``. Every function here turns the raw parameters into a
quantity that can actually be read as a prediction.

All summaries are on the **logit scale and conditional on the positive branch**.
That is not a simplification; ``logit(0)`` is undefined, so any logit-scale
summary is necessarily conditional on ``y > 0``, and ``pi`` must be reported
separately. The one summary that mixes both branches, ``exceedance_probability``,
is on the density scale by construction and is therefore unconditional.

Deliberately absent: the density-scale expectation
``E[y] = (1 - pi) * E[expit(Z)]``. ``E[expit(Z)]`` has no closed form, so it
requires Monte Carlo or Gauss-Hermite quadrature over every pixel — the only
genuinely expensive quantity in this module's problem space.

The parametrization matches ``scipy.stats.skewnorm(a=alpha, loc=mu,
scale=sigma)`` exactly, which is the same density as the positive branch of
``dann.losses.ZILNLoss``; ``dann/tests/test_ziln.py`` verifies that equivalence
rather than assuming it.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


__all__ = [
    "clipped_logit",
    "exceedance_probability",
    "hurdle_logit_quantile",
    "positive_logit_cdf",
    "positive_logit_mean",
    "positive_logit_pit",
    "positive_logit_quantile",
    "positive_logit_sd",
    "skew_delta",
]


_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


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


def skew_delta(alpha: np.ndarray | Sequence[float]) -> np.ndarray:
    """Compute the skew-normal ``delta`` reparametrization.

    ``delta`` is bounded in ``(-1, 1)`` and drives both the mean shift and the
    variance deflation of the skew-normal relative to a plain normal.

    Args:
        alpha (np.ndarray | Sequence[float]): Skew-normal shape parameters.

    Returns:
        np.ndarray: ``alpha / sqrt(1 + alpha ** 2)``.
    """

    array = _as_float_array(alpha, "alpha")
    return array / np.sqrt(1.0 + np.square(array))


def positive_logit_mean(
    mu: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    alpha: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Compute ``E[Z | y > 0]``, the skew-corrected logit-scale mean.

    This is the quantity ``mu`` is frequently mistaken for. It reduces to ``mu``
    exactly when ``alpha == 0``.

    Args:
        mu (np.ndarray | Sequence[float]): Positive-branch location parameters.
        sigma (np.ndarray | Sequence[float]): Positive-branch scale parameters.
        alpha (np.ndarray | Sequence[float]): Positive-branch shape parameters.

    Returns:
        np.ndarray: Logit-scale mean of the positive branch.
    """

    location = _as_float_array(mu, "mu")
    scale = _validate_sigma(sigma)
    return location + scale * skew_delta(alpha) * _SQRT_2_OVER_PI


def positive_logit_sd(
    sigma: np.ndarray | Sequence[float],
    alpha: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Compute ``SD[Z | y > 0]``, the skew-corrected logit-scale spread.

    Skewness always deflates the standard deviation below ``sigma``, so this is
    strictly smaller than ``sigma`` whenever ``alpha != 0``.

    Args:
        sigma (np.ndarray | Sequence[float]): Positive-branch scale parameters.
        alpha (np.ndarray | Sequence[float]): Positive-branch shape parameters.

    Returns:
        np.ndarray: Logit-scale standard deviation of the positive branch.
    """

    scale = _validate_sigma(sigma)
    delta = skew_delta(alpha)
    variance_factor = 1.0 - 2.0 * np.square(delta) / math.pi
    return scale * np.sqrt(np.clip(variance_factor, 0.0, None))


def positive_logit_quantile(
    mu: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    alpha: np.ndarray | Sequence[float],
    quantile: float,
) -> np.ndarray:
    """Compute a positive-branch quantile on the logit scale.

    Args:
        mu (np.ndarray | Sequence[float]): Positive-branch location parameters.
        sigma (np.ndarray | Sequence[float]): Positive-branch scale parameters.
        alpha (np.ndarray | Sequence[float]): Positive-branch shape parameters.
        quantile (float): Requested quantile in ``(0, 1)``.

    Returns:
        np.ndarray: Logit-scale quantile of ``Z`` given ``y > 0``.
    """

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must lie strictly between zero and one.")
    location = _as_float_array(mu, "mu")
    scale = _validate_sigma(sigma)
    shape = _as_float_array(alpha, "alpha")
    return stats.skewnorm.ppf(float(quantile), a=shape, loc=location, scale=scale)


def positive_logit_cdf(
    mu: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    alpha: np.ndarray | Sequence[float],
    logit_values: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Evaluate the positive-branch CDF at logit-scale points.

    Args:
        mu (np.ndarray | Sequence[float]): Positive-branch location parameters.
        sigma (np.ndarray | Sequence[float]): Positive-branch scale parameters.
        alpha (np.ndarray | Sequence[float]): Positive-branch shape parameters.
        logit_values (np.ndarray | Sequence[float]): Logit-scale evaluation points.

    Returns:
        np.ndarray: ``P(Z <= logit_values | y > 0)``.
    """

    location = _as_float_array(mu, "mu")
    scale = _validate_sigma(sigma)
    shape = _as_float_array(alpha, "alpha")
    points = _as_float_array(logit_values, "logit_values")
    return stats.skewnorm.cdf(points, a=shape, loc=location, scale=scale)


def positive_logit_pit(
    mu: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    alpha: np.ndarray | Sequence[float],
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
        alpha (np.ndarray | Sequence[float]): Positive-branch shape parameters.
        targets (np.ndarray | Sequence[float]): Observed positive densities.
        epsilon (float): Clamp applied before the observed-value logit.

    Returns:
        np.ndarray: PIT values in ``[0, 1]``.
    """

    observed = _as_float_array(targets, "targets")
    if np.any(observed <= 0.0):
        raise ValueError("positive_logit_pit requires strictly positive targets.")
    return positive_logit_cdf(mu, sigma, alpha, clipped_logit(observed, epsilon))


def hurdle_logit_quantile(
    pi: np.ndarray | Sequence[float],
    mu: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    alpha: np.ndarray | Sequence[float],
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
        alpha (np.ndarray | Sequence[float]): Positive-branch shape parameters.
        quantile (float): Requested quantile in ``(0, 1)``.

    Returns:
        np.ndarray: Logit-scale quantile, ``NaN`` where the quantile is a zero.
    """

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must lie strictly between zero and one.")
    zero_probability = _validate_probability(pi, "pi")
    location = _as_float_array(mu, "mu")
    scale = _validate_sigma(sigma)
    shape = _as_float_array(alpha, "alpha")
    location, scale, shape, zero_probability = np.broadcast_arrays(
        location, scale, shape, zero_probability
    )
    remaining = 1.0 - zero_probability
    result = np.full(zero_probability.shape, np.nan, dtype=np.float64)
    positive_branch = (remaining > 0.0) & (float(quantile) > zero_probability)
    if np.any(positive_branch):
        inner = (float(quantile) - zero_probability[positive_branch]) / remaining[
            positive_branch
        ]
        result[positive_branch] = stats.skewnorm.ppf(
            inner,
            a=shape[positive_branch],
            loc=location[positive_branch],
            scale=scale[positive_branch],
        )
    return result


def exceedance_probability(
    pi: np.ndarray | Sequence[float],
    mu: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    alpha: np.ndarray | Sequence[float],
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
        alpha (np.ndarray | Sequence[float]): Positive-branch shape parameters.
        threshold (float): Density threshold in ``(0, 1)``.
        epsilon (float): Clamp applied before the threshold logit.

    Returns:
        np.ndarray: Unconditional exceedance probabilities in ``[0, 1]``.
    """

    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must lie strictly between zero and one.")
    zero_probability = _validate_probability(pi, "pi")
    logit_threshold = float(clipped_logit(np.asarray([threshold]), epsilon)[0])
    survival = 1.0 - positive_logit_cdf(mu, sigma, alpha, logit_threshold)
    return (1.0 - zero_probability) * survival

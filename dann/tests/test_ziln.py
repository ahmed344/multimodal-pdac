"""Focused tests for derived zero-inflated skew-logit-normal summaries."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import stats

from dann import ziln
from dann.losses import ZILNLoss


def _parameter_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic parameter grid spanning skew signs and scales.

    Args:
        None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Broadcast mu, sigma, and alpha.
    """

    mu = np.asarray([-4.0, -1.0, 0.0, 0.5, 3.0])
    sigma = np.asarray([0.25, 0.75, 1.0, 2.0, 3.5])
    alpha = np.asarray([-6.0, -1.5, 0.0, 2.0, 8.0])
    return mu, sigma, alpha


def test_zero_alpha_collapses_to_gaussian() -> None:
    """Zero skew reproduces the plain normal mean, spread, and quantiles."""

    mu, sigma, _ = _parameter_grid()
    alpha = np.zeros_like(mu)
    np.testing.assert_allclose(ziln.positive_logit_mean(mu, sigma, alpha), mu)
    np.testing.assert_allclose(ziln.positive_logit_sd(sigma, alpha), sigma)
    for quantile in (0.05, 0.5, 0.95):
        np.testing.assert_allclose(
            ziln.positive_logit_quantile(mu, sigma, alpha, quantile),
            stats.norm.ppf(quantile, loc=mu, scale=sigma),
            atol=1e-9,
        )


def test_closed_forms_match_scipy_moments() -> None:
    """Closed-form mean and standard deviation match scipy's skew-normal."""

    mu, sigma, alpha = _parameter_grid()
    np.testing.assert_allclose(
        ziln.positive_logit_mean(mu, sigma, alpha),
        stats.skewnorm.mean(a=alpha, loc=mu, scale=sigma),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        ziln.positive_logit_sd(sigma, alpha),
        stats.skewnorm.std(a=alpha, loc=mu, scale=sigma),
        rtol=1e-10,
    )


def test_skew_shifts_mean_away_from_mu() -> None:
    """Nonzero skew moves the mean off ``mu`` and deflates the spread."""

    mu = np.asarray([0.0, 0.0])
    sigma = np.asarray([1.0, 1.0])
    alpha = np.asarray([5.0, -5.0])
    means = ziln.positive_logit_mean(mu, sigma, alpha)
    assert means[0] > 0.0
    assert means[1] < 0.0
    assert np.all(ziln.positive_logit_sd(sigma, alpha) < sigma)


def test_parametrization_matches_ziln_loss_positive_branch() -> None:
    """The scipy density equals the loss module's positive-branch density."""

    mu, sigma, alpha = _parameter_grid()
    targets = np.asarray([0.01, 0.2, 0.5, 0.75, 0.99])
    epsilon = 1e-6
    loss = ZILNLoss(
        target_weights=[1.0] * targets.size,
        logit_epsilon=epsilon,
        reduction="sum",
        include_normal_constant=True,
    )
    output = loss(
        pi_logits=torch.zeros(1, targets.size, dtype=torch.float64),
        mu=torch.as_tensor(mu, dtype=torch.float64).unsqueeze(0),
        sigma=torch.as_tensor(sigma, dtype=torch.float64).unsqueeze(0),
        alpha=torch.as_tensor(alpha, dtype=torch.float64).unsqueeze(0),
        targets=torch.as_tensor(targets, dtype=torch.float64).unsqueeze(0),
    )
    expected = -stats.skewnorm.logpdf(
        ziln.clipped_logit(targets, epsilon), a=alpha, loc=mu, scale=sigma
    ).sum()
    assert output.positive.item() == pytest.approx(float(expected), rel=1e-9)


def test_quantiles_are_ordered_and_monotone() -> None:
    """Quantiles increase with the requested probability level."""

    mu, sigma, alpha = _parameter_grid()
    previous = ziln.positive_logit_quantile(mu, sigma, alpha, 0.01)
    for quantile in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
        current = ziln.positive_logit_quantile(mu, sigma, alpha, quantile)
        assert np.all(current > previous)
        previous = current


def test_pit_is_uniform_for_correctly_specified_data() -> None:
    """Simulating from the model's own generative process yields uniform PIT."""

    rng = np.random.default_rng(20260730)
    size = 40_000
    mu = rng.normal(-2.0, 1.0, size)
    sigma = np.abs(rng.normal(1.0, 0.2, size)) + 0.1
    alpha = rng.normal(0.0, 3.0, size)
    draws = stats.skewnorm.rvs(a=alpha, loc=mu, scale=sigma, random_state=rng)
    densities = 1.0 / (1.0 + np.exp(-draws))
    pit = ziln.positive_logit_pit(mu, sigma, alpha, densities, 1e-9)
    assert float(stats.kstest(pit, "uniform").statistic) < 0.01
    assert float(pit.mean()) == pytest.approx(0.5, abs=0.01)


def test_interval_coverage_matches_nominal_level() -> None:
    """Central 90% intervals contain about 90% of correctly specified draws."""

    rng = np.random.default_rng(11)
    size = 40_000
    mu = rng.normal(0.0, 1.0, size)
    sigma = np.abs(rng.normal(1.0, 0.2, size)) + 0.1
    alpha = rng.normal(0.0, 2.0, size)
    draws = stats.skewnorm.rvs(a=alpha, loc=mu, scale=sigma, random_state=rng)
    lower = ziln.positive_logit_quantile(mu, sigma, alpha, 0.05)
    upper = ziln.positive_logit_quantile(mu, sigma, alpha, 0.95)
    coverage = float(((draws >= lower) & (draws <= upper)).mean())
    assert coverage == pytest.approx(0.90, abs=0.01)


def test_exceedance_probability_matches_monte_carlo() -> None:
    """Closed-form exceedance agrees with a direct simulation of the hurdle."""

    rng = np.random.default_rng(7)
    pi = np.asarray([0.0, 0.3, 0.8])
    mu = np.asarray([0.5, -1.0, 2.0])
    sigma = np.asarray([1.0, 0.5, 1.5])
    alpha = np.asarray([0.0, 3.0, -2.0])
    threshold = 0.4
    analytic = ziln.exceedance_probability(pi, mu, sigma, alpha, threshold)

    draws = 400_000
    simulated = np.empty_like(analytic)
    for index in range(pi.size):
        latent = stats.skewnorm.rvs(
            a=alpha[index],
            loc=mu[index],
            scale=sigma[index],
            size=draws,
            random_state=rng,
        )
        values = 1.0 / (1.0 + np.exp(-latent))
        values[rng.random(draws) < pi[index]] = 0.0
        simulated[index] = float((values > threshold).mean())
    np.testing.assert_allclose(analytic, simulated, atol=0.005)


def test_hurdle_quantile_returns_nan_below_the_zero_mass() -> None:
    """Unconditional quantiles inside the zero mass have no logit value."""

    pi = np.asarray([0.0, 0.4, 0.9])
    mu = np.zeros(3)
    sigma = np.ones(3)
    alpha = np.zeros(3)
    result = ziln.hurdle_logit_quantile(pi, mu, sigma, alpha, 0.5)
    assert np.isnan(result[2])
    assert np.isfinite(result[0]) and np.isfinite(result[1])
    # The zero mass occupies the bottom of the distribution, so the requested
    # probability is re-mapped onto a *lower* quantile of the positive branch:
    # a nonzero pi pulls the unconditional median down, not up.
    assert result[1] < result[0]
    assert result[0] == pytest.approx(0.0, abs=1e-9)
    assert result[1] == pytest.approx(stats.norm.ppf(0.1 / 0.6), rel=1e-9)


def test_clipped_logit_bounds_extreme_targets() -> None:
    """Boundary densities are clamped instead of producing infinities."""

    values = ziln.clipped_logit(np.asarray([0.0, 0.5, 1.0]), 1e-6)
    assert np.isfinite(values).all()
    assert values[0] == pytest.approx(-math.log(1e6 - 1.0), rel=1e-9)
    assert values[1] == pytest.approx(0.0, abs=1e-12)
    assert values[2] == pytest.approx(math.log(1e6 - 1.0), rel=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": np.asarray([0.0])},
        {"sigma": np.asarray([-1.0])},
    ],
)
def test_nonpositive_sigma_is_rejected(kwargs: dict[str, np.ndarray]) -> None:
    """A nonpositive scale is rejected rather than silently producing NaNs."""

    with pytest.raises(ValueError, match="sigma must be strictly positive"):
        ziln.positive_logit_mean(np.zeros(1), kwargs["sigma"], np.zeros(1))


def test_pit_rejects_zero_targets() -> None:
    """Zeros belong to the hurdle branch and must be masked by the caller."""

    with pytest.raises(ValueError, match="strictly positive targets"):
        ziln.positive_logit_pit(
            np.zeros(2), np.ones(2), np.zeros(2), np.asarray([0.0, 0.5]), 1e-6
        )

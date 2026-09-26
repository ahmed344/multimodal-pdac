"""Numerical contracts for density, fraction, and residual prediction exports."""

from __future__ import annotations

import numpy as np
import torch

from ihc_mvn.infer import (
    _export_denominators,
    expected_positive_fraction,
    hurdle_quantile_residuals,
)
from ihc_mvn.targets import TARGET_COLUMNS, haldane_to_fraction


def test_expected_positive_fraction_matches_monte_carlo_mean() -> None:
    """Compare Gauss-Hermite positive means with a large Monte Carlo sample.

    Args:
        None.

    Returns:
        None: Quadrature agrees with simulation and exceeds the median.
    """

    rng = np.random.default_rng(0)
    locations = np.asarray([-6.0, -2.0, 1.5])
    scales = np.asarray([1.3, 0.8, 2.0])
    denominators = np.asarray([36_100.0, 500.0, 20.0])
    quadrature = expected_positive_fraction(locations, scales, denominators, 0.5)
    draws = locations[:, None] + scales[:, None] * rng.standard_normal((3, 400_000))
    simulated = haldane_to_fraction(draws, denominators[:, None], 0.5).mean(axis=1)
    np.testing.assert_allclose(quadrature, simulated, rtol=5.0e-3)
    median = haldane_to_fraction(locations, denominators, 0.5)
    assert quadrature[0] > median[0]


def test_hurdle_quantile_residuals_are_calibrated_under_the_model() -> None:
    """Simulate from a hurdle Gaussian with censoring and check mid-quantiles.

    Under the model the mid-quantile ``U`` has expectation one half, exact
    positives follow the closed-form CDF, and zeros fall below the median.

    Args:
        None.

    Returns:
        None: Mid-quantile invariants are asserted.
    """

    rng = np.random.default_rng(1)
    rows = 200_000
    probability = rng.uniform(0.2, 0.9, size=(rows, 1))
    means = rng.normal(size=(rows, 1))
    scales = rng.uniform(0.5, 1.5, size=(rows, 1))
    positive = rng.uniform(size=(rows, 1)) < probability
    latent = means + scales * rng.standard_normal((rows, 1))
    boundary = means + 1.0 * scales
    censored = positive & (latent >= boundary)
    coordinates = np.where(censored, boundary, np.where(positive, latent, -9.0))
    residual = hurdle_quantile_residuals(
        probability, coordinates, means, scales, positive, censored
    )
    assert np.isfinite(residual).all()
    exact = positive & ~censored
    np.testing.assert_allclose(
        torch.special.ndtr(torch.from_numpy(residual)).numpy().mean(), 0.5, atol=3.0e-3
    )
    uniform = 1.0 - probability + probability * 0.5 * (
        1.0 + torch.special.erf(torch.from_numpy((coordinates - means) / scales / np.sqrt(2.0))).numpy()
    )
    np.testing.assert_allclose(
        residual[exact],
        torch.special.ndtri(torch.from_numpy(uniform[exact])).numpy(),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    zero_mid = np.sort(residual[~positive])
    assert np.all(zero_mid < 0.0)
    assert np.isnan(
        hurdle_quantile_residuals(
            probability[:1], np.full((1, 1), np.nan), means[:1], scales[:1],
            positive[:1], censored[:1],
        )
    ).all()


def test_export_denominators_use_observed_nontumor_count() -> None:
    """Use ``N`` for tumor and observed ``E`` otherwise, NaN when unavailable.

    Args:
        None.

    Returns:
        None: Denominator matrix is asserted.
    """

    batch = {
        "row_ids": torch.arange(3),
        "extratumoral_count": torch.tensor([36_100, 100, -1]),
        "target_available": torch.tensor([True, True, False]),
    }
    denominators = _export_denominators(batch, TARGET_COLUMNS, 36_100)
    np.testing.assert_array_equal(denominators[:, 0], 36_100.0)
    np.testing.assert_array_equal(denominators[:2, 1:], [[36_100.0] * 3, [100.0] * 3])
    assert np.isnan(denominators[2, 1:]).all()
    unlabeled = _export_denominators({"row_ids": torch.arange(2)}, TARGET_COLUMNS, 36_100)
    assert np.isnan(unlabeled[:, 1:]).all()


def test_output_override_places_latents_beside_predictions(tmp_path) -> None:
    """Keep latents next to an explicit prediction path, not the checkpoint's.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Default and override path resolution are asserted.
    """

    from ihc_mvn.infer import _default_paths

    config = {
        "data": {"inference_path": "tissue.h5ad"},
        "inference": {"output_dir": str(tmp_path / "checkpoint_run")},
    }
    _, output, latent = _default_paths(config, None, tmp_path / "run.parquet", None)
    assert output == tmp_path / "run.parquet"
    assert latent == tmp_path / "run_latent.parquet"
    _, output, latent = _default_paths(config, None, None, None)
    assert latent == tmp_path / "checkpoint_run" / "inference_latent.parquet"

"""Numerical contracts for exact masked MVN and conditional CD8 calculations."""

from __future__ import annotations

import itertools

import numpy as np
import torch
from scipy.stats import multivariate_normal, norm

from ihc_mvn.losses import (
    MVNHurdleLoss,
    censored_mvn_nll,
    conditional_gaussian_cd8,
    conditional_gaussian_moments,
    masked_mvn_nll,
)
from ihc_mvn.model import GlobalCorrelation


def test_masked_mvn_nll_matches_scipy_for_all_sixteen_masks() -> None:
    """Compare every four-target observation mask with SciPy marginal MVNs.

    Args:
        None.

    Returns:
        None: Assertions verify all 16 masks, including exact zero for no positives.
    """

    masks = np.asarray(list(itertools.product([False, True], repeat=4)), dtype=bool)
    values = np.asarray([0.2, -0.5, 1.1, 0.7], dtype=np.float64)
    means = np.asarray([-0.1, 0.3, 0.4, -0.2], dtype=np.float64)
    scales = np.asarray([0.7, 1.2, 0.9, 1.5], dtype=np.float64)
    correlation = np.asarray(
        [
            [1.0, 0.20, -0.10, 0.05],
            [0.20, 1.0, 0.25, -0.15],
            [-0.10, 0.25, 1.0, 0.30],
            [0.05, -0.15, 0.30, 1.0],
        ],
        dtype=np.float64,
    )
    covariance = scales[:, None] * correlation * scales[None, :]
    observed = masked_mvn_nll(
        torch.from_numpy(np.repeat(values[None, :], 16, axis=0)),
        torch.from_numpy(np.repeat(means[None, :], 16, axis=0)),
        torch.from_numpy(np.repeat(scales[None, :], 16, axis=0)),
        torch.from_numpy(correlation),
        torch.from_numpy(masks),
    ).numpy()

    expected = np.zeros(16, dtype=np.float64)
    for row_index, mask in enumerate(masks):
        if mask.any():
            selected = np.flatnonzero(mask)
            expected[row_index] = -multivariate_normal.logpdf(
                values[selected],
                mean=means[selected],
                cov=covariance[np.ix_(selected, selected)],
            )
    np.testing.assert_allclose(observed, expected, rtol=1.0e-10, atol=1.0e-10)
    assert observed[0] == 0.0


def test_hurdle_mvn_loss_has_finite_gradients_through_all_parameters() -> None:
    """Backpropagate hurdle and correlated positive losses through predictions.

    Args:
        None.

    Returns:
        None: Assertions verify finite non-null gradients for all branches.
    """

    torch.manual_seed(3)
    logits = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    means = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    raw_scales = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    scales = torch.nn.functional.softplus(raw_scales) + 0.1
    correlation_module = GlobalCorrelation(num_targets=4).double()
    with torch.no_grad():
        correlation_module.raw_lower[1, 0] = 0.25
        correlation_module.raw_lower[2, 0] = -0.15
        correlation_module.raw_lower[3, 1] = 0.20
    targets = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    standardized = torch.randn(4, 4, dtype=torch.float64)
    loss = MVNHurdleLoss()(
        {
            "hurdle_logits_total": logits,
            "mean_total": means,
            "scales": scales,
            "correlation": correlation_module(),
        },
        targets,
        standardized,
    )
    loss.total.backward()
    for parameter in (
        logits,
        means,
        raw_scales,
        correlation_module.raw_lower,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_conditional_cd8_matches_block_gaussian_formula_and_z_score() -> None:
    """Compare conditional CD8 moments and excess with direct block algebra.

    Args:
        None.

    Returns:
        None: Assertions verify full conditioning and no-conditioning behavior.
    """

    covariance = np.asarray(
        [
            [1.4, 0.2, -0.1, 0.3],
            [0.2, 1.1, 0.15, -0.2],
            [-0.1, 0.15, 0.9, 0.25],
            [0.3, -0.2, 0.25, 1.6],
        ],
        dtype=np.float64,
    )
    means = np.asarray(
        [[0.1, -0.2, 0.3, 0.4], [0.1, -0.2, 0.3, 0.4]],
        dtype=np.float64,
    )
    coordinates = np.asarray(
        [[0.7, -0.5, 1.2, 1.5], [9.0, 9.0, 9.0, -0.8]],
        dtype=np.float64,
    )
    masks = np.asarray(
        [[True, True, True], [False, False, False]],
        dtype=bool,
    )
    output = conditional_gaussian_cd8(
        torch.from_numpy(coordinates),
        torch.from_numpy(means),
        covariance=torch.from_numpy(np.repeat(covariance[None, :, :], 2, axis=0)),
        conditioning_mask=torch.from_numpy(masks),
    )

    cross = covariance[3, :3]
    residual = coordinates[0, :3] - means[0, :3]
    solved_residual = np.linalg.solve(covariance[:3, :3], residual)
    expected_mean = means[0, 3] + cross @ solved_residual
    expected_variance = covariance[3, 3] - cross @ np.linalg.solve(
        covariance[:3, :3],
        cross,
    )
    expected_z = (coordinates[0, 3] - expected_mean) / np.sqrt(expected_variance)
    np.testing.assert_allclose(output.conditional_mean[0].item(), expected_mean)
    np.testing.assert_allclose(output.conditional_variance[0].item(), expected_variance)
    np.testing.assert_allclose(output.excess[0].item(), expected_z)
    np.testing.assert_allclose(output.conditional_mean[1].item(), means[1, 3])
    np.testing.assert_allclose(output.conditional_variance[1].item(), covariance[3, 3])
    np.testing.assert_allclose(
        output.excess[1].item(),
        (coordinates[1, 3] - means[1, 3]) / np.sqrt(covariance[3, 3]),
    )
    assert output.conditioning_count.tolist() == [3, 0]


def _test_covariance() -> np.ndarray:
    """Return a fixed positive-definite four-target covariance.

    Args:
        None.

    Returns:
        np.ndarray: Covariance matrix with shape ``[4, 4]``.
    """

    return np.asarray(
        [
            [1.4, 0.2, -0.1, 0.3],
            [0.2, 1.1, 0.15, -0.2],
            [-0.1, 0.15, 0.9, 0.25],
            [0.3, -0.2, 0.25, 1.6],
        ],
        dtype=np.float64,
    )


def test_conditional_moments_match_block_formula_for_every_coordinate() -> None:
    """Compare all-coordinate conditional moments with explicit block algebra.

    Args:
        None.

    Returns:
        None: Moments of every unobserved coordinate match NumPy.
    """

    covariance = _test_covariance()
    means = np.asarray([0.1, -0.2, 0.3, 0.4])
    values = np.asarray([0.7, -0.5, 1.2, 1.5])
    for mask in itertools.product((False, True), repeat=4):
        observed = np.asarray(mask)
        mean, variance = conditional_gaussian_moments(
            torch.from_numpy(values[None]),
            torch.from_numpy(means[None]),
            torch.from_numpy(covariance[None]),
            torch.from_numpy(observed[None]),
        )
        for target in np.flatnonzero(~observed):
            if observed.any():
                cross = covariance[target, observed]
                block = covariance[np.ix_(observed, observed)]
                expected_mean = means[target] + cross @ np.linalg.solve(
                    block, values[observed] - means[observed]
                )
                expected_variance = covariance[target, target] - cross @ np.linalg.solve(
                    block, cross
                )
            else:
                expected_mean = means[target]
                expected_variance = covariance[target, target]
            np.testing.assert_allclose(
                mean[0, target].item(), expected_mean, atol=1.0e-12
            )
            np.testing.assert_allclose(
                variance[0, target].item(), expected_variance, atol=1.0e-12
            )


def test_censored_mvn_nll_adds_conditional_log_survival() -> None:
    """Match the censored objective with SciPy for one censored coordinate.

    Args:
        None.

    Returns:
        None: Censored, uncensored, and gradient behavior are asserted.
    """

    covariance = _test_covariance()
    scales = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scales, scales)
    means = np.asarray([[0.1, -0.2, 0.3, 0.4]])
    values = np.asarray([[0.7, -0.5, 2.5, 1.5]])
    observed = np.asarray([[True, False, False, True]])
    censored = np.asarray([[False, False, True, False]])
    means_tensor = torch.tensor(means, requires_grad=True)
    nll = censored_mvn_nll(
        torch.from_numpy(values),
        means_tensor,
        torch.from_numpy(scales[None]),
        torch.from_numpy(correlation),
        torch.from_numpy(observed),
        torch.from_numpy(censored),
    )

    kept = observed[0]
    marginal = multivariate_normal(
        mean=means[0, kept], cov=covariance[np.ix_(kept, kept)]
    ).logpdf(values[0, kept])
    cross = covariance[2, kept]
    block = covariance[np.ix_(kept, kept)]
    conditional_mean = means[0, 2] + cross @ np.linalg.solve(
        block, values[0, kept] - means[0, kept]
    )
    conditional_sd = np.sqrt(covariance[2, 2] - cross @ np.linalg.solve(block, cross))
    expected = -marginal - norm.logsf(values[0, 2], conditional_mean, conditional_sd)
    np.testing.assert_allclose(nll.item(), expected)
    nll.sum().backward()
    assert torch.isfinite(means_tensor.grad).all()
    np.testing.assert_allclose(
        censored_mvn_nll(
            torch.from_numpy(values),
            torch.from_numpy(means),
            torch.from_numpy(scales[None]),
            torch.from_numpy(correlation),
            torch.from_numpy(observed),
            torch.zeros_like(torch.from_numpy(censored)),
        ).item(),
        -marginal,
    )


def test_conditional_cd8_target_mask_removes_zero_and_censored_scores() -> None:
    """Return NaN excess where CD8 is not an exact positive observation.

    Args:
        None.

    Returns:
        None: Masked rows are NaN while moments stay defined.
    """

    covariance = np.repeat(_test_covariance()[None], 2, axis=0)
    output = conditional_gaussian_cd8(
        torch.zeros(2, 4, dtype=torch.float64),
        torch.zeros(2, 4, dtype=torch.float64),
        covariance=torch.from_numpy(covariance),
        conditioning_mask=torch.zeros(2, 4, dtype=torch.bool),
        target_mask=torch.tensor([True, False]),
    )
    assert torch.isfinite(output.excess[0])
    assert torch.isnan(output.excess[1])
    assert torch.isfinite(output.conditional_mean).all()

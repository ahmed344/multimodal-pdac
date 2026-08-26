"""Numerical contracts for exact masked MVN and conditional CD8 calculations."""

from __future__ import annotations

import itertools

import numpy as np
import torch
from scipy.stats import multivariate_normal

from ihc_mvn.losses import (
    MVNHurdleLoss,
    conditional_gaussian_cd8,
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

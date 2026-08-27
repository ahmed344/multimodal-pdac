"""Contracts for epoch metric aggregation and training console summaries."""

from __future__ import annotations

import math

import pytest
import torch

from ihc_mvn.train import (
    _empty_metrics,
    _finalize_metrics,
    _format_epoch_summary,
    _update_fit_statistics,
)


def test_fit_metrics_use_exact_epoch_statistics() -> None:
    """Verify balanced accuracy and positive R² from known predictions.

    Args:
        None.

    Returns:
        None: Assertions validate exact aggregate metric values.
    """

    target_names = ("Density_First", "Density_Second")
    metrics = _empty_metrics(len(target_names))
    metrics.update(
        {
            "samples": 4.0,
            "total": 8.0,
            "hurdle": 4.0,
            "positive": 3.0,
            "prior": 1.0,
            "positive_pixels": 4.0,
            "positive_targets": 6.0,
        }
    )
    predictions = {
        "hurdle_logits_total": torch.tensor(
            [[1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]]
        ),
        "mean_total": torch.tensor(
            [[0.0, 1.0], [1.0, 1.0], [20.0, 1.0], [20.0, 1.0]]
        ),
    }
    batch = {
        "target_densities": torch.tensor(
            [[0.1, 0.1], [0.2, 0.1], [0.0, 0.1], [0.0, 0.1]]
        ),
        "targets": torch.tensor(
            [[0.0, 1.0], [2.0, 1.0], [10.0, 1.0], [10.0, 1.0]]
        ),
    }

    _update_fit_statistics(metrics, predictions, batch)
    result = _finalize_metrics(metrics, target_names)

    assert result["total_loss"] == pytest.approx(2.0)
    assert result["presence_balanced_accuracy_first"] == pytest.approx(0.75)
    assert result["presence_balanced_accuracy_second"] == pytest.approx(0.75)
    assert result["presence_balanced_accuracy_macro"] == pytest.approx(0.75)
    assert result["positive_r2_first"] == pytest.approx(0.5)
    assert math.isnan(result["positive_r2_second"])
    assert result["positive_r2_macro"] == pytest.approx(0.5)


def test_epoch_summary_contains_losses_and_higher_is_better_metrics() -> None:
    """Verify the multiline epoch display labels and scientific notation.

    Args:
        None.

    Returns:
        None: Assertions validate the user-facing summary contract.
    """

    metrics = {
        "total_loss": 5.419068,
        "hurdle_loss": 2.0,
        "positive_loss": 3.4,
        "prior_loss": 0.019068,
        "presence_balanced_accuracy_macro": 0.625,
        "positive_r2_macro": 0.125,
        "presence_balanced_accuracy_tumor": 0.75,
        "positive_r2_tumor": 0.25,
    }

    summary = _format_epoch_summary(
        1,
        1000,
        metrics,
        {**metrics, "total_loss": 5.153543},
        ("Density_Tumor",),
    )

    assert summary.startswith(
        "epoch=1/1000 | t-loss: 5.419e+00, 5.154e+00"
    )
    assert "hurdle: 2.000e+00, 2.000e+00" in summary
    assert "metrics (train, validation) | balanced-accuracy: 0.625, 0.625" in summary
    assert "validation by target | Tumor: acc=0.750, R2=0.250" in summary
    assert len(summary.splitlines()) == 3

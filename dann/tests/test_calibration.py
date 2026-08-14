"""Focused tests for ZILN calibration metrics and report generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dann.calibration import (
    build_batch_metrics_frame,
    build_metrics_frame,
    compute_target_metrics,
    load_extracted_from_frame,
    run_calibration,
)


TARGETS = ("Density_CD8", "Density_Tumor")
EPSILON = 1e-6


def _well_specified_sample(
    size: int = 20_000,
    zero_probability: float = 0.3,
    seed: int = 4,
) -> dict[str, np.ndarray]:
    """Draw observations from the model's own generative process.

    Args:
        size (int): Number of simulated pixels.
        zero_probability (float): True structural-zero probability.
        seed (int): Random seed for reproducibility.

    Returns:
        dict[str, np.ndarray]: Targets and matching ZILN parameter arrays.
    """

    rng = np.random.default_rng(seed)
    mu = rng.normal(-1.0, 1.0, size)
    sigma = np.abs(rng.normal(1.0, 0.2, size)) + 0.1
    alpha = rng.normal(0.0, 2.0, size)
    latent = stats.skewnorm.rvs(a=alpha, loc=mu, scale=sigma, random_state=rng)
    densities = 1.0 / (1.0 + np.exp(-latent))
    densities[rng.random(size) < zero_probability] = 0.0
    return {
        "targets": densities,
        "pi": np.full(size, zero_probability),
        "mu": mu,
        "sigma": sigma,
        "alpha": alpha,
    }


def _stack(sample: dict[str, np.ndarray], num_targets: int) -> dict[str, np.ndarray]:
    """Replicate a single-target sample across several identical targets.

    Args:
        sample (dict[str, np.ndarray]): Single-target arrays.
        num_targets (int): Number of target columns to produce.

    Returns:
        dict[str, np.ndarray]: Two-dimensional arrays plus batch codes.
    """

    stacked = {
        key: np.column_stack([values] * num_targets) for key, values in sample.items()
    }
    size = sample["targets"].size
    stacked["batches"] = np.arange(size) % 2
    return stacked


def test_well_specified_model_is_calibrated() -> None:
    """A correctly specified model scores near its ideal calibration values."""

    sample = _well_specified_sample()
    metrics = compute_target_metrics(
        sample["targets"],
        sample["pi"],
        sample["mu"],
        sample["sigma"],
        sample["alpha"],
        EPSILON,
    )
    assert metrics["pit_ks"] < 0.02
    assert metrics["pit_mean"] == pytest.approx(0.5, abs=0.02)
    assert metrics["coverage_90"] == pytest.approx(0.90, abs=0.02)
    assert metrics["coverage_50"] == pytest.approx(0.50, abs=0.02)
    assert metrics["observed_zero_rate"] == pytest.approx(0.3, abs=0.02)
    assert metrics["bias"] == pytest.approx(0.0, abs=0.05)


def test_understated_sigma_produces_u_shaped_pit_and_undercoverage() -> None:
    """Overconfident scales inflate the KS distance and shrink coverage."""

    sample = _well_specified_sample()
    overconfident = compute_target_metrics(
        sample["targets"],
        sample["pi"],
        sample["mu"],
        sample["sigma"] * 0.5,
        sample["alpha"],
        EPSILON,
    )
    assert overconfident["coverage_90"] < 0.80
    assert overconfident["pit_ks"] > 0.05


def test_skew_correction_changes_reported_accuracy() -> None:
    """The corrected mean and the raw ``mu`` are scored as distinct quantities."""

    sample = _well_specified_sample()
    metrics = compute_target_metrics(
        sample["targets"],
        sample["pi"],
        sample["mu"],
        sample["sigma"],
        sample["alpha"],
        EPSILON,
    )
    assert not np.isclose(metrics["mean_skew_shift"], 0.0)
    assert np.isfinite(metrics["pearson_r_mu"])
    assert np.isfinite(metrics["pearson_r_corrected"])


def test_metrics_exclude_zeros_from_the_positive_branch() -> None:
    """Positive-branch counts and errors ignore structural zeros."""

    targets = np.asarray([0.0, 0.0, 0.5, 0.75])
    metrics = compute_target_metrics(
        targets,
        np.full(4, 0.5),
        np.zeros(4),
        np.ones(4),
        np.zeros(4),
        EPSILON,
    )
    assert metrics["n"] == 4.0
    assert metrics["n_zero"] == 2.0
    assert metrics["n_positive"] == 2.0
    assert metrics["observed_zero_rate"] == pytest.approx(0.5)


def test_all_zero_target_yields_nan_positive_metrics() -> None:
    """A target with no positive observations reports NaN rather than failing."""

    metrics = compute_target_metrics(
        np.zeros(8),
        np.full(8, 0.9),
        np.zeros(8),
        np.ones(8),
        np.zeros(8),
        EPSILON,
    )
    assert metrics["n_positive"] == 0.0
    assert np.isnan(metrics["pit_ks"])
    assert np.isnan(metrics["coverage_90"])
    assert np.isnan(metrics["auc"])
    assert np.isfinite(metrics["brier"])


def test_metrics_frames_cover_every_target_and_batch() -> None:
    """Global and per-batch tables enumerate the expected rows."""

    extracted = _stack(_well_specified_sample(size=2_000), len(TARGETS))
    frame = build_metrics_frame(extracted, TARGETS, "test", EPSILON)
    assert list(frame["target"]) == list(TARGETS)
    assert (frame["split"] == "test").all()

    batch_frame = build_batch_metrics_frame(
        extracted, TARGETS, ["slide_a", "slide_b"], "test", EPSILON
    )
    assert len(batch_frame) == 2 * len(TARGETS)
    assert set(batch_frame["batch"]) == {"slide_a", "slide_b"}


def test_round_trip_through_saved_wide_table() -> None:
    """A saved latent-UMAP table rebuilds the arrays the report needs."""

    extracted = _stack(_well_specified_sample(size=500), len(TARGETS))
    frame = pd.DataFrame({"batch_id": extracted["batches"]})
    frame["batch"] = np.where(extracted["batches"] == 0, "slide_a", "slide_b")
    frame["split"] = "test"
    for index, column in enumerate(TARGETS):
        frame[column] = extracted["targets"][:, index]
        for prefix, key in (
            ("pi_", "pi"),
            ("mu_", "mu"),
            ("sigma_", "sigma"),
            ("alpha_", "alpha"),
        ):
            frame[f"{prefix}{column}"] = extracted[key][:, index]

    restored, batch_names = load_extracted_from_frame(frame, TARGETS)
    assert batch_names == ["slide_a", "slide_b"]
    for key in ("targets", "pi", "mu", "sigma", "alpha"):
        np.testing.assert_allclose(restored[key], extracted[key])


def test_load_from_frame_rejects_missing_columns() -> None:
    """A table lacking a parameter column fails loudly instead of silently."""

    frame = pd.DataFrame({"batch_id": [0], "batch": ["a"], "Density_CD8": [0.5]})
    with pytest.raises(KeyError, match="pi_Density_CD8"):
        load_extracted_from_frame(frame, ("Density_CD8",))


def test_run_calibration_writes_every_artifact(tmp_path: Path) -> None:
    """The report writes both tables and all four figures."""

    extracted = _stack(_well_specified_sample(size=1_000), len(TARGETS))
    config = {
        "figure_dpi": 20,
        "point_size": 2.0,
        "calibration_pit_bins": 10,
        "calibration_coverage_levels": [0.50, 0.90],
    }
    output_dir = run_calibration(
        extracted=extracted,
        target_columns=TARGETS,
        batch_names=["slide_a", "slide_b"],
        split="test",
        output_dir=tmp_path / "calibration",
        config=config,
        epsilon=EPSILON,
    )
    expected = {
        "calibration_metrics.csv",
        "calibration_metrics_by_batch.csv",
        "calibration_pit.png",
        "calibration_scatter_corrected.png",
        "calibration_hurdle_reliability.png",
        "calibration_interval_coverage.png",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    written = pd.read_csv(output_dir / "calibration_metrics.csv")
    assert list(written["target"]) == list(TARGETS)
    assert "coverage_90" in written.columns

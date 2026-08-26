"""Contracts for configuration, targets, sparse scaling, and inference data."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from ihc_mvn.config import EXPECTED_TARGETS, load_config, validate_config
from ihc_mvn.data_loader import (
    CategoryMapping,
    create_inference_dataset,
    load_anndata_metadata,
    sparse_collate,
    within_slide_split_indices,
)
from ihc_mvn.scaling import SparseFeatureScaler, fit_sparse_feature_scaler
from ihc_mvn.targets import (
    TARGET_COLUMNS,
    TargetStandardizer,
    build_target_arrays,
    haldane_anscombe_coordinates,
    target_denominators,
)

from conftest import write_csr_anndata


def valid_config(tmp_path: Path) -> dict[str, Any]:
    """Build a valid isolated configuration mapping.

    Args:
        tmp_path (Path): Temporary root used for absolute input/output paths.

    Returns:
        dict[str, Any]: Configuration satisfying the current validator.
    """

    config_path = Path(__file__).parents[1] / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["data"]["path"] = str(tmp_path / "train.h5ad")
    config["data"]["inference_path"] = str(tmp_path / "infer.h5ad")
    config["output"]["directory"] = str(tmp_path / "outputs")
    return config


def fit_example_scaler(tmp_path: Path) -> tuple[Path, SparseFeatureScaler, np.ndarray]:
    """Fit a deterministic two-slide scaler on selected synthetic rows.

    Args:
        tmp_path (Path): Temporary directory for the backed AnnData file.

    Returns:
        tuple[Path, SparseFeatureScaler, np.ndarray]: Training path, fitted scaler,
            and dense count matrix used to calculate expectations.
    """

    counts = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 4.0],
            [0.0, 3.0, 0.0],
            [1.0, 0.0, 2.0],
            [3.0, 0.0, 0.0],
            [9.0, 9.0, 9.0],
        ],
        dtype=np.float32,
    )
    path = write_csr_anndata(
        tmp_path / "scaler_train.h5ad",
        counts,
        ["slide_a"] * 3 + ["slide_b"] * 3,
        ["patient_a"] * 3 + ["patient_b"] * 3,
        densities=np.full((6, 4), 0.1, dtype=np.float32),
    )
    scaler = fit_sparse_feature_scaler(
        path,
        "layers/counts",
        train_indices=np.asarray([0, 1, 2, 3, 4]),
        slide_codes=np.asarray([0, 0, 0, 1, 1, 1]),
        slide_names=("slide_a", "slide_b"),
        scale_floor=1.0e-6,
        row_chunk_size=2,
    )
    return path, scaler, counts


def test_config_validation_and_loading(tmp_path: Path) -> None:
    """Validate required schema, exact target order, and absolute paths.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        None: Assertions verify accepted and rejected configurations.
    """

    config = valid_config(tmp_path)
    validate_config(config)
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    assert load_config(path)["data"]["target_columns"] == EXPECTED_TARGETS

    reordered = copy.deepcopy(config)
    reordered["data"]["target_columns"] = list(reversed(EXPECTED_TARGETS))
    with pytest.raises(ValueError, match="exactly"):
        validate_config(reordered)

    relative = copy.deepcopy(config)
    relative["data"]["path"] = "relative.h5ad"
    with pytest.raises(ValueError, match="absolute"):
        validate_config(relative)


def test_within_slide_split_is_disjoint_complete_and_deterministic() -> None:
    """Check deterministic within-slide 80/10/10 row partitioning.

    Args:
        None.

    Returns:
        None: Assertions verify split set properties and per-slide counts.
    """

    slide_codes = np.repeat(np.arange(3), 20)
    first = within_slide_split_indices(slide_codes, 0.8, 0.1, 0.1, seed=41)
    second = within_slide_split_indices(slide_codes, 0.8, 0.1, 0.1, seed=41)
    for split_name in ("train", "validation", "test"):
        np.testing.assert_array_equal(first[split_name], second[split_name])
    sets = {name: set(values.tolist()) for name, values in first.items()}
    assert sets["train"].isdisjoint(sets["validation"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])
    assert set.union(*sets.values()) == set(range(slide_codes.size))
    for slide_code in range(3):
        assert np.sum(slide_codes[first["train"]] == slide_code) == 16
        assert np.sum(slide_codes[first["validation"]] == slide_code) == 2
        assert np.sum(slide_codes[first["test"]] == slide_code) == 2


def test_sparse_scaler_matches_standard_scaler_variance_with_implicit_zeros(
    tmp_path: Path,
) -> None:
    """Compare fitted scales with population variance over dense-equivalent rows.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        None: Assertions verify exact per-slide/global scales and fallback.
    """

    _, scaler, counts = fit_example_scaler(tmp_path)
    transformed = np.log1p(counts.astype(np.float64))
    expected_a = transformed[:3].std(axis=0, ddof=0)
    expected_b = transformed[3:5].std(axis=0, ddof=0)
    expected_global = transformed[:5].std(axis=0, ddof=0)
    floor = 1.0e-6
    expected_global[expected_global < floor] = 1.0
    expected_a[expected_a < floor] = expected_global[expected_a < floor]
    expected_b[expected_b < floor] = expected_global[expected_b < floor]
    np.testing.assert_allclose(scaler.per_slide_scales[0], expected_a)
    np.testing.assert_allclose(scaler.per_slide_scales[1], expected_b)
    np.testing.assert_allclose(scaler.global_scales, expected_global)
    np.testing.assert_array_equal(scaler.scales_for_slide(-1), scaler.global_scales)
    np.testing.assert_array_equal(scaler.scales_for_slide(99), scaler.global_scales)


def test_scaler_state_round_trip_and_frozen_known_unknown_inference(
    tmp_path: Path,
) -> None:
    """Ensure inference restores frozen scales and uses global unknown fallback.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        None: Assertions verify state fidelity, category codes, and scaled values.
    """

    _, fitted, _ = fit_example_scaler(tmp_path)
    restored = SparseFeatureScaler.from_state(fitted.to_state())
    frozen_before = copy.deepcopy(restored.to_state())
    inference_path = write_csr_anndata(
        tmp_path / "inference.h5ad",
        np.asarray([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
        ["slide_a", "new_slide"],
        ["patient_a", "new_patient"],
        feature_names=["peak_0", "peak_1", "peak_2"],
    )
    patient_mapping = CategoryMapping.from_names("patient", ("patient_a", "patient_b"))
    dataset = create_inference_dataset(inference_path, restored, patient_mapping)
    known = dataset[0]
    unknown = dataset[1]
    assert known["slide_code"] == 0
    assert unknown["slide_code"] == -1
    assert known["patient_code"] == 0
    assert unknown["patient_code"] == -1
    np.testing.assert_allclose(
        known["values"],
        np.asarray([np.log1p(3.0) / restored.per_slide_scales[0, 0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        unknown["values"],
        np.asarray([np.log1p(3.0) / restored.global_scales[0]], dtype=np.float32),
    )
    assert restored.to_state() == frozen_before
    dataset.close()


def test_haldane_denominators_cover_boundary_and_overfull_residual_cases() -> None:
    """Check finite target coordinates at zero, full, and residual edge cases.

    Args:
        None.

    Returns:
        None: Assertions verify denominator construction and corrected log odds.
    """

    counts = np.asarray(
        [
            [0, 0, 36_100, 1],
            [36_100, 0, 7, 36_100],
            [36_099, 2, 1, 0],
        ],
        dtype=np.int64,
    )
    denominators = target_denominators(counts)
    expected = np.asarray(
        [
            [36_100, 36_100, 36_100, 36_100],
            [36_100, 0, 7, 36_100],
            [36_100, 2, 1, 1],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(denominators, expected)
    coordinates = haldane_anscombe_coordinates(counts, denominators)
    assert np.isfinite(coordinates).all()
    np.testing.assert_allclose(coordinates[1, 1], 0.0)
    np.testing.assert_allclose(
        coordinates[0, 0],
        np.log(0.5 / 36_100.5),
    )


def test_target_standardizer_uses_training_positives_only() -> None:
    """Verify target statistics exclude zeros and held-out positive rows.

    Args:
        None.

    Returns:
        None: Assertions verify positive counts, moments, and affine inversion.
    """

    densities = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.3, 0.4],
            [0.2, 0.2, 0.0, 0.5],
            [0.9, 0.9, 0.9, 0.9],
        ],
        dtype=np.float32,
    )
    arrays = build_target_arrays(densities)
    train_indices = np.asarray([0, 1, 2])
    standardizer = TargetStandardizer.fit(
        arrays.coordinates,
        arrays.positive_mask,
        train_indices,
    )
    for target_index in range(4):
        selected = arrays.coordinates[
            train_indices[arrays.positive_mask[train_indices, target_index]],
            target_index,
        ]
        np.testing.assert_allclose(standardizer.means[target_index], selected.mean())
        np.testing.assert_allclose(standardizer.scales[target_index], selected.std(ddof=0))
        assert standardizer.positive_counts[target_index] == selected.size
    standardized = standardizer.transform(arrays.coordinates)
    np.testing.assert_allclose(
        standardizer.inverse_transform(standardized),
        arrays.coordinates,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_sparse_collator_preserves_empty_and_nonempty_rows() -> None:
    """Check sparse concatenation, membership indices, metadata, and targets.

    Args:
        None.

    Returns:
        None: Assertions verify the complete collated tensor schema.
    """

    samples = [
        {
            "feature_indices": np.asarray([0, 2], dtype=np.int64),
            "values": np.asarray([1.0, 3.0], dtype=np.float32),
            "slide_code": 0,
            "patient_code": 2,
            "row_id": 7,
            "target_densities": np.zeros(4, dtype=np.float32),
            "target_counts": np.zeros(4, dtype=np.int64),
            "target_denominators": np.ones(4, dtype=np.int64),
            "target_positive_mask": np.zeros(4, dtype=bool),
            "target_coordinates": np.zeros(4, dtype=np.float32),
            "targets": np.zeros(4, dtype=np.float32),
        },
        {
            "feature_indices": np.asarray([], dtype=np.int64),
            "values": np.asarray([], dtype=np.float32),
            "slide_code": 1,
            "patient_code": 3,
            "row_id": 8,
            "target_densities": np.ones(4, dtype=np.float32),
            "target_counts": np.ones(4, dtype=np.int64),
            "target_denominators": np.ones(4, dtype=np.int64),
            "target_positive_mask": np.ones(4, dtype=bool),
            "target_coordinates": np.ones(4, dtype=np.float32),
            "targets": np.ones(4, dtype=np.float32),
        },
    ]
    batch = sparse_collate(samples)
    assert set(batch) == {
        "feature_indices",
        "values",
        "sample_indices",
        "feature_counts",
        "slide_codes",
        "patient_codes",
        "row_ids",
        "target_densities",
        "target_counts",
        "target_denominators",
        "target_positive_mask",
        "target_coordinates",
        "targets",
    }
    assert batch["feature_indices"].tolist() == [0, 2]
    assert batch["sample_indices"].tolist() == [0, 0]
    assert batch["feature_counts"].tolist() == [2, 0]
    assert batch["targets"].shape == (2, 4)


def test_metadata_rejects_duplicate_feature_names(tmp_path: Path) -> None:
    """Require unique feature names before a training schema is accepted.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        None: The contract requires duplicate feature names to raise.
    """

    path = write_csr_anndata(
        tmp_path / "duplicate_features.h5ad",
        np.ones((2, 3), dtype=np.float32),
        ["slide_a", "slide_a"],
        ["patient_a", "patient_a"],
        feature_names=["peak_a", "peak_a", "peak_b"],
        densities=np.full((2, 4), 0.1, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="duplicate|unique"):
        load_anndata_metadata(
            path,
            TARGET_COLUMNS,
            "batch",
            "patient",
        )


def test_inference_rejects_reordered_features(tmp_path: Path) -> None:
    """Require inference feature identity and order to match the fitted schema.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        None: The contract requires a reordered feature axis to raise.
    """

    _, scaler, _ = fit_example_scaler(tmp_path)
    path = write_csr_anndata(
        tmp_path / "reordered.h5ad",
        np.ones((2, 3), dtype=np.float32),
        ["slide_a", "slide_a"],
        ["patient_a", "patient_a"],
        feature_names=["peak_2", "peak_1", "peak_0"],
    )
    patient_mapping = CategoryMapping.from_names("patient", ("patient_a",))
    with pytest.raises(ValueError, match="feature|order|schema"):
        create_inference_dataset(path, scaler, patient_mapping)


def test_inference_treats_nonfinite_optional_targets_as_unlabeled(
    tmp_path: Path,
) -> None:
    """Allow tissue inference when optional density columns contain missing values.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        None: Assertions verify frozen-scaler inference omits invalid labels.
    """

    _, scaler, _ = fit_example_scaler(tmp_path)
    densities = np.full((2, 4), 0.1, dtype=np.float32)
    densities[1, 3] = np.nan
    path = write_csr_anndata(
        tmp_path / "partially_labeled.h5ad",
        np.ones((2, 3), dtype=np.float32),
        ["slide_a", "slide_a"],
        ["patient_a", "patient_a"],
        densities=densities,
    )
    finite_arrays = build_target_arrays(np.full((2, 4), 0.1, dtype=np.float32))
    standardizer = TargetStandardizer.fit(
        finite_arrays.coordinates,
        finite_arrays.positive_mask,
        np.asarray([0, 1]),
    )
    dataset = create_inference_dataset(
        path,
        scaler,
        CategoryMapping.from_names("patient", ("patient_a",)),
        target_standardizer=standardizer,
    )
    assert dataset.metadata.target_arrays is None
    assert "targets" not in dataset[0]
    dataset.close()

"""Contracts for sparse encoding, hierarchy, covariance, and checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ihc_mvn.model import (
    GlobalCorrelation,
    IHCMultivariateModel,
    IntensityWeightedPeakEncoder,
    MultivariateHurdleHead,
    NonCenteredRandomIntercepts,
)
from ihc_mvn.scaling import SparseFeatureScaler
from ihc_mvn.targets import TARGET_COLUMNS, TargetStandardizer


def build_tiny_model() -> IHCMultivariateModel:
    """Construct a fast deterministic-shape model for unit tests.

    Args:
        None.

    Returns:
        IHCMultivariateModel: Small four-target model.
    """

    encoder = IntensityWeightedPeakEncoder(
        num_peaks=3,
        embedding_dim=2,
        peak_hidden_dims=(),
        peak_output_dim=2,
        aggregation_hidden_dims=(),
        latent_dim=2,
        activation="relu",
        dropout=0.0,
        use_layer_norm=False,
    )
    head = MultivariateHurdleHead(
        latent_dim=2,
        hidden_dims=(),
        num_targets=4,
        use_layer_norm=False,
    )
    random_intercepts = NonCenteredRandomIntercepts(
        num_patients=2,
        num_slides=2,
        num_targets=4,
    )
    correlation = GlobalCorrelation(num_targets=4)
    return IHCMultivariateModel(encoder, head, random_intercepts, correlation)


def sparse_batch() -> dict[str, torch.Tensor]:
    """Build a two-row sparse batch using data-loader key names.

    Args:
        None.

    Returns:
        dict[str, torch.Tensor]: Sparse feature and category tensors.
    """

    return {
        "feature_indices": torch.tensor([0, 1, 2], dtype=torch.long),
        "values": torch.tensor([2.0, 3.0, 4.0]),
        "sample_indices": torch.tensor([0, 0, 1], dtype=torch.long),
        "feature_counts": torch.tensor([2, 1], dtype=torch.long),
        "patient_codes": torch.tensor([0, -1], dtype=torch.long),
        "slide_codes": torch.tensor([1, -1], dtype=torch.long),
    }


def test_encoder_matches_hand_calculated_intensity_weighted_mean() -> None:
    """Compare Deep Sets encoding with a manually calculated linear example.

    Args:
        None.

    Returns:
        None: Assertions verify intensity weighting and per-row mean pooling.
    """

    encoder = IntensityWeightedPeakEncoder(
        num_peaks=3,
        embedding_dim=2,
        peak_hidden_dims=(),
        peak_output_dim=2,
        aggregation_hidden_dims=(),
        latent_dim=2,
        use_layer_norm=False,
    )
    with torch.no_grad():
        encoder.embedding.weight.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
        encoder.peak_mlp[0].weight.copy_(torch.eye(2))
        encoder.peak_mlp[0].bias.zero_()
        encoder.aggregation_mlp[0].weight.copy_(torch.eye(2))
        encoder.aggregation_mlp[0].bias.zero_()
    encoded = encoder(
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([2.0, 3.0, 4.0]),
        torch.tensor([0, 0, 1], dtype=torch.long),
        torch.tensor([2, 1], dtype=torch.long),
    )
    expected = torch.tensor([[5.5, 8.0], [20.0, 24.0]])
    torch.testing.assert_close(encoded, expected)


def test_model_accepts_loader_and_conventional_sparse_keys() -> None:
    """Require identical encoding for both supported sparse key vocabularies.

    Args:
        None.

    Returns:
        None: Assertions verify backward-compatible key aliases.
    """

    model = build_tiny_model()
    loader_batch = sparse_batch()
    conventional = {
        "peak_indices": loader_batch["feature_indices"],
        "intensities": loader_batch["values"],
        "sample_indices": loader_batch["sample_indices"],
        "peak_counts": loader_batch["feature_counts"],
        "patients": loader_batch["patient_codes"],
        "slides": loader_batch["slide_codes"],
    }
    torch.testing.assert_close(model.encode(loader_batch), model.encode(conventional))
    first = model(loader_batch)
    second = model(conventional)
    torch.testing.assert_close(first["mean_total"], second["mean_total"])


def test_model_shapes_positive_definite_correlation_and_unknown_zero_effects() -> None:
    """Check complete output schema, covariance positivity, and unknown groups.

    Args:
        None.

    Returns:
        None: Assertions verify dimensions, aliases, eigenvalues, and effects.
    """

    model = build_tiny_model()
    with torch.no_grad():
        model.random_intercepts.patient_hurdle_raw.weight.fill_(1.0)
        model.random_intercepts.patient_mean_raw.weight.fill_(1.0)
        model.random_intercepts.slide_hurdle_raw.weight.fill_(1.0)
        model.random_intercepts.slide_mean_raw.weight.fill_(1.0)
        model.correlation.raw_lower[1, 0] = 0.4
        model.correlation.raw_lower[2, 0] = -0.2
    output = model(sparse_batch())
    matrix_keys = (
        "hurdle_logits_msi",
        "mean_msi",
        "scales",
        "hurdle_logits_patient",
        "mean_patient",
        "hurdle_logits_total",
        "mean_total",
        "hurdle_logits",
        "pi_logits",
        "mean",
        "mu",
        "sigma",
    )
    for key in matrix_keys:
        assert output[key].shape == (2, 4)
    assert output["latent"].shape == (2, 2)
    assert output["correlation"].shape == (4, 4)
    assert output["covariance"].shape == (2, 4, 4)
    torch.testing.assert_close(
        output["correlation"].diagonal(),
        torch.ones(4),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    assert torch.linalg.eigvalsh(output["correlation"]).min() > 0.0
    assert torch.linalg.eigvalsh(output["covariance"]).min() > 0.0
    for key in (
        "patient_hurdle_effect",
        "patient_mean_effect",
        "slide_hurdle_effect",
        "slide_mean_effect",
    ):
        assert torch.count_nonzero(output[key][0]) > 0
        torch.testing.assert_close(output[key][1], torch.zeros(4))


def test_model_from_config_supports_conventional_and_legacy_dimension_keys() -> None:
    """Exercise model configuration aliases without importing another package.

    Args:
        None.

    Returns:
        None: Assertions verify both accepted independent config vocabularies.
    """

    conventional = {
        "model": {
            "num_peaks": 3,
            "embedding_dim": 2,
            "peak_hidden_dims": [],
            "peak_output_dim": 2,
            "aggregation_hidden_dims": [],
            "latent_dim": 2,
            "head_hidden_dims": [],
            "use_layer_norm": False,
        }
    }
    legacy = {
        "model": {
            "n_peaks": 3,
            "embedding_dim": 2,
            "peak_hidden_dims": [],
            "peak_output_dim": 2,
            "aggregation_hidden_dims": [],
            "latent_dim": 2,
            "biology_hidden_dims": [],
            "layer_norm": False,
        }
    }
    first = IHCMultivariateModel.from_config(conventional, num_patients=1, num_slides=1)
    second = IHCMultivariateModel.from_config(legacy, num_patients=1, num_slides=1)
    assert first.encoder.num_peaks == second.encoder.num_peaks == 3
    assert first.latent_dim == second.latent_dim == 2
    assert first.num_targets == second.num_targets == 4


def test_available_training_state_round_trips_through_checkpoint(tmp_path: Path) -> None:
    """Round-trip model, optimizer, frozen preprocessing, and ordered schema state.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        None: Assertions verify the checkpoint-ready components survive serialization.
    """

    model = build_tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    output = model(sparse_batch())
    objective = output["mean_total"].square().mean() + model.prior_nll() * 1.0e-4
    objective.backward()
    optimizer.step()

    scaler = SparseFeatureScaler(
        slide_names=("slide_a",),
        slide_to_code={"slide_a": 0},
            feature_names=("peak_0", "peak_1", "peak_2"),
        per_slide_scales=np.asarray([[1.0, 2.0, 3.0]]),
        global_scales=np.asarray([1.5, 2.5, 3.5]),
        slide_training_counts=np.asarray([8]),
        scale_floor=1.0e-6,
        matrix_key="layers/counts",
        transform="log1p",
        center=False,
        num_features=3,
        transform_metadata={"frozen_at_inference": True},
    )
    standardizer = TargetStandardizer(
        target_names=TARGET_COLUMNS,
        means=np.asarray([1.0, 2.0, 3.0, 4.0]),
        scales=np.asarray([0.5, 0.6, 0.7, 0.8]),
        positive_counts=np.asarray([5, 6, 7, 8]),
        standard_deviation_floor=1.0e-6,
    )
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "feature_names": ["peak_0", "peak_1", "peak_2"],
        "target_names": list(TARGET_COLUMNS),
        "scaler_state": scaler.to_state(),
        "target_standardizer_state": standardizer.to_state(),
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)

    restored_model = build_tiny_model()
    restored_model.load_state_dict(loaded["model_state"])
    restored_scaler = SparseFeatureScaler.from_state(loaded["scaler_state"])
    restored_standardizer = TargetStandardizer.from_state(
        loaded["target_standardizer_state"]
    )
    assert loaded["feature_names"] == ["peak_0", "peak_1", "peak_2"]
    assert tuple(loaded["target_names"]) == TARGET_COLUMNS
    assert restored_scaler.to_state() == scaler.to_state()
    assert restored_standardizer.to_state() == standardizer.to_state()
    original_output = model(sparse_batch())
    restored_output = restored_model(sparse_batch())
    torch.testing.assert_close(
        original_output["mean_total"],
        restored_output["mean_total"],
    )

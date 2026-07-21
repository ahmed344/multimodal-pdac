"""Focused unit tests for sparse loading, model math, and interpretation."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from dann.analyze import embedding_cosine_families
from dann.data_loader import (
    SparseAnnDataDataset,
    load_anndata_metadata,
    sparse_collate,
    stratified_split_indices,
)
from dann.losses import ZILNLoss
from dann.model import (
    AdversarialLatentFusion,
    GradientReversal,
    IntensityWeightedPeakEncoder,
)


@pytest.fixture
def tiny_h5ad(tmp_path: Path) -> Path:
    """Create a tiny sparse AnnData file with the production schema.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        Path: Written synthetic ``.h5ad`` path.
    """

    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [0.0, 2.0, 0.0, 4.0],
                [1.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [4.0, 0.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )
    )
    observations = pd.DataFrame(
        {
            "Density_Tumor": [0.0, 0.1, 0.2, 0.0, 0.4, 1.0],
            "Density_Stroma": [0.1, 0.2, 0.0, 0.4, 0.5, 0.6],
            "Density_CD8": [0.0, 0.01, 0.0, 0.03, 0.04, 0.05],
            "Density_Collagen": [0.2, 0.0, 0.4, 0.5, 0.6, 0.7],
            "batch": pd.Categorical(["a", "a", "a", "b", "b", "b"]),
        }
    )
    adata = ad.AnnData(X=matrix, obs=observations)
    adata.var_names = ["500.1", "501.2", "502.3", "503.4"]
    path = tmp_path / "tiny.h5ad"
    adata.write_h5ad(path)
    return path


def test_sparse_dataset_and_collator_never_expand_features(tiny_h5ad: Path) -> None:
    """Verify CSR rows remain concatenated nonzero lists through collation.

    Args:
        tiny_h5ad (Path): Synthetic sparse AnnData fixture.

    Returns:
        None: Assertions validate sparse values and labels.
    """

    targets = [
        "Density_Tumor",
        "Density_Stroma",
        "Density_CD8",
        "Density_Collagen",
    ]
    metadata = load_anndata_metadata(tiny_h5ad, targets, "batch")
    dataset = SparseAnnDataDataset(
        tiny_h5ad,
        np.asarray([0, 1, 2]),
        metadata,
        intensity_transform="none",
    )
    batch = sparse_collate([dataset[index] for index in range(3)])
    assert batch["peak_indices"].tolist() == [1, 3, 0, 2]
    assert batch["intensities"].tolist() == [2.0, 4.0, 1.0, 3.0]
    assert batch["sample_indices"].tolist() == [0, 0, 1, 1]
    assert batch["peak_counts"].tolist() == [2, 2, 0]
    assert batch["targets"].shape == (3, 4)
    dataset.close()
    log_dataset = SparseAnnDataDataset(
        tiny_h5ad,
        np.asarray([0]),
        metadata,
        intensity_transform="log1p",
    )
    np.testing.assert_allclose(
        log_dataset[0]["intensities"], np.log1p([2.0, 4.0]), rtol=1e-6
    )
    log_dataset.close()


def test_intensity_weighted_encoder_matches_pdf_mean() -> None:
    """Verify weighted embedding, peak MLP, mean pooling, and aggregation order.

    Args:
        None.

    Returns:
        None: Assertions compare encoder output to a hand calculation.
    """

    encoder = IntensityWeightedPeakEncoder(
        num_peaks=3,
        embedding_dim=2,
        peak_hidden_dims=[],
        peak_output_dim=2,
        aggregation_hidden_dims=[],
        latent_dim=2,
        dropout=0.0,
        use_layer_norm=False,
    )
    with torch.no_grad():
        encoder.embedding.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
        encoder.peak_mlp[0].weight.copy_(torch.eye(2))
        encoder.peak_mlp[0].bias.zero_()
        encoder.aggregation_mlp[0].weight.copy_(torch.eye(2))
        encoder.aggregation_mlp[0].bias.zero_()
    encoded = encoder(
        peak_indices=torch.tensor([0, 2, 1]),
        intensities=torch.tensor([2.0, 4.0, 3.0]),
        sample_indices=torch.tensor([0, 0, 1]),
        peak_counts=torch.tensor([2, 1]),
    )
    expected = torch.tensor([[3.0, 2.0], [0.0, 3.0]])
    torch.testing.assert_close(encoded, expected)


def test_gradient_reversal_changes_only_gradient_sign_and_scale() -> None:
    """Verify GRL is identity forward and negative-scaled backward.

    Args:
        None.

    Returns:
        None: Assertions validate values and gradients.
    """

    inputs = torch.tensor([1.0, -2.0], requires_grad=True)
    output = GradientReversal()(inputs, strength=0.25)
    torch.testing.assert_close(output, inputs)
    output.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.full_like(inputs, -0.25))


def test_ziln_branches_match_formula_and_handle_one() -> None:
    """Verify hurdle math, positive Gaussian math, and finite exact-one handling.

    Args:
        None.

    Returns:
        None: Assertions validate component values and gradients.
    """

    criterion = ZILNLoss([1.0], logit_epsilon=1e-4, reduction="mean")
    pi_logits = torch.zeros((3, 1), requires_grad=True)
    mu = torch.zeros((3, 1), requires_grad=True)
    sigma = torch.ones((3, 1), requires_grad=True)
    targets = torch.tensor([[0.0], [0.5], [1.0]])
    result = criterion(pi_logits, mu, sigma, targets)
    transformed_one = torch.logit(torch.tensor(1.0 - 1e-4))
    expected_hurdle = torch.tensor(np.log(2.0), dtype=torch.float32)
    expected_positive = transformed_one.square() / 6.0
    torch.testing.assert_close(result.hurdle, expected_hurdle)
    torch.testing.assert_close(result.positive, expected_positive, rtol=2e-4, atol=2e-4)
    assert torch.isfinite(result.total)
    result.total.backward()
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in (pi_logits, mu, sigma)
    )


def test_model_output_shapes() -> None:
    """Verify complete model emits four ZILN triplets and batch logits.

    Args:
        None.

    Returns:
        None: Assertions validate all public output shapes.
    """

    config = {
        "model": {
            "num_peaks": 4,
            "embedding_dim": 3,
            "peak_hidden_dims": [5],
            "peak_output_dim": 5,
            "aggregation_hidden_dims": [4],
            "latent_dim": 2,
            "biology_hidden_dims": [4],
            "discriminator_hidden_dims": [3],
            "activation": "relu",
            "dropout": 0.0,
            "use_layer_norm": False,
            "sigma_min": 1e-4,
        }
    }
    model = AdversarialLatentFusion.from_config(config, num_batches=3, num_targets=4)
    batch = {
        "peak_indices": torch.tensor([0, 2, 1]),
        "intensities": torch.tensor([1.0, 2.0, 3.0]),
        "sample_indices": torch.tensor([0, 0, 1]),
        "peak_counts": torch.tensor([2, 1]),
    }
    outputs = model(batch)
    assert outputs["latent"].shape == (2, 2)
    assert outputs["pi_logits"].shape == (2, 4)
    assert outputs["mu"].shape == (2, 4)
    assert outputs["sigma"].shape == (2, 4)
    assert outputs["batch_logits"].shape == (2, 3)
    assert torch.all(outputs["sigma"] > 0.0)


def test_stratified_splits_and_embedding_families() -> None:
    """Verify disjoint splits and stable cosine-family output dimensions.

    Args:
        None.

    Returns:
        None: Assertions validate partitioning and clustering.
    """

    batches = np.repeat(np.arange(3), 10)
    splits = stratified_split_indices(batches, 0.6, 0.2, 0.2, seed=7)
    combined = np.concatenate(list(splits.values()))
    assert np.unique(combined).size == batches.size
    assert set(splits) == {"train", "validation", "test"}
    for indices in splits.values():
        assert set(np.unique(batches[indices])) == {0, 1, 2}

    embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
    )
    similarity, labels, order = embedding_cosine_families(
        embeddings, family_count=2, linkage_method="average"
    )
    assert similarity.shape == (4, 4)
    np.testing.assert_allclose(np.diag(similarity), 1.0, atol=1e-6)
    assert labels.shape == (4,)
    assert sorted(order.tolist()) == [0, 1, 2, 3]

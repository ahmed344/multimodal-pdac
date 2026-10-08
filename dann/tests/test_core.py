"""Focused unit tests for sparse loading, model math, and interpretation."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse, stats

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
        inference_peak_chunk_size=2,
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
    with torch.inference_mode():
        chunked_encoded = encoder(
            peak_indices=torch.tensor([0, 2, 1]),
            intensities=torch.tensor([2.0, 4.0, 3.0]),
            sample_indices=torch.tensor([0, 0, 1]),
            peak_counts=torch.tensor([2, 1]),
        )
    torch.testing.assert_close(chunked_encoded, expected)


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
    """Verify hurdle math, positive Gaussian math, and exact-one handling.

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


def test_ziln_positive_branch_matches_scipy_normal() -> None:
    """Verify the positive branch is an exact normal negative log likelihood.

    Args:
        None.

    Returns:
        None: Assertions compare against ``scipy.stats.norm`` for several
        locations and scales.
    """

    criterion = ZILNLoss([1.0], logit_epsilon=1e-4, reduction="sum", include_normal_constant=True)
    x = np.array([-1.5, -0.3, 0.0, 0.4, 2.0], dtype=np.float64)
    mu_value, sigma_value = 0.2, 1.3
    targets = torch.from_numpy(1.0 / (1.0 + np.exp(-x))).reshape(-1, 1).to(torch.float32)
    pi_logits = torch.zeros_like(targets)
    mu = torch.full_like(targets, mu_value)
    sigma = torch.full_like(targets, sigma_value)
    result = criterion(pi_logits, mu, sigma, targets)
    expected = -stats.norm.logpdf(x, loc=mu_value, scale=sigma_value).sum()
    assert float(result.positive) == pytest.approx(float(expected), rel=1e-4)


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


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("constant", [False, True])
def test_normal_weighted_likelihood_and_gradients(reduction: str, constant: bool) -> None:
    """Check weighted Gaussian values and analytic derivatives, including zeros."""
    weights = torch.tensor([8., 0., 2., 1.], dtype=torch.float64)
    targets = torch.tensor([[0., .2, .5, 1.], [.3, 0., .8, .1]], dtype=torch.float64)
    logits = torch.full_like(targets, .3, requires_grad=True)
    mu = torch.full_like(targets, -.4, requires_grad=True)
    sigma = torch.full_like(targets, 1.2, requires_grad=True)
    loss = ZILNLoss(weights, logit_epsilon=1e-5, reduction=reduction,
                    include_normal_constant=constant)
    result = loss(logits, mu, sigma, targets)
    positive = targets > 0
    z = torch.logit(targets.clamp(1e-5, 1-1e-5))
    normal = -torch.distributions.Normal(mu, sigma).log_prob(z)
    if not constant:
        normal = normal - .5 * np.log(2 * np.pi)
    hurdle = torch.where(positive, torch.nn.functional.softplus(logits),
                        torch.nn.functional.softplus(-logits))
    divisor = weights.sum() * targets.shape[0] if reduction == "mean" else 1.
    torch.testing.assert_close(result.total, ((hurdle + normal * positive) * weights).sum() / divisor)
    result.total.backward()
    torch.testing.assert_close(mu.grad, (mu-z) / sigma.square() * positive * weights / divisor)
    torch.testing.assert_close(sigma.grad, (1/sigma-(z-mu).square()/sigma.pow(3)) * positive * weights / divisor)
    torch.testing.assert_close(logits.grad, (logits.sigmoid()-(~positive).to(logits)) * weights / divisor)


def test_normal_checkpoint_round_trip(tmp_path: Path, tiny_h5ad: Path) -> None:
    """Preserve normal predictions, target ordering, splits, and optimizer state."""
    from dann.config import load_config
    from dann.data_loader import create_data_bundle
    from dann.train import save_checkpoint, load_training_checkpoint
    config = load_config(Path("dann/config.yaml"))
    config["data"]["path"] = str(tiny_h5ad)
    config["training"].update(num_workers=0, batch_size=2, validation_batch_size=2)
    for key in ("spectral_encoder", "aggregation", "heads"):
        config["model"].pop(key)
    config["model"].update(num_peaks=4, embedding_dim=3, peak_hidden_dims=[4],
                           peak_output_dim=4, aggregation_hidden_dims=[4], latent_dim=3,
                           biology_hidden_dims=[3], discriminator_hidden_dims=[3], dropout=0.)
    bundle = create_data_bundle(config)
    model = AdversarialLatentFusion.from_config(config, num_batches=2, num_targets=4).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    batch = next(iter(bundle.loaders["train"]))
    expected = model(batch)
    path = tmp_path / "normal.pt"
    save_checkpoint(path, model, optimizer, 2, config, bundle, 1.25)
    restored = AdversarialLatentFusion.from_config(config, num_batches=2, num_targets=4).eval()
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    assert load_training_checkpoint(path, restored, restored_optimizer, torch.device("cpu"), config) == (3, 1.25)
    actual = restored(batch)
    assert set(actual) == {"latent", "pi_logits", "pi", "mu", "sigma", "batch_logits"}
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key])
    payload = torch.load(path, weights_only=False)
    assert payload["target_columns"] == tuple(config["data"]["target_columns"])
    for split, indices in bundle.split_indices.items():
        np.testing.assert_array_equal(payload["split_indices"][split], indices)
    for dataset in bundle.datasets.values():
        dataset.close()


def test_resume_training_config_keeps_checkpoint_learning_rate() -> None:
    """Apply only the resume allowlist and report every other requested change.

    Args:
        None.

    Returns:
        None: Assertions validate the effective configuration and ignored paths.
    """

    from dann.train import ignored_resume_settings, resume_training_config
    saved = {
        "execution": {"mode": "spatial", "num_workers": 2, "prefetch_factor": 2, "learning_rate": 0.01},
        "training": {
            "learning_rate": 0.01, "epochs": 30, "num_workers": 2, "prefetch_factor": 2,
            "early_stopping_patience": 5, "early_stopping_min_delta": 0.001,
            "resume_checkpoint": None, "weight_decay": 1e-5,
            "spatial": {"num_workers": 2, "prefetch_factor": 2, "learning_rate": 0.01},
        },
        "loss": {"biology_weight": 5.0},
    }
    requested = {
        "execution": {"mode": "spatial", "num_workers": 8, "prefetch_factor": 4, "learning_rate": 0.2},
        "training": {
            "learning_rate": 0.2, "epochs": 400, "num_workers": 8, "prefetch_factor": 4,
            "early_stopping_patience": 10, "early_stopping_min_delta": 0.002,
            "resume_checkpoint": "latest.pt", "weight_decay": 1e-4,
            "spatial": {"num_workers": 8, "prefetch_factor": 4, "learning_rate": 0.2},
        },
        "loss": {"biology_weight": 9.0},
    }
    effective = resume_training_config(saved, requested)
    assert effective["training"]["learning_rate"] == 0.01
    assert effective["training"]["spatial"]["learning_rate"] == 0.01
    assert effective["execution"]["learning_rate"] == 0.01
    assert effective["training"]["weight_decay"] == 1e-5
    assert effective["loss"]["biology_weight"] == 5.0
    assert effective["training"]["epochs"] == 400
    assert effective["training"]["num_workers"] == 8
    assert effective["training"]["prefetch_factor"] == 4
    assert effective["training"]["spatial"]["num_workers"] == 8
    assert effective["training"]["spatial"]["prefetch_factor"] == 4
    assert effective["execution"]["num_workers"] == 8
    assert effective["execution"]["prefetch_factor"] == 4
    assert effective["training"]["early_stopping_patience"] == 10
    assert effective["training"]["early_stopping_min_delta"] == 0.002
    assert effective["training"]["resume_checkpoint"] == "latest.pt"
    ignored = ignored_resume_settings(requested, effective)
    assert "training.learning_rate" in ignored
    assert "training.spatial.learning_rate" in ignored
    assert "execution.learning_rate" in ignored
    assert "loss.biology_weight" in ignored
    assert "training.epochs" not in ignored
    assert "training.num_workers" not in ignored
    assert "training.prefetch_factor" not in ignored


def test_resume_history_restores_streak_and_omits_later_rows(tmp_path: Path) -> None:
    """Keep completed history rows and continue the early-stopping streak.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions validate kept epochs, the omitted row, and the stale count.
    """

    from dann.train import EarlyStopping, load_resume_history, restore_resume_stopping, write_training_history
    path = tmp_path / "history.csv"
    rows = [
        {"epoch": 1, "validation_cd8_loss": 1.0, "train_total_loss": 3.0},
        {"epoch": 2, "validation_cd8_loss": 0.4, "train_total_loss": 2.0},
        {"epoch": 3, "validation_cd8_loss": 0.45, "train_total_loss": 1.9},
        {"epoch": 4, "validation_cd8_loss": 0.46, "train_total_loss": 1.8},
        {"epoch": 5, "validation_cd8_loss": 0.47, "train_total_loss": 1.7},
        {"epoch": 6, "validation_cd8_loss": 0.48, "train_total_loss": 1.6},
        {"epoch": 7, "validation_cd8_loss": 0.49, "train_total_loss": 1.5},
        {"epoch": 8, "validation_cd8_loss": 0.2, "train_total_loss": 1.4},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    raw = path.read_text(encoding="utf-8").splitlines()
    kept, columns, preserved = load_resume_history(path, 7)
    assert [row["epoch"] for row in kept] == [1, 2, 3, 4, 5, 6, 7]
    assert columns[0] == "epoch"
    stopper = EarlyStopping(patience=10, min_delta=0.001, best=0.4)
    best_epoch = restore_resume_stopping(stopper, kept, True)
    assert best_epoch == 2
    assert stopper.best == 0.4
    assert stopper.stale_epochs == 5
    strict = EarlyStopping(patience=10, min_delta=0.2, best=0.4)
    strict_rows = [
        {"epoch": 1, "validation_cd8_loss": 1.0},
        {"epoch": 2, "validation_cd8_loss": 0.9},
    ]
    assert restore_resume_stopping(strict, strict_rows, True) == 1
    assert strict.best == 0.4
    assert strict.stale_epochs == 1
    kept.append({"epoch": 9, "validation_cd8_loss": 0.3, "train_total_loss": 1.2})
    written = tmp_path / "continued.csv"
    write_training_history(written, kept, columns, preserved)
    text = written.read_text(encoding="utf-8").splitlines()
    assert text[:8] == raw[:8]
    assert float(text[-1].split(",")[0]) == 9
    reloaded, _, _ = load_resume_history(written, 9)
    assert [row["epoch"] for row in reloaded] == [1, 2, 3, 4, 5, 6, 7, 9]
    assert load_resume_history(tmp_path / "missing.csv", 7) == ([], [], {})


def test_smoke_resume_is_rejected() -> None:
    """Refuse a smoke run that would otherwise continue a full checkpoint.

    Args:
        None.

    Returns:
        None: The assertion validates the rejection.
    """

    from dann.train import train_model
    with pytest.raises(ValueError, match="smoke"):
        train_model({"training": {"resume_checkpoint": "missing.pt"}, "results": {"smoke": True}})


def test_resume_optimizer_keeps_saved_learning_rate(tmp_path: Path, tiny_h5ad: Path) -> None:
    """Restore Adam moments and the checkpoint learning rate together.

    Args:
        tmp_path (Path): Pytest temporary directory.
        tiny_h5ad (Path): Synthetic sparse AnnData fixture.

    Returns:
        None: Assertions validate the restored learning rate and moment buffers.
    """

    from dann.config import load_config
    from dann.data_loader import create_data_bundle
    from dann.train import load_training_checkpoint, save_checkpoint
    config = load_config(Path("dann/config.yaml"))
    config["data"]["path"] = str(tiny_h5ad)
    config["training"].update(num_workers=0, batch_size=2, validation_batch_size=2, learning_rate=0.01)
    for key in ("spectral_encoder", "aggregation", "heads"):
        config["model"].pop(key)
    config["model"].update(num_peaks=4, embedding_dim=3, peak_hidden_dims=[4],
                           peak_output_dim=4, aggregation_hidden_dims=[4], latent_dim=3,
                           biology_hidden_dims=[3], discriminator_hidden_dims=[3], dropout=0.)
    bundle = create_data_bundle(config)
    model = AdversarialLatentFusion.from_config(config, num_batches=2, num_targets=4).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    batch = next(iter(bundle.loaders["train"]))
    model(batch)["mu"].sum().backward()
    optimizer.step()
    moments = {
        name: optimizer.state[parameter]["exp_avg"].detach().clone()
        for name, parameter in model.named_parameters()
        if parameter in optimizer.state and "exp_avg" in optimizer.state[parameter]
    }
    assert moments
    path = tmp_path / "optimizer.pt"
    save_checkpoint(path, model, optimizer, 0, config, bundle, 1.0)
    restored = AdversarialLatentFusion.from_config(config, num_batches=2, num_targets=4).train()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.2, weight_decay=1e-5)
    load_training_checkpoint(path, restored, restored_optimizer, torch.device("cpu"), config)
    assert restored_optimizer.param_groups[0]["lr"] == 0.01
    for name, parameter in restored.named_parameters():
        if name in moments:
            torch.testing.assert_close(restored_optimizer.state[parameter]["exp_avg"], moments[name])
    for dataset in bundle.datasets.values():
        dataset.close()

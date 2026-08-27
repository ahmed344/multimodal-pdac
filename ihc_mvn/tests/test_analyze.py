"""Contracts for MVN held-out analysis and visualization artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ihc_mvn import analyze
from ihc_mvn.analyze import embedding_cosine_families, plot_training_curves
from ihc_mvn.tests.test_integration_contract import tiny_pipeline_config
from ihc_mvn.train import train_model


def test_embedding_cosine_families_returns_complete_partition() -> None:
    """Verify cosine clustering covers every peak exactly once.

    Args:
        None.

    Returns:
        None: Assertions validate matrix, labels, and ordering.
    """

    weights = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )
    similarity, families, order = embedding_cosine_families(weights, 2, "average")

    assert similarity.shape == (4, 4)
    assert np.allclose(np.diag(similarity), 1.0)
    assert families.shape == (4,)
    assert set(order) == set(range(4))


def test_plot_training_curves_writes_png(tmp_path: Path) -> None:
    """Verify all MVN history panels can be rendered.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: The expected nonempty PNG is produced.
    """

    history = pd.DataFrame({"epoch": [1.0, 2.0]})
    metrics = (
        "total_loss",
        "hurdle_loss",
        "positive_loss",
        "prior_loss",
        "presence_balanced_accuracy_macro",
        "positive_r2_macro",
    )
    for metric in metrics:
        history[f"train_{metric}"] = [2.0, 1.0]
        history[f"validation_{metric}"] = [2.5, 1.5]
    history_path = tmp_path / "history.csv"
    output_path = tmp_path / "curves.png"
    history.to_csv(history_path, index=False)

    plot_training_curves(history_path, output_path, best_epoch=2, dpi=72)

    assert output_path.stat().st_size > 0


def test_run_analysis_writes_representative_artifacts(
    tiny_h5ad: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train a tiny model and exercise frozen held-out analysis.

    Args:
        tiny_h5ad (Path): Synthetic labeled CSR AnnData.
        tmp_path (Path): Pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Returns:
        None: Representative UMAP, peptide, and MVN outputs are validated.
    """

    model_dir = tmp_path / "model"
    config = tiny_pipeline_config(tiny_h5ad, model_dir)
    config["analysis"].update(
        {
            "split": "train",
            "max_samples": 8,
            "batch_size": 4,
            "num_workers": 0,
            "activity_chunk_size": 4,
            "activity_max_nonzeros": 20,
            "heatmap_top_peaks": 3,
            "family_count": 2,
            "figure_dpi": 72,
            "point_size": 4.0,
            "output_dir": str(tmp_path / "analysis"),
        }
    )
    checkpoint = train_model(config)

    def fake_umap(
        latent: np.ndarray, analysis_config: object
    ) -> np.ndarray:
        """Return deterministic coordinates without fitting UMAP.

        Args:
            latent (np.ndarray): Extracted latent matrix.
            analysis_config (object): Unused analysis configuration.

        Returns:
            np.ndarray: First two latent dimensions.
        """

        del analysis_config
        return latent[:, :2].astype(np.float32)

    monkeypatch.setattr(analyze, "fit_latent_umap", fake_umap)
    output_dir = analyze.run_analysis(config, checkpoint)

    expected = (
        output_dir / "umap" / "latent_umap.csv",
        output_dir / "peptide_similarity_heatmap.png",
        output_dir / "peptide_families.csv",
        output_dir / "target_correlation_heatmaps.png",
        output_dir / "mahalanobis_qq.png",
        model_dir / "training_curves.png",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)

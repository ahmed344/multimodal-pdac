"""Contracts for configurable post-training stage orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from ihc_mvn import workflow
from ihc_mvn.tests.test_integration_contract import tiny_pipeline_config


def test_post_training_workflow_dispatches_all_stages(
    tiny_h5ad: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify analysis, inference, and spatial stages run in order.

    Args:
        tiny_h5ad (Path): Synthetic input path used in configuration.
        tmp_path (Path): Pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Function replacement fixture.

    Returns:
        None: Stage calls and returned paths are asserted.
    """

    config = tiny_pipeline_config(tiny_h5ad, tmp_path / "model")
    calls: list[str] = []
    predictions = tmp_path / "inference" / "inference.parquet"
    latents = tmp_path / "inference" / "inference_latent.parquet"

    def fake_analysis(
        config_value: Mapping[str, Any],
        checkpoint_override: Path | None = None,
    ) -> Path:
        """Record the analysis stage.

        Args:
            config_value (Mapping[str, Any]): Pipeline configuration.
            checkpoint_override (Path | None): Selected checkpoint.

        Returns:
            Path: Synthetic analysis directory.
        """

        del config_value, checkpoint_override
        calls.append("analysis")
        return tmp_path / "analysis"

    def fake_inference(**kwargs: object) -> tuple[Path, Path]:
        """Record the inference stage.

        Args:
            **kwargs (object): Inference keyword arguments.

        Returns:
            tuple[Path, Path]: Synthetic prediction and latent paths.
        """

        del kwargs
        calls.append("inference")
        return predictions, latents

    def fake_spatial(
        config_value: Mapping[str, Any], **kwargs: object
    ) -> Path:
        """Record the spatial stage.

        Args:
            config_value (Mapping[str, Any]): Pipeline configuration.
            **kwargs (object): Spatial path overrides.

        Returns:
            Path: Synthetic spatial directory.
        """

        del config_value, kwargs
        calls.append("spatial")
        return tmp_path / "spatial"

    monkeypatch.setattr(workflow, "run_analysis", fake_analysis)
    monkeypatch.setattr(workflow, "run_inference", fake_inference)
    monkeypatch.setattr(workflow, "run_spatial_heatmaps", fake_spatial)

    outputs = workflow.run_post_training_workflow(config, tmp_path / "best.pt")

    assert calls == ["analysis", "inference", "spatial"]
    assert outputs["predictions"] == predictions
    assert outputs["spatial"] == tmp_path / "spatial"

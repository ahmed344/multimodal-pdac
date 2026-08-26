"""Fast end-to-end contracts for training checkpoints and Parquet inference."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import yaml

from ihc_mvn.infer import run_inference
from ihc_mvn.train import train_model


def tiny_pipeline_config(input_path: Path, output_dir: Path) -> dict[str, Any]:
    """Create a one-epoch CPU configuration for a synthetic AnnData file.

    Args:
        input_path (Path): Labeled synthetic training and inference H5AD.
        output_dir (Path): Temporary artifact directory.

    Returns:
        dict[str, Any]: Complete validated-style pipeline configuration.
    """

    config_path = Path(__file__).parents[1] / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = copy.deepcopy(yaml.safe_load(handle))
    config["data"]["path"] = str(input_path)
    config["data"]["inference_path"] = str(input_path)
    config["data"]["max_train_samples"] = None
    config["data"]["max_validation_samples"] = None
    config["data"]["max_test_samples"] = None
    config["preprocessing"]["fit_row_chunk_size"] = 8
    config["model"].update(
        {
            "num_peaks": 3,
            "embedding_dim": 4,
            "peak_hidden_dims": [],
            "peak_output_dim": 4,
            "aggregation_hidden_dims": [],
            "latent_dim": 3,
            "head_hidden_dims": [],
            "dropout": 0.0,
            "use_layer_norm": False,
        }
    )
    config["training"].update(
        {
            "device": "cpu",
            "deterministic_algorithms": True,
            "epochs": 1,
            "batch_size": 4,
            "validation_batch_size": 4,
            "test_batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "early_stopping_patience": 1,
        }
    )
    config["output"]["directory"] = str(output_dir)
    config["inference"].update(
        {
            "batch_size": 4,
            "num_workers": 0,
            "output_dir": str(output_dir / "inference"),
        }
    )
    return config


def test_one_epoch_checkpoint_and_ordered_inference(
    tiny_h5ad: Path,
    tmp_path: Path,
) -> None:
    """Train once, restore frozen preprocessing, and stream ordered Parquets.

    Args:
        tiny_h5ad (Path): Synthetic labeled CSR AnnData fixture.
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions validate checkpoint and inference artifact contracts.
    """

    output_dir = tmp_path / "model"
    checkpoint_path = train_model(tiny_pipeline_config(tiny_h5ad, output_dir))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["scaler_state"]["center"] is False
    assert checkpoint["scaler_state"]["transform"] == "log1p"
    assert checkpoint["scaler_state"]["feature_names"] == [
        "peak_0",
        "peak_1",
        "peak_2",
    ]
    prediction_path, latent_path = run_inference(
        checkpoint_path=checkpoint_path,
        input_path=tiny_h5ad,
        output_path=tmp_path / "predictions.parquet",
        latent_output_path=tmp_path / "latent.parquet",
        device_override="cpu",
        batch_size_override=4,
        num_workers_override=0,
        max_rows=6,
    )
    predictions = pq.read_table(prediction_path)
    latent = pq.read_table(latent_path)
    assert predictions.num_rows == 6
    assert latent.num_rows == 6
    assert predictions["row_position"].to_pylist() == list(range(6))
    assert latent["row_position"].to_pylist() == list(range(6))
    assert "prob_presence_Density_CD8" in predictions.column_names
    assert "conditional_cd8_excess" in predictions.column_names
    assert len([name for name in latent.column_names if name.startswith("latent_")]) == 3

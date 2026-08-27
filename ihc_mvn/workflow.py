"""Coordinate optional analysis, tissue inference, and spatial visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .analyze import run_analysis
from .infer import run_inference
from .spatial_heatmaps import run_spatial_heatmaps


def run_post_training_workflow(
    config: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, Path]:
    """Run configured post-training stages in dependency order.

    Args:
        config (Mapping[str, Any]): Effective training configuration.
        checkpoint_path (Path): Best-validation checkpoint from training.

    Returns:
        dict[str, Path]: Paths produced by each successful enabled stage.
    """

    settings = config["post_training"]
    if not bool(settings["enabled"]):
        return {}
    outputs: dict[str, Path] = {}

    def execute(label: str, operation: Any) -> Any:
        """Execute one stage with configured failure handling.

        Args:
            label (str): Human-readable stage name.
            operation (Any): Zero-argument callable implementing the stage.

        Returns:
            Any: Stage return value, or ``None`` after an ignored failure.
        """

        print(f"Post-training stage: {label}", flush=True)
        try:
            return operation()
        except Exception as error:
            print(f"Post-training stage {label!r} failed: {error}", flush=True)
            if bool(settings["fail_on_error"]):
                raise
            return None

    if bool(settings["run_analysis"]):
        analysis_output = execute(
            "held-out analysis",
            lambda: run_analysis(config, checkpoint_override=checkpoint_path),
        )
        if analysis_output is not None:
            outputs["analysis"] = Path(analysis_output)

    inference_dir = Path(str(config["inference"]["output_dir"]))
    prediction_path = inference_dir / "inference.parquet"
    latent_path = inference_dir / "inference_latent.parquet"
    if bool(settings["run_inference"]):
        inference_output = execute(
            "tissue inference",
            lambda: run_inference(
                checkpoint_path=checkpoint_path,
                output_path=prediction_path,
                latent_output_path=latent_path,
                max_rows=config["inference"].get("max_rows"),
                overwrite=bool(settings["overwrite_inference"]),
            ),
        )
        if inference_output is not None:
            outputs["predictions"], outputs["latents"] = (
                Path(inference_output[0]),
                Path(inference_output[1]),
            )

    if bool(settings["run_spatial_heatmaps"]):
        spatial_output = execute(
            "spatial heatmaps",
            lambda: run_spatial_heatmaps(
                config,
                predictions_path=prediction_path,
                latent_path=latent_path,
            ),
        )
        if spatial_output is not None:
            outputs["spatial"] = Path(spatial_output)
    return outputs


__all__ = ["run_post_training_workflow"]

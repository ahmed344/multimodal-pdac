"""Capped downstream checks for a staged preprocessing generation."""
from __future__ import annotations

import argparse
from pathlib import Path

from dann.analyze import run_analysis
from dann.config import apply_smoke_overrides, load_config
from dann.latent_variance import compute_latent_variance
from dann.spatial import run_inference

from .io import json_write


def main() -> None:
    """Exercise analysis/calibration, inference and latent consumers without full fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = apply_smoke_overrides(load_config(Path(args.config)))
    checkpoint = Path(config["training"]["output_dir"]) / "best.pt"
    output = run_analysis(config, checkpoint)
    predictions = output / "spatial" / "spatial_predictions.parquet"
    latent = predictions.with_name("spatial_latent.parquet")
    tissue = Path(config["data"]["context_path"])
    cap = int(config["training"]["smoke_test_samples"])
    run_inference(tissue, checkpoint, predictions, max_rows=cap, latent_output_path=latent,
                  num_workers_override=0, overwrite=True, density_mc_samples=32)
    summary, spectrum = compute_latent_variance(latent, tissue, config["data"]["batch_column"])
    json_write(output / "capped_latent_variance.json", summary)
    spectrum.to_csv(output / "capped_latent_spectrum.csv", index=False)


if __name__ == "__main__":
    main()

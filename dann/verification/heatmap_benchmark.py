"""Reproduce heatmap worker screening, baseline timing, and full-size validation.

Run from the repository root. Generated inputs, PNGs, and timing records belong
under the gitignored data directory. The baseline is an exact pre-change copy of
spatial_heatmaps.py supplied explicitly with --baseline-script.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import h5py
import numpy as np

from dann import heatmap_execution as execution
from dann.spatial_heatmaps import DEFAULT_DENSITIES, run_spatial_heatmaps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("screen", "baseline", "full", "compare"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-script", type=Path)
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument("--optimized-output", type=Path)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--regions", nargs="+", default=["tissue", "lesion"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--slides", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    if args.mode == "compare":
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        baseline_files = sorted(args.baseline_output.glob("spatial_heatmap_*.png"))
        expected = {f"spatial_heatmap_{prefix}{density}.png"
                    for density in DEFAULT_DENSITIES for prefix in ("", "diagnostics_")}
        if {path.name for path in baseline_files} != expected:
            raise ValueError("The baseline must contain all eight expected tissue figures.")
        for old in baseline_files:
            new = args.optimized_output / (old.stem + "_tissue.png")
            with Image.open(old) as left, Image.open(new) as right:
                if left.size != right.size:
                    raise ValueError(f"Dimension mismatch: {old.name}")
                # Decode once, compare modest strips to bound comparison scratch.
                left.load()
                right.load()
                differing = 0
                maximum = 0
                for y in range(0, left.height, 64):
                    box = (0, y, left.width, min(y + 64, left.height))
                    a, b = np.asarray(left.crop(box)), np.asarray(right.crop(box))
                    differing += int(np.any(a != b, axis=-1).sum())
                    maximum = max(maximum, int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()))
                record = dict(file=old.name, differing_pixels=differing, maximum_channel_difference=maximum)
                records.append(record)
                print(record, flush=True)
        (args.output_dir / "comparison.json").write_text(json.dumps(records, indent=2))
        if any(record["differing_pixels"] for record in records):
            raise SystemExit("Rendered tissue images differ; inspect comparison.json")
        return
    if args.mode == "baseline":
        if args.baseline_script is None:
            parser.error("baseline requires --baseline-script")
        spec = importlib.util.spec_from_file_location("heatmap_baseline", args.baseline_script)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        progress = execution.Progress()
        progress.start()
        try:
            baseline.run_spatial_heatmaps(args.input, args.predictions, args.output_dir,
                                          DEFAULT_DENSITIES, 1e-4, args.dpi)
            for path in args.output_dir.glob("*.png"):
                execution.validate_png(path)
        finally:
            progress.close()
        record = dict(total_seconds=time.perf_counter() - progress.started,
                      peak_parent_worker_rss_gib=progress.peak_rss / execution.GIB,
                      input=execution.source_identity(args.input), predictions=execution.source_identity(args.predictions),
                      baseline=execution.source_identity(args.baseline_script), dpi=args.dpi)
        (args.output_dir / "baseline_run.json").write_text(json.dumps(record, indent=2))
        return
    if args.mode == "screen":
        # Include the largest slide and spread the rest across observed slide order.
        from anndata.io import read_elem
        with h5py.File(args.input, "r") as handle:
            batches = read_elem(handle["obs/batch"])
        import pandas as pd
        ordered = pd.Series(batches).unique().tolist()
        largest = pd.Series(batches).value_counts().index[0]
        candidates = [largest, *[ordered[i] for i in np.linspace(0, len(ordered) - 1, args.slides, dtype=int)]]
        selected = list(dict.fromkeys(candidates))[:args.slides]
        original = execution.load_plot_data
        def subset(*arguments):
            data = original(*arguments)
            data = data[data.obs.batch.isin(selected)].copy()
            data.uns["spatial"] = {batch: data.uns["spatial"][batch] for batch in selected}
            return data
        execution.load_plot_data = subset
        (args.output_dir / "manifest.json").write_text(json.dumps(dict(
            selected_slides=selected, input=execution.source_identity(args.input),
            predictions=execution.source_identity(args.predictions), dpi=args.dpi,
            repetitions=args.repetitions, workers=args.workers, regions=args.regions), indent=2))
    else:
        args.repetitions = 1
    for workers in args.workers:
        repetitions = range(-1, args.repetitions) if args.mode == "screen" else range(args.repetitions)
        for repetition in repetitions:
            destination = args.output_dir / f"workers-{workers}" / ("warmup" if repetition == -1 else f"repeat-{repetition}")
            run_spatial_heatmaps(args.input, args.predictions, destination, DEFAULT_DENSITIES,
                                 1e-4, args.dpi, workers=workers, regions=args.regions)
            record = json.loads((destination / "heatmap_run.json").read_text())
            record.update(requested_workers=workers, repetition=repetition)
            records.append(record)
            (args.output_dir / "timings.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

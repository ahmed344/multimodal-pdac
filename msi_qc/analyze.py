"""Run read-only MSI QC: python -m msi_qc.analyze --config msi_qc/config.yaml.

Use --resume to reuse verified per-slide intermediate results after interruption.
The source H5AD is never opened for writing. This module has no DANN/IHC imports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from .config import load_config
from .core import (adjacent_pairs, iter_blocks, read_element, robust_limits,
                   row_sums, select_threshold, spatial_diagnostics, threshold_histograms)


def log(message: str) -> None:
    """Emit flushed progress for long-running local analyses."""
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}", flush=True)


def source_identity(path: Path) -> dict[str, Any]:
    """Record source identity without rereading a large source just to hash it."""
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_json(path: Path, value: Any) -> None:
    """Write strict JSON, mapping undefined numerical summaries to null."""
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (tuple, list, np.ndarray)):
            return [clean(v) for v in item]
        if isinstance(item, (np.integer, np.bool_)):
            return item.item()
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        return item
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def inspect_input(handle: h5py.File, layer: str) -> tuple[pd.DataFrame, np.ndarray, h5py.Group]:
    """Validate source metadata and obtain only fields used by QC."""
    key = f"layers/{layer}"
    if key not in handle:
        raise ValueError(f"Missing required original layer: {key}")
    group = handle[key]
    if group.attrs.get("encoding-type") != "csr_matrix":
        raise ValueError("The selected layer must be CSR encoded.")
    shape = tuple(group.attrs["shape"])
    if (len(group["indptr"]) != shape[0] + 1 or int(group["indptr"][0]) != 0
            or int(group["indptr"][-1]) != len(group["data"])
            or len(group["data"]) != len(group["indices"])):
        raise ValueError("CSR dimensions do not match the stored arrays.")
    columns = ("batch", "patient", "x", "y", "Lesion")
    for name in columns:
        if name not in handle["obs"]:
            raise ValueError(f"Missing required obs column {name}")
    obs = pd.DataFrame({name: read_element(handle["obs"][name]) for name in columns})
    if obs.isna().any().any():
        raise ValueError("Required metadata must not contain missing values.")
    if obs["Lesion"].dtype != bool:
        raise ValueError("Lesion annotations must be boolean; they are not a background mask.")
    features = read_element(handle["var"][handle["var"].attrs["_index"]])
    if shape != (len(obs), len(features)) or any(v is None for v in features):
        raise ValueError("Feature and observation metadata must match the matrix.")
    if len(set(features)) != len(features):
        raise ValueError("Feature identities must be unique.")
    masses = np.asarray(features, dtype=float)
    if not np.isfinite(masses).all():
        raise ValueError("Peak names must be finite m/z values.")
    return obs, features.astype(str), group


def analyze_slide(handle: h5py.File, group: h5py.Group, obs: pd.DataFrame, rows: np.ndarray,
                  slide_index: int, config: dict, cache: Path) -> dict[str, Any]:
    """Stream one slide, retaining only sampled spectra for spatial diagnostics."""
    started = time.monotonic()
    frame = obs.iloc[rows]
    slide = str(frame["batch"].iloc[0])
    if frame["patient"].nunique() != 1:
        raise ValueError(f"Slide {slide} maps to multiple patients.")
    n, p = len(rows), int(group.attrs["shape"][1])
    thresholds = np.asarray(config["thresholds"], dtype=float)
    nt = len(thresholds)
    pair_rng = np.random.default_rng(np.random.SeedSequence([config["seed"], slide_index, 0]))
    sample_rng = np.random.default_rng(np.random.SeedSequence([config["seed"], slide_index, 1]))
    shuffle_rng = np.random.default_rng(np.random.SeedSequence([config["seed"], slide_index, 2]))
    pairs = adjacent_pairs(frame.x.to_numpy(), frame.y.to_numpy(), config["spatial_pairs"], pair_rng)
    spatial_rows = np.unique(pairs)
    mapped_pairs = np.searchsorted(spatial_rows, pairs)
    spatial_parts = []
    count_hist = np.zeros((p, nt + 1), dtype=np.int64)
    sum_hist = np.zeros_like(count_hist, dtype=np.float64)
    region_counts = np.zeros((2, nt + 1), dtype=np.int64)
    region_sums = np.zeros_like(region_counts, dtype=np.float64)
    peak_sum_squares = np.zeros(p)
    peak_max = np.zeros(p)
    tic = np.zeros(n)
    detected = np.zeros(n, dtype=np.int32)
    maximum = np.zeros(n)
    weak_fraction = np.zeros(n)
    samples = []
    lesion = frame.Lesion.to_numpy()
    stored_nnz = 0
    # Reading two pointer values for each row stays small compared with spectra.
    slide_ptr_lo = read_element(group["indptr"], rows)
    slide_ptr_hi = read_element(group["indptr"], rows + 1)
    estimated_nnz = int(np.sum(slide_ptr_hi - slide_ptr_lo))
    sample_probability = min(1., config["sample_values_per_slide"] / max(estimated_nnz, 1))
    for start, stop, matrix in iter_blocks(group, rows, config["block_rows"]):
        stored_nnz += int(np.sum(slide_ptr_hi[start:stop] - slide_ptr_lo[start:stop]))
        counts, sums = threshold_histograms(matrix, thresholds)
        count_hist += counts
        sum_hist += sums
        lengths = np.diff(matrix.indptr)
        detected[start:stop] = lengths
        tic[start:stop] = np.asarray(matrix.sum(axis=1)).ravel()
        maximum[start:stop] = matrix.max(axis=1).toarray().ravel()
        weak = row_sums(matrix.data <= config["weak_intensity_screen"], matrix.indptr)
        weak_fraction[start:stop] = np.divide(weak, lengths, out=np.zeros(len(lengths)), where=lengths > 0)
        peak_sum_squares += np.bincount(matrix.indices, weights=matrix.data ** 2, minlength=p)
        np.maximum.at(peak_max, matrix.indices, matrix.data)
        bins = np.searchsorted(thresholds, matrix.data, side="left")
        regions = np.repeat(lesion[start:stop], lengths).astype(np.int64)
        flat = regions * (nt + 1) + bins
        region_counts += np.bincount(flat, minlength=2 * (nt + 1)).reshape(2, nt + 1)
        region_sums += np.bincount(flat, weights=matrix.data, minlength=2 * (nt + 1)).reshape(2, nt + 1)
        selected = spatial_rows[(spatial_rows >= start) & (spatial_rows < stop)] - start
        if len(selected):
            spatial_parts.append(matrix[selected])
        sample_n = int(sample_rng.binomial(matrix.nnz, sample_probability))
        if sample_n:
            samples.append(matrix.data[sample_rng.choice(matrix.nnz, sample_n, replace=False)].astype(np.float32))
    sampled_matrix = (sparse.vstack(spatial_parts, format="csr") if spatial_parts
                      else sparse.csr_matrix((0, p), dtype=float))
    del spatial_parts
    log(f"{slide}: exact summaries complete; spatial controls on {len(pairs):,} pairs")
    spatial = spatial_diagnostics(sampled_matrix, mapped_pairs, tic[spatial_rows], config, shuffle_rng)
    del sampled_matrix
    count = count_hist.sum(axis=1)
    intensity_sum = sum_hist.sum(axis=1)
    losses = count_hist.cumsum(axis=1)[:, :nt]
    signal_losses = sum_hist.cumsum(axis=1)[:, :nt]
    weak_index = list(thresholds).index(config["weak_intensity_screen"])
    peak_weak_fraction = np.divide(losses[:, weak_index], count, out=np.zeros(p), where=count > 0)
    weak_unstructured = (spatial["spatial_tested"]
                         & (peak_weak_fraction >= config["weak_fraction_screen"])
                         & (spatial["raw_excess"] <= config["spatial_weak_excess"])
                         & (spatial["normalized_excess"] <= config["spatial_weak_excess"]))
    limits_tic = robust_limits(tic, config["mad_multiplier"])
    limits_detected = robust_limits(detected, config["mad_multiplier"])
    pixel = pd.DataFrame({"row_position": rows, "slide": slide,
                          "patient": str(frame.patient.iloc[0]), "x": frame.x.to_numpy(),
                          "y": frame.y.to_numpy(), "lesion": lesion, "summed_peak_intensity": tic,
                          "detected_peaks": detected, "maximum_intensity": maximum,
                          "weak_measurement_fraction": weak_fraction,
                          "zero_signal": tic == 0,
                          "low_signal": tic < limits_tic["lower"],
                          "low_peak_count": detected < limits_detected["lower"],
                          "high_signal": tic > limits_tic["upper"],
                          "high_peak_count": detected > limits_detected["upper"]})
    pixel["low_quality_review"] = pixel[["zero_signal", "low_signal", "low_peak_count"]].any(axis=1)
    index_element = handle["obs"][handle["obs"].attrs["_index"]]
    pixel.insert(1, "obs_name", read_element(index_element, rows))
    stem = cache / f"slide_{slide_index:02d}"
    pixel.to_parquet(f"{stem}_pixels.parquet", index=False)
    sample = np.concatenate(samples) if samples else np.array([], dtype=np.float32)
    np.savez_compressed(f"{stem}.npz", count=count, intensity_sum=intensity_sum,
                        sum_squares=peak_sum_squares, maximum=peak_max,
                        count_loss=losses, signal_loss=signal_losses,
                        region_counts=region_counts, region_sums=region_sums,
                        peak_weak_fraction=peak_weak_fraction,
                        weak_unstructured=weak_unstructured, intensity_sample=sample, **spatial)
    threshold_rows = []
    for region, region_mask in (("all", np.ones(n, bool)), ("lesion", lesion), ("non_lesion", ~lesion)):
        total_count = int(count.sum()) if region == "all" else int(region_counts[int(region == "lesion")].sum())
        total_sum = float(intensity_sum.sum()) if region == "all" else float(region_sums[int(region == "lesion")].sum())
        removed_counts = losses.sum(axis=0) if region == "all" else region_counts[int(region == "lesion")].cumsum()[:nt]
        removed_sums = signal_losses.sum(axis=0) if region == "all" else region_sums[int(region == "lesion")].cumsum()[:nt]
        for j, threshold in enumerate(thresholds):
            threshold_rows.append({"slide": slide, "region": region, "threshold": threshold,
                                   "pixels": int(region_mask.sum()), "positive_measurements": total_count,
                                   "summed_peak_intensity": total_sum,
                                   "measurements_removed": int(removed_counts[j]),
                                   "intensity_removed": float(removed_sums[j]),
                                   "measurement_loss_fraction": removed_counts[j] / max(total_count, 1),
                                   "intensity_loss_fraction": removed_sums[j] / total_sum if total_sum else 0.,
                                   "pixels_empty_after": int(np.sum((maximum <= threshold) & region_mask)),
                                   "newly_empty_pixels": int(np.sum((maximum > 0) & (maximum <= threshold) & region_mask)),
                                   "peaks_lost": int(np.sum((count > 0) & (count == losses[:, j]))) if region == "all" else None})
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(f"{stem}_thresholds.csv", index=False)
    summary = {"slide_index": slide_index, "slide": slide, "patient": str(frame.patient.iloc[0]),
               "pixels": n, "lesion_pixels": int(lesion.sum()), "stored_entries": stored_nnz,
               "positive_measurements": int(count.sum()), "summed_peak_intensity": float(intensity_sum.sum()),
               "present_peaks": int(np.sum(count > 0)),
               "median_summed_peak_intensity": float(np.median(tic)),
               "median_detected_peaks": float(np.median(detected)),
               "sample_positive_quantiles": dict(zip(("q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99"),
                    np.quantile(sample, [.01, .05, .1, .25, .5, .75, .9, .95, .99]).tolist())) if len(sample) else {},
               "sample_positive_values": len(sample), "sampled_spatial_pixels": len(spatial_rows),
               "sampled_spatial_pairs": len(pairs),
               "spatial_tested_peaks": int(spatial["spatial_tested"].sum()),
               "spatial_supported_peaks": int(spatial["spatial_supported"].sum()),
               "weak_unstructured_peaks": int(weak_unstructured.sum()),
               "pixel_signal_limits": limits_tic, "pixel_peak_count_limits": limits_detected,
               "pixel_flags": {key: int(pixel[key].sum()) for key in
                               ("zero_signal", "low_signal", "low_peak_count", "high_signal", "high_peak_count", "low_quality_review")},
               "regions": {name: {"pixels": int(mask.sum()),
                                  "median_summed_peak_intensity": float(np.median(tic[mask])) if mask.any() else None,
                                  "median_detected_peaks": float(np.median(detected[mask])) if mask.any() else None}
                           for name, mask in (("lesion", lesion), ("non_lesion", ~lesion))},
               "elapsed_seconds": time.monotonic() - started}
    write_json(Path(f"{stem}.json"), summary)
    log(f"{slide}: {count.sum():,} measurements; {summary['spatial_supported_peaks']:,} spatially supported peaks; {summary['elapsed_seconds']:.1f}s")
    return summary


def aggregate_results(output: Path, config: dict, features: np.ndarray,
                      summaries: list[dict]) -> dict[str, Any]:
    """Reconcile exact summaries, export tables, and select a conservative threshold."""
    cache = output / "cache"
    p = len(features)
    thresholds = np.asarray(config["thresholds"], dtype=float)
    peak_frames, threshold_frames, losses = [], [], []
    total_count, total_signal = np.zeros(p, np.int64), np.zeros(p)
    total_squares, peak_max = np.zeros(p), np.zeros(p)
    tested, supported, candidates = np.zeros(p, int), np.zeros(p, int), np.zeros(p, int)
    weak_count, weak_sum = np.zeros(p, np.int64), np.zeros(p)
    present_slides, max_prevalence = np.zeros(p, int), np.zeros(p)
    spatial_supported_loss = np.zeros((len(thresholds), 2))
    spatial_supported_denominator = np.zeros(2)
    pixel_writer = None
    try:
        for summary in summaries:
            stem = cache / f"slide_{summary['slide_index']:02d}"
            arrays = np.load(f"{stem}.npz")
            count, signal = arrays["count"], arrays["intensity_sum"]
            total_count += count
            total_signal += signal
            total_squares += arrays["sum_squares"]
            peak_max = np.maximum(peak_max, arrays["maximum"])
            tested += arrays["spatial_tested"]
            supported += arrays["spatial_supported"]
            candidates += arrays["weak_unstructured"]
            present_slides += count > 0
            prevalence = count / summary["pixels"]
            max_prevalence = np.maximum(max_prevalence, prevalence)
            wi = list(thresholds).index(config["weak_intensity_screen"])
            weak_count += arrays["count_loss"][:, wi]
            weak_sum += arrays["signal_loss"][:, wi]
            selected = arrays["spatial_supported"]
            spatial_supported_loss[:, 0] += arrays["count_loss"][selected].sum(axis=0)
            spatial_supported_loss[:, 1] += arrays["signal_loss"][selected].sum(axis=0)
            spatial_supported_denominator += [count[selected].sum(), signal[selected].sum()]
            spatial_keys = [k for k in arrays.files if k.startswith(("raw_", "normalized_", "binary_", "spatial_"))]
            data = {"feature_index": np.arange(p), "mz": features, "slide": summary["slide"],
                    "pixels": summary["pixels"], "detections": count, "prevalence": prevalence,
                    "intensity_sum": signal,
                    "mean_positive_intensity": np.divide(signal, count, out=np.full(p, np.nan), where=count > 0),
                    "maximum_intensity": arrays["maximum"], "weak_measurement_fraction": arrays["peak_weak_fraction"],
                    "weak_unstructured_review": arrays["weak_unstructured"]}
            data.update({k: arrays[k] for k in spatial_keys})
            peak_frames.append(pd.DataFrame(data))
            t = pd.read_csv(f"{stem}_thresholds.csv")
            threshold_frames.append(t)
            a = t[t.region == "all"]
            losses.append((a.measurement_loss_fraction.to_numpy(), a.intensity_loss_fraction.to_numpy()))
            pixel_table = pq.read_table(f"{stem}_pixels.parquet")
            if pixel_writer is None:
                pixel_writer = pq.ParquetWriter(output / "pixel_qc.parquet", pixel_table.schema, compression="zstd")
            pixel_writer.write_table(pixel_table)
            arrays.close()
    finally:
        if pixel_writer is not None:
            pixel_writer.close()
    threshold_frame = pd.concat(threshold_frames, ignore_index=True)
    all_thresholds = threshold_frame[threshold_frame.region == "all"]
    columns = ["pixels", "positive_measurements", "summed_peak_intensity", "measurements_removed",
               "intensity_removed", "pixels_empty_after", "newly_empty_pixels"]
    global_thresholds = all_thresholds.groupby("threshold", sort=True)[columns].sum().reset_index()
    global_thresholds["measurement_loss_fraction"] = global_thresholds.measurements_removed / max(total_count.sum(), 1)
    global_thresholds["intensity_loss_fraction"] = global_thresholds.intensity_removed / max(total_signal.sum(), 1e-300)
    global_thresholds["worst_slide_measurement_loss_fraction"] = all_thresholds.groupby("threshold").measurement_loss_fraction.max().to_numpy()
    global_thresholds["worst_slide_intensity_loss_fraction"] = all_thresholds.groupby("threshold").intensity_loss_fraction.max().to_numpy()
    global_thresholds["spatial_supported_measurement_loss_fraction"] = spatial_supported_loss[:, 0] / max(spatial_supported_denominator[0], 1)
    global_thresholds["spatial_supported_intensity_loss_fraction"] = spatial_supported_loss[:, 1] / max(spatial_supported_denominator[1], 1e-300)
    cutoff = select_threshold(losses, thresholds, config["retained_signal_min"], config["retained_measurements_min"])
    global_thresholds["conservative_policy_eligible"] = ((global_thresholds.worst_slide_measurement_loss_fraction <= 1 - config["retained_measurements_min"] + 1e-12)
                                                        & (global_thresholds.worst_slide_intensity_loss_fraction <= 1 - config["retained_signal_min"] + 1e-12))
    n = sum(s["pixels"] for s in summaries)
    mean = np.divide(total_signal, total_count, out=np.full(p, np.nan), where=total_count > 0)
    positive_variance = np.maximum(np.divide(total_squares, total_count, out=np.zeros(p), where=total_count > 0) - mean ** 2, 0)
    # Review requires consistent evidence and no contrary spatial support on any slide.
    review = (candidates >= np.maximum(2, np.ceil(.5 * tested))) & (supported == 0)
    peak_table = pd.DataFrame({"feature_index": np.arange(p), "mz": features,
                              "detections": total_count, "prevalence": total_count / n,
                              "intensity_sum": total_signal, "mean_positive_intensity": mean,
                              "sd_positive_intensity": np.sqrt(positive_variance),
                              "maximum_intensity": peak_max, "present_slides": present_slides,
                              "maximum_slide_prevalence": max_prevalence,
                              "spatial_tested_slides": tested, "spatial_supported_slides": supported,
                              "weak_unstructured_slides": candidates,
                              "weak_measurement_fraction": np.divide(weak_count, total_count, out=np.zeros(p), where=total_count > 0),
                              "weak_intensity_fraction": np.divide(weak_sum, total_signal, out=np.zeros(p), where=total_signal > 0),
                              "whole_peak_review": review,
                              "absent_everywhere": total_count == 0,
                              "insufficient_spatial_evidence": tested == 0})
    for screen in config["prevalence_screens"]:
        peak_table[f"below_{100 * screen:g}pct_on_every_slide"] = max_prevalence < screen
    peak_table.to_csv(output / "peak_statistics.csv", index=False)
    peak_slides = pd.concat(peak_frames, ignore_index=True)
    peak_slides.to_csv(output / "peak_slide_statistics.csv", index=False)
    peak_slides.to_parquet(output / "peak_slide_statistics.parquet", index=False)
    threshold_frame.to_csv(output / "thresholds_by_slide_and_region.csv", index=False)
    global_thresholds.to_csv(output / "threshold_tradeoffs.csv", index=False)
    flat_summaries = []
    for s in summaries:
        flat = {k: v for k, v in s.items() if not isinstance(v, dict)}
        for prefix in ("sample_positive_quantiles", "pixel_signal_limits", "pixel_peak_count_limits", "pixel_flags"):
            flat.update({f"{prefix}_{k}": v for k, v in s[prefix].items()})
        flat_summaries.append(flat)
    pd.DataFrame(flat_summaries).to_csv(output / "slide_statistics.csv", index=False)
    if total_count.sum() != sum(s["positive_measurements"] for s in summaries):
        raise AssertionError("Global detection counts do not reconcile.")
    if not np.isclose(total_signal.sum(), sum(s["summed_peak_intensity"] for s in summaries), rtol=1e-10):
        raise AssertionError("Global signal sums do not reconcile.")
    if (np.any(np.diff(global_thresholds.measurements_removed) < 0)
            or np.any(np.diff(global_thresholds.intensity_removed) < -1e-6)):
        raise AssertionError("Threshold removal curves must be monotone.")
    recommendation = {"conservative_raw_cutoff": cutoff,
                      "retention_policy": {"minimum_signal_fraction_each_slide": config["retained_signal_min"],
                                           "minimum_measurement_fraction_each_slide": config["retained_measurements_min"]},
                      "cutoff_semantics": "Keep original counts-layer intensities strictly greater than the cutoff; this is a retention policy, not an instrumental noise boundary.",
                      "whole_peak_review_count": int(review.sum()),
                      "whole_peak_review_measurement_fraction": float(total_count[review].sum() / max(total_count.sum(), 1)),
                      "whole_peak_review_signal_fraction": float(total_signal[review].sum() / max(total_signal.sum(), 1e-300)),
                      "spatial_supported_peak_count": int((supported > 0).sum()),
                      "insufficient_spatial_evidence_peak_count": int((tested == 0).sum()),
                      "absent_peak_count": int((total_count == 0).sum()),
                      "low_quality_pixel_review_count": sum(s["pixel_flags"]["low_quality_review"] for s in summaries),
                      "zero_signal_pixels": sum(s["pixel_flags"]["zero_signal"] for s in summaries),
                      "true_noise_fraction": None,
                      "automatic_filtering_applied": False,
                      "spatial_reference": "20 within-slide endpoint shuffles (configured); q95 envelopes are descriptive, not p-values. Strong support requires raw and normalized excess correlation >= configured support cutoff.",
                      "whole_peak_review_rule": "At least 2 and at least half of adequately sampled slides meet the weak/unstructured screen; no slide meets spatial support. Weak/unstructured means >= configured weak fraction below the weak intensity screen and both excess correlations <= configured weak excess. This is a review screen, not proof of noise.",
                      "prevalence_screen_counts": {str(t): int((max_prevalence < t).sum()) for t in config["prevalence_screens"]}}
    write_json(output / "recommendations.json", recommendation)
    write_json(output / "slide_details.json", summaries)
    return recommendation


def run(config: dict, resume: bool = False, render: bool = True) -> Path:
    """Run the complete reproducible QC workflow and return the report path."""
    from .report import render_report
    source, output = Path(config["input"]), Path(config["output"])
    identity = source_identity(source)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    implementation = hashlib.sha256(b"".join((Path(__file__).parent / f).read_bytes()
                                              for f in ("analyze.py", "core.py", "config.py"))).hexdigest()
    signature = {"input": identity, "config": config, "implementation_sha256": implementation}
    cache_manifest = output / "cache_manifest.json"
    if resume and cache_manifest.exists() and json.loads(cache_manifest.read_text()) != signature:
        raise ValueError("Resume cache does not match input, code, or configuration. Use a new output directory or rerun without --resume.")
    write_json(cache_manifest, signature)
    write_json(output / "run_manifest.json", {**signature, "status": "running",
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "versions": {name: importlib.metadata.version(name) for name in
                            ("numpy", "scipy", "h5py", "pandas", "pyarrow", "matplotlib")}})
    summaries = []
    with h5py.File(source, "r") as handle:
        obs, features, group = inspect_input(handle, config["layer"])
        slides = list(pd.unique(obs.batch))
        log(f"Source: {len(obs):,} pixels, {len(features):,} peaks, {len(slides)} slides; {len(group['data']):,} stored measurements")
        for i, slide in enumerate(slides):
            stem = cache / f"slide_{i:02d}"
            complete = all(Path(f"{stem}{suffix}").exists() for suffix in
                           (".json", ".npz", "_pixels.parquet", "_thresholds.csv"))
            if resume and complete:
                log(f"Reusing verified slide {i + 1}/{len(slides)}: {slide}")
                summaries.append(json.loads(Path(f"{stem}.json").read_text()))
            else:
                log(f"Analyzing slide {i + 1}/{len(slides)}: {slide}")
                rows = np.flatnonzero(obs.batch.to_numpy() == slide)
                summaries.append(analyze_slide(handle, group, obs, rows, i, config, cache))
    log("Reconciling full-file summaries and threshold budgets")
    recommendations = aggregate_results(output, config, features, summaries)
    if render:
        log("Rendering report, spatial examples, and standalone figures")
        render_report(output, config, features, summaries, recommendations)
    if source_identity(source) != identity:
        raise RuntimeError("Input identity changed during QC; outputs cannot be treated as a consistent source snapshot.")
    manifest = json.loads((output / "run_manifest.json").read_text())
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                    input_unchanged=True, observations=len(obs), features=len(features),
                    slides=len(slides), patients=int(obs.patient.nunique()),
                    output_files=sorted(str(p.relative_to(output)) for p in output.rglob("*")
                                        if p.is_file() and "cache" not in p.parts))
    write_json(output / "run_manifest.json", manifest)
    log(f"Complete. Conservative raw cutoff: {recommendations['conservative_raw_cutoff']:g}; report: {output / 'report.html'}")
    return output / "report.html"


def main() -> None:
    """Parse the standalone analysis CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="msi_qc/config.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()

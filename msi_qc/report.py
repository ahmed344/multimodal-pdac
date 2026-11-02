"""Publication-exportable QC figures and a local HTML analysis report."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd

from .core import iter_blocks, read_element


TEAL = "#137f83"
NAVY = "#233b53"
ORANGE = "#c9752b"
RED = "#b8464d"


def save_figure(fig: plt.Figure, directory: Path, name: str) -> None:
    """Save the same scientific figure as a PNG and standalone PDF."""
    fig.savefig(directory / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def image_grid(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """Rasterize original acquisition coordinates, leaving absent pixels masked."""
    x, y = frame.x.to_numpy(dtype=int), frame.y.to_numpy(dtype=int)
    x, y = x - x.min(), y - y.min()
    grid = np.full((int(y.max()) + 1, int(x.max()) + 1), np.nan)
    grid[y, x] = values
    return grid


def plot_distributions(output: Path, config: dict, summaries: list[dict], cutoff: float) -> None:
    """Plot weighted positive-intensity distributions and slide-level QC summaries."""
    figures = output / "figures"
    samples = []
    for s in summaries:
        with np.load(output / "cache" / f"slide_{s['slide_index']:02d}.npz") as a:
            samples.append(a["intensity_sample"])
    nonempty = [v for v in samples if len(v)]
    if not nonempty:
        return
    low = min(float(v.min()) for v in nonempty)
    high = max(float(v.max()) for v in nonempty)
    edges = np.geomspace(max(low, 1e-12), max(high, low * 1.1), 180)
    mids = np.sqrt(edges[1:] * edges[:-1])
    pooled = np.zeros(len(edges) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for s, values in zip(summaries, samples):
        if not len(values):
            continue
        histogram = np.histogram(values, edges)[0] / len(values)
        pooled += histogram * s["positive_measurements"]
        axes[1].plot(edges[1:], np.cumsum(histogram) * 100, color=TEAL, alpha=.18, lw=.8)
    pooled /= max(sum(s["positive_measurements"] for s in summaries), 1)
    axes[0].fill_between(mids, pooled * 100, color=TEAL, alpha=.75)
    axes[0].set(ylabel="Positive measurements per logarithmic bin (%)",
                title="Positive intensity distribution\nSlide-stratified sample, weighted by measurement count")
    axes[1].plot(edges[1:], np.cumsum(pooled) * 100, color=NAVY, lw=2.3, label="Weighted pooled")
    axes[1].set(ylabel="Positive measurements at or below intensity (%)",
                title="Cumulative distributions\nThin lines: individual slides", ylim=(0, 100))
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Original counts-layer intensity (fractional units)")
        if cutoff > 0:
            ax.axvline(cutoff, color=RED, ls="--", lw=1.5, label=f"Conservative cutoff {cutoff:g}")
        ax.grid(alpha=.15)
    axes[1].legend(fontsize=9)
    save_figure(fig, figures, "intensity_distributions")
    ordered = sorted(summaries, key=lambda s: s["sample_positive_quantiles"].get("q50", 0))
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 3, figsize=(14, max(7, .30 * len(ordered))), sharey=True, constrained_layout=True)
    q10 = [s["sample_positive_quantiles"].get("q10", np.nan) for s in ordered]
    q50 = [s["sample_positive_quantiles"].get("q50", np.nan) for s in ordered]
    q90 = [s["sample_positive_quantiles"].get("q90", np.nan) for s in ordered]
    axes[0].hlines(y, q10, q90, color=TEAL, alpha=.6, lw=2)
    axes[0].scatter(q50, y, color=NAVY, s=22)
    axes[0].set_xscale("log")
    axes[0].set(title="Positive intensity: median and 10–90%", xlabel="Raw intensity")
    axes[1].barh(y, [s["median_summed_peak_intensity"] for s in ordered], color=TEAL)
    axes[1].set(title="Median summed intensity per pixel", xlabel="Sum over retained m/z features")
    axes[2].barh(y, [s["median_detected_peaks"] for s in ordered], color=NAVY)
    axes[2].set(title="Median detected peaks per pixel", xlabel="Distinct peaks with intensity > 0")
    axes[0].set_yticks(y, [s["slide"] for s in ordered], fontsize=8)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=.15)
    save_figure(fig, figures, "slide_quality")


def plot_thresholds(output: Path, cutoff: float) -> None:
    """Show exact signal-loss and measurement-loss curves and supported-peak costs."""
    table = pd.read_csv(output / "threshold_tradeoffs.csv")
    t = table.threshold
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, field, ylabel in ((axes[0, 0], "measurement", "Positive measurements removed (%)"),
                              (axes[0, 1], "intensity", "Summed peak intensity removed (%)")):
        ax.plot(t, 100 * table[f"{field}_loss_fraction"], "o-", color=TEAL, label="All slides pooled")
        ax.plot(t, 100 * table[f"worst_slide_{field}_loss_fraction"], "s--", color=ORANGE, label="Worst affected slide")
        ax.set(xlabel="Raw intensity cutoff (remove values ≤ cutoff)", ylabel=ylabel)
        ax.set_xscale("symlog", linthresh=.01)
        ax.axvline(cutoff, color=RED, ls=":", label=f"Recommended {cutoff:g}")
        ax.legend(fontsize=9)
    axes[0, 0].set_title("How many measurements would be removed?")
    axes[0, 1].set_title("How much measured signal would be lost?")
    axes[1, 0].plot(100 * table.measurement_loss_fraction, 100 * table.intensity_loss_fraction, "o-", color=NAVY)
    for _, row in table.iterrows():
        if row.threshold >= .05:
            axes[1, 0].annotate(f"{row.threshold:g}", (100 * row.measurement_loss_fraction, 100 * row.intensity_loss_fraction),
                                 xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1, 0].set(xlabel="Positive measurements removed (%)", ylabel="Summed peak intensity lost (%)",
                    title="Removal trade-off (labels are raw cutoffs)")
    axes[1, 1].plot(t, 100 * table.spatial_supported_measurement_loss_fraction, "o-", color=TEAL, label="Measurements")
    axes[1, 1].plot(t, 100 * table.spatial_supported_intensity_loss_fraction, "s-", color=ORANGE, label="Summed intensity")
    axes[1, 1].set_xscale("symlog", linthresh=.01)
    axes[1, 1].axvline(cutoff, color=RED, ls=":")
    axes[1, 1].set(xlabel="Raw intensity cutoff", ylabel="Loss within supported peak/slide combinations (%)",
                    title="Filtering also removes spatially organized signals")
    axes[1, 1].legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=.15)
    save_figure(fig, output / "figures", "threshold_tradeoffs")


def plot_peak_landscape(output: Path) -> None:
    """Show abundance, prevalence, and concordance without labeling rarity as noise."""
    peaks = pd.read_csv(output / "peak_statistics.csv")
    slides = pd.read_parquet(output / "peak_slide_statistics.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    valid = peaks.detections > 0
    points = axes[0, 0].scatter(100 * peaks.loc[valid, "prevalence"], peaks.loc[valid, "mean_positive_intensity"],
                              c=peaks.loc[valid, "spatial_supported_slides"], cmap="viridis", s=9, alpha=.65, rasterized=True)
    axes[0, 0].set(xscale="log", yscale="log", xlabel="Pixels detecting peak (%)", ylabel="Mean positive raw intensity",
                    title="Weak and rare do not necessarily mean noise")
    fig.colorbar(points, ax=axes[0, 0], label="Slides with spatial support")
    axes[0, 1].scatter(peaks.mz, peaks.weak_measurement_fraction * 100,
                       c=peaks.spatial_supported_slides, cmap="viridis", s=8, alpha=.65, rasterized=True)
    axes[0, 1].set(xlabel="m/z", ylabel="Measurements at or below the weak screen (%)", title="Weak-signal burden across the mass range")
    tested = slides.spatial_tested & slides.raw_correlation.notna() & slides.normalized_correlation.notna()
    h = axes[1, 0].hexbin(slides.loc[tested, "raw_correlation"], slides.loc[tested, "normalized_correlation"],
                          gridsize=55, bins="log", cmap="magma", mincnt=1, rasterized=True)
    axes[1, 0].plot([-.2, 1], [-.2, 1], ls="--", color="grey", lw=1)
    axes[1, 0].set(xlabel="Adjacent-pixel correlation: log1p(raw)", ylabel="Correlation after summed-signal normalization",
                    title="Spatial structure and pixel-wide signal variation")
    fig.colorbar(h, ax=axes[1, 0], label="Peak/slide combinations")
    data = [int((peaks.spatial_supported_slides > 0).sum()),
            int(((peaks.spatial_tested_slides > 0) & (peaks.spatial_supported_slides == 0) & ~peaks.whole_peak_review).sum()),
            int(peaks.whole_peak_review.sum()), int((peaks.spatial_tested_slides == 0).sum())]
    labels = ["Spatial support on ≥1 slide", "Tested, mixed/weak evidence", "Weak/unstructured review", "Insufficient sampled detections"]
    axes[1, 1].barh(labels, data, color=[TEAL, NAVY, ORANGE, "#a5afb8"])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set(xlabel="Number of m/z features", title="Evidence categories (not a true noise classification)")
    for i, v in enumerate(data):
        axes[1, 1].text(v, i, f" {v:,}", va="center", fontsize=9)
    save_figure(fig, output / "figures", "peak_landscape")


def plot_pixel_quality(output: Path, config: dict, summaries: list[dict]) -> None:
    """Generate all-slide spatial QC maps and an overview of pixel flags."""
    rng = np.random.default_rng(config["seed"])
    pooled = []
    for s in summaries:
        frame = pd.read_parquet(output / "cache" / f"slide_{s['slide_index']:02d}_pixels.parquet")
        selected = rng.choice(len(frame), min(4000, len(frame)), replace=False)
        pooled.append(frame.iloc[selected][["summed_peak_intensity", "detected_peaks"]])
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        plots = [(np.log1p(frame.summed_peak_intensity.to_numpy()), "Summed peak intensity", "log1p(sum of retained peaks)", "magma"),
                 (frame.detected_peaks.to_numpy(), "Detected m/z peaks", "Number of positive peaks", "viridis"),
                 (frame.weak_measurement_fraction.to_numpy(), f"Fraction of measurements ≤ {config['weak_intensity_screen']:g}", "Fraction of positive measurements", "cividis")]
        for ax, (values, title, label, cmap) in zip(axes.flat, plots):
            im = ax.imshow(image_grid(frame, values), cmap=cmap, interpolation="none", origin="upper")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, shrink=.75, label=label)
        flags = np.zeros(len(frame))
        flags[frame.high_signal | frame.high_peak_count] = 1
        flags[frame.low_quality_review] = 2
        flags[frame.zero_signal] = 3
        cmap = colors.ListedColormap(["#dbe5e9", ORANGE, RED, "#222222"])
        axes[1, 1].imshow(image_grid(frame, flags), cmap=cmap, vmin=-.5, vmax=3.5, interpolation="none", origin="upper")
        axes[1, 1].set_title(f"Pixel review flags: {frame.low_quality_review.mean():.2%} low")
        from matplotlib.patches import Patch
        axes[1, 1].legend(handles=[Patch(color=c, label=t) for c, t in zip(cmap.colors,
                                 ["Within screens", "High outlier", "Low signal / peak count", "Zero signal"])],
                            loc="upper right", fontsize=8)
        for ax in axes.flat:
            ax.set(xlabel="Acquisition x (pixel offset)", ylabel="Acquisition y (pixel offset)")
        fig.suptitle(f"Slide {s['slide']} — counts-layer quality maps", fontsize=15)
        save_figure(fig, output / "figures", f"spatial_qc_{s['slide_index']:02d}")
        print(f"Rendered QC maps {s['slide_index'] + 1}/{len(summaries)}: {s['slide']}", flush=True)
    sample = pd.concat(pooled, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, .24 * len(summaries))), constrained_layout=True)
    h = axes[0].hexbin(np.log10(1 + sample.summed_peak_intensity), np.log10(1 + sample.detected_peaks),
                       gridsize=65, cmap="magma", bins="log", mincnt=1, rasterized=True)
    axes[0].set(xlabel="log10(1 + summed peak intensity)", ylabel="log10(1 + detected peaks)",
                title="Pixel signal and spectrum complexity\nUp to 4,000 sampled pixels per slide")
    fig.colorbar(h, ax=axes[0], label="Sampled pixels")
    y = np.arange(len(summaries))
    low = [100 * s["pixel_flags"]["low_quality_review"] / s["pixels"] for s in summaries]
    high = [100 * max(s["pixel_flags"]["high_signal"], s["pixel_flags"]["high_peak_count"]) / s["pixels"] for s in summaries]
    axes[1].barh(y, low, color=RED, label="Low-quality review (union)")
    axes[1].scatter(high, y, color=ORANGE, s=15, label="Larger of the two high-outlier rates")
    axes[1].set_yticks(y, [s["slide"] for s in summaries], fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Pixels flagged (%)", title="Slide-specific robust outlier screens")
    axes[1].legend(fontsize=8)
    save_figure(fig, output / "figures", "pixel_quality")


def render_examples(output: Path, config: dict, summaries: list[dict], cutoff: float) -> list[dict]:
    """Select reproducible examples and reread only bounded blocks for their maps."""
    order = sorted(summaries, key=lambda s: s["sample_positive_quantiles"].get("q50", 0))
    choices = [order[0], order[len(order) // 2], order[-1]]
    choices += sorted(summaries, key=lambda s: s["weak_unstructured_peaks"], reverse=True)[:2]
    choices += sorted(summaries, key=lambda s: s["pixel_flags"]["low_quality_review"] / s["pixels"], reverse=True)
    chosen = []
    for s in choices:
        if s["slide_index"] not in [v["slide_index"] for v in chosen]:
            chosen.append(s)
        if len(chosen) >= min(config["example_slides"], len(summaries)):
            break
    statistics = pd.read_parquet(output / "peak_slide_statistics.parquet")
    records = []
    with h5py.File(config["input"], "r") as handle:
        group = handle[f"layers/{config['layer']}"]
        for s in chosen:
            slide = statistics[statistics.slide == s["slide"]].copy()
            supported = slide[slide.spatial_supported]
            tested = slide[slide.spatial_tested]
            selection = []
            weak_supported = supported.sort_values(["weak_measurement_fraction", "mean_positive_intensity"], ascending=[False, True])
            unstructured = tested.sort_values(["weak_unstructured_review", "normalized_excess", "weak_measurement_fraction"], ascending=[False, True, False])
            strong = supported.sort_values(["mean_positive_intensity", "normalized_excess"], ascending=False)
            for label, candidates in (("Weak, spatially supported", weak_supported),
                                      ("Weak / low spatial support", unstructured),
                                      ("Strong, spatially supported", strong)):
                candidates = candidates[~candidates.feature_index.isin([r["feature_index"] for r in selection])]
                if not len(candidates):
                    continue
                row = candidates.iloc[0].to_dict()
                row["example_category"] = label
                selection.append(row)
            if not selection:
                continue
            frame = pd.read_parquet(output / "cache" / f"slide_{s['slide_index']:02d}_pixels.parquet")
            rows = frame.row_position.to_numpy()
            indices = [int(r["feature_index"]) for r in selection]
            intensities = np.empty((len(frame), len(indices)))
            for lo, hi, block in iter_blocks(group, rows, config["block_rows"]):
                intensities[lo:hi] = block[:, indices].toarray()
            fig, axes = plt.subplots(2, len(indices), figsize=(5 * len(indices), 9), squeeze=False, constrained_layout=True)
            for j, record in enumerate(selection):
                values = intensities[:, j]
                positive = values[values > 0]
                vmax = float(np.quantile(positive, .99)) if len(positive) else 1.
                norm = colors.SymLogNorm(linthresh=max(.02, vmax * .001), vmin=0, vmax=max(vmax, .02))
                keep = values > cutoff
                for i, plotted in enumerate((values, np.where(keep, values, 0.))):
                    im = axes[i, j].imshow(image_grid(frame, plotted), cmap="magma", norm=norm,
                                            origin="upper", interpolation="none")
                    axes[i, j].set(xlabel="Acquisition x", ylabel="Acquisition y")
                    if i == 0:
                        axes[i, j].set_title(f"{record['example_category']}\nm/z {float(record['mz']):.4f}; prevalence {record['prevalence']:.1%}\n"
                                            f"Adjacent r: raw {record['raw_correlation']:.2f}, normalized {record['normalized_correlation']:.2f}", fontsize=10)
                    else:
                        loss = np.sum((values > 0) & ~keep) / max(np.sum(values > 0), 1)
                        axes[i, j].set_title(f"After cutoff {cutoff:g}: {loss:.1%} detections removed", fontsize=10)
                fig.colorbar(im, ax=axes[:, j], shrink=.7, label="Raw intensity (symmetric-log color; capped at positive p99)")
                record.update(slide_index=s["slide_index"], cutoff=cutoff,
                              detection_loss_fraction=float(np.sum((values > 0) & ~keep) / max(np.sum(values > 0), 1)))
                records.append(record)
            fig.suptitle(f"Slide {s['slide']} — selected peak maps, before and after filtering", fontsize=15)
            save_figure(fig, output / "figures", f"peak_examples_{s['slide_index']:02d}")
            print(f"Rendered peak examples: {s['slide']}", flush=True)
    pd.DataFrame(records).to_csv(output / "peak_example_selection.csv", index=False)
    return records


def figure_html(name: str, caption: str) -> str:
    """Embed a local scientific plot with its export links."""
    return (f'<figure><a href="figures/{name}.png"><img loading="lazy" src="figures/{name}.png" alt="{escape(caption)}"></a>'
            f'<figcaption>{escape(caption)} · <a href="figures/{name}.pdf">PDF</a> · '
            f'<a href="figures/{name}.png">PNG</a></figcaption></figure>')


def render_report(output: Path, config: dict, features: np.ndarray, summaries: list[dict], recommendations: dict) -> None:
    """Generate all figures and a self-contained-in-directory HTML report."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "semibold", "savefig.dpi": 170,
                         "pdf.fonttype": 42})
    (output / "figures").mkdir(exist_ok=True)
    cutoff = recommendations["conservative_raw_cutoff"]
    plot_distributions(output, config, summaries, cutoff)
    plot_thresholds(output, cutoff)
    plot_peak_landscape(output)
    plot_pixel_quality(output, config, summaries)
    examples = render_examples(output, config, summaries, cutoff)
    thresholds = pd.read_csv(output / "threshold_tradeoffs.csv")
    peaks = pd.read_csv(output / "peak_statistics.csv")
    selected = thresholds[np.isclose(thresholds.threshold, cutoff)].iloc[0]
    weak = thresholds[np.isclose(thresholds.threshold, config["weak_intensity_screen"])].iloc[0]
    n = sum(s["pixels"] for s in summaries)
    nnz = sum(s["positive_measurements"] for s in summaries)
    count = len(features)
    supported = recommendations["spatial_supported_peak_count"]
    review = recommendations["whole_peak_review_count"]
    unknown = recommendations["insufficient_spatial_evidence_peak_count"]
    flag_fraction = recommendations["low_quality_pixel_review_count"] / n
    table = thresholds[["threshold", "measurement_loss_fraction", "intensity_loss_fraction",
                        "worst_slide_measurement_loss_fraction", "worst_slide_intensity_loss_fraction",
                        "newly_empty_pixels", "conservative_policy_eligible"]].copy()
    for field in table.columns:
        if "fraction" in field:
            table[field] = table[field].map(lambda x: f"{x:.3%}")
    table.columns = ["Raw cutoff ≤", "Measurements removed", "Summed signal lost", "Worst slide: measurements",
                     "Worst slide: signal", "Newly empty pixels", "Meets retention policy"]
    pixel_limits = pd.DataFrame([{"Slide": s["slide"], "Low summed intensity <": s["pixel_signal_limits"]["lower"],
                                   "Low detected peaks <": s["pixel_peak_count_limits"]["lower"],
                                   "Low-quality review": f"{s['pixel_flags']['low_quality_review'] / s['pixels']:.2%}"}
                                  for s in summaries])
    for key in ("Low summed intensity <", "Low detected peaks <"):
        pixel_limits[key] = pixel_limits[key].map(lambda x: f"{x:.2f}")
    rarity = pd.DataFrame([{"Maximum within-slide detection frequency below": f"{float(t):.1%}",
                            "Features flagged for review": v}
                          for t, v in recommendations["prevalence_screen_counts"].items()])
    peak_review = peaks[peaks.whole_peak_review].sort_values("detections", ascending=False)
    review_html = (peak_review[["mz", "detections", "mean_positive_intensity", "spatial_tested_slides",
                                "weak_unstructured_slides"]].head(30).to_html(index=False, float_format=lambda x: f"{x:.4g}")
                   if len(peak_review) else "<p>No whole peak meets the conservative weak/unstructured review rule.</p>")
    cards = [("Pixels", f"{n:,}"), ("m/z features", f"{count:,}"), ("Slides", str(len(summaries))),
             ("Conservative raw cutoff", f"{cutoff:g}")]
    card_html = "".join(f'<div class="card"><span>{escape(label)}</span><strong>{value}</strong></div>' for label, value in cards)
    regions = pd.DataFrame([{"slide": s["slide"], "region": region, **values}
                            for s in summaries for region, values in s["regions"].items()])
    regions.to_csv(output / "lesion_nonlesion_pixel_summaries.csv", index=False)
    sections = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MSI original-counts quality assessment</title><style>
body{{margin:0;background:#eef2f4;color:#203447;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:38px 30px;background:white}}h1{{font-size:34px;line-height:1.2}}h2{{margin-top:44px;border-top:1px solid #dce4e9;padding-top:26px}}h3{{margin-top:28px}}a{{color:#137f83}}.kicker{{font-size:12px;letter-spacing:2px;color:#137f83}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:25px 0}}.card{{background:#edf5f4;padding:18px;border-radius:8px}}.card span{{display:block;font-size:13px}}.card strong{{display:block;font-size:26px}}.note{{background:#fff5e7;border-left:4px solid #c9752b;padding:18px}}figure{{margin:28px 0}}img{{max-width:100%;height:auto}}figcaption{{font-size:13px;color:#61717c}}table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{padding:9px;border-bottom:1px solid #e3e9ed;text-align:right}}th{{background:#f0f5f7}}td:first-child,th:first-child{{text-align:left}}.scroll{{overflow-x:auto}}details{{margin:15px 0;padding:12px;border:1px solid #dce4e9;border-radius:7px}}summary{{cursor:pointer;font-weight:600}}code{{background:#edf2f5;padding:2px 4px}}li{{margin:6px 0}}@media(max-width:700px){{main{{padding:20px}}.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:28px}}}}</style></head><body><main>
<div class="kicker">MALDI MSI · ORIGINAL COUNTS LAYER · READ-ONLY ANALYSIS</div>
<h1>How much weak or noise-like signal is present?</h1>
<p>Source: <code>{escape(str(Path(config['input']).resolve()))}</code>, <code>layers["{escape(config['layer'])}"]</code>.
All {n:,} pixels and {nnz:,} positive, distinct pixel–peak measurements contribute to the exact summaries. The layer contains fractional intensities, not integer molecule counts.</p>
<div class="cards">{card_html}</div>
<p><strong>{weak.measurement_loss_fraction:.1%} of positive measurements are ≤ {config['weak_intensity_screen']:g}, accounting for {weak.intensity_loss_fraction:.1%} of summed intensity.</strong>
This describes a weak-signal burden; it is not a measured noise fraction. {supported:,} of {count:,} m/z features show spatial support on at least one slide, {review:,} meet the consistent weak/unstructured review rule, and {unknown:,} have insufficient sampled detections for spatial assessment.</p>
<p><strong>The conservative retention-policy cutoff is {cutoff:g}:</strong> keep values strictly greater than this raw counts-layer intensity.
It removes {selected.measurement_loss_fraction:.2%} of measurements and {selected.intensity_loss_fraction:.3%} of summed signal globally;
the worst slide loses {selected.worst_slide_measurement_loss_fraction:.2%} and {selected.worst_slide_intensity_loss_fraction:.3%}, respectively.
These thresholds have not been applied to the data.</p>
<p class="note"><strong>What this file can establish:</strong> weak-signal burden, spatial organization, slide differences, and the cost of filtering.
It cannot establish a true instrumental noise fraction, baseline noise, or spectral S/N because it contains aligned peak intensities without raw spectral baselines or known blank/off-tissue controls.
Non-lesion tissue is not background. Spatial support can also occur in acquisition artefacts; absence of support does not prove a peak is noise.</p>
<h2>1. Original intensity distributions and slide differences</h2>
<p>The intensity curves use approximately {config['sample_values_per_slide']:,} uniformly sampled positive measurements per slide, weighted back by each slide’s exact measurement count.
Quantiles are sample estimates. Detection counts, intensity sums, and threshold trade-offs use every measurement.
Summed peak intensity is the sum over these {count:,} aligned features; it is not the TIC of an unfiltered full spectrum.</p>""",
                figure_html("intensity_distributions", "Positive intensity distributions and slide-specific cumulative distributions."),
                figure_html("slide_quality", "Slide-to-slide differences in intensity scale and peak detection."),
                "<h2>2. Intensity thresholds and their measured costs</h2>",
                f"<p>The recommendation is the largest tested cutoff retaining ≥{config['retained_signal_min']:.0%} of summed intensity and ≥{config['retained_measurements_min']:.0%} of positive measurements on <em>every</em> slide. This is an explicit conservative retention policy, not an estimated noise boundary. Higher cutoffs below are sensitivity analyses, not validated cleaning settings.</p>",
                '<div class="scroll">' + table.to_html(index=False, escape=True) + '</div>',
                figure_html("threshold_tradeoffs", "Exact removal costs globally, on the worst affected slide, and within spatially supported peak/slide combinations."),
                "<h2>3. Which whole m/z peaks warrant review?</h2>",
                figure_html("peak_landscape", "Prevalence, intensity, and spatial evidence for the complete feature set."),
                f"<p>{review:,} peaks meet the review rule. Together they account for {recommendations['whole_peak_review_measurement_fraction']:.2%} of measurements and {recommendations['whole_peak_review_signal_fraction']:.3%} of summed intensity. The rule requires weak/unstructured evidence on at least two and at least half of adequately sampled slides, with no spatially supported slide. No whole peaks are automatically removed.</p>",
                review_html,
                "<p>Frequency screens below require rarity on every slide, so a feature localized to one slide is not lost merely through global dilution. These are review thresholds, not evidence of noise.</p>", rarity.to_html(index=False),
                "<h2>4. Pixel quality and spatial maps</h2>",
                f"""<p>{recommendations['low_quality_pixel_review_count']:,} pixels ({flag_fraction:.2%}) meet at least one low-quality review flag; {recommendations['zero_signal_pixels']:,} have zero signal.
For each slide, lower limits are median − {config['mad_multiplier']:g} × 1.4826 × MAD in log10(1+x) space, separately for summed intensity and detected-peak count, then converted back to raw units. Zero MAD disables the corresponding outlier screen. Upper-tail flags are displayed separately.</p>""",
                figure_html("pixel_quality", "Pixel complexity and slide-specific QC flag rates."),
                '<details><summary>Numerical pixel cutoffs for every slide</summary><div class="scroll">' + pixel_limits.to_html(index=False) + '</div></details>',
                "<h3>Selected peak maps before and after the conservative cutoff</h3><p>Examples are selected reproducibly from slides spanning intensity scale, weak/unstructured burden, and pixel outliers. Colors use the same range before and after filtering. Spatially supported weak peaks illustrate why low intensity alone is insufficient.</p>"]
    example_indices = list(dict.fromkeys(int(r["slide_index"]) for r in examples))
    for i in example_indices:
        slide = next(s["slide"] for s in summaries if s["slide_index"] == i)
        sections.append(figure_html(f"peak_examples_{i:02d}", f"Slide {slide}: original peak maps above and filtered maps below."))
    sections.append("<h3>Quality maps for all slides</h3>")
    for s in summaries:
        sections.append(f'<details><summary>{escape(s["slide"])} — {s["pixels"]:,} pixels; '
                        f'{s["pixel_flags"]["low_quality_review"] / s["pixels"]:.2%} flagged low</summary>'
                        + figure_html(f"spatial_qc_{s['slide_index']:02d}", "Acquisition coordinates; missing positions are blank; no smoothing applied.") + '</details>')
    sections += [f"""<h2>5. Methods, interpretation, and reproducibility</h2>
<p>Spatial diagnostics use up to {config['spatial_pairs']:,} uniformly sampled four-neighbor edges per slide, without connections across slides or missing pixels.
Pearson correlation includes absent peaks as zero and is evaluated on binary detection, log1p(raw intensity), and log1p(intensity scaled to the sampled median summed signal per pixel).
Each is compared with {config['shuffles']} shuffled endpoint controls. The 95th-percentile shuffled envelope is descriptive, not a p-value or FDR estimate; neighboring pairs may share pixels.
At least {config['min_spatial_detections']} distinct sampled pixels must detect a peak. Spatial support requires both raw and normalized correlation to exceed their shuffled envelopes by at least {config['spatial_support_excess']:g}.
Weak/unstructured screening requires ≥{config['weak_fraction_screen']:.0%} of that peak’s measurements to be ≤{config['weak_intensity_screen']:g}, and both excess correlations ≤{config['spatial_weak_excess']:g}. These effect-size screens are pragmatic diagnostics.</p>
<p>Weak biology, uniform tissue signals, ion suppression, and structured acquisition artefacts remain possible. Normalization may remove genuine abundance variation; raw and normalized evidence are both retained in the tables.
Lesion/non-lesion summaries are descriptive, and CD8 or other IHC outcomes are not used to select thresholds. Any future model preprocessing must be fitted only on its training partition and frozen for evaluation. Counts-layer cutoffs must not be transferred directly to a transformed <code>X</code> matrix.</p>
<p>Methodological context: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC6298046/">SPUTNIK: spatial filtering of MSI peaks</a> and
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10623552/">NECTAR: spectral and spatial characterization of MSI noise</a>.
These studies motivate complementary diagnostics; their instrument-specific thresholds are not transplanted into this dataset.</p>
<p>Run: <code>python -m msi_qc.analyze --config msi_qc/config.yaml</code>. Use <code>--resume</code> only with an unchanged source, analysis code, and configuration. Source size and modification time are checked before and after the run. Sparse duplicates are summed before positivity or thresholding; explicit zeros are removed from positive-measurement denominators.</p>
<h3>Download tables and parameters</h3><ul>"""]
    for name, description in [("threshold_tradeoffs.csv", "Global threshold costs and conservative-policy eligibility"),
                              ("thresholds_by_slide_and_region.csv", "Threshold costs by slide and lesion/non-lesion region"),
                              ("peak_statistics.csv", "All m/z features and whole-peak review flags"),
                              ("peak_slide_statistics.csv", "Per-slide peak statistics and spatial control envelopes"),
                              ("pixel_qc.parquet", "Every pixel’s metrics and flags, with original observation identity"),
                              ("slide_statistics.csv", "Slide summaries and raw pixel cutoffs"),
                              ("lesion_nonlesion_pixel_summaries.csv", "Lesion/non-lesion pixel summaries"),
                              ("peak_example_selection.csv", "Exact peak-map selections and their statistics"),
                              ("recommendations.json", "Numerical recommendations and decision rules"),
                              ("run_manifest.json", "Source identity, configuration, seed, versions, and run status")]:
        sections.append(f'<li><a href="{name}">{name}</a> — {escape(description)}</li>')
    sections.append("</ul></main></body></html>")
    (output / "report.html").write_text("\n".join(sections))

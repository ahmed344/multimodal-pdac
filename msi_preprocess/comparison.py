"""Exact regional summaries and deterministic pixel-level stage comparisons."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy import sparse

from .io import axes, json_write, read_rows
from .cohort import aligned_export_rows


class RegionDiagnostics:
    """Accumulate exact counts and threshold placement, including zero denominators."""

    def __init__(self, output: Path, features: np.ndarray, retained: np.ndarray, cuts: list[float]):
        self.output, self.features, self.retained = output, features, retained
        self.cuts = np.unique(np.r_[cuts, .1, .2, .3, .5, 1.])
        self.bins = np.r_[0, np.geomspace(.0001, 10000, 161), np.inf]
        self.summaries, self.thresholds, self.histograms, self.removals = [], [], [], []
        self.feature_header = True

    def add(self, slide: str, matrices: dict, lesion_positions: np.ndarray) -> None:
        """Assess all tissue and the exact matched lesion rows without label access."""
        feature_rows = []
        for region, positions in (("whole_tissue", None), ("lesion", lesion_positions)):
            for stage, source in matrices.items():
                matrix = source if positions is None else source[positions]
                n, p = matrix.shape
                count = np.zeros(len(self.cuts) + 1, dtype=np.int64)
                intensity = np.zeros(len(self.cuts) + 1)
                hist = np.zeros(len(self.bins) - 1, dtype=np.int64)
                for start in range(0, matrix.nnz, 1_000_000):
                    values = matrix.data[start:start+1_000_000]
                    slots = np.searchsorted(self.cuts, values, side="left")
                    count += np.bincount(slots, minlength=len(count))
                    intensity += np.bincount(slots, weights=values, minlength=len(count))
                    hist += np.bincount(np.searchsorted(self.bins, values, side="right") - 1,
                                        minlength=len(hist))
                total = float(intensity.sum())
                base = dict(region=region, slide=slide, stage=stage, rows=n, features=p,
                            entries=n*p, positive_measurements=matrix.nnz, intensity_sum=total,
                            zero_fraction=1-matrix.nnz/max(n*p, 1))
                self.summaries.append(base)
                below, mass = np.cumsum(count), np.cumsum(intensity)
                self.thresholds.extend(dict(**base, boundary=float(c), positive_below=int(below[k]),
                    fraction_positive_below=float(below[k]/max(matrix.nnz, 1)),
                    fraction_all_at_or_below=float((n*p-matrix.nnz+below[k])/max(n*p, 1)),
                    intensity_below=float(mass[k]), fraction_intensity_below=float(mass[k]/total) if total else 0.)
                    for k, c in enumerate(self.cuts))
                self.histograms.extend(dict(region=region, slide=slide, stage=stage,
                    lower=float(lo), upper=float(hi), count=int(v))
                    for lo, hi, v in zip(self.bins[:-1], self.bins[1:], hist))
            before = matrices["integrated"] if positions is None else matrices["integrated"][positions]
            after = matrices["cleaned"] if positions is None else matrices["cleaned"][positions]
            removed = before - after
            removed.eliminate_zeros()
            n, p = before.shape
            self.removals.append(dict(region=region, slide=slide, rows=n, retained_features=p,
                entries=n*p, positive_before=before.nnz, positive_after=after.nnz,
                measurements_removed=removed.nnz, intensity_before=float(before.sum()),
                intensity_removed=float(removed.sum())))
            before_n = np.bincount(before.indices, minlength=p)
            removed_n = np.bincount(removed.indices, minlength=p)
            before_mass = np.asarray(before.sum(axis=0)).ravel()
            removed_mass = np.asarray(removed.sum(axis=0)).ravel()
            feature_rows.extend(dict(region=region, slide=slide, feature=str(self.features[j]),
                rows=n, positive_before=int(before_n[k]), measurements_removed=int(removed_n[k]),
                fraction_positive_removed=float(removed_n[k]/before_n[k]) if before_n[k] else 0.,
                fraction_all_removed=float(removed_n[k]/n) if n else 0.,
                intensity_before=float(before_mass[k]), intensity_removed=float(removed_mass[k]))
                for k, j in enumerate(self.retained))
        pd.DataFrame(feature_rows).to_csv(self.output / "regional_feature_removal.csv", index=False,
            mode="w" if self.feature_header else "a", header=self.feature_header)
        self.feature_header = False

    def finish(self) -> None:
        """Persist exact statistics; percentages are calculated from pooled totals."""
        for name, values in (("regional_stage_summaries", self.summaries),
                             ("threshold_placement", self.thresholds),
                             ("regional_intensity_histograms", self.histograms)):
            pd.DataFrame(values).to_csv(self.output / f"{name}.csv", index=False)
        rows = pd.DataFrame(self.removals)
        total = rows.groupby("region", sort=False).sum(numeric_only=True).reset_index()
        total["slide"] = "ALL"
        total["retained_features"] = len(self.retained)
        rows = pd.concat([rows, total], ignore_index=True)
        for denominator, output in (("positive_before", "fraction_positive_removed"), ("entries", "fraction_all_removed")):
            rows[output] = rows.measurements_removed / rows[denominator].replace(0, np.nan)
        rows["fraction_intensity_removed"] = rows.intensity_removed / rows.intensity_before.replace(0, np.nan)
        rows.to_csv(self.output / "removal_summary.csv", index=False)
        peaks = pd.read_csv(self.output / "regional_feature_removal.csv", dtype={"feature": str})
        peaks = peaks.groupby(["region", "feature"], sort=False)[
            ["rows", "positive_before", "measurements_removed", "intensity_before", "intensity_removed"]].sum().reset_index()
        peaks["fraction_positive_removed"] = peaks.measurements_removed / peaks.positive_before.replace(0, np.nan)
        peaks["fraction_all_removed"] = peaks.measurements_removed / peaks.rows.replace(0, np.nan)
        peaks.to_csv(self.output / "peak_removal_summary.csv", index=False)


def sample_pixels(obs: pd.DataFrame, data: dict, per_slide: int, seed: int) -> np.ndarray:
    """Select label-free pixels equally per slide, then order by native coordinates."""
    batch = obs[data["batch_column"]].astype(str).to_numpy()
    selected = []
    for i, slide in enumerate(np.unique(batch)):
        rows = np.flatnonzero(batch == slide)
        rows = np.random.default_rng(seed + i).choice(rows, min(per_slide, len(rows)), replace=False)
        frame = obs.iloc[rows]
        order = np.lexsort((frame[data["x_column"]].to_numpy(), frame[data["y_column"]].to_numpy()))
        selected.extend(rows[order])
    return np.asarray(selected, dtype=np.int64)


def rank_retained_peaks(counts: np.ndarray, retained: np.ndarray,
                        limit: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Rank retained peaks by population log1p variance, including sampled zeros.

    Return original feature indices and their variances in descending variance
    order, breaking ties by original feature index regardless of retained order.
    """
    values = np.log1p(counts[:, retained].astype(np.float64))
    variances = np.var(values, axis=0)
    ranked = np.lexsort((retained, -variances))[:limit]
    return retained[ranked], variances[ranked]


def cluster_peaks(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster once on log1p counts; constant columns remain explicitly unclustered."""
    values = np.log1p(counts.astype(np.float64))
    variable = np.flatnonzero(np.ptp(values, axis=0) > 0)
    constant = np.flatnonzero(np.ptp(values, axis=0) == 0)
    tree = np.empty((0, 4))
    if len(variable) > 1:
        distances = np.clip(pdist(values[:, variable].T, metric="correlation"), 0, 2)
        if not np.isfinite(distances).all():
            raise ValueError("Nonfinite peak clustering distance.")
        tree = linkage(distances, method="average")
        variable = variable[leaves_list(tree)]
    return np.r_[variable, constant], tree, constant


def cluster_slides(counts: np.ndarray, slide_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster slide mean log1p-count profiles across the displayed peaks.

    Each slide is one observation, using all its sampled pixels including zeros.
    Return slide names in display order, the linkage on alphabetically ordered
    nonconstant profiles, and constant slide names appended alphabetically.
    """
    slide_values = np.asarray(slide_values, dtype=str)
    slides = np.unique(slide_values)
    tree = np.empty((0, 4))
    if not len(slides):
        return slides, tree, slides
    values = np.log1p(counts.astype(np.float64))
    profiles = np.stack([values[slide_values == slide].mean(axis=0) for slide in slides])
    if not np.isfinite(profiles).all():
        raise ValueError("Nonfinite slide clustering profile.")
    variable = np.flatnonzero(np.ptp(profiles, axis=1) > 0)
    constant = np.flatnonzero(np.ptp(profiles, axis=1) == 0)
    if len(variable) > 1:
        distances = np.clip(pdist(profiles[variable], metric="correlation"), 0, 2)
        if not np.isfinite(distances).all():
            raise ValueError("Nonfinite slide clustering distance.")
        tree = linkage(distances, method="average")
        variable = variable[leaves_list(tree)]
    return slides[np.r_[variable, constant]], tree, slides[constant]


def sampled_stages(counts: np.ndarray, cleaned: np.ndarray, retained: np.ndarray,
                   multipliers: np.ndarray, peaks: np.ndarray) -> dict[str, np.ndarray]:
    """Align all stages to displayed original indices and the same sampled rows."""
    retained_columns = pd.Index(retained).get_indexer(peaks)
    if np.any(retained_columns < 0):
        raise ValueError("Displayed peaks must belong to the retained subset.")
    selected = counts[:, peaks]
    integrated = selected / multipliers[:, None]
    return dict(counts=selected, selected=selected, integrated=integrated,
                cleaned=cleaned[:, retained_columns])


def render_heatmaps(output: Path, manifest: dict, state: dict) -> dict:
    """Render individual pixels with frozen clustering and common display scales."""
    config, data = manifest["config"], manifest["coordinate_columns"]
    obs, original_rows = aligned_export_rows(output / "lesion.h5ad", manifest, state, "lesion")
    settings = config.get("report", {"pixels_per_slide": 100, "upper_percentile": 99.5})
    rows = sample_pixels(obs, data, settings["pixels_per_slide"], config["seed"])
    sampled = obs.iloc[rows].copy()
    sampled.insert(0, "source_row", original_rows[rows])
    sampled.insert(1, "export_row", rows)
    with h5py.File(config["lesion_input"], "r") as h:
        counts = read_rows(h["layers/counts"], original_rows[rows], config["block_rows"]).toarray()
        old = read_rows(h["X"], original_rows[rows], config["block_rows"]).toarray()
    with h5py.File(output / "lesion.h5ad", "r") as h:
        cleaned = read_rows(h["X"], rows, config["block_rows"]).toarray()
    ranked, variances = rank_retained_peaks(counts, state["retained"], settings.get("max_peaks", 500))
    # Cluster in original feature order so tied distances and appended constants
    # have a deterministic input order independent of variance rank.
    cluster_input = np.sort(ranked)
    order, tree, constant = cluster_peaks(counts[:, cluster_input])
    peaks = cluster_input[order]
    ranks = pd.Index(ranked).get_indexer(peaks)
    pd.DataFrame(dict(feature=state["features"][peaks], original_index=peaks,
        retained_index=pd.Index(state["retained"]).get_indexer(peaks),
        cluster_position=np.arange(len(peaks)), variance_rank=ranks+1,
        log1p_counts_variance=variances[ranks], retained=True,
        constant_in_sample=np.isin(order, constant),
        anchor=np.isin(peaks, state["anchors"]))).to_csv(output / "heatmap_peaks.csv", index=False)
    np.save(output / "heatmap_linkage.npy", tree)
    slide_values = sampled[data["batch_column"]].astype(str).to_numpy()
    multipliers = dict(zip(state["slides"], state["multipliers"]))
    factors = np.array([multipliers[s] for s in slide_values])
    stages = sampled_stages(counts, cleaned, state["retained"], factors, peaks)
    stages["old_X"] = old[:, peaks]
    slide_order, slide_tree, constant_slides = cluster_slides(stages["counts"], slide_values)
    row_order = np.concatenate([np.flatnonzero(slide_values == slide) for slide in slide_order])
    stages = {stage: values[row_order] for stage, values in stages.items()}
    sampled = sampled.iloc[row_order]
    sampled.to_csv(output / "heatmap_pixels.csv", index=False)
    np.save(output / "heatmap_slide_linkage.npy", slide_tree)
    slide_values = slide_values[row_order]
    # Boundaries must follow displayed blocks, not alphabetical group order.
    starts = np.r_[0, np.flatnonzero(slide_values[1:] != slide_values[:-1]) + 1]
    slides = slide_values[starts]
    sizes = np.diff(np.r_[starts, len(slide_values)])
    positive = stages["counts"][stages["counts"] > 0]
    upper = float(np.percentile(positive, settings["upper_percentile"])) if len(positive) else 1.
    colors = colormaps["turbo"]
    metadata = dict(sampled_pixels=len(rows), pixels_per_slide=settings["pixels_per_slide"],
                    peaks=len(peaks), retained_features=len(state["retained"]),
                    requested_peaks=settings.get("max_peaks", 500),
                    selection="Highest population variance of log1p(counts) across sampled pixels, including zeros",
                    variance_ddof=0, variance_tie_break="original feature index ascending",
                    peak_metadata="heatmap_peaks.csv", index_base=0, variance_rank_base=1,
                    clustering="average linkage of correlation distances on sampled log1p(counts)",
                    linkage_input_original_indices=cluster_input[~np.isin(np.arange(len(cluster_input)), constant)].tolist(),
                    clustered_original_indices=peaks.tolist(),
                    slide_order=slides.tolist(),
                    slide_clustering=dict(source="original lesion layers/counts",
                        profile="mean log1p(counts) across sampled pixels, including zeros, for each displayed peak",
                        metric="correlation", method="average", input_order="alphabetical slide identifier",
                        linkage="heatmap_slide_linkage.npy",
                        linkage_input_slides=np.setdiff1d(slides, constant_slides).tolist(),
                        constant_slides=constant_slides.tolist(), constant_policy="append alphabetically",
                        within_slide_order="native coordinates (y, then x)"),
                    slide_blocks=[dict(slide=str(slide), start=int(start), size=int(size))
                                  for slide, start, size in zip(slides, starts, sizes)],
                    constant_peaks=len(constant), lower_linear=0., upper_linear=upper,
                    upper_percentile=settings["upper_percentile"],
                    upper_population="positive sampled counts of displayed peaks",
                    cmap="turbo", figure_inches=[14, 24], figure_dpi=180, dendrogram_visible=False,
                    raster_shape=[len(rows), len(peaks)],
                    display="log1p only; exported X remains linear", stages={})
    for stage, values in stages.items():
        vmax = upper
        if stage == "old_X":
            positive_old = values[values > 0]
            vmax = float(np.percentile(positive_old, settings["upper_percentile"])) if len(positive_old) else 1.
        display = np.log1p(values)
        # Save one source pixel per raster cell for zooming, independent of figure size.
        plt.imsave(output / "figures" / f"heatmap_{stage}_pixels.png", display,
                   cmap=colors, vmin=0, vmax=np.log1p(vmax))
        fig = plt.figure(figsize=(14, 24), layout="constrained")
        grid = fig.add_gridspec(1, 2, width_ratios=[12, .3])
        ax = fig.add_subplot(grid[0, 0])
        ax.set_title(f"{stage}: {len(peaks)} most-variable retained peaks\nIndividual lesion pixels · slides and peaks clustered from counts")
        im = ax.imshow(display, aspect="auto", interpolation="nearest", cmap=colors, vmin=0, vmax=np.log1p(vmax))
        ax.set_yticks(starts+(sizes-1)/2, slides, fontsize=7)
        ax.set_xlabel("Most-variable retained peaks clustered once from log1p(counts)"
                      + ("; constants appended at right" if len(constant) else ""))
        ax.set_ylabel("Sampled pixels grouped by clustered slide")
        for start in starts[1:]:
            ax.axhline(start-.5, color="white", lw=.35, alpha=.7)
        ax.set_xticks([])
        cb = fig.colorbar(im, cax=fig.add_subplot(grid[0, 1]))
        ticks = np.linspace(0, np.log1p(vmax), 6)
        cb.set_ticks(ticks, labels=[f"{v:.3g}" for v in np.expm1(ticks)])
        cb.set_label("Linear intensity (log1p color spacing)" + ("; old X own scale" if stage == "old_X" else "; shared scale"))
        fig.savefig(output / "figures" / f"heatmap_{stage}.png", dpi=180)
        plt.close(fig)
        valid = np.isfinite(values)
        metadata["stages"][stage] = dict(lower_linear=0., upper_linear=vmax,
            scale="own scale" if stage == "old_X" else "shared counts scale",
            values_saturated=int(np.sum(values > vmax)),
            fraction_saturated=float(np.sum(values > vmax)/max(valid.sum(), 1)))
    json_write(output / "heatmap.json", metadata)
    return metadata


def pooled_removal(output: Path) -> list[dict]:
    """Return explicit regional denominators and percentages for the report."""
    frame = pd.read_csv(output / "removal_summary.csv")
    return json.loads(frame[frame.slide == "ALL"].to_json(orient="records"))


def render_thresholds(output: Path, manifest: dict, state: dict) -> pd.DataFrame:
    """Plot exact cutoff counts against histogram-based positive intensity CDFs."""
    histogram = pd.read_csv(output / "regional_intensity_histograms.csv")
    thresholds = pd.read_csv(output / "threshold_placement.csv")
    pool = thresholds.groupby(["region", "stage", "boundary"], sort=False)[
        ["entries", "positive_measurements", "intensity_sum", "positive_below", "intensity_below"]].sum().reset_index()
    pool["percent_positive_below"] = 100*pool.positive_below/pool.positive_measurements.replace(0, np.nan)
    pool["percent_intensity_below"] = 100*pool.intensity_below/pool.intensity_sum.replace(0, np.nan)
    pool["percent_zero"] = 100*(1-pool.positive_measurements/pool.entries)
    pool.to_csv(output / "threshold_placement_pooled.csv", index=False)
    boundary = manifest["config"]["local"]["weak_boundary"]
    fig, axes_plot = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for col, region in enumerate(("lesion", "whole_tissue")):
        for stage in ("counts", "selected", "integrated", "cleaned"):
            group = histogram[(histogram.region == region) & (histogram.stage == stage)]
            bins = group.groupby("upper", sort=True)["count"].sum()
            finite = np.isfinite(bins.index)
            xs, ys = bins.index.to_numpy()[finite], bins.to_numpy()[finite]
            axes_plot[0, col].plot(xs, ys/max(bins.sum(), 1), label=stage)
            axes_plot[1, col].plot(xs, np.cumsum(ys)/max(bins.sum(), 1), label=stage)
        for row in range(2):
            ax = axes_plot[row, col]
            for cut in (.1, .2, .3, .5, 1.):
                ax.axvline(cut, color="#707787", alpha=.45, lw=.8)
                ax.text(cut, .97, str(cut), rotation=90, transform=ax.get_xaxis_transform(),
                        fontsize=8, va="top", ha="right", bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1))
            ax.axvline(boundary, color="#bc343e", lw=1.5)
            ax.set(xscale="log", xlim=(.001, 100), xlabel="Linear intensity; raw counts vs reference-adjusted stages")
            ax.grid(alpha=.15)
        axes_plot[0, col].set(title=f"{region}: positive intensities", yscale="log", ylabel="Fraction per log-spaced bin")
        axes_plot[0, col].legend()
        axes_plot[1, col].set(ylabel="Cumulative fraction of positive measurements", ylim=(0, 1.02))
        for stage in ("selected", "integrated"):
            row = pool[(pool.region == region) & (pool.stage == stage) & (pool.boundary == boundary)]
            if len(row):
                axes_plot[1, col].scatter([boundary], [row.iloc[0].percent_positive_below/100], s=30,
                    color={"selected": "#ff7f0e", "integrated": "#2ca02c"}[stage])
    fig.savefig(output / "figures" / "threshold_distributions.png", dpi=180)
    plt.close(fig)
    multipliers = pd.read_csv(output / "slide_multipliers.csv")
    multipliers["raw_equivalent_local_boundary"] = boundary*multipliers.multiplier
    multipliers.to_csv(output / "slide_thresholds.csv", index=False)
    settings = manifest["config"]["integration"]
    lower = settings.get("multiplier_min", 1/settings.get("max_correction", 2.))
    upper = settings.get("multiplier_max", settings.get("max_correction", 2.))
    fig, axs = plt.subplots(1, 2, figsize=(15, max(5, len(multipliers)*.18)), layout="constrained")
    y = np.arange(len(multipliers))
    axs[0].plot(multipliers.raw_multiplier, y, "o", ms=4, label="Estimated")
    axs[0].plot(multipliers.multiplier, y, "x", ms=4, label="Applied")
    for bound in (lower, upper):
        axs[0].axvline(bound, color="#bc343e", ls="--", lw=1)
    axs[0].axvline(1, color="gray", lw=1)
    axs[0].set(yticks=y, yticklabels=multipliers.slide, xlabel="Slide divisor: X = counts / multiplier")
    axs[0].tick_params(axis="y", labelsize=7)
    axs[0].legend()
    axs[1].plot(multipliers.raw_equivalent_local_boundary, y, "o", ms=4)
    axs[1].set(yticks=[], xlabel=f"Raw counts equivalent of local cutoff {boundary}")
    fig.savefig(output / "figures" / "multipliers_and_thresholds.png", dpi=170)
    plt.close(fig)
    return pool


def biological_summary(output: Path, retained: np.ndarray) -> dict:
    """Describe paired associations and constant-feature losses without fit feedback."""
    variation = pd.read_csv(output / "slide_variation.csv", dtype={"feature": str})
    decisions = pd.read_csv(output / "peak_decisions.csv", dtype={"feature": str})
    common = decisions.loc[decisions.retained, "feature"]
    variation = variation[variation.feature.isin(common)]
    eta = variation.groupby("stage").slide_eta_squared.median().to_dict()
    result = dict(median_slide_eta_squared_common_features={k: float(v) if np.isfinite(v) else None for k, v in eta.items()})
    path = output / "biological_associations.csv"
    if not path.exists():
        return result
    frame = pd.read_csv(path, dtype={"feature": str})
    frame = frame[frame.feature.isin(common)]
    keys = ["slide", "split", "label", "feature"]
    before = frame[frame.stage == "integrated"].set_index(keys).correlation
    after = frame[frame.stage == "cleaned"].set_index(keys).correlation
    paired = pd.concat([before.rename("before"), after.rename("after")], axis=1).reset_index()
    associations = []
    for (scope, label), group in paired.groupby(["split", "label"]):
        valid = group.before.notna() & group.after.notna()
        delta = (group.loc[valid, "after"] - group.loc[valid, "before"]).abs()
        associations.append(dict(scope=scope, label=label, paired_correlations=int(valid.sum()),
            median_absolute_change=float(delta.median()) if len(delta) else None,
            p95_absolute_change=float(delta.quantile(.95)) if len(delta) else None,
            became_constant=int((group.before.notna() & group.after.isna()).sum())))
    result["biological_associations"] = associations
    return result


def tic_report_section(output: Path, fingerprint: str) -> str:
    """Include optional TIC figures only when both belong to this fitted export."""
    path = output / "tic_maps.json"
    if not path.exists():
        return ""
    meta = json.loads(path.read_text())
    if meta.get("fingerprint") != fingerprint:
        return ""
    figures = [meta["stages"][stage]["figure"] for stage in ("counts", "cleaned")]
    if not all((output / figure).is_file() for figure in figures):
        return ""
    parts = [f'<section><h2>Whole-tissue spatial TIC</h2><p>Both figures show all {len(meta["slide_order"])} slides '
        f'and {meta["pixels"]:,} tissue pixels, summing the same {meta["retained_features"]:,} retained peaks per pixel. '
        'TIC is summed in linear units before the display transform. Counts already contain RMS normalization; '
        'cleaned intensities include slide scaling and local removal.</p>',
        f'<p>Turbo colors use log1p spacing with a shared range from zero to {meta["upper_linear"]:.6g} '
        f'(the {meta["upper_percentile"]:g}th percentile of positive counts TIC across all tissue pixels). '
        'Values above this limit saturate. Every panel uses the same scale and slide order. '
        'White regions are missing tissue; measured zeros use the lowest Turbo color. '
        'These maps include all retained peaks and tissue pixels, independently of the clustered-heatmap subsets.</p>']
    for label, figure in zip(("Retained-counts TIC", "Cleaned TIC"), figures):
        parts.append(f'<h3>{label}</h3><a href="{escape(figure)}"><img loading="lazy" '
                     f'src="{escape(figure)}" alt="{label}, all slides"></a>')
    parts.append('<p><a href="tic_maps.json">TIC sources, scale, slide order and saturation statistics</a></p></section>')
    return "\n".join(parts)


def render_report(fitted: str | Path, *, heatmaps: bool = True) -> Path:
    """Regenerate standalone figures and HTML from saved exact diagnostics and exports."""
    from .fit import load_fit
    manifest, state = load_fit(fitted)
    output = Path(fitted).resolve().parent
    meta = render_heatmaps(output, manifest, state) if heatmaps else json.loads((output / "heatmap.json").read_text())
    pool = render_thresholds(output, manifest, state)
    removals = pooled_removal(output)
    summary_path = output / "validation_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if summary.get("fingerprint") != manifest["fingerprint"]:
        summary = {}
    summary.update(fingerprint=manifest["fingerprint"], fitting=manifest.get("fitting", manifest.get("split")),
        retained_features=len(state["retained"]), removed_features=len(state["features"])-len(state["retained"]),
        reference_anchors=len(state["anchors"]), slides_at_correction_limit=int(state["correction_limited"].sum()),
        removal=removals, **biological_summary(output, state["retained"]))
    for name in ("verification", "smoke"):
        path = output / f"{name}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("fingerprint") == manifest["fingerprint"]:
                summary[name] = saved
    json_write(summary_path, summary)
    table = pd.DataFrame(removals)[["region", "measurements_removed", "positive_before", "fraction_positive_removed", "fraction_all_removed", "fraction_intensity_removed"]]
    for col in [c for c in table if c.startswith("fraction")]:
        table[col.replace("fraction", "percent")] = 100*table.pop(col)
    placement = pool[pool.boundary == manifest["config"]["local"]["weak_boundary"]][
        ["region", "stage", "boundary", "percent_positive_below", "percent_intensity_below", "percent_zero"]]
    sensitivity = pd.read_csv(output / "local_sensitivity.csv")
    if "region" not in sensitivity:
        sensitivity["region"] = "whole_tissue"
    sensitivity = sensitivity.groupby(["region", "boundary"])[["measurements_removed", "intensity_removed"]].sum().reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for region, group in sensitivity.groupby("region"):
        denom = next(r for r in removals if r["region"] == region)
        group = group.copy()
        group["percent_positive_removed"] = 100*group.measurements_removed/max(denom["positive_before"], 1)
        axs[0].plot(group.boundary, group.percent_positive_removed, "o-", label=region)
        axs[1].plot(group.boundary, 100*group.intensity_removed/max(denom["intensity_before"], 1), "o-", label=region)
    axs[0].set(xlabel="Weak boundary", ylabel="Positive measurements removed (%)")
    axs[1].set(xlabel="Weak boundary", ylabel="Reference-scaled intensity removed (%)")
    axs[0].legend()
    axs[1].legend()
    fig.savefig(output / "figures" / "removal_sensitivity.png", dpi=170)
    plt.close(fig)
    parts = ['''<!doctype html><html><head><meta charset="utf-8"><title>MSI preprocessing</title>
<style>body{font:16px system-ui;margin:36px auto;padding:0 24px;max-width:1400px;color:#192332;background:#f8fafc}h1,h2{color:#183351}p{max-width:1100px;line-height:1.55}img{max-width:100%;background:white}table{border-collapse:collapse;font-size:13px;background:white}th,td{padding:9px;border:1px solid #d8dee8;text-align:right}th:first-child,td:first-child{text-align:left}section{margin:36px 0}a{color:#205d9c}button,select{padding:8px;margin:6px}summary{cursor:pointer;font-weight:600}.zoom{overflow:auto;max-height:80vh;border:1px solid #ccd3df}.zoom img{max-width:none}code{background:#e8edf4;padding:2px 5px}</style></head><body>
<h1>MSI cleaning: retained-lesion fit, whole-tissue spatial context</h1>
<p>Every retained lesion pixel contributes to peak selection, patient-balanced reference anchors, slide multipliers, and random-location reference envelopes. These are frozen before application. Whole-tissue MSI values supply spatial neighborhoods during cleaning; lesion outputs are extracted from that same result. IHC and H&amp;E labels never determine cleaning. DANN retains its 90%/10% model split, but that split is not independent of preprocessing's MSI context.</p>
<p>Counts already contain RMS normalization. Exported <code>X = counts / slide_multiplier</code> remains linear, with weak unsupported measurements set to zero. DANN applies <code>log1p</code> exactly once.</p>''',
        f'<p><b>{len(state["retained"]):,} / {len(state["features"]):,} peaks retained; {len(state["anchors"]):,} anchors; {int(state["correction_limited"].sum())} slides clipped.</b> Fingerprint: <code>{manifest["fingerprint"]}</code></p>',
        '<section><h2>How many measurements were removed?</h2><p>The primary percentage divides removed measurements by positive measurements in retained peaks before local cleaning. The all-entry denominator also includes original zeros. Peak-selection losses are separate from these local-removal percentages.</p>',
        table.to_html(index=False, float_format=lambda v: f"{v:,.4f}"),
        '<p><a href="removal_summary.csv">Per-slide and pooled denominators</a> · <a href="regional_feature_removal.csv">Per-peak removal</a></p><img src="figures/removal_sensitivity.png"></section>',
        '<section><h2>Where the thresholds sit in your intensities</h2><p>The 0.5 intensity cutoffs, 50% weak-fraction screen, 0.02/0.10 spatial excess screens, anchor-prevalence requirements, and multiplier bounds are operational settings, not estimates of an instrumental noise floor. Random-location 95% spatial envelopes are fitted from lesion values, including zeros. A low intensity alone does not imply removal: spatial support and adequate neighborhood context are also assessed.</p><p>Counts include all original peaks; selected, integrated and cleaned stages contain retained peaks. Selection uses raw RMS-count units. Local removal uses reference-adjusted units; its equivalent raw-count boundary is 0.5 multiplied by each slide divisor. Curves use histogram bins; the table and cutoff markers use exact counts.</p>',
        placement.to_html(index=False, float_format=lambda v: f"{v:,.3f}"),
        '<img src="figures/threshold_distributions.png"><p><a href="threshold_placement.csv">Exact per-slide thresholds</a> · <a href="threshold_placement_pooled.csv">Pooled thresholds</a></p><img src="figures/multipliers_and_thresholds.png"><p><a href="slide_thresholds.csv">Raw/applied multipliers and raw-equivalent cutoffs</a></p></section>',
        f'<section><h2>Pixel-level clustered heatmaps</h2><p>{meta["sampled_pixels"]:,} sampled lesion pixels, up to {meta["pixels_per_slide"]} per slide. Every row is one pixel. The {meta["peaks"]:,} most-variable retained peaks are ranked by population variance of log1p(counts) across these pixels, including zeros, with ties broken by original feature order. The same pixels and peaks appear in every stage, including counts and old X. Columns are clustered once from sampled counts using correlation distance and average linkage; {meta["constant_peaks"]} constant sampled peaks are appended in original order. This is a visualization subset only: the dataset still contains {len(state["retained"]):,} retained peaks.</p>',
        f'<p>The {len(meta["slide_order"])} slides are clustered as independent observations, each represented by its mean log1p(original counts) across the sampled pixels for every displayed peak, including zeros. Slide profiles use correlation distance and average linkage with alphabetical input order; constant profiles are appended alphabetically. All five stages share this slide order and preserve native coordinate order within each slide. Pixel identities are saved in display order.</p>',
        f'<p>Portrait figures measure 14 × 24 inches, with centered slide labels and separators. Dendrograms are hidden; the heatmap fills their former space. Turbo colors use log1p spacing for display only. The common upper limit is {meta["upper_linear"]:.4g} linear units (the {meta["upper_percentile"]:g}th percentile of positive sampled counts of the displayed peaks). Old X uses its own identified scale, with upper limit {meta["stages"]["old_X"]["upper_linear"]:.4g}. Both scales start at zero; values above the upper limit saturate.</p>',
        '<p><a href="heatmap_pixels.csv">Pixel identities in display order</a> · <a href="heatmap_peaks.csv">Original peak indices, variance ranks and clustering order</a> · <a href="heatmap_linkage.npy">Peak linkage</a> · <a href="heatmap_slide_linkage.npy">Slide linkage</a> · <a href="heatmap.json">Selection, slide order, linkage inputs, color limits and saturation counts</a></p><label>Stage <select id="stage" onchange="showStage()">']
    for stage in ("counts", "selected", "integrated", "cleaned", "old_X"):
        parts.append(f'<option value="{stage}">{stage}</option>')
    parts.append('''</select></label><img id="overview" src="figures/heatmap_counts.png"><p><a id="download" href="figures/heatmap_counts.png">Open labeled figure</a></p><details><summary>Zoom to individual pixels (scroll horizontally and vertically)</summary><label>Zoom <input id="zoom" type="range" min="1" max="5" value="1" oninput="showStage()"></label><div class="zoom"><img id="pixels" src="figures/heatmap_counts_pixels.png"></div></details></section>
<script>function showStage(){let s=document.getElementById('stage').value;document.getElementById('overview').src='figures/heatmap_'+s+'.png';document.getElementById('download').href='figures/heatmap_'+s+'.png';let p=document.getElementById('pixels');p.src='figures/heatmap_'+s+'_pixels.png';p.style.width=('''+str(meta["peaks"])+'''*document.getElementById('zoom').value)+'px';p.style.imageRendering='pixelated';}showStage();</script>''')
    parts.append(tic_report_section(output, manifest["fingerprint"]))
    if "cohort" in manifest:
        decisions = pd.DataFrame(manifest["cohort"]["decisions"])
        parts.insert(1, '<section><h2>Manual cohort QC</h2><p>Only retained slides contribute to fitting, exports and reports. '
                     'These signal-summary and heatmap decisions do not establish a technical cause. '
                     '<a href="cohort.csv">Complete cohort decision table</a></p>' +
                     decisions.loc[~decisions.included].to_html(index=False) + '</section>')
    parts.append('''<section><h2>Biological structure and slide-associated variation</h2><p>Matched maps below retain native pixels and tissue gaps. CD8/stroma associations are evaluated after cleaning over the retained lesion. Association preservation is descriptive, not proof of biological preservation. Slide-associated variation includes patient biology as well as acquisition effects; lower variation alone does not establish a better representation.</p>''')
    parts.append(pd.DataFrame([summary["median_slide_eta_squared_common_features"]]).to_html(index=False, float_format=lambda v: f"{v:.6f}"))
    if summary.get("biological_associations"):
        parts.append(pd.DataFrame(summary["biological_associations"]).to_html(index=False, float_format=lambda v: f"{v:.6f}"))
    for path in sorted((output / "figures").glob("slide_*.png")):
        parts.append(f'<details><summary>{escape(path.stem)}</summary><a href="figures/{path.name}"><img loading="lazy" src="figures/{path.name}"></a></details>')
    parts.append('<p><a href="biological_associations.csv">Association table</a> · <a href="slide_variation.csv">Slide-associated variation</a> · <a href="validation_summary.json">Machine-readable summary</a></p></section><p>Preprocessing cannot recover information lost during RMS normalization or alignment.</p></body></html>')
    (output / "report.html").write_text("\n".join(parts))
    return output / "report.html"

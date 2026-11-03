"""Descriptive comparisons; report annotations never enter preprocessing fits."""
from __future__ import annotations

from pathlib import Path
import argparse

import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


def example_peaks(state: dict, slide: str, maximum: int) -> tuple[np.ndarray, list[str]]:
    """Choose informative fitted examples without using biological labels."""
    i = list(state["slides"]).index(slide)
    retained = np.isin(np.arange(len(state["features"])), state["retained"])
    abundance = state["median"][i] * state["prevalence"][i]
    supported, patches = state["supported"][i], state["patches"][i]
    groups = [
        ("reference anchor", np.isin(np.arange(len(retained)), state["anchors"]), abundance),
        ("spatial support", retained & supported, state["raw_excess"][i]),
        ("localized support", retained & patches & ~supported, abundance),
        ("weak retained", retained & (state["weak_fraction"][i] >= .5), abundance),
        ("removed peak", ~retained, abundance),
        ("retained comparison", retained, abundance),
    ]
    examples, reasons = [], []
    for reason, eligible, score in groups:
        candidates = np.flatnonzero(eligible & (state["prevalence"][i] > 0) & np.isfinite(score))
        for j in candidates[np.argsort(-score[candidates], kind="stable")]:
            if int(j) not in examples:
                examples.append(int(j))
                reasons.append(reason)
                break
        if len(examples) == maximum:
            break
    if not examples:
        examples, reasons = [int(state["retained"][0])], ["no fitting detections"]
    return np.array(examples), reasons


def plot_maps(path: Path, slide: str, coordinates: np.ndarray, names: np.ndarray,
              reasons: list[str], matrices: dict[str, np.ndarray]) -> None:
    """Render matched native-grid pixels, with holes masked and shared linear scales."""
    xy = coordinates.astype(np.int64) - coordinates.min(axis=0).astype(np.int64)
    shape = (int(xy[:, 1].max()) + 1, int(xy[:, 0].max()) + 1)
    native_grid = shape[0] * shape[1] <= 16_000_000
    fig, axes = plt.subplots(len(names), len(matrices), squeeze=False,
                             figsize=(3 * len(matrices), 2.7 * len(names)), constrained_layout=True)
    for row, (feature, reason) in enumerate(zip(names, reasons)):
        positives = [v[:, row][v[:, row] > 0] for stage, v in matrices.items() if stage != "old_X"]
        values = np.concatenate(positives)
        vmax = max(float(np.quantile(values, .99)) if len(values) else 1., 1e-12)
        for col, (stage, values) in enumerate(matrices.items()):
            values = values[:, row]
            limit = max(float(np.quantile(values[values > 0], .99)), 1e-12) if stage == "old_X" and np.any(values > 0) else vmax
            ax = axes[row, col]
            if native_grid:
                grid = np.full(shape, np.nan)
                grid[xy[:, 1], xy[:, 0]] = values
                artist = ax.imshow(grid, origin="upper", interpolation="nearest", cmap="viridis", vmin=0, vmax=limit)
            else:
                artist = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=.25, cmap="viridis", vmin=0, vmax=limit, rasterized=True)
                ax.invert_yaxis()
            ax.set_title(f"{stage}: {feature}\n{reason}", fontsize=8)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(artist, ax=ax, shrink=.7)
    fig.suptitle(slide)
    fig.savefig(path, dpi=130)
    plt.close(fig)


class Diagnostics:
    """Accumulate exact distributions and slide variance with bounded slide memory."""

    def __init__(self, output: Path, features: np.ndarray, retained: np.ndarray, peaks: int, state: dict):
        self.output, self.features, self.retained = output, features, retained
        self.state, self.peaks = state, peaks
        self.rows, self.histograms = [], []
        self.biology_header = True
        self.moments: dict[str, list] = {}
        self.bins = np.r_[0, np.geomspace(.001, 1000, 80), np.inf]
        (output / "figures").mkdir(exist_ok=True)

    def add(self, slide: str, coordinates: np.ndarray, matrices: dict[str, sparse.csr_matrix],
            lesion_positions: np.ndarray, labels: pd.DataFrame, fit_mask: np.ndarray) -> None:
        """Compare identical tissue positions, plus held-out biological associations."""
        biology = []
        for stage, matrix in matrices.items():
            n, p = matrix.shape
            positive = matrix.data
            sums = np.asarray(matrix.sum(axis=0)).ravel()
            squares = np.asarray(matrix.power(2).sum(axis=0)).ravel()
            self.moments.setdefault(stage, []).append((n, sums, squares))
            self.rows.append(dict(slide=slide, stage=stage, rows=n, features=p, positive_measurements=matrix.nnz,
                                  zero_fraction=1-matrix.nnz/max(n*p, 1), intensity_sum=float(positive.sum()),
                                  positive_mean=float(positive.mean()) if len(positive) else 0,
                                  maximum=float(positive.max()) if len(positive) else 0))
            hist = np.zeros(len(self.bins)-1, dtype=np.int64)
            # Bounded bin lookup avoids repeatedly sorting millions of values
            # for the unequal-width intensity bins used by np.histogram.
            for start in range(0, len(positive), 1_000_000):
                bins = np.searchsorted(self.bins, positive[start:start+1_000_000], side="right") - 1
                hist += np.bincount(bins, minlength=len(hist))
            self.histograms.extend(dict(slide=slide, stage=stage, lower=float(lo), upper=float(hi), count=int(v))
                                   for lo, hi, v in zip(self.bins[:-1], self.bins[1:], hist))
            feature_ids = np.arange(len(self.features)) if p == len(self.features) else self.retained
            for name, mask in (("train", fit_mask), ("validation", ~fit_mask)):
                if mask.sum() < 3:
                    continue
                spectra = matrix[lesion_positions[mask]]
                for label in labels.columns:
                    y = labels[label].to_numpy(dtype=float)[mask]
                    valid = np.isfinite(y)
                    if valid.sum() < 3 or np.std(y[valid]) == 0:
                        continue
                    a = spectra[valid]
                    y = y[valid] - y[valid].mean()
                    sa = np.asarray(a.sum(axis=0)).ravel()
                    variance = np.maximum(0, np.asarray(a.power(2).sum(axis=0)).ravel() - sa**2/len(y))
                    numerator = np.asarray(a.T @ y).ravel()
                    denom = np.sqrt(variance * np.dot(y, y))
                    correlation = np.divide(numerator, denom, out=np.full(len(sa), np.nan), where=denom > 1e-12)
                    biology.extend(dict(slide=slide, stage=stage, split=name, label=label,
                                             feature=self.features[j], correlation=float(v))
                                        for j, v in zip(feature_ids, correlation))
        if biology:
            pd.DataFrame(biology).to_csv(self.output / "biological_associations.csv", index=False,
                                        mode="w" if self.biology_header else "a", header=self.biology_header)
            self.biology_header = False
        examples, reasons = example_peaks(self.state, slide, self.peaks)
        plotted = {}
        for stage, matrix in matrices.items():
            columns = []
            for feature in examples:
                if matrix.shape[1] == len(self.features):
                    columns.append(matrix[:, feature].toarray().ravel())
                elif feature in self.retained:
                    columns.append(matrix[:, np.searchsorted(self.retained, feature)].toarray().ravel())
                else:
                    columns.append(np.zeros(matrix.shape[0]))
            plotted[stage] = np.column_stack(columns)
        plot_maps(self.output / "figures" / f"slide_{len(self.moments['counts']):03d}.png", slide,
                  coordinates, self.features[examples], reasons, plotted)

    def finish(self) -> None:
        """Write stage distributions, associations and descriptive slide eta-squared."""
        pd.DataFrame(self.rows).to_csv(self.output / "stage_summaries.csv", index=False)
        hist = pd.DataFrame(self.histograms)
        hist.to_csv(self.output / "intensity_histograms.csv", index=False)
        variation = []
        for stage, moments in self.moments.items():
            sizes = np.array([r[0] for r in moments])
            sums, squares = np.stack([r[1] for r in moments]), np.stack([r[2] for r in moments])
            mean = sums.sum(axis=0) / sizes.sum()
            between = (sizes[:, None] * (sums / sizes[:, None] - mean)**2).sum(axis=0)
            total = np.maximum(0, squares.sum(axis=0) - sizes.sum() * mean**2)
            eta = np.divide(between, total, out=np.full_like(total, np.nan), where=total > 1e-12)
            indices = np.arange(len(self.features)) if len(total) == len(self.features) else self.retained
            variation.extend(dict(stage=stage, feature=self.features[j], slide_eta_squared=float(v)) for j, v in zip(indices, eta))
        pd.DataFrame(variation).to_csv(self.output / "slide_variation.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for stage, group in hist.groupby("stage", sort=False):
            summed = group.groupby("lower")["count"].sum()
            ax.plot(summed.index.to_numpy()[1:], summed.to_numpy()[1:] / max(summed.sum(), 1), label=stage)
        ax.set(xscale="log", yscale="log", xlabel="Stored positive intensity (linear units)", ylabel="Fraction per histogram bin")
        ax.legend()
        fig.savefig(self.output / "figures" / "intensity_distributions.png", dpi=150)
        plt.close(fig)
        (self.output / "report.html").write_text('''<!doctype html><meta charset="utf-8"><title>MSI preprocessing</title>
<h1>MSI preprocessing comparisons</h1><p>Counts already contain RMS normalization.
X contains linear reference-adjusted intensities; DANN applies log1p once.</p>
<p>Thresholds and 95% random-location envelopes are operational, descriptive filters,
not an instrumental noise floor or calibrated hypothesis tests. Missing information
from RMS normalization and alignment cannot be recovered.</p>
<p>Fits use only DANN training rows. Application uses whole-tissue MSI neighborhoods,
including held-out and non-lesion image context; labels are never used for cleaning.
Biological-association tables use annotations only after preprocessing, with the
original train/validation partition. Correlation preservation is a diagnostic,
not proof of biological preservation. Slide variation confounds biology and acquisition.</p>
<p>See peak_decisions.csv, slide_multipliers.csv, stage_summaries.csv,
intensity_histograms.csv, local_sensitivity.csv, local_removal.csv,
biological_associations.csv and slide_variation.csv. Matched maps share the
counts-derived color scale; old X has a separate scale.</p>
<img src="figures/intensity_distributions.png" width="800">''' +
            "".join(f'<p>{path.stem}</p><img src="figures/{path.name}" width="1200">' for path in sorted((self.output / "figures").glob("slide_*.png"))))


def render_maps(fitted: str | Path) -> None:
    """Regenerate informative matched maps from completed exports, without refitting."""
    from .fit import load_fit, log
    from .io import axes, read_rows
    manifest, state = load_fit(fitted)
    config = manifest["config"]
    output = Path(fitted).resolve().parent
    data = manifest["coordinate_columns"]
    obs, _ = axes(output / "tissue.h5ad", data, config["patient_column"])
    batch = np.asarray(obs[data["batch_column"]], dtype=str)
    decisions = []
    with h5py.File(config["tissue_input"], "r") as source, h5py.File(output / "tissue.h5ad", "r") as result:
        for index, slide in enumerate(dict.fromkeys(batch)):
            i = list(state["slides"]).index(slide)
            rows = np.flatnonzero(batch == slide)
            examples, reasons = example_peaks(state, slide, config["report_peaks"])
            matrices = {}
            for stage, key in (("counts", "layers/counts"), ("old_X", "X")):
                matrices[stage] = read_rows(source[key], rows, config["block_rows"])[:, examples].toarray()
            retained = np.isin(examples, state["retained"])
            matrices["selected"] = matrices["counts"].copy()
            matrices["selected"][:, ~retained] = 0
            matrices["integrated"] = matrices["selected"] / state["multipliers"][i]
            matrices["cleaned"] = np.zeros_like(matrices["integrated"])
            columns = np.searchsorted(state["retained"], examples[retained])
            matrices["cleaned"][:, retained] = read_rows(result["X"], rows, config["block_rows"])[:, columns].toarray()
            coordinates = obs.iloc[rows][[data["x_column"], data["y_column"]]].to_numpy()
            plot_maps(output / "figures" / f"slide_{index+1:03d}.png", slide, coordinates, state["features"][examples], reasons, matrices)
            decisions.extend(dict(slide=slide, feature=state["features"][j], reason=reason) for j, reason in zip(examples, reasons))
            log(f"Rendered informative map {index+1}: {slide}")
    pd.DataFrame(decisions).to_csv(output / "map_examples.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate matched spatial maps from completed preprocessing exports.")
    parser.add_argument("--fit", default="data/PDAC/Processed/msi_preprocess/fit.json")
    render_maps(parser.parse_args().fit)

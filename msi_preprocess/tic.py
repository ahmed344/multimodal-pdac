"""Whole-tissue TIC comparisons from frozen retained-feature exports."""
from __future__ import annotations

from pathlib import Path

from anndata.io import read_elem
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
import numpy as np

from .fit import load_fit, log
from .io import axes, identity, json_write
from .qc import read_csr_range


def sum_tic(group: h5py.Group, block_rows: int) -> np.ndarray:
    """Sum linear intensities per pixel in float64, holding one sparse block."""
    if block_rows < 1:
        raise ValueError("TIC block_rows must be positive.")
    if group.attrs.get("encoding-type") != "csr_matrix":
        raise ValueError("TIC matrices must be CSR encoded.")
    n = int(group.attrs["shape"][0])
    totals = np.empty(n, dtype=np.float64)
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        block = read_csr_range(group, start, stop)
        totals[start:stop] = np.asarray(block.sum(axis=1, dtype=np.float64)).ravel()
    if not np.isfinite(totals).all():
        raise ValueError("Nonfinite TIC after summing intensities.")
    return totals


def plot_tic_grid(path: Path, coordinates: np.ndarray, slide_rows: dict[str, np.ndarray],
                  values: np.ndarray, norm: Normalize, title: str) -> None:
    """Render every slide on native coordinates with one shared log1p scale."""
    rows = (len(slide_rows) + 4) // 5
    fig = plt.figure(figsize=(20, 3 * rows), layout="constrained")
    layout = fig.add_gridspec(rows, 6, width_ratios=[1, 1, 1, 1, 1, .06])
    colors = colormaps["turbo"].copy()
    colors.set_bad("white")
    try:
        for index, (slide, positions) in enumerate(slide_rows.items()):
            ax = fig.add_subplot(layout[index // 5, index % 5])
            xy = coordinates[positions].astype(np.int64)
            xy -= xy.min(axis=0)
            shape = (int(xy[:, 1].max()) + 1, int(xy[:, 0].max()) + 1)
            display = np.log1p(values[positions])
            if shape[0] * shape[1] <= 16_000_000:
                grid = np.full(shape, np.nan)
                grid[xy[:, 1], xy[:, 0]] = display
                artist = ax.imshow(grid, origin="upper", interpolation="nearest", cmap=colors, norm=norm)
            else:
                artist = ax.scatter(xy[:, 0], xy[:, 1], c=display, s=.25,
                                    cmap=colors, norm=norm, rasterized=True)
                ax.invert_yaxis()
            ax.set(title=slide, xticks=[], yticks=[], aspect="equal")
            ax.title.set_fontsize(11)
        # Unused cells have no axes, labels, or colorbars.
        cb = fig.colorbar(artist, cax=fig.add_subplot(layout[:, 5]), extend="max")
        ticks = np.linspace(norm.vmin, norm.vmax, 6)
        cb.set_ticks(ticks, labels=[f"{v:.4g}" for v in np.expm1(ticks)])
        cb.set_label("TIC: sum of linear intensities (log1p color spacing; shared scale)")
        fig.suptitle(title, fontsize=17)
        fig.savefig(path, dpi=250)
    finally:
        plt.close(fig)


def render_tic_maps(fitted: str | Path) -> dict:
    """Write two all-slide TIC PNGs and metadata without changing fitted exports."""
    manifest, state = load_fit(fitted)
    output = Path(fitted).resolve().parent
    source = output / "tissue.h5ad"
    config, data = manifest["config"], manifest["coordinate_columns"]
    obs, features = axes(source, data, config["patient_column"])
    if not np.array_equal(features, state["features"][state["retained"]]):
        raise ValueError("TIC export feature order differs from fitted retained features.")
    if not len(obs):
        raise ValueError("TIC maps require tissue pixels.")
    sources = dict(counts="layers/counts", cleaned="X")
    totals = {}
    with h5py.File(source, "r") as handle:
        if str(read_elem(handle["uns/msi_preprocess/fingerprint"])) != manifest["fingerprint"]:
            raise ValueError("TIC export fingerprint differs from fitted state.")
        if str(read_elem(handle["uns/msi_preprocess/intensity_scale"])) != "linear":
            raise ValueError("TIC requires linear exported intensities.")
        for stage, key in sources.items():
            if tuple(handle[key].attrs["shape"]) != (len(obs), len(features)):
                raise ValueError("TIC matrix shape differs from tissue metadata.")
            log(f"Summing whole-tissue TIC: {stage}")
            totals[stage] = sum_tic(handle[key], config["block_rows"])
    slide_rows = dict(sorted((str(slide), positions) for slide, positions in
        obs.groupby(data["batch_column"], observed=True, sort=False).indices.items()))
    coordinates = obs[[data["x_column"], data["y_column"]]].to_numpy()
    percentile = config.get("report", {}).get("upper_percentile", 99.5)
    positive = totals["counts"][totals["counts"] > 0]
    upper = float(np.percentile(positive, percentile)) if len(positive) else 1.
    norm = Normalize(vmin=0, vmax=np.log1p(upper), clip=True)
    grid_rows = (len(slide_rows) + 4) // 5
    metadata = dict(fingerprint=manifest["fingerprint"], source=identity(source),
        matrix_sources=sources, region="whole_tissue", pixels=len(obs),
        retained_features=len(features), feature_scope="all retained peaks in export order",
        aggregation="float64 sum of linear intensities per pixel before log1p display",
        slide_order=list(slide_rows), cmap="turbo", display="log1p color spacing only",
        lower_linear=0., upper_linear=upper, upper_percentile=percentile,
        upper_population="positive counts TIC across all whole-tissue pixels",
        empty_counts_scale_fallback=not len(positive), missing_tissue_color="white",
        grid=[grid_rows, 5], figure_inches=[20, 3 * grid_rows], figure_dpi=250,
        image_pixels=[5000, 750 * grid_rows], stages={})
    (output / "figures").mkdir(exist_ok=True)
    for stage, values in totals.items():
        filename = f"figures/tic_{stage}_all_slides.png"
        label = "Retained-counts TIC" if stage == "counts" else "Cleaned TIC"
        units = "RMS-normalized counts" if stage == "counts" else "Slide-scaled intensities after local removal"
        title = (f"{label} · {len(slide_rows)} whole-tissue slides\n"
                 f"{len(features):,} retained peaks per pixel · {units}")
        plot_tic_grid(output / filename, coordinates, slide_rows, values, norm, title)
        statistics = []
        for slide, positions in slide_rows.items():
            selected = values[positions]
            saturated = int(np.count_nonzero(selected > upper))
            statistics.append(dict(slide=slide, pixels=len(positions),
                intensity_sum=float(selected.sum()), minimum=float(selected.min()),
                maximum=float(selected.max()), zero_pixels=int(np.count_nonzero(selected == 0)),
                saturated_pixels=saturated, fraction_saturated=saturated / len(positions)))
        metadata["stages"][stage] = dict(figure=filename, lower_linear=0., upper_linear=upper,
            pixels=len(values), intensity_sum=float(values.sum()),
            saturated_pixels=int(np.count_nonzero(values > upper)), slides=statistics)
        log(f"Rendered {label}: {output / filename}")
    json_write(output / "tic_maps.json", metadata)
    return metadata

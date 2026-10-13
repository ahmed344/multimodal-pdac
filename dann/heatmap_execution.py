"""Selective loading, shared grids, and bounded CPU spatial figure rendering."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid
from typing import Any, Mapping, Sequence

import anndata as ad
from anndata.io import read_elem
import h5py
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

GIB = 1024 ** 3
_WORKER_DATA = None
_THREAD_LIMIT = None


class Progress:
    """Report wall time and parent/worker memory even during long savefig calls."""

    def __init__(self):
        self.started = time.perf_counter()
        self.phase = "startup"
        self.phase_started = self.started
        self.durations = {}
        self.done = 0
        self.total = 0
        self.peak_rss = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self):
        root = psutil.Process()
        memory = 0
        for process in [root, *root.children(recursive=True)]:
            try:
                memory += process.memory_info().rss
            except psutil.Error:
                pass
        self.peak_rss = max(memory, self.peak_rss)
        return memory

    def report(self):
        elapsed = time.perf_counter() - self.phase_started
        memory = self.sample()
        print(f"[heatmaps {self.phase}] {self.done}/{self.total} completed; "
              f"{self.done / max(elapsed, .001):.2f}/s; {elapsed:.1f}s; "
              f"parent+worker RSS {memory / GIB:.2f} GiB", flush=True)

    def _loop(self):
        while not self.stop.wait(5):
            self.report()

    def start(self):
        self.thread.start()
        self.report()

    def set_phase(self, name, total):
        now = time.perf_counter()
        self.durations[self.phase] = now - self.phase_started
        self.phase, self.phase_started, self.done, self.total = name, now, 0, total
        self.report()

    def close(self):
        self.stop.set()
        self.thread.join()
        self.durations[self.phase] = time.perf_counter() - self.phase_started
        self.sample()


def load_plot_data(
    input_path: Path, predictions_path: Path, densities: Sequence[str],
    batch_column: str, settings: Mapping[str, Any],
) -> ad.AnnData:
    """Read plotting fields only; never access spectra, layers, or HES features."""
    from dann.spatial_heatmaps import attach_predictions
    from dann.targets import prepare_observed_targets
    from dann.tiles import _axis_names

    schema = pq.read_schema(predictions_path)
    metadata = schema.metadata or {}
    contract = json.loads(metadata[b"dann_target_transform"]) if b"dann_target_transform" in metadata else None
    if contract and contract.get("version") != 1:
        raise ValueError("Unsupported spatial target transformation version.")
    observed = list(dict.fromkeys([*densities, *(contract["target_columns"] if contract else [])]))
    columns = [batch_column, "x", "y", *observed]
    if "lesion" in settings["regions"]:
        columns.append(settings["lesion_column"])
    with h5py.File(input_path, "r") as handle:
        missing = set(columns) - set(handle["obs"])
        if missing:
            raise ValueError(f"Missing observation columns: {sorted(missing)}. "
                             "Lesion plots require lesion annotations; use --regions tissue for tissue only.")
        frame = pd.DataFrame({key: read_elem(handle["obs"][key]) for key in dict.fromkeys(columns)})
        frame.index = pd.Index(_axis_names(handle["obs"]))
        for key in (batch_column, "x", "y"):
            if frame[key].isna().any():
                raise ValueError(f"Missing spatial metadata in {key}.")
        xy = frame[["x", "y"]].to_numpy()
        if not np.isfinite(xy).all():
            raise ValueError("Spatial coordinates must be finite.")
        if frame.duplicated([batch_column, "x", "y"]).any():
            raise ValueError("Duplicate (batch, x, y) grid positions.")
        if "lesion" in settings["regions"]:
            lesion = frame[settings["lesion_column"]]
            if not pd.api.types.is_bool_dtype(lesion.dtype) or lesion.isna().any():
                raise ValueError("Lesion annotations must be nonmissing boolean values.")
        result = ad.AnnData(obs=frame)
        result.obsm["spatial"] = read_elem(handle["obsm/spatial"])
        if result.obsm["spatial"].shape != (len(frame), 2) or not np.isfinite(result.obsm["spatial"]).all():
            raise ValueError("obsm/spatial must contain finite two-dimensional coordinates.")
        result.uns["spatial"] = {}
        for batch in frame[batch_column].unique():
            library = handle["uns/spatial"][str(batch)]
            result.uns["spatial"][batch] = {
                "images": {"HES": read_elem(library["images/HES"])},
                "scalefactors": read_elem(library["scalefactors"]),
            }
    prefixes = ("mu_", "prob_of_presence_", "sigma_", "density_mean_", "logit_mean_", "logit_q05_", "logit_q95_")
    prediction_columns = [prefix + density for density in densities for prefix in prefixes]
    missing = set(prediction_columns) - set(schema.names)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}; rerun python -m dann.spatial to regenerate predictions.")
    predictions = pq.read_table(predictions_path, columns=["row_position", "obs_name", *prediction_columns]).to_pandas()
    attach_predictions(result, predictions)
    if contract:
        columns = contract["target_columns"]
        raw = result.obs[columns].to_numpy(copy=True)
        targets, valid = prepare_observed_targets(raw, columns, contract["cd8_normalization"])
        for i, column in enumerate(columns):
            result.obs[f"raw_{column}"] = raw[:, i]
            result.obs[f"valid_{column}"] = valid[:, i]
            result.obs[column] = np.where(valid[:, i], targets[:, i], np.nan)
        result.uns["dann_target_transform"] = contract
    return result


def prepare_grids(
    adata: ad.AnnData, densities: Sequence[str], epsilon: float, batch_column: str,
    settings: Mapping[str, Any], directory: Path, progress: Progress | None = None,
) -> dict[str, Any]:
    """Persist one set of numerical grids shared by all figure tasks."""
    from dann.spatial_heatmaps import add_logit_columns, add_residual_columns, add_interval_width_columns
    add_logit_columns(adata, densities, epsilon)
    add_logit_columns(adata, [f"density_mean_{d}" for d in densities], epsilon)
    add_residual_columns(adata, densities)
    add_interval_width_columns(adata, densities)
    manifest = {"slides": [], "contract": adata.uns.get("dann_target_transform"), "bytes": 0}

    def save(name, values):
        path = directory / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        manifest["bytes"] += path.stat().st_size
        return str(path)

    # Group once in first-observation order, never once per panel.
    groups = adata.obs.groupby(batch_column, sort=False, observed=True).indices
    for i, batch in enumerate(adata.obs[batch_column].unique()):
        rows = groups[batch]
        frame = adata.obs.iloc[rows]
        x, xi = np.unique(frame.x.to_numpy(), return_inverse=True)
        y, yi = np.unique(frame.y.to_numpy(), return_inverse=True)
        shape = (len(y), len(x))
        positions = yi * len(x) + xi
        lesion = np.zeros(shape, dtype=bool)
        bounds = None
        coordinates = np.asarray(adata.obsm["spatial"])[rows]
        if "lesion" in settings["regions"]:
            mask = frame[settings["lesion_column"]].to_numpy(dtype=bool)
            lesion.flat[positions] = mask
            if mask.any():
                bounds = [int(yi[mask].min()), int(yi[mask].max()) + 1,
                          int(xi[mask].min()), int(xi[mask].max()) + 1]
                lesion_hes = np.vstack([coordinates[mask].min(axis=0), coordinates[mask].max(axis=0)]).tolist()
            else:
                lesion_hes = None
        else:
            lesion_hes = None
        library = adata.uns["spatial"][batch]
        slide = dict(batch=str(batch), lesion=save(f"{i}-lesion", lesion), crop=bounds,
                     tissue_hes=np.vstack([coordinates.min(axis=0), coordinates.max(axis=0)]).tolist(),
                     lesion_hes=lesion_hes, image=save(f"{i}-HES", library["images"]["HES"]),
                     scalefactors=library["scalefactors"], grids={})
        for j, density in enumerate(densities):
            positive = frame[density].to_numpy() > 0
            mean_positive = frame[f"density_mean_{density}"].to_numpy() > 0
            columns = [f"logit_{density}", f"mu_{density}", f"prob_of_presence_{density}",
                       f"sigma_{density}", f"logit_density_mean_{density}",
                       f"logit_mean_{density}", f"residual_{density}", f"interval_width_{density}"]
            grids = np.full((8, *shape), np.nan, dtype=np.float32)
            for k, column in enumerate(columns):
                values = frame[column].to_numpy(dtype=np.float32, copy=True)
                if k in (0, 5, 6, 7):
                    values[~positive] = np.nan
                elif k == 4:
                    values[~mean_positive] = np.nan
                grids[k].flat[positions] = values
            slide["grids"][density] = save(f"{i}-{j}-grids", grids)
        manifest["slides"].append(slide)
        if progress:
            progress.done += 1
    return manifest


def _load_worker(manifest):
    global _WORKER_DATA, _THREAD_LIMIT
    from threadpoolctl import threadpool_limits
    _THREAD_LIMIT = threadpool_limits(limits=1)
    # Every task reuses these read-only maps; no large arrays cross process pipes.
    _WORKER_DATA = {**manifest, "slides": []}
    for slide in manifest["slides"]:
        _WORKER_DATA["slides"].append({**slide,
            "image": np.load(slide["image"], mmap_mode="r"),
            "lesion": np.load(slide["lesion"], mmap_mode="r"),
            "grids": {key: np.load(path, mmap_mode="r") for key, path in slide["grids"].items()},
        })


def region_grid(
    grids: np.ndarray, lesion: np.ndarray, crop: Sequence[int] | None, index: int, region: str,
) -> np.ndarray:
    """Slice the original grid, retaining interior gaps when zooming to a lesion."""
    values = grids[index]
    if region == "tissue":
        return values
    if crop is None:
        return np.full((1, 1), np.nan, dtype=np.float32)
    y0, y1, x0, x1 = crop
    return np.where(lesion[y0:y1, x0:x1], values[y0:y1, x0:x1], np.nan)


def validate_png(path: Path | str, expected_size: Sequence[int] | None = None) -> list[int]:
    """Check complete PNG chunks and dimensions without decoding a giant canvas."""
    from PIL import Image
    previous = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None  # Trusted images generated by this invocation.
        with Image.open(path) as image:
            if image.format != "PNG" or (expected_size and image.size != tuple(expected_size)):
                raise ValueError(f"Invalid PNG dimensions or format: {path}")
            size = image.size
            image.verify()
        return list(size)
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def render_task(task: tuple[str, str, str, str, int, int]) -> dict[str, Any]:
    """Render a single figure with independent process-local Matplotlib state."""
    from dann import spatial_heatmaps as plots
    from dann.targets import target_label
    plt = plots.plt
    data = _WORKER_DATA
    density, kind, region, destination, dpi, compression = task
    diagnostic = kind == "diagnostics"
    ncols = 5 if diagnostic else 6
    width = 30 if diagnostic else 36
    indices = [0, 5, 6, 7] if diagnostic else [0, 1, 2, 3, 4]
    titles = ([f"logit_{density} (observed)", f"logit_mean_{density}", f"residual_{density}",
               f"interval_width_{density}", "HES"] if diagnostic else
              [f"logit_{density}", f"mu_{density}", f"prob_of_presence_{density}", f"sigma_{density}",
               f"logit(sampled mean density): {density}", "HES"])
    figure = None
    temporary = Path(destination).with_name(f".{Path(destination).name}.{uuid.uuid4().hex}.tmp")
    started = time.perf_counter()
    try:
        figure, axes = plt.subplots(nrows=len(data["slides"]), ncols=ncols,
                                   figsize=(width, 4 * len(data["slides"])), tight_layout=True, squeeze=False)
        for row, slide in enumerate(data["slides"]):
            empty = region == "lesion" and slide["crop"] is None
            for col, index in enumerate(indices):
                values = region_grid(slide["grids"][density], slide["lesion"], slide["crop"], index, region)
                options = dict(cmap="jet", origin="upper")
                if diagnostic and index == 6:
                    finite = values[np.isfinite(values)]
                    limit = max(float(np.abs(finite).max()), 1e-6) if finite.size else 1.
                    options.update(cmap="coolwarm", vmin=-limit, vmax=limit)
                image = axes[row, col].imshow(values, **options)
                figure.colorbar(image, ax=axes[row, col])
            axes[row, 0].set_ylabel(slide["batch"], fontsize=12)
            if empty:
                for axis in axes[row]:
                    axis.text(.5, .5, "No lesion", ha="center", va="center", transform=axis.transAxes)
            else:
                bounds = ad.AnnData(obs=pd.DataFrame(index=["minimum", "maximum"]))
                bounds.obsm["spatial"] = np.asarray(slide[f"{region}_hes"])
                bounds.uns["spatial"] = {slide["batch"]: {
                    "images": {"HES": slide["image"]}, "scalefactors": slide["scalefactors"]}}
                plots._plot_hes_context(bounds, pd.Series([True, True]), slide["batch"], axes[row, -1])
        label = target_label(density, data["contract"])
        for axis, title in zip(axes[0], titles):
            axis.set_title(title.replace(density, label), fontsize=12)
        for axis in axes.flat:
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor("gray")
        for axis in axes[:, -1]:
            axis.invert_yaxis()
            axis.invert_xaxis()
        figure.savefig(temporary, format="png", dpi=dpi, pil_kwargs={"compress_level": compression})
        size = validate_png(temporary, [width * dpi, 4 * len(data["slides"]) * dpi])
        temporary.replace(destination)
        return dict(path=str(destination), size=size, seconds=time.perf_counter() - started,
                    bytes=Path(destination).stat().st_size)
    finally:
        if figure is not None:
            plt.close(figure)
        temporary.unlink(missing_ok=True)
        gc.collect()


@contextmanager
def single_thread_environment():
    """Limit numerical pools before spawned workers import NumPy/Scanpy."""
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(dict.fromkeys(keys, "1"))
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def choose_workers(
    requested: int, tasks: int, shared_bytes: int, nslides: int, dpi: int, budget_gib: float,
) -> tuple[int, dict[str, float]]:
    """Reserve four canvas buffers, image copies, and process/import overhead."""
    available = min(budget_gib * GIB, psutil.virtual_memory().available * .85)
    canvas = 36 * dpi * 4 * nslides * dpi * 4
    # Imshow owns one private copy of each image/grid used by the figure.
    # Reserving the entire shared set covers those copies even though each task
    # touches only one target. Four canvas buffers separately cover drawing,
    # resampling, and PNG encoding scratch.
    per_worker = 4 * canvas + shared_bytes + GIB
    affordable = int((available - shared_bytes - GIB) // per_worker)
    if affordable < 1:
        raise MemoryError(f"Heatmap budget cannot fit one renderer: shared={shared_bytes / GIB:.2f} GiB, "
                          f"worker estimate={per_worker / GIB:.2f} GiB, usable={available / GIB:.2f} GiB.")
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    workers = min(requested, tasks, cpus, affordable)
    return workers, dict(shared_gib=shared_bytes / GIB, per_worker_gib=per_worker / GIB, usable_gib=available / GIB)


def source_identity(path: Path) -> dict[str, Any]:
    """Record source identity without reading the large file payload."""
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def run_heatmaps(
    input_path: Path, predictions_path: Path, output_dir: Path, densities: Sequence[str],
    epsilon: float, dpi: int, batch_column: str, settings: Mapping[str, Any],
) -> Path:
    """Prepare once, render with bounded processes, and validate all outputs."""
    from dann.config import spatial_heatmap_settings
    global _WORKER_DATA, _THREAD_LIMIT
    settings = spatial_heatmap_settings(settings)
    if not densities or len(set(densities)) != len(densities):
        raise ValueError("Select at least one distinct density.")
    if not 0 < epsilon < .5 or isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("Require 0 < logit_epsilon < 0.5 and a positive integer DPI.")
    identities = {"input": source_identity(input_path), "predictions": source_identity(predictions_path)}
    progress = Progress()
    progress.start()
    results = []
    token = uuid.uuid4().hex
    try:
        # Reject impossible budgets before reading the plotting arrays.
        with h5py.File(input_path, "r") as handle:
            nrows = handle["obsm/spatial"].shape[0]
            batches = pd.unique(read_elem(handle["obs"][batch_column]))
            images = [handle["uns/spatial"][str(batch)]["images/HES"] for batch in batches]
            image_bytes = sum(int(np.prod(image.shape)) * image.dtype.itemsize for image in images)
            estimated_shared = nrows * (len(densities) * 8 * 4 * 2 + 128) + image_bytes
            choose_workers(1, 1, estimated_shared, len(batches), dpi, settings["host_budget_gib"])
        adata = load_plot_data(input_path, predictions_path, densities, batch_column, settings)
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".heatmaps-{token}-", dir=output_dir) as directory:
            directory = Path(directory)
            progress.set_phase("preparation", adata.obs[batch_column].nunique())
            manifest = prepare_grids(adata, densities, epsilon, batch_column, settings, directory, progress)
            del adata
            gc.collect()
            tasks = []
            for density in densities:
                for kind in ("parameters", "diagnostics"):
                    for region in settings["regions"]:
                        prefix = "spatial_heatmap_diagnostics" if kind == "diagnostics" else "spatial_heatmap"
                        name = f"{prefix}_{density}_{region}.png"
                        # All workers publish into an invocation-owned staging directory.
                        tasks.append((density, kind, region, str(directory / name), dpi, settings["png_compress_level"]))
            workers, memory = choose_workers(settings["workers"], len(tasks), manifest["bytes"],
                                              len(manifest["slides"]), dpi, settings["host_budget_gib"])
            print(f"[heatmaps] using {workers}/{settings['workers']} workers; memory estimates {memory}", flush=True)
            progress.set_phase("rendering", len(tasks))
            with single_thread_environment():
                if workers == 1:
                    try:
                        _load_worker(manifest)
                        for task in tasks:
                            results.append(render_task(task))
                            progress.done += 1
                    finally:
                        _WORKER_DATA = None
                        if _THREAD_LIMIT is not None:
                            _THREAD_LIMIT.restore_original_limits()
                            _THREAD_LIMIT = None
                else:
                    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                             initializer=_load_worker, initargs=(manifest,)) as pool:
                        pending = set()
                        iterator = iter(tasks)
                        for _ in range(min(2 * workers, len(tasks))):
                            pending.add(pool.submit(render_task, next(iterator)))
                        while pending:
                            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                            for future in completed:
                                results.append(future.result())
                                progress.done += 1
                                task = next(iterator, None)
                                if task is not None:
                                    pending.add(pool.submit(render_task, task))
            progress.set_phase("validation", len(results))
            for result in results:
                validate_png(result["path"], result["size"])
                progress.done += 1
            # No final PNG is touched until every requested image is complete.
            for result in results:
                source = Path(result["path"])
                destination = output_dir / source.name
                source.replace(destination)
                result["path"] = str(destination)
        progress.close()
        report = dict(settings=settings, effective_workers=workers, memory_estimates=memory,
                      sources=identities, densities=list(densities), dpi=dpi, logit_epsilon=epsilon,
                      phase_seconds=progress.durations, total_seconds=time.perf_counter() - progress.started,
                      peak_parent_worker_rss_gib=progress.peak_rss / GIB,
                      outputs=sorted(results, key=lambda result: result["path"]))
        temporary = output_dir / f".heatmap_run-{token}.json"
        try:
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(output_dir / "heatmap_run.json")
        finally:
            temporary.unlink(missing_ok=True)
        print(f"[heatmaps] validated {len(results)} figures in {report['total_seconds']:.1f}s; "
              f"peak parent+worker RSS {progress.peak_rss / GIB:.2f} GiB", flush=True)
        return output_dir
    finally:
        if not progress.stop.is_set():
            progress.close()

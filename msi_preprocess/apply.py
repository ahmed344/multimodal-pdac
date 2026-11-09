"""Apply frozen MSI preprocessing to tissue, then extract exactly matching lesions."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil

from anndata.io import read_elem, write_elem
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from .fit import load_fit, log
from .cohort import source_rows
from .io import (CSRWriter, axes, check_disk, check_memory, copy_metadata, identity,
                 json_write, match_rows, read_rows)
from .report import Diagnostics
from .spatial import neighborhoods, spatial_support


def clean_slide(counts: sparse.csr_matrix, retained: np.ndarray, multiplier: float,
                graphs: tuple, envelopes: np.ndarray, config: dict,
                lesion_positions: np.ndarray | None = None) -> tuple:
    """Select, scale and mask. Neither averaging nor imputation changes a retained cell."""
    selected = counts[:, retained].tocsr()
    integrated = selected.copy()
    integrated.data /= multiplier
    cleaned = integrated.tocsc()
    cuts = np.asarray(config["local"]["sensitivity"])
    sensitivity = np.zeros((len(cuts), 2))
    lesion_sensitivity = np.zeros_like(sensitivity)
    removed = np.zeros(len(retained), dtype=np.int64)
    removed_rows = np.zeros(counts.shape[0], dtype=np.int64)
    if config["stages"]["local"]:
        def assess_block(start: int) -> tuple:
            end = min(len(retained), start + config["feature_block"])
            values = cleaned[:, start:end].toarray()
            supported, assessed = spatial_support(values, graphs, envelopes[:, :, start:end], config["local"]["min_neighbors"])
            unsupported = (values > 0) & assessed & ~supported
            losses = np.zeros((len(cuts), 2))
            lesion_losses = np.zeros_like(losses)
            for k, cut in enumerate(cuts):
                mask = unsupported & (values <= cut)
                losses[k] = [mask.sum(), values[mask].sum()]
                if lesion_positions is not None:
                    local_mask = mask[lesion_positions]
                    lesion_losses[k] = [local_mask.sum(), values[lesion_positions][local_mask].sum()]
            remove = unsupported & (values <= config["local"]["weak_boundary"])
            lo = cleaned.indptr[start]
            entry_mask = np.zeros(cleaned.indptr[end] - lo, dtype=bool)
            for j in range(start, end):
                a, b = cleaned.indptr[j:j+2]
                entry_mask[a-lo:b-lo] = remove[cleaned.indices[a:b], j-start]
            return start, end, entry_mask, losses, remove.sum(axis=0), remove.sum(axis=1), lesion_losses

        starts = range(0, len(retained), config["feature_block"])
        # Blocks are independent features, and only the caller writes results.
        # Completed blocks never supply neighborhood context for another block.
        with ThreadPoolExecutor(max_workers=config.get("application_workers", 4)) as executor:
            for start, end, entry_mask, losses, peak_losses, row_losses, lesion_losses in executor.map(assess_block, starts):
                lo, hi = cleaned.indptr[start], cleaned.indptr[end]
                cleaned.data[lo:hi][entry_mask] = 0
                sensitivity += losses
                lesion_sensitivity += lesion_losses
                removed[start:end] = peak_losses
                removed_rows += row_losses
        cleaned.eliminate_zeros()
    return selected, integrated, cleaned.tocsr(), sensitivity, removed, removed_rows, lesion_sensitivity


def update_dann_config(path: str | Path, output: Path, manifest: dict) -> None:
    """Patch only preprocessing fields, preserving all unrelated YAML and comments."""
    from dann.config import read_config

    path = Path(path)
    text = path.read_text()
    parsed = read_config(path)
    if parsed["training"].get("resume_checkpoint"):
        raise ValueError("Changed MSI inputs require fresh training; clear resume_checkpoint before updating DANN.")
    if parsed["data"]["matrix_key"] != "X" or parsed["data"]["intensity_transform"] != "log1p":
        raise ValueError("DANN must retain matrix_key: X and intensity_transform: log1p.")
    replacements = {("data", "path"): str((output / "lesion.h5ad").resolve()),
                    ("data", "context_path"): str((output / "tissue.h5ad").resolve()),
                    ("data", "preprocessing_fingerprint"): manifest["fingerprint"],
                    ("model", "num_peaks"): manifest["retained_features"]}
    if manifest["schema"] >= 3 and not parsed["results"]["run_name"].endswith("_qc40"):
        replacements[("results", "run_name")] = parsed["results"]["run_name"] + "_qc40"
    lines = text.splitlines(keepends=True)
    section = None
    seen = set()
    result = []
    for line in lines:
        top = re.match(r"^(\w+):", line)
        if top:
            section = top[1]
        field = re.match(r"^  (\w+):", line)
        key = (section, field[1]) if field else None
        if key in replacements:
            comment = line[line.index("  #"):] if "  #" in line else "\n"
            if key == ("data", "path"):
                comment = "  # Cleaned lesion spectra; preprocessing used whole-tissue MSI image context.\n"
            elif key == ("data", "context_path"):
                comment = "  # Matching cleaned whole tissue for inference; fitted parameters are frozen.\n"
            value = json.dumps(replacements[key]) if isinstance(replacements[key], str) else str(replacements[key])
            line = f"  {key[1]}: {value}{comment}"
            seen.add(key)
        result.append(line)
        if line.startswith("data:\n") and "preprocessing_fingerprint" not in parsed["data"]:
            result.append(f"  preprocessing_fingerprint: {manifest['fingerprint']}  # Changed inputs require fresh training.\n")
            seen.add(("data", "preprocessing_fingerprint"))
    if seen != set(replacements):
        raise ValueError("Could not safely locate DANN fields to update.")
    before = yaml.safe_load(text)
    after = yaml.safe_load("".join(result))
    for (group, key), value in replacements.items():
        before[group][key] = value
    if before != after:
        raise ValueError("DANN update unexpectedly changed unrelated settings.")
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text("".join(result))
    temporary.replace(path)


def apply(fitted: str | Path, *, update_dann: bool = False) -> Path:
    """Stream immutable inputs to new sparse H5ADs with all original metadata.

    Contiguous slides write directly. Interleaved source rows use temporary slide
    files, then a bounded assembly pass restores exact original observation order.
    """
    manifest, state = load_fit(fitted)
    config = manifest["config"]
    output = Path(fitted).resolve().parent
    data = manifest["coordinate_columns"]
    for key, source in (("lesion_input", "source"), ("tissue_input", "tissue_source")):
        if identity(config[key]) != manifest[source]:
            raise ValueError(f"{key} changed since fitting; frozen source identity mismatch.")
    lesion, features = axes(config["lesion_input"], data, config["patient_column"])
    tissue, tissue_features = axes(config["tissue_input"], data, config["patient_column"])
    lesion_rows = source_rows(manifest, state, "lesion", len(lesion))
    tissue_rows = source_rows(manifest, state, "tissue", len(tissue))
    lesion, tissue = lesion.iloc[lesion_rows].reset_index(drop=True), tissue.iloc[tissue_rows].reset_index(drop=True)
    if not np.array_equal(features, state["features"]) or not np.array_equal(features, tissue_features):
        raise ValueError("Input feature order differs from the fitted feature order.")
    mapping = match_rows(lesion, tissue, data)
    if not np.array_equal(lesion[config["patient_column"]].astype(str), tissue.iloc[mapping][config["patient_column"]].astype(str)):
        raise ValueError("Matched lesion/tissue patients disagree.")
    batch = tissue[data["batch_column"]].astype(str).to_numpy()
    slides = list(dict.fromkeys(batch))
    if set(slides) != set(state["slides"]):
        raise ValueError("Whole tissue contains unknown/missing slides; application cannot fit new multipliers.")
    starts = np.r_[0, np.flatnonzero(batch[1:] != batch[:-1]) + 1, len(batch)]
    contiguous = len(starts) - 1 == len(slides)
    for name in ("lesion.h5ad", "tissue.h5ad", "lesion.partial.h5ad", "tissue.partial.h5ad"):
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite {output / name}")
    disk = check_disk([config["lesion_input"], config["tissue_input"]], output, config["disk_reserve_gib"])
    if not contiguous:
        # Additional uncompressed slide-cache upper bound for arbitrary row order.
        with h5py.File(config["tissue_input"], "r") as h:
            scratch_bytes = len(h["layers/counts/data"]) * 24 + (len(tissue) + len(slides)) * 16
        if shutil.disk_usage(output).free < disk["required_bytes"] + scratch_bytes:
            raise OSError("Insufficient disk for interleaved-row scratch export.")
        disk["scratch_bytes"] = scratch_bytes
    json_write(output / "disk_preflight.json", disk)
    log(f"Disk preflight: upper bound {disk['required_bytes']/2**30:.1f} GiB; free {disk['available_bytes']/2**30:.1f} GiB")
    retained = state["retained"]
    provenance = dict(fingerprint=manifest["fingerprint"], schema=manifest["schema"],
                      fit_scope=manifest.get("fitting", {}).get("scope", "dann_training"),
                      intensity_scale="linear", original_counts="already_RMS_normalized",
                      application_context="whole_tissue_MSI_without_IHC_or_HE_labels",
                      fresh_training_required=True, input_features=len(features), retained_features=len(retained),
                      manifest_json=json.dumps(manifest), anchors=state["anchors"], reference=state["reference"],
                      slides=state["slides"], multipliers=state["multipliers"], envelopes=state["envelopes"],
                      retained_original_indices=retained)
    train_mask = np.zeros(len(lesion), bool)
    train_mask[:] = np.isin(lesion_rows, state.get("fitting_rows", state.get("training_rows")))
    diagnostics = Diagnostics(output, features, retained, config["report_peaks"], state, manifest)
    sensitivity_rows, removal_rows = [], []
    per_row_removed = np.zeros(len(tissue), dtype=np.int32)
    scratch = output / "apply_work"
    partials = [output / "tissue.partial.h5ad", output / "lesion.partial.h5ad"]
    try:
        with ExitStack() as stack:
            th = stack.enter_context(h5py.File(config["tissue_input"], "r"))
            lh = stack.enter_context(h5py.File(config["lesion_input"], "r"))
            out = stack.enter_context(h5py.File(partials[0], "w"))
            copy_metadata(th, out, retained, dict(provenance, dataset_kind="tissue"), tissue_rows, config["block_rows"])
            shape = (len(tissue), len(retained))
            dtype = th["layers/counts/data"].dtype
            dtype = np.result_type(dtype, np.float32)
            cw, xw = CSRWriter(out["layers"], "counts", shape, dtype), CSRWriter(out, "X", shape, dtype)
            cache_rows = {}
            for slide in slides:
                i = list(state["slides"]).index(slide)
                rows = np.flatnonzero(batch == slide)
                check_memory(th["layers/counts"], tissue_rows[rows], config)
                counts = read_rows(th["layers/counts"], tissue_rows[rows], config["block_rows"])
                lrows = np.flatnonzero(lesion[data["batch_column"]].astype(str).to_numpy() == slide)
                local = np.searchsorted(rows, mapping[lrows])
                lesion_counts = read_rows(lh["layers/counts"], lesion_rows[lrows], config["block_rows"])
                if (lesion_counts != counts[local]).nnz:
                    raise ValueError(f"Matched counts differ between lesion and tissue on slide {slide}.")
                del lesion_counts
                frame = tissue.iloc[rows]
                coordinates = frame[[data["x_column"], data["y_column"]]].to_numpy()
                graphs = neighborhoods(coordinates[:, 0], coordinates[:, 1]) if config["stages"]["local"] else ()
                selected, integrated, cleaned, sensitivity, removed, pixel_removed, lesion_sensitivity = clean_slide(
                        counts, retained, float(state["multipliers"][i]), graphs, state["envelopes"][i], config, local)
                per_row_removed[rows] = pixel_removed
                for region, losses in (("whole_tissue", sensitivity), ("lesion", lesion_sensitivity)):
                    for cut, (number, intensity) in zip(config["local"]["sensitivity"], losses):
                        sensitivity_rows.append(dict(region=region, slide=slide, boundary=cut,
                            measurements_removed=int(number), intensity_removed=float(intensity),
                            local_enabled=config["stages"]["local"]))
                removal_rows.extend(dict(slide=slide, feature=features[j], measurements_removed=int(n)) for j, n in zip(retained, removed))
                old = read_rows(th["X"], tissue_rows[rows], config["block_rows"])
                label_names = [name for name in ("Density_CD8", "Density_Stroma") if name in lh["obs"]]
                labels = pd.DataFrame({name: np.asarray(read_elem(lh["obs"][name]))[lesion_rows[lrows]] for name in label_names})
                diagnostics.add(slide, coordinates, dict(counts=counts, old_X=old, selected=selected,
                                                       integrated=integrated, cleaned=cleaned), local, labels, train_mask[lrows])
                if contiguous:
                    for start in range(0, len(rows), config["block_rows"]):
                        stop = start + config["block_rows"]
                        cw.append(selected[start:stop])
                        xw.append(cleaned[start:stop])
                else:
                    scratch.mkdir(exist_ok=True)
                    path = scratch / f"slide_{i:03d}.h5"
                    with h5py.File(path, "w") as cache:
                        write_elem(cache, "counts", selected)
                        write_elem(cache, "X", cleaned)
                    cache_rows[slide] = (path, rows)
                log(f"Applied {slide}: removed {removed.sum():,} weak unsupported measurements")
                del counts, selected, integrated, cleaned, old, graphs
            if not contiguous:
                for start in range(0, len(tissue), config["block_rows"]):
                    rows = np.arange(start, min(len(tissue), start + config["block_rows"]))
                    positions, cp, xp = [], [], []
                    for slide in np.unique(batch[rows]):
                        mask = np.flatnonzero(batch[rows] == slide)
                        path, slide_rows = cache_rows[slide]
                        with h5py.File(path, "r") as cache:
                            indices = np.searchsorted(slide_rows, rows[mask])
                            cp.append(read_rows(cache["counts"], indices, config["block_rows"]))
                            xp.append(read_rows(cache["X"], indices, config["block_rows"]))
                        positions.extend(mask)
                    order = np.argsort(positions)
                    cw.append(sparse.vstack(cp, format="csr")[order])
                    xw.append(sparse.vstack(xp, format="csr")[order])
            if cw.rows != len(tissue) or xw.rows != len(tissue):
                raise RuntimeError("Incomplete tissue export.")
            obs = read_elem(out["obs"])
            obs["msi_removed_measurements"] = per_row_removed
            del out["obs"]
            write_elem(out, "obs", obs)
        # Derive lesion X and counts from the exact written tissue result, with
        # arbitrary lesion ordering supported and no second spatial cleaning pass.
        with h5py.File(partials[0], "r") as th, h5py.File(config["lesion_input"], "r") as lh, h5py.File(partials[1], "w") as out:
            copy_metadata(lh, out, retained, dict(provenance, dataset_kind="lesion"), lesion_rows, config["block_rows"])
            shape = (len(lesion), len(retained))
            cw, xw = CSRWriter(out["layers"], "counts", shape, dtype), CSRWriter(out, "X", shape, dtype)
            for start in range(0, len(lesion), config["block_rows"]):
                rows = mapping[start:start + config["block_rows"]]
                cw.append(read_rows(th["layers/counts"], rows, config["block_rows"]))
                xw.append(read_rows(th["X"], rows, config["block_rows"]))
            obs = read_elem(out["obs"])
            obs["msi_removed_measurements"] = per_row_removed[mapping]
            del out["obs"]
            write_elem(out, "obs", obs)
        diagnostics.finish()
        pd.DataFrame(sensitivity_rows).to_csv(output / "local_sensitivity.csv", index=False)
        pd.DataFrame(removal_rows).to_csv(output / "local_removal.csv", index=False)
        for path, name in zip(partials, ("tissue.h5ad", "lesion.h5ad")):
            path.replace(output / name)
        json_write(output / "export.json", dict(fingerprint=manifest["fingerprint"],
                    lesion_rows=len(lesion), tissue_rows=len(tissue), retained_features=len(retained),
                    lesion_tissue_agreement="exact_by_extraction_after_matching_all_original_counts",
                    intensity_scale="linear", image_context="whole_tissue_MSI", fresh_training_required=True))
    except BaseException:
        for path in partials:
            path.unlink(missing_ok=True)
        raise
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
    if update_dann:
        update_dann_config(config["dann_config"], output, manifest)
    return output / "export.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", default="data/PDAC/Processed/msi_preprocess/fit.json")
    parser.add_argument("--update-dann-config", action="store_true")
    args = parser.parse_args()
    apply(args.fit, update_dann=args.update_dann_config)


if __name__ == "__main__":
    main()

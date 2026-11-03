"""Fit peak decisions and spatial references using only DANN training spectra."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
import yaml

from . import SCHEMA_VERSION
from .config import load_config, validate_config
from .io import axes, check_memory, digest, identity, json_write, read_rows
from .qc import adjacent_pairs, spatial_diagnostics
from .spatial import neighborhoods, reference_envelopes, supported_patches


def log(message: str) -> None:
    """Report progress during a long fit or export."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def training_split(obs: pd.DataFrame, dann: dict) -> tuple[np.ndarray, dict]:
    """Use DANN's exact uncapped seeded split and stored categorical ordering."""
    from dann.data_loader import stratified_split_indices
    data = dann["data"]
    settings = {key: float(data[key]) for key in ("train_fraction", "validation_fraction", "test_fraction")}
    settings["seed"] = int(dann["training"]["seed"])
    codes = obs[data["batch_column"]].astype("category").cat.codes.to_numpy()
    rows = stratified_split_indices(codes, **settings)["train"]
    settings["training_rows_sha256"] = hashlib.sha256(np.sort(rows).astype("<i8").tobytes()).hexdigest()
    return rows, settings


def selection_decisions(tested: np.ndarray, weak: np.ndarray, supported: np.ndarray,
                        patches: np.ndarray, enabled: bool) -> tuple[np.ndarray, list[str]]:
    """Require repeated weak/unstructured evidence and veto any spatial support."""
    enough = weak.sum(axis=0) >= np.maximum(2, np.ceil(.5 * tested.sum(axis=0)))
    veto = (supported | patches).any(axis=0)
    remove = enough & ~veto if enabled else np.zeros(tested.shape[1], bool)
    reasons = ["selection_disabled" if not enabled else
               "spatial_support" if veto[j] else
               "weak_unstructured" if remove[j] else
               "insufficient_assessment" if tested[:, j].sum() < 2 else
               "inconsistent_weak_evidence" for j in range(len(remove))]
    return np.flatnonzero(~remove), reasons


def integration_parameters(medians: np.ndarray, prevalence: np.ndarray,
                           supported: np.ndarray, patients: np.ndarray,
                           retained: np.ndarray, settings: dict, enabled: bool) -> dict:
    """Build a median-of-patient-medians reference and robust slide multipliers."""
    eligible = ((prevalence >= settings["prevalence"]).mean(axis=0) >= settings["slide_fraction"])
    eligible &= supported.mean(axis=0) >= settings["support_fraction"]
    mask = np.zeros(medians.shape[1], bool)
    mask[retained] = True
    anchors = np.flatnonzero(eligible & mask)
    reference = np.zeros(medians.shape[1])
    for j in anchors:
        patient_values = []
        for patient in np.unique(patients):
            values = medians[patients == patient, j]
            values = values[values > 0]
            if len(values):
                patient_values.append(np.median(values))
        if patient_values:
            reference[j] = np.median(patient_values)
    usable_counts = np.zeros(len(medians), dtype=int)
    raw = np.ones(len(medians))
    for i in range(len(medians)):
        usable = anchors[(medians[i, anchors] > 0) & (reference[anchors] > 0)]
        usable_counts[i] = len(usable)
        if enabled:
            if len(usable) < settings["min_anchors"]:
                raise ValueError(f"Fitting slide {i}: only {len(usable)} usable anchors; "
                                 f"need {settings['min_anchors']}. Integration cannot be fitted.")
            raw[i] = np.median(medians[i, usable] / reference[usable])
    limit = settings["max_correction"]
    return dict(anchors=anchors, reference=reference, usable_anchors=usable_counts,
                raw_multipliers=raw, multipliers=np.clip(raw, 1 / limit, limit),
                correction_limited=(raw <= 1 / limit) | (raw >= limit))


def _save_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _fingerprint(manifest: dict, arrays: dict) -> str:
    """Bind frozen settings, source identities and every fitted parameter array."""
    inputs = {key: manifest[key] for key in
              ("schema", "config", "split", "coordinate_columns", "source", "tissue_source")}
    result = hashlib.sha256(digest(inputs).encode())
    for key in sorted(arrays):
        result.update(key.encode())
        result.update(np.ascontiguousarray(arrays[key]).tobytes())
    return result.hexdigest()


def fit(config: dict) -> Path:
    """Fit a reusable state. Incomplete fits resume only with identical inputs/settings."""
    validate_config(config)
    dann = yaml.safe_load(Path(config["dann_config"]).read_text())
    data = dann["data"]
    obs, features = axes(config["lesion_input"], data, config["patient_column"])
    train, split = training_split(obs, dann)
    fitting = np.zeros(len(obs), bool)
    fitting[train] = True
    batch = np.asarray(obs[data["batch_column"]], dtype=str)
    slides = np.unique(batch)
    if any(not np.any(fitting & (batch == slide)) for slide in slides):
        raise ValueError("Every slide must have fitting rows; unseen slides cannot receive frozen multipliers.")
    patients = np.array([str(obs.loc[batch == slide, config["patient_column"]].iloc[0]) for slide in slides])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "fit.json").exists():
        raise FileExistsError("Fitted state already exists; apply it or choose a new output directory.")
    work = output / "fit_work"
    work.mkdir(exist_ok=True)
    manifest = dict(schema=SCHEMA_VERSION, config=config, split=split,
                    coordinate_columns={k: data[k] for k in ("batch_column", "x_column", "y_column")},
                    source=identity(config["lesion_input"]), tissue_source=identity(config["tissue_input"]))
    token = digest(manifest)
    cache_identity = work / "identity.json"
    if cache_identity.exists() and json.loads(cache_identity.read_text()) != token:
        raise ValueError("Incomplete fit belongs to different inputs or settings; choose a new output directory.")
    json_write(cache_identity, token)
    summaries = []
    p = len(features)
    with h5py.File(config["lesion_input"], "r") as h:
        for i, slide in enumerate(slides):
            cache = work / f"slide_{i:03d}.npz"
            if cache.exists():
                with np.load(cache) as z:
                    summaries.append({k: z[k] for k in z.files})
                log(f"Reuse fitted QC: {slide}")
                continue
            rows = np.flatnonzero(fitting & (batch == slide))
            check_memory(h["layers/counts"], rows, config)
            matrix = read_rows(h["layers/counts"], rows, config["block_rows"])
            tic = np.asarray(matrix.sum(axis=1)).ravel()
            xy = obs.iloc[rows]
            graphs = neighborhoods(xy[data["x_column"]].to_numpy(), xy[data["y_column"]].to_numpy())
            pairs = adjacent_pairs(xy[data["x_column"]].to_numpy(), xy[data["y_column"]].to_numpy(),
                                   config["selection"]["spatial_pairs"], np.random.default_rng(config["seed"] + i))
            pair_rows = np.unique(pairs)
            mapped_pairs = np.searchsorted(pair_rows, pairs)
            result = {k: np.zeros(p) for k in ("median", "prevalence", "weak_fraction", "raw_excess", "normalized_excess")}
            result.update({k: np.zeros(p, bool) for k in ("tested", "supported", "patches", "weak")})
            csc = matrix.tocsc()
            for start in range(0, p, config["feature_block"]):
                end = min(p, start + config["feature_block"])
                block = csc[:, start:end].tocsr()
                count = block.getnnz(axis=0)
                result["prevalence"][start:end] = count / len(rows)
                weak_count = np.bincount(block.indices, weights=block.data <= config["selection"]["weak_intensity_screen"], minlength=end-start)
                result["weak_fraction"][start:end] = np.divide(weak_count, count, out=np.zeros(end-start), where=count > 0)
                for j in range(start, end):
                    positive = csc.data[csc.indptr[j]:csc.indptr[j+1]]
                    result["median"][j] = np.median(positive) if len(positive) else 0
                spatial = spatial_diagnostics(block[pair_rows], mapped_pairs, tic[pair_rows], config["selection"],
                                              np.random.default_rng(config["seed"] + i))
                for key in ("raw_excess", "normalized_excess"):
                    result[key][start:end] = spatial[key]
                result["tested"][start:end] = spatial["spatial_tested"]
                result["supported"][start:end] = spatial["spatial_supported"]
                dense = block.toarray()
                envelopes = reference_envelopes(dense, config["local"]["draws"], config["local"]["envelope"], config["seed"] + i)
                result["patches"][start:end] = supported_patches(dense, graphs, envelopes,
                        config["selection"]["patch_pixels"], config["local"]["min_neighbors"])
            result["weak"] = (result["tested"] & (result["weak_fraction"] >= config["selection"]["weak_fraction_screen"])
                              & (result["raw_excess"] <= config["selection"]["spatial_weak_excess"])
                              & (result["normalized_excess"] <= config["selection"]["spatial_weak_excess"]))
            _save_npz(cache, **result)
            summaries.append(result)
            log(f"Fitted QC {i+1}/{len(slides)}: {slide}; {len(rows):,} training rows")
            del matrix, csc, graphs
    arrays = {key: np.stack([r[key] for r in summaries]) for key in summaries[0]}
    retained, reasons = selection_decisions(arrays["tested"], arrays["weak"], arrays["supported"],
                                          arrays["patches"], config["stages"]["selection"])
    if not len(retained):
        raise ValueError("Selection removed every peak; inspect the recorded QC decisions.")
    integration = integration_parameters(arrays["median"], arrays["prevalence"],
                        arrays["supported"] | arrays["patches"], patients, retained,
                        config["integration"], config["stages"]["integration"])
    envelopes = np.full((len(slides), 2, 13, len(retained)), np.nan)
    if config["stages"]["local"]:
        with h5py.File(config["lesion_input"], "r") as h:
            for i, slide in enumerate(slides):
                rows = np.flatnonzero(fitting & (batch == slide))
                matrix = read_rows(h["layers/counts"], rows, config["block_rows"])[:, retained].tocsc()
                matrix.data /= integration["multipliers"][i]
                for start in range(0, len(retained), config["feature_block"]):
                    end = min(len(retained), start + config["feature_block"])
                    envelopes[i, :, :, start:end] = reference_envelopes(matrix[:, start:end].toarray(),
                            config["local"]["draws"], config["local"]["envelope"], config["seed"] + i)
                log(f"Frozen local envelopes {i+1}/{len(slides)}: {slide}")
    arrays.update(integration, envelopes=envelopes, retained=retained, features=features,
                  slides=slides, patients=patients, training_rows=np.sort(train))
    manifest["fingerprint"] = _fingerprint(manifest, arrays)
    manifest["retained_features"] = len(retained)
    manifest["input_features"] = p
    _save_npz(output / "fit.npz", **arrays)
    manifest["state_sha256"] = hashlib.sha256((output / "fit.npz").read_bytes()).hexdigest()
    pd.DataFrame(dict(feature=features, retained=np.isin(np.arange(p), retained), reason=reasons,
                      adequately_assessed_slides=arrays["tested"].sum(axis=0),
                      weak_unstructured_slides=arrays["weak"].sum(axis=0),
                      supported_slides=arrays["supported"].sum(axis=0), patch_slides=arrays["patches"].sum(axis=0),
                      anchor=np.isin(np.arange(p), integration["anchors"]))).to_csv(output / "peak_decisions.csv", index=False)
    pd.DataFrame(dict(slide=slides, patient=patients, raw_multiplier=integration["raw_multipliers"],
                      multiplier=integration["multipliers"], usable_anchors=integration["usable_anchors"],
                      correction_limited=integration["correction_limited"])).to_csv(output / "slide_multipliers.csv", index=False)
    for i, slide in enumerate(slides):
        pd.DataFrame({"feature": features, **{k: arrays[k][i] for k in summaries[0]}}).to_csv(work / f"slide_{i:03d}.csv", index=False)
    json_write(output / "fit.json", manifest)
    log(f"Fit complete: retained {len(retained)}/{p} peaks, {len(integration['anchors'])} reference anchors")
    return output / "fit.json"


def load_fit(path: str | Path) -> tuple[dict, dict]:
    """Load and verify a frozen state; application never refits missing parameters."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest["schema"] != SCHEMA_VERSION:
        raise ValueError("Unsupported preprocessing state version.")
    state_path = path.with_name("fit.npz")
    if hashlib.sha256(state_path.read_bytes()).hexdigest() != manifest["state_sha256"]:
        raise ValueError("Fitted parameter checksum mismatch.")
    with np.load(state_path, allow_pickle=False) as state:
        arrays = {key: state[key] for key in state.files}
    if _fingerprint(manifest, arrays) != manifest["fingerprint"]:
        raise ValueError("Frozen settings/parameter fingerprint mismatch; refit instead of editing fitted state.")
    return manifest, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="msi_preprocess/config.yaml")
    args = parser.parse_args()
    fit(load_config(args.config))


if __name__ == "__main__":
    main()

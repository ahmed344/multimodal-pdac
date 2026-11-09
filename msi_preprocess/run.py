"""Stage, verify, smoke-test and replace one complete preprocessing generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import yaml

from .apply import apply, update_dann_config
from .comparison import render_report
from .config import load_config
from .fit import fit, load_fit, log
from .io import check_disk, identity, json_write
from .verify import verify
from .report import render_maps
from .tic import render_tic_maps


def check_cohort(manifest: dict, output: Path, expected: tuple[int, ...] | None = None) -> dict:
    """Gate promotion on the declared cohort and optional study-specific totals."""
    included = [row for row in manifest["cohort"]["decisions"] if row["included"]]
    actual = dict(slides=len(included), patients=len({row["patient"] for row in included}),
                  lesion_rows=sum(row["lesion_rows"] for row in included),
                  tissue_rows=sum(row["tissue_rows"] for row in included))
    exported = json.loads((output / "export.json").read_text())
    if any(exported[key] != actual[key] for key in ("lesion_rows", "tissue_rows")):
        raise ValueError("Export counts disagree with retained cohort.")
    if expected is not None and tuple(actual.values()) != tuple(expected):
        raise ValueError(f"Retained cohort differs from requested counts: {actual}")
    return actual


def check_promoted(output: Path, config_path: Path, manifest: dict) -> None:
    """Validate installed paths and the DANN input contract before deleting backups."""
    from dann.preprocessing import validate_preprocessing
    from .io import axes
    config = yaml.safe_load(config_path.read_text())
    for key, name in (("path", "lesion.h5ad"), ("context_path", "tissue.h5ad")):
        path = output / name
        if Path(config["data"][key]).resolve() != path.resolve():
            raise ValueError("DANN path does not reference the promoted generation.")
        validate_preprocessing(path, config["data"])
        _, features = axes(path, manifest["coordinate_columns"], manifest["config"]["patient_column"])
        if len(features) != config["model"]["num_peaks"]:
            raise ValueError("DANN feature count differs from promoted input.")


def promote(staging: Path, output: Path, config_path: Path, manifest: dict) -> None:
    """Promote a verified generation with rollback of both directory and YAML."""
    backup = output.with_name(f".{output.name}.previous")
    if backup.exists():
        raise FileExistsError(f"Previous replacement backup still exists: {backup}")
    original_config = config_path.read_bytes()
    had_output = output.exists()
    if had_output:
        archive = staging / "previous_parameters_and_summaries.zip"
        names = {p.name for pattern in ("*.json", "*.csv", "*.npz", "*.npy", "*.zip", "report.html")
                 for p in output.glob(pattern)}
        names.update(str(p.relative_to(output)) for p in (output / "figures").glob("*.png"))
        names.update(str(p.relative_to(output)) for p in (output / "fit_work").glob("*.csv"))
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name in sorted(names):
                if (output / name).is_file():
                    z.write(output / name, name)
            z.writestr("dann_config_before.yaml", original_config)
    installed = False
    try:
        if had_output:
            output.rename(backup)
        staging.rename(output)
        installed = True
        update_dann_config(config_path, output, manifest)
        check_promoted(output, config_path, manifest)
    except BaseException:
        temporary = config_path.with_suffix(".rollback.tmp")
        temporary.write_bytes(original_config)
        temporary.replace(config_path)
        if installed:
            output.rename(staging)
        if had_output and backup.exists():
            backup.rename(output)
        raise
    json_write(output / "replacement.json", dict(fingerprint=manifest["fingerprint"],
        output=str(output.resolve()), previous_outputs_replaced=had_output, final_contract_passed=True))
    if backup.exists():
        shutil.rmtree(backup)
    log(f"Installed verified preprocessing at {output}")


def smoke(staging: Path, config_path: Path, manifest: dict) -> None:
    """Run the actual capped DANN entry point with outputs isolated in this generation."""
    from dann.config import read_config

    smoke_path = staging / "smoke_config.yaml"
    smoke_path.write_text(yaml.safe_dump(read_config(config_path), sort_keys=False))
    update_dann_config(smoke_path, staging, manifest)
    config = yaml.safe_load(smoke_path.read_text())
    config["results"]["root"] = str((staging / "dann_smoke").resolve())
    config["results"]["run_name"] = "retained_lesion_preprocessing_qc40"
    config["analysis"]["peaks"].update(enabled=False, spatial=False)
    config["analysis"].update(figure_dpi=100, density_mc_samples=32, activity_max_nonzeros=2000000)
    config["analysis"]["execution"]["render_workers"] = 2
    smoke_path.write_text(yaml.safe_dump(config, sort_keys=False))
    log("Running capped DANN smoke: configured training/validation caps and smoke epochs")
    with (staging / "smoke.log").open("w") as stream:
        subprocess.run([sys.executable, "-m", "dann.train", "--config", str(smoke_path), "--smoke-test"],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, "-m", "msi_preprocess.smoke", "--config", str(smoke_path)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    json_write(staging / "smoke.json", dict(fingerprint=manifest["fingerprint"], passed=True, version=2,
        latent_calibration_analysis=True, capped_spatial_inference=True, latent_variance=True,
        train_cap=config["training"]["smoke_train_samples"],
        validation_cap=config["training"]["smoke_validation_samples"],
        epochs=config["training"]["smoke_epochs"],
        note="Staging paths in smoke artifacts record the paths used at execution; fresh production training is required."))


def run(config: dict, *, replace: bool = False, expected_cohort: tuple[int, ...] | None = None) -> Path:
    """Resume a matching staged generation; never overwrite original input files."""
    output = Path(config["output"]).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() and not replace:
        raise FileExistsError("Output exists; use --replace to stage and replace derived outputs.")
    if output.exists():
        load_fit(output / "fit.json")  # Never replace an unrelated directory.
    if output.with_name(f".{output.name}.previous").exists():
        raise FileExistsError("An earlier promotion backup exists; restore it before another replacement.")
    staging.mkdir(parents=True, exist_ok=True)
    fitted = staging / "fit.json"
    if not fitted.exists():
        check_disk([config["lesion_input"], config["tissue_input"]], staging, config["disk_reserve_gib"])
        fit(config, output_dir=staging)
    manifest, _ = load_fit(fitted)
    if manifest["config"] != config:
        raise ValueError("Staged fit settings differ; do not reuse stale fitted parameters.")
    for key, source in (("lesion_input", "source"), ("tissue_input", "tissue_source")):
        if identity(config[key]) != manifest[source]:
            raise ValueError("Input changed since the staged fit.")
    if not (staging / "export.json").exists():
        apply(fitted)
    cohort = check_cohort(manifest, staging, expected_cohort)
    log(f"Confirmed retained cohort: {cohort}")
    log("Rendering full report and pixel heatmaps")
    render_maps(fitted)
    render_tic_maps(fitted)
    render_report(fitted)
    log("Verifying all written tissue values and all matched lesion rows")
    verification = verify(fitted)
    marker = staging / "smoke.json"
    if (not marker.exists() or json.loads(marker.read_text()).get("fingerprint") != manifest["fingerprint"]
            or json.loads(marker.read_text()).get("version") != 2):
        smoke(staging, Path(config["dann_config"]), manifest)
    summary = json.loads((staging / "validation_summary.json").read_text())
    summary["verification"] = verification
    summary["smoke"] = json.loads(marker.read_text())
    summary["cohort"] = cohort
    json_write(staging / "validation_summary.json", summary)
    promote(staging, output, Path(config["dann_config"]), manifest)
    return output / "report.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="msi_preprocess/config.yaml")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--expected-cohort", type=int, nargs=4, metavar=("SLIDES", "PATIENTS", "LESION_ROWS", "TISSUE_ROWS"))
    args = parser.parse_args()
    print(run(load_config(args.config), replace=args.replace, expected_cohort=args.expected_cohort), flush=True)


if __name__ == "__main__":
    main()

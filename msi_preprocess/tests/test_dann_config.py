"""Preprocessing edits and smoke snapshots support linked DANN settings."""

from pathlib import Path
import shutil

import pytest
import yaml

from dann.config import load_config, read_config
from msi_preprocess.apply import update_dann_config
from msi_preprocess.run import smoke


@pytest.fixture
def linked_config(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("config.yaml", "config_trainging.yaml", "config_analysis.yaml"):
        shutil.copyfile(Path("dann") / name, source / name)
    return source / "config.yaml"


def test_updates_only_shared_fields_and_checks_linked_resume(linked_config, tmp_path):
    fragments = [linked_config.with_name(name) for name in ("config_trainging.yaml", "config_analysis.yaml")]
    before = {path: path.read_bytes() for path in fragments}
    expected = read_config(linked_config)
    output = tmp_path / "processed"
    manifest = dict(schema=3, fingerprint="b" * 64, retained_features=123)
    update_dann_config(linked_config, output, manifest)
    expected["data"].update(path=str(output / "lesion.h5ad"), context_path=str(output / "tissue.h5ad"),
                            preprocessing_fingerprint=manifest["fingerprint"])
    expected["model"]["num_peaks"] = 123
    expected["results"]["run_name"] += "_qc40"
    assert read_config(linked_config) == expected
    assert yaml.safe_load(linked_config.read_text())["includes"] == ["config_trainging.yaml", "config_analysis.yaml"]
    assert {path: path.read_bytes() for path in fragments} == before
    assert "# Shared DANN settings." in linked_config.read_text()
    training = yaml.safe_load(fragments[0].read_text())
    training["training"]["resume_checkpoint"] = "existing.pt"
    fragments[0].write_text(yaml.safe_dump(training))
    unchanged = linked_config.read_bytes()
    with pytest.raises(ValueError, match="fresh training"):
        update_dann_config(linked_config, output, manifest)
    assert linked_config.read_bytes() == unchanged


def test_smoke_config_is_portable_and_preserves_sources(linked_config, tmp_path, monkeypatch):
    import msi_preprocess.run as runner

    before = {path: path.read_bytes() for path in linked_config.parent.iterdir()}
    staging = tmp_path / "staging"
    staging.mkdir()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        config = load_config(Path(command[command.index("--config") + 1]))
        assert config["data"]["path"] == str(staging / "lesion.h5ad")
        assert config["results"]["root"] == str(staging / "dann_smoke")

    monkeypatch.setattr(runner.subprocess, "run", run)
    smoke(staging, linked_config, dict(schema=3, fingerprint="b" * 64, retained_features=123))
    assert [command[2] for command in calls] == ["dann.train", "msi_preprocess.smoke"]
    assert {path: path.read_bytes() for path in before} == before
    shutil.rmtree(linked_config.parent)
    snapshot = staging / "smoke_config.yaml"
    assert "includes" not in yaml.safe_load(snapshot.read_text())
    assert load_config(snapshot)["analysis"]["density_mc_samples"] == 32

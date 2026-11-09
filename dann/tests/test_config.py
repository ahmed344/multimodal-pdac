"""Linked configuration loading and portable snapshot contracts."""

from pathlib import Path
import shutil

import pytest
import yaml

from dann.config import apply_smoke_overrides, load_config, read_config


@pytest.fixture
def linked_config(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("config.yaml", "config_trainging.yaml", "config_analysis.yaml"):
        shutil.copyfile(Path("dann") / name, source / name)
    return source / "config.yaml"


def test_includes_are_relative_to_main_file(tmp_path, monkeypatch):
    source = tmp_path / "source"
    fragments = source / "settings"
    fragments.mkdir(parents=True)
    path = source / "config.yaml"
    path.write_text("includes: [settings/config_trainging.yaml, settings/config_analysis.yaml]\ndata: {path: relative.h5ad}\n")
    (fragments / "config_trainging.yaml").write_text("training: {epochs: 7}\n")
    (fragments / "config_analysis.yaml").write_text("analysis: {max_samples: 12}\n")
    monkeypatch.chdir(tmp_path)
    assert read_config(Path("source/config.yaml")) == {
        "data": {"path": "relative.h5ad"}, "training": {"epochs": 7},
        "analysis": {"max_samples": 12},
    }


def test_source_and_resolved_snapshots_are_self_contained(linked_config, tmp_path):
    raw = read_config(linked_config)
    effective = load_config(linked_config)
    assert "includes" not in raw and "includes" not in effective
    assert "execution" not in raw
    assert "output_dir" not in raw["training"]
    source_snapshot = tmp_path / "source_config.yaml"
    resolved_snapshot = tmp_path / "resolved_config.yaml"
    source_snapshot.write_text(yaml.safe_dump(raw))
    resolved_snapshot.write_text(yaml.safe_dump(effective))
    shutil.rmtree(linked_config.parent)
    assert read_config(source_snapshot) == raw
    assert load_config(source_snapshot) == load_config(resolved_snapshot) == effective
    assert apply_smoke_overrides(load_config(source_snapshot)) == apply_smoke_overrides(effective)


def test_fragment_edits_control_runtime_and_smoke(linked_config):
    training_path = linked_config.with_name("config_trainging.yaml")
    training = yaml.safe_load(training_path.read_text())
    training["training"].update(epochs=17, smoke_epochs=2)
    training_path.write_text(yaml.safe_dump(training))
    analysis_path = linked_config.with_name("config_analysis.yaml")
    analysis = yaml.safe_load(analysis_path.read_text())
    analysis["analysis"]["max_samples"] = 23
    analysis["post_training"].update(enabled=True, run_latent_variance=False)
    analysis_path.write_text(yaml.safe_dump(analysis))
    config = load_config(linked_config)
    assert config["training"]["epochs"] == 17
    assert config["analysis"]["max_samples"] == 23
    assert not config["post_training"]["run_latent_variance"]
    smoke = apply_smoke_overrides(config)
    assert smoke["training"]["epochs"] == 2
    assert not smoke["post_training"]["enabled"]
    assert config["post_training"]["enabled"]
    analysis["analysis"]["max_samples"] = -1
    analysis_path.write_text(yaml.safe_dump(analysis))
    with pytest.raises(ValueError, match="analysis.max_samples"):
        load_config(linked_config)


@pytest.mark.parametrize("document", ["", "null", "[]", "scalar", "true", "1: value"])
@pytest.mark.parametrize("included", [False, True])
def test_rejects_non_mapping_sources(tmp_path, document, included):
    path = tmp_path / "config.yaml"
    invalid = path
    if included:
        path.write_text("includes: [fragment.yaml]\n")
        invalid = tmp_path / "fragment.yaml"
    invalid.write_text(document)
    with pytest.raises(ValueError, match=rf"{invalid.name}.*mapping"):
        read_config(path)


@pytest.mark.parametrize("includes", [None, "config_trainging.yaml", {}, True, [None], [1], [False], [""], [" "]])
def test_rejects_invalid_include_lists(tmp_path, includes):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"includes": includes}))
    with pytest.raises(ValueError, match="includes must be a list of nonempty file paths"):
        read_config(path)


@pytest.mark.parametrize("main_section", [False, True])
def test_rejects_duplicate_sections_across_sources(tmp_path, main_section):
    path = tmp_path / "config.yaml"
    root = {"includes": ["config_trainging.yaml"]}
    if main_section:
        root["training"] = {"epochs": 1}
    else:
        root["includes"].append("other.yaml")
        (tmp_path / "other.yaml").write_text("training: {epochs: 2}\n")
    path.write_text(yaml.safe_dump(root))
    (tmp_path / "config_trainging.yaml").write_text("training: {epochs: 3}\n")
    with pytest.raises(ValueError, match="Duplicate configuration sections.*training"):
        read_config(path)


def test_rejects_nested_includes(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("includes: [config_trainging.yaml]\n")
    (tmp_path / "config_trainging.yaml").write_text("includes: [config.yaml]\n")
    with pytest.raises(ValueError, match="Nested includes.*config_trainging.yaml"):
        read_config(path)


def test_missing_and_invalid_yaml_files_name_the_source(tmp_path):
    path = tmp_path / "config.yaml"
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        read_config(path)
    path.write_text("includes: [config_trainging.yaml]\n")
    with pytest.raises(FileNotFoundError, match="config_trainging.yaml"):
        read_config(path)
    (tmp_path / "config_trainging.yaml").write_text("training: [\n")
    with pytest.raises(ValueError, match="Invalid YAML.*config_trainging.yaml"):
        read_config(path)

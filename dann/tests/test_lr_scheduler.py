"""Plateau numerics, immutable snapshots, and production resume boundaries."""

import copy
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
import yaml

from dann import train
from dann.config import load_config, lr_scheduler_settings, resolve_execution
from dann.data_loader import create_data_bundle
from dann.lr_scheduler import create_lr_scheduler, step_lr_scheduler
from dann.model import AdversarialLatentFusion
from dann.tests.test_components import spatial_config
from dann.tests.test_training_artifacts import assert_nested_equal, worker


def scheduler_pair(starting_lr=1e-6, **settings):
    config = dict(training=dict(
        learning_rate=starting_lr, lr_scheduler=dict(enabled=True, **settings)))
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=starting_lr)
    return optimizer, create_lr_scheduler(optimizer, config)


def test_config_defaults_and_mode_floors(tmp_path):
    config = load_config(Path("dann/config.yaml"))
    expected = dict(enabled=True, factor=.5, patience=5, cooldown=2,
                    threshold=.001, min_lr_factor=.01, eps=0.)
    assert config["training"]["lr_scheduler"] == expected
    assert config["training"]["early_stopping_patience"] == 20
    for mode, starting_lr in (("spatial", 1e-6), ("pixel", .001)):
        config["model"]["aggregation"]["type"] = "cnn" if mode == "spatial" else "mlp"
        for group in config["model"]["heads"].values():
            group["type"] = "mlp"
        config["training"][mode]["learning_rate"] = starting_lr
        resolve_execution(config)
        assert config["execution"]["mode"] == mode
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=starting_lr)
        assert create_lr_scheduler(optimizer, config).min_lrs == [starting_lr * .01]
    del config["training"]["lr_scheduler"]
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(config))
    legacy = load_config(path)
    assert not legacy["training"]["lr_scheduler"]["enabled"]
    assert create_lr_scheduler(optimizer, legacy) is None


@pytest.mark.parametrize("value", [None, True, [], "plateau", 1])
def test_invalid_scheduler_blocks(value):
    with pytest.raises(ValueError, match="lr_scheduler must be a mapping"):
        lr_scheduler_settings(dict(lr_scheduler=value))


@pytest.mark.parametrize("key, value", [
    ("enabled", 1), ("enabled", "true"), ("enabled", None),
    ("factor", 0), ("factor", 1), ("factor", -1),
    ("min_lr_factor", 0), ("min_lr_factor", 1), ("min_lr_factor", -1),
    ("patience", -1), ("patience", 1.5), ("patience", True),
    ("cooldown", -1), ("cooldown", 2.), ("cooldown", False),
    ("threshold", -1), ("eps", -1),
    ("unknown", 0), ("mode", "max"), ("threshold_mode", "rel"),
] + [(key, value) for key in ("factor", "min_lr_factor", "threshold", "eps")
     for value in (float("nan"), float("inf"), -float("inf"), True, "0.5", None)])
def test_invalid_scheduler_settings(key, value):
    with pytest.raises(ValueError, match="lr_scheduler"):
        lr_scheduler_settings(dict(lr_scheduler={key: value}))


def test_config_loader_validates_scheduler(tmp_path):
    config = yaml.safe_load(Path("dann/config.yaml").read_text())
    config["training"]["lr_scheduler"]["patience"] = 1.5
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="lr_scheduler.patience"):
        load_config(path)


def test_default_patience_cooldown_and_absolute_threshold():
    optimizer, scheduler = scheduler_pair()
    # Negative objectives distinguish absolute thresholds from relative thresholds.
    values = [-10.] + [-10.0005] * 6 + [-10.0005] * 2 + [-10.002] + [-10.002] * 6
    states = []
    for value in values:
        step_lr_scheduler(scheduler, dict(supervised_objective=value))
        states.append((optimizer.param_groups[0]["lr"], scheduler.num_bad_epochs,
                       scheduler.cooldown_counter, scheduler.best))
    assert [state[0] for state in states] == [1e-6] * 6 + [5e-7] * 9 + [2.5e-7]
    assert states[6][1:] == (0, 2, -10.)
    assert states[7][1:3] == (0, 1)
    assert states[8][1:3] == (0, 0)
    assert states[9][1:] == (0, 0, -10.002)


def test_threshold_requires_strict_improvement():
    optimizer, scheduler = scheduler_pair(patience=0, threshold=.125, cooldown=0)
    for value in (2., 1.875):  # Exactly threshold-sized improvement does not count.
        step_lr_scheduler(scheduler, dict(supervised_objective=value))
    assert optimizer.param_groups[0]["lr"] == 5e-7
    assert scheduler.best == 2.
    step_lr_scheduler(scheduler, dict(supervised_objective=1.874))
    assert scheduler.best == 1.874
    assert optimizer.param_groups[0]["lr"] == 5e-7


@pytest.mark.parametrize("starting_lr", [1e-6, .001, .0001])
def test_plateaus_reach_exact_original_floor(starting_lr):
    optimizer, scheduler = scheduler_pair(starting_lr)
    rates = []
    for _ in range(100):
        step_lr_scheduler(scheduler, dict(supervised_objective=1.))
        rates.append(optimizer.param_groups[0]["lr"])
    floor = starting_lr * .01
    assert rates[-1] == floor
    assert all(rate >= floor for rate in rates)
    assert all(a >= b for a, b in zip(rates, rates[1:]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_validation_does_not_advance_scheduler(value):
    optimizer, scheduler = scheduler_pair()
    before = copy.deepcopy(scheduler.state_dict())
    with pytest.raises(ValueError, match="finite validation supervised_objective"):
        step_lr_scheduler(scheduler, dict(supervised_objective=value))
    assert scheduler.state_dict() == before
    assert optimizer.param_groups[0]["lr"] == 1e-6
    step_lr_scheduler(None, {})  # Disabled scheduling does not require a monitor.


@pytest.fixture
def checkpoint_setup(spatial_config):
    config = spatial_config
    config["training"]["lr_scheduler"].update(enabled=True, patience=1, cooldown=2)
    config["training"]["spatial"]["learning_rate"] = 1e-6
    resolve_execution(config)
    bundle = create_data_bundle(config)
    model = AdversarialLatentFusion.from_config(config, 2, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    # Give the optimizer real moment state as well as parameter groups to restore.
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    scheduler = create_lr_scheduler(optimizer, config)
    yield config, bundle, model, optimizer, scheduler
    for dataset in bundle.datasets.values():
        dataset.close()


@pytest.mark.parametrize("completed", [2, 3, 4, 5, 7])
def test_checkpoint_resume_matches_uninterrupted(checkpoint_setup, tmp_path, completed):
    config, bundle, model, optimizer, scheduler = checkpoint_setup
    values = [3.] * 8 + [2.] + [2.] * 35
    for value in values[:completed]:
        step_lr_scheduler(scheduler, dict(supervised_objective=value))
    path = tmp_path / "scheduled.pt"
    train.save_checkpoint(path, model, optimizer, completed - 1, config, bundle, 3.,
                          lr_scheduler=scheduler)
    payload = torch.load(path, weights_only=False)
    requested = copy.deepcopy(config)
    requested["training"]["lr_scheduler"].update(enabled=False, factor=.8, min_lr_factor=.3)
    requested["training"]["early_stopping_metric"] = "cd8_loss"
    effective = train.resume_training_config(payload["config"], requested)
    assert effective["training"]["lr_scheduler"] == config["training"]["lr_scheduler"]
    assert "training.lr_scheduler.factor" in train.ignored_resume_settings(requested, effective)
    restored = AdversarialLatentFusion.from_config(effective, 2, 4)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.1)
    restored_scheduler = create_lr_scheduler(restored_optimizer, effective)
    assert train.load_training_checkpoint(
        path, restored, restored_optimizer, torch.device("cpu"), effective,
        lr_scheduler=restored_scheduler,
    )[0] == completed
    assert_nested_equal(optimizer.state_dict(), restored_optimizer.state_dict())
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    for value in values[completed:]:
        for current in (scheduler, restored_scheduler):
            step_lr_scheduler(current, dict(supervised_objective=value))
        assert optimizer.param_groups[0]["lr"] == restored_optimizer.param_groups[0]["lr"]
        assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.param_groups[0]["lr"] == 1e-8
    assert restored_scheduler.min_lrs == [1e-8]


def test_async_scheduler_snapshot_is_frozen(checkpoint_setup, worker, tmp_path):
    config, bundle, model, optimizer, scheduler = checkpoint_setup
    for _ in range(3):  # Save immediately after reduction, with cooldown active.
        step_lr_scheduler(scheduler, dict(supervised_objective=3.))
    path = tmp_path / "sync.pt"
    train.save_checkpoint(path, model, optimizer, 2, config, bundle, 3., lr_scheduler=scheduler)
    frozen = worker.snapshot(model, optimizer, scheduler)
    payload = train.checkpoint_payload(train.checkpoint_metadata(model, config, bundle), frozen, 2, 3.)
    worker.submit_epoch([dict(epoch=3)], ["epoch"], {}, {}, payload, ["latest.pt"])
    for _ in range(8):
        step_lr_scheduler(scheduler, dict(supervised_objective=3.))
    scheduler.min_lrs[0] = .1  # Mutable state lists must also be independent.
    worker.flush()
    saved = torch.load(worker.directory / "latest.pt", weights_only=False)
    assert_nested_equal(saved, torch.load(path, weights_only=False))
    assert saved["lr_scheduler_state"]["cooldown_counter"] == 2
    assert saved["lr_scheduler_state"]["min_lrs"] == [1e-8]


@pytest.mark.parametrize("async_artifacts", [False, True])
@pytest.mark.parametrize("metric, count", [("supervised_objective", 5), ("cd8_loss", 7),
                                         ("batch_aware_objective", 7)])
def test_training_history_checkpoint_timing_and_stopping(
    spatial_config, monkeypatch, capsys, async_artifacts, metric, count,
):
    config = spatial_config
    for group in [config["model"]["aggregation"], *config["model"]["heads"].values()]:
        group["type"] = "mlp"
    config["training"].update(epochs=7, early_stopping_patience=4,
                              early_stopping_metric=metric, checkpoint_every=20)
    config["training"]["lr_scheduler"].update(enabled=True, patience=1, cooldown=1)
    config["performance"]["async_artifacts"] = async_artifacts
    config["analysis"]["figure_dpi"] = 12
    resolve_execution(config)
    monkeypatch.setattr(train, "refresh_loss_curves", Mock())
    real_epoch = train.run_epoch
    used_rates = []

    def epoch(**kwargs):
        # Exercise real sparse training and validation, but control plateau scores.
        metrics = real_epoch(**kwargs)
        if kwargs["optimizer"] is None:
            metrics.update(supervised_objective=10., cd8_loss=10. - kwargs["epoch_index"],
                           batch_aware_objective=12. - .1 * kwargs["epoch_index"],
                           total_loss=20. - kwargs["epoch_index"],
                           batch_weighted_loss=10. - kwargs["epoch_index"])
        else:
            used_rates.append(kwargs["optimizer"].param_groups[0]["lr"])
        return metrics

    monkeypatch.setattr(train, "run_epoch", epoch)
    best = train.train_model(config)
    history = pd.read_csv(best.with_name("history.csv"))
    starting_lr = config["training"]["learning_rate"]
    expected_used = [starting_lr * factor for factor in [1, 1, 1, .5, .5, .5, .25]][:count]
    expected_next = [starting_lr * factor for factor in [1, 1, .5, .5, .5, .25, .25]][:count]
    assert used_rates == expected_used
    assert history.learning_rate.tolist() == expected_used
    assert history.next_learning_rate.tolist() == expected_next
    latest = torch.load(best.with_name("latest.pt"), weights_only=False)
    assert latest["epoch"] + 1 == count
    assert latest["lr_scheduler_state"]["last_epoch"] == count
    assert latest["optimizer_state"]["param_groups"][0]["lr"] == expected_next[-1]
    assert latest["early_stopping_state"]["stale_epochs"] == (4 if count == 5 else 0)
    assert latest["training_progress"]["epoch"] == count
    assert latest["training_progress"]["optimizer_updates"] == history.train_optimizer_updates.sum()
    selected = torch.load(best, weights_only=False)
    assert selected["lr_scheduler_state"]["last_epoch"] == selected["epoch"] + 1
    if not async_artifacts:
        output = capsys.readouterr().out
        assert "learning_rate = " in output and "next_learning_rate = " in output


@pytest.mark.parametrize("missing", [None, {}, {"best": 1.}])
def test_production_resume_requires_scheduler_state(checkpoint_setup, tmp_path, monkeypatch, missing):
    config, bundle, model, optimizer, scheduler = checkpoint_setup
    path = tmp_path / "broken.pt"
    train.save_checkpoint(path, model, optimizer, 0, config, bundle, 1., lr_scheduler=scheduler)
    payload = torch.load(path, weights_only=False)
    if missing is None:
        del payload["lr_scheduler_state"]
    else:
        payload["lr_scheduler_state"] = missing
    torch.save(payload, path)
    config["training"].update(resume_checkpoint=str(path), epochs=3)
    prepare = Mock(side_effect=AssertionError("must reject before preparing data"))
    monkeypatch.setattr(train, "create_data_bundle", prepare)
    message = "lr_scheduler_state is incomplete" if missing else "lr_scheduler_state is missing"
    with pytest.raises(ValueError, match=message):
        train.train_model(config)
    prepare.assert_not_called()
    with pytest.raises(ValueError, match=message):
        train.load_training_checkpoint(path, model, optimizer, torch.device("cpu"), config,
                                       lr_scheduler=scheduler)


def test_legacy_production_resume_stays_fixed_lr(checkpoint_setup, tmp_path, monkeypatch, capsys):
    config, bundle, model, optimizer, scheduler = checkpoint_setup
    del config["training"]["lr_scheduler"]
    path = tmp_path / "legacy.pt"
    train.save_checkpoint(path, model, optimizer, 0, config, bundle, 1.)
    requested = copy.deepcopy(config)
    requested["training"].update(resume_checkpoint=str(path), epochs=3,
                                  early_stopping_enabled=False,
                                  lr_scheduler=dict(enabled=True, patience=0))
    monkeypatch.setattr(train, "refresh_loss_curves", Mock())
    monkeypatch.setattr(train, "_format_epoch_summary", lambda *args: "synthetic epoch")
    monkeypatch.setattr(train, "run_epoch", lambda **kwargs: dict(
        supervised_objective=2., cd8_loss=2., batch_aware_objective=2.))
    history_path = Path(config["training"]["output_dir"]) / "history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("epoch,validation_supervised_objective\n1,1.0\n")
    best = train.train_model(requested)
    history = pd.read_csv(best.with_name("history.csv"))
    assert pd.isna(history.learning_rate.iloc[0]) and pd.isna(history.next_learning_rate.iloc[0])
    assert history.learning_rate.tolist()[1:] == history.next_learning_rate.tolist()[1:] == [1e-6, 1e-6]
    latest = torch.load(best.with_name("latest.pt"), weights_only=False)
    assert not latest["config"]["training"]["lr_scheduler"]["enabled"]
    assert "lr_scheduler_state" not in latest
    assert "training.lr_scheduler.enabled" in capsys.readouterr().out


def test_production_metric_switch_retains_scheduler_without_history(spatial_config, monkeypatch, capsys):
    config = spatial_config
    config["training"].update(epochs=3, early_stopping_enabled=False)
    config["training"]["lr_scheduler"].update(enabled=True, patience=1, cooldown=2)
    monkeypatch.setattr(train, "refresh_loss_curves", Mock())
    monkeypatch.setattr(train, "_format_epoch_summary", lambda *args: "synthetic epoch")
    monkeypatch.setattr(train, "run_epoch", lambda **kwargs: dict(
        supervised_objective=10., cd8_loss=10. - kwargs["epoch_index"], batch_aware_objective=10.))
    best = train.train_model(config)
    latest_path = best.with_name("latest.pt")
    initial = torch.load(latest_path, weights_only=False)
    assert initial["lr_scheduler_state"]["cooldown_counter"] == 2
    assert initial["lr_scheduler_state"]["last_epoch"] == 3
    best.with_name("history.csv").unlink()
    requested = copy.deepcopy(config)
    requested["training"].update(epochs=8, resume_checkpoint=str(latest_path),
                                  early_stopping_metric="cd8_loss")
    requested["training"]["lr_scheduler"].update(enabled=False, patience=0, min_lr_factor=.5)
    train.train_model(requested)
    final = torch.load(latest_path, weights_only=False)
    _, expected_scheduler = scheduler_pair(config["training"]["learning_rate"], patience=1, cooldown=2)
    for _ in range(8):
        step_lr_scheduler(expected_scheduler, dict(supervised_objective=10.))
    assert final["lr_scheduler_state"] == expected_scheduler.state_dict()
    assert final["optimizer_state"]["param_groups"][0]["lr"] == expected_scheduler.get_last_lr()[0]
    assert final["config"]["training"]["lr_scheduler"] == config["training"]["lr_scheduler"]
    assert final["selection_start_epoch"] == 4
    assert final["best_epoch"] == 8
    assert final["early_stopping_state"]["stale_epochs"] == 0
    assert "training.lr_scheduler.enabled" in capsys.readouterr().out
    history = pd.read_csv(best.with_name("history.csv"))
    assert history.epoch.tolist() == [4, 5, 6, 7, 8]
    assert history.learning_rate.iloc[0] == initial["optimizer_state"]["param_groups"][0]["lr"]


def test_production_rejects_nonfinite_scheduler_score_before_saving(spatial_config, monkeypatch):
    config = spatial_config
    config["training"].update(epochs=1, early_stopping_metric="cd8_loss")
    monkeypatch.setattr(train, "run_epoch", lambda **kwargs: dict(
        supervised_objective=float("nan"), cd8_loss=1.))
    with pytest.raises(ValueError, match="finite validation supervised_objective"):
        train.train_model(config)
    assert not list(Path(config["training"]["output_dir"]).glob("*.pt"))

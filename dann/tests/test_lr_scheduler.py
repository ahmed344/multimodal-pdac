"""Plateau numerics, immutable snapshots, and production resume boundaries."""

import copy
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
import yaml

from dann import train
from dann.config import load_config, lr_scheduler_settings, read_config, resolve_execution
from dann.data_loader import create_data_bundle
from dann.lr_scheduler import create_lr_scheduler, step_lr_scheduler
from dann.model import AdversarialLatentFusion
from dann.selection import OBJECTIVE_METRICS
from dann.tests.test_components import spatial_config
from dann.tests.test_training_artifacts import assert_nested_equal, worker


def scheduler_pair(starting_lr=1e-6, **settings):
    config = dict(training=dict(
        learning_rate=starting_lr, lr_scheduler=dict(enabled=True, **settings)))
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=starting_lr)
    return optimizer, create_lr_scheduler(optimizer, config)


def test_config_defaults_and_mode_floors(tmp_path):
    config = load_config(Path("dann/config.yaml"))
    expected = dict(enabled=True, metric="total_loss", factor=.5, patience=5, cooldown=2,
                    threshold=.001, min_lr_factor=.01, eps=0.)
    assert config["training"]["lr_scheduler"] == expected
    assert lr_scheduler_settings({})["metric"] == "total_loss"
    assert lr_scheduler_settings(dict(lr_scheduler=dict(enabled=True)))["metric"] == "total_loss"
    assert config["training"]["early_stopping_patience"] == 10
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
    *[("metric", value) for value in ("unknown_loss", "", None, True, 1, [], {})],
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


@pytest.mark.parametrize("key,value", [("patience", 1.5), ("metric", "unknown_loss")])
def test_config_loader_validates_scheduler(tmp_path, key, value):
    config = read_config(Path("dann/config.yaml"))
    config["training"]["lr_scheduler"][key] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=f"lr_scheduler.{key}"):
        load_config(path)


@pytest.mark.parametrize("metric", OBJECTIVE_METRICS)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("discriminator_enabled", [False, True])
def test_scheduler_objectives_and_discriminator_requirement(spatial_config, metric, enabled, discriminator_enabled):
    config = spatial_config
    config["model"]["heads"]["discriminator"]["enabled"] = discriminator_enabled
    config["training"]["lr_scheduler"].update(enabled=enabled, metric=metric)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-6)
    if enabled and metric == "batch_aware_objective" and not discriminator_enabled:
        with pytest.raises(ValueError, match="lr_scheduler.metric.*requires an enabled discriminator"):
            resolve_execution(config)
        with pytest.raises(ValueError, match="lr_scheduler.metric.*requires an enabled discriminator"):
            create_lr_scheduler(optimizer, config)
    else:
        assert resolve_execution(config)["training"]["lr_scheduler"]["metric"] == metric
        scheduler = create_lr_scheduler(optimizer, config)
        assert (scheduler.metric if enabled else scheduler) == (metric if enabled else None)


def test_resume_validates_effective_scheduler_discriminator_requirement(spatial_config):
    saved = spatial_config
    saved["model"]["heads"]["discriminator"]["enabled"] = False
    resolve_execution(saved)
    requested = copy.deepcopy(saved)
    requested["training"]["lr_scheduler"].update(enabled=False, metric="batch_aware_objective")
    resolve_execution(requested)  # The current YAML alone permits a disabled scheduler.
    with pytest.raises(ValueError, match="lr_scheduler.metric.*requires an enabled discriminator"):
        train.resume_training_config(saved, requested)  # The saved run keeps scheduling enabled.


def test_default_patience_cooldown_and_absolute_threshold():
    optimizer, scheduler = scheduler_pair()
    # Negative objectives distinguish absolute thresholds from relative thresholds.
    values = [-10.] + [-10.0005] * 6 + [-10.0005] * 2 + [-10.002] + [-10.002] * 6
    states = []
    for value in values:
        step_lr_scheduler(scheduler, dict(total_loss=value))
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
        step_lr_scheduler(scheduler, dict(total_loss=value))
    assert optimizer.param_groups[0]["lr"] == 5e-7
    assert scheduler.best == 2.
    step_lr_scheduler(scheduler, dict(total_loss=1.874))
    assert scheduler.best == 1.874
    assert optimizer.param_groups[0]["lr"] == 5e-7


@pytest.mark.parametrize("starting_lr", [1e-6, .001, .0001])
def test_plateaus_reach_exact_original_floor(starting_lr):
    optimizer, scheduler = scheduler_pair(starting_lr)
    rates = []
    for _ in range(100):
        step_lr_scheduler(scheduler, dict(total_loss=1.))
        rates.append(optimizer.param_groups[0]["lr"])
    floor = starting_lr * .01
    assert rates[-1] == floor
    assert all(rate >= floor for rate in rates)
    assert all(a >= b for a, b in zip(rates, rates[1:]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("metric", OBJECTIVE_METRICS)
def test_nonfinite_validation_does_not_advance_scheduler(value, metric):
    optimizer, scheduler = scheduler_pair(metric=metric)
    step_lr_scheduler(scheduler, {metric: 1.})
    before = copy.deepcopy(scheduler.state_dict())
    with pytest.raises(ValueError, match=f"finite validation {metric}"):
        step_lr_scheduler(scheduler, {**dict.fromkeys(OBJECTIVE_METRICS, 1.), metric: value})
    assert scheduler.state_dict() == before
    assert optimizer.param_groups[0]["lr"] == 1e-6
    step_lr_scheduler(None, {})  # Disabled scheduling does not require a monitor.


@pytest.mark.parametrize("metric", OBJECTIVE_METRICS)
def test_missing_validation_does_not_advance_scheduler(metric):
    optimizer, scheduler = scheduler_pair(metric=metric)
    step_lr_scheduler(scheduler, {metric: 1.})
    before = copy.deepcopy(scheduler.state_dict())
    with pytest.raises(ValueError, match=f"validation {metric}.*missing"):
        step_lr_scheduler(scheduler, {key: 1. for key in OBJECTIVE_METRICS if key != metric})
    assert scheduler.state_dict() == before
    assert optimizer.param_groups[0]["lr"] == 1e-6


@pytest.mark.parametrize("metric,selection", [(metric, selection) for metric in OBJECTIVE_METRICS
                                             for selection in OBJECTIVE_METRICS if metric != selection])
def test_scheduler_and_stopping_follow_opposing_trajectories(metric, selection):
    optimizer, scheduler = scheduler_pair(metric=metric, patience=0, cooldown=0)
    stopper = train.EarlyStopping(patience=1, min_delta=.001)
    for row in ({metric: 3., selection: 2.}, {metric: 4., selection: 1.}):
        step_lr_scheduler(scheduler, row)
        assert stopper.update(row[selection]) == (True, False)
    assert optimizer.param_groups[0]["lr"] == 5e-7
    assert scheduler.best == 3.
    step_lr_scheduler(scheduler, {metric: 2., selection: 5.})
    assert stopper.update(5.) == (False, True)
    assert optimizer.param_groups[0]["lr"] == 5e-7
    assert scheduler.best == 2.


def test_scheduler_threshold_is_independent_of_stopping_delta():
    optimizer, scheduler = scheduler_pair(patience=0, cooldown=0, threshold=.125)
    stopper = train.EarlyStopping(patience=1, min_delta=.01)
    for value in (2., 1.95):
        step_lr_scheduler(scheduler, dict(total_loss=value))
        assert stopper.update(value) == (True, False)
    assert optimizer.param_groups[0]["lr"] == 5e-7
    assert scheduler.best == 2.
    assert stopper.best == 1.95


@pytest.fixture
def checkpoint_setup(spatial_config, request):
    config = spatial_config
    config["training"]["lr_scheduler"].update(enabled=True, patience=1, cooldown=2,
                                               metric=getattr(request, "param", "total_loss"))
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
@pytest.mark.parametrize("checkpoint_setup", OBJECTIVE_METRICS, indirect=True)
def test_checkpoint_resume_matches_uninterrupted(checkpoint_setup, tmp_path, completed):
    config, bundle, model, optimizer, scheduler = checkpoint_setup
    values = [3.] * 8 + [2.] + [2.] * 35
    for value in values[:completed]:
        step_lr_scheduler(scheduler, {scheduler.metric: value})
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
            step_lr_scheduler(current, {current.metric: value})
        assert optimizer.param_groups[0]["lr"] == restored_optimizer.param_groups[0]["lr"]
        assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.param_groups[0]["lr"] == 1e-8
    assert restored_scheduler.min_lrs == [1e-8]


@pytest.mark.parametrize("checkpoint_setup", OBJECTIVE_METRICS, indirect=True)
def test_async_scheduler_snapshot_is_frozen(checkpoint_setup, worker, tmp_path):
    config, bundle, model, optimizer, scheduler = checkpoint_setup
    for _ in range(3):  # Save immediately after reduction, with cooldown active.
        step_lr_scheduler(scheduler, {scheduler.metric: 3.})
    path = tmp_path / "sync.pt"
    train.save_checkpoint(path, model, optimizer, 2, config, bundle, 3., lr_scheduler=scheduler)
    frozen = worker.snapshot(model, optimizer, scheduler)
    payload = train.checkpoint_payload(train.checkpoint_metadata(model, config, bundle), frozen, 2, 3.)
    worker.submit_epoch([dict(epoch=3)], ["epoch"], {}, {}, payload, ["latest.pt"])
    for _ in range(8):
        step_lr_scheduler(scheduler, {scheduler.metric: 3.})
    scheduler.min_lrs[0] = .1  # Mutable state lists must also be independent.
    scheduler.metric = "cd8_loss" if scheduler.metric != "cd8_loss" else "total_loss"
    worker.flush()
    saved = torch.load(worker.directory / "latest.pt", weights_only=False)
    assert_nested_equal(saved, torch.load(path, weights_only=False))
    assert saved["lr_scheduler_state"]["cooldown_counter"] == 2
    assert saved["lr_scheduler_state"]["min_lrs"] == [1e-8]
    assert saved["lr_scheduler_state"]["metric"] == config["training"]["lr_scheduler"]["metric"]


@pytest.mark.parametrize("completed", [3, 6])
@pytest.mark.parametrize("metric", [None, "supervised_objective"])
@pytest.mark.parametrize("requested_metric", [None, *OBJECTIVE_METRICS])
def test_historical_checkpoint_resumes_requested_metric(
    checkpoint_setup, tmp_path, completed, metric, requested_metric, capsys,
):
    config, bundle, model, optimizer, scheduler = checkpoint_setup
    for _ in range(completed):
        step_lr_scheduler(scheduler, dict(total_loss=-10.))
    path = tmp_path / "supervised.pt"
    train.save_checkpoint(path, model, optimizer, completed - 1, config, bundle, -10.,
                          lr_scheduler=scheduler)
    payload = torch.load(path, weights_only=False)
    state = payload["lr_scheduler_state"]
    if metric is None:
        del state["metric"]  # Historical checkpoints did not name the monitored objective.
        del payload["config"]["training"]["lr_scheduler"]["metric"]
    else:
        state["metric"] = metric
    torch.save(payload, path)

    requested = copy.deepcopy(config)
    if requested_metric is None:
        del requested["training"]["lr_scheduler"]["metric"]
    else:
        requested["training"]["lr_scheduler"]["metric"] = requested_metric
    effective = train.resume_training_config(payload["config"], requested)
    restored_optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    restored_scheduler = create_lr_scheduler(restored_optimizer, effective)
    train.load_training_checkpoint(path, model, restored_optimizer, torch.device("cpu"), effective,
                                   lr_scheduler=restored_scheduler)
    assert_nested_equal(optimizer.state_dict(), restored_optimizer.state_dict())
    resolved_metric = requested_metric or "total_loss"
    switched = resolved_metric != "supervised_objective"
    expected = {**state, "metric": resolved_metric}
    if switched:
        expected.update(best=float("inf"), num_bad_epochs=0, cooldown_counter=0)
    assert restored_scheduler.state_dict() == expected
    assert ("LR scheduler changed from supervised_objective" in capsys.readouterr().out) == switched
    saved_lr = restored_optimizer.param_groups[0]["lr"]
    step_lr_scheduler(restored_scheduler, {resolved_metric: 20.})
    assert restored_scheduler.last_epoch == completed + 1
    if switched:
        assert restored_scheduler.best == 20.
        assert restored_optimizer.param_groups[0]["lr"] == saved_lr


@pytest.mark.parametrize("saved_metric,requested_metric", [(saved, requested) for saved in OBJECTIVE_METRICS
                                                         for requested in OBJECTIVE_METRICS if saved != requested])
def test_metric_switch_preserves_lr_floor_and_epoch_count(saved_metric, requested_metric):
    optimizer, scheduler = scheduler_pair(metric=saved_metric, patience=0, cooldown=2)
    for _ in range(2):
        step_lr_scheduler(scheduler, {saved_metric: -10.})
    state = copy.deepcopy(scheduler.state_dict())
    restored_optimizer, restored = scheduler_pair(starting_lr=.1, metric=requested_metric)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    restored.load_state_dict(state)
    assert restored.state_dict() == {**state, "metric": requested_metric, "best": float("inf"),
                                     "num_bad_epochs": 0, "cooldown_counter": 0}
    assert restored.min_lrs == [1e-8]
    assert restored_optimizer.param_groups[0]["lr"] == 5e-7
    assert state == scheduler.state_dict()  # Loading does not mutate the caller's snapshot.
    step_lr_scheduler(restored, {requested_metric: 20., saved_metric: -10.})
    assert restored.best == 20.
    assert restored.last_epoch == 3
    assert restored_optimizer.param_groups[0]["lr"] == 5e-7


@pytest.mark.parametrize("requested_metric", [None, *OBJECTIVE_METRICS])
def test_resume_metric_comes_from_current_yaml(spatial_config, requested_metric):
    saved = spatial_config
    saved["training"]["lr_scheduler"]["metric"] = "cd8_loss"
    requested = copy.deepcopy(saved)
    requested["training"]["lr_scheduler"] = dict(enabled=False, factor=.8, patience=0, min_lr_factor=.5)
    if requested_metric is not None:
        requested["training"]["lr_scheduler"]["metric"] = requested_metric
    effective = train.resume_training_config(saved, requested)
    assert effective["training"]["lr_scheduler"] == {
        **saved["training"]["lr_scheduler"], "metric": requested_metric or "total_loss"}
    assert saved["training"]["lr_scheduler"]["metric"] == "cd8_loss"
    ignored = train.ignored_resume_settings(requested, effective)
    assert "training.lr_scheduler.factor" in ignored
    assert "training.lr_scheduler.metric" not in ignored


@pytest.mark.parametrize("metric", ["unknown_loss", None, True, [], {}])
def test_unknown_scheduler_metric_rejected_before_data_preparation(checkpoint_setup, tmp_path, monkeypatch, metric):
    config, bundle, model, optimizer, scheduler = checkpoint_setup
    path = tmp_path / "unknown.pt"
    train.save_checkpoint(path, model, optimizer, 0, config, bundle, 1., lr_scheduler=scheduler)
    payload = torch.load(path, weights_only=False)
    payload["lr_scheduler_state"]["metric"] = metric
    torch.save(payload, path)
    config["training"].update(resume_checkpoint=str(path), epochs=3)
    prepare = Mock(side_effect=AssertionError("must reject before preparing data"))
    monkeypatch.setattr(train, "create_data_bundle", prepare)
    with pytest.raises(ValueError, match="unknown LR scheduler metric"):
        train.train_model(config)
    prepare.assert_not_called()
    before = copy.deepcopy(scheduler.state_dict())
    with pytest.raises(ValueError, match="unknown LR scheduler metric"):
        scheduler.load_state_dict(payload["lr_scheduler_state"])
    assert scheduler.state_dict() == before


@pytest.mark.parametrize("async_artifacts", [False, True])
@pytest.mark.parametrize("scheduler_metric, metric, count", [
    ("total_loss", "total_loss", 5),
    ("total_loss", "supervised_objective", 7),
    ("supervised_objective", "cd8_loss", 7),
    ("cd8_loss", "batch_aware_objective", 7),
    ("batch_aware_objective", "total_loss", 7),
])
def test_training_history_checkpoint_timing_and_stopping(
    spatial_config, monkeypatch, capsys, async_artifacts, scheduler_metric, metric, count,
):
    config = spatial_config
    for group in [config["model"]["aggregation"], *config["model"]["heads"].values()]:
        group["type"] = "mlp"
    config["training"].update(epochs=7, early_stopping_patience=4,
                              early_stopping_metric=metric, checkpoint_every=20)
    config["training"]["lr_scheduler"].update(enabled=True, metric=scheduler_metric, patience=1, cooldown=1)
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
            metrics.update(dict.fromkeys(OBJECTIVE_METRICS, 10. - kwargs["epoch_index"]))
            metrics[scheduler_metric] = 20.
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
    assert latest["lr_scheduler_state"]["metric"] == scheduler_metric
    assert latest["selection_metric"] == metric
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


@pytest.mark.parametrize("metric", OBJECTIVE_METRICS)
def test_legacy_production_resume_stays_fixed_lr(checkpoint_setup, tmp_path, monkeypatch, capsys, metric):
    config, bundle, model, optimizer, scheduler = checkpoint_setup
    del config["training"]["lr_scheduler"]
    path = tmp_path / "legacy.pt"
    train.save_checkpoint(path, model, optimizer, 0, config, bundle, 1.)
    requested = copy.deepcopy(config)
    requested["training"].update(resume_checkpoint=str(path), epochs=3,
                                  early_stopping_enabled=False,
                                  lr_scheduler=dict(enabled=True, metric=metric, patience=0))
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


def test_production_stopping_metric_switch_retains_scheduler_without_history(spatial_config, monkeypatch, capsys):
    config = spatial_config
    config["training"].update(epochs=3, early_stopping_enabled=False)
    config["training"]["lr_scheduler"].update(enabled=True, patience=1, cooldown=2)
    monkeypatch.setattr(train, "refresh_loss_curves", Mock())
    monkeypatch.setattr(train, "_format_epoch_summary", lambda *args: "synthetic epoch")
    monkeypatch.setattr(train, "run_epoch", lambda **kwargs: dict(
        total_loss=20., supervised_objective=10., cd8_loss=10. - kwargs["epoch_index"], batch_aware_objective=10.))
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
        step_lr_scheduler(expected_scheduler, dict(total_loss=20.))
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


@pytest.mark.parametrize("saved_metric,requested_metric", [
    ("total_loss", "supervised_objective"), ("supervised_objective", "cd8_loss"),
    ("cd8_loss", "batch_aware_objective"), ("batch_aware_objective", None),
])
def test_production_scheduler_metric_switch_retains_stopping_without_history(
    spatial_config, monkeypatch, capsys, saved_metric, requested_metric,
):
    config = spatial_config
    config["training"].update(epochs=3, early_stopping_enabled=False)
    config["training"]["lr_scheduler"].update(enabled=True, metric=saved_metric, patience=1, cooldown=2)
    monkeypatch.setattr(train, "refresh_loss_curves", Mock())
    monkeypatch.setattr(train, "_format_epoch_summary", lambda *args: "synthetic epoch")
    monkeypatch.setattr(train, "run_epoch", lambda **kwargs: dict.fromkeys(OBJECTIVE_METRICS, 20.))
    best = train.train_model(config)
    latest_path = best.with_name("latest.pt")
    initial = torch.load(latest_path, weights_only=False)
    assert initial["lr_scheduler_state"]["cooldown_counter"] == 2
    assert initial["early_stopping_state"]["stale_epochs"] == 2
    best.with_name("history.csv").unlink()
    requested = copy.deepcopy(config)
    requested["training"].update(epochs=4, resume_checkpoint=str(latest_path))
    requested["training"]["lr_scheduler"].update(enabled=False, patience=0, min_lr_factor=.5)
    if requested_metric is None:
        del requested["training"]["lr_scheduler"]["metric"]
    else:
        requested["training"]["lr_scheduler"]["metric"] = requested_metric
    train.train_model(requested)
    final = torch.load(latest_path, weights_only=False)
    resolved_metric = requested_metric or "total_loss"
    assert final["lr_scheduler_state"] == {
        **initial["lr_scheduler_state"], "metric": resolved_metric, "last_epoch": 4,
        "best": 20., "num_bad_epochs": 0, "cooldown_counter": 0,
    }
    assert_nested_equal(final["optimizer_state"], initial["optimizer_state"])
    assert final["early_stopping_state"] == {**initial["early_stopping_state"], "stale_epochs": 3}
    for key in ("selection_metric", "selection_start_epoch", "best_epoch", "best_selection_score"):
        assert final[key] == initial[key]
    assert final["config"]["training"]["lr_scheduler"] == {
        **config["training"]["lr_scheduler"], "metric": resolved_metric}
    assert f"LR scheduler changed from {saved_metric} to {resolved_metric}" in capsys.readouterr().out
    history = pd.read_csv(best.with_name("history.csv"))
    assert history.epoch.tolist() == [4]
    assert history.learning_rate.iloc[0] == initial["optimizer_state"]["param_groups"][0]["lr"]


@pytest.mark.parametrize("metric", OBJECTIVE_METRICS)
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf")])
def test_production_rejects_invalid_scheduler_score_before_saving(spatial_config, monkeypatch, metric, value):
    config = spatial_config
    config["training"].update(epochs=1, early_stopping_metric="cd8_loss" if metric != "cd8_loss" else "total_loss")
    config["training"]["lr_scheduler"]["metric"] = metric
    scores = dict.fromkeys(OBJECTIVE_METRICS, 1.)
    if value is None:
        del scores[metric]
    else:
        scores[metric] = value
    monkeypatch.setattr(train, "run_epoch", lambda **kwargs: scores)
    with pytest.raises(ValueError, match=f"validation {metric}"):
        train.train_model(config)
    assert not list(Path(config["training"]["output_dir"]).glob("*.pt"))

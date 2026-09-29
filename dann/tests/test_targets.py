"""CD8 normalization, masked likelihood, and checkpoint/export regression tests."""

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from dann.analyze import build_latent_umap_frame
from dann.calibration import build_metrics_frame, load_extracted_from_frame
from dann.losses import ZILNLoss
from dann.targets import (
    checkpoint_analysis_config, checkpoint_contract, normalization_settings,
    prepare_targets, target_contract,
)
from dann.train import load_training_checkpoint, run_epoch, summarize_target_validity

COLUMNS = ["Density_CD8", "Density_Tumor", "Density_Collagen", "Density_Stroma"]
SETTINGS = {"enabled": True, "min_non_tumor_fraction": .01}


def test_normalization_boundaries_and_target_order():
    raw = np.array([
        [.2, 0, .4, .6], [0, 1, .4, .6], [.001, .999, .4, .6],
        [.005, .99, .4, .6], [0, .5, .4, .6], [.5, .5, .4, .6],
        [.51, .5, .4, .6], [0, np.nextafter(.99, 1), .4, .6],
        [np.nextafter(.5, 1), .5, .4, .6],
    ])
    before = raw.copy()
    with np.errstate(divide="raise", invalid="raise"):
        targets, valid, reasons = prepare_targets(raw, COLUMNS, SETTINGS)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_allclose(targets[:, 1:], raw[:, 1:])
    np.testing.assert_allclose(targets[:, 0], [.2, 0, 0, .5, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(reasons, [0, 1, 1, 0, 0, 0, 2, 1, 2])
    np.testing.assert_array_equal(valid[:, 0], reasons == 0)
    assert valid[:, 1:].all()
    order = [2, 1, 3, 0]
    reordered = prepare_targets(raw[:, order], [COLUMNS[i] for i in order], SETTINGS)
    np.testing.assert_array_equal(reordered[0], targets[:, order])
    np.testing.assert_array_equal(reordered[1], valid[:, order])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inclusive_threshold_at_source_precision(dtype):
    raw = np.array([[0, .99], [0, np.nextafter(dtype(.99), dtype(1))]], dtype=dtype)
    _, valid, reasons = prepare_targets(raw, COLUMNS[:2], SETTINGS)
    np.testing.assert_array_equal(valid[:, 0], [True, False])
    np.testing.assert_array_equal(reasons, [0, 1])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -.01, 1.001])
def test_raw_validation_precedes_float32_rounding_and_exclusion(bad):
    raw = np.array([[bad, 1.]])
    with pytest.raises(ValueError, match="Raw density"):
        prepare_targets(raw, COLUMNS[:2], SETTINGS)
    with pytest.raises(ValueError, match="Raw density"):
        prepare_targets(np.array([[1. + 1e-12, 0.]]), COLUMNS[:2], SETTINGS)


@pytest.mark.parametrize("settings", [{"enabled": "yes"}, {"min_non_tumor_fraction": 0},
    {"min_non_tumor_fraction": np.nan}, {"min_non_tumor_fraction": 1.1}, {"unknown": 1}])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        normalization_settings(settings)


def test_legacy_raw_and_required_columns():
    raw = np.array([[.2, 1.]])
    targets, valid, reasons = prepare_targets(raw, COLUMNS[:2])
    np.testing.assert_allclose(targets, raw)
    assert valid.all() and not reasons.any()
    with pytest.raises(ValueError, match="requires"):
        prepare_targets(raw[:, :1], COLUMNS[:1], SETTINGS)
    with pytest.raises(ValueError, match="unique"):
        prepare_targets(raw, [COLUMNS[0], COLUMNS[0]], SETTINGS)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("all_masked", [False, True])
def test_masked_loss_matches_retained_entries_and_gradients(reduction, all_masked):
    targets = torch.tensor([[float("nan"), .2], [float("nan"), 0.]])
    valid = torch.tensor([[False, True], [False, True]])
    if all_masked:
        valid[:] = False
    logits = torch.zeros_like(targets, requires_grad=True)
    mu = torch.zeros_like(targets, requires_grad=True)
    sigma = torch.ones_like(targets, requires_grad=True)
    loss = ZILNLoss([8., 2.], reduction=reduction)
    result = loss(logits, mu, sigma, targets, valid)
    expected = ZILNLoss([2.], reduction=reduction)(logits[:, 1:], mu[:, 1:], sigma[:, 1:], targets[:, 1:])
    torch.testing.assert_close(result.total, expected.total if not all_masked else torch.tensor(0.))
    assert result.positive_count == (0 if all_masked else 1)
    assert result.valid_weight_mass == (0 if all_masked else 4)
    result.total.backward()
    for tensor in (logits, mu, sigma):
        assert torch.isfinite(tensor.grad).all()
        assert (tensor.grad[~valid] == 0).all()


def test_epoch_mean_uses_valid_mass_not_batch_size():
    class FixedModel(torch.nn.Module):
        def forward(self, batch, grl_strength):
            y = batch["targets"]
            return {"pi_logits": torch.zeros_like(y), "mu": torch.zeros_like(y),
                    "sigma": torch.ones_like(y), "batch_logits": torch.zeros((len(y), 2))}
    y = torch.tensor([[0., .1], [.8, .9], [.3, .5]])
    mask = torch.tensor([[False, True], [True, True], [True, True]])
    def evaluate(slices):
        loader = [{"targets": y[s], "target_valid_mask": mask[s], "batches": torch.zeros(len(y[s]), dtype=torch.long)} for s in slices]
        return run_epoch(FixedModel(), loader, ZILNLoss([8., 1.]), torch.device("cpu"),
                         5., 1., None, None, 0, 1, "constant", .25, 10., None)
    whole, split = evaluate([slice(None)]), evaluate([slice(0, 1), slice(1, 3)])
    for key in whole:
        assert whole[key] == pytest.approx(split[key], rel=1e-6)


def config_and_checkpoint():
    config = {"data": {"target_columns": COLUMNS, "cd8_normalization": SETTINGS},
              "model": {"latent_dim": 2}, "loss": {"logit_epsilon": 1e-5}}
    checkpoint = {"config": copy.deepcopy(config), "target_columns": COLUMNS,
                  "target_transform": target_contract(config)}
    return config, checkpoint


def test_checkpoint_semantics_and_legacy_analysis():
    config, checkpoint = config_and_checkpoint()
    assert checkpoint_contract(checkpoint) == target_contract(config)
    legacy = copy.deepcopy(checkpoint)
    del legacy["target_transform"]
    assert not checkpoint_analysis_config(config, legacy)["data"]["cd8_normalization"]["enabled"]
    mismatch = copy.deepcopy(checkpoint)
    mismatch["target_columns"] = COLUMNS[::-1]
    with pytest.raises(ValueError, match="order"):
        checkpoint_contract(mismatch)
    checkpoint["target_transform"]["version"] = 2
    with pytest.raises(ValueError, match="version"):
        checkpoint_contract(checkpoint)


@pytest.mark.parametrize("change", ["enabled", "threshold", "order", "legacy"])
def test_resume_rejects_changes_before_loading_weights(tmp_path, change):
    config, checkpoint = config_and_checkpoint()
    altered = copy.deepcopy(config)
    if change == "enabled":
        altered["data"]["cd8_normalization"]["enabled"] = False
    elif change == "threshold":
        altered["data"]["cd8_normalization"]["min_non_tumor_fraction"] = .02
    elif change == "order":
        altered["data"]["target_columns"] = COLUMNS[::-1]
    else:
        del checkpoint["target_transform"]
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_training_checkpoint(path, None, None, torch.device("cpu"), altered)


@pytest.mark.parametrize("cd8, tumor, message", [(0., 1., "no valid CD8"), (0., .5, "no valid positive CD8")])
def test_training_requires_observed_and_positive_cd8(cd8, tumor, message):
    raw = np.array([[cd8, tumor, 0., 0.]])
    targets, valid, reasons = prepare_targets(raw, COLUMNS, SETTINGS)
    bundle = SimpleNamespace(metadata=SimpleNamespace(targets=targets, target_valid_mask=valid,
        exclusion_reasons=reasons), split_indices={"train": np.array([0])})
    config, _ = config_and_checkpoint()
    with pytest.raises(ValueError, match=message):
        summarize_target_validity(bundle, config)


def test_export_calibration_round_trip_preserves_raw_and_missing(tmp_path):
    raw = np.array([[.1, .5], [.2, 1.], [0., .2], [.8, .5]])
    targets, valid, _ = prepare_targets(raw, COLUMNS[:2], SETTINGS)
    contract = target_contract({"data": {"target_columns": COLUMNS[:2], "cd8_normalization": SETTINGS}})
    extracted = {"targets": targets, "raw_targets": raw, "target_valid_mask": valid,
        "target_transform": contract, "pi": np.full_like(targets, .5), "mu": np.zeros_like(targets),
        "sigma": np.ones_like(targets), "density_mean": np.full_like(targets, .25),
        "row_ids": np.arange(4), "batches": np.zeros(4, dtype=int)}
    frame = build_latent_umap_frame(extracted, np.zeros((4, 2)), COLUMNS[:2], ["a"], "test")
    assert frame.Density_CD8.isna().tolist() == [False, True, False, True]
    np.testing.assert_array_equal(frame.raw_Density_CD8, raw[:, 0])
    assert frame.label_Density_CD8.eq("Normalized CD8 (non-tumor area)").all()
    path = tmp_path / "roundtrip.csv"
    frame.to_csv(path, index=False)
    restored, _ = load_extracted_from_frame(pd.read_csv(path), COLUMNS[:2])
    assert restored["target_transform"] == contract
    expected = build_metrics_frame(extracted, COLUMNS[:2], "test", 1e-5)
    actual = build_metrics_frame(restored, COLUMNS[:2], "test", 1e-5)
    pd.testing.assert_frame_equal(expected, actual, check_exact=False)
    assert actual.iloc[0]["n"] == 2 and actual.iloc[0]["n_excluded"] == 2
    assert actual.iloc[0]["n_zero"] == 1 and actual.iloc[0]["n_positive"] == 1
    assert actual.iloc[1]["n"] == 4


def test_calibration_all_cd8_excluded_keeps_other_targets():
    extracted = {"targets": np.array([[0., .2], [0., .4]]),
        "target_valid_mask": np.array([[False, True], [False, True]]),
        "pi": np.full((2, 2), .5), "mu": np.zeros((2, 2)), "sigma": np.ones((2, 2))}
    metrics = build_metrics_frame(extracted, COLUMNS[:2], "test", 1e-5)
    assert metrics.iloc[0]["n"] == 0
    assert metrics.iloc[0]["n_excluded"] == 2
    assert np.isnan(metrics.iloc[0]["brier"])
    assert np.isnan(metrics.iloc[0]["rmse"])
    assert metrics.iloc[1]["n"] == 2


def test_spatial_heatmaps_transform_observations_from_export_metadata(tmp_path, monkeypatch):
    import anndata as ad
    import json
    import pyarrow.parquet as pq
    from dann import spatial_heatmaps
    from dann.spatial import build_output_table
    raw = np.array([[.1, .5, .3, .2], [.2, 1., .3, .2], [0., .2, .3, .2]])
    adata = ad.AnnData(np.zeros((3, 1)), obs=pd.DataFrame(raw, columns=COLUMNS, index=["a", "b", "c"]))
    adata.obs["batch"] = "a"
    input_path = tmp_path / "tissue.h5ad"
    input_path.touch()
    monkeypatch.setattr(spatial_heatmaps.sc, "read_h5ad", lambda path: adata)
    config, _ = config_and_checkpoint()
    contract = target_contract(config)
    table = build_output_table(np.arange(3), np.array(["a", "b", "c"]),
        np.zeros((3, 4)), np.full((3, 4), .5), np.ones((3, 4)), COLUMNS,
        density_mean=np.full((3, 4), .25))
    table = table.replace_schema_metadata({b"dann_target_transform": json.dumps(contract).encode()})
    path = tmp_path / "predictions.parquet"
    pq.write_table(table, path)
    captured = []
    def capture(transformed, *args, **kwargs):
        captured.append(transformed.obs.copy())
    monkeypatch.setattr(spatial_heatmaps, "plot_density_heatmap", capture)
    monkeypatch.setattr(spatial_heatmaps, "plot_diagnostics_density_heatmap", capture)
    spatial_heatmaps.run_spatial_heatmaps(input_path, path, tmp_path / "plots", COLUMNS, 1e-5, 72)
    assert len(captured) == 8
    np.testing.assert_allclose(captured[0].Density_CD8, [.2, np.nan, 0.], equal_nan=True)
    np.testing.assert_array_equal(captured[0].raw_Density_CD8, raw[:, 0])
    np.testing.assert_allclose(captured[0].Density_Tumor, raw[:, 1])
    assert np.isnan(captured[0].logit_Density_CD8.iloc[1])
    assert adata.uns["dann_target_transform"] == contract


def test_nearly_constant_predictions_survive_csv_round_trip(tmp_path):
    from dann.calibration import compute_target_metrics
    n = 32
    targets = np.linspace(.001, .9, n, dtype=np.float32).reshape(-1, 1)
    mu = np.linspace(-.6694, -.6693, n, dtype=np.float32).reshape(-1, 1)
    sigma = np.full_like(mu, 1.2)
    pi = np.full_like(mu, .3)
    extracted = {"targets": targets, "pi": pi, "mu": mu, "sigma": sigma,
        "density_mean": np.full_like(mu, .2), "row_ids": np.arange(n),
        "batches": np.zeros(n, dtype=int)}
    frame = build_latent_umap_frame(extracted, np.zeros((n, 2)), COLUMNS[:1], ["a"], "test")
    path = tmp_path / "precise.csv"
    frame.to_csv(path, index=False, float_format="%.17g")
    restored, _ = load_extracted_from_frame(pd.read_csv(path), COLUMNS[:1])
    before = compute_target_metrics(targets, pi, mu, sigma, 1e-5)
    after = compute_target_metrics(restored["targets"], restored["pi"], restored["mu"], restored["sigma"], 1e-5)
    for key in before:
        assert after[key] == pytest.approx(before[key], abs=1e-10, nan_ok=True)


@pytest.mark.parametrize('normalized', [False, True])
def test_missing_tissue_annotations_are_masked_per_target(normalized):
    from dann.targets import prepare_observed_targets
    raw = np.array([
        [np.nan, np.nan, np.nan, np.nan],
        [.1, np.nan, .3, .2],
        [np.nan, .5, .3, .2],
        [.1, .5, np.nan, .2],
        [.1, 1., .3, .2],
    ])
    before = raw.copy()
    targets, valid = prepare_observed_targets(raw, COLUMNS, {'enabled': normalized})
    np.testing.assert_array_equal(raw, before)
    expected = np.isfinite(raw)
    if normalized:
        expected[[1, 4], 0] = False
    np.testing.assert_array_equal(valid, expected)
    np.testing.assert_array_equal(np.isnan(targets), ~valid)
    assert targets[3, 0] == pytest.approx(.2 if normalized else .1)
    np.testing.assert_allclose(targets[:, 1:], raw[:, 1:], equal_nan=True)
    with pytest.raises(ValueError, match='Raw density'):
        prepare_targets(raw, COLUMNS, {'enabled': normalized})


@pytest.mark.parametrize('invalid', [np.inf, -np.inf, 1.1, -.1])
def test_tissue_observed_values_still_require_valid_density(invalid):
    from dann.targets import prepare_observed_targets
    raw = np.array([[invalid, .5, np.nan, .2]])
    with pytest.raises(ValueError, match='Raw density'):
        prepare_observed_targets(raw, COLUMNS, SETTINGS)

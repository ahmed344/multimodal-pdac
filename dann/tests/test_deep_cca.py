"""Numerical CCA, frozen preprocessing, architecture and training contracts."""
import copy
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dann.checkpoints import architecture_contract, checkpoint_config, validate_checkpoint
from dann.config import resolve_execution, seed_everything
from dann.data_loader import create_data_bundle
from dann.deep_cca import (CCAProjectors, CCASummary, SlideCovariance, cca_settings,
                           effective_weight, evaluate_cca)
from dann.initialization import initialize_supervised_heads
from dann.losses import ZILNLoss
from dann.model import AdversarialLatentFusion
from dann.tests.test_components import spatial_config
from dann.tests.test_he import he_config
from dann.train import (load_training_checkpoint, run_epoch, save_checkpoint,
                        resume_training_config, train_model)


def sample(d=4):
    g = torch.Generator().manual_seed(19)
    u = torch.randn(96, d, generator=g, dtype=torch.float64)
    v = u @ torch.randn(d, d, generator=g, dtype=torch.float64) + .3 * torch.randn(96, d, generator=g)
    return u, v, torch.tensor([0]*32 + [1]*64)


def test_reference_equal_slide_weight_offsets_order_and_gradients():
    u, v, slides = sample()
    result = evaluate_cca(u, v, slides)
    z = torch.cat((u, v), 1).numpy()
    cov = np.mean([np.cov(z[slides.numpy() == s].T, ddof=1) for s in (0, 1)], 0)
    d = u.shape[1]
    a, b, ab = cov[:d, :d], cov[d:, d:], cov[:d, d:]
    la, lb = [np.linalg.cholesky(x + (1e-3 * np.trace(x)/d + 1e-8)*np.eye(d)) for x in (a, b)]
    t = np.linalg.solve(lb, np.linalg.solve(la, ab).T).T
    reference = 1 - np.linalg.svd(t, compute_uv=False).mean() + .1 * np.maximum(0, 1-np.sqrt(np.diag(cov)+1e-8)).mean()
    assert result.loss.item() == pytest.approx(reference, abs=1e-12)
    offset = slides[:, None] * 17.
    shifted = evaluate_cca(u + offset, v - 2*offset, slides)
    torch.testing.assert_close(result.loss, shifted.loss, rtol=1e-12, atol=1e-12)
    order = torch.randperm(len(u))
    torch.testing.assert_close(result.loss, evaluate_cca(u[order], v[order], slides[order]).loss)
    u.requires_grad_(); v.requires_grad_()
    assert torch.autograd.gradcheck(lambda x, y: evaluate_cca(x, y, slides).loss, (u, v), fast_mode=True)
    # Deliberately unequal slide sizes must not silently become row weighting.
    pooled = np.cov(z.T)
    assert not np.allclose(cov, pooled)


def test_shuffled_pairs_and_streaming_batch_boundaries():
    u, v, slides = sample()
    expected = evaluate_cca(u, v, slides)
    stream = SlideCovariance(u.shape[1])
    for i in range(0, len(u), 3):
        stream.update(u[i:i+3], v[i:i+3], slides[i:i+3])
    torch.testing.assert_close(stream.finish().loss, expected.loss, rtol=1e-12, atol=1e-12)
    shuffled = v.clone()
    g = torch.Generator().manual_seed(7)
    for s in slides.unique():
        selected = torch.where(slides == s)[0]
        shuffled[selected] = v[selected[torch.randperm(len(selected), generator=g)]]
    assert evaluate_cca(u, shuffled, slides).diagnostics['correlation'] < expected.diagnostics['correlation'] - .3


@pytest.mark.parametrize('collapsed', [False, True])
def test_rank_deficiency_has_finite_loss_and_gradients(collapsed):
    u, _, slides = sample()
    u = (torch.zeros_like(u) if collapsed else u[:, :1].expand_as(u).clone()).requires_grad_()
    v = u.detach().clone().requires_grad_()
    result = evaluate_cca(u, v, slides)
    result.loss.backward()
    assert torch.isfinite(u.grad).all() and torch.isfinite(v.grad).all()
    assert result.diagnostics['msi_rank'] <= 1
    if collapsed:
        assert result.diagnostics['correlation'] == 0
        assert result.diagnostics['variance_penalty'] > .99


def test_insufficient_samples_and_ramp():
    u, v, slides = sample(16)
    assert evaluate_cca(u[:8], v[:8], slides[:8]).loss is None  # 7 df < 17
    assert evaluate_cca(u[:7, :2], v[:7, :2], slides[:7]).loss is None
    summary = CCASummary(.05)
    summary.add(evaluate_cca(u[:8], v[:8], slides[:8]))
    with pytest.raises(ValueError, match='no valid evaluations'):
        summary.finish()
    settings = cca_settings({})
    assert [effective_weight(.05, settings, i, True) for i in range(4)] == [0, .025, .05, .05]
    assert effective_weight(.05, settings, 0, False) == .05


@pytest.fixture
def cca_config(he_config):
    c = he_config
    with h5py.File(c['data']['path'], 'r+') as f:
        f['obsm/hes_features'][:] = np.random.default_rng(41).normal(size=f['obsm/hes_features'].shape)
    for group in [c['model']['aggregation'], *c['model']['heads'].values()]:
        group['type'] = 'mlp'
    c['model']['heads']['he'].update(objective='deep_cca', deep_cca=dict(dimension=2, msi_hidden_dim=5, he_hidden_dim=8))
    c['training']['pixel'].update(batch_size=128, validation_batch_size=3)
    c['training']['spatial'].update(tiles_per_batch=64)
    c['loss']['he_weight'] = .05
    return resolve_execution(c)


def test_training_only_standardization_and_provenance(cca_config, tmp_path):
    c = cca_config
    bundle = create_data_bundle(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    with h5py.File(c['data']['path'], 'r+') as f:
        rows = bundle.split_indices['validation']
        for row in rows:
            f['obsm/hes_features'][int(row)] = 1e5
    report = initialize_supervised_heads(model, bundle, c)
    x = bundle.he_reader.read(bundle.split_indices['train']).astype(np.float64)
    np.testing.assert_allclose(model.he_cca.feature_mean.numpy(), x.mean(0), atol=1e-12)
    np.testing.assert_allclose(model.he_cca.feature_scale.numpy(), x.std(0), atol=1e-12)
    assert report['cca_preprocessing']['fit_split'] == 'train'
    with pytest.raises(ValueError, match='never refit'):
        model.he_cca.fit(bundle.he_reader, rows)
    path = tmp_path/'cca.pt'
    opt = torch.optim.AdamW(model.parameters())
    save_checkpoint(path, model, opt, 0, c, bundle, 1.)
    payload = torch.load(path, weights_only=False)
    assert checkpoint_config(payload)['model']['heads']['he']['objective'] == 'deep_cca'
    payload['model_state']['he_cca.fit_count'] += 1
    with pytest.raises(ValueError, match='provenance'):
        checkpoint_config(payload)


@pytest.mark.parametrize('kind', ['mlp', 'cnn', 'gatv2'])
def test_encoder_both_projector_gradients_and_msi_inference(cca_config, kind, tmp_path):
    c = cca_config
    c['model']['aggregation']['type'] = c['model']['heads']['biology']['type'] = kind
    resolve_execution(c)
    bundle = create_data_bundle(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    initialize_supervised_heads(model, bundle, c)
    batch = next(iter(bundle.loaders['train']))
    output = model(batch)
    assert 'he_prediction' not in output
    loss = evaluate_cca(output['cca_msi'], output['cca_he'], batch['batches']).loss
    assert loss is not None
    loss.backward()
    for module in (model.encoder, model.he_cca.msi, model.he_cca.he):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in model.biology_predictor.parameters())
    model.eval()
    expected = model(batch, predict_he=False)
    msi_batch = {k: v for k, v in batch.items() if k != 'he_targets'}
    for key, value in model(msi_batch).items():
        torch.testing.assert_close(value, expected[key])
    path = tmp_path/'cca.pt'
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.step()
    save_checkpoint(path, model, optimizer, 1, c, bundle, 1.)
    restored = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    restored_opt = torch.optim.AdamW(restored.parameters())
    assert load_training_checkpoint(path, restored, restored_opt, torch.device('cpu'), c)[0] == 2
    for key, value in model(batch).items():
        torch.testing.assert_close(value, restored(batch)[key])


def epoch(model, loader, c, optimizer=None, index=0):
    return run_epoch(model, loader, ZILNLoss.from_config(c), torch.device('cpu'), 1., 1.,
                     None, optimizer, index, 3, 'constant', 0., 10., None, he_weight=.05)


def test_validation_batching_and_supervised_objective(cca_config):
    c = cca_config
    bundle = create_data_bundle(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4).double()
    initialize_supervised_heads(model, bundle, c)
    loader = bundle.loaders['validation']
    scores = epoch(model, loader, c)
    regrouped = DataLoader(loader.dataset, batch_size=100, collate_fn=loader.collate_fn)
    other = epoch(model, regrouped, c)
    for key in ('cca_loss', 'cca_correlation', 'total_loss'):
        assert scores[key] == pytest.approx(other[key], abs=2e-7)
    assert scores['supervised_objective'] == scores['biology_weighted_loss']
    assert scores['cca_effective_weight'] == .05
    scores = epoch(model, bundle.loaders['train'], c, torch.optim.AdamW(model.parameters()), index=1)
    assert scores['cca_effective_weight'] == .025 and scores['cca_valid_count'] > 0


def test_legacy_initialization_and_objective_contracts(cca_config):
    c = cca_config
    raw = copy.deepcopy(c)
    raw['model']['heads']['he']['objective'] = 'raw_mse'
    seed_everything(13)
    a = AdversarialLatentFusion.from_config(raw, 2, 4)
    seed_everything(13)
    b = AdversarialLatentFusion.from_config(c, 2, 4)
    for name in ('encoder', 'biology_predictor'):
        for x, y in zip(getattr(a, name).parameters(), getattr(b, name).parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    with pytest.raises(ValueError, match='fresh training'):
        resume_training_config(raw, c)
    bad = copy.deepcopy(c)
    bad['analysis']['export_he_predictions'] = True
    with pytest.raises(ValueError, match='export'):
        resolve_execution(bad)
    bad = copy.deepcopy(c)
    bad['model']['aggregation']['type'] = 'cnn'
    bad['training']['spatial'].update(sampling_strategy='proportional_slide_tiles', supervised_rows_per_batch=32, microbatch_rows=16)
    with pytest.raises(ValueError, match='unsplit'):
        resolve_execution(bad)
    # Version 9 raw-MSE checkpoints migrate without changing their architecture.
    legacy = architecture_contract(raw)
    legacy['version'] = 9
    legacy['components']['he'].pop('objective')
    validate_checkpoint(dict(config=raw, architecture=legacy), raw, a)


def test_resume_keeps_preprocessing_ramp_and_dropout(cca_config, tmp_path):
    c = cca_config
    c['model']['dropout'] = .1
    c['training'].update(epochs=3, checkpoint_every=1, model_seed=92, early_stopping_enabled=False)
    c['monitoring']['enabled'] = False
    c['post_training']['enabled'] = False
    c['training']['lr_scheduler']['enabled'] = False
    c['results']['run_name'] = 'full'
    resolve_execution(c)
    train_model(c)
    full = torch.load(Path(c['training']['output_dir'])/'latest.pt', weights_only=False)
    c['results']['run_name'] = 'resumed'
    resolve_execution(c)
    train_model(c, stop_after_epoch=1)
    c['training']['resume_checkpoint'] = str(Path(c['training']['output_dir'])/'latest.pt')
    train_model(c)
    resumed = torch.load(c['training']['resume_checkpoint'], weights_only=False)
    for key, value in full['model_state'].items():
        torch.testing.assert_close(value, resumed['model_state'][key], rtol=0, atol=0)
    assert resumed['epoch'] == 2


def test_separate_projector_rng_and_raw_default(cca_config):
    c = cca_config
    no_he = copy.deepcopy(c)
    no_he['model']['heads']['he']['enabled'] = False
    seed_everything(45)
    AdversarialLatentFusion.from_config(no_he, 2, 4)
    expected_cpu = torch.get_rng_state()
    expected_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    seed_everything(45)
    AdversarialLatentFusion.from_config(c, 2, 4)
    assert torch.equal(expected_cpu, torch.get_rng_state())
    for before, after in zip(expected_cuda, torch.cuda.get_rng_state_all() if expected_cuda else []):
        assert torch.equal(before, after)
    c['model']['heads']['he']['objective'] = 'raw_mse'
    seed_everything(22)
    explicit = AdversarialLatentFusion.from_config(c, 2, 4)
    c['model']['heads']['he'].pop('objective')
    seed_everything(22)
    omitted = AdversarialLatentFusion.from_config(c, 2, 4)
    for key, value in explicit.state_dict().items():
        torch.testing.assert_close(value, omitted.state_dict()[key], rtol=0, atol=0)


def test_analysis_summary_and_export_rejection(cca_config, tmp_path):
    import json
    from dann.he import extract_with_he
    c = cca_config
    bundle = create_data_bundle(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    initialize_supervised_heads(model, bundle, c)
    extract_with_he(model, bundle.loaders['validation'], torch.device('cpu'), c, tmp_path, 'validation')
    metrics = json.loads((tmp_path/'he/validation_cca_metrics.json').read_text())
    assert metrics['cca_valid_count'] == 1
    assert metrics['cca_effective_weight'] == .05
    assert not (tmp_path/'he/validation_predictions.parquet').exists()
    c['analysis']['export_he_predictions'] = True
    with pytest.raises(ValueError, match='export'):
        extract_with_he(model, bundle.loaders['validation'], torch.device('cpu'), c, tmp_path, 'validation')


def test_analysis_insufficient_pairs_remain_unavailable(tmp_path):
    import json
    from dann.he import report_he
    arrays = dict(cca_msi=np.zeros((7, 2)), cca_he=np.zeros((7, 2)), batches=np.zeros(7, dtype=int))
    report_he(arrays, tmp_path, 'validation', .05)
    scores = json.loads((tmp_path/'he/validation_cca_metrics.json').read_text())
    assert scores['cca_loss'] is None
    assert scores['cca_valid_count'] == 0
    assert scores['cca_invalid_insufficient_samples'] == 1

"""Set Transformer paper equations, sparse membership, and public DANN contracts."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dann.analyze import load_model
from dann.checkpoints import architecture_contract, checkpoint_config, validate_checkpoint
from dann.config import component_settings, load_config, resolve_execution
from dann.data_loader import create_data_bundle
from dann.model import AdversarialLatentFusion, IntensityWeightedPeakEncoder
from dann.model_components import set_transformer
from dann.model_components.set_transformer import ISAB, MAB, SAB, SetTransformerEncoder
from dann.spatial import run_inference
from dann.tests.test_components import spatial_config
from dann.tests.test_spatial_execution import options
from dann.train import load_training_checkpoint, resume_training_config, save_checkpoint


def select(config, block, width):
    config['model']['spectral_encoder'] = dict(type='set_transformer', set_transformer=dict(
        block_type=block, d_model=width, num_inducing_points=3))
    config['model']['latent_dim'] = 4 * width
    return resolve_execution(config)


def mab_reference(module, a, b):
    # Evaluate one unpadded pixel, one head at a time. Use model width in the scale.
    q, k, v = (proj(x).chunk(module.heads, dim=-1) for proj, x in
               [(module.query, a), (module.key, b), (module.value, b)])
    heads = [(qi @ ki.T / a.shape[-1] ** .5).softmax(-1) @ vi for qi, ki, vi in zip(q, k, v)]
    h = module.attention_norm(a + module.output(torch.cat(heads, dim=-1)))
    return module.output_norm(h + module.rff(h))


def test_mab_and_induced_return_against_explicit_attention():
    torch.manual_seed(19)
    x = torch.randn(5, 8, dtype=torch.float64)
    y = torch.randn(3, 8, dtype=torch.float64)
    mab = MAB(8, 2).double()
    torch.testing.assert_close(mab(x[None], y[None])[0], mab_reference(mab, x, y))
    block = ISAB(8, 2, 3).double()
    h = mab_reference(block.to_inducing, block.inducing, x)
    expected = mab_reference(block.to_peaks, x, h)
    torch.testing.assert_close(block(x[None])[0], expected)


def sparse_batch():
    # Interleaved pixels, a repeated bin, explicit zero, zero-only and empty rows.
    return dict(peak_indices=torch.tensor([3, 2, 1, 3, 4, 0, 2, 1]),
                intensities=torch.tensor([.4, 2., 3., .6, .5, 4., 0., 0.]),
                sample_indices=torch.tensor([2, 0, 2, 2, 0, 2, 1, 2]),
                peak_counts=torch.tensor([2, 1, 5, 0]))


@pytest.mark.parametrize('block', ['sab', 'isab'])
@pytest.mark.parametrize('width', [16, 32])
def test_factory_shape_trace_and_optimizer(spatial_config, block, width):
    config = select(spatial_config, block, width)
    config['model']['aggregation']['type'] = config['model']['heads']['biology']['type'] = 'mlp'
    model = AdversarialLatentFusion.from_config(config, 2, 4).train()
    encoder = model.encoder
    assert type(encoder) is SetTransformerEncoder
    assert len(encoder.blocks) == 2 and all(type(b) is (SAB if block == 'sab' else ISAB) for b in encoder.blocks)
    assert encoder.blocks[0] is not encoder.blocks[1]
    assert isinstance(encoder.decoder_sab, SAB)
    batch = sparse_batch()
    config_trace = {}
    def trace(name):
        def hook(module, inputs, output):
            config_trace[name] = (inputs[0].shape, output.shape)
        return hook
    modules = dict(adapter=encoder.input_projection, block1=encoder.blocks[0], block2=encoder.blocks[1],
                   pma=encoder.pma, seed_sab=encoder.decoder_sab, out=encoder.rff_out,
                   aggregation=encoder.aggregation_mlp)
    hooks = [module.register_forward_hook(trace(name)) for name, module in modules.items()]
    before = encoder.seeds.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    result = model(batch)
    for hook in hooks:
        hook.remove()
    assert result['latent'].shape == (4, 4 * width)
    assert config_trace['adapter'][0][-1] == 9
    for name in ('block1', 'block2'):
        assert config_trace[name][1][-2:] == (3, width)
    for name in ('pma', 'seed_sab', 'out'):
        assert config_trace[name][1][-2:] == (4, width)
    assert config_trace['aggregation'][0][-1] == 4 * width
    assert encoder.rff_in[0].out_features == 2 * width
    result['mu'].square().mean().backward()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    optimizer.step()
    assert torch.isfinite(result['mu']).all() and not torch.equal(before, encoder.seeds)


@pytest.mark.parametrize('block', ['sab', 'isab'])
def test_membership_padding_duplicates_empty_and_query_chunks(spatial_config, block, monkeypatch):
    encoder = AdversarialLatentFusion.from_config(select(spatial_config, block, 16), 2, 4).encoder.double().eval()
    batch = sparse_batch()
    batch['intensities'] = batch['intensities'].double()
    pooled = encoder(**batch)
    assert torch.count_nonzero(pooled[[1, 3]]) == 0
    for row, (ids, values) in enumerate([([2, 4], [2., .5]), ([], []), ([0, 1, 3], [4., 3., 1.]), ([], [])]):
        count = len(ids)
        alone = encoder(torch.tensor(ids, dtype=torch.long), torch.tensor(values, dtype=torch.double),
                        torch.zeros(count, dtype=torch.long), torch.tensor([count]))
        torch.testing.assert_close(pooled[row], alone[0])
    perm = torch.tensor([7, 5, 3, 1, 2, 0, 6, 4])
    permuted = {k: (v if k == 'peak_counts' else v[perm]) for k, v in batch.items()}
    torch.testing.assert_close(pooled, encoder(**permuted))
    monkeypatch.setattr(set_transformer, 'MAX_ATTENTION_SCORES', 24)
    encoder.train()
    chunked = encoder.microbatched(**permuted, peak_budget=1, checkpointing=True)
    torch.testing.assert_close(chunked, pooled)
    grad = torch.autograd.grad(chunked.square().sum(), encoder.embedding.weight)[0]
    reference = torch.autograd.grad(pooled.square().sum(), encoder.embedding.weight)[0]
    torch.testing.assert_close(grad, reference)


def test_main_selector_contract_and_results():
    config = load_config('dann/config.yaml')
    selector = config['model']['spectral_encoder']
    original = architecture_contract(config)
    assert config['results']['run_name'] == 'set_transformer_sab_128'
    assert type(AdversarialLatentFusion.from_config(config, 2, 4).encoder) is SetTransformerEncoder
    assert config['execution']['combination'].startswith('set_transformer__')
    assert original['components']['spectral_encoder']['semantics']['pooled_dim'] == 128
    assert config['execution']['spectral_encoder'] == original['components']['spectral_encoder']
    inactive_edit = copy.deepcopy(config)
    inactive_edit['model']['spectral_encoder']['set_transformer']['num_inducing_points'] += 1
    inactive_edit['model']['spectral_encoder']['deep_sets']['embedding_dim'] += 1
    assert architecture_contract(inactive_edit) == original
    selector['type'] = 'deep_sets'
    resolve_execution(config)
    encoder = AdversarialLatentFusion.from_config(config, 2, 4).encoder
    assert isinstance(encoder, IntensityWeightedPeakEncoder) and encoder.pooling == selector['deep_sets']['pooling']
    assert config['execution']['combination'].startswith('deep_sets__')
    historic = architecture_contract(config)
    selector['set_transformer']['num_heads'] = 0
    assert architecture_contract(config) == historic
    legacy = copy.deepcopy(config)
    del legacy['model']['spectral_encoder']['set_transformer']
    historic['version'] = 11
    restored = checkpoint_config(dict(config=legacy, architecture=historic, cca_contract=historic['components']['he']['deep_cca']))
    assert architecture_contract(restored) == architecture_contract(config)


@pytest.mark.parametrize('key,value', [('embedding_dim', True), ('d_model', 17), ('num_blocks', 0),
                                       ('num_heads', 0), ('num_seeds', -1), ('num_inducing_points', 2.5),
                                       ('block_type', 'cross')])
def test_active_settings_validation(spatial_config, key, value):
    config = select(spatial_config, 'isab', 32)
    config['model']['spectral_encoder']['set_transformer'][key] = value
    with pytest.raises(ValueError, match='set_transformer'):
        component_settings(config['model'])


@pytest.mark.parametrize('block,spatial', [('sab', False), ('isab', True)])
def test_training_restore_analysis_and_tissue_exports(spatial_config, tmp_path, block, spatial):
    config = select(spatial_config, block, 16)
    config['model']['latent_dim'] = 6  # Pooled features and exported latents have independent widths.
    if not spatial:
        config['model']['aggregation']['type'] = config['model']['heads']['biology']['type'] = 'mlp'
    resolve_execution(config)
    bundle = create_data_bundle(config)
    model = AdversarialLatentFusion.from_config(config, 2, 4)
    opt = torch.optim.AdamW(model.parameters())
    batch = next(iter(bundle.loaders['train']))
    model(batch)['mu'].square().mean().backward()
    opt.step()
    path = tmp_path/'checkpoint.pt'
    save_checkpoint(path, model, opt, 0, config, bundle, 1.)
    restored, epoch = load_model(path, config, bundle, torch.device('cpu'))
    assert epoch == 1
    model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(batch)['latent'], model(batch)['latent'])
    new_opt = torch.optim.AdamW(restored.parameters())
    load_training_checkpoint(path, restored, new_opt, torch.device('cpu'), config)
    torch.testing.assert_close(new_opt.state[restored.encoder.seeds]['exp_avg'], opt.state[model.encoder.seeds]['exp_avg'])
    changed = copy.deepcopy(config)
    changed['model']['spectral_encoder']['set_transformer']['num_heads'] = 2
    with pytest.raises(ValueError, match='fresh training'):
        resume_training_config(config, changed)
    payload = torch.load(path, weights_only=False)
    with pytest.raises(ValueError, match='incompatible'):
        validate_checkpoint(payload, changed)
    with pytest.raises(ValueError, match='incompatible'):
        validate_checkpoint(payload, config, AdversarialLatentFusion.from_config(changed, 2, 4))
    switched = copy.deepcopy(config)
    switched['model']['spectral_encoder']['set_transformer']['block_type'] = 'isab' if block == 'sab' else 'sab'
    with pytest.raises(ValueError, match='incompatible'):
        validate_checkpoint(payload, switched)
    payload['architecture']['components']['spectral_encoder']['semantics']['attention_scale'] = '1/sqrt(head_dim)'
    with pytest.raises(ValueError, match='architecture metadata'):
        checkpoint_config(payload)
    for dataset in bundle.datasets.values():
        dataset.close()
    source = Path(config['data']['context_path'])
    def export(label, overrides):
        output = tmp_path/label/'prediction.parquet'
        run_inference(source, path, output, max_rows=7, execution_overrides=overrides)
        predictions = pd.read_parquet(output)
        latents = pd.read_parquet(output.with_name('spatial_latent.parquet'))
        assert latents.row_position.tolist() == predictions.row_position.tolist() == list(range(7))
        assert latents.filter(regex='^latent_').shape == (7, 6)
        return predictions.select_dtypes('number'), latents.filter(regex='^latent_')
    baseline = export('direct', {})
    if spatial:
        cached = export('cached', options(density_workers=0, feature_cache=True))
        for a, b in zip(baseline, cached):
            np.testing.assert_allclose(a, b, atol=5e-5, rtol=5e-5)

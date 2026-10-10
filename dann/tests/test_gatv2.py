"""Graph geometry, numerical behavior, dependency isolation, and v5 contracts."""
import builtins
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from dann.checkpoints import architecture_contract, checkpoint_config, validate_checkpoint
from dann.config import resolve_execution
from dann.model import AdversarialLatentFusion
from dann.model_components.aggregation_gatv2 import AggregationGATv2
from dann.model_components.biology_gatv2 import BiologyGATv2
from dann.model_components.discriminator_gatv2 import DiscriminatorGATv2
from dann.model_components.layers_gatv2 import grid_graph, grid_offsets, graph_reach
from dann.tests.test_components import spatial_config, predict_tiles


@pytest.mark.parametrize('radius,count', [(1., 5), (1.5, 9), (2., 13)])
def test_graph_geometry(radius, count):
    assert len(grid_offsets(radius)) == count
    mask = torch.ones(2, 1, 5, 6)
    mask[:, :, 2, 2] = 0  # Missing endpoints; gaps can still be spanned.
    positions, edges, features = grid_graph(mask, radius, torch.float64)
    coords = [(int(p)//30, int(p)%6, int(p)%30//6) for p in positions]
    expected = {(j, i) for j, (tj, xj, yj) in enumerate(coords)
                for i, (ti, xi, yi) in enumerate(coords)
                if ti == tj and math.hypot(xj-xi, yj-yi) <= radius}
    assert set(map(tuple, edges.T.tolist())) == expected
    assert len(expected) == edges.shape[1]  # No duplicate edges, including self.
    for (j, i), feature in zip(edges.T.tolist(), features):
        tj, xj, yj = coords[j]
        ti, xi, yi = coords[i]
        torch.testing.assert_close(feature, torch.tensor(
            [(xj-xi)/radius, (yj-yi)/radius, math.hypot(xj-xi, yj-yi)/radius], dtype=feature.dtype))
    self_edges = edges[0] == edges[1]
    assert self_edges.sum() == len(positions)
    assert features[self_edges].count_nonzero() == 0
    assert graph_reach(3, radius) == 3 * int(radius)


@pytest.mark.parametrize('constructor', [AggregationGATv2, BiologyGATv2, DiscriminatorGATv2])
@pytest.mark.parametrize('empty', [False, True])
def test_residual_masking_empty_and_translation(constructor, empty):
    torch.manual_seed(31)
    model = constructor(3, 5, channels=8, depth=2, heads=4, dropout=0.).double().eval()
    other = copy.deepcopy(model)
    other.residual = False
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in other.parameters())
    assert model.radius == other.radius == 2
    x = torch.randn(1, 3, 5, 6, dtype=torch.float64, requires_grad=True)
    mask = torch.zeros(1, 1, 5, 6, dtype=x.dtype)
    if not empty:
        mask[:, :, 1:3, 1:4] = 1
        mask[:, :, 4, 5] = 1  # Isolated node.
        x.data[:, :, 4, 5] = 0  # Occupied, empty spectral feature vector.
    output = model(x, mask)
    assert output.shape == (1, 5, 5, 6)
    assert torch.isfinite(output).all()
    assert (output * (1-mask)).count_nonzero() == 0
    shifted = model(torch.nn.functional.pad(x, (2, 1, 3, 1)),
                    torch.nn.functional.pad(mask, (2, 1, 3, 1)))[:, :, 3:8, 2:8]
    torch.testing.assert_close(output, shifted, atol=1e-12, rtol=1e-12)
    if not empty:
        assert not torch.allclose(output, other(x, mask))
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad * (1-mask)).count_nonzero() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_attention_incoming_normalization():
    model = AggregationGATv2(3, 5, channels=8, heads=4, depth=1, attention_dropout=.5).eval()
    positions, edges, attrs = grid_graph(torch.ones(2, 1, 3, 4), 1.5)
    _, (returned, alpha) = model.attentions[0](torch.randn(len(positions), 8), edges, attrs,
                                             return_attention_weights=True)
    assert torch.equal(edges, returned)
    sums = torch.zeros(len(positions), 4).index_add_(0, edges[1], alpha)
    torch.testing.assert_close(sums, torch.ones_like(sums))
    layer = model.attentions[0]
    assert layer.edge_dim == 3 and layer.concat and not layer.share_weights
    assert not layer.add_self_loops and layer.res is None


@pytest.mark.parametrize('key,value', [
    ('channels', True), ('channels', 0), ('heads', False), ('heads', 3),
    ('depth', 0), ('depth', 1.5), ('residual', 1), ('neighbor_radius', True),
    ('neighbor_radius', .9), ('neighbor_radius', float('inf')), ('neighbor_radius', float('nan')),
    ('dropout', True), ('dropout', -1), ('dropout', 1), ('dropout', float('nan')),
    ('attention_dropout', True), ('attention_dropout', 1), ('attention_dropout', float('inf'))])
def test_invalid_settings(spatial_config, key, value):
    spatial_config['model']['aggregation']['gatv2'][key] = value
    with pytest.raises(ValueError):
        resolve_execution(spatial_config)
    with pytest.raises(ValueError):
        AggregationGATv2(3, 4, **{key: value})


def test_dependency_is_lazy(spatial_config, monkeypatch):
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name.startswith('torch_geometric'):
            raise ImportError('simulated missing PyG')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    AdversarialLatentFusion.from_config(spatial_config, 2, 4)
    spatial_config['model']['aggregation']['type'] = 'gatv2'
    resolve_execution(spatial_config)  # File-based consumers need no PyG.
    architecture_contract(spatial_config)
    with pytest.raises(ImportError, match='pip install torch-geometric'):
        AdversarialLatentFusion.from_config(spatial_config, 2, 4)


@pytest.mark.parametrize('name', ['aggregation', 'biology', 'discriminator'])
def test_v5_contracts(spatial_config, name):
    c = spatial_config
    group = c['model']['aggregation'] if name == 'aggregation' else c['model']['heads'][name]
    group['type'] = 'gatv2'
    resolve_execution(c)
    payload = {'config': c, 'architecture': architecture_contract(c)}
    assert checkpoint_config(payload) == c
    for key, value in dict(channels=16, heads=4, neighbor_radius=2., depth=3,
                           residual=False, dropout=.2, attention_dropout=.1).items():
        bad = copy.deepcopy(payload)
        bad['architecture']['components'][name]['gatv2'][key] = value
        with pytest.raises(ValueError, match='disagrees'):
            checkpoint_config(bad)
        bad = copy.deepcopy(payload)
        g = bad['config']['model']['aggregation'] if name == 'aggregation' else bad['config']['model']['heads'][name]
        del g['gatv2'][key]
        with pytest.raises(ValueError, match='complete'):
            checkpoint_config(bad)
    for version in (None, 2, 3, 4):
        bad = copy.deepcopy(payload)
        if version is None:
            del bad['architecture']
        else:
            bad['architecture']['version'] = version
        with pytest.raises(ValueError, match='version 5'):
            checkpoint_config(bad)
    for key, value in [('residual', 1), ('channels', 8.0), ('attention_dropout', False)]:
        bad = copy.deepcopy(payload)
        bad['architecture']['components'][name]['gatv2'][key] = value
        with pytest.raises(ValueError):
            checkpoint_config(bad)
    for key in ('graph_semantics', 'reach'):
        bad = copy.deepcopy(payload)
        bad['architecture']['components'][name][key] = None
        with pytest.raises(ValueError, match='disagrees'):
            checkpoint_config(bad)
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    validate_checkpoint(payload, c, model)
    component = {'aggregation': model.encoder.aggregation_mlp, 'biology': model.biology_predictor,
                 'discriminator': model.batch_discriminator}[name]
    component.residual = False
    with pytest.raises(ValueError, match='GATv2 settings'):
        validate_checkpoint(payload, c, model)


def test_different_graph_reaches_and_tile_batching(spatial_config):
    from dann.tiles import SpatialTileDataset, tile_collate
    c = spatial_config
    for group, radius, depth in zip([c['model']['aggregation'], *c['model']['heads'].values()],
                                    [2., 1., 1.5], [1, 2, 3]):
        group['type'] = 'gatv2'
        group['gatv2'].update(neighbor_radius=radius, depth=depth)
    resolve_execution(c)
    assert c['execution']['halo'] == 5
    model = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    rows = np.arange(156)
    small = predict_tiles(c, model, rows)
    c['training']['core_size'] = 32
    reference = predict_tiles(c, model, rows)
    for row in rows:
        for key in reference[row]:
            torch.testing.assert_close(small[row][key], reference[row][key], atol=2e-6, rtol=2e-5)
    c['training']['core_size'] = 4
    dataset = SpatialTileDataset(c, rows[:17], input_path=c['data']['context_path'])
    with torch.no_grad():
        batch = tile_collate([dataset[i] for i in range(len(dataset))])
        output = model(batch)
    for i, row in enumerate(batch['row_ids'].tolist()):
        for key in output:
            torch.testing.assert_close(output[key][i], reference[row][key], atol=2e-6, rtol=2e-5)
    dataset.close()


def test_end_to_end_gatv2_workflow(spatial_config, tmp_path):
    from dann.train import train_model
    from dann.analyze import run_analysis
    from dann.spatial import run_inference
    from dann.latent_variance import main as variance_main
    from dann.spatial_heatmaps import run_spatial_heatmaps
    c = spatial_config
    import anndata as ad
    tissue = ad.read_h5ad(c['data']['context_path'])
    tissue.obsm['spatial'] = tissue.obs[['x', 'y']].to_numpy()
    tissue.uns['spatial'] = {
        slide: {'images': {'HES': np.zeros((10, 10, 3))},
                'scalefactors': {'tissue_HES_scalef': 1., 'spot_diameter_fullres': 1.}}
        for slide in ('a', 'b')}
    tissue.write_h5ad(c['data']['context_path'])
    c['model']['aggregation']['type'] = 'gatv2'
    c['model']['heads']['discriminator']['type'] = 'cnn'
    c['model']['latent_dim'] = 256  # Production export contract.
    c['training'].update(epochs=1, checkpoint_every=1)
    c['analysis'].update(figure_dpi=20, umap_neighbors=3, heatmap_top_peaks=5, family_count=2)
    c['latent_variance']['figure_dpi'] = 20
    resolve_execution(c)
    path = train_model(c)
    payload = torch.load(path, weights_only=False)
    assert payload['architecture']['version'] == 5
    assert payload['software_versions']['torch_geometric']
    c['training'].update(epochs=2, resume_checkpoint=str(Path(c['training']['output_dir'])/'latest.pt'))
    train_model(c)
    latest = torch.load(Path(c['training']['output_dir'])/'latest.pt', weights_only=False)
    assert latest['epoch'] == 1 and latest['sampler_state']['next_epoch'] == 2
    for split in payload['split_indices']:
        np.testing.assert_array_equal(payload['split_indices'][split], latest['split_indices'][split])
    # Saved selectors must win for analysis even when runtime selectors differ.
    runtime = copy.deepcopy(c)
    for group in [runtime['model']['aggregation'], *runtime['model']['heads'].values()]:
        group['type'] = 'mlp'
    analysis_dir = run_analysis(runtime, path)
    assert list(analysis_dir.rglob('*.png'))
    input_path = Path(c['data']['context_path'])
    predictions, latents = Path(c['spatial']['output']), Path(c['spatial']['latents'])
    run_inference(input_path, path, predictions, latent_output_path=latents)
    frame = pd.read_parquet(latents)
    assert len(frame) == 156
    assert len([key for key in frame if key.startswith('latent_')]) == 256
    assert frame.row_position.tolist() == list(range(156))
    prediction_frame = pd.read_parquet(predictions)
    assert np.isfinite(prediction_frame.select_dtypes('number')).all().all()
    config_path = tmp_path/'consumer.yaml'
    config_path.write_text(yaml.safe_dump(c))
    assert variance_main(['--config', str(config_path)]) == 0
    assert (Path(c['latent_variance']['output_dir'])/'explained_variance.png').is_file()
    heatmaps = run_spatial_heatmaps(input_path, predictions, tmp_path/'heatmaps',
                                   c['data']['target_columns'], c['loss']['logit_epsilon'], 20)
    assert len(list(heatmaps.glob('*.png'))) == 8
    assert not list(predictions.parent.glob('.dann-stage-*'))


@pytest.mark.parametrize('residual', [False, True])
@pytest.mark.parametrize('dropout', [0., .2])
def test_attention_checkpoint_preserves_outputs_gradients_rng(monkeypatch, residual, dropout):
    import dann.model_components.layers_gatv2 as layers
    torch.manual_seed(53)
    model = AggregationGATv2(3, 4, channels=8, heads=2, depth=2, residual=residual,
                             dropout=dropout, attention_dropout=dropout).double().train()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 3, 5, 5, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    mask = torch.ones(2, 1, 5, 5, dtype=x.dtype)
    mask[:, :, 1, 2] = 0
    torch.manual_seed(57)
    actual = model(x, mask)
    actual.square().sum().backward()
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(layers, 'checkpoint', lambda fn, *args, **kwargs: fn(*args))
    torch.manual_seed(57)
    expected = reference(y, mask)
    expected.square().sum().backward()
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize('residual', [False, True])
def test_graph_tile_budget_preserves_all_outputs_and_gradients(residual):
    torch.manual_seed(67)
    model = AggregationGATv2(3, 4, channels=8, heads=2, depth=2,
                             neighbor_radius=2., dropout=0., residual=residual).double()
    reference = copy.deepcopy(model)
    model.tile_budget = 2
    reference.tile_budget = 20
    x = torch.randn(5, 3, 6, 7, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    mask = (torch.rand(5, 1, 6, 7) > .2).double()
    actual, expected = model(x, mask), reference(y, mask)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-12)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-10)
    model.eval()
    reference.eval()
    torch.testing.assert_close(model(x, mask), reference(y, mask), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('value', [True, 0, -1, 1.5])
def test_invalid_graph_tile_budget(spatial_config, value):
    spatial_config['model']['gatv2_tile_budget'] = value
    with pytest.raises(ValueError, match='positive integers'):
        resolve_execution(spatial_config)

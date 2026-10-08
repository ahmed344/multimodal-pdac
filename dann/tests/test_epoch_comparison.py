"""Controls for full-epoch source and execution comparisons."""
import copy
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import torch
import yaml
from torch import nn

from dann.config import resolve_execution
from dann.data_loader import create_data_bundle, sparse_collate
from dann.model import AdversarialLatentFusion
from dann.tiles import SpatialTileDataset, tile_collate
from dann.train import save_checkpoint
from dann.tests.test_components import spatial_config
from dann.verification.epoch_comparison import (
    flattened_tile_collate, spatial_control,
    equivalence_audit, ConvertedNetwork, main,
)


def pointwise_config(config):
    for group in [config['model']['aggregation'], *config['model']['heads'].values()]:
        group['type'] = 'mlp'
    return resolve_execution(config)


@pytest.mark.parametrize('name', ['aggregation', 'biology'])
def test_historical_spatial_control_stays_plain(spatial_config, name):
    config = pointwise_config(spatial_config)
    reference = AdversarialLatentFusion.from_config(config, 2, 4)
    model = spatial_control(reference, config, production=name)
    module = model.encoder.aggregation_mlp if name == 'aggregation' else model.biology_predictor
    assert not module.residual
    group = config['model']['aggregation'] if name == 'aggregation' else config['model']['heads'][name]
    assert group['cnn']['residual'] is True


def test_flat_and_grid_values_gradients_adamw(spatial_config):
    torch.manual_seed(109)
    c = pointwise_config(spatial_config)
    bundle = create_data_bundle(c)
    c['training']['core_size'] = 4
    tiles = SpatialTileDataset(c, bundle.split_indices['train'], bundle.metadata)
    selected = [tiles[i] for i in range(2)]
    flat, grid = flattened_tile_collate(selected), tile_collate(selected)
    for key in ('row_ids','targets','target_valid_mask','batches','raw_targets'):
        torch.testing.assert_close(flat[key],grid[key], equal_nan=True)
    assert set(flat['row_ids'].tolist()) <= set(bundle.split_indices['train'])
    a = AdversarialLatentFusion.from_config(c, 2, 4)
    b = spatial_control(a, c)
    result = equivalence_audit(a, b, flat, grid, c, bundle.batch_class_weights)
    assert result['passed']
    # Finite garbage at absent grid sites cannot leak through masked biases.
    network = b.encoder.aggregation_mlp.double().eval()
    mask = grid['occupancy'].double()
    x = torch.randn(mask.shape[0], 6, *mask.shape[-2:], dtype=torch.double)
    torch.testing.assert_close(network(x, mask), network(x + (1-mask)*123, mask))
    assert torch.count_nonzero(network(x, mask)*(1-mask)) == 0
    tiles.close()


@pytest.mark.parametrize('mixing', ['aggregation','biology'])
def test_center_initialized_mixing_preserves_mlp_at_boundaries(spatial_config, mixing):
    torch.manual_seed(73)
    c = pointwise_config(spatial_config)
    bundle = create_data_bundle(c)
    a = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    b = spatial_control(a, c, mixing=mixing).eval()
    geometry = copy.deepcopy(c)
    group = geometry['model']['aggregation'] if mixing=='aggregation' else geometry['model']['heads']['biology']
    group['type']='cnn'; group['cnn']['depth']=2
    geometry['training']['core_size']=4
    tiles = SpatialTileDataset(geometry,bundle.split_indices['train'],bundle.metadata)
    selected = [tiles[0],tiles[1]]
    with torch.no_grad():
        left=a(flattened_tile_collate(selected));right=b(tile_collate(selected))
    for key in left:
        torch.testing.assert_close(left[key],right[key],atol=2e-6,rtol=2e-5)
    component=b.encoder.aggregation_mlp if mixing=='aggregation' else b.biology_predictor
    assert component.radius==2
    for layer in component.layers:
        if isinstance(layer,nn.Conv2d):
            neighbors=layer.weight.detach().clone();neighbors[:,:,1,1]=0
            assert torch.count_nonzero(neighbors)==0
    tiles.close()


def test_three_epochs_preserve_schedule_and_production_checkpoint(spatial_config,tmp_path,monkeypatch):
    c=pointwise_config(spatial_config)
    c['training']['epochs']=300
    bundle=create_data_bundle(c)
    model=AdversarialLatentFusion.from_config(c,2,4)
    opt=torch.optim.AdamW(model.parameters())
    source=tmp_path/'source.pt'
    save_checkpoint(source,model,opt,0,c,bundle,1.)
    config_path=tmp_path/'config.yaml'
    config_path.write_text(yaml.safe_dump(c))
    from dann import train
    seen=[]
    original=train.grl_strength
    def record(progress,*args):
        seen.append(progress)
        return original(progress,*args)
    monkeypatch.setattr(train,'grl_strength',record)
    output=tmp_path/'comparison'
    main(['--config',str(config_path),'--reference-checkpoint',str(source),'--output',str(output),
          '--input-source','labeled','--execution-path','pixel','--epoch-limit','3'])
    history=json.loads((output/'history.json').read_text())
    assert [h['epoch'] for h in history]==[0,1,2,3]
    steps=len(bundle.loaders['train'])
    assert history[-1]['updates']==3*steps
    assert max(seen)==pytest.approx((3*steps-1)/(300*steps-1))
    saved=torch.load(output/'epoch_3.pt',weights_only=False)
    restored=AdversarialLatentFusion.from_config(c,2,4)
    restored.load_state_dict(saved['model_state'])
    assert saved['config']['training']['epochs']==300
    for epoch in range(4):
        prediction=torch.load(output/f'validation_epoch_{epoch}.pt',weights_only=False)
        np.testing.assert_array_equal(prediction['row_ids'],bundle.split_indices['validation'])
        assert not np.isin(prediction['row_ids'],bundle.split_indices['test']).any()
    assert (output/'source_config.yaml').read_bytes()==config_path.read_bytes()


def test_legacy_seeded_initialization_order(spatial_config):
    """The old encoder's final embedding draw follows aggregation construction."""
    from dann.model_components.layers import build_mlp
    from dann.model import IntensityWeightedPeakEncoder
    torch.manual_seed(91)
    embedding=nn.Embedding(5,4)
    peak=build_mlp(4,[7],6,'gelu',.1,True)
    aggregation=build_mlp(6,[7],6,'gelu',.1,True)
    nn.init.normal_(embedding.weight,mean=0.,std=.02)
    expected_rng=torch.get_rng_state().clone()
    torch.manual_seed(91)
    encoder=IntensityWeightedPeakEncoder(5,4,[7],6,[7],6,'gelu',.1,True)
    torch.testing.assert_close(encoder.embedding.weight,embedding.weight,rtol=0,atol=0)
    for expected,actual in zip(list(peak.parameters())+list(aggregation.parameters()),
                               list(encoder.peak_mlp.parameters())+list(encoder.aggregation_mlp.parameters())):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),expected_rng)


def test_constant_hurdle_scores_have_exact_chance_auc():
    from dann.verification.epoch_comparison import extra_metrics
    y=torch.zeros(10003,1)
    y[::3]=.1
    out={'pi_logits':torch.tensor([[-.36632925]]).expand(len(y),1),
         'mu':torch.zeros_like(y)}
    result=extra_metrics(out,y,torch.ones_like(y,dtype=torch.bool),1e-5)
    assert result['cd8_zero_auc']==.5
    assert result['positive_cd8_correlation'] is None
    assert result['positive_cd8_r2'] is None


def test_flat_reader_matches_grid_core_batches_without_context_reads(spatial_config):
    from dann.verification.epoch_comparison import FlatTileDataset
    c=pointwise_config(spatial_config)
    c['training']['core_size']=4
    bundle=create_data_bundle(c)
    grid=SpatialTileDataset(c,bundle.split_indices['train'],bundle.metadata)
    flat=FlatTileDataset(c,bundle.split_indices['train'],bundle.metadata,geometry=grid.geometry)
    indices=list(range(min(3,len(grid))))
    expected=flattened_tile_collate([grid[i] for i in indices])
    actual=flattened_tile_collate([flat[i] for i in indices])
    for key in expected:
        torch.testing.assert_close(actual[key],expected[key],equal_nan=True)
    assert sum(len(flat[i]['samples']) for i in indices)==len(actual['row_ids'])
    grid.close();flat.close()

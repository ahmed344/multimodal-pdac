"""Supervised source isolation and explicit inference/provenance boundaries."""
import copy
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pytest
import torch

from dann.config import inference_input, resolve_execution
from dann.data_loader import create_data_bundle, sparse_collate
from dann.model import AdversarialLatentFusion
from dann.tiles import SpatialTileDataset, tile_collate
from dann.train import save_checkpoint, load_training_checkpoint
from dann.tests.test_components import spatial_config
from dann.checkpoints import validate_training_data_contract


@pytest.mark.parametrize('mode', ['pixel', 'spatial'])
@pytest.mark.parametrize('inference', ['absent', 'missing', 'unrelated'])
def test_supervised_loaders_never_open_inference(spatial_config, monkeypatch, mode, inference):
    c = spatial_config
    if mode == 'pixel':
        for group in [c['model']['aggregation'], *c['model']['heads'].values()]:
            group['type'] = 'mlp'
    resolve_execution(c)
    baseline = create_data_bundle(c)
    expected = {s: [d[i] for i in range(len(d))] for s, d in baseline.datasets.items()}
    for d in baseline.datasets.values():
        d.close()
    path = Path(c['data']['context_path'])
    if inference == 'absent':
        del c['data']['context_path']
    elif inference == 'missing':
        c['data']['context_path'] = str(path.with_name('nonexistent.h5ad'))
    else:
        unrelated = ad.read_h5ad(path)[:, ::-1].copy()
        unrelated.X *= 13
        unrelated.obs['x'] = .123
        for target in c['data']['target_columns']:
            unrelated.obs[target] = np.nan
        unrelated.write_h5ad(path)
    original = h5py.File.__init__
    def guarded(self, name, *args, **kwargs):
        if isinstance(name, (str, Path)):
            assert Path(name).resolve() == Path(c['data']['path']).resolve()
        return original(self, name, *args, **kwargs)
    monkeypatch.setattr(h5py.File, '__init__', guarded)
    resolve_execution(c)
    actual = create_data_bundle(c, baseline.split_indices)
    model = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    collate = tile_collate if mode == 'spatial' else sparse_collate
    for split, dataset in actual.datasets.items():
        left = collate(expected[split])
        right = collate([dataset[i] for i in range(len(dataset))])
        for key in left:
            torch.testing.assert_close(left[key], right[key], equal_nan=True, rtol=0, atol=0)
        with torch.no_grad():
            a, b = model(left), model(right)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        dataset.close()


def test_cached_geometry_and_explicit_source(spatial_config):
    c = spatial_config
    bundle = create_data_bundle(c)
    inference = SpatialTileDataset(c, np.array([0]), input_path=c['data']['context_path'])
    with pytest.raises(ValueError, match='source identity'):
        SpatialTileDataset(c, np.array([0]), bundle.metadata, geometry=inference.geometry)
    with pytest.raises(ValueError, match='data.path'):
        SpatialTileDataset(c, np.array([0]), bundle.metadata, input_path=c['data']['context_path'])
    with pytest.raises(ValueError, match='explicit input'):
        SpatialTileDataset(c, np.array([0]))
    for dataset in bundle.datasets.values():
        dataset.close()
    inference.close()


def test_held_out_context_and_labels(spatial_config):
    c = spatial_config
    bundle = create_data_bundle(c)
    rows = bundle.split_indices['train']
    dataset = bundle.datasets['train']
    expected = tile_collate([dataset[i] for i in range(len(dataset))])
    # Every occupied site is a native labeled-file coordinate, including held-out rows.
    seen = set()
    for (slide, tx, ty), _ in dataset.tiles:
        x0, y0 = tx*dataset.core_size-dataset.halo, ty*dataset.core_size-dataset.halo
        seen.update(row for (batch, x, y), row in dataset.lookup.items()
                    if batch == slide and x0 <= x < x0+dataset.size and y0 <= y < y0+dataset.size)
    held_out = np.concatenate([bundle.split_indices[s] for s in ('validation', 'test')])
    assert set(held_out) & seen
    assert len(dataset.keys) == bundle.metadata.num_observations
    for d in bundle.datasets.values():
        d.close()
    labeled = ad.read_h5ad(c['data']['path'])
    for target in c['data']['target_columns']:
        labeled.obs.iloc[held_out, labeled.obs.columns.get_loc(target)] = .9
    labeled.write_h5ad(c['data']['path'])
    updated = create_data_bundle(c, bundle.split_indices)
    actual = tile_collate([updated.datasets['train'][i] for i in range(len(dataset))])
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], equal_nan=True, rtol=0, atol=0)
    from dann.losses import ZILNLoss
    model = AdversarialLatentFusion.from_config(c, 2, 4).eval()
    loss = ZILNLoss.from_config(c)
    scores = []
    for batch in (expected, actual):
        out = model(batch)
        scores.append(loss(out['pi_logits'], out['mu'], out['sigma'], batch['targets'], batch['target_valid_mask']).total)
    torch.testing.assert_close(*scores, rtol=0, atol=0)
    for d in updated.datasets.values():
        d.close()


def test_provenance_blocks_supervision_preserves_inference(spatial_config, tmp_path):
    from dann.spatial import load_checkpoint_model
    from dann.analyze import load_model
    c = spatial_config
    bundle = create_data_bundle(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path/'checkpoint.pt'
    save_checkpoint(path, model, opt, 0, c, bundle, 1.)
    checkpoint = torch.load(path, weights_only=False)
    validate_training_data_contract(checkpoint, c)
    bad = copy.deepcopy(checkpoint)
    bad['training_data_contract']['context_source'] = 'data.context_path'
    with pytest.raises(ValueError, match='Incompatible training_data_contract'):
        validate_training_data_contract(bad, c)
    del checkpoint['training_data_contract']
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='Missing training_data_contract'):
        load_training_checkpoint(path, model, opt, torch.device('cpu'), c)
    with pytest.raises(ValueError, match='Missing training_data_contract'):
        load_model(path, c, bundle, torch.device('cpu'))
    loaded, *_ = load_checkpoint_model(path, 'cpu')
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key])
    for d in bundle.datasets.values():
        d.close()


def test_inference_resolution(spatial_config):
    c = spatial_config
    assert inference_input(c) == Path(c['data']['context_path'])
    explicit = Path(c['data']['path'])
    assert inference_input(c, explicit) == explicit
    del c['data']['context_path']
    resolve_execution(c)
    with pytest.raises(ValueError, match='Inference requires'):
        inference_input(c)
    c['data']['context_path'] = '/missing/inference.h5ad'
    with pytest.raises(FileNotFoundError, match='Inference input'):
        inference_input(c)


def test_native_grid_and_direct_labeled_spectra(spatial_config):
    from dann.data_loader import SparseAnnDataDataset
    from dann.verification.epoch_comparison import flattened_tile_collate
    c = spatial_config
    bundle = create_data_bundle(c)
    data = c['data']
    for split in ('train', 'validation'):
        dataset = bundle.datasets[split]
        direct = SparseAnnDataDataset(Path(data['path']), np.arange(bundle.metadata.num_observations),
            bundle.metadata, data['matrix_key'], data['intensity_transform'],
            data.get('intensity_clip_max'), data['nonzero_threshold'])
        for i in range(len(dataset)):
            tile = dataset[i]
            (slide, tx, ty), _ = dataset.tiles[i]
            x0, y0 = tx*dataset.core_size-dataset.halo, ty*dataset.core_size-dataset.halo
            occupied = {}
            for (batch, x, y), row in dataset.lookup.items():
                if batch == slide and x0 <= x < x0+dataset.size and y0 <= y < y0+dataset.size:
                    occupied[(y-y0)*dataset.size+x-x0] = row
            assert set(tile['spatial_positions']) == set(occupied)
            for position, sample in zip(tile['spatial_positions'], tile['samples']):
                expected = direct[occupied[position]]
                for key in ('peak_indices', 'intensities'):
                    np.testing.assert_array_equal(sample[key], expected[key])
            flat = flattened_tile_collate([tile])
            expected = sparse_collate([direct[row] for row in tile['row_ids']])
            for key in flat:
                torch.testing.assert_close(flat[key], expected[key], equal_nan=True, rtol=0, atol=0)
        direct.close()
    for dataset in bundle.datasets.values():
        dataset.close()


def test_load_and_reload_config_without_inference_path(spatial_config, tmp_path):
    import yaml
    from dann.config import load_config, apply_smoke_overrides
    c = spatial_config
    del c['data']['context_path']
    path = tmp_path/'config.yaml'
    for _ in range(2):
        path.write_text(yaml.safe_dump(c))
        c = load_config(path)
        assert c['latent_variance']['input'] is None
    smoke = apply_smoke_overrides(c)
    bundle = create_data_bundle(smoke)
    next(iter(bundle.loaders['train']))
    for dataset in bundle.datasets.values():
        dataset.close()


def test_tissue_training_rejected_before_file_access(tmp_path):
    from dann.verification.epoch_comparison import main
    with pytest.raises(SystemExit):
        main(['--reference-checkpoint', str(tmp_path/'missing.pt'), '--output', str(tmp_path/'unused'),
              '--input-source', 'tissue', '--execution-path', 'flat-tiles'])
    assert not (tmp_path/'unused').exists()


def test_checkpoint_does_not_restore_inference_path(spatial_config):
    from dann.targets import checkpoint_analysis_config
    c = spatial_config
    saved = copy.deepcopy(c)
    saved['data']['context_path'] = '/historical/inference.h5ad'
    checkpoint = {'config': saved, 'target_columns': c['data']['target_columns']}
    result = checkpoint_analysis_config(c, checkpoint)
    assert result['data']['context_path'] == c['data']['context_path']


def test_legacy_export_records_unknown_provenance_without_training_file(spatial_config, tmp_path):
    import json
    from dann.spatial import run_inference
    c = spatial_config
    bundle = create_data_bundle(c)
    model = AdversarialLatentFusion.from_config(c, 2, 4)
    path = tmp_path/'legacy.pt'
    save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), 0, c, bundle, 1.)
    payload = torch.load(path, weights_only=False)
    del payload['training_data_contract']
    torch.save(payload, path)
    for dataset in bundle.datasets.values():
        dataset.close()
    # Frozen feature names suffice; inference never coordinate-matches training.
    Path(c['data']['path']).rename(tmp_path/'unavailable-training.h5ad')
    output = tmp_path/'prediction.parquet'
    run_inference(Path(c['data']['context_path']), path, output, max_rows=3,
                  num_workers_override=0, density_mc_samples=3)
    provenance = json.loads(output.with_suffix('.provenance.json').read_text())
    assert provenance['training_data_contract']['status'] == 'unknown_legacy_provenance'


def test_inference_cli_keeps_runtime_input_with_checkpoint_override(spatial_config, tmp_path, monkeypatch):
    import yaml
    from dann import spatial
    c = spatial_config
    config_path = tmp_path/'runtime.yaml'
    config_path.write_text(yaml.safe_dump(c))
    called = {}
    def run(**kwargs):
        called.update(kwargs)
        return kwargs['output_path']
    monkeypatch.setattr(spatial, 'run_inference', run)
    spatial.main(['--config', str(config_path), '--checkpoint', str(tmp_path/'old.pt')])
    assert called['input_path'] == Path(c['data']['context_path'])
    spatial.main(['--config', str(config_path), '--checkpoint', str(tmp_path/'old.pt'),
                  '--input', c['data']['path']])
    assert called['input_path'] == Path(c['data']['path'])

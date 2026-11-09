"""Exercise schema 3 through fresh training and every downstream workflow stage."""
from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from msi_preprocess.tests.test_cohort import cohort_fixture
from msi_preprocess.fit import fit
from msi_preprocess.apply import apply
from dann.config import load_config
from dann.train import train_model
from dann.workflow import run_post_training_workflow


def test_retained_cohort_complete_workflow(tmp_path):
    preprocessing, _ = cohort_fixture(tmp_path)
    fitted = fit(preprocessing)
    apply(fitted, update_dann=True)
    path = Path(preprocessing['dann_config'])
    config = yaml.safe_load(path.read_text())
    config['model']['heads']['he']['enabled'] = False  # This workflow tests biology-only exports.
    config['results'].update(root=str(tmp_path / 'runs'), run_name='retained_synthetic')
    config['model']['latent_dim'] = 4
    config['model']['spectral_encoder']['deep_sets'].update(embedding_dim=4, peak_hidden_dims=[8], peak_output_dim=8)
    config['model']['aggregation']['mlp']['hidden_dims'] = [8]
    for head in config['model']['heads'].values():
        head['mlp']['hidden_dims'] = [8]
    config['training'].update(epochs=1, device='cpu')
    config['training']['pixel'].update(batch_size=32, validation_batch_size=32, num_workers=0)
    config['monitoring']['enabled'] = False
    config['data'].update(max_train_samples=64, max_validation_samples=24)
    config['analysis'].update(max_samples=24, figure_dpi=20, density_mc_samples=4, num_workers=0,
                              batch_size=64, umap_neighbors=3, heatmap_top_peaks=4, family_count=2)
    config['analysis']['peaks'].update(enabled=True, spatial=True, min_observations=2, umap_neighbors=3)
    config['analysis']['execution'].update(render_workers=0, sampling_workers=0, feature_cache=False)
    config['analysis']['execution']['performance'].update(ram_cache=False, preparation_workers=0)
    config['analysis']['spatial_heatmaps'].update(workers=1, enabled=True)
    config['post_training'].update(run_inference=True, run_spatial_heatmaps=True)
    config['analysis']['spatial_execution'].update(density_workers=1, feature_cache=False)
    path.write_text(yaml.safe_dump(config))
    config = load_config(path)
    checkpoint = train_model(config)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['batch_names'] == ('d', 'b') or saved['batch_names'] == ['d', 'b']
    history = pd.read_csv(checkpoint.parent / 'history.csv')
    chance = [name for name in history if name.endswith('batch_chance_loss')]
    assert not chance
    assert history['validation_alignment_nhsic'].notna().all()
    assert history['validation_alignment_nhsic_sample_slides'].tolist() == [2]
    assert saved['architecture']['enabled_heads'] == ['biology']
    from dann.checkpoints import validate_checkpoint
    stale = copy.deepcopy(config)
    stale['data']['preprocessing_fingerprint'] = 'old_cohort'
    with pytest.raises(ValueError, match='fresh training'):
        validate_checkpoint(saved, stale)
    outputs = run_post_training_workflow(config, checkpoint)
    from dann.preprocessing import validate_artifact_preprocessing
    for key in ('inference', 'latents'):
        validate_artifact_preprocessing(Path(config['data']['context_path']), pq.read_schema(outputs[key]).metadata)
    from dann.artifact_layout import peak_file, analysis_file
    peak_root = outputs['analysis'] / 'peaks'
    spatial = pd.read_parquet(peak_file(peak_root, 'family_spatial_activity.parquet'))
    assert set(spatial['batch']) == {'b', 'd'}
    latent = pd.read_csv(analysis_file(outputs['analysis'], 'latent_embeddings.csv'))
    assert set(latent['batch']) == {'b', 'd'}
    summary = pd.read_csv((outputs['analysis'] / 'latent_variance') / 'latent_variance_summary.csv')
    assert summary.n_slides.tolist() == [2]
    assert (outputs['analysis'] / 'preprocessing.json').is_file()
    # Reject incompatible measured-peak input even when dimensions/names match.
    from dann.config import analysis_execution_settings
    from dann.peak_analysis import run_peak_analysis
    import h5py
    with h5py.File(config['data']['context_path'], 'r+') as handle:
        handle['uns/msi_preprocess/fingerprint'][()] = 'stale'
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        run_peak_analysis(config, checkpoint, analysis_execution_settings({'render_workers': 0}),
                          output_override=tmp_path / 'rejected')

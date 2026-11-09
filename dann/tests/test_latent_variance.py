"""Lesion-only covariance, complete PCA spectra and explicit Scanpy graph inputs."""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import pytest
from anndata.io import write_elem

from dann.analysis_latents import (covariance_pca, variance_from_pca, select_pc_count,
    select_sample_rows, fit_pca_umap, cached_pca, plot_explained_variance)
from dann.analysis_execution import StageCache
from dann.spatial_execution import InferenceProgress


def test_covariance_full_spectrum_and_pixel_weighted_slides():
    rng = np.random.default_rng(19)
    values = rng.normal(size=(37, 9)) + 1e8
    slides = np.array(['a']*5 + ['b']*32)
    pca = covariance_pca(values)
    summary, spectrum = variance_from_pca(values, slides, pca)
    expected = np.linalg.eigvalsh(np.cov(values, rowvar=False))[::-1]
    np.testing.assert_allclose(spectrum.eigenvalue, expected, rtol=1e-12, atol=1e-12)
    centered = values - values[0]
    grand = centered.mean(axis=0)
    between = sum(np.sum(slides == s) * np.sum((centered[slides == s].mean(axis=0)-grand)**2) for s in np.unique(slides))
    assert summary['variance_explained_by_slide_means'] == pytest.approx(between / np.sum((centered-grand)**2))
    assert summary['participation_ratio'] == pytest.approx(expected.sum()**2 / np.sum(expected**2))
    assert len(spectrum) == values.shape[1]


def test_constant_and_rank_deficient_latents(tmp_path):
    values = np.full((10, 13), 1e15)
    pca = covariance_pca(values)
    summary, spectrum = variance_from_pca(values, ['a']*10, pca)
    assert summary['total_variance'] == 0
    assert summary['pcs_99'] is None
    assert summary['participation_ratio'] is None
    assert len(spectrum) == 13
    assert spectrum.eigenvalue.eq(0).all()
    plot_explained_variance(spectrum, summary, tmp_path/'constant.png', 20)
    cache = StageCache(tmp_path/'cache', 'reuse', InferenceProgress())
    assert fit_pca_umap(values, pca, cache)[1]['skipped'] == 'zero total variance'
    assert fit_pca_umap(values[:3], None, cache)[1]['skipped'] == 'fewer than four observations'
    rank_one = np.arange(5.)[:, None] * np.ones((1, 30))
    pca = covariance_pca(rank_one)
    assert len(pca['eigenvalues']) == 30
    assert select_pc_count(pca['eigenvalues'])[0] == 1


def test_pc_selection_reaches_99_percent_without_cap():
    count, fraction = select_pc_count(np.ones(128))
    assert count == 127 and fraction >= .99
    # Do not round a value just below the threshold into an acceptable component count.
    assert select_pc_count(np.array([.99 - 1e-14, .01 + 1e-14]))[0] == 2
    assert select_pc_count(np.array([99., 1.])) == (1, .99)


def test_saved_row_selection_preserves_order_and_alignment(tmp_path):
    path = tmp_path/'source.h5ad'
    with h5py.File(path, 'w') as h:
        obs = h.create_group('obs')
        write_elem(obs, 'batch', np.array(['a','b','a','b'], dtype=object))
        write_elem(obs, 'Lesion', np.array([True, False, True, True]))
    c = dict(data=dict(path=str(path), batch_column='batch'), analysis=dict(split='validation', features={'variance':{'enabled':True}}))
    values = dict(row_ids=np.array([3,1,2]), batches=np.array([1,1,0]), latent=np.arange(6.).reshape(3,2))
    subset, slides, info = select_sample_rows(values, c, ['a','b'])
    np.testing.assert_array_equal(subset['row_ids'], [3,2])
    np.testing.assert_array_equal(subset['latent'], values['latent'][[0,2]])
    assert info['scope'] == 'lesion' and info['split'] == 'validation'
    assert slides.tolist() == ['b','a']
    values['batches'][0] = 0
    with pytest.raises(ValueError, match='batch identities'):
        select_sample_rows(values, c, ['a','b'])
    values['row_ids'][0] = 99
    with pytest.raises(ValueError, match='outside'):
        select_sample_rows(values, c, ['a','b'])


def test_scanpy_explicit_scores_and_separate_caches(tmp_path, monkeypatch):
    import scanpy as sc
    from scipy.sparse import eye
    values = np.random.default_rng(3).normal(size=(140, 70))
    progress = InferenceProgress()
    cache = StageCache(tmp_path, 'reuse', progress)
    pca = cached_pca(values, cache)
    calls = []
    def neighbors(adata, **kwargs):
        calls.append('neighbors')
        assert kwargs == dict(n_neighbors=15, use_rep='X_pca', metric='euclidean', random_state=0, knn=True, method='umap')
        count, _ = select_pc_count(pca['eigenvalues'])
        assert count > 50 and adata.obsm['X_pca'].shape == (140, count)
        adata.obsp['distances'] = eye(140, format='csr')
        adata.obsp['connectivities'] = eye(140, format='csr')
        adata.uns['neighbors'] = dict(connectivities_key='connectivities', distances_key='distances')
    def umap(adata, **kwargs):
        calls.append('umap')
        assert kwargs == dict(min_dist=.5, spread=1., n_components=2, init_pos='spectral', random_state=0)
        adata.obsm['X_umap'] = np.ones((140, 2))
    monkeypatch.setattr(sc.pp, 'neighbors', neighbors)
    monkeypatch.setattr(sc.tl, 'umap', umap)
    coordinates, info = fit_pca_umap(values, pca, cache)
    fit_pca_umap(values, cached_pca(values, cache), cache)
    assert calls == ['neighbors','umap']
    assert set(progress.details['cache_hits']) >= {'sample-pca','sample-pca-neighbors','sample-pca-umap'}
    assert info['achieved_variance'] >= .99
    assert cached_pca(values + np.eye(140,70), cache)['eigenvalues'][0] != pca['eigenvalues'][0]


def test_legacy_split_keeps_repeated_training_samples(tmp_path):
    path=tmp_path/'labels.h5ad'
    with h5py.File(path,'w') as h:
        obs=h.create_group('obs')
        write_elem(obs,'batch',np.array(['a','a'],dtype=object))
        write_elem(obs,'Lesion',np.ones(2,dtype=bool))
    c=dict(data=dict(path=str(path),batch_column='batch'),analysis=dict(split='train'))
    sample=dict(row_ids=np.array([1,0,1]),batches=np.zeros(3,dtype=int),latent=np.arange(6.).reshape(3,2))
    selected,_,_=select_sample_rows(sample,c,['a'])
    np.testing.assert_array_equal(selected['latent'],sample['latent'])
    c['analysis']['sample_scope']='lesion'
    with pytest.raises(ValueError,match='unique'):
        select_sample_rows(sample,c,['a'])

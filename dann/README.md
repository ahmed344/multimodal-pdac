# Adversarial Latent Fusion

This package implements the PDF's adversarial predictive bottleneck for sparse
MALDI-MSI. MSI rows stay in backed CSR form: the loader reads only nonzero peak
indices and intensities, and the encoder applies
`MLP_agg(mean(MLP_peak(intensity * embedding[peak])))`.

The four biology outputs use a structural-zero hurdle and a logit-normal
positive component. A gradient reversal layer trains the shared latent space to
confuse the 43-class batch discriminator while preserving IHC prediction.

## Configuration

All adjustable paths, architecture sizes, optimization settings, numerical
stabilizers, smoke limits, clustering controls, and plot settings are described
with inline comments in `dann/config.yaml`.

The PDF does not specify preprocessing, dimensions, optimization, splitting,
GRL scheduling, or clustering. Their defaults are therefore practical,
documented choices. `intensity_transform: log1p` transforms every stored
nonzero MSI intensity immediately after sparse row loading and before the
embedding model. Implicit zeros remain zero and no dense matrix is created.

Rare exact-one target values are clamped only for the otherwise undefined logit
operation. The normal-density constant remains disabled by default because the
PDF omits it.

## Commands

Run from the repository root:

```bash
python -m dann.train --config dann/config.yaml
python -m dann.analyze --config dann/config.yaml
```

Fast end-to-end verification on capped rows from the real AnnData:

```bash
python -m dann.train --config dann/config.yaml --smoke-test
python -m dann.analyze --config dann/config.yaml --smoke-test
```

Run focused tests:

```bash
pytest -q dann/tests
```

The default results location is `data/PDAC/Results/dann/`. Analysis writes:

- `latent_umap_batch.png`
- `latent_umap_densities.png`
- `peptide_similarity_heatmap.png`
- `loss_curves.png`
- `ziln_cd8_scatter.png`
- `peptide_families.csv`
- `embedding_cosine_similarity.npy`
- `latent_predictions.npz`

The complete 5,808 × 5,808 cosine matrix is intentionally retained. The
activity scan is exact when `activity_max_nonzeros` is null; smoke mode caps it.

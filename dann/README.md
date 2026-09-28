# Adversarial Latent Fusion

This package implements the PDF's adversarial predictive bottleneck for sparse
MALDI-MSI. MSI rows stay in backed CSR form: the loader reads only nonzero peak
indices and intensities, and the encoder applies
`MLP_agg(mean(MLP_peak(intensity * embedding[peak])))`.

The four biology targets each use a structural-zero hurdle with three parameters:
zero probability `pi`, positive logit mean `mu`, and positive logit standard
deviation `sigma`. A gradient reversal layer trains the shared latent space to
confuse the batch discriminator while preserving IHC prediction.

## Configuration

`analysis.max_samples: null` analyzes all rows in the test and validation splits.
A positive integer selects a stratified cap.

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
python -m dann.spatial
python -m dann.spatial_heatmaps
```

Override both spatial output files when needed:

```bash
python -m dann.spatial \
  --output /path/to/spatial_inference.parquet \
  --latent-output /path/to/spatial_latent.parquet
```

Fast end-to-end verification on capped rows from the real AnnData:

```bash
python -m dann.train --config dann/config.yaml --smoke-test
python -m dann.analyze --config dann/config.yaml --smoke-test
```

Tissue-wide spatial inference writes ordered ZILN predictions and the complete
latent representation for every AnnData row. Heatmaps join the predictions back
onto the tissue AnnData and plot per-batch logit / mean / presence / sigma
panels plus HES.

Run focused tests:

```bash
pytest -q dann/tests
```

The default results location is `data/PDAC/Results/dann/`, with training artifacts
under `model/` and analysis under `analysis/` (UMAP exports in `analysis/umap/`):

`model/`:

- `best.pt`, `latest.pt`
- `history.csv`, `data_schema.json`, `resolved_config.yaml`
- `loss_curves.png` (written by analysis)

`analysis/umap/`:

- `latent_umap_batch.png`
- `latent_umap_densities.png`
- `latent_umap_densities_logit.png`
- `latent_umap_ziln_Density_*.png` (one 2x3 Logit/1-pi/mu/sigma/batch figure per target)
- `latent_umap_ziln_diagnostics_Density_*.png` (one 2x3 Logit/logit_mean/presence/residual/interval-width/batch figure per target, masked to observed positives)
- `latent_umap.csv`

`analysis/` (non-UMAP):

- `peptide_similarity_heatmap.png`
- `ziln_density_scatter.png`
- `latent_embeddings.csv`
- `ziln_density_scatter.csv`
- `peptide_families.csv`
- `embedding_cosine_similarity.npy`

`analysis/calibration/` (test metrics; validation reports in `validation/`):

- `calibration_metrics.csv`, `calibration_metrics_by_batch.csv`
- `calibration_pit.png` (PIT histogram and uniform QQ per target)
- `calibration_hurdle_reliability.png`, `calibration_interval_coverage.png`

`analysis/spatial/`:

- `spatial_inference.parquet` (ordered mu/presence/sigma plus the derived
  logit_mean/logit_sd/logit_median/logit_q05/logit_q95 summaries)
- `spatial_latent.parquet` (ordered full latent vectors)
- `spatial_heatmap_Density_*.png` (one 5-column per-batch raw-parameter figure per target)
- `spatial_heatmap_diagnostics_Density_*.png` (one 5-column per-batch normal diagnostic figure per target)

## Predictions versus parameters

The biology head emits a distribution: `pi` is the probability of a structural
zero, and `(mu, sigma)` define a normal on the logit scale. The positive-branch
logit mean and median both equal `mu`; its standard deviation equals `sigma`.

`dann/ziln.py` holds the conversions (mean, standard deviation, quantiles, PIT,
exceedance probability). All of them are conditional on the positive branch and on
the logit scale, because `logit(0)` is undefined — `prob_of_presence` carries the
hurdle separately. The density-scale expectation `E[y] = (1 - pi) * E[expit(Z)]`
has no closed form and is deliberately not implemented; it needs Monte Carlo or
Gauss-Hermite quadrature per pixel.

`dann/calibration.py` scores whether those distributions are trustworthy: PIT
uniformity and interval coverage for the positive branch, Brier/AUC/reliability for
the hurdle, each globally and per batch. It runs inside `python -m dann.analyze`
and also standalone from a saved table:

```bash
python -m dann.calibration --config dann/config.yaml
```

Sample-level outputs are CSV tables that include batch labels and targets.
`latent_umap.csv` holds UMAP coordinates plus ZILN parameters; `latent_embeddings.csv`
holds the full latent vectors. The complete 5,808 × 5,808 cosine matrix remains
`.npy` because a dense square matrix is not practical as CSV. The activity scan is
exact when `activity_max_nonzeros` is null; smoke mode caps it.

Attach the tissue-wide latent representation to the exact AnnData used for
inference as a standard `obsm` matrix:

```python
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

adata_path = Path("/path/to/adata_assembled_tissue.h5ad")
latent_path = Path("/path/to/spatial_latent.parquet")

adata = ad.read_h5ad(adata_path)
latent_frame = pd.read_parquet(latent_path)

expected_positions = np.arange(adata.n_obs, dtype=np.int64)
if not np.array_equal(latent_frame["row_position"].to_numpy(), expected_positions):
    raise ValueError("Latent rows are not in contiguous AnnData order.")
if not np.array_equal(
    latent_frame["obs_name"].to_numpy(dtype=str),
    adata.obs_names.to_numpy(dtype=str),
):
    raise ValueError("Latent observation names do not match the AnnData order.")

latent_columns = sorted(
    (name for name in latent_frame.columns if name.startswith("latent_")),
    key=lambda name: int(name.removeprefix("latent_")),
)
adata.obsm["X_dann_latent"] = latent_frame[latent_columns].to_numpy(
    dtype=np.float32,
    copy=True,
)
adata.uns["dann_latent"] = {
    "source": str(latent_path),
    "latent_dim": len(latent_columns),
}
adata.write_h5ad("/path/to/adata_with_dann_latent.h5ad")
```

An inference run capped with `--max-rows` produces only a leading-row prefix;
attach that output only to the matching AnnData subset.

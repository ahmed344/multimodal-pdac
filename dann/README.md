# Adversarial Latent Fusion

This package implements the PDF's adversarial predictive bottleneck for sparse
MALDI-MSI. MSI rows stay in backed CSR form: the loader reads only nonzero peak
indices and intensities, and the encoder applies
`MLP_agg(mean(MLP_peak(intensity * embedding[peak])))`.

The four biology targets each use a structural-zero hurdle with three parameters:
zero probability `pi`, positive logit mean `mu`, and positive logit standard
deviation `sigma`. A gradient reversal layer trains the shared latent space to
confuse the batch discriminator while preserving IHC prediction.

## Standalone latent variance

After running `python -m dann.spatial`, compute diagnostics from its existing
`spatial_latent.parquet` and the exact tissue AnnData used for inference:

```bash
python -m dann.latent_variance --config dann/config.yaml
# Optional overrides:
python -m dann.latent_variance --latents /path/to/spatial_latent.parquet \
  --input /path/to/tissue.h5ad --output-dir /path/to/latent_variance \
  --slide-column batch
```

This independent script does not run inference or require a checkpoint/GPU.
Settings live in `latent_variance` in the YAML. Defaults write to
`data/PDAC/Results/dann/latent_variance/`, separately from analysis:

- `explained_variance.png`: per-PC and cumulative explained variance, with
  90%, 99%, and 99.9% thresholds marked.
- `latent_variance_summary.csv`: latent width, participation ratio, minimum PCs
  reaching each threshold (`pcs_90`, `pcs_99`, `pcs_99_9`),
  `variance_explained_by_slide_means`, total variance, observation/slide counts,
  source paths, grouping column, and inference coverage.
- `pca_variance.csv`: one-based PC numbers, covariance eigenvalues, explained
  variance fractions, and cumulative fractions.

PCA centers the raw latents without scaling individual dimensions. Total variance
is the trace of the sample covariance (denominator `n - 1`). Participation ratio
is the squared sum of eigenvalues divided by the sum of squared eigenvalues;
it measures effective dimensionality. Slide-mean explained variance divides
`sum_s n_s * ||mean_s - global_mean||^2` by
`sum_i ||latent_i - global_mean||^2`. Slides are defined by `batch` by default
and weighted by their pixel counts. This measures variation associated with
slide means, not necessarily technical batch effects or all slide differences.
CSV variance fractions range from 0 to 1; the plot uses percentages.

Parquet is streamed in chunks to compute the full covariance and spectrum;
only observation names and slide labels are read from AnnData. Row positions
and names must match, including for duplicate names. Leading-row inference
prefixes from `--max-rows` are accepted and recorded; their statistics describe
only that prefix. Missing labels and nonfinite latents are errors. Constant
latents produce zero total variance, empty undefined CSV metrics, and an
annotated figure. Repeated runs replace the three diagnostic output files.

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
- `latent_umap_density_means.png` (logit of sampled density means for every target)
- `latent_umap_ziln_Density_*.png` (one 2x3 Logit/1-pi/mu/sigma/sampled-mean/batch figure per target)
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
  logit_mean/logit_sd/logit_median/logit_q05/logit_q95 summaries and density_mean)
- `spatial_latent.parquet` (ordered full latent vectors)
- `spatial_heatmap_Density_*.png` (one 6-column per-batch parameter/sampled-mean/HES figure per target)
- `spatial_heatmap_diagnostics_Density_*.png` (one 5-column per-batch normal diagnostic figure per target)

## Predictions versus parameters

The biology head emits a distribution: `pi` is the probability of a structural
zero, and `(mu, sigma)` define a normal on the logit scale. The positive-branch
logit mean and median both equal `mu`; its standard deviation equals `sigma`.

`dann/ziln.py` provides positive-branch logit summaries, unconditional exceedance
probabilities, and a Monte Carlo density mean. For each pixel and target, each
draw is zero with probability `pi`; otherwise it is `expit(Normal(mu, sigma))`.
The arithmetic mean estimates the full expected density, including absence.
Only means are saved, using bounded sampling blocks and float64 accumulation.

`analysis.density_mc_samples` defaults to `1000`; `analysis.density_mc_seed`
defaults to `20260719`. Both are integers (positive count, nonnegative seed).
Older checkpoints use these defaults. Spatial inference accepts
`--density-mc-samples` and `--density-mc-seed` overrides. The isolated generator
reproduces results with the same settings, parameters, processing order, and
batch boundaries; changing batch size or analyzed splits can change sampled
means. Independent analysis and spatial runs can therefore differ slightly.

The UMAP CSV and spatial Parquet add `density_mean_<target>`; the positive-only
scatter CSV adds `predicted_density_mean`. Spatial prediction columns remain
float32. Plots reuse exported means and apply the observed-density clipped logit transform
with the same epsilon in each figure. A sampled mean of zero is uncolored (gray
in per-target UMAPs and spatial maps, omitted in the combined UMAP), just like
an observed zero; positive predictions remain visible even where observations
are zero. UMAP colors use the configured percentiles, and spatial colors use
per-panel limits, matching the observed panels. Saved means stay on the original
density scale: the plotted quantity is `logit(mean density)`, not a mean of logits.
Existing logit diagnostics retain their original meaning.

Regenerate outputs from an existing checkpoint without retraining:

```bash
python -m dann.analyze --config dann/config.yaml
python -m dann.spatial --overwrite --density-mc-samples 1000 --density-mc-seed 20260719
python -m dann.spatial_heatmaps
```

Set configuration/checkpoint and input/output paths for your data as needed.
Old spatial Parquet files lack mean columns and must be regenerated before
plotting the new mean panel.

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

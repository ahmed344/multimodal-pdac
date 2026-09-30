# Adversarial Latent Fusion

This package fits sparse MALDI-MSI to four IHC targets and exports a shared latent
representation. MSI stays in backed CSR form until Deep Sets mean-pools the
intensity-weighted learned peak embeddings. Aggregation and the two prediction
heads can independently use MLPs or local spatial CNNs.

The four biology targets each use a structural-zero hurdle with three parameters:
zero probability `pi`, positive logit mean `mu`, and positive logit standard
deviation `sigma`. A gradient reversal layer trains the shared latent space to
confuse the batch discriminator while preserving IHC prediction.

## Components and spatial execution

There are three groups and four selectors, all in `dann/config.yaml`:

| Group | Selector | Choices | Default |
|---|---|---|---|
| Spectral processing | `model.spectral_encoder.type` | `deep_sets` | `deep_sets` |
| Latent aggregation | `model.aggregation.type` | `mlp`, `cnn` | `cnn` |
| Prediction heads | `model.heads.biology.type` | `mlp`, `cnn` | `cnn` |
| Prediction heads | `model.heads.discriminator.type` | `mlp`, `cnn` | `mlp` |

All eight aggregation/head combinations are supported. Selectable neural modules
live in `dann/model_components/`; `dann/model.py` composes them. The Deep Sets stage uses
256-dimensional learned peak embeddings, multiplies each by its transformed
intensity, applies the original per-peak MLP with `[512, 512]` hidden widths and
512 output channels, and mean-pools over active peaks. Empty spectra pool to zero.
The selectable aggregation produces the **exported 512-dimensional latent**.
The MLP aggregation preserves `[512, 512]` hidden widths; both MLP heads preserve
`[512, 256, 128]`. The shared ZILN transformation, weighted four-target objective,
class-weighted batch cross-entropy, and gradient reversal schedule are unchanged.
Only the discriminator input receives gradient reversal.

Each CNN independently projects to 128 channels, applies three 3×3 stride-one
convolutions, and finishes with a pointwise output projection. Hidden stages use
per-pixel channel LayerNorm, GELU, and dropout 0.10. Occupancy masks suppress
features in missing tissue after every hidden stage and the output projection.
There is no spatial pooling or normalization over a whole tile.

All-MLP configurations use the original sparse pixel loader and the settings in
`training.pixel`: 1,024 pixels per optimizer step, 4,096 per validation batch, and
learning rate 0.01. Any CNN selects `training.spatial`: **eight tiles** per optimizer
step, 32×32 core positions per tile, learning rate 0.001, four workers, and prefetch
factor two. Eight tiles contain at most 8,192 core positions; actual supervised
counts depend on tissue occupancy, the selected split, and per-target validity.
Tiles shuffle across slides without requiring distinct slides within a batch.

Eight tiles is a fitting baseline, not a GPU-memory ceiling. A 2026-09-29 RTX 5090
comparison used the same 32 real-data tiles (17,926 supervised core rows and
19,953,249 active peaks), one warm-up per batch size, and three timed passes.
Median times for all 32 tiles were 3.72, 3.73, and 3.79 seconds at batch sizes
8, 16, and 32; peak allocated memory was 1.37, 1.54, and 2.58 GiB respectively.
Timing included collation, transfer, forward/backward, gradient clipping, and
AdamW, but excluded HDF5 reads. Larger logical batches showed no throughput
improvement in this check. They do not enlarge the spectral microbatches and
reduce the number of optimizer updates per epoch. Keep eight as the default;
evaluate larger batches using held-out CD8 loss if changing the fitting setup.

The halo is derived as `aggregation radius + max(biology radius, discriminator
radius)`. An MLP has radius zero; a CNN has radius equal to its depth. Defaults
therefore use halo six and 44×44 input tiles. Each biology prediction sees a
13×13 spectral-input neighborhood, and each exported latent sees 7×7. Both heads
run before core rows are selected, so CNN heads retain their latent halo.

Context comes from `data.context_path` (`adata_assembled_tissue.h5ad`), while
supervision comes from `data.path` (`adata_assembled.h5ad`). `batch_column`,
`x_column`, and `y_column` define unique integer grid identities. Feature order
and labeled-to-context alignment are validated. Observation names may repeat.
Coordinates are never compressed, interpolated, or joined across slides. Held-out
and unlabeled MSI can provide context, but only the selected core rows contribute
biology and batch losses. No edge rows are excluded.

`model.spectral_peak_budget: 65536` limits spectral microbatches to complete
spectra targeting that many active peaks. A single spectrum exceeding the budget
stays intact. `model.spectral_checkpointing: true` recomputes these microbatches
with preserved dropout RNG during spatial training. Pooled representations remain
differentiable and are recomputed every step. Raw MSI is never densified. Inference
also bounds spectral execution. Disabling checkpointing increases activation
memory; it does not change the selected architecture.

For the original architecture, set aggregation, biology, and discriminator types
to `mlp`. For a CNN batch discriminator, change only
`model.heads.discriminator.type` to `cnn`. Alternative component settings remain
next to the selectors. Changing depth automatically changes the required halo.
Edit mode-specific settings in `training.pixel` or `training.spatial`; the loader
writes the selected effective settings in `execution` and `resolved_config.yaml`.
Smoke overrides reduce label counts, cores (8×8), tiles (two), epochs, and workers,
and record all effective values.

Checkpoint selection uses **mean validation CD8 ZILN loss over valid CD8 rows**.
Training still optimizes all four weighted targets plus the adversarial objective.
Checkpoints save the exact split positions plus an ordered row/feature identity
fingerprint; analysis reuses those splits and may cap them further. There is no
refit using validation or test labels. Version-two architecture metadata records
selected components, numerical settings, receptive fields, batch order, targets,
and optimizer parameter names. Legacy flat configurations mean Deep Sets with
MLP aggregation and heads. Legacy parameter-key conversion is identity because
keys and registration order were retained; optimizer ordering and state shapes
are checked before resume. Incompatible components or targets are rejected.
A legacy selection score is reset when resuming under the CD8 selection metric.

All prediction consumers should use `model(batch)`. Calling `encode()` and then
the biology head loses the spatial halo when a CNN head is selected. Peak-embedding
analysis remains available for every combination through the Deep Sets dictionary.

## Result routing

`results.root`, the four selectors, and `results.run_name` (default `fit`) derive:

```text
data/PDAC/Results/dann/<combination>/<run_name>/
    model/
    analysis/
    analysis/spatial/
    latent_variance/
```

The default combination is `deep_sets__agg-cnn__bio-cnn__disc-mlp`. Smoke artifacts
insert `smoke/` after the run name. Training, analysis, inference, heatmaps, and
latent variance use this root; no separate architecture YAML files are required.
For a smoke checkpoint, pass its `model/resolved_config.yaml` to downstream CLIs.
Checkpoint overrides use saved architecture and run routing.

Spatial exports preserve existing schemas and original tissue row order. CNN
inference stages tile-order latents/parameters in temporary disk arrays, then
streams them in row order. Temporary staging is removed on success or failure;
allow roughly four bytes per row per exported latent/parameter channel on disk.
`--max-rows` limits exported cores while retaining the full tissue context.
`spatial_predictions.support.parquet` reports occupied-neighborhood fractions for
biology and latent receptive fields. `spatial_predictions.provenance.json` records
checkpoint/input paths, context and output counts, and zero edge exclusions.

Analysis adds `<split>_density_errors.csv` with density-scale MAE, RMSE, bias, and
hurdle Brier scores for all rows and for boundary/interior subsets. Boundary means
incomplete occupied support in the biology receptive field.
`<split>_spatial_contrasts.csv` compares observed and predicted differences between
adjacent held-out pixels; a cap may leave zero eligible pairs. Calibration uses the
existing hurdle and positive-branch metrics. Spatial smoothness alone is not
evidence of denoising or improved biological accuracy.

## CD8 per non-tumor area

The supplied configuration enables `data.cd8_normalization.enabled` with
`min_non_tumor_fraction: 0.01`. The CD8 target is `Density_CD8 / (1 - Density_Tumor)`.
Rows with less non-tumor area or a ratio above one lose only CD8 supervision;
the other targets and domain objective remain active. The threshold is inclusive.
Valid zeros retain the hurdle-zero label, and exact-one ratios use existing logit
clipping. Both CD8 loss branches are masked. Mean biology losses and epoch metrics
use valid target-weight mass as the denominator.

Raw input densities must be finite and in `[0, 1]`. Training reports exclusions
separately by cause and split in the console and `model/data_schema.json`, and
fails if its selected rows have no valid CD8 observations or no valid positive CD8.

Retrain to obtain normalized-CD8 predictions. Checkpoints store the transformation
and ordered targets; incompatible resumes are rejected. Checkpoints without this
contract retain raw-CD8 semantics, including when analyzed using the new YAML.
Prediction formulas and machine-readable prediction column names are unchanged.
Plot labels identify normalized CD8. Wide analysis CSVs retain `raw_*`, `valid_*`,
and `label_*` columns, with excluded observations missing in the model-target
columns. They also include the transformation contract for standalone calibration.
Spatial Parquet metadata carries the same contract so heatmaps transform observed
CD8 consistently. Missing tissue annotations stay missing in heatmaps while
predictions remain available; normalized observed CD8 requires both CD8 and tumor
annotations. Older exports without metadata are interpreted as raw CD8.

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
`<run-root>/latent_variance/`, separately from analysis:

- `explained_variance.png`: stacked per-PC (logarithmic y-axis) and cumulative
  explained variance, with 90%, 99%, 99.9%, 99.99%, and 99.999% thresholds marked.
- `latent_variance_summary.csv`: latent width, participation ratio, minimum PCs
  reaching each threshold (`pcs_90`, `pcs_99`, `pcs_99_9`, `pcs_99_99`, `pcs_99_999`),
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
in `dann/config.yaml` and this README.

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
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q dann/tests ihc_mvn/tests
```

Within the resolved run root, training artifacts go under `model/` and analysis
under `analysis/` (UMAP exports in `analysis/umap/`):

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

- `spatial_predictions.parquet` (ordered mu/presence/sigma plus the derived
  logit_mean/logit_sd/logit_median/logit_q05/logit_q95 summaries and density_mean)
- `spatial_latent.parquet` (ordered full latent vectors)
- `heatmaps/spatial_heatmap_Density_*.png` (one 6-column per-batch parameter/sampled-mean/HES figure per target)
- `heatmaps/spatial_heatmap_diagnostics_Density_*.png` (one 5-column per-batch normal diagnostic figure per target)

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
reproduces analysis results with the same settings, parameters, processing order,
and batch boundaries. Spatial exports use row-keyed RNG streams so caps, tile
order, and output batch sizes do not change shared-row sampled means. Independent
analysis and spatial runs can differ slightly because their RNG streams differ.

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

## Reproducible implementation checks

```bash
python -m dann.verification.legacy_equivalence
python -m dann.verification.real_smoke --mode benchmark
python -m dann.verification.real_smoke --mode mlp
python -m dann.verification.real_smoke --mode cnn
```

The baseline comparison loads the original model from Git commit `256dd33`, copies
weights, and compares predictions, gradients, and legacy optimizer resume. Real
smoke runs use 128 training, 64 validation, and 64 test labels, one epoch, and a
128-row tissue export. They write under each combination's `verification/smoke/`.
The benchmark uses the uncapped training split with the full default logical
spatial batch and records GPU memory, wall time, occupancy, and supervised counts.
These checks validate execution and contracts; they do not compare converged fits.

### Training-only fit diagnosis

Before another full fit, run the bounded diagnostic separately from the completed
run (the output directory must not already exist):

```bash
python -m dann.verification.fit_diagnosis \
  --checkpoint data/PDAC/Results/dann/deep_sets__agg-cnn__bio-cnn__disc-mlp/fit/model/best.pt \
  --output data/PDAC/Results/dann/diagnostics/new-fit-diagnosis
```

This uses 64 saved training rows, balanced between zero and positive CD8 in four
8×8 cores on two slides, retaining the full CNN halo. All four target masks and
the configured biology loss are preserved. Both CNN and MLP fits use the same
tissue spectra; any differences from the labeled file are recorded explicitly.
The runner audits fresh/fitted variation, MSI sensitivity, biology gradients,
and updates, then tries the three prescribed dropout-free, non-adversarial fits
(up to 500 updates or five minutes each). A CNN gate pass triggers separate
fresh dropout and adversarial follow-ups. Otherwise it audits the final stages
without searching additional hyperparameters. The GPU experiment budget is
30 minutes, excluding CPU input preparation. Early success requires both CD8
and weighted biology improvements of at least 0.05 over subset-only marginal
constants and positive-CD8 logit correlation above 0.8.

Settings, identities, input batches, baselines, audit JSON, losses, timings, and
diagnostic-only checkpoints are saved in the new directory. These checkpoints
are evidence for the experiment and are not production resume/export artifacts.

For a completed diagnosis in which neither CNN passes, inspect layer-wise
variation and positive-CD8 residuals without additional fitting:

```bash
python -m dann.verification.fit_stage_audit \
  --run data/PDAC/Results/dann/diagnostics/new-fit-diagnosis
```

This records its own timing and the combined experiment time in
`layer_followup.json`. It retains the original loss and model parameters.

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

All-MLP configurations retain the sparse pixel loader and `training.pixel` settings.
Any CNN selects `training.spatial`. The checked-in configuration now uses
`sampling_strategy: proportional_slide_tiles`, 8×8 cores, and
`supervised_rows_per_batch: 2048`. Every selected training row appears once per
epoch, including rows with invalid targets (validity masks still control loss).
Only the final batch may contain fewer rows. No labels influence sampling.

For zero-based epoch `e`, an isolated NumPy generator uses `seed + 50000 + e`.
Slides are visited in sorted order and their tile queues independently shuffled.
Tiles merge by normalized cumulative row midpoint, seeded random tie-break, then
tile index. Their existing row order is preserved. Larger slides contribute
proportionally; an update need not contain every slide. Batch boundaries can split
a tile's supervision across updates, each retaining the entire original context.
Requests are generated in the main process and sent immutably to persistent
workers; the dataset reads CSR spectra lazily without a dense MSI matrix.

Missing `sampling_strategy` retains legacy `shuffled_tiles`. Under that strategy,
`tiles_per_batch` controls training. Under the corrected strategy it controls only
validation/test tile batches (default eight), never caps training tiles.
`execution` metadata records training row and evaluation tile units separately.
Smoke mode caps split membership and evaluation tiles but preserves the training
row budget and architecture-derived halo. The activated run name is
`proportional-slide-tiles-8x8-2048`, separate from earlier `fit` directories.

Production retains its epoch-based GRL horizon:
`(epoch * len(train_loader) + step) / (epochs * len(train_loader) - 1)` (with the
existing denominator guard). The diagnostic used a fixed update horizon.
Dropout, AdamW, clipping, loss weights, peak budget and checkpointing are unchanged.
New checkpoints have a separate versioned `sampling_contract` and epoch-boundary
`sampler_state`. Optimizer resume rejects changes in strategy, algorithm, seed,
core size, training batch budget, or ordered selected-row identities. Old
checkpoints imply legacy sampling using their saved configuration. Inference and
architecture checks do not depend on sampler compatibility; no mid-epoch resume
is supported.

Training writes `model/sampling/epoch_NNNN_batches.jsonl` with supervised rows,
unique tiles, distinct slides, slide entropy (natural logarithm), largest slide
fraction, cumulative rows and epoch position. The corresponding summary records
composition ranges/means and SHA-256 hashes of actual row order and cumulative
boundaries (little-endian int64, including the initial zero boundary).

The preceding diagnostic supports this sampling correction for spatial MLP and
discriminator-only CNN controls. It does not confirm convergence of the active
CNN aggregation/CNN biology/MLP discriminator architecture. Global pixel
shuffling performed better in that study.

The halo is derived as `aggregation radius + max(biology radius, discriminator
radius)`. An MLP has radius zero; a CNN has radius equal to its depth. Defaults
therefore use halo six and 20×20 input tiles. Each biology prediction sees a
13×13 spectral-input neighborhood, and each exported latent sees 7×7. Both heads
run before core rows are selected, so CNN heads retain their latent halo.

All supervised spectra, neighborhoods, labels, and masks come from `data.path`
(`adata_assembled.h5ad`). `data.context_path` is a backward-compatible name for
an **inference-only** input (`adata_assembled_tissue.h5ad`); training works when it
is absent or nonexistent. Inference resolves explicit `--input` before the runtime
`data.context_path`, never from a checkpoint or the training-file default.
`batch_column`, `x_column`, and `y_column` define unique integer grid identities
within each file independently. Coordinate matches across these files do not
establish biological correspondence. Observation names may repeat. Coordinates
are never compressed, interpolated, or joined across slides. All occupied
labeled-file rows, including held-out spectra, may provide context; only selected
split core labels contribute biology and batch losses. Native gaps remain masked,
and no lesion filter or edge exclusion is applied.

Checkpoints record a versioned `training_data_contract` separately from model
architecture and preprocessing. Missing or incompatible provenance prevents
training resume and supervised analysis. Legacy weights remain loadable for
inference/inspection, with unknown provenance recorded in export metadata; saved
splits may still initialize fresh experiments. Historical checkpoints are never
rewritten.

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
weights, and compares predictions, gradients, and legacy weight loading. Real
smoke runs use 128 training, 64 validation, and 64 test labels and one epoch.
An optional `--export-inference` adds a 128-row independent inference export. They write under each combination's `verification/smoke/`.
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
labeled-file spectra; supervised core values are checked against direct labeled-file reads.
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

### Full-epoch CNN isolation controls

`python -m dann.verification.epoch_comparison` runs one isolated comparison in a
**new** output directory. Use the staged all-MLP configuration and the preserved
MLP checkpoint for its saved split and row/feature identity contract. For example:

```bash
python -m dann.verification.epoch_comparison \
  --config dann/config.yaml \
  --reference-checkpoint <preserved-MLP-model/best.pt> \
  --output <new-diagnostic-directory>/A \
  --input-source labeled --execution-path pixel --epoch-limit 3
```

Run controls sequentially and review each completed three-epoch trajectory before
launching its successor. Corrected A uses `labeled/pixel`; C uses
`labeled/flat-tiles`; D uses `labeled/spatial`. Tissue training is rejected. C and D retain exactly the same
supervised tile-core row ordering and eight-tile logical batches. D copies the
MLP into 1×1 convolutions and checks deterministic outputs, parameter gradients,
and an AdamW update in float64 before training with configured dropout. Missing
sites are masked after every layer. Spectral microbatching remains enabled in D.

Progression requires epoch-three validation CD8 loss to improve over the
training-fitted constant by at least `early_stopping_min_delta`, biology loss to
beat its constant, and positive-CD8 R² to be positive. Report MSI shuffle
sensitivity alongside this decision. Repeat a failed control once unchanged;
stop downstream training on a second failure or disagreeing outcomes. The source
correction does not authorize 3×3 controls, depth changes, learning-rate searches,
or production fits.

**Historical correction:** previous B/C comparisons used inappropriate tissue
training spectra. They cannot establish tile-batching failure in the intended
PDAC pipeline; tissue-trained CNN results do not validate the corrected pipeline.
A's labeled-file measurements and the seeded-initialization fix remain relevant.
See the correction notice in
[`2026-09-30-dann-three-epoch-isolation.md`](../markdowns/2026-09-30-dann-three-epoch-isolation.md).

The runner retains `training.epochs` as the GRL schedule horizon (300 for the
staged reference); `--epoch-limit 3` stops after three complete epochs. It always
uses the reference's 0.01 learning rate, never the spatial learning rate. No test
evaluation, early stopping, sample caps, or tissue exports occur. Constants are
fit exclusively on training labels and scored on validation labels. Stage,
shuffle, and gradient audits use a recorded fixed validation batch, with no
optimizer update and with training RNG state preserved. Full validation metrics
and predictions are saved before training and after every epoch. Histories,
source snapshots, split arrays, initial MLP weights, checkpoints, settings,
update counts, row-order hashes, timings, and preservation checks remain under
the new directory. Historical working-run metrics provide context; the old
64-row overfit threshold is not used as a full-data gate.

Pixel and flattened-tile checkpoints retain the production architecture schema;
their source/batching protocol is recorded separately in `settings.json` and
must be respected when reproducing training. Converted spatial controls use the
explicit `diagnostic-spatial-v1` format and must be reconstructed using
`spatial_control`, not passed to production inference/resume. Production
checkpoint keys and parameter registration order have not changed. The encoder
now preserves the original seeded initialization order: construct peak and
aggregation MLPs before the final normal draw for the embedding table.

Large flattened tile batches can fragment the CUDA allocator. Prefix the same
command with `PYTORCH_ALLOC_CONF=expandable_segments:True` when needed; this is
recorded in diagnostic settings and leaves logical batches and model numerics
unchanged. The flat reader loads only supervised core spectra, with regression
coverage against the full grid reader's core extraction.

Corrected source-isolation tests and the required sequential comparisons are
reported in [`2026-09-30-dann-labeled-source-correction.md`](../markdowns/2026-09-30-dann-labeled-source-correction.md).
The two corrected A runs disagreed, so the stopping rule withheld C and D.

### Matched discriminator controls

The separately authorized discriminator diagnostic isolates tile execution,
shared gradient clipping, and adversarial reversal. It starts fresh models using
only the reference checkpoint's saved splits, class ordering, and data identity:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m dann.verification.discriminator_controls \
  --config dann/config.yaml \
  --reference-checkpoint <MLP-model/best.pt> \
  --output <new-diagnostic-directory>
```

The four primary controls use the same MLP encoder/biology initialization,
eight 32×32 training tiles with a three-pixel halo, learning rate 0.001, and
configured dropout and global gradient clipping. A forces the all-MLP model
through spatial execution with discriminator loss disabled. B replaces only the
discriminator with a CNN, still disabling its loss. C enables discriminator loss
with zero GRL; D enables the configured GRL ramp. The ramp keeps the original
300-epoch horizon. Controls have matching tile-order generators and per-update
dropout seeds, preventing discriminator random draws from shifting the next
update's biology randomness. Disabling discriminator loss omits its backward
path entirely, so AdamW also skips its parameters.

Each control performs 500 updates, cycling the training loader if needed, and
scores every saved validation row initially and every 100 updates. Since the
encoder and biology head are pointwise in every control, validation uses the
equivalent pixel biology path; a numerical test checks this against full spatial
execution. A fixed training batch supplies representation variation, spectrum
shuffle sensitivity, and separate biology/discriminator gradient audits. The
audits preserve training RNG and gradients. Actual training logs include global
gradient norms, clipping factors, slide counts, and rows per update.

If both A and B finish with nonpositive positive-CD8 logit R², the runner adds
A and B at learning rate 0.0001 and a shuffled-pixel MLP at 0.001, each with
discriminator loss disabled and the same 500-update limit. Pixel batches retain
the configured pixel batch size; compare both update counts and processed rows.
`--updates` and `--evaluate-every` permit smaller synthetic or smoke checks.

Outputs include a frozen reference snapshot, source/config snapshots, split
identities, training-fitted marginal baselines, initial weights, histories,
validation predictions, and final diagnostic checkpoints. The latter use
`diagnostic-discriminator-v1` and must be reconstructed with `make_control`;
they are not production resume/inference checkpoints. Existing production files
are never written, and concurrently running training is left untouched. These
bounded comparisons use no test labels and do not launch a full fit.

The completed 2026-09-30 run performed all seven 500-update controls; evidence is
in `data/PDAC/Results/dann/diagnostics/2026-09-30-discriminator-controls/`, with
`diagnosis.txt` and `summary.csv` summarizing the results. At learning rate 0.001,
the forced-spatial MLP and all three CNN-discriminator controls converged to
almost constant CD8 predictions (validation positive-logit R² approximately
−0.00009). The shuffled-pixel MLP at the **same** learning rate reached R² 0.01979
and CD8 loss 1.23655, beating the training-fitted marginal baseline of 1.24841.
The 0.0001 spatial controls retained MSI sensitivity but failed the prediction
baseline within the update cap (R² −0.75495).

This implicates the spatial training regime, without identifying one sampler or
execution mechanism: tile updates averaged 4,353 supervised rows from 7.188
slides, versus 2,048 rows from 42.996 slides for shuffled pixels. A matched
flat-tile MLP control would separate this batch-composition change from spatial
microbatch/dropout execution. The measured early ramp reached only 0.00666;
these experiments do not evaluate the full 0.25 adversarial strength. They do
not establish a production fix. The full synthetic suites passed 265 tests.

### Batch composition and execution diagnosis

`dann.verification.batch_execution_diagnosis` freezes the preceding diagnostic's
configuration, split identities, and fresh initialization. It separates ordinary
pixel execution, core spectral microbatching, checkpointing, context processing,
and spatial grid execution. Production settings and checkpoint APIs are unchanged.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m dann.verification.batch_execution_diagnosis \
  --output <new-diagnostic-directory> --stage B
```

Stage A performs strict float64 forward/gradient/clipped-AdamW equivalence and
checkpoint RNG checks, then preflights each seed and flat schedule at its largest planned peak load.
Stage B performs 500-update P/F/S comparisons for seeds 20260719–20260721. Stages
C1 and C2 are explicitly selected after reviewing B: C1 compares complete-pass
composition schedules; C2 runs the intermediate execution ladder. `--controls`
and `--seeds` select individual recorded controls. `--repeat <suffix>` creates
an unchanged repeat without replacing evidence.

Stage D accepts an evidence-selected `--dropout spectral-off` or
`--dropout aggregation-off`, or `--sampler small`, `rounds`, or `rounds-small`.
The rounds sampler uses 8×8 cores, complete halos, and seeded rounds over active
slides without oversampling. `--placement discriminator`, `aggregation`, or
`biology` constructs paired CNN confirmations with restored adversarial loss
and the fixed original GRL horizon. A correction must be selected from completed
causal evidence; these switches are not recommended production settings.

Sampling plans contain ordered row IDs, batch boundaries, pass boundaries, tile
identities, and hashes. Spectra remain sparse and are read lazily. Full validation
checks exact row coverage and uses complete spatial inference for CNN aggregation
or biology. Fixed training-only audits preserve RNG and existing gradients.
Diagnostic states use `batch-execution-diagnostic-v1`, carry optimizer and plan
positions, and are not production checkpoints.

The 120-minute persistent wall-clock budget starts with the first real GPU audit.
Causal stages reserve 35 minutes; correction stages reserve five minutes. Group
admission uses `--estimate-seconds` with an additional 20% margin. `--resume`
retains that deadline and skips only completed runs whose scientific contracts
and artifact hashes match. Incomplete attempts remain immutable; a new explicit
repeat is required. `--stage report` regenerates the comparison CSV and plots
from recorded evidence without GPU training.

The completed study is recorded in
`data/PDAC/Results/dann/diagnostics/2026-10-01-batch-execution-audit-fix/`;
`diagnosis.txt`, `comparison.csv`, and the per-seed `curves_*.png` contain the
results. All three globally mixed native-sized controls learned, while all
three within-slide-dispersed controls failed at identical batch sizes, both
at update 500 and after two complete passes. Smaller local batches also failed
in all three seeds. This supports a between-slide composition effect under the
frozen optimizer/dropout settings; larger batches alone do not force collapse.

The specified 8×8 active-slide-round correction **failed all three spatial MLP
seeds**. Its mean slide diversity rose to about 30, but fell to one near pass
ends, and final CD8 spectrum-shuffle sensitivity remained negligible. A paired
first-seed discriminator-CNN screen also failed; other CNN confirmations and the
stochastic execution ladder were budget-limited. No production change is
validated. One third-seed native monolithic control was memory-infeasible and
is explicitly excluded from completed comparisons. The initial diagnostic audit
OOM and its correction are preserved in the sibling original-attempt directory.

GPU experiments stopped within the 120-minute persistent deadline. Final checks
verified 27 complete controls and 256 exact validation exports; 284 synthetic
tests passed. The final runner preflights each seed/schedule separately, so an
OOM in a native batch does not block a feasible smaller schedule, and rejects
incompatible stage/execution/correction choices. These administrative safeguards
were added after the experiment; its exact implementation remains frozen under
`<study>/source/`. Resume intentionally rejects changed source contracts. To
resume that frozen implementation, run from `<study>/source/`, passing absolute
`--previous-run` and `--output` paths together with `--resume`. The source archive
contains the package modules needed for that invocation.

### Extended 1,000-update composition/execution study

`python -m dann.verification.extended_batch_diagnosis --output <new-directory>`
launches the isolated follow-up with a persistent 240-minute cap. It preserves the
archived data/split/initialization contract and uses three prescribed seeds. Every
fit has 1,000 optimizer updates; complete-pass plans include final partial batches.
The original diagnostic CLI also accepts `--updates` without changing its existing
default stage lengths.

The follow-up compares globally shuffled pixels and native spatial batches, then
pairs flat/spatial execution on identical 2,048-row batches from 8×8 tile cores.
Per-slide shuffled queues are interleaved by normalized cumulative supervised-row
midpoints, spreading slide contributions throughout each pass without oversampling.
Halo three, dropout, learning rate, target masks, and the biology loss are preserved.
Training-only density/spectral audits do not exclude data or assert registration
correctness. An exact globally shuffled spatial-context comparison requires a
separate memory/time gate; blocked controls are never silently approximated.

Subsequent complete three-seed groups follow the observed contrast: execution
ladder, within-slide dispersion, or local ordering with smaller spatial supervision.
An unchanged decisive-pair repeat precedes discriminator-CNN confirmation. The
complete three-seed CNN group is admitted when the budget permits; otherwise only
the prescribed first-seed preliminary screen is considered, before seeing CNN
outcomes. Admission includes a 25% margin and a ten-minute final reserve. All
fits are sequential; immutable failures and incomplete comparisons are reported.
No production defaults or preprocessing are changed.

Use `--controls Qflat Qspatial --seeds 20260719` with a new output directory to
replay an individual matched MLP comparison. Available controls also include
`P`, `S`, `QX1`, `QX2`, `QX3`, `LocalSmall`, and `Qdispersed`. Explicit `--resume`
skips completed contract-matching controls and retains the original deadline;
incomplete attempts are never overwritten. Full studies require all three seeds.

The 1,000-update results are stored in
`data/PDAC/Results/dann/diagnostics/2026-10-01-batch-execution-1000/`.
The original spatial regime and its smaller-local-batch variant failed all three
MLP seeds. The proportional 8×8 spatial regime passed all three, with an unchanged
first-seed repeat confirming the original/corrected failure/success contrast.
All corrected spatial MLP seeds had failed at 500 updates, making the longer
diagnostic horizon consequential. Globally shuffled pixels still performed best.

The restored-adversarial discriminator-only CNN comparison also passed all three
corrected seeds and failed all three original seeds. Aggregation/biology CNN
placements and full-strength GRL remain untested. Dispersion within slides improved
the flat controls at exactly matched batch sizes and slide counts. Execution
ladder outcomes were seed-sensitive; strict numerical and checkpoint RNG checks
found no production implementation defect. Registration errors and the causal
effects of low-signal slides were not ruled out.

See `diagnosis.txt`, `final_assessment.json`, `matched_pass_endpoints.csv`, and
`composition_summary.png` / `execution_summary.png` / `cnn_summary.png` for
per-seed evidence, exposure comparisons, and limitations. The complete sampling
regime is a supported diagnostic candidate, not an automatic production change.
The exact experiment source remains under `source/`, with separately executed
follow-up orchestration under `analysis_source/`. The final integrated runner and
tests are archived separately under `final_implementation/`; resume deliberately
rejects a different source contract. Use the frozen source archive to reproduce
the recorded implementation.

## Production sampler integration verification

The production parity tests are in `dann/tests/test_sampling.py`. Run the bounded
real-data audit with a new output directory:

```bash
python -m dann.verification.production_sampling_audit \
  --output data/PDAC/Results/dann/diagnostics/<new-audit-name>
```

The audit has a 20-minute maximum including checks and validation. It compares
six full passes with archived Q plans, checks original batch fields and full
halos, preflights the largest context workload, executes ten updates through
`run_epoch` with the production worker settings, and evaluates a bounded spatial
validation subset. It records failures without changing execution settings and
never scores test labels. Its partial training audit summaries deliberately say
`complete: false`; no partial epoch is saved as a resumable training checkpoint.

The initial integration audit passed in about one minute; all 327 repository
tests passed. See
`data/PDAC/Results/dann/diagnostics/2026-10-01-production-sampling-audit/summary.txt`
and `report.json` for plan hashes, composition, GPU memory and bounded results.

## Two-architecture production fits

Run the CNN aggregation/biology model, then the MLP aggregation/CNN biology model,
with a fresh orchestration directory:

```bash
python -m dann.verification.production_fits \
  --output data/PDAC/Results/dann/production/<new-study-name>
```

This entry point copies the current YAML, changes only run names and the second
aggregation selector, and verifies the prescribed seed, 300-epoch cap, five-epoch
patience, 0.001 selection delta, 8×8 cores and 2,048-row proportional sampling.
Architecture-derived halos remain six and three pixels. It verifies both splits
and the first sampling pass against the archived audit, and fits a training-only
constant hurdle baseline before starting either fit. Existing run directories
are rejected, and training is never resumed or retried. This is a two-fit,
single-seed experiment, without the diagnostic audit's 20-minute limit.

The orchestration directory contains `A.yaml`, `B.yaml`, `manifest.json`,
`constant_baseline.json`, `splits.npz`, and `results.json`. Each architecture's
unique results directory contains `status.json`, stage logs, sampled GPU/disk
usage, resource summaries, and the usual model artifacts. Training failures
preserve completed artifacts and allow the other architecture to run unchanged.
The launcher waits for existing GPU compute processes without interrupting them.

Each normally completed best checkpoint is independently evaluated on the entire
validation split using full spatial execution. Downstream work requires CD8 loss
at least 0.001 below the constant baseline, weighted biology loss below baseline,
and positive-CD8 logit R² above zero. `assessment.json` records the three checks,
checkpoint hash, stopping/best epochs, metrics and exact validation coverage.
Only passing runs execute configured analysis, full tissue prediction/latent
export, spatial heatmaps and latent variance diagnostics. The configured 1,000
Monte Carlo samples and test analysis settings are retained. Export validators
check complete tissue row coverage, finite values and all target heatmaps.

Routing and threshold tests run without launching production fits:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q dann/tests/test_production_fits.py
```

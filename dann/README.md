# Adversarial Latent Fusion

This package fits sparse MALDI-MSI to four IHC targets and exports a shared latent
representation. MSI stays sparse, either backed or resident in RAM, until the spectral encoder
pools learned peak responses. Aggregation and the prediction
heads can independently use MLPs, local spatial CNNs, or residual GATv2 attention.

The four biology targets each use a structural-zero hurdle with three parameters:
zero probability `pi`, positive logit mean `mu`, and positive logit standard
deviation `sigma`. A gradient reversal layer trains the shared latent space to
confuse the batch discriminator while preserving IHC prediction.

## Spectral pooling

`model.spectral_encoder.deep_sets` selects either `pooling: occurrence_mlp`
with `normalization: count` (the historical default), or
`pooling: intensity_weighted_dictionary` with `normalization: l1` or `l2`.
Omitted fields reconstruct occurrence pooling; the checked-in configuration
retains that selection. Changing pooling, normalization, centering, or the
normalization floor requires fresh training.

Occurrence pooling applies the peak MLP separately to each intensity-scaled
embedding, retaining its original dropout, stored-entry counts, and
`pool_centering: none | zero_input` behavior. Dictionary pooling transforms all
embeddings once per encoder invocation, then multiplies the sparse transformed
intensities by that dictionary. `pool_centering: dictionary_mean` subtracts the
mean transformed peak vector; `none` leaves it uncentered. A dictionary has one
shared dropout realization across occurrences and internal spectral chunks.
No persistent dictionary cache survives an invocation or a parameter update.

With transformed intensities `I` (currently `log1p`), dictionary pooling is
`g = I @ v / d`, where `v[p] = peak_mlp(e[p]) - mean_q peak_mlp(e[q])` when
centered. L1 uses `d = max(sum(I), epsilon)`; L2 uses
`d = sqrt(max(sum(I**2), epsilon**2))`, with
`normalization_epsilon: 1.0e-8`. Duplicate entries are added **after** intensity
transformation and **before** row norms. Explicit zeros do not change the result;
empty spectra pool to zero with finite gradients. Sparse multiplication avoids
an active-occurrences by output-width tensor. There is no abundance channel.
Both norms remove positive global scaling above the floor; L2 has a different
row-dependent scale, so absolute latent variance alone is not a quality metric.

Checkpoint architecture version 8 records these semantics. Versions 2–7 keep
their original occurrence pooling and residual/centering interpretation, with
unchanged parameter registration and seeded draws. Existing raw-embedding
visualizations remain raw embeddings; an occurrence encoder does not have a
single intensity-independent transformed dictionary.

## Matched dictionary comparison

The isolated comparison runner uses `train_model` and optional observational
hooks. It checks identical seeded parameters, train-only head initialization,
feature order, targets, and saved splits against
`he10_biology2_discriminator1_non_centered` before fitting. The root YAML keeps
occurrence pooling. Explicit L1/L2 configurations, immutable manifest, initial
snapshots, logs, and per-stage completion records live beneath a new directory:

```bash
python -m dann.verification.dictionary_comparison --output NEW_DIRECTORY --prepare
python -m dann.verification.dictionary_comparison --output NEW_DIRECTORY --run
```

`--run` executes isolated real-data smoke workflows for both norms, sequential
full fits (300-epoch maximum, existing early stopping, full 224,100-update GRL
horizon), then the configured downstream workflows and matched diagnostics.
Rerunning validates completed stages; an interrupted fit resumes its compatible
`latest.pt`. To retry one stage, use `--arm l1 --stage analysis` (also `l2` and
`baseline`; stages include `smoke`, `train`, `inference`, `spatial_heatmaps`,
`latent_variance`, and `diagnostics`). Historical outputs are validated and reused;
missing diagnostics are written only in the comparison directory.

`report.html` is an offline report with CSV/JSON metrics and PNG/PDF figures.
All probes use the same at-most-65,536 proportionally slide-stratified training
rows, train-fitted scaling, and complete validation split. CD8 thresholds and
stromal quartiles are frozen from training. Patient bootstrap intervals are
paired across the fitted models and do not estimate variation across training
seeds. Initial/early observations preserve Python, NumPy, and PyTorch RNG and
module modes; historical early measurements that were not saved stay unavailable.
The comparison is specific to the configured PDAC data and baseline. H&E-to-Visium
transfer is assessed separately.

To extend a completed comparison with peak-vector and family quality diagnostics:

```bash
python -m dann.verification.peak_comparison --comparison COMPARISON_DIRECTORY
```

This writes `peak_comparison/report.html`, CSV tables, numerical arrays, and
PNG/PDF figures. It compares the original raw-embedding families with families
formed from the effective dictionaries; the historical encoder instead uses
intensity-conditioned responses at three training-derived intensity levels.
Family IDs are matched by peak overlap. Held-out within-slide IHC associations
are reported both directly and after controlling for total transformed intensity
and positive peak count, with paired patient intervals. Neighbor and family
association coherence use training-prevalence/intensity-matched permutation
controls and an unlearned nearest-m/z control. Baseline initialization comparisons
use a reconstructed common-seed reference, not a saved historical initialization.
These are descriptive peak-quality checks, not molecular annotations
or independent validation of the selected models.

## Raw H&E auxiliary prediction

`model.heads.he.enabled: true` adds a standard head from the shared MSI latent to
1,536 stored UNI2-h features. Its `type` and independent `mlp`, `cnn`, and `gatv2`
blocks follow the biology head interfaces; the initial settings copy biology.
`output_dim: 1536` must match `obsm[data.he_features_key]` (default
`hes_features`) in **data.path**, at exactly the MSI observation positions.
Targets are read lazily with handles local to each read, safe across preparation
threads and loader processes. Only selected core targets enter batches; large
H&E arrays are never retained in cached tile geometry. The reader rejects
incorrect shape, nonnumeric values, and nonfinite supervised values, reporting
invalid source row positions. Stored column order is preserved.

Outputs are unrestricted linear values. There is no feature normalization or
standardization, no IHC-validity masking of H&E, and no gradient reversal on the
H&E branch. The objective is:

```text
total = biology_weight * IHC_loss + batch_weight * batch_cross_entropy
        + he_weight * mean((he_prediction - raw_he_targets)**2)
```

The mean covers selected core pixels and feature dimensions. `loss.he_weight:
1.0` is the initial baseline, not a tuned optimum. Training and validation history
record `he_mse`, `he_weighted_loss`, `he_r2`, and the updated `total_loss`; squared-error
sums and element counts produce batch-size-independent H&E metrics. Loss plots
show `he_weighted_loss` as **H&E objective**, without a separate MSE panel.
The checked-in early-stopping criterion is the validation supervised objective
(weighted IHC plus H&E). The study split remains 90:10.

Analysis always writes `analysis/he/{split}_metrics.json` for an enabled head.
`analysis.export_he_predictions: true` additionally streams vectors into
`analysis/he/{split}_predictions.parquet`. Independently,
`analysis.spatial_execution.export_he_predictions: true` writes
`he_predictions.parquet` beside spatial outputs. Both switches default to false.
Each file contains `row_position`, `obs_name`, and a fixed-size float32 list
`he_prediction` in stored feature order. Row coverage is checked; observation
names may be duplicated because source positions are the unique identities.
Spatial exports use disk-backed staging to restore source order. Analysis caches
retain H&E error sums/counts, not full vectors; requesting vector exports reruns
extraction to stream them. Spatial inference needs no observed H&E features and
skips the head when feature export is disabled.

Enabled runs add `__he-{type}` to the experiment directory selector. To compare
against the existing model, set `model.heads.he.enabled: false` and choose a
separate `results.run_name`. Disabled or absent H&E settings preserve legacy
initialization, output keys, checkpoint contracts, and loading requirements.
Version-six enabled checkpoints record the head architecture, width, target key,
and raw-target convention. Adding the head to an older architecture requires a
fresh run (`training.resume_checkpoint: null`); compatible H&E checkpoints resume
normally using their saved objective settings. Do not compare the baseline weight
as though it had been optimized.

## Components and spatial execution

There are three groups and five selectors, all in `dann/config.yaml`:

| Group | Selector | Choices | Default |
|---|---|---|---|
| Spectral processing | `model.spectral_encoder.type` | `deep_sets` | `deep_sets` |
| Latent aggregation | `model.aggregation.type` | `mlp`, `cnn`, `gatv2` | `gatv2` |
| Prediction heads | `model.heads.biology.type` | `mlp`, `cnn`, `gatv2` | `gatv2` |
| Prediction heads | `model.heads.discriminator.type` | `mlp`, `cnn`, `gatv2` | `gatv2` |
| Optional H&E head | `model.heads.he.type` | `mlp`, `cnn`, `gatv2` | `gatv2` (enabled) |

All aggregation/head combinations are supported. Selectable neural modules
live in `dann/model_components/`; `dann/model.py` composes them. The Deep Sets stage uses
128-dimensional learned peak embeddings, multiplies each by its transformed
intensity, applies the original per-peak MLP with `[256, 256]` hidden widths and
256 output channels, and mean-pools over active peaks. Empty spectra pool to zero.
The selectable aggregation produces the **exported 256-dimensional latent**.
The MLP aggregation uses `[256, 256]` hidden widths; all MLP heads use
`[256, 128, 64]`. The shared ZILN transformation, weighted four-target objective,
class-weighted batch cross-entropy, and gradient reversal schedule are unchanged.
Only the discriminator input receives gradient reversal.

Each CNN independently projects to 128 channels, applies three 3×3 stride-one
convolutions, and finishes with a pointwise output projection. Hidden stages use
per-pixel channel LayerNorm, GELU, and dropout 0.10. Occupancy masks suppress
features in missing tissue after every hidden stage and the output projection.
There is no spatial pooling or normalization over a whole tile.

All three CNN components default to residual connections for new training.
The independent boolean options are `model.aggregation.cnn.residual`,
`model.heads.biology.cnn.residual`, and `model.heads.discriminator.cnn.residual`.
Each hidden 3×3 block adds its input to its normalized, activated, dropped-out
output, divides by `sqrt(2)`, and reapplies the tissue mask. Input/output
projections, parameter counts, receptive fields, and halo sizes are unchanged.
Set a component's `residual: false` to use its original plain CNN. These settings
apply only when that component selects `type: cnn`.

GATv2 uses [PyTorch Geometric 2.8.0.post1](https://pypi.org/project/torch-geometric/2.8.0.post1/),
pinned in `.devcontainer/requirements.txt`. Install it in an existing environment:

```bash
python -m pip install torch-geometric==2.8.0.post1
```

The implementation uses ordinary tensor edge lists; optional compiled neighbor
extensions are unnecessary. PyG is imported only when constructing a GATv2
component. MLP/CNN models and file-based consumers work without PyG installed.

Each component has an independent `gatv2` configuration block:

```yaml
gatv2:
  channels: 128
  depth: 3
  heads: 4
  neighbor_radius: 1.5
  residual: true
  dropout: 0.10
  attention_dropout: 0.10
```

`channels` is the total concatenated hidden width and must divide evenly by
`heads`. `depth` counts attention layers. `neighbor_radius` is a finite per-layer
Euclidean cutoff in grid units, at least 1: radius 1 includes the four cardinal
neighbors, 1.5 includes all eight immediate neighbors, and 2 also reaches cardinal
neighbors two grid units away. Every occupied pixel is a node, including empty
spectra. All eligible occupied endpoints connect in both directions plus exactly
one self-edge. Gaps may be spanned if endpoints are within the cutoff. Tiles have
independent graphs, including overlapping copies and tiles from different slides.
No label, absolute coordinate, slide identifier, spectral similarity, neighbor cap,
or random edge sampling enters graph construction.

Edges from source j to receiver i carry `[(x_j-x_i)/r, (y_j-y_i)/r, distance/r]`;
self-edge features are zero. Finite integer offset lookups avoid all-pairs distance
or adjacency matrices. Edges are built once per component forward and reused by
its layers; only offset templates are cached across calls.

A pointwise linear projection, node-wise LayerNorm, GELU and feature dropout
precede hidden attention layers. Each hidden update uses attention, LayerNorm,
GELU and feature `dropout`; `residual: true` returns `(hidden + update)/sqrt(2)`.
A final pointwise projection emits latent channels, raw biology parameters, or
batch logits. Missing positions remain zero. Residual toggles change neither
parameters nor reach. `attention_dropout` affects attention coefficients only;
the full-fit configuration uses 0.10 during training. During
training, each attention update uses activation checkpointing with preserved RNG:
backward recomputes edge activations while retaining all eligible neighbors and the
original optimizer batch. Evaluation does not checkpoint attention updates.
`model.gatv2_tile_budget: 8` bounds the number of independent tile graphs processed
per internal attention call, preventing a large supervised row batch from requiring
one enormous edge-activation tensor. This execution control retains every node,
neighbor and optimizer row, and is saved with the configuration. It does not change
reach, graph semantics, parameter count, or evaluation predictions. Changing this
budget can reorder dropout random draws during training; resume retains the saved
value. Each tile graph is built once per component forward and reused by its layers.
[PyG GATv2Conv](https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.nn.conv.GATv2Conv.html)
uses concatenated heads, edge dimension 3, separate source/receiver weights,
negative slope 0.2 and bias, with automatic self-loops and internal residuals disabled.

The objective remains `biology_weight * weighted_masked_ZILN + batch_weight *
class_weighted_cross_entropy`. There are no graph, smoothness, contrastive, or
attention regularization losses. Held-out spectra may supply context; only selected
training core labels contribute training losses. GATv2 performance improvements
require matched full-training experiments; smoke checks establish execution only.

In the matched three-epoch biology diagnosis, the plain biology CNN collapsed
pixel-to-pixel feature variation and failed all three seeds; residual connections
passed all three, including fresh fits with adversarial loss restored. This
evidence is specific to biology; aggregation and discriminator residuals have
not been evaluated in that experiment. Historical plain-CNN diagnostic controls
explicitly disable residuals to retain their original comparisons.

All-MLP configurations retain the sparse pixel loader and `training.pixel` settings.
Any CNN or GATv2 selects `training.spatial`. The checked-in configuration now uses
`sampling_strategy: proportional_slide_tiles`, 32×32 cores, and
`supervised_rows_per_batch: 32768`. Every selected training row appears once per
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
Smoke mode caps split membership, evaluation tiles, and the training row budget
(`training.smoke_supervised_rows_per_batch: 64`) while preserving the
architecture-derived halo. Capped labels are scattered across many footprints,
so the smaller smoke batch avoids excessive spatial activation memory.
The activated run name is `core32_attention010_radius15`, separate from the
baseline and pilot. Aggregation, biology, H&E and discriminator all use GATv2
with radius 1.5 and attention dropout 0.10. The full fit uses starting learning rate
0.0001, one physical 32,768-row batch (`microbatch_rows: null`), and eight physical
validation tiles. It starts fresh with the seeded 90/10 split, at most 300 epochs
and supervised-objective early stopping. `grl_horizon_updates: null` retains the
full epoch-based DANN schedule (14,100 updates at 47 updates/epoch); the accelerated
500-update pilot ramp is not used. Configured post-training analysis and tissue
inference run after fitting when invoking `python -m dann.train --config dann/config.yaml`.

`training.spatial.microbatch_rows` optionally subdivides a proportional sampler
request **before** spectra gathering. `null` (the default) retains a single
forward/backward pass; `supervised_rows_per_batch` always remains the effective
optimizer batch. Every microbatch keeps its requested tiles' complete context.
Biology means use the effective batch's valid target-weight mass, weighted
discriminator cross-entropy uses its class-weight mass, and H&E MSE uses its
feature-element count. ZILN sum reduction remains additive. Parameters and GRL
strength stay fixed until the group finishes; clipping and AdamW occur once,
including for a partial final batch. Training logs describe logical updates.
With dropout enabled the stochastic draws differ from a monolithic batch.

Production defaults to its epoch-based GRL horizon:
`completed_updates / (epochs * logical_updates_per_epoch - 1)` (with the
existing denominator guard). An explicit `training.grl_horizon_updates` uses
`completed_updates / (grl_horizon_updates - 1)` instead. Checkpoints record the
horizon, physical microbatch setting, logical updates per epoch, and cumulative
optimizer-update/row counters in `optimization` and `training_progress`.
`training.early_stopping_enabled: false` disables diagnostic early stopping.
Dropout, AdamW, clipping, loss weights, peak budget and checkpointing are unchanged.
New checkpoints have a separate versioned `sampling_contract` and epoch-boundary
`sampler_state`. Training resume keeps the checkpoint sampling contract, so a YAML change to
strategy, algorithm, seed, core size, training batch budget, or selected-row
identity is ignored. The contract check still rejects a checkpoint whose saved
sampler state is inconsistent. Old checkpoints imply legacy sampling using their
saved configuration. Inference and
architecture checks do not depend on sampler compatibility. The production
`train_model` entry point resumes completed epochs only; partial pilot checkpoints
also carry the update position and RNG needed by update-oriented callers.

Training writes `model/sampling/epoch_NNNN_batches.jsonl` with supervised rows,
unique tiles, distinct slides, slide entropy (natural logarithm), largest slide
fraction, cumulative rows and epoch position. The corresponding summary records
composition ranges/means and SHA-256 hashes of actual row order and cumulative
boundaries (little-endian int64, including the initial zero boundary).

The bounded core/batch pilot uses fresh identical all-GATv2 weights, the saved
90/10 split, radius 1.5 throughout, learning rate 0.0001, and a shared 500-update
GRL horizon. Its reference uses core 8 / 2,048 rows / attention dropout 0; its
candidate uses core 32 / 32,768 rows / attention dropout 0.10, initially in 4,096-row
microbatches (falling back to 2,048 then 1,024 on memory failures). Both validate
with eight physical tiles and retain the other configured architecture/loss
settings. It compares the combined changes, not an isolated dropout effect.

```bash
python -m dann.verification.core_batch_pilot \
  --split-checkpoint data/PDAC/Results/dann/deep_sets__agg-gatv2__bio-gatv2__disc-gatv2__he-gatv2/he_mse_baseline/model/latest.pt \
  --output data/PDAC/Results/dann/core32-pilot \
  --minutes 60 --updates 500
```

Use a fresh output directory. The runner alternates 50-update blocks, validates
initially and after every completed pair, and reserves measured validation time
before starting another pair. `results.json` identifies the largest completed
paired checkpoint; unmatched work is excluded from comparisons. Separate arm
directories contain resolved configs, update composition/row hashes, checkpoints,
and held-out diagnostics. `comparison.csv` reports learning by updates, rows and
elapsed time. Automatic full-tissue inference and plotting are disabled.

To compare physical accumulation while holding the candidate's effective batch
at 32,768 rows, use its untrained `candidate/model/update_0000.pt` checkpoint:

```bash
python -m dann.verification.accumulation_benchmark \
  --checkpoint /path/to/candidate/model/update_0000.pt \
  --output data/PDAC/Results/dann/accumulation-benchmark \
  --minutes 30 --repeats 3 --warmup 5
```

The benchmark tests 1/2/4/8 accumulation steps (32,768/16,384/8,192/4,096 physical
rows). Core 32, radius 1.5, attention dropout 0.10, spectral/GAT execution budgets,
optimizer settings, saved split, initial weights, and update-based GRL schedule
stay fixed. Every repetition uses a fresh process and five warmup updates, then
restores weights, AdamW, gradients, random streams and counters before one timed
full epoch. Repetitions use sampler epochs 0/1/2, paired across settings. Timing
includes CPU preparation, transfers, training and the final partial update;
startup, report writing, validation and checkpointing are excluded.

`results.json`, `comparison.csv` and `report.txt` retain isolated process timings,
warmup and training memory peaks, actual supervision hashes, optimizer counters,
and failures. CUDA allocation failures are distinct from exceeding the configured
24 GiB GPU budget. A failing setting never silently changes its microbatch size.
All repetitions must complete within budget to qualify; within 3% of the fastest
median epoch, the recommendation prefers lower allocated GPU memory. These are
observed memory and throughput results, not convergence or biological-quality
comparisons. Existing production configurations and checkpoints are not modified.
Latent effective rank uses a fixed row-ID subsample (row ID modulo 23); variance
and within/between-slide decomposition use all validation rows. CD8 diagnostics
include presence/exceedance Brier scores, presence calibration bins and
positive-branch PIT/interval coverage. These are pilot diagnostics, not evidence
of improved biological interpretation or H&E-mediated transfer.

The preceding diagnostic supports this sampling correction for spatial MLP and
discriminator-only CNN controls. It does not confirm convergence of the active
GATv2 aggregation/CNN biology/CNN discriminator architecture. Global pixel
shuffling performed better in that study.

The halo is derived as `aggregation radius + max(biology radius, discriminator
radius, enabled H&E radius)`. An MLP has reach zero; a CNN has reach equal to its depth. A GATv2
component has reach `depth * maximum_step`, where `maximum_step` is the largest
absolute x/y offset among integer offsets within its Euclidean cutoff. The same
offset definition builds edges and calculates reach. Aggregation reach is the
latent radius; aggregation plus biology reach is the biology radius. Defaults
use halo six and 20×20 input tiles for 8×8 cores. The enclosing biology window
is 13×13 and latent window is 7×7. Spatial heads run before core rows are selected,
retaining their complete latent halo.

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

Fresh runs use a seeded within-slide 90% training / 10% validation split. Training
gets `floor(0.9 × slide_pixels)` rows per slide and validation gets every remaining
row, including rounding remainders. The test array remains present but empty. For
the current dataset this gives 1,528,567 training and 169,861 validation pixels
(including the 37 rounding leftovers), with zero test pixels. A positive
`test_fraction` retains the original rounding rule: floor training and validation
counts, then assign the remainder to test. Loading or resuming historical
checkpoints preserves their saved split membership even if configured fractions
have changed. H&E-mediated transfer is assessed separately.

Checkpoint selection and early stopping both use **validation batch-aware objective**
with `training.early_stopping_metric: batch_aware_objective` in the checked-in YAML.
The optimized loss is unchanged:

```text
IHC_weighted = biology_weight * IHC_loss
HE_weighted = he_weight * he_mse                 # omitted when disabled
Batch_weighted = batch_weight * batch_cross_entropy
Supervised = IHC_weighted + HE_weighted          # LR scheduling score
Total = Supervised + Batch_weighted             # backpropagation objective
Batch_chance = ln(K)                            # discriminator output class count
Batch_selection_penalty = grl_max_lambda * batch_weight * max(0, Batch_chance - batch_cross_entropy)
Batch_aware = Supervised + Batch_selection_penalty  # stopping / best.pt score
```

The score uses finalized epoch losses and the discriminator's full output width
for `K`, including classes absent from a particular minibatch or validation split.
The fixed coefficient uses the configured maximum GRL strength, not the current
schedule value. With the current weights and 43 classes, the penalty is
`0.25 * max(0, 3.7612 - batch_cross_entropy)`. It decreases as the discriminator
approaches chance cross-entropy and remains zero above chance. Both `batch_weight`
and `grl_max_lambda` must be finite and nonnegative. Selection starts at the first
completed validation epoch, with no additional warmup.

`supervised_objective` and `cd8_loss` remain available for comparison. An absent
setting retains legacy `cd8_loss` selection: mean validation CD8 ZILN loss over
valid CD8 rows. `early_stopping_min_delta` measures units of the selected score.
The selection policy does not change the optimized loss, gradients, GRL schedule,
or LR scheduling on `supervised_objective`.

New runs enable validation-driven LR reduction through the shared block:

```yaml
training:
  lr_scheduler:
    enabled: true
    factor: 0.5
    patience: 5
    cooldown: 2
    threshold: 0.001
    min_lr_factor: 0.01
    eps: 0.0
  early_stopping_patience: 20
```

`ReduceLROnPlateau` minimizes validation `supervised_objective`, regardless of the
early-stopping metric. Improvement must exceed the absolute `threshold`; after
six non-improving epochs (`patience: 5`) the LR halves, followed by two cooldown
epochs. The scheduler steps once after completed validation, before checkpoint
snapshots. Nonfinite monitored scores are rejected. LR reductions do not reset
early-stopping patience or change GRL/optimizer-update accounting.

The floor is computed once as `min_lr_factor ×` the selected execution mode's
starting LR. With the checked-in spatial LR `0.0001` the floor is `0.000001`;
pixel LR `0.001` gives `0.00001`. A spatial starting LR of `0.000001` gives a
`0.00000001` floor. `eps: 0.0` permits reductions to these small floors.
Omitting `lr_scheduler` or setting `enabled: false` keeps a fixed LR. Unknown
settings, nonfinite numbers, invalid factors, and negative thresholds, epsilon,
patience, or cooldown are rejected; patience and cooldown must be integers.

Checkpoints save the exact split positions plus an ordered row/feature identity
fingerprint; analysis reuses those splits and may cap them further. There is no
refit using validation or test labels. Version-five architecture metadata records
selected components, numerical settings, receptive fields, batch order, targets,
and optimizer parameter names. Legacy flat configurations mean Deep Sets with
MLP aggregation and heads. Legacy parameter-key conversion is identity because
keys and registration order were retained; optimizer ordering and state shapes
are checked before resume. Incompatible components or targets are rejected.
Checkpoints explicitly record `selection_metric`, `best_selection_score`,
`best_epoch` (one-based), `selection_start_epoch` (one-based), and
`early_stopping_state` (stale epochs, patience, minimum improvement, stopped flag).
Selection scores are never written under CD8-labelled fields. Compatible resumes
restore the score, selected epoch, and stale-epoch count even without `history.csv`;
recognized old CD8 checkpoints can replay their saved CD8 history for patience.
Changing the metric resets the score and patience at the next epoch, records that
selection boundary, and selects only newly evaluated epochs. Set `training.epochs`
beyond the resumed checkpoint epoch; a metric switch with no additional epoch is
rejected. Unknown legacy selection rules also start a new selection period.

Training resume copies the checkpoint configuration, then applies only the current
YAML values for `epochs`, `num_workers`, `prefetch_factor`,
`early_stopping_metric`, `early_stopping_patience`, and `early_stopping_min_delta`,
plus the documented performance, post-training, and monitoring overrides. The checkpoint learning
rate and other training settings stay in place; differing YAML values are reported
and ignored. Batch-aware selection uses the saved loss weights and maximum GRL
strength, with the saved class count enforced by the checkpoint's batch order
and discriminator state. `--smoke-test` cannot be combined with a resume checkpoint. Completed
`history.csv` rows are kept and both `loss_curves.png` and `ihc_curves.png` are refreshed from that full history.

Scheduled checkpoints save `lr_scheduler_state` with the optimizer, including the
best supervised score, counters, cooldown, and original LR floor. Resume restores
both states without stepping the scheduler or deriving a new floor from the
reduced optimizer LR. Saved scheduler settings take precedence over conflicting
YAML settings, which are reported and ignored. Legacy checkpoints without a
scheduler block remain fixed-LR even when current YAML enables scheduling.
Production resume rejects checkpoints that enable scheduling but lack its state.
Changing the early-stopping metric resets only selection state; scheduler state
continues because its monitored score remains `supervised_objective`.

`performance.async_artifacts: true` moves checkpoint serialization, history and
sampling-audit writes, and curve rendering into two persistent CPU workers. One
worker writes files in order; the other renders immutable history snapshots. The
next epoch can start while either worker is busy. Validation and stopping decisions
remain synchronous, as does copying a consistent model/optimizer/scheduler snapshot to CPU
before training mutates it. CPU tensors are also copied, and CUDA transfers finish
before a snapshot is handed to a worker. This covers MLP, CNN, GATv2, their mixed
combinations, and optional H&E heads. Dataset identity and checkpoint contracts are
computed once per run. One snapshot serves both best/latest saves when applicable.

If outputs fall behind, pending plots and superseded saves for each of `best.pt`
and `latest.pt` are replaced by the newest request for that destination. A latest
save never discards a pending best save. Every history row and sampling audit is
retained. Queues account for snapshots, worker memory, and rendering buffers against
`performance.host_budget_gib`; training waits only when required records or a new
snapshot cannot otherwise fit. Files are published with atomic replacements, so
readers see complete files. Outputs can lag the running epoch until the workers
catch up.

Workers are drained on completion, early stopping, and graceful interruption before
post-training analysis can consume the final checkpoint. Worker errors surface in
training; a partially completed epoch is never saved as completed. An abrupt process
or machine failure can lose pending outputs, leaving the last published files.
Configurations without `async_artifacts` retain synchronous output, and setting it
to `false` provides the comparison path. This execution-only setting can change on
resume without changing model or optimizer semantics. `model/artifact_timings.json`
records snapshot and epoch-boundary times, worker jobs, replaced requests, worker
startup, and final drain time. A matched three-epoch capped GPU comparison is:

```bash
python -m dann.verification.artifact_benchmark --root data/PDAC/Results/dann/artifact-check --deterministic
```

The result root must be new. The command checks all-MLP, all-CNN, and all-GATv2
runs with synchronous and asynchronous output, comparing numerical states,
histories, sampling audits, and objective identities. `--deterministic` enables
deterministic GPU kernels only in these comparison runs and requires exact state
and metric equality; production training settings are unchanged.

History retains raw loss and fit columns and adds `biology_weighted_loss`,
`hurdle_weighted_loss`, `positive_weighted_loss`, `batch_weighted_loss`,
`supervised_objective`, `batch_chance_loss`, `batch_selection_penalty`, and
`batch_aware_objective`, each prefixed by `train_` or `validation_`. Per-signal
`hurdle_weighted_loss_<signal>` and `positive_weighted_loss_<signal>` include both
the signal's target weight and `biology_weight`. Their sums equal the aggregate
IHC objective. For mean reduction, every signal and both branches share the total
valid target-weight mass denominator, accumulated across the epoch. Positive
contributions are not divided by positive counts. Sum reduction retains the
existing sample-weighted average of batch sums for aggregate and per-signal reports.
`selected_epoch` and `selection_start_epoch` preserve selection metadata in history.
`learning_rate` records the LR used during the completed epoch, and
`next_learning_rate` records the LR after validation scheduling (also on the final
epoch). Both appear in epoch logs and are equal for fixed-LR runs. Older history
rows without these columns remain blank when resumed; no past LR is inferred.

The four-column overall plot pairs IHC balanced hurdle accuracy, IHC positive-logit
R², H&E R², and discriminator accuracy with their objectives directly below.
The next row shows total objective, supervised objective, actual GRL strength,
and batch-aware objective. The active selection metric is labelled in the figure
header, and the supervised or batch-aware panel is marked when selected.
Epoch summaries also report the new score.
Historical rows without batch-aware metrics remain gaps, even if raw batch losses
are available. The separate IHC figure places each signal's fit metrics above its
weighted hurdle and positive likelihood objectives. The combined IHC objective
remains in history and the additive formulas. Disabled H&E panels are hidden. One shared legend
identifies Train, Validation, and the selected checkpoint epoch across all panels.
Epoch axes are aligned, with numeric ticks and no repeated "Epoch" axis labels;
no minimum is inferred from incomplete history.

`train_he_r2` and `validation_he_r2` are variance-weighted feature R² values:
`1 - sum((target - prediction)**2) / sum((target - feature_mean)**2)`, where each
feature mean is computed over the entire respective epoch. Float64 per-feature
means and centered sums of squares are merged across batches, including the
variation between batch means. No target or prediction matrix is retained.
Fewer than two observations or zero total feature variance yields an unavailable
value. R² is higher-is-better and may be negative; it does not change the MSE
objective, gradients, or checkpoint selection. Historical MSE alone cannot
reconstruct H&E R², so those entries remain unavailable.

IHC positive-logit R² measures the positive mean, while its likelihood objective
depends on both location and scale and may also be negative. Additive formulas
clarify how the detailed panels decompose IHC. Both analysis paths use the
checkpoint's saved weights, target order, and selection metadata. Historical
aggregate losses can be weighted from saved settings; unavailable per-signal
losses remain gaps or unavailable panels.
This applies to future training and requested analysis; completed-run figures are
not regenerated automatically.

The architecture contract includes the residual setting for every selected CNN.
Version-three checkpoints retain their recorded biology setting and default missing
aggregation/discriminator flags to false. Version-two and unversioned checkpoints
default all missing residual flags to false. This preserves their original
calculations during analysis and inference. Training resume keeps checkpoint
residual settings, so different YAML values are ignored. Version-four checkpoints
require explicit boolean residual settings for selected CNNs in both saved
configuration and architecture metadata, and those settings must agree.

Version-five GATv2 checkpoints require all selected settings, versioned fixed graph
semantics, and derived reach to agree with the saved configuration. Versions 2–4
and unversioned payloads cannot select GATv2. Old MLP/CNN metadata is normalized
in memory without rewriting historical checkpoints. PyTorch and PyG versions are
informational reproducibility metadata, separate from architecture compatibility.
Switching a component from CNN to GATv2 requires fresh weights and optimizer state.
Resume retains saved architecture and the existing runtime override whitelist.

All prediction consumers should use `model(batch)`. Calling `encode()` and then
the biology head loses the spatial halo when a CNN or GATv2 head is selected. Peak-embedding
analysis remains available for every combination through the Deep Sets dictionary.

## Result routing

`results.root`, the four selectors, and `results.run_name` (checked-in `residual-gatv2-aggregation`) derive:

```text
data/PDAC/Results/dann/<combination>/<run_name>/
    model/
    analysis/
    analysis/spatial/
    analysis/latent_variance/
```

The default run is
`deep_sets__agg-gatv2__bio-cnn__disc-cnn/residual-gatv2-aggregation/`. Smoke artifacts
insert `smoke/` after the run name. Training, analysis, inference, heatmaps, and
latent variance use this root; no separate architecture YAML files are required.
For a smoke checkpoint, pass its `model/resolved_config.yaml` to downstream CLIs.
Checkpoint overrides use saved architecture and run routing.

Spatial exports preserve existing schemas and original tissue row order. Spatial
inference stages tile-order latents/parameters in temporary disk arrays, then
streams them in row order. Temporary staging is removed on success or failure;
allow roughly four bytes per row per exported latent/parameter channel on disk.
`--max-rows` limits exported cores while retaining the full tissue context.
`spatial_predictions.support.parquet` reports occupied-neighborhood fractions for
biology and latent enclosing square windows. These are geometric occupancy
diagnostics, not exact graph reachability or attention mass. Boundary/interior
analysis uses this same square-window definition, including for GATv2.
`spatial_predictions.provenance.json` records graph settings and semantics,
support semantics, checkpoint/input paths, context/output counts, and zero edge
exclusions. Prediction and latent Parquet columns remain unchanged.

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
`<run-root>/analysis/latent_variance/`:

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

`analysis.split` and `analysis.ziln_scatter_split` default to `validation`.
`analysis.max_samples: null` analyzes all rows in the selected saved splits.
A positive integer selects a stratified cap.

### Analysis execution and reuse

`analysis.execution.enabled: true` enables the optimized `python -m dann.analyze`
pipeline. Missing or disabled sections retain the legacy execution path. Model
architecture, target order, preprocessing and spatial context remain checkpoint
controlled. Analysis reads the labeled `data.path`, including its full lesion
context around held-out rows; it does not reuse tissue-wide spatial predictions.
With the default validation-based analysis, UMAP and prediction scatter use the
saved validation rows: 169,861 for a fresh 90:10 split, or 169,824 for the
historical 80:10:10 checkpoint. Historical test rows remain saved and can still
be selected explicitly. Peak activity for peptide selection uses all
1,698,428 labeled lesion rows, including training rows. Splits are within slides,
not held-out patients; analysis does not add a new `Lesion` filter.

The execution profile uses resident spectra, a shared evaluation-only spectral
feature cache, independent tile/spectral/GAT budgets, and spawned figure/table
workers. `analysis.execution.performance` overlays the current `performance`
controls for this invocation without changing training settings. The default
RAM budget is 192 GiB and the allocated VRAM budget is 24 GiB. Requested caches
that exceed their budget fail explicitly. Worker memory and pending buffers are
included in budget checks. Pixel mode and training-split analysis keep the legacy
model extraction behavior.

Numerical stage caches live under `<analysis-output>/.analysis-cache/`. Changing
plot colors, DPI or point size reuses predictions, sampled means and UMAP
coordinates. Changing the checkpoint, input file, row selection, relevant
execution settings or numerical software versions invalidates affected stages.
Arrays and RNG metadata have checksums; incomplete or corrupt entries are
recomputed. The source identity includes file size, nanosecond modification/change
times and row/feature fingerprints, rather than a full hash of the large CSR file.
The cache is regenerable and can be removed between invocations to reclaim disk.

```bash
# Redirect all artifacts, including loss_curves.png; production outputs stay intact.
python -m dann.analyze --config dann/config.yaml --output-dir data/analysis-review
# Recompute numerical stages, or disable persistent cache reads and writes.
python -m dann.analyze --config dann/config.yaml --cache-policy refresh
python -m dann.analyze --config dann/config.yaml --cache-policy off
```

`run_analysis(config, checkpoint_override, *, execution_overrides=None,
output_override=None)` preserves existing two-argument Python callers. Explicit
execution overrides take precedence over current YAML controls. Saved training
execution annotations are not inference settings. Validated temporary outputs
replace only their own destination files; unrelated spatial artifacts are retained.

The seeded UMAP algorithm is unchanged and fits once per latent input. Density
sampling preserves the original shared stream: main split first, then the scatter
split, including on partial cache hits. Identical parameters give bitwise-identical
float64 means across worker counts. The default remains 1,000 draws. GPU batching
can change predictions within FP32 tolerance and therefore change the seeded UMAP
layout; small `pi` changes can also flip rare Monte Carlo hurdles. CSV blocks retain
the original precision, column order and target-major scatter ordering.

`analysis_execution.json` records source/checkpoint identities, effective settings,
cache generations, preparation waits, phase times, total runtime through output
validation, peak allocated/reserved VRAM and sampled parent-plus-worker RSS.
Five-second heartbeats continue during UMAP and worker execution. RSS includes
shared mapped pages in each process, so it is a conservative memory measure.

Stage screens use a warmup and three timed repetitions. After generating an
isolated optimized analysis directory, reproduce them with:

```bash
python -m dann.verification.analysis_benchmark --stage gpu --output data/analysis-bench/gpu
python -m dann.verification.analysis_benchmark --stage sampling --analysis-dir data/analysis-review --output data/analysis-bench/sampling
python -m dann.verification.analysis_benchmark --stage rendering --analysis-dir data/analysis-review --output data/analysis-bench/rendering
python -m dann.verification.analysis_benchmark --stage compare --reference data/analysis-legacy --analysis-dir data/analysis-review --output data/analysis-bench/comparison
python -m dann.verification.analysis_benchmark --stage audit-density --reference data/analysis-legacy --analysis-dir data/analysis-review --output data/analysis-bench/comparison
python -m dann.verification.analysis_benchmark --stage verify-umap --analysis-dir data/analysis-review --output data/analysis-bench/comparison
```

Run screens separately with other GPU/CPU workloads idle. The density audit replays
the unchanged sampler against both versions' exported parameters, explicitly
checking discrete hurdle differences rather than silently relaxing their tolerance.

The [recorded measurements](verification/analysis_benchmark_results.json) use the
full lesion dataset, checkpoint epoch 47, 1,000 draws, 300-DPI figures, an RTX 5090,
and 16 physical / 32 logical CPUs with 249 GiB usable RAM. Complete CLI timings
include imports, startup, every export and final validation:

| Run | Wall time | Peak parent + worker RSS | Peak allocated VRAM |
| --- | ---: | ---: | ---: |
| Pre-change legacy | 411.8 s | 18.7 GiB | 0.93 GiB |
| Optimized, numerical caches rebuilt | 192.5 s | 23.5 GiB | 2.28 GiB |
| Cached reruns, median of three | 37.4 s | 19.8 GiB | 0.00 GiB |

This is a 2.14× fresh-run and 11.0× cached speedup. Cached repetitions took 37.41,
37.36 and 37.82 seconds. Filesystem caches were warm for these comparisons. The
selected profile uses 64 tiles, 262,144 peaks, 32 GAT tiles, 1,024 feature-cache
rows, two preparation threads, one sampling worker and 16 rendering workers.
Rendering/table screens with 4/8/16/32 workers took median 31.60/22.98/18.21/22.65
seconds; extra workers did not improve the selected profile.

All 750 repository tests passed. All 37 scientific artifacts retained their schemas,
identities, ordering and support; peptide families and raw activity were exact.
Predicted parameters and latents passed `rtol=5e-5, atol=5e-5`. Of 1,188,808 exported
sampled means, 13 differed beyond that tolerance after FP32 parameter changes
(maximum absolute difference 0.000661). Replaying the unchanged sampler reproduced
both versions' means bitwise, with identical final RNG states. Refitting the full
169,878-row frozen latent input also reproduced its seeded UMAP coordinates bitwise.

The largest remaining cold-run cost is seeded UMAP (93.4 seconds), followed by
geometry preparation (20.3 seconds). Cached runs primarily spend time on figure/
table workers and Python/library startup. The pre-change source/config snapshot
and isolated full outputs are preserved under `data/verification/analysis_optimization/`.

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

## Automatic post-training workflow

`python -m dann.train --config dann/config.yaml` runs the enabled downstream
stages after successful training and resource cleanup, using the **best validation
checkpoint returned by training** (including resumes and early stopping). The
order is analysis → spatial inference → spatial heatmaps → latent variance.
Programmatic `train_model(config)` remains training-only. The reusable entry point
is `dann.workflow.run_post_training_workflow(config, checkpoint_path)`; it returns
a mapping of completed output names to `Path` locations and prints those locations.

```yaml
post_training:
  enabled: true
  run_analysis: true
  run_inference: true
  run_spatial_heatmaps: true
  run_latent_variance: true
  overwrite_inference: true
```

Omitted fields default to `true`, including when the entire section is absent.
All values must be YAML booleans; unknown keys are rejected. Set `enabled: false`
to disable the workflow, or disable individual stages with their `run_*` switch.
`--smoke-test` always sets `post_training.enabled: false`.

Analysis runs once. Peak analysis, calibration, density diagnostics, and the
interactive latent reports remain within that stage under their existing
`analysis` controls, including `analysis.peaks.enabled` and
`analysis.latent_interactive`. Numerical and browser audits remain manual.
Each stage retains its configured execution and memory budgets.

Workflow inference derives its execution mode from the checkpoint architecture.
`analysis.batch_size` counts rows for pixel models. Spatial models use
`analysis.spatial_execution.tiles_per_batch` (currently 64) when optimized
inference is enabled, or the checkpoint's saved `training.tiles_per_batch`
(default 8) for legacy inference. A current YAML architecture that differs from
the checkpoint does not change these batch units. Explicit standalone CLI
`--batch-size` and `--tiles-per-batch` overrides retain their documented precedence.

The current YAML supplies downstream switches, report settings, execution
settings, and density sampling settings. The checkpoint supplies frozen model,
preprocessing, targets, training data, and splits. Inference uses the runtime
`data.context_path` tissue input and the configured `spatial.output` and
`spatial.latents` destinations. Resolved paths within the requested run tree are
rebased into the actual returned checkpoint's run tree, so a resume with a changed
run name still writes beside its original model. Explicit destinations outside
the run tree remain explicit. Latent variance consumes that same spatial latent
export and runtime tissue input; its standalone CLI delegates to
`dann.latent_variance.run_latent_variance(config)`.

With `run_inference: false`, enabled heatmaps and latent variance consume existing
exports. They require complete tissue coverage, matching schemas and row order,
and the prediction-adjacent `.provenance.json` sidecar matching the checkpoint and
tissue file identities. The sidecar also binds the required prediction/latent
file identities; prediction reuse checks target semantics and current density
sampling settings. Heatmaps require only predictions; latent variance requires
only latents plus the provenance sidecar. Missing, changed, partial, or older
exports without these identities fail with instructions to regenerate them.
Inference is never implicitly enabled. With inference enabled,
`overwrite_inference: false` preserves its refusal to replace existing artifacts;
the default `true` permits replacement after successful inference validation.

The first failed stage stops the workflow, identifies the stage, and gives the
training CLI a nonzero exit status. Completed training artifacts and completed
downstream outputs remain available. Failed training never starts downstream
work. This workflow does not change the independent IHC-MVN pipeline.

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
# Use the saved smoke configuration for analysis and capped tissue inference.
smoke_config=data/PDAC/Results/dann/deep_sets__agg-gatv2__bio-cnn__disc-cnn/residual-gatv2-aggregation/smoke/model/resolved_config.yaml
python -m dann.analyze --config "$smoke_config"
python -m dann.spatial --config "$smoke_config" --max-rows 512
```

Tissue-wide spatial inference writes ordered ZILN predictions and the complete
latent representation for every AnnData row. Heatmaps join the predictions back
onto the tissue AnnData and plot per-batch logit / mean / presence / sigma
panels plus HES.

Run both synthetic suites:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q dann/tests ihc_mvn/tests
```

GATv2 implementation validation (2026-10-08): **627 synthetic tests passed**,
including all 27 architectures, legacy checkpoints, halo equivalence, optimizer
reload, and the complete synthetic downstream workflow. On an RTX 5090 with
PyTorch 2.11.0+cu128 and PyG 2.8.0.post1, the configured one-epoch smoke trained
2,048 core rows across 1,997 tiles and 43 slides without reducing neighbor coverage
or optimizer batch size. Training took 174.2 s with 25.38 GiB peak allocated and
27.53 GiB peak reserved GPU memory. Train/validation total losses were finite
(54.725424 / 17.685151); validation CD8 loss was 2.434584. All 80 optimizer
parameter states reloaded. Saved-config analysis passed in 105.6 s, and capped
512-row tissue inference passed in 19.0 s with 36 prediction and 256 latent channels.

The initial unbounded graph execution exceeded GPU memory; attention recomputation
plus the internal tile budget above resolved it. Logs, resource measurements, and
`validation_summary.json` live in the smoke run directory. Full latent-variance
and all eight heatmap outputs were verified on the aligned 156-row synthetic
input. These checks establish execution and compatibility, not predictive gains.


Within the resolved run root, training artifacts go under `model/` and analysis
under `analysis/` (UMAP exports in `analysis/umap/`):

`model/`:

- `best.pt`, `latest.pt`
- `history.csv` (resume keeps completed epochs and appends new ones), `data_schema.json`, `resolved_config.yaml`
- `loss_curves.png` (overall objectives, shared gradients, and validation latent diversity)
- `ihc_curves.png` (per-target fit, weighted losses, and target/branch gradient composition; both figures regenerate from history during training and analysis)

`analysis/umap/`:

- `overview/latent_umap_batch.png`
- `overview/latent_umap_densities.png`
- `overview/latent_umap_densities_logit.png`
- `overview/latent_umap_density_means.png` (logit of sampled density means for every target)
- `parameters/latent_umap_ziln_Density_*.png` (top: logit/sampled-mean/batch; bottom: 1-pi/mu/sigma)
- `diagnostics/latent_umap_ziln_diagnostics_Density_*.png` (one 2x3 Logit/logit_mean/presence/residual/interval-width/batch figure per target, masked to observed positives)
- `tables/latent_umap.csv`

`analysis/` (non-UMAP):

- `density/ziln_density_scatter.png`
- `tables/latent_embeddings.csv`
- `tables/ziln_density_scatter.csv`
- `similarity/peptide_families.csv`
- `similarity/embedding_cosine_similarity.npy`

`analysis/calibration/` (test metrics; validation reports in `validation/`):

- `calibration_metrics.csv`, `calibration_metrics_by_batch.csv`
- `calibration_pit.png` (PIT histogram and uniform QQ per target)
- `calibration_hurdle_reliability.png`, `calibration_interval_coverage.png`

`analysis/spatial/`:

- `spatial_predictions.parquet` (ordered mu/presence/sigma plus the derived
  logit_mean/logit_sd/logit_median/logit_q05/logit_q95 summaries and density_mean)
- `spatial_latent.parquet` (ordered full latent vectors)
- `heatmaps/{tissue,lesion}/spatial_heatmap_Density_*_{tissue,lesion}.png` (6-column parameter/sampled-mean/HES figures)
- `heatmaps/{tissue,lesion}/spatial_heatmap_diagnostics_Density_*_{tissue,lesion}.png` (5-column diagnostic figures)

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

## Frozen biology neighborhood benefit

Evaluate two saved validation checkpoints without retraining:

```bash
python -m dann.verification.neighborhood_benefit \
  --cnn-checkpoint <cnn-run>/model/best.pt \
  --mlp-checkpoint <mlp-run>/model/best.pt \
  --output data/PDAC/Results/dann_neighborhood_benefit/<new-run>
```

The runner requires pointwise aggregation in both models, a depth-three biology
CNN, identical saved splits/input identity/target transformations, and 169,824
validation rows by default. It uses the legacy-compatible inference loader with
restricted checkpoint deserialization and preserves unknown training provenance.
Only validation labels are scored; labeled-grid spectra supply frozen context.

It verifies original losses and patch/tile equivalence, then preserves each
center and occupancy mask while replacing neighbors with a complete distant
same-slide patch, rearranging occupied neighbors, or replicating the center.
The random interventions use seeds 20261003–20261007. MLP controls must remain
exactly invariant. Results include per-row parameters and intervention identities,
quadrature density means, paired slide-bootstrap loss intervals, per-slide and
boundary metrics, adjacent-pixel contrast summaries, effect plots, and a report.

GPU execution uses evaluation mode, float32, two CPU threads, no loader workers,
and a 3 GiB PyTorch allocator cap. It waits for 16 GiB free before starting, pauses
between chunks below 12 GiB, and waits for training to exit after an analysis OOM
before retrying. CUDA context/library overhead is outside the allocator cap.
Completed arrays and progress records are retained on failure; existing output
directories are refused. Neither production settings nor checkpoints are changed.

After all inference conditions complete, regenerate the CPU tables and report
with `--summarize-only --output <existing-run>` (checkpoint arguments omitted).
This verifies frozen checkpoint/data identities and saves a separate reporting
source snapshot; inference arrays and the original source snapshot are retained.

## Resident execution and validation caching

`performance` controls RAM residency, CPU preparation, CUDA prefetching, and
validation execution independently of the architecture. Old configurations that
omit this section retain backed loading. `ram_cache` reads CSR arrays once,
filters raw entries only when the existing threshold is positive, and applies
the configured transform and clip once. Stored zeros, peak order, duplicate
entries, empty spectra, and target masks are preserved. Training, validation,
and test share the same read-only arrays; the tissue inference file is never
used for supervised context.

With residency enabled, `preparation_workers` CPU **threads** replace DataLoader
process workers. `prepared_batches` bounds outstanding work; futures are consumed
in sampler order. `cache_geometry` shares occupied tile footprints and graphs
between splits and precomputes core positions and neighborhood support. Complete
halos and every eligible graph edge remain present. `torch_threads` optionally
bounds PyTorch's intra-operation CPU threads to avoid oversubscription.

Pixel datasets prepare complete batches through `SparseAnnDataDataset.get_batch`
when the prepared loader uses `sparse_collate`. Resident spectra use one ordered
CSR gather and H&E targets use one `HETargetReader.read` per nonempty batch.
Backed spectra sort/deduplicate source rows, read only contiguous requested runs
using one invocation-local read-only HDF5 handle, and restore request order and
duplicates. Ordinary DataLoaders use `__getitems__` with the same batch gathering,
returning NumPy/scalar sample dictionaries for existing and custom collators.
Scalar `__getitem__` remains available as a reference. Empty splits produce no
batches; explicit empty `get_batch` requests raise the collator's empty-sequence
error. No dense spectra or full-dataset H&E cache is constructed.

This is automatic for pixel training, validation, and testing, with unchanged
samplers, batch boundaries, preprocessing, targets, losses, and checkpoint schema.
Spatial tile/graph preparation is unchanged. Resident preprocessing errors also
identify the original source row.

Reproduce the pixel comparison in a fresh diagnostic directory on an idle GPU:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python -m dann.verification.pixel_loader_benchmark \
  --output data/PDAC/Results/dann/diagnostics/pixel-loader-comparison --full-epoch
```

The harness freezes **row-index requests**, including the full sampler order,
largest stored-entry footprint and actual final partial batch. Each pair uses
identical initial weights, requests, seeds, losses, and execution settings. Five
warm-up batches per phase precede 64 measured training and 24 validation batches;
three trials alternate implementation order. `--full-epoch` additionally checks
one complete resident training/validation epoch per implementation. Tensor
gathering, H&E reads, collation, transfers and model/optimizer execution all remain
inside the measurement. Scalar fetching is retained in a benchmark-only wrapper.
Resident runs use the configured two preparation threads and two queued batches;
backed runs use zero preparation threads, as required by the existing performance
validator, with the same 2,048/4,096 row batches. Neither comparison tunes settings.

Artifacts include source/config snapshots, row requests, initial/final model
states, per-phase preparation time, loader wait, throughput, sampled utilization,
and the existing host/CUDA memory measurements. Preparation time sums overlapping
thread durations; loader wait is time blocked awaiting prepared batches. GPU
utilization samples every 0.5 seconds are coarse for short validation passes.
Peak host RSS and reserved CUDA memory can retain earlier phases' high-water marks;
no allocator changes or cache flushing are performed. The `--deterministic` flag
supports a separate numerical audit with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; its timings should be kept separate from the
default training profile.

The 2026-10-19 comparison used the labeled file's 1,698,428 rows and 686,730,743
stored MSI entries, the configured H&E head, and an RTX 5090 with PyTorch
2.11.0+cu128. Median default-profile results over three pairs were:

| Storage / phase | Scalar seconds | Batched seconds | Speedup | Batched loader wait |
| --- | ---: | ---: | ---: | ---: |
| Resident, 64 training batches | 26.40 | 2.86 | 9.22× | 0.98% |
| Resident, 24 validation batches | 19.57 | 0.87 | 22.58× | 7.65% |
| Backed, 64 training batches | 41.12 | 5.50 | 7.47× | 73.88% |
| Backed, 24 validation batches | 30.57 | 3.56 | 8.59× | 79.71% |
| Resident, complete training epoch (one pair) | 305.46 | 32.57 | 9.38× | 0.12% |
| Resident, complete validation epoch (one pair) | 33.36 | 1.44 | 23.10× | 4.94% |

The complete training/validation pass fell from **338.8 to 34.0 seconds**. Median
training throughput rose from 4,966 to 45,780 rows/s resident and from 3,188 to
23,823 rows/s backed. Resident preparation summed 52.31 versus 1.04 CPU wall
seconds across threads for 64 batches; backed preparation fell from 39.68 to
4.06 seconds. Sampled training GPU utilization rose from 9.3% to 85.3% resident
and from 6.4% to 47.6% backed. Allocated CUDA peaks were essentially unchanged:
7.49 GiB for training, 1.12 GiB for validation, and 7.57 GiB for the full training
epoch. Memory optimization remains deferred.

All three resident training speedups were 9.0–9.3×; no reproducible throughput
regression above 5% occurred. The resident throughput and wait targets were met.
The backed path still spends most time preparing inputs with its unchanged
zero-thread setting. Frozen shuffled requests have medians of 2,046 distinct
contiguous CSR runs per training batch and 4,086 per validation batch, so bounded
HDF5 reads remain significant. Worker-count/batch-size tuning was not part of
this comparison. These are prepared-loader measurements with normal OS caching;
cold-disk and multi-process throughput were not measured. The timings include
model execution and are separate
from the earlier isolated 38× H&E-reading probe.

Correctness checks compare every field's value, dtype, shape and order to scalar
collation, including repeated rows, empty spectra, duplicate peaks, explicit
zeros, all transforms, filtering/clipping, masks and optional H&E. Tests cover
source-row errors, invocation-local handle cleanup, ordinary/persistent process
workers, custom collators, ordered preparation threads, unchanged RNG state,
budget checks, and seeded outputs, losses, gradients and an optimizer update.
Existing CUDA transfer/stream regressions are included in the DANN suite.
Final verification passed **962 tests** with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m pytest -q dann/tests`.

Default CUDA training is not bitwise reproducible: the short paired runs had
parameter differences up to 3.46e-5, and repeated scalar runs also differed. The
full default-profile pair accumulated a maximum parameter difference of 0.395
and a validation-loss difference of 0.146. A separate deterministic CUDA replay
matched **all final model parameters and training/validation losses exactly**
over the complete resident epoch (747 updates, 1,528,567 training rows and
169,861 validation rows), as well as the 64-update resident and backed checks.
This isolates the default-profile drift from input preparation. Determinism was
enabled only for this verification; production behavior and the timing table above retain
the existing CUDA settings.

Raw measurements, requests, snapshots and model states are under
`data/PDAC/Results/dann/diagnostics/2026-10-19-pixel-batches/`. The separate
deterministic numerical audit is under the same parent with the directory name
`2026-10-19-pixel-batches-deterministic/`.

`transfer_prefetch` lazily creates one dedicated CUDA stream per loader and device,
then reuses it across training, validation, or inference passes. Unindexed `cuda`
resolves to the current device, sharing its entry with the explicit device index.
The stream cache lives with the loader as runtime state; it is absent from model
weights, checkpoints, and configuration. PyTorch's native allocator reuses cached
blocks on their allocation stream, so keeping the stream avoids populating new
stream-specific caches each epoch. Reserved memory can still exceed live tensor
memory and vary with batch sizes; routine cache flushing is unnecessary.

There are at most two pinned staging batches. The compute stream waits on a
transfer event, device allocations are recorded on the consuming stream, and
source buffers stay alive through DMA.
Completion, failure, and explicit iterator closure synchronize outstanding
transfers and close the source iterator while retaining the stream for reuse.
CPU execution uses the same prepared interface without CUDA pinning. The transfer
path follows [PyTorch’s pinned-memory and stream guidance](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
and [native allocator stream ownership](https://github.com/pytorch/pytorch/blob/v2.11.0/c10/cuda/CUDACachingAllocator.cpp).

The 2026-10-19 CUDA regression repeats identical batches for 40 passes, both on
one loader and alternating two loaders. Live allocations return to baseline and
reserved memory stays stable after warm-up; reinstating stream creation per pass
fails this check (40 to 1,280 MiB reserved for the synthetic workload). Tests also
cover device aliases, empty loaders, early closure, source/transfer/consumer
exceptions, subsequent reuse, and CNN/GATv2 output and gradient equivalence.

A capped real-data replay on an RTX 5090 with PyTorch 2.11.0+cu128 and the native
allocator used three fixed training batches of 2,048 rows and three validation
batches of 4,096 rows per cycle, including the largest stored-entry footprint
from each capped candidate set (up to 1.68 million peaks in a validation batch).
Across 40 alternating passes, persistent streams stabilized at 8.71 GiB reserved
after four passes. Median throughput over three independent runs, excluding the
first two cycles, was:

| Transfer stream lifetime | Peak reserved GiB | Peak allocated GiB (train / validation) | Training rows/s | Validation rows/s |
| --- | ---: | ---: | ---: | ---: |
| New stream each pass | 12.79 | 7.60 / 1.12 | 44,447 | 134,574 |
| Reused per loader/device | 8.71 | 7.60 / 1.12 | 44,828 | 137,848 |

This replay reduced reserved memory by 31.9%, with no throughput regression;
live allocations after each pass were 0.041 GiB for both prefetch variants.
A separate deterministic real-data check matched transferred values, model
outputs, and training gradients exactly against disabled prefetch. Default
nondeterministic training produced small final-weight differences after 60
updates, also observed between unchanged repeats. These bounded replays exclude
CPU preparation timing and do not establish full-epoch memory or convergence.
Inputs, scripts, per-pass metrics, and numerical comparisons are saved under
`data/PDAC/Results/dann/transfer-stream-reuse-2026-10-19/`.

`model.spectral_peak_budget` and `model.gatv2_tile_budget` tune execution without
changing spatial dimensions or optimizer rows. `spectral_checkpointing` and
`gatv2_checkpointing` independently control activation recomputation. FP32 and
all layer settings are unchanged. With dropout, changing execution groups can
change random draw ordering and therefore the optimization trajectory.

Validation has separate `validation_tiles`, `validation_peak_budget`, and
`validation_gat_tiles` controls. Physical tile batches are reassembled into the
original `training.tiles_per_batch` loss groups before reporting. This preserves
weighted discriminator reductions, mean/sum biology reductions, and the mean
valid CD8 early-stopping score, including partial groups. Validation covers the
entire configured validation split every epoch in evaluation and inference mode.

`validation_feature_cache` requires resident spectra and geometry. It encodes
only the unique required context rows, stores pooled spectral features on the
compute device, and skips subsequent spectral gathering and transfer. The cache
is local to one validation pass, rebuilt after training updates, and rejected by
model execution with gradients or training mode enabled. Its construction is
included in validation timing. `feature_cache_rows` bounds each construction
batch. No cache is saved in checkpoint weights.

The host budget covers resident spectra, graph payloads, and a conservative
bound on preparation/staging tensors. Construction rejects oversized caches.
Peak CUDA allocation is checked against `vram_budget_gib`; this is an execution
acceptance limit, not an allocator reservation. Oversized candidates can still
raise CUDA OOM and should be reduced. Defaults for the budgets are 128 GiB host
and 24 GiB CUDA. `execution_startup.json` records preparation cost and residency;
`resolved_config.yaml` records effective settings. Prepared runs add elapsed
time, observed rows, rows/second, loader waits, maximum prepared-batch payload,
process peak RSS, and peak allocated/reserved VRAM to `history.csv`. Process RSS
is a lifetime high-water mark, not incremental cache size.

Prepared batches optionally carry `spectral_chunks` (CPU boundaries),
`context_row_ids`, `prepared_graphs`, `graph_tile_budget`, and `reduction_groups`.
`move_batch_to_device` supports these nested structures; legacy tensor-only
batches remain valid. `pooled_features` is accepted only in gradient-disabled
evaluation. These fields are internal execution metadata, not model inputs or
checkpoint parameters.

Resume keeps the checkpoint's sampling contract, optimizer row budget, learning
rate, losses, geometry, and epoch-based adversarial schedule. Current
`performance` controls and model execution budgets/checkpointing can be applied
without modifying historical checkpoint files or parameter names. Optimizer-row
experiments belong to fresh fits.

Benchmark production footprints (including the largest stored-entry workload),
then compare complete matched capped epochs:

```bash
python -m dann.verification.execution_benchmark \
  --output data/PDAC/Results/dann/execution-benchmark.json \
  --warmup 1 --steps 3
```

The coordinate sweep tests training rows 1,024–8,192, peak budgets
65,536–524,288, GAT groups 8–64, and both checkpoint switches independently.
Validation compares 8, 32, 64, and 128 tiles with and without pooled features and
checks every reported fit metric against an FP32 reference. It does not rewrite
production YAML. Reuse a sweep for complete uncapped epoch comparisons:

```bash
python -m dann.verification.execution_benchmark \
  --selection-from data/PDAC/Results/dann/execution-benchmark.json \
  --full-epochs --output data/PDAC/Results/dann/execution-full-epochs.json
```

Matched comparisons start from the same model weights and splits, preserve the
original optimizer-row budget, and use the configured learning rate, class
weights, losses, and GRL schedule. Execution regrouping with dropout can produce
different trained weights; prediction/metric equivalence is checked with frozen
evaluation weights separately. Generated JSON and datasets stay under `data/`.


### Measured production profile (RTX 5090)

One matched full-epoch comparison on the assembled dataset covered 1,358,726
training rows and all 169,824 validation rows. Both runs used the same initial
weights, split identities, 2,048-row optimizer batches, and configured objectives.
The reference used backed loading and the original 65,536-peak / 8-tile execution
budgets. The selected profile uses 524,288 peaks / 64 GAT tiles, spectral
checkpointing on, GAT checkpointing off, two preparation threads, and validation
feature caching with 64 physical tiles (262,144 peaks / 32 GAT tiles internally).

| Measurement | Backed reference | Selected profile |
|---|---:|---:|
| Training time | 414.89 s | 300.02 s |
| Training rows/second | 3,274.87 | 4,528.82 |
| Full validation, including feature construction | 126.58 s | 18.86 s |
| Train + validation epoch | 541.47 s | 318.88 s |
| Training loader wait | 3.48 s | 0.33 s |
| Validation loader wait | 24.14 s | 0.008 s |
| Peak allocated VRAM | 0.88 GiB | 4.89 GiB |
| Peak reserved VRAM | 1.56 GiB | 8.23 GiB |

This is a 1.70× epoch speedup (41% less elapsed time). Cache/geometry startup
cost 33.04 s separately: 7.69 GiB resident CSR and 2.54 GiB graph/geometry payload.
Process peak RSS was 26.51 GiB. Both stages ran in one process after constructing
the cache, so RSS includes it even during the reference stage; it is not a
measurement of the reference-only working set. The filesystem cache was warm.
These are single-run timings, not statistical confidence intervals. Disabling
spectral checkpointing exhausted VRAM and was rejected. Larger optimizer-row
budgets did not improve the initial throughput sweep, so the existing budget
was retained.

Raw measurements are in the gitignored
`data/PDAC/Results/dann/execution-benchmark.json` and
`data/PDAC/Results/dann/execution-full-epochs.json`. Frozen-model validation
predictions and metrics are checked separately across physical batch sizes;
trained weights differ because regrouped dropout changes random draw ordering.

### Repeated larger-batch experiment

The joint sweep compares against the accelerated resident profile in the input
YAML. It keeps FP32, spatial geometry, learning rate, class weights, losses, and
the epoch-based adversarial schedule fixed. Run it without a concurrent GPU job:

```bash
python -m dann.verification.execution_benchmark --larger-batches \
  --output data/PDAC/Results/dann/larger-batch-benchmark.json
```

The defaults are five training warmup steps, twenty timed steps, and three
independent repetitions per candidate. Training tests rows 2,048/4,096/8,192,
peak budgets 524,288/1,048,576/2,097,152, and GAT groups 64/128/256 jointly;
identical GAT partitions are skipped. The largest complete sampler request is
also exercised with transfer prefetching enabled. Timing synchronizes CUDA at
window boundaries, rather than after each optimizer step. Training-screen
`seconds` is seconds per supervised row; each trial also records the actual
`window_seconds`, observed rows, and rows/second.

Validation separately screens complete feature-cache construction and complete
spatial passes. It tests cache batches 1,024–8,192 rows, peak budgets
262,144–2,097,152, physical batches 64–512 tiles, and GAT groups 32–256 tiles.
An explicit `validation_peak_budget` also controls the inner inference chunk
limit for that validation pass; both encoder limits are restored afterward.
Winning upper boundaries are doubled until memory rejection or two consecutive
increases each improve throughput by less than 3%. Invalid outputs/gradients
and memory-budget violations reject a candidate.

The best cache/spatial combinations undergo full validation, including cache
construction, prediction comparisons, and every reported metric at FP32
`rtol=5e-5, atol=5e-5`. The two fastest training candidates then undergo three
complete training-plus-validation comparisons against the current baseline,
with independent model/optimizer resets and paired seeds. A selection requires
at least a 3% reduction in median epoch time; candidates within 3% of the fastest
are ranked by peak allocated memory. Optimizer-batch changes apply to fresh fits;
existing resumes retain their saved row budget. Equal trained weights or
convergence are not implied by these throughput comparisons.

Reports preserve all repetitions, effective configurations, split hashes,
startup costs, memory peaks, and rejection reasons. Existing report files are
refused unless `--resume` is supplied. `--stage training`, `--stage validation`,
and `--stage epochs` allow resuming individual phases using the same output and
repetition settings. The harness never rewrites production YAML or checkpoints.

### Larger-batch results (2026-10-13)

The expanded experiment completed 18 passing joint training configurations,
18 cache-construction configurations, 13 spatial-validation configurations,
and five combined full-validation configurations, each with three measured
repetitions. A 4,194,304-peak training boundary probe exhausted VRAM and was
rejected; subsequent candidates ran successfully. All frozen-model validation
predictions and reported metrics passed the FP32 equivalence checks.

The baseline and two training finalists then completed three independent,
interleaved training-plus-validation trials each. Every trial covered 1,358,726
training rows and all 169,824 validation rows, with matched initial weights,
splits, and paired random seeds. Validation retained the original settings:
larger validation configurations did not clear the 3% full-pass threshold.

| Measurement | Current: 2,048 rows / 524,288 peaks | 4,096 rows / 1,048,576 peaks | 4,096 rows / 2,097,152 peaks |
|---|---:|---:|---:|
| Median training time | 300.47 s | 294.26 s | 292.48 s |
| Median training rows/second | 4,521.96 | 4,617.48 | 4,645.53 |
| Median complete validation | 19.27 s | 19.21 s | 19.22 s |
| Median complete epoch | 319.70 s | 313.44 s | 311.75 s |
| Complete epoch range | 319.69–321.55 s | 313.39–313.47 s | 311.59–312.04 s |
| Epoch time reduction | — | 1.96% | 2.48% |
| Peak allocated VRAM | 4.89 GiB | 9.72 GiB | 18.74 GiB |
| Peak reserved VRAM | 8.21 GiB | 14.08 GiB | 24.23 GiB |
| Median training loader wait | 0.360 s | 0.332 s | 0.331 s |
| Median validation loader wait | 0.0082 s | 0.0083 s | 0.0084 s |

Each table median is calculated independently from the three trials; median
stage times need not sum exactly to the median total. The full-epoch process
peaked at 26.51 GiB host RSS, a lifetime high-water mark shared across profiles.
The separate screening process peaked at 27.06 GiB host RSS.
Resident/geometry startup cost about 33.4 seconds per process, separately from
epoch timing. Screening and full-epoch confirmation ran in separate processes
using the same saved baseline and splits. The filesystem cache was warm; no
other CUDA compute process was observed during the comparisons.

Larger allocations were feasible: the 8,192-row screening workloads reached
19.34 GiB allocated VRAM, but did not outperform the training finalists. In
validation, isolated cache construction improved from about 13.05 to 12.02
seconds, while the best combined full pass improved only from 19.18 to 19.07
seconds. GAT groups of 64–256 tiles were slower than 32-tile groups in the
spatial-stage sweep. Full validation timing includes cache construction,
spatial execution, and the original loss-reduction/reporting groups.

**Decision: retain the existing production batch and execution settings.**
Neither training finalist achieved the required 3% median complete-epoch
reduction, and no validation candidate achieved a 3% full-pass reduction.
The larger controls remain available for future experiments. The explicit
validation peak budget now correctly governs both encoder chunk limits;
its current default equals the prior inner limit, so default execution size
is unchanged.

Larger optimizer batches also changed the training trajectory: median CD8
validation loss after the first epoch was 1.2211 for the baseline versus
1.2490 and 1.2489 for the two 4,096-row profiles. These single-epoch comparisons
do not establish final convergence. Existing checkpoint sampling contracts
and weights remain untouched.

The resumable raw report is
`data/PDAC/Results/dann/larger-batch-benchmark.json`; earlier benchmark reports
are preserved. Verification: **663 tests passed** across `dann/tests` and
`ihc_mvn/tests`, including validation chunk restoration, partial groups,
prediction equivalence, and benchmark selection thresholds.

## Tissue inference execution

`analysis.spatial_execution` enables a separate export profile. Omitting this
section preserves legacy inference; saved `performance.validation_*` annotations
do not enable it. The CLI passes current YAML execution settings explicitly.
Architecture, core size/halo, transforms, feature and target order, and density
sampling settings still come from the checkpoint. Sampling flags remain explicit
scientific overrides.

Execution precedence is explicit CLI tile/batch flags, current YAML settings,
saved inference-specific settings, then legacy defaults. `--tiles-per-batch`
takes precedence over `--batch-size` in spatial mode. Without that flag,
`--batch-size` keeps its historical meaning (tiles in spatial mode, rows in pixel
mode). Python callers may supply `execution_overrides={"spatial_execution": {...},
"performance": {...}}`; existing calls continue to work.

The optimized profile uses independent physical tile, spectral-peak and GAT
budgets. An explicit spectral budget controls both the outer complete-spectrum
batches and the encoder's inner no-gradient chunk limit. `feature_cache_rows`
bounds cache construction; `output_rows` bounds ordered export blocks.
`feature_cache` requires `performance.ram_cache` and `cache_geometry`. Residency,
geometry, CPU preparation threads, queued batches, transfer prefetch, PyTorch CPU
threads, and host/VRAM budgets reuse the shared `performance` controls.

Grid discovery reads only the required HDF5 observation columns and feature
index, including categorical and nullable encodings. It does not load unrelated
layers or HES arrays. Frozen preprocessing runs once on resident CSR storage.
Only the unique context rows required by the selected output tiles are encoded;
a leading-row export cap retains its complete external halo. Features and
prepared graphs are local to one invocation and cannot be used with gradients.
Spatial aggregation, support semantics and exported schemas are unchanged.

Prediction staging uses a temporary memory map in original source-row order,
with duplicate and missing-row checks. Model outputs share one CPU transfer per
tile batch. Preparation resources and pooled features are released before density
workers start. `density_workers: 0` runs serially; positive values use spawned
processes with at most twice that many outstanding tasks. Workers read only
prediction parameters from the staging map. Row-keyed NumPy generators, draw
order, sample counts and float64 accumulation are unchanged, so identical
parameters produce bitwise-identical sampled means across worker counts and
output block sizes. The parent consumes blocks in row order and writes Parquet.

Progress reports startup, feature encoding, spatial prediction, density sampling
and writing, and validation every five seconds during sustained work. Provenance
records effective settings, file identities, sampling seed/count, phase durations,
preparation waits, peak allocated/reserved VRAM, and sampled peak host RSS summed
across the parent and workers (shared pages can be counted more than once). Total
time starts before checkpoint loading and ends after complete output validation.
Host checks include CSR construction scratch, actual geometry/metadata, bounded
queues, staging and worker reserves. Budget failures report the requested memory;
there is no automatic precision, neighborhood or draw-count reduction.

Prediction, latent and support artifacts are fully validated before publication.
Each invocation owns uniquely named temporary files and removes them on failure.
Existing output or companion artifacts require `--overwrite`; a failed validation
leaves previously published artifacts intact.

Reproduce stage screening (one warmup and three timed repetitions per candidate),
then complete exports and compare every row:

```bash
python -m dann.verification.spatial_execution_benchmark screen \
  --output data/PDAC/Results/dann/spatial_execution_benchmark/screen
python -m dann.verification.spatial_execution_benchmark export --legacy \
  --output data/PDAC/Results/dann/spatial_execution_benchmark/legacy
python -m dann.verification.spatial_execution_benchmark export \
  --output data/PDAC/Results/dann/spatial_execution_benchmark/optimized
python -m dann.verification.spatial_execution_benchmark compare \
  --left data/PDAC/Results/dann/spatial_execution_benchmark/legacy \
  --right data/PDAC/Results/dann/spatial_execution_benchmark/optimized \
  --output data/PDAC/Results/dann/spatial_execution_benchmark/comparison
```

Run GPU measurements after other GPU workloads finish. Screening includes
a leading 16,384-row workload and the highest-entry 131,072-row window
(on a quarter-window scan stride), tile batches 8–256, spectral limits
262,144–2,097,152, feature batches 1,024–4,096, nonredundant GAT groups 32–128,
preparation threads 2/4/8 with twice as many queued batches, and density workers
0/2/4/8/16. Export comparison requires exact identities, order, schemas and
support, and FP32 `rtol=5e-5, atol=5e-5` for predictions, latents and derived
columns. Timing artifacts and data stay outside commits.

For density scaling beyond the eight default blocks in the large screen, reuse
its frozen parameter file with 32 bounded tasks and a persistent pool. This
reports startup in the warmup and three steady throughput repetitions:

```bash
python -m dann.verification.spatial_execution_benchmark density-steady \
  --parameters data/PDAC/Results/dann/spatial_execution_benchmark/screen/density_parameters.bin \
  --output data/PDAC/Results/dann/spatial_execution_benchmark/density_steady
```

Full exports always include pool startup in their measured total time. Small
exports may be faster with `density_workers: 0` because spawning costs more than
their sampling work.

### Measured tissue export profile (2026-10-13)

Measurements used the RTX 5090 (32 GiB), 16 physical CPU cores, PyTorch
2.11.0+cu128, the epoch-47 `deep_sets__agg-gatv2__bio-gatv2__disc-mlp` checkpoint
from `no-residual-gatv2-aggregation`, and all **5,893,291** tissue rows. Sampling
remained **1,000 draws**, seed **20260719**. Both optimized runs include startup,
cache construction, prediction, sampling, writing and complete output validation.

| Full export | Total | Peak allocated VRAM | Peak host RSS including workers |
|---|---:|---:|---:|
| Preserved pre-change source | 2,276.1 s (37m 56s) | Not instrumented | Startup peak not instrumented |
| Initial optimized profile | 246.5 s (4m 6s) | 6.94 GiB | 47.12 GiB |
| Selected profile | **207.3 s (3m 27s)** | **6.29 GiB** | **47.05 GiB** |

The selected profile is **10.98× faster** than the preserved source and 15.9%
faster than the initial optimized profile. Its peak reserved VRAM was 7.41 GiB.
Full exports were measured once per profile; the 39.2-second profile difference
substantially exceeds the observed stage timing variability. The original
baseline's CPU export overlapped stage screening, while its GPU phase and both
optimized complete exports had exclusive GPU execution. The selected full run
had no competing benchmark workload. Filesystem caches were not flushed.

The installed `analysis.spatial_execution` defaults are:

| Control | Selected value |
|---|---:|
| `enabled` | `true` in current YAML; absent sections remain legacy |
| `tiles_per_batch` | 64 |
| `spectral_peak_budget` | 262,144 |
| `gat_tiles` | 32 |
| `feature_cache` | `true` |
| `feature_cache_rows` | 1,024 |
| `output_rows` | 16,384 |
| `density_workers` | 16 |

Shared training performance settings remain unchanged: two preparation threads,
two queued batches, four Torch CPU threads, residency, geometry caching and CUDA
prefetch enabled, with 128 GiB host / 24 GiB allocated-VRAM budgets. The full
resident CSR was 27.73 GiB and the FP32 feature cache 5.62 GiB. Preparation waits
totaled 0.026 seconds; the largest prepared batch was 6.50 MiB.

Screening covered **60 configurations with 180 timed repetitions**, each after
one warmup, plus five density-worker configurations with another 15 repetitions.
The representative window was rows `[0, 16384)` (4,150,349 entries); the largest
window was `[4161536, 4292608)` (106,328,590 entries). On the latter, encoding was
fastest with 262,144 peaks / 1,024 cache rows (median 1.693 s). Tile batches
64–256 with GAT groups of 32 were within 1%; 64 retained the lower-memory choice.
Larger GAT groups and more preparation threads did not improve throughput.

With 32 density tasks and warm pools, median times for 131,072 rows were
15.05 / 7.75 / 3.99 / 2.09 / **1.41 seconds** for 0 / 2 / 4 / 8 / 16 workers.
The selected complete export's phase times were:

| Phase | Seconds |
|---|---:|
| Startup, resident inputs and geometry | 82.35 |
| Encode unique spectra | 38.23 |
| Spatial prediction and staging | 20.33 |
| Density sampling and writing | 55.61 |
| Complete output validation | 10.77 |

Startup/geometry preparation and density sampling are the remaining dominant
costs. Increasing physical GPU batches offers little benefit on this machine.

**Numerical results:** all selected-profile parameters, latents and continuous
logit summaries pass `rtol=5e-5, atol=5e-5`. The maximum latent difference is
`4.94e-5`. Row identities, schemas, ordering and support are exact. The initial
profile had one latent entry outside that tolerance; it was not selected.

The strict comparison also reports **322 of 23,573,164 sampled density values**
(0.00137%) outside that initial tolerance, with maximum difference `0.000998795`.
Tiny FP32 parameter differences can move a uniform draw across the hurdle
threshold, changing a 1,000-draw mean by up to 0.001. The unchanged sampler is discontinuous at this threshold. The audit replays the
original row streams using the exact inverse rounding intervals of exported
float32 presence probabilities. **All 322 exceptions are explained by at most
one hurdle switch**, with maximum continuous perturbation bound `1.37e-6` and
no unexplained rows. The strict comparison lists these exceptions explicitly. Frozen identical parameters produce bitwise
identical float64 means across serial/parallel workers, task order and block size.

All artifacts are isolated under
`data/PDAC/Results/dann/spatial_execution_benchmark/`: `baseline/` preserves the
pre-change source/config and commit; `screen/manifest.json` records source hashes,
configuration, workload positions, software/hardware and input/checkpoint stat
identities; `screen/screen.json` and `density_steady/density_steady.json` retain
individual trials. `full_legacy/`, `full_initial/`, and `full_tuned/` contain
complete exports. Their comparison JSON, the density audit and run logs remain
there with the original production outputs untouched.

The strict comparison returns a nonzero status for the documented sampled-mean
exceptions. Reproduce their independent audit with:

```bash
python -m dann.verification.spatial_density_audit \
  --left data/PDAC/Results/dann/spatial_execution_benchmark/full_legacy/predictions.parquet \
  --right data/PDAC/Results/dann/spatial_execution_benchmark/full_tuned/predictions.parquet \
  --output data/PDAC/Results/dann/spatial_execution_benchmark/density_audit_recheck.json
```

To reproduce the original pre-change baseline, run `python -m dann.spatial` from
`data/PDAC/Results/dann/spatial_execution_benchmark/baseline/`, using its saved
config and **explicit fresh prediction and latent output paths**. The current
`export --legacy` benchmark mode keeps the backed inference reference but already
benefits from selective metadata loading. Benchmark subprocesses used
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4`.

A further 131,072-row frozen-parameter check produced the **same SHA-256 of all
float64 density means** for 0/2/4/8/16 workers with block sizes ranging from
2,048 to 16,384 rows. Its evidence is in
`density_equivalence/density_equivalence.json`; reproduce it with:

```bash
python -m dann.verification.spatial_execution_benchmark density-exact \
  --parameters data/PDAC/Results/dann/spatial_execution_benchmark/screen/density_parameters.bin \
  --output data/PDAC/Results/dann/spatial_execution_benchmark/density_equivalence_recheck
```

Final verification:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q dann/tests ihc_mvn/tests`
completed with **687 passed**. Failure tests cover spawned worker exceptions,
writer-close errors, budget rejection, temporary cleanup and preservation of
existing exports. The input tissue and checkpoint stat identities were unchanged;
no invocation-owned staging files remain. `final_source/` preserves the final
implementation/configuration and source hashes.


## Tissue and lesion spatial heatmaps

`python -m dann.spatial_heatmaps --config dann/config.yaml` generates both tissue
and lesion views from the existing spatial prediction Parquet. Four targets
produce 16 PNGs: `spatial_heatmap_<density>_{tissue,lesion}.png` and
`spatial_heatmap_diagnostics_<density>_{tissue,lesion}.png`. Existing unsuffixed
figures and replaced flat copies are removed after successful validation. Outputs
are grouped into `heatmaps/tissue/` and `heatmaps/lesion/`. No inference or density
resampling is performed.

Tissue views retain the existing grid, masks, and HES orientation. Sequential
IHC spatial panels use `turbo`; signed residual panels retain their diverging scale. Lesion
views use the boolean `Lesion` observation column, zoom each slide to its lesion
bounds, and calculate each panel's color limits from visible lesion values.
Non-lesion pixels and holes remain gray inside the crop. The HES image is cropped
to the lesion's spatial bounds as context; its background is not segmented.
Residual limits remain symmetric around zero. A slide without any lesion retains
its row with “No lesion” labels. Missing/nonboolean/null lesion annotations fail
clearly; select `--regions tissue` when annotations are unavailable.

The loader reads only required observation fields, spatial coordinates, HES
images/scalefactors, and prediction columns. Spectra, layers, other images, and
HES embeddings are never loaded. Row positions and observation names must match
the exact inference input; duplicate names are supported. The Parquet's target
transformation contract controls observed CD8 normalization, including missing
annotations. The current YAML cannot change those scientific semantics.

Slide indices and numerical grids are prepared once and shared through read-only
memory maps. Spawned processes render independent PNGs with one numerical-library
thread each. File-backed maps use the operating system's RAM cache and avoid the
container's smaller `/dev/shm` limit. Temporary maps and figures live in a unique
subdirectory of the output directory and are cleaned on failure. Existing output
PNGs are replaced only after all requested images have been generated and
validated. Publication uses individual atomic file replacements, not an atomic
transaction across the entire set.

`analysis.spatial_heatmaps` controls `regions`, `lesion_column`, `workers`,
`host_budget_gib`, `png_compress_level`, and `cmap` (default `turbo`). Explicit CLI options override YAML;
Python callers may pass the same settings as keyword-only arguments to
`run_spatial_heatmaps`. One worker uses the identical renderer serially.

```bash
python -m dann.spatial_heatmaps --workers 8 --host-budget-gib 192
python -m dann.spatial_heatmaps --regions lesion --png-compress-level 1
python -m dann.spatial_heatmaps --regions tissue --workers 1
```

The host budget is independent of inference's RAM/VRAM budgets. Concurrency is
capped using CPU availability, task count, available RAM, and estimates for shared
arrays, image copies, four canvas buffers, and process overhead. A full 43-slide
parameter canvas at 300 DPI alone occupies approximately 2.08 GiB. PNG compression
level 1 is lossless and favors encoding speed over file size; use a higher level
(up to 9) for smaller files. Plot resolution and numerical precision are unchanged.

Progress is reported every five seconds during loading, preparation, rendering,
and validation. `heatmap_run.json` records source identities, settings, effective
workers, phase/total seconds, output dimensions/sizes, and sampled peak aggregate
parent/worker RSS. Aggregate RSS conservatively counts shared mapped pages in
each process; it is not a measurement of unique physical RAM.

Reproduce worker screening and full runs with
`python -m dann.verification.heatmap_benchmark {screen,full}` using `--input`,
`--predictions`, and an isolated `--output-dir`. Screening includes the largest
slide plus slides spread through source order, with warmup and three timed
repetitions for 1, 4, 8, and 16 workers. The `baseline` mode additionally takes
`--baseline-script` pointing to a preserved pre-change script. The `compare` mode
uses `--baseline-output` and `--optimized-output` to compare all tissue PNG pixels.


### Measured worker screening

On the Ryzen 9 9950X3D (16 physical cores / 32 logical CPUs, 249 GiB usable
system RAM), four representative slides at 300 DPI produced these end-to-end
medians for all 16 tissue/lesion figures. Each setting had a warmup and three
timed repetitions, with identical input and predictions:

| Rendering processes | Median seconds | Peak aggregate RSS (GiB) |
|---:|---:|---:|
| 1 | 59.42 | 4.35 |
| 4 | 24.96 | 8.04 |
| 8 | 19.92 | 13.65 |
| 16 | 19.64 | 21.11 |

The full 43-slide run took 121.27 seconds with eight workers and 105.45 seconds
with sixteen workers. A preliminary run capped at fifteen workers took 104.36
seconds, effectively tied with sixteen. The default remains sixteen workers,
with a separate 192 GiB host budget. Its measured peak aggregate RSS was
68.68 GiB. Loading took 4.78 seconds, preparation 4.19 seconds, rendering/writing
95.60 seconds, and final validation 0.89 seconds. These timings start at plotting
initialization and include input loading through output validation; interpreter
and module-import startup is excluded.

Rendering and encoding the large PNGs account for approximately 91% of total
runtime. The sixteen full-resolution files total about 4.58 GiB. Increasing worker
count beyond sixteen cannot help this sixteen-task workload. Reducing DPI would
reduce rendering cost, but the default preserves the existing resolution.

Detailed records and the pre-change source/config snapshot are under
`data/PDAC/Results/dann/heatmap_benchmark/`: `screen/timings.json`,
`full/timings.json`, and `full_16/timings.json`. The first full run used a more
conservative memory estimate; the final estimate retains four complete canvas
buffers and reserves one private copy of the shared image/grid set per worker,
plus parent/process overhead.


The preserved original script required **637.78 seconds** and **99.18 GiB** peak
RSS for the eight tissue figures. The optimized tissue-only run required
**82.03 seconds** and **35.73 GiB** peak aggregate RSS: **7.78× faster** for the same
eight outputs. Compression level 1 increased their combined file size from
2.71 GiB to 3.13 GiB (about 16%); PNG decoding preserves every pixel.

The normal production CLI subsequently wrote all sixteen figures to the
configured `analysis/spatial/heatmaps` directory in **112.67 seconds**, with
**57.59 GiB** peak aggregate RSS. The benchmark and normal CLI were measured in separate end-to-end runs; their
timing and memory measurements are reported separately. Existing unsuffixed PNGs and the input/prediction files
were preserved. `full_baseline/baseline_run.json`,
`tissue_only/workers-16/repeat-0/heatmap_run.json`, and the production directory's
`heatmap_run.json` contain the complete measurements.


Final verification: **711 tests passed** with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q dann/tests ihc_mvn/tests`.
All eight production tissue PNGs were compared against the preserved original
renderer at full resolution: **zero differing pixels** across 4,086,720,000 pixels.
Synthetic tests also cover lesion masks/crops/color limits, empty lesions,
normalization, selective reads, duplicate names, serial/parallel image equality,
worker/initializer failures, cleanup, budgets, and CLI precedence.

`comparison/comparison.json`, `final_suites.log`, `summary.json`, and
`source_sha256.json` under the benchmark directory retain the verification
results, software versions, measurements, and final source hashes. `final_source/`
contains the implemented source/config snapshot.

### Peak families and measured spatial activity

The project configuration enables `analysis.peaks` and requests **32** families.
The stage interprets the saved Deep Sets embedding dictionary without retraining.
To run it without latent inference or Monte Carlo prediction sampling:

```bash
python -m dann.analyze --config dann/config.yaml --peaks-only
python -m dann.analyze --config dann/config.yaml --peaks-only \
  --peak-spatial-input data/PDAC/Raw/adata_assembled_tissue.h5ad \
  --output-dir data/peak-review --cache-policy reuse
```

Normal analysis also runs this stage after its existing outputs. Configurations
without `analysis.peaks` retain their previous behavior; `--peaks-only` explicitly
requests the stage regardless of `enabled`. `spatial: false` omits tissue exports.
The existing `analysis.execution.render_workers` and
`analysis.execution.performance.host_budget_gib` govern execution (the project
uses 16 workers and 192 GiB). Peak statistics and scores ignore smoke sample caps:
statistics always cover all labeled lesion pixels, and spatial scores always
cover the supplied full tissue input.

Outputs live in `analysis/peaks/`:

- `peak_embeddings.csv` contains peak identity and every FP32 embedding coordinate,
  with enough decimal precision for exact FP32 round-tripping.
- `peak_annotations.csv`, `peak_slide_statistics.csv`, and
  `peak_slide_ihc_associations.csv` provide lesion statistics and associations.
  Missing correlations remain NaN; contributing slide counts are exported.
- `family_membership.csv`, `family_sizes.csv`, `peak_linkage.npy`, and
  `peak_order.npy` preserve the average-linkage cosine hierarchy and `maxclust`
  cut. `peak_run.json` records both requested and actual counts (ties may give
  fewer families). Family colors are shared across tables and figures.
- `peak_umap.csv` stores the single cosine UMAP used by every color variant
  (30 neighbors, minimum distance 0.30, seed 20260719). PNG panels and standalone
  offline HTML include family, m/z, prevalence, detected intensity, embedding norm,
  held-out IHC associations, and slide prevalence. HTML includes a color/slide
  selector. Prevalence-versus-intensity plots use a logarithmic intensity axis;
  never-detected peaks remain in tables and their plot exclusion count is reported.
- Heatmaps include every embedding coordinate, the full cosine overview, the
  original top-frequency selection with its pruned original dendrogram, every
  complete family matrix (also in zoomable offline HTML), family mean pairwise
  similarity, and batch prevalence. Family-summary diagonals exclude self pairs;
  singleton diagonals are missing. Global PNGs omit m/z ticks; family PNGs retain spaced m/z labels. Every matrix retains all selected members.
- `family_spatial_activity.parquet` preserves original row positions, observation
  names, slide/coordinates, lesion flags, and FP32 mean signal and detection
  percentage for every family. `family_spatial_settings.json` records common limits.
  `family_01_spatial.png` contains all slides as rows and six columns: tissue
  signal/detection/HES and lesion signal/detection/HES. Each canvas is 36 inches
  wide and 4 inches per slide at the configured 300 DPI. At 43 slides and 32
  families, there are 32 family spatial PNGs plus two CD8 comparison PNGs,
  78 total PNGs, and 36 offline HTML files.
- UMAP and prevalence–intensity each produce three PNGs: all slides in six
  columns, four IHC targets in 2×2 panels, and five other annotations in 2×3 panels.
  Sequential colors use `turbo`; detection remains linear from 0–100%.
- `ihc_display_settings.json` records per-target symmetric limits at the 99th
  percentile of absolute finite peak correlations, `RdBu_r`, and signed square-root
  coloring. These limits are shared across every view and strip. Colorbar ticks
  and hover show original correlations, with clipped tails indicated. Missing or
  all-zero targets use ±1. CSV exports retain unmodified correlations.
- `full_similarity.html` stores the symmetric FP32 upper triangle once (about
  86 MiB of base64 for 5,808 peaks). Its bounded 800×800 overview samples cell
  centers and redraws on zoom; hover reads the exact original cells. Matrix,
  annotation strips, and original dendrogram share one row viewport. The top-512
  and family viewers use offline Plotly with vertically rotated strip headings;
  top-512 retains the pruned original tree. Global full/top PNGs also show trees.
- `family_slide_ihc_associations.csv` and `family_ihc_associations.csv` contain
  associations of mean transformed family signal, including valid observation
  and contributing-slide counts. `peak_family_ihc_correlations.png` compares
  lightly jittered individual peak dots with outlined family-signal diamonds in
  four target panels. Family colors remain consistent and correlation axes linear.

Statistics count distinct positive peaks after sparse duplicate entries are summed.
Prevalence uses all lesion pixels, including empty spectra; detected intensity is
the positive raw intensity sum divided by detections. The top-frequency heatmap
retains the older selection based on stored occurrences across all labeled rows.
Associations use saved checkpoint rows selected by `analysis.split` (default
`validation`), intersected with `Lesion=True`.
They compute within-slide Spearman correlations including absent peaks as zero,
require at least 20 valid observations and nonconstant inputs, and average valid
slide correlations with equal weight. Frozen target transformations and masks are
applied, including explicitly labeled normalized CD8. Family correlations use mean
transformed family signal from labeled spectra on those same saved rows, with a
separate numerical cache. Both correlation cache keys include the selected split;
`peak_run.json` records `correlation_split` and `correlation_lesion_rows`.
Plotting options never invalidate numerical stages.
These are associations on
held-out pixels, not held-out patients or evidence of causality.

Spatial scores sum transformed measured intensities over family members, divided
by total family size. Positivity and the saved raw threshold determine detection;
the saved intensity transform and clipping determine signal. Duplicates are
coalesced before thresholding and transformation. All-zero spectra remain valid
zero-valued pixels, while missing tissue and lesion holes remain blank. Lesion
views mask and crop the exact same tissue scores. Detection uses 0–100% everywhere;
signal uses zero to the family's positive-score 99th percentile over all tissue
rows, with saturation labeled. All-zero families use 0–1. Exports retain uncapped
scores. Feature identities and ordering must exactly match the checkpoint.

Numerical caches are separate from rendering and bind the checkpoint, source file
identities, feature order, selected correlation split and rows, preprocessing,
and annotation settings.
Resident CSR arrays and bounded blocks avoid a dense pixels-by-peaks matrix;
family sums accumulate in float64 before FP32 storage. Figures use shared immutable
mmap arrays and a single bounded worker pool. The requested 16 workers are reduced
when actual tall canvases and encoding buffers would exceed the 192 GiB budget.
The legacy `slides_per_page` setting is accepted but no longer paginates. Outputs are staged and validated
before publication; failures clean up invocation-owned temporary files.
`peak_run.json` records cache reuse, effective settings, phase times, parent-plus-
worker peak RSS, artifact names, and exclusion counts. Progress prints every five
seconds, including during long computations and rendering.

After a complete real-data run, an independent reference audit is available:

```bash
python -m dann.verification.peak_analysis_audit --config dann/config.yaml
```

It checks selected peaks over every labeled row, all corresponding slide/target
Spearman associations, sampled tissue spectra, all family display percentiles,
family-signal correlations, exact output counts, and spatial canvas dimensions. Its results and the measured run time/memory are saved
in `analysis/peaks/peak_audit.json`.

For browser verification with Playwright and Chromium installed, run:

```bash
python -m dann.verification.peak_browser_audit --output-dir path/to/analysis/peaks
```

The audit opens all HTML files with network access disabled, checks exact matrix
and annotation values, and exercises zoom alignment and selectors. It writes
`peak_browser_audit.json`; optional `--screenshots` saves browser previews outside
the peak output directory.


### Organized figure outputs

Peak outputs are grouped under `analysis/peaks/`:

- `spatial/families/`: one six-column tissue/lesion figure per family.
- `spatial/cd8_comparison/`: `cd8_families_tissue.png` and `cd8_families_lesion.png`.
  Each has one row per slide and 33 columns: observed checkpoint-normalized CD8
  density followed by mean signal from all 32 families. All columns use `turbo`.
  Overview PNGs are capped at an 8,000-pixel longest edge. Companion offline HTML
  viewers load lossless native-grid panels only near the visible viewport; keep
  each HTML file beside its `_assets` directory when copying the outputs.
  CD8 uses a linear zero-to-positive-P99 density scale; family signals reuse their
  established tissue limits. Each column shares its scale across slides and regions.
  Missing observations, physical holes, valid zeros, and empty lesions are preserved.
  The overviews and detail panels run in a separate memory-budgeted render queue.
- `umap/`, `prevalence_intensity/`, and `correlations/`: consolidated figures.
- `similarity/global/` and `similarity/families/`: PNG and offline HTML matrices.
- `tables/`, `hierarchy/`, `settings/`, and `reports/`: numerical exports, saved
  ordering, display settings, and audits. `peak_run.json` remains at the peak root.

Latent UMAP figures live in `analysis/umap/overview/`, `parameters/`, and
`diagnostics/`; the saved UMAP table is in `umap/tables/`. The six parameter
panels are **logit | sampled mean | batch** above **1-pi | mu | sigma**. Reordering
uses the same saved values and coordinates. Other recognized analysis artifacts
are grouped into `tables/`, `similarity/`, `density/`, and split-specific
`diagnostics/` directories. Generated flat copies are moved or replaced, while
unrecognized files and numerical caches are preserved.


### Offline latent explorers and number formatting

`analysis.latent_interactive: true` generates five viewers under `analysis/latent/`:
`sample_umap.html`, `latent_ihc_scatter.html`, `prediction_scatter.html`,
`feature_correlations.html`, and `feature_umap.html`. Open these directly from
disk; retain the accompanying `assets/` directory. Sample UMAP coordinates and
all saved samples are preserved. Feature columns load on demand, with at most
twelve columns retained in the browser cache. Prediction scatter uses the
configured scatter split; sample UMAP and latent–IHC scatter use the analysis split.

Regenerate these viewers without model inference:

```bash
python -m dann.latent_analysis --config dann/config.yaml
```

Feature correlations use saved analysis rows intersected with lesions, FP64
Spearman ranks, at least 20 observations, and equal weighting across valid slides.
IHC masks and normalized target values come from the saved export. The new
`latent-feature-associations` and `latent-feature-umap` caches are independent of
existing peak and inference caches. Undefined correlations remain missing;
layout alone uses neutral distance 1 for undefined pairs. Constant features stay
in the heatmap and are reported as unavailable for feature UMAP.

HTML displays integer identifiers/counts without decimals, m/z with four decimals,
and other values with two decimals at absolute values ≥1, three at 0.01–1, and
four below 0.01. Values below 0.0001 use scientific notation. Raw embedded values
and exported numerical tables retain their precision.

Recognized legacy variance files migrate into `analysis/latent_variance/` without
recomputation. Conflicting destination contents stop migration; unrelated files
remain untouched. Explicit standalone `--output-dir` overrides remain supported.

### Repeat analysis for existing checkpoints

`python -m dann.verification.rerun_analysis --roots COMBINATION_DIRECTORY ...
--archive-root data/temp` discovers each immediate run's `model/best.pt`, validates
all destinations, and moves old `analysis/` and sibling `latent_variance/` trees
into a timestamped archive. It preserves `model/` byte-for-byte and never trains.
Fresh outputs use current analysis settings with each checkpoint's frozen model,
preprocessing, target contract, and saved splits.

The runner executes analysis, spatial inference, spatial heatmaps, latent variance,
and numerical/browser audits sequentially. Browser audits require Playwright and
Chromium in the calling environment. Configurations, logs, stage timings, sampled
process-tree memory peaks, and a summary live in the archive's `batch_records/`
and manifest. Resume with `python -m dann.verification.rerun_analysis --resume
ARCHIVE_DIRECTORY`; completed stages are skipped only when their recorded files,
configuration, inputs, and training artifacts remain unchanged. Archived caches
are never reused. A failed stage skips that run's dependent stages and allows
the remaining runs to proceed.


### Gradient and representation monitoring

The current YAML enables observational monitoring:

```yaml
monitoring:
  enabled: true
  gradient_updates_per_epoch: 5
```

Omitting the block disables collection for legacy callers. A positive sample count
chooses unique, evenly spaced **logical optimizer updates**, including the first
and last when the count is at least two. Short epochs use every update; `null`
requests every update. Microbatch size and resume position do not change those
positions. Enablement and frequency can change on resume, including when starting
from an older checkpoint. Earlier history entries remain missing.

For each sampled effective batch, additional `autograd.grad` traversals use the
existing forward graph and accumulate gradients on every trainable encoder
parameter, including latent aggregation. Prediction heads are excluded from this
decomposition. Eight IHC branches include target weights, the biology multiplier,
validity/positive masks, the original common loss denominator, and accumulation
scales. Target and aggregate IHC gradients are sums of their branches. H&E and
batch gradients include their objective weights; the batch gradient also includes
the actual reversed GRL multiplier. These are objective gradients, not percentages
of AdamW parameter updates. Normal backward, clipping, optimization, checkpoint
selection, and LR scheduling are unchanged.

`history.csv` stores per-sampled-update means, population standard deviations, and
finite counts (`*_mean`, `*_std`, `*_count`) for weighted L2 magnitudes, magnitude
shares, pairwise cosines, retained magnitude (norm of sum / sum of norms), and
signed contributions (component dot sum / squared norm of sum). Pairwise conflict
frequency is the mean of `cosine < 0`, excluding undefined cosines. Signed values
may be negative or exceed one; they are missing if the combined norm is at most
1e-6 of the summed component norms. Zero magnitudes remain zero; undefined ratios
remain missing. Plotting omits zeros only on logarithmic magnitude axes and labels
that omission. Whole-model pre-clipping norm, actual clipping multiplier, clipping
fraction, and actual GRL strength summarize **all** optimizer updates.

Validation uses every selected core observation with dropout disabled and fixed
model weights. Centered float64 covariance merging records latent sample variance
and participation ratio without exporting vectors. Participation ratio is missing
for zero variance; covariance is unavailable for fewer than two observations.

Only latent total variance and participation ratio are exported from validation
monitoring. Covariance retains its internal observation count for sample-variance
normalization. Training monitoring does not calculate CD8 Brier scores, interval
coverage, reliability bins, probability observation counts, or training quantiles.
Standalone post-training calibration remains available and unchanged.

Version-two `monitoring_definitions` in the resolved configuration and checkpoints
record this reduced scope. Resume accepts version-one definitions, retains the
scientific definitions of the remaining diagnostics, and retires probability
metadata. Existing history values are preserved; retired columns remain empty in
new epochs. Fresh histories omit those columns entirely.

Both four-column figures use one immutable history snapshot in synchronous
training, asynchronous artifacts, and post-training analysis. The overall figure
places IHC hurdle, IHC positive, H&E, and discriminator fit metrics directly above
their objectives, followed by total/supervised objectives and GRL. The next two
rows align objective pairwise cosine above conflict frequency and single-column
latent participation ratio above total variance. Pre-clipping norm and clipping
activity occupy the final row. The IHC figure keeps CD8, Tumor, Collagen, and
Stroma in fixed columns: four fit/objective rows, two cross-target
gradient rows, and four hurdle/positive gradient rows. Cross-target retained
magnitude and signed contributions each span two columns.

Figures are 24 inches wide with 14-point subplot titles, 12-point ticks/legends,
and 18-point figure titles. Shared legends identify train/validation line styles,
selected checkpoints, and component colors; only clipping and GRL use local
legends. Target colors remain blue/orange/green/purple; hurdle is charcoal,
positive is red, and adversarial is brown. Epoch ticks and checkpoint markers
remain, without repeated axis labels. Missing values, negative objectives and
signed contributions, zero reference lines, and logarithmic magnitudes are
preserved. Dimensions in `curve_layout.py` also determine the artifact worker's
canvas memory reservation. Old histories render unavailable panels without
prediction exports.

Run an isolated synthetic runtime/memory comparison without touching study data:

```bash
python -m dann.verification.monitoring_benchmark --output /tmp/dann-monitoring-benchmark
```

This runs disabled, five-sample, and every-update arms in fresh processes using the
production model/training loop with deterministic CUDA operations (or `--device cpu`). Each arm warms up before timing three 20-update epochs with validation.
The report includes median runtime, process peak RSS, allocated/reserved VRAM,
overheads, and final parameter hashes. This small Deep Sets/MLP benchmark isolates
monitoring cost; its relative overhead is not an estimate for a full spatial fit.


Historical measurement of version-one monitoring (including the now-retired CD8
probability statistics), on 2026-10-22 with PyTorch 2.11.0/CUDA 12.8 and an RTX 5090
(135,247 model parameters, 74,752 shared parameters; 2,560 training and 512
validation rows per epoch, three timed epochs after train/validation warmup):

| Monitoring | Median epoch seconds | Runtime overhead | Peak allocated GPU MiB | Process peak RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| Disabled | 0.088 | — | 74.94 | 2293.94 |
| Five updates | 0.504 | +474.5% | 79.52 (+4.57) | 2477.67 (+183.73) |
| Every update (20) | 1.735 | +1877.0% | 79.52 (+4.57) | 2520.05 (+226.11) |

All three deterministic arms produced identical final parameter hashes. The high
relative runtime cost reflects the very short unmonitored synthetic epoch and the
extra component backward passes. Peak RSS includes process/library initialization;
allocated GPU memory isolates tensor memory more directly. The configured default
remains five sampled updates per epoch.

### Training stabilization and version 7 checkpoints

`model.spectral_encoder.deep_sets.peak_mlp_residual` and each component's
`mlp.residual` enable `(input + update) / sqrt(2)` around equal-width hidden
blocks. Unequal-width blocks and final projections stay ordinary transformations.
Omitting these flags preserves the original MLP construction, parameter names,
registration order, and seeded draws. CNN and GATv2 residual flags remain separate.

`deep_sets.pool_centering: zero_input` subtracts the dropout-free `peak_mlp(0)`
from each nonempty spectrum's active-peak mean. Empty spectra remain exactly zero.
The reference retains gradients, uses no dropout RNG, and never changes module
training modes. Direct pooling, spectral checkpointing, and pass-local inference
caches share this operation; cached features are already centered. There are no
batch statistics or fitted preprocessing state. Omission means `none`.

Fresh training can initialize the biology and H&E heads from selected training
rows using `training.head_initialization`:

```yaml
head_initialization:
  enabled: true
  output_weight_scale: 0.01
  min_positive_sigma: 0.1
warmup:
  epochs: 1
  start_factor: 0.1
```

Biology initialization uses the transformed targets and per-target validity masks:
zero-fraction logits, positive-logit means, and population standard deviations
floored at `min_positive_sigma`. The raw scale inverts
`sigma = softplus(raw_sigma) + model.sigma_min`. Probabilities use the loss epsilon.
Unobserved targets use zero probability 0.5; targets without positives use
`mu=0, sigma=1`. H&E biases use streamed means of raw
training features in stored column order. Both supervised final projection weights
are scaled, preserving nonzero upstream gradients; discriminator parameters retain
their original initialization. Counts, means, scales, and explicit fallbacks are
saved in `head_initialization.json` and production checkpoint metadata. Omission
disables initialization, and resume never recomputes these statistics.

Warmup uses actual logical optimizer updates per epoch, including the final partial
batch. For N > 1 warmup updates, the first uses `start_factor * configured_lr` and
the Nth uses the configured LR; N=1 uses the start rate then installs the base rate.
Accumulated microbatches share a rate. Validation epochs containing any warmup
updates do not advance the plateau scheduler. Subsequent plateau behavior and its
floor relative to the original configured LR are unchanged. Histories record
`train_learning_rate_min`, `train_learning_rate_max`, `train_warmup_updates`, and
`next_learning_rate`. `epochs: 0` (the omitted default) disables warmup. Both
synchronous and asynchronous checkpoints save its resolved horizon, base rates,
and completed updates. Enabled-warmup resumes reject missing or inconsistent state.
Existing epoch-only resume restrictions still apply to partial-epoch checkpoints.

Stabilization settings were introduced in architecture contract version 7;
new checkpoints use version 8 with explicit spectral pooling and normalization.
Versions 2–7 reconstruct their historical MLP and pooling behavior. Changing
pooling, normalization, centering, or residual semantics is
incompatible with loading those weights as a new architecture; start a fresh fit.
Inference target order, prediction fields, sparse processing, and frozen target
transformation contracts are unchanged.

Learning rates remain explicit in `training.pixel.learning_rate` and
`training.spatial.learning_rate`; there is no automatic width-based LR scaling.
Use wider-model rates as an empirical tuning choice, assessed with latent variation
and predictive distribution calibration as well as loss.

In the matched 2026-10-28 three-epoch width-1024 experiment, explicit LR 0.0005
produced participation ratio 16.81 versus 13.45 at 0.001, presence Brier 0.1831
versus 0.1835, and between-slide variance fraction 0.475 versus 0.497. CD8 loss
was slightly worse (0.9750 versus 0.9730). These single-seed early results support
trying the lower rate for representation preservation; they do not establish a
universal scaling rule or final convergence. Both wide arms used identical
checkpointed spectral chunks and the same 300-epoch GRL horizon.

A reproducible three-epoch comparison driver creates new directories only:

```bash
python -m dann.verification.stabilization --output NEW_DIRECTORY --prepare-only
python -m dann.verification.stabilization --output NEW_DIRECTORY --run-all
python -m dann.verification.stabilization --output NEW_DIRECTORY --summarize
```

It runs six width-512 MLP ablations, width-1024 combined runs at explicit 0.001 and
0.0005 rates, and baseline/combined CNN and GATv2 runs (GATv2 aggregation depth 4).
Peak hidden/output, latent, and MLP aggregation hidden widths follow the experiment
width; CNN/GATv2 channels and all head hidden widths retain the source configuration.
Baseline CNN residual flags
are preserved; baseline GATv2 residuals are disabled. All arms retain seed, split,
sampling, and losses, and save initial parameter hashes before marginal
initialization. The GRL horizon stays at 300 full epochs. Tissue exports and
post-training analysis are disabled. Per-arm histories and early probes, final
CSV comparisons, and PNG/PDF plots separate latent variation from loss and CD8
calibration. These runs establish early behavior, not final convergence or
biological usefulness of the representation.

Wide pixel MLPs may opt into `model.pixel_spectral_microbatching: true` to apply the
existing `spectral_peak_budget` and `spectral_checkpointing` controls before MLP
aggregation. The default is false, preserving historical pixel execution. This
changes memory use and dropout draw grouping, without changing parameter names,
logical optimizer batches, the sampling order, or the architecture contract.
Evaluation and gradients match direct pooling with dropout disabled; checkpoint
recomputation preserves RNG with dropout enabled. The width-1024 comparisons use
this path with a 262144-peak budget in both learning-rate arms because the direct
path exhausted the RTX 5090 during gradient monitoring. The failed attempt is
retained separately and excluded from comparison tables.

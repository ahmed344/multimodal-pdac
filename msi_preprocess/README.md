# MSI preprocessing

This standalone package fits three optional stages on `layers/counts` and writes
new lesion and whole-tissue H5AD files. Counts already have RMS normalization.
The output `X` is **linear**: retained counts divided by one frozen slide
multiplier, with unsupported weak measurements set to zero. DANN continues to
read `X` and apply `log1p` once. No new RMS/TIC normalization, feature variance
scaling, averaging, imputation, or stored logarithm is introduced.

```bash
python -m msi_preprocess.fit --config msi_preprocess/config.yaml
python -m msi_preprocess.apply \
  --fit data/PDAC/Processed/msi_preprocess/fit.json --update-dann-config
python -m msi_preprocess.report --fit data/PDAC/Processed/msi_preprocess/fit.json
python -m msi_preprocess.verify --fit data/PDAC/Processed/msi_preprocess/fit.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q msi_preprocess/tests
```

Use the configured scientific/devcontainer environment. Run from the repository
root. The input/output paths and budgets are in `config.yaml`. Source files are
opened read-only. Standalone fit/apply refuse to overwrite existing exports or
complete fits; replacement is an explicit staged operation.
The replacement runner stages a new generation, verifies it and runs a capped DANN
smoke before swapping outputs and updating DANN. Failed promotion restores the old
output directory and YAML. Superseded large files are removed only after success;
previous parameters and summaries are retained in a compact ZIP archive.

```bash
python -m msi_preprocess.run --config msi_preprocess/config.yaml --replace \
  --expected-cohort 40 19 1665807 5468414
```

An interrupted fit reuses completed per-slide QC caches only when source
identities and settings match. Application removes incomplete output files on
failure. Change `output` to run a different stage combination or refit.

## Fitting and image context

Fitting uses **all retained lesion rows**, independently of every downstream model's split,
seed or smoke caps. Its own YAML defines the batch/x/y columns. Only lesion
counts, slide/patient identities and coordinates enter fitting, and fitting
neighborhoods contain lesion pixels only. Non-lesion tissue values cannot supply
anchor statistics, peak-selection evidence or reference draws. Version 3
provenance records the `retained_lesion` scope, the cohort decisions, and exact
source-row mappings for both exports, all bound into the fingerprint. Parameters remain frozen during application. Version 1 artifacts retain
their original DANN-training-only contract; version 2 retains its all-lesion contract.

Application uses the entire MSI image on each slide, including validation and
non-lesion pixels, for spatial support. This is explicitly **whole-image MSI
context**, not an evaluation independent of the image. Neither IHC nor H&E
labels enter cleaning. Lesion rows are matched by `(batch, x, y)`; all original
matched counts must agree. Lesion `X` and retained-feature `layers/counts` are
then extracted from the written tissue result, preserving each file's original
row and feature order. Duplicate keys, missing lesion matches, reordered peaks,
changed sources, unknown slides, or missing fitted parameters fail explicitly.
New slides require a new study fit; application never estimates new multipliers.

## Stages

1. **Selection.** Reuses the QC raw-log and TIC-adjusted-log spatial excess
   correlation screens. The latter is diagnostic only and never changes output
   intensities. A peak is removable when at least two and at least half of
   adequately assessed slides have at least 50% of positive measurements at
   intensity <=0.5, and both excess correlations <=0.02. Any slide with both
   excess correlations >=0.10 vetoes removal. A connected patch of at least four
   locally supported pixels also vetoes removal, including peaks too sparse for
   the global correlation screen. Peaks without enough assessment are retained.
   The retained count follows the evidence; there is no removal quota.
2. **Integration.** Retained anchors have prevalence >=10% on >=80% of fitting
   slides and global or patch spatial support on >=50%. For each anchor, first
   take the median positive intensity per slide, then the median of slides
   within each patient, then the median across patients. Each slide multiplier
   is the median of its usable anchor/reference ratios. At least 32 anchors are
   required on every slide; insufficient anchors stop integration. Multipliers
   are clipped to explicit bounds [0.5, 2.5], with the raw ratio, usable-anchor
   count, and clipping recorded. Bounds restrict the divisor, so applied intensity
   factors range from 2 to 0.4; scaling is identity only at divisor 1. Zero values stay zero.
3. **Local removal.** A measurement is removed only if positive, <=0.5 in the
   reference-adjusted linear units, sufficiently assessed, and unsupported on
   both neighborhood scales. One-step and cumulative two-step neighborhoods
   follow four-neighbor paths through occupied pixels; self, duplicate paths,
   slide crossings, and jumps across gaps are excluded. A scale requires at
   least two neighbors. A pixel with neither scale assessable is retained.

Local detection support is the fraction of positive neighbors. Intensity
concordance is that fraction multiplied by
`exp(-abs(log1p(center) - mean(log1p(positive_neighbors))))`; a zero center has
zero concordance. The fit makes 1,024 deterministic independent random-location
draws with replacement: a center and twelve neighbor locations, **including
zeros**. Separate 95% upper envelopes are saved per slide, feature, and actual
neighborhood size (1..12). Either observed statistic reaching its envelope on
either scale supports the measurement. A statistic must also be strictly
positive, so a zero envelope does not protect an isolated measurement. Equality
preserves constant coherent fields. The same draws are used across feature
blocks, making the result independent of block size. The envelopes are
descriptive settings, not calibrated significance tests or multiple-testing
adjusted p-values. Strong isolated signals and weak coherent regions survive.
Sensitivity at 0.1, 0.2, 0.3, 0.5, and 1.0 is reported using the same frozen
support decisions.

Each `stages` toggle is independent. Disabling integration sets multipliers to
one; local thresholds then use original RMS-count units. Disabling selection
retains every feature. Disabling all stages yields `X == layers/counts`.

## Artifacts and DANN contract

- `fit.json`, `fit.npz`, and `fit_work/`: settings, source identities, exact
  fitting rows, decisions, patient references, multipliers, spatial envelopes,
  and a SHA-256 preprocessing fingerprint with a parameter-file checksum.
- `peak_decisions.csv`, `slide_multipliers.csv`, and per-slide fitting CSVs:
  auditable selection/support evidence and correction limits.
- `tissue.h5ad`, `lesion.h5ad`: sparse `X`, original retained-feature counts,
  consistently subset observation/embedding/pairwise metadata and ordered feature annotations,
  `obs.msi_source_row`, `obs.msi_removed_measurements`, original feature indices, and
  `uns.msi_preprocess` with fitted parameters and provenance.
- `report.html`, `figures/`, and CSVs: counts versus old `X` versus selection,
  integration, and cleaning; intensity histograms, slide-associated variance,
  threshold sensitivity, per-feature removals, and matched spatial maps.
  Map examples come from fitted anchors, globally supported peaks, localized
  patches, weak retained peaks, and removed peaks. Native-grid rendering masks
  gaps and does not interpolate pixel values. To regenerate maps independently,
  run `python -m msi_preprocess.report --fit <output>/fit.json --maps`.
  CD8/stroma associations are computed **after** cleaning over the full lesion. They are diagnostics of structural preservation;
  slide variation also contains patient biology, so lower variation alone is
  not evidence of a better representation.
- `disk_preflight.json`, `export.json`: conservative disk estimate and completed
  export contract. A full sparse slide and small dense feature blocks are held
  in memory; the full dataset is never materialized. Arbitrarily interleaved
  slides use a temporary sparse cache with an additional disk check. Unknown
  extra layers/raw matrices fail rather than being silently dropped.
  `application_workers` bounds parallel feature blocks and is included in the
  memory estimate. CSR row pointers and cumulative entry offsets use int64, so
  exports can exceed 2.1 billion nonzeros while feature indices remain int32.
- `verification.json`: independent written-schema checks, a streaming check of
  **every** tissue nonzero against linear counts/scaling and allowed removals,
  exhaustive lesion/tissue agreement, and the DANN single-log1p loader check. Full exported-value checks and sampled DANN-loader checks are recorded separately.

`--update-dann-config` patches only `data.path`, `data.context_path`,
`model.num_peaks`, and `data.preprocessing_fingerprint`, and appends `_qc40` to the run name for this
schema 3 generation, preserving unrelated
YAML and comments. It verifies `matrix_key: X`, `intensity_transform: log1p`, and
that no resume checkpoint was requested. DANN validates the fingerprint and declared fit scope (including the uncapped
training split for legacy version 1 artifacts only); checkpoint identities include preprocessing provenance.
Changed fingerprints require fresh training. Existing unprocessed inputs and
their checkpoint contracts remain supported.

These thresholds are operational settings, **not a measured instrumental noise
floor**. The pipeline cannot recover information lost during RMS normalization
or feature alignment. A short DANN smoke run validates data/model plumbing; it
does not establish biological preservation, CD8 calibration, or latent quality.

## Pixel heatmaps, thresholds and percentages

The report samples 100 lesion pixels per slide without replacement using seed 42
(or all pixels when fewer are available). Every heatmap row is one pixel, grouped
by clustered slide and ordered by coordinates within each slide. The 500 retained
peaks with the highest population variance of `log1p(counts)` across those pixels, including zeros, are
displayed (`report.max_peaks`, default 500; all retained peaks if fewer exist).
Variance ties are broken by original feature index. The same peaks and pixels
appear in every stage, including counts and old `X`. This display subset does not
alter the retained dataset, fitted parameters, or exported matrices.

Columns are clustered once from sampled `log1p(counts)`, using correlation
distance and average linkage with original feature order as input; constant
sampled peaks are appended in original order. `heatmap_peaks.csv` records original
and retained-column indices, population variances, variance ranks (one-based),
and clustering positions (zero-based). `heatmap_linkage.npy` and its input feature
indices in `heatmap.json` preserve the clustering. Pixel identities, shared color
limits and saturation counts are also saved.

Slides are clustered as independent observations, each represented by its mean
`log1p(original counts)` across the sampled pixels, including zeros, for each
displayed peak. Correlation distance and average linkage use alphabetical slide
order as input; constant profiles are appended alphabetically. Every stage uses
this same slide order, preserving coordinate order within each slide.
`heatmap_slide_linkage.npy` saves the slide linkage; `heatmap.json` records its
input slides, clustering details, display order and block boundaries.
`heatmap_pixels.csv` records source and export rows in display order.

Labeled figures use a 14 × 24-inch portrait layout and Turbo colors, with centered
slide labels, separators, and an intensity colorbar. Dendrograms are hidden and
their space is given to the heatmap. Stage colors share the 99.5th percentile of
positive sampled counts of the displayed peaks as their upper limit; `log1p` is a
display transform only.
Old `X` has a separately identified scale calculated on the same displayed subset.
Native-cell PNGs contain one column per displayed peak and one row per sampled
pixel (500 × 4,000 for the 40-slide cohort), supporting full-resolution HTML zoom.

`removal_summary.csv` reports local removals within retained peaks, per slide and
pooled, separately for lesion and whole tissue. The main percentage divides the
number removed by positive measurements before local cleaning. Secondary
percentages divide by all entries (including zeros) and by scaled intensity mass.
`regional_feature_removal.csv` and `peak_removal_summary.csv` give per-slide and
pooled per-peak results. Peak selection is reported separately.

`threshold_placement.csv` and its pooled table give exact measurement and intensity
fractions at each boundary; distribution/CDF curves use recorded histogram bins.
Zero fractions are explicit. Raw selection thresholds and reference-adjusted
local thresholds use different units: a local cutoff of 0.5 corresponds to
`0.5 * slide_multiplier` in original counts. `slide_thresholds.csv` records that
mapping. `local_sensitivity.csv` assesses the same frozen spatial support at all
configured cutoffs, separately for lesion and whole tissue. Weak eligibility is
not the same as actual removal.

Existing validation-summary fields are preserved when their preprocessing
fingerprint matches; report-derived fields are refreshed.

Regenerate the HTML, heatmaps and distribution figures without refitting:

```bash
python -m msi_preprocess.report --fit data/PDAC/Processed/msi_preprocess/fit.json
```

Add the two whole-tissue spatial TIC grids with `--tic`:

```bash
python -m msi_preprocess.report --fit data/PDAC/Processed/msi_preprocess/fit.json --tic
```

`figures/tic_counts_all_slides.png` and `figures/tic_cleaned_all_slides.png`
each show every slide in alphabetical order. Both sum **all retained features**
at **every tissue pixel** from the exported `tissue.h5ad`: `layers/counts` before
slide scaling/cleaning and `X` after it. The 500-peak and sampled-pixel heatmap
subsets do not apply. Sparse blocks are summed in float64 before applying the
display-only `log1p` transform. Counts already contain RMS normalization.

Both figures use the same Turbo scale from zero to the positive counts TIC's
pooled 99.5th percentile (`report.upper_percentile`). Higher values saturate;
missing tissue is white and measured zeros use the lowest Turbo color. Panels
preserve native pixel coordinates and tissue gaps. The 40-slide layout
uses five columns and eight rows at 20 × 24 inches, 250 DPI (5,000 × 6,000 pixels).
`tic_maps.json` records sources, fingerprint, feature/pixel counts, slide order,
shared limits, and per-slide TIC totals and saturation. Subsequent report runs
include these figures when their saved fingerprint matches, without recomputing
TIC unless `--tic` is requested. Fitted parameters and H5AD files remain unchanged.

DANN still uses a 90%/10% model split. Because preprocessing fits all retained lesion MSI
and application uses whole-image MSI context, validation is not independent of
that context. Biological labels are used only for downstream descriptive reports.

## Manual cohort QC and replacement checks

`exclude_slides` is optional and defaults to an empty mapping. This study excludes
549857-19 and 549857-22 for severely depleted lesion spectra, and 546332-17 for
spatial striping (the corrected ID for the earlier 549332-17 reference). These
are manual decisions from signal summaries and heatmaps, without a confirmed
technical cause. IDs must exist in both source datasets and leave a nonempty
matching cohort. `cohort.csv` records patients, decisions, reasons and source row
counts. All fitting calculations use retained slides only.

The exports contain 40 slides from 19 patients: 1,665,807 lesion pixels and
5,468,414 whole-tissue pixels. Original relative row order, observation names,
coordinates and feature order are preserved. H&E embeddings are copied in row
blocks; observation pairwise arrays are subset on both axes. Unused categories
and excluded `uns.spatial` entries are removed. `obs.msi_source_row` identifies
each observation's row in its respective immutable original input. Reports use
these mappings and verify pixel keys before comparing raw and cleaned values.

Replacement regenerates maps, both TIC grids and sampled heatmaps, checks every
exported retained count against its source and every cleaned tissue nonzero,
then checks all lesion/tissue matches. Capped DANN training is followed by capped
latent/calibration analysis, spatial inference and latent variance analysis.
Full-tissue measured-peak analysis is disabled for real-data smoke runs and is
exercised by the synthetic workflow test. The previous figures, signal summaries,
fit provenance and earlier parameter archive are archived before promotion.

DANN uses the preprocessing cohort as its exclusion record. Domains, class
weights and discriminator width derive from observed retained slides; the seeded
90%/10% within-slide split is unchanged. Checkpoints, analysis CSV sidecars,
Parquet exports and caches carry preprocessing provenance. Schema 3 consumers
reject incompatible artifacts even if names or widths match. Full training and
production analysis remain separate; these checks establish pipeline compatibility.

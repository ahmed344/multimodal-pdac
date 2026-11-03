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
python -m msi_preprocess.verify --fit data/PDAC/Processed/msi_preprocess/fit.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q msi_preprocess/tests
```

Use the configured scientific/devcontainer environment. Run from the repository
root. The input/output paths and budgets are in `config.yaml`. Source files are
opened read-only, and existing exports or complete fits are never overwritten.
An interrupted fit reuses completed per-slide QC caches only when source
identities and settings match. Application removes incomplete output files on
failure. Change `output` to run a different stage combination or refit.

## Fitting and image context

The fitting adapter calls DANN's `stratified_split_indices`, with the categorical
slide order, seed, and fractions from the specified DANN YAML. It uses the full,
uncapped training partition (normally 90%), before smoke-test or label caps.
Only training counts, slide/patient identities, and integer coordinates enter
the fit. Fitting neighborhoods contain training pixels only. Held-out pixels
cannot supply anchor statistics, peak-selection evidence, or reference draws.

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
   are clipped to [0.5, 2], with the raw ratio, usable-anchor count, and reached
   limit recorded. Zero values stay zero.
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
  training rows, decisions, patient references, multipliers, spatial envelopes,
  and a SHA-256 preprocessing fingerprint with a parameter-file checksum.
- `peak_decisions.csv`, `slide_multipliers.csv`, and per-slide fitting CSVs:
  auditable selection/support evidence and correction limits.
- `tissue.h5ad`, `lesion.h5ad`: sparse `X`, original retained-feature counts,
  preserved observation/embedding metadata and ordered feature annotations,
  `obs.msi_removed_measurements`, original feature indices, and
  `uns.msi_preprocess` with fitted parameters and provenance.
- `report.html`, `figures/`, and CSVs: counts versus old `X` versus selection,
  integration, and cleaning; intensity histograms, slide-associated variance,
  threshold sensitivity, per-feature removals, and matched spatial maps.
  Map examples come from fitted anchors, globally supported peaks, localized
  patches, weak retained peaks, and removed peaks. Native-grid rendering masks
  gaps and does not interpolate pixel values. To regenerate maps independently,
  run `python -m msi_preprocess.report --fit <output>/fit.json`.
  CD8/stroma associations are computed **after** cleaning, separately for
  training and validation. They are diagnostics of structural preservation;
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
  sampled lesion/tissue agreement, and the DANN single-log1p loader check. Full
  and sampled coverage are recorded separately.

`--update-dann-config` patches only `data.path`, `data.context_path`,
`model.num_peaks`, and `data.preprocessing_fingerprint`, preserving unrelated
YAML and comments. It verifies `matrix_key: X`, `intensity_transform: log1p`, and
that no resume checkpoint was requested. DANN validates the fingerprint and
uncapped training split; checkpoint identities include preprocessing provenance.
Changed fingerprints require fresh training. Existing unprocessed inputs and
their checkpoint contracts remain supported.

These thresholds are operational settings, **not a measured instrumental noise
floor**. The pipeline cannot recover information lost during RMS normalization
or feature alignment. A short DANN smoke run validates data/model plumbing; it
does not establish biological preservation, CD8 calibration, or latent quality.

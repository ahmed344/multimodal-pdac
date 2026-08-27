# IHC multivariate-normal model

`ihc_mvn` is a standalone package for predicting four ordered IHC density
targets from sparse MALDI MSI counts:

1. `Density_Tumor`
2. `Density_Stroma`
3. `Density_Collagen`
4. `Density_CD8`

It owns its configuration, sparse loading, preprocessing, target construction,
model, and loss. It does not import or reuse `dann`.

## Mathematical summary

For pixel \(i\), the sparse encoder maps active MSI peaks to a latent vector:

\[
z_i =
\operatorname{MLP}_{agg}\left[
\frac{1}{|A_i|}
\sum_{p\in A_i}
\operatorname{MLP}_{peak}\left(x_{ip}e_p\right)
\right].
\]

The model has a factorized hurdle branch and a multivariate Gaussian positive
branch. For target \(j\),

\[
\Pr(k_{ij}>0)=\operatorname{logit}^{-1}(\eta_{ij}),
\]

where the total logit and standardized positive-coordinate mean are the sums
of MSI-only predictions, patient random intercepts, and slide random
intercepts. Unknown patients and slides receive zero random effects.

The positive branch uses row-specific marginal scales and one global
correlation matrix:

\[
\Sigma_i = D_i R D_i,\qquad D_i=\operatorname{diag}(\sigma_i).
\]

`GlobalCorrelation` constructs a positive-definite covariance from a learned
Cholesky factor and normalizes it to a correlation matrix. The positive loss
is the exact marginal multivariate-normal negative log likelihood for whichever
subset of the four targets is positive. There are 16 possible masks, including
the all-zero mask, whose Gaussian contribution is exactly zero. The objective
adds binary cross entropy over all hurdle indicators and optional normalized
priors for non-centered patient and slide random effects.

The CD8 conditional diagnostic uses standard Gaussian conditioning on any
available subset of tumor, stroma, and collagen. Its excess score is

\[
z_{\mathrm{CD8}} =
\frac{y_{\mathrm{CD8}}-\mathbb{E}[Y_{\mathrm{CD8}}\mid Y_C]}
{\sqrt{\operatorname{Var}(Y_{\mathrm{CD8}}\mid Y_C)}}.
\]

## Frozen sparse preprocessing

The input matrix must be CSR encoded in `adata.layers["counts"]`.

1. Stored counts are transformed with `log1p`. Implicit sparse zeros remain
   zero because `log1p(0) = 0`.
2. A separate standard deviation is fitted for every slide and feature.
   Variance is population variance,
   \(E[x^2]-E[x]^2\), over all selected training rows in that slide, including
   implicit zeros. This matches `StandardScaler(with_mean=False)` scale
   semantics.
3. Values are divided by their fitted scales only. Means are never subtracted,
   so sparse rows remain sparse.
4. The scaler is fitted only on the actual selected training rows, after split
   caps. Validation and test rows never contribute.
5. The complete scaler state must be persisted in the training checkpoint.
   Inference restores this state and must never refit it.
6. A known slide uses its frozen per-slide scales. An unknown slide uses the
   frozen global per-feature scales fitted on all selected training rows.
   A slide-constant feature falls back to that global scale; a globally
   constant or unseen training feature uses unit scale.

Feature names and order are part of the fitted schema. Inference must reject
duplicate feature names, missing or extra features, and a reordered feature
axis unless an integration layer explicitly restores the training order before
constructing the dataset.

## Haldane-Anscombe targets

Each density is converted to a count with

\[
k_j=\operatorname{round}(36100\,y_j).
\]

Tumor uses denominator \(n_T=36100\). The remaining targets use the residual
extratumoral compartment while remaining valid when a component count exceeds
that residual:

\[
n_j=\max(36100-k_T,\;k_j),\qquad j\in\{S,C,CD8\}.
\]

The finite coordinate is

\[
q_j=\log\frac{k_j+0.5}{n_j-k_j+0.5}.
\]

Presence is defined from the uncorrected count, `k_j > 0`; the correction does
not turn structural zeros into positives. Coordinate means and population
standard deviations are fitted independently per target from positive
training rows only, then frozen and persisted.

## Split interpretation and leakage

The configured split is random 80/10/10 within each slide. It is deterministic
for a fixed seed and produces disjoint row sets, but it is deliberately not a
patient-held-out or slide-held-out evaluation. Pixels from the same patient,
slide, specimen, and nearby tissue regions may occur in every split. Reported
validation/test performance therefore measures interpolation within observed
slides and may be optimistic for new patients, slides, or tissues. Do not
describe it as independent external generalization.

## Commands

Train from the repository root:

```bash
python -m ihc_mvn.train --config ihc_mvn/config.yaml
```

Run a capped one-epoch real-data smoke train:

```bash
python -m ihc_mvn.train --config ihc_mvn/config.yaml --smoke-test
```

Run ordered inference on the configured tissue file:

```bash
python -m ihc_mvn.infer \
  --checkpoint data/PDAC/Results/ihc_mvn/best.pt \
  --overwrite
```

Run held-out checkpoint analysis or spatial plotting independently:

```bash
python -m ihc_mvn.analyze --config ihc_mvn/config.yaml
python -m ihc_mvn.spatial_heatmaps --config ihc_mvn/config.yaml
```

By default, the ordinary training CLI automatically runs held-out analysis,
tissue-wide inference, and spatial visualization after `best.pt` has been
selected and test metrics have been recorded. The `post_training` configuration
section enables or disables the complete workflow and each individual stage.
`fail_on_error: true` makes a failed post-training stage return a failing
command status without deleting the already completed checkpoints or history.
Smoke training applies caps to analysis, peak-activity scanning, and tissue
inference as well as to the training splits.

Pass `--input data/PDAC/Raw/adata_assembled.h5ad` to infer on the training
AnnData instead. Run the fast synthetic contract tests with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q ihc_mvn/tests
```

Compile the tests without executing them:

```bash
python -m compileall -q ihc_mvn/tests
```

## Checkpoints and outputs

The configured output root is
`/workspaces/multimodal-pdac/data/PDAC/Results/ihc_mvn`. Training writes:

- model and optimizer state;
- resolved configuration and ordered target names;
- ordered feature names;
- slide and patient category mappings;
- frozen `SparseFeatureScaler.to_state()` state;
- frozen `TargetStandardizer.to_state()` state;
- split indices and training history.

Inference writes an ordered prediction Parquet and a separate latent Parquet.
The prediction artifact reports presence probabilities, MSI-only,
patient-adjusted and fully adjusted Gaussian means, marginal uncertainty,
density-scale positive medians/intervals, the shared correlations and
conditional-CD8 diagnostics. Row positions and possibly duplicated observation
names remain aligned exactly with the source H5AD.

Held-out analysis writes:

- `training_curves.png` beside `history.csv`, marking the selected best epoch;
- `analysis/umap/latent_umap.csv`, group UMAPs, one six-panel MVN UMAP per
  target, and a conditional-CD8 excess UMAP;
- `analysis/peptide_similarity_heatmap.png`,
  `embedding_cosine_similarity.npy`, and `peptide_families.csv`;
- learned and empirical target-correlation tables/heatmaps;
- patient and slide random-intercept tables/heatmaps;
- hurdle reliability, positive prediction, interval coverage, and
  mask-specific Mahalanobis diagnostics.

Spatial inference continues to write `inference/inference.parquet` and
`inference/inference_latent.parquet`. Spatial visualization validates both
artifacts against the tissue AnnData row positions and observation names,
then writes one figure per target under `spatial/`. Panels distinguish
MSI-only, patient-adjusted, and fully adjusted positive medians and include
presence probability, uncertainty width, observed truth/residuals when
available, and HES when present. A separate conditional-CD8 excess map is
written when that diagnostic is finite.

## Known and unknown inference groups

Known slides use frozen per-slide preprocessing and their learned slide random
effects. Known patients use their learned patient random effects. Unknown
slides use global frozen training scales and zero slide random effects; unknown
patients use zero patient random effects. No inference data, whether labeled or
unlabeled, may update feature scales, target standardization, category
mappings, model weights, or random effects.

## Visualization scope

The visualization pipeline is downstream of the fitted statistical model. It
does not add HES, spatial coordinates, neighborhoods, or Fourier features as
model inputs and does not refit frozen preprocessing. HES and coordinates are
used only to render tissue predictions after inference. Existing compatible
checkpoints and `history.csv` files can therefore be analyzed without
retraining; spatial figures additionally require compatible inference
Parquets.

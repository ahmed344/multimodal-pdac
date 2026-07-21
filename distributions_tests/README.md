# Bounded density distribution tests

This analysis compares seven bounded models for four density columns, both
globally and within each `batch`:

- beta
- generalized Beta of the first kind (GB1)
- Kumaraswamy
- logit-normal
- skew-logit-normal
- Johnson SB
- normal truncated to `[0, 1]`

Standard Beta already permits left and right skew. GB1 adds a third positive
shape parameter, while skew-logit-normal replaces the Gaussian distribution on
the logit scale with a skew-normal distribution.

Every candidate is a hurdle model. It estimates `P(X = 0)` separately, so
`P(X > 0)` directly represents existence. The few observations that are
exactly one do not receive a separate probability parameter. Instead, the
Smithson-Verkuilen boundary correction is applied to all positive observations,
mapping them into `(0, 1)` before fitting the continuous positive-density
family. For a beta continuous component, this is a standard zero-inflated
Beta model with a boundary correction.

## Run

From the repository root:

```bash
python distributions_tests/fit_density_distributions.py
```

The defaults read:

`data/PDAC/Raw/adata_assembled.h5ad`

and write to:

`data/PDAC/Results/distributions_tests/`

Use `--help` to see options. The default fitting cap of 250,000 observations
per density and scope mainly affects global fits; most batches use all their
observations. Set `--max-fit-observations 0` to disable the cap.

## Selection and diagnostics

Parameters are estimated on a deterministic 80% training split. The selected
family minimizes negative log likelihood per observation on the held-out 20%.
The zero-versus-positive hurdle probabilities are included in the likelihood
and are identical in form across candidates.

`candidate_fits.csv` also reports:

- AIC and BIC on the training sample
- held-out Kolmogorov-Smirnov distance for the positive continuous component
- convergence status and any fitting error
- JSON-encoded fitted parameters

The KS value is an effect-size diagnostic, not a reported p-value. At this
dataset's sample size, formal goodness-of-fit tests can reject models for
scientifically negligible deviations.

## Outputs

- `descriptive_statistics.csv`: ranges, quantiles, and zero/existence/one rates
- `candidate_fits.csv`: all seven fits for every global and batch group
- `selected_models.csv`: held-out likelihood winner for every group
- `global_distribution_fits.png`: global interior histograms and fitted curves
- `batch_distribution_heatmaps.png`: batch zero rates, winners, and fit quality
- `summary.md`: concise global and per-batch interpretation

## Logit-transformed positive-value analysis

The separate logit-scale analysis excludes zeros, applies the same
Smithson-Verkuilen correction to positive values, and then applies the logit
transform. Run it from the repository root:

```bash
python distributions_tests/fit_logit_positive_distributions.py
```

Results are written by default to:

`data/PDAC/Results/distributions_tests/logit_positive/`

It compares real-line distributions with a small number of parameters:

- Normal: location and scale (2)
- logistic: location and scale (2)
- Student-t: degrees of freedom, location, and scale (3)
- generalized normal: shape, location, and scale (3)
- skew-normal: skew shape, location, and scale (3)
- Johnson SU: two shape parameters, location, and scale (4)

Beta is intentionally omitted: after the logit transform the observations have
real-line support, whereas Beta has bounded support. As in the bounded
analysis, parameters are fit on a deterministic 80% split and candidates are
selected by held-out negative log likelihood.

The logit-scale output contains:

- `transformed_descriptive_statistics.csv`
- `candidate_fits.csv`
- `selected_models.csv`
- `global_logit_distribution_diagnostics.png`
- `batch_logit_distribution_heatmaps.png`
- `summary.md`

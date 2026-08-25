# cbmr-proofs

Machine-checked derivations behind a closed-form observed information matrix and covariance step
for NiMARE's coordinate-based meta-regression (CBMR).

## Why

`CBMRModel.information_matrix` differentiates the log-likelihood twice with
`torch.func.hessian`. That is `jacfwd(jacrev(...))`, and forward mode pushes every tangent
through the likelihood at once, so the intermediate has shape
`(n_parameters x n_patterns x n_voxels)`. On a 21-group model over 8,444 voxels that is 9.7 GB;
on the real 21-group diagnosis pool (8,420 coefficients, 21,789 voxels) it is **28.7 GB**, which
is an out-of-memory kill rather than a slow fit.

All three of NiMARE's observation distributions have a closed-form Hessian. Deriving it removes
the intermediate entirely and exposes the next bottleneck — a diagnostic SVD costing more than
the quantity it diagnoses. This repository is the proof that the replacements are the same
mathematics.

Measured on the real 21-group pool, complete run (load, fit, covariance, 20 one-versus-rest
contrasts):

| | time | peak RSS |
|---|---|---|
| stock NiMARE | — | 28.7 GB intermediate, does not run |
| chunked autodiff | 769 s | 5.49 GB |
| closed form | **44 s** | **2.67 GB** |

## What is proved

59 claims, every residual exactly zero.

| module | proves |
|---|---|
| `proofs/poisson.py` | the Poisson Hessian, and that it does not depend on the foci at all |
| `proofs/negative_binomial.py` | the NB Hessian, via the `(R, A)` reparameterisation |
| `proofs/clustered_negative_binomial.py` | the ClusteredNB Hessian, diagonal plus rank one |
| `proofs/covariance.py` | structural separation, blockwise inverse, Schur/bordered inverse, block-diagonal spectrum, and the reduction of the condition number to a symmetric eigenproblem |

Nothing here is numerical. Each claim is a residual between what NiMARE's likelihood
differentiates to and what the closed form asserts, simplified by `sympy` to the symbol `0`.

## Running it

```bash
pip install -r requirements.txt
make proofs     # verify; nonzero exit if any claim fails
make papers     # verify, then build both PDFs (needs pdflatex)
```

`run_proofs.py` also writes `docs/generated/*.tex`. Those fragments are rendered from the same
sympy objects the claims are checked against, so the papers cannot drift from the verification —
a sign flip or a moved index would have to change both at once. CI runs the proofs and fails if
the checked-in fragments are stale.

## Papers

- `docs/cbmr_hessian.tex` — the three closed-form Hessians. One lemma covers all of them: the
  second differential in the log-intensity is diagonal (Poisson, NB) or diagonal-plus-rank-one
  (ClusteredNB), and the chain rule through the log-bilinear predictor turns each into
  `n_patterns` symmetric rank-updates of width `n_bases`.
- `docs/cbmr_covariance.tex` — the covariance step: which designs let the information matrix
  separate, the four inversion strategies, and what it costs in time and memory.

Both end with an appendix of the generated fragments.

## A result that came from measuring rather than deriving

LAPACK's `pocon` condition estimator is the fastest route by a wide margin, and the standard
bound `n^-1 k1 <= k2 <= n k1` makes it look harmless. Measured here it reports 90.09 against a
true 2-norm condition number of 5.02, and 2.42e5 against 1657.7 — errors of 18x and 146x, both
comfortably inside the bound. A bound that admits a 264x error admits behaviour that cannot be
shown to a user as a condition number. What generalises instead is simpler: the information
matrix is symmetric, so its singular values are its absolute eigenvalues and a symmetric
eigensolver gives the exact same number 3-4x faster than the SVD.

## Implementation

The corresponding NiMARE changes live on the `cbmr-closed-form-information` branch of
<https://github.com/jdkent/NiMARE>.

## Scope

The derivations hold with the overdispersion parameters held fixed at their fitted values, which
is what `information_matrix` does and what CBMR reports regression standard errors from. They do
not propagate uncertainty in the overdispersion.

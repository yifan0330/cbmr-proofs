# cbmr-proofs

Machine-checked derivations for a closed-form information matrix and covariance step in
NiMARE's coordinate-based meta-regression (CBMR).

## Why

`CBMRModel.information_matrix` differentiates the log-likelihood twice with
`torch.func.hessian`. That is `jacfwd(jacrev(...))`, and forward mode carries one tangent per
parameter at once. Its intermediate has shape `(n_parameters, n_patterns, n_voxels)`. On the
real 21-group diagnosis pool — 8,420 coefficients, 21,789 voxels — that is **28.7 GB**. The
model cannot be fitted at all.

All three of NiMARE's distributions have a closed-form Hessian. Using it removes the
intermediate. That exposes the next problem: a diagnostic SVD that costs more than the inverse
it checks. This repository proves the replacements are the same mathematics.

Complete run on the real pool, including 20 one-versus-rest contrasts:

| | time | peak memory |
|---|---|---|
| stock NiMARE | — | 28.7 GB, does not run |
| chunked autodiff | 769 s | 5.49 GB |
| closed form | **44 s** | **2.67 GB** |

## What is proved

59 claims. Every residual is exactly zero.

| module | proves |
|---|---|
| `proofs/poisson.py` | the Poisson Hessian, and that the counts drop out of it |
| `proofs/negative_binomial.py` | the NB Hessian, using the `(R, A)` reparameterisation |
| `proofs/clustered_negative_binomial.py` | the ClusteredNB Hessian, diagonal plus rank one |
| `proofs/covariance.py` | structural separation, the blockwise inverse, the Schur inverse, the block-diagonal spectrum, and the condition number as a symmetric eigenproblem |

Nothing here is numerical. Each claim is a residual between what NiMARE's likelihood
differentiates to and what the closed form says. `sympy` reduces it to the symbol `0`.

## Running it

```bash
pip install -r requirements.txt
make proofs     # verify; exits nonzero if any claim fails
make papers     # verify, then build both PDFs (needs pdflatex)
```

`run_proofs.py` also writes `docs/generated/*.tex`. Those fragments are rendered from the same
sympy objects the claims are checked against, so the papers cannot drift from the proofs. CI runs
the proofs and fails if the checked-in fragments are stale.

## Papers

- `docs/cbmr_hessian.tex` — the three Hessians. One lemma covers all of them.
- `docs/cbmr_covariance.tex` — which designs let the matrix separate, the four inversion
  strategies, and what each costs in time and memory.

Both end with an appendix of the generated fragments.

## One result came from measuring, not deriving

LAPACK's `pocon` condition estimator is much the fastest route, and the bound
`k1/n <= k2 <= n*k1` makes it look safe. Measured here, it reports 90.09 where the true 2-norm
condition number is 5.02, and 2.42e5 where it is 1657.7. Both are inside the bound. A bound that
allows a 264x error allows numbers you cannot show a user.

What generalises instead is simpler. The information matrix is symmetric, so its singular values
are its absolute eigenvalues. A symmetric eigensolver gives the same number 3-4x faster than the
SVD.

## Implementation

The matching NiMARE changes are on the `cbmr-closed-form-information` branch of
<https://github.com/jdkent/NiMARE>.

## Scope

The derivations hold with the overdispersion parameters fixed at their fitted values. That is
what `information_matrix` does and what CBMR reports standard errors from. They do not carry
uncertainty in the overdispersion.

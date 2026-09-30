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

The original four suites contain 59 claims. The integrated appendix adds checks
for `docs/cbmr_paper.pdf` and the published CBMR article `docs/imag.a.1057.pdf`
in one consistent notation; the runner reports the combined count.
Every accepted residual is exactly zero.

| module | proves |
|---|---|
| `proofs/poisson.py` | the Poisson Hessian, and that the counts drop out of it |
| `proofs/negative_binomial.py` | the NB Hessian, using the `(R, A)` reparameterisation |
| `proofs/clustered_negative_binomial.py` | the ClusteredNB Hessian, diagonal plus rank one |
| `proofs/covariance.py` | structural separation, the blockwise inverse, the Schur inverse, the block-diagonal spectrum, and the condition number as a symmetric eigenproblem |
| `proofs/appendix.py` | the paper's independent NB score and information weights, affine and separable block assembly, Kronecker factorisation, limits, and contrast/sandwich identities |
| `proofs/published_cbmr.py` | grouped design and Poisson compression, NB moment matching, clustered-NB factorisation and curvature, roughness penalties, shared-covariate uncertainty, and delta-method Jacobians |

Nothing here is numerical. Each claim checks an exact symbolic identity, using scalar
likelihood derivatives or matrix algebra. `sympy` reduces the residual to zero.
Most matrix checks use fixed small dimensions with symbolic entries; the Schur and
Kronecker inverse checks use formal matrix symbols. These are computer-algebra checks,
not proof-assistant certification or direct tests of NiMARE's implementation.

## Running it

```bash
pip install -r requirements.txt
make proofs     # verify; exits nonzero if any claim fails
make appendix   # verify, then build docs/cbmr_appendix.pdf (needs pdflatex)
make expanded-appendix  # build the separate step-by-step derivations (needs pdflatex)
make paper-appendices   # build separate expanded appendices for each reference paper
make papers     # verify, then build all three PDFs (needs pdflatex)
```

## Lean proof checking

A pinned Lean/Mathlib project formalises a **subset**, not yet all derivations,
of the integrated appendix. With Elan installed:

```bash
make lean-deps  # download the pinned dependencies and compiled Mathlib cache
make lean      # compile theorem proofs and audit their axiom dependencies
make lean-complete  # additionally fail if any inventoried derivation remains unproved
```

The audit rejects admitted proofs and additional axioms, while permitting the
standard `propext`, `Classical.choice` and `Quot.sound` foundations.
See [the coverage inventory](docs/lean_coverage.md) for the theorem modules,
their assumptions, and the remaining derivative, probability and inference
proof obligations. A successful Lean build is not a claim of full-appendix coverage.
CI checks both the SymPy suites and the Lean development.
The equation-level inventory is [docs/lean_coverage.json](docs/lean_coverage.json).
It covers all 54 displayed mathematical blocks and additional prose derivations,
checks that referenced theorems were actually audited, and detects changes to the
displayed mathematics, notation preamble or generated formula macros. This is a reviewed
LaTeX-to-Lean correspondence, not an automatic proof that the two languages agree.
The latest audit output is saved in `.lake/lean-audit.log`; the coverage
check writes `.lake/lean-coverage-report.json`, including source lines and remaining
obligations. A successful `make lean` may still report incomplete coverage.

## Generated documentation

`run_proofs.py` also writes `docs/generated/*.tex`, including the appendix's checked scalar
formulas and verification summary. Generated expressions come from the checked SymPy objects;
handwritten exposition is not mechanically certified. CI runs the proofs and fails if the
checked-in fragments are stale.

## Papers

- `docs/cbmr_paper_appendix.pdf` — self-contained expanded appendix for
  `cbmr_paper.pdf` (*Scalable Spatial Model for Brain MRI Lesion Mask Data*).
  Uses its spatially varying covariate maps and binary-image setup, with Poisson
  and independent-cell NB working likelihoods, separable initialization,
  Kronecker preconditioning, the subject-centered Taylor derivation, and
  subject-robust inference. Includes a manuscript notation crosswalk,
  matrix-free Hessian products, QR sandwich calculations, and an expanded,
  corrected derivation of Supplement S2.3's toy slope-map variance.
  Distinguishes normalized binary models from count
  working likelihoods and flags formula qualifications explicitly. Its editable
  source is `docs/cbmr_paper_appendix.tex`; build with `make cbmr-paper-appendix`.
- `docs/imag_a_1057_appendix.pdf` — self-contained expanded appendix for
  `imag.a.1057.pdf` (*CBMR: Coordinate-based meta-regression for group and covariate
  inference*). Uses its group-specific spatial maps and globally constant study
  covariates, with Poisson, moment-matched aggregated NB, clustered NB, roughness
  penalties, and group/global covariance and inference. Includes a paper-native
  notation crosswalk, explicit aggregation and study-total derivations, and
  qualifications about totals-only covariate identification. Its editable source is
  `docs/imag_a_1057_appendix.tex`; build with `make imag-appendix`.

`make paper-appendices` builds both. These handwritten, paper-specific derivations
are intended for manual review, not claimed to be fully formally certified.
The reference papers and combined expanded appendix are preserved.

- `docs/cbmr_expanded_appendix.pdf` — separate, expanded derivations for manual
  mathematical review, with editable sources `docs/cbmr_expanded_appendix.tex`
  and `docs/expanded_covariance.tex`. Includes scalar differentiation and
  coefficient-by-coefficient Hessian assembly for constant and spatially varying
  effects; distinguishes independent, aggregated, and clustered NB likelihoods;
  derives structured covariance, nuisance adjustment, penalties, robust inference,
  and contrasts. Worked small examples and explicit approximation boundaries aid
  review. This handwritten expansion is not covered in full by the existing
  SymPy or Lean checks. `make expanded-appendix` leaves the other PDFs unchanged.

- `docs/cbmr_hessian.tex` — the three Hessians. One lemma covers all of them.
- `docs/cbmr_covariance.tex` — which designs let the matrix separate, the four inversion
  strategies, and what each costs in time and memory.
- `docs/cbmr_appendix.tex` — integrated information, covariance and inference summary using
  the notation of `docs/cbmr_paper.pdf`, incorporating the published multi-group CBMR
  model in `docs/imag.a.1057.pdf` (doi:10.1162/IMAG.a.1057). A contents page,
  notation translation table and matched example distinguish **GC-CBMR** (covariate
  effects constant across voxels) from **SV-CBMR** (spatial coefficient maps).
  A spatially varying group baseline is not a spatially varying covariate effect;
  these predictor choices are separate from the Poisson and NB likelihood choices.
  The appendix covers exact versus approximate reductions, penalties, structured
  covariance, shared-global uncertainty, contrasts and bootstrap limitations.

The two original notes end with generated proof fragments. The integrated appendix includes
generated scalar formulas and a report of all proof suites.
The original PDFs are reference inputs and are not modified. The supplied published
article refers to a separate supplement; its detailed roughness operator is not assumed
to be available here, so the appendix specifies its quadratic-penalty convention explicitly.

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

### Opt-in numerical implementation

[`cbmr_improved/`](cbmr_improved/) is an independent, import-isolated copy of the
supplied `cbmr_code/` implementation. **`cbmr_code/` is unchanged.** Internal
imports resolve to `cbmr_improved`, without monkey-patching the originals or
the installed NiMARE package. The original already implements the three
closed-form Hessians and block/Schur inversion; the new version carries that
structure through assembly and inference instead of first allocating dense
information and covariance matrices.

| Improvement | Derivation |
|---|---|
| Assemble only connected spatial blocks and the global border for all three existing likelihoods | `proofs/poisson.py`, `negative_binomial.py`, `clustered_negative_binomial.py`, `covariance.py` |
| Factor once, solve for requested contrasts, and compute coefficient variances without a full inverse | Schur/block identities in `proofs/covariance.py`; contrast identities in `proofs/appendix.py` |
| Preserve the low-rank covariance between spatial groups induced by shared global covariates | `proofs/published_cbmr.py` |
| Compute Poisson experiment scores from sparse count projections, and sandwich covariance as a Gram matrix of solved scores | Affine scores and sandwich identities in `proofs/appendix.py` |
| Evaluate moment-matched NB probabilities in log space and use log-scale curvature plus gamma recurrences near the Poisson limit | `(R, A)` derivatives in `proofs/negative_binomial.py` |

From the repository root:

```bash
pip install -r requirements-numerical.txt
make numerical
```

```python
import numpy as np
import pandas as pd
from cbmr_improved import CBMRModel, CBMRPredictor, Design
from cbmr_improved.terms import bind

annotations = pd.DataFrame({"group": ["a", "a", "b", "b"]})
bases = np.column_stack([np.ones(3), [-1.0, 0.0, 1.0]])
foci = np.array([[2, 1, 3], [1, 2, 2], [3, 2, 1], [2, 3, 2]])
design = bind(Design.from_formula("~ s(group)"), annotations)
model = CBMRModel(CBMRPredictor(design, bases), "poisson").fit(foci)

information = model.information_operator(foci)
C = np.array([[1.0, 0.0, -1.0, 0.0]])
contrast_covariance = information.contrast_covariance(C)
coefficient_variances = information.diagonal()
standard_errors_by_term = model.standard_errors(foci)
```

`model.information_matrix(foci)` and `model.covariance(foci)` remain dense
compatibility APIs. `information.solve(rhs)` and
`information.sandwich_covariance(scores, contrast=C)` avoid a full inverse;
the latter expects **actual likelihood score contributions** with one row
per independent unit. For Poisson experiment clustering these are available
as `poisson_cluster_scores(model, foci)`. Operators are snapshots: reuse one
for multiple contrasts, and request a new one after modifying a fit or its
counts. `information.condition_number()` requests the symmetric spectral
diagnostic explicitly; it is not a hidden cubic-cost step in each solve.

The copied high-level `CBMR`, `CBMRResult`, and `evaluate_hypotheses` APIs
also use the improved implementation. They additionally require a NiMARE
installation with the interfaces used by the supplied source snapshot;
the numerical core above does not require NiMARE. Fisher hypothesis tests
project covariance before forming voxelwise statistics, including joint
hypotheses. The core and structured numerical checks run in CI separately
from the symbolic proofs.

**Scope and safeguards.** Dispersion is held fixed for inference, as in the
proofs and original code. Unknown distribution subclasses use row-wise
reverse-mode differentiation rather than borrowing a built-in Hessian.
Non-positive-definite block factorizations trigger a logged dense fallback;
singularity raises an error, never a pseudoinverse. Numerical ridge is
explicit, nonnegative, and applied consistently to Fisher and sandwich
bread, including HC3 leverage. The Poisson residual sandwich is rejected
for NB likelihoods: applying it to their observed information is not the
likelihood sandwich proved in the appendix.

The appendix's independent-cell NB model is **not** substituted for the
existing moment-matched NB likelihood. Penalties, spatially varying
predictors, and exact Kronecker separability have their own assumptions;
no penalty or factorization is imposed on a fitted model merely because
its formula appears in the proofs. NB shape derivatives use finite gamma
recurrences for integer pattern counts up to 4096, and the general
special-function identities otherwise. This improves small-dispersion
curvature without silently switching likelihoods to Poisson; it does not
guarantee accurate nuisance fitting at arbitrarily small dispersion.
The moment-matched likelihood tends to an **aggregated** Poisson likelihood:
its global curvature still differs from full Poisson when study allocation
carries covariate information. Singular or nonpositive contrast covariance
is reported as an error rather than converted into an apparently null result.
The numerical checks are regression evidence, not additional Lean proofs.

### Floating-point comparisons, performance and calibration

[`comparison/`](comparison/README.md) complements Lean and SymPy with independent
float64 likelihood/derivative references and comparisons against **both actual
source implementations**, without requiring NiMARE or editing either directory.
Run `make comparison` for the small seeded CI suite. Its JSON report includes
scores, Hessian block errors, directional finite differences, inverse residuals,
conditioning diagnostics, difficult cases and complete same-objective fits.

`make benchmark` runs small sequential benchmarks in separate processes, with
fixed thread counts and runtime/peak-memory tables. `make calibration` is a
separate Monte Carlo experiment for bias, coverage and false-positive rates.
Timing equality and derivative equality are not evidence of inferential
calibration; see the comparison guide for the model and dispersion assumptions.

## Scope

The derivations hold with the overdispersion parameters fixed at their fitted values. That is
what `information_matrix` does and what CBMR reports standard errors from. They do not carry
uncertainty in the overdispersion.

# Lean coverage of the CBMR appendix

**Status: partial formalisation, not a Lean proof of the entire appendix.**
`make lean` checks the included theorems. Its success does not close the
outstanding obligations below, and the SymPy claim count is not a Lean theorem count.

## Reproduce the checks

Install [Elan](https://github.com/leanprover/elan) and run:

```bash
make lean-deps
make lean
```

`lean-toolchain`, `lakefile.toml` and `lake-manifest.json` pin Lean, Mathlib and
their dependency revisions. The Makefile also recognises a repository-local
`.elan/bin/lake`; this avoids changing an existing user toolchain installation.
`make lean-deps` runs the Mathlib cache downloader through the Lean interpreter,
so it does not require the bundled Clang to run on an older Linux host.

The build elaborates proofs and checks their proof terms with Lean. The subsequent
`scripts/LeanAudit.lean` inspects theorem dependencies transitively and rejects
admissions (`sorryAx`) and additional axioms. Only Mathlib's usual foundational
axioms `propext`, `Classical.choice` and `Quot.sound` are accepted. This is not an
axiom-free constructive development, nor an independent recheck of the entire
downloaded Mathlib library.
The printed audit count includes compiler-generated equational/helper theorems;
it is not a one-to-one count of appendix derivations.

## Equation-level coverage and saved outcomes

`docs/lean_coverage.json` maps all 54 displayed mathematical blocks and 15
additional prose derivations to theorem names. `D01`--`D54` follow display order;
the generated report also gives current source line numbers. Entries distinguish
`proved`, `partial` and `pending`, with explicit outstanding obligations.
For a model specification, the mapping verifies its formal representation, not
the truth of the model for real observations. For an approximation, the intended
exact result concerns the specified linearisation, not exact nonlinear sampling
covariance.

`make lean` validates this inventory against its successful axiom-audit output.
It also detects changed displayed mathematics, notation macros and generated scalar formulas,
missing displays and references to unaudited theorems. The semantic correspondence
between a displayed formula and its referenced Lean theorem is reviewed by humans;
the checker does not interpret LaTeX as Lean.

After running it, inspect `.lake/lean-audit.log` for individual checked declarations
and `.lake/lean-coverage-report.json` for equation-level outcomes.
`make lean-complete` additionally requires **every inventory entry** to be `proved`;
it exits nonzero while any entry remains partial or pending. Empirical benchmarks,
model adequacy and bootstrap-calibration disclaimers are not formalised as
mathematical identities.

## Files and scope

| Lean file | Formalised material |
|---|---|
| `lean_proof_checks/Models.lean` | Finite-index predictor/weighted-entry algebra, GC factorisation, constant-function restriction, Poisson compression and allocation, scale non-identifiability, NB weight-grid nonseparability |
| `lean_proof_checks/Design.lean` | Arbitrary finite stacked GC/SV designs, row-major coefficient indexing, parameter dimensions, effect signs and coefficient-level identification shifts |
| `lean_proof_checks/Calculus.lean` | Real-variable affine, Poisson, independent-NB and scalar penalty derivatives; scalar affine mixed-derivative chain rule |
| `lean_proof_checks/Hessian.lean` | Actual finite-sum coordinate scores/Hessians, general vector affine chain rule, repeated-design compression, GC blocks and fixed-dispersion convexity |
| `lean_proof_checks/Assembly.lean` | Full subjectwise-score and Kronecker weighted-Gram matrix equalities, linked to actual mixed likelihood derivatives |
| `lean_proof_checks/GroupedPoisson.lean` | Arbitrary-group likelihood compression, allocation derivatives, coefficient scores and all spatial/global information blocks |
| `lean_proof_checks/Clustered.lean` | Actual clustered likelihood derivatives, projected and grouped information blocks, finite-sum likelihood collection and regression convexity |
| `lean_proof_checks/NegativeBinomial.lean` | NB matching-parameter algebra, higher-moment algebra, selected Gamma/aggregated-likelihood derivatives and clustered-curvature identities |
| `lean_proof_checks/Covariance.lean` | Schur and block inverse formulas, Kronecker Gram factorisation, determinant identities, covariance and sandwich matrix algebra |
| `lean_proof_checks/Structure.lean` | Supported design partitions and arbitrary unequal-sized block-family inverses, characteristic polynomials and spectra |
| `lean_proof_checks/Spectral.lean` | Actual induced Euclidean and one-norm conditioning, symmetric eigenvalue ratios, block extrema, singularity and spectral counterexamples |
| `lean_proof_checks/Penalty.lean` | Actual vector quadratic gradients/mixed Hessians, additive penalised derivatives, grouped roughness blocks, positive-semidefinite penalties and kernel-direction invariance |

Finite index sets replace fixed-size symbolic matrices wherever the theorem
statement permits. Positivity, nonzero denominators, differentiability and
invertibility appear explicitly as hypotheses. A theorem assuming differentiability
of a generic function is a chain-rule result, not a proof that every likelihood
satisfies those hypotheses. An algebraic covariance identity is not a statistical
sampling-distribution theorem.

## Complete appendix inventory and remaining obligations

The section references below follow the LaTeX source's section labels, so PDF
pagination changes do not change their meaning. Each row covers the displayed
equations and the surrounding derivations in that part of the appendix; it is
deliberately not a claim that every display has a completed Lean translation.

| Appendix area | Included results | Still required for full formalisation |
|---|---|---|
| S1: GC-CBMR versus SV-CBMR (`sec:notation`) | Predictor and effect identities; constant-function restriction; coefficient shift invariance; matrix/vectorisation correspondence for both designs and arbitrary group encodings | No remaining displayed design identity; statistical identification conditions are separate from these algebraic identities |
| S2: observed Hessian assembly (`sec:master`) | Actual finite-sum coordinate scores and mixed Hessians, weighted Gram entries, SV Kronecker entries, general vector affine chain rule and repeated-row aggregation | No remaining coordinate-derivative identity; numerical cost and implementation behaviour are separate |
| S3: independent Poisson and NB (`sec:families`) | Actual scalar likelihood/score differentiation and rational weight identities | Normalised distribution statements, binary Gamma-factor cancellation, expectations/score variances as probability theorems, and all Poisson limits |
| S3: separable information and NB nonseparability | Actual Poisson Hessian at a separable mean, Kronecker inverse, NB weight-grid determinant and nonzero result for unequal positive factors | Probabilistic expectation for the independent-NB Fisher weights |
| S4: grouped Poisson (`sec:group-poisson`) | Arbitrary-group compression, omitted allocation and its derivative, all coefficient scores/Hessian blocks and cross-group zeros | Probabilistic sufficient-statistic/allocation interpretation, as distinct from the checked likelihood identities |
| S4: aggregated NB (`sec:aggregate`) | Matching mean/variance/effective-dispersion algebra; actual directional log-intensity and shape derivatives of `Psi` for arbitrary voxelwise intensities | Remaining rate/second partials of `Psi`, all moderator derivatives of `rho` and `A`, their full composite Hessian, and probabilistic convolution/approximation statements |
| S4: clustered NB (`sec:clustered`) | Actual local and projected Hessians, all grouped coefficient blocks, likelihood collection, positive-curvature identity and full affine-regression convexity | Gamma-mixture probability model, integrated joint mass function, NB-total/multinomial factorisation and expectation theorems |
| S5: penalties (`sec:penalty`) | Full vector quadratic derivatives, grouped penalty blocks, kernel invariance and penalised matrix covariance identities | Statistical linearisation yielding the penalised sandwich; smoothing-bias/tuning-uncertainty statements |
| S6: structured covariance (`sec:covariance`) | Supported partitions, arbitrary block inverses/spectra, Schur and shared-group covariance, actual induced-norm spectral ratios and norm-equivalence bounds | The partition-support criterion is proved; a particular graph-finding or numerical factorisation implementation is not verified |
| S7: contrasts and sandwich (`sec:inference`) | Matrix sandwich/projection algebra | Random-vector covariance transformations, complete vector exponential Jacobian/delta theorem, Wald/chi-square asymptotics, robust estimating-equation theory, bootstrap/tail-calibration theory |
| S8: verification and empirical qualifications (`sec:verification`) | Build and axiom-audit tooling | Empirical timing, memory, foci thresholds and model adequacy are not consequences of algebraic Lean proofs; computational-complexity claims would need a separately formalised cost model |

The missing items are not represented by axioms or admitted theorems. They remain
documented proof obligations. In particular, defining a quantity by the desired
closed form, or accepting its derivative formula as a hypothesis, would not count
as deriving that formula from the likelihood.

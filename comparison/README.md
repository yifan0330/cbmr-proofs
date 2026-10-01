# Floating-point implementation comparisons

These checks complement the exact SymPy identities and the inventoried Lean
theorems. They exercise finite-precision programs, not proof-assistant kernels.
Neither `cbmr_code/` nor `cbmr_improved/` is edited by this suite.

Run from the repository root:

```bash
pip install -r requirements-numerical.txt
make comparison
```

The default is a small seeded CPU regression suite. It exits nonzero for a
required discrepancy and writes `comparison/results/regression.json`, including
numerical errors, diagnostic observations, environment versions and SHA-256
fingerprints of both source directories. CI uploads this report even when a
required comparison fails. A snapshot of the generated reports is committed under
`comparison/results/`; rerunning a target overwrites the corresponding file.

## Four independent questions

| Layer | What is measured | What it does not establish |
|---|---|---|
| Derivatives | Independent closed formulas versus float64 autodiff and directional differences | Correctness at all parameters, exact arithmetic, or statistical coverage |
| Covariance | Block/Schur/Kronecker algebra, inverse residuals, eigenvalue/SVD conditioning and projected sandwiches | Stable inversion of unidentifiable models |
| Difficult cases and fits | Finite-precision breakdown and agreement of identifiable same-objective optima | Equality of distinct NB models or optimization paths |
| Experiments | Isolated timing/memory, and separately simulation bias/coverage/false positives | Production-scale speedups or calibration from a small Monte Carlo run |

## 1. Likelihoods and derivatives

`likelihoods.py` implements unsimplified PyTorch likelihoods using `float64`.
`closed_form.py` independently implements NumPy/SciPy scores and observed
negative-log-likelihood Hessians. It imports neither source implementation nor
the autodiff reference. Derivatives are evaluated at exactly the same regression
coefficients and fixed dispersion.

| Model | Oracle and assembly | Actual source comparison |
|---|---|---|
| Poisson | Full cell objective versus pattern compression; affine score/Hessian | Both packages |
| Independent NB | Cell PMF; weighted score, observed and expected information; expected score variance | Reference/helper only: neither package implements this likelihood |
| Aggregated NB | Pattern-total working PMF with both `rho` and `A` depending on global coefficients; all Hessian blocks | Both packages |
| Clustered NB | Full joint Gamma-Poisson marginal; diagonal-minus-rank-one log-mean curvature | Both packages |
| Fixed quadratic penalty | Add `0.5 theta.T K theta`; NLL gradient adds `K theta`, Hessian adds `K` | Explicit external wrapper, not a claim of native package penalty support |

Scores have the **log-likelihood** sign (the negative of the NLL gradient);
penalized scores therefore add `-K theta`. Expected objective curvature includes
`K`, but likelihood score variance does not. Test penalties are fixed symmetric
PSD coefficient-difference matrices with a zero global block; no smoothing
parameter is estimated or inferred from a paper's unspecified roughness operator.

Constants match the original programs. Poisson and clustered NB omit cell
factorials and the parameter-independent exposure log term. Aggregated NB keeps
the factorial of its pattern-total observation. Independent NB keeps the cell
factorial. Objective values are compared **within** these conventions, never
between different likelihood families.

GC designs have group-specific spatial maps plus a shared scalar covariate.
SV designs add a spatial slope and overlapping spatial loadings. For the two
original marginal NB classes, SV loadings repeat within groups so each pattern
has multiple experiments. Continuous, unique-pattern SV rows are additionally
checked for Poisson and independent-cell NB. Unequal group sizes, varying
exposure and shared covariates are present in the seeded designs.

Default elementwise tolerances are `rtol=1e-8`, `atol=1e-10`. Reports also include
maximum absolute error and

```text
||actual - reference||_F / max(1, ||reference||_F).
```

Directional first and second objective differences use steps from `1e-2` through
`1e-6`. Every step and its error is retained; only the best agreement is required
to meet the looser finite-difference thresholds (`1e-6` and `1e-5` respectively).
The smallest step is not assumed to be the most accurate.

## 2. Covariance and conditioning

`covariance.py` compares actual original covariance functions and the improved
`StructuredInformation` with dense NumPy inversion/solves. Cases include SPD
blocks, an SPD border, an indefinite invertible border, a nearly singular
Schur complement and an exactly singular block. Checks retain shared-global
cross-group uncertainty.

The report includes `||H V - I||_F`, its identity-normalized version and the
normwise backward residual

```text
||H V - I||_F / (||H||_F ||V||_F + ||I||_F).
```

Near singularity, inverse-entry equality and close SVD/eigenvalue condition
agreement are **not** required. Condition estimates, `condition * eps` and
residuals are reported instead; a tiny backward error does not imply accurate
inverse entries. Deliberately singular solves must raise, without pseudoinverses
or hidden ridge. Indefinite inverse agreement is an algebra check, not permission
to interpret an indefinite matrix as a valid covariance.

The Kronecker case uses actual Poisson information with an exactly common
spatial shape and no global border. A separate counterexample confirms that
separable NB means generally yield nonseparable weights. Projected score
sandwiches are compared with a full dense meat and inverse. These algebra tests
use arbitrary score matrices; calibration is a different question.

## 3. Stability and full fits

Zero-count cases retain required derivative checks. Extreme means and small
dispersion retain all finite/nonfinite results, warnings and exceptions as
**diagnostics**, rather than pretending an unstable autodiff reference is exact.
Specific required checks cover finite improved aggregated-NB output at spatial
log means of `-1000` and `1000`, and its curvature near the correct aggregated
Poisson limit.

**Aggregated NB is a moment-matched working likelihood, not independent-cell
NB.** Even at zero dispersion its global curvature need not equal full Poisson:
the full study-allocation likelihood carries additional covariate information.
The limit check differentiates the aggregate Poisson objective explicitly.
Redundant-intercept cases are diagnosed by design rank and Hessian singular
values and are not fitted as identifiable models.

`fits.py` uses the same initial coefficients and SciPy L-BFGS-B configuration for
the unsimplified reference, independent closed formulas, and each actual source
likelihood where supported. This isolates objective/derivative differences from
optimizer differences. Dispersion is **fixed**, not jointly refitted by one
implementation and held fixed by another.

Reports retain optimizer status, final objective, all coefficients, fitted
means, scores, model-based standard errors and information conditioning. Fit
agreement has separate tolerances (`rtol=1e-5`, `atol=2e-6`; score `atol=2e-5`),
and requires final score infinity norm below `2e-5` and positive-definite
fitted information. The score scale accommodates objective-roundoff stopping
without weakening derivative checks at identical coefficients. No equality of
iteration counts or trajectories is assumed. Penalty derivatives are covered
above; these fit comparisons are unpenalized.

Target individual layers without rerunning unrelated work:

```bash
python -m comparison.run --sections derivatives covariance
python -m comparison.run --sections stability fits --output comparison/results/fits.json
```

## 4. Performance and statistical calibration are separate

### Benchmarks

```bash
python -m comparison.benchmark --sizes tiny --repeats 1 --models poisson \
  --threads 1 --output comparison/results/benchmark-smoke.json
make benchmark
python -m comparison.benchmark --sizes tiny small --repeats 3 --models poisson \
  --workloads hessian contrast_covariance --output comparison/results/scaling.json
```

Benchmark sizes are deliberately bounded. Workers run **sequentially**, with
fixed BLAS/OpenMP/Torch thread settings established before numerical imports.
Each measured repetition gets a fresh process. Reference calculations happen
in separate processes so reference Hessians do not inflate measured worker
peak memory. Runtime excludes import/setup costs, which are reported separately.

Peak RSS is the absolute process high-water mark, including Python, Torch,
imports and setup; it is **not** a precise incremental allocation measurement.
Subtracting an import high-water mark is not treated as operation peak memory.
JSON records and the printed table label the workload, model, dimensions,
numerical error, timings and peak RSS. Do not infer large-pool speedups from
these deliberately small examples.

### Calibration

```bash
python -m comparison.calibration --replicates 3 --models poisson \
  --output comparison/results/calibration-smoke.json
make calibration
```

Simulation is not a CI coverage threshold and is not part of timed benchmarks.
It reports coefficient bias, 95% interval coverage, false-positive rates for a
true null coefficient, Monte Carlo uncertainty and failed-fit counts.
Monte Carlo rate uncertainty includes Wilson intervals, which remain
nondegenerate when every observed replicate covers or none rejects.
Dispersion stays fixed, and the fitted mean design is correctly specified.
Conditional-on-success summaries must be interpreted together with failures.
A few replicates are a smoke test, not evidence for or against nominal coverage.
If a requested model has no accepted inference replicates, the command writes
the failure report and exits nonzero; partial failures retain explicit denominators.

Poisson, independent-cell NB and clustered NB require their respective sampling
models. An aggregated-NB experiment samples its **own pattern-total working
distribution**, not independent-cell NB, and explicitly labels the experiment
counts used only to represent those totals. This distinction prevents a
misspecified-data simulation from being mistaken for an implementation bug.

### Expanded experiment (opt-in)

```bash
make comparison-expanded
# Run either part independently:
make benchmark-expanded
make calibration-expanded
```

This is a separate, larger experiment, not an expansion of the default regression
suite or an overwrite of the original benchmark/calibration reports.

`comparison/explanation.py` generates result tables for the collaborator
explanation (`docs/explanation.tex`, maintained with the docs) from the saved
expanded reports, without rerunning either experiment. The document reports both the mixed performance findings and the expanded
calibration acceptance failures. The generator checks source fingerprints and
stops for review if the saved outcomes no longer support the document's explanation.

| Size | Experiments | Voxels | Bases | Groups | GC parameters |
|---|---:|---:|---:|---:|---:|
| Small | 32 | 32 | 4 | 3 | 13 |
| Moderate | 64 | 96 | 8 | 4 | 33 |
| Large | 128 | 256 | 8 | 4 | 33 |

The scaling benchmark uses all three sizes, GC designs, and Poisson, aggregated
NB and clustered NB. It compares the original, improved and unsimplified
autodiff implementations for both Hessian assembly and contrast covariance, with
three fresh-process repetitions per configuration. Results go to
`comparison/results/benchmark-expanded.json` (54 records, 162 measured workers
plus 18 separate reference workers). Small-to-moderate scaling changes both data
size and parameter count; moderate-to-large keeps the parameter count fixed.
This is bounded synthetic scaling, not a whole-brain production benchmark.

Calibration uses the large GC design and 100 replicates for each of Poisson,
independent-cell NB, aggregated NB and clustered NB, saving all 400 attempts to
`comparison/results/calibration-expanded.json`. Sampling, fixed dispersion,
zero initial coefficients, optimizer settings, inference acceptance criteria and
failure denominators are unchanged. The three NB names still denote
different sampling models, not interchangeable likelihoods. At 100 accepted
replicates, the nominal 5% false-positive rate has an approximate Monte Carlo
standard error of 2.2 percentage points; Wilson intervals and failure counts
remain necessary, and 100 replicates do not establish precise calibration.

Both parts run sequentially with one numerical thread. The benchmark retains
the 64-parameter guard, uses a 128 MiB estimated design/tangent limit and allows
600 seconds per child, including imports. This estimate is **not** a bound on
total process RSS. The combined target attempts calibration even if the
benchmark fails, and returns nonzero if either command fails. These opt-in runs
can take substantially longer than the smoke suite.

Calibration also accepts `--size smoke|tiny|small|moderate|large`; its default
remains `smoke` (24 experiments, 9 voxels, 2 bases, 2 groups). For example:

```bash
python -m comparison.calibration --size moderate --replicates 100 \
  --models poisson independent_nb aggregated_nb clustered_nb \
  --output comparison/results/calibration-moderate.json
```

## Source loading and provenance

`adapters.py` loads the original files through a private package namespace,
redirecting only their `from nimare.meta.cbmr...` import nodes. Statistical code
bodies remain unchanged. This is necessary because the supplied files were
written as NiMARE modules, not as a standalone package. The loader neither
replaces installed NiMARE modules nor rewrites source files. Infrastructure
checks compare non-import ASTs and retain original source paths and hashes.

High-level NiMARE image preprocessing is outside this suite. The actual
likelihood, information, covariance and model-standard-error paths are exercised
with small explicit design blocks. No source directory is copied or patched
as part of a comparison run.

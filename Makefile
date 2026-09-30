PY ?= python
LAKE ?= $(if $(wildcard .elan/bin/lake),ELAN_HOME="$(CURDIR)/.elan" "$(CURDIR)/.elan/bin/lake",lake)

.PHONY: all proofs papers appendix expanded-appendix paper-appendices cbmr-paper-appendix imag-appendix explanation numerical comparison benchmark calibration comparison-expanded benchmark-expanded calibration-expanded lean-deps lean lean-complete clean

all: proofs papers

## Run every symbolic proof and regenerate the LaTeX fragments. Nonzero exit if any claim fails.
proofs:
	$(PY) run_proofs.py

## Numerical equivalence and structured-inference checks for the opt-in implementation.
numerical:
	$(PY) -m unittest scripts.test_cbmr_improved scripts.test_cbmr_improved_integration scripts.test_structured_information

## Seeded four-layer floating-point comparisons; detailed errors are saved as JSON.
comparison:
	$(PY) -m unittest comparison.test_comparison comparison.test_experiments
	$(PY) -m comparison.run --output comparison/results/regression.json

## Small, sequential, fresh-process timings; larger experiments are opt-in.
benchmark:
	$(PY) -m comparison.benchmark --output comparison/results/benchmark.json

## Separate Monte Carlo diagnostics, not a Hessian-equivalence assertion.
calibration:
	$(PY) -m comparison.calibration --output comparison/results/calibration.json

## Opt-in larger experiment; keep workloads sequential even under make -j.
comparison-expanded:
	$(MAKE) -k -j1 benchmark-expanded calibration-expanded

## Three design sizes, three source models, two workloads and three fresh-process repeats.
benchmark-expanded:
	$(PY) -m comparison.benchmark --sizes small moderate large \
		--models poisson aggregated_nb clustered_nb --designs gc \
		--workloads hessian contrast_covariance --repeats 3 --threads 1 \
		--timeout 600 --max-estimated-mib 128 \
		--output comparison/results/benchmark-expanded.json

## Larger correctly specified data, 100 replicates for each of the four sampling models.
calibration-expanded:
	$(PY) -m comparison.calibration --size large --design gc --replicates 100 \
		--models poisson independent_nb aggregated_nb clustered_nb --threads 1 \
		--output comparison/results/calibration-expanded.json

## Update the collaborator PDF from saved reports, without rerunning experiments.
explanation:
	$(PY) -m unittest comparison.test_explanation
	$(PY) -m comparison.explanation
	cd docs && pdflatex -halt-on-error -interaction=nonstopmode explanation.tex >/dev/null || { cat explanation.log; exit 1; }
	cd docs && pdflatex -halt-on-error -interaction=nonstopmode explanation.tex >/dev/null || { cat explanation.log; exit 1; }
	cp docs/explanation.pdf explanation.pdf
	rm -f docs/explanation.pdf
	@echo "built explanation.pdf"

## Download dependencies without requiring the toolchain's native C compiler.
lean-deps:
	MATHLIB_NO_CACHE_ON_UPDATE=1 $(LAKE) update
	$(LAKE) build +Cache.Main:olean
	$(LAKE) env lean --run .lake/packages/mathlib/Cache/Main.lean get

## Check the formalised subset with Lean and reject additional axiom dependencies.
lean:
	$(LAKE) build
	@$(LAKE) env lean scripts/LeanAudit.lean > .lake/lean-audit.log 2>&1; status=$$?; cat .lake/lean-audit.log; exit "$$status"
	$(PY) scripts/test_lean_audit.py
	$(PY) scripts/test_lean_coverage.py
	$(PY) scripts/check_lean_coverage.py .lake/lean-audit.log --report .lake/lean-coverage-report.json

## Also require every inventoried appendix derivation to have completed coverage.
lean-complete: lean
	$(PY) scripts/check_lean_coverage.py .lake/lean-audit.log --require-complete

## Verify, then build the integrated appendix.
appendix: proofs
	cd docs && pdflatex -halt-on-error -interaction=nonstopmode cbmr_appendix.tex >/dev/null
	cd docs && pdflatex -halt-on-error -interaction=nonstopmode cbmr_appendix.tex >/dev/null
	@echo "built docs/cbmr_appendix.pdf"

## Build the standalone expanded derivations without changing the other PDFs.
expanded-appendix:
	cd docs && pdflatex -halt-on-error -interaction=nonstopmode cbmr_expanded_appendix.tex >/dev/null || { cat cbmr_expanded_appendix.log; exit 1; }
	cd docs && pdflatex -halt-on-error -interaction=nonstopmode cbmr_expanded_appendix.tex >/dev/null || { cat cbmr_expanded_appendix.log; exit 1; }
	cd docs && pdflatex -halt-on-error -interaction=nonstopmode cbmr_expanded_appendix.tex >/dev/null || { cat cbmr_expanded_appendix.log; exit 1; }
	@echo "built docs/cbmr_expanded_appendix.pdf"

## Build the two independent, paper-specific expanded appendices.
paper-appendices: cbmr-paper-appendix imag-appendix

cbmr-paper-appendix:
	cd docs && pdflatex -halt-on-error -interaction=nonstopmode cbmr_paper_appendix.tex >/dev/null || { cat cbmr_paper_appendix.log; exit 1; }
	cd docs && pdflatex -halt-on-error -interaction=nonstopmode cbmr_paper_appendix.tex >/dev/null || { cat cbmr_paper_appendix.log; exit 1; }
	cd docs && pdflatex -halt-on-error -interaction=nonstopmode cbmr_paper_appendix.tex >/dev/null || { cat cbmr_paper_appendix.log; exit 1; }
	@echo "built docs/cbmr_paper_appendix.pdf"

imag-appendix:
	cd docs && pdflatex -halt-on-error -interaction=nonstopmode imag_a_1057_appendix.tex >/dev/null || { cat imag_a_1057_appendix.log; exit 1; }
	cd docs && pdflatex -halt-on-error -interaction=nonstopmode imag_a_1057_appendix.tex >/dev/null || { cat imag_a_1057_appendix.log; exit 1; }
	cd docs && pdflatex -halt-on-error -interaction=nonstopmode imag_a_1057_appendix.tex >/dev/null || { cat imag_a_1057_appendix.log; exit 1; }
	@echo "built docs/imag_a_1057_appendix.pdf"

## Build the two derivation papers and the integrated appendix after verification.
papers: appendix
	cd docs && pdflatex -interaction=nonstopmode cbmr_hessian.tex >/dev/null
	cd docs && pdflatex -interaction=nonstopmode cbmr_hessian.tex >/dev/null
	cd docs && pdflatex -interaction=nonstopmode cbmr_covariance.tex >/dev/null
	cd docs && pdflatex -interaction=nonstopmode cbmr_covariance.tex >/dev/null
	@echo "built docs/cbmr_hessian.pdf and docs/cbmr_covariance.pdf"

clean:
	rm -f docs/*.aux docs/*.log docs/*.out docs/*.toc

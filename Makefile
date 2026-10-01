PY ?= python

.PHONY: all proofs papers numerical comparison benchmark calibration comparison-expanded benchmark-expanded calibration-expanded clean

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

## Build both papers. Depends on proofs: the fragments are inputs, not checked-in artefacts.
papers: proofs
	cd docs && pdflatex -interaction=nonstopmode cbmr_hessian.tex >/dev/null
	cd docs && pdflatex -interaction=nonstopmode cbmr_hessian.tex >/dev/null
	cd docs && pdflatex -interaction=nonstopmode cbmr_covariance.tex >/dev/null
	cd docs && pdflatex -interaction=nonstopmode cbmr_covariance.tex >/dev/null
	@echo "built docs/cbmr_hessian.pdf and docs/cbmr_covariance.pdf"

clean:
	rm -f docs/*.aux docs/*.log docs/*.out docs/*.toc

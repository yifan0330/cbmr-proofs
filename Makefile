PY ?= python

.PHONY: all proofs papers clean

all: proofs papers

## Run every symbolic proof and regenerate the LaTeX fragments. Nonzero exit if any claim fails.
proofs:
	$(PY) run_proofs.py

## Build both papers. Depends on proofs: the fragments are inputs, not checked-in artefacts.
papers: proofs
	cd docs && pdflatex -interaction=nonstopmode cbmr_hessian.tex >/dev/null
	cd docs && pdflatex -interaction=nonstopmode cbmr_hessian.tex >/dev/null
	cd docs && pdflatex -interaction=nonstopmode cbmr_covariance.tex >/dev/null
	cd docs && pdflatex -interaction=nonstopmode cbmr_covariance.tex >/dev/null
	@echo "built docs/cbmr_hessian.pdf and docs/cbmr_covariance.pdf"

clean:
	rm -f docs/*.aux docs/*.log docs/*.out docs/*.toc

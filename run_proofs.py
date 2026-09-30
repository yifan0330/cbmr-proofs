#!/usr/bin/env python
"""Run every proof, write its LaTeX fragment, and exit nonzero if any claim fails.

This is the contract of the repository. The papers in ``docs/`` are only as good as this
script's exit code, and CI runs it on every push.
"""
import sys
from pathlib import Path

from proofs import (
    appendix, clustered_negative_binomial, covariance, negative_binomial, poisson,
    published_cbmr,
)
from proofs.latex import BANNER

MODULES = [
    ("Poisson Hessian", poisson),
    ("Negative Binomial Hessian", negative_binomial),
    ("Clustered Negative Binomial Hessian", clustered_negative_binomial),
    ("Covariance strategies", covariance),
    ("Integrated appendix", appendix),
    ("Published CBMR integration", published_cbmr),
]


def main():
    failures, claims, results = [], 0, []
    for title, module in MODULES:
        print(f"\n=== {title} ===")
        proof = module.run()
        proof.write()
        claims += len(proof.claims)
        results.append((title, len(proof.claims), proof.closed))
        if not proof.closed:
            failures.append(title)
    report = Path("docs/generated/verification.tex")
    report.write_text(BANNER + "\n".join([
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Symbolic suite & Claims & Closed\\",
        r"\midrule",
        *[rf"{title} & {count} & {'yes' if closed else 'NO'}\\"
          for title, count, closed in results],
        r"\midrule",
        rf"Total & {claims} & {'yes' if not failures else 'NO'}\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n")
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED ({claims} claims checked): {', '.join(failures)}")
        return 1
    print(f"All {claims} claims closed. Every residual is exactly zero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

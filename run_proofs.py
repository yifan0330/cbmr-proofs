#!/usr/bin/env python
"""Run every proof, write its LaTeX fragment, and exit nonzero if any claim fails.

This is the contract of the repository. The papers in ``docs/`` are only as good as this
script's exit code, and CI runs it on every push.
"""
import sys

from proofs import (
    clustered_negative_binomial, covariance, negative_binomial, poisson,
)

MODULES = [
    ("Poisson Hessian", poisson),
    ("Negative Binomial Hessian", negative_binomial),
    ("Clustered Negative Binomial Hessian", clustered_negative_binomial),
    ("Covariance strategies", covariance),
]


def main():
    failures, claims = [], 0
    for title, module in MODULES:
        print(f"\n=== {title} ===")
        proof = module.run()
        proof.write()
        claims += len(proof.claims)
        if not proof.closed:
            failures.append(title)
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED ({claims} claims checked): {', '.join(failures)}")
        return 1
    print(f"All {claims} claims closed. Every residual is exactly zero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

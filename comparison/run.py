"""Small seeded CI regression suite: python -m comparison.run."""

import argparse
import os


SECTIONS = ("derivatives", "covariance", "stability", "fits")


def run_regression(seed=731, rtol=1e-8, atol=1e-10, sections=SECTIONS):
    import numpy as np
    from .adapters import source_manifest
    from .cases import MODELS, make_case
    from .metrics import environment

    before = source_manifest()
    rows = []
    if "derivatives" in sections:
        from .derivatives import derivative_checks

        for model in MODELS:
            for design in ("gc", "sv"):
                for penalty in (False, True):
                    case = make_case(model, design, seed=seed, penalty=penalty)
                    case.name += "-penalized" if penalty else "-unpenalized"
                    rows.extend(derivative_checks(case, rtol, atol))
        for model in ("poisson", "independent_nb"):
            case = make_case(model, "sv", seed=seed)
            case.spatial[:, -1] = np.random.default_rng(seed + 1).normal(size=len(case.counts))
            case.name += "-continuous-unique-patterns"
            rows.extend(derivative_checks(case, rtol, atol))
    if "covariance" in sections:
        from .covariance import covariance_checks
        rows.extend(covariance_checks(seed, rtol, atol))
    if "stability" in sections:
        from .stability import stability_checks
        rows.extend(stability_checks(seed))
    if "fits" in sections:
        from .fits import fit_checks
        for model in MODELS:
            for design in ("gc", "sv"):
                rows.extend(fit_checks(make_case(model, design, seed=seed)))
    after = source_manifest()
    rows.append({
        "layer": "provenance", "case": "source-implementations",
        "comparison": "Both implementation directories unchanged during comparisons",
        "passed": before == after,
    })
    required = [row for row in rows if row.get("passed") is not None]
    failures = [row for row in required if not row["passed"]]
    return {
        "purpose": "Floating-point implementation regression; not a symbolic proof or calibration study.",
        "seed": seed, "derivative_rtol": rtol, "derivative_atol": atol,
        "fit_tolerances": {"rtol": 1e-5, "atol": 2e-6, "score_atol": 2e-5,
                           "maximum_score_infinity_norm": 2e-5},
        "dispersion": "Fixed, including throughout fit comparisons.",
        "sections": list(sections), "environment": environment(),
        "source_hashes": after,
        "applicability": {
            "actual_source_likelihoods": ["poisson", "aggregated_nb", "clustered_nb"],
            "reference_and_helper_only": ["independent_nb"],
            "penalties": "External fixed quadratic wrapper; neither source gains penalty support.",
        },
        "summary": {"required": len(required), "failed": len(failures),
                    "diagnostic": len(rows) - len(required)},
        "rows": rows,
    }


def print_report(report):
    print("| Layer | Required | Failed | Diagnostic | Largest reported relative error |")
    print("|---|---:|---:|---:|---:|")
    for layer in dict.fromkeys(row["layer"] for row in report["rows"]):
        rows = [row for row in report["rows"] if row["layer"] == layer]
        required = [row for row in rows if row.get("passed") is not None]
        errors = [row["relative_frobenius"] for row in rows if "relative_frobenius" in row]
        largest = f"{max(errors):.3e}" if errors else "-"
        print(f"| {layer} | {len(required)} | {sum(not r['passed'] for r in required)} "
              f"| {len(rows) - len(required)} | {largest} |")
    for row in report["rows"]:
        if row.get("passed") is False:
            print(f"FAIL: {row['case']}: {row['comparison']}: "
                  f"{row.get('reason', row.get('message', row.get('relative_frobenius', 'see report')))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument("--output", default="comparison/results/regression.json")
    args = parser.parse_args()
    import math
    if not all(math.isfinite(value) and value > 0 for value in (args.rtol, args.atol)):
        parser.error("Tolerances must be finite and positive.")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from .metrics import write_report

    report = run_regression(args.seed, args.rtol, args.atol, args.sections)
    report["thread_environment"] = {name: os.environ[name] for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )}
    write_report(args.output, report)
    print_report(report)
    print(f"Detailed numerical errors: {args.output}")
    return int(report["summary"]["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())

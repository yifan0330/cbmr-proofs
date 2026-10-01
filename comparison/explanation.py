"""Generate explanation-PDF tables from the saved expanded experiment reports."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
SIZES = ("small", "moderate", "large")
MODELS = ("poisson", "aggregated_nb", "clustered_nb")
IMPLEMENTATIONS = ("original", "improved", "unsimplified_autodiff")
WORKLOADS = ("hessian", "contrast_covariance")
LABELS = {
    "poisson": "Poisson",
    "independent_nb": "Independent NB",
    "aggregated_nb": "Aggregated NB",
    "clustered_nb": "Clustered NB",
}


def scientific(value):
    if not math.isfinite(value):
        raise ValueError(f"Cannot typeset a nonfinite summary value: {value}")
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def benchmark_summary(report):
    if report["status"] != "ok":
        raise ValueError("Review failed benchmark records before rebuilding the explanation.")
    indexed = {}
    for row in report["records"]:
        key = tuple(row[field] for field in ("size", "model", "workload", "implementation"))
        if key in indexed:
            raise ValueError(f"Duplicate benchmark configuration: {key}")
        if row["status"] != "ok" or row["successful_repeats"] != 3 or len(row["runs"]) != 3:
            raise ValueError(f"Expected three successful repetitions for {key}")
        times = [run["seconds"] for run in row["runs"]]
        if not all(run["status"] == "ok" for run in row["runs"]):
            raise ValueError(f"Failed benchmark repetition for {key}")
        if not all(run["numerical_error"]["within_tolerance"] for run in row["runs"]):
            raise ValueError(f"Benchmark numerical disagreement for {key}")
        if not all(math.isfinite(value) and value > 0 for value in times):
            raise ValueError(f"Invalid benchmark time for {key}")
        if not math.isclose(statistics.median(times), row["seconds"]["median"],
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Inconsistent saved median for {key}")
        indexed[key] = row
    expected = set(itertools.product(SIZES, MODELS, WORKLOADS, IMPLEMENTATIONS))
    if set(indexed) != expected:
        raise ValueError("Expected the complete 54-record expanded benchmark grid.")
    cases = list(itertools.product(SIZES, MODELS, WORKLOADS))
    return {
        "indexed": indexed,
        "records": len(indexed),
        "runs": sum(len(row["runs"]) for row in indexed.values()),
        "pairs": len(cases),
        "faster_original": sum(
            indexed[(*case, "improved")]["seconds"]["median"] <
            indexed[(*case, "original")]["seconds"]["median"] for case in cases
        ),
        "faster_autodiff": sum(
            indexed[(*case, "improved")]["seconds"]["median"] <
            indexed[(*case, "unsimplified_autodiff")]["seconds"]["median"] for case in cases
        ),
    }


def calibration_summary(report):
    models = {row["model"]: row for row in report["models"]}
    if len(models) != len(report["models"]) or set(models) != set(LABELS):
        raise ValueError("Expected four distinct calibration models.")
    threshold = report["fit"]["score_tolerance"]
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid calibration score threshold.")
    attempts = accepted = inferred = optimizer_success = above_threshold = relative_stops = 0
    for model in models.values():
        rows = model["replicates"]
        aggregate = model["aggregate"]
        if len(rows) != aggregate["attempts"]:
            raise ValueError("Calibration attempt count disagrees with saved replicates.")
        if sum(row["fit_success"] for row in rows) != aggregate["successful_fits"]:
            raise ValueError("Calibration fit count disagrees with saved replicates.")
        if sum(row["inference_success"] for row in rows) != aggregate["successful_inferences"]:
            raise ValueError("Calibration inference count disagrees with saved replicates.")
        for field, count in (("bias_denominator", aggregate["successful_fits"]),
                             ("coverage_denominator", aggregate["successful_inferences"]),
                             ("false_positive_denominator", aggregate["successful_inferences"])):
            if aggregate[field] != count:
                raise ValueError(f"Calibration {field} disagrees with accepted replicates.")
        if not aggregate["successful_fits"] and aggregate["coefficient_bias"] is not None:
            raise ValueError("Bias must be unavailable when no fits were accepted.")
        if not aggregate["successful_inferences"] and (
            aggregate["coverage"] is not None or
            aggregate["null_moderator_false_positive_rate"] is not None
        ):
            raise ValueError("Calibration rates must be unavailable without accepted inference.")
        attempts += len(rows)
        accepted += aggregate["successful_fits"]
        inferred += aggregate["successful_inferences"]
        for row in rows:
            optimizer = row["optimizer"]
            if optimizer is not None:
                optimizer_success += bool(optimizer["success"])
                above_threshold += optimizer["score_infinity_norm"] > threshold
                relative_stops += "REL_REDUCTION_OF_F_" in optimizer["message"]
    return {
        "models": models, "attempts": attempts, "accepted": accepted, "inferred": inferred,
        "optimizer_success": optimizer_success, "above_threshold": above_threshold,
        "relative_stops": relative_stops, "threshold": threshold,
    }


def macro(name, value):
    return "\\newcommand{\\" + name + "}{" + str(value) + "}\n"


def timing_table(summary, workload):
    rows = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Size & Model & Original & Improved & Autodiff & Ratio$^\dagger$\\",
        r"\midrule",
    ]
    for size, model in itertools.product(SIZES, MODELS):
        entries = [summary["indexed"][(size, model, workload, impl)] for impl in IMPLEMENTATIONS]
        times = [entry["seconds"]["median"] * 1000 for entry in entries]
        rows.append(
            f"{size.title()} & {LABELS[model]} & " +
            " & ".join(f"{value:.2f}" for value in times) +
            f" & {times[0] / times[1]:.2f}" + r"$\times$\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(rows)


def calibration_table(summary):
    rows = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & \shortstack{Optimiser\\success} & \shortstack{Accepted\\fit / inference}"
        r" & \shortstack{Score norm\\minimum} & Median & Maximum\\",
        r"\midrule",
    ]
    for name in LABELS:
        model = summary["models"][name]
        optimizers = [row["optimizer"] for row in model["replicates"]
                      if row["optimizer"] is not None]
        norms = [row["score_infinity_norm"] for row in optimizers]
        if not norms or not all(math.isfinite(value) for value in norms):
            raise ValueError(f"Review missing/nonfinite optimiser diagnostics for {name}.")
        aggregate = model["aggregate"]
        cells = [
            LABELS[name], str(sum(row["success"] for row in optimizers)),
            f"{aggregate['successful_fits']} / {aggregate['successful_inferences']}",
            *(f"${scientific(value)}$" for value in
              (min(norms), statistics.median(norms), max(norms))),
        ]
        rows.append(" & ".join(cells) + r"\\")
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(rows)


def render(benchmark, calibration):
    b = benchmark_summary(benchmark)
    c = calibration_summary(calibration)
    if b["faster_original"] != 4 or b["faster_autodiff"] != 18:
        raise ValueError("Benchmark rankings changed: revise the explanation's interpretation.")
    # The prose explains this specific all-rejected run; do not silently reuse it for new outcomes.
    if not (c["attempts"] == c["optimizer_success"] == c["above_threshold"] ==
            c["relative_stops"] and c["accepted"] == c["inferred"] == 0):
        raise ValueError("Calibration outcomes changed: revise the explanation's interpretation.")
    if {row["aggregate"]["attempts"] for row in c["models"].values()} != {100}:
        raise ValueError("The explanation describes 100 attempts per calibration model.")
    expected_dimensions = {
        "small": (32, 32, 4, 3, 13),
        "moderate": (64, 96, 8, 4, 33),
        "large": (128, 256, 8, 4, 33),
    }
    keys = ("experiments", "voxels", "bases", "groups", "parameters")
    for row in benchmark["records"]:
        if row["design"] != "gc" or row["threads"] != 1:
            raise ValueError("The explanation describes single-threaded GC benchmarks.")
        if tuple(row["dimensions"][key] for key in keys) != expected_dimensions[row["size"]]:
            raise ValueError("Benchmark dimensions changed: update the explanation's setup table.")
    for row in calibration["models"]:
        if row["design"] != "gc" or (
            tuple(row["dimensions"][key] for key in keys) != expected_dimensions["large"]
        ):
            raise ValueError("Calibration dimensions changed: update the explanation's setup.")
    output = "% Generated by python -m comparison.explanation; do not edit.\n"
    for name, value in (
        ("ExpandedRecords", b["records"]), ("ExpandedRuns", b["runs"]),
        ("ExpandedPairs", b["pairs"]), ("ExpandedFasterOriginal", b["faster_original"]),
        ("ExpandedFasterAutodiff", b["faster_autodiff"]),
        ("ExpandedAttempts", c["attempts"]), ("ExpandedAcceptedFits", c["accepted"]),
        ("ExpandedAcceptedInferences", c["inferred"]),
        ("ExpandedScoreThreshold", scientific(c["threshold"])),
        ("ExpandedHessianTable", timing_table(b, "hessian")),
        ("ExpandedContrastTable", timing_table(b, "contrast_covariance")),
        ("ExpandedCalibrationTable", calibration_table(c)),
    ):
        output += macro(name, value)
    source_rows = [row for row in benchmark["records"]
                   if row["implementation"] in ("original", "improved")]
    for name, field in (("ExpandedMaxAbsolute", "max_absolute"),
                        ("ExpandedMaxRelative", "relative_frobenius")):
        output += macro(name, scientific(max(row["numerical_error"][field] for row in source_rows)))
    for implementation, name in (("original", "ExpandedOriginalRSS"),
                                 ("improved", "ExpandedImprovedRSS")):
        values = [row["peak_rss_mib"]["median"] for row in source_rows
                  if row["implementation"] == implementation]
        output += macro(name, f"{min(values):.1f}--{max(values):.1f}")
    example = b["indexed"][("moderate", "poisson", "contrast_covariance", "improved")]
    output += macro("ExpandedTimingSpread", ", ".join(
        f"{value * 1000:.2f}" for value in sorted(run["seconds"] for run in example["runs"])
    ))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "comparison/results")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/generated/explanation_results.tex")
    args = parser.parse_args()
    inputs = [args.results / f"{name}-expanded.json" for name in ("benchmark", "calibration")]
    reports = [json.loads(path.read_text()) for path in inputs]
    for path, report in zip(inputs, reports):
        if not report["metadata"]["source_hashes"]:
            raise ValueError(f"{path.name}: missing source fingerprints")
        for source, expected in report["metadata"]["source_hashes"].items():
            if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != expected:
                raise ValueError(f"{path.name}: source snapshot no longer matches {source}")
    text = render(*reports)
    for path, name in zip(inputs, ("ExpandedBenchmarkDigest", "ExpandedCalibrationDigest")):
        text += macro(name, hashlib.sha256(path.read_bytes()).hexdigest()[:16])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"Generated {args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output}")


if __name__ == "__main__":
    main()

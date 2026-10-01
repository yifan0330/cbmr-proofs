"""Checks for the saved-report summaries used by the collaborator PDF."""

from copy import deepcopy
import itertools
import unittest

from .explanation import (
    IMPLEMENTATIONS, LABELS, MODELS, SIZES, WORKLOADS,
    benchmark_summary, calibration_summary, render, scientific,
)


def benchmark_fixture():
    dimensions = {
        "small": (32, 32, 4, 3, 13),
        "moderate": (64, 96, 8, 4, 33),
        "large": (128, 256, 8, 4, 33),
    }
    keys = ("experiments", "voxels", "bases", "groups", "parameters")
    records = []
    for index, (size, model, workload) in enumerate(itertools.product(SIZES, MODELS, WORKLOADS)):
        for implementation in IMPLEMENTATIONS:
            seconds = {"original": 1.0, "improved": 0.5 if index < 4 else 2.0,
                       "unsimplified_autodiff": 3.0}[implementation]
            error = {"within_tolerance": True, "max_absolute": 1e-11,
                     "relative_frobenius": 1e-14}
            records.append({
                "size": size, "model": model, "workload": workload,
                "implementation": implementation, "status": "ok", "successful_repeats": 3,
                "runs": [{"status": "ok", "seconds": seconds, "numerical_error": error.copy()}
                         for _ in range(3)],
                "seconds": {"median": seconds}, "peak_rss_mib": {"median": 260.0},
                "numerical_error": error, "design": "gc", "threads": 1,
                "dimensions": dict(zip(keys, dimensions[size])),
            })
    return {"status": "ok", "records": records}


def calibration_fixture():
    models = []
    for name in LABELS:
        row = {
            "fit_success": False, "inference_success": False,
            "optimizer": {"success": True, "score_infinity_norm": 0.002,
                          "message": "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH"},
        }
        models.append({
            "model": name, "design": "gc",
            "dimensions": {"experiments": 128, "voxels": 256, "bases": 8,
                           "groups": 4, "parameters": 33},
            "aggregate": {
                "attempts": 100, "successful_fits": 0, "successful_inferences": 0,
                "bias_denominator": 0, "coverage_denominator": 0, "false_positive_denominator": 0,
                "coefficient_bias": None, "coverage": None,
                "null_moderator_false_positive_rate": None,
            },
            "replicates": [deepcopy(row) for _ in range(100)],
        })
    return {"models": models, "fit": {"score_tolerance": 1e-4}}


class ExplanationTests(unittest.TestCase):
    def test_complete_grid_and_paired_counts(self):
        summary = benchmark_summary(benchmark_fixture())
        self.assertEqual((summary["records"], summary["runs"], summary["pairs"]), (54, 162, 18))
        self.assertEqual(summary["faster_original"], 4)
        self.assertEqual(summary["faster_autodiff"], 18)

    def test_reject_duplicate_or_missing_configuration(self):
        for duplicate in (True, False):
            with self.subTest(duplicate=duplicate):
                report = benchmark_fixture()
                if duplicate:
                    report["records"].append(deepcopy(report["records"][0]))
                else:
                    report["records"].pop()
                with self.assertRaises(ValueError):
                    benchmark_summary(report)

    def test_reject_inconsistent_median(self):
        report = benchmark_fixture()
        report["records"][0]["seconds"]["median"] += 1
        with self.assertRaisesRegex(ValueError, "median"):
            benchmark_summary(report)

    def test_reject_failed_numerical_agreement(self):
        report = benchmark_fixture()
        report["records"][0]["runs"][0]["numerical_error"]["within_tolerance"] = False
        with self.assertRaisesRegex(ValueError, "numerical disagreement"):
            benchmark_summary(report)

    def test_optimizer_success_is_not_fit_acceptance(self):
        summary = calibration_summary(calibration_fixture())
        self.assertEqual(summary["attempts"], 400)
        self.assertEqual(summary["optimizer_success"], 400)
        self.assertEqual(summary["above_threshold"], 400)
        self.assertEqual((summary["accepted"], summary["inferred"]), (0, 0))

    def test_missing_estimates_are_not_zero(self):
        report = calibration_fixture()
        report["models"][0]["aggregate"]["null_moderator_false_positive_rate"] = 0.0
        with self.assertRaisesRegex(ValueError, "unavailable"):
            calibration_summary(report)

    def test_changed_failures_require_prose_review(self):
        report = calibration_fixture()
        report["models"][0]["replicates"][0]["optimizer"]["success"] = False
        with self.assertRaisesRegex(ValueError, "outcomes changed"):
            render(benchmark_fixture(), report)

    def test_render_measured_values_and_counts(self):
        text = render(benchmark_fixture(), calibration_fixture())
        self.assertIn(r"\newcommand{\ExpandedFasterOriginal}{4}", text)
        self.assertIn(r"\newcommand{\ExpandedAcceptedInferences}{0}", text)
        self.assertIn(r"Small & Poisson & 1000.00 & 500.00 & 3000.00 & 2.00", text)

    def test_changed_rankings_require_prose_review(self):
        report = benchmark_fixture()
        row = report["records"][1]
        row["seconds"]["median"] = 2.0
        for run in row["runs"]:
            run["seconds"] = 2.0
        with self.assertRaisesRegex(ValueError, "rankings changed"):
            render(report, calibration_fixture())

    def test_reject_nonfinite_typesetting(self):
        with self.assertRaises(ValueError):
            scientific(float("nan"))


if __name__ == "__main__":
    unittest.main()

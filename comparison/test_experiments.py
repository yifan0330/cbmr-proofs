"""Deterministic orchestration and simulation checks, not coverage assertions."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from .benchmark import SIZES, THREAD_VARIABLES, _errors, _launch, parser as benchmark_parser
from .calibration import (
    CALIBRATION_SIZES, fit_replicate, main as calibration_main,
    parser as calibration_parser, simulate_counts, summarize,
)
from .cases import make_case


class ExperimentTests(unittest.TestCase):
    def test_expanded_sizes_preserve_defaults_and_stay_bounded(self):
        self.assertEqual(benchmark_parser().parse_args([]).sizes, ["tiny"])
        self.assertEqual(calibration_parser().parse_args([]).size, "smoke")
        self.assertEqual(CALIBRATION_SIZES["smoke"],
                         dict(experiments=24, voxels=9, bases=2, groups=2))
        self.assertEqual(SIZES["large"],
                         dict(experiments=128, voxels=256, bases=8, groups=4))
        self.assertEqual(benchmark_parser().parse_args(["--sizes", "large"]).sizes, ["large"])
        self.assertEqual(calibration_parser().parse_args(["--size", "large"]).size, "large")
        for design in ("gc", "sv"):
            case = make_case(design=design, **SIZES["large"])
            self.assertEqual(case.counts.shape, (128, 256))
            self.assertEqual(case.n_parameters, 33 if design == "gc" else 41)
            self.assertTrue(np.all(np.isfinite(case.means())))
            self.assertEqual(np.linalg.matrix_rank(case.bases), 8)
            _, assignment = case.patterns
            self.assertGreaterEqual(np.bincount(assignment).min(), 2)
            n, v = case.counts.shape
            p = case.n_parameters
            estimate = 8 * (6 * n * v * p + 12 * p**2 + 12 * n * v)
            self.assertLessEqual(p, 64)
            self.assertLessEqual(estimate, 128 * 1024**2)

    def test_calibration_uses_selected_size_and_retains_failed_attempts(self):
        failed = {"status": "error", "fit_success": False, "inference_success": False,
                  "optimizer": None, "errors": ["Deliberate test failure."]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calibration.json"
            with patch("torch.set_num_threads"), patch("torch.set_num_interop_threads"), \
                    patch.dict("os.environ"), \
                    patch("comparison.calibration.fit_replicate",
                          side_effect=lambda *args, **kwargs: dict(failed)) as fit:
                status = calibration_main([
                    "--size", "large", "--replicates", "2", "--models", "poisson",
                    "--output", str(output),
                ])
            self.assertEqual(status, 1)
            self.assertEqual(fit.call_count, 2)
            for call in fit.call_args_list:
                self.assertEqual(call.args[0].counts.shape, (128, 256))
                self.assertEqual(call.args[0].coefficients[-1], 0)
            report = json.loads(output.read_text())
            model = report["models"][0]
            self.assertEqual(model["size"], "large")
            self.assertEqual(model["dimensions"], dict(SIZES["large"], parameters=33))
            self.assertEqual(model["aggregate"]["total_failed_replicates"], 2)
            self.assertEqual(len(model["replicates"]), 2)

    def test_orchestrators_do_not_import_numerical_libraries(self):
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; import comparison.benchmark; import comparison.calibration; "
            "assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules",
        ], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_worker_launch_configures_threads_before_imports(self):
        complete = SimpleNamespace(returncode=0, stdout=json.dumps({"status": "ok"}), stderr="")
        with patch("comparison.benchmark.subprocess.run", return_value=complete) as run:
            self.assertEqual(_launch({"threads": 2}, 10)["status"], "ok")
        _, options = run.call_args
        for name in THREAD_VARIABLES:
            self.assertEqual(options["env"][name], "2")
        self.assertEqual(options["timeout"], 10)
        self.assertNotIn("shell", options)
        with patch("comparison.benchmark.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 10)):
            self.assertEqual(_launch({"threads": 1}, 10)["status"], "error")

    def test_benchmark_error_uses_same_scaled_norm(self):
        result = _errors([[0.2]], [[0.1]])
        self.assertAlmostEqual(result["relative_frobenius"], 0.1)
        self.assertFalse(result["within_tolerance"])
        with self.assertRaises(ValueError):
            _errors([[1, 2]], [[1]])

    def test_sampling_models_keep_the_correct_dependence_and_totals(self):
        class RecordingRng:
            def __init__(self):
                self.gamma_shapes = []
                self.poisson_means = []

            def gamma(self, shape, scale, size):
                self.gamma_shapes.append((shape, scale, size))
                return np.full(size, 1.5)

            def poisson(self, mean):
                self.poisson_means.append(np.asarray(mean))
                return np.ones_like(mean, dtype=int)

        for name in ("poisson", "independent_nb", "clustered_nb", "aggregated_nb"):
            case = make_case(name)
            rng = RecordingRng()
            sampled = simulate_counts(case, rng)
            self.assertEqual(sampled.shape, case.counts.shape)
            if name == "poisson":
                self.assertFalse(rng.gamma_shapes)
                assert_allclose(rng.poisson_means[0], case.means())
            elif name in ("independent_nb", "clustered_nb"):
                shape = case.counts.shape if name == "independent_nb" else (len(case.counts), 1)
                self.assertEqual(rng.gamma_shapes[0][2], shape)
                assert_allclose(rng.poisson_means[0], 1.5 * case.means())
            else:
                patterns, assignment = case.patterns
                for p in range(len(patterns)):
                    assert_allclose(sampled[assignment == p].sum(axis=0), 1)
                    assert_allclose(rng.poisson_means[p], 1.5 * case.means()[assignment == p].sum(axis=0))

    def test_calibration_denominators_and_endpoint_uncertainty(self):
        case = make_case()
        records = [
            {"fit_success": True, "inference_success": True,
             "coefficients": (case.coefficients + 0.1).tolist(),
             "covered": [True] * case.n_parameters, "null_moderator_rejected": False},
            {"fit_success": True, "inference_success": False,
             "coefficients": (case.coefficients + 0.3).tolist()},
            {"fit_success": False, "inference_success": False},
        ]
        result = summarize(case, records)
        self.assertEqual(result["bias_denominator"], 2)
        self.assertEqual(result["coverage_denominator"], 1)
        self.assertEqual(result["total_failed_replicates"], 2)
        assert_allclose(result["coefficient_bias"], 0.2)
        self.assertGreater(result["null_moderator_false_positive_interval95"][1], 0)
        self.assertLess(result["coverage_monte_carlo_interval95"][0][0], 1)

    def test_expected_failures_are_recorded_but_programming_errors_propagate(self):
        case = make_case()
        with patch("comparison.adapters.package_model", side_effect=ValueError("numerical failure")):
            result = fit_replicate(case)
            self.assertEqual(result["status"], "error")
            self.assertIn("numerical failure", result["errors"][0])
        with patch("comparison.adapters.package_model", side_effect=AssertionError("bug")):
            with self.assertRaises(AssertionError):
                fit_replicate(case)


if __name__ == "__main__":
    unittest.main()

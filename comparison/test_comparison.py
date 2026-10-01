"""Small infrastructure regressions; numerical case records come from comparison.run."""

import ast
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from .adapters import PRIVATE_PACKAGE, ROOT, _Imports, modules, package_model, source_manifest
from .cases import MODELS, make_case
from .metrics import errors, serializable, write_report


class ComparisonInfrastructureTests(unittest.TestCase):
    def test_seeded_designs_and_parameter_layout(self):
        for model in MODELS:
            for design in ("gc", "sv"):
                case = make_case(model, design)
                assert_allclose(case.counts, make_case(model, design).counts)
                matrix = case.cell_design().reshape(-1, case.n_parameters)
                expected = case.exposure[:, None] * np.exp(
                    (matrix @ case.coefficients).reshape(case.counts.shape)
                )
                assert_allclose(case.means(), expected)
                self.assertEqual(np.linalg.matrix_rank(matrix), case.n_parameters)
                _, assignment = case.patterns
                self.assertGreaterEqual(np.bincount(assignment).min(), 2)

    def test_original_adapter_redirects_only_import_nodes(self):
        source = (ROOT / "cbmr_code/distributions.py").read_text()
        original = ast.parse(source)
        changed = _Imports().visit(ast.parse(source))

        def without_imports(tree):
            class Remove(ast.NodeTransformer):
                def visit_ImportFrom(self, node):
                    return None
            return ast.dump(Remove().visit(tree), include_attributes=False)

        self.assertEqual(without_imports(original), without_imports(changed))
        before = source_manifest()
        case = make_case()
        old, new = package_model(case, "original"), package_model(case, "improved")
        self.assertTrue(type(old).__module__.startswith(PRIVATE_PACKAGE))
        self.assertEqual(type(new).__module__, "cbmr_improved.model")
        self.assertTrue(modules("original").model.__file__.endswith("/cbmr_code/model.py"))
        self.assertEqual(source_manifest(), before)
        with self.assertRaisesRegex(ValueError, "absent"):
            package_model(replace(case, model="independent_nb"), "original")

    def test_finite_metric_and_report_semantics(self):
        matrix = np.eye(2)
        report = errors(matrix + 1e-7, matrix)
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["relative_frobenius"], 2e-7 / np.sqrt(2))
        self.assertFalse(errors(np.array([np.nan]), np.array([1.0]))["passed"])
        self.assertFalse(errors(np.ones(3), np.ones(2))["passed"])
        raw = {"values": np.array([np.nan, np.inf, -np.inf]), "flag": np.bool_(True)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            write_report(output, raw)
            parsed = json.loads(output.read_text())
        self.assertEqual(parsed, serializable(raw))
        self.assertEqual(parsed["values"], ["NaN", "Infinity", "-Infinity"])

    def test_penalty_is_psd_and_does_not_penalize_globals(self):
        case = make_case(penalty=True)
        self.assertGreaterEqual(np.linalg.eigvalsh(case.penalty).min(), -1e-14)
        assert_allclose(case.penalty[case.n_spatial:], 0)

    def test_incorrect_actual_hessian_is_reported_as_failure(self):
        from . import derivatives

        case = make_case()
        actual = derivatives.package_information

        def perturbed(*args, **kwargs):
            return actual(*args, **kwargs) + 1e-3 * np.eye(case.n_parameters)

        with patch.object(derivatives, "package_information", side_effect=perturbed):
            rows = derivatives.derivative_checks(case)
        failures = [row for row in rows if row.get("passed") is False]
        self.assertTrue(any("actual Hessian" in row["comparison"] for row in failures))
        self.assertTrue(all(row["max_absolute"] > 0 for row in failures))


if __name__ == "__main__":
    unittest.main()

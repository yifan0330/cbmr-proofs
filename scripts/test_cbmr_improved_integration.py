"""Optional comparisons with the original source and NiMARE's inference API."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from scripts.test_cbmr_improved import fixture


@unittest.skipUnless(importlib.util.find_spec("nimare"), "High-level API needs compatible NiMARE")
class OriginalImplementationTests(unittest.TestCase):
    def test_original_hessians_are_preserved_numerically(self):
        from cbmr_code import information as original
        from cbmr_code.distributions import DISTRIBUTIONS
        from cbmr_code.covariance import fisher_covariance

        functions = {
            "poisson": lambda model, foci: original.poisson_information_matrix(model),
            "negativebinomial": original.negative_binomial_information_matrix,
            "clusterednegativebinomial": original.clustered_negative_binomial_information_matrix,
        }
        for distribution, function in functions.items():
            for formula in ("~ s(group)", "~ s(group) + age + exposure(duration)", "~ sz(group) + age"):
                with self.subTest(distribution=distribution, formula=formula):
                    model, foci = fixture(distribution, formula)
                    spatial, global_coef = model.unpack(model.coefficients)
                    nuisance = (
                        None if model.nuisance is None
                        else model.distribution.transform_nuisance(model.nuisance)
                    )
                    expected_likelihood = DISTRIBUTIONS[distribution]().log_likelihood(
                        model.predictor, spatial, global_coef, nuisance, foci
                    )
                    assert_allclose(model.log_likelihood(foci).detach().numpy(),
                                    expected_likelihood.detach().numpy(), rtol=2e-12, atol=2e-12)
                    expected = function(model, foci)
                    assert_allclose(model.information_matrix(foci), expected,
                                    rtol=2e-10, atol=2e-10)
                    assert_allclose(model.covariance(foci), fisher_covariance(model, expected),
                                    rtol=2e-9, atol=2e-9)

    def test_projected_hypotheses_match_original_statistics(self):
        from cbmr_code.contrasts import evaluate_hypotheses as original
        from cbmr_improved.contrasts import evaluate_hypotheses

        model, foci = fixture()
        for options in (
            {"term": "group", "method": "pairwise"},
            {"term": "age", "method": "zero"},
            {"hypotheses": ["a = 0", "b = 0"]},
        ):
            with self.subTest(options=options):
                expected = original(model, foci=foci, **options)
                with patch.object(model, "covariance", side_effect=AssertionError):
                    actual = evaluate_hypotheses(model, foci=foci, **options)
                self.assertEqual(actual["maps"].keys(), expected["maps"].keys())
                self.assertEqual(actual["tables"].keys(), expected["tables"].keys())
                for kind in ("maps", "tables"):
                    for key in actual[kind]:
                        assert_allclose(actual[kind][key], expected[kind][key], rtol=1e-10, atol=1e-10)

    def test_estimator_constructs_local_model(self):
        from cbmr_improved import CBMR
        from cbmr_improved.model import CBMRModel

        estimator = CBMR("~ s(group)", n_iter=40, random_state=0)
        model, foci = fixture(formula="~ s(group)")
        annotations = pd.DataFrame({"group": np.repeat(["a", "b"], 8)})
        estimator.inputs_ = {"foci": foci, "coef_spline_bases": model.predictor.bases}
        with patch.object(estimator, "_experiment_annotations", return_value=annotations):
            maps, tables, description = estimator._fit(None)
        self.assertIsInstance(estimator.cbmr_model, CBMRModel)
        self.assertTrue(maps)
        self.assertTrue(tables)
        self.assertTrue(description)

    def test_redundant_hypotheses_raise_instead_of_reporting_null(self):
        from cbmr_improved.contrasts import ContrastError, evaluate_hypotheses

        model, foci = fixture()
        with self.assertRaisesRegex(ContrastError, "positive definite"):
            evaluate_hypotheses(model, hypotheses=["a = 0", "a = 0"], foci=foci)


if __name__ == "__main__":
    unittest.main()

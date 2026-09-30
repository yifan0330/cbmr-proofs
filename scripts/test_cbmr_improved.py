"""Numerical regression checks for the opt-in implementation (not symbolic proofs)."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy import sparse

from cbmr_improved import CBMRModel, CBMRPredictor, Design, Poisson
from cbmr_improved._torch import torch
from cbmr_improved.covariance import (
    CovarianceError,
    _meat_iid,
    _scores_by_cluster,
    poisson_cluster_scores,
)
from cbmr_improved.information import closed_form_information
from cbmr_improved.terms import bind


def fixture(distribution="poisson", formula="~ s(group) + age + exposure(duration)"):
    rng = np.random.default_rng(18)
    annotations = pd.DataFrame({
        "group": np.repeat(["a", "b"], 8),
        "age": rng.normal(size=16),
        "duration": rng.uniform(0.5, 2, size=16),
    })
    bases = np.column_stack([np.ones(7), np.linspace(-1, 1, 7)])
    predictor = CBMRPredictor(bind(Design.from_formula(formula), annotations), bases)
    model = CBMRModel(predictor, distribution)
    with torch.no_grad():
        model.coefficients.copy_(torch.linspace(-0.4, 0.3, model.n_parameters))
        if model.nuisance is not None:
            model.nuisance.fill_(np.log(0.4))
    foci = rng.poisson(1.2, size=(16, 7)).astype(float)
    return model, foci


class InformationTests(unittest.TestCase):
    def test_three_likelihoods_against_autodiff(self):
        for distribution in ("poisson", "negativebinomial", "clusterednegativebinomial"):
            for formula in (
                "~ s(group)",
                "~ s(group) + age + exposure(duration)",
                "~ sz(group) + age",
            ):
                for use_sparse in (False, True):
                    with self.subTest(distribution=distribution, formula=formula, sparse=use_sparse):
                        model, foci = fixture(distribution, formula)
                        counts = sparse.csr_matrix(foci) if use_sparse else foci
                        expected = torch.func.hessian(
                            lambda flat: -model.log_likelihood(counts, flat=flat)
                        )(model.coefficients.detach()).detach().numpy()
                        operator = model.information_operator(counts)
                        assert_allclose(operator.to_dense(), expected, rtol=2e-10, atol=2e-10)
                        assert_allclose(model.information_matrix(counts), expected, rtol=2e-10, atol=2e-10)
                        inverse = np.linalg.inv(expected)
                        assert_allclose(operator.covariance(), inverse, rtol=2e-9, atol=2e-9)
                        assert_allclose(operator.diagonal(), np.diag(inverse), rtol=2e-9, atol=2e-9)

    def test_no_dense_information_for_standard_errors(self):
        model, foci = fixture()
        expected = np.sqrt(np.diag(np.linalg.inv(model.information_matrix(foci))))
        with patch("cbmr_improved.structured.StructuredInformation.to_dense", side_effect=AssertionError):
            errors = model.standard_errors(foci)
        assert_allclose(np.concatenate([value.ravel() for value in errors.values()]), expected)

    def test_many_groups_store_only_blocks_and_border(self):
        annotations = pd.DataFrame({
            "group": np.repeat([f"g{i:02}" for i in range(40)], 3),
            "age": np.tile([-1.0, 0.0, 1.0], 40),
        })
        bases = np.column_stack([np.ones(7), np.linspace(-1, 1, 7)])
        predictor = CBMRPredictor(bind(Design.from_formula("~ s(group) + age"), annotations), bases)
        model = CBMRModel(predictor)
        foci = sparse.csr_matrix(np.ones((120, 7)))
        with patch("cbmr_improved.structured.StructuredInformation.to_dense", side_effect=AssertionError):
            operator = model.information_operator(foci)
            self.assertEqual(len(operator.blocks), 40)
            self.assertEqual(sum(block.size for block in operator.blocks), 40 * 2 * 2)
            self.assertEqual(operator.cross.size, 80)
            operator.diagonal()
            contrast = np.zeros((1, model.n_parameters))
            contrast[0, 0], contrast[0, 2] = 1, -1
            self.assertGreater(operator.contrast_covariance(contrast)[0, 0], 0)

    def test_poisson_counts_drop_out_and_spatial_design(self):
        model, foci = fixture(formula="~ s(group) + s(age)")
        assert_allclose(model.information_matrix(foci), model.information_matrix(foci * 9))
        expected = torch.func.hessian(
            lambda flat: -model.log_likelihood(foci, flat=flat)
        )(model.coefficients.detach()).detach().numpy()
        assert_allclose(model.information_matrix(foci), expected)

    def test_custom_subclass_uses_own_hessian(self):
        class ChangedPoisson(Poisson):
            def log_likelihood(self, *args):
                return 2 * super().log_likelihood(*args)

        model, foci = fixture(ChangedPoisson())
        self.assertIsNone(closed_form_information(model.distribution))
        ordinary, _ = fixture()
        assert_allclose(model.information_matrix(foci), 2 * ordinary.information_matrix(foci))

    def test_mutations_do_not_reuse_stale_covariance(self):
        model, foci = fixture("negativebinomial")
        before = model.covariance(foci)
        foci *= 2
        after_counts = model.covariance(foci)
        self.assertFalse(np.allclose(before, after_counts))
        with torch.no_grad():
            model.coefficients.add_(0.1)
        self.assertFalse(np.allclose(after_counts, model.covariance(foci)))

    def test_ridge_is_explicit_and_consistent(self):
        model, foci = fixture()
        expected = np.linalg.inv(model.information_matrix(foci) + 0.2 * np.eye(model.n_parameters))
        assert_allclose(model.covariance(foci, ridge=0.2), expected)
        for ridge in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                model.covariance(foci, ridge=ridge)

    def test_nb_extreme_log_odds_remain_finite(self):
        model, foci = fixture("negativebinomial")
        with torch.no_grad():
            model.coefficients.fill_(0)
        for log_intensity in (-1000, -100, 100, 1000):
            with self.subTest(log_intensity=log_intensity):
                with torch.no_grad():
                    model.coefficients[:model.n_spatial:2] = log_intensity
                value = model.log_likelihood(foci)
                self.assertTrue(torch.isfinite(value))
                self.assertTrue(torch.isfinite(torch.autograd.grad(value, model.coefficients)[0]).all())
                self.assertTrue(np.isfinite(model.information_matrix(foci)).all())

    def test_nb_information_poisson_limit(self):
        model, foci = fixture("negativebinomial")
        poisson, _ = fixture()
        predictor = model.predictor
        marginal = torch.tensor(predictor.patterns.marginal_by_pattern(foci))

        def aggregate_poisson(flat):
            spatial, global_coef = model.unpack(flat)
            log_intensity = predictor.log_intensity_by_pattern(spatial)
            weights = predictor.experiment_weights(global_coef)
            totals = torch.stack([
                weights[predictor.patterns.assignment == p].sum()
                for p in range(predictor.patterns.n_patterns)
            ])
            return (
                torch.exp(log_intensity) * totals[:, None]
                - marginal * (log_intensity + torch.log(totals)[:, None])
            ).sum()

        expected = torch.func.hessian(aggregate_poisson)(
            model.coefficients.detach()
        ).detach().numpy()
        with torch.no_grad():
            model.nuisance.fill_(np.log(1e-12))
        actual = model.information_matrix(foci)
        assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
        full_poisson = poisson.information_matrix(foci)
        assert_allclose(actual[:model.n_spatial], full_poisson[:model.n_spatial],
                        rtol=2e-10, atol=2e-10)
        self.assertFalse(np.allclose(actual[-1:, -1:], full_poisson[-1:, -1:]))

    def test_noninteger_nb_counts_use_general_derivatives(self):
        model, foci = fixture("negativebinomial")
        foci = foci + 0.25
        expected = torch.func.hessian(
            lambda flat: -model.log_likelihood(foci, flat=flat)
        )(model.coefficients.detach()).detach().numpy()
        assert_allclose(model.information_matrix(foci), expected, rtol=2e-10, atol=2e-10)

    def test_fit_and_projected_covariance(self):
        model, foci = fixture()
        before = float(model(foci).detach())
        model.fit(foci, n_iter=60)
        self.assertLess(float(model(foci).detach()), before)
        contrast = np.zeros((2, model.n_parameters))
        contrast[:, :2] = np.eye(2)
        contrast[:, 2:4] = -np.eye(2)
        dense = model.covariance(foci)
        assert_allclose(model.contrast_covariance(foci, contrast), contrast @ dense @ contrast.T)


class SandwichTests(unittest.TestCase):
    def test_sparse_cluster_scores_and_sandwich(self):
        model, foci = fixture()
        spatial, global_coef = model.unpack(model.coefficients.detach())
        predictor = model.predictor
        mean = predictor.fitted_mean(spatial, global_coef).detach().numpy()
        scores = _scores_by_cluster(
            foci - mean, predictor.spatial_block, predictor.global_block, predictor.bases,
            model.n_spatial,
        )
        counts = sparse.csr_matrix(foci)
        with patch.object(predictor, "fitted_mean", side_effect=AssertionError):
            assert_allclose(poisson_cluster_scores(model, counts), scores)
            covariance = model.covariance(counts, cov_type="sandwich", correction="hc0")
        inverse = np.linalg.inv(model.information_matrix(foci))
        assert_allclose(covariance, inverse @ scores.T @ scores @ inverse, atol=1e-14)
        corrected = model.covariance(counts, cov_type="sandwich")
        assert_allclose(corrected, covariance * len(foci) / (len(foci) - model.n_parameters))

    def test_iid_sandwich_and_regularized_hc3(self):
        model, foci = fixture()
        predictor = model.predictor
        spatial, global_coef = model.unpack(model.coefficients.detach())
        mean = predictor.fitted_mean(spatial, global_coef).detach().numpy()
        cell_design = np.concatenate([
            np.einsum("ic,vk->ivck", predictor.spatial_block, predictor.bases).reshape(
                len(foci), predictor.n_voxels, model.n_spatial
            ),
            np.broadcast_to(predictor.global_block[:, None, :], (*foci.shape, model.n_global)),
        ], axis=2)
        inverse = np.linalg.inv(model.information_matrix(foci) + 0.1 * np.eye(model.n_parameters))
        for correction in ("hc0", "hc3"):
            residual = foci - mean
            if correction == "hc3":
                leverage = mean * np.einsum("ivp,pq,ivq->iv", cell_design, inverse, cell_design)
                residual /= np.clip(1 - leverage, 1e-6, None)
            meat = _meat_iid(residual, predictor.spatial_block, predictor.global_block,
                             predictor.bases, model.n_spatial)
            actual = model.covariance(foci, cov_type="sandwich", meat="iid",
                                      correction=correction, ridge=0.1)
            assert_allclose(actual, inverse @ meat @ inverse)

    def test_nb_rejects_poisson_sandwich(self):
        for distribution in ("negativebinomial", "clusterednegativebinomial"):
            model, foci = fixture(distribution)
            with self.assertRaisesRegex(CovarianceError, "Poisson scores"):
                model.covariance(foci, cov_type="sandwich")


if __name__ == "__main__":
    unittest.main()

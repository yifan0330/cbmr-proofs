"""Run with python -m unittest discover -s scripts -p test_structured_information.py."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from cbmr_improved.structured import StructuredInformation


class StructuredInformationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(940)
        self.indices = [np.array([4, 0]), np.array([2, 5, 1]), np.array([3])]
        self.blocks = []
        spatial = np.zeros((6, 6))
        for index in self.indices:
            a = self.rng.normal(size=(len(index), len(index)))
            block = a @ a.T + np.eye(len(index))
            self.blocks.append(block)
            spatial[np.ix_(index, index)] = block
        self.cross = self.rng.normal(size=(6, 2))
        a = self.rng.normal(size=(2, 2))
        self.global_block = (
            self.cross.T @ np.linalg.solve(spatial, self.cross) + a @ a.T + np.eye(2)
        )
        self.dense = np.block([[spatial, self.cross], [self.cross.T, self.global_block]])
        self.inverse = np.linalg.inv(self.dense)
        self.info = StructuredInformation(
            self.blocks, self.indices, self.cross, self.global_block
        )

    def test_dense_and_solves(self):
        self.assertEqual(self.info.shape, (8, 8))
        self.assertEqual((self.info.n_spatial, self.info.n_global), (6, 2))
        assert_allclose(self.info.to_dense(), self.dense)
        for rhs in [self.rng.normal(size=8), self.rng.normal(size=(8, 4))]:
            with self.subTest(shape=rhs.shape):
                assert_allclose(self.info.solve(rhs), np.linalg.solve(self.dense, rhs))
        assert_allclose(self.info.covariance(), self.inverse, atol=1e-14)
        assert_allclose(self.info.diagonal(), np.diag(self.inverse))
        assert_allclose(self.info.condition_number(), np.linalg.cond(self.dense))

    def test_contrasts_and_sandwich(self):
        scores = self.rng.normal(size=(17, 8))
        contrast = self.rng.normal(size=(3, 8))
        expected = self.inverse @ scores.T @ scores @ self.inverse
        assert_allclose(self.info.sandwich_covariance(scores), expected, atol=1e-13)
        assert_allclose(
            self.info.sandwich_covariance(scores, contrast),
            contrast @ expected @ contrast.T,
        )
        assert_allclose(
            self.info.contrast_covariance(contrast),
            contrast @ self.inverse @ contrast.T,
        )
        self.assertEqual(self.info.contrast_covariance(contrast[0]).shape, (1, 1))
        assert_allclose(
            self.info.sandwich_covariance(scores, contrast[0]),
            [[contrast[0] @ expected @ contrast[0]]],
        )
        assert_allclose(self.info.sandwich_covariance(np.empty((0, 8))), np.zeros((8, 8)))
        self.assertEqual(self.info.solve(np.empty((8, 0))).shape, (8, 0))
        self.assertEqual(self.info.contrast_covariance(np.empty((0, 8))).shape, (0, 0))

    def test_fast_path_never_assembles_dense(self):
        with patch.object(StructuredInformation, "to_dense", side_effect=AssertionError):
            info = StructuredInformation(self.blocks, self.indices, self.cross, self.global_block)
            assert_allclose(info.solve(np.eye(8)), self.inverse, atol=1e-14)
            assert_allclose(info.diagonal(), np.diag(self.inverse))
            info.contrast_covariance(np.ones(8))
            info.sandwich_covariance(np.ones((3, 8)))
            info.sandwich_covariance(np.ones((3, 8)), np.ones(8))

    def test_factorization_is_lazy_and_reused(self):
        from cbmr_improved.structured import linalg

        with patch.object(linalg, "cho_factor", wraps=linalg.cho_factor) as factor:
            info = StructuredInformation(self.blocks, self.indices, self.cross, self.global_block)
            info.to_dense()
            factor.assert_not_called()
            info.solve(np.ones(8))
            self.assertEqual(factor.call_count, len(self.blocks) + 1)
            info.diagonal()
            info.contrast_covariance(np.ones(8))
            self.assertEqual(factor.call_count, len(self.blocks) + 1)

    def test_no_border_and_empty_dimensions(self):
        info = StructuredInformation(self.blocks, self.indices, np.empty((6, 0)), np.empty((0, 0)))
        expected = np.linalg.inv(self.dense[:6, :6])
        with patch.object(info, "to_dense", side_effect=AssertionError):
            assert_allclose(info.covariance(), expected, atol=1e-14)
            assert_allclose(info.diagonal(), np.diag(expected))
            assert_allclose(info.condition_number(), np.linalg.cond(self.dense[:6, :6]))
        only_global = StructuredInformation([], [], np.empty((0, 2)), self.global_block)
        assert_allclose(only_global.covariance(), np.linalg.inv(self.global_block))
        assert_allclose(only_global.diagonal(), np.diag(np.linalg.inv(self.global_block)))
        empty = StructuredInformation([], [], np.empty((0, 0)), np.empty((0, 0)))
        self.assertEqual(empty.covariance().shape, (0, 0))
        self.assertEqual(empty.solve(np.empty(0)).shape, (0,))
        self.assertEqual(empty.diagonal().shape, (0,))
        self.assertEqual(empty.condition_number(), 1)
        with_empty_block = StructuredInformation([np.empty((0, 0))], [[]], np.empty((0, 0)), np.empty((0, 0)))
        self.assertEqual(with_empty_block.covariance().shape, (0, 0))

    def test_indefinite_and_singular_block_fallback(self):
        for d, c, e in [(-2., 1., 3.), (0., 1., 2.), (2., 1., -1.)]:
            with self.subTest(d=d, e=e):
                with self.assertLogs("cbmr_improved.structured", level="WARNING"):
                    info = StructuredInformation([[[d]]], [[0]], [[c]], [[e]])
                    info.solve([1, 2])
                dense = np.array([[d, c], [c, e]])
                expected = np.linalg.inv(dense)
                assert_allclose(info.covariance(), expected)
                assert_allclose(info.solve([1, 2]), np.linalg.solve(dense, [1, 2]))
                assert_allclose(info.diagonal(), np.diag(expected))
                assert_allclose(info.condition_number(), np.linalg.cond(dense))
                scores = np.array([[1., 2.], [3., 4.]])
                assert_allclose(info.sandwich_covariance(scores), expected @ scores.T @ scores @ expected)

    def test_singular_full_error(self):
        info = StructuredInformation([[[1.]]], [[0]], [[1.]], [[1.]])
        with self.assertLogs("cbmr_improved.structured", level="WARNING"):
            with self.assertRaisesRegex(np.linalg.LinAlgError, "singular.*identifiable"):
                info.solve([1., 1.])

    def test_global_uncertainty_correlates_spatial_groups(self):
        info = StructuredInformation([[[2.]], [[3.]]], [[0], [1]], [[1.], [1.]], [[2.]])
        covariance = info.covariance()
        self.assertGreater(covariance[0, 1], 0)
        self.assertGreater(covariance[0, 0], 1 / 2)
        assert_allclose(covariance, np.linalg.inv(info.to_dense()))
        assert_allclose(info.contrast_covariance([1, -1, 0]), [[
            covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
        ]])

    def test_defensive_immutability(self):
        self.blocks[0][:] = 0
        self.indices[0][:] = 0
        self.cross[:] = 0
        self.global_block[:] = 0
        assert_allclose(self.info.to_dense(), self.dense)
        assert_allclose(self.info.covariance(), self.inverse, atol=1e-14)
        for array in [*self.info.blocks, *self.info.indices, self.info.cross, self.info.global_block]:
            with self.assertRaises(ValueError):
                array.flat[0] = 0
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(AttributeError):
            self.info.cross = np.zeros((6, 2))
        self.info.to_dense()[:] = 0
        assert_allclose(self.info.to_dense(), self.dense)

    def test_invalid_constructor_inputs(self):
        valid = dict(blocks=[np.eye(2)], indices=[[0, 1]], cross=np.ones((2, 1)), global_block=[[3.]])
        invalid = [
            {"blocks": [np.ones((2, 3))]},
            {"blocks": [[[1., 1.], [0., 1.]]]},
            {"blocks": [[[1., np.nan], [np.nan, 1.]]]},
            {"blocks": [np.eye(2, dtype=complex)]},
            {"indices": []},
            {"indices": [[0, 0]]},
            {"indices": [[0, 2]]},
            {"indices": [[-1, 0]]},
            {"indices": [[0.0, 1.0]]},
            {"indices": [[[0, 1]]]},
            {"indices": [[0]]},
            {"cross": np.ones(2)},
            {"cross": [[np.inf], [1.]]},
            {"global_block": np.eye(2)},
            {"global_block": [1.]},
            {"global_block": [[np.nan]]},
            {"blocks": [], "indices": []},
        ]
        for change in invalid:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    StructuredInformation(**dict(valid, **change))

    def test_invalid_method_inputs(self):
        for rhs in [1, np.ones(7), np.ones((8, 1, 1)), np.full(8, np.nan), np.ones(8, dtype=complex)]:
            with self.subTest(rhs=np.shape(rhs)):
                with self.assertRaises(ValueError):
                    self.info.solve(rhs)
        for contrast in [1, np.ones(7), np.ones((8, 1)), np.full((1, 8), np.inf)]:
            with self.assertRaises(ValueError):
                self.info.contrast_covariance(contrast)
            with self.assertRaises(ValueError):
                self.info.sandwich_covariance(np.ones((3, 8)), contrast)
        for scores in [np.ones(8), np.ones((3, 7)), np.ones((1, 1, 8)), np.full((2, 8), np.nan)]:
            with self.assertRaises(ValueError):
                self.info.sandwich_covariance(scores)


if __name__ == "__main__":
    unittest.main()

"""Bordered block information, with regression nuisance parameters held fixed.

The block/Schur identities are verified in ``proofs/covariance.py``; projected
and sandwich identities in ``proofs/appendix.py``. Shared global uncertainty
and its induced spatial covariance are illustrated in ``proofs/published_cbmr.py``.
"""

import logging

import numpy as np
from scipy import linalg


_LOGGER = logging.getLogger(__name__)


def _real_array(value, name):
    """Convert real numeric input without silently dropping imaginary parts."""
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real")
        result = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real numbers") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite real numbers")
    return result


def _immutable(array):
    # A bytes-backed array cannot be made writable again with setflags().
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _symmetric(value, name):
    array = _real_array(value, name)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.allclose(array, array.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return _immutable(array * 0.5 + array.T * 0.5)


class StructuredInformation:
    """Represent ``H = [[D, C], [C.T, E]]`` without dense spatial storage.

    ``blocks`` are symmetric diagonal blocks of D; corresponding ``indices``
    partition spatial positions ``0 .. n_spatial - 1`` in any order.
    ``cross`` is C and ``global_block`` is E. Public arrays are immutable
    copies, and block/index collections are tuples. Empty spatial or global
    portions are supported, including a completely empty information matrix.

    SPD block and Schur Cholesky factors are computed lazily and cached. A non-SPD block or Schur
    complement triggers a logged fallback to a dense symmetric solve, allowing
    invertible indefinite observed Hessians. No pseudoinverse or regularization
    is used. Singular fallback systems raise ``numpy.linalg.LinAlgError`` on
    solving. See ``proofs/covariance.py`` for the Schur/block inverse identities.
    """

    def __init__(self, blocks, indices, cross, global_block):
        self._blocks = tuple(
            _symmetric(block, f"blocks[{i}]") for i, block in enumerate(blocks)
        )
        raw_indices = tuple(indices)
        if len(raw_indices) != len(self._blocks):
            raise ValueError("blocks and indices must have the same length")
        self._global_block = _symmetric(global_block, "global_block")
        cross_array = _real_array(cross, "cross")
        if cross_array.ndim != 2:
            raise ValueError("cross must be a two-dimensional matrix")
        self._n_spatial, self._n_global = cross_array.shape
        if self._global_block.shape != (self.n_global, self.n_global):
            raise ValueError("cross and global_block dimensions do not match")
        converted_indices = []
        for i, (raw, block) in enumerate(zip(raw_indices, self.blocks)):
            index = np.asarray(raw)
            if index.ndim != 1 or (index.size and index.dtype.kind not in "iu"):
                raise ValueError(f"indices[{i}] must be a one-dimensional integer array")
            if index.size != block.shape[0]:
                raise ValueError(f"indices[{i}] length must match its block dimension")
            if np.any(index < 0) or np.any(index >= self.n_spatial):
                raise ValueError("indices must partition spatial positions 0..n_spatial-1")
            converted_indices.append(_immutable(index.astype(np.intp)))
        self._indices = tuple(converted_indices)
        joined = np.concatenate(self.indices) if self.indices else np.empty(0, dtype=int)
        if not np.array_equal(np.sort(joined), np.arange(self.n_spatial)):
            raise ValueError("indices must partition spatial positions 0..n_spatial-1")
        self._cross = _immutable(cross_array)
        self._dense_fallback = None
        self._block_factors = ()
        self._schur_factor = None
        self._factorized = False

    def _factorize(self):
        if self._factorized:
            return
        try:
            self._block_factors = tuple(
                linalg.cho_factor(block, lower=True, check_finite=False)
                if block.size else None
                for block in self.blocks
            )
            self._w = self._solve_blocks(self.cross)
            if self.n_global:
                schur = self.global_block - self.cross.T @ self._w
                if not np.all(np.isfinite(schur)):
                    raise np.linalg.LinAlgError("nonfinite Schur complement")
                schur = schur * 0.5 + schur.T * 0.5
                self._schur_factor = linalg.cho_factor(
                    schur, lower=True, check_finite=False
                )
        except np.linalg.LinAlgError as exc:
            _LOGGER.warning(
                "Structured information Cholesky failed; using dense symmetric "
                "solve (non-SPD or singular block/Schur complement): %s", exc
            )
            self._dense_fallback = _immutable(self.to_dense())
        self._factorized = True

    @property
    def n_spatial(self):
        return self._n_spatial

    @property
    def n_global(self):
        return self._n_global

    @property
    def shape(self):
        size = self.n_spatial + self.n_global
        return (size, size)

    @property
    def blocks(self):
        return self._blocks

    @property
    def indices(self):
        return self._indices

    @property
    def cross(self):
        return self._cross

    @property
    def global_block(self):
        return self._global_block

    def to_dense(self):
        """Return a fresh dense information matrix (not its covariance)."""
        dense = np.zeros(self.shape)
        for block, index in zip(self.blocks, self.indices):
            dense[np.ix_(index, index)] = block
        dense[:self.n_spatial, self.n_spatial:] = self.cross
        dense[self.n_spatial:, :self.n_spatial] = self.cross.T
        dense[self.n_spatial:, self.n_spatial:] = self.global_block
        return dense

    def _solve_blocks(self, rhs):
        result = np.empty_like(rhs)
        for factor, index in zip(self._block_factors, self.indices):
            if index.size:
                result[index] = linalg.cho_solve(
                    factor, rhs[index], check_finite=False
                )
        return result

    def solve(self, rhs):
        """Solve H x = rhs for a finite vector or matrix, preserving its shape."""
        rhs = _real_array(rhs, "rhs")
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.shape[0]:
            raise ValueError("rhs must have shape (n_parameters,) or (n_parameters, k)")
        self._factorize()
        if self._dense_fallback is not None:
            try:
                return linalg.solve(
                    self._dense_fallback, rhs, assume_a="sym", check_finite=False
                )
            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    "Full structured information matrix is singular; "
                    "covariance requires identifiable parameters"
                ) from exc
        vector = rhs.ndim == 1
        rhs = rhs[:, None] if vector else rhs
        spatial = self._solve_blocks(rhs[:self.n_spatial])
        if self.n_global:
            global_solution = linalg.cho_solve(
                self._schur_factor,
                rhs[self.n_spatial:] - self.cross.T @ spatial,
                check_finite=False,
            )
            spatial -= self._w @ global_solution
            result = np.concatenate((spatial, global_solution), axis=0)
        else:
            result = spatial
        return result[:, 0] if vector else result

    def covariance(self):
        """Explicit compatibility API returning the full dense inverse of H."""
        covariance = self.solve(np.eye(self.shape[0]))
        return 0.5 * (covariance + covariance.T)

    def _contrast(self, contrast):
        contrast = _real_array(contrast, "contrast")
        if contrast.ndim == 1:
            contrast = contrast[None, :]
        if contrast.ndim != 2 or contrast.shape[1] != self.shape[0]:
            raise ValueError("contrast must have shape (k, n_parameters) or (n_parameters,)")
        return contrast

    def contrast_covariance(self, contrast):
        """Return R H^-1 R.T; a 1D contrast is one row and returns shape (1, 1).

        Uses the projected covariance identity in ``proofs/appendix.py``.
        """
        contrast = self._contrast(contrast)
        covariance = contrast @ self.solve(contrast.T)
        return 0.5 * (covariance + covariance.T)

    def diagonal(self):
        """Return inverse diagonal using block solves and Schur corrections.

        No full information/covariance matrix is assembled on the SPD path.
        The correction diag(W S^-1 W.T) retains shared global uncertainty
        (``proofs/published_cbmr.py``); it must not be dropped for spatial groups.
        """
        self._factorize()
        if self._dense_fallback is not None:
            return np.diag(self.covariance()).copy()
        result = np.empty(self.shape[0])
        for factor, block, index in zip(self._block_factors, self.blocks, self.indices):
            if index.size:
                result[index] = np.diag(linalg.cho_solve(
                    factor, np.eye(block.shape[0]), check_finite=False
                ))
        if self.n_global:
            correction = linalg.cho_solve(
                self._schur_factor, self._w.T, check_finite=False
            )
            result[:self.n_spatial] += np.einsum("ij,ji->i", self._w, correction)
            result[self.n_spatial:] = np.diag(linalg.cho_solve(
                self._schur_factor, np.eye(self.n_global), check_finite=False
            ))
        return result

    def sandwich_covariance(self, scores, contrast=None):
        """Return an unscaled score Gram sandwich, optionally contrast-projected.

        ``scores`` has shape (n_observations, n_parameters); rows may also be
        aggregated cluster scores. No dense inverse or meat matrix is formed.
        A 1D contrast returns shape (1, 1), as in ``contrast_covariance``.
        The score-projection identity is verified in ``proofs/appendix.py``.
        """
        scores = _real_array(scores, "scores")
        if scores.ndim != 2 or scores.shape[1] != self.shape[0]:
            raise ValueError("scores must have shape (n_observations, n_parameters)")
        if contrast is None:
            projected = self.solve(scores.T).T
        else:
            contrast = self._contrast(contrast)
            projected = scores @ self.solve(contrast.T)
        return projected.T @ projected

    def condition_number(self):
        """Return the exact spectral 2-norm condition number (empty H: 1).

        Bordered matrices require full symmetric eigenvalues: Schur eigenvalues
        are not the eigenvalues of H. Without a border, block spectra suffice.
        """
        if not self.shape[0]:
            return 1.0
        if self.n_global:
            eigenvalues = linalg.eigvalsh(self.to_dense(), check_finite=False)
        else:
            eigenvalues = np.concatenate([
                linalg.eigvalsh(block, check_finite=False) for block in self.blocks
            ])
        magnitudes = np.abs(eigenvalues)
        smallest = magnitudes.min()
        return float(magnitudes.max() / smallest) if smallest else float("inf")

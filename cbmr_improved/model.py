"""Fitting and covariance for a term-based CBMR model.

Holds one flat coefficient vector, sliced according to the design's parameter layout.


Information is stored as connected spatial blocks plus a global border.
Dense matrices are materialized only by the compatibility APIs or when a
non-positive-definite factorization requires a general fallback.
"""

import logging

import numpy as np

from cbmr_improved._torch import torch
from cbmr_improved.distributions import resolve_distribution
from cbmr_improved.information import closed_form_information

LGR = logging.getLogger(__name__)


class CBMRModel(torch.nn.Module):
    """A CBMR model over a term-based design.

    Parameters
    ----------
    predictor : :class:`~cbmr_improved.predictor.CBMRPredictor`
        Assembled linear predictor.
    distribution : :obj:`str`, :class:`~cbmr_improved.distributions.Distribution`, or class
        Observation distribution. Resolved by
        :func:`~cbmr_improved.distributions.resolve_distribution`.
    device : :obj:`str`, optional
        Torch device. Default is ``"cpu"``.
    """

    def __init__(self, predictor, distribution="poisson", device="cpu"):
        super().__init__()
        self.predictor = predictor
        self.distribution = resolve_distribution(distribution)
        self.distribution.check_design(predictor)
        self.device = device

        self.n_spatial = predictor.n_spatial_columns * predictor.n_bases
        self.n_global = predictor.n_global_columns
        self.n_nuisance = self.distribution.n_nuisance_parameters(predictor.patterns.n_patterns)

        # Small nonzero starting values, matching the historical uniform(-0.01, 0.01) init.
        generator = torch.Generator().manual_seed(0)
        start = (
            torch.rand(self.n_spatial + self.n_global, generator=generator, dtype=torch.float64)
            * 0.02
            - 0.01
        )
        self.coefficients = torch.nn.Parameter(start.to(device))

        nuisance = self.distribution.initial_nuisance(predictor.patterns.n_patterns)
        self.nuisance = (
            torch.nn.Parameter(nuisance.detach().to(device)) if nuisance is not None else None
        )
        self._iterations = 0

    @property
    def n_parameters(self):
        """Number of regression coefficients, excluding nuisance parameters."""
        return self.n_spatial + self.n_global

    def unpack(self, flat):
        """Split a flat coefficient vector into spatial and non-spatial parts."""
        spatial = flat[: self.n_spatial].reshape(
            self.predictor.n_spatial_columns, self.predictor.n_bases
        )
        global_coef = flat[self.n_spatial :] if self.n_global else None
        return spatial, global_coef

    def log_likelihood(self, foci, flat=None, nuisance=None):
        """Return the log-likelihood at ``flat``, defaulting to the current coefficients."""
        flat = self.coefficients if flat is None else flat
        raw_nuisance = self.nuisance if nuisance is None else nuisance
        spatial, global_coef = self.unpack(flat)
        transformed = (
            None if raw_nuisance is None else self.distribution.transform_nuisance(raw_nuisance)
        )
        return self.distribution.log_likelihood(
            self.predictor, spatial, global_coef, transformed, foci
        )

    def forward(self, foci):
        """Return the negative log-likelihood, the quantity being minimized."""
        return -self.log_likelihood(foci)

    def fit(self, foci, n_iter=1000, lr=1.0, tol=1e-8):
        """Fit by L-BFGS.

        Parameters
        ----------
        foci : :obj:`scipy.sparse.spmatrix` or :obj:`numpy.ndarray`
            Foci counts, of shape ``(n_experiments, n_voxels)``.
        n_iter : :obj:`int`, optional
            Maximum L-BFGS iterations. Default is 1000.
        lr : :obj:`float`, optional
            Learning rate. Default is 1.0.
        tol : :obj:`float`, optional
            Stopping tolerance on the change in the objective. Default is 1e-8.
        """
        parameters = [self.coefficients] + ([] if self.nuisance is None else [self.nuisance])
        optimizer = torch.optim.LBFGS(
            params=parameters,
            lr=lr,
            max_iter=n_iter,
            tolerance_change=tol,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = self(foci)
            loss.backward()
            return loss

        optimizer.step(closure)
        state = optimizer.state.get(parameters[0], {})
        self._iterations = int(state.get("n_iter", 0))

        loss = self(foci)
        if not torch.isfinite(loss):
            raise ValueError(
                f"The {self.distribution.name} log-likelihood became "
                f"{float(loss.detach())} during optimization. Try a smaller lr, a coarser "
                "spline_spacing, or the Poisson distribution."
            )
        return self

    def information_operator(self, foci, ridge=0.0):
        """Assemble block information and reusable solves without a dense Hessian.

        Dispersion stays fixed. Unknown likelihoods use row-wise reverse-mode
        differentiation, never a formula inherited from a different likelihood.
        ``ridge`` is explicit numerical regularization, not a fitted penalty.
        """
        from cbmr_improved.structured import StructuredInformation

        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative.")
        closed_form = closed_form_information(self.distribution)
        if closed_form is not None:
            information = closed_form(self, foci, structured=True)
        else:
            flat = self.coefficients.detach().clone()
            nuisance = None if self.nuisance is None else self.nuisance.detach().clone()

            def objective(vector):
                return -self.log_likelihood(foci, flat=vector, nuisance=nuisance)

            hessian = torch.func.jacrev(torch.func.jacrev(objective), chunk_size=1)(flat)
            dense = hessian.detach().cpu().numpy()
            dense = 0.5 * (dense + dense.T)
            information = StructuredInformation(
                [dense[: self.n_spatial, : self.n_spatial]],
                [np.arange(self.n_spatial)],
                dense[: self.n_spatial, self.n_spatial :],
                dense[self.n_spatial :, self.n_spatial :],
            )
        if ridge:
            information = StructuredInformation(
                [block + ridge * np.eye(len(block)) for block in information.blocks],
                information.indices,
                information.cross,
                information.global_block + ridge * np.eye(self.n_global),
            )
        return information

    def information_matrix(self, foci):
        """Return the observed Fisher information over the flat coefficient vector.

        One matrix over all coefficients, so the cross blocks between terms are present rather
        than assumed away. Nuisance parameters are held fixed at their fitted values.

        Computed in closed form for every distribution with automatic differentiation as the
        fallback for distributions added without a derivation.
        """
        return self.information_operator(foci).to_dense()

    def contrast_covariance(self, foci, contrast, ridge=0.0):
        """Return ``C H^-1 C.T`` by solving only for the requested contrasts."""
        return self.information_operator(foci, ridge=ridge).contrast_covariance(contrast)

    def covariance(self, foci, cov_type="fisher", meat="cluster", correction="hc1", ridge=0.0):
        """Return the coefficient covariance.

        Parameters
        ----------
        foci : array_like
            Foci counts the model was fitted to.
        cov_type : {"fisher", "sandwich"}, optional
            ``"fisher"`` inverts the selected likelihood's observed information.
            ``"sandwich"`` uses empirical Poisson scores and is available only
            for Poisson fits. Default is ``"fisher"``.
        meat, correction, ridge
            Sandwich score grouping and correction. ``ridge`` adds an explicit
            nonnegative diagonal term to the bread for either covariance type.

        Notes
        -----
        This compatibility API explicitly materializes the full covariance. Use
        ``information_operator`` or ``contrast_covariance`` to avoid it. Operators
        are snapshots; model calls rebuild them so parameter and foci mutations
        cannot reuse stale covariance.
        """
        from cbmr_improved.covariance import sandwich_covariance

        if cov_type == "sandwich":
            value = sandwich_covariance(self, foci, meat=meat, correction=correction, ridge=ridge)
        elif cov_type == "fisher":
            value = self.information_operator(foci, ridge=ridge).covariance()
        else:
            raise ValueError(f"cov_type must be 'fisher' or 'sandwich', got {cov_type!r}.")

        return value

    def standard_errors(self, foci, **covariance_kwargs):
        """Return coefficient standard errors, keyed by term.

        Parameters
        ----------
        foci : array_like
            Foci counts the model was fitted to.
        **covariance_kwargs
            Passed to :meth:`covariance`, so ``cov_type="sandwich"`` gives robust errors.

        Returns
        -------
        :obj:`dict`
            Maps the rendered term to an array of standard errors, shaped
            ``(n_columns, n_bases)`` for a spatial term and ``(n_columns,)`` otherwise.
        """
        if covariance_kwargs.get("cov_type", "fisher") == "fisher":
            unknown = set(covariance_kwargs) - {"cov_type", "meat", "correction", "ridge"}
            if unknown:
                raise TypeError(f"Unexpected covariance options: {sorted(unknown)}")
            variances = self.information_operator(
                foci, ridge=covariance_kwargs.get("ridge", 0.0)
            ).diagonal()
        else:
            variances = np.diag(self.covariance(foci, **covariance_kwargs))
        if np.any(variances < 0):
            raise ValueError(
                "Negative coefficient variance: the fitted information is not positive definite."
            )
        errors = np.sqrt(variances)
        result = {}
        for name, term_slice in self.predictor.design.parameter_slices(
            self.predictor.n_bases
        ).items():
            block = errors[term_slice]
            term = next(t for t in self.predictor.design.terms if str(t) == name)
            columns = next(
                b.n_columns for b in self.predictor.design.blocks if str(b.term) == name
            )
            result[name] = block.reshape(columns, -1) if term.spatial else block
        return result

    def overdispersion(self):
        """Return the fitted overdispersion per spatial pattern, or None if there is none.

        Reported on the statistical scale, not the unconstrained scale the optimizer works on.
        """
        if self.nuisance is None:
            return None
        with torch.no_grad():
            return self.distribution.transform_nuisance(self.nuisance).cpu().numpy()

    def fitted_coefficients(self):
        """Return the fitted coefficients, keyed by term, in the design's layout."""
        flat = self.coefficients.detach().cpu().numpy()
        result = {}
        for name, term_slice in self.predictor.design.parameter_slices(
            self.predictor.n_bases
        ).items():
            term = next(t for t in self.predictor.design.terms if str(t) == name)
            columns = next(
                b.n_columns for b in self.predictor.design.blocks if str(b.term) == name
            )
            block = flat[term_slice]
            result[name] = block.reshape(columns, -1) if term.spatial else block
        return result

    def log_intensity(self):
        """Return the fitted log spatial intensity, one row per spatial pattern."""
        spatial, _ = self.unpack(self.coefficients.detach())
        return self.predictor.log_intensity_by_pattern(spatial).cpu().numpy()

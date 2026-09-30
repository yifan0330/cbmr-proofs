"""Closed-form observed information for the CBMR distributions.

``torch.func.hessian`` is ``jacfwd(jacrev(...))``. Forward mode carries one tangent per
parameter, so its intermediate has shape ``(n_parameters, n_patterns, n_voxels)`` -- 28.7 GB for
a 21-group model over 21,789 voxels. All three distributions have a closed form that avoids it::

    H_bb[(c,k),(c',k')] = sum_p L_pc L_pc' (B^T Sigma_p B)[k,k']

Only ``Sigma_p``, the second derivative of the pattern's term in the log-intensity, differs
between them. ``structured=True`` stores only coupled spatial components and
the global border; the default materializes the same dense matrix as the
original implementation.
"""

import numpy as np
from scipy.special import digamma, expit, polygamma

from cbmr_improved.distributions import (
    ClusteredNegativeBinomial,
    NegativeBinomial,
    Poisson,
)
from cbmr_improved.predictor import experiment_totals
from cbmr_improved.covariance import spatial_components
from cbmr_improved.structured import StructuredInformation


def _intensity_pieces(model, *, log_scale=False):
    """Return the fitted intensity and per-experiment weights all closed forms share.

    ``weight`` is ``E_i exp(m_i)``, and it comes from
    :meth:`~cbmr_improved.predictor.CBMRPredictor.experiment_weights` rather than being
    rebuilt here. Rebuilding it was how this function could disagree with the likelihood: the
    exposure would have been present in the fit and absent from every closed form, while the
    autodiff fallback stayed correct and so agreed with neither.
    """
    predictor = model.predictor
    flat = model.coefficients.detach().cpu().numpy()
    n_spatial, n_global = model.n_spatial, model.n_global

    spatial = flat[:n_spatial].reshape(predictor.n_spatial_columns, predictor.n_bases)
    loadings = predictor.patterns.loadings
    intensity = (loadings @ spatial) @ predictor.bases.T
    if not log_scale:
        intensity = np.exp(intensity)

    global_coef = model.unpack(model.coefficients.detach())[1]
    weight = predictor.experiment_weights(global_coef).detach().cpu().numpy()
    global_block = predictor.global_block if n_global else None
    return predictor, loadings, intensity, global_block, weight


def _symmetrize(matrix):
    """Return the exact symmetric part of a matrix that is symmetric in exact arithmetic.

    ``B^T diag(w) B`` is symmetric on paper, but BLAS sums the ``[i, j]`` and ``[j, i]`` dot
    products in different orders, so they can differ in the last bit. ``eigvalsh`` and the
    Cholesky read one triangle only, so that difference would quietly decide which value is used.
    """
    return 0.5 * (matrix + matrix.T)


class _InformationBuilder:
    """Assemble only structurally coupled spatial blocks and the global border."""

    def __init__(self, model):
        self.indices = spatial_components(model.predictor.patterns.loadings, model.predictor.n_bases)
        self.blocks = [np.zeros((len(index), len(index))) for index in self.indices]
        self.cross = np.zeros((model.n_spatial, model.n_global))
        self.global_block = np.zeros((model.n_global, model.n_global))
        self.locations = {}
        width = model.predictor.n_bases
        for component, index in enumerate(self.indices):
            for offset in range(0, len(index), width):
                self.locations[index[offset] // width] = (component, offset)

    def finish(self, structured):
        value = StructuredInformation(
            self.blocks, self.indices, self.cross, _symmetrize(self.global_block)
        )
        return value if structured else value.to_dense()


def _scatter_spatial(information, row, n_bases, block):
    """Add ``L_pc L_pc' * block`` into every spatial cross block the pattern's support touches."""
    block = _symmetrize(block)
    support = np.flatnonzero(row)
    for c in support:
        component, offset = information.locations[c]
        rows = slice(offset, offset + n_bases)
        for d in support:
            other, start = information.locations[d]
            if other != component:
                raise ValueError("Pattern support crosses supposedly disconnected components.")
            information.blocks[component][rows, start : start + n_bases] += (
                row[c] * row[d]
            ) * block


def _scatter_cross(information, row, n_bases, n_spatial, vector, moderator_row):
    """Add ``L_pc * vector_k * moderator_q`` into the spatial-by-global cross block."""
    outer = np.outer(vector, moderator_row)
    for c in np.flatnonzero(row):
        information.cross[c * n_bases : (c + 1) * n_bases] += row[c] * outer


def _nuisance(model):
    """Return the fitted nuisance parameters on the statistical scale."""
    return model.distribution.transform_nuisance(model.nuisance).detach().cpu().numpy()


def _nb_shape_derivatives(r, counts, normalizer):
    """Derivatives in log(R), using gamma recurrences for ordinary small counts.

    Summing ``r/(r+j)`` and ``r*j/(r+j)^2`` avoids subtracting
    nearly identical special functions near the Poisson limit. Large or
    noninteger counts retain the general special-function identity.
    """
    if np.all(counts == np.floor(counts)) and np.all(counts >= 0) and counts.max(initial=0) <= 4096:
        histogram = np.bincount(counts.astype(int))
        multiplicity = np.cumsum(histogram[:0:-1])[::-1]
        j = np.arange(len(multiplicity), dtype=float)
        fraction = r / (r + j)
        first = normalizer - multiplicity @ fraction
        second = normalizer - multiplicity @ (fraction * (j / (r + j)))
    else:
        first = normalizer - r * (digamma(counts + r) - digamma(r)).sum()
        second = first + r**2 * (polygamma(1, r) - polygamma(1, counts + r)).sum()
    return first, second


def poisson_information_matrix(model, *, structured=False):
    """Return the observed Fisher information of a fitted Poisson model.

    Takes no foci. Under a log link the Poisson information depends only on the fitted
    coefficients, so the counts drop out.

    Parameters
    ----------
    model : :class:`~cbmr_improved.model.CBMRModel`
        Fitted model.
    structured : bool, optional
        Return a ``StructuredInformation`` operator instead of a dense matrix.

    Returns
    -------
    :obj:`numpy.ndarray`
        Shape ``(n_parameters, n_parameters)``.
    """
    predictor, loadings, intensity, global_block, weight = _intensity_pieces(model)
    n_spatial, n_global, n_bases = model.n_spatial, model.n_global, predictor.n_bases
    bases = predictor.bases
    assignment = predictor.patterns.assignment

    total = np.zeros(predictor.patterns.n_patterns)  # T_p
    np.add.at(total, assignment, weight)

    information = _InformationBuilder(model)
    for p, row in enumerate(loadings):
        _scatter_spatial(
            information, row, n_bases, bases.T @ (bases * (total[p] * intensity[p])[:, None])
        )

    if n_global:
        marginal_basis = intensity @ bases  # a_pk
        moderator = np.zeros((predictor.patterns.n_patterns, n_global))  # U_pq
        np.add.at(moderator, assignment, weight[:, None] * global_block)
        information.cross = np.einsum(
            "pc,pk,pq->ckq", loadings, marginal_basis, moderator, optimize=True
        ).reshape(n_spatial, n_global)

        energy = intensity.sum(axis=1)  # E_p
        information.global_block = global_block.T @ (
            global_block * (weight * energy[assignment])[:, None]
        )
    return information.finish(structured)


def negative_binomial_information_matrix(model, foci, *, structured=False):
    """Return the observed Fisher information of a fitted NegativeBinomial model.

    Parameters
    ----------
    model : :class:`~cbmr_improved.model.CBMRModel`
        Fitted model.
    foci : array_like or :obj:`scipy.sparse.spmatrix`
        Foci counts. Used, unlike the Poisson case, but only through the pattern marginals.
    structured : bool, optional
        Return a ``StructuredInformation`` operator instead of a dense matrix.

    Returns
    -------
    :obj:`numpy.ndarray`
        Shape ``(n_parameters, n_parameters)``.

    Notes
    -----
    Uses ``A = S1/(theta S2)`` and ``R = S1 A`` in place of NiMARE's shape and probability. In
    those terms the gamma functions depend only on ``R``, the voxels only on the log-intensity,
    and the moderators only on ``R`` and ``A``.

    Log-scale chain factors avoid large powers of ``A``. Gamma recurrences
    stabilize the shape derivatives for small integer counts near the Poisson
    limit; general noninteger or large counts still use special functions.
    """
    predictor, loadings, log_intensity, global_block, weight = _intensity_pieces(model, log_scale=True)
    n_spatial, n_global, n_bases = model.n_spatial, model.n_global, predictor.n_bases
    bases = predictor.bases
    assignment = predictor.patterns.assignment
    overdispersion = _nuisance(model)
    marginal = predictor.patterns.marginal_by_pattern(foci)

    information = _InformationBuilder(model)
    for p, row in enumerate(loadings):
        members = np.flatnonzero(assignment == p)
        member_weight = weight[members]
        sum_1 = member_weight.sum()
        sum_2 = np.square(member_weight).sum()
        a = sum_1 / (overdispersion[p] * sum_2)
        r = sum_1 * a

        counts = marginal[p]
        log_ratio = log_intensity[p] - np.log(a)
        probability = expit(log_ratio)
        complement = expit(-log_ratio)

        weight_v = (r + counts) * probability * complement
        _scatter_spatial(information, row, n_bases, bases.T @ (bases * weight_v[:, None]))
        if not n_global:
            continue

        # dR/dgamma and dA/dgamma, through their log derivatives.
        member_block = global_block[members]
        u_1 = member_weight @ member_block
        u_2 = np.square(member_weight) @ member_block
        u_1_2 = member_block.T @ (member_block * member_weight[:, None])
        u_2_2 = member_block.T @ (member_block * np.square(member_weight)[:, None])
        d_log_1, d_log_2 = u_1 / sum_1, 2 * u_2 / sum_2
        dd_log_1 = u_1_2 / sum_1 - np.outer(u_1, u_1) / sum_1**2
        dd_log_2 = 4 * u_2_2 / sum_2 - 4 * np.outer(u_2, u_2) / sum_2**2
        d_log_a, d_log_r = d_log_1 - d_log_2, 2 * d_log_1 - d_log_2
        _scatter_cross(
            information, row, n_bases, n_spatial, bases.T @ (r * probability), d_log_r
        )
        _scatter_cross(information, row, n_bases, n_spatial, -(bases.T @ weight_v), d_log_a)

        psi_r, psi_rr = _nb_shape_derivatives(
            r, counts, r * np.logaddexp(0, log_ratio).sum()
        )
        psi_a = (counts * complement - r * probability).sum()
        psi_ra = -(r * probability).sum()
        information.global_block += (
            psi_rr * np.outer(d_log_r, d_log_r)
            + psi_ra * (np.outer(d_log_r, d_log_a) + np.outer(d_log_a, d_log_r))
            + weight_v.sum() * np.outer(d_log_a, d_log_a)
            + psi_r * (2 * dd_log_1 - dd_log_2)
            + psi_a * (dd_log_1 - dd_log_2)
        )

    return information.finish(structured)


def clustered_negative_binomial_information_matrix(model, foci, *, structured=False):
    """Return the observed Fisher information of a fitted ClusteredNegativeBinomial model.

    Parameters
    ----------
    model : :class:`~cbmr_improved.model.CBMRModel`
        Fitted model.
    foci : array_like or :obj:`scipy.sparse.spmatrix`
        Foci counts. Only the per-experiment totals matter here.
    structured : bool, optional
        Return a ``StructuredInformation`` operator instead of a dense matrix.

    Returns
    -------
    :obj:`numpy.ndarray`
        Shape ``(n_parameters, n_parameters)``.

    Notes
    -----
    The spatial coefficients reach this likelihood only through the scalar
    ``E_p = sum_v exp(S_pv)``. That makes the spatial block ``B^T diag(w) B`` plus a rank-one
    correction, where the other two distributions need only the first term.
    """
    predictor, loadings, intensity, global_block, weight = _intensity_pieces(model)
    n_spatial, n_global, n_bases = model.n_spatial, model.n_global, predictor.n_bases
    bases = predictor.bases
    assignment = predictor.patterns.assignment
    overdispersion = _nuisance(model)
    per_experiment = experiment_totals(foci)

    information = _InformationBuilder(model)
    for p, row in enumerate(loadings):
        members = np.flatnonzero(assignment == p)
        precision = 1.0 / overdispersion[p]
        u = intensity[p]
        energy = u.sum()  # E_p

        member_weight = weight[members]
        excess = per_experiment[members] + precision
        denominator = energy * member_weight + precision

        gradient = (excess * member_weight / denominator).sum()
        curvature = -(excess * np.square(member_weight) / denominator**2).sum()
        marginal_basis = bases.T @ u  # a_k

        block = gradient * (bases.T @ (bases * u[:, None])) + curvature * np.outer(
            marginal_basis, marginal_basis
        )
        _scatter_spatial(information, row, n_bases, block)
        if not n_global:
            continue

        member_block = global_block[members]
        scale = excess * precision * member_weight / denominator**2
        _scatter_cross(information, row, n_bases, n_spatial, marginal_basis, scale @ member_block)
        information.global_block += member_block.T @ (
            member_block * (scale * energy)[:, None]
        )

    return information.finish(structured)


def _poisson_entry(model, foci, *, structured=False):
    """Adapt the Poisson signature to the ``(model, foci)`` the dispatch uses."""
    return poisson_information_matrix(model, structured=structured)


# Exact classes only: a subclass may override the likelihood.
CLOSED_FORMS = (
    (ClusteredNegativeBinomial, clustered_negative_binomial_information_matrix),
    (NegativeBinomial, negative_binomial_information_matrix),
    (Poisson, _poisson_entry),
)


def closed_form_information(distribution):
    """Return the closed-form information function for ``distribution``, or None.

    Parameters
    ----------
    distribution : :class:`~cbmr_improved.distributions.Distribution`
        Observation distribution.

    Returns
    -------
    callable or None
        None means no derivation covers this distribution, and the caller should fall back to
        automatic differentiation rather than use another distribution's formula.
    """
    for klass, function in CLOSED_FORMS:
        if type(distribution) is klass:
            return function
    return None

"""Independent NumPy likelihoods and analytic appendix derivatives.

Scores differentiate the log likelihood; Hessians differentiate its negative.
Dispersion and the optional quadratic penalty matrix are held fixed.
"""

import numpy as np
from scipy.special import digamma, expit, gammaln, logsumexp, polygamma


def _coefficients(case, coefficients):
    values = case.coefficients if coefficients is None else coefficients
    # Accept tensor inputs without importing or using their differentiation API.
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _design(case):
    # Construct the study-major cell Jacobian independently of Case helpers.
    spatial = (case.spatial[:, None, :, None] * case.bases[None, :, None, :]).reshape(
        *case.counts.shape, case.n_spatial
    )
    global_design = np.broadcast_to(
        case.global_design[:, None, :], (*case.counts.shape, case.n_global)
    )
    return np.concatenate((spatial, global_design), axis=-1)


def _aggregate(case, theta):
    patterns, inverse = case.patterns
    p = len(theta)
    k = case.n_spatial
    beta = theta[:k].reshape(case.spatial.shape[1], case.bases.shape[1])
    log_weights = np.log(case.exposure) + case.global_design @ theta[k:]
    value = 0.0
    gradient = np.zeros(p)
    hessian = np.zeros((p, p))
    for index, pattern in enumerate(patterns):
        mask = inverse == index
        g = case.global_design[mask]
        lw = log_weights[mask]
        l1, l2 = logsumexp(lw), logsumexp(2 * lw)
        w1, w2 = np.exp(lw - l1), np.exp(2 * lw - l2)
        mean1, mean2 = w1 @ g, w2 @ g
        centered1, centered2 = g - mean1, g - mean2
        cov1 = centered1.T @ (w1[:, None] * centered1)
        cov2 = centered2.T @ (w2[:, None] * centered2)
        log_rho = 2 * l1 - l2 - np.log(case.dispersion)
        log_a = l1 - l2 - np.log(case.dispersion)
        rho = np.exp(log_rho)
        y = case.counts[mask].sum(axis=0)
        s = pattern @ beta @ case.bases.T
        t = s - log_a
        probability, complement = expit(t), expit(-t)
        softplus = np.logaddexp(0, t)
        value += np.sum(
            gammaln(rho) - gammaln(y + rho) + gammaln(y + 1)
            + rho * softplus + y * np.logaddexp(0, -t)
        )

        # Differentiate F(t, r), where t=s-log(A) and r=log(rho).
        f_t = (rho + y) * probability - y
        f_r = rho * (digamma(rho) - digamma(y + rho) + softplus)
        f_tt = (rho + y) * probability * complement
        f_tr = rho * probability
        f_rr = f_r + rho ** 2 * (polygamma(1, rho) - polygamma(1, y + rho))
        j_t = np.zeros((len(case.bases), p))
        j_t[:, :k] = (pattern[None, :, None] * case.bases[:, None, :]).reshape(-1, k)
        j_t[:, k:] = -(mean1 - 2 * mean2)
        j_r = np.zeros(p)
        j_r[k:] = 2 * mean1 - 2 * mean2
        gradient += j_t.T @ f_t + j_r * np.sum(f_r)
        hessian += j_t.T @ (f_tt[:, None] * j_t)
        cross = j_t.T @ f_tr
        hessian += np.outer(cross, j_r) + np.outer(j_r, cross)
        hessian += np.sum(f_rr) * np.outer(j_r, j_r)
        # These second-chain terms do not vanish away from a stationary point.
        hessian[k:, k:] += (
            np.sum(f_r) * (2 * cov1 - 4 * cov2)
            - np.sum(f_t) * (cov1 - 4 * cov2)
        )
    return value, -gradient, hessian


def evaluate(case, coefficients=None):
    """Return stable objective, penalized score, and observed NLL Hessian."""
    theta = _coefficients(case, coefficients)
    if case.model == "aggregated_nb":
        value, score, hessian = _aggregate(case, theta)
    elif case.model in ("poisson", "independent_nb", "clustered_nb"):
        x = _design(case)
        eta = x @ theta
        log_mean = eta + np.log(case.exposure)[:, None]
        y = np.asarray(case.counts, dtype=np.float64)
        nu = 1.0 / case.dispersion
        if case.model == "poisson":
            means = np.exp(log_mean)
            value = np.sum(means - y * eta)
            residual, curvature = y - means, means
        elif case.model == "independent_nb":
            t = log_mean - np.log(nu)
            probability, complement = expit(t), expit(-t)
            value = np.sum(
                gammaln(nu) - gammaln(y + nu) + gammaln(y + 1)
                + nu * np.logaddexp(0, t) + y * np.logaddexp(0, -t)
            )
            residual = y - (y + nu) * probability
            curvature = (y + nu) * probability * complement
        else:
            totals = y.sum(axis=1)
            log_d = np.logaddexp(np.log(nu), logsumexp(log_mean, axis=1))
            fractions = np.exp(log_mean - log_d[:, None])
            fitted_term = (totals + nu)[:, None] * fractions
            value = -np.sum(
                nu * np.log(nu) - gammaln(nu) + gammaln(totals + nu)
                - (totals + nu) * log_d + np.sum(y * eta, axis=1)
            )
            residual, curvature = y - fitted_term, fitted_term
        score = np.einsum("ivp,iv->p", x, residual)
        hessian = np.einsum("ivp,iv,ivq->pq", x, curvature, x)
        if case.model == "clustered_nb":
            projected = np.einsum("ivp,iv->ip", x, fractions)
            hessian -= np.einsum("ip,i,iq->pq", projected, totals + nu, projected)
    else:
        raise ValueError(f"Unknown model: {case.model!r}")
    if case.penalty is not None:
        penalty = np.asarray(case.penalty, dtype=np.float64)
        value += 0.5 * theta @ penalty @ theta
        score -= penalty @ theta
        hessian += penalty
    return float(value), score, hessian


def fisher_information(case, coefficients=None):
    """Expected fixed-dispersion information, including fixed penalty curvature.

    The independent NB model is helper-only (no original CBMR equivalent).
    For these cellwise models expectation replaces counts by fitted means.
    Moment-matched aggregated NB is deliberately not treated as independent NB.
    """
    if case.model not in ("poisson", "independent_nb", "clustered_nb"):
        raise ValueError(f"Fisher information is not implemented for {case.model!r}.")
    theta = _coefficients(case, coefficients)
    x = _design(case)
    log_mean = x @ theta + np.log(case.exposure)[:, None]
    means = np.exp(log_mean)
    if case.model == "independent_nb":
        curvature = expit(log_mean + np.log(case.dispersion)) / case.dispersion
    else:
        curvature = means
    information = np.einsum("ivp,iv,ivq->pq", x, curvature, x)
    if case.model == "clustered_nb":
        log_d = np.logaddexp(-np.log(case.dispersion), logsumexp(log_mean, axis=1))
        scaled = np.exp(log_mean - 0.5 * log_d[:, None])
        projected = np.einsum("ivp,iv->ip", x, scaled)
        information -= projected.T @ projected
    if case.penalty is not None:
        information += case.penalty
    return information

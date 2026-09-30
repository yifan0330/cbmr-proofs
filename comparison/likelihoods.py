"""Independent float64 likelihood references, with no model-package imports.

``objective`` intentionally uses the original probability parametrizations,
including their numerical weaknesses. ``stable_objective`` is a separate,
algebraically equivalent implementation suitable for optimization.
"""

import torch


def _tensor(value, like=None):
    return torch.as_tensor(
        value, dtype=torch.float64, device=None if like is None else like.device
    )


def _inputs(case, coefficients):
    theta = _tensor(case.coefficients if coefficients is None else coefficients)
    spatial = _tensor(case.spatial, theta)
    bases = _tensor(case.bases, theta)
    global_design = _tensor(case.global_design, theta)
    beta = theta[:case.n_spatial].reshape(spatial.shape[1], bases.shape[1])
    spatial_eta = spatial @ beta @ bases.T
    global_eta = global_design @ theta[case.n_spatial:]
    return theta, spatial_eta, global_eta, _tensor(case.counts, theta)


def _penalty(case, theta):
    if case.penalty is None:
        return theta.new_zeros(())
    return 0.5 * theta @ _tensor(case.penalty, theta) @ theta


def _objective(case, coefficients, stable):
    theta, spatial_eta, global_eta, counts = _inputs(case, coefficients)
    eta = spatial_eta + global_eta[:, None]
    exposure = _tensor(case.exposure, theta)
    log_mean = eta + torch.log(exposure)[:, None]
    nu = theta.new_tensor(1.0 / case.dispersion)
    if case.model == "poisson":
        means = exposure[:, None] * torch.exp(eta)
        result = torch.sum(means - counts * eta)
    elif case.model == "independent_nb":
        constant = torch.lgamma(counts + nu) - torch.lgamma(counts + 1) - torch.lgamma(nu)
        if stable:
            t = log_mean - torch.log(nu)
            loglik = constant - nu * torch.logaddexp(torch.zeros_like(t), t)
            loglik -= counts * torch.logaddexp(torch.zeros_like(t), -t)
        else:
            means = exposure[:, None] * torch.exp(eta)
            probability = means / (means + nu)
            loglik = constant + nu * torch.log(1 - probability) + counts * torch.log(probability)
        result = -torch.sum(loglik)
    elif case.model == "aggregated_nb":
        patterns, inverse = case.patterns
        beta = theta[:case.n_spatial].reshape(case.spatial.shape[1], case.bases.shape[1])
        pattern_eta = _tensor(patterns, theta) @ beta @ _tensor(case.bases, theta).T
        log_weights = torch.log(exposure) + global_eta
        result = theta.new_zeros(())
        for index in range(len(patterns)):
            mask = _tensor(inverse == index, theta).bool()
            y = torch.sum(counts[mask], dim=0)
            if stable:
                log_s1 = torch.logsumexp(log_weights[mask], dim=0)
                log_s2 = torch.logsumexp(2 * log_weights[mask], dim=0)
                log_alpha = theta.new_tensor(case.dispersion).log()
                rho = torch.exp(2 * log_s1 - log_s2 - log_alpha)
                log_a = log_s1 - log_s2 - log_alpha
                t = pattern_eta[index] - log_a
                loglik = torch.lgamma(y + rho) - torch.lgamma(y + 1) - torch.lgamma(rho)
                loglik -= rho * torch.logaddexp(torch.zeros_like(t), t)
                loglik -= y * torch.logaddexp(torch.zeros_like(t), -t)
            else:
                weights = exposure[mask] * torch.exp(global_eta[mask])
                s1 = torch.sum(weights)
                s2 = torch.sum(weights ** 2)
                rho = s1 ** 2 / (case.dispersion * s2)
                a = s1 / (case.dispersion * s2)
                spatial_mean = torch.exp(pattern_eta[index])
                probability = spatial_mean / (spatial_mean + a)
                loglik = torch.lgamma(y + rho) - torch.lgamma(y + 1) - torch.lgamma(rho)
                loglik += rho * torch.log(1 - probability) + y * torch.log(probability)
            result = result - torch.sum(loglik)
    elif case.model == "clustered_nb":
        totals = torch.sum(counts, dim=1)
        if stable:
            log_total_mean = torch.logsumexp(log_mean, dim=1)
            log_denominator = torch.logaddexp(torch.log(nu), log_total_mean)
        else:
            means = exposure[:, None] * torch.exp(eta)
            log_denominator = torch.log(nu + torch.sum(means, dim=1))
        loglik = nu * torch.log(nu) - torch.lgamma(nu) + torch.lgamma(totals + nu)
        loglik -= (totals + nu) * log_denominator
        loglik += torch.sum(counts * eta, dim=1)
        result = -torch.sum(loglik)
    else:
        raise ValueError(f"Unknown model: {case.model!r}")
    return result + _penalty(case, theta)


def objective(case, coefficients=None):
    """Unsimplified negative log likelihood plus a fixed quadratic penalty."""
    return _objective(case, coefficients, stable=False)


def stable_objective(case, coefficients=None):
    """Log-domain equivalent; never substituted for the unsimplified reference."""
    return _objective(case, coefficients, stable=True)


def compressed_poisson(case, coefficients=None):
    """Poisson objective using pattern voxel marginals and experiment totals."""
    if case.model != "poisson":
        raise ValueError("compressed_poisson requires a Poisson case.")
    theta, _, global_eta, counts = _inputs(case, coefficients)
    patterns, inverse = case.patterns
    beta = theta[:case.n_spatial].reshape(case.spatial.shape[1], case.bases.shape[1])
    pattern_eta = _tensor(patterns, theta) @ beta @ _tensor(case.bases, theta).T
    weights = _tensor(case.exposure, theta) * torch.exp(global_eta)
    result = -torch.sum(torch.sum(counts, dim=1) * global_eta)
    for index in range(len(patterns)):
        mask = _tensor(inverse == index, theta).bool()
        result += torch.sum(weights[mask]) * torch.sum(torch.exp(pattern_eta[index]))
        result -= torch.sum(torch.sum(counts[mask], dim=0) * pattern_eta[index])
    return result + _penalty(case, theta)


def autodiff(case, coefficients=None):
    """Return objective, penalized log-likelihood score, and NLL Hessian."""
    values = case.coefficients if coefficients is None else coefficients
    theta = _tensor(values).detach().clone().requires_grad_(True)
    value = objective(case, theta)
    gradient = torch.autograd.grad(value, theta)[0]
    hessian = torch.autograd.functional.hessian(lambda x: objective(case, x), theta)
    return (
        float(value.detach().cpu()),
        -gradient.detach().cpu().numpy(),
        hessian.detach().cpu().numpy(),
    )

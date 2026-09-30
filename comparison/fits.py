"""Layer 3: fit identical, unpenalized objectives with fixed dispersion."""

import numpy as np
from scipy.optimize import minimize
import torch

from .adapters import package_information, package_model, package_objective
from .closed_form import evaluate
from .likelihoods import autodiff, objective
from .metrics import errors


def fit_case(case, implementation):
    """Use the same L-BFGS-B settings and starting vector for every implementation.

    Source likelihoods use their actual PyTorch gradients, while the independent
    simplified objective uses its closed score. Nuisance dispersion is fixed,
    so this is not a comparison of different joint nuisance optimizers.
    """
    if case.penalty is not None:
        raise ValueError("Fit comparisons are unpenalized; penalty derivatives are checked separately.")
    model = package_model(case, implementation) if implementation in ("original", "improved") else None
    if implementation not in ("reference", "closed", "original", "improved"):
        raise ValueError("Unknown fit implementation.")

    def value_gradient(vector):
        if implementation == "closed":
            value, score, _ = evaluate(case, vector)
            return float(value), -score
        theta = torch.tensor(vector, dtype=torch.float64, requires_grad=True)
        value = objective(case, theta) if implementation == "reference" else package_objective(
            case, implementation, theta, model=model
        )
        gradient = torch.autograd.grad(value, theta)[0]
        return float(value.detach()), gradient.detach().numpy()

    result = minimize(
        value_gradient, case.coefficients.copy(), jac=True, method="L-BFGS-B",
        options={"maxiter": 400, "maxls": 40, "ftol": 1e-14, "gtol": 1e-9},
    )
    fitted = case.with_coefficients(result.x)
    value, gradient = value_gradient(result.x)
    if implementation == "reference":
        _, _, information = autodiff(fitted)
    elif implementation == "closed":
        _, _, information = evaluate(fitted)
    else:
        with torch.no_grad():
            model.coefficients.copy_(torch.tensor(result.x, dtype=torch.float64))
        information = package_information(fitted, implementation, model=model)
    eigenvalues = np.linalg.eigvalsh(information)
    positive = bool(np.isfinite(eigenvalues).all() and eigenvalues.min() > 0)
    standard_errors = None
    if positive:
        if model is None:
            standard_errors = np.sqrt(np.diag(np.linalg.solve(
                information, np.eye(case.n_parameters)
            )))
        else:
            standard_errors = np.concatenate([
                value.ravel() for value in model.standard_errors(case.counts).values()
            ])
    score_norm = float(np.linalg.norm(gradient, ord=np.inf))
    return {
        "implementation": implementation, "optimizer_success": bool(result.success),
        "message": str(result.message), "iterations": int(result.nit),
        "objective": value, "score": -gradient, "score_infinity_norm": score_norm,
        "parameters": result.x, "fitted_means": fitted.means(),
        "standard_errors": standard_errors,
        "information_positive_definite": positive,
        "condition_number": float(np.linalg.cond(information)),
        "passed": bool(result.success and positive and np.isfinite(value) and score_norm < 2e-5),
    }


def fit_checks(case):
    implementations = ["reference", "closed"]
    if case.model != "independent_nb":
        implementations += ["original", "improved"]
    results = [fit_case(case, implementation) for implementation in implementations]
    rows = [{
        "layer": "fit", "case": case.name, "comparison": result["implementation"],
        "dispersion_fixed": case.dispersion, **result,
    } for result in results]
    reference = results[0]
    for result in results[1:]:
        for field in ("objective", "parameters", "fitted_means", "score", "standard_errors"):
            if reference[field] is None or result[field] is None:
                report = {"passed": False, "reason": "No identifiable positive-definite fitted covariance."}
            else:
                # Objective-roundoff stopping can leave slightly different small
                # scores; use the same absolute scale as the stationarity check.
                absolute = 2e-5 if field == "score" else 2e-6
                report = errors(result[field], reference[field], rtol=1e-5, atol=absolute)
            rows.append({
                "layer": "fit_agreement", "case": case.name,
                "comparison": f"{result['implementation']} / reference {field}", **report,
            })
    return rows

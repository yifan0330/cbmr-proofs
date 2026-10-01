"""Layer 3: hard cases are diagnostics, not unreliable-reference equality tests."""

from dataclasses import replace
import warnings

import numpy as np
import torch

from .adapters import package_information, package_model, package_objective
from .cases import make_case
from .closed_form import evaluate
from .derivatives import derivative_checks
from .likelihoods import autodiff
from .metrics import errors


def _observe(case, implementation):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if implementation == "reference":
                value, score, hessian = autodiff(case)
            elif implementation == "closed":
                value, score, hessian = evaluate(case)
            else:
                model = package_model(case, implementation)
                value_tensor = package_objective(case, implementation, model=model)
                score = -torch.autograd.grad(value_tensor, model.coefficients)[0].detach().numpy()
                value = float(value_tensor.detach())
                hessian = package_information(case, implementation, model=model)
            result = {
                "objective": value, "score": score, "hessian": hessian,
                "finite": bool(np.isfinite(value) and np.isfinite(score).all()
                               and np.isfinite(hessian).all()),
            }
        except (ValueError, FloatingPointError, OverflowError, RuntimeError, np.linalg.LinAlgError) as error:
            result = {"finite": False, "exception": type(error).__name__, "message": str(error)}
    result["warnings"] = sorted(set(str(item.message) for item in caught))
    return result


def _aggregate_poisson_limit(case):
    """Autodiff of aggregate Poisson, NOT the full Poisson allocation likelihood."""
    loadings, assignment = case.patterns
    y = np.stack([case.counts[assignment == p].sum(axis=0) for p in range(len(loadings))])
    y = torch.tensor(y, dtype=torch.float64)
    loadings = torch.tensor(loadings, dtype=torch.float64)
    basis = torch.tensor(case.bases, dtype=torch.float64)
    global_design = torch.tensor(case.global_design, dtype=torch.float64)
    exposure = torch.tensor(case.exposure, dtype=torch.float64)

    def objective(vector):
        beta = vector[:case.n_spatial].reshape(case.spatial.shape[1], case.bases.shape[1])
        spatial = loadings @ beta @ basis.T
        weights = exposure * torch.exp(global_design @ vector[case.n_spatial:])
        totals = torch.stack([weights[assignment == p].sum() for p in range(len(loadings))])
        return (totals[:, None] * torch.exp(spatial) - y * (
            spatial + torch.log(totals)[:, None]
        )).sum()

    return torch.func.hessian(objective)(
        torch.tensor(case.coefficients, dtype=torch.float64)
    ).detach().numpy()


def stability_checks(seed=731):
    rows = []
    for model in ("poisson", "independent_nb", "aggregated_nb", "clustered_nb"):
        zero = make_case(model, seed=seed)
        zero = replace(zero, name=f"{model}-zero-counts", counts=np.zeros_like(zero.counts))
        rows.extend(derivative_checks(zero))
        for log_mean in (-20, 20):
            case = make_case(model, seed=seed)
            case.coefficients[:case.n_spatial:case.bases.shape[1]] = log_mean
            case.name = f"{model}-log-mean-{log_mean}"
            implementations = ("reference", "closed") if model == "independent_nb" else (
                "reference", "closed", "original", "improved"
            )
            observed = {name: _observe(case, name) for name in implementations}
            for name, observation in observed.items():
                rows.append({
                    "layer": "stability", "case": case.name, "comparison": name,
                    "passed": None, "interpretation": "Floating-point diagnostic, not a trusted-oracle assertion.",
                    **observation,
                    "reference_error": errors(observation["hessian"], observed["reference"]["hessian"])
                    if "hessian" in observation and "hessian" in observed["reference"] else None,
                })

    for dispersion in (1e-4, 1e-8, 1e-12):
        case = make_case("aggregated_nb", seed=seed, dispersion=dispersion)
        case.name = f"aggregated-nb-dispersion-{dispersion}"
        limit = _aggregate_poisson_limit(case)
        for name in ("reference", "closed", "original", "improved"):
            observation = _observe(case, name)
            rows.append({
                "layer": "stability", "case": case.name, "comparison": name,
                "passed": None, **observation,
                "aggregate_poisson_limit_error": errors(observation["hessian"], limit)
                if "hessian" in observation else None,
            })
            if name == "improved" and dispersion == 1e-12:
                report = errors(observation.get("hessian", np.full_like(limit, np.nan)), limit,
                                rtol=1e-8, atol=1e-10)
                rows.append({"layer": "stability", "case": case.name,
                             "comparison": "improved / correct aggregated Poisson limit", **report})

    for log_mean in (-1000, 1000):
        case = make_case("aggregated_nb", seed=seed)
        case.coefficients[:case.n_spatial:case.bases.shape[1]] = log_mean
        case.name = f"aggregated-nb-extreme-{log_mean}"
        for name in ("reference", "original", "improved"):
            observed = _observe(case, name)
            rows.append({
                "layer": "stability", "case": case.name, "comparison": name,
                "passed": observed["finite"] if name == "improved" else None,
                "interpretation": "Only finiteness is asserted for the improved extreme-mean path.",
                **observed,
            })

    redundant = make_case("poisson", seed=seed)
    redundant.global_design = np.column_stack((redundant.global_design, np.ones(len(redundant.counts))))
    redundant.coefficients = np.append(redundant.coefficients, 0)
    design = redundant.cell_design().reshape(-1, redundant.n_parameters)
    hessian = evaluate(redundant)[2]
    singular_values = np.linalg.svd(hessian, compute_uv=False)
    rows.append({
        "layer": "identifiability", "case": "redundant-global-intercept",
        "comparison": "Do not fit or interpret standard errors for redundant coefficients",
        "passed": bool(np.linalg.matrix_rank(design) < redundant.n_parameters
                       and singular_values[-1] < 1e-12 * singular_values[0]),
        "design_rank": int(np.linalg.matrix_rank(design)), "parameters": redundant.n_parameters,
        "hessian_singular_values": singular_values,
    })
    for name in ("original", "improved"):
        observed = _observe(redundant, name)
        rows.append({"layer": "identifiability", "case": "redundant-global-intercept",
                     "comparison": f"{name} actual information", "passed": None, **observed})
    return rows

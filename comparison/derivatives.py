"""Layer 1: independent derivative references and directional differences."""

import numpy as np
import torch

from .adapters import package_information, package_model, package_objective
from .closed_form import evaluate, fisher_information
from .likelihoods import autodiff, compressed_poisson, objective
from .metrics import errors


def derivative_checks(case, rtol=1e-8, atol=1e-10):
    rows = []

    def compare(label, actual, expected):
        rows.append({"layer": "derivatives", "case": case.name, "comparison": label,
                     **errors(actual, expected, rtol, atol)})

    value, score, hessian = autodiff(case)
    closed_value, closed_score, closed_hessian = evaluate(case)
    compare("independent closed objective / unsimplified", closed_value, value)
    compare("independent closed score / autodiff", closed_score, score)
    compare("independent closed Hessian / autodiff", closed_hessian, hessian)
    if case.model == "poisson":
        vector = torch.tensor(case.coefficients, dtype=torch.float64)
        compressed = lambda theta: compressed_poisson(case, theta)
        compare("full / compressed Poisson objective", float(compressed(vector)), value)
        compare("full / compressed Poisson score",
                -torch.func.jacrev(compressed)(vector).detach().numpy(), score)
        compare("full / compressed Poisson Hessian",
                torch.func.hessian(compressed)(vector).detach().numpy(), hessian)
    if case.model == "independent_nb":
        # Observed weights are affine in y, so substitution of E[y] gives E[H].
        from dataclasses import replace
        mean_case = replace(case, counts=case.means())
        expected = evaluate(mean_case)[2]
        compare("expected observed / Fisher information", fisher_information(case), expected)
        x = case.cell_design().reshape(-1, case.n_parameters)
        means = case.means().ravel()
        score_variance = x.T @ (x * (means / (1 + case.dispersion * means))[:, None])
        fisher = fisher_information(case)
        if case.penalty is not None:
            fisher = fisher - case.penalty
        compare("score variance / unpenalized Fisher", score_variance, fisher)
    else:
        for implementation in ("original", "improved"):
            model = package_model(case, implementation)
            vector = model.coefficients.detach().clone().requires_grad_(True)
            package_value = package_objective(case, implementation, vector, model=model)
            package_score = -torch.autograd.grad(package_value, vector)[0].detach().numpy()
            package_hessian = package_information(case, implementation, model=model)
            compare(f"{implementation} actual objective / reference", float(package_value.detach()), value)
            compare(f"{implementation} actual score / reference", package_score, score)
            compare(f"{implementation} actual Hessian / reference", package_hessian, hessian)
            s = case.n_spatial
            for block, selection in (
                ("spatial", np.s_[:s, :s]), ("cross", np.s_[:s, s:]), ("global", np.s_[s:, s:]),
            ):
                compare(f"{implementation} {block} Hessian block", package_hessian[selection],
                        hessian[selection])
    if case.penalty is not None:
        from dataclasses import replace
        unpenalized = replace(case, penalty=None)
        raw_value, raw_score, raw_hessian = autodiff(unpenalized)
        compare("penalty objective increment", value - raw_value,
                0.5 * case.coefficients @ case.penalty @ case.coefficients)
        compare("penalty score increment", score - raw_score, -case.penalty @ case.coefficients)
        compare("penalty Hessian increment", hessian - raw_hessian, case.penalty)
    rows.append(directional_check(case, score, hessian))
    return rows


def directional_check(case, score, hessian):
    direction = np.random.default_rng(482).normal(size=case.n_parameters)
    direction /= np.linalg.norm(direction)
    gradient = float(-score @ direction)
    curvature = float(direction @ hessian @ direction)
    samples = []
    vector = torch.tensor(case.coefficients, dtype=torch.float64)
    delta = torch.tensor(direction, dtype=torch.float64)
    center = float(objective(case, vector).detach())
    for step in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        plus = float(objective(case, vector + step * delta).detach())
        minus = float(objective(case, vector - step * delta).detach())
        first, second = (plus - minus) / (2 * step), (plus - 2 * center + minus) / step**2
        samples.append({
            "step": step, "directional_gradient": first, "directional_curvature": second,
            "gradient_error": abs(first - gradient) / max(1, abs(gradient)),
            "curvature_error": abs(second - curvature) / max(1, abs(curvature)),
        })
    best_gradient = min(sample["gradient_error"] for sample in samples)
    best_curvature = min(sample["curvature_error"] for sample in samples)
    return {
        "layer": "finite_difference", "case": case.name,
        "comparison": "directional central differences over multiple steps",
        "passed": bool(best_gradient < 1e-6 and best_curvature < 1e-5),
        "expected_gradient": gradient, "expected_curvature": curvature,
        "best_gradient_error": best_gradient, "best_curvature_error": best_curvature,
        "samples": samples,
    }

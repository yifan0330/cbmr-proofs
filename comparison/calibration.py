"""Separate small Monte Carlo study of coefficient inference at fixed dispersion."""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import platform
import time

from .benchmark import SIZES, THREAD_VARIABLES, _bounded_integer, finite_json


MODELS = ("poisson", "independent_nb", "aggregated_nb", "clustered_nb")
CALIBRATION_SIZES = {
    "smoke": dict(experiments=24, voxels=9, bases=2, groups=2),
    **SIZES,
}


def simulate_counts(case, rng):
    """Draw from the selected likelihood's own distribution, not a common NB surrogate."""
    import numpy as np
    means = case.means()
    if case.model == "poisson":
        return rng.poisson(means).astype(float)
    if case.model == "independent_nb":
        return rng.poisson(means * rng.gamma(1 / case.dispersion, case.dispersion,
                                           size=means.shape)).astype(float)
    if case.model == "clustered_nb":
        multipliers = rng.gamma(1 / case.dispersion, case.dispersion, size=(len(means), 1))
        return rng.poisson(means * multipliers).astype(float)
    if case.model != "aggregated_nb":
        raise ValueError(f"Unsupported model: {case.model}")
    weights = case.exposure * np.exp(case.global_design @ case.coefficients[case.n_spatial:])
    counts = np.zeros_like(means)
    _, assignments = case.patterns
    for pattern in np.unique(assignments):
        members = np.flatnonzero(assignments == pattern)
        member_weights = weights[members]
        rho = member_weights.sum()**2 / (case.dispersion * (member_weights**2).sum())
        totals = rng.poisson(means[members].sum(axis=0) * rng.gamma(
            shape=rho, scale=1 / rho, size=means.shape[1],
        ))
        # Deterministic representation only; the likelihood uses pattern-voxel totals.
        counts[members[0]] = totals
    return counts


def fit_replicate(case, maxiter=200, score_tolerance=1e-4):
    """Common optimizer, actual improved source objective/H where available; no ridge."""
    import numpy as np
    from scipy.optimize import minimize
    import torch
    from .adapters import package_information, package_model, package_objective
    from .closed_form import evaluate

    started = time.perf_counter()
    record = {"status": "error", "errors": [], "fit_success": False,
              "inference_success": False, "optimizer": None}
    try:
        if case.model == "independent_nb":
            def objective(vector):
                value, score, _ = evaluate(case, vector)
                return value, -score
            model = None
        else:
            model = package_model(case, "improved")
            def objective(vector):
                coefficients = torch.tensor(vector, dtype=torch.float64, requires_grad=True)
                value = package_objective(case, "improved", coefficients=coefficients, model=model)
                gradient, = torch.autograd.grad(value, coefficients)
                return float(value.detach()), gradient.detach().numpy()

        result = minimize(
            objective, np.zeros(case.n_parameters), jac=True, method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxls": 40},
        )
        record["optimizer"] = {
            "success": bool(result.success), "status": int(result.status),
            "message": str(result.message), "iterations": int(result.nit),
            "function_evaluations": int(result.nfev), "objective": float(result.fun),
            "score": (-np.asarray(result.jac)).tolist(),
            "score_infinity_norm": float(np.linalg.norm(result.jac, ord=np.inf)),
        }
        record["coefficients"] = result.x.tolist()
        record["fit_success"] = bool(
            result.success and np.all(np.isfinite(result.x)) and np.isfinite(result.fun)
            and np.all(np.isfinite(result.jac))
            and np.linalg.norm(result.jac, ord=np.inf) <= score_tolerance
        )
        if not record["fit_success"]:
            record["errors"].append("Optimizer failed or final score exceeded tolerance.")
            return record
        if model is None:
            information = evaluate(case, result.x)[2]
        else:
            with torch.no_grad():
                model.coefficients.copy_(torch.as_tensor(result.x, dtype=torch.float64))
            information = package_information(case, "improved", model=model)
        information = np.asarray(information)
        if not np.all(np.isfinite(information)):
            raise ValueError("Non-finite observed information.")
        asymmetry = float(np.max(np.abs(information - information.T)))
        record["information_max_asymmetry"] = asymmetry
        if not np.allclose(information, information.T, rtol=1e-10, atol=1e-10):
            raise ValueError("Observed information is not symmetric.")
        information = (information + information.T) / 2
        cholesky = np.linalg.cholesky(information)
        inverse_factor = np.linalg.solve(cholesky, np.eye(case.n_parameters))
        covariance = inverse_factor.T @ inverse_factor
        se = np.sqrt(np.diag(covariance))
        if not np.all(np.isfinite(se)) or np.any(se <= 0):
            raise ValueError("Non-finite or nonpositive standard errors.")
        lower, upper = result.x - 1.959963984540054 * se, result.x + 1.959963984540054 * se
        record.update(
            status="ok", inference_success=True, standard_errors=se.tolist(),
            interval_lower=lower.tolist(), interval_upper=upper.tolist(),
            covered=((lower <= case.coefficients) & (case.coefficients <= upper)).tolist(),
            null_moderator_rejected=bool(lower[-1] > 0 or upper[-1] < 0),
        )
    except (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError, RuntimeError) as error:
        record["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        record["seconds"] = time.perf_counter() - started
    return record


def summarize(case, records):
    import numpy as np
    fitted = [record for record in records if record["fit_success"]]
    inferred = [record for record in records if record["inference_success"]]
    output = {
        "attempts": len(records), "successful_fits": len(fitted),
        "successful_inferences": len(inferred),
        "fit_failures": len(records) - len(fitted),
        "inference_failures_after_successful_fit": len(fitted) - len(inferred),
        "total_failed_replicates": len(records) - len(inferred),
        "bias_denominator": len(fitted), "coverage_denominator": len(inferred),
        "false_positive_denominator": len(inferred),
        "conditioning": "Bias conditional on accepted fit; coverage/FPR conditional on accepted "
                        "fit and SPD observed information. Failed attempts retained in replicates.",
        "coefficient_bias": None, "coefficient_bias_mcse": None,
        "coverage": None, "coverage_mcse": None,
        "coverage_monte_carlo_interval95": None,
        "null_moderator_false_positive_rate": None, "null_moderator_false_positive_mcse": None,
        "null_moderator_false_positive_interval95": None,
    }
    if fitted:
        estimates = np.array([record["coefficients"] for record in fitted])
        output["coefficient_bias"] = (estimates.mean(axis=0) - case.coefficients).tolist()
        if len(fitted) > 1:
            output["coefficient_bias_mcse"] = (
                estimates.std(axis=0, ddof=1) / np.sqrt(len(fitted))
            ).tolist()
    if inferred:
        coverage = np.mean([record["covered"] for record in inferred], axis=0)
        fpr = float(np.mean([record["null_moderator_rejected"] for record in inferred]))
        output.update(
            coverage=coverage.tolist(),
            coverage_mcse=np.sqrt(coverage * (1 - coverage) / len(inferred)).tolist(),
            coverage_monte_carlo_interval95=[
                _wilson_interval(float(rate), len(inferred)) for rate in coverage
            ],
            null_moderator_false_positive_rate=fpr,
            null_moderator_false_positive_mcse=math.sqrt(fpr * (1 - fpr) / len(inferred)),
            null_moderator_false_positive_interval95=_wilson_interval(fpr, len(inferred)),
        )
    return output


def _wilson_interval(rate, count):
    """Finite Monte Carlo uncertainty even when the observed rate is zero or one."""
    z = 1.959963984540054
    denominator = 1 + z**2 / count
    center = (rate + z**2 / (2 * count)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / count + z**2 / (4 * count**2)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--replicates", type=_bounded_integer(1, 1000), default=20)
    result.add_argument("--seed", type=_bounded_integer(0, 2**32 - 1), default=731)
    result.add_argument("--models", nargs="+", choices=MODELS,
                        default=["poisson", "independent_nb", "clustered_nb"])
    result.add_argument("--output", type=Path, default=Path("comparison/results/calibration.json"))
    result.add_argument("--threads", type=_bounded_integer(1, 8), default=1)
    result.add_argument("--maxiter", type=_bounded_integer(1, 1000), default=200)
    result.add_argument("--design", choices=("gc", "sv"), default="gc")
    result.add_argument("--size", choices=CALIBRATION_SIZES, default="smoke",
                        help="Bounded design dimensions; smoke preserves the original setup.")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    # No numerical imports occur before these settings in this module's CLI path.
    os.environ.update({name: str(arguments.threads) for name in THREAD_VARIABLES})
    import numpy as np
    import scipy
    import torch
    torch.set_num_threads(arguments.threads)
    torch.set_num_interop_threads(1)
    from .cases import make_case
    from .adapters import source_manifest

    models = []
    dimensions = CALIBRATION_SIZES[arguments.size]
    for model_name in dict.fromkeys(arguments.models):
        case = make_case(model=model_name, design=arguments.design, seed=arguments.seed,
                         **dimensions)
        truth = case.coefficients.copy()
        truth[-1] = 0.0
        case = case.with_coefficients(truth)
        # Stable model-specific substreams, independent of CLI model ordering.
        stream = np.random.SeedSequence([arguments.seed, MODELS.index(model_name)])
        records = []
        for replicate, child_seed in enumerate(stream.spawn(arguments.replicates)):
            try:
                counts = simulate_counts(case, np.random.default_rng(child_seed))
                record = fit_replicate(replace(case, counts=counts), maxiter=arguments.maxiter)
                record["total_count"] = int(counts.sum())
            except (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError, RuntimeError) as error:
                record = {"status": "error", "fit_success": False, "inference_success": False,
                          "optimizer": None, "errors": [f"{type(error).__name__}: {error}"]}
            record.update(replicate=replicate, seed_entropy=child_seed.entropy,
                          seed_spawn_key=list(child_seed.spawn_key))
            records.append(record)
        aggregate = summarize(case, records)
        models.append({
            "model": model_name, "design": arguments.design, "seed": arguments.seed,
            "size": arguments.size,
            "dimensions": dict(dimensions, parameters=case.n_parameters),
            "true_coefficients": truth.tolist(), "null_moderator_index": case.n_parameters - 1,
            "fixed_dispersion": case.dispersion,
            "implementation": "independent_closed" if model_name == "independent_nb" else "improved",
            "status": "ok" if not aggregate["total_failed_replicates"] else "completed_with_failures",
            "aggregate": aggregate, "replicates": records,
        })
    report = {
        "schema_version": 1, "kind": "statistical_calibration",
        "status": "ok" if all(model["status"] == "ok" for model in models) else "completed_with_failures",
        "question": "Do fixed-dispersion model-based Wald intervals cover and does the null "
                    "moderator reject at its nominal rate? Matching Hessians does not imply coverage.",
        "warning": "Small Monte Carlo runs are smoke tests, not calibration evidence. Plug-in "
                   "binomial MCSE can be zero at observed rates 0 or 1; Wilson Monte Carlo "
                   "intervals quantify uncertainty even at those endpoints.",
        "fit": {"optimizer": "scipy L-BFGS-B", "initial_coefficients": "all zero",
                "maxiter": arguments.maxiter, "score_tolerance": 1e-4,
                "dispersion": "Held fixed at true value, never optimized.",
                "source_models": "Actual improved package objective with autograd gradient and "
                                 "actual improved observed information at fitted coefficients.",
                "independent_nb": "Independent closed-form objective, score and observed information.",
                "covariance": "Inverse observed information only if SPD; no pseudoinverse or ridge.",
                "interval": "Two-sided 95% coefficient-wise Wald; no multiplicity adjustment."},
        "data_generating_distributions": {
            "poisson": "Independent cells, Poisson(case.means()).",
            "independent_nb": "Independent Gamma(mean=1, variance=dispersion) multiplier per cell.",
            "clustered_nb": "One Gamma(mean=1, variance=dispersion) multiplier per experiment, "
                            "shared across all voxels; experiments independent.",
            "aggregated_nb": "Independent pattern-voxel NB totals: rho=(sum w)^2/"
                             "(dispersion*sum w^2), w=exposure*exp(global_design*true_gamma); "
                             "Gamma(rho, scale=1/rho) multiplied by the total mean, then Poisson. "
                             "Totals placed deterministically in the first member experiment only "
                             "as a representation. NOT an independent-cell DGP.",
        },
        "metadata": {"python": platform.python_version(), "platform": platform.platform(),
                     "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
                     "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
                     "torch_threads": torch.get_num_threads(), "torch_interop_threads": 1,
                     "source_hashes": source_manifest()},
        "models": models,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False) + "\n")
    print("| Model | Attempts | Accepted inference | Failures | Null FPR (MCSE) |")
    print("|---|---:|---:|---:|---:|")
    for model in models:
        a = model["aggregate"]
        rate, mcse = a["null_moderator_false_positive_rate"], a["null_moderator_false_positive_mcse"]
        display = "unavailable" if rate is None else f"{rate:.3f} ({mcse:.3f})"
        print(f"| {model['model']} | {a['attempts']} | {a['successful_inferences']} | "
              f"{a['total_failed_replicates']} | {display} |")
    print(f"Small runs are not calibration evidence. Failed replicates retained. JSON: {arguments.output}")
    return int(any(model["aggregate"]["successful_inferences"] == 0 for model in models))


if __name__ == "__main__":
    raise SystemExit(main())

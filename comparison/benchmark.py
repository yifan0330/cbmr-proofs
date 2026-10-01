"""Small, sequential, fresh-process benchmarks; no numerical imports in the parent."""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
SIZES = {
    "tiny": dict(experiments=16, voxels=7, bases=2, groups=2),
    "small": dict(experiments=32, voxels=32, bases=4, groups=3),
    "moderate": dict(experiments=64, voxels=96, bases=8, groups=4),
    "large": dict(experiments=128, voxels=256, bases=8, groups=4),
}
MODELS = ("poisson", "aggregated_nb", "clustered_nb")


def finite_json(value):
    """Replace non-finite scalars, never emit nonstandard JSON NaN/Infinity."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def _bounded_integer(low, high):
    def parse(value):
        result = int(value)
        if not low <= result <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return result
    return parse


def _rss_mib():
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def _worker(config):
    # The launching parent sets these before Python (and any numerical imports).
    started = time.perf_counter()
    import numpy as np
    import scipy
    import torch
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    from .cases import make_case
    from .adapters import package_information, package_model, modules, source_manifest

    method = config["implementation"]
    model = None
    if method in ("original", "improved"):
        modules(method)
    elif method in ("unsimplified_autodiff", "reference"):
        from .likelihoods import objective
    elif method == "independent_closed":
        from .closed_form import evaluate
    else:
        raise ValueError(f"Unknown method: {method}")
    import_seconds = time.perf_counter() - started
    import_peak_rss = _rss_mib()
    case = make_case(
        model=config["model"], design=config["design"], seed=config["seed"],
        **SIZES[config["size"]],
    )
    if method in ("original", "improved"):
        model = package_model(case, method)
    coefficients = torch.tensor(case.coefficients, dtype=torch.float64)
    contrast = np.zeros((2, case.n_parameters))
    contrast[0, 0], contrast[0, case.bases.shape[1]] = 1.0, -1.0
    contrast[1, -1] = 1.0
    setup_peak_rss = _rss_mib()
    started = time.perf_counter()
    if config["workload"] == "contrast_covariance" and method == "improved":
        result = model.information_operator(case.counts).contrast_covariance(contrast)
    else:
        if method in ("original", "improved"):
            information = package_information(case, method, model=model)
        elif method == "independent_closed":
            information = evaluate(case)[2]
        else:
            information = torch.autograd.functional.hessian(
                lambda vector: objective(case, vector), coefficients
            ).detach().numpy()
        if config["workload"] == "contrast_covariance":
            if method == "original":
                covariance = modules("original").covariance.fisher_covariance(model, information)
                result = contrast @ covariance @ contrast.T
            else:
                result = contrast @ np.linalg.solve(information, contrast.T)
        else:
            result = information
    seconds = time.perf_counter() - started
    # Capture the workload high-water mark BEFORE serialization/metadata; no reference here.
    peak_rss = _rss_mib()
    result = np.asarray(result, dtype=float)
    valid = bool(np.all(np.isfinite(result))) and math.isfinite(seconds)
    return {
        "status": "ok" if valid else "error",
        "errors": [] if valid else ["Workload returned non-finite values."],
        "seconds": seconds, "peak_rss_mib": peak_rss,
        "import_seconds": import_seconds, "import_peak_rss_mib": import_peak_rss,
        "setup_peak_rss_mib": setup_peak_rss,
        "result": result.tolist(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "torch": torch.__version__, "python": platform.python_version()},
        "platform": platform.platform(),
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "source_hashes": source_manifest(),
    }


def _launch(config, timeout):
    environment = os.environ.copy()
    environment.update({name: str(config["threads"]) for name in THREAD_VARIABLES})
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "comparison.benchmark", "--worker-json", json.dumps(config)],
            cwd=Path(__file__).resolve().parents[1], env=environment,
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return {"status": "error", "errors": [f"Worker timeout after {timeout}s", stderr]}
    if completed.returncode:
        return {"status": "error", "errors": [
            f"Worker exited {completed.returncode}", completed.stderr, completed.stdout,
        ]}
    try:
        output = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"status": "error", "errors": [
            "Worker did not return JSON", completed.stdout, completed.stderr,
        ]}
    if completed.stderr:
        output["stderr"] = completed.stderr
    return output


def _errors(actual, expected):
    pairs = [(a, b) for row_a, row_b in zip(actual, expected)
             for a, b in zip(row_a, row_b)]
    if len(actual) != len(expected) or any(
        len(a) != len(b) for a, b in zip(actual, expected)
    ):
        raise ValueError("Reference and result shapes differ.")
    if not pairs or any(a is None or b is None for a, b in pairs):
        raise ValueError("Non-finite or empty numerical output.")
    difference = math.sqrt(sum((a - b) ** 2 for a, b in pairs))
    norm = math.sqrt(sum(b ** 2 for _, b in pairs))
    return {
        "max_absolute": max(abs(a - b) for a, b in pairs),
        "relative_frobenius": difference / max(1.0, norm),
        "within_tolerance": all(abs(a - b) <= 1e-10 + 1e-8 * abs(b) for a, b in pairs),
    }


def _summary(values):
    return {"median": statistics.median(values), "min": min(values), "all": values} if values else {
        "median": None, "min": None, "all": [],
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=Path("comparison/results/benchmark.json"))
    result.add_argument("--sizes", nargs="+", choices=SIZES, default=["tiny"])
    result.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    result.add_argument("--designs", nargs="+", choices=("gc", "sv"), default=["gc"])
    result.add_argument("--workloads", nargs="+", choices=("hessian", "contrast_covariance"),
                        default=["hessian"])
    result.add_argument("--include-closed-form", action="store_true")
    result.add_argument("--repeats", type=_bounded_integer(1, 20), default=3)
    result.add_argument("--threads", type=_bounded_integer(1, 8), default=1)
    result.add_argument("--timeout", type=_bounded_integer(1, 600), default=120)
    result.add_argument("--max-estimated-mib", type=_bounded_integer(1, 256), default=64)
    result.add_argument("--seed", type=_bounded_integer(0, 2**32 - 1), default=731)
    result.add_argument("--worker-json", help=argparse.SUPPRESS)
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.worker_json is not None:
        result = _worker(json.loads(arguments.worker_json))
        print(json.dumps(finite_json(result), allow_nan=False))
        return 0
    records = []
    metadata = None
    for size in dict.fromkeys(arguments.sizes):
        dimensions = SIZES[size]
        for design in dict.fromkeys(arguments.designs):
            n, v, b, g = (dimensions[key] for key in ("experiments", "voxels", "bases", "groups"))
            parameters = (g + (design == "sv")) * b + 1
            estimate = 8 * (6 * n * v * parameters + 12 * parameters**2 + 12 * n * v)
            if parameters > 64 or estimate > arguments.max_estimated_mib * 1024**2:
                parser().error(f"{size}/{design} exceeds the parameter/tangent memory guard")
            for model in dict.fromkeys(arguments.models):
                for workload in dict.fromkeys(arguments.workloads):
                    config = dict(size=size, model=model, design=design, workload=workload,
                                  seed=arguments.seed, threads=arguments.threads)
                    reference = _launch(dict(config, implementation="reference"), arguments.timeout)
                    methods = ["original", "improved", "unsimplified_autodiff"]
                    if arguments.include_closed_form:
                        methods.append("independent_closed")
                    for implementation in methods:
                        runs = []
                        for repeat in range(arguments.repeats):
                            run = _launch(dict(config, implementation=implementation), arguments.timeout)
                            if metadata is None and "versions" in run:
                                metadata = {key: run[key] for key in (
                                    "versions", "platform", "threads", "torch_threads",
                                    "torch_interop_threads", "source_hashes",
                                )}
                            if run["status"] == "ok":
                                try:
                                    if reference["status"] != "ok":
                                        raise ValueError(f"Reference failed: {reference.get('errors')}")
                                    run["numerical_error"] = _errors(run["result"], reference["result"])
                                except (ValueError, TypeError, OverflowError) as error:
                                    run["status"] = "error"
                                    run["errors"].append(str(error))
                            run.pop("result", None)
                            for key in ("versions", "platform", "threads", "source_hashes",
                                        "torch_threads", "torch_interop_threads"):
                                run.pop(key, None)
                            run["repeat"] = repeat
                            runs.append(run)
                        valid = [run for run in runs if run["status"] == "ok"]
                        numerically_close = (
                            len(valid) == arguments.repeats and
                            all(run["numerical_error"]["within_tolerance"] for run in valid)
                        )
                        record = dict(
                            config, implementation=implementation,
                            dimensions=dict(dimensions, parameters=parameters),
                            estimated_problem_bytes=estimate,
                            status="ok" if numerically_close else "error",
                            repeats=arguments.repeats, successful_repeats=len(valid), runs=runs,
                            seconds=_summary([run["seconds"] for run in valid]),
                            peak_rss_mib=_summary([run["peak_rss_mib"] for run in valid]),
                            import_seconds=_summary([run["import_seconds"] for run in valid]),
                            import_peak_rss_mib=_summary([run["import_peak_rss_mib"] for run in valid]),
                            numerical_error={
                                key: max((run["numerical_error"][key] for run in valid
                                          if run["numerical_error"][key] is not None), default=None)
                                for key in ("max_absolute", "relative_frobenius")
                            },
                            reference_status=reference["status"],
                            reference_errors=reference.get("errors", []),
                            errors=[error for run in runs for error in run.get("errors", [])],
                        )
                        if valid and not numerically_close:
                            record["errors"].append("Numerical comparison failed at rtol=1e-8, atol=1e-10.")
                        records.append(record)
    report = {
        "schema_version": 1, "kind": "performance_benchmark",
        "measurement": {
            "isolation": "One operation per sequential fresh subprocess; reference in separate child.",
            "timing": "Cold kernel after imports and case/model setup; no warmup.",
            "memory": "Absolute ru_maxrss high-water RSS includes interpreter, imports and setup. "
                      "Import/setup marks are separate high-water marks, NOT incremental operation memory.",
            "rss_conversion": "Linux ru_maxrss KiB / 1024; macOS bytes / 1048576.",
            "guard": "Whitelist, P<=64, estimated design/tangent bytes, per-child timeout; "
                     "estimate is not a bound on interpreter/runtime RSS; no RLIMIT_AS.",
            "contrast_covariance": "Same 2-row C: group intercept difference and global moderator. "
                                   "Original: assemble H, full fisher_covariance, C V C.T; "
                                   "improved: assemble information_operator, projected solves.",
            "reference": "Unsimplified float64 autodiff observed Hessian; solve for covariance.",
            "autodiff": "torch.autograd.functional.hessian; reverse-mode row-wise Hessian, no separate score calculation.",
            "independent_closed": "Optional helper evaluate() computes value, score and Hessian together.",
            "inference": "Performance or matching Hessians does not establish statistical calibration.",
        },
        "metadata": metadata, "records": records,
        "status": "ok" if all(record["status"] == "ok" for record in records) else "error",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False) + "\n")
    print("| Model/design | N/V/P | Method/workload | Rel. error | Median s | Peak MiB* |")
    print("|---|---:|---|---:|---:|---:|")
    for record in records:
        def display(value):
            return f"{value:.4g}" if value is not None else "ERROR"
        d = record["dimensions"]
        print(f"| {record['model']}/{record['design']} | "
              f"{d['experiments']}/{d['voxels']}/{d['parameters']} | "
              f"{record['implementation']}/{record['workload']} | "
              f"{display(record['numerical_error']['relative_frobenius'])} | "
              f"{display(record['seconds']['median'])} | "
              f"{display(max(record['peak_rss_mib']['all'], default=None))} |")
        if record["errors"]:
            print(f"ERROR {record['implementation']}: {'; '.join(record['errors'])}", file=sys.stderr)
    print(f"*Absolute process high-water RSS, not incremental memory. JSON: {arguments.output}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

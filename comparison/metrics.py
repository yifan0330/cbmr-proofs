"""Error magnitudes and strict, portable JSON reports."""

import json
from pathlib import Path
import platform
import sys

import numpy as np


def errors(actual, expected, rtol=1e-8, atol=1e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return {"passed": False, "reason": f"shape mismatch: {actual.shape} != {expected.shape}"}
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        return {"passed": False, "reason": "nonfinite actual or reference",
                "actual_finite": bool(np.isfinite(actual).all()),
                "reference_finite": bool(np.isfinite(expected).all())}
    difference = actual - expected
    return {
        "passed": bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
        "max_absolute": float(np.max(np.abs(difference), initial=0)),
        "relative_frobenius": float(
            np.linalg.norm(difference.ravel()) / max(1.0, np.linalg.norm(expected.ravel()))
        ),
        "reference_norm": float(np.linalg.norm(expected.ravel())),
        "rtol": rtol, "atol": atol,
    }


def inverse_residual(information, covariance):
    identity = np.eye(len(information))
    residual = np.linalg.norm(information @ covariance - identity)
    return {
        "frobenius": float(residual),
        "relative_frobenius": float(residual / max(1, np.linalg.norm(identity))),
        "backward_error": float(residual / (
            np.linalg.norm(information) * np.linalg.norm(covariance) + np.linalg.norm(identity)
        )),
    }


def serializable(value):
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable(report), indent=2, allow_nan=False) + "\n")


def environment():
    import scipy
    import torch

    return {"python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__}

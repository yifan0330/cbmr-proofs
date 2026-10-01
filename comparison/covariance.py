"""Layer 2: solves, inverse residuals, spectra and contrast projection."""

from types import SimpleNamespace

import numpy as np

from cbmr_improved.structured import StructuredInformation

from .adapters import modules, package_information
from .cases import make_case
from .metrics import errors, inverse_residual


def covariance_checks(seed=932, rtol=1e-8, atol=1e-10):
    rng = np.random.default_rng(seed)
    original = modules("original").covariance
    rows = []
    indices = [np.arange(i * 2, i * 2 + 2) for i in range(3)]
    blocks = []
    for _ in indices:
        root = rng.normal(size=(2, 2))
        blocks.append(root @ root.T + 2 * np.eye(2))
    d = np.zeros((6, 6))
    for index, block in zip(indices, blocks):
        d[np.ix_(index, index)] = block
    cross = rng.normal(size=(6, 2)) * 0.2
    e = cross.T @ np.linalg.solve(d, cross) + np.eye(2)

    def compare(case, label, actual, expected):
        rows.append({"layer": "covariance", "case": case, "comparison": label,
                     **errors(actual, expected, rtol, atol)})

    matrices = {
        "positive_definite_block": (blocks, np.empty((6, 0)), np.empty((0, 0))),
        "positive_definite_border": (blocks, cross, e),
        "indefinite_border": ([np.diag([-2.0, 2.0]), *blocks[1:]], cross, e),
        "nearly_singular_border": (
            blocks, cross, cross.T @ np.linalg.solve(d, cross) + np.diag([1.0, 1e-12])
        ),
        "deliberately_singular_block": (
            [*blocks[:2], np.diag([1.0, 0.0])], np.empty((6, 0)), np.empty((0, 0))
        ),
    }
    for name, (pieces, border, global_block) in matrices.items():
        operator = StructuredInformation(pieces, indices, border, global_block)
        information = operator.to_dense()
        norm_condition = float(np.linalg.cond(information, 2))
        eigen_condition = original.symmetric_condition_number(information)
        near = name.startswith("nearly")
        singular = name.startswith("deliberately")
        rows.append({
            "layer": "conditioning", "case": name, "comparison": "SVD / absolute eigenvalues",
            "passed": None if near else (
                bool(np.isinf(norm_condition) and np.isinf(eigen_condition)) if singular
                else errors(eigen_condition, norm_condition, rtol, atol)["passed"]
            ),
            "svd_condition": norm_condition, "eigen_condition": eigen_condition,
            "condition_times_epsilon": norm_condition * np.finfo(float).eps,
            "error": errors(eigen_condition, norm_condition, rtol, atol),
            "interpretation": "Near singularity, condition estimates and inverse entries are diagnostics."
            if near else "Symmetric singular values are absolute eigenvalues.",
        })
        if singular:
            for label, calculate in (
                ("dense inverse", lambda: np.linalg.inv(information)),
                ("structured solve", lambda: operator.solve(np.ones(len(information)))),
                ("original block inverse", lambda: original.blockwise_inverse(information, indices)),
            ):
                try:
                    calculate()
                except np.linalg.LinAlgError as error:
                    outcome, detail = True, str(error)
                else:
                    outcome, detail = False, "Singular matrix unexpectedly accepted."
                rows.append({"layer": "covariance", "case": name, "comparison": label,
                             "passed": outcome, "exception": detail})
            continue

        dense_inverse = np.linalg.inv(information)
        structured_inverse = operator.covariance()
        model = SimpleNamespace(
            n_spatial=6, n_global=border.shape[1], n_parameters=len(information),
            predictor=SimpleNamespace(
                patterns=SimpleNamespace(loadings=np.eye(3)), n_bases=2
            ),
        )
        original_inverse = original.fisher_covariance(model, information)
        for label, inverse in (
            ("dense", dense_inverse), ("original", original_inverse), ("structured", structured_inverse),
        ):
            residual = inverse_residual(information, inverse)
            rows.append({
                "layer": "inverse_residual", "case": name, "comparison": label,
                "passed": bool(residual["backward_error"] < 1e-12),
                **residual,
                "entry_error": errors(inverse, dense_inverse, rtol, atol),
                "entry_equality_required": not near,
            })
            if not near:
                compare(name, f"{label} / dense inverse", inverse, dense_inverse)
        rhs = rng.normal(size=(len(information), 3))
        if not near:
            compare(name, "structured / dense multiple-RHS solve", operator.solve(rhs),
                    np.linalg.solve(information, rhs))
            compare(name, "diagonal without full inverse", operator.diagonal(),
                    np.diag(dense_inverse))
            contrast = rng.normal(size=(2, len(information)))
            scores = rng.normal(size=(17, len(information)))
            full = dense_inverse @ scores.T @ scores @ dense_inverse.T
            compare(name, "full / projected Fisher covariance",
                    operator.contrast_covariance(contrast), contrast @ dense_inverse @ contrast.T)
            compare(name, "full / projected score sandwich", operator.sandwich_covariance(
                scores, contrast=contrast), contrast @ full @ contrast.T)
        if not border.shape[1]:
            compare(name, "union of block eigenvalues", operator.condition_number(), norm_condition)

    # This actual Poisson design has a common spatial shape and no global border.
    case = make_case("poisson", seed=seed)
    case.global_design = np.empty((len(case.counts), 0))
    case.coefficients = np.tile(np.array([0.2, -0.1]), case.spatial.shape[1])
    basis_information = case.bases.T @ (
        case.bases * np.exp(case.bases @ np.array([0.2, -0.1]))[:, None]
    )
    study_information = case.spatial.T @ (case.spatial * case.exposure[:, None])
    kronecker = np.kron(study_information, basis_information)
    for implementation in ("original", "improved"):
        compare("separable_poisson", f"{implementation} actual information / Kronecker",
                package_information(case, implementation), kronecker)
    compare("separable_poisson", "dense / Kronecker inverse", np.linalg.inv(kronecker),
            np.kron(np.linalg.inv(study_information), np.linalg.inv(basis_information)))
    mean_grid = np.outer([1.0, 3.0], [2.0, 5.0])
    nb_weights = mean_grid / (1 + 0.4 * mean_grid)
    rows.append({
        "layer": "covariance", "case": "nonseparable_nb_weights",
        "comparison": "Do not infer Kronecker weights from separable NB means",
        "passed": bool(np.linalg.matrix_rank(nb_weights) == 2),
        "weight_determinant": float(np.linalg.det(nb_weights)),
        "weight_rank": int(np.linalg.matrix_rank(nb_weights)),
    })
    return rows

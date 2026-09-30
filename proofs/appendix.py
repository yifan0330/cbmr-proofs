"""Exact checks for the integrated appendix in the lesion paper's notation.

Scalar likelihood identities and a generic affine-chain-rule check are separate:
the paper's independent NB likelihood is not NiMARE's aggregated NB likelihood.
"""
from pathlib import Path

import sympy as sp

from proofs.covariance import blkdiag
from proofs.latex import BANNER, Proof


class AppendixProof(Proof):
    def __init__(self):
        super().__init__(
            "appendix", "Integrated appendix",
            preamble="Additional checks in the notation of the lesion-model paper. "
                     "All calculations use exact symbolic expressions.")
        self.formulas = []

    def formula(self, name, label, derived, expected):
        self.claim(label, sp.simplify(derived - expected))
        self.formulas.append((name, expected))

    def write(self, directory="docs/generated"):
        path = super().write(directory)
        equations = Path(directory) / "appendix_equations.tex"
        equations.write_text(BANNER + "\n".join(
            rf"\newcommand{{\{name}}}{{{sp.latex(expression)}}}"
            for name, expression in self.formulas
        ) + "\n")
        print(f"   -> wrote {equations}")
        return path


def scalar_likelihoods(proof):
    eta = sp.Symbol("eta", real=True)
    mu, alpha = sp.symbols("mu alpha", positive=True)
    Y = sp.Symbol("Y", nonnegative=True)
    poisson = sp.exp(eta) - Y * eta + sp.loggamma(Y + 1)
    nb = (-sp.loggamma(Y + 1 / alpha) + sp.loggamma(1 / alpha)
          + sp.loggamma(Y + 1) - Y * sp.log(alpha) - Y * eta
          + (Y + 1 / alpha) * sp.log(1 + alpha * sp.exp(eta)))

    def in_mu(expression):
        return sp.simplify(expression).subs(sp.exp(eta), mu)

    poisson_score = Y - mu
    nb_score = (Y - mu) / (1 + alpha * mu)
    nb_observed = mu * (1 + alpha * Y) / (1 + alpha * mu)**2
    nb_fisher = mu / (1 + alpha * mu)
    proof.formula("AppendixPoissonScore", "Poisson log-mean score",
                  in_mu(-sp.diff(poisson, eta)), poisson_score)
    proof.formula("AppendixPoissonWeight", "Poisson observed equals expected weight",
                  in_mu(sp.diff(poisson, eta, eta)), mu)
    proof.formula("AppendixNBScore", "Independent NB log-mean score",
                  in_mu(-sp.diff(nb, eta)), nb_score)
    proof.formula("AppendixNBObserved", "Independent NB observed weight",
                  in_mu(sp.diff(nb, eta, eta)), nb_observed)
    # The observed weight is affine in Y, so E[Y] = mu suffices.
    proof.claim("NB observed weight is affine in Y", sp.diff(nb_observed, Y, Y))
    proof.formula("AppendixNBFisher", "Expected NB weight using E[Y] = mu",
                  nb_observed.subs(Y, mu), nb_fisher)
    proof.claim("NB score variance equals Fisher weight",
                sp.simplify((mu + alpha * mu**2) / (1 + alpha * mu)**2 - nb_fisher))
    proof.claim("NB IRLS weighted residual equals score",
                sp.simplify(nb_fisher * (Y - mu) / mu - nb_score))
    proof.claim("NB score Poisson limit",
                sp.limit(nb_score, alpha, 0, dir="+") - poisson_score)
    proof.claim("NB observed weight Poisson limit",
                sp.limit(nb_observed, alpha, 0, dir="+") - mu)
    proof.claim("NB Fisher weight Poisson limit",
                sp.limit(nb_fisher, alpha, 0, dir="+") - mu)
    binary_nll = -Y * eta + (Y + 1 / alpha) * sp.log(1 + alpha * sp.exp(eta))
    for value in (0, 1):
        residual = (nb - binary_nll).subs(Y, value)
        residual = residual.replace(sp.loggamma, lambda x: sp.log(sp.gamma(x)))
        proof.claim(f"Binary NB likelihood reduction Y={value}",
                    sp.simplify(sp.expand_func(residual)))


def affine_assembly(proof):
    M, N, P, R = 2, 2, 2, 2
    Z = sp.Matrix(M, R, lambda i, r: sp.Symbol(f"Z{i}{r}", real=True))
    B = sp.Matrix(N, P, lambda j, p: sp.Symbol(f"B{j}{p}", real=True))
    beta = sp.Matrix(sp.symbols(f"beta0:{P * R}", real=True))
    X = sp.kronecker_product(Z, B)
    eta = X * beta
    beta_matrix = sp.Matrix(R, P, list(beta))
    expected_eta = Z * beta_matrix * B.T
    proof.claim("Subject-major predictor and coefficient ordering",
                sp.simplify(eta - sp.Matrix(list(expected_eta))))

    f = [sp.Function(f"f{k}")(eta[k]) for k in range(M * N)]
    nll = sum(f)
    w = []
    for k in range(M * N):
        t = sp.Symbol(f"t{k}", real=True)
        w.append(sp.diff(sp.Function(f"f{k}")(t), t, t).subs(t, eta[k]))
    assembled = sp.zeros(P * R)
    for i in range(M):
        wi = sp.diag(*w[i * N:(i + 1) * N])
        assembled += sp.kronecker_product(Z.row(i).T * Z.row(i), B.T * wi * B)
    proof.claim("Generic affine Hessian equals weighted Kronecker blocks",
                sp.simplify(sp.hessian(nll, beta) - assembled))
    proof.claim("Block assembly equals X transpose W X",
                sp.simplify(assembled - X.T * sp.diag(*w) * X))

    mu_Z = sp.symbols(f"mu_Z0:{M}", positive=True)
    mu_B = sp.symbols(f"mu_B0:{N}", positive=True)
    WZ, WB = sp.diag(*mu_Z), sp.diag(*mu_B)
    proof.claim("Separable Poisson information factorisation",
                sp.simplify(X.T * sp.kronecker_product(WZ, WB) * X
                            - sp.kronecker_product(Z.T * WZ * Z, B.T * WB * B)))
    a, b = sp.symbols("a b", positive=True)
    c, d = sp.symbols("c d", positive=True)
    alpha = sp.Symbol("alpha", positive=True)
    weight_grid = sp.Matrix([
        [a * c / (1 + alpha * a * c), a * d / (1 + alpha * a * d)],
        [b * c / (1 + alpha * b * c), b * d / (1 + alpha * b * d)],
    ])
    determinant = (alpha * a * b * c * d * (a - b) * (d - c)
                   / ((1 + alpha * a * c) * (1 + alpha * a * d)
                      * (1 + alpha * b * c) * (1 + alpha * b * d)))
    proof.formula("AppendixNBWeightDet", "NB separable means need not give separable weights",
                  sp.factor(weight_grid.det()), determinant)
    assert determinant != 0


def separable_assembly(proof):
    M, N, P, R = 2, 2, 2, 2
    B = sp.Matrix(N, P, lambda j, p: sp.Symbol(f"B{j}{p}", real=True))
    Z = sp.Matrix(M, R, lambda i, r: sp.Symbol(f"Z{i}{r}", real=True))
    xi = sp.Matrix(sp.symbols(f"xi0:{P}", real=True))
    gamma = sp.Matrix(sp.symbols(f"gamma0:{R}", real=True))
    eta_B, eta_Z = B * xi, Z * gamma
    mu_B, mu_Z = eta_B.applyfunc(sp.exp), eta_Z.applyfunc(sp.exp)
    Y = sp.Matrix(M, N, lambda i, j: sp.Symbol(f"Y{i}{j}", nonnegative=True))
    nll = sum(mu_B[j] * mu_Z[i] - Y[i, j] * (eta_B[j] + eta_Z[i])
              for i in range(M) for j in range(N))
    a, u = B.T * mu_B, Z.T * mu_Z
    expected = sp.BlockMatrix([
        [sum(mu_Z) * B.T * sp.diag(*mu_B) * B, a * u.T],
        [u * a.T, sum(mu_B) * Z.T * sp.diag(*mu_Z) * Z],
    ]).as_explicit()
    proof.claim("Paper separable Poisson full Hessian",
                sp.simplify(sp.hessian(nll, list(xi) + list(gamma)) - expected))
    proof.claim("Separable Poisson score in xi",
                sp.simplify(-sp.Matrix([sp.diff(nll, x) for x in xi])
                            - B.T * (Y.T * sp.ones(M, 1) - sum(mu_Z) * mu_B)))
    proof.claim("Separable Poisson score in gamma",
                sp.simplify(-sp.Matrix([sp.diff(nll, x) for x in gamma])
                            - Z.T * (Y * sp.ones(N, 1) - sum(mu_B) * mu_Z)))

    # Check all three weighted blocks for an arbitrary cellwise likelihood.
    parameters = list(xi) + list(gamma)
    design = sp.Matrix([
        list(B.row(j)) + list(Z.row(i))
        for i in range(M) for j in range(N)
    ])
    linear = design * sp.Matrix(parameters)
    functions, weights = [], []
    for k in range(M * N):
        t = sp.Symbol(f"t{k}", real=True)
        fn = sp.Function(f"l{k}")
        functions.append(fn(linear[k]))
        weights.append(sp.diff(fn(t), t, t).subs(t, linear[k]))
    xi_block = B.T * sp.diag(*[
        sum(weights[i * N + j] for i in range(M)) for j in range(N)
    ]) * B
    gamma_block = Z.T * sp.diag(*[
        sum(weights[i * N + j] for j in range(N)) for i in range(M)
    ]) * Z
    cross = sp.zeros(P, R)
    for i in range(M):
        for j in range(N):
            cross += weights[i * N + j] * B.row(j).T * Z.row(i)
    blocks = sp.BlockMatrix([[xi_block, cross], [cross.T, gamma_block]]).as_explicit()
    proof.claim("Generic separable-model weighted Hessian",
                sp.simplify(sp.hessian(sum(functions), parameters) - blocks))


def clustered_and_aggregated(proof):
    eta = sp.symbols("eta0:2", real=True)
    mu_B = sp.Matrix([sp.exp(x) for x in eta])
    mu_Z = sp.symbols("mu_Z0:2", positive=True)
    y = sp.symbols("y0:2", nonnegative=True)
    alpha = sp.Symbol("alpha", positive=True)
    t = sp.Symbol("t", real=True)
    E = sum(mu_B)
    g = sum((y[i] + 1 / alpha) * sp.log(t * mu_Z[i] + 1 / alpha)
            for i in range(2))
    gp, gpp = sp.diff(g, t).subs(t, E), sp.diff(g, t, t).subs(t, E)
    expected = gp * sp.diag(*mu_B) + gpp * mu_B * mu_B.T
    proof.claim("Clustered NB log-spatial Hessian is diagonal plus rank one",
                sp.simplify(sp.hessian(g.subs(t, E), eta) - expected))
    proof.claim("Clustered NB spatial Hessian Poisson limit",
                expected.applyfunc(lambda x: sp.simplify(sp.limit(x, alpha, 0, dir="+")))
                - sum(mu_Z) * sp.diag(*mu_B))

    s1, s2, mu, Y = sp.symbols("s_1 s_2 mu Y", positive=True)
    rho = s1**2 / (alpha * s2)
    A = s1 / (alpha * s2)
    proof.claim("Aggregated NB spatial weight Poisson limit",
                sp.limit((rho + Y) * mu * A / (mu + A)**2, alpha, 0, dir="+") - s1 * mu)


def covariance_and_contrasts(proof):
    m, n = sp.symbols("m n", positive=True, integer=True)
    A = sp.MatrixSymbol("A", m, m)
    B = sp.MatrixSymbol("B", n, n)
    product = sp.kronecker_product(A, B) * sp.kronecker_product(A.I, B.I)
    proof.claim("Kronecker inverse product for conformable invertible factors",
                sp.simplify(product.doit() - sp.Identity(m * n)))

    H1 = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"h1{i}{j}", real=True))
    H2 = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"h2{i}{j}", real=True))
    V = blkdiag(H1, H2)
    c1, c2, b1, b2 = sp.symbols("c1 c2 b1 b2", real=True)
    b = sp.Matrix([[b1, b2]])
    T = sp.kronecker_product(sp.Matrix([[c1, c2]]), b)
    proof.claim("Voxel contrast variance with independent coefficient blocks",
                sp.simplify(T * V * T.T - c1**2 * b * H1 * b.T - c2**2 * b * H2 * b.T))

    inv = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"v{min(i,j)}{max(i,j)}", real=True))
    scores = [sp.Matrix([sp.Symbol(f"U{i}{j}", real=True) for j in range(2)])
              for i in range(2)]
    contrast = sp.Matrix([[c1, c2]])
    meat = sum((u * u.T for u in scores), sp.zeros(2))
    projected = sum(((contrast * inv * u) * (contrast * inv * u).T for u in scores),
                    sp.zeros(1))
    proof.claim("Sandwich contrast from projected subject scores",
                sp.simplify(contrast * inv * meat * inv * contrast.T - projected))

    d1, d2, e = sp.symbols("d1 d2 e", nonzero=True, real=True)
    c = sp.Matrix(sp.symbols("c0:2", real=True))
    D = sp.diag(d1, d2)
    schur = e - (c.T * D.inv() * c)[0]
    H = sp.BlockMatrix([[D, c], [c.T, sp.Matrix([[e]])]]).as_explicit()
    lam = sp.Symbol("lambda", real=True)
    discrepancy = sp.expand((H - lam * sp.eye(3)).det()
                            - (D - lam * sp.eye(2)).det() * (schur - lam))
    proof.claim("Bordered determinant is not a block-spectrum factorisation",
                sp.simplify(discrepancy
                            - lam * (c[0]**2 * (lam - d2) / d1
                                     + c[1]**2 * (lam - d1) / d2)))
    assert discrepancy != 0


def covariate_effects(proof):
    x, gamma = sp.symbols("x gamma_x", real=True)
    baseline = sp.Matrix(sp.symbols("f0:2", real=True))
    B = sp.Matrix(2, 2, lambda j, p: sp.Symbol(f"B{j}{p}", real=True))
    beta_x = sp.Matrix(sp.symbols("beta_x0:2", real=True))
    spatial_effect = B * beta_x
    eta_gc = baseline + x * gamma * sp.ones(2, 1)
    eta_sv = baseline + x * spatial_effect
    proof.claim("GC covariate effect is constant across voxels",
                eta_gc.diff(x) - gamma * sp.ones(2, 1))
    proof.claim("SV covariate effect is a spatial coefficient map",
                eta_sv.diff(x) - spatial_effect)
    for label, predictor, effect in [
        ("GC", eta_gc, gamma * sp.ones(2, 1)),
        ("SV", eta_sv, spatial_effect),
    ]:
        mean = predictor.applyfunc(sp.exp)
        ratio = sp.Matrix([mean[j].subs(x, x + 1) / mean[j] for j in range(2)])
        proof.claim(f"{label} one-unit covariate mean ratio",
                    sp.simplify(ratio - effect.applyfunc(sp.exp)))
    # With an explicit constant basis column, beta_x = (gamma_x, 0)
    # represents a constant function even when the other feature varies.
    z0, z1 = sp.symbols("z0 z1", real=True)
    constant_basis = sp.Matrix([[1, z0], [1, z1]])
    restricted_effect = constant_basis * sp.Matrix([gamma, 0])
    proof.claim("Constant-function restriction reduces SV to GC",
                baseline + x * restricted_effect - eta_gc)
    proof.claim("Restricted SV mean is exactly separable",
                sp.simplify((baseline + x * restricted_effect).applyfunc(sp.exp)
                            - sp.exp(x * gamma) * baseline.applyfunc(sp.exp)))


def run():
    proof = AppendixProof()
    scalar_likelihoods(proof)
    affine_assembly(proof)
    separable_assembly(proof)
    clustered_and_aggregated(proof)
    covariance_and_contrasts(proof)
    covariate_effects(proof)
    return proof

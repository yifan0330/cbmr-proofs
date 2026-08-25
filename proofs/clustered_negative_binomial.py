"""Clustered Negative Binomial: the observed information, in one pass.

Cheaper to close than the negative binomial case. With the nuisance parameters held fixed, every
special function in this likelihood depends only on the counts and the precision, so the whole
composite Hessian differentiates and simplifies without decomposition.

This is also the only one of the three whose second differential in the log-intensity is not
diagonal: the spatial coefficients enter through the scalar E = sum_v u_v, giving a diagonal
plus rank-one structure and hence a rank-one correction to the spatial block.
"""
import sympy as sp

from proofs.latex import Proof

V, N, K, Q = 3, 2, 2, 2

PREAMBLE = r"""The latent effect is a property of the experiment rather than of the voxel, so
the spatial coefficients reach the likelihood only through the scalar $E=\sum_v u_v$. Writing
$\Gamma(E)=\sum_i (y_i+\nu)\log(Ee_i+\nu)$, the second differential in $S$ is
$\Gamma''(E)u_vu_{v'}+\Gamma'(E)u_v\delta_{vv'}$ --- diagonal plus rank one."""


def run():
    proof = Proof("clustered_negative_binomial", "Clustered Negative Binomial",
                  preamble=PREAMBLE)

    beta = sp.symbols(f"beta0:{K}", real=True)
    gamma = sp.symbols(f"gamma0:{Q}", real=True)
    B = sp.Matrix(V, K, lambda v, k: sp.Symbol(f"B{v}{k}", real=True))
    G = sp.Matrix(N, Q, lambda i, q: sp.Symbol(f"G{i}{q}", real=True))
    Y = sp.symbols(f"Y0:{V}", nonnegative=True)
    y = sp.symbols(f"y0:{N}", nonnegative=True)
    nu = sp.Symbol("nu", positive=True)

    S = [sum(B[v, k] * beta[k] for k in range(K)) for v in range(V)]
    u = [sp.exp(S[v]) for v in range(V)]
    m = [sum(G[i, q] * gamma[q] for q in range(Q)) for i in range(N)]
    e = [sp.exp(m[i]) for i in range(N)]
    E = sum(u)

    loglik = (N * nu * sp.log(nu) - N * sp.loggamma(nu)
              + sum(sp.loggamma(y[i] + nu) for i in range(N))
              - sum((y[i] + nu) * sp.log(E * e[i] + nu) for i in range(N))
              + sum(Y[v] * S[v] for v in range(V))
              + sum(y[i] * m[i] for i in range(N)))
    nll = -loglik
    proof.define(r"-\log\mathcal{L}_p", nll)
    params = list(beta) + list(gamma)

    a = [sum(u[v] * B[v, k] for v in range(V)) for k in range(K)]
    M = [[sum(u[v] * B[v, k] * B[v, l] for v in range(V)) for l in range(K)] for k in range(K)]
    D = [E * e[i] + nu for i in range(N)]
    gp = sum((y[i] + nu) * e[i] / D[i] for i in range(N))
    gpp = -sum((y[i] + nu) * e[i]**2 / D[i]**2 for i in range(N))
    h = [sum((y[i] + nu) * nu * e[i] * G[i, q] / D[i]**2 for i in range(N)) for q in range(Q)]
    kappa = [(y[i] + nu) * nu * E * e[i] / D[i]**2 for i in range(N)]
    proof.define(r"g'", gp)
    proof.define(r"g''", gpp)

    claim = sp.zeros(len(params), len(params))
    for k in range(K):
        for l in range(K):
            claim[k, l] = gp * M[k][l] + gpp * a[k] * a[l]
    for k in range(K):
        for q in range(Q):
            claim[k, K + q] = claim[K + q, k] = a[k] * h[q]
    for q in range(Q):
        for r in range(Q):
            claim[K + q, K + r] = sum(kappa[i] * G[i, q] * G[i, r] for i in range(N))

    names = [f"beta{k}" for k in range(K)] + [f"gamma{q}" for q in range(Q)]
    shown = [rf"\beta_{k}" for k in range(K)] + [rf"\gamma_{q}" for q in range(Q)]
    for i in range(len(params)):
        for j in range(i, len(params)):
            proof.claim(
                f"d2/d{names[i]} d{names[j]}",
                sp.simplify(sp.diff(nll, params[i], params[j]) - claim[i, j]),
                rf"\partial^2_{{{shown[i]}{shown[j]}}} - \textrm{{closed form}}")
    return proof

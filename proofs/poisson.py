"""Poisson: the observed information, and its independence from the data.

The Poisson case was previously verified only numerically. It is the cheapest of the three to
close symbolically and it carries the claim the other two do not -- that the Hessian does not
depend on the foci at all -- so it belongs here.
"""
import sympy as sp

from proofs.latex import Proof

V, N, K, Q = 3, 2, 2, 2


def run():
    proof = Proof(
        "poisson", "Poisson",
        preamble=r"""The log-likelihood on marginals, dropping the parameter-free
$-\sum\log y!$, is $\log\mathcal{L}=\sum_{v}Y_{v}S_{v}+\sum_i y_i m_i-(\sum_v e^{S_v})T$ with
$T=\sum_i e^{m_i}$. The first two terms are linear in the coefficients, so they differentiate
away entirely; that is the whole of the data dependence.""")

    beta = sp.symbols(f"beta0:{K}", real=True)
    gamma = sp.symbols(f"gamma0:{Q}", real=True)
    B = sp.Matrix(V, K, lambda v, k: sp.Symbol(f"B{v}{k}", real=True))
    G = sp.Matrix(N, Q, lambda i, q: sp.Symbol(f"G{i}{q}", real=True))
    Y = sp.symbols(f"Y0:{V}", nonnegative=True)
    y = sp.symbols(f"y0:{N}", nonnegative=True)

    S = [sum(B[v, k] * beta[k] for k in range(K)) for v in range(V)]
    u = [sp.exp(S[v]) for v in range(V)]
    m = [sum(G[i, q] * gamma[q] for q in range(Q)) for i in range(N)]
    e = [sp.exp(m[i]) for i in range(N)]
    T = sum(e)
    E = sum(u)

    nll = -(sum(Y[v] * S[v] for v in range(V)) + sum(y[i] * m[i] for i in range(N)) - E * T)
    params = list(beta) + list(gamma)
    proof.define(r"-\log\mathcal{L}", nll)

    a = [sum(u[v] * B[v, k] for v in range(V)) for k in range(K)]
    U = [sum(e[i] * G[i, q] for i in range(N)) for q in range(Q)]

    for k in range(K):
        for l in range(K):
            claim = T * sum(u[v] * B[v, k] * B[v, l] for v in range(V))
            proof.claim(f"d2/dbeta{k} dbeta{l}",
                        sp.simplify(sp.diff(nll, beta[k], beta[l]) - claim),
                        rf"\partial^2_{{\beta_{k}\beta_{l}}} - T\,(B^\top\mathrm{{diag}}(u)B)_{{{k}{l}}}")
    for k in range(K):
        for q in range(Q):
            proof.claim(f"d2/dbeta{k} dgamma{q}",
                        sp.simplify(sp.diff(nll, beta[k], gamma[q]) - a[k] * U[q]),
                        rf"\partial^2_{{\beta_{k}\gamma_{q}}} - a_{k}U_{q}")
    for q in range(Q):
        for r in range(Q):
            claim = E * sum(e[i] * G[i, q] * G[i, r] for i in range(N))
            proof.claim(f"d2/dgamma{q} dgamma{r}",
                        sp.simplify(sp.diff(nll, gamma[q], gamma[r]) - claim),
                        rf"\partial^2_{{\gamma_{q}\gamma_{r}}} - E\sum_i e_iG_{{i{q}}}G_{{i{r}}}")

    # The claim that distinguishes Poisson from the other two.
    hessian = sp.Matrix(len(params), len(params),
                        lambda i, j: sp.diff(nll, params[i], params[j]))
    free = set().union(*[h.free_symbols for h in hessian])
    residual = sp.Integer(0) if not (free & set(Y) | free & set(y)) else sp.Integer(1)
    proof.claim("Hessian contains no foci symbol", residual,
                r"\{Y_v, y_i\} \cap \mathrm{free}(H)")
    return proof

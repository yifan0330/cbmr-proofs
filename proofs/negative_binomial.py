"""Negative Binomial: the observed information, via the (R, A) reparameterisation.

Closed in three pieces rather than one. Writing the likelihood as Psi(R, A, S) separates the
special-function algebra from the chain rule, and each piece is small enough for sympy:

  1. NiMARE's summand *is* Psi(R, A, S) under R = S1^2/(theta S2), A = S1/(theta S2);
  2. Psi's second partials in (S, R, A) are what the implementation uses;
  3. R and A's first and second derivatives in gamma are what the implementation uses.

The composite is then the multivariate chain rule applied to 2 and 3, which is a theorem rather
than a conjecture. Attempting the composite directly does not terminate under `simplify`.
"""
import sympy as sp

from proofs.latex import Proof

V, N, Q = 3, 2, 2

PREAMBLE = r"""NiMARE's negative binomial uses shape $r=s_1^2/(\theta s_2)$ and success
probability $\pi_v=(1+s_1/(\theta u_v s_2))^{-1}$. Under the reparameterisation
$A=s_1/(\theta s_2)$, $R=s_1A$ we have $\pi_v=u_v/(u_v+A)$, and the special functions depend on
$R$ alone, the voxels on $S$ alone, and the moderators on $(R,A)$ alone."""


def run():
    proof = Proof("negative_binomial", "Negative Binomial", preamble=PREAMBLE)

    R, A, theta = sp.symbols("R A theta", positive=True)
    S = sp.symbols(f"S0:{V}", real=True)
    y = sp.symbols(f"y0:{V}", nonnegative=True)
    gamma = sp.symbols(f"gamma0:{Q}", real=True)
    G = sp.Matrix(N, Q, lambda i, q: sp.Symbol(f"G{i}{q}", real=True))
    u = [sp.exp(S[v]) for v in range(V)]
    d = [u[v] + A for v in range(V)]

    # --- 1. the reparameterisation is exact -------------------------------------------------
    S1, S2 = sp.symbols("S1 S2", positive=True)
    nimare = 0
    for v in range(V):
        r_ = S1**2 / (theta * S2)
        p_ = 1 / (1 + S1 / (theta * u[v] * S2))
        nimare += (sp.loggamma(y[v] + r_) - sp.loggamma(y[v] + 1) - sp.loggamma(r_)
                   + r_ * sp.log(1 - p_) + y[v] * sp.log(p_))
    Psi = (-sum(sp.loggamma(y[v] + R) for v in range(V)) + V * sp.loggamma(R)
           - V * R * sp.log(A) + sum((R + y[v]) * sp.log(d[v]) for v in range(V))
           - sum(y[v] * S[v] for v in range(V)))
    proof.define(r"\Psi(R, A, S)", Psi)

    # Substitute the definitions into Psi, not the other way round: given (s1, s2, theta), R and
    # A are dependent, so eliminating only one relation leaves equal expressions looking unequal.
    substituted = Psi.subs({R: S1**2 / (theta * S2), A: S1 / (theta * S2)})
    target = -nimare - sum(sp.loggamma(y[v] + 1) for v in range(V))
    proof.claim("Psi(R,A,S) equals NiMARE's summand",
                sp.simplify(sp.expand_log(sp.expand(substituted - target), force=True)),
                r"\Psi\big|_{R,A} - \big(-\log\mathcal{L} - \textrm{const}\big)")

    # --- 2. second partials of Psi ----------------------------------------------------------
    for v in range(V):
        proof.claim(f"d2Psi/dS{v}^2",
                    sp.simplify(sp.diff(Psi, S[v], S[v])
                                - (R + y[v]) * u[v] * A / d[v]**2),
                    rf"\partial^2_{{S_{v}}}\Psi - \frac{{(R+Y_{v})u_{v}A}}{{(u_{v}+A)^2}}")
        proof.claim(f"d2Psi/dS{v} dR", sp.simplify(sp.diff(Psi, S[v], R) - u[v] / d[v]),
                    rf"\partial^2_{{S_{v}R}}\Psi - \frac{{u_{v}}}{{u_{v}+A}}")
        proof.claim(f"d2Psi/dS{v} dA",
                    sp.simplify(sp.diff(Psi, S[v], A) + (R + y[v]) * u[v] / d[v]**2),
                    rf"\partial^2_{{S_{v}A}}\Psi + \frac{{(R+Y_{v})u_{v}}}{{(u_{v}+A)^2}}")

    for label, expected, wrt, shown in [
        ("Psi_R", -sum(sp.digamma(y[v] + R) for v in range(V)) + V * sp.digamma(R)
         - V * sp.log(A) + sum(sp.log(d[v]) for v in range(V)), (R,), r"\Psi_R"),
        ("Psi_A", -V * R / A + sum((R + y[v]) / d[v] for v in range(V)), (A,), r"\Psi_A"),
        ("Psi_RR", -sum(sp.polygamma(1, y[v] + R) for v in range(V)) + V * sp.polygamma(1, R),
         (R, R), r"\Psi_{RR}"),
        ("Psi_RA", -V / A + sum(1 / d[v] for v in range(V)), (R, A), r"\Psi_{RA}"),
        ("Psi_AA", V * R / A**2 - sum((R + y[v]) / d[v]**2 for v in range(V)), (A, A),
         r"\Psi_{AA}"),
    ]:
        proof.claim(label, sp.simplify(sp.diff(Psi, *wrt) - expected),
                    shown + r"\ \textrm{as implemented}")

    # --- 3. derivatives of R and A in gamma -------------------------------------------------
    e = [sp.exp(sum(G[i, q] * gamma[q] for q in range(Q))) for i in range(N)]
    s1, s2 = sum(e), sum(ei**2 for ei in e)
    R_of, A_of = s1**2 / (theta * s2), s1 / (theta * s2)
    U = [sum(e[i] * G[i, q] for i in range(N)) for q in range(Q)]
    W = [sum(e[i]**2 * G[i, q] for i in range(N)) for q in range(Q)]
    U2 = [[sum(e[i] * G[i, q] * G[i, r] for i in range(N)) for r in range(Q)] for q in range(Q)]
    W2 = [[sum(e[i]**2 * G[i, q] * G[i, r] for i in range(N)) for r in range(Q)]
          for q in range(Q)]
    ell = [U[q] / s1 for q in range(Q)]
    h = [2 * W[q] / s2 for q in range(Q)]
    ell2 = [[U2[q][r] / s1 - U[q] * U[r] / s1**2 for r in range(Q)] for q in range(Q)]
    h2 = [[4 * W2[q][r] / s2 - 4 * W[q] * W[r] / s2**2 for r in range(Q)] for q in range(Q)]
    dA = [ell[q] - h[q] for q in range(Q)]
    dR = [2 * ell[q] - h[q] for q in range(Q)]

    for q in range(Q):
        proof.claim(f"dR/dgamma{q}", sp.simplify(sp.diff(R_of, gamma[q]) - R_of * dR[q]),
                    rf"\partial_{{\gamma_{q}}}R - R(2\ell_{q}-h_{q})")
        proof.claim(f"dA/dgamma{q}", sp.simplify(sp.diff(A_of, gamma[q]) - A_of * dA[q]),
                    rf"\partial_{{\gamma_{q}}}A - A(\ell_{q}-h_{q})")
        for r in range(Q):
            proof.claim(
                f"d2R/dgamma{q} dgamma{r}",
                sp.simplify(sp.diff(R_of, gamma[q], gamma[r])
                            - R_of * (dR[q] * dR[r] + 2 * ell2[q][r] - h2[q][r])),
                rf"\partial^2_{{\gamma_{q}\gamma_{r}}}R\ \textrm{{as implemented}}")
            proof.claim(
                f"d2A/dgamma{q} dgamma{r}",
                sp.simplify(sp.diff(A_of, gamma[q], gamma[r])
                            - A_of * (dA[q] * dA[r] + ell2[q][r] - h2[q][r])),
                rf"\partial^2_{{\gamma_{q}\gamma_{r}}}A\ \textrm{{as implemented}}")
    return proof

"""The covariance step: structural separation, and the four inversion strategies.

Once the Hessian is closed form, the remaining cost is a condition number and an inverse. These
are the five claims those two calls rest on.
"""
import sympy as sp

from proofs.latex import Proof

PREAMBLE = r"""Let $\mathrm{supp}(p)=\{c: L_{pc}\neq0\}$ and let $\sim$ be the transitive
closure of $c\sim c' \iff \exists p:\,c,c'\in\mathrm{supp}(p)$. Every distribution's Hessian has
$\beta\beta$ block $\sum_p L_{pc}L_{pc'}(B^\top\Sigma_pB)_{kk'}$, so the factor $L_{pc}L_{pc'}$
decides which entries can be nonzero at all."""


def blkdiag(*blocks):
    """Block diagonal matrix, written out rather than assembled by a library helper."""
    out = sp.zeros(sum(b.rows for b in blocks), sum(b.cols for b in blocks))
    r = c = 0
    for b in blocks:
        out[r:r + b.rows, c:c + b.cols] = b
        r, c = r + b.rows, c + b.cols
    return out


def run():
    proof = Proof("covariance", "Covariance strategies", preamble=PREAMBLE)
    lam = sp.Symbol("lambda")

    # --- 1. structural separation -----------------------------------------------------------
    C, K, P = 3, 2, 2
    L = sp.Matrix(P, C, lambda p, c: sp.Symbol(f"L{p}{c}", real=True))
    M = [sp.Matrix(K, K, lambda k, l: sp.Symbol(f"M{p}_{k}{l}", real=True)) for p in range(P)]
    # Column 0 loaded only by pattern 0, column 2 only by pattern 1, column 1 by both.
    L = L.subs({sp.Symbol("L02", real=True): 0, sp.Symbol("L10", real=True): 0})
    H = sp.zeros(C * K, C * K)
    for p in range(P):
        for c in range(C):
            for d in range(C):
                H[c * K:(c + 1) * K, d * K:(d + 1) * K] += L[p, c] * L[p, d] * M[p]
    proof.claim("columns sharing no pattern have a zero cross block",
                sp.simplify(H[0:K, 2 * K:3 * K]),
                r"H_{(0k),(2k')}\ \textrm{with}\ 0\not\sim 2")
    assert not sp.simplify(H[0:K, K:2 * K]).is_zero_matrix, (
        "columns that do share a pattern must not vanish, or the claim is vacuous")

    # --- 2. block diagonal spectrum ---------------------------------------------------------
    A = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"a{i}{j}", real=True))
    Bm = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"b{i}{j}", real=True))
    proof.claim(
        "characteristic polynomial factorises over the blocks",
        sp.simplify(sp.expand((blkdiag(A, Bm) - lam * sp.eye(4)).det()
                              - (A - lam * sp.eye(2)).det() * (Bm - lam * sp.eye(2)).det())),
        r"\det(\mathrm{blkdiag}(A,B)-\lambda I)-\det(A-\lambda I)\det(B-\lambda I)")

    # --- 3. symmetry suffices ---------------------------------------------------------------
    S = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"s{min(i,j)}{max(i,j)}", real=True))
    proof.claim("symmetric H satisfies H^T H = H^2", sp.simplify(S.T * S - S * S),
                r"H^\top H - H^2")
    proof.claim(
        "the singular values are the absolute eigenvalues",
        sp.simplify(sp.expand((S * S - lam**2 * sp.eye(3)).det()
                              - (S - lam * sp.eye(3)).det() * (S + lam * sp.eye(3)).det())),
        r"\det(H^2-\lambda^2I)-\det(H-\lambda I)\det(H+\lambda I)")

    # --- 4. blockwise inverse ---------------------------------------------------------------
    proof.claim("blockwise inverse", sp.simplify(blkdiag(A, Bm).inv()
                                                 - blkdiag(A.inv(), Bm.inv())),
                r"\mathrm{blkdiag}(A,B)^{-1}-\mathrm{blkdiag}(A^{-1},B^{-1})")

    # --- 5. bordered (Schur) inverse --------------------------------------------------------
    # Over matrix symbols: filling in a symbolic D^-1 inside the product makes `simplify`
    # non-terminating, and it proves less. The identity needs only invertibility of D and Sigma
    # and conforming shapes, so D^-1 and Sigma^-1 stay formal.
    m, q = sp.symbols("m q", positive=True, integer=True)
    Dm, Cb, Eb = sp.MatrixSymbol("D", m, m), sp.MatrixSymbol("C", m, q), sp.MatrixSymbol("E", q, q)
    Sigma = sp.MatrixSymbol("Sigma", q, q)
    Di, Si = Dm.I, Sigma.I
    top_left = Di + Di * Cb * Si * Cb.T * Di
    top_right = -Di * Cb * Si
    bottom_left = -Si * Cb.T * Di
    bottom_right = Si
    E_expanded = Sigma + Cb.T * Di * Cb          # the definition of the Schur complement
    for label, left, right, shown in [
        ("D.TL + C.BL = I", Dm * top_left + Cb * bottom_left, sp.Identity(m),
         r"D\,\mathrm{TL}+C\,\mathrm{BL}-I_m"),
        ("D.TR + C.BR = 0", Dm * top_right + Cb * bottom_right, sp.ZeroMatrix(m, q),
         r"D\,\mathrm{TR}+C\,\mathrm{BR}"),
        ("C^T.TL + E.BL = 0", (Cb.T * top_left + Eb * bottom_left).subs(Eb, E_expanded),
         sp.ZeroMatrix(q, m), r"C^\top\mathrm{TL}+E\,\mathrm{BL}"),
        ("C^T.TR + E.BR = I", (Cb.T * top_right + Eb * bottom_right).subs(Eb, E_expanded),
         sp.Identity(q), r"C^\top\mathrm{TR}+E\,\mathrm{BR}-I_q"),
    ]:
        proof.claim(label, sp.simplify((left - right).doit().expand()), shown)
    return proof

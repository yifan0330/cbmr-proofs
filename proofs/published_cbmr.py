"""Exact identities linking the separable lesion and grouped CBMR models.

Moment-matched NB aggregation and delta-method uncertainty remain approximations;
the algebraic identities used to construct them are what this module verifies.
Design A uses the full R-by-P coefficient matrix beta (PR parameters).
Design B instead stacks group spatial xi and shared gamma (GP+R parameters);
its study-major design is [L kron B, Z kron ones_N], not Z kron B.
"""

import sympy as sp

from proofs.latex import Proof


def _grouped_design(proof):
    """Check Design B with interleaved groups and fully symbolic B, Z, and theta."""
    membership = sp.Matrix([[1, 0], [0, 1], [1, 0], [0, 1]])
    basis = sp.Matrix(2, 2, lambda j, p: sp.Symbol(f"B_{j}{p}", real=True))
    global_design = sp.Matrix(4, 2, lambda i, r: sp.Symbol(f"Z_{i}{r}", real=True))
    spatial = [sp.Matrix(sp.symbols(f"xi_{g}1 xi_{g}2", real=True)) for g in (1, 2)]
    gamma = sp.Matrix(sp.symbols("gamma_1:3", real=True))
    theta = sp.Matrix([*spatial[0], *spatial[1], *gamma])
    design = sp.kronecker_product(membership, basis).row_join(
        sp.kronecker_product(global_design, sp.ones(2, 1)))
    expected = sp.Matrix([
        (basis.row(j)*spatial[g])[0] + (global_design.row(i)*gamma)[0]
        for i, g in enumerate((0, 1, 0, 1)) for j in range(2)
    ])
    proof.claim("Grouped design study-major parameter ordering",
                (design*theta-expected).applyfunc(sp.expand),
                r"[L\otimes B,\;Z\otimes\mathbf{1}_N]\vartheta"
                r"-\operatorname{vec}_{\rm study}(B_j^\top\xi_{g(i)}+Z_i^\top\gamma)")


PREAMBLE = r"""
Write $B\in\mathbb R^{N\times P}$, $Z\in\mathbb R^{M\times R}$,
$\mu_{B,gj}=\exp(B_j^\top\xi_g)$, $\mu_{Z,i}=\exp(Z_i^\top\gamma)$ and
$\mu_{ij}=\mu_{B,gj}\mu_{Z,i}$ for $i\in g$.
Here $\gamma$ is shared, $\alpha_g>0$ is the original NB dispersion and
$\nu_g=1/\alpha_g$. All checks use exact arithmetic and small symbolic instances.
Poisson likelihood comparisons omit only data-dependent factorial constants.
Set $Y_{gj}=\sum_{i\in g}Y_{ij}$, $Y_{i\cdot}=\sum_jY_{ij}$,
$E_g=\sum_j\mu_{B,gj}$ and $T_g=\sum_{i\in g}\mu_{Z,i}$.
The Poisson check has two groups of two studies and two voxels.
Independent NB aggregation below is moment matching, not an exact likelihood
unless the study factors coincide: $a,c>0$ are the two study factors,
$s_k=a^k+c^k$, and $b=\mu_B$ in its displayed definitions.
Clustered Gamma--Poisson formulas instead use one shared, mean-one latent
multiplier per study, with $t=\sum_j\mu_j$ and $y=\sum_jY_j$.
Convexity statements hold at fixed dispersion. Roughness matrices are symmetric
and positive semidefinite, with fixed nonnegative smoothing parameters.
Covariance inverses require identifiability; the Schur example assumes
$d_1,d_2,S>0$. The sandwich difference is checked in the scalar case
with positive information $I$ and penalty curvature $K=k$.
Delta-method covariance is a first-order approximation.
"""


def _poisson(proof):
    b = sp.Matrix(2, 2, lambda g, j: sp.Symbol(f"b_{g}{j}", positive=True))
    u = sp.symbols("u_0:4", positive=True)
    counts = sp.Matrix(
        4, 2, lambda i, j: sp.Symbol(f"Y_{i}{j}", nonnegative=True, integer=True))
    totals = [sum(counts[i, j] for j in range(2)) for i in range(4)]
    group_counts = [[counts[2*g, j] + counts[2*g+1, j] for j in range(2)]
                    for g in range(2)]
    spatial = [sum(b[g, j] for j in range(2)) for g in range(2)]
    exposure = [u[2*g] + u[2*g+1] for g in range(2)]
    full = sum(b[i//2, j]*u[i] - counts[i, j]
               * (sp.log(b[i//2, j]) + sp.log(u[i]))
               for i in range(4) for j in range(2))
    grouped = sum(spatial[g]*exposure[g]
                  - sum(group_counts[g][j]*sp.log(b[g, j]) for j in range(2))
                  for g in range(2)) - sum(totals[i]*sp.log(u[i]) for i in range(4))
    proof.claim("Grouped Poisson full likelihood", sp.expand(full - grouped),
                r"\ell^-_{\rm full}-\{\sum_g(E_gT_g-\sum_jY_{gj}\eta_{B,gj})"
                r"-\sum_iY_{i\cdot}\eta_{Z,i}\}")

    voxel_nll = sum(b[g, j]*exposure[g]
                    - group_counts[g][j]*sp.log(b[g, j]*exposure[g])
                    for g in range(2) for j in range(2))
    allocation = sum(totals[i]*sp.log(u[i]/exposure[i//2]) for i in range(4))
    proof.claim("Study allocation is the omitted likelihood",
                sp.expand(sp.expand_log(-full + voxel_nll - allocation)),
                r"\ell_{\rm full}-\ell_{\rm voxel}"
                r"-\sum_iY_{i\cdot}\log(\mu_{Z,i}/T_{g(i)})")

    gamma = sp.Matrix(sp.symbols("gamma_1:3", real=True))
    design = sp.Matrix([[0, 0], [1, 1], [1, 0], [0, 1]])
    factors = (design*gamma).applyfunc(sp.exp)
    allocation_eta = allocation.subs(dict(zip(u, factors)))
    expected = sum((totals[i]*design.row(i).T for i in range(4)), sp.zeros(2, 1))
    for g in range(2):
        indices = range(2*g, 2*g+2)
        expected -= (sum(totals[i] for i in indices)
                     * sum((factors[i]*design.row(i).T for i in indices), sp.zeros(2, 1))
                     / sum(factors[i] for i in indices))
    gradient = sp.Matrix([sp.diff(allocation_eta, v) for v in gamma])
    proof.define("Z_{\\rm check}", design)
    proof.claim("Conditional allocation global score",
                (gradient - expected).applyfunc(sp.simplify),
                r"\nabla_\gamma\ell_{\rm alloc}"
                r"-\sum_g\{\sum_{i\in g}Y_{i\cdot}Z_i"
                r"-Y_{g\cdot}\sum_{i\in g}\mu_{Z,i}Z_i/T_g\}")
    first_group = sum(totals[i]*sp.log(factors[i]/(factors[0]+factors[1]))
                      for i in range(2))
    witness = sp.diff(first_group, gamma[0]).subs(dict.fromkeys(gamma, 0))
    proof.claim("Allocation score need not vanish",
                sp.simplify(witness - (totals[1] - totals[0])/2),
                r"\left.\partial_{\gamma_1}\ell_{{\rm alloc},1}\right|_{\gamma=0}"
                r"-(Y_{1\cdot}-Y_{0\cdot})/2")
    assert witness != 0, "The variable-global-effect counterexample must be nonvacuous."


def _independent_nb(proof):
    alpha, b, a, c = sp.symbols("alpha b a c", positive=True)
    s1, s2, s3 = a+c, a**2+c**2, a**3+c**3
    rho = proof.define(r"\rho", s1**2/(alpha*s2))
    rate = proof.define("A", s1/(alpha*s2))
    probability = proof.define("p", b/(b+rate))
    proof.claim("Matched NB mean",
                sp.cancel(rho*probability/(1-probability) - b*s1),
                r"\rho p/(1-p)-\mu_Bs_1")
    proof.claim("Matched NB variance",
                sp.cancel(rho*probability/(1-probability)**2 - b*s1-alpha*b**2*s2),
                r"\rho p/(1-p)^2-(\mu_Bs_1+\alpha\mu_B^2s_2)")
    proof.claim("Aggregate effective dispersion", sp.cancel(1/rho-alpha*s2/s1**2),
                r"\rho^{-1}-\alpha s_2/s_1^2")

    t = sp.Symbol("t", real=True)
    mean, shape = sp.symbols("m r", positive=True)
    cgf = -shape*sp.log(1-mean/shape*(sp.exp(t)-1))
    third = sp.diff(cgf, t, 3).subs(t, 0)
    independent = sum(third.subs({mean: b*u, shape: 1/alpha}) for u in (a, c))
    matched = third.subs({mean: b*s1, shape: rho})
    gap = proof.define(r"\Delta\kappa_3", 2*alpha**2*b**3*(s3-s2**2/s1),
                       r"The independent sum minus the moment-matched NB third cumulant.")
    proof.claim("Third cumulant aggregation mismatch",
                sp.cancel(independent - matched - gap),
                r"\kappa_{3,\rm independent}-\kappa_{3,\rm matched}"
                r"-2\alpha^2\mu_B^3(s_3-s_2^2/s_1)")
    proof.claim("Mismatch positive for unequal factors",
                sp.cancel(s3-s2**2/s1-a*c*(a-c)**2/(a+c)),
                r"s_3-s_2^2/s_1-ac(a-c)^2/(a+c)")
    proof.claim("Equal factors remove cumulant mismatch",
                sp.simplify((independent-matched).subs(c, a)),
                r"\left.(\kappa_{3,\rm independent}-\kappa_{3,\rm matched})\right|_{c=a}")


def _clustered(proof):
    nu, alpha = sp.symbols("nu alpha", positive=True)
    mu = sp.Matrix(sp.symbols("mu_1:4", positive=True))
    counts = sp.Matrix(sp.symbols("y_1:4", nonnegative=True, integer=True))
    total, y = sum(mu), sum(counts)
    diagonal = sp.diag(*mu)
    # E[Cov(Y|U)] + Cov(E[Y|U]), with E[U]=1 and E[U^2]=1+alpha.
    second = diagonal + (1+alpha)*mu*mu.T
    covariance = diagonal + alpha*mu*mu.T
    proof.define(r"\operatorname{Cov}(Y_i)", covariance)
    proof.claim("Latent multiplier covariance",
                (second-mu*mu.T-covariance).applyfunc(sp.expand),
                r"E[YY^\top]-\mu\mu^\top"
                r"-\{\operatorname{diag}(\mu)+\alpha\mu\mu^\top\}")

    joint_log = (sp.loggamma(nu+y)-sp.loggamma(nu)+nu*sp.log(nu)
                 -(nu+y)*sp.log(nu+total)
                 + sum(counts[j]*sp.log(mu[j])-sp.loggamma(counts[j]+1)
                       for j in range(3)))
    nb_log = (sp.loggamma(nu+y)-sp.loggamma(nu)-sp.loggamma(y+1)
              +nu*sp.log(nu/(nu+total))+y*sp.log(total/(nu+total)))
    multinomial_log = (sp.loggamma(y+1)
                       -sum(sp.loggamma(v+1) for v in counts)
                       +sum(counts[j]*sp.log(mu[j]/total) for j in range(3)))
    proof.define(r"\log p_{\rm joint}", joint_log,
                 r"Gamma--Poisson marginal with latent shape and rate $\nu=1/\alpha$.")
    proof.claim("Joint NB-total multinomial factorisation",
                sp.expand(sp.expand_log(joint_log-nb_log-multinomial_log)),
                r"\log p_{\rm joint}-\log p_{\rm NB}(y;\nu,t)"
                r"-\log p_{\rm Mult}(Y;y,\mu/t)")

    eta = sp.Matrix(sp.symbols("eta_1:4", real=True))
    exp_mu = eta.applyfunc(sp.exp)
    nll = (y+nu)*sp.log(nu+sum(exp_mu))-(counts.T*eta)[0]
    observed = (y+nu)/(nu+total)*(diagonal-mu*mu.T/(nu+total))
    direct = sp.hessian(nll, eta).subs(dict(zip(exp_mu, mu)))
    proof.define("W", diagonal-mu*mu.T/(nu+total))
    proof.claim("Clustered observed log-mean Hessian",
                (direct-observed).applyfunc(sp.cancel),
                r"\nabla_\eta^2 f-\frac{y+\nu}{\nu+t}"
                r"\{\operatorname{diag}(\mu)-\mu\mu^\top/(\nu+t)\}")
    expected = observed.subs(dict(zip(counts, mu)))
    weight = diagonal-mu*mu.T/(nu+total)
    proof.claim("Expected clustered curvature",
                (expected-weight).applyfunc(sp.cancel),
                r"E[\nabla_\eta^2 f]-W\quad(E[y]=t)")
    v = sp.Matrix(sp.symbols("v_1:4", real=True))
    positive_form = (nu*sum(mu[j]*v[j]**2 for j in range(3))
                     +sum(mu[j]*mu[k]*(v[j]-v[k])**2
                          for j in range(3) for k in range(j+1, 3)))
    proof.claim("Clustered curvature positive quadratic form",
                sp.cancel((nu+total)*(v.T*weight*v)[0]-positive_form),
                r"(\nu+t)v^\top Wv-\nu\sum_j\mu_jv_j^2"
                r"-\sum_{j<k}\mu_j\mu_k(v_j-v_k)^2")


def _penalty(proof):
    j11, j12, j22 = sp.symbols("J_11 J_12 J_22", real=True)
    roughness = sp.Matrix([[j11, j12], [j12, j22]])
    lambdas = sp.symbols("lambda_1:3", nonnegative=True)
    xi = [sp.Matrix(sp.symbols(f"xi_{g}1 xi_{g}2", real=True)) for g in (1, 2)]
    gamma = sp.Symbol("gamma", real=True)
    theta = sp.Matrix([*xi[0], *xi[1], gamma])
    penalty = sum(lambdas[g]*(xi[g].T*roughness*xi[g])[0]/2 for g in range(2))
    curvature = sp.diag(lambdas[0]*roughness, lambdas[1]*roughness, sp.zeros(1))
    proof.define(r"\mathcal P(\vartheta)", penalty)
    proof.define("K", curvature)
    expected_gradient = sp.Matrix([*(lambdas[0]*roughness*xi[0]),
                                   *(lambdas[1]*roughness*xi[1]), 0])
    proof.claim("Fixed roughness gradient",
                (sp.Matrix([sp.diff(penalty, v) for v in theta])
                 -expected_gradient).applyfunc(sp.expand),
                r"\nabla_\vartheta\mathcal P"
                r"-(\lambda_1J_B\xi_1,\lambda_2J_B\xi_2,0)^\top")
    proof.claim("Penalty Hessian blocks and zero global block",
                sp.hessian(penalty, theta)-curvature,
                r"\nabla_\vartheta^2\mathcal P"
                r"-\operatorname{diag}(\lambda_1J_B,\lambda_2J_B,0)")
    likelihood = sp.Function("L")(*theta)
    proof.claim("Penalized bread adds fixed curvature",
                sp.hessian(likelihood+penalty, theta)-sp.hessian(likelihood, theta)-curvature,
                r"\nabla_\vartheta^2(L+\mathcal P)-(H+K)")
    information, k = sp.symbols("I k", positive=True)
    inverse = 1/(information+k)
    proof.claim("Curvature inverse differs from sampling variance",
                sp.cancel(inverse-inverse*information*inverse-inverse*k*inverse),
                r"(I+K)^{-1}-(I+K)^{-1}I(I+K)^{-1}"
                r"-(I+K)^{-1}K(I+K)^{-1}")


def _schur_and_delta(proof):
    d1, d2, e = sp.symbols("d_1 d_2 e", positive=True)
    f1, f2, c1, c2 = sp.symbols("f_1 f_2 c_1 c_2", real=True)
    hessian = sp.Matrix([[d1, 0, f1], [0, d2, f2], [f1, f2, e]])
    schur = e-f1**2/d1-f2**2/d2
    inverse = hessian.inv()
    proof.define("H_{\\rm Schur}", hessian)
    proof.define(r"\mathcal S", schur)
    proof.claim("Shared global inverse is Schur inverse",
                sp.cancel(inverse[2, 2]-1/schur), r"(H^{-1})_{\gamma\gamma}-S^{-1}")
    proof.claim("Ignoring spatial coupling changes global variance",
                sp.cancel(inverse[2, 2]-1/e-(f1**2/d1+f2**2/d2)/(e*schur)),
                r"(H^{-1})_{\gamma\gamma}-e^{-1}"
                r"-(f_1^2/d_1+f_2^2/d_2)/(eS)")
    proof.claim("Cross-group spatial covariance",
                sp.cancel(inverse[0, 1]-f1*f2/(d1*d2*schur)),
                r"(H^{-1})_{12}-f_1f_2/(d_1d_2S)")
    contrast = sp.Matrix([c1, c2, 0])
    variance = c1**2/d1+c2**2/d2+(c1*f1/d1+c2*f2/d2)**2/schur
    proof.claim("Spatial contrast includes global uncertainty",
                sp.cancel((contrast.T*inverse*contrast)[0]-variance),
                r"c^\top H^{-1}c-c_1^2/d_1-c_2^2/d_2"
                r"-(c_1f_1/d_1+c_2f_2/d_2)^2/S")

    basis = sp.Matrix(2, 2, lambda j, p: sp.Symbol(f"B_{j}{p}", real=True))
    xi = sp.Matrix(sp.symbols("xi_1:3", real=True))
    mean = (basis*xi).applyfunc(sp.exp)
    jacobian = mean.jacobian(xi)
    expected = sp.diag(*mean)*basis
    proof.define(r"\mu_B", mean)
    proof.claim("Spatial exponential map exact Jacobian",
                (jacobian-expected).applyfunc(sp.simplify),
                r"D_\xi\mu_B-\operatorname{diag}(\mu_B)B")
    v11, v12, v22 = sp.symbols("V_11 V_12 V_22", real=True)
    covariance = sp.Matrix([[v11, v12], [v12, v22]])
    projected = sp.diag(*mean)*basis*covariance*basis.T*sp.diag(*mean)
    proof.claim("First-order delta covariance projection",
                (jacobian*covariance*jacobian.T-projected).applyfunc(sp.expand),
                r"D_\xi\mu_B\,V_\xi(D_\xi\mu_B)^\top"
                r"-\operatorname{diag}(\mu_B)BV_\xi B^\top\operatorname{diag}(\mu_B)")


def run() -> Proof:
    proof = Proof("published_cbmr", "Published CBMR integration", preamble=PREAMBLE)
    _grouped_design(proof)
    _poisson(proof)
    _independent_nb(proof)
    _clustered(proof)
    _penalty(proof)
    _schur_and_delta(proof)
    return proof


if __name__ == "__main__":
    result = run()
    result.write()
    raise SystemExit(0 if result.closed else 1)

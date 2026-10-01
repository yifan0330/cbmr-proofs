import lean_proof_checks.NegativeBinomial
import Mathlib.Analysis.Complex.CauchyIntegral
import Mathlib.Analysis.Calculus.ContDiff.Deriv

/-! Real Gamma smoothness and the logarithmic derivatives used by the matched
likelihood. Smoothness is proved from complex holomorphy, not postulated. -/
namespace CbmrProofs.AggregatedNB

open scoped Topology
open Filter
noncomputable section
set_option backward.isDefEq.respectTransparency false

theorem contDiffAt_gamma {x : ℝ} (hx : 0 < x) (n : WithTop ℕ∞) :
    ContDiffAt ℝ n Real.Gamma x := by
  have hc : AnalyticAt ℂ Complex.Gamma (x : ℂ) := by
    apply Complex.analyticAt_iff_eventually_differentiableAt.mpr
    have hp : ∀ᶠ z : ℂ in 𝓝 (x : ℂ), 0 < z.re :=
      (Complex.continuous_re.continuousAt.eventually (lt_mem_nhds hx))
    filter_upwards [hp] with z hz
    apply Complex.differentiableAt_Gamma z
    intro m hm
    have hre := congrArg Complex.re hm
    simp only [Complex.neg_re, Complex.natCast_re] at hre
    have : (0 : ℝ) ≤ m := Nat.cast_nonneg m
    linarith
  exact hc.contDiffAt.real_of_complex

def digamma (x : ℝ) : ℝ := deriv Real.Gamma x / Real.Gamma x

def trigamma (x : ℝ) : ℝ :=
  (deriv (deriv Real.Gamma) x * Real.Gamma x - (deriv Real.Gamma x) ^ 2) /
    Real.Gamma x ^ 2

theorem logGamma_hasDerivAt {x : ℝ} (hx : 0 < x) :
    HasDerivAt (fun x : ℝ => Real.log (Real.Gamma x)) (digamma x) x :=
  NegativeBinomial.hasDerivAt_log_gamma hx

theorem digamma_hasDerivAt {x : ℝ} (hx : 0 < x) :
    HasDerivAt digamma (trigamma x) x := by
  have hG := (contDiffAt_gamma hx 2).differentiableAt (by norm_num)
  have hG' := ((contDiffAt_gamma hx 2).derivWithin
    (m := 1) (by norm_num)).differentiableAt (by norm_num)
  convert! hG'.hasDerivAt.div hG.hasDerivAt (ne_of_gt (Real.Gamma_pos_of_pos hx)) using 1
  simp [trigamma, pow_two]

theorem digamma_eq_deriv_logGamma {x : ℝ} (hx : 0 < x) :
    digamma x = deriv (fun t : ℝ => Real.log (Real.Gamma t)) x :=
  (logGamma_hasDerivAt hx).deriv.symm

theorem trigamma_eq_deriv_digamma {x : ℝ} (hx : 0 < x) :
    trigamma x = deriv digamma x :=
  (digamma_hasDerivAt hx).deriv.symm

/-- The binary gamma-factor contribution, including `y log alpha`, vanishes. -/
theorem binary_gamma_log_cancellation {alpha y : ℝ} (ha : 0 < alpha)
    (hy : y = 0 ∨ y = 1) :
    Real.log (Real.Gamma (y + 1 / alpha)) - Real.log (Real.Gamma (1 / alpha)) -
      Real.log (Real.Gamma (y + 1)) + y * Real.log alpha = 0 := by
  rcases hy with rfl | rfl
  · simp [Real.Gamma_one]
  · have hi : 0 < 1 / alpha := one_div_pos.mpr ha
    rw [add_comm 1 (1 / alpha), Real.Gamma_add_one (ne_of_gt hi),
      Real.log_mul (ne_of_gt hi) (ne_of_gt (Real.Gamma_pos_of_pos hi))]
    simp [Real.Gamma_add_one, Real.Gamma_one]

theorem binary_nb_loglik_reduction {alpha y : ℝ} (eta : ℝ) (ha : 0 < alpha)
    (hy : y = 0 ∨ y = 1) :
    Real.log (Real.Gamma (y + 1 / alpha)) - Real.log (Real.Gamma (1 / alpha)) -
      Real.log (Real.Gamma (y + 1)) + y * Real.log alpha + y * eta -
      (y + 1 / alpha) * Real.log (1 + alpha * Real.exp eta) =
      y * eta - (y + 1 / alpha) * Real.log (1 + alpha * Real.exp eta) := by
  rw [binary_gamma_log_cancellation ha hy, zero_add]

end
end CbmrProofs.AggregatedNB

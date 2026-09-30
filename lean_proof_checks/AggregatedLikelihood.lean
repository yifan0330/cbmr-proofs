import lean_proof_checks.AggregatedModel
import lean_proof_checks.Calculus
import lean_proof_checks.ProbabilityMixture

/-! Identification of the differentiated expressions with the normalized NB
mass constructed from the Gamma--Poisson mixture. The omitted aggregated term
depends only on the observed natural-number counts. -/
namespace CbmrProofs.AggregatedNB

open scoped BigOperators
open NegativeBinomial ProbabilityMixture
noncomputable section
set_option backward.isDefEq.respectTransparency false

theorem log_nb_mass (r m : ℝ) (y : ℕ) (hr : 0 < r) (hm : 0 < m) :
    Real.log (nbTotalMass r m y) =
      Real.log (Real.Gamma (r + y)) - Real.log (Real.Gamma r) -
        Real.log (Real.Gamma ((y : ℝ) + 1)) +
        r * Real.log r + (y : ℝ) * Real.log m - (r + y) * Real.log (r + m) := by
  unfold nbTotalMass
  rw [Real.log_mul (by positivity) (by positivity),
    Real.log_mul (by positivity) (by positivity),
    Real.log_div (by positivity) (by positivity),
    Real.log_mul (by positivity) (by positivity),
    Real.log_rpow (by positivity), Real.log_pow,
    Real.log_div (ne_of_gt hr) (ne_of_gt (add_pos hr hm)),
    Real.log_div (ne_of_gt hm) (ne_of_gt (add_pos hr hm)),
    Real.Gamma_nat_eq_factorial]
  ring

def independentNBConstant (alpha y : ℝ) : ℝ :=
  Real.log (Real.Gamma (y + 1 / alpha)) - Real.log (Real.Gamma (1 / alpha)) -
    Real.log (Real.Gamma (y + 1)) + y * Real.log alpha

def fullIndependentNBLogLik (alpha y eta : ℝ) : ℝ :=
  Real.log (Real.Gamma (y + 1 / alpha)) - Real.log (Real.Gamma (1 / alpha)) -
    Real.log (Real.Gamma (y + 1)) + y * Real.log alpha + y * eta -
    (y + 1 / alpha) * Real.log (1 + alpha * Real.exp eta)

theorem independent_nb_constant_identification (alpha y eta : ℝ) :
    fullIndependentNBLogLik alpha y eta =
      independentNBConstant alpha y -
        (-y * eta + (y + 1 / alpha) * Real.log (1 + alpha * Real.exp eta)) := by
  unfold fullIndependentNBLogLik independentNBConstant
  ring

theorem independent_nb_full_loglik_hasDerivAt (alpha y eta : ℝ) (ha : 0 < alpha) :
    HasDerivAt (fullIndependentNBLogLik alpha y)
      ((y - Real.exp eta) / (1 + alpha * Real.exp eta)) eta := by
  have h := (Calculus.nb_nll_hasDerivAt alpha y eta ha).const_sub
    (independentNBConstant alpha y)
  convert! h using 1
  · funext t
    exact independent_nb_constant_identification alpha y t
  · ring

theorem independent_nb_log_mass (alpha eta : ℝ) (y : ℕ) (ha : 0 < alpha) :
    Real.log (nbTotalMass (1 / alpha) (Real.exp eta) y) =
      fullIndependentNBLogLik alpha y eta := by
  have hd : 0 < 1 + alpha * Real.exp eta := by positivity
  have he : 1 / alpha + Real.exp eta = (1 + alpha * Real.exp eta) / alpha := by
    field_simp
  rw [log_nb_mass _ _ y (by positivity) (Real.exp_pos _), he,
    Real.log_div (ne_of_gt hd) (ne_of_gt ha)]
  simp only [Real.log_exp, one_div, Real.log_inv]
  unfold fullIndependentNBLogLik
  simp only [one_div]
  rw [add_comm alpha⁻¹ (y : ℝ)]
  ring

theorem aggregated_cell_log_mass (r A eta : ℝ) (y : ℕ) (hr : 0 < r) (hA : 0 < A) :
    -Real.log (nbTotalMass r (r * Real.exp eta / A) y) =
      -Real.log (Real.Gamma ((y : ℝ) + r)) + Real.log (Real.Gamma r) -
        r * Real.log A + (r + y) * Real.log (Real.exp eta + A) - (y : ℝ) * eta +
        Real.log (Real.Gamma ((y : ℝ) + 1)) := by
  have he : r + r * Real.exp eta / A = r * (Real.exp eta + A) / A := by
    field_simp
    ring
  have hd : 0 < Real.exp eta + A := by positivity
  rw [log_nb_mass r _ y hr (by positivity), he,
    Real.log_div (ne_of_gt (mul_pos hr hd)) (ne_of_gt hA),
    Real.log_mul (ne_of_gt hr) (ne_of_gt hd),
    Real.log_div (ne_of_gt (mul_pos hr (Real.exp_pos eta))) (ne_of_gt hA),
    Real.log_mul (ne_of_gt hr) (ne_of_gt (Real.exp_pos eta)), Real.log_exp,
    add_comm r (y : ℝ)]
  ring

variable {ι : Type*} [Fintype ι]

theorem aggregatedPsi_neg_log_mass (r A : ℝ) (y : ι → ℕ) (S : ι → ℝ)
    (hr : 0 < r) (hA : 0 < A) :
    -(∑ i, Real.log (nbTotalMass r (r * Real.exp (S i) / A) (y i))) =
      aggregatedPsi (fun i => (y i : ℝ)) r A S +
        ∑ i, Real.log (Real.Gamma ((y i : ℝ) + 1)) := by
  rw [← Finset.sum_neg_distrib]
  simp_rw [aggregated_cell_log_mass r A _ _ hr hA]
  simp only [aggregatedPsi, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    Finset.sum_neg_distrib, Finset.sum_const, nsmul_eq_mul, Finset.card_univ]
  ring

/-- The independent product of normalized NB masses gives exactly Psi plus
the sum of log-factorials; this term is independent of every fitted parameter. -/
theorem aggregatedPsi_neg_log_product_mass (r A : ℝ) (y : ι → ℕ) (S : ι → ℝ)
    (hr : 0 < r) (hA : 0 < A) :
    -Real.log (∏ i, nbTotalMass r (r * Real.exp (S i) / A) (y i)) =
      aggregatedPsi (fun i => (y i : ℝ)) r A S +
        ∑ i, Real.log (Real.Gamma ((y i : ℝ) + 1)) := by
  rw [Real.log_prod (fun i _ => ne_of_gt (nb_mass_pos r _ hr (by positivity) (y i)))]
  exact aggregatedPsi_neg_log_mass r A y S hr hA

theorem rateA_eq_rho_div_s1 (alpha s1 s2 : ℝ) (hs : s1 ≠ 0) :
    rateA alpha s1 s2 = rho alpha s1 s2 / s1 := by
  rw [rho_eq_s1_mul_rateA]
  field_simp

theorem aggregated_mean_parameter (alpha s1 s2 mu : ℝ)
    (ha : 0 < alpha) (h1 : 0 < s1) (h2 : 0 < s2) :
    rho alpha s1 s2 * mu / rateA alpha s1 s2 = mu * s1 := by
  have hA := ne_of_gt (parameter_pos ha h1 h2).2.1
  rw [rho_eq_s1_mul_rateA]
  field_simp

theorem moment_matchedPsi_neg_log_mass (alpha s1 s2 : ℝ) (y : ι → ℕ) (S : ι → ℝ)
    (ha : 0 < alpha) (h1 : 0 < s1) (h2 : 0 < s2) :
    -Real.log (∏ i, nbTotalMass (rho alpha s1 s2) (Real.exp (S i) * s1) (y i)) =
      aggregatedPsi (fun i => (y i : ℝ)) (rho alpha s1 s2) (rateA alpha s1 s2) S +
        ∑ i, Real.log (Real.Gamma ((y i : ℝ) + 1)) := by
  have h := aggregatedPsi_neg_log_product_mass (rho alpha s1 s2) (rateA alpha s1 s2) y S
    (parameter_pos ha h1 h2).1 (parameter_pos ha h1 h2).2.1
  simpa only [aggregated_mean_parameter alpha s1 s2 _ ha h1 h2] using h

section FixedEffectiveDispersion

variable {κ : Type*} [Fintype κ] [Nonempty ι]

/-- If effective dispersion (hence shape) is held fixed, the rate derivative
is different from `moderatorA_hasDerivAt`, which holds original dispersion fixed. -/
theorem fixedShapeRate_hasDerivAt (r : ℝ) (Z : ι → κ → ℝ)
    (gamma v : κ → ℝ) (x : ℝ) :
    HasDerivAt (fun t => r / moment 1 Z (shift gamma v t))
      (-(r / moment 1 Z (shift gamma v x)) * momentLogGradient 1 Z (shift gamma v x) v) x := by
  have hd := ne_of_gt (moment_pos 1 Z (shift gamma v x))
  convert! const_div_hasDerivAt (moment_hasDerivAt 1 Z gamma v x) r hd using 1
  unfold momentLogGradient
  field_simp

theorem fixedShapeRate_gradient_hasDerivAt (r : ℝ) (Z : ι → κ → ℝ)
    (gamma v w : κ → ℝ) (x : ℝ) :
    HasDerivAt (fun t => -(r / moment 1 Z (shift gamma w t)) *
      momentLogGradient 1 Z (shift gamma w t) v)
      ((r / moment 1 Z (shift gamma w x)) *
        (momentLogGradient 1 Z (shift gamma w x) v * momentLogGradient 1 Z (shift gamma w x) w -
          momentLogHessian 1 Z (shift gamma w x) v w)) x := by
  convert! ((fixedShapeRate_hasDerivAt r Z gamma w x).neg).mul
    (momentLogGradient_hasDerivAt 1 Z gamma v w x) using 1
  simp only [Pi.neg_apply]
  ring

end FixedEffectiveDispersion

end
end CbmrProofs.AggregatedNB

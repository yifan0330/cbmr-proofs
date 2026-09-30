import Mathlib.Analysis.SpecialFunctions.Gamma.Deriv
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Aggregated negative-binomial and clustered curvature calculations

The moment and cumulant results below are identities for the displayed scalar
expressions, not assertions about the law of a sum of random variables.
The quadratic-form results apply to an arbitrary finite index type.
-/

namespace CbmrProofs.NegativeBinomial

open scoped BigOperators

noncomputable section

def rho (alpha s1 s2 : ℝ) : ℝ := s1 ^ 2 / (alpha * s2)
def rateA (alpha s1 s2 : ℝ) : ℝ := s1 / (alpha * s2)
def prob (alpha s1 s2 mu : ℝ) : ℝ := mu / (mu + rateA alpha s1 s2)
def alphaEff (alpha s1 s2 : ℝ) : ℝ := alpha * s2 / s1 ^ 2

theorem parameter_pos {alpha s1 s2 : ℝ}
    (ha : 0 < alpha) (h1 : 0 < s1) (h2 : 0 < s2) :
    0 < rho alpha s1 s2 ∧ 0 < rateA alpha s1 s2 ∧ 0 < alphaEff alpha s1 s2 := by
  dsimp [rho, rateA, alphaEff]
  exact ⟨div_pos (sq_pos_of_pos h1) (mul_pos ha h2),
    div_pos h1 (mul_pos ha h2), div_pos (mul_pos ha h2) (sq_pos_of_pos h1)⟩

theorem prob_bounds {alpha s1 s2 mu : ℝ}
    (ha : 0 < alpha) (h1 : 0 < s1) (h2 : 0 < s2) (hm : 0 < mu) :
    0 < prob alpha s1 s2 mu ∧ prob alpha s1 s2 mu < 1 := by
  have hA := (parameter_pos ha h1 h2).2.1
  exact ⟨div_pos hm (add_pos hm hA), (div_lt_one (add_pos hm hA)).2 (by linarith)⟩

theorem denominator_guards {alpha s1 s2 mu : ℝ}
    (ha : 0 < alpha) (h1 : 0 < s1) (h2 : 0 < s2) (hm : 0 < mu) :
    alpha * s2 ≠ 0 ∧ mu + rateA alpha s1 s2 ≠ 0 ∧
      1 - prob alpha s1 s2 mu ≠ 0 := by
  exact ⟨ne_of_gt (mul_pos ha h2),
    ne_of_gt (add_pos hm (parameter_pos ha h1 h2).2.1),
    ne_of_gt (sub_pos.mpr (prob_bounds ha h1 h2 hm).2)⟩

theorem rho_eq_s1_mul_rateA (alpha s1 s2 : ℝ) :
    rho alpha s1 s2 = s1 * rateA alpha s1 s2 := by
  unfold rho rateA
  ring

theorem reciprocal_rho (alpha s1 s2 : ℝ) :
    1 / rho alpha s1 s2 = alphaEff alpha s1 s2 := by
  simp [rho, alphaEff, one_div, inv_div]

theorem prob_reparameterization {alpha s1 s2 mu : ℝ}
    (ha : 0 < alpha) (h1 : 0 < s1) (h2 : 0 < s2) (hm : 0 < mu) :
    prob alpha s1 s2 mu = alpha * mu * s2 / (s1 + alpha * mu * s2) := by
  have hA := (parameter_pos ha h1 h2).2.1
  have hd : 0 < s1 + alpha * mu * s2 := by positivity
  unfold prob rateA at *
  field_simp
  ring

theorem complement_reparameterization {alpha s1 s2 mu : ℝ}
    (ha : 0 < alpha) (h1 : 0 < s1) (h2 : 0 < s2) (hm : 0 < mu) :
    1 - prob alpha s1 s2 mu = s1 / (s1 + alpha * mu * s2) := by
  rw [prob_reparameterization ha h1 h2 hm]
  have hd : s1 + alpha * mu * s2 ≠ 0 := ne_of_gt (by positivity)
  field_simp
  ring

theorem mean_expression {alpha s1 s2 mu : ℝ}
    (ha : 0 < alpha) (h1 : 0 < s1) (h2 : 0 < s2) (hm : 0 < mu) :
    rho alpha s1 s2 * prob alpha s1 s2 mu / (1 - prob alpha s1 s2 mu) =
      mu * s1 := by
  rw [prob_reparameterization ha h1 h2 hm,
    show 1 - alpha * mu * s2 / (s1 + alpha * mu * s2) =
      s1 / (s1 + alpha * mu * s2) from by
        rw [← prob_reparameterization ha h1 h2 hm]
        exact complement_reparameterization ha h1 h2 hm]
  unfold rho
  have hd : s1 + alpha * mu * s2 ≠ 0 := ne_of_gt (by positivity)
  field_simp

theorem variance_expression {alpha s1 s2 mu : ℝ}
    (ha : 0 < alpha) (h1 : 0 < s1) (h2 : 0 < s2) (hm : 0 < mu) :
    rho alpha s1 s2 * prob alpha s1 s2 mu / (1 - prob alpha s1 s2 mu) ^ 2 =
      mu * s1 + alpha * mu ^ 2 * s2 := by
  rw [complement_reparameterization ha h1 h2 hm,
    prob_reparameterization ha h1 h2 hm]
  unfold rho
  have hd : s1 + alpha * mu * s2 ≠ 0 := ne_of_gt (by positivity)
  field_simp

theorem moment_matching_prob {alpha s1 s2 mu : ℝ}
    (ha : 0 < alpha) (h1 : 0 < s1) (h2 : 0 < s2) (hm : 0 < mu) :
    prob alpha s1 s2 mu =
      ((mu * s1 + alpha * mu ^ 2 * s2) - mu * s1) /
        (mu * s1 + alpha * mu ^ 2 * s2) := by
  rw [prob_reparameterization ha h1 h2 hm]
  have hd : s1 + alpha * mu * s2 ≠ 0 := ne_of_gt (by positivity)
  have hv : mu * s1 + alpha * mu ^ 2 * s2 ≠ 0 := ne_of_gt (by positivity)
  field_simp
  ring

theorem two_factor_moment_gap {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    (a ^ 3 + b ^ 3) - (a ^ 2 + b ^ 2) ^ 2 / (a + b) =
      a * b * (a - b) ^ 2 / (a + b) := by
  have hab : a + b ≠ 0 := ne_of_gt (add_pos ha hb)
  field_simp
  ring

theorem two_factor_moment_gap_nonneg {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    0 ≤ (a ^ 3 + b ^ 3) - (a ^ 2 + b ^ 2) ^ 2 / (a + b) := by
  rw [two_factor_moment_gap ha hb]
  positivity

/-- The usual third-cumulant expression, treated here as an algebraic function. -/
def thirdCumulantExpression (mean shape : ℝ) : ℝ :=
  mean + 3 * mean ^ 2 / shape + 2 * mean ^ 3 / shape ^ 2

theorem two_factor_third_cumulant_gap {alpha mu a b : ℝ}
    (_halpha : 0 < alpha) (ha : 0 < a) (hb : 0 < b) :
    thirdCumulantExpression (mu * a) (1 / alpha) +
      thirdCumulantExpression (mu * b) (1 / alpha) -
      thirdCumulantExpression (mu * (a + b)) (rho alpha (a + b) (a ^ 2 + b ^ 2)) =
      2 * alpha ^ 2 * mu ^ 3 *
        ((a ^ 3 + b ^ 3) - (a ^ 2 + b ^ 2) ^ 2 / (a + b)) := by
  have hab : a + b ≠ 0 := ne_of_gt (add_pos ha hb)
  have hab2 : a ^ 2 + b ^ 2 ≠ 0 := ne_of_gt (by positivity)
  unfold thirdCumulantExpression rho
  field_simp
  ring

theorem two_factor_third_cumulant_gap_nonneg {alpha mu a b : ℝ}
    (halpha : 0 < alpha) (hmu : 0 ≤ mu) (ha : 0 < a) (hb : 0 < b) :
    0 ≤ thirdCumulantExpression (mu * a) (1 / alpha) +
      thirdCumulantExpression (mu * b) (1 / alpha) -
      thirdCumulantExpression (mu * (a + b)) (rho alpha (a + b) (a ^ 2 + b ^ 2)) := by
  rw [two_factor_third_cumulant_gap halpha ha hb]
  exact mul_nonneg (by positivity) (two_factor_moment_gap_nonneg ha hb)

def weightEntry (nu t muJ muK diagonal : ℝ) : ℝ :=
  diagonal * muJ - muJ * muK / (nu + t)

def observedEntry (nu t y muJ muK diagonal : ℝ) : ℝ :=
  (y + nu) / (nu + t) * weightEntry nu t muJ muK diagonal

/-- Set `diagonal` to the Kronecker delta for a matrix entry. -/
theorem observed_curvature_affine (nu t y muJ muK diagonal : ℝ) :
    observedEntry nu t y muJ muK diagonal =
      y * (weightEntry nu t muJ muK diagonal / (nu + t)) +
      nu * (weightEntry nu t muJ muK diagonal / (nu + t)) := by
  unfold observedEntry
  ring

/-- Algebraic substitution of the mean count, not a probabilistic expectation theorem. -/
theorem observed_curvature_at_mean {nu t : ℝ} (h : nu + t ≠ 0)
    (muJ muK diagonal : ℝ) :
    observedEntry nu t t muJ muK diagonal = weightEntry nu t muJ muK diagonal := by
  simp [observedEntry, add_comm t nu, h]

section FiniteQuadratic

variable {ι : Type*} [Fintype ι]

def weightedQuadratic (nu : ℝ) (mu v : ι → ℝ) : ℝ :=
  (∑ j, mu j * v j ^ 2) - (∑ j, mu j * v j) ^ 2 / (nu + ∑ j, mu j)

theorem pairwise_square_sum (mu v : ι → ℝ) :
    (∑ j, ∑ k, mu j * mu k * (v j - v k) ^ 2) =
      2 * (∑ j, mu j) * (∑ j, mu j * v j ^ 2) - 2 * (∑ j, mu j * v j) ^ 2 := by
  calc
    _ = ∑ j, ∑ k, ((mu j * v j ^ 2) * mu k +
        mu j * (mu k * v k ^ 2) - 2 * (mu j * v j) * (mu k * v k)) := by
      apply Finset.sum_congr rfl
      intro j _
      apply Finset.sum_congr rfl
      intro k _
      ring
    _ = _ := by
      simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib,
        ← Finset.mul_sum, ← Finset.sum_mul]
      ring

theorem weighted_quadratic_identity (nu : ℝ) (mu v : ι → ℝ)
    (hd : nu + ∑ j, mu j ≠ 0) :
    (nu + ∑ j, mu j) * weightedQuadratic nu mu v =
      nu * (∑ j, mu j * v j ^ 2) +
        (1 / 2 : ℝ) * ∑ j, ∑ k, mu j * mu k * (v j - v k) ^ 2 := by
  rw [pairwise_square_sum]
  unfold weightedQuadratic
  field_simp
  ring

theorem weighted_quadratic_nonneg {nu : ℝ} (mu v : ι → ℝ)
    (hnu : 0 < nu) (hmu : ∀ j, 0 ≤ mu j) :
    0 ≤ weightedQuadratic nu mu v := by
  have hsum : 0 ≤ ∑ j, mu j := Finset.sum_nonneg (fun j _ => hmu j)
  have hd : 0 < nu + ∑ j, mu j := add_pos_of_pos_of_nonneg hnu hsum
  have hq : 0 ≤ ∑ j, mu j * v j ^ 2 :=
    Finset.sum_nonneg (fun j _ => mul_nonneg (hmu j) (sq_nonneg _))
  have hp : 0 ≤ ∑ j, ∑ k, mu j * mu k * (v j - v k) ^ 2 :=
    Finset.sum_nonneg (fun j _ => Finset.sum_nonneg (fun k _ =>
      mul_nonneg (mul_nonneg (hmu j) (hmu k)) (sq_nonneg _)))
  have hh := weighted_quadratic_identity nu mu v (ne_of_gt hd)
  have : 0 ≤ (nu + ∑ j, mu j) * weightedQuadratic nu mu v := by
    rw [hh]
    positivity
  exact nonneg_of_mul_nonneg_right this hd

end FiniteQuadratic

/-- Real log-Gamma is differentiable at every positive argument. The derivative is
expressed through Mathlib's derivative of `Real.Gamma`; no trigamma formula is claimed. -/
theorem hasDerivAt_log_gamma {x : ℝ} (hx : 0 < x) :
    HasDerivAt (fun z : ℝ => Real.log (Real.Gamma z))
      (deriv Real.Gamma x / Real.Gamma x) x := by
  have hg : DifferentiableAt ℝ Real.Gamma x :=
    Real.differentiableAt_Gamma (fun m => by
      have hm : (0 : ℝ) ≤ m := Nat.cast_nonneg m
      linarith)
  exact hg.hasDerivAt.log (ne_of_gt (Real.Gamma_pos_of_pos hx))

/-- The aggregated negative log-likelihood expression, up to parameter-independent terms. -/
def aggregatedPsi {ι : Type*} [Fintype ι] (y : ι → ℝ) (shape A : ℝ)
    (S : ι → ℝ) : ℝ :=
  -(∑ i, Real.log (Real.Gamma (y i + shape))) +
    (Fintype.card ι : ℝ) * Real.log (Real.Gamma shape) -
    (Fintype.card ι : ℝ) * shape * Real.log A +
    (∑ i, (shape + y i) * Real.log (Real.exp (S i) + A)) -
    (∑ i, y i * S i)

/-- Directional differentiation of the full vector of log intensities.
The shape, rate, and Gamma terms are fixed along the direction. -/
theorem hasDerivAt_aggregatedPsi_log_intensity {ι : Type*} [Fintype ι]
    (y : ι → ℝ) (shape A : ℝ) (S v : ι → ℝ) (hA : 0 < A) :
    HasDerivAt (fun t => aggregatedPsi y shape A (fun i => S i + t * v i))
      (∑ i, ((shape + y i) * Real.exp (S i) / (Real.exp (S i) + A) - y i) * v i)
      0 := by
  have hpath : ∀ i, HasDerivAt (fun t : ℝ => S i + t * v i) (v i) 0 := by
    intro i
    convert! ((hasDerivAt_id (0 : ℝ)).mul_const (v i)).const_add (S i) using 1
    simp
  have hl : ∀ i, HasDerivAt (fun t : ℝ => Real.log (Real.exp (S i + t * v i) + A))
      (Real.exp (S i) * v i / (Real.exp (S i) + A)) 0 := by
    intro i
    convert! ((hpath i).exp.add_const A).log (ne_of_gt (by positivity)) using 1
    simp
  have hs := HasDerivAt.fun_sum (u := Finset.univ)
    (fun i _ => (hl i).const_mul (shape + y i))
  have hy := HasDerivAt.fun_sum (u := Finset.univ)
    (fun i _ => (hpath i).const_mul (y i))
  convert! ((hs.const_add
    (-(∑ i, Real.log (Real.Gamma (y i + shape))) +
      (Fintype.card ι : ℝ) * Real.log (Real.Gamma shape) -
      (Fintype.card ι : ℝ) * shape * Real.log A)).sub hy) using 1
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- Actual shape differentiation. Gamma logarithmic derivatives are left as
`deriv Real.Gamma / Real.Gamma`, rather than claiming a trigamma closed form. -/
theorem hasDerivAt_aggregatedPsi_shape {ι : Type*} [Fintype ι]
    (y : ι → ℝ) (shape A : ℝ) (S : ι → ℝ) (hr : 0 < shape) (hy : ∀ i, 0 ≤ y i) :
    HasDerivAt (fun r => aggregatedPsi y r A S)
      (-(∑ i, deriv Real.Gamma (y i + shape) / Real.Gamma (y i + shape)) +
        (Fintype.card ι : ℝ) * (deriv Real.Gamma shape / Real.Gamma shape) -
        (Fintype.card ι : ℝ) * Real.log A +
        ∑ i : ι, Real.log (Real.exp (S i) + A)) shape := by
  have hg : ∀ i, HasDerivAt (fun r : ℝ => Real.log (Real.Gamma (y i + r)))
      (deriv Real.Gamma (y i + shape) / Real.Gamma (y i + shape)) shape := by
    intro i
    convert! (hasDerivAt_log_gamma (add_pos_of_nonneg_of_pos (hy i) hr)).comp
      shape ((hasDerivAt_id shape).const_add (y i)) using 1
    simp
  have hsum := HasDerivAt.fun_sum (u := Finset.univ) (fun i _ => hg i)
  have hshape := (hasDerivAt_log_gamma hr).const_mul (Fintype.card ι : ℝ)
  have hlinear := ((hasDerivAt_id shape).const_mul (Fintype.card ι : ℝ)).mul_const
    (Real.log A)
  have hlogs := HasDerivAt.fun_sum (u := Finset.univ)
    (fun i _ => ((hasDerivAt_id shape).add_const (y i)).mul_const
      (Real.log (Real.exp (S i) + A)))
  convert! (((hsum.neg.add hshape).sub hlinear).add hlogs).sub_const
    (∑ i, y i * S i) using 1
  simp

end

end CbmrProofs.NegativeBinomial

import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Convex.Deriv
import Mathlib.Tactic.Convert
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity

/-!
# Scalar calculus for the CBMR appendix

These results differentiate with respect to a scalar covariate or log mean.
Dispersion and the observed count are fixed. The NB mean substitution below is
an algebraic substitution, not a probabilistic expectation theorem.
-/

namespace CbmrProofs.Calculus

set_option backward.isDefEq.respectTransparency false

open Real

theorem gc_predictor_hasDerivAt (f gamma x : ℝ) :
    HasDerivAt (fun t => f + t * gamma) gamma x := by
  simpa using ((hasDerivAt_id x).mul_const gamma).const_add f

theorem sv_predictor_hasDerivAt (f b x : ℝ) :
    HasDerivAt (fun t => f + t * b) b x :=
  gc_predictor_hasDerivAt f b x

/-- The same identity applies to a global coefficient or a voxelwise coefficient. -/
theorem affine_mean_ratio (f b x : ℝ) :
    exp (f + (x + 1) * b) / exp (f + x * b) = exp b := by
  rw [show f + (x + 1) * b = (f + x * b) + b by ring, exp_add]
  field_simp [exp_ne_zero]

theorem affine_mean_hasDerivAt (f b x : ℝ) :
    HasDerivAt (fun t => exp (f + t * b)) (exp (f + x * b) * b) x :=
  (sv_predictor_hasDerivAt f b x).exp

theorem affine_chain_hasDerivAt (f : ℝ → ℝ) (eta slope value : ℝ)
    (hf : HasDerivAt f value eta) :
    HasDerivAt (fun t => f (eta + t * slope)) (slope * value) 0 := by
  have h := (show HasDerivAt f value (eta + 0 * slope) by simpa using hf).comp
    0 (gc_predictor_hasDerivAt eta slope 0)
  simpa only [Function.comp_def, zero_mul, add_zero, mul_comm] using h

/-- The first-coordinate derivative is identified at every second-coordinate
offset; its derivative at zero is the mixed affine-chain curvature. -/
theorem affine_chain_mixed_hasDerivAt (f df : ℝ → ℝ) (eta a b curvature : ℝ)
    (hf : ∀ x, HasDerivAt f (df x) x)
    (hdf : HasDerivAt df curvature eta) :
    (∀ t, HasDerivAt (fun s => f (eta + s * a + t * b))
      (a * df (eta + t * b)) 0) ∧
    HasDerivAt (fun t => a * df (eta + t * b)) (a * b * curvature) 0 := by
  constructor
  · intro t
    convert! affine_chain_hasDerivAt f (eta + t * b) a (df (eta + t * b))
      (hf (eta + t * b)) using 1
    funext s
    congr 1
    ring
  · convert! (affine_chain_hasDerivAt df eta b curvature hdf).const_mul a using 1 <;>
      ring

theorem poisson_nll_hasDerivAt (y eta : ℝ) :
    HasDerivAt (fun t => exp t - y * t) (exp eta - y) eta := by
  convert! (hasDerivAt_exp eta).sub ((hasDerivAt_id eta).const_mul y) using 1 <;>
    simp

theorem poisson_score_hasDerivAt (y eta : ℝ) :
    HasDerivAt (fun t => y - exp t) (-exp eta) eta :=
  (hasDerivAt_exp eta).const_sub y

theorem poisson_nll_second_hasDerivAt (y eta : ℝ) :
    HasDerivAt (fun t => exp t - y) (exp eta) eta := by
  simpa using (hasDerivAt_exp eta).sub_const y

/-- Any log-normalising term independent of eta can be included in `c`. -/
theorem poisson_loglik_hasDerivAt (y c eta : ℝ) :
    HasDerivAt (fun t => y * t - exp t + c) (y - exp eta) eta := by
  simpa using
    (((hasDerivAt_id eta).const_mul y).sub (hasDerivAt_exp eta)).add_const c

private theorem nb_denominator_pos (alpha eta : ℝ) (ha : 0 < alpha) :
    0 < 1 + alpha * exp eta := by
  positivity

theorem nb_nll_hasDerivAt (alpha y eta : ℝ) (ha : 0 < alpha) :
    HasDerivAt
      (fun t => -y * t + (y + 1 / alpha) * log (1 + alpha * exp t))
      ((exp eta - y) / (1 + alpha * exp eta)) eta := by
  have hd : 1 + alpha * exp eta ≠ 0 :=
    ne_of_gt (nb_denominator_pos alpha eta ha)
  have hlog := (((hasDerivAt_exp eta).const_mul alpha).const_add 1).log hd
  have h := ((hasDerivAt_id eta).const_mul (-y)).add
    (hlog.const_mul (y + 1 / alpha))
  convert! h using 1 <;> (try dsimp) <;> field_simp [ne_of_gt ha, hd] <;> ring

theorem nb_score_hasDerivAt (alpha y eta : ℝ) (ha : 0 < alpha) :
    HasDerivAt (fun t => (y - exp t) / (1 + alpha * exp t))
      (-exp eta * (1 + alpha * y) / (1 + alpha * exp eta) ^ 2) eta := by
  have hd : 1 + alpha * exp eta ≠ 0 :=
    ne_of_gt (nb_denominator_pos alpha eta ha)
  have h := ((hasDerivAt_exp eta).const_sub y).div
    (((hasDerivAt_exp eta).const_mul alpha).const_add 1) hd
  convert! h using 1 <;> (try dsimp) <;> ring

theorem nb_nll_second_hasDerivAt (alpha y eta : ℝ) (ha : 0 < alpha) :
    HasDerivAt (fun t => (exp t - y) / (1 + alpha * exp t))
      (exp eta * (1 + alpha * y) / (1 + alpha * exp eta) ^ 2) eta := by
  have h := (nb_score_hasDerivAt alpha y eta ha).neg
  simpa only [Pi.neg_def, neg_div', neg_sub, neg_mul_eq_neg_mul, neg_neg] using h

theorem nb_observed_weight_pos (alpha y eta : ℝ)
    (ha : 0 < alpha) (hy : 0 ≤ y) :
    0 < exp eta * (1 + alpha * y) / (1 + alpha * exp eta) ^ 2 := by
  have hd := nb_denominator_pos alpha eta ha
  positivity

/-- Substituting the mean for the count; no expectation operator is asserted. -/
theorem nb_observed_weight_mean_substitution (alpha mu : ℝ)
    (ha : 0 < alpha) (hmu : 0 < mu) :
    mu * (1 + alpha * mu) / (1 + alpha * mu) ^ 2 =
      mu / (1 + alpha * mu) := by
  have hd : 1 + alpha * mu ≠ 0 := by positivity
  field_simp [hd]

/-- Fisher weight times the IRLS working residual is the NB likelihood score. -/
theorem nb_irls_weight_residual (alpha mu y eta : ℝ)
    (ha : 0 < alpha) (hmu : 0 < mu) :
    (mu / (1 + alpha * mu)) * ((eta + (y - mu) / mu) - eta) =
      (y - mu) / (1 + alpha * mu) := by
  have hd : 1 + alpha * mu ≠ 0 := by positivity
  field_simp [hd, ne_of_gt hmu]
  ring

theorem quadratic_penalty_hasDerivAt (k x : ℝ) :
    HasDerivAt (fun t => (1 / 2 : ℝ) * k * t ^ 2) (k * x) x := by
  convert! ((hasDerivAt_id x).pow 2).const_mul ((1 / 2 : ℝ) * k) using 1 <;>
    (try dsimp) <;> ring

theorem quadratic_penalty_second_hasDerivAt (k x : ℝ) :
    HasDerivAt (fun t => k * t) k x := by
  simpa using (hasDerivAt_id x).const_mul k

theorem poisson_nll_convex (y : ℝ) :
    ConvexOn ℝ Set.univ (fun eta => exp eta - y * eta) := by
  apply convexOn_of_hasDerivWithinAt2_nonneg
    (f' := fun eta => exp eta - y) (f'' := exp) convex_univ
  · intro x _
    exact (poisson_nll_hasDerivAt y x).continuousAt.continuousWithinAt
  · intro x _
    exact (poisson_nll_hasDerivAt y x).hasDerivWithinAt
  · intro x _
    exact (poisson_nll_second_hasDerivAt y x).hasDerivWithinAt
  · intro x _
    exact le_of_lt (exp_pos x)

theorem nb_nll_convex (alpha y : ℝ) (ha : 0 < alpha) (hy : 0 ≤ y) :
    ConvexOn ℝ Set.univ
      (fun eta => -y * eta + (y + 1 / alpha) * log (1 + alpha * exp eta)) := by
  apply convexOn_of_hasDerivWithinAt2_nonneg
    (f' := fun eta => (exp eta - y) / (1 + alpha * exp eta))
    (f'' := fun eta => exp eta * (1 + alpha * y) / (1 + alpha * exp eta) ^ 2)
    convex_univ
  · intro x _
    exact (nb_nll_hasDerivAt alpha y x ha).continuousAt.continuousWithinAt
  · intro x _
    exact (nb_nll_hasDerivAt alpha y x ha).hasDerivWithinAt
  · intro x _
    exact (nb_nll_second_hasDerivAt alpha y x ha).hasDerivWithinAt
  · intro x _
    exact le_of_lt (nb_observed_weight_pos alpha y x ha hy)

/-- Two-sided continuity implies, in particular, the positive-dispersion Poisson limit. -/
theorem nb_score_poisson_limit (y mu : ℝ) :
    Filter.Tendsto (fun alpha : ℝ => (y - mu) / (1 + alpha * mu))
      (nhds 0) (nhds (y - mu)) := by
  have h : ContinuousAt (fun alpha : ℝ => (y - mu) / (1 + alpha * mu)) 0 :=
    continuousAt_const.div (continuousAt_const.add
      (continuousAt_id.mul continuousAt_const)) (by simp)
  simpa using h.tendsto

theorem nb_observed_weight_poisson_limit (y mu : ℝ) :
    Filter.Tendsto (fun alpha : ℝ => mu * (1 + alpha * y) / (1 + alpha * mu) ^ 2)
      (nhds 0) (nhds mu) := by
  have h : ContinuousAt
      (fun alpha : ℝ => mu * (1 + alpha * y) / (1 + alpha * mu) ^ 2) 0 :=
    (continuousAt_const.mul (continuousAt_const.add
      (continuousAt_id.mul continuousAt_const))).div
        ((continuousAt_const.add (continuousAt_id.mul continuousAt_const)).pow 2) (by simp)
  simpa using h.tendsto

theorem nb_fisher_weight_poisson_limit (mu : ℝ) :
    Filter.Tendsto (fun alpha : ℝ => mu / (1 + alpha * mu)) (nhds 0) (nhds mu) := by
  have h : ContinuousAt (fun alpha : ℝ => mu / (1 + alpha * mu)) 0 :=
    continuousAt_const.div (continuousAt_const.add
      (continuousAt_id.mul continuousAt_const)) (by simp)
  simpa using h.tendsto

end CbmrProofs.Calculus

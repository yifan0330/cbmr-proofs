import lean_proof_checks.NegativeBinomial
import Mathlib.Topology.Algebra.Order.Field

/-! Genuine right-hand Poisson limits and arbitrary-finite cumulant-expression
algebra. No probability-law identification is asserted in this module. -/
namespace CbmrProofs.AggregatedNB

open scoped BigOperators Topology
open Filter NegativeBinomial
noncomputable section

theorem nb_score_poisson_limit (mu y : ℝ) :
    Tendsto (fun alpha : ℝ => (y - mu) / (1 + alpha * mu))
      (𝓝[>] 0) (𝓝 (y - mu)) := by
  have h : ContinuousAt (fun alpha : ℝ => (y - mu) / (1 + alpha * mu)) 0 :=
    continuousAt_const.div (continuousAt_const.add (continuousAt_id.mul continuousAt_const))
      (by norm_num)
  simpa using h.tendsto.mono_left nhdsWithin_le_nhds

theorem nb_observed_weight_poisson_limit (mu y : ℝ) :
    Tendsto (fun alpha : ℝ => mu * (1 + alpha * y) / (1 + alpha * mu) ^ 2)
      (𝓝[>] 0) (𝓝 mu) := by
  have h : ContinuousAt (fun alpha : ℝ => mu * (1 + alpha * y) / (1 + alpha * mu) ^ 2) 0 :=
    (continuousAt_const.mul (continuousAt_const.add (continuousAt_id.mul continuousAt_const))).div
      ((continuousAt_const.add (continuousAt_id.mul continuousAt_const)).pow 2) (by norm_num)
  simpa using h.tendsto.mono_left nhdsWithin_le_nhds

theorem nb_fisher_weight_poisson_limit (mu : ℝ) :
    Tendsto (fun alpha : ℝ => mu / (1 + alpha * mu)) (𝓝[>] 0) (𝓝 mu) := by
  have h : ContinuousAt (fun alpha : ℝ => mu / (1 + alpha * mu)) 0 :=
    continuousAt_const.div (continuousAt_const.add (continuousAt_id.mul continuousAt_const))
      (by norm_num)
  simpa using h.tendsto.mono_left nhdsWithin_le_nhds

theorem aggregated_weight_regularization {alpha s1 s2 mu : ℝ} (y : ℝ)
    (ha : 0 < alpha) (h1 : 0 < s1) (h2 : 0 < s2) (hm : 0 < mu) :
    (rho alpha s1 s2 + y) * mu * rateA alpha s1 s2 /
        (mu + rateA alpha s1 s2) ^ 2 =
      mu * s1 * (s1 ^ 2 + y * alpha * s2) / (s1 + alpha * s2 * mu) ^ 2 := by
  unfold rho rateA
  have hd : 0 < s1 + alpha * s2 * mu := by positivity
  field_simp
  ring

theorem aggregated_weight_poisson_limit {s1 s2 mu : ℝ} (y : ℝ)
    (h1 : 0 < s1) (h2 : 0 < s2) (hm : 0 < mu) :
    Tendsto (fun alpha : ℝ => (rho alpha s1 s2 + y) * mu * rateA alpha s1 s2 /
      (mu + rateA alpha s1 s2) ^ 2) (𝓝[>] 0) (𝓝 (s1 * mu)) := by
  have hc : ContinuousAt (fun alpha : ℝ =>
      mu * s1 * (s1 ^ 2 + y * alpha * s2) / (s1 + alpha * s2 * mu) ^ 2) 0 := by
    apply ContinuousAt.div
    · fun_prop
    · fun_prop
    · simpa using pow_ne_zero 2 (ne_of_gt h1)
  have ht : Tendsto (fun alpha : ℝ =>
      mu * s1 * (s1 ^ 2 + y * alpha * s2) / (s1 + alpha * s2 * mu) ^ 2)
      (𝓝[>] 0) (𝓝 (s1 * mu)) := by
    simpa [ne_of_gt h1, mul_comm] using hc.tendsto.mono_left nhdsWithin_le_nhds
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin] with alpha ha
  exact (aggregated_weight_regularization y ha h1 h2 hm).symm

variable {ι : Type*} [Fintype ι]

theorem finite_moment_gap_identity (a : ι → ℝ) (hs : (∑ i, a i) ≠ 0) :
    (∑ i, a i) * ((∑ i, a i ^ 3) - (∑ i, a i ^ 2) ^ 2 / (∑ i, a i)) =
      (1 / 2 : ℝ) * ∑ i, ∑ j, a i * a j * (a i - a j) ^ 2 := by
  have h := weighted_quadratic_identity 0 a a (by simpa using hs)
  simpa [weightedQuadratic, pow_succ, pow_two, mul_assoc] using h

theorem finite_moment_gap_nonneg (a : ι → ℝ)
    (ha : ∀ i, 0 ≤ a i) (hs : 0 < ∑ i, a i) :
    0 ≤ (∑ i, a i ^ 3) - (∑ i, a i ^ 2) ^ 2 / (∑ i, a i) := by
  have hh := finite_moment_gap_identity a (ne_of_gt hs)
  have hp : 0 ≤ ∑ i, ∑ j, a i * a j * (a i - a j) ^ 2 :=
    Finset.sum_nonneg (fun i _ => Finset.sum_nonneg (fun j _ =>
      mul_nonneg (mul_nonneg (ha i) (ha j)) (sq_nonneg _)))
  exact nonneg_of_mul_nonneg_right (by rw [hh]; positivity) hs

theorem finite_moment_gap_zero_iff (a : ι → ℝ)
    (ha : ∀ i, 0 < a i) (hs : 0 < ∑ i, a i) :
    ((∑ i, a i ^ 3) - (∑ i, a i ^ 2) ^ 2 / (∑ i, a i) = 0) ↔
      ∀ i j, a i = a j := by
  have hh := finite_moment_gap_identity a (ne_of_gt hs)
  constructor
  · intro hz
    have hp : (∑ i, ∑ j, a i * a j * (a i - a j) ^ 2) = 0 := by
      rw [hz, mul_zero] at hh
      linarith
    have hn : ∀ i j, 0 ≤ a i * a j * (a i - a j) ^ 2 := fun i j =>
      mul_nonneg (mul_nonneg (ha i).le (ha j).le) (sq_nonneg _)
    have hrows := (Finset.sum_eq_zero_iff_of_nonneg (fun i _ =>
      Finset.sum_nonneg (fun j _ => hn i j))).mp hp
    intro i j
    have hij := (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => hn i j)).mp
      (hrows i (Finset.mem_univ i)) j (Finset.mem_univ j)
    have hzsq : (a i - a j) ^ 2 = 0 :=
      (mul_eq_zero.mp hij).resolve_left (ne_of_gt (mul_pos (ha i) (ha j)))
    exact sub_eq_zero.mp (sq_eq_zero_iff.mp hzsq)
  · intro heq
    have hp : (∑ i, ∑ j, a i * a j * (a i - a j) ^ 2) = 0 := by
      apply Finset.sum_eq_zero
      intro i _
      apply Finset.sum_eq_zero
      intro j _
      rw [heq i j]
      simp
    rw [hp, mul_zero] at hh
    exact (mul_eq_zero.mp hh).resolve_left (ne_of_gt hs)

theorem finite_third_cumulant_gap (alpha mu : ℝ) (a : ι → ℝ)
    (hs : (∑ i, a i) ≠ 0) :
    (∑ i, thirdCumulantExpression (mu * a i) (1 / alpha)) -
      thirdCumulantExpression (mu * ∑ i, a i) (rho alpha (∑ i, a i) (∑ i, a i ^ 2)) =
      2 * alpha ^ 2 * mu ^ 3 *
        ((∑ i, a i ^ 3) - (∑ i, a i ^ 2) ^ 2 / (∑ i, a i)) := by
  have hi : ∀ i, thirdCumulantExpression (mu * a i) (1 / alpha) =
      mu * a i + (3 * alpha * mu ^ 2) * a i ^ 2 +
        (2 * alpha ^ 2 * mu ^ 3) * a i ^ 3 := by
    intro i
    simp only [thirdCumulantExpression, div_eq_mul_inv, one_mul, inv_pow, inv_inv]
    ring
  simp_rw [hi]
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
  unfold thirdCumulantExpression rho
  field_simp
  ring

theorem finite_third_cumulant_gap_nonneg (alpha mu : ℝ) (a : ι → ℝ)
    (hm : 0 ≤ mu) (ha : ∀ i, 0 ≤ a i) (hs : 0 < ∑ i, a i) :
    0 ≤ (∑ i, thirdCumulantExpression (mu * a i) (1 / alpha)) -
      thirdCumulantExpression (mu * ∑ i, a i) (rho alpha (∑ i, a i) (∑ i, a i ^ 2)) := by
  rw [finite_third_cumulant_gap alpha mu a (ne_of_gt hs)]
  exact mul_nonneg (by positivity) (finite_moment_gap_nonneg a ha hs)

theorem finite_third_cumulant_gap_zero_iff [Nonempty ι]
    (alpha mu : ℝ) (a : ι → ℝ) (halpha : 0 < alpha) (hmu : 0 < mu)
    (ha : ∀ i, 0 < a i) :
    ((∑ i, thirdCumulantExpression (mu * a i) (1 / alpha)) -
      thirdCumulantExpression (mu * ∑ i, a i) (rho alpha (∑ i, a i) (∑ i, a i ^ 2)) = 0) ↔
      ∀ i j, a i = a j := by
  have hs : 0 < ∑ i, a i := Finset.sum_pos (fun i _ => ha i) Finset.univ_nonempty
  rw [finite_third_cumulant_gap alpha mu a (ne_of_gt hs),
    mul_eq_zero, or_iff_right (ne_of_gt (by positivity : 0 < 2 * alpha ^ 2 * mu ^ 3))]
  exact finite_moment_gap_zero_iff a ha hs

end
end CbmrProofs.AggregatedNB

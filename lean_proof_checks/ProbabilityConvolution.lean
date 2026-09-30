import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Data.Nat.Choose.Sum
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Normalized integer-shape NB masses and a common-probability convolution

`integerNBMass r p` has shape `r + 1` in the appendix's convention
`(1-p)^shape p^count`. Adding an independent geometric count with the same
`p` increments the shape. The finite convolution is proved by the
hockey-stick identity, not by assuming a law for the sum. This establishes
the repeated-geometric construction of integer-shape NB distributions; it
does not assert arbitrary real-shape NB convolution.
-/

noncomputable section

open scoped BigOperators

namespace CbmrProofs.ProbabilityConvolution

def integerNBMass (r : ℕ) (p : ℝ) (n : ℕ) : ℝ :=
  ((n + r).choose r : ℝ) * (1 - p) ^ (r + 1) * p ^ n

theorem integer_nb_nonneg (r : ℕ) (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1) (n : ℕ) :
    0 ≤ integerNBMass r p n := by
  unfold integerNBMass
  positivity

/-- Actual summability and total mass one, including the degenerate `p = 0` case. -/
theorem integer_nb_hasSum_one (r : ℕ) (p : ℝ) (hp : 0 ≤ p) (hp1 : p < 1) :
    HasSum (integerNBMass r p) 1 := by
  have hn : ‖p‖ < 1 := by simpa [Real.norm_eq_abs, abs_of_nonneg hp] using hp1
  have h := (hasSum_choose_mul_geometric_of_norm_lt_one r hn).mul_left
    ((1 - p) ^ (r + 1))
  have heq : (fun n => (1 - p) ^ (r + 1) * (((n + r).choose r : ℝ) * p ^ n)) =
      integerNBMass r p := by
    funext n
    unfold integerNBMass
    ring
  rw [heq] at h
  have hd : (1 - p) ^ (r + 1) ≠ 0 := pow_ne_zero _ (ne_of_gt (sub_pos.mpr hp1))
  simpa only [mul_one_div_cancel hd] using h

/-- A genuine normalized probability mass function, not just a closed-form expression. -/
def integerNBPMF (r : ℕ) (p : ℝ) (hp : 0 ≤ p) (hp1 : p < 1) : PMF ℕ :=
  ⟨fun n => ENNReal.ofReal (integerNBMass r p n), by
    have hs := integer_nb_hasSum_one r p hp hp1
    apply ENNReal.summable.hasSum_iff.mpr
    rw [← ENNReal.ofReal_tsum_of_nonneg (integer_nb_nonneg r p hp hp1.le) hs.summable,
      hs.tsum_eq, ENNReal.ofReal_one]⟩

theorem integer_nb_geometric (p : ℝ) (n : ℕ) :
    integerNBMass 0 p n = (1 - p) * p ^ n := by
  simp [integerNBMass]

/-- Exact common-`p` NB(shape `r+1`) plus NB(shape one) convolution. -/
theorem common_probability_geometric_convolution (r n : ℕ) (p : ℝ) :
    (∑ k ∈ Finset.range (n + 1), integerNBMass r p k * integerNBMass 0 p (n - k)) =
      integerNBMass (r + 1) p n := by
  simp only [integer_nb_geometric]
  have hterm : ∀ k ∈ Finset.range (n + 1),
      integerNBMass r p k * ((1 - p) * p ^ (n - k)) =
        ((k + r).choose r : ℝ) * ((1 - p) ^ (r + 2) * p ^ n) := by
    intro k hk
    have hk' : k ≤ n := Nat.le_of_lt_succ (Finset.mem_range.mp hk)
    unfold integerNBMass
    rw [show r + 2 = (r + 1) + 1 by omega, pow_succ]
    have hp : p ^ k * p ^ (n - k) = p ^ n := by
      rw [← pow_add, Nat.add_sub_of_le hk']
    calc
      _ = ((k + r).choose r : ℝ) * ((1 - p) ^ (r + 1) * (1 - p)) *
          (p ^ k * p ^ (n - k)) := by ring
      _ = _ := by rw [hp]; ring
  rw [Finset.sum_congr rfl hterm, ← Finset.sum_mul]
  have hchoose : (∑ k ∈ Finset.range (n + 1), ((k + r).choose r : ℝ)) =
      ((n + r + 1).choose (r + 1) : ℝ) := by
    exact_mod_cast Nat.sum_range_add_choose n r
  rw [hchoose]
  simp only [integerNBMass, Nat.add_assoc]
  ring

/-- The finite convolution of `r + 1` geometric mass functions. -/
def geometricSumMass : ℕ → ℝ → ℕ → ℝ
  | 0, p, n => (1 - p) * p ^ n
  | r + 1, p, n => ∑ k ∈ Finset.range (n + 1),
      geometricSumMass r p k * ((1 - p) * p ^ (n - k))

theorem repeated_geometric_convolution (r : ℕ) (p : ℝ) :
    geometricSumMass r p = integerNBMass r p := by
  induction r with
  | zero =>
      funext n
      exact (integer_nb_geometric p n).symm
  | succ r ih =>
      funext n
      simp only [geometricSumMass, ih, ← integer_nb_geometric]
      exact common_probability_geometric_convolution r n p

end CbmrProofs.ProbabilityConvolution

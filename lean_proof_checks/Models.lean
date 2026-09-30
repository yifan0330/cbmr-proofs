import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! Finite-index predictor and Poisson-compression identities.

Theorems here concern algebra of the likelihood, not sampling distributions.
-/

namespace CbmrProofs.Models

open scoped BigOperators

variable {I J R P : Type*} [Fintype I] [Fintype J] [Fintype R] [Fintype P]

def svPredictor (Z : I → R → ℝ) (B : J → P → ℝ) (β : R → P → ℝ)
    (i : I) (j : J) : ℝ :=
  ∑ r, Z i r * ∑ p, B j p * β r p

def gcPredictor (spatial : J → ℝ) (global : I → ℝ) (i : I) (j : J) : ℝ :=
  spatial j + global i

omit [Fintype I] [Fintype J] in
theorem sv_predictor_coefficients (Z : I → R → ℝ) (B : J → P → ℝ)
    (β : R → P → ℝ) (i : I) (j : J) :
    svPredictor Z B β i j = ∑ r, ∑ p, (Z i r * B j p) * β r p := by
  simp only [svPredictor, Finset.mul_sum, mul_assoc]

omit [Fintype I] [Fintype J] in
theorem gc_mean_factorisation (spatial : J → ℝ) (global : I → ℝ)
    (i : I) (j : J) :
    Real.exp (gcPredictor spatial global i j) =
      Real.exp (spatial j) * Real.exp (global i) := by
  exact Real.exp_add _ _

omit [Fintype J] in
theorem constant_basis_effect (B : J → P → ℝ) (c : P → ℝ)
    (hconstant : ∀ j, ∑ p, B j p * c p = 1) (γ : ℝ) (j : J) :
    (∑ p, B j p * (γ * c p)) = γ := by
  calc
    (∑ p, B j p * (γ * c p)) = γ * ∑ p, B j p * c p := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro p _
      ring
    _ = γ := by rw [hconstant, mul_one]

omit [Fintype J] in
theorem restricted_sv_equals_gc (B : J → P → ℝ) (c : P → ℝ)
    (hconstant : ∀ j, ∑ p, B j p * c p = 1)
    (baseline : J → ℝ) (x γ : ℝ) (j : J) :
    baseline j + x * (∑ p, B j p * (γ * c p)) =
      baseline j + x * γ := by
  rw [constant_basis_effect B c hconstant]

theorem poisson_marginal_compression
    (Y : I → J → ℝ) (u S : J → ℝ) (e m : I → ℝ) :
    (∑ i, ∑ j, (u j * e i - Y i j * (S j + m i))) =
      (∑ j, u j) * (∑ i, e i)
      - (∑ j, (∑ i, Y i j) * S j)
      - (∑ i, (∑ j, Y i j) * m i) := by
  have hmean : (∑ i, ∑ j, u j * e i) = (∑ j, u j) * (∑ i, e i) := by
    simp only [Finset.sum_mul, Finset.mul_sum]
  have hspatial : (∑ i, ∑ j, Y i j * S j) = ∑ j, (∑ i, Y i j) * S j := by
    simp only [Finset.sum_mul]
    exact Finset.sum_comm
  have hglobal : (∑ i, ∑ j, Y i j * m i) = ∑ i, (∑ j, Y i j) * m i := by
    simp only [Finset.sum_mul]
  simp only [mul_add, Finset.sum_sub_distrib, Finset.sum_add_distrib]
  rw [hmean, hspatial, hglobal]
  ring

theorem poisson_exp_marginal_compression
    (Y : I → J → ℝ) (S : J → ℝ) (m : I → ℝ) :
    (∑ i, ∑ j, (Real.exp (S j + m i) - Y i j * (S j + m i))) =
      (∑ j, Real.exp (S j)) * (∑ i, Real.exp (m i))
      - (∑ j, (∑ i, Y i j) * S j)
      - (∑ i, (∑ j, Y i j) * m i) := by
  simp only [Real.exp_add]
  exact poisson_marginal_compression Y (fun j => Real.exp (S j)) S
    (fun i => Real.exp (m i)) m

theorem poisson_allocation_difference
    (Y : I → J → ℝ) (u : J → ℝ) (e : I → ℝ)
    (hu : ∀ j, 0 < u j) (he : ∀ i, 0 < e i)
    (hT : 0 < ∑ i, e i) :
    ((∑ j, (∑ i, Y i j) * Real.log (u j))
      + (∑ i, (∑ j, Y i j) * Real.log (e i)))
      - (∑ j, (∑ i, Y i j) * Real.log (u j * ∑ i, e i)) =
      ∑ i, (∑ j, Y i j) * Real.log (e i / ∑ i, e i) := by
  simp_rw [Real.log_mul (ne_of_gt (hu _)) (ne_of_gt hT),
    Real.log_div (ne_of_gt (he _)) (ne_of_gt hT)]
  simp only [mul_add, mul_sub, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  have htotal :
      (∑ j, (∑ i, Y i j) * Real.log (∑ i, e i)) =
        ∑ i, (∑ j, Y i j) * Real.log (∑ i, e i) := by
    simp only [Finset.sum_mul]
    exact Finset.sum_comm
  rw [htotal]
  ring

omit [Fintype I] [Fintype J] in
theorem spatial_scale_unidentified (S : J → ℝ) (m : I → ℝ)
    (shift : ℝ) (i : I) (j : J) :
    gcPredictor (fun j => S j + shift) (fun i => m i - shift) i j =
      gcPredictor S m i j := by
  unfold gcPredictor
  ring

omit [Fintype R] [Fintype P] in
theorem weighted_design_entry (Z : I → R → ℝ) (B : J → P → ℝ)
    (w : I → J → ℝ) (r r' : R) (p p' : P) :
    (∑ i, ∑ j, (Z i r * B j p) * w i j * (Z i r' * B j p')) =
      ∑ i, (Z i r * Z i r') * ∑ j, w i j * B j p * B j p' := by
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem nb_weight_grid_determinant (a b c d α : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hα : 0 < α) :
    (a * c / (1 + α * a * c)) * (b * d / (1 + α * b * d)) -
      (a * d / (1 + α * a * d)) * (b * c / (1 + α * b * c)) =
      α * a * b * c * d * (a - b) * (d - c) /
        ((1 + α * a * c) * (1 + α * a * d) *
          (1 + α * b * c) * (1 + α * b * d)) := by
  have hac : 1 + α * a * c ≠ 0 := ne_of_gt (by positivity)
  have had : 1 + α * a * d ≠ 0 := ne_of_gt (by positivity)
  have hbc : 1 + α * b * c ≠ 0 := ne_of_gt (by positivity)
  have hbd : 1 + α * b * d ≠ 0 := ne_of_gt (by positivity)
  field_simp
  ring

theorem nb_weight_grid_not_separable (a b c d α : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hα : 0 < α)
    (hab : a ≠ b) (hdc : d ≠ c) :
    (a * c / (1 + α * a * c)) * (b * d / (1 + α * b * d)) -
      (a * d / (1 + α * a * d)) * (b * c / (1 + α * b * c)) ≠ 0 := by
  rw [nb_weight_grid_determinant a b c d α ha hb hc hd hα]
  apply div_ne_zero
  · exact mul_ne_zero
      (mul_ne_zero (by positivity) (sub_ne_zero.mpr hab)) (sub_ne_zero.mpr hdc)
  · positivity

theorem separable_grid_determinant (a b c d : ℝ) :
    (a * c) * (b * d) - (a * d) * (b * c) = 0 := by
  ring

end CbmrProofs.Models

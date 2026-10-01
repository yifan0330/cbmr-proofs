import lean_proof_checks.Models
import Mathlib.Data.Matrix.ColumnRowPartitioned
import Mathlib.LinearAlgebra.Matrix.Kronecker
import Mathlib.Logic.Equiv.Fin.Basic

/-! Stacked designs, row-major coefficient indexing and identification constraints. -/

namespace CbmrProofs.Design

set_option backward.isDefEq.respectTransparency false

open Matrix
open scoped BigOperators Kronecker

variable {I J G R P : Type*}
variable [Fintype I] [Fintype J] [Fintype G] [Fintype R] [Fintype P]

omit [Fintype I] [Fintype J] [Fintype G] in
theorem sv_matrix_predictor (Z : Matrix I R ℝ) (B : Matrix J P ℝ)
    (beta : Matrix R P ℝ) (i : I) (j : J) :
    (Z * beta * Bᵀ) i j = CbmrProofs.Models.svPredictor Z B beta i j := by
  simp only [Matrix.mul_apply, Matrix.transpose_apply, Finset.sum_mul,
    CbmrProofs.Models.svPredictor, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro r _
  apply Finset.sum_congr rfl
  intro p _
  ring

omit [Fintype I] [Fintype J] [Fintype G] in
theorem sv_kronecker_predictor (Z : Matrix I R ℝ) (B : Matrix J P ℝ)
    (beta : Matrix R P ℝ) (i : I) (j : J) :
    ((Z ⊗ₖ B) *ᵥ (fun rp => beta rp.1 rp.2)) (i, j) =
      (Z * beta * Bᵀ) i j := by
  rw [sv_matrix_predictor]
  rw [CbmrProofs.Models.sv_predictor_coefficients Z B beta i j]
  simp [Matrix.mulVec, dotProduct, Fintype.sum_prod_type, Matrix.kroneckerMap_apply]

theorem row_major_index {r p : ℕ} (a : Fin r) (b : Fin p) :
    (finProdFinEquiv (a, b)).val = a.val * p + b.val := by
  change b.val + p * a.val = a.val * p + b.val
  simp [Nat.mul_comm, Nat.add_comm]

theorem row_major_coefficient {r p : ℕ} (beta : Fin r → Fin p → ℝ)
    (a : Fin r) (b : Fin p) :
    (fun k : Fin (r * p) =>
      beta (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2)
        (finProdFinEquiv (a, b)) = beta a b := by
  simp

theorem positive_effect_increases_mean (f b x : ℝ) :
    Real.exp (f + x * b) < Real.exp (f + (x + 1) * b) ↔ 0 < b := by
  rw [Real.exp_lt_exp, show f + (x + 1) * b = (f + x * b) + b by ring]
  simp

theorem negative_effect_decreases_mean (f b x : ℝ) :
    Real.exp (f + (x + 1) * b) < Real.exp (f + x * b) ↔ b < 0 := by
  rw [Real.exp_lt_exp, show f + (x + 1) * b = (f + x * b) + b by ring]
  simp

theorem zero_effect_preserves_mean (f b x : ℝ) :
    Real.exp (f + (x + 1) * b) = Real.exp (f + x * b) ↔ b = 0 := by
  rw [Real.exp_injective.eq_iff, show f + (x + 1) * b = (f + x * b) + b by ring]
  simp

omit [Fintype I] [Fintype J] [Fintype R] [Fintype P] in
def membership [DecidableEq G] (group : I → G) : Matrix I G ℝ :=
  fun i g => if group i = g then 1 else 0

omit [Fintype I] [Fintype J] [Fintype G] [Fintype R] [Fintype P] in
def globalDesign (Z : Matrix I R ℝ) : Matrix (I × J) R ℝ :=
  fun ij r => Z ij.1 r

omit [Fintype I] [Fintype J] [Fintype G] [Fintype R] [Fintype P] in
def groupedDesign [DecidableEq G] (group : I → G)
    (B : Matrix J P ℝ) (Z : Matrix I R ℝ) : Matrix (I × J) ((G × P) ⊕ R) ℝ :=
  Matrix.fromCols (membership group ⊗ₖ B) (globalDesign Z)

omit [Fintype I] [Fintype J] in
theorem grouped_design_predictor [DecidableEq G] (group : I → G)
    (B : Matrix J P ℝ) (Z : Matrix I R ℝ)
    (xi : G → P → ℝ) (gamma : R → ℝ) (i : I) (j : J) :
    (groupedDesign group B Z *ᵥ Sum.elim (fun gp => xi gp.1 gp.2) gamma) (i, j) =
      (∑ p, B j p * xi (group i) p) + ∑ r, Z i r * gamma r := by
  simp [groupedDesign, membership, globalDesign,
    Matrix.mulVec, dotProduct, Fintype.sum_prod_type, Matrix.kroneckerMap_apply,
    ite_mul]

omit [Fintype I] [Fintype J] [Fintype G] [Fintype R] [Fintype P] in
theorem global_design_kronecker (Z : Matrix I R ℝ) (i : I) (j : J) (r : R) :
    (Z ⊗ₖ (fun (_ : J) (_ : Unit) => (1 : ℝ))) (i, j) (r, ()) =
      globalDesign (J := J) Z (i, j) r := by
  change Z i r * 1 = Z i r
  exact mul_one _

omit [Fintype I] [Fintype J] in
theorem grouped_parameter_count :
    Fintype.card ((G × P) ⊕ R) =
      Fintype.card G * Fintype.card P + Fintype.card R := by
  simp

omit [Fintype I] [Fintype J] [Fintype G] in
theorem sv_parameter_count :
    Fintype.card (R × P) = Fintype.card R * Fintype.card P := by
  simp

omit [Fintype I] [Fintype J] [Fintype R] in
theorem matched_parameter_counts :
    Fintype.card ((G × P) ⊕ Unit) = Fintype.card G * Fintype.card P + 1 ∧
    Fintype.card ((G ⊕ Unit) × P) = (Fintype.card G + 1) * Fintype.card P := by
  simp

omit [Fintype I] [Fintype J] [Fintype G] in
theorem coefficient_shift_unidentified (B : Matrix J P ℝ) (Z : Matrix I R ℝ)
    (cB xi : P → ℝ) (cZ gamma : R → ℝ)
    (hB : ∀ j, ∑ p, B j p * cB p = 1)
    (hZ : ∀ i, ∑ r, Z i r * cZ r = 1) (t : ℝ) (i : I) (j : J) :
    (∑ p, B j p * (xi p + t * cB p)) +
        (∑ r, Z i r * (gamma r - t * cZ r)) =
      (∑ p, B j p * xi p) + ∑ r, Z i r * gamma r := by
  simp only [mul_add, mul_sub, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  have hb : (∑ p, B j p * (t * cB p)) = t := by
    exact CbmrProofs.Models.constant_basis_effect B cB hB t j
  have hz : (∑ r, Z i r * (t * cZ r)) = t := by
    exact CbmrProofs.Models.constant_basis_effect Z cZ hZ t i
  rw [hb, hz]
  ring

omit [Fintype I] [Fintype J] [Fintype G] in
theorem coefficient_shift_mean_unidentified
    (B : Matrix J P ℝ) (Z : Matrix I R ℝ)
    (cB xi : P → ℝ) (cZ gamma : R → ℝ)
    (hB : ∀ j, ∑ p, B j p * cB p = 1)
    (hZ : ∀ i, ∑ r, Z i r * cZ r = 1) (t : ℝ) (i : I) (j : J) :
    Real.exp ((∑ p, B j p * (xi p + t * cB p)) +
        (∑ r, Z i r * (gamma r - t * cZ r))) =
      Real.exp ((∑ p, B j p * xi p) + ∑ r, Z i r * gamma r) := by
  rw [coefficient_shift_unidentified B Z cB xi cZ gamma hB hZ]

omit [Fintype G] [Fintype R] [Fintype P] in
theorem count_marginal_total (Y : I → J → ℝ) :
    (∑ i, ∑ j, Y i j) = ∑ j, ∑ i, Y i j :=
  Finset.sum_comm

end CbmrProofs.Design

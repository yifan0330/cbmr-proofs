import Mathlib.Data.Real.Basic
import Mathlib.LinearAlgebra.Matrix.Kronecker
import Mathlib.LinearAlgebra.Matrix.SchurComplement
import lean_proof_checks.Structure

/-!
# Finite-dimensional covariance algebra

These identities verify the appendix's matrix algebra over `ℝ`. Inverse
identities assume nonsingular determinants explicitly. They do not assert
positive definiteness, sampling-distribution validity, or a spectral-norm
condition-number formula.
-/

namespace CbmrProofs.Covariance

open Matrix
open scoped Kronecker

variable {m n p q : Type*}
variable [Fintype m] [Fintype n] [Fintype p] [Fintype q]
variable [DecidableEq m] [DecidableEq n] [DecidableEq p] [DecidableEq q]

/-- The Schur complement of the spatial information block. -/
noncomputable def schur (D : Matrix m m ℝ) (F : Matrix m n ℝ)
    (E : Matrix n n ℝ) : Matrix n n ℝ :=
  E - Fᵀ * D⁻¹ * F

/-- Full bordered inverse; symmetry of `D` and `E` is not needed for this identity. -/
theorem bordered_inverse (D : Matrix m m ℝ) (F : Matrix m n ℝ)
    (E : Matrix n n ℝ) (hD : IsUnit D.det) (hS : IsUnit (schur D F E).det) :
    (fromBlocks D F Fᵀ E)⁻¹ =
      fromBlocks
        (D⁻¹ + D⁻¹ * F * (schur D F E)⁻¹ * Fᵀ * D⁻¹)
        (-(D⁻¹ * F * (schur D F E)⁻¹))
        (-((schur D F E)⁻¹ * Fᵀ * D⁻¹))
        (schur D F E)⁻¹ := by
  cases (D.isUnit_iff_isUnit_det.mpr hD).nonempty_invertible
  have hS' : IsUnit (E - Fᵀ * ⅟D * F) := by
    simpa only [invOf_eq_nonsing_inv, schur] using
      (schur D F E).isUnit_iff_isUnit_det.mpr hS
  cases hS'.nonempty_invertible
  let := Matrix.fromBlocks₁₁Invertible D F Fᵀ E
  simpa only [invOf_eq_nonsing_inv, schur] using
    Matrix.invOf_fromBlocks₁₁_eq D F Fᵀ E

/-- The Schur determinant formula needs only the leading block nonsingular. -/
theorem bordered_determinant (D : Matrix m m ℝ) (F : Matrix m n ℝ)
    (E : Matrix n n ℝ) (hD : IsUnit D.det) :
    (fromBlocks D F Fᵀ E).det = D.det * (schur D F E).det := by
  cases (D.isUnit_iff_isUnit_det.mpr hD).nonempty_invertible
  simpa only [invOf_eq_nonsing_inv, schur] using
    Matrix.det_fromBlocks₁₁ D F Fᵀ E

theorem spatial_covariance_block (D : Matrix m m ℝ) (F : Matrix m n ℝ)
    (E : Matrix n n ℝ) (hD : IsUnit D.det) (hS : IsUnit (schur D F E).det) :
    ((fromBlocks D F Fᵀ E)⁻¹).submatrix Sum.inl Sum.inl =
      D⁻¹ + D⁻¹ * F * (schur D F E)⁻¹ * Fᵀ * D⁻¹ := by
  rw [bordered_inverse D F E hD hS]
  rfl

theorem global_covariance_block (D : Matrix m m ℝ) (F : Matrix m n ℝ)
    (E : Matrix n n ℝ) (hD : IsUnit D.det) (hS : IsUnit (schur D F E).det) :
    ((fromBlocks D F Fᵀ E)⁻¹).submatrix Sum.inr Sum.inr = (schur D F E)⁻¹ := by
  rw [bordered_inverse D F E hD hS]
  rfl

theorem spatial_global_covariance_block (D : Matrix m m ℝ) (F : Matrix m n ℝ)
    (E : Matrix n n ℝ) (hD : IsUnit D.det) (hS : IsUnit (schur D F E).det) :
    ((fromBlocks D F Fᵀ E)⁻¹).submatrix Sum.inl Sum.inr =
      -(D⁻¹ * F * (schur D F E)⁻¹) := by
  rw [bordered_inverse D F E hD hS]
  rfl

theorem two_block_inverse (A : Matrix m m ℝ) (B : Matrix n n ℝ)
    (hA : IsUnit A.det) (hB : IsUnit B.det) :
    (fromBlocks A 0 0 B)⁻¹ = fromBlocks A⁻¹ 0 0 B⁻¹ := by
  apply Matrix.inv_eq_left_inv
  simp only [fromBlocks_multiply, Matrix.nonsing_inv_mul _ hA,
    Matrix.nonsing_inv_mul _ hB, Matrix.mul_zero, Matrix.zero_mul,
    add_zero, zero_add, fromBlocks_one]

theorem two_block_determinant (A : Matrix m m ℝ) (B : Matrix n n ℝ) :
    (fromBlocks A 0 0 B).det = A.det * B.det :=
  Matrix.det_fromBlocks_zero₂₁ A 0 B

/-- A characteristic-determinant factorization, without a claim about norms. -/
theorem two_block_characteristic_determinant (A : Matrix m m ℝ)
    (B : Matrix n n ℝ) (t : ℝ) :
    (t • (1 : Matrix (m ⊕ n) (m ⊕ n) ℝ) - fromBlocks A 0 0 B).det =
      (t • (1 : Matrix m m ℝ) - A).det *
        (t • (1 : Matrix n n ℝ) - B).det := by
  have h :
      t • (1 : Matrix (m ⊕ n) (m ⊕ n) ℝ) - fromBlocks A 0 0 B =
        fromBlocks (t • 1 - A) 0 0 (t • 1 - B) := by
    ext i j
    cases i <;> cases j <;> simp [Matrix.one_apply, Matrix.fromBlocks]
  rw [h, two_block_determinant]

omit [Fintype p] [Fintype q] [DecidableEq m] [DecidableEq n]
  [DecidableEq p] [DecidableEq q] in
theorem kronecker_weighted_gram (Z : Matrix m p ℝ) (B : Matrix n q ℝ)
    (WZ : Matrix m m ℝ) (WB : Matrix n n ℝ) :
    (Z ⊗ₖ B)ᵀ * (WZ ⊗ₖ WB) * (Z ⊗ₖ B) =
      (Zᵀ * WZ * Z) ⊗ₖ (Bᵀ * WB * B) := by
  rw [← Matrix.kroneckerMap_transpose, ← Matrix.mul_kronecker_mul,
    ← Matrix.mul_kronecker_mul]

theorem kronecker_inverse (A : Matrix m m ℝ) (B : Matrix n n ℝ)
    (hA : IsUnit A.det) (hB : IsUnit B.det) :
    (A ⊗ₖ B)⁻¹ = A⁻¹ ⊗ₖ B⁻¹ := by
  apply Matrix.inv_eq_left_inv
  rw [← Matrix.mul_kronecker_mul, Matrix.nonsing_inv_mul _ hA,
    Matrix.nonsing_inv_mul _ hB, Matrix.one_kronecker_one]

omit [DecidableEq m] in
theorem symmetric_gram_eq_square (H : Matrix m m ℝ) (hH : Hᵀ = H) :
    Hᵀ * H = H * H := by
  rw [hH]

/-- Valid for every real square matrix, not just symmetric matrices. -/
theorem gram_determinant_square (H : Matrix m m ℝ) :
    (Hᵀ * H).det = H.det ^ 2 := by
  rw [Matrix.det_mul, Matrix.det_transpose, pow_two]

/-- Inverse curvature minus penalized sampling covariance, as a matrix identity. -/
theorem penalty_covariance_difference (info penalty : Matrix m m ℝ)
    (h : IsUnit (info + penalty).det) :
    (info + penalty)⁻¹ - (info + penalty)⁻¹ * info * (info + penalty)⁻¹ =
      (info + penalty)⁻¹ * penalty * (info + penalty)⁻¹ := by
  have hsum :
      (info + penalty)⁻¹ * info * (info + penalty)⁻¹ +
        (info + penalty)⁻¹ * penalty * (info + penalty)⁻¹ =
          (info + penalty)⁻¹ := by
    rw [← Matrix.add_mul, ← Matrix.mul_add, Matrix.nonsing_inv_mul _ h, Matrix.one_mul]
  exact sub_eq_iff_eq_add.mpr (by simpa only [add_comm] using hsum.symm)

omit [Fintype n] [DecidableEq m] [DecidableEq n] in
/-- Projecting a sandwich is equivalent to projecting its outer factors. -/
theorem project_sandwich (T : Matrix n m ℝ) (V meat : Matrix m m ℝ) :
    T * (V * meat * Vᵀ) * Tᵀ = (T * V) * meat * (T * V)ᵀ := by
  simp only [Matrix.transpose_mul, Matrix.mul_assoc]

omit [Fintype n] [DecidableEq n] in
theorem projected_penalty_covariance_difference (T : Matrix n m ℝ)
    (info penalty : Matrix m m ℝ) (h : IsUnit (info + penalty).det) :
    T * (info + penalty)⁻¹ * Tᵀ -
        T * ((info + penalty)⁻¹ * info * (info + penalty)⁻¹) * Tᵀ =
      T * ((info + penalty)⁻¹ * penalty * (info + penalty)⁻¹) * Tᵀ := by
  rw [← Matrix.sub_mul, ← Matrix.mul_sub, penalty_covariance_difference info penalty h]

section SharedGroups

variable {g : Type*} [Fintype g] [DecidableEq g]

/-- The global-covariate information blocks stacked by group. -/
def stackGroups (F : g → Matrix m n ℝ) : Matrix (m × g) n ℝ :=
  fun i j => F i.2 i.1 j

omit [Fintype n] [DecidableEq m] [DecidableEq n] in
theorem blockDiagonal_mul_stack (D : g → Matrix m m ℝ) (F : g → Matrix m n ℝ) :
    blockDiagonal D * stackGroups F = stackGroups (fun a => D a * F a) := by
  ext ⟨i, a⟩ j
  simp [Matrix.mul_apply, ← Finset.univ_product_univ, Finset.sum_product,
    Matrix.blockDiagonal_apply, stackGroups]

omit [Fintype n] [DecidableEq m] [DecidableEq n] in
theorem stack_transpose_mul_blockDiagonal (F : g → Matrix m n ℝ)
    (D : g → Matrix m m ℝ) (j : n) (i : m) (a : g) :
    ((stackGroups F)ᵀ * blockDiagonal D) j (i, a) = ((F a)ᵀ * D a) j i := by
  simp [Matrix.mul_apply, ← Finset.univ_product_univ, Finset.sum_product,
    Matrix.blockDiagonal_apply, stackGroups]

/-- Exact group-to-group spatial covariance, including the shared-global
correction when the spatial information itself is block diagonal. -/
theorem shared_group_covariance (D : g → Matrix m m ℝ)
    (F : g → Matrix m n ℝ) (E : Matrix n n ℝ)
    (hD : ∀ a, IsUnit (D a).det)
    (hS : IsUnit (schur (blockDiagonal D) (stackGroups F) E).det)
    (a b : g) :
    ((fromBlocks (blockDiagonal D) (stackGroups F) (stackGroups F)ᵀ E)⁻¹).submatrix
        (fun i => Sum.inl (i, a)) (fun j => Sum.inl (j, b)) =
      (if a = b then (D a)⁻¹ else 0) +
        (D a)⁻¹ * F a * (schur (blockDiagonal D) (stackGroups F) E)⁻¹ *
          (F b)ᵀ * (D b)⁻¹ := by
  have hDb : IsUnit (blockDiagonal D).det := by
    rw [Matrix.det_blockDiagonal]
    exact isUnit_iff_ne_zero.mpr (Finset.prod_ne_zero_iff.mpr
      (fun a _ => (hD a).ne_zero))
  rw [bordered_inverse _ _ _ hDb hS, Structure.uniform_family_inverse D hD]
  change (blockDiagonal (fun a => (D a)⁻¹) +
    (blockDiagonal (fun a => (D a)⁻¹) * stackGroups F *
      (schur (blockDiagonal D) (stackGroups F) E)⁻¹ *
      (stackGroups F)ᵀ * blockDiagonal (fun a => (D a)⁻¹))).submatrix
        (fun i => (i, a)) (fun j => (j, b)) = _
  rw [Matrix.submatrix_add, blockDiagonal_mul_stack, Matrix.mul_assoc
    (stackGroups (fun a => (D a)⁻¹ * F a) * _) (stackGroups F)ᵀ]
  simp only [Pi.add_apply]
  rw [Matrix.submatrix_mul _ _ _ id _ Function.bijective_id,
    Matrix.submatrix_mul _ _ _ id id Function.bijective_id]
  have hr : ((stackGroups F)ᵀ * blockDiagonal (fun a => (D a)⁻¹)).submatrix
      id (fun j => (j, b)) = (F b)ᵀ * (D b)⁻¹ := by
    ext i j
    exact stack_transpose_mul_blockDiagonal F (fun a => (D a)⁻¹) i j b
  rw [hr]
  have hd : (blockDiagonal (fun a => (D a)⁻¹)).submatrix
      (fun i => (i, a)) (fun j => (j, b)) = if a = b then (D a)⁻¹ else 0 := by
    ext i j
    by_cases hab : a = b <;> simp [Matrix.submatrix_apply, Matrix.blockDiagonal_apply, hab]
  rw [hd]
  change _ + ((D a)⁻¹ * F a * _) * ((F b)ᵀ * (D b)⁻¹) = _
  rw [Matrix.mul_assoc _ (F b)ᵀ (D b)⁻¹]
  simp only [Matrix.submatrix_id_id]

end SharedGroups

omit [Fintype n] [DecidableEq n] in
/-- A global design contained in the spatial design has zero profile
information, regardless of its coefficient values. -/
theorem confounded_schur_zero (D : Matrix m m ℝ) (X : Matrix m n ℝ)
    (hD : IsUnit D.det) (hsym : Dᵀ = D) :
    schur D (D * X) (Xᵀ * D * X) = 0 := by
  rw [schur, Matrix.transpose_mul, hsym, Matrix.mul_assoc Xᵀ D D⁻¹,
    Matrix.mul_nonsing_inv D hD, Matrix.mul_one, ← Matrix.mul_assoc, sub_self]

theorem confounded_border_singular [Nonempty n] (D : Matrix m m ℝ)
    (X : Matrix m n ℝ) (hD : IsUnit D.det) (hsym : Dᵀ = D) :
    ¬ IsUnit (fromBlocks D (D * X) (D * X)ᵀ (Xᵀ * D * X)).det := by
  rw [bordered_determinant D (D * X) (Xᵀ * D * X) hD,
    confounded_schur_zero D X hD hsym]
  simp

end CbmrProofs.Covariance

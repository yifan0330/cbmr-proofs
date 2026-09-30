import Mathlib.Data.Real.Basic
import Mathlib.LinearAlgebra.Matrix.Block
import Mathlib.LinearAlgebra.Matrix.Charpoly.Eigs

/-! # Design-supported partitions and arbitrary finite block families -/

namespace CbmrProofs.Structure

open Matrix

section Design

variable {ι r p k : Type*} [Fintype ι]

/-- Subject-specific spatial curvature may be arbitrary; design co-loading is the
only factor used in the structural-zero argument. -/
def designInformation (Z : Matrix ι r ℝ) (C : ι → Matrix p p ℝ) :
    Matrix (r × p) (r × p) ℝ :=
  fun a b => ∑ i, Z i a.1 * Z i b.1 * C i a.2 b.2

theorem design_zero (Z : Matrix ι r ℝ) (C : ι → Matrix p p ℝ)
    (a b : r) (h : ∀ i, Z i a * Z i b = 0) (u v : p) :
    designInformation Z C (a, u) (b, v) = 0 := by
  simp [designInformation, h]

/-- Any partition containing every co-loading edge has zero cross-component
blocks. Thus it applies in particular to a connected-component partition. -/
theorem partition_design_zero (Z : Matrix ι r ℝ) (C : ι → Matrix p p ℝ)
    (component : r → k)
    (hsupport : ∀ i a b, Z i a ≠ 0 → Z i b ≠ 0 → component a = component b)
    (a b : r) (hab : component a ≠ component b) (u v : p) :
    designInformation Z C (a, u) (b, v) = 0 := by
  apply design_zero
  intro i
  by_cases ha : Z i a = 0
  · simp [ha]
  · have hb : Z i b = 0 := by
      by_contra hb
      exact hab (hsupport i a b ha hb)
    simp [hb]

end Design

section Partition

variable {n k : Type*} [DecidableEq k]

/-- Reordering by the fibers of a supported partition really produces a block
diagonal matrix, including unequal component sizes and empty fibers. -/
theorem partition_reindex (H : Matrix n n ℝ) (component : n → k)
    (hzero : ∀ i j, component i ≠ component j → H i j = 0) :
    H.submatrix (Equiv.sigmaFiberEquiv component) (Equiv.sigmaFiberEquiv component) =
      blockDiagonal' (fun a => H.submatrix
        (fun i : {i // component i = a} => i.val)
        (fun i : {i // component i = a} => i.val)) := by
  ext ⟨a, i⟩ ⟨b, j⟩
  by_cases hab : a = b
  · subst b
    simp [Matrix.submatrix, Equiv.sigmaFiberEquiv]
  · rw [Matrix.blockDiagonal'_apply_ne _ _ _ hab]
    exact hzero i.val j.val (by simpa only [i.property, j.property] using hab)

end Partition

section Families

variable {k : Type*} {d : k → Type*}
variable [Fintype k] [DecidableEq k] [∀ a, Fintype (d a)] [∀ a, DecidableEq (d a)]

theorem family_inverse (A : ∀ a, Matrix (d a) (d a) ℝ)
    (hA : ∀ a, IsUnit (A a).det) :
    (blockDiagonal' A)⁻¹ = blockDiagonal' (fun a => (A a)⁻¹) := by
  apply Matrix.inv_eq_left_inv
  rw [← Matrix.blockDiagonal'_mul]
  simp only [Matrix.nonsing_inv_mul _ (hA _)]
  exact Matrix.blockDiagonal'_one

theorem family_determinant {R : Type*} [CommRing R]
    (A : ∀ a, Matrix (d a) (d a) R) :
    (blockDiagonal' A).det = ∏ a, (A a).det := by
  let : LinearOrder k := LinearOrder.lift' (Fintype.equivFin k)
    (Fintype.equivFin k).injective
  have ht : (blockDiagonal' A).BlockTriangular Sigma.fst := by
    intro i j hij
    exact Matrix.blockDiagonal'_apply_ne A i.2 j.2 (ne_of_gt hij)
  rw [ht.det_fintype]
  apply Finset.prod_congr rfl
  intro a _
  rw [← Matrix.det_reindex_self (Equiv.sigmaSubtype a)]
  congr 1
  ext i j
  simp [Matrix.reindex_apply, Matrix.toSquareBlock_def, Matrix.submatrix,
    Equiv.sigmaSubtype]

theorem family_charpoly (A : ∀ a, Matrix (d a) (d a) ℝ) :
    (blockDiagonal' A).charpoly = ∏ a, (A a).charpoly := by
  have h : (blockDiagonal' A).charmatrix = blockDiagonal' (fun a => (A a).charmatrix) := by
    ext ⟨a, i⟩ ⟨b, j⟩
    by_cases hab : a = b
    · subst b
      simp [Matrix.charmatrix, Matrix.blockDiagonal'_apply, Matrix.diagonal_apply]
    · simp [Matrix.charmatrix, Matrix.blockDiagonal'_apply, hab]
  simp only [Matrix.charpoly, h, family_determinant]

theorem family_spectrum (A : ∀ a, Matrix (d a) (d a) ℝ) (t : ℝ) :
    t ∈ spectrum ℝ (blockDiagonal' A) ↔ ∃ a, t ∈ spectrum ℝ (A a) := by
  simp only [Matrix.mem_spectrum_iff_isRoot_charpoly, Polynomial.IsRoot,
    family_charpoly, Polynomial.eval_prod, Finset.prod_eq_zero_iff,
    Finset.mem_univ, true_and]

theorem family_nonsingular_iff (A : ∀ a, Matrix (d a) (d a) ℝ) :
    IsUnit (blockDiagonal' A).det ↔ ∀ a, IsUnit (A a).det := by
  simp [family_determinant, isUnit_iff_ne_zero, Finset.prod_ne_zero_iff]

end Families

section PartitionInverse

variable {n k : Type*} [Fintype n] [DecidableEq n] [Fintype k] [DecidableEq k]

theorem partition_reindex_inverse (H : Matrix n n ℝ) (component : n → k)
    (hzero : ∀ i j, component i ≠ component j → H i j = 0)
    (hblocks : ∀ a, IsUnit (H.submatrix
      (fun i : {i // component i = a} => i.val)
      (fun i : {i // component i = a} => i.val)).det) :
    H⁻¹.submatrix (Equiv.sigmaFiberEquiv component) (Equiv.sigmaFiberEquiv component) =
      blockDiagonal' (fun a => (H.submatrix
        (fun i : {i // component i = a} => i.val)
        (fun i : {i // component i = a} => i.val))⁻¹) := by
  rw [← Matrix.inv_submatrix_equiv, partition_reindex H component hzero]
  exact family_inverse _ hblocks

end PartitionInverse

section EqualSizedFamilies

variable {k n : Type*} [Fintype k] [DecidableEq k]
variable [Fintype n] [DecidableEq n]

theorem uniform_family_inverse (A : k → Matrix n n ℝ)
    (hA : ∀ a, IsUnit (A a).det) :
    (blockDiagonal A)⁻¹ = blockDiagonal (fun a => (A a)⁻¹) := by
  apply Matrix.inv_eq_left_inv
  rw [← Matrix.blockDiagonal_mul]
  simp only [Matrix.nonsing_inv_mul _ (hA _)]
  exact Matrix.blockDiagonal_one

theorem uniform_family_charpoly (A : k → Matrix n n ℝ) :
    (blockDiagonal A).charpoly = ∏ a, (A a).charpoly := by
  have h : (blockDiagonal A).charmatrix = blockDiagonal (fun a => (A a).charmatrix) := by
    ext ⟨i, a⟩ ⟨j, b⟩
    by_cases hab : a = b
    · subst b
      simp [Matrix.charmatrix, Matrix.blockDiagonal_apply, Matrix.diagonal_apply]
    · simp [Matrix.charmatrix, Matrix.blockDiagonal_apply, hab]
  simp only [Matrix.charpoly, h, Matrix.det_blockDiagonal]

theorem uniform_family_spectrum (A : k → Matrix n n ℝ) (t : ℝ) :
    t ∈ spectrum ℝ (blockDiagonal A) ↔ ∃ a, t ∈ spectrum ℝ (A a) := by
  simp only [Matrix.mem_spectrum_iff_isRoot_charpoly, Polynomial.IsRoot,
    uniform_family_charpoly, Polynomial.eval_prod, Finset.prod_eq_zero_iff,
    Finset.mem_univ, true_and]

end EqualSizedFamilies

end CbmrProofs.Structure

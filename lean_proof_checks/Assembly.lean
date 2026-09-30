import lean_proof_checks.Hessian

/-! Matrix equalities connecting the coordinate derivatives to the appendix's block assembly. -/

namespace CbmrProofs.Assembly

open Matrix
open scoped BigOperators Kronecker

set_option backward.isDefEq.respectTransparency false

variable {I J R P : Type*}
variable [Fintype I] [Fintype J] [Fintype R] [Fintype P]

def studyDesign (Z : Matrix I R ℝ) (B : Matrix J P ℝ) (i : I) :
    Matrix J (R × P) ℝ :=
  fun j rp => Z i rp.1 * B j rp.2

theorem subjectwise_score_projection (Z : Matrix I R ℝ) (B : Matrix J P ℝ)
    (score : I → J → ℝ) :
    (Z ⊗ₖ B)ᵀ *ᵥ (fun ij => score ij.1 ij.2) =
      ∑ i, (studyDesign Z B i)ᵀ *ᵥ score i := by
  ext ⟨r, p⟩
  simp [Matrix.mulVec, dotProduct, Matrix.transpose_apply, Matrix.kroneckerMap_apply,
    Fintype.sum_prod_type, studyDesign, Finset.sum_apply]

theorem sv_weighted_gram [DecidableEq I] [DecidableEq J]
    (Z : Matrix I R ℝ) (B : Matrix J P ℝ) (w : I → J → ℝ) :
    (Z ⊗ₖ B)ᵀ * Matrix.diagonal (fun ij => w ij.1 ij.2) * (Z ⊗ₖ B) =
      ∑ i, Matrix.vecMulVec (Z i) (Z i) ⊗ₖ
        (Bᵀ * Matrix.diagonal (w i) * B) := by
  ext ⟨r, p⟩ ⟨s, q⟩
  rw [Hessian.weighted_gram_entry]
  simp only [Matrix.sum_apply, Matrix.kroneckerMap_apply, Matrix.vecMulVec_apply,
    Hessian.weighted_gram_entry, Fintype.sum_prod_type, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem sv_hessian_kronecker_blocks [DecidableEq J]
    (f df : I → J → ℝ → ℝ) (eta w : I → J → ℝ)
    (Z : Matrix I R ℝ) (B : Matrix J P ℝ) (r s : R) (p q : P)
    (hf : ∀ i j x, HasDerivAt (f i j) (df i j x) x)
    (hdf : ∀ i j, HasDerivAt (df i j) (w i j) (eta i j)) :
    HasDerivAt (fun t => deriv (fun u => ∑ i, ∑ j,
      f i j (eta i j + u * (Z i r * B j p) + t * (Z i s * B j q))) 0)
      ((∑ i, Matrix.vecMulVec (Z i) (Z i) ⊗ₖ
        (Bᵀ * Matrix.diagonal (w i) * B)) (r, p) (s, q)) 0 := by
  have h := Hessian.sv_hessian_entry_hasDerivAt f df eta w Z B r s p q hf hdf
  convert! h using 1
  simp only [Matrix.sum_apply, Matrix.kroneckerMap_apply, Matrix.vecMulVec_apply,
    Hessian.weighted_gram_entry, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

end CbmrProofs.Assembly

import Mathlib.Analysis.CStarAlgebra.Matrix
import Mathlib.Analysis.CStarAlgebra.Basic
import Mathlib.Analysis.Matrix.Spectrum
import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FinCases
import lean_proof_checks.Structure

/-! # Euclidean operator-norm conditioning of symmetric real matrices

`condition₂` uses Mathlib's induced Euclidean operator norm, not an eigenvalue
ratio as its definition. Nonsingularity is required by the ratio theorem.
-/

namespace CbmrProofs.Spectral

open Matrix
open scoped Matrix.Norms.L2Operator

variable {n : Type*} [Fintype n] [DecidableEq n]

noncomputable def condition₂ (H : Matrix n n ℝ) : ℝ := ‖H‖ * ‖H⁻¹‖

section Extrema

variable [Nonempty n]

noncomputable def maxAbs (v : n → ℝ) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty (fun i => |v i|)

noncomputable def minAbs (v : n → ℝ) : ℝ :=
  Finset.univ.inf' Finset.univ_nonempty (fun i => |v i|)

omit [DecidableEq n] in
theorem abs_le_maxAbs (v : n → ℝ) (i : n) : |v i| ≤ maxAbs v :=
  Finset.le_sup' (fun i => |v i|) (Finset.mem_univ i)

omit [DecidableEq n] in
theorem minAbs_le_abs (v : n → ℝ) (i : n) : minAbs v ≤ |v i| :=
  Finset.inf'_le _ (Finset.mem_univ i)

omit [DecidableEq n] in
theorem norm_eq_maxAbs (v : n → ℝ) : ‖v‖ = maxAbs v := by
  apply le_antisymm
  · apply (pi_norm_le_iff_of_nonneg
      ((abs_nonneg (v (Classical.arbitrary n))).trans
        (abs_le_maxAbs v (Classical.arbitrary n)))).mpr
    intro i
    exact abs_le_maxAbs v i
  · exact Finset.sup'_le _ _ (fun i _ => norm_le_pi_norm v i)

omit [DecidableEq n] in
theorem minAbs_pos (v : n → ℝ) (hv : ∀ i, v i ≠ 0) : 0 < minAbs v := by
  obtain ⟨i, _, hi⟩ := Finset.exists_mem_eq_inf' Finset.univ_nonempty (fun i => |v i|)
  rw [minAbs, hi]
  exact abs_pos.mpr (hv i)

omit [DecidableEq n] in
theorem norm_inv_eq_inv_minAbs (v : n → ℝ) (hv : ∀ i, v i ≠ 0) :
    ‖(fun i => (v i)⁻¹)‖ = (minAbs v)⁻¹ := by
  apply le_antisymm
  · apply (pi_norm_le_iff_of_nonneg (inv_nonneg.mpr (minAbs_pos v hv).le)).mpr
    intro i
    rw [Real.norm_eq_abs, abs_inv]
    simpa only [one_div] using
      one_div_le_one_div_of_le (minAbs_pos v hv) (minAbs_le_abs v i)
  · obtain ⟨i, _, hi⟩ := Finset.exists_mem_eq_inf' Finset.univ_nonempty (fun i => |v i|)
    calc
      (minAbs v)⁻¹ = ‖(v i)⁻¹‖ := by rw [minAbs, hi, Real.norm_eq_abs, abs_inv]
      _ ≤ _ := norm_le_pi_norm (fun i => (v i)⁻¹) i

end Extrema

theorem unitary_conjugate_norm (Q : Matrix.unitaryGroup n ℝ) (A : Matrix n n ℝ) :
    ‖Unitary.conjStarAlgAut ℝ (Matrix n n ℝ) Q A‖ = ‖A‖ := by
  simp only [Unitary.conjStarAlgAut_apply, ← Unitary.coe_star,
    CStarRing.norm_mul_coe_unitary, CStarRing.norm_coe_unitary_mul]

theorem symmetric_norm_eigenvalues (H : Matrix n n ℝ) (hH : H.IsHermitian) :
    ‖H‖ = ‖hH.eigenvalues‖ := by
  conv_lhs => rw [hH.spectral_theorem]
  rw [unitary_conjugate_norm]
  exact Matrix.l2_opNorm_diagonal hH.eigenvalues

theorem eigenvalues_ne_zero (H : Matrix n n ℝ) (hH : H.IsHermitian)
    (hdet : IsUnit H.det) : ∀ i, hH.eigenvalues i ≠ 0 := by
  have h : (∏ i, hH.eigenvalues i) ≠ 0 := by
    simpa only [hH.det_eq_prod_eigenvalues, RCLike.ofReal_real_eq_id, id_eq] using
      hdet.ne_zero
  exact fun i => (Finset.prod_ne_zero_iff.mp h) i (Finset.mem_univ i)

theorem symmetric_inverse_diagonalization (H : Matrix n n ℝ) (hH : H.IsHermitian)
    (hdet : IsUnit H.det) :
    H⁻¹ = Unitary.conjStarAlgAut ℝ (Matrix n n ℝ) hH.eigenvectorUnitary
      (diagonal (fun i => (hH.eigenvalues i)⁻¹)) := by
  apply Matrix.inv_eq_left_inv
  conv_lhs => arg 2; rw [hH.spectral_theorem]
  rw [← map_mul, Matrix.diagonal_mul_diagonal]
  simp [RCLike.ofReal_real_eq_id, eigenvalues_ne_zero H hH hdet]

theorem condition₂_eq_eigenvalue_ratio [Nonempty n] (H : Matrix n n ℝ)
    (hH : H.IsHermitian) (hdet : IsUnit H.det) :
    condition₂ H = maxAbs hH.eigenvalues / minAbs hH.eigenvalues := by
  rw [condition₂, symmetric_norm_eigenvalues H hH, norm_eq_maxAbs,
    symmetric_inverse_diagonalization H hH hdet, unitary_conjugate_norm,
    Matrix.l2_opNorm_diagonal,
    norm_inv_eq_inv_minAbs _ (eigenvalues_ne_zero H hH hdet), div_eq_mul_inv]

/-- A zero eigenvalue precludes a two-sided inverse; the totalized matrix inverse
and `condition₂` must not be interpreted as finite conditioning in this case. -/
theorem zero_eigenvalue_not_invertible (H : Matrix n n ℝ) (hH : H.IsHermitian)
    (i : n) (hi : hH.eigenvalues i = 0) : ¬ IsUnit H.det := by
  intro hdet
  exact eigenvalues_ne_zero H hH hdet i hi

theorem singular_has_no_right_inverse (H : Matrix n n ℝ) (hH : ¬ IsUnit H.det) :
    ¬ ∃ B : Matrix n n ℝ, H * B = 1 := by
  rintro ⟨B, hB⟩
  apply hH
  apply isUnit_iff_ne_zero.mpr
  intro hz
  have hh := congrArg Matrix.det hB
  simp [Matrix.det_mul, hz] at hh

theorem gram_of_orthogonal_diagonalization (H Q : Matrix n n ℝ) (v : n → ℝ)
    (hQ : Qᵀ * Q = 1) (hH : H = Q * diagonal v * Qᵀ) :
    Hᵀ * H = Q * diagonal (fun i => v i ^ 2) * Qᵀ := by
  rw [hH]
  simp only [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.diagonal_transpose,
    Matrix.mul_assoc]
  rw [← Matrix.mul_assoc Qᵀ Q, hQ, Matrix.one_mul,
    ← Matrix.mul_assoc (diagonal v) (diagonal v),
    Matrix.diagonal_mul_diagonal]
  simp only [pow_two]

/-! The induced `ℓ¹` norm is the operator norm of the same matrix acting on
`PiLp 1`, so the following comparison is between actual operator norms. -/

noncomputable def toL1CLM (A : Matrix n n ℝ) :
    PiLp 1 (fun _ : n => ℝ) →L[ℝ] PiLp 1 (fun _ : n => ℝ) :=
  LinearMap.toContinuousLinearMap (Matrix.toLpLin 1 1 A)

theorem toL1CLM_apply (A : Matrix n n ℝ) (x : n → ℝ) :
    toL1CLM A (WithLp.toLp 1 x) = WithLp.toLp 1 (A *ᵥ x) := rfl

noncomputable def opNorm₁ (A : Matrix n n ℝ) : ℝ := ‖toL1CLM A‖

noncomputable def condition₁ (A : Matrix n n ℝ) : ℝ := opNorm₁ A * opNorm₁ A⁻¹

omit [DecidableEq n] in
theorem vector_norm₂_le_norm₁ (x : n → ℝ) :
    ‖WithLp.toLp 2 x‖ ≤ ‖WithLp.toLp 1 x‖ := by
  apply (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  rw [PiLp.norm_sq_eq_of_L2, PiLp.norm_eq_of_L1]
  exact Finset.sum_sq_le_sq_sum_of_nonneg (fun i _ => norm_nonneg (x i))

omit [DecidableEq n] in
theorem vector_norm₁_le_sqrt_card_mul_norm₂ (x : n → ℝ) :
    ‖WithLp.toLp 1 x‖ ≤ Real.sqrt (Fintype.card n) * ‖WithLp.toLp 2 x‖ := by
  apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (Real.sqrt_nonneg _) (norm_nonneg _))).mp
  rw [mul_pow, Real.sq_sqrt (Nat.cast_nonneg _), PiLp.norm_eq_of_L1,
    PiLp.norm_sq_eq_of_L2]
  exact_mod_cast (sq_sum_le_card_mul_sum_sq
    (s := Finset.univ) (f := fun i => ‖x i‖))

theorem l1_action_bound (A : Matrix n n ℝ) (x : n → ℝ) :
    ‖WithLp.toLp 1 (A *ᵥ x)‖ ≤ opNorm₁ A * ‖WithLp.toLp 1 x‖ := by
  simpa only [toL1CLM_apply, opNorm₁] using (toL1CLM A).le_opNorm (WithLp.toLp 1 x)

theorem l2_action_bound (A : Matrix n n ℝ) (x : n → ℝ) :
    ‖WithLp.toLp 2 (A *ᵥ x)‖ ≤ ‖A‖ * ‖WithLp.toLp 2 x‖ := by
  simpa only [Matrix.toEuclideanCLM_toLp, Matrix.l2_opNorm_toEuclideanCLM] using
    (Matrix.toEuclideanCLM (n := n) (𝕜 := ℝ) A).le_opNorm (WithLp.toLp 2 x)

theorem opNorm₂_le_sqrt_card_mul_opNorm₁ (A : Matrix n n ℝ) :
    ‖A‖ ≤ Real.sqrt (Fintype.card n) * opNorm₁ A := by
  have h (x : n → ℝ) :
      ‖WithLp.toLp 2 (A *ᵥ x)‖ ≤
        (Real.sqrt (Fintype.card n) * opNorm₁ A) * ‖WithLp.toLp 2 x‖ := by
    calc
      _ ≤ ‖WithLp.toLp 1 (A *ᵥ x)‖ := vector_norm₂_le_norm₁ _
      _ ≤ opNorm₁ A * ‖WithLp.toLp 1 x‖ := l1_action_bound A x
      _ ≤ opNorm₁ A * (Real.sqrt (Fintype.card n) * ‖WithLp.toLp 2 x‖) :=
        mul_le_mul_of_nonneg_left (vector_norm₁_le_sqrt_card_mul_norm₂ x)
          (norm_nonneg (toL1CLM A))
      _ = _ := by rw [mul_left_comm, mul_assoc]
  rw [← Matrix.l2_opNorm_toEuclideanCLM]
  refine (Matrix.toEuclideanCLM (n := n) (𝕜 := ℝ) A).opNorm_le_bound
    (mul_nonneg (Real.sqrt_nonneg _) (norm_nonneg (toL1CLM A))) ?_
  intro x
  simpa only [← Matrix.toEuclideanCLM_toLp, WithLp.toLp_ofLp] using h (WithLp.ofLp x)

theorem opNorm₁_le_sqrt_card_mul_opNorm₂ (A : Matrix n n ℝ) :
    opNorm₁ A ≤ Real.sqrt (Fintype.card n) * ‖A‖ := by
  have h (x : n → ℝ) :
      ‖WithLp.toLp 1 (A *ᵥ x)‖ ≤
        (Real.sqrt (Fintype.card n) * ‖A‖) * ‖WithLp.toLp 1 x‖ := by
    calc
      _ ≤ Real.sqrt (Fintype.card n) * ‖WithLp.toLp 2 (A *ᵥ x)‖ :=
        vector_norm₁_le_sqrt_card_mul_norm₂ _
      _ ≤ Real.sqrt (Fintype.card n) * (‖A‖ * ‖WithLp.toLp 2 x‖) :=
        mul_le_mul_of_nonneg_left (l2_action_bound A x) (Real.sqrt_nonneg _)
      _ ≤ Real.sqrt (Fintype.card n) * (‖A‖ * ‖WithLp.toLp 1 x‖) :=
        mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_left (vector_norm₂_le_norm₁ x) (norm_nonneg A))
          (Real.sqrt_nonneg _)
      _ = _ := (mul_assoc _ _ _).symm
  refine (toL1CLM A).opNorm_le_bound
    (mul_nonneg (Real.sqrt_nonneg _) (norm_nonneg A)) ?_
  intro x
  simpa only [← toL1CLM_apply, WithLp.toLp_ofLp] using h (WithLp.ofLp x)

theorem condition₁_le_card_mul_condition₂ (A : Matrix n n ℝ) :
    condition₁ A ≤ Fintype.card n * condition₂ A := by
  calc
    _ ≤ (Real.sqrt (Fintype.card n) * ‖A‖) *
        (Real.sqrt (Fintype.card n) * ‖A⁻¹‖) :=
      mul_le_mul (opNorm₁_le_sqrt_card_mul_opNorm₂ A)
        (opNorm₁_le_sqrt_card_mul_opNorm₂ A⁻¹) (norm_nonneg _)
        (mul_nonneg (Real.sqrt_nonneg _) (norm_nonneg _))
    _ = _ := by
      rw [mul_mul_mul_comm, ← pow_two, Real.sq_sqrt (Nat.cast_nonneg _)]
      rfl

theorem condition₂_le_card_mul_condition₁ (A : Matrix n n ℝ) :
    condition₂ A ≤ Fintype.card n * condition₁ A := by
  calc
    _ ≤ (Real.sqrt (Fintype.card n) * opNorm₁ A) *
        (Real.sqrt (Fintype.card n) * opNorm₁ A⁻¹) :=
      mul_le_mul (opNorm₂_le_sqrt_card_mul_opNorm₁ A)
        (opNorm₂_le_sqrt_card_mul_opNorm₁ A⁻¹) (norm_nonneg _)
        (mul_nonneg (Real.sqrt_nonneg _) (norm_nonneg _))
    _ = _ := by
      rw [mul_mul_mul_comm, ← pow_two, Real.sq_sqrt (Nat.cast_nonneg _)]
      rfl

theorem condition_norm_equivalence [Nonempty n] (A : Matrix n n ℝ)
    (_hA : IsUnit A.det) :
    (Fintype.card n : ℝ)⁻¹ * condition₁ A ≤ condition₂ A ∧
      condition₂ A ≤ Fintype.card n * condition₁ A := by
  refine ⟨?_, condition₂_le_card_mul_condition₁ A⟩
  rw [inv_mul_eq_div]
  exact (div_le_iff₀ (Nat.cast_pos.mpr Fintype.card_pos)).mpr
    (by simpa only [mul_comm] using condition₁_le_card_mul_condition₂ A)

section Families

variable {k : Type*} {d : k → Type*}
variable [Fintype k] [DecidableEq k] [Nonempty k]
variable [∀ a, Fintype (d a)] [∀ a, DecidableEq (d a)] [∀ a, Nonempty (d a)]

local instance : Nonempty (Sigma d) :=
  ⟨⟨Classical.arbitrary k, Classical.arbitrary (d (Classical.arbitrary k))⟩⟩

omit [DecidableEq k] [∀ a, DecidableEq (d a)] in
theorem maxAbs_sigma (v : ∀ a, d a → ℝ) :
    maxAbs (fun i : Sigma d => v i.1 i.2) =
      Finset.univ.sup' Finset.univ_nonempty (fun a => maxAbs (v a)) := by
  apply le_antisymm
  · apply Finset.sup'_le
    intro i _
    exact (abs_le_maxAbs (v i.1) i.2).trans
      (Finset.le_sup' (fun a => maxAbs (v a)) (Finset.mem_univ i.1))
  · apply Finset.sup'_le
    intro a _
    apply Finset.sup'_le
    intro i _
    exact abs_le_maxAbs (fun i : Sigma d => v i.1 i.2) ⟨a, i⟩

omit [DecidableEq k] [∀ a, DecidableEq (d a)] in
theorem minAbs_sigma (v : ∀ a, d a → ℝ) :
    minAbs (fun i : Sigma d => v i.1 i.2) =
      Finset.univ.inf' Finset.univ_nonempty (fun a => minAbs (v a)) := by
  apply le_antisymm
  · apply Finset.le_inf'
    intro a _
    apply Finset.le_inf'
    intro i _
    exact minAbs_le_abs (fun i : Sigma d => v i.1 i.2) ⟨a, i⟩
  · apply Finset.le_inf'
    intro i _
    exact (Finset.inf'_le (fun a => minAbs (v a)) (Finset.mem_univ i.1)).trans
      (minAbs_le_abs (v i.1) i.2)

omit [Fintype k] [Nonempty k] [∀ a, Fintype (d a)]
  [∀ a, DecidableEq (d a)] [∀ a, Nonempty (d a)] in
theorem family_isHermitian (A : ∀ a, Matrix (d a) (d a) ℝ)
    (hA : ∀ a, (A a).IsHermitian) : (blockDiagonal' A).IsHermitian := by
  change (blockDiagonal' A)ᴴ = blockDiagonal' A
  rw [Matrix.blockDiagonal'_conjTranspose]
  congr 1
  funext a
  exact hA a

omit [Nonempty k] [∀ a, Nonempty (d a)] in
theorem family_eigenvalue_range (A : ∀ a, Matrix (d a) (d a) ℝ)
    (hA : ∀ a, (A a).IsHermitian) :
    Set.range (family_isHermitian A hA).eigenvalues =
      Set.range (fun i : Sigma d => (hA i.1).eigenvalues i.2) := by
  rw [← (family_isHermitian A hA).spectrum_real_eq_range_eigenvalues]
  ext t
  simp only [Structure.family_spectrum, Set.mem_range]
  constructor
  · rintro ⟨a, ha⟩
    rw [(hA a).spectrum_real_eq_range_eigenvalues] at ha
    obtain ⟨i, hi⟩ := ha
    exact ⟨⟨a, i⟩, hi⟩
  · rintro ⟨⟨a, i⟩, hi⟩
    refine ⟨a, ?_⟩
    rw [(hA a).spectrum_real_eq_range_eigenvalues]
    exact ⟨i, hi⟩

theorem family_condition₂ (A : ∀ a, Matrix (d a) (d a) ℝ)
    (hA : ∀ a, (A a).IsHermitian) (hdet : ∀ a, IsUnit (A a).det) :
    condition₂ (blockDiagonal' A) =
      (Finset.univ.sup' Finset.univ_nonempty (fun a => maxAbs (hA a).eigenvalues)) /
        (Finset.univ.inf' Finset.univ_nonempty (fun a => minAbs (hA a).eigenvalues)) := by
  have hfull : IsUnit (blockDiagonal' A).det := by
    rw [Structure.family_determinant]
    exact isUnit_iff_ne_zero.mpr
      (Finset.prod_ne_zero_iff.mpr (fun a _ => (hdet a).ne_zero))
  rw [condition₂_eq_eigenvalue_ratio _ (family_isHermitian A hA) hfull]
  have hr := family_eigenvalue_range A hA
  have hmax :
      maxAbs (family_isHermitian A hA).eigenvalues =
        maxAbs (fun i : Sigma d => (hA i.1).eigenvalues i.2) := by
    apply le_antisymm <;> apply Finset.sup'_le <;> intro i _
    · obtain ⟨j, hj⟩ := hr ▸ Set.mem_range_self i
      rw [← hj]
      exact abs_le_maxAbs (fun i : Sigma d => (hA i.1).eigenvalues i.2) j
    · obtain ⟨j, hj⟩ := hr.symm ▸ Set.mem_range_self i
      change |(hA i.1).eigenvalues i.2| ≤ _
      rw [← hj]
      exact abs_le_maxAbs (family_isHermitian A hA).eigenvalues j
  have hmin :
      minAbs (family_isHermitian A hA).eigenvalues =
        minAbs (fun i : Sigma d => (hA i.1).eigenvalues i.2) := by
    apply le_antisymm <;> apply Finset.le_inf' <;> intro i _
    · obtain ⟨j, hj⟩ := hr.symm ▸ Set.mem_range_self i
      change _ ≤ |(hA i.1).eigenvalues i.2|
      rw [← hj]
      exact minAbs_le_abs (family_isHermitian A hA).eigenvalues j
    · obtain ⟨j, hj⟩ := hr ▸ Set.mem_range_self i
      rw [← hj]
      exact minAbs_le_abs (fun i : Sigma d => (hA i.1).eigenvalues i.2) j
  rw [hmax, hmin, maxAbs_sigma (fun a => (hA a).eigenvalues),
    minAbs_sigma (fun a => (hA a).eigenvalues)]

end Families

theorem diagonal_condition₂ [Nonempty n] (v : n → ℝ) (hv : ∀ i, v i ≠ 0) :
    condition₂ (diagonal v) = maxAbs v / minAbs v := by
  have hinv : (diagonal v)⁻¹ = diagonal (fun i => (v i)⁻¹) := by
    apply Matrix.inv_eq_left_inv
    rw [Matrix.diagonal_mul_diagonal]
    simp [hv]
  rw [condition₂, hinv, Matrix.l2_opNorm_diagonal, Matrix.l2_opNorm_diagonal,
    norm_eq_maxAbs, norm_inv_eq_inv_minAbs v hv, div_eq_mul_inv]

/-- Individually perfectly conditioned scalar blocks can have a badly
conditioned block diagonal sum. All three condition numbers use operator norms. -/
theorem block_condition_not_max :
    condition₂ (diagonal ![(1 : ℝ), 100]) ≠
      max (condition₂ (diagonal (fun _ : Fin 1 => (1 : ℝ))))
        (condition₂ (diagonal (fun _ : Fin 1 => (100 : ℝ)))) := by
  rw [diagonal_condition₂ _ (by intro i; fin_cases i <;> norm_num),
    diagonal_condition₂ _ (by intro i; norm_num),
    diagonal_condition₂ _ (by intro i; norm_num)]
  norm_num [maxAbs, minAbs, Finset.univ_fin2, Finset.univ_unique,
    Finset.sup'_insert, Finset.inf'_insert]

/-- The leading block need not be invertible even when the full symmetric
bordered matrix is invertible. -/
theorem invertible_border_singular_leading_block :
    IsUnit (!![(0 : ℝ), 1; 1, 0]).det ∧
      ¬ IsUnit (!![(0 : ℝ)]).det := by
  norm_num [Matrix.det_fin_two, Matrix.det_fin_one]

/-- In this border `D = F = 1`, `E = 2`, both `D` and its Schur
complement have eigenvalue `1`, whereas the full matrix does not. -/
theorem schur_blocks_do_not_give_full_spectrum :
    (2 : ℝ) - 1 * 1⁻¹ * 1 = 1 ∧
      (1 : ℝ) ∈ spectrum ℝ (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
      (1 : ℝ) ∉ spectrum ℝ (!![(1 : ℝ), 1; 1, 2]) := by
  refine ⟨by norm_num, ?_, ?_⟩
  · rw [Matrix.mem_spectrum_iff_isRoot_charpoly, Polynomial.IsRoot,
      Matrix.eval_charpoly]
    norm_num [Matrix.det_fin_one, Matrix.scalar_apply]
  · rw [Matrix.mem_spectrum_iff_isRoot_charpoly, Polynomial.IsRoot,
      Matrix.eval_charpoly]
    norm_num [Matrix.det_fin_two, Matrix.scalar_apply]

end CbmrProofs.Spectral

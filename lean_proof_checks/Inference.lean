import Mathlib.Probability.Moments.Variance
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.FDeriv.Prod
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.MeasureTheory.Function.ConvergenceInDistribution
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.HasLaw
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Exact probability and linearization identities for S7

The covariance identities concern actual random vectors with finite second
moments and deterministic transformations. For nonlinear intensities the
covariance asserted is that of the *linearization*, not the exact nonlinear
finite-sample covariance. A delta-method or Wald limiting law additionally
requires distributional convergence and nonsingularity hypotheses; neither
follows from these identities. The plus-one Monte Carlo bounds do not assert
calibration of a fitted-null bootstrap.
-/

noncomputable section

open MeasureTheory ProbabilityTheory
open scoped BigOperators Matrix

namespace CbmrProofs.Inference

set_option backward.isDefEq.respectTransparency false
set_option backward.isDefEq.respectTransparency.types false

variable {Ω ι κ ρ : Type*} [MeasurableSpace Ω] [Fintype ι] [Fintype κ]
    [Fintype ρ] {P : Measure Ω} [IsProbabilityMeasure P]

/-- Entrywise covariance transformation for deterministic linear contrasts. -/
theorem covariance_linear_transform (X : ι → Ω → ℝ) (Y : κ → Ω → ℝ)
    (a : ι → ℝ) (b : κ → ℝ)
    (hX : ∀ i, MemLp (X i) 2 P) (hY : ∀ j, MemLp (Y j) 2 P) :
    covariance (fun ω => ∑ i, a i * X i ω)
      (fun ω => ∑ j, b j * Y j ω) P =
      ∑ i, ∑ j, a i * covariance (X i) (Y j) P * b j := by
  rw [covariance_fun_sum_fun_sum
    (fun i => (hX i).const_mul (a i)) (fun j => (hY j).const_mul (b j))]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  rw [covariance_const_mul_left, covariance_const_mul_right]
  ring

omit [Fintype κ] in
/-- This is `Cov(TX) = T Cov(X) Tᵀ`, not a definition of covariance. -/
theorem covariance_matrix_transform (X : ι → Ω → ℝ)
    (T : Matrix κ ι ℝ) (hX : ∀ i, MemLp (X i) 2 P) :
    (fun k l => covariance (fun ω => ∑ i, T k i * X i ω)
      (fun ω => ∑ i, T l i * X i ω) P) =
      T * Matrix.of (fun i j => covariance (X i) (X j) P) * T.transpose := by
  ext k l
  rw [covariance_linear_transform X X _ _ hX hX]
  simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.of_apply, Finset.sum_mul]
  exact Finset.sum_comm

theorem contrast_reference_covariance (X Y : Ω → ℝ) (a b : ℝ)
    (hX : Integrable X P) (hY : Integrable Y P) :
    covariance (fun ω => X ω - a) (fun ω => Y ω - b) P =
      covariance X Y P := by
  rw [covariance_sub_const_left hX, covariance_sub_const_right hY]

omit [Fintype κ] in
/-- Deterministic reference values and offsets change means, not covariance. -/
theorem covariance_affine_transform (X : ι → Ω → ℝ)
    (T : Matrix κ ι ℝ) (offset : κ → ℝ) (reference : ι → ℝ)
    (hX : ∀ i, MemLp (X i) 2 P) :
    Matrix.of (fun k l => covariance
      (fun ω => offset k + ∑ i, T k i * (X i ω - reference i))
      (fun ω => offset l + ∑ i, T l i * (X i ω - reference i)) P) =
        T * Matrix.of (fun i j => covariance (X i) (X j) P) * T.transpose := by
  let Y := fun i ω => X i ω - reference i
  have hY (i : ι) : MemLp (Y i) 2 P := (hX i).sub (memLp_const _)
  have hint (k : κ) : Integrable (fun ω => ∑ i, T k i * Y i ω) P :=
    (memLp_finsetSum _ (fun i _ => (hY i).const_mul (T k i))).integrable (by norm_num)
  have hc : Matrix.of (fun i j => covariance (Y i) (Y j) P) =
      Matrix.of (fun i j => covariance (X i) (X j) P) := by
    ext i j
    exact contrast_reference_covariance _ _ _ _
      ((hX i).integrable (by norm_num)) ((hX j).integrable (by norm_num))
  calc
    _ = Matrix.of (fun k l => covariance (fun ω => ∑ i, T k i * Y i ω)
        (fun ω => ∑ i, T l i * Y i ω) P) := by
      ext k l
      exact (covariance_const_add_left (hint k) (offset k)).trans
        (covariance_const_add_right (hint l) (offset l))
    _ = T * Matrix.of (fun i j => covariance (Y i) (Y j) P) * T.transpose :=
      covariance_matrix_transform Y T hY
    _ = _ := by rw [hc]

/-- The displayed penalized estimator solves the linearized score equation,
not necessarily the original nonlinear estimating equation. -/
theorem penalized_linearized_equation [DecidableEq ι]
    (H K : Matrix ι ι ℝ) (center score : ι → ℝ) (hA : IsUnit (H + K).det) :
    (H + K) *ᵥ ((center + (H + K)⁻¹ *ᵥ (score - K *ᵥ center)) - center) =
      score - K *ᵥ center := by
  rw [add_sub_cancel_left, Matrix.mulVec_mulVec,
    Matrix.mul_nonsing_inv _ hA, Matrix.one_mulVec]

/-- Exact covariance of the penalized estimating-equation linearization.
The score covariance is the unpenalized data-score covariance. -/
theorem penalized_linearization_covariance [DecidableEq ι]
    (U : ι → Ω → ℝ) (H K : Matrix ι ι ℝ) (center : ι → ℝ)
    (hU : ∀ i, MemLp (U i) 2 P) :
    Matrix.of (fun k l => covariance
      (fun ω => center k + ∑ i, (H + K)⁻¹ k i * (U i ω - (K *ᵥ center) i))
      (fun ω => center l + ∑ i, (H + K)⁻¹ l i * (U i ω - (K *ᵥ center) i)) P) =
      (H + K)⁻¹ * Matrix.of (fun i j => covariance (U i) (U j) P) * ((H + K)⁻¹).transpose :=
  covariance_affine_transform U (H + K)⁻¹ center (K *ᵥ center) hU

/-- Symmetry permits omitting the transpose in the model-based specialization. -/
theorem model_based_penalized_linearization_covariance [DecidableEq ι]
    (U : ι → Ω → ℝ) (I K : Matrix ι ι ℝ) (center : ι → ℝ)
    (hU : ∀ i, MemLp (U i) 2 P)
    (hI : Matrix.of (fun i j => covariance (U i) (U j) P) = I)
    (hsym : (I + K).transpose = I + K) :
    Matrix.of (fun k l => covariance
      (fun ω => center k + ∑ i, (I + K)⁻¹ k i * (U i ω - (K *ᵥ center) i))
      (fun ω => center l + ∑ i, (I + K)⁻¹ l i * (U i ω - (K *ᵥ center) i)) P) =
      (I + K)⁻¹ * I * (I + K)⁻¹ := by
  rw [penalized_linearization_covariance U I K center hU, hI,
    Matrix.transpose_nonsing_inv, hsym]

/-- The unpenalized sandwich is the covariance of the corresponding
linearized estimating-equation solution, with deterministic bread. -/
theorem estimating_equation_linearization_covariance [DecidableEq ι]
    (U : ι → Ω → ℝ) (A : Matrix ι ι ℝ) (center : ι → ℝ)
    (hU : ∀ i, MemLp (U i) 2 P) :
    Matrix.of (fun k l => covariance
      (fun ω => center k + ∑ i, A⁻¹ k i * U i ω)
      (fun ω => center l + ∑ i, A⁻¹ l i * U i ω) P) =
      A⁻¹ * Matrix.of (fun i j => covariance (U i) (U j) P) * (A⁻¹).transpose := by
  simpa only [Pi.zero_apply, sub_zero] using
    covariance_affine_transform U A⁻¹ center 0 hU

omit [Fintype ι] in
/-- Independence is required between subject score vectors, not between
their coordinates. Within-subject score cross terms are retained. -/
theorem independent_subject_score_covariance
    (U : ρ → ι → Ω → ℝ) (hU : ∀ r i, MemLp (U r i) 2 P)
    (hind : Pairwise (fun r s => IndepFun (fun ω i => U r i ω) (fun ω i => U s i ω) P))
    (i j : ι) :
    covariance (fun ω => ∑ r, U r i ω) (fun ω => ∑ r, U r j ω) P =
      ∑ r, covariance (U r i) (U r j) P := by
  classical
  rw [covariance_fun_sum_fun_sum (fun r => hU r i) (fun r => hU r j)]
  apply Finset.sum_congr rfl
  intro r _
  apply Finset.sum_eq_single r
  · intro s _ hsr
    have h : IndepFun (U r i) (U s j) P :=
      (hind hsr.symm).comp (measurable_pi_apply i) (measurable_pi_apply j)
    exact h.covariance_eq_zero (hU r i) (hU s j)
  · simp

/-- Both cross terms are retained even if the two groups share coefficients. -/
theorem group_difference_variance (X Y : Ω → ℝ)
    (hX : MemLp X 2 P) (hY : MemLp Y 2 P) :
    variance (fun ω => X ω - Y ω) P =
      variance X P + variance Y P - covariance X Y P - covariance Y X P := by
  rw [variance_fun_sub hX hY, covariance_comm (X := Y) (Y := X)]
  ring

theorem group_map_difference_variance (X Y : ι → Ω → ℝ) (b : ι → ℝ)
    (hX : ∀ i, MemLp (X i) 2 P) (hY : ∀ i, MemLp (Y i) 2 P) :
    variance (fun ω => ∑ i, b i * (X i ω - Y i ω)) P =
      ∑ i, ∑ j, b i *
        (covariance (X i) (X j) P + covariance (Y i) (Y j) P -
          covariance (X i) (Y j) P - covariance (Y i) (X j) P) * b j := by
  have hdiff : ∀ i, MemLp (fun ω => X i ω - Y i ω) 2 P :=
    fun i => (hX i).sub (hY i)
  have hs : MemLp (fun ω => ∑ i, b i * (X i ω - Y i ω)) 2 P :=
    memLp_finsetSum _ (fun i _ => (hdiff i).const_mul (b i))
  rw [← covariance_self hs.aemeasurable,
    covariance_linear_transform _ _ _ _ hdiff hdiff]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  rw [covariance_fun_sub_fun_sub (hX i) (hY i) (hX j) (hY j)]
  ring

omit [Fintype κ] in
/-- Projection of the empirical sandwich is a sum of projected score outer products. -/
theorem projected_sandwich_outer_products (A : Matrix κ ι ℝ)
    (U : ρ → ι → ℝ) (k l : κ) :
    (A * Matrix.of (fun i j => ∑ r, U r i * U r j) * A.transpose) k l =
      ∑ r, (∑ i, A k i * U r i) * (∑ j, A l j * U r j) := by
  simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.of_apply,
    Finset.mul_sum, Finset.sum_mul]
  calc
    _ = ∑ j, ∑ i, ∑ r, (A k i * U r i) * (A l j * U r j) := by
      apply Finset.sum_congr rfl
      intro j _
      apply Finset.sum_congr rfl
      intro i _
      apply Finset.sum_congr rfl
      intro r _
      ring
    _ = ∑ j, ∑ r, ∑ i, (A k i * U r i) * (A l j * U r j) := by
      apply Finset.sum_congr rfl
      intro j _
      exact Finset.sum_comm
    _ = ∑ r, ∑ j, ∑ i, (A k i * U r i) * (A l j * U r j) :=
      Finset.sum_comm
    _ = _ := by
      apply Finset.sum_congr rfl
      intro r _
      rfl

/-- The continuous linear map represented by a matrix, in finite coordinates. -/
def matrixLinearMap (B : Matrix κ ι ℝ) : (ι → ℝ) →L[ℝ] (κ → ℝ) :=
  ContinuousLinearMap.pi (fun j => ∑ i, B j i • ContinuousLinearMap.proj i)

omit [Fintype κ] in
theorem matrixLinearMap_apply (B : Matrix κ ι ℝ) (x : ι → ℝ) (j : κ) :
    matrixLinearMap B x j = ∑ i, B j i * x i := by
  simp [matrixLinearMap]

omit [Fintype κ] in
/-- Actual Fréchet derivative, with Jacobian `diag(exp(Bx)) B`. -/
theorem exp_matrix_hasFDerivAt (B : Matrix κ ι ℝ) (x : ι → ℝ) :
    HasFDerivAt (fun z j => Real.exp (matrixLinearMap B z j))
      (ContinuousLinearMap.pi (fun j =>
        Real.exp (matrixLinearMap B x j) •
          (ContinuousLinearMap.proj j).comp (matrixLinearMap B))) x := by
  apply hasFDerivAt_pi.mpr
  intro j
  exact (((ContinuousLinearMap.proj j).comp (matrixLinearMap B)).hasFDerivAt).exp

omit [Fintype κ] in
theorem exp_matrix_jacobian (B : Matrix κ ι ℝ) (x : ι → ℝ) :
    HasFDerivAt (fun z j => Real.exp (matrixLinearMap B z j))
      (matrixLinearMap (Matrix.of (fun j i => Real.exp (matrixLinearMap B x j) * B j i))) x := by
  convert! exp_matrix_hasFDerivAt B x using 1
  ext h j
  simp [matrixLinearMap_apply, Finset.mul_sum, mul_assoc]

theorem scalar_wald_square (contrast v : ℝ) (hv : 0 < v) :
    (contrast / Real.sqrt v) ^ 2 = contrast ^ 2 / v := by
  rw [div_pow, Real.sq_sqrt hv.le]

omit [Fintype κ] in
/-- Exact covariance of the first-order exponential linearization. -/
theorem exp_linearization_covariance (X : κ → Ω → ℝ) (eta : κ → ℝ)
    (hX : ∀ i, MemLp (X i) 2 P) (i j : κ) :
    covariance
      (fun ω => Real.exp (eta i) + Real.exp (eta i) * (X i ω - eta i))
      (fun ω => Real.exp (eta j) + Real.exp (eta j) * (X j ω - eta j)) P =
      Real.exp (eta i) * covariance (X i) (X j) P * Real.exp (eta j) := by
  have hi0 : Integrable (X i) P := (hX i).integrable (by norm_num)
  have hj0 : Integrable (X j) P := (hX j).integrable (by norm_num)
  have hi : Integrable (fun ω => Real.exp (eta i) * (X i ω - eta i)) P :=
    (hi0.sub (integrable_const (eta i))).const_mul _
  have hj : Integrable (fun ω => Real.exp (eta j) * (X j ω - eta j)) P :=
    (hj0.sub (integrable_const (eta j))).const_mul _
  rw [covariance_const_add_left hi,
    covariance_const_add_right hj,
    covariance_const_mul_left, covariance_const_mul_right,
    contrast_reference_covariance _ _ _ _ hi0 hj0]
  ring

/-- Bounds only: a fitted-null bootstrap is not automatically an exact test. -/
theorem monte_carlo_plus_one_bounds (n k : ℕ) (hk : k ≤ n) :
    0 < (1 + (k : ℝ)) / ((n : ℝ) + 1) ∧
    1 / ((n : ℝ) + 1) ≤ (1 + (k : ℝ)) / ((n : ℝ) + 1) ∧
    (1 + (k : ℝ)) / ((n : ℝ) + 1) ≤ 1 := by
  have hn : 0 < (n : ℝ) + 1 := by positivity
  refine ⟨div_pos (by positivity) hn, ?_, ?_⟩
  · exact (div_le_div_iff_of_pos_right hn).mpr (by linarith [Nat.cast_nonneg (α := ℝ) k])
  · apply (div_le_one hn).mpr
    exact_mod_cast (show 1 + k ≤ n + 1 by omega)

theorem monte_carlo_exceedance_bounds (n : ℕ) (observed : ℝ) (replicate : Fin n → ℝ) :
    0 < (1 + ((Finset.univ.filter (fun b => observed ≤ replicate b)).card : ℝ)) /
        ((n : ℝ) + 1) ∧
    1 / ((n : ℝ) + 1) ≤
      (1 + ((Finset.univ.filter (fun b => observed ≤ replicate b)).card : ℝ)) /
        ((n : ℝ) + 1) ∧
    (1 + ((Finset.univ.filter (fun b => observed ≤ replicate b)).card : ℝ)) /
        ((n : ℝ) + 1) ≤ 1 := by
  apply monte_carlo_plus_one_bounds
  exact (Finset.card_filter_le _ _).trans (by simp)

/-- Distributional delta step with its genuinely probabilistic remainder
hypothesis stated explicitly. Derivative identities alone do not give this
convergence-in-probability premise. -/
theorem delta_limit_with_remainder {Ω' : Type*} [MeasurableSpace Ω']
    (Q : Measure Ω') [IsProbabilityMeasure Q]
    (X : ℕ → Ω → (ι → ℝ)) (Z : Ω' → (ι → ℝ))
    (R : ℕ → Ω → (κ → ℝ)) (D : (ι → ℝ) →L[ℝ] (κ → ℝ))
    (hCLT : TendstoInDistribution X Filter.atTop Z (fun _ => P) Q)
    (hrem : TendstoInMeasure P R Filter.atTop (fun _ => 0))
    (hR : ∀ n, AEMeasurable (R n) P) :
    TendstoInDistribution (fun n ω => D (X n ω) + R n ω) Filter.atTop
      (fun ω => D (Z ω)) (fun _ => P) Q := by
  have h := (hCLT.continuous_comp D.continuous).add_of_tendstoInMeasure_const hrem hR
  simpa only [Function.comp_def, Pi.add_apply, add_zero] using! h

/-- The chi-square law represented by the sum of squares of `d` independent
standard normal coordinates. No finite-sample estimator law is asserted. -/
def squaredStandardNormalLaw (d : ℕ) : Measure ℝ :=
  (Measure.pi (fun _ : Fin d => gaussianReal 0 1)).map (fun z => ∑ i, z i ^ 2)

/-- A whitened contrast CLT implies the Wald squared-norm limit. This premise
includes the rank/nonsingularity and consistent-whitening obligations of a
statistical application, rather than deducing them from Hessian algebra. -/
theorem wald_limit_from_whitened_clt {Ω' : Type*} [MeasurableSpace Ω']
    (Q : Measure Ω') [IsProbabilityMeasure Q] (d : ℕ)
    (W : ℕ → Ω → (Fin d → ℝ)) (Z : Ω' → (Fin d → ℝ))
    (hCLT : TendstoInDistribution W Filter.atTop Z (fun _ => P) Q)
    (hZ : HasLaw Z (Measure.pi (fun _ : Fin d => gaussianReal 0 1)) Q) :
    TendstoInDistribution (fun n ω => ∑ i, W n ω i ^ 2) Filter.atTop
      (fun ω => ∑ i, Z ω i ^ 2) (fun _ => P) Q ∧
    HasLaw (fun ω => ∑ i, Z ω i ^ 2) (squaredStandardNormalLaw d) Q := by
  constructor
  · exact hCLT.continuous_comp (g := fun z : Fin d → ℝ => ∑ i, z i ^ 2) (by fun_prop)
  · have hs : HasLaw (fun z : Fin d → ℝ => ∑ i, z i ^ 2)
        (squaredStandardNormalLaw d) (Measure.pi (fun _ : Fin d => gaussianReal 0 1)) :=
      ⟨(show Continuous (fun z : Fin d → ℝ => ∑ i, z i ^ 2) by fun_prop).measurable.aemeasurable,
        rfl⟩
    exact hs.comp hZ

/-- A vector CLT plus consistency of the estimated whitening matrix gives
the studentized Wald limit. These are explicit probabilistic hypotheses;
neither is asserted to follow from the Hessian calculation. -/
theorem wald_limit_with_consistent_whitener {Ω' : Type*} [MeasurableSpace Ω']
    (Q : Measure Ω') [IsProbabilityMeasure Q] (d : ℕ)
    (X : ℕ → Ω → (ι → ℝ)) (Z : Ω' → (ι → ℝ))
    (W : ℕ → Ω → (Fin d → ι → ℝ)) (L : Fin d → ι → ℝ)
    (hCLT : TendstoInDistribution X Filter.atTop Z (fun _ => P) Q)
    (hW : TendstoInMeasure P W Filter.atTop (fun _ => L))
    (hWmeas : ∀ n, AEMeasurable (W n) P)
    (hNormal : HasLaw (fun ω k => ∑ i, L k i * Z ω i)
      (Measure.pi (fun _ : Fin d => gaussianReal 0 1)) Q) :
    TendstoInDistribution
      (fun n ω => ∑ k : Fin d, (∑ i, W n ω k i * X n ω i) ^ 2) Filter.atTop
      (fun ω => ∑ k : Fin d, (∑ i, L k i * Z ω i) ^ 2) (fun _ => P) Q ∧
    HasLaw (fun ω => ∑ k : Fin d, (∑ i, L k i * Z ω i) ^ 2)
      (squaredStandardNormalLaw d) Q := by
  constructor
  · have h := hCLT.continuous_comp_prodMk_of_tendstoInMeasure_const
      (g := fun z : (ι → ℝ) × (Fin d → ι → ℝ) =>
        ∑ k : Fin d, (∑ i, z.2 k i * z.1 i) ^ 2) (by fun_prop) hW hWmeas
    simpa only using! h
  · have hs : HasLaw (fun z : Fin d → ℝ => ∑ k, z k ^ 2)
        (squaredStandardNormalLaw d) (Measure.pi (fun _ : Fin d => gaussianReal 0 1)) :=
      ⟨(show Continuous (fun z : Fin d → ℝ => ∑ k, z k ^ 2) by fun_prop).measurable.aemeasurable,
        rfl⟩
    exact hs.comp hNormal

end CbmrProofs.Inference

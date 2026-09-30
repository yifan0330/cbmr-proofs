import lean_proof_checks.Calculus
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.Kronecker
import Mathlib.Tactic.Linarith

/-!
# Actual affine-design likelihood derivatives

All indices are arbitrary finite types. Mixed derivatives below differentiate
the first derivative of the likelihood, whose existence is proved separately.
Thus no convention for the derivative of a nondifferentiable function is used.
-/

namespace CbmrProofs.Hessian

set_option autoImplicit false

open Finset Real
open CbmrProofs.Calculus
open scoped Matrix Kronecker

set_option backward.isDefEq.respectTransparency false
set_option backward.isDefEq.respectTransparency.types false

variable {I P G : Type*} [Fintype I] [Fintype P] [Fintype G]

theorem finite_sum_hasDerivAt (f : I → ℝ → ℝ) (v : I → ℝ) (x : ℝ)
    (h : ∀ i, HasDerivAt (f i) (v i) x) :
    HasDerivAt (fun t => ∑ i, f i t) (∑ i, v i) x := by
  exact HasDerivAt.fun_sum (fun i _ => h i)

theorem summed_affine_hasDerivAt (f : I → ℝ → ℝ) (eta a value : I → ℝ)
    (hf : ∀ i, HasDerivAt (f i) (value i) (eta i)) :
    HasDerivAt (fun t => ∑ i, f i (eta i + t * a i))
      (∑ i, a i * value i) 0 :=
  finite_sum_hasDerivAt _ _ _ (fun i =>
    affine_chain_hasDerivAt (f i) (eta i) (a i) (value i) (hf i))

theorem coordinate_gradient_hasDerivAt (f : I → ℝ → ℝ) (eta value : I → ℝ)
    (X : Matrix I P ℝ) (p : P)
    (hf : ∀ i, HasDerivAt (f i) (value i) (eta i)) :
    HasDerivAt (fun t => ∑ i, f i (eta i + t * X i p))
      ((X.transpose *ᵥ value) p) 0 := by
  simpa [Matrix.mulVec, dotProduct, Matrix.transpose_apply] using
    summed_affine_hasDerivAt f eta (fun i => X i p) value hf

theorem summed_affine_mixed_hasDerivAt (f df : I → ℝ → ℝ)
    (eta a b w : I → ℝ)
    (hf : ∀ i x, HasDerivAt (f i) (df i x) x)
    (hdf : ∀ i, HasDerivAt (df i) (w i) (eta i)) :
    HasDerivAt
      (fun t => deriv (fun s => ∑ i, f i (eta i + s * a i + t * b i)) 0)
      (∑ i, a i * b i * w i) 0 := by
  have hid : (fun t => deriv
      (fun s => ∑ i, f i (eta i + s * a i + t * b i)) 0) =
      (fun t => ∑ i, a i * df i (eta i + t * b i)) := by
    funext t
    exact (finite_sum_hasDerivAt _ _ _ (fun i =>
      (affine_chain_mixed_hasDerivAt (f i) (df i) (eta i) (a i) (b i) (w i)
        (hf i) (hdf i)).1 t)).deriv
  rw [hid]
  exact finite_sum_hasDerivAt _ _ _ (fun i =>
    (affine_chain_mixed_hasDerivAt (f i) (df i) (eta i) (a i) (b i) (w i)
      (hf i) (hdf i)).2)

theorem weighted_gram_entry [DecidableEq I]
    (X : Matrix I P ℝ) (w : I → ℝ) (p q : P) :
    (X.transpose * Matrix.diagonal w * X) p q =
      ∑ i, X i p * X i q * w i := by
  classical
  rw [Matrix.mul_apply]
  simp only [Matrix.mul_diagonal, Matrix.transpose_apply]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- The mixed coordinate derivative is an entry of the weighted Gram matrix. -/
theorem coordinate_hessian_hasDerivAt [DecidableEq I] (f df : I → ℝ → ℝ)
    (eta w : I → ℝ) (X : Matrix I P ℝ) (p q : P)
    (hf : ∀ i x, HasDerivAt (f i) (df i x) x)
    (hdf : ∀ i, HasDerivAt (df i) (w i) (eta i)) :
    HasDerivAt
      (fun t => deriv (fun s =>
        ∑ i, f i (eta i + s * X i p + t * X i q)) 0)
      ((X.transpose * Matrix.diagonal w * X) p q) 0 := by
  rw [weighted_gram_entry]
  exact summed_affine_mixed_hasDerivAt f df eta _ _ w hf hdf

theorem poisson_directional_hasDerivAt (y eta a : I → ℝ) :
    HasDerivAt (fun t => ∑ i,
      (exp (eta i + t * a i) - y i * (eta i + t * a i)))
      (∑ i, a i * (exp (eta i) - y i)) 0 := by
  convert! summed_affine_hasDerivAt (fun i x => exp x - y i * x) eta a
    (fun i => exp (eta i) - y i) (fun i => poisson_nll_hasDerivAt (y i) (eta i)) using 1

theorem poisson_coordinate_hessian_hasDerivAt [DecidableEq I] (y eta : I → ℝ)
    (X : Matrix I P ℝ) (p q : P) :
    HasDerivAt
      (fun t => deriv (fun s => ∑ i,
        (exp (eta i + s * X i p + t * X i q) -
          y i * (eta i + s * X i p + t * X i q))) 0)
      ((X.transpose * Matrix.diagonal (fun i => exp (eta i)) * X) p q) 0 := by
  convert! coordinate_hessian_hasDerivAt _ _ _ _ X p q
    (fun i x => poisson_nll_hasDerivAt (y i) x)
    (fun i => poisson_nll_second_hasDerivAt (y i) (eta i)) using 1

theorem nb_directional_hasDerivAt (alpha y eta a : I → ℝ)
    (ha : ∀ i, 0 < alpha i) :
    HasDerivAt (fun t => ∑ i,
      (-y i * (eta i + t * a i) +
      (y i + 1 / alpha i) * log (1 + alpha i * exp (eta i + t * a i))))
      (∑ i, a i * ((exp (eta i) - y i) / (1 + alpha i * exp (eta i)))) 0 := by
  convert! summed_affine_hasDerivAt
    (fun i x => -y i * x + (y i + 1 / alpha i) * log (1 + alpha i * exp x))
    eta a (fun i => (exp (eta i) - y i) / (1 + alpha i * exp (eta i)))
    (fun i => nb_nll_hasDerivAt _ _ _ (ha i)) using 1

theorem poisson_coefficient_score_hasDerivAt (y eta : I → ℝ)
    (X : Matrix I P ℝ) (p : P) :
    HasDerivAt (fun t => ∑ i,
      (y i * (eta i + t * X i p) - exp (eta i + t * X i p)))
      ((X.transpose *ᵥ (fun i => y i - exp (eta i))) p) 0 := by
  have hi (i : I) : HasDerivAt (fun t => y i * t - exp t)
      (y i - exp (eta i)) (eta i) := by
    simpa using poisson_loglik_hasDerivAt (y i) 0 (eta i)
  have h := summed_affine_hasDerivAt (fun i t => y i * t - exp t) eta
    (fun i => X i p) (fun i => y i - exp (eta i)) hi
  simpa [Matrix.mulVec, dotProduct, Matrix.transpose_apply] using h

theorem nb_coefficient_score_hasDerivAt (alpha y eta : I → ℝ)
    (X : Matrix I P ℝ) (p : P) (ha : ∀ i, 0 < alpha i) :
    HasDerivAt (fun t => ∑ i,
      (y i * (eta i + t * X i p) -
        (y i + 1 / alpha i) * log (1 + alpha i * exp (eta i + t * X i p))))
      ((X.transpose *ᵥ (fun i =>
        (y i - exp (eta i)) / (1 + alpha i * exp (eta i)))) p) 0 := by
  have hi (i : I) : HasDerivAt
      (fun t => y i * t - (y i + 1 / alpha i) * log (1 + alpha i * exp t))
      ((y i - exp (eta i)) / (1 + alpha i * exp (eta i))) (eta i) := by
    convert! (nb_nll_hasDerivAt (alpha i) (y i) (eta i) (ha i)).neg using 1
    · funext t
      simp only [Pi.neg_apply]
      ring
    · ring
  have h := summed_affine_hasDerivAt
    (fun i t => y i * t - (y i + 1 / alpha i) * log (1 + alpha i * exp t))
    eta (fun i => X i p)
    (fun i => (y i - exp (eta i)) / (1 + alpha i * exp (eta i))) hi
  simpa [Matrix.mulVec, dotProduct, Matrix.transpose_apply] using h

theorem nb_coordinate_hessian_hasDerivAt [DecidableEq I] (alpha y eta : I → ℝ)
    (X : Matrix I P ℝ) (p q : P) (ha : ∀ i, 0 < alpha i) :
    HasDerivAt
      (fun t => deriv (fun s => ∑ i,
        (-y i * (eta i + s * X i p + t * X i q) +
        (y i + 1 / alpha i) *
          log (1 + alpha i * exp (eta i + s * X i p + t * X i q)))) 0)
      (((X.transpose * Matrix.diagonal (fun i : I =>
        exp (eta i) * (1 + alpha i * y i) /
          (1 + alpha i * exp (eta i)) ^ 2) * X) : Matrix P P ℝ) p q) 0 := by
  convert! coordinate_hessian_hasDerivAt _ _ _ _ X p q
    (fun i x => nb_nll_hasDerivAt (alpha i) (y i) x (ha i))
    (fun i => nb_nll_second_hasDerivAt (alpha i) (y i) (eta i) (ha i)) using 1

/-- Summing weights over exactly equal design rows changes no Gram entry. -/
theorem repeated_design_rows [DecidableEq G] (row : G → P → ℝ) (group : I → G)
    (w : I → ℝ) (p q : P) :
    (∑ i, row (group i) p * row (group i) q * w i) =
      ∑ g, row g p * row g q * ∑ i, if group i = g then w i else 0 := by
  classical
  simp_rw [Finset.mul_sum, mul_ite, mul_zero]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  simp

theorem repeated_row_hessian_hasDerivAt [DecidableEq G]
    (f df : I → ℝ → ℝ) (eta w : I → ℝ)
    (row : G → P → ℝ) (group : I → G) (p q : P)
    (hf : ∀ i x, HasDerivAt (f i) (df i x) x)
    (hdf : ∀ i, HasDerivAt (df i) (w i) (eta i)) :
    HasDerivAt
      (fun t => deriv (fun s =>
        ∑ i, f i (eta i + s * row (group i) p + t * row (group i) q)) 0)
      (∑ g, row g p * row g q * ∑ i, if group i = g then w i else 0) 0 := by
  rw [← repeated_design_rows]
  exact summed_affine_mixed_hasDerivAt f df eta _ _ w hf hdf

theorem sv_hessian_entry_hasDerivAt {J R Q : Type*}
    [Fintype J] [Fintype R] [Fintype Q]
    (f df : I → J → ℝ → ℝ) (eta w : I → J → ℝ)
    (Z : I → R → ℝ) (B : J → Q → ℝ) (r r' : R) (p p' : Q)
    (hf : ∀ i j x, HasDerivAt (f i j) (df i j x) x)
    (hdf : ∀ i j, HasDerivAt (df i j) (w i j) (eta i j)) :
    HasDerivAt
      (fun t => deriv (fun s => ∑ i, ∑ j,
        f i j (eta i j + s * (Z i r * B j p) + t * (Z i r' * B j p'))) 0)
      (∑ i, Z i r * Z i r' * ∑ j, w i j * B j p * B j p') 0 := by
  have h := summed_affine_mixed_hasDerivAt (I := I × J)
    (fun ij => f ij.1 ij.2) (fun ij => df ij.1 ij.2)
    (fun ij => eta ij.1 ij.2) (fun ij => Z ij.1 r * B ij.2 p)
    (fun ij => Z ij.1 r' * B ij.2 p') (fun ij => w ij.1 ij.2)
    (fun ij => hf ij.1 ij.2) (fun ij => hdf ij.1 ij.2)
  convert! h using 1
  · simp only [Fintype.sum_prod_type]
  · simp only [Fintype.sum_prod_type, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro j _
    ring

theorem sv_repeated_design_rows {J R : Type*} [Fintype J] [Fintype R] [DecidableEq G]
    (row : G → R → ℝ) (group : I → G) (B : J → P → ℝ)
    (w : I → J → ℝ) (r r' : R) (p p' : P) :
    (∑ i, row (group i) r * row (group i) r' * ∑ j, w i j * B j p * B j p') =
      ∑ g, row g r * row g r' *
        ∑ j, (∑ i, if group i = g then w i j else 0) * B j p * B j p' := by
  rw [repeated_design_rows row group (fun i => ∑ j, w i j * B j p * B j p')]
  apply Finset.sum_congr rfl
  intro g _
  congr 1
  calc
    _ = ∑ i, ∑ j, if group i = g then w i j * B j p * B j p' else 0 := by
      apply Finset.sum_congr rfl
      intro i _
      by_cases h : group i = g <;> simp [h]
    _ = ∑ j, ∑ i, if group i = g then w i j * B j p * B j p' else 0 :=
      Finset.sum_comm
    _ = _ := by simp only [Finset.sum_mul, ite_mul, zero_mul]

theorem sv_repeated_row_hessian_hasDerivAt {J R : Type*}
    [Fintype J] [Fintype R] [DecidableEq G]
    (f df : I → J → ℝ → ℝ) (eta w : I → J → ℝ)
    (row : G → R → ℝ) (group : I → G) (B : J → P → ℝ)
    (r r' : R) (p p' : P)
    (hf : ∀ i j x, HasDerivAt (f i j) (df i j x) x)
    (hdf : ∀ i j, HasDerivAt (df i j) (w i j) (eta i j)) :
    HasDerivAt (fun t => deriv (fun s => ∑ i, ∑ j,
      f i j (eta i j + s * (row (group i) r * B j p) +
        t * (row (group i) r' * B j p'))) 0)
      (∑ g, row g r * row g r' *
        ∑ j, (∑ i, if group i = g then w i j else 0) * B j p * B j p') 0 := by
  rw [← sv_repeated_design_rows]
  exact sv_hessian_entry_hasDerivAt f df eta w (fun i => row (group i)) B r r' p p' hf hdf

/-- General independent-cell GC curvature: no observation-family restriction. -/
theorem gc_cellwise_mixed_hasDerivAt {J : Type*} [Fintype J]
    (f df : I → J → ℝ → ℝ) (eta w : I → J → ℝ)
    (b c : J → ℝ) (z v : I → ℝ)
    (hf : ∀ i j x, HasDerivAt (f i j) (df i j x) x)
    (hdf : ∀ i j, HasDerivAt (df i j) (w i j) (eta i j)) :
    HasDerivAt (fun t => deriv (fun s => ∑ i, ∑ j,
      f i j (eta i j + s * (b j + z i) + t * (c j + v i))) 0)
      (∑ i, ∑ j, (b j + z i) * (c j + v i) * w i j) 0 := by
  have h := summed_affine_mixed_hasDerivAt (I := I × J)
    (fun ij => f ij.1 ij.2) (fun ij => df ij.1 ij.2) (fun ij => eta ij.1 ij.2)
    (fun ij => b ij.2 + z ij.1) (fun ij => c ij.2 + v ij.1)
    (fun ij => w ij.1 ij.2) (fun ij => hf ij.1 ij.2) (fun ij => hdf ij.1 ij.2)
  simpa only [Fintype.sum_prod_type] using h

theorem gc_cellwise_spatial_block_hasDerivAt {J : Type*} [Fintype J] [DecidableEq J]
    (f df : I → J → ℝ → ℝ) (eta w : I → J → ℝ)
    (B : Matrix J P ℝ) (p q : P)
    (hf : ∀ i j x, HasDerivAt (f i j) (df i j x) x)
    (hdf : ∀ i j, HasDerivAt (df i j) (w i j) (eta i j)) :
    HasDerivAt (fun t => deriv (fun s => ∑ i, ∑ j,
      f i j (eta i j + s * B j p + t * B j q)) 0)
      (((B.transpose * Matrix.diagonal (fun j => ∑ i, w i j) * B) : Matrix P P ℝ) p q) 0 := by
  rw [weighted_gram_entry]
  have h := gc_cellwise_mixed_hasDerivAt f df eta w
    (fun j => B j p) (fun j => B j q) 0 0 hf hdf
  simp only [Pi.zero_apply, add_zero] at h
  convert! h using 1
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]

theorem gc_cellwise_cross_block_hasDerivAt {J R : Type*} [Fintype J] [Fintype R]
    (f df : I → J → ℝ → ℝ) (eta w : I → J → ℝ)
    (B : Matrix J P ℝ) (Z : Matrix I R ℝ) (p : P) (r : R)
    (hf : ∀ i j x, HasDerivAt (f i j) (df i j x) x)
    (hdf : ∀ i j, HasDerivAt (df i j) (w i j) (eta i j)) :
    HasDerivAt (fun t => deriv (fun s => ∑ i, ∑ j,
      f i j (eta i j + s * B j p + t * Z i r)) 0)
      (∑ i, ∑ j, w i j * B j p * Z i r) 0 := by
  have h := gc_cellwise_mixed_hasDerivAt f df eta w
    (fun j => B j p) 0 0 (fun i => Z i r) hf hdf
  simp only [Pi.zero_apply, add_zero, zero_add] at h
  convert! h using 1
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem gc_cellwise_global_block_hasDerivAt {J R : Type*}
    [Fintype J] [Fintype R] [DecidableEq I]
    (f df : I → J → ℝ → ℝ) (eta w : I → J → ℝ)
    (Z : Matrix I R ℝ) (r s : R)
    (hf : ∀ i j x, HasDerivAt (f i j) (df i j x) x)
    (hdf : ∀ i j, HasDerivAt (df i j) (w i j) (eta i j)) :
    HasDerivAt (fun t => deriv (fun u => ∑ i, ∑ j,
      f i j (eta i j + u * Z i r + t * Z i s)) 0)
      (((Z.transpose * Matrix.diagonal (fun i => ∑ j, w i j) * Z) : Matrix R R ℝ) r s) 0 := by
  rw [weighted_gram_entry]
  have h := gc_cellwise_mixed_hasDerivAt f df eta w
    0 0 (fun i => Z i r) (fun i => Z i s) hf hdf
  simpa only [Pi.zero_apply, zero_add, Finset.mul_sum] using h

theorem poisson_separable_kronecker_hasDerivAt {J R : Type*}
    [Fintype J] [Fintype R] [DecidableEq I] [DecidableEq J]
    (y : I → J → ℝ) (etaB : J → ℝ) (etaZ : I → ℝ)
    (Z : Matrix I R ℝ) (B : Matrix J P ℝ) (r r' : R) (p p' : P) :
    HasDerivAt (fun t => deriv (fun s => ∑ i, ∑ j,
      (exp (etaB j + etaZ i + s * (Z i r * B j p) + t * (Z i r' * B j p')) -
        y i j * (etaB j + etaZ i + s * (Z i r * B j p) + t * (Z i r' * B j p')))) 0)
      (((Z.transpose * Matrix.diagonal (fun i => exp (etaZ i)) * Z) ⊗ₖ
        (B.transpose * Matrix.diagonal (fun j => exp (etaB j)) * B)) (r, p) (r', p')) 0 := by
  have h := sv_hessian_entry_hasDerivAt
    (fun i j x => exp x - y i j * x) (fun i j x => exp x - y i j)
    (fun i j => etaB j + etaZ i) (fun i j => exp (etaB j + etaZ i))
    Z B r r' p p'
    (fun i j x => poisson_nll_hasDerivAt (y i j) x)
    (fun i j => poisson_nll_second_hasDerivAt (y i j) (etaB j + etaZ i))
  convert! h using 1
  simp only [Matrix.kroneckerMap_apply, weighted_gram_entry, exp_add]
  simp only [Finset.sum_mul]
  simp only [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- Finite study summation commutes with both actual derivatives. -/
theorem finite_sum_mixed_hasDerivAt (f : I → ℝ → ℝ → ℝ)
    (first : I → ℝ → ℝ) (second : I → ℝ)
    (hfirst : ∀ i t, HasDerivAt (fun s => f i s t) (first i t) 0)
    (hsecond : ∀ i, HasDerivAt (fun t => deriv (fun s => f i s t) 0) (second i) 0) :
    HasDerivAt (fun t => deriv (fun s => ∑ i, f i s t) 0) (∑ i, second i) 0 := by
  have hid : (fun t => deriv (fun s => ∑ i, f i s t) 0) =
      (fun t => ∑ i, deriv (fun s => f i s t) 0) := by
    funext t
    rw [(finite_sum_hasDerivAt _ _ _ (fun i => hfirst i t)).deriv]
    apply Finset.sum_congr rfl
    intro i _
    exact (hfirst i t).deriv.symm
  rw [hid]
  exact finite_sum_hasDerivAt _ _ _ hsecond

theorem affine_sum_convex (f : I → ℝ → ℝ) (eta : I → ℝ) (X : I → P → ℝ)
    (hf : ∀ i, ConvexOn ℝ Set.univ (f i)) :
    ConvexOn ℝ Set.univ
      (fun beta : P → ℝ => ∑ i, f i (eta i + ∑ p, X i p * beta p)) := by
  refine ⟨convex_univ, ?_⟩
  intro beta _ gamma _ a b ha hb hab
  have hlin (i : I) :
      eta i + (∑ p, X i p * (a • beta + b • gamma) p) =
      a * (eta i + ∑ p, X i p * beta p) +
        b * (eta i + ∑ p, X i p * gamma p) := by
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, mul_add,
      Finset.sum_add_distrib]
    have hsum (c : ℝ) (v : P → ℝ) :
        (∑ p, X i p * (c * v p)) = c * ∑ p, X i p * v p := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro p _
      ring
    rw [hsum a beta, hsum b gamma]
    have heq : a = 1 - b := by linarith
    rw [heq]
    ring
  simp_rw [hlin]
  calc
    _ ≤ ∑ i, (a * f i (eta i + ∑ p, X i p * beta p) +
        b * f i (eta i + ∑ p, X i p * gamma p)) := by
      apply Finset.sum_le_sum
      intro i _
      exact (hf i).2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab
    _ = _ := by simp [Finset.sum_add_distrib, Finset.mul_sum, smul_eq_mul]

theorem poisson_regression_convex (y eta : I → ℝ) (X : I → P → ℝ) :
    ConvexOn ℝ Set.univ (fun beta : P → ℝ => ∑ i,
      (exp (eta i + ∑ p, X i p * beta p) -
        y i * (eta i + ∑ p, X i p * beta p))) := by
  exact affine_sum_convex _ eta X (fun i => poisson_nll_convex (y i))

theorem nb_regression_convex (alpha y eta : I → ℝ) (X : I → P → ℝ)
    (ha : ∀ i, 0 < alpha i) (hy : ∀ i, 0 ≤ y i) :
    ConvexOn ℝ Set.univ (fun beta : P → ℝ => ∑ i,
      (-y i * (eta i + ∑ p, X i p * beta p) +
        (y i + 1 / alpha i) * log (1 + alpha i * exp (eta i + ∑ p, X i p * beta p)))) := by
  exact affine_sum_convex _ eta X (fun i => nb_nll_convex (alpha i) (y i) (ha i) (hy i))

section Vector

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem vector_affine_hasDerivAt (f : E → ℝ) (df : E →L[ℝ] ℝ)
    (eta a : E) (hf : HasFDerivAt f df eta) :
    HasDerivAt (fun t : ℝ => f (eta + t • a)) (df a) 0 := by
  have hi : HasDerivAt (fun t : ℝ => eta + t • a) a 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const a).const_add eta
  simpa only [Function.comp_def] using
    hf.comp_hasDerivAt_of_eq 0 hi (by simp)

/-- Joint study likelihoods need not be sums of independent-cell likelihoods. -/
theorem vector_affine_mixed_hasDerivAt (f : E → ℝ) (df : E → E →L[ℝ] ℝ)
    (ddf : E →L[ℝ] E →L[ℝ] ℝ) (eta a b : E)
    (hf : ∀ x, HasFDerivAt f (df x) x)
    (hdf : HasFDerivAt df ddf eta) :
    (∀ t : ℝ, HasDerivAt (fun s : ℝ => f (eta + t • b + s • a))
      (df (eta + t • b) a) 0) ∧
    HasDerivAt (fun t : ℝ => df (eta + t • b) a) (ddf b a) 0 := by
  constructor
  · intro t
    exact vector_affine_hasDerivAt f _ _ a (hf _)
  · have hi : HasDerivAt (fun t : ℝ => eta + t • b) b 0 := by
      simpa using ((hasDerivAt_id (0 : ℝ)).smul_const b).const_add eta
    have hd := hdf.comp_hasDerivAt_of_eq 0 hi (by simp)
    simpa [Function.comp_def] using hd.clm_apply (hasDerivAt_const (0 : ℝ) a)

theorem sum_vector_affine_mixed_hasDerivAt (f : I → E → ℝ)
    (df : I → E → E →L[ℝ] ℝ) (ddf : I → E →L[ℝ] E →L[ℝ] ℝ)
    (eta a b : I → E)
    (hf : ∀ i x, HasFDerivAt (f i) (df i x) x)
    (hdf : ∀ i, HasFDerivAt (df i) (ddf i) (eta i)) :
    HasDerivAt (fun t : ℝ => deriv
      (fun s : ℝ => ∑ i, f i (eta i + t • b i + s • a i)) 0)
      (∑ i, ddf i (b i) (a i)) 0 := by
  apply finite_sum_mixed_hasDerivAt
    (first := fun i t => df i (eta i + t • b i) (a i))
  · intro i t
    exact (vector_affine_mixed_hasDerivAt (f i) (df i) (ddf i)
      (eta i) (a i) (b i) (hf i) (hdf i)).1 t
  · intro i
    have h := vector_affine_mixed_hasDerivAt (f i) (df i) (ddf i)
      (eta i) (a i) (b i) (hf i) (hdf i)
    have heq :
        (fun t : ℝ => deriv (fun s : ℝ => f i (eta i + t • b i + s • a i)) 0) =
        (fun t : ℝ => df i (eta i + t • b i) (a i)) := by
      funext t
      exact (h.1 t).deriv
    rw [heq]
    exact h.2

end Vector

end CbmrProofs.Hessian

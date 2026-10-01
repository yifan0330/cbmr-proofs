import lean_proof_checks.Hessian
import lean_proof_checks.Models

/-!
# Grouped Poisson scores, curvature, and the omitted allocation likelihood

`I` may be the finite type of studies in any one group. The spatial directions
`b,c` and global directions `z,v` can be arbitrary design columns or linear
combinations of columns. All curvature statements differentiate the likelihood
itself twice, rather than postulating information matrices.
-/

namespace CbmrProofs.GroupedPoisson

set_option autoImplicit false

open Finset Real
open CbmrProofs.Calculus CbmrProofs.Hessian

set_option backward.isDefEq.respectTransparency false
set_option backward.isDefEq.respectTransparency.types false

variable {I J G : Type*} [Fintype I] [Fintype J] [Fintype G]

/-- Exact compression for arbitrary groups, including different study counts in each group. -/
theorem all_groups_marginal_compression {S : G → Type*} [∀ g, Fintype (S g)]
    (etaB : G → J → ℝ) (etaZ : ∀ g, S g → ℝ) (y : ∀ g, S g → J → ℝ) :
    (∑ g, ∑ i, ∑ j,
      (y g i j * (etaB g j + etaZ g i) - exp (etaB g j + etaZ g i))) =
      (∑ g, ∑ j, (∑ i, y g i j) * etaB g j) +
        (∑ g, ∑ i, (∑ j, y g i j) * etaZ g i) -
          ∑ g, (∑ j, exp (etaB g j)) * (∑ i, exp (etaZ g i)) := by
  have hg (g : G) :
      (∑ i, ∑ j, (y g i j * (etaB g j + etaZ g i) - exp (etaB g j + etaZ g i))) =
      (∑ j, (∑ i, y g i j) * etaB g j) +
        (∑ i, (∑ j, y g i j) * etaZ g i) -
          (∑ j, exp (etaB g j)) * (∑ i, exp (etaZ g i)) := by
    have h := congrArg (fun x : ℝ => -x)
      (CbmrProofs.Models.poisson_exp_marginal_compression (y g) (etaB g) (etaZ g))
    simp only [← Finset.sum_neg_distrib, neg_sub] at h
    convert! h using 1
    ring
  simp_rw [hg, Finset.sum_sub_distrib, Finset.sum_add_distrib]

noncomputable def nll (etaB : J → ℝ) (etaZ : I → ℝ) (y : I → J → ℝ)
    (b c : J → ℝ) (z v : I → ℝ) (s t : ℝ) : ℝ :=
  ∑ i, ∑ j,
    (exp (etaB j + etaZ i + s * (b j + z i) + t * (c j + v i)) -
      y i j * (etaB j + etaZ i + s * (b j + z i) + t * (c j + v i)))

theorem score_expansion (etaB : J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (b : J → ℝ) (z : I → ℝ) :
    (∑ i, ∑ j, (b j + z i) * (exp (etaB j) * exp (etaZ i) - y i j)) =
      (∑ j, b j * ((∑ i, exp (etaZ i)) * exp (etaB j) - ∑ i, y i j)) +
      ∑ i, z i * ((∑ j, exp (etaB j)) * exp (etaZ i) - ∑ j, y i j) := by
  simp_rw [add_mul, mul_sub, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm (f := fun i j => b j * (exp (etaB j) * exp (etaZ i))),
    Finset.sum_comm (f := fun i j => b j * y i j)]
  congr 1
  · congr 1
    apply Finset.sum_congr rfl
    intro j _
    apply Finset.sum_congr rfl
    intro i _
    ring
/-- The negative of this derivative is the full spatial plus global score. -/
theorem full_score_hasDerivAt (etaB : J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (b : J → ℝ) (z : I → ℝ) :
    HasDerivAt (fun s => nll etaB etaZ y b 0 z 0 s 0)
      ((∑ j, b j * ((∑ i, exp (etaZ i)) * exp (etaB j) - ∑ i, y i j)) +
        ∑ i, z i * ((∑ j, exp (etaB j)) * exp (etaZ i) - ∑ j, y i j)) 0 := by
  rw [← score_expansion]
  have h := poisson_directional_hasDerivAt (I := I × J)
    (fun ij => y ij.1 ij.2) (fun ij => etaB ij.2 + etaZ ij.1)
    (fun ij => b ij.2 + z ij.1)
  simpa [nll, Fintype.sum_prod_type, exp_add] using h

theorem full_loglik_score_hasDerivAt (etaB : J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (b : J → ℝ) (z : I → ℝ) :
    HasDerivAt (fun s => -nll etaB etaZ y b 0 z 0 s 0)
      ((∑ j, b j * ((∑ i, y i j) - (∑ i, exp (etaZ i)) * exp (etaB j))) +
        ∑ i, z i * ((∑ j, y i j) - (∑ j, exp (etaB j)) * exp (etaZ i))) 0 := by
  convert! (full_score_hasDerivAt etaB etaZ y b z).neg using 1
  simp only [mul_sub, Finset.sum_sub_distrib]
  ring

theorem all_groups_score_hasDerivAt {S : G → Type*} [∀ g, Fintype (S g)]
    (etaB : G → J → ℝ) (etaZ : ∀ g, S g → ℝ) (y : ∀ g, S g → J → ℝ)
    (b : G → J → ℝ) (z : ∀ g, S g → ℝ) :
    HasDerivAt (fun s => ∑ g, -nll (etaB g) (etaZ g) (y g) (b g) 0 (z g) 0 s 0)
      (∑ g, ((∑ j, b g j *
          ((∑ i, y g i j) - (∑ i, exp (etaZ g i)) * exp (etaB g j))) +
        ∑ i, z g i *
          ((∑ j, y g i j) - (∑ j, exp (etaB g j)) * exp (etaZ g i)))) 0 :=
  finite_sum_hasDerivAt _ _ _ (fun g => full_loglik_score_hasDerivAt _ _ _ _ _)

theorem first_at_offset (etaB : J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (b c : J → ℝ) (z v : I → ℝ) (t : ℝ) :
    HasDerivAt (fun s => nll etaB etaZ y b c z v s t)
      (∑ i, ∑ j, (b j + z i) *
        (exp (etaB j + etaZ i + t * (c j + v i)) - y i j)) 0 := by
  have h := poisson_directional_hasDerivAt (I := I × J)
    (fun ij => y ij.1 ij.2)
    (fun ij => etaB ij.2 + etaZ ij.1 + t * (c ij.2 + v ij.1))
    (fun ij => b ij.2 + z ij.1)
  simp only [Fintype.sum_prod_type] at h
  convert! h using 1
  funext s
  unfold nll
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  congr 1 <;> congr 1 <;> ring

theorem mixed_hasDerivAt (etaB : J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (b c : J → ℝ) (z v : I → ℝ) :
    HasDerivAt (fun t => deriv (fun s => nll etaB etaZ y b c z v s t) 0)
      (∑ i, ∑ j, (b j + z i) * (c j + v i) *
        (exp (etaB j) * exp (etaZ i))) 0 := by
  have h := summed_affine_mixed_hasDerivAt (I := I × J)
    (fun ij x => exp x - y ij.1 ij.2 * x)
    (fun ij x => exp x - y ij.1 ij.2)
    (fun ij => etaB ij.2 + etaZ ij.1)
    (fun ij => b ij.2 + z ij.1) (fun ij => c ij.2 + v ij.1)
    (fun ij => exp (etaB ij.2 + etaZ ij.1))
    (fun ij x => poisson_nll_hasDerivAt _ _)
    (fun ij => poisson_nll_second_hasDerivAt _ _)
  simpa [nll, Fintype.sum_prod_type, exp_add] using h

theorem spatial_block_hasDerivAt (etaB : J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (b c : J → ℝ) :
    HasDerivAt (fun t => deriv (fun s => nll etaB etaZ y b c 0 0 s t) 0)
      ((∑ i, exp (etaZ i)) * ∑ j, exp (etaB j) * b j * c j) 0 := by
  convert! mixed_hasDerivAt etaB etaZ y b c 0 0 using 1
  simp only [Pi.zero_apply, add_zero]
  simp only [Finset.sum_mul]
  simp only [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem spatial_global_block_hasDerivAt (etaB : J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (b : J → ℝ) (z : I → ℝ) :
    HasDerivAt (fun t => deriv (fun s => nll etaB etaZ y b 0 0 z s t) 0)
      ((∑ j, exp (etaB j) * b j) * ∑ i, exp (etaZ i) * z i) 0 := by
  convert! mixed_hasDerivAt etaB etaZ y b 0 0 z using 1
  simp only [Pi.zero_apply, add_zero, zero_add]
  simp only [Finset.sum_mul]
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem global_block_hasDerivAt (etaB : J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (z v : I → ℝ) :
    HasDerivAt (fun t => deriv (fun s => nll etaB etaZ y 0 0 z v s t) 0)
      ((∑ j, exp (etaB j)) * ∑ i, exp (etaZ i) * z i * v i) 0 := by
  convert! mixed_hasDerivAt etaB etaZ y 0 0 z v using 1
  simp only [Pi.zero_apply, zero_add]
  simp only [Finset.sum_mul]
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- Groups can have different finite study types; global information adds over groups. -/
theorem all_groups_global_block_hasDerivAt {S : G → Type*}
    [∀ g, Fintype (S g)] (etaB : G → J → ℝ) (etaZ : ∀ g, S g → ℝ)
    (y : ∀ g, S g → J → ℝ) (z v : ∀ g, S g → ℝ) :
    HasDerivAt (fun t => deriv
      (fun s => ∑ g, nll (etaB g) (etaZ g) (y g) 0 0 (z g) (v g) s t) 0)
      (∑ g, (∑ j, exp (etaB g j)) *
        ∑ i, exp (etaZ g i) * z g i * v g i) 0 := by
  apply finite_sum_mixed_hasDerivAt
    (first := fun g t => ∑ i, ∑ j, (0 + z g i) *
      (exp (etaB g j + etaZ g i + t * (0 + v g i)) - y g i j))
  · intro g t
    exact first_at_offset _ _ _ _ _ _ _ t
  · intro g
    exact global_block_hasDerivAt _ _ _ _ _

/-- Distinct groups' spatial coefficients have zero mixed derivative. -/
theorem cross_group_spatial_zero_hasDerivAt [DecidableEq G]
    (group : I → G) (etaB : G → J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (b c : J → ℝ) (g h : G) (hgh : g ≠ h) :
    HasDerivAt
      (fun t => deriv (fun s => ∑ i, ∑ j,
        (exp (etaB (group i) j + etaZ i +
          s * (if group i = g then b j else 0) +
          t * (if group i = h then c j else 0)) -
        y i j * (etaB (group i) j + etaZ i +
          s * (if group i = g then b j else 0) +
          t * (if group i = h then c j else 0)))) 0) 0 0 := by
  have hd := summed_affine_mixed_hasDerivAt (I := I × J)
    (fun ij x => exp x - y ij.1 ij.2 * x)
    (fun ij x => exp x - y ij.1 ij.2)
    (fun ij => etaB (group ij.1) ij.2 + etaZ ij.1)
    (fun ij => if group ij.1 = g then b ij.2 else 0)
    (fun ij => if group ij.1 = h then c ij.2 else 0)
    (fun ij => exp (etaB (group ij.1) ij.2 + etaZ ij.1))
    (fun ij x => poisson_nll_hasDerivAt _ _)
    (fun ij => poisson_nll_second_hasDerivAt _ _)
  convert! hd using 1
  · simp only [Fintype.sum_prod_type]
  · symm
    apply Finset.sum_eq_zero
    intro ij _
    by_cases hi : group ij.1 = g
    · have hj : group ij.1 ≠ h := by simpa [hi] using hgh
      simp [hi, hgh]
    · simp [hi]

theorem exp_sum_hasDerivAt (eta z : I → ℝ) (x : ℝ) :
    HasDerivAt (fun t => ∑ i, exp (eta i + t * z i))
      (∑ i, exp (eta i + x * z i) * z i) x :=
  finite_sum_hasDerivAt _ _ _ (fun i => affine_mean_hasDerivAt _ _ _)

theorem exp_sum_pos [Nonempty I] (eta : I → ℝ) :
    0 < ∑ i, exp (eta i) := by
  exact Finset.sum_pos (fun i _ => exp_pos _) Finset.univ_nonempty

/-- Differentiate the actual log allocation probabilities omitted by totals-only fitting. -/
theorem allocation_hasDerivAt [Nonempty I] (eta z count : I → ℝ) :
    HasDerivAt
      (fun t => ∑ i, count i *
        log (exp (eta i + t * z i) / ∑ k, exp (eta k + t * z k)))
      ((∑ i, count i * z i) -
        (∑ i, count i) * (∑ i, exp (eta i) * z i) / (∑ i, exp (eta i))) 0 := by
  have hpos (t : ℝ) : 0 < ∑ i, exp (eta i + t * z i) := exp_sum_pos _
  have hid (t : ℝ) :
      (∑ i, count i *
        log (exp (eta i + t * z i) / ∑ k, exp (eta k + t * z k))) =
      ∑ i, count i * (eta i + t * z i - log (∑ k, exp (eta k + t * z k))) := by
    apply Finset.sum_congr rfl
    intro i _
    rw [log_div (exp_ne_zero _) (ne_of_gt (hpos t)), log_exp]
  simp_rw [hid]
  have hs := (exp_sum_hasDerivAt eta z 0).log (ne_of_gt (hpos 0))
  have hd := finite_sum_hasDerivAt _ _ 0 (fun i =>
    ((gc_predictor_hasDerivAt (eta i) (z i) 0).sub hs).const_mul (count i))
  convert! hd using 1
  simp only [zero_mul, add_zero, mul_sub, Finset.sum_sub_distrib]
  rw [← Finset.sum_mul]
  ring

theorem all_groups_allocation_hasDerivAt {S : G → Type*}
    [∀ g, Fintype (S g)] [∀ g, Nonempty (S g)]
    (eta z count : ∀ g, S g → ℝ) :
    HasDerivAt
      (fun t => ∑ g, ∑ i, count g i *
        log (exp (eta g i + t * z g i) / ∑ k, exp (eta g k + t * z g k)))
      (∑ g, ((∑ i, count g i * z g i) -
        (∑ i, count g i) * (∑ i, exp (eta g i) * z g i) /
          (∑ i, exp (eta g i)))) 0 :=
  finite_sum_hasDerivAt _ _ _ (fun g => allocation_hasDerivAt (eta g) (z g) (count g))

theorem all_groups_allocation_difference {S : G → Type*} [∀ g, Fintype (S g)]
    (y : ∀ g, S g → J → ℝ) (u : G → J → ℝ) (e : ∀ g, S g → ℝ)
    (hu : ∀ g j, 0 < u g j) (he : ∀ g i, 0 < e g i)
    (hT : ∀ g, 0 < ∑ i, e g i) :
    ((∑ g, ∑ j, (∑ i, y g i j) * log (u g j)) +
      (∑ g, ∑ i, (∑ j, y g i j) * log (e g i))) -
        (∑ g, ∑ j, (∑ i, y g i j) * log (u g j * ∑ i, e g i)) =
      ∑ g, ∑ i, (∑ j, y g i j) * log (e g i / ∑ k, e g k) := by
  rw [← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro g _
  exact CbmrProofs.Models.poisson_allocation_difference
    (y g) (u g) (e g) (hu g) (he g) (hT g)

end CbmrProofs.GroupedPoisson

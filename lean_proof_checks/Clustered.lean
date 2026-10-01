import lean_proof_checks.Hessian
import Mathlib.Analysis.Convex.Deriv
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic.Linarith

/-!
# Clustered NB regression calculus at fixed shape

The joint negative log likelihood retains every log-mean-dependent term.
The omitted gamma/factorial terms are constant in the regression parameters.
The arbitrary directions in the results include individual coordinates and
columns of either GC or SV design matrices.
-/

namespace CbmrProofs.Clustered

set_option autoImplicit false

open Finset Real
open CbmrProofs.Calculus CbmrProofs.Hessian

set_option backward.isDefEq.respectTransparency false
set_option backward.isDefEq.respectTransparency.types false

variable {I J : Type*} [Fintype I] [Fintype J]

noncomputable def total (eta : J → ℝ) : ℝ := ∑ j, exp (eta j)

noncomputable def moment (eta a : J → ℝ) : ℝ := ∑ j, exp (eta j) * a j

noncomputable def studyNll (eta y : J → ℝ) (nu : ℝ) : ℝ :=
  ((∑ j, y j) + nu) * log (nu + total eta) - ∑ j, y j * eta j

theorem total_nonneg (eta : J → ℝ) : 0 ≤ total eta :=
  Finset.sum_nonneg (fun j _ => le_of_lt (exp_pos _))

theorem denominator_pos (eta : J → ℝ) (nu : ℝ) (hn : 0 < nu) :
    0 < nu + total eta := add_pos_of_pos_of_nonneg hn (total_nonneg eta)

theorem total_hasDerivAt (eta a : J → ℝ) (x : ℝ) :
    HasDerivAt (fun t => total (fun j => eta j + t * a j))
      (moment (fun j => eta j + x * a j) a) x :=
  finite_sum_hasDerivAt _ _ _ (fun j => affine_mean_hasDerivAt _ _ _)

theorem moment_hasDerivAt (eta a b : J → ℝ) (x : ℝ) :
    HasDerivAt (fun t => moment (fun j => eta j + t * b j) a)
      (moment (fun j => eta j + x * b j) (fun j => a j * b j)) x := by
  apply finite_sum_hasDerivAt
  intro j
  convert! (affine_mean_hasDerivAt (eta j) (b j) x).mul_const (a j) using 1
  ring

theorem study_directional_hasDerivAt (eta y a : J → ℝ) (nu x : ℝ) (hn : 0 < nu) :
    HasDerivAt (fun t => studyNll (fun j => eta j + t * a j) y nu)
      (((∑ j, y j) + nu) * moment (fun j => eta j + x * a j) a /
        (nu + total (fun j => eta j + x * a j)) - ∑ j, y j * a j) x := by
  have hp := denominator_pos (fun j => eta j + x * a j) nu hn
  have hlog := ((total_hasDerivAt eta a x).const_add nu).log (ne_of_gt hp)
  have hlinear := finite_sum_hasDerivAt _ _ x (fun j =>
    (gc_predictor_hasDerivAt (eta j) (a j) x).const_mul (y j))
  convert! (hlog.const_mul ((∑ j, y j) + nu)).sub hlinear using 1
  ring

/-- `c` includes all fixed-shape gamma and factorial terms of the joint log likelihood. -/
theorem joint_loglik_hasDerivAt (eta y a : J → ℝ) (nu c x : ℝ) (hn : 0 < nu) :
    HasDerivAt (fun t => c - studyNll (fun j => eta j + t * a j) y nu)
      ((∑ j, y j * a j) -
        ((∑ j, y j) + nu) * moment (fun j => eta j + x * a j) a /
          (nu + total (fun j => eta j + x * a j))) x := by
  convert! (study_directional_hasDerivAt eta y a nu x hn).const_sub c using 1
  ring

theorem study_first_at_offset (eta y a b : J → ℝ) (nu t : ℝ) (hn : 0 < nu) :
    HasDerivAt
      (fun s => studyNll (fun j => eta j + s * a j + t * b j) y nu)
      (((∑ j, y j) + nu) * moment (fun j => eta j + t * b j) a /
        (nu + total (fun j => eta j + t * b j)) - ∑ j, y j * a j) 0 := by
  convert! study_directional_hasDerivAt (fun j => eta j + t * b j) y a nu 0 hn
    using 1
  · funext s
    congr 1
    funext j
    ring
  · simp

theorem study_score_hasDerivAt (eta y a b : J → ℝ) (nu x : ℝ) (hn : 0 < nu) :
    HasDerivAt
      (fun t => ((∑ j, y j) + nu) * moment (fun j => eta j + t * b j) a /
        (nu + total (fun j => eta j + t * b j)) - ∑ j, y j * a j)
      (((∑ j, y j) + nu) *
        (moment (fun j => eta j + x * b j) (fun j => a j * b j) /
          (nu + total (fun j => eta j + x * b j)) -
        moment (fun j => eta j + x * b j) a * moment (fun j => eta j + x * b j) b /
          (nu + total (fun j => eta j + x * b j)) ^ 2)) x := by
  have hp := ne_of_gt (denominator_pos (fun j => eta j + x * b j) nu hn)
  have h := (((moment_hasDerivAt eta a b x).const_mul ((∑ j, y j) + nu)).div
    ((total_hasDerivAt eta b x).const_add nu) hp).sub_const (∑ j, y j * a j)
  convert! h using 1
  field_simp

/-- The full log-mean curvature is diagonal minus rank one, in arbitrary directions. -/
theorem study_mixed_hasDerivAt (eta y a b : J → ℝ) (nu : ℝ) (hn : 0 < nu) :
    HasDerivAt
      (fun t => deriv
        (fun s => studyNll (fun j => eta j + s * a j + t * b j) y nu) 0)
      (((∑ j, y j) + nu) *
        (moment eta (fun j => a j * b j) / (nu + total eta) -
          moment eta a * moment eta b / (nu + total eta) ^ 2)) 0 := by
  have hid : (fun t => deriv
      (fun s => studyNll (fun j => eta j + s * a j + t * b j) y nu) 0) =
      (fun t => ((∑ j, y j) + nu) * moment (fun j => eta j + t * b j) a /
        (nu + total (fun j => eta j + t * b j)) - ∑ j, y j * a j) := by
    funext t
    exact (study_first_at_offset eta y a b nu t hn).deriv
  rw [hid]
  simpa using study_score_hasDerivAt eta y a b nu 0 hn

/-- Entries of the local observed information matrix, not a definition of a derivative. -/
noncomputable def localInformation (eta y : J → ℝ) (nu : ℝ) : Matrix J J ℝ := by
  classical
  exact fun j k => (((∑ l, y l) + nu) / (nu + total eta)) *
    ((if j = k then exp (eta j) else 0) -
      exp (eta j) * exp (eta k) / (nu + total eta))

theorem local_information_bilinear (eta y a b : J → ℝ) (nu : ℝ)
    (hn : 0 < nu) :
    (∑ j, ∑ k, a j * localInformation eta y nu j k * b k) =
      (((∑ j, y j) + nu) *
        (moment eta (fun j => a j * b j) / (nu + total eta) -
          moment eta a * moment eta b / (nu + total eta) ^ 2)) := by
  classical
  have hentry (j k : J) :
      a j * localInformation eta y nu j k * b k =
      (((∑ l, y l) + nu) / (nu + total eta)) *
        ((if j = k then exp (eta j) * a j * b j else 0) -
          ((exp (eta j) * a j) * (exp (eta k) * b k)) / (nu + total eta)) := by
    unfold localInformation
    split_ifs with h
    · subst k
      ring
    · ring
  simp only [hentry, mul_sub, Finset.sum_sub_distrib, ← Finset.mul_sum,
    ← Finset.sum_div]
  simp only [Finset.sum_ite_eq, Finset.mem_univ, if_true]
  simp only [← Finset.mul_sum, ← Finset.sum_mul]
  simp only [moment, mul_assoc]
  have hp := ne_of_gt (denominator_pos eta nu hn)
  field_simp

theorem study_information_hasDerivAt (eta y a b : J → ℝ) (nu : ℝ) (hn : 0 < nu) :
    HasDerivAt
      (fun t => deriv
        (fun s => studyNll (fun j => eta j + s * a j + t * b j) y nu) 0)
      (∑ j, ∑ k, a j * localInformation eta y nu j k * b k) 0 := by
  rw [local_information_bilinear eta y a b nu hn]
  exact study_mixed_hasDerivAt eta y a b nu hn

theorem study_coordinate_hasDerivAt [DecidableEq J] (eta y : J → ℝ)
    (nu : ℝ) (hn : 0 < nu) (p q : J) :
    HasDerivAt
      (fun t => deriv (fun s => studyNll
        (fun j => eta j + s * (if j = p then 1 else 0) +
          t * (if j = q then 1 else 0)) y nu) 0)
      (localInformation eta y nu p q) 0 := by
  have h := study_information_hasDerivAt eta y
    (fun j => if j = p then 1 else 0) (fun j => if j = q then 1 else 0) nu hn
  simpa [mul_ite, ite_mul] using h

theorem study_projected_coordinate_hasDerivAt {P : Type*} [Fintype P]
    (eta y : J → ℝ) (nu : ℝ) (hn : 0 < nu) (X : Matrix J P ℝ) (p q : P) :
    HasDerivAt
      (fun t => deriv (fun s =>
        studyNll (fun j => eta j + s * X j p + t * X j q) y nu) 0)
      ((X.transpose * localInformation eta y nu * X) p q) 0 := by
  have heq : (X.transpose * localInformation eta y nu * X) p q =
      ∑ j, ∑ k, X j p * localInformation eta y nu j k * X k q := by
    simp only [Matrix.mul_apply, Matrix.transpose_apply, Finset.sum_mul]
    exact Finset.sum_comm
  rw [heq]
  exact study_information_hasDerivAt eta y (fun j => X j p) (fun j => X j q) nu hn

/-- Algebraic mean substitution only; the expectation theorem belongs to probability. -/
theorem local_information_mean_substitution [DecidableEq J] (eta y : J → ℝ) (nu : ℝ)
    (hn : 0 < nu) (hy : (∑ j, y j) = total eta) (j k : J) :
    localInformation eta y nu j k =
      (if j = k then exp (eta j) else 0) -
        exp (eta j) * exp (eta k) / (nu + total eta) := by
  unfold localInformation
  rw [hy, add_comm (total eta) nu, div_self (ne_of_gt (denominator_pos eta nu hn))]
  simp only [one_mul]
  by_cases h : j = k <;> simp [h]

noncomputable def g (etaZ count : I → ℝ) (nu E : ℝ) : ℝ :=
  ∑ i, (count i + nu) * log (nu + E * exp (etaZ i))

noncomputable def gprime (etaZ count : I → ℝ) (nu E : ℝ) : ℝ :=
  ∑ i, (count i + nu) * exp (etaZ i) / (nu + E * exp (etaZ i))

noncomputable def gsecond (etaZ count : I → ℝ) (nu E : ℝ) : ℝ :=
  -∑ i, (count i + nu) * exp (etaZ i) ^ 2 / (nu + E * exp (etaZ i)) ^ 2

theorem g_hasDerivAt (etaZ count : I → ℝ) (nu E : ℝ)
    (hn : 0 < nu) (hE : 0 ≤ E) :
    HasDerivAt (g etaZ count nu) (gprime etaZ count nu E) E := by
  apply finite_sum_hasDerivAt
  intro i
  have hp : nu + E * exp (etaZ i) ≠ 0 := by positivity
  have h := (((hasDerivAt_id E).mul_const (exp (etaZ i))).const_add nu).log hp
  convert! h.const_mul (count i + nu) using 1
  simp only [id_eq]
  ring

theorem gprime_hasDerivAt (etaZ count : I → ℝ) (nu E : ℝ)
    (hn : 0 < nu) (hE : 0 ≤ E) :
    HasDerivAt (gprime etaZ count nu) (gsecond etaZ count nu E) E := by
  unfold gprime gsecond
  rw [← Finset.sum_neg_distrib]
  apply finite_sum_hasDerivAt
  intro i
  have hp : nu + E * exp (etaZ i) ≠ 0 := by positivity
  have h := (hasDerivAt_const E ((count i + nu) * exp (etaZ i))).div
    (((hasDerivAt_id E).mul_const (exp (etaZ i))).const_add nu) hp
  convert! h using 1
  simp only [id_eq]
  ring

theorem gsecond_nonpos (etaZ count : I → ℝ) (nu E : ℝ)
    (hn : 0 < nu) (hc : ∀ i, 0 ≤ count i) :
    gsecond etaZ count nu E ≤ 0 := by
  unfold gsecond
  apply neg_nonpos.mpr
  apply Finset.sum_nonneg
  intro i _
  have hi := hc i
  positivity

theorem gsecond_neg [Nonempty I] (etaZ count : I → ℝ) (nu E : ℝ)
    (hn : 0 < nu) (hE : 0 ≤ E) (hc : ∀ i, 0 ≤ count i) :
    gsecond etaZ count nu E < 0 := by
  unfold gsecond
  apply neg_lt_zero.mpr
  apply Finset.sum_pos _ Finset.univ_nonempty
  intro i _
  have hi := hc i
  positivity

theorem total_shift (eta : J → ℝ) (c : ℝ) :
    total (fun j => eta j + c) = total eta * exp c := by
  simp [total, exp_add, Finset.sum_mul]

theorem moment_shift (eta a : J → ℝ) (c : ℝ) :
    moment (fun j => eta j + c) a = moment eta a * exp c := by
  unfold moment
  simp only [exp_add, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem moment_const (eta : J → ℝ) (c : ℝ) :
    moment eta (fun _ => c) = total eta * c := by
  simp [moment, total, Finset.sum_mul]

theorem moment_mul_const (eta a : J → ℝ) (c : ℝ) :
    moment eta (fun j => a j * c) = moment eta a * c := by
  simp [moment, Finset.sum_mul, mul_assoc]

/-- Finite likelihood collection, retaining arbitrary shared and study-specific
normalising constants. Gamma terms can be substituted without discarding them. -/
theorem gc_likelihood_collection (etaB : J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (nu shared : ℝ) (normalising : I → ℝ) :
    (∑ i, (shared + normalising i -
      studyNll (fun j => etaB j + etaZ i) (y i) nu)) =
      (Fintype.card I : ℝ) * shared + (∑ i, normalising i) -
        g etaZ (fun i => ∑ j, y i j) nu (total etaB) +
        (∑ j, (∑ i, y i j) * etaB j) +
        ∑ i, (∑ j, y i j) * etaZ i := by
  have hterm (i : I) :
      shared + normalising i - studyNll (fun j => etaB j + etaZ i) (y i) nu =
      shared + normalising i -
        ((∑ j, y i j) + nu) * log (nu + total etaB * exp (etaZ i)) +
          ∑ j, y i j * (etaB j + etaZ i) := by
    unfold studyNll
    rw [total_shift]
    ring
  have hlinear :
      (∑ i, ∑ j, y i j * (etaB j + etaZ i)) =
      (∑ j, (∑ i, y i j) * etaB j) + ∑ i, (∑ j, y i j) * etaZ i := by
    simp only [mul_add, Finset.sum_add_distrib, Finset.sum_mul]
    congr 1
    exact Finset.sum_comm
  simp_rw [hterm, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  rw [hlinear]
  simp only [g, add_mul, Finset.sum_add_distrib, Finset.sum_const,
    Finset.card_univ, nsmul_eq_mul]
  ring

noncomputable def gcNll (etaB : J → ℝ) (etaZ : I → ℝ) (y : I → J → ℝ)
    (nu : ℝ) (b c : J → ℝ) (z v : I → ℝ) (s t : ℝ) : ℝ :=
  ∑ i, studyNll
    (fun j => etaB j + etaZ i + s * (b j + z i) + t * (c j + v i)) (y i) nu

theorem gc_first_hasDerivAt (etaB : J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (nu : ℝ) (b c : J → ℝ) (z v : I → ℝ)
    (t : ℝ) (hn : 0 < nu) :
    HasDerivAt (fun s => gcNll etaB etaZ y nu b c z v s t)
      (∑ i, (((∑ j, y i j) + nu) *
        moment (fun j => etaB j + etaZ i + t * (c j + v i)) (fun j => b j + z i) /
        (nu + total (fun j => etaB j + etaZ i + t * (c j + v i))) -
          ∑ j, y i j * (b j + z i))) 0 :=
  finite_sum_hasDerivAt _ _ _ (fun i => study_first_at_offset _ _ _ _ nu t hn)

theorem gc_mixed_hasDerivAt (etaB : J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (nu : ℝ) (b c : J → ℝ) (z v : I → ℝ) (hn : 0 < nu) :
    HasDerivAt
      (fun t => deriv (fun s => gcNll etaB etaZ y nu b c z v s t) 0)
      (∑ i, ((∑ j, y i j) + nu) *
        (moment (fun j => etaB j + etaZ i) (fun j => (b j + z i) * (c j + v i)) /
          (nu + total (fun j => etaB j + etaZ i)) -
        moment (fun j => etaB j + etaZ i) (fun j => b j + z i) *
          moment (fun j => etaB j + etaZ i) (fun j => c j + v i) /
            (nu + total (fun j => etaB j + etaZ i)) ^ 2)) 0 := by
  apply finite_sum_mixed_hasDerivAt
    (first := fun i t => ((∑ j, y i j) + nu) *
      moment (fun j => etaB j + etaZ i + t * (c j + v i)) (fun j => b j + z i) /
      (nu + total (fun j => etaB j + etaZ i + t * (c j + v i))) -
        ∑ j, y i j * (b j + z i))
  · intro i t
    exact study_first_at_offset _ _ _ _ nu t hn
  · intro i
    exact study_mixed_hasDerivAt _ _ _ _ nu hn

/-- GC spatial-coordinate curvature is a weighted Gram plus the negative rank-one correction. -/
theorem gc_spatial_block_hasDerivAt (etaB : J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (nu : ℝ) (b c : J → ℝ) (hn : 0 < nu) :
    HasDerivAt
      (fun t => deriv (fun s => gcNll etaB etaZ y nu b c 0 0 s t) 0)
      (gprime etaZ (fun i => ∑ j, y i j) nu (total etaB) *
          moment etaB (fun j => b j * c j) +
        gsecond etaZ (fun i => ∑ j, y i j) nu (total etaB) *
          moment etaB b * moment etaB c) 0 := by
  convert! gc_mixed_hasDerivAt etaB etaZ y nu b c 0 0 hn using 1
  simp only [Pi.zero_apply, add_zero, total_shift, moment_shift, gprime, gsecond,
    Finset.sum_mul, neg_mul, ← Finset.sum_neg_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem gc_spatial_global_block_hasDerivAt (etaB : J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (nu : ℝ) (b : J → ℝ) (z : I → ℝ) (hn : 0 < nu) :
    HasDerivAt
      (fun t => deriv (fun s => gcNll etaB etaZ y nu b 0 0 z s t) 0)
      (moment etaB b * ∑ i, ((∑ j, y i j) + nu) * nu * exp (etaZ i) * z i /
        (nu + total etaB * exp (etaZ i)) ^ 2) 0 := by
  convert! gc_mixed_hasDerivAt etaB etaZ y nu b 0 0 z hn using 1
  simp only [Pi.zero_apply, add_zero, zero_add, total_shift, moment_shift,
    moment_const, moment_mul_const, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  have hE := total_nonneg etaB
  have hp : nu + total etaB * exp (etaZ i) ≠ 0 := by positivity
  field_simp
  ring

theorem gc_global_block_hasDerivAt (etaB : J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (nu : ℝ) (z v : I → ℝ) (hn : 0 < nu) :
    HasDerivAt
      (fun t => deriv (fun s => gcNll etaB etaZ y nu 0 0 z v s t) 0)
      (∑ i, ((∑ j, y i j) + nu) * nu * total etaB * exp (etaZ i) * z i * v i /
        (nu + total etaB * exp (etaZ i)) ^ 2) 0 := by
  convert! gc_mixed_hasDerivAt etaB etaZ y nu 0 0 z v hn using 1
  simp only [Pi.zero_apply, zero_add, total_shift, moment_shift, moment_const]
  apply Finset.sum_congr rfl
  intro i _
  have hE := total_nonneg etaB
  have hp : nu + total etaB * exp (etaZ i) ≠ 0 := by positivity
  field_simp
  ring

theorem all_groups_global_block_hasDerivAt {G : Type*} [Fintype G]
    {S : G → Type*} [∀ g, Fintype (S g)]
    (etaB : G → J → ℝ) (etaZ : ∀ g, S g → ℝ)
    (y : ∀ g, S g → J → ℝ) (nu : G → ℝ) (z v : ∀ g, S g → ℝ)
    (hn : ∀ g, 0 < nu g) :
    HasDerivAt
      (fun t => deriv (fun s => ∑ g,
        gcNll (etaB g) (etaZ g) (y g) (nu g) 0 0 (z g) (v g) s t) 0)
      (∑ g, ∑ i, ((∑ j, y g i j) + nu g) * nu g * total (etaB g) *
        exp (etaZ g i) * z g i * v g i /
          (nu g + total (etaB g) * exp (etaZ g i)) ^ 2) 0 := by
  apply finite_sum_mixed_hasDerivAt
    (first := fun g t => ∑ i, (((∑ j, y g i j) + nu g) *
      moment (fun j => etaB g j + etaZ g i + t * (0 + v g i)) (fun _ => 0 + z g i) /
      (nu g + total (fun j => etaB g j + etaZ g i + t * (0 + v g i))) -
        ∑ j, y g i j * (0 + z g i)))
  · intro g t
    exact gc_first_hasDerivAt _ _ _ _ _ _ _ _ t (hn g)
  · intro g
    exact gc_global_block_hasDerivAt _ _ _ _ _ _ (hn g)

theorem cross_group_spatial_zero_hasDerivAt {G : Type*} [DecidableEq G]
    (group : I → G) (etaB : G → J → ℝ) (etaZ : I → ℝ)
    (y : I → J → ℝ) (nu : G → ℝ) (hn : ∀ g, 0 < nu g)
    (b c : J → ℝ) (g h : G) (hgh : g ≠ h) :
    HasDerivAt
      (fun t => deriv (fun s => ∑ i, studyNll
        (fun j => etaB (group i) j + etaZ i +
          s * (if group i = g then b j else 0) +
          t * (if group i = h then c j else 0)) (y i) (nu (group i))) 0) 0 0 := by
  let f := fun i s t => studyNll
    (fun j => etaB (group i) j + etaZ i +
      s * (if group i = g then b j else 0) +
      t * (if group i = h then c j else 0)) (y i) (nu (group i))
  suffices hd : HasDerivAt (fun t => deriv (fun s => ∑ i, f i s t) 0)
      (∑ _i : I, (0 : ℝ)) 0 by simpa [f] using hd
  apply finite_sum_mixed_hasDerivAt (first := fun i t => deriv (fun s => f i s t) 0)
  · intro i t
    exact (study_first_at_offset (fun j => etaB (group i) j + etaZ i) (y i)
      (fun j => if group i = g then b j else 0)
      (fun j => if group i = h then c j else 0) (nu (group i)) t
        (hn (group i))).differentiableAt.hasDerivAt
  · intro i
    have hd := study_mixed_hasDerivAt (fun j => etaB (group i) j + etaZ i) (y i)
      (fun j => if group i = g then b j else 0)
      (fun j => if group i = h then c j else 0) (nu (group i)) (hn (group i))
    by_cases hi : group i = g
    · simpa [f, hi, hgh, moment] using hd
    · simpa [f, hi, moment] using hd

/-- An unordered-pair equivalent of the appendix's `j < k` identity. -/
theorem weighted_variance_identity (mu a : J → ℝ) :
    (∑ j, ∑ k, mu j * mu k * (a j - a k) ^ 2) =
      2 * ((∑ j, mu j) * (∑ j, mu j * a j ^ 2) - (∑ j, mu j * a j) ^ 2) := by
  calc
    _ = ∑ j, ∑ k,
        ((mu j * a j ^ 2) * mu k + mu j * (mu k * a k ^ 2) -
          2 * (mu j * a j) * (mu k * a k)) := by
      apply Finset.sum_congr rfl
      intro j _
      apply Finset.sum_congr rfl
      intro k _
      ring
    _ = _ := by
      simp_rw [Finset.sum_sub_distrib, Finset.sum_add_distrib,
        ← Finset.mul_sum, ← Finset.sum_mul]
      simp only [← Finset.mul_sum]
      ring

theorem weighted_variance_strict_pairs [LinearOrder J] (mu a : J → ℝ) :
    (∑ j, mu j) * (∑ j, mu j * a j ^ 2) - (∑ j, mu j * a j) ^ 2 =
      ∑ j, ∑ k, if j < k then mu j * mu k * (a j - a k) ^ 2 else 0 := by
  have hsplit (j k : J) :
      mu j * mu k * (a j - a k) ^ 2 =
      (if j < k then mu j * mu k * (a j - a k) ^ 2 else 0) +
        (if k < j then mu k * mu j * (a k - a j) ^ 2 else 0) := by
    rcases lt_trichotomy j k with hjk | hjk | hkj
    · simp [hjk, not_lt_of_gt hjk]
    · subst k
      simp
    · simp [hkj, not_lt_of_gt hkj] <;> ring
  have hs : (∑ j, ∑ k, mu j * mu k * (a j - a k) ^ 2) =
      (∑ j, ∑ k, if j < k then mu j * mu k * (a j - a k) ^ 2 else 0) +
        (∑ j, ∑ k, if j < k then mu j * mu k * (a j - a k) ^ 2 else 0) := by
    calc
      _ = ∑ j, ∑ k,
          ((if j < k then mu j * mu k * (a j - a k) ^ 2 else 0) +
            (if k < j then mu k * mu j * (a k - a j) ^ 2 else 0)) := by
        apply Finset.sum_congr rfl
        intro j _
        apply Finset.sum_congr rfl
        intro k _
        exact hsplit j k
      _ = _ := by
        simp only [Finset.sum_add_distrib]
        congr 1
        exact Finset.sum_comm
  have hv := weighted_variance_identity mu a
  linarith

/-- Exactly the appendix's strict-pair positive quadratic-form identity. -/
theorem fisher_information_quadratic_identity [LinearOrder J] (eta a : J → ℝ)
    (nu : ℝ) (hn : 0 < nu) :
    (nu + total eta) *
      (∑ j, ∑ k, a j * localInformation eta (fun j => exp (eta j)) nu j k * a k) =
      nu * (∑ j, exp (eta j) * a j ^ 2) +
        ∑ j, ∑ k, if j < k then
          exp (eta j) * exp (eta k) * (a j - a k) ^ 2 else 0 := by
  rw [local_information_bilinear eta (fun j => exp (eta j)) a a nu hn]
  rw [← weighted_variance_strict_pairs (fun j => exp (eta j)) a]
  have hp := ne_of_gt (denominator_pos eta nu hn)
  simp only [total, moment, ← sq] at *
  field_simp <;> ring

theorem study_directional_curvature_nonneg (eta y a : J → ℝ) (nu : ℝ)
    (hn : 0 < nu) (hy : ∀ j, 0 ≤ y j) :
    0 ≤ ((∑ j, y j) + nu) *
      (moment eta (fun j => a j * a j) / (nu + total eta) -
        moment eta a * moment eta a / (nu + total eta) ^ 2) := by
  have hvar := weighted_variance_identity (fun j => exp (eta j)) a
  have hpairs : 0 ≤ ∑ j, ∑ k, exp (eta j) * exp (eta k) * (a j - a k) ^ 2 :=
    Finset.sum_nonneg (fun j _ => Finset.sum_nonneg (fun k _ => by positivity))
  have hs : 0 ≤ ∑ j, exp (eta j) * a j ^ 2 :=
    Finset.sum_nonneg (fun j _ => by positivity)
  have hnum : 0 ≤ (nu + total eta) * moment eta (fun j => a j * a j) -
      moment eta a * moment eta a := by
    have hprod := mul_nonneg (le_of_lt hn) hs
    unfold total moment
    simp only [← sq] at *
    nlinarith
  have hd := ne_of_gt (denominator_pos eta nu hn)
  have heq :
      moment eta (fun j => a j * a j) / (nu + total eta) -
        moment eta a * moment eta a / (nu + total eta) ^ 2 =
      ((nu + total eta) * moment eta (fun j => a j * a j) -
        moment eta a * moment eta a) / (nu + total eta) ^ 2 := by
    field_simp
  rw [heq]
  apply mul_nonneg
  · exact add_nonneg (Finset.sum_nonneg (fun j _ => hy j)) (le_of_lt hn)
  · exact div_nonneg hnum (sq_nonneg _)

theorem study_line_convex (eta y a : J → ℝ) (nu : ℝ)
    (hn : 0 < nu) (hy : ∀ j, 0 ≤ y j) :
    ConvexOn ℝ Set.univ (fun t : ℝ => studyNll (fun j => eta j + t * a j) y nu) := by
  apply convexOn_of_hasDerivWithinAt2_nonneg
    (f' := fun x => ((∑ j, y j) + nu) * moment (fun j => eta j + x * a j) a /
      (nu + total (fun j => eta j + x * a j)) - ∑ j, y j * a j)
    (f'' := fun x => ((∑ j, y j) + nu) *
      (moment (fun j => eta j + x * a j) (fun j => a j * a j) /
        (nu + total (fun j => eta j + x * a j)) -
      moment (fun j => eta j + x * a j) a * moment (fun j => eta j + x * a j) a /
        (nu + total (fun j => eta j + x * a j)) ^ 2))
    convex_univ
  · intro x _
    exact (study_directional_hasDerivAt eta y a nu x hn).continuousAt.continuousWithinAt
  · intro x _
    exact (study_directional_hasDerivAt eta y a nu x hn).hasDerivWithinAt
  · intro x _
    exact (study_score_hasDerivAt eta y a a nu x hn).hasDerivWithinAt
  · intro x _
    exact study_directional_curvature_nonneg _ y a nu hn hy

/-- Fixed-shape convexity in the entire finite log-mean vector, not just at one point. -/
theorem study_convex (y : J → ℝ) (nu : ℝ) (hn : 0 < nu) (hy : ∀ j, 0 ≤ y j) :
    ConvexOn ℝ Set.univ (fun eta : J → ℝ => studyNll eta y nu) := by
  refine ⟨convex_univ, ?_⟩
  intro x _ z _ a b ha hb hab
  have h := (study_line_convex x y (fun j => z j - x j) nu hn hy).2
    (Set.mem_univ (0 : ℝ)) (Set.mem_univ (1 : ℝ)) ha hb hab
  have hx : (fun j => x j + (0 : ℝ) * (z j - x j)) = x := by
    funext j
    ring
  have hz : (fun j => x j + (1 : ℝ) * (z j - x j)) = z := by
    funext j
    ring
  have hm : (fun j => x j + (a • (0 : ℝ) + b • (1 : ℝ)) * (z j - x j)) =
      a • x + b • z := by
    funext j
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    have heq : a = 1 - b := by linarith
    rw [heq]
    ring
  simpa only [hx, hz, hm] using h

/-- Any finite sum of clustered study objectives remains convex under an affine design. -/
theorem regression_convex {P : Type*} [Fintype P]
    (offset : I → J → ℝ) (X : I → J → P → ℝ) (y : I → J → ℝ) (nu : I → ℝ)
    (hn : ∀ i, 0 < nu i) (hy : ∀ i j, 0 ≤ y i j) :
    ConvexOn ℝ Set.univ (fun beta : P → ℝ =>
      ∑ i, studyNll (fun j => offset i j + ∑ p, X i j p * beta p) (y i) (nu i)) := by
  refine ⟨convex_univ, ?_⟩
  intro beta _ gamma _ a b ha hb hab
  have hlin (i : I) :
      (fun j => offset i j + ∑ p, X i j p * (a • beta + b • gamma) p) =
      a • (fun j => offset i j + ∑ p, X i j p * beta p) +
        b • (fun j => offset i j + ∑ p, X i j p * gamma p) := by
    funext j
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, mul_add,
      Finset.sum_add_distrib]
    have hsum (c : ℝ) (v : P → ℝ) :
        (∑ p, X i j p * (c * v p)) = c * ∑ p, X i j p * v p := by
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
    _ ≤ ∑ i, (a * studyNll (fun j => offset i j + ∑ p, X i j p * beta p) (y i) (nu i) +
        b * studyNll (fun j => offset i j + ∑ p, X i j p * gamma p) (y i) (nu i)) := by
      apply Finset.sum_le_sum
      intro i _
      exact (study_convex (y i) (nu i) (hn i) (hy i)).2
        (Set.mem_univ _) (Set.mem_univ _) ha hb hab
    _ = _ := by simp [Finset.sum_add_distrib, Finset.mul_sum, smul_eq_mul]

end CbmrProofs.Clustered

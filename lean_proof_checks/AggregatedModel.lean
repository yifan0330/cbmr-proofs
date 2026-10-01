import lean_proof_checks.AggregatedChain
import lean_proof_checks.AggregatedModerators
import Mathlib.Data.Matrix.Mul

/-! Concrete matched-likelihood moderator and spatial Hessians. The iterated
derivative statements explicitly identify the algebraic blocks with derivatives
of the original likelihood, including both inner-map curvature terms. -/
namespace CbmrProofs.AggregatedNB

open scoped BigOperators Topology
open NegativeBinomial Filter
noncomputable section
set_option backward.isDefEq.respectTransparency false

variable {ι κ ν : Type*} [Fintype ι] [Fintype κ] [Fintype ν]

omit [Fintype ι] [Fintype κ] [Fintype ν] in
theorem shift_zero (gamma v : κ → ℝ) : CbmrProofs.AggregatedNB.shift gamma v 0 = gamma := by
  funext k
  simp [CbmrProofs.AggregatedNB.shift]

omit [Fintype κ] [Fintype ν] in
theorem psi_rho_rho_second (y S : ι → ℝ) (r A : ℝ)
    (hr : 0 < r) (hy : ∀ i, 0 ≤ y i) :
    HasDerivAt (fun t => deriv (fun q => aggregatedPsi y q A S) t) (rhoRho y r) r := by
  apply (rhoScore_rho_hasDerivAt y r A S hr hy).congr_of_eventuallyEq
  filter_upwards [lt_mem_nhds hr] with t ht
  exact (psi_rho_hasDerivAt y t A S ht hy).deriv

omit [Fintype κ] [Fintype ν] in
theorem psi_A_A_second (y S : ι → ℝ) (r A : ℝ) (hA : 0 < A) :
    HasDerivAt (fun t => deriv (fun a => aggregatedPsi y r a S) t) (aA y r A S) A := by
  apply (aScore_A_hasDerivAt y r A S hA).congr_of_eventuallyEq
  filter_upwards [lt_mem_nhds hA] with t ht
  exact (psi_A_hasDerivAt y r t S ht).deriv

omit [Fintype κ] [Fintype ν] in
theorem psi_rho_A_second (y S : ι → ℝ) (r A : ℝ)
    (hr : 0 < r) (hA : 0 < A) (hy : ∀ i, 0 ≤ y i) :
    HasDerivAt (fun a => deriv (fun q => aggregatedPsi y q a S) r) (rhoA A S) A := by
  apply (rhoScore_A_hasDerivAt y r A S hA).congr_of_eventuallyEq
  exact Filter.Eventually.of_forall (fun a => (psi_rho_hasDerivAt y r a S hr hy).deriv)

omit [Fintype κ] [Fintype ν] in
theorem psi_A_rho_second (y S : ι → ℝ) (r A : ℝ) (hA : 0 < A) :
    HasDerivAt (fun q => deriv (fun a => aggregatedPsi y q a S) A) (rhoA A S) r := by
  apply (aScore_rho_hasDerivAt y r A S).congr_of_eventuallyEq
  exact Filter.Eventually.of_forall (fun q => (psi_A_hasDerivAt y q A S hA).deriv)

omit [Fintype κ] [Fintype ν] in
theorem psi_spatial_iterated_deriv (y S v w : ι → ℝ) (r A : ℝ) (hA : 0 < A) :
    HasDerivAt (fun t => deriv (fun s => aggregatedPsi y r A
      (fun i => S i + t * w i + s * v i)) 0)
      (∑ i, ((r + y i) * Real.exp (S i) * A / (Real.exp (S i) + A) ^ 2) *
        w i * v i) 0 := by
  apply (psi_eta_direction_second y r A S v w hA).congr_of_eventuallyEq
  apply Filter.Eventually.of_forall
  intro t
  exact (hasDerivAt_aggregatedPsi_log_intensity y r A (fun i => S i + t * w i) v hA).deriv

omit [Fintype κ] [Fintype ν] in
theorem psi_spatial_matrix_hessian [DecidableEq ι]
    (y S : ι → ℝ) (B : Matrix ι κ ℝ) (p q : κ) (r A : ℝ) (hA : 0 < A) :
    HasDerivAt (fun t => deriv (fun s => aggregatedPsi y r A
      (fun i => S i + t * B i q + s * B i p)) 0)
      ((B.transpose * Matrix.diagonal
        (fun i => (r + y i) * Real.exp (S i) * A / (Real.exp (S i) + A) ^ 2) * B) p q) 0 := by
  have he : (B.transpose * Matrix.diagonal
      (fun i => (r + y i) * Real.exp (S i) * A / (Real.exp (S i) + A) ^ 2) * B) p q =
      ∑ i, ((r + y i) * Real.exp (S i) * A / (Real.exp (S i) + A) ^ 2) * B i q * B i p := by
    rw [Matrix.mul_apply]
    simp only [Matrix.mul_diagonal, Matrix.transpose_apply]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [he]
  exact psi_spatial_iterated_deriv y S (fun i => B i p) (fun i => B i q) r A hA

omit [Fintype κ] [Fintype ν] in
theorem psi_rho_eta_second (y S v : ι → ℝ) (r A : ℝ)
    (hr : 0 < r) (hA : 0 < A) (hy : ∀ i, 0 ≤ y i) :
    HasDerivAt (fun t => deriv (fun q => aggregatedPsi y q A (fun i => S i + t * v i)) r)
      (∑ i, Real.exp (S i) / (Real.exp (S i) + A) * v i) 0 := by
  apply (rhoScore_eta_direction_hasDerivAt y r A S v hA).congr_of_eventuallyEq
  exact Filter.Eventually.of_forall (fun t =>
    (psi_rho_hasDerivAt y r A (fun i => S i + t * v i) hr hy).deriv)

omit [Fintype κ] [Fintype ν] in
theorem psi_A_eta_second (y S v : ι → ℝ) (r A : ℝ) (hA : 0 < A) :
    HasDerivAt (fun t => deriv (fun a => aggregatedPsi y r a (fun i => S i + t * v i)) A)
      (∑ i, (-(r + y i) * Real.exp (S i) / (Real.exp (S i) + A) ^ 2) * v i) 0 := by
  apply (aScore_eta_direction_hasDerivAt y r A S v hA).congr_of_eventuallyEq
  exact Filter.Eventually.of_forall (fun t =>
    (psi_A_hasDerivAt y r A (fun i => S i + t * v i) hA).deriv)

omit [Fintype κ] [Fintype ν] in
theorem psi_eta_rho_second (y S v : ι → ℝ) (r A : ℝ) (hA : 0 < A) :
    HasDerivAt (fun q => deriv (fun t => aggregatedPsi y q A (fun i => S i + t * v i)) 0)
      (∑ i, Real.exp (S i) / (Real.exp (S i) + A) * v i) r := by
  have h := HasDerivAt.fun_sum (u := Finset.univ) (fun i _ =>
    (etaScore_rho_hasDerivAt (y i) r A (S i)).mul_const (v i))
  apply h.congr_of_eventuallyEq
  exact Filter.Eventually.of_forall (fun q =>
    (hasDerivAt_aggregatedPsi_log_intensity y q A S v hA).deriv)

omit [Fintype κ] [Fintype ν] in
theorem psi_eta_A_second (y S v : ι → ℝ) (r A : ℝ) (hA : 0 < A) :
    HasDerivAt (fun a => deriv (fun t => aggregatedPsi y r a (fun i => S i + t * v i)) 0)
      (∑ i, (-(r + y i) * Real.exp (S i) / (Real.exp (S i) + A) ^ 2) * v i) A := by
  have h := HasDerivAt.fun_sum (u := Finset.univ) (fun i _ =>
    (etaScore_A_hasDerivAt (y i) r A (S i) hA).mul_const (v i))
  apply h.congr_of_eventuallyEq
  filter_upwards [lt_mem_nhds hA] with a ha
  exact (hasDerivAt_aggregatedPsi_log_intensity y r a S v ha).deriv

def matchedPsi (alpha : ℝ) (Z : ι → κ → ℝ) (y S : ν → ℝ) (gamma : κ → ℝ) : ℝ :=
  aggregatedPsi y (moderatorRho alpha Z gamma) (moderatorA alpha Z gamma) S

def matchedGradient (alpha : ℝ) (Z : ι → κ → ℝ) (y S : ν → ℝ)
    (gamma v : κ → ℝ) : ℝ :=
  rhoScore y (moderatorRho alpha Z gamma) (moderatorA alpha Z gamma) S *
    rGradient alpha Z gamma v +
  aScore y (moderatorRho alpha Z gamma) (moderatorA alpha Z gamma) S *
    aGradient alpha Z gamma v

def matchedHessian (alpha : ℝ) (Z : ι → κ → ℝ) (y S : ν → ℝ)
    (gamma v w : κ → ℝ) : ℝ :=
  rhoRho y (moderatorRho alpha Z gamma) * rGradient alpha Z gamma v * rGradient alpha Z gamma w +
    rhoA (moderatorA alpha Z gamma) S *
      (rGradient alpha Z gamma v * aGradient alpha Z gamma w +
        aGradient alpha Z gamma v * rGradient alpha Z gamma w) +
    aA y (moderatorRho alpha Z gamma) (moderatorA alpha Z gamma) S *
      aGradient alpha Z gamma v * aGradient alpha Z gamma w +
    rhoScore y (moderatorRho alpha Z gamma) (moderatorA alpha Z gamma) S *
      rHessian alpha Z gamma v w +
    aScore y (moderatorRho alpha Z gamma) (moderatorA alpha Z gamma) S *
      aHessian alpha Z gamma v w

theorem matchedPsi_hasDerivAt [Nonempty ι] (alpha : ℝ) (Z : ι → κ → ℝ)
    (y S : ν → ℝ) (gamma v : κ → ℝ) (x : ℝ) (ha : 0 < alpha) (hy : ∀ i, 0 ≤ y i) :
    HasDerivAt (fun t => matchedPsi alpha Z y S (shift gamma v t))
      (matchedGradient alpha Z y S (shift gamma v x) v) x :=
  psi_path_hasDerivAt y S (moderatorRho_hasDerivAt alpha Z gamma v x ha)
    (moderatorA_hasDerivAt alpha Z gamma v x ha)
    (moderatorRho_pos alpha Z _ ha) (moderatorA_pos alpha Z _ ha) hy

theorem matchedGradient_hasDerivAt [Nonempty ι] (alpha : ℝ) (Z : ι → κ → ℝ)
    (y S : ν → ℝ) (gamma v w : κ → ℝ) (x : ℝ) (ha : 0 < alpha) (hy : ∀ i, 0 ≤ y i) :
    HasDerivAt (fun t => matchedGradient alpha Z y S (shift gamma w t) v)
      (matchedHessian alpha Z y S (shift gamma w x) v w) x :=
  composite_score_hasDerivAt y S (moderatorRho_hasDerivAt alpha Z gamma w x ha)
    (moderatorA_hasDerivAt alpha Z gamma w x ha)
    (rGradient_hasDerivAt alpha Z gamma v w x ha) (aGradient_hasDerivAt alpha Z gamma v w x ha)
    (moderatorRho_pos alpha Z _ ha) (moderatorA_pos alpha Z _ ha) hy

/-- The concrete Hessian is a mixed second derivative of the likelihood itself. -/
theorem matchedPsi_iterated_deriv [Nonempty ι] (alpha : ℝ) (Z : ι → κ → ℝ)
    (y S : ν → ℝ) (gamma v w : κ → ℝ) (ha : 0 < alpha) (hy : ∀ i, 0 ≤ y i) :
    HasDerivAt (fun t => deriv (fun s => matchedPsi alpha Z y S
      (shift (shift gamma w t) v s)) 0) (matchedHessian alpha Z y S gamma v w) 0 := by
  have h := matchedGradient_hasDerivAt alpha Z y S gamma v w 0 ha hy
  rw [shift_zero] at h
  apply h.congr_of_eventuallyEq
  apply Filter.Eventually.of_forall
  intro t
  simpa only [shift_zero] using
    (matchedPsi_hasDerivAt alpha Z y S (shift gamma w t) v 0 ha hy).deriv

/-- Differentiating the actual spatial derivative gives the spatial/global block. -/
theorem matched_spatial_global_iterated_deriv [Nonempty ι] (alpha : ℝ)
    (Z : ι → κ → ℝ) (y S B : ν → ℝ) (gamma v : κ → ℝ) (ha : 0 < alpha) :
    HasDerivAt (fun t => deriv (fun s => matchedPsi alpha Z y
      (fun i => S i + s * B i) (shift gamma v t)) 0)
      (∑ i, B i * (Real.exp (S i) /
          (Real.exp (S i) + moderatorA alpha Z gamma) * rGradient alpha Z gamma v -
        (moderatorRho alpha Z gamma + y i) * Real.exp (S i) /
          (Real.exp (S i) + moderatorA alpha Z gamma) ^ 2 * aGradient alpha Z gamma v)) 0 := by
  have h := spatial_global_score_hasDerivAt y S B
    (moderatorRho_hasDerivAt alpha Z gamma v 0 ha)
    (moderatorA_hasDerivAt alpha Z gamma v 0 ha)
    (moderatorA_pos alpha Z _ ha)
  rw [shift_zero] at h
  apply h.congr_of_eventuallyEq
  apply Filter.Eventually.of_forall
  intro t
  have ht := (hasDerivAt_aggregatedPsi_log_intensity y
    (moderatorRho alpha Z (shift gamma v t)) (moderatorA alpha Z (shift gamma v t))
    S B (moderatorA_pos alpha Z _ ha)).deriv
  simpa [matchedPsi, etaScore, mul_comm] using ht

omit [Fintype ι] in
theorem grouped_matchedPsi_hasDerivAt {G : Type*} [Fintype G]
    {I : G → Type*} [∀ g, Fintype (I g)] [∀ g, Nonempty (I g)]
    (alpha : G → ℝ) (Z : ∀ g, I g → κ → ℝ) (y S : G → ν → ℝ)
    (gamma v : κ → ℝ) (x : ℝ) (ha : ∀ g, 0 < alpha g) (hy : ∀ g i, 0 ≤ y g i) :
    HasDerivAt (fun t => ∑ g, matchedPsi (alpha g) (Z g) (y g) (S g) (shift gamma v t))
      (∑ g, matchedGradient (alpha g) (Z g) (y g) (S g) (shift gamma v x) v) x :=
  HasDerivAt.fun_sum (u := Finset.univ) (fun g _ =>
    matchedPsi_hasDerivAt (alpha g) (Z g) (y g) (S g) gamma v x (ha g) (hy g))

omit [Fintype ι] in
theorem grouped_matchedPsi_iterated_deriv {G : Type*} [Fintype G]
    {I : G → Type*} [∀ g, Fintype (I g)] [∀ g, Nonempty (I g)]
    (alpha : G → ℝ) (Z : ∀ g, I g → κ → ℝ) (y S : G → ν → ℝ)
    (gamma v w : κ → ℝ) (ha : ∀ g, 0 < alpha g) (hy : ∀ g i, 0 ≤ y g i) :
    HasDerivAt (fun t => deriv (fun s =>
      ∑ g, matchedPsi (alpha g) (Z g) (y g) (S g) (shift (shift gamma w t) v s)) 0)
      (∑ g, matchedHessian (alpha g) (Z g) (y g) (S g) gamma v w) 0 := by
  have h := HasDerivAt.fun_sum (u := Finset.univ) (fun g _ =>
    matchedGradient_hasDerivAt (alpha g) (Z g) (y g) (S g) gamma v w 0 (ha g) (hy g))
  simp only [shift_zero] at h
  apply h.congr_of_eventuallyEq
  apply Filter.Eventually.of_forall
  intro t
  simpa only [shift_zero] using
    (grouped_matchedPsi_hasDerivAt alpha Z y S (shift gamma w t) v 0 ha hy).deriv

end
end CbmrProofs.AggregatedNB

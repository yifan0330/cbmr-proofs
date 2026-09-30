import lean_proof_checks.AggregatedNB

/-! Moderator first and mixed second derivatives for arbitrary finite study and
coefficient index types. The original dispersion `alpha` is held fixed.
The bilinear Hessian formulas specialize to matrix entries by basis directions. -/
namespace CbmrProofs.AggregatedNB

open scoped BigOperators
open NegativeBinomial
noncomputable section
set_option backward.isDefEq.respectTransparency false

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

def linearPredictor (Z : ι → κ → ℝ) (gamma : κ → ℝ) (i : ι) : ℝ :=
  ∑ k, Z i k * gamma k

def shift (gamma v : κ → ℝ) (t : ℝ) : κ → ℝ := fun k => gamma k + t * v k

def moment (k : ℝ) (Z : ι → κ → ℝ) (gamma : κ → ℝ) : ℝ :=
  ∑ i, Real.exp (k * linearPredictor Z gamma i)

def momentGradient (k : ℝ) (Z : ι → κ → ℝ) (gamma v : κ → ℝ) : ℝ :=
  ∑ i, k * Real.exp (k * linearPredictor Z gamma i) * linearPredictor Z v i

def momentHessian (k : ℝ) (Z : ι → κ → ℝ) (gamma v w : κ → ℝ) : ℝ :=
  ∑ i, k ^ 2 * Real.exp (k * linearPredictor Z gamma i) *
    linearPredictor Z v i * linearPredictor Z w i

theorem momentGradient_one (Z : ι → κ → ℝ) (gamma v : κ → ℝ) :
    momentGradient 1 Z gamma v =
      ∑ i, Real.exp (linearPredictor Z gamma i) * linearPredictor Z v i := by
  simp [momentGradient]

theorem momentHessian_one (Z : ι → κ → ℝ) (gamma v w : κ → ℝ) :
    momentHessian 1 Z gamma v w =
      ∑ i, Real.exp (linearPredictor Z gamma i) *
        linearPredictor Z v i * linearPredictor Z w i := by
  simp [momentHessian]

theorem moment_one (Z : ι → κ → ℝ) (gamma : κ → ℝ) :
    moment 1 Z gamma = ∑ i, Real.exp (linearPredictor Z gamma i) := by
  simp [moment]

theorem moment_two (Z : ι → κ → ℝ) (gamma : κ → ℝ) :
    moment 2 Z gamma = ∑ i, Real.exp (linearPredictor Z gamma i) ^ 2 := by
  simp [moment, two_mul, Real.exp_add, pow_two]

theorem moment_pos [Nonempty ι] (k : ℝ) (Z : ι → κ → ℝ) (gamma : κ → ℝ) :
    0 < moment k Z gamma := by
  apply Finset.sum_pos
  · intro i _
    exact Real.exp_pos _
  · exact Finset.univ_nonempty

omit [Fintype ι] in
theorem linearPredictor_hasDerivAt (Z : ι → κ → ℝ) (gamma v : κ → ℝ)
    (i : ι) (x : ℝ) :
    HasDerivAt (fun t => linearPredictor Z (shift gamma v t) i)
      (linearPredictor Z v i) x := by
  convert! HasDerivAt.fun_sum (u := Finset.univ) (fun k _ =>
    (((hasDerivAt_id x).mul_const (v k)).const_add (gamma k)).const_mul (Z i k)) using 1
  simp [linearPredictor]

theorem moment_hasDerivAt (k : ℝ) (Z : ι → κ → ℝ) (gamma v : κ → ℝ) (x : ℝ) :
    HasDerivAt (fun t => moment k Z (shift gamma v t))
      (momentGradient k Z (shift gamma v x) v) x := by
  convert! HasDerivAt.fun_sum (u := Finset.univ) (fun i _ =>
    ((linearPredictor_hasDerivAt Z gamma v i x).const_mul k).exp) using 1
  unfold momentGradient
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem momentGradient_hasDerivAt (k : ℝ) (Z : ι → κ → ℝ)
    (gamma v w : κ → ℝ) (x : ℝ) :
    HasDerivAt (fun t => momentGradient k Z (shift gamma w t) v)
      (momentHessian k Z (shift gamma w x) v w) x := by
  convert! HasDerivAt.fun_sum (u := Finset.univ) (fun i _ =>
    ((((linearPredictor_hasDerivAt Z gamma w i x).const_mul k).exp).const_mul k).mul_const
      (linearPredictor Z v i)) using 1
  unfold momentHessian
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- `momentGradient 1` is `u₁·v`; `momentGradient 2` is `2 u₂·v`. -/
theorem momentGradient_two (Z : ι → κ → ℝ) (gamma v : κ → ℝ) :
    momentGradient 2 Z gamma v =
      2 * ∑ i, Real.exp (linearPredictor Z gamma i) ^ 2 * linearPredictor Z v i := by
  simp [momentGradient, two_mul, Real.exp_add, pow_two, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- `momentHessian 2` is the bilinear form `4 vᵀ M₂ w`. -/
theorem momentHessian_two (Z : ι → κ → ℝ) (gamma v w : κ → ℝ) :
    momentHessian 2 Z gamma v w =
      4 * ∑ i, Real.exp (linearPredictor Z gamma i) ^ 2 *
        linearPredictor Z v i * linearPredictor Z w i := by
  simp only [momentHessian, two_mul, Real.exp_add, pow_two, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

def momentLogGradient (k : ℝ) (Z : ι → κ → ℝ) (gamma v : κ → ℝ) : ℝ :=
  momentGradient k Z gamma v / moment k Z gamma

def momentLogHessian (k : ℝ) (Z : ι → κ → ℝ) (gamma v w : κ → ℝ) : ℝ :=
  momentHessian k Z gamma v w / moment k Z gamma -
    momentGradient k Z gamma v * momentGradient k Z gamma w / moment k Z gamma ^ 2

theorem logMoment_hasDerivAt [Nonempty ι] (k : ℝ) (Z : ι → κ → ℝ)
    (gamma v : κ → ℝ) (x : ℝ) :
    HasDerivAt (fun t => Real.log (moment k Z (shift gamma v t)))
      (momentLogGradient k Z (shift gamma v x) v) x :=
  (moment_hasDerivAt k Z gamma v x).log (ne_of_gt (moment_pos k Z _))

theorem momentLogGradient_hasDerivAt [Nonempty ι] (k : ℝ) (Z : ι → κ → ℝ)
    (gamma v w : κ → ℝ) (x : ℝ) :
    HasDerivAt (fun t => momentLogGradient k Z (shift gamma w t) v)
      (momentLogHessian k Z (shift gamma w x) v w) x := by
  have hd := ne_of_gt (moment_pos k Z (shift gamma w x))
  convert! (momentGradient_hasDerivAt k Z gamma v w x).div
    (moment_hasDerivAt k Z gamma w x) hd using 1
  unfold momentLogHessian
  field_simp

def moderatorA (alpha : ℝ) (Z : ι → κ → ℝ) (gamma : κ → ℝ) : ℝ :=
  rateA alpha (moment 1 Z gamma) (moment 2 Z gamma)

def moderatorRho (alpha : ℝ) (Z : ι → κ → ℝ) (gamma : κ → ℝ) : ℝ :=
  rho alpha (moment 1 Z gamma) (moment 2 Z gamma)

def aGradient (alpha : ℝ) (Z : ι → κ → ℝ) (gamma v : κ → ℝ) : ℝ :=
  moderatorA alpha Z gamma *
    (momentLogGradient 1 Z gamma v - momentLogGradient 2 Z gamma v)

def rGradient (alpha : ℝ) (Z : ι → κ → ℝ) (gamma v : κ → ℝ) : ℝ :=
  moderatorRho alpha Z gamma *
    (2 * momentLogGradient 1 Z gamma v - momentLogGradient 2 Z gamma v)

def aHessian (alpha : ℝ) (Z : ι → κ → ℝ) (gamma v w : κ → ℝ) : ℝ :=
  moderatorA alpha Z gamma *
    ((momentLogGradient 1 Z gamma v - momentLogGradient 2 Z gamma v) *
      (momentLogGradient 1 Z gamma w - momentLogGradient 2 Z gamma w) +
      momentLogHessian 1 Z gamma v w - momentLogHessian 2 Z gamma v w)

def rHessian (alpha : ℝ) (Z : ι → κ → ℝ) (gamma v w : κ → ℝ) : ℝ :=
  moderatorRho alpha Z gamma *
    ((2 * momentLogGradient 1 Z gamma v - momentLogGradient 2 Z gamma v) *
      (2 * momentLogGradient 1 Z gamma w - momentLogGradient 2 Z gamma w) +
      2 * momentLogHessian 1 Z gamma v w - momentLogHessian 2 Z gamma v w)

theorem moderatorA_pos [Nonempty ι] (alpha : ℝ) (Z : ι → κ → ℝ)
    (gamma : κ → ℝ) (ha : 0 < alpha) : 0 < moderatorA alpha Z gamma :=
  (parameter_pos ha (moment_pos 1 Z gamma) (moment_pos 2 Z gamma)).2.1

theorem moderatorRho_pos [Nonempty ι] (alpha : ℝ) (Z : ι → κ → ℝ)
    (gamma : κ → ℝ) (ha : 0 < alpha) : 0 < moderatorRho alpha Z gamma :=
  (parameter_pos ha (moment_pos 1 Z gamma) (moment_pos 2 Z gamma)).1

theorem log_moderatorA [Nonempty ι] (alpha : ℝ) (Z : ι → κ → ℝ)
    (gamma : κ → ℝ) (ha : 0 < alpha) :
    Real.log (moderatorA alpha Z gamma) =
      Real.log (moment 1 Z gamma) - Real.log (moment 2 Z gamma) - Real.log alpha := by
  unfold moderatorA rateA
  rw [Real.log_div (ne_of_gt (moment_pos 1 Z gamma))
    (mul_ne_zero (ne_of_gt ha) (ne_of_gt (moment_pos 2 Z gamma))),
    Real.log_mul (ne_of_gt ha) (ne_of_gt (moment_pos 2 Z gamma))]
  ring

theorem log_moderatorRho [Nonempty ι] (alpha : ℝ) (Z : ι → κ → ℝ)
    (gamma : κ → ℝ) (ha : 0 < alpha) :
    Real.log (moderatorRho alpha Z gamma) =
      2 * Real.log (moment 1 Z gamma) - Real.log (moment 2 Z gamma) - Real.log alpha := by
  unfold moderatorRho rho
  rw [Real.log_div (pow_ne_zero 2 (ne_of_gt (moment_pos 1 Z gamma)))
    (mul_ne_zero (ne_of_gt ha) (ne_of_gt (moment_pos 2 Z gamma))),
    Real.log_mul (ne_of_gt ha) (ne_of_gt (moment_pos 2 Z gamma)), Real.log_pow]
  push_cast
  ring

theorem moderatorA_hasDerivAt [Nonempty ι] (alpha : ℝ) (Z : ι → κ → ℝ)
    (gamma v : κ → ℝ) (x : ℝ) (ha : 0 < alpha) :
    HasDerivAt (fun t => moderatorA alpha Z (shift gamma v t))
      (aGradient alpha Z (shift gamma v x) v) x := by
  have h1 := ne_of_gt (moment_pos 1 Z (shift gamma v x))
  have h2 := ne_of_gt (moment_pos 2 Z (shift gamma v x))
  convert! (moment_hasDerivAt 1 Z gamma v x).div
    ((moment_hasDerivAt 2 Z gamma v x).const_mul alpha) (mul_ne_zero (ne_of_gt ha) h2) using 1
  unfold aGradient moderatorA rateA momentLogGradient
  field_simp

theorem moderatorRho_hasDerivAt [Nonempty ι] (alpha : ℝ) (Z : ι → κ → ℝ)
    (gamma v : κ → ℝ) (x : ℝ) (ha : 0 < alpha) :
    HasDerivAt (fun t => moderatorRho alpha Z (shift gamma v t))
      (rGradient alpha Z (shift gamma v x) v) x := by
  have h1 := ne_of_gt (moment_pos 1 Z (shift gamma v x))
  have h2 := ne_of_gt (moment_pos 2 Z (shift gamma v x))
  convert! ((moment_hasDerivAt 1 Z gamma v x).pow 2).div
    ((moment_hasDerivAt 2 Z gamma v x).const_mul alpha) (mul_ne_zero (ne_of_gt ha) h2) using 1
  simp only [Pi.pow_apply]
  unfold rGradient moderatorRho rho momentLogGradient
  field_simp
  ring

theorem aGradient_hasDerivAt [Nonempty ι] (alpha : ℝ) (Z : ι → κ → ℝ)
    (gamma v w : κ → ℝ) (x : ℝ) (ha : 0 < alpha) :
    HasDerivAt (fun t => aGradient alpha Z (shift gamma w t) v)
      (aHessian alpha Z (shift gamma w x) v w) x := by
  convert! (moderatorA_hasDerivAt alpha Z gamma w x ha).mul
    ((momentLogGradient_hasDerivAt 1 Z gamma v w x).sub
      (momentLogGradient_hasDerivAt 2 Z gamma v w x)) using 1
  unfold aHessian aGradient
  simp only [Pi.sub_apply]
  ring

theorem rGradient_hasDerivAt [Nonempty ι] (alpha : ℝ) (Z : ι → κ → ℝ)
    (gamma v w : κ → ℝ) (x : ℝ) (ha : 0 < alpha) :
    HasDerivAt (fun t => rGradient alpha Z (shift gamma w t) v)
      (rHessian alpha Z (shift gamma w x) v w) x := by
  convert! (moderatorRho_hasDerivAt alpha Z gamma w x ha).mul
    (((momentLogGradient_hasDerivAt 1 Z gamma v w x).const_mul 2).sub
      (momentLogGradient_hasDerivAt 2 Z gamma v w x)) using 1
  unfold rHessian rGradient
  simp only [Pi.sub_apply]
  ring

end
end CbmrProofs.AggregatedNB

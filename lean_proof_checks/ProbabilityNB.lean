import lean_proof_checks.ProbabilityMixture
import Mathlib.RingTheory.Binomial
import Mathlib.Probability.ProbabilityMassFunction.Integrals

/-!
# Negative-binomial convolution and distributional moments

The shape is any positive real number. The probability parameter follows the
appendix convention `(1-p)^shape p^count`. Normalization is inherited from the
evaluated Gamma--Poisson integral; convolution is proved by Chu--Vandermonde.
-/

noncomputable section

open MeasureTheory ProbabilityTheory Real Polynomial
open scoped BigOperators

namespace CbmrProofs.ProbabilityNB

def mass (shape p : ℝ) (n : ℕ) : ℝ :=
  Gamma (shape + n) / (Gamma shape * (n.factorial : ℝ)) *
    (1 - p) ^ shape * p ^ n

theorem mass_pos (shape p : ℝ) (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) (n : ℕ) :
    0 < mass shape p n := by
  unfold mass
  positivity

theorem mean_parameter_ratios (shape p : ℝ)
    (hs : 0 < shape) (hp1 : p < 1) :
    shape / (shape + shape * p / (1 - p)) = 1 - p ∧
    (shape * p / (1 - p)) / (shape + shape * p / (1 - p)) = p := by
  have hq : 1 - p ≠ 0 := ne_of_gt (sub_pos.mpr hp1)
  have hden : shape + shape * p / (1 - p) = shape / (1 - p) := by
    field_simp
    ring
  rw [hden]
  constructor <;> field_simp

theorem mass_eq_mean_parameter (shape p : ℝ)
    (hs : 0 < shape) (hp1 : p < 1) (n : ℕ) :
    mass shape p n = ProbabilityMixture.nbTotalMass shape (shape * p / (1 - p)) n := by
  unfold mass ProbabilityMixture.nbTotalMass
  rw [(mean_parameter_ratios shape p hs hp1).1, (mean_parameter_ratios shape p hs hp1).2]

theorem mass_hasSum_one (shape p : ℝ) (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) :
    HasSum (mass shape p) 1 := by
  have hsum : (∑' n, ENNReal.ofReal (mass shape p n)) = 1 := by
    simp_rw [mass_eq_mean_parameter shape p hs hp1]
    exact ProbabilityMixture.nb_mass_sum_one shape _ hs (by positivity)
  have h := ENNReal.hasSum_toReal (hsum ▸ ENNReal.one_ne_top)
  have he : (∑' n, (ENNReal.ofReal (mass shape p n)).toReal) = 1 := by
    rw [← ENNReal.tsum_toReal_eq (fun _ => ENNReal.ofReal_ne_top), hsum, ENNReal.toReal_one]
  rw [he] at h
  simpa only [ENNReal.toReal_ofReal (le_of_lt (mass_pos shape p hs hp hp1 _))] using h

theorem gamma_pochhammer (shape : ℝ) (hs : 0 < shape) (n : ℕ) :
    Gamma (shape + n) = Gamma shape * (ascPochhammer ℝ n).eval shape := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Nat.cast_succ, ← add_assoc, Gamma_add_one (by positivity),
        ascPochhammer_succ_eval, ih]
      ring

theorem gamma_coefficient_multichoose (shape : ℝ) (hs : 0 < shape) (n : ℕ) :
    Gamma (shape + n) / (Gamma shape * (n.factorial : ℝ)) =
      Ring.multichoose shape n := by
  rw [gamma_pochhammer shape hs]
  have h := Ring.factorial_nsmul_multichoose_eq_ascPochhammer shape n
  rw [Polynomial.ascPochhammer_smeval_eq_eval, nsmul_eq_mul] at h
  rw [← h]
  field_simp

theorem multichoose_add (r s : ℝ) (n : ℕ) :
    Ring.multichoose (r + s) n =
      ∑ ij ∈ Finset.antidiagonal n, Ring.multichoose r ij.1 * Ring.multichoose s ij.2 := by
  have h := Ring.add_choose_eq (r := -r) (s := -s) n (Commute.all _ _)
  rw [show -r + -s = -(r + s) by ring] at h
  simp only [Ring.choose_neg', Units.smul_def, zsmul_eq_mul, Int.cast_negOnePow_natCast] at h
  have heq :
      (∑ ij ∈ Finset.antidiagonal n,
        ((-1 : ℝ) ^ ij.1 * Ring.multichoose r ij.1) *
          ((-1 : ℝ) ^ ij.2 * Ring.multichoose s ij.2)) =
      (-1 : ℝ) ^ n *
        ∑ ij ∈ Finset.antidiagonal n, Ring.multichoose r ij.1 * Ring.multichoose s ij.2 := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro ij hij
    have hn : ij.1 + ij.2 = n := Finset.mem_antidiagonal.mp hij
    rw [← hn, pow_add]
    ring
  rw [heq] at h
  exact mul_left_cancel₀ (pow_ne_zero _ (by norm_num : (-1 : ℝ) ≠ 0)) h

/-- Exact NB convolution for arbitrary positive real shapes and common probability. -/
theorem common_probability_convolution (r s p : ℝ)
    (hr : 0 < r) (hs : 0 < s) (hp1 : p < 1) (n : ℕ) :
    (∑ ij ∈ Finset.antidiagonal n, mass r p ij.1 * mass s p ij.2) =
      mass (r + s) p n := by
  unfold mass
  simp_rw [gamma_coefficient_multichoose r hr, gamma_coefficient_multichoose s hs,
    gamma_coefficient_multichoose (r + s) (add_pos hr hs)]
  rw [multichoose_add, rpow_add (sub_pos.mpr hp1), Finset.sum_mul, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro ij hij
  have hn : ij.1 + ij.2 = n := Finset.mem_antidiagonal.mp hij
  rw [← hn, pow_add]
  ring

theorem mass_successor (shape p : ℝ) (hs : 0 < shape) (hp1 : p < 1) (n : ℕ) :
    ((n : ℝ) + 1) * mass shape p (n + 1) =
      (shape * p / (1 - p)) * mass (shape + 1) p n := by
  unfold mass
  rw [Nat.cast_add, Nat.cast_one,
    show shape + ((n : ℝ) + 1) = shape + 1 + n by ring,
    Gamma_add_one (ne_of_gt hs), Nat.factorial_succ, Nat.cast_mul, Nat.cast_succ,
    rpow_add (sub_pos.mpr hp1), rpow_one, pow_succ]
  have hq : 1 - p ≠ 0 := ne_of_gt (sub_pos.mpr hp1)
  field_simp

theorem first_moment_hasSum (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) :
    HasSum (fun n : ℕ => (n : ℝ) * mass shape p n) (shape * p / (1 - p)) := by
  have h := (mass_hasSum_one (shape + 1) p (by positivity) hp hp1).mul_left
    (shape * p / (1 - p))
  simp only [mul_one] at h
  have heq : (fun n : ℕ => ((n + 1 : ℕ) : ℝ) * mass shape p (n + 1)) =
      (fun n => (shape * p / (1 - p)) * mass (shape + 1) p n) := by
    funext n
    simpa only [Nat.cast_add, Nat.cast_one] using mass_successor shape p hs hp1 n
  rw [← heq] at h
  simpa only [Nat.cast_zero, zero_mul, zero_add] using
    (HasSum.zero_add (f := fun n : ℕ => (n : ℝ) * mass shape p n) h)

theorem second_moment_hasSum (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) :
    HasSum (fun n : ℕ => (n : ℝ) ^ 2 * mass shape p n)
      ((shape * p / (1 - p)) * ((shape + 1) * p / (1 - p) + 1)) := by
  have h := ((first_moment_hasSum (shape + 1) p (by positivity) hp hp1).add
    (mass_hasSum_one (shape + 1) p (by positivity) hp hp1)).mul_left
      (shape * p / (1 - p))
  have heq : (fun n : ℕ => ((n + 1 : ℕ) : ℝ) ^ 2 * mass shape p (n + 1)) =
      (fun n : ℕ => (shape * p / (1 - p)) *
        ((n : ℝ) * mass (shape + 1) p n + mass (shape + 1) p n)) := by
    funext n
    simp only [Nat.cast_add, Nat.cast_one]
    calc
      _ = ((n : ℝ) + 1) * (((n : ℝ) + 1) * mass shape p (n + 1)) := by ring
      _ = _ := by rw [mass_successor shape p hs hp1]; ring
  rw [← heq] at h
  simpa only [Nat.cast_zero, zero_pow (by decide : 2 ≠ 0), zero_mul, zero_add] using
    (HasSum.zero_add (f := fun n : ℕ => (n : ℝ) ^ 2 * mass shape p n) h)

/-- A measure with the evaluated NB masses; normalization is proved below. -/
def nbMeasure (shape p : ℝ) : Measure ℕ :=
  Measure.sum (fun n => ENNReal.ofReal (mass shape p n) • Measure.dirac n)

theorem nbMeasure_probability (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) :
    IsProbabilityMeasure (nbMeasure shape p) :=
  (mass_hasSum_one shape p hs hp hp1).isProbabilityMeasure_sum_dirac
    (fun n => le_of_lt (mass_pos shape p hs hp hp1 n))

theorem nbMeasure_mean (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) :
    (∫ n, (n : ℝ) ∂nbMeasure shape p) = shape * p / (1 - p) := by
  rw [nbMeasure, integral_sum_dirac (fun _ => ENNReal.ofReal_ne_top)]
  simp only [ENNReal.toReal_ofReal (le_of_lt (mass_pos shape p hs hp hp1 _)),
    smul_eq_mul, mul_comm (mass shape p _) (_ : ℝ)]
  exact (first_moment_hasSum shape p hs hp hp1).tsum_eq

theorem nbMeasure_second_moment (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) :
    (∫ n, (n : ℝ) ^ 2 ∂nbMeasure shape p) =
      (shape * p / (1 - p)) * ((shape + 1) * p / (1 - p) + 1) := by
  rw [nbMeasure, integral_sum_dirac (fun _ => ENNReal.ofReal_ne_top)]
  simp only [ENNReal.toReal_ofReal (le_of_lt (mass_pos shape p hs hp hp1 _)),
    smul_eq_mul, mul_comm (mass shape p _) (_ : ℝ)]
  exact (second_moment_hasSum shape p hs hp hp1).tsum_eq

theorem nbMeasure_memLp_two (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) :
    MemLp (fun n : ℕ => (n : ℝ)) 2 (nbMeasure shape p) := by
  apply (memLp_two_iff_integrable_sq (by fun_prop)).mpr
  apply Integrable.of_integral_ne_zero
  rw [nbMeasure_second_moment shape p hs hp hp1]
  have hq : 0 < 1 - p := sub_pos.mpr hp1
  positivity

theorem nbMeasure_variance (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) :
    variance (fun n : ℕ => (n : ℝ)) (nbMeasure shape p) = shape * p / (1 - p) ^ 2 := by
  let := nbMeasure_probability shape p hs hp hp1
  rw [variance_eq_sub (nbMeasure_memLp_two shape p hs hp hp1)]
  change (∫ n, (n : ℝ) ^ 2 ∂nbMeasure shape p) -
    (∫ n, (n : ℝ) ∂nbMeasure shape p) ^ 2 = _
  rw [nbMeasure_second_moment shape p hs hp hp1, nbMeasure_mean shape p hs hp hp1]
  have hq : 1 - p ≠ 0 := ne_of_gt (sub_pos.mpr hp1)
  field_simp
  ring

theorem nbMeasure_variance_mean_dispersion (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) :
    variance (fun n : ℕ => (n : ℝ)) (nbMeasure shape p) =
      (shape * p / (1 - p)) + (1 / shape) * (shape * p / (1 - p)) ^ 2 := by
  rw [nbMeasure_variance shape p hs hp hp1]
  have hq : 1 - p ≠ 0 := ne_of_gt (sub_pos.mpr hp1)
  field_simp
  ring

theorem nbMeasure_singleton (shape p : ℝ) (n : ℕ) :
    nbMeasure shape p {n} = ENNReal.ofReal (mass shape p n) := by
  exact Measure.sum_smul_dirac_singleton

/-- Distribution-level convolution, not just an identity of mass expressions. -/
theorem nbMeasure_convolution (r s p : ℝ)
    (hr : 0 < r) (hs : 0 < s) (hp : 0 < p) (hp1 : p < 1) :
    nbMeasure r p ∗ nbMeasure s p = nbMeasure (r + s) p := by
  let := nbMeasure_probability r p hr hp hp1
  let := nbMeasure_probability s p hs hp hp1
  apply Measure.ext_of_singleton
  intro n
  rw [Measure.conv, Measure.map_apply (by fun_prop) (MeasurableSet.singleton n)]
  have hpre : (fun ij : ℕ × ℕ => ij.1 + ij.2) ⁻¹' {n} =
      (Finset.antidiagonal n : Set (ℕ × ℕ)) := by
    ext ij
    simp
  rw [hpre, ← sum_measure_singleton]
  have hcell (ij : ℕ × ℕ) :
      ((nbMeasure r p).prod (nbMeasure s p)) {ij} =
        ENNReal.ofReal (mass r p ij.1 * mass s p ij.2) := by
    rcases ij with ⟨i, j⟩
    rw [← Set.singleton_prod_singleton, Measure.prod_prod, nbMeasure_singleton,
      nbMeasure_singleton, ENNReal.ofReal_mul (le_of_lt (mass_pos r p hr hp hp1 i))]
  simp_rw [hcell]
  rw [nbMeasure_singleton]
  rw [← ENNReal.ofReal_sum_of_nonneg (fun ij _ =>
    mul_nonneg (le_of_lt (mass_pos r p hr hp hp1 ij.1))
      (le_of_lt (mass_pos s p hs hp hp1 ij.2))),
    common_probability_convolution r s p hr hs hp1 n]

theorem independent_nb_add_hasLaw {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (X Y : Ω → ℕ) (r s p : ℝ)
    (hr : 0 < r) (hs : 0 < s) (hp : 0 < p) (hp1 : p < 1)
    (hX : HasLaw X (nbMeasure r p) P) (hY : HasLaw Y (nbMeasure s p) P)
    (hind : IndepFun X Y P) :
    HasLaw (fun ω => X ω + Y ω) (nbMeasure (r + s) p) P := by
  let := nbMeasure_probability r p hr hp hp1
  let := nbMeasure_probability s p hs hp hp1
  rw [← nbMeasure_convolution r s p hr hs hp hp1]
  exact hind.hasLaw_add hX hY

/-- Any nonempty finite family of independent NB counts with common `p`
sums to an NB count with the sum of the positive real shapes. -/
theorem independent_nb_finset_sum_hasLaw {Ω ι : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (Y : ι → Ω → ℕ) (shape : ι → ℝ) (p : ℝ)
    (hs : ∀ i, 0 < shape i) (hp : 0 < p) (hp1 : p < 1)
    (hlaw : ∀ i, HasLaw (Y i) (nbMeasure (shape i) p) P)
    (hind : iIndepFun Y P) (s : Finset ι) (hne : s.Nonempty) :
    HasLaw (∑ i ∈ s, Y i) (nbMeasure (∑ i ∈ s, shape i) p) P := by
  classical
  induction s using Finset.induction with
  | empty => simp at hne
  | @insert i s hi ih =>
      by_cases hnonempty : s.Nonempty
      · rw [Finset.sum_insert hi, Finset.sum_insert hi]
        have hsum : 0 < ∑ j ∈ s, shape j := Finset.sum_pos (fun j _ => hs j) hnonempty
        have hi' : IndepFun (Y i) (∑ j ∈ s, Y j) P :=
          (hind.indepFun_finsetSum_of_notMem₀ (fun j => (hlaw j).aemeasurable) hi).symm
        exact independent_nb_add_hasLaw P _ _ _ _ p (hs i) hsum hp hp1
          (hlaw i) (ih hnonempty) hi'
      · have hempty : s = ∅ := Finset.not_nonempty_iff_eq_empty.mp hnonempty
        subst s
        simpa using hlaw i

theorem nb_count_moments {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (Y : Ω → ℕ) (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1)
    (hlaw : HasLaw Y (nbMeasure shape p) P) :
    (∫ ω, (Y ω : ℝ) ∂P) = shape * p / (1 - p) ∧
    variance (fun ω => (Y ω : ℝ)) P =
      (shape * p / (1 - p)) + (1 / shape) * (shape * p / (1 - p)) ^ 2 := by
  constructor
  · exact (hlaw.integral_comp (f := fun n : ℕ => (n : ℝ)) (by fun_prop)).trans
      (nbMeasure_mean shape p hs hp hp1)
  · have h := variance_map (X := fun n : ℕ => (n : ℝ)) (μ := P) (Y := Y)
      (by fun_prop) hlaw.aemeasurable
    rw [hlaw.map_eq] at h
    exact h.symm.trans (nbMeasure_variance_mean_dispersion shape p hs hp hp1)

/-- Identifies the derived measure with the previously normalized
Gamma--Poisson NB PMF in its mean/shape parameterization. -/
theorem nbMeasure_eq_mixture_PMF (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) :
    nbMeasure shape p =
      (ProbabilityMixture.nbPMF shape (shape * p / (1 - p)) hs (by positivity)).toMeasure := by
  apply Measure.ext_of_singleton
  intro n
  rw [nbMeasure_singleton, PMF.toMeasure_apply_singleton _ _ (MeasurableSet.singleton n)]
  change ENNReal.ofReal (mass shape p n) =
    ENNReal.ofReal (ProbabilityMixture.nbTotalMass shape (shape * p / (1 - p)) n)
  rw [mass_eq_mean_parameter shape p hs hp1]

/-- First two moments of the normalized Gamma--Poisson NB PMF itself, without
any moment assumptions. -/
theorem mixture_PMF_moments (shape mu : ℝ) (hs : 0 < shape) (hm : 0 < mu) :
    (∫ n, (n : ℝ) ∂(ProbabilityMixture.nbPMF shape mu hs hm).toMeasure) = mu ∧
    variance (fun n : ℕ => (n : ℝ))
      (ProbabilityMixture.nbPMF shape mu hs hm).toMeasure =
        mu + (1 / shape) * mu ^ 2 := by
  let p := mu / (shape + mu)
  have hp : 0 < p := by dsimp [p]; positivity
  have hp1 : p < 1 := (div_lt_one (by positivity)).mpr (by linarith)
  have hmean : shape * p / (1 - p) = mu := by
    dsimp [p]
    have hd : shape + mu ≠ 0 := by positivity
    field_simp [ne_of_gt hs, hd]
    ring
  have heq := nbMeasure_eq_mixture_PMF shape p hs hp hp1
  simp only [hmean] at heq
  constructor
  · rw [← heq, nbMeasure_mean shape p hs hp hp1, hmean]
  · rw [← heq, nbMeasure_variance_mean_dispersion shape p hs hp hp1, hmean]

end CbmrProofs.ProbabilityNB

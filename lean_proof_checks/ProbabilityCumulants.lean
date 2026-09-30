import lean_proof_checks.ProbabilityNB
import Mathlib.Probability.Moments.Basic

/-!
# Third central moments and the moment-matching boundary

The third cumulant of a distribution with a finite third moment is its third
central moment. Here that is Mathlib's actual centered integral, not a
definition equal to the NB formula.
-/

noncomputable section

open MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace CbmrProofs.ProbabilityCumulants

set_option backward.isDefEq.respectTransparency false
set_option backward.isDefEq.respectTransparency.types false

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]

omit [IsProbabilityMeasure P] in
theorem integrable_cube_of_memLp (X : Ω → ℝ) (hX : MemLp X 3 P) :
    Integrable (fun ω => X ω ^ 3) P := by
  apply (integrable_norm_iff (hX.aestronglyMeasurable.pow 3)).mp
  simpa only [Pi.pow_apply, norm_pow] using hX.integrable_norm_pow (by decide : 3 ≠ 0)

theorem third_central_moment_eq (X : Ω → ℝ)
    (h1 : Integrable X P) (h2 : Integrable (fun ω => X ω ^ 2) P)
    (h3 : Integrable (fun ω => X ω ^ 3) P) :
    centralMoment X 3 P = (∫ ω, X ω ^ 3 ∂P) -
      3 * (∫ ω, X ω ∂P) * (∫ ω, X ω ^ 2 ∂P) + 2 * (∫ ω, X ω ∂P) ^ 3 := by
  let m := ∫ ω, X ω ∂P
  change (∫ ω, (X ω - m) ^ 3 ∂P) = _
  have heq (ω : Ω) : (X ω - m) ^ 3 =
      (X ω ^ 3 - (3 * m) * X ω ^ 2) + (3 * m ^ 2) * X ω - m ^ 3 := by ring
  have hsub : Integrable (fun ω => X ω ^ 3 - (3 * m) * X ω ^ 2) P :=
    h3.sub (h2.const_mul (3 * m))
  have hleft : Integrable (fun ω => X ω ^ 3 - (3 * m) * X ω ^ 2 + (3 * m ^ 2) * X ω) P :=
    hsub.add (h1.const_mul (3 * m ^ 2))
  simp_rw [heq]
  rw [integral_sub hleft (integrable_const (m ^ 3)),
    integral_add hsub (h1.const_mul (3 * m ^ 2)),
    integral_sub h3 (h2.const_mul (3 * m)), integral_const_mul, integral_const_mul, integral_const]
  simp only [probReal_univ, smul_eq_mul, one_mul]
  dsimp [m]
  ring

theorem independent_cube_integral (X Y : Ω → ℝ)
    (hX : MemLp X 3 P) (hY : MemLp Y 3 P) (hind : IndepFun X Y P) :
    (∫ ω, (X ω + Y ω) ^ 3 ∂P) =
      (∫ ω, X ω ^ 3 ∂P) + (∫ ω, Y ω ^ 3 ∂P) +
        3 * (∫ ω, X ω ^ 2 ∂P) * (∫ ω, Y ω ∂P) +
        3 * (∫ ω, X ω ∂P) * (∫ ω, Y ω ^ 2 ∂P) := by
  have hX1 : Integrable X P := hX.integrable (by norm_num)
  have hY1 : Integrable Y P := hY.integrable (by norm_num)
  have hX2 : Integrable (fun ω => X ω ^ 2) P :=
    (hX.mono_exponent (by norm_num : (2 : ENNReal) ≤ 3)).integrable_sq
  have hY2 : Integrable (fun ω => Y ω ^ 2) P :=
    (hY.mono_exponent (by norm_num : (2 : ENNReal) ≤ 3)).integrable_sq
  have hXY : IndepFun (fun ω => X ω ^ 2) Y P :=
    hind.comp (by fun_prop : Measurable (fun x : ℝ => x ^ 2)) measurable_id
  have hYX : IndepFun X (fun ω => Y ω ^ 2) P :=
    hind.comp measurable_id (by fun_prop : Measurable (fun y : ℝ => y ^ 2))
  have hiXY : Integrable (fun ω => X ω ^ 2 * Y ω) P := hXY.integrable_mul hX2 hY1
  have hiYX : Integrable (fun ω => X ω * Y ω ^ 2) P := hYX.integrable_mul hX1 hY2
  have heq (ω : Ω) : (X ω + Y ω) ^ 3 =
      (X ω ^ 3 + Y ω ^ 3) + 3 * (X ω ^ 2 * Y ω) + 3 * (X ω * Y ω ^ 2) := by ring
  have hcubes : Integrable (fun ω => X ω ^ 3 + Y ω ^ 3) P :=
    (integrable_cube_of_memLp X hX).add (integrable_cube_of_memLp Y hY)
  have hfirst : Integrable (fun ω => X ω ^ 3 + Y ω ^ 3 + 3 * (X ω ^ 2 * Y ω)) P :=
    hcubes.add (hiXY.const_mul 3)
  simp_rw [heq]
  rw [integral_add hfirst (hiYX.const_mul 3),
    integral_add hcubes (hiXY.const_mul 3),
    integral_add (integrable_cube_of_memLp X hX) (integrable_cube_of_memLp Y hY),
    integral_const_mul, integral_const_mul]
  have hxy := hXY.integral_mul_eq_mul_integral hX2.aestronglyMeasurable hY1.aestronglyMeasurable
  have hyx := hYX.integral_mul_eq_mul_integral hX1.aestronglyMeasurable hY2.aestronglyMeasurable
  change (∫ ω, X ω ^ 2 * Y ω ∂P) = _ at hxy
  change (∫ ω, X ω * Y ω ^ 2 ∂P) = _ at hyx
  rw [hxy, hyx]
  ring

/-- Third cumulants add for independent random variables with finite third moments. -/
theorem independent_third_central_moment_add (X Y : Ω → ℝ)
    (hX : MemLp X 3 P) (hY : MemLp Y 3 P) (hind : IndepFun X Y P) :
    centralMoment (fun ω => X ω + Y ω) 3 P =
      centralMoment X 3 P + centralMoment Y 3 P := by
  let A := fun ω => X ω - ∫ w, X w ∂P
  let B := fun ω => Y ω - ∫ w, Y w ∂P
  have hA : MemLp A 3 P := hX.sub (memLp_const _)
  have hB : MemLp B 3 P := hY.sub (memLp_const _)
  have hAB : IndepFun A B P := by
    exact hind.comp (show Measurable (fun x : ℝ => x - ∫ w, X w ∂P) by fun_prop)
      (show Measurable (fun y : ℝ => y - ∫ w, Y w ∂P) by fun_prop)
  have ha0 : (∫ ω, A ω ∂P) = 0 := by
    dsimp [A]
    rw [integral_sub (hX.integrable (by norm_num)) (integrable_const _)]
    simp
  have hb0 : (∫ ω, B ω ∂P) = 0 := by
    dsimp [B]
    rw [integral_sub (hY.integrable (by norm_num)) (integrable_const _)]
    simp
  have heq : centralMoment (fun ω => X ω + Y ω) 3 P =
      ∫ ω, (A ω + B ω) ^ 3 ∂P := by
    unfold centralMoment
    rw [integral_add (hX.integrable (by norm_num)) (hY.integrable (by norm_num))]
    congr 1
    funext ω
    dsimp [A, B]
    ring
  rw [heq, independent_cube_integral A B hA hB hAB, ha0, hb0]
  simp only [mul_zero, zero_mul, add_zero]
  rfl

theorem independent_third_central_moment_sum {ι : Type*} (X : ι → Ω → ℝ)
    (hX : ∀ i, MemLp (X i) 3 P) (hind : iIndepFun X P) (s : Finset ι) :
    centralMoment (∑ i ∈ s, X i) 3 P = ∑ i ∈ s, centralMoment (X i) 3 P := by
  classical
  induction s using Finset.induction with
  | empty => simp [centralMoment_zero]
  | @insert i s hi ih =>
      rw [Finset.sum_insert hi, Finset.sum_insert hi]
      have hsum : MemLp (∑ j ∈ s, X j) 3 P := memLp_finsetSum' s (fun j _ => hX j)
      have hind' : IndepFun (X i) (∑ j ∈ s, X j) P :=
        (hind.indepFun_finsetSum_of_notMem₀ (fun j => (hX j).aemeasurable) hi).symm
      simp only [Pi.add_def]
      rw [independent_third_central_moment_add (X i) _ (hX i) hsum hind', ih]

open ProbabilityNB in
theorem nb_third_moment_hasSum (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) :
    HasSum (fun n : ℕ => (n : ℝ) ^ 3 * mass shape p n)
      ((shape * p / (1 - p)) *
        (((shape + 1) * p / (1 - p)) * ((shape + 2) * p / (1 - p) + 1) +
          2 * ((shape + 1) * p / (1 - p)) + 1)) := by
  have h := (((second_moment_hasSum (shape + 1) p (by positivity) hp hp1).add
    ((first_moment_hasSum (shape + 1) p (by positivity) hp hp1).mul_left 2)).add
      (mass_hasSum_one (shape + 1) p (by positivity) hp hp1)).mul_left (shape * p / (1 - p))
  have heq : (fun n : ℕ => ((n + 1 : ℕ) : ℝ) ^ 3 * mass shape p (n + 1)) =
      (fun n : ℕ => (shape * p / (1 - p)) *
        ((n : ℝ) ^ 2 * mass (shape + 1) p n +
          2 * ((n : ℝ) * mass (shape + 1) p n) + mass (shape + 1) p n)) := by
    funext n
    simp only [Nat.cast_add, Nat.cast_one]
    calc
      _ = ((n : ℝ) + 1) ^ 2 * (((n : ℝ) + 1) * mass shape p (n + 1)) := by ring
      _ = _ := by rw [mass_successor shape p hs hp1]; ring
  rw [← heq] at h
  simpa only [Nat.cast_zero, zero_pow (by decide : 3 ≠ 0), zero_mul, zero_add,
    show shape + 1 + 1 = shape + 2 by ring] using
    (HasSum.zero_add (f := fun n : ℕ => (n : ℝ) ^ 3 * mass shape p n) h)

open ProbabilityNB in
theorem nbMeasure_third_moment (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) :
    (∫ n, (n : ℝ) ^ 3 ∂nbMeasure shape p) =
      (shape * p / (1 - p)) *
        (((shape + 1) * p / (1 - p)) * ((shape + 2) * p / (1 - p) + 1) +
          2 * ((shape + 1) * p / (1 - p)) + 1) := by
  rw [nbMeasure, integral_sum_dirac (fun _ => ENNReal.ofReal_ne_top)]
  simp only [ENNReal.toReal_ofReal (le_of_lt (mass_pos shape p hs hp hp1 _)),
    smul_eq_mul, mul_comm (mass shape p _) (_ : ℝ)]
  exact (nb_third_moment_hasSum shape p hs hp hp1).tsum_eq

open ProbabilityNB in
theorem nbMeasure_third_central_moment (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) :
    centralMoment (fun n : ℕ => (n : ℝ)) 3 (nbMeasure shape p) =
      (shape * p / (1 - p)) + 3 * (shape * p / (1 - p)) ^ 2 / shape +
        2 * (shape * p / (1 - p)) ^ 3 / shape ^ 2 := by
  let := nbMeasure_probability shape p hs hp hp1
  have h2 := nbMeasure_memLp_two shape p hs hp hp1
  have h3 : Integrable (fun n : ℕ => (n : ℝ) ^ 3) (nbMeasure shape p) := by
    apply Integrable.of_integral_ne_zero
    rw [nbMeasure_third_moment shape p hs hp hp1]
    have hq : 0 < 1 - p := sub_pos.mpr hp1
    positivity
  rw [third_central_moment_eq _ (h2.integrable (by norm_num)) h2.integrable_sq h3,
    nbMeasure_third_moment shape p hs hp hp1, nbMeasure_second_moment shape p hs hp hp1,
    nbMeasure_mean shape p hs hp hp1]
  have hq : 1 - p ≠ 0 := ne_of_gt (sub_pos.mpr hp1)
  field_simp
  ring

open ProbabilityNB in
theorem nbMeasure_memLp_three (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1) :
    MemLp (fun n : ℕ => (n : ℝ)) 3 (nbMeasure shape p) := by
  have h3 : Integrable (fun n : ℕ => (n : ℝ) ^ 3) (nbMeasure shape p) := by
    apply Integrable.of_integral_ne_zero
    rw [nbMeasure_third_moment shape p hs hp hp1]
    have hq : 0 < 1 - p := sub_pos.mpr hp1
    positivity
  have hnorm : MemLp (fun n : ℕ => ‖(n : ℝ)‖ ^ (3 : ℝ)) 1 (nbMeasure shape p) := by
    rw [memLp_one_iff_integrable]
    simpa using h3
  have h := (memLp_norm_rpow_iff (f := fun n : ℕ => (n : ℝ))
    (μ := nbMeasure shape p) (p := (3 : ENNReal)) (q := (3 : ENNReal))
    (by fun_prop) (by norm_num) ENNReal.ofNat_ne_top).mp
  rw [ENNReal.div_self (by norm_num : (3 : ENNReal) ≠ 0) ENNReal.ofNat_ne_top] at h
  exact h (by simpa using hnorm)

open ProbabilityNB in
omit [IsProbabilityMeasure P] in
theorem nb_count_third_central_moment (Y : Ω → ℕ) (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1)
    (hlaw : HasLaw Y (nbMeasure shape p) P) :
    centralMoment (fun ω => (Y ω : ℝ)) 3 P =
      (shape * p / (1 - p)) + 3 * (shape * p / (1 - p)) ^ 2 / shape +
        2 * (shape * p / (1 - p)) ^ 3 / shape ^ 2 := by
  have hc := nbMeasure_third_central_moment shape p hs hp hp1
  unfold centralMoment at hc ⊢
  rw [nbMeasure_mean shape p hs hp hp1] at hc
  rw [(nb_count_moments P Y shape p hs hp hp1 hlaw).1]
  exact (hlaw.integral_comp
    (f := fun n : ℕ => ((n : ℝ) - shape * p / (1 - p)) ^ 3) (by fun_prop)).trans hc

open ProbabilityNB in
theorem independent_nb_third_central_moment_sum {ι : Type*} [Fintype ι]
    (Y : ι → Ω → ℕ) (shape : ℝ) (p : ι → ℝ)
    (hs : 0 < shape) (hp : ∀ i, 0 < p i) (hp1 : ∀ i, p i < 1)
    (hlaw : ∀ i, HasLaw (Y i) (nbMeasure shape (p i)) P) (hind : iIndepFun Y P) :
    centralMoment (fun ω => ∑ i : ι, (Y i ω : ℝ)) 3 P =
      ∑ i : ι, ((shape * p i / (1 - p i)) + 3 * (shape * p i / (1 - p i)) ^ 2 / shape +
        2 * (shape * p i / (1 - p i)) ^ 3 / shape ^ 2) := by
  have hmem (i : ι) : MemLp (fun ω => (Y i ω : ℝ)) 3 P := by
    have hmap : MemLp (fun n : ℕ => (n : ℝ)) 3 (P.map (Y i)) := by
      rw [(hlaw i).map_eq]
      exact nbMeasure_memLp_three shape (p i) hs (hp i) (hp1 i)
    exact hmap.comp_of_map (hlaw i).aemeasurable
  have hi : iIndepFun (fun i ω => (Y i ω : ℝ)) P :=
    hind.comp (fun _ (n : ℕ) => (n : ℝ)) (fun _ => by fun_prop)
  have heq : (fun ω => ∑ i, (Y i ω : ℝ)) = ∑ i, fun ω => (Y i ω : ℝ) := by
    funext ω
    simp
  rw [heq, independent_third_central_moment_sum _ hmem hi]
  apply Finset.sum_congr rfl
  intro i _
  exact nb_count_third_central_moment (Y i) shape (p i) hs (hp i) (hp1 i) (hlaw i)

open ProbabilityNB in
theorem mixture_PMF_third_central_moment (shape mu : ℝ) (hs : 0 < shape) (hm : 0 < mu) :
    centralMoment (fun n : ℕ => (n : ℝ)) 3
      (ProbabilityMixture.nbPMF shape mu hs hm).toMeasure =
        mu + 3 * mu ^ 2 / shape + 2 * mu ^ 3 / shape ^ 2 := by
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
  rw [← heq, nbMeasure_third_central_moment shape p hs hp hp1, hmean]

open ProbabilityNB in
/-- The appendix's higher-cumulant discrepancy, now for an actual independent
sum and an actual normalized moment-matched NB law. The mean-parameter
equalities specify the study factors; they do not assume any sum moments. -/
theorem moment_matching_third_cumulant_gap {ι : Type*} [Fintype ι]
    (Y : ι → Ω → ℕ) (p e : ι → ℝ) (alpha mu : ℝ)
    (ha : 0 < alpha) (hm : 0 < mu)
    (h1 : 0 < ∑ i, e i) (h2 : 0 < ∑ i, e i ^ 2)
    (hp : ∀ i, 0 < p i) (hp1 : ∀ i, p i < 1)
    (hlaw : ∀ i, HasLaw (Y i) (nbMeasure (1 / alpha) (p i)) P)
    (hind : iIndepFun Y P)
    (hmean : ∀ i, (1 / alpha) * p i / (1 - p i) = mu * e i) :
    centralMoment (fun ω => ∑ i, (Y i ω : ℝ)) 3 P -
      centralMoment (fun n : ℕ => (n : ℝ)) 3
        (ProbabilityMixture.nbPMF
          ((∑ i, e i) ^ 2 / (alpha * ∑ i, e i ^ 2))
          (mu * ∑ i, e i) (by positivity) (by positivity)).toMeasure =
      2 * alpha ^ 2 * mu ^ 3 *
        ((∑ i, e i ^ 3) - (∑ i, e i ^ 2) ^ 2 / (∑ i, e i)) := by
  rw [independent_nb_third_central_moment_sum Y (1 / alpha) p (by positivity) hp hp1 hlaw hind,
    mixture_PMF_third_central_moment]
  simp_rw [hmean]
  have hsum :
      (∑ i, (mu * e i + 3 * (mu * e i) ^ 2 / (1 / alpha) +
        2 * (mu * e i) ^ 3 / (1 / alpha) ^ 2)) =
      mu * (∑ i, e i) + (3 * alpha * mu ^ 2) * (∑ i, e i ^ 2) +
        (2 * alpha ^ 2 * mu ^ 3) * (∑ i, e i ^ 3) := by
    calc
      _ = ∑ i, (mu * e i + (3 * alpha * mu ^ 2) * e i ^ 2 +
          (2 * alpha ^ 2 * mu ^ 3) * e i ^ 3) := by
        apply Finset.sum_congr rfl
        intro i _
        field_simp
      _ = _ := by simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [hsum]
  have hn1 : ∑ i, e i ≠ 0 := ne_of_gt h1
  have hn2 : ∑ i, e i ^ 2 ≠ 0 := ne_of_gt h2
  field_simp
  ring

end CbmrProofs.ProbabilityCumulants

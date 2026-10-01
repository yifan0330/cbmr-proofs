import lean_proof_checks.ProbabilityCumulants
import Mathlib.MeasureTheory.Function.ConditionalExpectation.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Sampling moments used in S3 and S4

All expectations, variances and covariances here are Mathlib's measure-theoretic
operators. NB moment assumptions are explicit: no distributional conclusion
about a sum follows from matching its first two moments. Conditional first and
second moment assumptions below describe the moments supplied by the
conditionally independent Poisson model.
-/

noncomputable section

open MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace CbmrProofs.Probability

variable {Ω : Type*} {m₀ : MeasurableSpace Ω} {P : Measure Ω} [IsProbabilityMeasure P]

theorem nb_expected_observed_weight (Y : Ω → ℝ) (alpha mu : ℝ)
    (hY : Integrable Y P) (hmean : ∫ ω, Y ω ∂P = mu)
    (_hd : 1 + alpha * mu ≠ 0) :
    (∫ ω, mu * (1 + alpha * Y ω) / (1 + alpha * mu) ^ 2 ∂P) =
      mu / (1 + alpha * mu) := by
  rw [integral_div, integral_const_mul,
    integral_add (integrable_const 1) (hY.const_mul alpha),
    integral_const_mul, integral_const, hmean]
  simp only [probReal_univ, smul_eq_mul, one_mul]
  field_simp [_hd]

theorem nb_expected_score (Y : Ω → ℝ) (alpha mu : ℝ)
    (hY : Integrable Y P) (hmean : ∫ ω, Y ω ∂P = mu) :
    (∫ ω, (Y ω - mu) / (1 + alpha * mu) ∂P) = 0 := by
  rw [integral_div, integral_sub hY (integrable_const mu), hmean]
  simp

/-- Finite weighted information assembly commutes with the actual NB
observed-weight expectation. -/
theorem nb_expected_information_entry {ι : Type*} [Fintype ι]
    (Y : ι → Ω → ℝ) (alpha mu x z : ι → ℝ)
    (hY : ∀ i, Integrable (Y i) P) (hmean : ∀ i, ∫ ω, Y i ω ∂P = mu i)
    (hd : ∀ i, 1 + alpha i * mu i ≠ 0) :
    (∫ ω, (∑ i, x i * (mu i * (1 + alpha i * Y i ω) /
      (1 + alpha i * mu i) ^ 2) * z i) ∂P) =
        ∑ i, x i * (mu i / (1 + alpha i * mu i)) * z i := by
  have hw (i : ι) : Integrable
      (fun ω => mu i * (1 + alpha i * Y i ω) / (1 + alpha i * mu i) ^ 2) P :=
    (((integrable_const 1).add ((hY i).const_mul (alpha i))).const_mul (mu i)).div_const _
  rw [integral_finsetSum _ (fun i _ => ((hw i).const_mul (x i)).mul_const (z i))]
  apply Finset.sum_congr rfl
  intro i _
  rw [integral_mul_const, integral_const_mul,
    nb_expected_observed_weight _ _ _ (hY i) (hmean i) (hd i)]

theorem nb_score_variance (Y : Ω → ℝ) (alpha mu : ℝ)
    (hY : AEStronglyMeasurable Y P)
    (hvar : variance Y P = mu + alpha * mu ^ 2)
    (_hd : 1 + alpha * mu ≠ 0) :
    variance (fun ω => (Y ω - mu) / (1 + alpha * mu)) P =
      mu / (1 + alpha * mu) := by
  simp only [div_eq_mul_inv]
  rw [variance_mul_const, variance_sub_const hY, hvar]
  field_simp [_hd]

theorem independent_nb_sum_moments {ι : Type*} [Fintype ι]
    (Y : ι → Ω → ℝ) (e : ι → ℝ) (mu alpha : ℝ)
    (hY : ∀ i, MemLp (Y i) 2 P)
    (hind : Pairwise (fun i j => IndepFun (Y i) (Y j) P))
    (hmean : ∀ i, ∫ ω, Y i ω ∂P = mu * e i)
    (hvar : ∀ i, variance (Y i) P = mu * e i + alpha * (mu * e i) ^ 2) :
    (∫ ω, (∑ i, Y i ω) ∂P) = mu * ∑ i, e i ∧
    variance (fun ω => ∑ i, Y i ω) P =
      mu * ∑ i, e i + alpha * mu ^ 2 * ∑ i, e i ^ 2 := by
  constructor
  · rw [integral_finsetSum _ (fun i _ => (hY i).integrable (by norm_num))]
    simp only [hmean, Finset.mul_sum]
  · have hv := IndepFun.variance_sum (s := Finset.univ)
      (fun i _ => hY i) (fun i _ j _ hij => hind hij)
    have heq : (fun ω => ∑ i, Y i ω) = ∑ i, Y i := by
      funext ω
      simp
    rw [heq, hv]
    simp only [hvar, Finset.sum_add_distrib, Finset.mul_sum]
    congr 1
    apply Finset.sum_congr rfl
    intro i _
    ring

theorem clustered_expected_information_entry (Y : Ω → ℝ)
    (nu total entry : ℝ) (hY : Integrable Y P)
    (hmean : ∫ ω, Y ω ∂P = total) (hd : total + nu ≠ 0) :
    (∫ ω, (Y ω + nu) / (total + nu) * entry ∂P) = entry := by
  rw [integral_mul_const, integral_div,
    integral_add hY (integrable_const nu), hmean]
  simp [hd]

/-- Integrating the conditional Poisson mean uses the tower property. -/
theorem shared_effect_mean {m : MeasurableSpace Ω} (hm : m ≤ m₀)
    (Y L : Ω → ℝ) (mu : ℝ)
    (hcond : P[Y | m] =ᵐ[P] fun ω => L ω * mu)
    (hLmean : ∫ ω, L ω ∂P = 1) :
    (∫ ω, Y ω ∂P) = mu := by
  rw [← integral_condExp hm (f := Y), integral_congr_ae hcond, integral_mul_const, hLmean,
    one_mul]

/-- Conditional Poisson product moments plus the shared effect's first two
moments imply the diagonal and off-diagonal clustered covariance. `d` is
the Kronecker delta (one on the diagonal and zero off it). -/
theorem shared_effect_covariance {m : MeasurableSpace Ω}
    (hm : m ≤ m₀) (Y Z L : Ω → ℝ)
    (mu eta alpha d : ℝ) (hY : MemLp Y 2 P) (hZ : MemLp Z 2 P)
    (hL : MemLp L 2 P)
    (hYcond : P[Y | m] =ᵐ[P] fun ω => L ω * mu)
    (hZcond : P[Z | m] =ᵐ[P] fun ω => L ω * eta)
    (hprod : P[(fun ω => Y ω * Z ω) | m] =ᵐ[P]
      fun ω => d * (L ω * mu) + L ω ^ 2 * (mu * eta))
    (hLmean : ∫ ω, L ω ∂P = 1)
    (hLvar : variance L P = alpha) :
    covariance Y Z P = d * mu + alpha * mu * eta := by
  have hsecond : (∫ ω, L ω ^ 2 ∂P) = alpha + 1 := by
    have hv := variance_eq_sub hL
    rw [hLmean, hLvar] at hv
    change alpha = (∫ ω, L ω ^ 2 ∂P) - 1 ^ 2 at hv
    linarith
  have hproduct : (∫ ω, Y ω * Z ω ∂P) =
      d * mu + (alpha + 1) * (mu * eta) := by
    rw [← integral_condExp hm (f := fun ω => Y ω * Z ω), integral_congr_ae hprod,
      integral_add (((hL.integrable (by norm_num)).mul_const mu).const_mul d)
        (hL.integrable_sq.mul_const (mu * eta)),
      integral_const_mul, integral_mul_const, integral_mul_const, hLmean, hsecond]
    ring
  rw [covariance_eq_sub hY hZ]
  change (∫ ω, Y ω * Z ω ∂P) - (∫ ω, Y ω ∂P) * (∫ ω, Z ω ∂P) = _
  rw [
    shared_effect_mean hm Y L mu hYcond hLmean,
    shared_effect_mean hm Z L eta hZcond hLmean, hproduct]
  ring

omit [IsProbabilityMeasure P] in
theorem independent_studies_covariance_zero (Y Z : Ω → ℝ)
    (hY : MemLp Y 2 P) (hZ : MemLp Z 2 P) (hind : IndepFun Y Z P) :
    covariance Y Z P = 0 :=
  hind.covariance_eq_zero hY hZ

/-- Shared Gamma law plus the conditional Poisson first/product moments yields
the full clustered covariance, without assuming the marginal covariance. -/
theorem gamma_poisson_covariance {m : MeasurableSpace Ω} (hm : m ≤ m₀)
    (Y Z L : Ω → ℝ) (mu eta nu d : ℝ) (hnu : 0 < nu)
    (hY : MemLp Y 2 P) (hZ : MemLp Z 2 P)
    (hlaw : HasLaw L (gammaMeasure nu nu) P)
    (hYcond : P[Y | m] =ᵐ[P] fun ω => L ω * mu)
    (hZcond : P[Z | m] =ᵐ[P] fun ω => L ω * eta)
    (hprod : P[(fun ω => Y ω * Z ω) | m] =ᵐ[P]
      fun ω => d * (L ω * mu) + L ω ^ 2 * (mu * eta)) :
    covariance Y Z P = d * mu + (1 / nu) * mu * eta := by
  let : MeasurableSpace Ω := m₀
  have hmap : MemLp (id : ℝ → ℝ) 2 (P.map L) := by
    rw [hlaw.map_eq]
    exact ProbabilityMixture.gamma_memLp_two nu hnu
  have hL : MemLp L 2 P := by
    simpa using hmap.comp_of_map hlaw.aemeasurable
  have hmom := ProbabilityMixture.gamma_random_effect_moments P L nu hnu hlaw
  exact shared_effect_covariance hm Y Z L mu eta (1 / nu) d hY hZ hL
    hYcond hZcond hprod hmom.1 hmom.2

/-- Fisher-weight and score-variance identities derived directly from an NB
sampling law. No NB mean/variance assumptions are needed in this corollary. -/
theorem nb_fisher_from_sampling_law (Y : Ω → ℕ) (shape p : ℝ)
    (hs : 0 < shape) (hp : 0 < p) (hp1 : p < 1)
    (hlaw : HasLaw Y (ProbabilityNB.nbMeasure shape p) P) :
    let mu := shape * p / (1 - p)
    let alpha := 1 / shape
    (∫ ω, mu * (1 + alpha * (Y ω : ℝ)) / (1 + alpha * mu) ^ 2 ∂P) =
        mu / (1 + alpha * mu) ∧
    variance (fun ω => ((Y ω : ℝ) - mu) / (1 + alpha * mu)) P =
        mu / (1 + alpha * mu) := by
  dsimp only
  have hq : 0 < 1 - p := sub_pos.mpr hp1
  have hmap : MemLp (fun n : ℕ => (n : ℝ)) 2 (P.map Y) := by
    rw [hlaw.map_eq]
    exact ProbabilityNB.nbMeasure_memLp_two shape p hs hp hp1
  have hY : MemLp (fun ω => (Y ω : ℝ)) 2 P := hmap.comp_of_map hlaw.aemeasurable
  have hmom := ProbabilityNB.nb_count_moments P Y shape p hs hp hp1 hlaw
  have hd : 1 + (1 / shape) * (shape * p / (1 - p)) ≠ 0 := by positivity
  exact ⟨nb_expected_observed_weight _ _ _ (hY.integrable (by norm_num)) hmom.1 hd,
    nb_score_variance _ _ _ hY.aestronglyMeasurable hmom.2 hd⟩

end CbmrProofs.Probability

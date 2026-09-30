import Mathlib.Probability.Distributions.Gamma
import Mathlib.Probability.Moments.Variance
import Mathlib.Probability.HasLaw
import Mathlib.Probability.Distributions.Poisson.Basic
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Exact finite-count Gamma--Poisson marginalization (S4)

The joint mass is an actual integral of the Gamma density times the product
of Poisson masses. Counts are natural numbers on an arbitrary finite index
type. Its evaluation uses the Gamma integral, not an assumed marginal law.
-/

noncomputable section

open MeasureTheory ProbabilityTheory Real Set
open scoped BigOperators

namespace CbmrProofs.ProbabilityMixture

variable {ι : Type*} [Fintype ι]

def poissonMass (rate : ℝ) (n : ℕ) : ℝ :=
  exp (-rate) * rate ^ n / (n.factorial : ℝ)

def jointMass (nu : ℝ) (mu : ι → ℝ) (y : ι → ℕ) : ℝ :=
  ∫ l in Ioi (0 : ℝ), gammaPDFReal nu nu l * ∏ j, poissonMass (l * mu j) (y j)

/-- Product of conditionally independent Poisson masses at a finite count vector. -/
theorem poisson_product (mu : ι → ℝ) (y : ι → ℕ) (l : ℝ) :
    (∏ j, poissonMass (l * mu j) (y j)) =
      exp (-(l * ∑ j, mu j)) * l ^ (∑ j, y j) *
        (∏ j, mu j ^ y j) / (∏ j, (y j).factorial : ℝ) := by
  simp only [poissonMass, Finset.prod_div_distrib, Finset.prod_mul_distrib,
    mul_pow, ← exp_sum, Finset.sum_neg_distrib, ← Finset.mul_sum,
    Finset.prod_pow_eq_pow_sum]
  ring

theorem gamma_poisson_integrand (nu : ℝ) (mu : ι → ℝ) (y : ι → ℕ)
    (l : ℝ) (hl : 0 < l) :
    gammaPDFReal nu nu l * (∏ j, poissonMass (l * mu j) (y j)) =
      (nu ^ nu / Gamma nu * ((∏ j, mu j ^ y j) /
        (∏ j, (y j).factorial : ℝ))) *
        (l ^ (nu + (∑ j, y j : ℕ) - 1) * exp (-((nu + ∑ j, mu j) * l))) := by
  rw [poisson_product, gammaPDFReal, if_pos hl.le]
  have hp : l ^ (nu - 1) * l ^ (∑ j, y j) =
      l ^ (nu + (∑ j, y j : ℕ) - 1) := by
    rw [← rpow_natCast, ← rpow_add hl]
    congr 1
    ring
  have he : exp (-(nu * l)) * exp (-(l * ∑ j, mu j)) =
      exp (-((nu + ∑ j, mu j) * l)) := by
    rw [← exp_add]
    congr 1
    ring
  calc
    _ = (nu ^ nu / Gamma nu * ((∏ j, mu j ^ y j) /
          (∏ j, (y j).factorial : ℝ))) *
        ((l ^ (nu - 1) * l ^ (∑ j, y j)) *
          (exp (-(nu * l)) * exp (-(l * ∑ j, mu j)))) := by ring
    _ = _ := by rw [hp, he]

/-- Exact integrated joint mass; valid even for zero individual cell means. -/
theorem joint_mass_integral (nu : ℝ) (mu : ι → ℝ) (y : ι → ℕ)
    (hnu : 0 < nu) (hmu : ∀ j, 0 ≤ mu j) :
    jointMass nu mu y =
      (nu ^ nu / Gamma nu * ((∏ j, mu j ^ y j) /
        (∏ j, (y j).factorial : ℝ))) *
        ((1 / (nu + ∑ j, mu j)) ^ (nu + (∑ j, y j : ℕ)) *
          Gamma (nu + (∑ j, y j : ℕ))) := by
  unfold jointMass
  rw [setIntegral_congr_fun measurableSet_Ioi
    (fun l hl => gamma_poisson_integrand nu mu y l hl), integral_const_mul]
  rw [integral_rpow_mul_exp_neg_mul_Ioi (by positivity)
    (add_pos_of_pos_of_nonneg hnu (Finset.sum_nonneg fun j _ => hmu j))]

/-- Logarithm of the evaluated joint Gamma--Poisson mass, including the
count-factorial term which is constant in the regression coefficients. -/
theorem log_joint_mass (nu : ℝ) (mu : ι → ℝ) (y : ι → ℕ)
    (hnu : 0 < nu) (hmu : ∀ j, 0 < mu j) :
    log (jointMass nu mu y) =
      nu * log nu - log (Gamma nu) + log (Gamma (nu + (∑ j, y j : ℕ))) -
        (nu + (∑ j, y j : ℕ)) * log (nu + ∑ j, mu j) +
        (∑ j, (y j : ℝ) * log (mu j)) - ∑ j, log ((y j).factorial : ℝ) := by
  have hsum : 0 ≤ ∑ j, mu j := Finset.sum_nonneg (fun j _ => (hmu j).le)
  have hd : 0 < nu + ∑ j, mu j := add_pos_of_pos_of_nonneg hnu hsum
  have hg : 0 < Gamma nu := Gamma_pos_of_pos hnu
  have hgn : 0 < Gamma (nu + (∑ j, y j : ℕ)) := Gamma_pos_of_pos (by positivity)
  have hprod : 0 < ∏ j, mu j ^ y j := Finset.prod_pos (fun j _ => pow_pos (hmu j) _)
  have hfact : 0 < (∏ j, (y j).factorial : ℝ) := by positivity
  rw [joint_mass_integral nu mu y hnu (fun j => (hmu j).le),
    log_mul (by positivity) (by positivity),
    log_mul (by positivity) (by positivity),
    log_mul (by positivity) (ne_of_gt hgn),
    log_div (by positivity) (ne_of_gt hg),
    log_div (ne_of_gt hprod) (ne_of_gt hfact),
    log_rpow hnu, log_rpow (by positivity),
    log_div one_ne_zero (ne_of_gt hd), log_one,
    log_prod (fun j _ => ne_of_gt (pow_pos (hmu j) _)),
    log_prod (fun j _ => by positivity)]
  simp only [log_pow]
  ring

/-- Studywise joint masses give the grouped clustered log-likelihood, up to
the displayed data-only factorial sum. -/
theorem grouped_log_joint_mass {I : Type*} [Fintype I]
    (nu : ℝ) (etaB : ι → ℝ) (etaZ : I → ℝ) (y : I → ι → ℕ)
    (hnu : 0 < nu) :
    (∑ i, log (jointMass nu (fun j => exp (etaB j + etaZ i)) (y i))) =
      (Fintype.card I : ℝ) * (nu * log nu - log (Gamma nu)) +
      (∑ i, log (Gamma (nu + (∑ j, y i j : ℕ)))) -
      (∑ i, (nu + (∑ j, y i j : ℕ)) *
        log (nu + (∑ j, exp (etaB j)) * exp (etaZ i))) +
      (∑ j, (∑ i, (y i j : ℝ)) * etaB j) +
      (∑ i, (∑ j, (y i j : ℝ)) * etaZ i) -
      ∑ i, ∑ j, log ((y i j).factorial : ℝ) := by
  simp_rw [log_joint_mass nu _ _ hnu (fun _ => exp_pos _), log_exp]
  have htotal (i : I) :
      (∑ j, exp (etaB j + etaZ i)) = (∑ j, exp (etaB j)) * exp (etaZ i) := by
    simp only [exp_add, Finset.sum_mul]
  simp_rw [htotal, mul_add, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib]
  have hspatial : (∑ i, ∑ j, (y i j : ℝ) * etaB j) =
      ∑ j, (∑ i, (y i j : ℝ)) * etaB j := by
    simp only [Finset.sum_mul]
    exact Finset.sum_comm
  rw [hspatial]
  simp only [← Finset.sum_mul, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  ring

def nbTotalMass (nu total : ℝ) (n : ℕ) : ℝ :=
  Gamma (nu + n) / (Gamma nu * (n.factorial : ℝ)) *
    (nu / (nu + total)) ^ nu * (total / (nu + total)) ^ n

def multinomialMass (mu : ι → ℝ) (y : ι → ℕ) : ℝ :=
  ((∑ j, y j : ℕ).factorial : ℝ) / (∏ j, (y j).factorial : ℝ) *
    ∏ j, (mu j / ∑ k, mu k) ^ y j

/-- Normalized allocation probabilities for the multinomial factor. -/
theorem allocation_probabilities (mu : ι → ℝ) (hmu : ∀ j, 0 ≤ mu j)
    (ht : 0 < ∑ j, mu j) :
    (∀ j, 0 ≤ mu j / ∑ k, mu k) ∧ (∑ j, mu j / ∑ k, mu k) = 1 := by
  constructor
  · intro j
    exact div_nonneg (hmu j) ht.le
  · rw [← Finset.sum_div, div_self (ne_of_gt ht)]

/-- The allocation mass is normalized on the finite set of count vectors with
specified total, by the finite multinomial theorem. -/
theorem multinomial_mass_sum_one [DecidableEq ι] (mu : ι → ℝ) (n : ℕ)
    (hmu : ∀ j, 0 ≤ mu j) (ht : 0 < ∑ j, mu j) :
    (∑ y ∈ Finset.piAntidiag Finset.univ n, multinomialMass mu y) = 1 := by
  have h := Finset.sum_pow_eq_sum_piAntidiag Finset.univ
    (fun j => mu j / ∑ k, mu k) n
  rw [(allocation_probabilities mu hmu ht).2, one_pow] at h
  rw [h]
  apply Finset.sum_congr rfl
  intro y _
  unfold multinomialMass Nat.multinomial
  rw [Nat.cast_div (Nat.prod_factorial_dvd_factorial_sum Finset.univ y),
    Nat.cast_prod]
  positivity

/-- The exact Gamma-integrated joint mass is NB(total) times multinomial(allocation). -/
theorem joint_mass_nb_multinomial (nu : ℝ) (mu : ι → ℝ) (y : ι → ℕ)
    (hnu : 0 < nu) (hmu : ∀ j, 0 ≤ mu j) (ht : 0 < ∑ j, mu j) :
    jointMass nu mu y =
      nbTotalMass nu (∑ j, mu j) (∑ j, y j) * multinomialMass mu y := by
  rw [joint_mass_integral nu mu y hnu hmu]
  unfold nbTotalMass multinomialMass
  simp only [div_pow, Finset.prod_div_distrib, Finset.prod_pow_eq_pow_sum]
  rw [div_rpow hnu.le (by positivity), rpow_add (by positivity),
    one_div, inv_rpow (by positivity), rpow_natCast]
  have hn : ((∑ j, y j : ℕ).factorial : ℝ) ≠ 0 := by positivity
  have hd : (nu + ∑ j, mu j) ^ nu ≠ 0 := ne_of_gt (rpow_pos_of_pos (by positivity) _)
  have hdn : nu + ∑ j, mu j ≠ 0 := by positivity
  have ht' : ∑ j, mu j ≠ 0 := ne_of_gt ht
  rw [inv_pow]
  field_simp

/-- Summing the actual joint mass over all allocations of a fixed total gives
the NB total-count mass. This also checks that the allocation factor has the
correct combinatorial support, not merely the correct algebraic powers. -/
theorem joint_mass_total [DecidableEq ι] (nu : ℝ) (mu : ι → ℝ) (n : ℕ)
    (hnu : 0 < nu) (hmu : ∀ j, 0 ≤ mu j) (ht : 0 < ∑ j, mu j) :
    (∑ y ∈ Finset.piAntidiag Finset.univ n, jointMass nu mu y) =
      nbTotalMass nu (∑ j, mu j) n := by
  calc
    _ = ∑ y ∈ Finset.piAntidiag Finset.univ n,
        nbTotalMass nu (∑ j, mu j) n * multinomialMass mu y := by
      apply Finset.sum_congr rfl
      intro y hy
      rw [joint_mass_nb_multinomial nu mu y hnu hmu ht, (Finset.mem_piAntidiag.mp hy).1]
    _ = _ := by
      rw [← Finset.mul_sum, multinomial_mass_sum_one mu n hmu ht, mul_one]

/-- The mixing law itself is normalized, by Mathlib's Gamma density integral. -/
theorem gamma_mixing_probability (nu : ℝ) (hnu : 0 < nu) :
    IsProbabilityMeasure (gammaMeasure nu nu) :=
  isProbabilityMeasure_gammaMeasure hnu hnu

theorem gamma_integral_density (nu : ℝ) (hnu : 0 < nu) (f : ℝ → ℝ) :
    (∫ l, f l ∂gammaMeasure nu nu) =
      ∫ l in Ioi (0 : ℝ), gammaPDFReal nu nu l * f l := by
  unfold gammaMeasure gammaPDF
  rw [integral_withDensity_eq_integral_toReal_smul
    ((measurable_gammaPDFReal nu nu).ennreal_ofReal)
    (Filter.Eventually.of_forall fun _ => ENNReal.ofReal_lt_top)]
  simp only [ENNReal.toReal_ofReal (gammaPDFReal_nonneg hnu hnu _),
    smul_eq_mul]
  rw [← setIntegral_eq_integral_of_forall_compl_eq_zero
    (s := Ici (0 : ℝ)) (fun l hl => by
      simp only [mem_Ici, not_le] at hl
      simp [gammaPDFReal, not_le.mpr hl]), integral_Ici_eq_integral_Ioi]

/-- All nonnegative integer moments are evaluated against the normalized Gamma measure. -/
theorem gamma_nat_moment (nu : ℝ) (hnu : 0 < nu) (n : ℕ) :
    (∫ l, l ^ n ∂gammaMeasure nu nu) =
      nu ^ nu / Gamma nu * ((1 / nu) ^ (nu + n) * Gamma (nu + n)) := by
  rw [gamma_integral_density nu hnu]
  have heq : ∀ l ∈ Ioi (0 : ℝ),
      gammaPDFReal nu nu l * l ^ n =
        (nu ^ nu / Gamma nu) * (l ^ (nu + n - 1) * exp (-(nu * l))) := by
    intro l hl
    rw [gammaPDFReal, if_pos (le_of_lt hl)]
    have hp : l ^ (nu - 1) * l ^ n = l ^ (nu + n - 1) := by
      rw [← rpow_natCast, ← rpow_add hl]
      congr 1
      ring
    calc
      _ = nu ^ nu / Gamma nu * ((l ^ (nu - 1) * l ^ n) * exp (-(nu * l))) := by ring
      _ = _ := by rw [hp]
  rw [setIntegral_congr_fun measurableSet_Ioi heq, integral_const_mul,
    integral_rpow_mul_exp_neg_mul_Ioi (by positivity) hnu]

theorem gamma_mean_one (nu : ℝ) (hnu : 0 < nu) :
    (∫ l, l ∂gammaMeasure nu nu) = 1 := by
  have h := gamma_nat_moment nu hnu 1
  simp only [pow_one, Nat.cast_one] at h
  rw [h, Gamma_add_one (ne_of_gt hnu), rpow_add (by positivity), rpow_one,
    one_div, inv_rpow hnu.le]
  field_simp

theorem gamma_second_moment (nu : ℝ) (hnu : 0 < nu) :
    (∫ l, l ^ 2 ∂gammaMeasure nu nu) = 1 + 1 / nu := by
  rw [gamma_nat_moment nu hnu]
  norm_num only [Nat.cast_ofNat]
  rw [show nu + 2 = (nu + 1) + 1 by ring,
    Gamma_add_one (by positivity), Gamma_add_one (ne_of_gt hnu)]
  rw [show nu + 1 + 1 = nu + (2 : ℕ) by norm_num; ring,
    rpow_add (by positivity), rpow_natCast, one_div, inv_rpow hnu.le, inv_pow]
  field_simp

theorem gamma_memLp_two (nu : ℝ) (hnu : 0 < nu) :
    MemLp (id : ℝ → ℝ) 2 (gammaMeasure nu nu) := by
  apply (memLp_two_iff_integrable_sq aestronglyMeasurable_id).mpr
  apply Integrable.of_integral_ne_zero
  change (∫ l, l ^ 2 ∂gammaMeasure nu nu) ≠ 0
  rw [gamma_second_moment nu hnu]
  positivity

theorem gamma_variance (nu : ℝ) (hnu : 0 < nu) :
    variance (id : ℝ → ℝ) (gammaMeasure nu nu) = 1 / nu := by
  let := gamma_mixing_probability nu hnu
  rw [variance_eq_sub (gamma_memLp_two nu hnu)]
  change (∫ l, l ^ 2 ∂gammaMeasure nu nu) -
    (∫ l, l ∂gammaMeasure nu nu) ^ 2 = _
  rw [gamma_second_moment nu hnu, gamma_mean_one nu hnu]
  ring

theorem gamma_random_effect_moments {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (L : Ω → ℝ) (nu : ℝ) (hnu : 0 < nu)
    (hlaw : HasLaw L (gammaMeasure nu nu) P) :
    (∫ ω, L ω ∂P) = 1 ∧ variance L P = 1 / nu := by
  exact ⟨hlaw.integral_eq.trans (gamma_mean_one nu hnu),
    hlaw.variance_eq.trans (gamma_variance nu hnu)⟩

/-- The scalar NB mass is itself obtained by integrating the Poisson law.
The maximum only extends the kernel harmlessly outside the Gamma support. -/
theorem nb_mass_poisson_mixture (nu total : ℝ) (hnu : 0 < nu) (ht : 0 < total) (n : ℕ) :
    (∫ l, poissonMass (max l 0 * total) n ∂gammaMeasure nu nu) =
      nbTotalMass nu total n := by
  have h := joint_mass_nb_multinomial (ι := Fin 1) nu (fun _ => total)
    (fun _ => n) hnu (fun _ => ht.le) (by simpa using ht)
  simp only [jointMass, Fin.prod_univ_one, Fin.sum_univ_one, multinomialMass,
    div_self (ne_of_gt ht), one_pow, mul_one] at h
  have hn : (n.factorial : ℝ) ≠ 0 := by positivity
  simp only [div_self hn, mul_one] at h
  rw [gamma_integral_density nu hnu]
  calc
    _ = ∫ l in Ioi (0 : ℝ), gammaPDFReal nu nu l * poissonMass (l * total) n := by
      apply setIntegral_congr_fun measurableSet_Ioi
      intro l hl
      have hl' : 0 ≤ l := (Set.mem_Ioi.mp hl).le
      simp only [max_eq_left hl']
    _ = _ := h

theorem nb_mass_pos (nu total : ℝ) (hnu : 0 < nu) (ht : 0 < total) (n : ℕ) :
    0 < nbTotalMass nu total n := by
  unfold nbTotalMass
  positivity

theorem conditional_allocation_mass (nu : ℝ) (mu : ι → ℝ) (y : ι → ℕ)
    (hnu : 0 < nu) (hmu : ∀ j, 0 ≤ mu j) (ht : 0 < ∑ j, mu j) :
    jointMass nu mu y / nbTotalMass nu (∑ j, mu j) (∑ j, y j) =
      multinomialMass mu y := by
  rw [joint_mass_nb_multinomial nu mu y hnu hmu ht]
  exact mul_div_cancel_left₀ _ (ne_of_gt (nb_mass_pos nu _ hnu ht _))

/-- Tonelli's theorem transfers Poisson normalization through the Gamma mixture. -/
theorem nb_mass_sum_one (nu total : ℝ) (hnu : 0 < nu) (ht : 0 < total) :
    (∑' n, ENNReal.ofReal (nbTotalMass nu total n)) = 1 := by
  let := gamma_mixing_probability nu hnu
  have hint (n : ℕ) : Integrable (fun l => poissonMass (max l 0 * total) n)
      (gammaMeasure nu nu) := by
    apply Integrable.of_integral_ne_zero
    rw [nb_mass_poisson_mixture nu total hnu ht]
    exact ne_of_gt (nb_mass_pos nu total hnu ht n)
  have hnonneg (n : ℕ) (l : ℝ) : 0 ≤ poissonMass (max l 0 * total) n := by
    unfold poissonMass
    positivity
  have heq (n : ℕ) : ENNReal.ofReal (nbTotalMass nu total n) =
      ∫⁻ l, ENNReal.ofReal (poissonMass (max l 0 * total) n) ∂gammaMeasure nu nu := by
    rw [← nb_mass_poisson_mixture nu total hnu ht]
    exact ofReal_integral_eq_lintegral_ofReal (hint n) (Filter.Eventually.of_forall (hnonneg n))
  simp_rw [heq]
  rw [← lintegral_tsum (fun n => by
    apply Measurable.aemeasurable
    unfold poissonMass
    fun_prop)]
  have hsum (l : ℝ) : (∑' n, ENNReal.ofReal (poissonMass (max l 0 * total) n)) = 1 := by
    have hs : HasSum (fun n => poissonMass (max l 0 * total) n) 1 :=
      hasSum_one_poissonMeasure ⟨max l 0 * total, by positivity⟩
    rw [← ENNReal.ofReal_tsum_of_nonneg (fun n => hnonneg n l) hs.summable,
      hs.tsum_eq, ENNReal.ofReal_one]
  simp_rw [hsum]
  simp

/-- A normalized NB distribution for every positive real shape and mean. -/
def nbPMF (nu total : ℝ) (hnu : 0 < nu) (ht : 0 < total) : PMF ℕ :=
  ⟨fun n => ENNReal.ofReal (nbTotalMass nu total n),
    ENNReal.summable.hasSum_iff.mpr (nb_mass_sum_one nu total hnu ht)⟩

theorem joint_mass_normalized_by_total [DecidableEq ι]
    (nu : ℝ) (mu : ι → ℝ) (hnu : 0 < nu) (hmu : ∀ j, 0 ≤ mu j)
    (ht : 0 < ∑ j, mu j) :
    (∑' n, ENNReal.ofReal
      (∑ y ∈ Finset.piAntidiag Finset.univ n, jointMass nu mu y)) = 1 := by
  simp_rw [joint_mass_total nu mu _ hnu hmu ht]
  exact nb_mass_sum_one nu _ hnu ht

end CbmrProofs.ProbabilityMixture

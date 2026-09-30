import lean_proof_checks.AggregatedGamma
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Pow

/-! First and second partials of the full cellwise matched NB likelihood.
Every score expression below is identified as an actual derivative. -/
namespace CbmrProofs.AggregatedNB

open scoped BigOperators
open NegativeBinomial
noncomputable section
set_option backward.isDefEq.respectTransparency false

variable {ι : Type*} [Fintype ι]

theorem const_div_hasDerivAt {f : ℝ → ℝ} {f' x : ℝ} (hf : HasDerivAt f f' x)
    (c : ℝ) (h : f x ≠ 0) :
    HasDerivAt (fun x => c / f x) (-c * f' / f x ^ 2) x := by
  convert! (hasDerivAt_const x c).div hf h using 1
  ring

def rhoScore (y : ι → ℝ) (r A : ℝ) (S : ι → ℝ) : ℝ :=
  -(∑ i, digamma (y i + r)) + (Fintype.card ι : ℝ) * digamma r -
    (Fintype.card ι : ℝ) * Real.log A + ∑ i, Real.log (Real.exp (S i) + A)

def aScore (y : ι → ℝ) (r A : ℝ) (S : ι → ℝ) : ℝ :=
  -(Fintype.card ι : ℝ) * r / A + ∑ i, (r + y i) / (Real.exp (S i) + A)

def etaScore (y r A eta : ℝ) : ℝ :=
  (r + y) * Real.exp eta / (Real.exp eta + A) - y

def rhoRho (y : ι → ℝ) (r : ℝ) : ℝ :=
  -(∑ i, trigamma (y i + r)) + (Fintype.card ι : ℝ) * trigamma r

def rhoA (A : ℝ) (S : ι → ℝ) : ℝ :=
  -(Fintype.card ι : ℝ) / A + ∑ i, 1 / (Real.exp (S i) + A)

def aA (y : ι → ℝ) (r A : ℝ) (S : ι → ℝ) : ℝ :=
  (Fintype.card ι : ℝ) * r / A ^ 2 - ∑ i, (r + y i) / (Real.exp (S i) + A) ^ 2

theorem psi_rho_hasDerivAt (y : ι → ℝ) (r A : ℝ) (S : ι → ℝ)
    (hr : 0 < r) (hy : ∀ i, 0 ≤ y i) :
    HasDerivAt (fun r => aggregatedPsi y r A S) (rhoScore y r A S) r :=
  hasDerivAt_aggregatedPsi_shape y r A S hr hy

theorem psi_A_hasDerivAt (y : ι → ℝ) (r A : ℝ) (S : ι → ℝ) (hA : 0 < A) :
    HasDerivAt (fun A => aggregatedPsi y r A S) (aScore y r A S) A := by
  have hlog := (Real.hasDerivAt_log (ne_of_gt hA)).const_mul
    ((Fintype.card ι : ℝ) * r)
  have hsum := HasDerivAt.fun_sum (u := Finset.univ) (fun i _ =>
    (((hasDerivAt_id A).const_add (Real.exp (S i))).log
      (ne_of_gt (by positivity))).const_mul (r + y i))
  convert! ((hlog.const_sub
    (-(∑ i, Real.log (Real.Gamma (y i + r))) +
      (Fintype.card ι : ℝ) * Real.log (Real.Gamma r))).add hsum).sub_const
      (∑ i, y i * S i) using 1
  simp only [aScore, div_eq_mul_inv, id_eq, one_mul]
  ring

theorem rhoScore_rho_hasDerivAt (y : ι → ℝ) (r A : ℝ) (S : ι → ℝ)
    (hr : 0 < r) (hy : ∀ i, 0 ≤ y i) :
    HasDerivAt (fun r => rhoScore y r A S) (rhoRho y r) r := by
  have hg : ∀ i, HasDerivAt (fun r : ℝ => digamma (y i + r))
      (trigamma (y i + r)) r := by
    intro i
    convert! (digamma_hasDerivAt (add_pos_of_nonneg_of_pos (hy i) hr)).comp
      r ((hasDerivAt_id r).const_add (y i)) using 1
    simp
  have hs := HasDerivAt.fun_sum (u := Finset.univ) (fun i _ => hg i)
  exact ((hs.neg.add ((digamma_hasDerivAt hr).const_mul (Fintype.card ι : ℝ))).sub_const
    ((Fintype.card ι : ℝ) * Real.log A)).add_const
      (∑ i, Real.log (Real.exp (S i) + A))

theorem rhoScore_A_hasDerivAt (y : ι → ℝ) (r A : ℝ) (S : ι → ℝ) (hA : 0 < A) :
    HasDerivAt (fun A => rhoScore y r A S) (rhoA A S) A := by
  have hs := HasDerivAt.fun_sum (u := Finset.univ) (fun i _ =>
    ((hasDerivAt_id A).const_add (Real.exp (S i))).log (ne_of_gt (by positivity)))
  convert! (((Real.hasDerivAt_log (ne_of_gt hA)).const_mul
    (Fintype.card ι : ℝ)).const_sub
      (-(∑ i, digamma (y i + r)) + (Fintype.card ι : ℝ) * digamma r)).add hs using 1
  simp [rhoA, div_eq_mul_inv]

theorem aScore_rho_hasDerivAt (y : ι → ℝ) (r A : ℝ) (S : ι → ℝ) :
    HasDerivAt (fun r => aScore y r A S) (rhoA A S) r := by
  have hs := HasDerivAt.fun_sum (u := Finset.univ) (fun i _ =>
    ((hasDerivAt_id r).add_const (y i)).div_const (Real.exp (S i) + A))
  convert! (((hasDerivAt_id r).const_mul (-(Fintype.card ι : ℝ))).div_const A).add hs using 1
  simp [rhoA]

theorem aScore_A_hasDerivAt (y : ι → ℝ) (r A : ℝ) (S : ι → ℝ) (hA : 0 < A) :
    HasDerivAt (fun A => aScore y r A S) (aA y r A S) A := by
  have hs := HasDerivAt.fun_sum (u := Finset.univ) (fun i _ =>
    const_div_hasDerivAt ((hasDerivAt_id A).const_add (Real.exp (S i))) (r + y i)
      (ne_of_gt (by positivity)))
  convert! ((const_div_hasDerivAt (hasDerivAt_id A) (-(Fintype.card ι : ℝ) * r)
    (ne_of_gt hA)).add hs) using 1
  simp only [aA, mul_one, neg_mul, neg_neg, neg_div, Finset.sum_neg_distrib, id_eq]
  ring

theorem etaScore_eta_hasDerivAt (y r A eta : ℝ) (hA : 0 < A) :
    HasDerivAt (etaScore y r A)
      ((r + y) * Real.exp eta * A / (Real.exp eta + A) ^ 2) eta := by
  have hd : Real.exp eta + A ≠ 0 := ne_of_gt (by positivity)
  convert! ((((Real.hasDerivAt_exp eta).const_mul (r + y)).div
    ((Real.hasDerivAt_exp eta).add_const A) hd).sub_const y) using 1
  ring

theorem etaScore_rho_hasDerivAt (y r A eta : ℝ) :
    HasDerivAt (fun r => etaScore y r A eta)
      (Real.exp eta / (Real.exp eta + A)) r := by
  convert! ((((hasDerivAt_id r).add_const y).mul_const (Real.exp eta)).div_const
    (Real.exp eta + A)).sub_const y using 1
  simp

theorem etaScore_A_hasDerivAt (y r A eta : ℝ) (hA : 0 < A) :
    HasDerivAt (fun A => etaScore y r A eta)
      (-(r + y) * Real.exp eta / (Real.exp eta + A) ^ 2) A := by
  convert! (const_div_hasDerivAt ((hasDerivAt_id A).const_add (Real.exp eta))
    ((r + y) * Real.exp eta) (ne_of_gt (by positivity))).sub_const y using 1
  simp only [id_eq]
  ring

/-- The coordinate partial is extracted from the already verified arbitrary
directional derivative, not from a common-intensity restriction. -/
theorem psi_eta_coordinate_hasDerivAt [DecidableEq ι] (y : ι → ℝ)
    (r A : ℝ) (S : ι → ℝ) (j : ι) (hA : 0 < A) :
    HasDerivAt (fun t => aggregatedPsi y r A
      (fun i => S i + t * if i = j then 1 else 0)) (etaScore (y j) r A (S j)) 0 := by
  simpa [etaScore] using hasDerivAt_aggregatedPsi_log_intensity
    y r A S (fun i => if i = j then 1 else 0) hA

/-- A second directional derivative of the full vector likelihood score. -/
theorem psi_eta_direction_second (y : ι → ℝ) (r A : ℝ)
    (S v w : ι → ℝ) (hA : 0 < A) :
    HasDerivAt (fun t => ∑ i, etaScore (y i) r A (S i + t * w i) * v i)
      (∑ i, ((r + y i) * Real.exp (S i) * A / (Real.exp (S i) + A) ^ 2) *
        w i * v i) 0 := by
  have hh : ∀ i, HasDerivAt (fun t : ℝ => etaScore (y i) r A (S i + t * w i) * v i)
      (((r + y i) * Real.exp (S i) * A / (Real.exp (S i) + A) ^ 2) * w i * v i) 0 := by
    intro i
    convert! ((etaScore_eta_hasDerivAt (y i) r A (S i + 0 * w i) hA).comp 0
      (((hasDerivAt_id (0 : ℝ)).mul_const (w i)).const_add (S i))).mul_const (v i) using 1
    simp
  exact HasDerivAt.fun_sum (u := Finset.univ) (fun i _ => hh i)

theorem rhoScore_eta_direction_hasDerivAt (y : ι → ℝ) (r A : ℝ)
    (S v : ι → ℝ) (hA : 0 < A) :
    HasDerivAt (fun t => rhoScore y r A (fun i => S i + t * v i))
      (∑ i, Real.exp (S i) / (Real.exp (S i) + A) * v i) 0 := by
  have hl : ∀ i, HasDerivAt
      (fun t : ℝ => Real.log (Real.exp (S i + t * v i) + A))
      (Real.exp (S i) / (Real.exp (S i) + A) * v i) 0 := by
    intro i
    convert! (((((hasDerivAt_id (0 : ℝ)).mul_const (v i)).const_add (S i)).exp).add_const A).log
      (ne_of_gt (by positivity)) using 1
    simp only [id_eq, zero_mul, add_zero, one_mul]
    ring
  exact (HasDerivAt.fun_sum (u := Finset.univ) (fun i _ => hl i)).const_add
    (-(∑ i, digamma (y i + r)) + (Fintype.card ι : ℝ) * digamma r -
      (Fintype.card ι : ℝ) * Real.log A)

theorem aScore_eta_direction_hasDerivAt (y : ι → ℝ) (r A : ℝ)
    (S v : ι → ℝ) (hA : 0 < A) :
    HasDerivAt (fun t => aScore y r A (fun i => S i + t * v i))
      (∑ i, (-(r + y i) * Real.exp (S i) / (Real.exp (S i) + A) ^ 2) * v i) 0 := by
  have hl : ∀ i, HasDerivAt
      (fun t : ℝ => (r + y i) / (Real.exp (S i + t * v i) + A))
      ((-(r + y i) * Real.exp (S i) / (Real.exp (S i) + A) ^ 2) * v i) 0 := by
    intro i
    convert! const_div_hasDerivAt
      (((((hasDerivAt_id (0 : ℝ)).mul_const (v i)).const_add (S i)).exp).add_const A)
      (r + y i) (ne_of_gt (by positivity)) using 1
    simp only [id_eq, zero_mul, add_zero, one_mul]
    ring
  exact (HasDerivAt.fun_sum (u := Finset.univ) (fun i _ => hl i)).const_add
    (-(Fintype.card ι : ℝ) * r / A)

end
end CbmrProofs.AggregatedNB

import lean_proof_checks.AggregatedNB

/-! Actual derivatives along shape/rate paths, and the complete composite
Hessian chain rule including the curvature of the inner parameter map. -/
namespace CbmrProofs.AggregatedNB

open scoped BigOperators
open NegativeBinomial
noncomputable section
set_option backward.isDefEq.respectTransparency false

variable {ι : Type*} [Fintype ι]
variable {r A : ℝ → ℝ} {r' A' x : ℝ}

theorem psi_path_hasDerivAt (y S : ι → ℝ)
    (hr : HasDerivAt r r' x) (hA : HasDerivAt A A' x)
    (hposr : 0 < r x) (hposA : 0 < A x) (hy : ∀ i, 0 ≤ y i) :
    HasDerivAt (fun t => aggregatedPsi y (r t) (A t) S)
      (rhoScore y (r x) (A x) S * r' + aScore y (r x) (A x) S * A') x := by
  have hg := HasDerivAt.fun_sum (u := Finset.univ) (fun i _ =>
    (logGamma_hasDerivAt (add_pos_of_nonneg_of_pos (hy i) hposr)).comp x
      (hr.const_add (y i)))
  have hn := ((logGamma_hasDerivAt hposr).comp x hr).const_mul (Fintype.card ι : ℝ)
  have hl := (hr.const_mul (Fintype.card ι : ℝ)).mul (hA.log (ne_of_gt hposA))
  have hs := HasDerivAt.fun_sum (u := Finset.univ) (fun i _ =>
    (hr.add_const (y i)).mul ((hA.const_add (Real.exp (S i))).log
      (ne_of_gt (by positivity))))
  convert! (((hg.neg.add hn).sub hl).add hs).sub_const (∑ i, y i * S i) using 1
  simp only [rhoScore, aScore, div_eq_mul_inv, add_mul, sub_mul, neg_mul,
    Finset.sum_add_distrib, Finset.sum_mul]
  ring_nf
  simp only [add_comm, mul_comm]
  ring

theorem rhoScore_path_hasDerivAt (y S : ι → ℝ)
    (hr : HasDerivAt r r' x) (hA : HasDerivAt A A' x)
    (hposr : 0 < r x) (hposA : 0 < A x) (hy : ∀ i, 0 ≤ y i) :
    HasDerivAt (fun t => rhoScore y (r t) (A t) S)
      (rhoRho y (r x) * r' + rhoA (A x) S * A') x := by
  have hg := HasDerivAt.fun_sum (u := Finset.univ) (fun i _ =>
    (digamma_hasDerivAt (add_pos_of_nonneg_of_pos (hy i) hposr)).comp x
      (hr.const_add (y i)))
  have hn := ((digamma_hasDerivAt hposr).comp x hr).const_mul (Fintype.card ι : ℝ)
  have hl := (hA.log (ne_of_gt hposA)).const_mul (Fintype.card ι : ℝ)
  have hs := HasDerivAt.fun_sum (u := Finset.univ) (fun i _ =>
    (hA.const_add (Real.exp (S i))).log (ne_of_gt (by positivity)))
  convert! ((hg.neg.add hn).sub hl).add hs using 1
  simp only [rhoRho, rhoA, div_eq_mul_inv, add_mul, neg_mul, Finset.sum_mul, one_mul]
  ring_nf
  simp only [add_comm, mul_comm]
  ring

theorem aScore_path_hasDerivAt (y S : ι → ℝ)
    (hr : HasDerivAt r r' x) (hA : HasDerivAt A A' x) (hposA : 0 < A x) :
    HasDerivAt (fun t => aScore y (r t) (A t) S)
      (rhoA (A x) S * r' + aA y (r x) (A x) S * A') x := by
  have hn := (hr.const_mul (-(Fintype.card ι : ℝ))).div hA (ne_of_gt hposA)
  have hs := HasDerivAt.fun_sum (u := Finset.univ) (fun i _ =>
    (hr.add_const (y i)).div (hA.const_add (Real.exp (S i))) (ne_of_gt (by positivity)))
  have he : ∀ i, (r' * (Real.exp (S i) + A x) - (r x + y i) * A') /
      (Real.exp (S i) + A x) ^ 2 =
      (1 / (Real.exp (S i) + A x)) * r' -
        ((r x + y i) / (Real.exp (S i) + A x) ^ 2) * A' := by
    intro i
    have hd : Real.exp (S i) + A x ≠ 0 := ne_of_gt (by positivity)
    field_simp
  convert! hn.add hs using 1
  simp only [he]
  simp only [rhoA, aA, add_mul, sub_mul, Finset.sum_sub_distrib, Finset.sum_mul]
  field_simp
  ring

theorem etaScore_path_hasDerivAt (y eta : ℝ)
    (hr : HasDerivAt r r' x) (hA : HasDerivAt A A' x) (hposA : 0 < A x) :
    HasDerivAt (fun t => etaScore y (r t) (A t) eta)
      (Real.exp eta / (Real.exp eta + A x) * r' -
        (r x + y) * Real.exp eta / (Real.exp eta + A x) ^ 2 * A') x := by
  have hd : Real.exp eta + A x ≠ 0 := ne_of_gt (by positivity)
  convert! (((hr.add_const y).mul_const (Real.exp eta)).div
    (hA.const_add (Real.exp eta)) hd).sub_const y using 1
  field_simp

/-- Mixed differentiation of a first-direction composite score. `R` and `D`
are the inner map's first-direction derivatives; their derivatives `R'`, `D'`
are its mixed curvatures. These are genuine derivative hypotheses on the inner
map, not assumptions of the composite Hessian formula. -/
theorem composite_score_hasDerivAt (y S : ι → ℝ)
    {R D : ℝ → ℝ} {R' D' : ℝ}
    (hr : HasDerivAt r r' x) (hA : HasDerivAt A A' x)
    (hR : HasDerivAt R R' x) (hD : HasDerivAt D D' x)
    (hposr : 0 < r x) (hposA : 0 < A x) (hy : ∀ i, 0 ≤ y i) :
    HasDerivAt (fun t => rhoScore y (r t) (A t) S * R t +
      aScore y (r t) (A t) S * D t)
      (rhoRho y (r x) * R x * r' +
        rhoA (A x) S * (R x * A' + D x * r') +
        aA y (r x) (A x) S * D x * A' +
        rhoScore y (r x) (A x) S * R' + aScore y (r x) (A x) S * D') x := by
  convert! ((rhoScore_path_hasDerivAt y S hr hA hposr hposA hy).mul hR).add
    ((aScore_path_hasDerivAt y S hr hA hposA).mul hD) using 1
  ring

/-- The spatial/global block, for an arbitrary spatial design direction `B`. -/
theorem spatial_global_score_hasDerivAt (y S B : ι → ℝ)
    (hr : HasDerivAt r r' x) (hA : HasDerivAt A A' x) (hposA : 0 < A x) :
    HasDerivAt (fun t => ∑ i, B i * etaScore (y i) (r t) (A t) (S i))
      (∑ i, B i * (Real.exp (S i) / (Real.exp (S i) + A x) * r' -
        (r x + y i) * Real.exp (S i) / (Real.exp (S i) + A x) ^ 2 * A')) x :=
  HasDerivAt.fun_sum (u := Finset.univ) (fun i _ =>
    (etaScore_path_hasDerivAt (y i) (S i) hr hA hposA).const_mul (B i))

end
end CbmrProofs.AggregatedNB

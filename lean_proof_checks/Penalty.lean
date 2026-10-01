import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Tactic.Convert
import Mathlib.Tactic.Ring

/-! Actual directional and mixed derivatives of finite-dimensional quadratic penalties. -/

namespace CbmrProofs.Penalty

set_option backward.isDefEq.respectTransparency false

open scoped BigOperators

variable {A : Type*} [Fintype A]

noncomputable def quadratic (K : A → A → ℝ) (theta : A → ℝ) : ℝ :=
  (1 / 2 : ℝ) * ∑ a, ∑ b, theta a * K a b * theta b

theorem quadratic_directional_raw (K : A → A → ℝ) (theta v : A → ℝ) :
    HasDerivAt (fun t => quadratic K (fun a => theta a + t * v a))
      ((1 / 2 : ℝ) * ∑ a, ∑ b,
        (v a * K a b * theta b + theta a * K a b * v b)) 0 := by
  have h (a : A) : HasDerivAt (fun t : ℝ => theta a + t * v a) (v a) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (v a)).const_add (theta a)
  have ht (a b : A) :
      HasDerivAt (fun t => (theta a + t * v a) * K a b * (theta b + t * v b))
        (v a * K a b * theta b + theta a * K a b * v b) 0 := by
    (convert! ((h a).mul_const (K a b)).mul (h b) using 1; simp)
  exact ((HasDerivAt.fun_sum (u := Finset.univ) fun a _ =>
    HasDerivAt.fun_sum (u := Finset.univ) fun b _ => ht a b).const_mul (1 / 2 : ℝ))

theorem symmetric_quadratic_gradient (K : A → A → ℝ)
    (hK : ∀ a b, K a b = K b a) (theta v : A → ℝ) :
    (1 / 2 : ℝ) * (∑ a, ∑ b,
        (v a * K a b * theta b + theta a * K a b * v b)) =
      ∑ a, (∑ b, K a b * theta b) * v a := by
  have hs :
      (∑ a, ∑ b, theta a * K a b * v b) =
        ∑ a, ∑ b, v a * K a b * theta b := by
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro a _
    apply Finset.sum_congr rfl
    intro b _
    rw [hK b a]
    ring
  have hv :
      (∑ a, ∑ b, v a * K a b * theta b) =
        ∑ a, (∑ b, K a b * theta b) * v a := by
    simp only [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro a _
    apply Finset.sum_congr rfl
    intro b _
    ring
  simp only [Finset.sum_add_distrib]
  rw [hs, hv]
  ring

theorem quadratic_directional (K : A → A → ℝ)
    (hK : ∀ a b, K a b = K b a) (theta v : A → ℝ) :
    HasDerivAt (fun t => quadratic K (fun a => theta a + t * v a))
      (∑ a, (∑ b, K a b * theta b) * v a) 0 := by
  rw [← symmetric_quadratic_gradient K hK theta v]
  exact quadratic_directional_raw K theta v

theorem quadratic_gradient_directional (K : A → A → ℝ) (theta v u : A → ℝ) :
    HasDerivAt (fun t => ∑ a, (∑ b, K a b * (theta b + t * u b)) * v a)
      (∑ a, (∑ b, K a b * u b) * v a) 0 := by
  apply HasDerivAt.fun_sum
  intro a _
  apply HasDerivAt.mul_const
  apply HasDerivAt.fun_sum
  intro b _
  (convert! (((hasDerivAt_id (0 : ℝ)).mul_const (u b)).const_add
    (theta b)).const_mul (K a b) using 1; simp)

theorem quadratic_mixed (K : A → A → ℝ)
    (hK : ∀ a b, K a b = K b a) (theta v u : A → ℝ) :
    HasDerivAt
      (fun t => deriv (fun s => quadratic K
        (fun a => theta a + s * v a + t * u a)) 0)
      (∑ a, (∑ b, K a b * u b) * v a) 0 := by
  have hfirst (t : ℝ) :
      deriv (fun s => quadratic K (fun a => theta a + s * v a + t * u a)) 0 =
        ∑ a, (∑ b, K a b * (theta b + t * u b)) * v a := by
    have h := (quadratic_directional K hK (fun a => theta a + t * u a) v).deriv
    convert! h using 1
    congr 1
    funext s
    congr 1
    funext a
    ring
  simp_rw [hfirst]
  exact quadratic_gradient_directional K theta v u

theorem quadratic_coordinate_gradient [DecidableEq A] (K : A → A → ℝ)
    (hK : ∀ a b, K a b = K b a) (theta : A → ℝ) (a : A) :
    HasDerivAt (fun t => quadratic K (fun b => theta b + t * if b = a then 1 else 0))
      (∑ b, K a b * theta b) 0 := by
  simpa using quadratic_directional K hK theta (fun b => if b = a then 1 else 0)

theorem quadratic_coordinate_hessian [DecidableEq A] (K : A → A → ℝ)
    (hK : ∀ a b, K a b = K b a) (theta : A → ℝ) (a b : A) :
    HasDerivAt
      (fun t => deriv (fun s => quadratic K (fun c =>
        theta c + s * (if c = a then 1 else 0) + t * (if c = b then 1 else 0))) 0)
      (K a b) 0 := by
  simpa using quadratic_mixed K hK theta
    (fun c => if c = a then 1 else 0) (fun c => if c = b then 1 else 0)

theorem penalised_directional (F : (A → ℝ) → ℝ) (K : A → A → ℝ)
    (hK : ∀ a b, K a b = K b a) (theta v : A → ℝ) (gradient : ℝ)
    (hF : HasDerivAt (fun t => F (fun a => theta a + t * v a)) gradient 0) :
    HasDerivAt (fun t => F (fun a => theta a + t * v a) +
      quadratic K (fun a => theta a + t * v a))
      (gradient + ∑ a, (∑ b, K a b * theta b) * v a) 0 :=
  hF.add (quadratic_directional K hK theta v)

theorem penalty_normalisation (K : A → A → ℝ) (theta : A → ℝ) :
    (∑ a, ∑ b, theta a * K a b * theta b) = 2 * quadratic K theta := by
  unfold quadratic
  ring

theorem quadratic_kernel_invariance (K : A → A → ℝ)
    (hK : ∀ a b, K a b = K b a) (theta c : A → ℝ)
    (hc : ∀ a, ∑ b, K a b * c b = 0) (t : ℝ) :
    quadratic K (fun a => theta a + t * c a) = quadratic K theta := by
  have right (x : A → ℝ) : (∑ a, ∑ b, x a * K a b * c b) = 0 := by
    simp only [mul_assoc, ← Finset.mul_sum, hc, mul_zero, Finset.sum_const_zero]
  have left (x : A → ℝ) : (∑ a, ∑ b, c a * K a b * x b) = 0 := by
    rw [Finset.sum_comm]
    calc
      (∑ b, ∑ a, c a * K a b * x b) = ∑ b, ∑ a, x b * K b a * c a := by
        apply Finset.sum_congr rfl
        intro b _
        apply Finset.sum_congr rfl
        intro a _
        rw [hK a b]
        ring
      _ = 0 := right x
  have term (a b : A) :
      (theta a + t * c a) * K a b * (theta b + t * c b) =
        theta a * K a b * theta b + t * (c a * K a b * theta b) +
          t * (theta a * K a b * c b) + t ^ 2 * (c a * K a b * c b) := by
    ring
  unfold quadratic
  simp_rw [term]
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum, left, right, mul_zero, add_zero]

theorem penalised_mixed (F : (A → ℝ) → ℝ) (K : A → A → ℝ)
    (hK : ∀ a b, K a b = K b a) (theta v u : A → ℝ)
    (gradient : ℝ → ℝ) (curvature : ℝ)
    (hF : ∀ t, HasDerivAt
      (fun s => F (fun a => theta a + s * v a + t * u a)) (gradient t) 0)
    (hH : HasDerivAt gradient curvature 0) :
    HasDerivAt (fun t => deriv (fun s =>
      F (fun a => theta a + s * v a + t * u a) +
      quadratic K (fun a => theta a + s * v a + t * u a)) 0)
      (curvature + ∑ a, (∑ b, K a b * u b) * v a) 0 := by
  have hQ (t : ℝ) : HasDerivAt
      (fun s => quadratic K (fun a => theta a + s * v a + t * u a))
      (∑ a, (∑ b, K a b * (theta b + t * u b)) * v a) 0 := by
    convert! quadratic_directional K hK (fun a => theta a + t * u a) v using 1
    funext s
    congr 1
    funext a
    ring
  have hderiv (t : ℝ) :
      deriv (fun s => F (fun a => theta a + s * v a + t * u a) +
        quadratic K (fun a => theta a + s * v a + t * u a)) 0 =
      gradient t + deriv
        (fun s => quadratic K (fun a => theta a + s * v a + t * u a)) 0 := by
    rw [deriv_fun_add (hF t).differentiableAt (hQ t).differentiableAt, (hF t).deriv]
  simp_rw [hderiv]
  exact hH.add (quadratic_mixed K hK theta v u)

section Grouped

variable {G P R : Type*} [Fintype G] [Fintype P] [Fintype R] [DecidableEq G]

noncomputable def groupedMatrix (J : P → P → ℝ) (weight : G → ℝ) :
    ((G × P) ⊕ R) → ((G × P) ⊕ R) → ℝ
  | Sum.inl (g, p), Sum.inl (h, q) => if g = h then weight g * J p q else 0
  | _, _ => 0

theorem grouped_quadratic (J : P → P → ℝ) (weight : G → ℝ)
    (xi : G → P → ℝ) (gamma : R → ℝ) :
    quadratic (groupedMatrix J weight) (Sum.elim (fun gp => xi gp.1 gp.2) gamma) =
      (1 / 2 : ℝ) * ∑ g, weight g * ∑ p, ∑ q, xi g p * J p q * xi g q := by
  simp only [quadratic, Fintype.sum_sum_type, Fintype.sum_prod_type, groupedMatrix,
    Sum.elim_inl, Sum.elim_inr, mul_ite, ite_mul, mul_zero, zero_mul,
    Finset.sum_ite_irrel, Finset.sum_ite_eq, Finset.mem_univ, if_true,
    Finset.sum_const_zero, add_zero]
  congr 1
  apply Finset.sum_congr rfl
  intro g _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro p _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro q _
  ring

omit [Fintype G] [Fintype P] [Fintype R] in
theorem grouped_symmetric (J : P → P → ℝ) (weight : G → ℝ)
    (hJ : ∀ p q, J p q = J q p) :
    ∀ a b : (G × P) ⊕ R, groupedMatrix J weight a b = groupedMatrix J weight b a := by
  intro a b
  rcases a with ⟨g, p⟩ | r <;> rcases b with ⟨h, q⟩ | s
  · by_cases hgh : g = h
    · subst h
      simp [groupedMatrix, hJ p q]
    · simp [groupedMatrix, hgh, Ne.symm hgh]
  · rfl
  · rfl
  · rfl

omit [Fintype G] [Fintype P] [Fintype R] in
theorem grouped_global_unpenalised (J : P → P → ℝ) (weight : G → ℝ)
    (r : R) (a : (G × P) ⊕ R) :
    groupedMatrix J weight (Sum.inr r) a = 0 ∧
      groupedMatrix J weight a (Sum.inr r) = 0 := by
  cases a <;> exact ⟨rfl, rfl⟩

omit [Fintype G] [Fintype P] [Fintype R] in
theorem grouped_cross_group_zero (J : P → P → ℝ) (weight : G → ℝ)
    (g h : G) (p q : P) (hgh : g ≠ h) :
    groupedMatrix (R := R) J weight (Sum.inl (g, p)) (Sum.inl (h, q)) = 0 := by
  simp [groupedMatrix, hgh]

theorem grouped_nonnegative (J : P → P → ℝ) (weight : G → ℝ)
    (hJ : ∀ x : P → ℝ, 0 ≤ ∑ p, ∑ q, x p * J p q * x q)
    (hw : ∀ g, 0 ≤ weight g) (xi : G → P → ℝ) (gamma : R → ℝ) :
    0 ≤ quadratic (groupedMatrix J weight) (Sum.elim (fun gp => xi gp.1 gp.2) gamma) := by
  rw [grouped_quadratic]
  exact mul_nonneg (by norm_num) (Finset.sum_nonneg fun g _ => mul_nonneg (hw g) (hJ _))

end Grouped

end CbmrProofs.Penalty

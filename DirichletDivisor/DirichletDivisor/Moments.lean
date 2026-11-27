import DirichletDivisor.Window

set_option linter.style.whitespace false

open Finset MeasureTheory

namespace DirichletDivisor

theorem eta_neg (u : ℝ) : eta (-u) = eta u := by
  unfold eta bump
  rw [neg_sq]

/-- σ² = ∫ η(u) u² du. -/
noncomputable def etaVar : ℝ := ∫ u in (-1:ℝ)..1, eta u * u ^ 2

/-- ∫ η(u) u du = 0, by oddness. -/
theorem eta_mean_zero : ∫ u in (-1:ℝ)..1, eta u * u = 0 := by
  have h := intervalIntegral.integral_comp_neg (a := (-1:ℝ)) (b := 1)
    (fun u => eta u * u)
  simp only [neg_neg, eta_neg, mul_neg, intervalIntegral.integral_neg] at h
  linarith

theorem etaVar_nonneg : 0 ≤ etaVar :=
  intervalIntegral.integral_nonneg (by norm_num)
    (fun u _ => mul_nonneg (eta_nonneg u) (sq_nonneg u))

theorem etaVar_le_one : etaVar ≤ 1 := by
  have h1 : etaVar ≤ ∫ u in (-1:ℝ)..1, eta u * 1 :=
    intervalIntegral.integral_mono_on (by norm_num)
      ((eta_continuous.mul (continuous_id.pow 2)).intervalIntegrable _ _)
      ((eta_continuous.mul continuous_const).intervalIntegrable _ _)
      (fun u hu => mul_le_mul_of_nonneg_left
        (by nlinarith [hu.1, hu.2]) (eta_nonneg u))
  have h2 : ∫ u in (-1:ℝ)..1, eta u * 1 = 1 := by
    simp only [mul_one]
    exact eta_integral
  linarith

/-- The smoothing centre: ∫ η (s + L u)² = s² + L² σ². -/
theorem eta_moment (s L : ℝ) :
    ∫ u in (-1:ℝ)..1, eta u * (s + L * u) ^ 2 = s ^ 2 + L ^ 2 * etaVar := by
  have i0 : IntervalIntegrable (fun u => s ^ 2 * eta u) volume (-1) 1 :=
    (continuous_const.mul eta_continuous).intervalIntegrable _ _
  have i1 : IntervalIntegrable (fun u => 2 * s * L * (eta u * u)) volume (-1) 1 :=
    (continuous_const.mul (eta_continuous.mul continuous_id)).intervalIntegrable _ _
  have i2 : IntervalIntegrable (fun u => L ^ 2 * (eta u * u ^ 2)) volume (-1) 1 :=
    (continuous_const.mul (eta_continuous.mul (continuous_id.pow 2))).intervalIntegrable _ _
  have i01 : IntervalIntegrable
      (fun u => s ^ 2 * eta u + 2 * s * L * (eta u * u)) volume (-1) 1 :=
    ((continuous_const.mul eta_continuous).add
      (continuous_const.mul (eta_continuous.mul continuous_id))).intervalIntegrable _ _
  have hexp : ∀ u : ℝ, eta u * (s + L * u) ^ 2
      = s ^ 2 * eta u + 2 * s * L * (eta u * u) + L ^ 2 * (eta u * u ^ 2) :=
    fun u => by ring
  simp only [hexp]
  rw [intervalIntegral.integral_add i01 i2, intervalIntegral.integral_add i0 i1,
    intervalIntegral.integral_const_mul, intervalIntegral.integral_const_mul,
    intervalIntegral.integral_const_mul, eta_integral, eta_mean_zero]
  unfold etaVar
  ring

/-- The window identity at the explicit centre N̄ = s² + L² σ². -/
theorem R_window_center (s L : ℝ) (hL0 : 0 ≤ L) (hLs : L ≤ s) (K : ℕ)
    (hK : (s + L) ^ 2 ≤ K) :
    R (s ^ 2 + L ^ 2 * etaVar)
      = (∫ u in (-1:ℝ)..1, eta u * R ((s + L * u) ^ 2))
        + ∑ k ∈ (Icc 1 K).filter
              (fun k : ℕ => (s - L) ^ 2 < k ∧ (k : ℝ) < (s + L) ^ 2),
            (d k : ℝ) * corr eta (fun u => (s + L * u) ^ 2)
              (s ^ 2 + L ^ 2 * etaVar) k := by
  have h := R_window_sqrt s L hL0 hLs K hK
  rw [eta_moment] at h
  exact h

end DirichletDivisor

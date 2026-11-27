import DirichletDivisor.Smoothing

set_option linter.style.whitespace false

open MeasureTheory

namespace DirichletDivisor

/-- Unnormalized bump: exp(-1/(1-u²)) on |u| < 1, zero elsewhere. -/
noncomputable def bump (u : ℝ) : ℝ := expNegInvGlue (1 - u ^ 2)

theorem bump_continuous : Continuous bump := by
  have h : Continuous expNegInvGlue :=
    contDiff_zero.mp (expNegInvGlue.contDiff (n := (0 : ℕ∞)))
  exact h.comp (continuous_const.sub (continuous_pow 2))

theorem bump_nonneg (u : ℝ) : 0 ≤ bump u :=
  expNegInvGlue.nonneg _

theorem bump_pos {u : ℝ} (h : u ^ 2 < 1) : 0 < bump u :=
  expNegInvGlue.pos_of_pos (by linarith)

/-- I₀ = ∫ exp(-1/(1-t²)) dt. -/
noncomputable def bumpMass : ℝ := ∫ u in (-1:ℝ)..1, bump u

theorem bumpMass_pos : 0 < bumpMass := by
  unfold bumpMass
  apply intervalIntegral.intervalIntegral_pos_of_pos_on
  · exact bump_continuous.intervalIntegrable _ _
  · intro u hu
    exact bump_pos (by nlinarith [hu.1, hu.2])
  · norm_num

/-- The normalized bump η = bump / I₀. -/
noncomputable def eta (u : ℝ) : ℝ := bump u / bumpMass

theorem eta_continuous : Continuous eta :=
  bump_continuous.div_const _

theorem eta_nonneg (u : ℝ) : 0 ≤ eta u :=
  div_nonneg (bump_nonneg u) bumpMass_pos.le

theorem eta_integral : ∫ u in (-1:ℝ)..1, eta u = 1 := by
  unfold eta
  rw [intervalIntegral.integral_div]
  exact div_self bumpMass_pos.ne'

end DirichletDivisor

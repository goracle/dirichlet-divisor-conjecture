import DirichletDivisor.DualInt
import DirichletDivisor.FsDecay
import DirichletDivisor.TailSpecialize

set_option linter.style.whitespace false

namespace DirichletDivisor

open MeasureTheory

/-! # Per-`m` bound on the smoothed dual term

Goal (feeds `tail_param_le`): for `L = tailL N`, `x(u) = (s + L u)²`,

  |∫ η(u) · mainTerm(x(u)) m du| ≤ 2^16 N^{3/4} · d(m) · m^{-3/4} · exp(-(tailC N · m^{1/4})).

Layers:
1. `∫ η (1+δu)^a sin/cos(θ + 2πξu) du` is `Im/Re (e^{iθ} · Fs a δ ξ)`, so bounded by `‖Fs‖`.
2. Pointwise: `mainTerm` is an exact `sin`/`cos` combination (Hankel two-term main part).
3. Integrate: `|∫ η·mainTerm| ≤ d(m)(K₁‖F_{3/2}‖ + K₂‖F_{1/2}‖)`.
4. Plug in `Fs_norm_le_exp` and bound `K₁ + K₂ ≤ N^{3/4} m^{-3/4}`.
-/

/-! ## Layer 1: trigonometric integrals against the bump -/

theorem continuous_eta_pow (a δ : ℝ) (ha : 0 ≤ a) :
    Continuous fun t : ℝ => eta t * (1 + δ * t) ^ a :=
  eta_continuous.mul
    ((continuous_const.add (continuous_const.mul continuous_id)).rpow_const
      (fun _ => Or.inr ha))

theorem continuous_eta_pow_sin (a δ θ ξ : ℝ) (ha : 0 ≤ a) :
    Continuous fun u : ℝ => eta u * (1 + δ * u) ^ a * Real.sin (θ + 2 * Real.pi * ξ * u) :=
  (continuous_eta_pow a δ ha).mul
    (Real.continuous_sin.comp (continuous_const.add (continuous_const.mul continuous_id)))

theorem continuous_eta_pow_cos (a δ θ ξ : ℝ) (ha : 0 ≤ a) :
    Continuous fun u : ℝ => eta u * (1 + δ * u) ^ a * Real.cos (θ + 2 * Real.pi * ξ * u) :=
  (continuous_eta_pow a δ ha).mul
    (Real.continuous_cos.comp (continuous_const.add (continuous_const.mul continuous_id)))

theorem phase_integrand_continuous (a δ θ ξ : ℝ) (ha : 0 ≤ a) :
    Continuous fun t : ℝ => (((eta t * (1 + δ * t) ^ a : ℝ) : ℂ) *
      Complex.exp (((θ + 2 * Real.pi * ξ * t : ℝ) : ℂ) * Complex.I)) := by
  refine (Complex.continuous_ofReal.comp (continuous_eta_pow a δ ha)).mul ?_
  refine Complex.continuous_exp.comp ?_
  refine Continuous.mul (Complex.continuous_ofReal.comp ?_) continuous_const
  exact continuous_const.add (continuous_const.mul continuous_id)

/-- `e^{iθ} · Fs = ∫ η(1+δt)^a e^{i(θ + 2πξ t)}`. -/
theorem Fs_phase_integral (a δ θ ξ : ℝ) :
    Complex.exp ((θ : ℂ) * Complex.I) * Fs a δ ξ =
      ∫ t in (-1:ℝ)..1, (((eta t * (1 + δ * t) ^ a : ℝ) : ℂ) *
        Complex.exp (((θ + 2 * Real.pi * ξ * t : ℝ) : ℂ) * Complex.I)) := by
  unfold Fs
  rw [← intervalIntegral.integral_const_mul]
  have key : ∀ t : ℝ, Complex.exp ((θ : ℂ) * Complex.I) *
      (((eta t * (1 + δ * t) ^ a : ℝ) : ℂ) *
        Complex.exp (((2 * Real.pi * ξ * t : ℝ) : ℂ) * Complex.I))
      = (((eta t * (1 + δ * t) ^ a : ℝ) : ℂ) *
        Complex.exp (((θ + 2 * Real.pi * ξ * t : ℝ) : ℂ) * Complex.I)) := by
    intro t
    have h : ((θ + 2 * Real.pi * ξ * t : ℝ) : ℂ) * Complex.I
        = (θ : ℂ) * Complex.I + ((2 * Real.pi * ξ * t : ℝ) : ℂ) * Complex.I := by
      push_cast; ring
    rw [h, Complex.exp_add]; ring
  exact intervalIntegral.integral_congr (fun t _ => key t)

theorem eta_pow_sin_integral (a δ θ ξ : ℝ) (ha : 0 ≤ a) :
    (∫ u in (-1:ℝ)..1, eta u * (1 + δ * u) ^ a * Real.sin (θ + 2 * Real.pi * ξ * u))
      = (Complex.exp ((θ : ℂ) * Complex.I) * Fs a δ ξ).im := by
  rw [Fs_phase_integral]
  have h := ContinuousLinearMap.intervalIntegral_comp_comm Complex.imCLM
    ((phase_integrand_continuous a δ θ ξ ha).intervalIntegrable (μ := volume) (-1) 1)
  simp only [Complex.imCLM_apply] at h
  rw [← h]
  have key : ∀ t : ℝ, (((eta t * (1 + δ * t) ^ a : ℝ) : ℂ) *
        Complex.exp (((θ + 2 * Real.pi * ξ * t : ℝ) : ℂ) * Complex.I)).im
      = eta t * (1 + δ * t) ^ a * Real.sin (θ + 2 * Real.pi * ξ * t) := by
    intro t
    rw [Complex.im_ofReal_mul, Complex.exp_ofReal_mul_I_im]
  exact intervalIntegral.integral_congr (fun t _ => (key t).symm)

theorem eta_pow_cos_integral (a δ θ ξ : ℝ) (ha : 0 ≤ a) :
    (∫ u in (-1:ℝ)..1, eta u * (1 + δ * u) ^ a * Real.cos (θ + 2 * Real.pi * ξ * u))
      = (Complex.exp ((θ : ℂ) * Complex.I) * Fs a δ ξ).re := by
  rw [Fs_phase_integral]
  have h := ContinuousLinearMap.intervalIntegral_comp_comm Complex.reCLM
    ((phase_integrand_continuous a δ θ ξ ha).intervalIntegrable (μ := volume) (-1) 1)
  simp only [Complex.reCLM_apply] at h
  rw [← h]
  have key : ∀ t : ℝ, (((eta t * (1 + δ * t) ^ a : ℝ) : ℂ) *
        Complex.exp (((θ + 2 * Real.pi * ξ * t : ℝ) : ℂ) * Complex.I)).re
      = eta t * (1 + δ * t) ^ a * Real.cos (θ + 2 * Real.pi * ξ * t) := by
    intro t
    rw [Complex.re_ofReal_mul, Complex.exp_ofReal_mul_I_re]
  exact intervalIntegral.integral_congr (fun t _ => (key t).symm)

theorem abs_eta_pow_sin_le (a δ θ ξ : ℝ) (ha : 0 ≤ a) :
    |∫ u in (-1:ℝ)..1, eta u * (1 + δ * u) ^ a * Real.sin (θ + 2 * Real.pi * ξ * u)|
      ≤ ‖Fs a δ ξ‖ := by
  rw [eta_pow_sin_integral a δ θ ξ ha]
  calc _ ≤ ‖Complex.exp ((θ : ℂ) * Complex.I) * Fs a δ ξ‖ := Complex.abs_im_le_norm _
    _ = ‖Fs a δ ξ‖ := by rw [norm_mul, Complex.norm_exp_ofReal_mul_I, one_mul]

theorem abs_eta_pow_cos_le (a δ θ ξ : ℝ) (ha : 0 ≤ a) :
    |∫ u in (-1:ℝ)..1, eta u * (1 + δ * u) ^ a * Real.cos (θ + 2 * Real.pi * ξ * u)|
      ≤ ‖Fs a δ ξ‖ := by
  rw [eta_pow_cos_integral a δ θ ξ ha]
  calc _ ≤ ‖Complex.exp ((θ : ℂ) * Complex.I) * Fs a δ ξ‖ := Complex.abs_re_le_norm _
    _ = ‖Fs a δ ξ‖ := by rw [norm_mul, Complex.norm_exp_ofReal_mul_I, one_mul]

/-! ## Root helpers -/

theorem sqrt_sqrt_eq_rpow (x : ℝ) (hx : 0 ≤ x) :
    Real.sqrt (Real.sqrt x) = x ^ (1 / 4 : ℝ) := by
  rw [Real.sqrt_eq_rpow, Real.sqrt_eq_rpow, ← Real.rpow_mul hx]
  norm_num

theorem pow_sqrt_sqrt (x : ℝ) (hx : 0 ≤ x) (k : ℕ) :
    (Real.sqrt (Real.sqrt x)) ^ k = x ^ ((k : ℝ) / 4) := by
  rw [sqrt_sqrt_eq_rpow x hx, ← Real.rpow_natCast, ← Real.rpow_mul hx]
  congr 1; ring

theorem rpow_three_quarters (x : ℝ) (hx : 0 ≤ x) :
    x ^ (3 / 4 : ℝ) = (Real.sqrt (Real.sqrt x)) ^ 3 := by
  rw [pow_sqrt_sqrt x hx 3]; norm_num

theorem one_le_sqrt_sqrt (x : ℝ) (hx : 1 ≤ x) : 1 ≤ Real.sqrt (Real.sqrt x) := by
  have h1 : 1 ≤ Real.sqrt x := by simpa using Real.sqrt_le_sqrt hx
  simpa using Real.sqrt_le_sqrt h1

theorem rpow_three_halves_eq (x : ℝ) (hx : 0 ≤ x) :
    x ^ (3 / 2 : ℝ) = (Real.sqrt x) ^ 3 := by
  rw [Real.sqrt_eq_rpow, ← Real.rpow_natCast, ← Real.rpow_mul hx]
  norm_num

/-! ## Layer 2: `mainTerm` as an exact `sin`/`cos` combination -/

/-- With `y = s + L u > 0`: `(y²/m) · main(4π√(m y²))/(2π)` in closed form,
`a = √y`, `b = m^{1/4}`. -/
theorem mainTerm_pt (y : ℝ) (hy : 0 < y) (m : ℕ) (hm : 1 ≤ m) :
    ((y ^ 2) / (m : ℝ)) *
        (hankelMain (4 * Real.pi * Real.sqrt ((m : ℝ) * y ^ 2)) / (2 * Real.pi))
      = 1 / (2 * Real.sqrt 2 * Real.pi ^ 2) *
          (((Real.sqrt (Real.sqrt (m : ℝ))) ^ 5)⁻¹ * (Real.sqrt y) ^ 3 *
            Real.sin (4 * Real.pi * (Real.sqrt (m : ℝ) * y) - 5 * Real.pi / 4))
        + 15 / (64 * Real.sqrt 2 * Real.pi ^ 3) *
          (((Real.sqrt (Real.sqrt (m : ℝ))) ^ 7)⁻¹ * Real.sqrt y *
            Real.cos (4 * Real.pi * (Real.sqrt (m : ℝ) * y) - 5 * Real.pi / 4)) := by
  have hm0 : (0 : ℝ) < m := by exact_mod_cast hm
  set a := Real.sqrt y with ha_def
  set b := Real.sqrt (Real.sqrt (m : ℝ)) with hb_def
  have ha : 0 < a := Real.sqrt_pos.2 hy
  have hay : a ^ 2 = y := Real.sq_sqrt hy.le
  have hsm0 : 0 < Real.sqrt (m : ℝ) := Real.sqrt_pos.2 hm0
  have hb : 0 < b := Real.sqrt_pos.2 hsm0
  have hbs : b ^ 2 = Real.sqrt (m : ℝ) := Real.sq_sqrt hsm0.le
  have hbm : b ^ 4 = (m : ℝ) := by
    calc b ^ 4 = (b ^ 2) ^ 2 := by ring
      _ = (Real.sqrt (m : ℝ)) ^ 2 := by rw [hbs]
      _ = m := Real.sq_sqrt hm0.le
  have hsq : Real.sqrt ((m : ℝ) * y ^ 2) = b ^ 2 * y := by
    rw [Real.sqrt_mul hm0.le, Real.sqrt_sq hy.le, hbs]
  have h2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have h2pos : 0 < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)
  have hZ : Real.sqrt (2 / (Real.pi * (4 * Real.pi * (b ^ 2 * y))))
      = 1 / (Real.sqrt 2 * Real.pi * a * b) := by
    have h1 : 2 / (Real.pi * (4 * Real.pi * (b ^ 2 * y)))
        = (1 / (Real.sqrt 2 * Real.pi * a * b)) ^ 2 := by
      rw [← hay, div_pow, one_pow, mul_pow, mul_pow, mul_pow, h2]
      rw [div_eq_div_iff (by positivity) (by positivity)]
      ring
    rw [h1]
    exact Real.sqrt_sq (by positivity)
  unfold hankelMain
  rw [hsq, hZ, ← hbs]
  rw [← hay, ← hbm]
  have h2ne : Real.sqrt 2 ≠ 0 := h2pos.ne'
  have hpine : Real.pi ≠ 0 := Real.pi_pos.ne'
  have hane : a ≠ 0 := ha.ne'
  have hbne : b ≠ 0 := hb.ne'
  first
    | (field_simp; ring)
    | field_simp

noncomputable def dualK1 (s : ℝ) (m : ℕ) : ℝ :=
  1 / (2 * Real.sqrt 2 * Real.pi ^ 2) * ((Real.sqrt (Real.sqrt (m : ℝ))) ^ 5)⁻¹
    * (Real.sqrt s) ^ 3

noncomputable def dualK2 (s : ℝ) (m : ℕ) : ℝ :=
  15 / (64 * Real.sqrt 2 * Real.pi ^ 3) * ((Real.sqrt (Real.sqrt (m : ℝ))) ^ 7)⁻¹
    * Real.sqrt s

theorem dualK1_nonneg (s : ℝ) (m : ℕ) : 0 ≤ dualK1 s m := by
  unfold dualK1; positivity

theorem dualK2_nonneg (s : ℝ) (m : ℕ) : 0 ≤ dualK2 s m := by
  unfold dualK2; positivity

/-- Pointwise form of `η · mainTerm` on the smoothing support.
`θ = 4π√m s - 5π/4`, `ξ = 2 L √m`, `δ = L/s`. -/
theorem eta_mainTerm_pt (s L u : ℝ) (m : ℕ) (hm : 1 ≤ m) (hs : 0 < s) (hL0 : 0 ≤ L)
    (hLs : L < s) (hu : u ∈ Set.Icc (-1 : ℝ) 1) (θ ξ : ℝ)
    (hθ : θ = 4 * Real.pi * Real.sqrt (m : ℝ) * s - 5 * Real.pi / 4)
    (hξ : ξ = 2 * L * Real.sqrt (m : ℝ)) :
    eta u * mainTerm ((s + L * u) ^ 2) m
      = (d m : ℝ) * (dualK1 s m * (eta u * (1 + (L / s) * u) ^ (3 / 2 : ℝ)
            * Real.sin (θ + 2 * Real.pi * ξ * u))
          + dualK2 s m * (eta u * (1 + (L / s) * u) ^ (1 / 2 : ℝ)
            * Real.cos (θ + 2 * Real.pi * ξ * u))) := by
  have hu1 := hu.1
  have hLu : -L ≤ L * u := by nlinarith [hu.1]
  have hy : 0 < s + L * u := by linarith
  have hδu : 0 < 1 + (L / s) * u := by
    have : 1 + (L / s) * u = (s + L * u) / s := by field_simp
    rw [this]; positivity
  have e3 : Real.sqrt (s + L * u) = Real.sqrt s * Real.sqrt (1 + (L / s) * u) := by
    rw [← Real.sqrt_mul hs.le]
    congr 1
    first
      | (field_simp; ring)
      | field_simp
  have e1 : (1 + (L / s) * u) ^ (3 / 2 : ℝ) = (Real.sqrt (1 + (L / s) * u)) ^ 3 :=
    rpow_three_halves_eq _ hδu.le
  have e2 : (1 + (L / s) * u) ^ (1 / 2 : ℝ) = Real.sqrt (1 + (L / s) * u) :=
    (Real.sqrt_eq_rpow _).symm
  have eX : 4 * Real.pi * (Real.sqrt (m : ℝ) * (s + L * u)) - 5 * Real.pi / 4
      = θ + 2 * Real.pi * ξ * u := by
    rw [hθ, hξ]; ring
  unfold mainTerm
  rw [mainTerm_pt (s + L * u) hy m hm, eX, e1, e2, e3]
  unfold dualK1 dualK2
  ring

/-! ## Layer 3: integrate -/

theorem dual_integral_bound (s L : ℝ) (m : ℕ) (hm : 1 ≤ m) (hs : 0 < s) (hL0 : 0 ≤ L)
    (hLs : L < s) :
    |∫ u in (-1 : ℝ)..1, eta u * mainTerm ((s + L * u) ^ 2) m|
      ≤ (d m : ℝ) * (dualK1 s m * ‖Fs (3 / 2) (L / s) (2 * L * Real.sqrt (m : ℝ))‖
          + dualK2 s m * ‖Fs (1 / 2) (L / s) (2 * L * Real.sqrt (m : ℝ))‖) := by
  obtain ⟨θ, hθ⟩ : ∃ θ : ℝ, θ = 4 * Real.pi * Real.sqrt (m : ℝ) * s - 5 * Real.pi / 4 :=
    ⟨_, rfl⟩
  obtain ⟨ξ, hξ⟩ : ∃ ξ : ℝ, ξ = 2 * L * Real.sqrt (m : ℝ) := ⟨_, rfl⟩
  rw [← hξ]
  have hint : (∫ u in (-1 : ℝ)..1, eta u * mainTerm ((s + L * u) ^ 2) m)
      = ∫ u in (-1 : ℝ)..1, (d m : ℝ) *
          (dualK1 s m * (eta u * (1 + (L / s) * u) ^ (3 / 2 : ℝ)
              * Real.sin (θ + 2 * Real.pi * ξ * u))
            + dualK2 s m * (eta u * (1 + (L / s) * u) ^ (1 / 2 : ℝ)
              * Real.cos (θ + 2 * Real.pi * ξ * u))) := by
    refine intervalIntegral.integral_congr (fun u hu => ?_)
    rw [Set.uIcc_of_le (by norm_num : (-1 : ℝ) ≤ 1)] at hu
    exact eta_mainTerm_pt s L u m hm hs hL0 hLs hu θ ξ hθ hξ
  have c1 := continuous_eta_pow_sin (3 / 2) (L / s) θ ξ (by norm_num)
  have c2 := continuous_eta_pow_cos (1 / 2) (L / s) θ ξ (by norm_num)
  have i1 : IntervalIntegrable (fun u : ℝ => dualK1 s m * (eta u * (1 + (L / s) * u) ^ (3 / 2 : ℝ)
      * Real.sin (θ + 2 * Real.pi * ξ * u))) volume (-1) 1 :=
    (continuous_const.mul c1).intervalIntegrable _ _
  have i2 : IntervalIntegrable (fun u : ℝ => dualK2 s m * (eta u * (1 + (L / s) * u) ^ (1 / 2 : ℝ)
      * Real.cos (θ + 2 * Real.pi * ξ * u))) volume (-1) 1 :=
    (continuous_const.mul c2).intervalIntegrable _ _
  rw [hint, intervalIntegral.integral_const_mul, intervalIntegral.integral_add i1 i2,
    intervalIntegral.integral_const_mul, intervalIntegral.integral_const_mul]
  have hA := abs_eta_pow_sin_le (3 / 2) (L / s) θ ξ (by norm_num)
  have hB := abs_eta_pow_cos_le (1 / 2) (L / s) θ ξ (by norm_num)
  have hK1 := dualK1_nonneg s m
  have hK2 := dualK2_nonneg s m
  have hd0 : (0 : ℝ) ≤ d m := Nat.cast_nonneg _
  rw [abs_mul, abs_of_nonneg hd0]
  refine mul_le_mul_of_nonneg_left ?_ hd0
  calc _ ≤ |dualK1 s m * ∫ u in (-1 : ℝ)..1, eta u * (1 + (L / s) * u) ^ (3 / 2 : ℝ)
                * Real.sin (θ + 2 * Real.pi * ξ * u)|
        + |dualK2 s m * ∫ u in (-1 : ℝ)..1, eta u * (1 + (L / s) * u) ^ (1 / 2 : ℝ)
                * Real.cos (θ + 2 * Real.pi * ξ * u)| := abs_add_le _ _
    _ = dualK1 s m * |∫ u in (-1 : ℝ)..1, eta u * (1 + (L / s) * u) ^ (3 / 2 : ℝ)
                * Real.sin (θ + 2 * Real.pi * ξ * u)|
        + dualK2 s m * |∫ u in (-1 : ℝ)..1, eta u * (1 + (L / s) * u) ^ (1 / 2 : ℝ)
                * Real.cos (θ + 2 * Real.pi * ξ * u)| := by
        rw [abs_mul, abs_mul, abs_of_nonneg hK1, abs_of_nonneg hK2]
    _ ≤ _ := add_le_add (mul_le_mul_of_nonneg_left hA hK1) (mul_le_mul_of_nonneg_left hB hK2)

/-! ## Layer 4: the `tail_param_le` shape -/

theorem dualK_sum_le (N s : ℝ) (m : ℕ) (hN : 1 ≤ N) (hs : 0 < s) (hsN : s ^ 2 ≤ N)
    (hm : 1 ≤ m) :
    dualK1 s m + dualK2 s m ≤ N ^ (3 / 4 : ℝ) * (m : ℝ) ^ (-(3 / 4 : ℝ)) := by
  have hm0 : (0 : ℝ) < m := by exact_mod_cast hm
  have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast hm
  have hN0 : (0 : ℝ) ≤ N := by linarith
  have hb1 : 1 ≤ Real.sqrt (Real.sqrt (m : ℝ)) := one_le_sqrt_sqrt _ hm1
  have hq1 : 1 ≤ Real.sqrt (Real.sqrt N) := one_le_sqrt_sqrt _ hN
  have hm34 : (m : ℝ) ^ (-(3 / 4 : ℝ)) = ((Real.sqrt (Real.sqrt (m : ℝ))) ^ 3)⁻¹ := by
    rw [Real.rpow_neg hm0.le, rpow_three_quarters (m : ℝ) hm0.le]
  have hN34 : N ^ (3 / 4 : ℝ) = (Real.sqrt (Real.sqrt N)) ^ 3 := rpow_three_quarters N hN0
  have hq4 : (Real.sqrt (Real.sqrt N)) ^ 4 = N := by
    rw [pow_sqrt_sqrt N hN0 4]; norm_num
  have hq0 : 0 ≤ Real.sqrt (Real.sqrt N) := Real.sqrt_nonneg _
  have hsq : s ≤ (Real.sqrt (Real.sqrt N)) ^ 2 := by
    by_contra h
    have h := not_le.mp h
    have h' : ((Real.sqrt (Real.sqrt N)) ^ 2) ^ 2 < s ^ 2 :=
      pow_lt_pow_left₀ h (by positivity) (by norm_num)
    have h'' : ((Real.sqrt (Real.sqrt N)) ^ 2) ^ 2 = N := by
      calc ((Real.sqrt (Real.sqrt N)) ^ 2) ^ 2 = (Real.sqrt (Real.sqrt N)) ^ 4 := by ring
        _ = N := hq4
    linarith
  have hsqrt : Real.sqrt s ≤ Real.sqrt (Real.sqrt N) :=
    calc Real.sqrt s ≤ Real.sqrt ((Real.sqrt (Real.sqrt N)) ^ 2) := Real.sqrt_le_sqrt hsq
      _ = Real.sqrt (Real.sqrt N) := Real.sqrt_sq hq0
  have hr0 : 0 ≤ Real.sqrt s := Real.sqrt_nonneg _
  -- numeric constants
  have h2 : 1 ≤ Real.sqrt 2 := by simpa using Real.sqrt_le_sqrt (show (1 : ℝ) ≤ 2 by norm_num)
  have hπ : (3 : ℝ) < Real.pi := Real.pi_gt_three
  have hπ2 : (9 : ℝ) ≤ Real.pi ^ 2 := by nlinarith
  have hπ3 : (27 : ℝ) ≤ Real.pi ^ 3 := by
    have := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 3) hπ.le 3
    norm_num at this; linarith
  have hC1 : 1 / (2 * Real.sqrt 2 * Real.pi ^ 2) ≤ 1 / 2 := by
    have : 1 * 9 ≤ Real.sqrt 2 * Real.pi ^ 2 := mul_le_mul h2 hπ2 (by norm_num) (by positivity)
    exact one_div_le_one_div_of_le (by norm_num) (by linarith)
  have hC2 : 15 / (64 * Real.sqrt 2 * Real.pi ^ 3) ≤ 1 / 2 := by
    have : 1 * 27 ≤ Real.sqrt 2 * Real.pi ^ 3 := mul_le_mul h2 hπ3 (by norm_num) (by positivity)
    rw [div_le_iff₀ (by positivity)]
    linarith
  have hC1nn : 0 ≤ 1 / (2 * Real.sqrt 2 * Real.pi ^ 2) := by positivity
  have hC2nn : 0 ≤ 15 / (64 * Real.sqrt 2 * Real.pi ^ 3) := by positivity
  rw [hm34, hN34]
  unfold dualK1 dualK2
  generalize Real.sqrt (Real.sqrt (m : ℝ)) = b at hb1 ⊢
  generalize Real.sqrt (Real.sqrt N) = q at hq1 hsqrt ⊢
  generalize Real.sqrt s = r at hr0 hsqrt ⊢
  have hb0 : 0 < b := by linarith
  have e5 : (b ^ 5)⁻¹ ≤ (b ^ 3)⁻¹ :=
    inv_anti₀ (by positivity) (pow_le_pow_right₀ hb1 (by norm_num))
  have e7 : (b ^ 7)⁻¹ ≤ (b ^ 3)⁻¹ :=
    inv_anti₀ (by positivity) (pow_le_pow_right₀ hb1 (by norm_num))
  have r3 : r ^ 3 ≤ q ^ 3 := pow_le_pow_left₀ hr0 hsqrt 3
  have rq : r ≤ q ^ 3 := by
    have : q ≤ q ^ 3 := by
      calc q = q ^ 1 := (pow_one q).symm
        _ ≤ q ^ 3 := pow_le_pow_right₀ hq1 (by norm_num)
    linarith
  have hq0' : 0 < q := by linarith
  calc 1 / (2 * Real.sqrt 2 * Real.pi ^ 2) * (b ^ 5)⁻¹ * r ^ 3
        + 15 / (64 * Real.sqrt 2 * Real.pi ^ 3) * (b ^ 7)⁻¹ * r
      ≤ 1 / (2 * Real.sqrt 2 * Real.pi ^ 2) * (b ^ 3)⁻¹ * q ^ 3
        + 15 / (64 * Real.sqrt 2 * Real.pi ^ 3) * (b ^ 3)⁻¹ * q ^ 3 :=
        add_le_add
          (mul_le_mul (mul_le_mul_of_nonneg_left e5 hC1nn) r3 (by positivity) (by positivity))
          (mul_le_mul (mul_le_mul_of_nonneg_left e7 hC2nn) rq hr0 (by positivity))
    _ = (1 / (2 * Real.sqrt 2 * Real.pi ^ 2) + 15 / (64 * Real.sqrt 2 * Real.pi ^ 3))
          * ((b ^ 3)⁻¹ * q ^ 3) := by ring
    _ ≤ 1 * ((b ^ 3)⁻¹ * q ^ 3) :=
        mul_le_mul_of_nonneg_right (by linarith) (by positivity)
    _ = q ^ 3 * (b ^ 3)⁻¹ := by ring

/-- **Per-`m` dual term bound**, in the shape consumed by `tail_param_le`. -/
theorem dual_term_bound (N s : ℝ) (m : ℕ) (hN : 1 ≤ N) (hm : 1 ≤ m) (hs : 0 < s)
    (hsN : s ^ 2 ≤ N) (hLs : tailL N < s) (hδ : tailL N / s ≤ 1 / 2) :
    |∫ u in (-1 : ℝ)..1, eta u * mainTerm ((s + tailL N * u) ^ 2) m|
      ≤ 2 ^ 16 * N ^ (3 / 4 : ℝ) *
        ((d m : ℝ) * ((m : ℝ) ^ (-(3 / 4 : ℝ))
          * Real.exp (-(tailC N * (m : ℝ) ^ (1 / 4 : ℝ))))) := by
  have hN0 : 0 < N := by linarith
  have hm0 : (0 : ℝ) < m := by exact_mod_cast hm
  have hL0 : 0 ≤ tailL N := (tailL_pos N hN0).le
  have h0 := dual_integral_bound s (tailL N) m hm hs hL0 hLs
  have hξ0 : 0 ≤ 2 * tailL N * Real.sqrt (m : ℝ) := by
    have := Real.sqrt_nonneg (m : ℝ); positivity
  have hF1 := Fs_norm_le_exp (3 / 2) (tailL N / s) (2 * tailL N * Real.sqrt (m : ℝ))
    (by norm_num) (by norm_num) (div_nonneg hL0 hs.le) hδ
  have hF2 := Fs_norm_le_exp (1 / 2) (tailL N / s) (2 * tailL N * Real.sqrt (m : ℝ))
    (by norm_num) (by norm_num) (div_nonneg hL0 hs.le) hδ
  have hX : 2 * Real.sqrt |2 * tailL N * Real.sqrt (m : ℝ)|
      = tailC N * (m : ℝ) ^ (1 / 4 : ℝ) := by
    rw [abs_of_nonneg hξ0, tailC, ← sqrt_sqrt_eq_rpow (m : ℝ) hm0.le,
      Real.sqrt_mul (by linarith : (0 : ℝ) ≤ 2 * tailL N)]
    ring
  rw [hX] at hF1 hF2
  have hK := dualK_sum_le N s m hN hs hsN hm
  have hK1 := dualK1_nonneg s m
  have hK2 := dualK2_nonneg s m
  have hd0 : (0 : ℝ) ≤ d m := Nat.cast_nonneg _
  have step1 : (d m : ℝ) * (dualK1 s m * ‖Fs (3 / 2) (tailL N / s) (2 * tailL N * Real.sqrt (m : ℝ))‖
        + dualK2 s m * ‖Fs (1 / 2) (tailL N / s) (2 * tailL N * Real.sqrt (m : ℝ))‖)
      ≤ (d m : ℝ) * (dualK1 s m * (2 ^ 16 * Real.exp (-(tailC N * (m : ℝ) ^ (1 / 4 : ℝ))))
        + dualK2 s m * (2 ^ 16 * Real.exp (-(tailC N * (m : ℝ) ^ (1 / 4 : ℝ))))) :=
    mul_le_mul_of_nonneg_left
      (add_le_add (mul_le_mul_of_nonneg_left hF1 hK1) (mul_le_mul_of_nonneg_left hF2 hK2)) hd0
  refine h0.trans (step1.trans ?_)
  calc (d m : ℝ) * (dualK1 s m * (2 ^ 16 * Real.exp (-(tailC N * (m : ℝ) ^ (1 / 4 : ℝ))))
        + dualK2 s m * (2 ^ 16 * Real.exp (-(tailC N * (m : ℝ) ^ (1 / 4 : ℝ)))))
      = (d m : ℝ) * ((dualK1 s m + dualK2 s m)
          * (2 ^ 16 * Real.exp (-(tailC N * (m : ℝ) ^ (1 / 4 : ℝ))))) := by ring
    _ ≤ (d m : ℝ) * ((N ^ (3 / 4 : ℝ) * (m : ℝ) ^ (-(3 / 4 : ℝ)))
          * (2 ^ 16 * Real.exp (-(tailC N * (m : ℝ) ^ (1 / 4 : ℝ))))) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right hK (by positivity)) hd0
    _ = _ := by ring

end DirichletDivisor

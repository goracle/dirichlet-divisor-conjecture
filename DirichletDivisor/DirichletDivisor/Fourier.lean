import DirichletDivisor.Displacement

set_option linter.style.whitespace false

open MeasureTheory

namespace DirichletDivisor

/-- F_{s,δ}(ξ) = ∫_{-1}^{1} η(t) (1+δt)^s e(ξ t) dt, with e(u) = exp(2πiu). -/
noncomputable def Fs (s δ ξ : ℝ) : ℂ :=
  ∫ t in (-1:ℝ)..1, ((eta t * (1 + δ * t) ^ s : ℝ) : ℂ) *
    Complex.exp (((2 * Real.pi * ξ * t : ℝ) : ℂ) * Complex.I)

/-- Trivial bound: ‖F_{s,δ}(ξ)‖ ≤ (3/2)^s for s ≥ 0, 0 ≤ δ ≤ 1/2. -/
theorem Fs_norm_le (s δ ξ : ℝ) (hs : 0 ≤ s) (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2) :
    ‖Fs s δ ξ‖ ≤ (3 / 2 : ℝ) ^ s := by
  have h1 : Continuous fun t : ℝ => eta t * (1 + δ * t) ^ s :=
    eta_continuous.mul
      ((continuous_const.add (continuous_const.mul continuous_id)).rpow_const
        (fun _ => Or.inr hs))
  have hc : Continuous fun t : ℝ => ((eta t * (1 + δ * t) ^ s : ℝ) : ℂ) *
      Complex.exp (((2 * Real.pi * ξ * t : ℝ) : ℂ) * Complex.I) :=
    (Complex.continuous_ofReal.comp h1).mul
      (Complex.continuous_exp.comp
        ((Complex.continuous_ofReal.comp (by fun_prop)).mul continuous_const))
  unfold Fs
  calc ‖∫ t in (-1:ℝ)..1, ((eta t * (1 + δ * t) ^ s : ℝ) : ℂ) *
          Complex.exp (((2 * Real.pi * ξ * t : ℝ) : ℂ) * Complex.I)‖
      ≤ ∫ t in (-1:ℝ)..1, ‖((eta t * (1 + δ * t) ^ s : ℝ) : ℂ) *
          Complex.exp (((2 * Real.pi * ξ * t : ℝ) : ℂ) * Complex.I)‖ :=
        intervalIntegral.norm_integral_le_integral_norm (by norm_num)
    _ ≤ ∫ t in (-1:ℝ)..1, eta t * (3 / 2 : ℝ) ^ s := by
        refine intervalIntegral.integral_mono_on (by norm_num)
          (hc.norm.intervalIntegrable _ _)
          ((eta_continuous.mul continuous_const).intervalIntegrable _ _) ?_
        intro t ht
        have hlo : 0 ≤ 1 + δ * t := by
          nlinarith [mul_nonneg hδ0 (by linarith [ht.1] : 0 ≤ t + 1)]
        have hhi : 1 + δ * t ≤ 3 / 2 := by
          nlinarith [mul_nonneg hδ0 (by linarith [ht.2] : 0 ≤ 1 - t)]
        have hn : ‖((eta t * (1 + δ * t) ^ s : ℝ) : ℂ) *
            Complex.exp (((2 * Real.pi * ξ * t : ℝ) : ℂ) * Complex.I)‖
            = |eta t * (1 + δ * t) ^ s| := by
          rw [norm_mul, Complex.norm_exp_ofReal_mul_I, mul_one, Complex.norm_real,
            Real.norm_eq_abs]
        rw [hn, abs_of_nonneg (mul_nonneg (eta_nonneg t) (Real.rpow_nonneg hlo _))]
        exact mul_le_mul_of_nonneg_left (Real.rpow_le_rpow hlo hhi hs) (eta_nonneg t)
    _ = (3 / 2 : ℝ) ^ s := by
        rw [intervalIntegral.integral_mul_const, eta_integral, one_mul]

/-- (3/2)^(3/2) < 2. -/
theorem three_halves_rpow_lt_two : (3 / 2 : ℝ) ^ (3 / 2 : ℝ) < 2 := by
  have hnn : 0 ≤ (3 / 2 : ℝ) ^ (3 / 2 : ℝ) := Real.rpow_nonneg (by norm_num) _
  have hsq : ((3 / 2 : ℝ) ^ (3 / 2 : ℝ)) ^ (2 : ℝ) = (3 / 2 : ℝ) ^ (3 : ℝ) := by
    rw [← Real.rpow_mul (by norm_num)]
    norm_num
  have h27 : ((3 / 2 : ℝ) ^ (3 / 2 : ℝ)) ^ 2 = 27 / 8 := by
    have : ((3 / 2 : ℝ) ^ (3 / 2 : ℝ)) ^ 2 = ((3 / 2 : ℝ) ^ (3 / 2 : ℝ)) ^ (2 : ℝ) := by
      rw [Real.rpow_two]
    rw [this, hsq, show (3 : ℝ) = ((3 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    norm_num
  generalize (3 / 2 : ℝ) ^ (3 / 2 : ℝ) = x at hnn h27 ⊢
  by_contra h
  push_neg at h
  nlinarith 



/-- For 0 ≤ s ≤ 3/2 and 0 ≤ δ ≤ 1/2, ‖F_{s,δ}(ξ)‖ < 2 for every real ξ. -/
theorem Fs_norm_lt_two (s δ ξ : ℝ) (hs : 0 ≤ s) (hs' : s ≤ 3 / 2)
    (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2) : ‖Fs s δ ξ‖ < 2 :=
  (Fs_norm_le s δ ξ hs hδ0 hδ).trans_lt
    ((Real.rpow_le_rpow_of_exponent_le (by norm_num) hs').trans_lt
      three_halves_rpow_lt_two)

end DirichletDivisor

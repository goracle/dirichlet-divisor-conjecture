import DirichletDivisor.Bluestein

set_option linter.style.whitespace false

namespace DirichletDivisor

theorem sqrt_half_ge (N : ℝ) (hN : 0 ≤ N) :
    (7 / 10) * Real.sqrt N ≤ Real.sqrt (N / 2) := by
  have hr := Real.sqrt_nonneg N
  have hs := Real.sqrt_nonneg (N / 2)
  have h1 := Real.sq_sqrt hN
  have h2 := Real.sq_sqrt (show (0 : ℝ) ≤ N / 2 by linarith)
  by_contra h
  push_neg at h
  have h3 := mul_self_lt_mul_self hs h
  nlinarith [h1, h2, h3]

/-- The K₂ contribution is tiny for N ≥ 2^65536. -/
theorem k2_small (N : ℝ) (hN : (2 : ℝ) ^ 65536 ≤ N) :
    2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * Real.sqrt (N / 2))) ≤ 1 / 100 := by
  have h20 : (2 : ℝ) ^ 20 ≤ 2 ^ 65536 := pow_le_pow_right₀ (by norm_num) (by norm_num)
  have hN6 : (1000000 : ℝ) ≤ N := by
    have : (2 : ℝ) ^ 20 = 1048576 := by norm_num
    linarith
  have hN1 : (1 : ℝ) ≤ N := by linarith
  have hN0 : 0 ≤ N := by linarith
  have hr0 : 0 ≤ Real.sqrt N := Real.sqrt_nonneg N
  have hrr : Real.sqrt N ^ 2 = N := Real.sq_sqrt hN0
  have hs := sqrt_half_ge N hN0
  have hs0 : 0 ≤ Real.sqrt (N / 2) := Real.sqrt_nonneg _
  generalize Real.sqrt N = r at hr0 hrr hs
  generalize Real.sqrt (N / 2) = s at hs0 hs ⊢
  have hr1000 : 1000 ≤ r := by
    by_contra h
    push_neg at h
    nlinarith [mul_self_lt_mul_self hr0 h]
  have hpow : N ^ (3 / 4 : ℝ) ≤ N := by
    have := Real.rpow_le_rpow_of_exponent_le hN1 (show (3 / 4 : ℝ) ≤ 1 by norm_num)
    simpa using this
  have hpis : 3.14 * ((7 / 10) * r) ≤ Real.pi * s :=
    mul_le_mul Real.pi_gt_d2.le hs (by positivity) Real.pi_pos.le
  have hexp : Real.exp (-(2 * Real.pi * s)) ≤ Real.exp (-(4 * r)) := by
    apply Real.exp_le_exp.2
    nlinarith [hpis, hr0]
  have he1 : r + 1 ≤ Real.exp r := Real.add_one_le_exp r
  have he4 : Real.exp (4 * r) = Real.exp r ^ 4 := by
    rw [← Real.exp_nat_mul]
    norm_num
  have hr4 : r ^ 4 ≤ Real.exp (4 * r) := by
    rw [he4]
    exact pow_le_pow_left₀ hr0 (by linarith) 4
  have hq0 : 0 < Real.exp (-(4 * r)) := Real.exp_pos _
  have hqe : Real.exp (-(4 * r)) * Real.exp (4 * r) = 1 := by
    rw [← Real.exp_add]
    simp
  generalize Real.exp (-(4 * r)) = q at hq0 hqe hexp
  have hq4 : q * r ^ 4 ≤ 1 := by
    calc q * r ^ 4 ≤ q * Real.exp (4 * r) := mul_le_mul_of_nonneg_left hr4 hq0.le
      _ = 1 := hqe
  have hr2 : (1000000 : ℝ) ≤ r ^ 2 := by nlinarith
  have hr2pos : 0 < r ^ 2 := by positivity
  have hbound : 2 * r ^ 2 * q ≤ 1 / 100 := by
    by_contra h
    push_neg at h
    have := mul_lt_mul_of_pos_right h hr2pos
    nlinarith [this, hq4, hr2]
  have hE0 : 0 < Real.exp (-(2 * Real.pi * s)) := Real.exp_pos _
  have hN3 : 0 ≤ N ^ (3 / 4 : ℝ) := Real.rpow_nonneg hN0 _
  calc 2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * s))
      ≤ 2 * N * q := by
        apply mul_le_mul (by linarith) hexp hE0.le (by linarith)
    _ = 2 * r ^ 2 * q := by rw [hrr]
    _ ≤ 1 / 100 := hbound

/-- The non-Hankel polynomial error 2^60 N^{-4} is tiny. -/
theorem non_hankel_small (N : ℝ) (hN : (2 : ℝ) ^ 65536 ≤ N) :
    (2 : ℝ) ^ 60 * (N ^ 4)⁻¹ ≤ 1 / 100 := by
  have h120 : (2 : ℝ) ^ 120 ≤ 2 ^ 65536 := pow_le_pow_right₀ (by norm_num) (by norm_num)
  have hbig : (100 : ℝ) * 2 ^ 60 ≤ 2 ^ 120 := by norm_num
  have hN1 : (1 : ℝ) ≤ N := by
    have : (1 : ℝ) ≤ 2 ^ 120 := by norm_num
    linarith
  have hN2 : N ≤ N ^ 2 := by nlinarith
  have h2 : (1 : ℝ) ≤ N ^ 2 := by nlinarith
  have hN4 : N ^ 2 ≤ N ^ 4 := by nlinarith [h2]
  have hpos : 0 < N ^ 4 := by positivity
  have hfin : (100 : ℝ) * 2 ^ 60 ≤ N ^ 4 := by linarith
  rw [← div_eq_mul_inv]
  first
    | rw [div_le_iff₀ hpos]
    | rw [div_le_iff hpos]
  nlinarith [hfin]

/-- Eq. (33): the closing inequality of §6, from the three component bounds. -/
theorem ledger_lt_quarter (N Ehank Ek2 Enon Err : ℝ)
    (hN : (2 : ℝ) ^ 65536 ≤ N)
    (hsum : Err ≤ Ehank + Ek2 + Enon)
    (hH : Ehank ≤ 4.8e-4 * N ^ (-(1 / 4) : ℝ))
    (hK : Ek2 ≤ 2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * Real.sqrt (N / 2))))
    (hn : Enon ≤ (2 : ℝ) ^ 60 * (N ^ 4)⁻¹) :
    Err < 1 / 4 := by
  have hN1 : (1 : ℝ) ≤ N := by
    have h : (1 : ℝ) ≤ 2 ^ 65536 := one_le_pow₀ (by norm_num)
    linarith
  have hN14 : N ^ (-(1 / 4) : ℝ) ≤ 1 :=
    Real.rpow_le_one_of_one_le_of_nonpos hN1 (by norm_num)
  have h1 : Ehank ≤ 4.8e-4 := by nlinarith [hH, hN14]
  have h2 := k2_small N hN
  have h3 := non_hankel_small N hN
  linarith

/-! Exponent bookkeeping behind (21), (23), and the interpolation bound. -/

theorem table_exponent (A : ℝ) : 9 / 28 + 1 + alpha * (A + 4) - PTf A = -(A + 5) := by
  unfold PTf
  ring

theorem coeff_exponent (A : ℝ) : 5 / 14 + beta * (A + 4) - Phf A = -(A + 5) := by
  unfold Phf
  ring

theorem interp_exponent (A : ℝ) : -A - 3 + 5 / 14 ≤ -A - 2 := by
  linarith

end DirichletDivisor

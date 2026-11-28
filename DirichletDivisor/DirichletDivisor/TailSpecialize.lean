import DirichletDivisor.TailParam

set_option linter.style.whitespace false

namespace DirichletDivisor

/-! ## Specializing the dual tail to the note's parameters

With `H = N^{9/28}`, `L = H / (4 √N)`, `c = 2 √(2 L)` and `M₀ = 1024 N^{5/14} (ln N)^4`
one has `c · M₀^{1/4} = 8 ln N`, so `e^{-c M^{1/4}} ≤ N^{-8}` for `M ≥ M₀`.
Here `tail_weighted_exp` is applied with `y = c M^{1/4}`.  Note `y M^{3/4} = c M`, and
`c ≤ 2`, which gives `N^{-193/28}` (up to `(ln N)^4`), then `≤ N^{-6}`.
-/

/-- `L = H / (4 √N)` with `H = N^{9/28}`. -/
noncomputable def tailL (N : ℝ) : ℝ := N ^ (9 / 28 : ℝ) / (4 * Real.sqrt N)

/-- `c = 2 √(2 L)`, so that `e^{-c m^{1/4}} = e^{-2 √ξ}` with `ξ = 2 L √m`. -/
noncomputable def tailC (N : ℝ) : ℝ := 2 * Real.sqrt (2 * tailL N)

/-- The dual cutoff `M₀ = 1024 N^{5/14} (ln N)^4`. -/
noncomputable def tailM0 (N : ℝ) : ℝ := 1024 * N ^ (5 / 14 : ℝ) * Real.log N ^ 4

theorem tailL_pos (N : ℝ) (hN : 0 < N) : 0 < tailL N := by
  unfold tailL
  positivity

theorem tailC_pos (N : ℝ) (hN : 0 < N) : 0 < tailC N := by
  have := tailL_pos N hN
  unfold tailC
  positivity

theorem tailL_sq (N : ℝ) (hN : 0 < N) : tailL N ^ 2 * (16 * N) = N ^ (9 / 14 : ℝ) := by
  have ha : (N ^ (9 / 28 : ℝ)) ^ 2 = N ^ (9 / 14 : ℝ) := by
    rw [← Real.rpow_natCast (N ^ (9 / 28 : ℝ)) 2, ← Real.rpow_mul hN.le]
    norm_num
  have hs : Real.sqrt N ^ 2 = N := Real.sq_sqrt hN.le
  unfold tailL
  rw [div_pow, mul_pow, hs, ha, div_mul_eq_mul_div, div_eq_iff (by positivity)]
  ring

theorem tailL_le (N : ℝ) (hN1 : 1 ≤ N) : tailL N ≤ 1 / 4 := by
  have hN0 : 0 < N := by linarith
  have h := tailL_sq N hN0
  have hw : N ^ (9 / 14 : ℝ) ≤ N := by
    have := Real.rpow_le_rpow_of_exponent_le hN1 (show (9 / 14 : ℝ) ≤ 1 by norm_num)
    simpa using this
  have hL0 := (tailL_pos N hN0).le
  have h16 : tailL N ^ 2 * 16 ≤ 1 := by
    have h' : (tailL N ^ 2 * 16) * N ≤ 1 * N := by nlinarith [h, hw]
    exact le_of_mul_le_mul_right h' hN0
  nlinarith [h16, hL0]

theorem tailC_le_two (N : ℝ) (hN1 : 1 ≤ N) : tailC N ≤ 2 := by
  have hL := tailL_le N hN1
  have hs : Real.sqrt (2 * tailL N) ≤ 1 := by
    have := Real.sqrt_le_sqrt (show 2 * tailL N ≤ 1 by linarith)
    simpa using this
  unfold tailC
  linarith

theorem tailC_pow4 (N : ℝ) (hN : 0 < N) : tailC N ^ 4 = 64 * tailL N ^ 2 := by
  have hL := (tailL_pos N hN).le
  have h := Real.sq_sqrt (show 0 ≤ 2 * tailL N by linarith)
  unfold tailC
  calc (2 * Real.sqrt (2 * tailL N)) ^ 4
      = 16 * (Real.sqrt (2 * tailL N) ^ 2) ^ 2 := by ring
    _ = 16 * (2 * tailL N) ^ 2 := by rw [h]
    _ = 64 * tailL N ^ 2 := by ring

/-- The key identity: `c · M₀^{1/4} = 8 ln N`. -/
theorem tailC_root (N : ℝ) (hN1 : 1 ≤ N) :
    tailC N * tailM0 N ^ (1 / 4 : ℝ) = 8 * Real.log N := by
  have hN0 : 0 < N := by linarith
  have hℓ0 : 0 ≤ Real.log N := Real.log_nonneg hN1
  have hM0 : 0 ≤ tailM0 N := by unfold tailM0; positivity
  have hc0 : 0 ≤ tailC N := (tailC_pos N hN0).le
  have hr0 : 0 ≤ tailM0 N ^ (1 / 4 : ℝ) := Real.rpow_nonneg hM0 _
  have hr4 : (tailM0 N ^ (1 / 4 : ℝ)) ^ 4 = tailM0 N := by
    rw [← Real.rpow_natCast (tailM0 N ^ (1 / 4 : ℝ)) 4, ← Real.rpow_mul hM0]
    norm_num
  have hprod : N ^ (9 / 14 : ℝ) * N ^ (5 / 14 : ℝ) = N := by
    rw [← Real.rpow_add hN0]
    norm_num
  have h16 : 16 * tailL N ^ 2 * N ^ (5 / 14 : ℝ) = 1 := by
    apply mul_left_cancel₀ hN0.ne'
    calc N * (16 * tailL N ^ 2 * N ^ (5 / 14 : ℝ))
        = (tailL N ^ 2 * (16 * N)) * N ^ (5 / 14 : ℝ) := by ring
      _ = N ^ (9 / 14 : ℝ) * N ^ (5 / 14 : ℝ) := by rw [tailL_sq N hN0]
      _ = N := hprod
      _ = N * 1 := by ring
  have h4 : (tailC N * tailM0 N ^ (1 / 4 : ℝ)) ^ 4 = (8 * Real.log N) ^ 4 := by
    rw [mul_pow, hr4, tailC_pow4 N hN0]
    calc 64 * tailL N ^ 2 * tailM0 N
        = 4096 * Real.log N ^ 4 * (16 * tailL N ^ 2 * N ^ (5 / 14 : ℝ)) := by
          unfold tailM0
          ring
      _ = (8 * Real.log N) ^ 4 := by rw [h16]; ring
  have ha : 0 ≤ tailC N * tailM0 N ^ (1 / 4 : ℝ) := mul_nonneg hc0 hr0
  have hb : 0 ≤ 8 * Real.log N := by linarith
  -- a⁴ = b⁴ with a, b ≥ 0 gives a = b
  generalize tailC N * tailM0 N ^ (1 / 4 : ℝ) = a at ha h4 ⊢
  generalize 8 * Real.log N = b at hb h4 ⊢
  have hf : (a - b) * ((a + b) * (a ^ 2 + b ^ 2)) = 0 := by linear_combination h4
  rcases mul_eq_zero.1 hf with h | h
  · linarith
  rcases mul_eq_zero.1 h with h | h
  · linarith
  · have ha0 : a = 0 := by nlinarith [sq_nonneg a, sq_nonneg b]
    have hb0 : b = 0 := by nlinarith [sq_nonneg a, sq_nonneg b]
    rw [ha0, hb0]

/-- `N ≥ 2^65536` gives `ln N ≥ 45000`. -/
theorem log_ge_45000 (N : ℝ) (hN : (2 : ℝ) ^ 65536 ≤ N) : 45000 ≤ Real.log N := by
  have hlog : Real.log ((2 : ℝ) ^ 65536) ≤ Real.log N := Real.log_le_log (by positivity) hN
  rw [Real.log_pow] at hlog
  clear hN
  have h2 := Real.log_two_gt_d9
  push_cast at hlog
  linarith

theorem exp_two_le_eight : Real.exp 2 ≤ 8 := by
  have e1 := Real.exp_one_lt_d9
  have e2 : Real.exp 2 = Real.exp 1 * Real.exp 1 := by
    rw [← Real.exp_add]
    norm_num
  have hm := mul_self_le_mul_self (Real.exp_pos 1).le e1.le
  rw [e2]
  calc Real.exp 1 * Real.exp 1 ≤ 2.7182818286 * 2.7182818286 := hm
    _ ≤ 8 := by norm_num

/-- **Tail at the note's parameters.**  For `N ≥ 2^65536` and any natural `M` with
`M₀ ≤ M ≤ M₀ + 1` (e.g. `M = ⌈M₀⌉`), the weighted dual tail, times the Fourier constant `2^16`
(from `Fs_norm_le_exp`) and the weight prefactor `N^{3/4}`, is at most
`2^34 (ln N)^4 N^{-193/28}`. -/
theorem tail_param_bound (N : ℝ) (hN : (2 : ℝ) ^ 65536 ≤ N) (M n : ℕ)
    (hM1 : tailM0 N ≤ (M : ℝ)) (hM2 : (M : ℝ) ≤ tailM0 N + 1) :
    (2 : ℝ) ^ 16 * N ^ (3 / 4 : ℝ) *
      ∑ j ∈ Finset.range n, (d (M + j + 1) : ℝ)
        * (((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ))
            * Real.exp (-(tailC N * ((M + j + 1 : ℕ) : ℝ) ^ (1 / 4 : ℝ))))
      ≤ 2 ^ 34 * Real.log N ^ 4 * N ^ (-(193 / 28 : ℝ)) := by
  have hN0 : 0 < N := lt_of_lt_of_le (by positivity) hN
  have hℓ := log_ge_45000 N hN
  clear hN
  have hN1 : 1 ≤ N := by
    by_contra h
    have := Real.log_neg hN0 (not_le.mp h)
    linarith
  have hM0nn : 0 ≤ tailM0 N := by unfold tailM0; positivity
  have hM0one : 1 ≤ tailM0 N := by
    unfold tailM0
    have h1 : 1 ≤ N ^ (5 / 14 : ℝ) := Real.one_le_rpow hN1 (by norm_num)
    have h2 : (1 : ℝ) ≤ Real.log N ^ 4 := by
      have := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 1) (by linarith : (1 : ℝ) ≤ Real.log N) 4
      simpa using this
    have h3 : (1 : ℝ) * 1 ≤ N ^ (5 / 14 : ℝ) * Real.log N ^ 4 :=
      mul_le_mul h1 h2 (by norm_num) (by linarith)
    linarith
  have hMpos : (0 : ℝ) < M := by linarith
  have hMnat : 1 ≤ M := by
    have : (1 : ℝ) ≤ M := by linarith
    exact_mod_cast this
  have hcpos := tailC_pos N hN0
  have hc2 := tailC_le_two N hN1
  have hroot := tailC_root N hN1
  obtain ⟨y, hyd⟩ : ∃ y : ℝ, y = tailC N * (M : ℝ) ^ (1 / 4 : ℝ) := ⟨_, rfl⟩
  have hy8 : 8 * Real.log N ≤ y := by
    rw [hyd, ← hroot]
    exact mul_le_mul_of_nonneg_left (Real.rpow_le_rpow hM0nn hM1 (by norm_num)) hcpos.le
  have hy4 : 4 ≤ y := by linarith
  have hsum := tail_weighted_exp (tailC N) y hcpos M n hMnat hyd hy4
  have hyM : y * (M : ℝ) ^ (3 / 4 : ℝ) = tailC N * M := by
    rw [hyd, mul_assoc, ← Real.rpow_add hMpos]
    norm_num
  have hexp : Real.exp (-y) ≤ Real.exp (-(8 * Real.log N)) :=
    Real.exp_le_exp.2 (by linarith)
  have hexp8 : Real.exp (-(8 * Real.log N)) = N ^ (-8 : ℝ) := by
    rw [Real.rpow_def_of_pos hN0]
    congr 1
    ring
  have hexp' : Real.exp (-y) ≤ N ^ (-8 : ℝ) := hexp.trans hexp8.le
  have he2 := exp_two_le_eight
  have hM2' : (M : ℝ) ≤ 2 * tailM0 N := by linarith
  have hcM : tailC N * M ≤ 2 * (2 * tailM0 N) :=
    mul_le_mul hc2 hM2' hMpos.le (by norm_num)
  have hq : 8 * Real.exp 2 * (tailC N * M) ≤ 8 * 8 * (2 * (2 * tailM0 N)) :=
    mul_le_mul (by linarith) hcM (mul_nonneg hcpos.le hMpos.le) (by norm_num)
  have hbig : ∑ j ∈ Finset.range n, (d (M + j + 1) : ℝ)
        * (((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ))
            * Real.exp (-(tailC N * ((M + j + 1 : ℕ) : ℝ) ^ (1 / 4 : ℝ))))
      ≤ 256 * tailM0 N * N ^ (-8 : ℝ) := by
    calc _ ≤ 8 * Real.exp 2 * y * (M : ℝ) ^ (3 / 4 : ℝ) * Real.exp (-y) := hsum
      _ = 8 * Real.exp 2 * (tailC N * M) * Real.exp (-y) := by rw [← hyM]; ring
      _ ≤ 8 * 8 * (2 * (2 * tailM0 N)) * N ^ (-8 : ℝ) :=
          mul_le_mul hq hexp' (Real.exp_pos _).le (by linarith)
      _ = 256 * tailM0 N * N ^ (-8 : ℝ) := by ring
  have hcomb : N ^ (3 / 4 : ℝ) * N ^ (5 / 14 : ℝ) * N ^ (-8 : ℝ) = N ^ (-(193 / 28 : ℝ)) := by
    rw [← Real.rpow_add hN0, ← Real.rpow_add hN0]
    norm_num
  refine le_trans (mul_le_mul_of_nonneg_left hbig (by positivity)) (le_of_eq ?_)
  calc (2 : ℝ) ^ 16 * N ^ (3 / 4 : ℝ) * (256 * tailM0 N * N ^ (-8 : ℝ))
      = 2 ^ 34 * Real.log N ^ 4 * (N ^ (3 / 4 : ℝ) * N ^ (5 / 14 : ℝ) * N ^ (-8 : ℝ)) := by
        unfold tailM0
        ring
    _ = 2 ^ 34 * Real.log N ^ 4 * N ^ (-(193 / 28 : ℝ)) := by rw [hcomb]

/-- Absorbing the `2^34 (ln N)^4`: the tail is at most `N^{-6}` for `N ≥ 2^65536`. -/
theorem tail_param_le (N : ℝ) (hN : (2 : ℝ) ^ 65536 ≤ N) (M n : ℕ)
    (hM1 : tailM0 N ≤ (M : ℝ)) (hM2 : (M : ℝ) ≤ tailM0 N + 1) :
    (2 : ℝ) ^ 16 * N ^ (3 / 4 : ℝ) *
      ∑ j ∈ Finset.range n, (d (M + j + 1) : ℝ)
        * (((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ))
            * Real.exp (-(tailC N * ((M + j + 1 : ℕ) : ℝ) ^ (1 / 4 : ℝ))))
      ≤ N ^ (-6 : ℝ) := by
  refine (tail_param_bound N hN M n hM1 hM2).trans ?_
  have hN0 : 0 < N := lt_of_lt_of_le (by positivity) hN
  have hu := log_ge_45000 N hN
  clear hN
  have hx : (2 : ℝ) ^ 34 * Real.log N ^ 4 ≤ Real.exp (25 / 28 * Real.log N) := by
    have h8 := Real.pow_div_factorial_le_exp (x := 25 / 28 * Real.log N) (by linarith) 8
    have hf : ((Nat.factorial 8 : ℕ) : ℝ) = 40320 := by norm_num [Nat.factorial]
    rw [hf] at h8
    have hv : (45000 : ℝ) ^ 4 ≤ Real.log N ^ 4 := pow_le_pow_left₀ (by norm_num) hu 4
    have key : (2 : ℝ) ^ 34 ≤ (25 / 28) ^ 8 * Real.log N ^ 4 / 40320 := by
      have h1 := mul_le_mul_of_nonneg_left hv (show (0 : ℝ) ≤ (25 / 28) ^ 8 by norm_num)
      have hn : (2 : ℝ) ^ 34 ≤ (25 / 28) ^ 8 * 45000 ^ 4 / 40320 := by norm_num
      linarith
    calc (2 : ℝ) ^ 34 * Real.log N ^ 4
        ≤ ((25 / 28) ^ 8 * Real.log N ^ 4 / 40320) * Real.log N ^ 4 :=
          mul_le_mul_of_nonneg_right key (by positivity)
      _ = (25 / 28 * Real.log N) ^ 8 / 40320 := by ring
      _ ≤ Real.exp (25 / 28 * Real.log N) := h8
  calc (2 : ℝ) ^ 34 * Real.log N ^ 4 * N ^ (-(193 / 28 : ℝ))
      ≤ Real.exp (25 / 28 * Real.log N) * N ^ (-(193 / 28 : ℝ)) :=
        mul_le_mul_of_nonneg_right hx (Real.rpow_nonneg hN0.le _)
    _ = N ^ (-6 : ℝ) := by
        rw [Real.rpow_def_of_pos hN0, Real.rpow_def_of_pos hN0, ← Real.exp_add]
        congr 1
        ring

end DirichletDivisor

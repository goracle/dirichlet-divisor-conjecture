import DirichletDivisor.DualTruncate

set_option linter.style.whitespace false

namespace DirichletDivisor

open MeasureTheory

/-! # The oracle interface and the assembly of `hdual`

1. `dual_term_eq`: the exact identity
   `∫ η·mainTerm_m = d(m)(K₁·Im(e^{iθ}F_{3/2}(δ,ξ)) + K₂·Re(e^{iθ}F_{1/2}(δ,ξ)))`.
2. `DualOracle`: the numerical algorithm (quadrature for `F_s`, Bluestein, Chebyshev, phases) is
   modelled by per-term computed values `comp j` with error bounds `err j` against the closed
   form. This is a hypothesis structure, like `CitedInputs`.
3. `hdual_of_oracle`: `|Σ_{j<M} comp j - ∫ η·dualMain| ≤ N^{-6} + Σ_{j<M} err j`.
4. `hoyingSum_eq_round_oracle`: final theorem at `H = N^{9/28}`.
-/

/-! ## 1. Exact identity -/

/-- Phase `θ_m = 4π√m·s - 5π/4`. -/
noncomputable def dualPhase (s : ℝ) (m : ℕ) : ℝ :=
  4 * Real.pi * Real.sqrt (m : ℝ) * s - 5 * Real.pi / 4

/-- Frequency `ξ_m = 2L√m`. -/
noncomputable def dualFreq (L : ℝ) (m : ℕ) : ℝ := 2 * L * Real.sqrt (m : ℝ)

/-- Closed form of the `m`-th smoothed dual term. -/
noncomputable def dualTermExact (s L : ℝ) (m : ℕ) : ℝ :=
  (d m : ℝ) *
    (dualK1 s m * (Complex.exp ((dualPhase s m : ℂ) * Complex.I)
          * Fs (3 / 2) (L / s) (dualFreq L m)).im
      + dualK2 s m * (Complex.exp ((dualPhase s m : ℂ) * Complex.I)
          * Fs (1 / 2) (L / s) (dualFreq L m)).re)

theorem dual_term_eq (s L : ℝ) (m : ℕ) (hm : 1 ≤ m) (hs : 0 < s) (hL0 : 0 ≤ L)
    (hLs : L < s) :
    (∫ u in (-1 : ℝ)..1, eta u * mainTerm ((s + L * u) ^ 2) m) = dualTermExact s L m := by
  have hint : (∫ u in (-1 : ℝ)..1, eta u * mainTerm ((s + L * u) ^ 2) m)
      = ∫ u in (-1 : ℝ)..1, (d m : ℝ) *
          (dualK1 s m * (eta u * (1 + (L / s) * u) ^ (3 / 2 : ℝ)
              * Real.sin (dualPhase s m + 2 * Real.pi * dualFreq L m * u))
            + dualK2 s m * (eta u * (1 + (L / s) * u) ^ (1 / 2 : ℝ)
              * Real.cos (dualPhase s m + 2 * Real.pi * dualFreq L m * u))) := by
    refine intervalIntegral.integral_congr (fun u hu => ?_)
    rw [Set.uIcc_of_le (by norm_num : (-1 : ℝ) ≤ 1)] at hu
    exact eta_mainTerm_pt s L u m hm hs hL0 hLs hu (dualPhase s m) (dualFreq L m) rfl rfl
  have c1 := continuous_eta_pow_sin (3 / 2) (L / s) (dualPhase s m) (dualFreq L m)
    (by norm_num)
  have c2 := continuous_eta_pow_cos (1 / 2) (L / s) (dualPhase s m) (dualFreq L m)
    (by norm_num)
  have i1 : IntervalIntegrable (fun u : ℝ => dualK1 s m * (eta u * (1 + (L / s) * u) ^ (3 / 2 : ℝ)
      * Real.sin (dualPhase s m + 2 * Real.pi * dualFreq L m * u))) volume (-1) 1 :=
    (continuous_const.mul c1).intervalIntegrable _ _
  have i2 : IntervalIntegrable (fun u : ℝ => dualK2 s m * (eta u * (1 + (L / s) * u) ^ (1 / 2 : ℝ)
      * Real.cos (dualPhase s m + 2 * Real.pi * dualFreq L m * u))) volume (-1) 1 :=
    (continuous_const.mul c2).intervalIntegrable _ _
  rw [hint, intervalIntegral.integral_const_mul, intervalIntegral.integral_add i1 i2,
    intervalIntegral.integral_const_mul, intervalIntegral.integral_const_mul,
    eta_pow_sin_integral (3 / 2) (L / s) (dualPhase s m) (dualFreq L m) (by norm_num),
    eta_pow_cos_integral (1 / 2) (L / s) (dualPhase s m) (dualFreq L m) (by norm_num)]
  all_goals rfl

/-! ## 2. The oracle interface -/

/-- **[Oracle]** Per-term computed values for the dual sum, with certified error bounds against the
closed form. Index `j` corresponds to `m = j + 1`. Models the quadrature for `F_{s,δ}`, the
Bluestein/Chebyshev block evaluation and the phase computation. Not proved here. -/
structure DualOracle (s L : ℝ) (M : ℕ) where
  comp : ℕ → ℝ
  err : ℕ → ℝ
  hcomp : ∀ j : ℕ, j < M → |comp j - dualTermExact s L (j + 1)| ≤ err j

/-! ## 3. `hdual` from the oracle -/

theorem hdual_of_oracle (N s L : ℝ) (M : ℕ) (hN : (2 : ℝ) ^ 65536 ≤ N)
    (hM1 : tailM0 N ≤ (M : ℝ)) (hM2 : (M : ℝ) ≤ tailM0 N + 1)
    (hs : 0 < s) (hsN : s ^ 2 ≤ N) (hL : L = tailL N) (hLs : tailL N < s)
    (hδ : tailL N / s ≤ 1 / 2) (O : DualOracle s L M) :
    |(∑ j ∈ Finset.range M, O.comp j)
        - ∫ u in (-1 : ℝ)..1, eta u * dualMain ((s + L * u) ^ 2)|
      ≤ N ^ (-6 : ℝ) + ∑ j ∈ Finset.range M, O.err j := by
  subst hL
  have hN0 : 0 < N := lt_of_lt_of_le (by positivity) hN
  have hL0 : 0 ≤ tailL N := (tailL_pos N hN0).le
  have ht := dual_truncation N s M hN hM1 hM2 hs hsN hLs hδ
  have he : ∑ j ∈ Finset.range M, ∫ u in (-1 : ℝ)..1, Fterm s (tailL N) j u
      = ∑ j ∈ Finset.range M, dualTermExact s (tailL N) (j + 1) :=
    Finset.sum_congr rfl (fun j _ =>
      dual_term_eq s (tailL N) (j + 1) (by omega) hs hL0 hLs)
  have h2 : |∑ j ∈ Finset.range M, O.comp j
        - ∑ j ∈ Finset.range M, dualTermExact s (tailL N) (j + 1)|
      ≤ ∑ j ∈ Finset.range M, O.err j := by
    rw [← Finset.sum_sub_distrib]
    exact (Finset.abs_sum_le_sum_abs _ _).trans
      (Finset.sum_le_sum (fun j hj => O.hcomp j (Finset.mem_range.mp hj)))
  rw [← he] at h2
  have h3 : |∑ j ∈ Finset.range M, O.comp j
        - ∫ u in (-1 : ℝ)..1, eta u * dualMain ((s + tailL N * u) ^ 2)|
      ≤ |∑ j ∈ Finset.range M, O.comp j
          - ∑ j ∈ Finset.range M, ∫ u in (-1 : ℝ)..1, Fterm s (tailL N) j u|
        + |(∫ u in (-1 : ℝ)..1, eta u * dualMain ((s + tailL N * u) ^ 2))
          - ∑ j ∈ Finset.range M, ∫ u in (-1 : ℝ)..1, Fterm s (tailL N) j u| := by
    have := abs_sub_le (∑ j ∈ Finset.range M, O.comp j)
      (∑ j ∈ Finset.range M, ∫ u in (-1 : ℝ)..1, Fterm s (tailL N) j u)
      (∫ u in (-1 : ℝ)..1, eta u * dualMain ((s + tailL N * u) ^ 2))
    rwa [abs_sub_comm (∑ j ∈ Finset.range M, ∫ u in (-1 : ℝ)..1, Fterm s (tailL N) j u)
      (∫ u in (-1 : ℝ)..1, eta u * dualMain ((s + tailL N * u) ^ 2))] at this
  exact h3.trans ((add_comm _ _).le.trans (add_le_add ht h2))

/-! ## 4. Side conditions at the note's parameters -/

theorem two_le_two_pow_65536 : (2 : ℝ) ≤ 2 ^ 65536 := by
  calc (2 : ℝ) = 2 ^ 1 := by norm_num
    _ ≤ 2 ^ 65536 := pow_le_pow_right₀ (by norm_num) (by norm_num)

theorem tail_side_conditions (N s σ : ℝ) (hN2 : 2 ≤ N) (hσ0 : 0 ≤ σ) (hσ1 : σ ≤ 1)
    (hs0 : 0 ≤ s) (hs : s ^ 2 = N - tailL N ^ 2 * σ) :
    0 < s ∧ s ^ 2 ≤ N ∧ tailL N < s ∧ tailL N / s ≤ 1 / 2 := by
  have hN1 : 1 ≤ N := by linarith
  have hN0 : 0 < N := by linarith
  have hL0 : 0 ≤ tailL N := (tailL_pos N hN0).le
  have hL4 : tailL N ≤ 1 / 4 := tailL_le N hN1
  have hL2 : tailL N ^ 2 ≤ 1 / 16 :=
    calc tailL N ^ 2 ≤ (1 / 4) ^ 2 := pow_le_pow_left₀ hL0 hL4 2
      _ = 1 / 16 := by norm_num
  have hLσ : tailL N ^ 2 * σ ≤ 1 / 16 :=
    calc tailL N ^ 2 * σ ≤ tailL N ^ 2 * 1 := mul_le_mul_of_nonneg_left hσ1 (sq_nonneg _)
      _ = tailL N ^ 2 := mul_one _
      _ ≤ 1 / 16 := hL2
  have hLσ0 : 0 ≤ tailL N ^ 2 * σ := mul_nonneg (sq_nonneg _) hσ0
  have hs2 : 1 ≤ s ^ 2 := by rw [hs]; linarith
  have hs1 : 1 ≤ s := by
    by_contra h
    have h' := not_le.mp h
    nlinarith [hs2, hs0, h']
  have hspos : 0 < s := by linarith
  refine ⟨hspos, ?_, ?_, ?_⟩
  · rw [hs]; linarith
  · linarith
  · rw [div_le_iff₀ hspos]; linarith

theorem budget_arith (q a b c : ℝ) (hq : 0 ≤ q) (ha : a ≤ q) (hb : b ≤ 2 ^ 58 * q)
    (hc : c ≤ 2 ^ 58 * q) : a + b + c ≤ 2 ^ 60 * q := by
  nlinarith [hq]

theorem parL_eq_tailL (n : ℕ) (H : ℝ) (hH : H = ((n : ℝ) + 1) ^ (9 / 28 : ℝ)) :
    parL (n + 1) H = tailL ((n : ℝ) + 1) := by
  subst hH
  first
    | simp [parL, tailL]
    | (unfold parL tailL; push_cast; rfl)

theorem rpow_neg_six_le (N : ℝ) (hN1 : 1 ≤ N) : N ^ (-6 : ℝ) ≤ (N ^ 4)⁻¹ := by
  have hN0 : 0 < N := lt_of_lt_of_le one_pos hN1
  have e : N ^ (-6 : ℝ) = (N ^ 6)⁻¹ := by
    rw [Real.rpow_neg hN0.le]
    congr 1
    exact_mod_cast Real.rpow_natCast N 6
  rw [e]
  exact inv_anti₀ (by positivity) (pow_le_pow_right₀ hN1 (by norm_num))

/-! ## 5. Final theorem -/

/-- **Correctness of rounding for the `N^{9/28}` algorithm, modulo `CitedInputs` and a `DualOracle`.**
`H = N^{9/28}`, `N = n + 1 ≥ 2^65536`, `M ∈ [tailM0 N, tailM0 N + 1]`. The oracle's total error and
the `σ̂²` error enter only through the two budget hypotheses. -/
theorem hoyingSum_eq_round_oracle (ci : CitedInputs) (n : ℕ) (H sigh ε : ℝ) (M : ℕ)
    (hH : H = ((n : ℝ) + 1) ^ (9 / 28 : ℝ))
    (hN : 2 ^ 65536 ≤ n + 1)
    (hsig0 : 0 ≤ sigh) (hsig1 : sigh ≤ 1)
    (hsigε : |etaVar - sigh| ≤ ε)
    (hM1 : tailM0 ((n : ℝ) + 1) ≤ (M : ℝ)) (hM2 : (M : ℝ) ≤ tailM0 ((n : ℝ) + 1) + 1)
    (O : DualOracle (parS (n + 1) H sigh) (parL (n + 1) H) M)
    (hεsum : ∑ j ∈ Finset.range M, O.err j ≤ 2 ^ 58 * (((n : ℝ) + 1) ^ 4)⁻¹)
    (hσbud : H ^ 2 * ε * (1 + Real.log (2 * ((n : ℝ) + 1))) / 8
      ≤ 2 ^ 58 * (((n : ℝ) + 1) ^ 4)⁻¹) :
    (hoyingSum n : ℤ)
      = round (idealAhat (parS (n + 1) H sigh) (parL (n + 1) H) (parK (n + 1) H sigh)
          (∑ j ∈ Finset.range M, O.comp j)) := by
  have hN' : (2 : ℝ) ^ 65536 ≤ (n : ℝ) + 1 := by exact_mod_cast hN
  have hN2 : (2 : ℝ) ≤ (n : ℝ) + 1 := le_trans two_le_two_pow_65536 hN'
  have hN1 : (1 : ℝ) ≤ (n : ℝ) + 1 := le_trans (by norm_num) hN2
  have hN0 : (0 : ℝ) < (n : ℝ) + 1 := lt_of_lt_of_le one_pos hN1
  have hH0 : 0 ≤ H := by rw [hH]; exact Real.rpow_nonneg hN0.le _
  have hHN : H ≤ (n : ℝ) + 1 := by
    rw [hH]
    calc ((n : ℝ) + 1) ^ (9 / 28 : ℝ) ≤ ((n : ℝ) + 1) ^ (1 : ℝ) :=
          Real.rpow_le_rpow_of_exponent_le hN1 (by norm_num)
      _ = (n : ℝ) + 1 := Real.rpow_one _
  have hNpos : (0 : ℝ) < ((n + 1 : ℕ) : ℝ) := by positivity
  have hHN' : H ≤ ((n + 1 : ℕ) : ℝ) := by push_cast; exact hHN
  have hs := parS_sq (n + 1) H sigh hNpos hH0 hHN' hsig0 hsig1
  push_cast at hs
  rw [parL_eq_tailL n H hH] at hs
  obtain ⟨hs_pos, hsN, hLs, hδ⟩ := tail_side_conditions ((n : ℝ) + 1)
    (parS (n + 1) H sigh) sigh hN2 hsig0 hsig1 (Real.sqrt_nonneg _) hs
  have hdual := hdual_of_oracle ((n : ℝ) + 1) (parS (n + 1) H sigh) (parL (n + 1) H) M hN'
    hM1 hM2 hs_pos hsN (parL_eq_tailL n H hH) hLs hδ O
  have hq : 0 ≤ (((n : ℝ) + 1) ^ 4)⁻¹ := by positivity
  have h6 := rpow_neg_six_le ((n : ℝ) + 1) hN1
  have hn := budget_arith (((n : ℝ) + 1) ^ 4)⁻¹ (((n : ℝ) + 1) ^ (-6 : ℝ))
    (∑ j ∈ Finset.range M, O.err j)
    (H ^ 2 * ε * (1 + Real.log (2 * ((n : ℝ) + 1))) / 8) hq h6 hεsum hσbud
  exact hoyingSum_eq_round_ideal' ci n H sigh ε
    (((n : ℝ) + 1) ^ (-6 : ℝ) + ∑ j ∈ Finset.range M, O.err j)
    (∑ j ∈ Finset.range M, O.comp j)
    ((((n : ℝ) + 1) ^ (-6 : ℝ) + ∑ j ∈ Finset.range M, O.err j)
      + H ^ 2 * ε * (1 + Real.log (2 * ((n : ℝ) + 1))) / 8)
    hN hH0 hHN hsig0 hsig1 hsigε hdual le_rfl hn

end DirichletDivisor

import DirichletDivisor.Assembly

set_option linter.style.whitespace false

namespace DirichletDivisor

/-- Smoothing half-width L = H / (4 √N). -/
noncomputable def parL (N : ℕ) (H : ℝ) : ℝ := H / (4 * Real.sqrt N)

/-- Curve parameter s = √(N - L² σ̂²). -/
noncomputable def parS (N : ℕ) (H sigh : ℝ) : ℝ :=
  Real.sqrt ((N : ℝ) - parL N H ^ 2 * sigh)

/-- Correction-window cutoff K = ⌈(s+L)²⌉. -/
noncomputable def parK (N : ℕ) (H sigh : ℝ) : ℕ :=
  ⌈(parS N H sigh + parL N H) ^ 2⌉₊

theorem parL_nonneg (N : ℕ) (H : ℝ) (hH0 : 0 ≤ H) : 0 ≤ parL N H :=
  div_nonneg hH0 (by positivity)

theorem parL_sq (N : ℕ) (H : ℝ) (hN : (0 : ℝ) < N) :
    parL N H ^ 2 = H ^ 2 / (16 * N) := by
  unfold parL
  rw [div_pow, mul_pow, Real.sq_sqrt hN.le]
  norm_num

theorem parL_sq_le (N : ℕ) (H : ℝ) (hN : (0 : ℝ) < N) (hH0 : 0 ≤ H)
    (hHN : H ≤ N) : parL N H ^ 2 ≤ (N : ℝ) / 16 := by
  have hmul : parL N H ^ 2 * (16 * N) = H ^ 2 := by
    rw [parL_sq N H hN]
    field_simp
  have hHH : H * H ≤ (N : ℝ) * N := mul_self_le_mul_self hH0 hHN
  by_contra h
  push_neg at h
  have h2 := mul_lt_mul_of_pos_right h hN
  nlinarith [hmul, hHH, h2]

theorem parS_sq (N : ℕ) (H sigh : ℝ) (hN : (0 : ℝ) < N) (hH0 : 0 ≤ H)
    (hHN : H ≤ N) (hsig0 : 0 ≤ sigh) (hsig1 : sigh ≤ 1) :
    parS N H sigh ^ 2 = (N : ℝ) - parL N H ^ 2 * sigh := by
  have hL := parL_sq_le N H hN hH0 hHN
  have h1 : parL N H ^ 2 * sigh ≤ parL N H ^ 2 * 1 :=
    mul_le_mul_of_nonneg_left hsig1 (sq_nonneg _)
  unfold parS
  apply Real.sq_sqrt
  nlinarith [h1, hL]

theorem parL_le_parS (N : ℕ) (H sigh : ℝ) (hN : (0 : ℝ) < N) (hH0 : 0 ≤ H)
    (hHN : H ≤ N) (hsig0 : 0 ≤ sigh) (hsig1 : sigh ≤ 1) :
    parL N H ≤ parS N H sigh := by
  have hL := parL_sq_le N H hN hH0 hHN
  have hs2 := parS_sq N H sigh hN hH0 hHN hsig0 hsig1
  have hs0 : 0 ≤ parS N H sigh := Real.sqrt_nonneg _
  have h1 : parL N H ^ 2 * sigh ≤ parL N H ^ 2 * 1 :=
    mul_le_mul_of_nonneg_left hsig1 (sq_nonneg _)
  by_contra h
  push_neg at h
  have h3 := mul_self_lt_mul_self hs0 h
  nlinarith [h3, hs2, hL, h1]

theorem par_bar (N : ℕ) (H sigh : ℝ) (hN : (0 : ℝ) < N) (hH0 : 0 ≤ H)
    (hHN : H ≤ N) (hsig0 : 0 ≤ sigh) (hsig1 : sigh ≤ 1) :
    parS N H sigh ^ 2 + parL N H ^ 2 * etaVar ≤ 2 * N := by
  have hL := parL_sq_le N H hN hH0 hHN
  have hs2 := parS_sq N H sigh hN hH0 hHN hsig0 hsig1
  have h1 : parL N H ^ 2 * etaVar ≤ parL N H ^ 2 * 1 :=
    mul_le_mul_of_nonneg_left etaVar_le_one (sq_nonneg _)
  have h2 : 0 ≤ parL N H ^ 2 * sigh := mul_nonneg (sq_nonneg _) hsig0
  nlinarith [hL, hs2, h1, h2]

theorem par_K (N : ℕ) (H sigh : ℝ) :
    (parS N H sigh + parL N H) ^ 2 ≤ (parK N H sigh : ℝ) :=
  Nat.le_ceil _

/-- Eq. (25) in closed form. -/
theorem par_displacement (N : ℕ) (H sigh ε : ℝ) (hN : (0 : ℝ) < N)
    (hN1 : (1 : ℝ) ≤ N) (hε : |etaVar - sigh| ≤ ε) :
    parL N H ^ 2 * |etaVar - sigh| * (2 * N * (1 + Real.log (2 * N)))
      ≤ H ^ 2 * ε * (1 + Real.log (2 * N)) / 8 := by
  have hlog : 0 ≤ Real.log (2 * N) := Real.log_nonneg (by linarith)
  have hX : 0 ≤ 2 * (N : ℝ) * (1 + Real.log (2 * N)) := by positivity
  calc parL N H ^ 2 * |etaVar - sigh| * (2 * N * (1 + Real.log (2 * N)))
      ≤ parL N H ^ 2 * ε * (2 * N * (1 + Real.log (2 * N))) :=
        mul_le_mul_of_nonneg_right
          (mul_le_mul_of_nonneg_left hε (sq_nonneg _)) hX
    _ = H ^ 2 * ε * (1 + Real.log (2 * N)) / 8 := by
        rw [parL_sq N H hN]
        field_simp
        ring

/-- The final theorem with the explicit parameter choice: the remaining inputs are
`hε` (algorithm vs smoothed approximation), the numerical σ̂² accuracy, and the
three ledger component bounds. -/
theorem hoyingSum_eq_round_param (n : ℕ)
    (H sigh ε Ahat Ehank Ek2 Enon1 Enon : ℝ)
    (hN : 2 ^ 65536 ≤ n + 1)
    (hH0 : 0 ≤ H) (hHN : H ≤ (n : ℝ) + 1)
    (hsig0 : 0 ≤ sigh) (hsig1 : sigh ≤ 1)
    (hsigε : |etaVar - sigh| ≤ ε)
    (hε : |Ahat - smoothedApprox (parS (n + 1) H sigh) (parL (n + 1) H)
            (parK (n + 1) H sigh)| ≤ Ehank + Ek2 + Enon1)
    (hE : Enon1 + H ^ 2 * ε * (1 + Real.log (2 * ((n : ℝ) + 1))) / 8 ≤ Enon)
    (hH : Ehank ≤ 4.8e-4 * ((n : ℝ) + 1) ^ (-(1 / 4) : ℝ))
    (hKb : Ek2 ≤ 2 * ((n : ℝ) + 1) ^ (3 / 4 : ℝ) *
      Real.exp (-(2 * Real.pi * Real.sqrt (((n : ℝ) + 1) / 2))))
    (hn : Enon ≤ (2 : ℝ) ^ 60 * (((n : ℝ) + 1) ^ 4)⁻¹) :
    (hoyingSum n : ℤ) = round Ahat := by
  have hNpos : (0 : ℝ) < ((n + 1 : ℕ) : ℝ) := by positivity
  have hN1 : (1 : ℝ) ≤ ((n + 1 : ℕ) : ℝ) := by
    push_cast
    have : (0 : ℝ) ≤ n := Nat.cast_nonneg n
    linarith
  have hHN' : H ≤ ((n + 1 : ℕ) : ℝ) := by push_cast; exact hHN
  have hs := parS_sq (n + 1) H sigh hNpos hH0 hHN' hsig0 hsig1
  have hbar := par_bar (n + 1) H sigh hNpos hH0 hHN' hsig0 hsig1
  have hδ := par_displacement (n + 1) H sigh ε hNpos hN1 hsigε
  push_cast at hs hbar hδ
  exact hoyingSum_eq_round_assembled n (parK (n + 1) H sigh) (parL (n + 1) H) sigh
    (parS (n + 1) H sigh) Ahat Ehank Ek2 Enon1
    (H ^ 2 * ε * (1 + Real.log (2 * ((n : ℝ) + 1))) / 8) Enon hN
    (parL_nonneg (n + 1) H hH0) (Real.sqrt_nonneg _) hs
    (parL_le_parS (n + 1) H sigh hNpos hH0 hHN' hsig0 hsig1)
    (par_K (n + 1) H sigh) hbar hε hδ hE hH hKb hn

end DirichletDivisor

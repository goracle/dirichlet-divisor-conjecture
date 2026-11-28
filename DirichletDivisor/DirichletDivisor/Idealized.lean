import DirichletDivisor.SmoothLink
import DirichletDivisor.Parameters

set_option linter.style.whitespace false

namespace DirichletDivisor

open Finset MeasureTheory

/-! ## Idealized algorithm output and the discharge of `hε`

The algorithm's output is modelled as
`idealAhat = ∫ η·P(x(u)) du + (local correction sum) - dualApprox`,
where `dualApprox` is *any* real within `Edual` of the exact dual integral `∫ η·dualMain(x(u))`.
All numerical errors of §4/§5 (quadrature, tables, phases, tail cutoff) enter only through `Edual`.
-/

/-- The local correction sum appearing in `smoothedApprox`. -/
noncomputable def corrSum (s L : ℝ) (K : ℕ) : ℝ :=
  ∑ k ∈ (Icc 1 K).filter (fun k : ℕ => (s - L) ^ 2 < k ∧ (k : ℝ) < (s + L) ^ 2),
    (d k : ℝ) * corr eta (fun u => (s + L * u) ^ 2) (s ^ 2 + L ^ 2 * etaVar) k

theorem smoothedApprox_eq (s L : ℝ) (K : ℕ) :
    smoothedApprox s L K
      = (∫ u in (-1 : ℝ)..1, eta u * R ((s + L * u) ^ 2)) + corrSum s L K := rfl

/-- The idealized output of the algorithm. -/
noncomputable def idealAhat (s L : ℝ) (K : ℕ) (dualApprox : ℝ) : ℝ :=
  (∫ u in (-1 : ℝ)..1, eta u * voronoiP ((s + L * u) ^ 2)) + corrSum s L K - dualApprox

/-- `hε` for the idealized output. -/
theorem ideal_hε (ci : CitedInputs) (N s L σ : ℝ) (K : ℕ) (dualApprox Edual : ℝ)
    (hN128 : 128 ≤ N) (hs0 : 0 ≤ s) (hL0 : 0 ≤ L) (hL : L ^ 2 ≤ N / 16)
    (hs : s ^ 2 = N - L ^ 2 * σ) (hσ0 : 0 ≤ σ) (hσ1 : σ ≤ 1) (hK : (s + L) ^ 2 ≤ K)
    (hD : IntervalIntegrable (fun u => eta u * dualMain ((s + L * u) ^ 2)) volume (-1) 1)
    (hdual : |dualApprox - ∫ u in (-1 : ℝ)..1, eta u * dualMain ((s + L * u) ^ 2)| ≤ Edual) :
    |idealAhat s L K dualApprox - smoothedApprox s L K|
      ≤ 4.8e-4 * N ^ (-(1 / 4 : ℝ))
          + 2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * Real.sqrt (N / 2)))
          + Edual := by
  have h := smooth_integral_error ci N s L σ K hN128 hs0 hL0 hL hs hσ0 hσ1 hK hD
  rw [smoothedApprox_eq]
  unfold idealAhat
  obtain ⟨A, hA⟩ : ∃ A : ℝ, A = ∫ u in (-1 : ℝ)..1, eta u * R ((s + L * u) ^ 2) := ⟨_, rfl⟩
  obtain ⟨B, hB⟩ : ∃ B : ℝ, B = ∫ u in (-1 : ℝ)..1, eta u * voronoiP ((s + L * u) ^ 2) :=
    ⟨_, rfl⟩
  obtain ⟨D, hDd⟩ : ∃ D : ℝ, D = ∫ u in (-1 : ℝ)..1, eta u * dualMain ((s + L * u) ^ 2) :=
    ⟨_, rfl⟩
  rw [← hA, ← hB, ← hDd] at h
  rw [← hDd] at hdual
  rw [← hA, ← hB]
  have e : B + corrSum s L K - dualApprox - (A + corrSum s L K)
      = -(A - (B - D)) + (D - dualApprox) := by ring
  rw [e]
  have h1 := abs_add_le (-(A - (B - D))) (D - dualApprox)
  rw [abs_neg, abs_sub_comm D dualApprox] at h1
  linarith

/-- **Final theorem for the idealized algorithm.** Everything numerical is packaged in
`Edual` (error of the dual sum), `ε` (error of `σ̂²`), and the final ledger bound `hn`. -/
theorem hoyingSum_eq_round_ideal (ci : CitedInputs) (n : ℕ)
    (H sigh ε Edual dualApprox Enon : ℝ)
    (hN : 2 ^ 65536 ≤ n + 1)
    (hH0 : 0 ≤ H) (hHN : H ≤ (n : ℝ) + 1)
    (hsig0 : 0 ≤ sigh) (hsig1 : sigh ≤ 1)
    (hsigε : |etaVar - sigh| ≤ ε)
    (hD : IntervalIntegrable
      (fun u => eta u * dualMain ((parS (n + 1) H sigh + parL (n + 1) H * u) ^ 2))
      volume (-1) 1)
    (hdual : |dualApprox - ∫ u in (-1 : ℝ)..1,
      eta u * dualMain ((parS (n + 1) H sigh + parL (n + 1) H * u) ^ 2)| ≤ Edual)
    (hE : Edual + H ^ 2 * ε * (1 + Real.log (2 * ((n : ℝ) + 1))) / 8 ≤ Enon)
    (hn : Enon ≤ (2 : ℝ) ^ 60 * (((n : ℝ) + 1) ^ 4)⁻¹) :
    (hoyingSum n : ℤ)
      = round (idealAhat (parS (n + 1) H sigh) (parL (n + 1) H) (parK (n + 1) H sigh)
          dualApprox) := by
  have hN' : (2 : ℝ) ^ 65536 ≤ (n : ℝ) + 1 := by exact_mod_cast hN
  have h7 : (128 : ℝ) ≤ 2 ^ 65536 := by
    calc (128 : ℝ) = 2 ^ 7 := by norm_num
      _ ≤ 2 ^ 65536 := pow_le_pow_right₀ (by norm_num) (by norm_num)
  have h128 : (128 : ℝ) ≤ (n : ℝ) + 1 := le_trans h7 hN'
  have hNpos : (0 : ℝ) < ((n + 1 : ℕ) : ℝ) := by positivity
  have hHN' : H ≤ ((n + 1 : ℕ) : ℝ) := by push_cast; exact hHN
  have hs := parS_sq (n + 1) H sigh hNpos hH0 hHN' hsig0 hsig1
  have hL := parL_sq_le (n + 1) H hNpos hH0 hHN'
  have hK := par_K (n + 1) H sigh
  push_cast at hs hL
  have hε := ideal_hε ci ((n : ℝ) + 1) (parS (n + 1) H sigh) (parL (n + 1) H) sigh
    (parK (n + 1) H sigh) dualApprox Edual h128 (Real.sqrt_nonneg _)
    (parL_nonneg (n + 1) H hH0) hL hs hsig0 hsig1 hK hD hdual
  exact hoyingSum_eq_round_param n H sigh ε
    (idealAhat (parS (n + 1) H sigh) (parL (n + 1) H) (parK (n + 1) H sigh) dualApprox)
    (4.8e-4 * ((n : ℝ) + 1) ^ (-(1 / 4 : ℝ)))
    (2 * ((n : ℝ) + 1) ^ (3 / 4 : ℝ)
      * Real.exp (-(2 * Real.pi * Real.sqrt (((n : ℝ) + 1) / 2))))
    Edual Enon hN hH0 hHN hsig0 hsig1 hsigε hε hE le_rfl le_rfl hn

end DirichletDivisor

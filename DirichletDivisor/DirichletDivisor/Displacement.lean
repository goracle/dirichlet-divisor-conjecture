import DirichletDivisor.Width

set_option linter.style.whitespace false

open Finset MeasureTheory

namespace DirichletDivisor

/-- What the algorithm evaluates: smooth integral plus local correction sum. -/
noncomputable def smoothedApprox (s L : ℝ) (K : ℕ) : ℝ :=
  (∫ u in (-1:ℝ)..1, eta u * R ((s + L * u) ^ 2))
    + ∑ k ∈ (Icc 1 K).filter
          (fun k : ℕ => (s - L) ^ 2 < k ∧ (k : ℝ) < (s + L) ^ 2),
        (d k : ℝ) * corr eta (fun u => (s + L * u) ^ 2)
          (s ^ 2 + L ^ 2 * etaVar) k

theorem R_window_center' (s L : ℝ) (hL0 : 0 ≤ L) (hLs : L ≤ s) (K : ℕ)
    (hK : (s + L) ^ 2 ≤ K) :
    R (s ^ 2 + L ^ 2 * etaVar) = smoothedApprox s L K :=
  R_window_center s L hL0 hLs K hK

/-- Eq. (25) with the centre shift: if s² = N - L² σ̂² (the numerically chosen
curve parameter) then R(N) is within L² |σ² - σ̂²| · 2N(1 + log 2N) of the
smoothed approximation. -/
theorem R_main_decomp (N K : ℕ) (L sigh s : ℝ) (hL0 : 0 ≤ L) (hs0 : 0 ≤ s)
    (hs : s ^ 2 = (N : ℝ) - L ^ 2 * sigh) (hLs : L ≤ s)
    (hK : (s + L) ^ 2 ≤ K)
    (hbar : s ^ 2 + L ^ 2 * etaVar ≤ 2 * N) :
    |R (N : ℝ) - smoothedApprox s L K|
      ≤ L ^ 2 * |etaVar - sigh| * (2 * N * (1 + Real.log (2 * N))) := by
  have hw := R_window_center' s L hL0 hLs K hK
  have hlip := R_lipschitz N (N : ℝ) (s ^ 2 + L ^ 2 * etaVar)
    (by linarith [Nat.cast_nonneg (α := ℝ) N]) hbar
  have habs : |(N : ℝ) - (s ^ 2 + L ^ 2 * etaVar)| = L ^ 2 * |etaVar - sigh| := by
    have h : (N : ℝ) - (s ^ 2 + L ^ 2 * etaVar) = L ^ 2 * (sigh - etaVar) := by
      rw [hs]; ring
    rw [h, abs_mul, abs_of_nonneg (sq_nonneg L), abs_sub_comm]
  rw [← hw]
  calc |R (N : ℝ) - R (s ^ 2 + L ^ 2 * etaVar)|
      ≤ |(N : ℝ) - (s ^ 2 + L ^ 2 * etaVar)| * (2 * N * (1 + Real.log (2 * N))) := hlip
    _ = L ^ 2 * |etaVar - sigh| * (2 * N * (1 + Real.log (2 * N))) := by rw [habs]

/-- Rounding form: if the smoothed approximation is evaluated to within ε and the
displacement is below δ, then A_{N-1} = R(N) is within ε + δ of the evaluated
value. -/
theorem R_eval_error (N K : ℕ) (L sigh s : ℝ) (hL0 : 0 ≤ L) (hs0 : 0 ≤ s)
    (hs : s ^ 2 = (N : ℝ) - L ^ 2 * sigh) (hLs : L ≤ s)
    (hK : (s + L) ^ 2 ≤ K) (hbar : s ^ 2 + L ^ 2 * etaVar ≤ 2 * N)
    (Ahat ε δ : ℝ) (hε : |Ahat - smoothedApprox s L K| ≤ ε)
    (hδ : L ^ 2 * |etaVar - sigh| * (2 * N * (1 + Real.log (2 * N))) ≤ δ) :
    |Ahat - R (N : ℝ)| ≤ ε + δ := by
  have h := R_main_decomp N K L sigh s hL0 hs0 hs hLs hK hbar
  calc |Ahat - R (N : ℝ)|
      = |(Ahat - smoothedApprox s L K) + (smoothedApprox s L K - R (N : ℝ))| := by ring_nf
    _ ≤ |Ahat - smoothedApprox s L K| + |smoothedApprox s L K - R (N : ℝ)| :=
        abs_add_le _ _
    _ ≤ ε + δ := by
        rw [abs_sub_comm (smoothedApprox s L K)]
        linarith

end DirichletDivisor

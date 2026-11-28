import DirichletDivisor.EtaAverage
import DirichletDivisor.Displacement

set_option linter.style.whitespace false

namespace DirichletDivisor

open Finset MeasureTheory

/-! ## Linking `eta_average_voronoi` to the algorithm's curve `x(u) = (s + L u)²`

Parameter-free window facts (`L² ≤ N/16`, `s² = N - L²σ`, `0 ≤ σ ≤ 1`) give
`(s-L)² ≥ N/2` and `(s+L)² ≤ 2N`, so the smoothing support lies in the range where the
pointwise Voronoï bounds hold. -/

theorem window_bounds (N s L σ : ℝ) (hN : 0 < N) (hs0 : 0 ≤ s) (hL0 : 0 ≤ L)
    (hL : L ^ 2 ≤ N / 16) (hs : s ^ 2 = N - L ^ 2 * σ) (hσ0 : 0 ≤ σ) (hσ1 : σ ≤ 1) :
    0 ≤ s - L ∧ N / 2 ≤ (s - L) ^ 2 ∧ (s + L) ^ 2 ≤ 2 * N := by
  obtain ⟨r, hr⟩ : ∃ r : ℝ, r = Real.sqrt N := ⟨_, rfl⟩
  have hr0 : 0 < r := by rw [hr]; exact Real.sqrt_pos.2 hN
  have hr2 : r ^ 2 = N := by rw [hr]; exact Real.sq_sqrt hN.le
  have hLr : L ≤ r / 4 := by
    by_contra h
    have h' := not_le.mp h
    have := mul_self_lt_mul_self (by linarith : 0 ≤ r / 4) h'
    nlinarith
  have hLσ : L ^ 2 * σ ≤ L ^ 2 := by nlinarith [sq_nonneg L]
  have hLσ0 : 0 ≤ L ^ 2 * σ := mul_nonneg (sq_nonneg L) hσ0
  have hsr : s ≤ r := by
    by_contra h
    have h' := not_le.mp h
    have := mul_self_lt_mul_self hr0.le h'
    nlinarith
  have hs96 : 24 / 25 * r ≤ s := by
    by_contra h
    have h' := not_le.mp h
    have := mul_self_lt_mul_self hs0 h'
    nlinarith
  have h1 : 71 / 100 * r ≤ s - L := by linarith
  refine ⟨by linarith, ?_, ?_⟩
  · have := mul_self_le_mul_self (by linarith : 0 ≤ 71 / 100 * r) h1
    nlinarith
  · have h2 : s + L ≤ 5 / 4 * r := by linarith
    have := mul_self_le_mul_self (by linarith : 0 ≤ s + L) h2
    nlinarith

theorem xu_bounds (s L u : ℝ) (h0 : 0 ≤ s - L) (hL0 : 0 ≤ L)
    (hu : u ∈ Set.Icc (-1 : ℝ) 1) :
    (s - L) ^ 2 ≤ (s + L * u) ^ 2 ∧ (s + L * u) ^ 2 ≤ (s + L) ^ 2 := by
  have a : s - L ≤ s + L * u := by
    nlinarith [mul_nonneg hL0 (by linarith [hu.1] : (0 : ℝ) ≤ u + 1)]
  have b : s + L * u ≤ s + L := by
    nlinarith [mul_nonneg hL0 (by linarith [hu.2] : (0 : ℝ) ≤ 1 - u)]
  exact ⟨pow_le_pow_left₀ h0 a 2, pow_le_pow_left₀ (by linarith) b 2⟩

/-- The smooth-integral part of `smoothedApprox`, compared with the Voronoï main terms. -/
theorem smooth_integral_error (ci : CitedInputs) (N s L σ : ℝ) (K : ℕ)
    (hN128 : 128 ≤ N) (hs0 : 0 ≤ s) (hL0 : 0 ≤ L) (hL : L ^ 2 ≤ N / 16)
    (hs : s ^ 2 = N - L ^ 2 * σ) (hσ0 : 0 ≤ σ) (hσ1 : σ ≤ 1) (hK : (s + L) ^ 2 ≤ K)
    (hD : IntervalIntegrable (fun u => eta u * dualMain ((s + L * u) ^ 2)) volume (-1) 1) :
    |(∫ u in (-1 : ℝ)..1, eta u * R ((s + L * u) ^ 2))
        - ((∫ u in (-1 : ℝ)..1, eta u * voronoiP ((s + L * u) ^ 2))
            - ∫ u in (-1 : ℝ)..1, eta u * dualMain ((s + L * u) ^ 2))|
      ≤ 4.8e-4 * N ^ (-(1 / 4 : ℝ))
          + 2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * Real.sqrt (N / 2))) := by
  obtain ⟨h0, hlo, hhi⟩ := window_bounds N s L σ (by linarith) hs0 hL0 hL hs hσ0 hσ1
  have hxc : Continuous (fun u : ℝ => (s + L * u) ^ 2) :=
    (continuous_const.add (continuous_const.mul continuous_id)).pow 2
  refine eta_average_voronoi ci N (fun u : ℝ => (s + L * u) ^ 2) hxc K ?_ hN128 ?_ ?_ hD
  · intro u hu
    have := (xu_bounds s L u h0 hL0 hu).2
    show ⌊(s + L * u) ^ 2⌋₊ ≤ K
    exact Nat.floor_le_of_le (le_trans this hK)
  · intro u hu
    have := (xu_bounds s L u h0 hL0 hu).1
    show N / 2 ≤ (s + L * u) ^ 2
    linarith
  · intro u hu
    have := (xu_bounds s L u h0 hL0 hu).2
    show (s + L * u) ^ 2 ≤ 2 * N
    linarith

end DirichletDivisor

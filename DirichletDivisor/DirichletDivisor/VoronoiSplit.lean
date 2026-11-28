import DirichletDivisor.K2Sum

set_option linter.style.whitespace false

namespace DirichletDivisor

open Finset

/-! ## Pointwise Voronoï split

For `N/2 ≤ x ≤ 2N`, `N ≥ 128`:
`R x = P x - dualMain x - ∑ rem + ∑ k2`, hence
`|R x - (P x - dualMain x)| ≤ 4.8e-4 N^{-1/4} + 2 N^{3/4} e^{-2π√(N/2)}`.
Summability of the oscillatory main series is *derived* (it is the Voronoï series minus two
absolutely summable ones), so no extra analysis is needed. -/

/-- The smooth part `P(x) = x²/2 (log x + 2γ - 3/2) + x/4 - 1/144`. -/
noncomputable def voronoiP (x : ℝ) : ℝ :=
  x ^ 2 / 2 * (Real.log x + 2 * Real.eulerMascheroniConstant - 3 / 2) + x / 4 - 1 / 144

/-- The `(j+1)`-st Voronoï term, in the shape used by `CitedInputs`. -/
noncomputable def voronoiTerm (ci : CitedInputs) (x : ℝ) (j : ℕ) : ℝ :=
  (d (j + 1) : ℝ) * (x / ((j : ℝ) + 1)) *
    (ci.Y2 (4 * Real.pi * Real.sqrt (((j : ℝ) + 1) * x)) / (2 * Real.pi)
      - ci.K2 (4 * Real.pi * Real.sqrt (((j : ℝ) + 1) * x)) / Real.pi ^ 2)

/-- Oscillatory main term `d(m)(x/m) · main(Z)/(2π)`. -/
noncomputable def mainTerm (x : ℝ) (m : ℕ) : ℝ :=
  (d m : ℝ) * ((x / m) * (hankelMain (4 * Real.pi * Real.sqrt (m * x)) / (2 * Real.pi)))

/-- The dual main sum (what the algorithm's `W_H` series approximates). -/
noncomputable def dualMain (x : ℝ) : ℝ := ∑' j : ℕ, mainTerm x (j + 1)

theorem voronoi_term_split (ci : CitedInputs) (x : ℝ) (j : ℕ) :
    voronoiTerm ci x j
      = mainTerm x (j + 1) + hankelRem ci x (j + 1) - k2Term ci x (j + 1) := by
  unfold voronoiTerm mainTerm hankelRem k2Term
  push_cast
  ring

theorem voronoi_eq (ci : CitedInputs) (x : ℝ) (hx : 0 < x) :
    R x = voronoiP x - ∑' j : ℕ, voronoiTerm ci x j := by
  unfold voronoiP voronoiTerm
  exact ci.voronoi x hx

theorem voronoi_summable (ci : CitedInputs) (x : ℝ) (hx : 0 < x) :
    Summable (voronoiTerm ci x) := by
  unfold voronoiTerm
  exact ci.summable_voronoi x hx

theorem voronoi_pointwise (ci : CitedInputs) (N x : ℝ) (hN : 128 ≤ N) (hx : N / 2 ≤ x)
    (hx2 : x ≤ 2 * N) :
    Summable (fun j : ℕ => mainTerm x (j + 1)) ∧
      |R x - (voronoiP x - dualMain x)|
        ≤ 4.8e-4 * N ^ (-(1 / 4 : ℝ))
          + 2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * Real.sqrt (N / 2))) := by
  have hx0 : 0 < x := by linarith
  obtain ⟨hrs, hrt⟩ := hankel_remainder_sum_N ci N x hN hx
  obtain ⟨hks, hkt⟩ := k2_remainder_sum_N ci N x hN hx hx2
  have hT := voronoi_summable ci x hx0
  have hrem : Summable (fun j : ℕ => hankelRem ci x (j + 1)) := summable_abs_iff.1 hrs
  have hk2 : Summable (fun j : ℕ => k2Term ci x (j + 1)) := summable_abs_iff.1 hks
  have hmain : Summable (fun j : ℕ => mainTerm x (j + 1)) := by
    have h : (fun j : ℕ => mainTerm x (j + 1))
        = fun j : ℕ => voronoiTerm ci x j - hankelRem ci x (j + 1) + k2Term ci x (j + 1) := by
      funext j
      rw [voronoi_term_split ci x j]
      ring
    rw [h]
    exact (hT.sub hrem).add hk2
  have htsum : ∑' j : ℕ, voronoiTerm ci x j
      = dualMain x + ∑' j : ℕ, hankelRem ci x (j + 1) - ∑' j : ℕ, k2Term ci x (j + 1) := by
    calc ∑' j : ℕ, voronoiTerm ci x j
        = ∑' j : ℕ, (mainTerm x (j + 1) + hankelRem ci x (j + 1) - k2Term ci x (j + 1)) :=
          tsum_congr (fun j => voronoi_term_split ci x j)
      _ = (∑' j : ℕ, mainTerm x (j + 1)) + ∑' j : ℕ, hankelRem ci x (j + 1)
            - ∑' j : ℕ, k2Term ci x (j + 1) :=
          ((hmain.hasSum.add hrem.hasSum).sub hk2.hasSum).tsum_eq
      _ = dualMain x + ∑' j : ℕ, hankelRem ci x (j + 1) - ∑' j : ℕ, k2Term ci x (j + 1) := rfl
  have hdiff : R x - (voronoiP x - dualMain x)
      = -(∑' j : ℕ, hankelRem ci x (j + 1)) + ∑' j : ℕ, k2Term ci x (j + 1) := by
    rw [voronoi_eq ci x hx0, htsum]
    ring
  have hA : |∑' j : ℕ, hankelRem ci x (j + 1)| ≤ ∑' j : ℕ, |hankelRem ci x (j + 1)| := by
    have := norm_tsum_le_tsum_norm (f := fun j : ℕ => hankelRem ci x (j + 1))
      (by simpa only [Real.norm_eq_abs] using hrs)
    simpa only [Real.norm_eq_abs] using this
  have hB : |∑' j : ℕ, k2Term ci x (j + 1)| ≤ ∑' j : ℕ, |k2Term ci x (j + 1)| := by
    have := norm_tsum_le_tsum_norm (f := fun j : ℕ => k2Term ci x (j + 1))
      (by simpa only [Real.norm_eq_abs] using hks)
    simpa only [Real.norm_eq_abs] using this
  refine ⟨hmain, ?_⟩
  rw [hdiff]
  have h3 := abs_add_le (-(∑' j : ℕ, hankelRem ci x (j + 1))) (∑' j : ℕ, k2Term ci x (j + 1))
  rw [abs_neg] at h3
  linarith

end DirichletDivisor

import DirichletDivisor.NearestInteger

open Finset

namespace DirichletDivisor

/-- [Cited] Inputs from the literature / DLMF that Mathlib does not have.
These are hypotheses, not axioms, so nothing downstream is inconsistent
if one of them is stated wrongly. Discharge each field later. -/
structure CitedInputs where
  /-- Bessel functions Y₂ and K₂ (absent from Mathlib). -/
  Y2 : ℝ → ℝ
  K2 : ℝ → ℝ
  /-- §2: explicit Hankel remainder for Y₂ on z ≥ 100 (DLMF 10.17(iii)). -/
  hankel_Y2 : ∀ z : ℝ, 100 ≤ z →
    |Y2 z - Real.sqrt (2 / (Real.pi * z)) *
        (Real.sin (z - 5 * Real.pi / 4)
          + 15 / (8 * z) * Real.cos (z - 5 * Real.pi / 4))|
      ≤ Real.sqrt (2 / (Real.pi * z)) * (0.83 / z ^ 2)
  /-- §2: first-omitted-term bound for K₂ on z ≥ 100 (DLMF 10.40(ii)). -/
  bound_K2 : ∀ z : ℝ, 100 ≤ z →
    |K2 z| ≤ 1.02 * Real.sqrt (Real.pi / (2 * z)) * Real.exp (-z)
  /-- §1: differentiated order-two Voronoï identity (the note's main identity). -/
  summable_voronoi : ∀ x : ℝ, 0 < x → Summable fun m : ℕ =>
    (d (m + 1) : ℝ) * (x / (m + 1)) *
      (Y2 (4 * Real.pi * Real.sqrt ((m + 1) * x)) / (2 * Real.pi)
        - K2 (4 * Real.pi * Real.sqrt ((m + 1) * x)) / Real.pi ^ 2)
  voronoi : ∀ x : ℝ, 0 < x →
    R x = x ^ 2 / 2 * (Real.log x + 2 * Real.eulerMascheroniConstant - 3 / 2)
            + x / 4 - 1 / 144
          - ∑' m : ℕ, (d (m + 1) : ℝ) * (x / (m + 1)) *
              (Y2 (4 * Real.pi * Real.sqrt ((m + 1) * x)) / (2 * Real.pi)
                - K2 (4 * Real.pi * Real.sqrt ((m + 1) * x)) / Real.pi ^ 2)

end DirichletDivisor

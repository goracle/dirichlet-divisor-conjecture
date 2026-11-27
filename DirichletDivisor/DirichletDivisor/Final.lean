import DirichletDivisor.Main
import DirichletDivisor.Ledger

set_option linter.style.whitespace false

namespace DirichletDivisor

/-- Top-level statement, modulo the algorithmic error decomposition.
Given the ledger's component bounds for `N = n + 1 ≥ 2^65536`, the nearest integer
to `Ahat` is the Hoying sum. -/
theorem hoyingSum_eq_round_of_ledger (n : ℕ) (Ahat Ehank Ek2 Enon : ℝ)
    (hN : (2 : ℝ) ^ 65536 ≤ (n : ℝ) + 1)
    (hsum : |Ahat - R ((n : ℝ) + 1)| ≤ Ehank + Ek2 + Enon)
    (hH : Ehank ≤ 4.8e-4 * ((n : ℝ) + 1) ^ (-(1 / 4) : ℝ))
    (hK : Ek2 ≤ 2 * ((n : ℝ) + 1) ^ (3 / 4 : ℝ) *
      Real.exp (-(2 * Real.pi * Real.sqrt (((n : ℝ) + 1) / 2))))
    (hn : Enon ≤ (2 : ℝ) ^ 60 * (((n : ℝ) + 1) ^ 4)⁻¹) :
    (hoyingSum n : ℤ) = round Ahat :=
  hoyingSum_eq_round n Ahat
    (ledger_lt_quarter ((n : ℝ) + 1) Ehank Ek2 Enon _ hN hsum hH hK hn)

/-- Same, phrased for a natural-number threshold `2^65536 ≤ n + 1`. -/
theorem hoyingSum_eq_round_of_nat (n : ℕ) (Ahat Ehank Ek2 Enon : ℝ)
    (hN : 2 ^ 65536 ≤ n + 1)
    (hsum : |Ahat - R ((n : ℝ) + 1)| ≤ Ehank + Ek2 + Enon)
    (hH : Ehank ≤ 4.8e-4 * ((n : ℝ) + 1) ^ (-(1 / 4) : ℝ))
    (hK : Ek2 ≤ 2 * ((n : ℝ) + 1) ^ (3 / 4 : ℝ) *
      Real.exp (-(2 * Real.pi * Real.sqrt (((n : ℝ) + 1) / 2))))
    (hn : Enon ≤ (2 : ℝ) ^ 60 * (((n : ℝ) + 1) ^ 4)⁻¹) :
    (hoyingSum n : ℤ) = round Ahat := by
  have hN' : (2 : ℝ) ^ 65536 ≤ (n : ℝ) + 1 := by exact_mod_cast hN
  exact hoyingSum_eq_round_of_ledger n Ahat Ehank Ek2 Enon hN' hsum hH hK hn

end DirichletDivisor

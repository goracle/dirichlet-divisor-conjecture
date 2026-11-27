import DirichletDivisor.Displacement
import DirichletDivisor.Final

set_option linter.style.whitespace false

namespace DirichletDivisor

/-- `hsum` discharged from `R_eval_error`. The only remaining inputs are:
* `hε`: `Ahat` is within `Ehank + Ek2 + Enon1` of `smoothedApprox` (Voronoï remainder,
  tail, quadrature, tables, phases: the algorithmic ledger terms);
* `hδ`: the centre-displacement term is at most `Enon2` (eq. 25);
* `Enon1 + Enon2 ≤ Enon`. -/
theorem hsum_of_eval (n K : ℕ) (L sigh s Ahat Ehank Ek2 Enon1 Enon2 Enon : ℝ)
    (hL0 : 0 ≤ L) (hs0 : 0 ≤ s)
    (hs : s ^ 2 = ((n : ℝ) + 1) - L ^ 2 * sigh) (hLs : L ≤ s)
    (hK : (s + L) ^ 2 ≤ K)
    (hbar : s ^ 2 + L ^ 2 * etaVar ≤ 2 * ((n : ℝ) + 1))
    (hε : |Ahat - smoothedApprox s L K| ≤ Ehank + Ek2 + Enon1)
    (hδ : L ^ 2 * |etaVar - sigh| *
        (2 * ((n : ℝ) + 1) * (1 + Real.log (2 * ((n : ℝ) + 1)))) ≤ Enon2)
    (hE : Enon1 + Enon2 ≤ Enon) :
    |Ahat - R ((n : ℝ) + 1)| ≤ Ehank + Ek2 + Enon := by
  have h := R_eval_error (n + 1) K L sigh s hL0 hs0
    (by push_cast; exact hs) hLs hK (by push_cast; exact hbar)
    Ahat (Ehank + Ek2 + Enon1) Enon2 hε (by push_cast; exact hδ)
  push_cast at h
  linarith

/-- The assembled theorem: from the algorithmic error hypotheses to the exact Hoying sum.
All remaining hypotheses are exactly the analytic/numerical ledger inputs. -/
theorem hoyingSum_eq_round_assembled (n K : ℕ)
    (L sigh s Ahat Ehank Ek2 Enon1 Enon2 Enon : ℝ)
    (hN : 2 ^ 65536 ≤ n + 1)
    (hL0 : 0 ≤ L) (hs0 : 0 ≤ s)
    (hs : s ^ 2 = ((n : ℝ) + 1) - L ^ 2 * sigh) (hLs : L ≤ s)
    (hK : (s + L) ^ 2 ≤ K)
    (hbar : s ^ 2 + L ^ 2 * etaVar ≤ 2 * ((n : ℝ) + 1))
    (hε : |Ahat - smoothedApprox s L K| ≤ Ehank + Ek2 + Enon1)
    (hδ : L ^ 2 * |etaVar - sigh| *
        (2 * ((n : ℝ) + 1) * (1 + Real.log (2 * ((n : ℝ) + 1)))) ≤ Enon2)
    (hE : Enon1 + Enon2 ≤ Enon)
    (hH : Ehank ≤ 4.8e-4 * ((n : ℝ) + 1) ^ (-(1 / 4) : ℝ))
    (hKb : Ek2 ≤ 2 * ((n : ℝ) + 1) ^ (3 / 4 : ℝ) *
      Real.exp (-(2 * Real.pi * Real.sqrt (((n : ℝ) + 1) / 2))))
    (hn : Enon ≤ (2 : ℝ) ^ 60 * (((n : ℝ) + 1) ^ 4)⁻¹) :
    (hoyingSum n : ℤ) = round Ahat :=
  hoyingSum_eq_round_of_nat n Ahat Ehank Ek2 Enon hN
    (hsum_of_eval n K L sigh s Ahat Ehank Ek2 Enon1 Enon2 Enon
      hL0 hs0 hs hLs hK hbar hε hδ hE)
    hH hKb hn

end DirichletDivisor

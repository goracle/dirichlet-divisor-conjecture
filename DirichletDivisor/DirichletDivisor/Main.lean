import DirichletDivisor.Cited
import DirichletDivisor.Lipschitz

namespace DirichletDivisor

/-- Skeleton of the main theorem. The remaining work is producing `Ahat`
(the interval-arithmetic algorithm of §3–§5) and proving `herr`, which is
exactly the §6 error ledger (eq. 33). Everything after `herr` is already proved. -/
theorem hoyingSum_eq_round (n : ℕ) (Ahat : ℝ)
    (herr : |Ahat - R ((n : ℝ) + 1)| < 1 / 4) :
    (hoyingSum n : ℤ) = round Ahat :=
  (round_eq_hoyingSum_of_R n Ahat herr).symm

end DirichletDivisor

module

public import Mathlib

@[expose] public section

open Finset

namespace DirichletDivisor

/-- The divisor function d(k). -/
def d (k : ℕ) : ℕ := (Nat.divisors k).card

/-- Hoying sum: A_n = ∑_{k ≤ n} (n+1-k) d(k). -/
def hoyingSum (n : ℕ) : ℕ := ∑ k ∈ Icc 1 n, (n + 1 - k) * d k

/-- R(x) = ∑_{k ≥ 1} d(k) (x-k)_+, written as a finite sum over k ≤ ⌊x⌋. -/
noncomputable def R (x : ℝ) : ℝ := ∑ k ∈ Icc 1 ⌊x⌋₊, (d k : ℝ) * (x - k)

/-- A_n = R(n+1). -/
theorem R_succ (n : ℕ) : R ((n : ℝ) + 1) = hoyingSum n := by
  have hfl : ⌊((n : ℝ) + 1)⌋₊ = n + 1 := by
    rw [← Nat.cast_one, ← Nat.cast_add]
    exact Nat.floor_natCast _
  unfold R hoyingSum
  rw [hfl, Finset.sum_Icc_succ_top (by omega)]
  simp only [Nat.cast_add, Nat.cast_one, sub_self, mul_zero, add_zero]
  push_cast
  refine Finset.sum_congr rfl fun k hk => ?_
  have hkn : k ≤ n := (Finset.mem_Icc.mp hk).2
  rw [Nat.cast_sub (by omega)]
  push_cast
  ring

end DirichletDivisor

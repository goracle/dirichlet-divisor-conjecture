import DirichletDivisor.Defs

namespace DirichletDivisor

/-- If x is within 1/4 of an integer z, then round x = z. -/
theorem round_eq_of_abs_sub_lt (z : ℤ) (x : ℝ) (h : |x - z| < 1 / 4) :
    round x = z := by
  rw [round_eq, Int.floor_eq_iff]
  have h' := abs_lt.mp h
  constructor <;> linarith [h'.1, h'.2]

/-- The final step of the algorithm: A_n is the nearest integer to any
approximation Ahat with |Ahat - A_n| < 1/4. -/
theorem round_hoyingSum (n : ℕ) (x : ℝ) (h : |x - (hoyingSum n : ℝ)| < 1 / 4) :
    round x = (hoyingSum n : ℤ) := by
  apply round_eq_of_abs_sub_lt
  simpa using h

/-- Same statement phrased through R, using A_n = R(n+1). -/
theorem round_eq_hoyingSum_of_R (n : ℕ) (x : ℝ) (h : |x - R ((n : ℝ) + 1)| < 1 / 4) :
    round x = (hoyingSum n : ℤ) := by
  rw [R_succ] at h
  exact round_hoyingSum n x h

end DirichletDivisor

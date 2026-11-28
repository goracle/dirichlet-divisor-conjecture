module

public import Mathlib

@[expose] public section


set_option linter.style.whitespace false

namespace DirichletDivisor

/-! ## Pointwise tail estimates (step toward Section 3's tail cutoff)

These are the project-independent analytic inputs: `e^{-y} ≤ k!/y^k`, and the resulting
termwise bound for `m^{-3/4} e^{-c m^{1/4}}`.  Only Mathlib is imported. -/

/-- `e^{-y} ≤ k! / y^k` for `y > 0`. -/
theorem exp_neg_le_factorial_div (y : ℝ) (hy : 0 < y) (k : ℕ) :
    Real.exp (-y) ≤ (k.factorial : ℝ) / y ^ k := by
  have hF : (0 : ℝ) < (k.factorial : ℝ) := Nat.cast_pos.mpr (Nat.factorial_pos _)
  have h := Real.pow_div_factorial_le_exp y hy.le k
  rw [div_le_iff₀ hF] at h
  have hE := Real.exp_pos y
  rw [le_div_iff₀ (pow_pos hy k), Real.exp_neg, inv_mul_le_iff₀ hE]
  linarith

/-- `m^{-3/4} e^{-c m^{1/4}} ≤ (k!/c^k) · m^{-3/4 - k/4}` for `m > 0`. -/
theorem term_le (c : ℝ) (hc : 0 < c) (k : ℕ) (m : ℝ) (hm : 0 < m) :
    m ^ (-(3 / 4 : ℝ)) * Real.exp (-(c * m ^ (1 / 4 : ℝ)))
      ≤ (k.factorial : ℝ) / c ^ k * m ^ (-(3 / 4 : ℝ) - (k : ℝ) / 4) := by
  have hm4 : 0 < m ^ (1 / 4 : ℝ) := Real.rpow_pos_of_pos hm _
  have hy : 0 < c * m ^ (1 / 4 : ℝ) := mul_pos hc hm4
  have hexp := exp_neg_le_factorial_div (c * m ^ (1 / 4 : ℝ)) hy k
  have e1 : (m ^ (1 / 4 : ℝ)) ^ k = m ^ ((k : ℝ) / 4) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hm.le]
    congr 1
    ring
  have hmk : 0 < m ^ ((k : ℝ) / 4) := Real.rpow_pos_of_pos hm _
  have hck : 0 < c ^ k := pow_pos hc k
  calc m ^ (-(3 / 4 : ℝ)) * Real.exp (-(c * m ^ (1 / 4 : ℝ)))
      ≤ m ^ (-(3 / 4 : ℝ)) * ((k.factorial : ℝ) / (c * m ^ (1 / 4 : ℝ)) ^ k) :=
        mul_le_mul_of_nonneg_left hexp (Real.rpow_nonneg hm.le _)
    _ = (k.factorial : ℝ) / c ^ k * m ^ (-(3 / 4 : ℝ) - (k : ℝ) / 4) := by
        rw [mul_pow, e1, Real.rpow_sub hm]
        field_simp

end DirichletDivisor

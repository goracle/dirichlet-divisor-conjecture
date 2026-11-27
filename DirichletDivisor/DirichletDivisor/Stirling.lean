import DirichletDivisor.Fourier

set_option linter.style.whitespace false

namespace DirichletDivisor

/-- e ≤ (1 + 1/n)^(n+1) for n ≥ 1. -/
theorem exp_one_le_pow (n : ℕ) (hn : 1 ≤ n) :
    Real.exp 1 ≤ (1 + 1 / (n : ℝ)) ^ (n + 1) := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hx : 0 < 1 + 1 / (n : ℝ) := by positivity
  have hlog : 1 / ((n : ℝ) + 1) ≤ Real.log (1 + 1 / (n : ℝ)) := by
    have h := Real.log_le_sub_one_of_pos (inv_pos.2 hx)
    rw [Real.log_inv] at h
    have e : (1 + 1 / (n : ℝ))⁻¹ - 1 = -(1 / ((n : ℝ) + 1)) := by
      field_simp
      ring
    rw [e] at h
    linarith
  have h1 : 1 ≤ ((n : ℝ) + 1) * Real.log (1 + 1 / (n : ℝ)) := by
    have h2 : ((n : ℝ) + 1) * (1 / ((n : ℝ) + 1)) = 1 := by field_simp
    have h3 : (0 : ℝ) < (n : ℝ) + 1 := by positivity
    nlinarith [hlog, h2, h3]
  calc Real.exp 1
      ≤ Real.exp (((n : ℝ) + 1) * Real.log (1 + 1 / (n : ℝ))) := Real.exp_le_exp.2 h1
    _ = (1 + 1 / (n : ℝ)) ^ (n + 1) := by
        have hpow : ∀ (k : ℕ) (x : ℝ), Real.exp ((k : ℝ) * x) = Real.exp x ^ k := by
          intro k x
          induction k with
          | zero => simp
          | succ m ih => rw [Nat.cast_succ, add_mul, one_mul, Real.exp_add, ih, pow_succ]
        have h := hpow (n + 1) (Real.log (1 + 1 / (n : ℝ)))
        rw [Real.exp_log hx] at h
        push_cast at h
        exact h        



/-- Stirling-type upper bound: r! · e^r ≤ e · r^(r+1) for r ≥ 1. -/
theorem factorial_mul_exp_le (r : ℕ) (hr : 1 ≤ r) :
    (r.factorial : ℝ) * Real.exp r ≤ Real.exp 1 * (r : ℝ) ^ (r + 1) := by
  induction r, hr using Nat.le_induction with
  | base => simp
  | succ n hn ih =>
    have hn' : (0 : ℝ) < n := by exact_mod_cast hn
    have hp : Real.exp 1 * (n : ℝ) ^ (n + 1) ≤ ((n : ℝ) + 1) ^ (n + 1) := by
      have h := exp_one_le_pow n hn
      have hnn : 0 ≤ (n : ℝ) ^ (n + 1) := by positivity
      calc Real.exp 1 * (n : ℝ) ^ (n + 1)
          ≤ (1 + 1 / (n : ℝ)) ^ (n + 1) * (n : ℝ) ^ (n + 1) :=
            mul_le_mul_of_nonneg_right h hnn
        _ = ((n : ℝ) + 1) ^ (n + 1) := by
            rw [← mul_pow]
            congr 1
            field_simp
    have he1 : 0 ≤ Real.exp 1 := (Real.exp_pos 1).le
    have hn1 : (0 : ℝ) ≤ (n : ℝ) + 1 := by positivity
    rw [Nat.factorial_succ]
    push_cast
    rw [Real.exp_add]
    calc ((n : ℝ) + 1) * (n.factorial : ℝ) * (Real.exp n * Real.exp 1)
        = ((n : ℝ) + 1) * ((n.factorial : ℝ) * Real.exp n) * Real.exp 1 := by ring
      _ ≤ ((n : ℝ) + 1) * (Real.exp 1 * (n : ℝ) ^ (n + 1)) * Real.exp 1 :=
          mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left ih hn1) he1
      _ = Real.exp 1 * (((n : ℝ) + 1) * (Real.exp 1 * (n : ℝ) ^ (n + 1))) := by ring
      _ ≤ Real.exp 1 * (((n : ℝ) + 1) * ((n : ℝ) + 1) ^ (n + 1)) :=
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hp hn1) he1
      _ = Real.exp 1 * ((n : ℝ) + 1) ^ (n + 1 + 1) := by ring

/-- Squared form used in the decay optimisation: (r!)² e^{2r} ≤ e² r^{2r+2}. -/
theorem factorial_sq_mul_exp_le (r : ℕ) (hr : 1 ≤ r) :
    ((r.factorial : ℝ) * Real.exp r) ^ 2 ≤ (Real.exp 1 * (r : ℝ) ^ (r + 1)) ^ 2 :=
  pow_le_pow_left₀ (by positivity) (factorial_mul_exp_le r hr) 2

end DirichletDivisor

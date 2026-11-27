import DirichletDivisor.Schedule

set_option linter.style.whitespace false

namespace DirichletDivisor

/-- Bernoulli-type upper bound: (1+x)^j (1 - j x) ≤ 1 for x ≥ 0. -/
theorem one_add_pow_mul_le_one (x : ℝ) (hx : 0 ≤ x) :
    ∀ j : ℕ, (1 + x) ^ j * (1 - (j : ℝ) * x) ≤ 1 := by
  intro j
  induction j with
  | zero => simp
  | succ n ih =>
    have hp : 0 < (1 + x) ^ n := pow_pos (by linarith) n
    have hsq : 0 ≤ ((n : ℝ) + 1) * x ^ 2 := by positivity
    have hstep : (1 + x) * (1 - ((n : ℝ) + 1) * x) ≤ 1 - (n : ℝ) * x := by
      nlinarith [hsq]
    have h2 : (1 + x) ^ (n + 1) * (1 - (((n + 1 : ℕ) : ℝ)) * x)
        ≤ (1 + x) ^ n * (1 - (n : ℝ) * x) := by
      push_cast
      calc (1 + x) ^ (n + 1) * (1 - ((n : ℝ) + 1) * x)
          = (1 + x) ^ n * ((1 + x) * (1 - ((n : ℝ) + 1) * x)) := by ring
        _ ≤ (1 + x) ^ n * (1 - (n : ℝ) * x) :=
          mul_le_mul_of_nonneg_left hstep hp.le
    linarith

/-- (2+u)^j ≤ (3/2) 2^j for j ≤ S when u S ≤ 1/4. -/
theorem pow_two_add_le (u : ℝ) (S : ℕ) (hu0 : 0 ≤ u) (hu : u * S ≤ 1 / 4) :
    ∀ j : ℕ, j ≤ S → (2 + u) ^ j ≤ (3 / 2) * 2 ^ j := by
  intro j hjS
  have hx : 0 ≤ u / 2 := by linarith
  have hp : 0 < (1 + u / 2) ^ j := pow_pos (by linarith) j
  have hjS' : (j : ℝ) ≤ S := by exact_mod_cast hjS
  have hju : (j : ℝ) * u ≤ (S : ℝ) * u := mul_le_mul_of_nonneg_right hjS' hu0
  have hjx : (j : ℝ) * (u / 2) ≤ 1 / 8 := by nlinarith
  have key := one_add_pow_mul_le_one (u / 2) hx j
  have h1 : (1 + u / 2) ^ j * (7 / 8) ≤ 1 :=
    calc (1 + u / 2) ^ j * (7 / 8)
        ≤ (1 + u / 2) ^ j * (1 - (j : ℝ) * (u / 2)) :=
          mul_le_mul_of_nonneg_left (by linarith) hp.le
      _ ≤ 1 := key
  have h2 : (1 + u / 2) ^ j ≤ 3 / 2 := by linarith
  have h3 : (2 + u) ^ j = 2 ^ j * (1 + u / 2) ^ j := by
    rw [← mul_pow]
    congr 1
    ring
  rw [h3]
  have h4 : 0 < (2 : ℝ) ^ j := by positivity
  nlinarith

end DirichletDivisor

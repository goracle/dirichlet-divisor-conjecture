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
  

/-- Eq. (15): unrolled FFT stage recurrence. L = 2^S. -/
theorem stage_recurrence (E : ℕ → ℝ) (u M0 : ℝ) (S : ℕ)
    (hu0 : 0 ≤ u) (hM0 : 0 ≤ M0) (hE0 : 0 ≤ E 0) (huS : u * S ≤ 1 / 4)
    (hrec : ∀ s, s < S → E (s + 1) ≤ (2 + u) * E s + u * 2 ^ s * M0 + 7 * u) :
    E S ≤ 2 * 2 ^ S * E 0 + 32 * u * 2 ^ S * (1 + M0 * S) := by
  have inv : ∀ k : ℕ, k ≤ S →
      E k ≤ (2 + u) ^ k * (E 0 + (k : ℝ) * u * M0 / 2 + 7 * u * (1 - 1 / 2 ^ k)) := by
    intro k
    induction k with
    | zero => intro _; simp
    | succ n ih =>
      intro hn
      have hn' : n < S := Nat.lt_of_succ_le hn
      have ih' := ih (by omega)
      have hr := hrec n hn'
      have hP : (2 : ℝ) ^ (n + 1) ≤ (2 + u) ^ (n + 1) := by
        first
          | exact pow_le_pow_left₀ (by norm_num) (by linarith) _
          | exact pow_le_pow_left (by norm_num) (by linarith) _
      have h3 : (2 + u) * E n ≤ (2 + u) ^ (n + 1) *
          (E 0 + (n : ℝ) * u * M0 / 2 + 7 * u * (1 - 1 / 2 ^ n)) := by
        calc (2 + u) * E n
            ≤ (2 + u) * ((2 + u) ^ n *
                (E 0 + (n : ℝ) * u * M0 / 2 + 7 * u * (1 - 1 / 2 ^ n))) :=
              mul_le_mul_of_nonneg_left ih' (by linarith)
          _ = (2 + u) ^ (n + 1) *
                (E 0 + (n : ℝ) * u * M0 / 2 + 7 * u * (1 - 1 / 2 ^ n)) := by ring
      have h4 : u * 2 ^ n * M0 ≤ (2 + u) ^ (n + 1) * (u * M0 / 2) := by
        calc u * 2 ^ n * M0 = 2 ^ (n + 1) * (u * M0 / 2) := by ring
          _ ≤ (2 + u) ^ (n + 1) * (u * M0 / 2) :=
              mul_le_mul_of_nonneg_right hP (by positivity)
      have h5 : 7 * u ≤ (2 + u) ^ (n + 1) * (7 * u / 2 ^ (n + 1)) := by
        calc 7 * u = 2 ^ (n + 1) * (7 * u / 2 ^ (n + 1)) := by field_simp
          _ ≤ (2 + u) ^ (n + 1) * (7 * u / 2 ^ (n + 1)) :=
              mul_le_mul_of_nonneg_right hP (by positivity)
      have h6 : (2 + u) ^ (n + 1) *
            (E 0 + ((n : ℝ) + 1) * u * M0 / 2 + 7 * u * (1 - 1 / 2 ^ (n + 1)))
          = (2 + u) ^ (n + 1) *
              (E 0 + (n : ℝ) * u * M0 / 2 + 7 * u * (1 - 1 / 2 ^ n))
            + (2 + u) ^ (n + 1) * (u * M0 / 2)
            + (2 + u) ^ (n + 1) * (7 * u / 2 ^ (n + 1)) := by
        field_simp
        ring
      push_cast
      linarith [h3, h4, h5, h6, hr]
  have h := inv S le_rfl
  have hj := pow_two_add_le u S hu0 huS S le_rfl
  have hL : (0 : ℝ) < 2 ^ S := by positivity
  have h1S : (1 : ℝ) ≤ 2 ^ S := by exact_mod_cast (Nat.one_le_two_pow : 1 ≤ 2 ^ S)
  have hfrac : (1 : ℝ) / 2 ^ S ≤ 1 := by
    rw [div_le_one hL]; exact h1S
  have hS0 : (0 : ℝ) ≤ S := Nat.cast_nonneg S
  have hI0 : 0 ≤ E 0 + (S : ℝ) * u * M0 / 2 + 7 * u * (1 - 1 / 2 ^ S) := by
    have : 0 ≤ 7 * u * (1 - 1 / 2 ^ S) :=
      mul_nonneg (by linarith) (by linarith)
    have : 0 ≤ (S : ℝ) * u * M0 / 2 := by positivity
    linarith
  have hIup : E 0 + (S : ℝ) * u * M0 / 2 + 7 * u * (1 - 1 / 2 ^ S)
      ≤ E 0 + (S : ℝ) * u * M0 / 2 + 7 * u := by
    have : 0 ≤ 7 * u * (1 / 2 ^ S) := by positivity
    nlinarith
  have a : 0 ≤ (2 : ℝ) ^ S * E 0 := by positivity
  have b : 0 ≤ u * 2 ^ S := by positivity
  have c : 0 ≤ u * 2 ^ S * (M0 * S) := by positivity
  calc E S ≤ (2 + u) ^ S *
          (E 0 + (S : ℝ) * u * M0 / 2 + 7 * u * (1 - 1 / 2 ^ S)) := h
    _ ≤ (3 / 2 * 2 ^ S) *
          (E 0 + (S : ℝ) * u * M0 / 2 + 7 * u * (1 - 1 / 2 ^ S)) :=
        mul_le_mul_of_nonneg_right hj hI0
    _ ≤ (3 / 2 * 2 ^ S) * (E 0 + (S : ℝ) * u * M0 / 2 + 7 * u) :=
        mul_le_mul_of_nonneg_left hIup (by positivity)
    _ ≤ 2 * 2 ^ S * E 0 + 32 * u * 2 ^ S * (1 + M0 * S) := by
        nlinarith [a, b, c]

end DirichletDivisor

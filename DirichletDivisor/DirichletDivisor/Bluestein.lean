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
        


/-- (15) specialised: forward transform of inputs of modulus ≤ 1 and initial error ≤ 10u. -/
theorem fft_forward_error (E : ℕ → ℝ) (u : ℝ) (S : ℕ)
    (hu0 : 0 ≤ u) (hE0 : 0 ≤ E 0) (hE0' : E 0 ≤ 10 * u) (huS : u * S ≤ 1 / 4)
    (hrec : ∀ s, s < S → E (s + 1) ≤ (2 + u) * E s + u * 2 ^ s * 1 + 7 * u) :
    E S ≤ 64 * u * 2 ^ S * (1 + S) := by
  have h := stage_recurrence E u 1 S hu0 zero_le_one hE0 huS hrec
  have h1 : 2 * (2 : ℝ) ^ S * E 0 ≤ 2 * 2 ^ S * (10 * u) :=
    mul_le_mul_of_nonneg_left hE0' (by positivity)
  have b : 0 ≤ u * (2 : ℝ) ^ S := by positivity
  have c : 0 ≤ u * (2 : ℝ) ^ S * S := by positivity
  nlinarith [b, c, h, h1]

/-- Pointwise product error: Eprod ≤ 2Lδ + δ² + 5u ≤ 200 u L² (1+S). -/
theorem prod_error (u L δ Eprod S : ℝ) (hu0 : 0 ≤ u) (hS : 0 ≤ S) (hL : 2 ≤ L)
    (hδ0 : 0 ≤ δ) (hδ : δ ≤ 64 * u * L * (1 + S))
    (hsmall : 64 * u * (1 + S) ≤ 1)
    (hE : Eprod ≤ 2 * L * δ + δ ^ 2 + 5 * u) :
    Eprod ≤ 200 * u * L ^ 2 * (1 + S) := by
  have hL0 : 0 ≤ L := by linarith
  have hδL : δ ≤ L := by
    calc δ ≤ 64 * u * L * (1 + S) := hδ
      _ = L * (64 * u * (1 + S)) := by ring
      _ ≤ L * 1 := mul_le_mul_of_nonneg_left hsmall hL0
      _ = L := by ring
  have hsq : δ * δ ≤ (64 * u * L * (1 + S)) * L :=
    mul_le_mul hδ hδL hδ0 (by positivity)
  have hLδ : L * δ ≤ L * (64 * u * L * (1 + S)) :=
    mul_le_mul_of_nonneg_left hδ hL0
  have hW : 0 ≤ u * L ^ 2 * (1 + S) := by positivity
  have hL2 : 4 ≤ L ^ 2 := by nlinarith
  have h5 : 4 * u ≤ u * L ^ 2 := by nlinarith
  have h5' : u * L ^ 2 ≤ u * L ^ 2 * (1 + S) := by nlinarith [mul_nonneg (mul_nonneg hu0 (sq_nonneg L)) hS]
  nlinarith [hsq, hLδ, h5, h5', hW]

/-- Inverse transform: Einv ≤ 3/2 Eprod + 3/4 u L² S + 21/2 u ≤ 304 u L² (1+S). -/
theorem inv_error (u L Einv Eprod S : ℝ) (hu0 : 0 ≤ u) (hS : 0 ≤ S) (hL : 2 ≤ L)
    (hE : Einv ≤ 3 / 2 * Eprod + 3 / 4 * u * L ^ 2 * S + 21 / 2 * u)
    (hP : Eprod ≤ 200 * u * L ^ 2 * (1 + S)) :
    Einv ≤ 304 * u * L ^ 2 * (1 + S) := by
  have hL2 : 4 ≤ L ^ 2 := by nlinarith
  have h1 : 4 * u ≤ u * L ^ 2 := by nlinarith
  have h2 : u * L ^ 2 ≤ u * L ^ 2 * (1 + S) := by
    nlinarith [mul_nonneg (mul_nonneg hu0 (sq_nonneg L)) hS]
  have h3 : 0 ≤ u * L ^ 2 * S := mul_nonneg (mul_nonneg hu0 (sq_nonneg L)) hS
  nlinarith [h1, h2, h3, hE, hP]

/-- Final chirp multiplication: ≤ 512 u L² (1+S). -/
theorem chirp_error (u L Efin Einv S : ℝ) (hu0 : 0 ≤ u) (hS : 0 ≤ S) (hL : 2 ≤ L)
    (hsmall : u * (1 + S) ≤ 1 / 64)
    (hE : Efin ≤ (1 + u) * Einv + u * L + 5 * u)
    (hI : Einv ≤ 304 * u * L ^ 2 * (1 + S)) :
    Efin ≤ 512 * u * L ^ 2 * (1 + S) := by
  have hW : 0 ≤ u * L ^ 2 * (1 + S) := by positivity
  have hu : u ≤ 1 / 64 := by nlinarith [mul_nonneg hu0 hS]
  have h1 : (1 + u) * Einv ≤ (1 + u) * (304 * u * L ^ 2 * (1 + S)) :=
    mul_le_mul_of_nonneg_left hI (by linarith)
  have h2 : (1 + u) * (304 * u * L ^ 2 * (1 + S))
      ≤ (65 / 64) * (304 * u * L ^ 2 * (1 + S)) := by
    nlinarith [hW]
  have hL2 : 4 ≤ L ^ 2 := by nlinarith
  have hLL : L ≤ L ^ 2 := by nlinarith
  have h3 : u * L ≤ u * L ^ 2 := mul_le_mul_of_nonneg_left hLL hu0
  have h4 : 4 * u ≤ u * L ^ 2 := by
    have := mul_le_mul_of_nonneg_left hL2 hu0
    linarith

  have h5 : u * L ^ 2 ≤ u * L ^ 2 * (1 + S) := by
    nlinarith [mul_nonneg (mul_nonneg hu0 (sq_nonneg L)) hS]
  nlinarith [h1, h2, h3, h4, h5, hW]

/-- Conversion to the (16) shape, with L = 2^S, S ≥ 1. -/
theorem chirp_to_final (u : ℝ) (S : ℕ) (hu0 : 0 ≤ u) (hS : 1 ≤ S) :
    512 * u * ((2 : ℝ) ^ S) ^ 2 * (1 + S)
      ≤ 2 ^ 12 * u * ((2 : ℝ) ^ S) ^ 3 * (Real.log (2 * 2 ^ S)) ^ 2 := by
  have hL1 : (2 : ℝ) ≤ 2 ^ S := by
    calc (2 : ℝ) = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ S := pow_le_pow_right₀ (by norm_num) hS
  have hS' : (1 : ℝ) ≤ S := by exact_mod_cast hS
  have hlog : Real.log (2 * 2 ^ S) = ((S : ℝ) + 1) * Real.log 2 := by
    rw [Real.log_mul (by norm_num) (by positivity), Real.log_pow]
    ring
  have hl2 : (1 : ℝ) / 2 < Real.log 2 := by
    have := Real.log_two_gt_d9
    linarith
  set ℓ := Real.log (2 * 2 ^ S) with hℓ
  set L : ℝ := 2 ^ S with hLdef
  have hℓ1 : (1 : ℝ) ≤ ℓ := by
    rw [hlog]; nlinarith
  have h2ℓ : (1 : ℝ) + S ≤ 2 * ℓ := by
    rw [hlog]; nlinarith
  have hLl : (1 : ℝ) ≤ L * ℓ := by nlinarith
  have hW : 0 ≤ u * L ^ 2 := by positivity
  have h1 : u * L ^ 2 * (1 + S) ≤ u * L ^ 2 * (2 * ℓ) :=
    mul_le_mul_of_nonneg_left h2ℓ hW
  have h2 : u * L ^ 2 * (2 * ℓ) ≤ u * L ^ 2 * (2 * ℓ) * (L * ℓ) :=
    le_mul_of_one_le_right (by positivity) hLl
  have h3 : 0 ≤ u * L ^ 3 * ℓ ^ 2 := by positivity
  nlinarith [h1, h2, h3]

/-- Eq. (16), assembled: from the chain of hypotheses in §5 to the 2^12 constant. -/
theorem bluestein_table_error (u : ℝ) (S : ℕ) (δ Eprod Einv Efin : ℝ)
    (hS : 1 ≤ S) (hu0 : 0 ≤ u) (hδ0 : 0 ≤ δ)
    (hsmall : 64 * u * (1 + S) ≤ 1)
    (hδ : δ ≤ 64 * u * 2 ^ S * (1 + S))
    (hP : Eprod ≤ 2 * 2 ^ S * δ + δ ^ 2 + 5 * u)
    (hI : Einv ≤ 3 / 2 * Eprod + 3 / 4 * u * ((2 : ℝ) ^ S) ^ 2 * S + 21 / 2 * u)
    (hC : Efin ≤ (1 + u) * Einv + u * 2 ^ S + 5 * u) :
    Efin ≤ 2 ^ 12 * u * ((2 : ℝ) ^ S) ^ 3 * (Real.log (2 * 2 ^ S)) ^ 2 := by
  have hSr : (0 : ℝ) ≤ S := Nat.cast_nonneg S
  have hL : (2 : ℝ) ≤ 2 ^ S := by
    calc (2 : ℝ) = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ S := pow_le_pow_right₀ (by norm_num) hS
  have hP' := prod_error u (2 ^ S) δ Eprod S hu0 hSr hL hδ0 hδ hsmall hP
  have hI' := inv_error u (2 ^ S) Einv Eprod S hu0 hSr hL hI hP'
  have hs2 : u * (1 + S) ≤ 1 / 64 := by linarith
  have hC' := chirp_error u (2 ^ S) Efin Einv S hu0 hSr hL hs2 hC hI'
  exact hC'.trans (chirp_to_final u S hu0 hS)



end DirichletDivisor

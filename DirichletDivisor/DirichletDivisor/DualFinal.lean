import DirichletDivisor.DualQuad

set_option linter.style.whitespace false

namespace DirichletDivisor

/-! # Clean numerical hypotheses

`hoyingSum_eq_round_quad` still carries two combined budget inequalities. Here they are replaced
by one tolerance each on the three primitive accuracies:

* quadrature `εQ ≤ N^{-5}/4`,
* phase factor `εφ ≤ N^{-5}/4`,
* variance `|η̂var − σ̂²| ≤ ε ≤ N^{-7}`.

The note's values (`N^{-29}`, `N^{-A-4}`, `N^{-Aq-2}`) satisfy these with enormous slack.
-/

theorem quad_simple (N εQ εφ : ℝ) (hN1 : 1 ≤ N) (hQ0 : 0 ≤ εQ) (hφ0 : 0 ≤ εφ)
    (hQ : εQ ≤ (N ^ 5)⁻¹ / 4) (hφ : εφ ≤ (N ^ 5)⁻¹ / 4) :
    (1 + εφ) * εQ + 2 * εφ ≤ (N ^ 5)⁻¹ := by
  have h5 : (1 : ℝ) ≤ N ^ 5 := one_le_pow₀ hN1
  have hr : (N ^ 5)⁻¹ ≤ 1 := by
    have := inv_anti₀ one_pos h5
    simpa using this
  have h1 : (1 + εφ) * εQ ≤ (5 / 4) * ((N ^ 5)⁻¹ / 4) :=
    mul_le_mul (by linarith) hQ hQ0 (by norm_num)
  linarith

theorem sigma_budget (N H ε : ℝ) (hH0 : 0 ≤ H) (hHN : H ≤ N) (hN1 : 1 ≤ N)
    (hε0 : 0 ≤ ε) (hε : ε ≤ (N ^ 7)⁻¹) :
    H ^ 2 * ε * (1 + Real.log (2 * N)) / 8 ≤ 2 ^ 58 * (N ^ 4)⁻¹ := by
  have hN0 : 0 < N := lt_of_lt_of_le one_pos hN1
  have hN0' : N ≠ 0 := hN0.ne'
  have hlog : 1 + Real.log (2 * N) ≤ 2 * N := by
    have := Real.log_le_sub_one_of_pos (by positivity : (0 : ℝ) < 2 * N)
    linarith
  have hlogn : 0 ≤ 1 + Real.log (2 * N) := by
    have : 0 ≤ Real.log (2 * N) := Real.log_nonneg (by linarith)
    linarith
  have hH2 : H ^ 2 ≤ N ^ 2 := pow_le_pow_left₀ hH0 hHN 2
  have hstep : H ^ 2 * ε * (1 + Real.log (2 * N)) ≤ N ^ 2 * (N ^ 7)⁻¹ * (2 * N) :=
    mul_le_mul (mul_le_mul hH2 hε hε0 (sq_nonneg N)) hlog hlogn (by positivity)
  have e : N ^ 2 * (N ^ 7)⁻¹ * (2 * N) = 2 * (N ^ 4)⁻¹ := by
    field_simp <;> ring
  have hq : 0 ≤ (N ^ 4)⁻¹ := by positivity
  rw [e] at hstep
  linarith

theorem H_facts (n : ℕ) (H : ℝ) (hH : H = ((n : ℝ) + 1) ^ (9 / 28 : ℝ))
    (hN : 2 ^ 65536 ≤ n + 1) : 0 ≤ H ∧ H ≤ (n : ℝ) + 1 := by
  have hN' : (2 : ℝ) ^ 65536 ≤ (n : ℝ) + 1 := by exact_mod_cast hN
  have hN2 : (2 : ℝ) ≤ (n : ℝ) + 1 := le_trans two_le_two_pow_65536 hN'
  have hN1 : (1 : ℝ) ≤ (n : ℝ) + 1 := le_trans (by norm_num) hN2
  have hN0 : (0 : ℝ) < (n : ℝ) + 1 := lt_of_lt_of_le one_pos hN1
  refine ⟨by rw [hH]; exact Real.rpow_nonneg hN0.le _, ?_⟩
  rw [hH]
  calc ((n : ℝ) + 1) ^ (9 / 28 : ℝ) ≤ ((n : ℝ) + 1) ^ (1 : ℝ) :=
        Real.rpow_le_rpow_of_exponent_le hN1 (by norm_num)
    _ = (n : ℝ) + 1 := Real.rpow_one _

/-- **Top-level statement (modulo `CitedInputs` and the three numerical primitives).**
For `N = n + 1 ≥ 2^65536`, `H = N^{9/28}`, `M ∈ [tailM0 N, tailM0 N + 1]`, the nearest integer to the
algorithm's output is the Hoying sum, provided the quadrature values, phase factors and variance
estimate are accurate to `N^{-5}/4`, `N^{-5}/4` and `N^{-7}` respectively. -/
theorem hoyingSum_eq_round_final (ci : CitedInputs) (n : ℕ) (H sigh ε εQ εφ : ℝ) (M : ℕ)
    (hH : H = ((n : ℝ) + 1) ^ (9 / 28 : ℝ))
    (hN : 2 ^ 65536 ≤ n + 1)
    (hsig0 : 0 ≤ sigh) (hsig1 : sigh ≤ 1)
    (hsigε : |etaVar - sigh| ≤ ε) (hε : ε ≤ (((n : ℝ) + 1) ^ 7)⁻¹)
    (hM1 : tailM0 ((n : ℝ) + 1) ≤ (M : ℝ)) (hM2 : (M : ℝ) ≤ tailM0 ((n : ℝ) + 1) + 1)
    (hεQ0 : 0 ≤ εQ) (hεφ0 : 0 ≤ εφ)
    (hεQ : εQ ≤ (((n : ℝ) + 1) ^ 5)⁻¹ / 4) (hεφ : εφ ≤ (((n : ℝ) + 1) ^ 5)⁻¹ / 4)
    (O : FsQuadOracle (parS (n + 1) H sigh) (parL (n + 1) H) M εQ εφ) :
    (hoyingSum n : ℤ)
      = round (idealAhat (parS (n + 1) H sigh) (parL (n + 1) H) (parK (n + 1) H sigh)
          (∑ j ∈ Finset.range M, O.comp j)) := by
  have hN' : (2 : ℝ) ^ 65536 ≤ (n : ℝ) + 1 := by exact_mod_cast hN
  have hN2 : (2 : ℝ) ≤ (n : ℝ) + 1 := le_trans two_le_two_pow_65536 hN'
  have hN1 : (1 : ℝ) ≤ (n : ℝ) + 1 := le_trans (by norm_num) hN2
  obtain ⟨hH0, hHN⟩ := H_facts n H hH hN
  have hε0 : 0 ≤ ε := (abs_nonneg _).trans hsigε
  exact hoyingSum_eq_round_quad ci n H sigh ε εQ εφ M hH hN hsig0 hsig1 hsigε hM1 hM2
    hεQ0 hεφ0 (quad_simple _ εQ εφ hN1 hεQ0 hεφ0 hεQ hεφ) O
    (sigma_budget _ H ε hH0 hHN hN1 hε0 hε)

end DirichletDivisor

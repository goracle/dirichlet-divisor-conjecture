import DirichletDivisor.DivisorTail
import DirichletDivisor.Stirling

set_option linter.style.whitespace false

namespace DirichletDivisor

/-! ## Exponential tail bound: choosing `k = ⌊y⌋` with `y = c M^{1/4}` -/

/-- `⌊y⌋! / y^⌊y⌋ ≤ e² y e^{-y}` for `y ≥ 4` (Stirling-type bound `r! e^r ≤ e r^{r+1}`). -/
theorem factorial_div_pow_le (y : ℝ) (hy : 4 ≤ y) :
    ((⌊y⌋₊).factorial : ℝ) / y ^ ⌊y⌋₊ ≤ Real.exp 2 * y * Real.exp (-y) := by
  have hk1 : 1 ≤ ⌊y⌋₊ := Nat.le_floor (by exact_mod_cast (by linarith : (1 : ℝ) ≤ y))
  have hkle : ((⌊y⌋₊ : ℕ) : ℝ) ≤ y := Nat.floor_le (by linarith)
  have hylt : y < ((⌊y⌋₊ : ℕ) : ℝ) + 1 := Nat.lt_floor_add_one y
  generalize ⌊y⌋₊ = k at hk1 hkle hylt ⊢
  have hk0 : (0 : ℝ) < k := by exact_mod_cast hk1
  have hy0 : 0 < y := by linarith
  have hst := factorial_mul_exp_le k hk1
  have e2 : Real.exp 2 = Real.exp 1 * Real.exp 1 := by
    rw [← Real.exp_add]
    norm_num
  have hpow : (k : ℝ) ^ (k + 1) ≤ y * y ^ k := by
    rw [pow_succ']
    exact mul_le_mul hkle (pow_le_pow_left₀ hk0.le hkle k) (pow_nonneg hk0.le k) hy0.le
  have hex : Real.exp (-(k : ℝ)) ≤ Real.exp 1 * Real.exp (-y) := by
    rw [← Real.exp_add]
    exact Real.exp_le_exp.2 (by linarith)
  rw [div_le_iff₀ (pow_pos hy0 k)]
  calc (k.factorial : ℝ)
      = ((k.factorial : ℝ) * Real.exp k) * Real.exp (-(k : ℝ)) := by
        rw [mul_assoc, ← Real.exp_add]
        simp
    _ ≤ (Real.exp 1 * (k : ℝ) ^ (k + 1)) * Real.exp (-(k : ℝ)) :=
        mul_le_mul_of_nonneg_right hst (Real.exp_pos _).le
    _ ≤ (Real.exp 1 * (y * y ^ k)) * (Real.exp 1 * Real.exp (-y)) :=
        mul_le_mul (mul_le_mul_of_nonneg_left hpow (Real.exp_pos 1).le) hex
          (Real.exp_pos _).le (by positivity)
    _ = Real.exp 2 * y * Real.exp (-y) * y ^ k := by
        rw [e2]
        ring

/-- Exponential tail bound for the weighted dual sum.  If `y = c M^{1/4} ≥ 4` then
`∑_{j<n} d(m_j) m_j^{-3/4} e^{-c m_j^{1/4}} ≤ 8 e² y M^{3/4} e^{-y}`, `m_j = M + j + 1`. -/
theorem tail_weighted_exp (c y : ℝ) (hc : 0 < c) (M n : ℕ) (hM : 1 ≤ M)
    (hyd : y = c * (M : ℝ) ^ (1 / 4 : ℝ)) (hy : 4 ≤ y) :
    ∑ j ∈ Finset.range n, (d (M + j + 1) : ℝ)
        * (((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ))
            * Real.exp (-(c * ((M + j + 1 : ℕ) : ℝ) ^ (1 / 4 : ℝ))))
      ≤ 8 * Real.exp 2 * y * (M : ℝ) ^ (3 / 4 : ℝ) * Real.exp (-y) := by
  have hMpos : (0 : ℝ) < M := by exact_mod_cast hM
  have hk4 : 4 ≤ ⌊y⌋₊ := Nat.le_floor (by exact_mod_cast hy)
  have hk4' : (4 : ℝ) ≤ (⌊y⌋₊ : ℝ) := by exact_mod_cast hk4
  have h1 := tail_weighted_partial c hc ⌊y⌋₊ hk4 M n hM
  have hfd := factorial_div_pow_le y hy
  have hq : (1 / 4 : ℝ) ≤ (1 / 4 + (⌊y⌋₊ : ℝ) / 4) - 1 := by linarith
  have hMp : 0 < (M : ℝ) ^ (1 - (1 / 4 + (⌊y⌋₊ : ℝ) / 4)) := Real.rpow_pos_of_pos hMpos _
  have hp4 : (M : ℝ) ^ (1 - (1 / 4 + (⌊y⌋₊ : ℝ) / 4)) / ((1 / 4 + (⌊y⌋₊ : ℝ) / 4) - 1)
      ≤ 4 * (M : ℝ) ^ (1 - (1 / 4 + (⌊y⌋₊ : ℝ) / 4)) := by
    rw [div_le_iff₀ (by linarith)]
    nlinarith [hMp, hq]
  have hid : ∀ k : ℕ, (k.factorial : ℝ) / c ^ k * (M : ℝ) ^ (1 - (1 / 4 + (k : ℝ) / 4))
      = (k.factorial : ℝ) / y ^ k * (M : ℝ) ^ (3 / 4 : ℝ) := by
    intro k
    have hmk : (M : ℝ) ^ ((k : ℝ) / 4) ≠ 0 := (Real.rpow_pos_of_pos hMpos _).ne'
    have hck : c ^ k ≠ 0 := (pow_pos hc _).ne'
    have hyk : y ^ k = c ^ k * (M : ℝ) ^ ((k : ℝ) / 4) := by
      rw [hyd, mul_pow, ← Real.rpow_natCast ((M : ℝ) ^ (1 / 4 : ℝ)) k,
        ← Real.rpow_mul hMpos.le,
        show (1 / 4 : ℝ) * (k : ℝ) = (k : ℝ) / 4 by ring]
    rw [hyk, show 1 - (1 / 4 + (k : ℝ) / 4) = 3 / 4 - (k : ℝ) / 4 by ring,
      Real.rpow_sub hMpos]
    field_simp
  calc ∑ j ∈ Finset.range n, (d (M + j + 1) : ℝ)
        * (((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ))
            * Real.exp (-(c * ((M + j + 1 : ℕ) : ℝ) ^ (1 / 4 : ℝ))))
      ≤ 2 * (((⌊y⌋₊).factorial : ℝ) / c ^ ⌊y⌋₊)
          * ((M : ℝ) ^ (1 - (1 / 4 + (⌊y⌋₊ : ℝ) / 4))
              / ((1 / 4 + (⌊y⌋₊ : ℝ) / 4) - 1)) := h1
    _ ≤ 2 * (((⌊y⌋₊).factorial : ℝ) / c ^ ⌊y⌋₊)
          * (4 * (M : ℝ) ^ (1 - (1 / 4 + (⌊y⌋₊ : ℝ) / 4))) :=
        mul_le_mul_of_nonneg_left hp4 (by positivity)
    _ = 8 * (((⌊y⌋₊).factorial : ℝ) / c ^ ⌊y⌋₊
          * (M : ℝ) ^ (1 - (1 / 4 + (⌊y⌋₊ : ℝ) / 4))) := by ring
    _ = 8 * (((⌊y⌋₊).factorial : ℝ) / y ^ ⌊y⌋₊ * (M : ℝ) ^ (3 / 4 : ℝ)) := by rw [hid ⌊y⌋₊]
    _ ≤ 8 * ((Real.exp 2 * y * Real.exp (-y)) * (M : ℝ) ^ (3 / 4 : ℝ)) :=
        mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_right hfd (Real.rpow_nonneg hMpos.le _)) (by norm_num)
    _ = 8 * Real.exp 2 * y * (M : ℝ) ^ (3 / 4 : ℝ) * Real.exp (-y) := by ring

end DirichletDivisor

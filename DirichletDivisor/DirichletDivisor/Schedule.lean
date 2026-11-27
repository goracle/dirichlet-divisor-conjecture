import DirichletDivisor.Fourier

set_option linter.style.whitespace false

namespace DirichletDivisor

/-- Λ = 3 + √10, the dominant root of x² = 6x + 1. -/
noncomputable def lam : ℝ := 3 + Real.sqrt 10

theorem lam_sq : lam ^ 2 = 6 * lam + 1 := by
  unfold lam
  have h := Real.sq_sqrt (show (0 : ℝ) ≤ 10 by norm_num)
  nlinarith [h]

theorem lam_gt : 6.16 < lam := by
  unfold lam
  have : (3.16 : ℝ) < Real.sqrt 10 := (Real.lt_sqrt (by norm_num)).2 (by norm_num)
  linarith

theorem lam_lt : lam < 6.17 := by
  unfold lam
  have : Real.sqrt 10 < (3.17 : ℝ) := (Real.sqrt_lt' (by norm_num)).2 (by norm_num)
  linarith

theorem lam_pos : 0 < lam := by linarith [lam_gt]

/-- Coefficient norm bound (18): L_{j+2} ≤ 6 L_{j+1} + L_j implies L_j ≤ Λ^j. -/
theorem coeff_norm_le (L : ℕ → ℝ) (h0 : L 0 ≤ 1) (h1 : L 1 ≤ lam)
    (hrec : ∀ j, L (j + 2) ≤ 6 * L (j + 1) + L j) :
    ∀ j, L j ≤ lam ^ j ∧ L (j + 1) ≤ lam ^ (j + 1) := by
  intro j
  induction j with
  | zero => exact ⟨by simpa using h0, by simpa using h1⟩
  | succ n ih =>
    refine ⟨ih.2, ?_⟩
    show L (n + 2) ≤ lam ^ (n + 2)
    have e : lam ^ (n + 2) = 6 * lam ^ (n + 1) + lam ^ n := by
      calc lam ^ (n + 2) = lam ^ n * lam ^ 2 := by ring
        _ = lam ^ n * (6 * lam + 1) := by rw [lam_sq]
        _ = 6 * lam ^ (n + 1) + lam ^ n := by ring
    have := hrec n
    linarith [ih.1, ih.2]

/-- α = log₂(Λ/2) and β = log₂ Λ. -/
noncomputable def alpha : ℝ := Real.logb 2 (lam / 2)
noncomputable def beta : ℝ := Real.logb 2 lam

theorem beta_eq : beta = alpha + 1 := by
  have hx : lam / 2 ≠ 0 := (div_pos lam_pos (by norm_num)).ne'
  have h : Real.logb 2 lam = Real.logb 2 (2 * (lam / 2)) := by
    congr 1
    ring
  unfold beta alpha
  rw [h, Real.logb_mul (by norm_num) hx, Real.logb_self_eq_one (by norm_num)]
  ring

theorem alpha_lt : alpha < 17 / 10 := by
  have hx : 0 < lam / 2 := div_pos lam_pos (by norm_num)
  have hlt : lam / 2 < 16 / 5 := by linarith [lam_lt]
  have h1 : (lam / 2) ^ 10 < (2 : ℝ) ^ 17 := by
    calc (lam / 2) ^ 10 < (16 / 5 : ℝ) ^ 10 := by
          first
            | exact pow_lt_pow_left hlt hx.le (by norm_num)
            | exact pow_lt_pow_left₀ hlt hx.le (by norm_num)
      _ < (2 : ℝ) ^ 17 := by norm_num
  have h2 := Real.logb_lt_logb (b := 2) (by norm_num) (pow_pos hx 10) h1
  rw [Real.logb_pow, Real.logb_pow, Real.logb_self_eq_one (by norm_num)] at h2
  unfold alpha
  push_cast at h2
  linarith

theorem alpha_gt : 3 / 2 < alpha := by
  have hx : 0 < lam / 2 := div_pos lam_pos (by norm_num)
  have hgt : 3 < lam / 2 := by linarith [lam_gt]
  have h1 : (2 : ℝ) ^ 3 < (lam / 2) ^ 2 := by nlinarith
  have h2 := Real.logb_lt_logb (b := 2) (by norm_num) (pow_pos (by norm_num) 3) h1
  rw [Real.logb_pow, Real.logb_pow, Real.logb_self_eq_one (by norm_num)] at h2
  unfold alpha
  push_cast at h2
  linarith

/-- The two precision exponents of §5/§6 as functions of A. -/
noncomputable def PTf (A : ℝ) : ℝ := A + 5 + 9 / 28 + 1 + alpha * (A + 4)
noncomputable def Phf (A : ℝ) : ℝ := A + 5 + 5 / 14 + beta * (A + 4)

theorem PTf_two_lt : PTf 2 < 19 := by
  have := alpha_lt
  unfold PTf
  linarith

theorem Phf_two_lt : Phf 2 < 24 := by
  have := alpha_lt
  unfold Phf
  rw [beta_eq]
  linarith

/-- The rounded-up implementation schedule at A = 2 covers the requirements:
bT = 23 log₂ N ≥ (PT + 2) log₂ N, bD = 28 log₂ N ≥ (Ph + 4) log₂ N,
Aq = 27 ≥ Ph + 2. -/
theorem schedule_ok :
    PTf 2 + 2 < 23 ∧ Phf 2 + 4 < 28 ∧ Phf 2 + 2 < 27 := by
  have h1 := PTf_two_lt
  have h2 := Phf_two_lt
  refine ⟨by linarith, by linarith, by linarith⟩

end DirichletDivisor

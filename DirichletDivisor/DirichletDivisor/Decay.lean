import DirichletDivisor.Stirling

set_option linter.style.whitespace false

namespace DirichletDivisor

/-- The optimisation step: from the family of bounds Φ ≤ 16 (r!)² / X^{2r} (all r),
the trivial bound Φ ≤ 2, and y ≤ 1.64 X, conclude Φ ≤ 2^16 e^{-y}. -/
theorem decay_optimise (Φ X y : ℝ) (hX : 0 < X) (hy0 : 0 ≤ y) (hy : y ≤ 1.64 * X)
    (hΦ0 : 0 ≤ Φ) (htriv : Φ ≤ 2)
    (hr : ∀ r : ℕ, Φ ≤ 16 * ((r.factorial : ℝ)) ^ 2 / X ^ (2 * r)) :
    Φ ≤ 2 ^ 16 * Real.exp (-y) := by
  by_cases hX1 : X < 1
  · have h1 := Real.add_one_le_exp (-y / 2)
    have h2 : Real.exp (-y) = Real.exp (-y / 2) * Real.exp (-y / 2) := by
      rw [← Real.exp_add]
      ring_nf
    have h3 : (0.18 : ℝ) ≤ Real.exp (-y / 2) := by linarith
    have h4 : (0.18 : ℝ) * 0.18 ≤ Real.exp (-y / 2) * Real.exp (-y / 2) :=
      mul_self_le_mul_self (by norm_num) h3
    rw [h2]
    nlinarith [h4]
  · push_neg at hX1
    have hr1 : 1 ≤ ⌊X⌋₊ := Nat.le_floor (by simpa using hX1)
    have hrX : ((⌊X⌋₊ : ℕ) : ℝ) ≤ X := Nat.floor_le hX.le
    have hXr : X < ((⌊X⌋₊ : ℕ) : ℝ) + 1 := Nat.lt_floor_add_one X
    generalize ⌊X⌋₊ = r at hr1 hrX hXr
    have hr0 : (0 : ℝ) < r := Nat.cast_pos.2 hr1
    have h1 : Φ * X ^ (2 * r) ≤ 16 * ((r.factorial : ℝ)) ^ 2 := by
      have := hr r
      rwa [le_div_iff₀ (by positivity)] at this
    have h2 : (r : ℝ) ^ (2 * r) ≤ X ^ (2 * r) := pow_le_pow_left₀ hr0.le hrX _
    have h3 := factorial_sq_mul_exp_le r hr1
    have hE : 0 < Real.exp (r : ℝ) := Real.exp_pos _
    have he1pos : 0 < Real.exp 1 := Real.exp_pos 1
    have a1 : (r : ℝ) ^ (2 * r) * Φ ≤ Φ * X ^ (2 * r) := by
      have := mul_le_mul_of_nonneg_left h2 hΦ0
      linarith
    have a2 : (r : ℝ) ^ (2 * r) * Φ * Real.exp (r : ℝ) ^ 2
        ≤ 16 * ((r.factorial : ℝ)) ^ 2 * Real.exp (r : ℝ) ^ 2 :=
      mul_le_mul_of_nonneg_right (a1.trans h1) (sq_nonneg _)
    have a3 : 16 * ((r.factorial : ℝ)) ^ 2 * Real.exp (r : ℝ) ^ 2
        ≤ 16 * (Real.exp 1 * (r : ℝ) ^ (r + 1)) ^ 2 := by
      calc 16 * ((r.factorial : ℝ)) ^ 2 * Real.exp (r : ℝ) ^ 2
          = 16 * ((r.factorial : ℝ) * Real.exp (r : ℝ)) ^ 2 := by ring
        _ ≤ 16 * (Real.exp 1 * (r : ℝ) ^ (r + 1)) ^ 2 := by linarith [h3]
    have a4 : (Real.exp 1 * (r : ℝ) ^ (r + 1)) ^ 2
        = (r : ℝ) ^ (2 * r) * (Real.exp 1 ^ 2 * (r : ℝ) ^ 2) := by ring
    have h4 : (r : ℝ) ^ (2 * r) * (Φ * Real.exp (r : ℝ) ^ 2)
        ≤ (r : ℝ) ^ (2 * r) * (16 * Real.exp 1 ^ 2 * (r : ℝ) ^ 2) := by
      calc (r : ℝ) ^ (2 * r) * (Φ * Real.exp (r : ℝ) ^ 2)
          = (r : ℝ) ^ (2 * r) * Φ * Real.exp (r : ℝ) ^ 2 := by ring
        _ ≤ 16 * ((r.factorial : ℝ)) ^ 2 * Real.exp (r : ℝ) ^ 2 := a2
        _ ≤ 16 * (Real.exp 1 * (r : ℝ) ^ (r + 1)) ^ 2 := a3
        _ = (r : ℝ) ^ (2 * r) * (16 * Real.exp 1 ^ 2 * (r : ℝ) ^ 2) := by
            rw [a4]; ring
    have hkey : Φ * Real.exp (r : ℝ) ^ 2 ≤ 16 * Real.exp 1 ^ 2 * (r : ℝ) ^ 2 :=
      le_of_mul_le_mul_left h4 (pow_pos hr0 _)
    -- exponent bookkeeping
    have hyr : y ≤ (r : ℝ) + r + (2 - 0.36 * X) := by linarith
    have hexp2 : Real.exp 2 = Real.exp 1 * Real.exp 1 := by
      rw [show (2 : ℝ) = 1 + 1 by norm_num, Real.exp_add]
    have hexp : Real.exp y ≤ Real.exp (r : ℝ) * Real.exp (r : ℝ) *
        (Real.exp 1 * Real.exp 1 * Real.exp (-(0.36 * X))) := by
      calc Real.exp y ≤ Real.exp ((r : ℝ) + r + (2 - 0.36 * X)) := Real.exp_le_exp.2 hyr
        _ = Real.exp (r : ℝ) * Real.exp (r : ℝ) *
            (Real.exp 1 * Real.exp 1 * Real.exp (-(0.36 * X))) := by
          rw [Real.exp_add, Real.exp_add, ← hexp2,
            show (2 : ℝ) - 0.36 * X = 2 + -(0.36 * X) by ring, Real.exp_add]
    -- X² e^{-0.36 X} ≤ 16
    have ht : (0 : ℝ) ≤ 0.36 * X := by positivity
    have hq := Real.quadratic_le_exp_of_nonneg ht
    have hp : X ^ 2 ≤ 16 * Real.exp (0.36 * X) := by nlinarith [hq, sq_nonneg X]
    have hqp : Real.exp (-(0.36 * X)) * Real.exp (0.36 * X) = 1 := by
      rw [← Real.exp_add]; simp
    have hq0 : 0 < Real.exp (-(0.36 * X)) := Real.exp_pos _
    have hXq : X ^ 2 * Real.exp (-(0.36 * X)) ≤ 16 := by
      calc X ^ 2 * Real.exp (-(0.36 * X))
          ≤ 16 * Real.exp (0.36 * X) * Real.exp (-(0.36 * X)) :=
            mul_le_mul_of_nonneg_right hp hq0.le
        _ = 16 := by rw [mul_assoc, mul_comm (Real.exp (0.36 * X)), hqp]; ring
    have hr2 : (r : ℝ) ^ 2 ≤ X ^ 2 := pow_le_pow_left₀ hr0.le hrX 2
    have he1 : Real.exp 1 < 2.72 := by
      have := Real.exp_one_lt_d9
      linarith
    have he4 : Real.exp 1 ^ 4 ≤ (2.72 : ℝ) ^ 4 := pow_le_pow_left₀ he1pos.le he1.le 4
    have hΦy : Φ * Real.exp y ≤ 2 ^ 16 := by
      calc Φ * Real.exp y
          ≤ Φ * (Real.exp (r : ℝ) * Real.exp (r : ℝ) *
              (Real.exp 1 * Real.exp 1 * Real.exp (-(0.36 * X)))) :=
            mul_le_mul_of_nonneg_left hexp hΦ0
        _ = (Φ * Real.exp (r : ℝ) ^ 2) *
              (Real.exp 1 * Real.exp 1 * Real.exp (-(0.36 * X))) := by ring
        _ ≤ (16 * Real.exp 1 ^ 2 * (r : ℝ) ^ 2) *
              (Real.exp 1 * Real.exp 1 * Real.exp (-(0.36 * X))) :=
            mul_le_mul_of_nonneg_right hkey (by positivity)
        _ ≤ (16 * Real.exp 1 ^ 2 * X ^ 2) *
              (Real.exp 1 * Real.exp 1 * Real.exp (-(0.36 * X))) :=
            mul_le_mul_of_nonneg_right
              (mul_le_mul_of_nonneg_left hr2 (by positivity)) (by positivity)
        _ = 16 * Real.exp 1 ^ 4 * (X ^ 2 * Real.exp (-(0.36 * X))) := by ring
        _ ≤ 16 * Real.exp 1 ^ 4 * 16 :=
            mul_le_mul_of_nonneg_left hXq (by positivity)
        _ = 256 * Real.exp 1 ^ 4 := by ring
        _ ≤ 256 * (2.72 : ℝ) ^ 4 := mul_le_mul_of_nonneg_left he4 (by norm_num)
        _ ≤ 2 ^ 16 := by norm_num
    calc Φ = Φ * Real.exp y * Real.exp (-y) := by
          rw [mul_assoc, ← Real.exp_add]; simp
      _ ≤ 2 ^ 16 * Real.exp (-y) :=
          mul_le_mul_of_nonneg_right hΦy (Real.exp_pos _).le

/-- Link between the two scales: with X² = 2π a / (38/9), we have 2√a ≤ 1.64 X. -/
theorem sqrt_le_scale (a : ℝ) (ha : 0 ≤ a) :
    2 * Real.sqrt a ≤ 1.64 * Real.sqrt (2 * Real.pi * a / (38 / 9)) := by
  have hX0 : 0 ≤ 2 * Real.pi * a / (38 / 9) := by
    have := Real.pi_pos
    positivity
  have hX2 := Real.sq_sqrt hX0
  have ha2 := Real.sq_sqrt ha
  have hXn := Real.sqrt_nonneg (2 * Real.pi * a / (38 / 9))
  have han := Real.sqrt_nonneg a
  have hpi := Real.pi_gt_d2
  have hlow : (1.4873 : ℝ) * a ≤ 2 * Real.pi * a / (38 / 9) := by
    rw [le_div_iff₀ (by norm_num)]
    nlinarith [mul_nonneg ha (by linarith : (0 : ℝ) ≤ Real.pi - 3.14)]
  generalize Real.sqrt (2 * Real.pi * a / (38 / 9)) = X at hX2 hXn ⊢
  generalize Real.sqrt a = s at ha2 han ⊢
  by_contra h
  push_neg at h
  have h' : 0.82 * X < s := by linarith
  have h3 := mul_self_lt_mul_self (by positivity) h'
  nlinarith [h3, hX2, ha2, hlow]

end DirichletDivisor

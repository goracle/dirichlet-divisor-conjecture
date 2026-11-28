import DirichletDivisor.Final

set_option linter.style.whitespace false

namespace DirichletDivisor

/-! # Why the runtime is `N^{9/28}`: the shell/block cost arithmetic

For a dyadic shell `X ∈ [N^{9/28}, M]`, `M ≤ N^{5/14}·P` (`P = 1024 (ln N)^4` is the polylog
factor), the note uses blocks of length `K_X = ⌊N^{-11/112} X^{3/8}⌋`. The cost components are

* number of blocks per shell  ~ `X / K_X`,
* moment-table size           ~ `K_X^6 · min(K_X^4, 1 + A_X)` with `A_X = √N K_X^4 X^{-3/2}`,
* terminal (incomplete) blocks ~ `K_X · √X`.

Here every one is bounded by `N^{9/28}` times a power of `P`. All proofs are exponent bookkeeping in
`exp`/`log` form (so `linarith` finishes them).
-/

/-- Unfloored block length `N^{-11/112} X^{3/8}`. -/
noncomputable def blkLen (N X : ℝ) : ℝ := N ^ (-(11 / 112 : ℝ)) * X ^ (3 / 8 : ℝ)

theorem blkLen_exp (N X : ℝ) (hN : 0 < N) (hX : 0 < X) :
    blkLen N X = Real.exp (Real.log N * (-(11 / 112)) + Real.log X * (3 / 8)) := by
  unfold blkLen
  rw [Real.rpow_def_of_pos hN, Real.rpow_def_of_pos hX, ← Real.exp_add]

theorem logX_le (N P X : ℝ) (hN : 0 < N) (hP : 1 ≤ P) (hX : 0 < X)
    (hXhi : X ≤ N ^ (5 / 14 : ℝ) * P) :
    Real.log X ≤ 5 / 14 * Real.log N + Real.log P := by
  have hP0 : 0 < P := lt_of_lt_of_le one_pos hP
  have h := Real.log_le_log hX hXhi
  rwa [Real.log_mul (Real.rpow_pos_of_pos hN _).ne' hP0.ne', Real.log_rpow hN] at h

theorem logX_ge (N X : ℝ) (hN : 0 < N) (hXlo : N ^ (9 / 28 : ℝ) ≤ X) :
    9 / 28 * Real.log N ≤ Real.log X := by
  have h := Real.log_le_log (Real.rpow_pos_of_pos hN _) hXlo
  rwa [Real.log_rpow hN] at h

/-- Block length is at most `N^{1/28}` (up to `P^{3/8}`). -/
theorem shell_len (N P X : ℝ) (hN : 0 < N) (hP : 1 ≤ P) (hX : 0 < X)
    (hXhi : X ≤ N ^ (5 / 14 : ℝ) * P) :
    blkLen N X ≤ N ^ (1 / 28 : ℝ) * P ^ (3 / 8 : ℝ) := by
  have hP0 : 0 < P := lt_of_lt_of_le one_pos hP
  have hx := logX_le N P X hN hP hX hXhi
  rw [blkLen_exp N X hN hX, Real.rpow_def_of_pos hN, Real.rpow_def_of_pos hP0,
    ← Real.exp_add]
  exact Real.exp_le_exp.2 (by linarith)

/-- Block length is at least 1 once `X ≥ N^{9/28}` (indeed `≥ N^{5/224}`). -/
theorem blkLen_ge_one (N X : ℝ) (hN1 : 1 ≤ N) (hX : 0 < X)
    (hXlo : N ^ (9 / 28 : ℝ) ≤ X) : 1 ≤ blkLen N X := by
  have hN : 0 < N := lt_of_lt_of_le one_pos hN1
  have hx := logX_ge N X hN hXlo
  have ha : 0 ≤ Real.log N := Real.log_nonneg hN1
  rw [blkLen_exp N X hN hX]
  exact Real.one_le_exp (by linarith)

/-- Number of blocks per shell: `X / K_X ≤ N^{9/28} P^{5/8}`. -/
theorem shell_blocks (N P X : ℝ) (hN : 0 < N) (hP : 1 ≤ P) (hX : 0 < X)
    (hXhi : X ≤ N ^ (5 / 14 : ℝ) * P) :
    X / blkLen N X ≤ N ^ (9 / 28 : ℝ) * P ^ (5 / 8 : ℝ) := by
  have hP0 : 0 < P := lt_of_lt_of_le one_pos hP
  have hx := logX_le N P X hN hP hX hXhi
  rw [blkLen_exp N X hN hX, Real.rpow_def_of_pos hN, Real.rpow_def_of_pos hP0,
    ← Real.exp_add]
  calc X / Real.exp (Real.log N * (-(11 / 112)) + Real.log X * (3 / 8))
      = Real.exp (Real.log X - (Real.log N * (-(11 / 112)) + Real.log X * (3 / 8))) := by
        rw [Real.exp_sub, Real.exp_log hX]
    _ ≤ _ := Real.exp_le_exp.2 (by linarith)

/-- The key cancellation: `A_X = √N K_X^4 X^{-3/2} = N^{3/28}` exactly (independent of `X`). -/
theorem shell_A (N X : ℝ) (hN : 0 < N) (hX : 0 < X) :
    Real.sqrt N * blkLen N X ^ 4 * X ^ (-(3 / 2 : ℝ)) = N ^ (3 / 28 : ℝ) := by
  rw [blkLen_exp N X hN hX, Real.sqrt_eq_rpow]
  simp only [Real.rpow_def_of_pos hN, Real.rpow_def_of_pos hX]
  rw [← Real.exp_nat_mul, ← Real.exp_add, ← Real.exp_add]
  congr 1
  push_cast
  ring

/-- Moment-table size: `K_X^6 · N^{3/28} ≤ N^{9/28} P^{9/4}` (and `min(K^4, 1+A) ≤ 1 + A`,
`1 + A ≤ 2 N^{3/28}` for `N ≥ 1`). -/
theorem shell_table (N P X : ℝ) (hN : 0 < N) (hP : 1 ≤ P) (hX : 0 < X)
    (hXhi : X ≤ N ^ (5 / 14 : ℝ) * P) :
    blkLen N X ^ 6 * N ^ (3 / 28 : ℝ) ≤ N ^ (9 / 28 : ℝ) * P ^ (9 / 4 : ℝ) := by
  have hP0 : 0 < P := lt_of_lt_of_le one_pos hP
  have hx := logX_le N P X hN hP hX hXhi
  rw [blkLen_exp N X hN hX]
  simp only [Real.rpow_def_of_pos hN, Real.rpow_def_of_pos hP0]
  rw [← Real.exp_nat_mul, ← Real.exp_add, ← Real.exp_add]
  refine Real.exp_le_exp.2 ?_
  push_cast
  linarith

theorem one_add_A_le (N : ℝ) (hN1 : 1 ≤ N) : 1 + N ^ (3 / 28 : ℝ) ≤ 2 * N ^ (3 / 28 : ℝ) := by
  have : (1 : ℝ) ≤ N ^ (3 / 28 : ℝ) := Real.one_le_rpow hN1 (by norm_num)
  linarith

/-- Terminal incomplete blocks: `K_X √X ≤ N^{3/14} P^{7/8}` (exponent `6/28 < 9/28`). -/
theorem shell_terminal (N P X : ℝ) (hN : 0 < N) (hP : 1 ≤ P) (hX : 0 < X)
    (hXhi : X ≤ N ^ (5 / 14 : ℝ) * P) :
    blkLen N X * Real.sqrt X ≤ N ^ (3 / 14 : ℝ) * P ^ (7 / 8 : ℝ) := by
  have hP0 : 0 < P := lt_of_lt_of_le one_pos hP
  have hx := logX_le N P X hN hP hX hXhi
  rw [blkLen_exp N X hN hX, Real.sqrt_eq_rpow]
  simp only [Real.rpow_def_of_pos hN, Real.rpow_def_of_pos hP0, Real.rpow_def_of_pos hX]
  rw [← Real.exp_add, ← Real.exp_add]
  exact Real.exp_le_exp.2 (by linarith)

/-! ## From the real block length to `⌊·⌋` -/
theorem floor_ge_half (K : ℝ) (hK : 1 ≤ K) : K / 2 ≤ (⌊K⌋₊ : ℝ) := by
  have h1 : (1 : ℝ) ≤ ⌊K⌋₊ := by
    exact_mod_cast (Nat.le_floor (by simpa using hK : (1 : ℝ) ≤ K))
  have h2 := Nat.lt_floor_add_one K
  linarith


/-- With the actual (floored) block length the block count at most doubles. -/
theorem shell_blocks_floor (N P X : ℝ) (hN1 : 1 ≤ N) (hP : 1 ≤ P) (hX : 0 < X)
    (hXlo : N ^ (9 / 28 : ℝ) ≤ X) (hXhi : X ≤ N ^ (5 / 14 : ℝ) * P) :
    X / (⌊blkLen N X⌋₊ : ℝ) ≤ 2 * (N ^ (9 / 28 : ℝ) * P ^ (5 / 8 : ℝ)) := by
  have hN : 0 < N := lt_of_lt_of_le one_pos hN1
  have hK1 := blkLen_ge_one N X hN1 hX hXlo
  have hKpos : 0 < blkLen N X / 2 := by linarith
  have hfl := floor_ge_half _ hK1
  have h1 : X / (⌊blkLen N X⌋₊ : ℝ) ≤ X / (blkLen N X / 2) := by
    first
      | exact div_le_div_of_nonneg_left hX.le hKpos hfl
      | exact div_le_div_of_le_left hX.le hKpos hfl
  have h2 : X / (blkLen N X / 2) = 2 * (X / blkLen N X) := by
    have : blkLen N X ≠ 0 := by linarith
    field_simp
  rw [h2] at h1
  have h3 := shell_blocks N P X hN hP hX hXhi
  linarith

/-! ## Rows: the number of `(row, block)` pairs in a shell -/

theorem row_count (X K : ℝ) (hX : 0 ≤ X) (hK : 0 < K) (R : ℕ) :
    ∑ a ∈ Finset.Icc 1 R, (X / ((a : ℝ) * K) + 1)
      ≤ (X / K) * (1 + Real.log R) + R := by
  have hH : ∑ a ∈ Finset.Icc 1 R, ((a : ℝ))⁻¹ = (harmonic R : ℝ) := by
    simp [harmonic_eq_sum_Icc]
  have e : ∀ a ∈ Finset.Icc 1 R, X / ((a : ℝ) * K) + 1 = (X / K) * ((a : ℝ))⁻¹ + 1 := by
    intro a ha
    have ha1 : 1 ≤ a := (Finset.mem_Icc.1 ha).1
    have : (a : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
    field_simp
  have hs1 : ∑ a ∈ Finset.Icc 1 R, (1 : ℝ) = R := by simp
  rw [Finset.sum_congr rfl e, Finset.sum_add_distrib, ← Finset.mul_sum, hH, hs1]
  have := mul_le_mul_of_nonneg_left (harmonic_le_one_add_log R) (div_nonneg hX hK.le)
  linarith

end DirichletDivisor

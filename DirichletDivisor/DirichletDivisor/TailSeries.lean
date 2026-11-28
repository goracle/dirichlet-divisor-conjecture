import DirichletDivisor.TailSum

set_option linter.style.whitespace false

open intervalIntegral

namespace DirichletDivisor

/-! ## p-series tail via integral comparison, and the resulting exponential-tail bound -/

/-- Partial sums of the `p`-series tail past `M`:
`∑_{k<n} (M+k+1)^{-p} ≤ M^{1-p}/(p-1)` for `p > 1`, `M ≥ 1`. -/
theorem pseries_partial_le (p : ℝ) (hp : 1 < p) (M n : ℕ) (hM : 1 ≤ M) :
    ∑ k ∈ Finset.range n, ((M + k + 1 : ℕ) : ℝ) ^ (-p) ≤ (M : ℝ) ^ (1 - p) / (p - 1) := by
  have hMpos : (0 : ℝ) < M := by exact_mod_cast hM
  have hanti : AntitoneOn (fun x : ℝ => x ^ (-p)) (Set.Icc (M : ℝ) ((M + n : ℕ) : ℝ)) := by
    intro x hx y hy hxy
    have hx0 : 0 < x := lt_of_lt_of_le hMpos hx.1
    have hy0 : 0 ≤ y := hx0.le.trans hxy
    dsimp only
    rw [Real.rpow_neg hx0.le, Real.rpow_neg hy0]
    exact inv_anti₀ (Real.rpow_pos_of_pos hx0 _) (Real.rpow_le_rpow hx0.le hxy (by linarith))
  have hsum := AntitoneOn.sum_le_integral_Ico (by omega : M ≤ M + n) hanti
  rw [Finset.sum_Ico_eq_sum_range, Nat.add_sub_cancel_left] at hsum
  have h0 : (0 : ℝ) ∉ Set.uIcc (M : ℝ) ((M + n : ℕ) : ℝ) := by
    rw [Set.uIcc_of_le (by exact_mod_cast Nat.le_add_right M n)]
    intro h
    linarith [h.1]
  have hint : ∫ x in (M : ℝ)..((M + n : ℕ) : ℝ), x ^ (-p)
      = (((M + n : ℕ) : ℝ) ^ (-p + 1) - (M : ℝ) ^ (-p + 1)) / (-p + 1) :=
    integral_rpow (Or.inr ⟨by intro h; linarith, h0⟩)
  rw [hint] at hsum
  have hE : 0 ≤ ((M + n : ℕ) : ℝ) ^ (-p + 1) := Real.rpow_nonneg (by positivity) _
  have hc : 0 < p - 1 := by linarith
  rw [show (1 : ℝ) - p = -p + 1 by ring]
  calc ∑ k ∈ Finset.range n, ((M + k + 1 : ℕ) : ℝ) ^ (-p)
      ≤ (((M + n : ℕ) : ℝ) ^ (-p + 1) - (M : ℝ) ^ (-p + 1)) / (-p + 1) := hsum
    _ = ((M : ℝ) ^ (-p + 1) - ((M + n : ℕ) : ℝ) ^ (-p + 1)) / (p - 1) := by
        rw [show -p + 1 = -(p - 1) by ring, div_neg, ← neg_div, neg_sub]
    _ ≤ (M : ℝ) ^ (-p + 1) / (p - 1) :=
        (div_le_div_iff_of_pos_right hc).2 (sub_le_self _ hE)

/-- The full `p`-series tail: `∑'_{k} (M+k+1)^{-p} ≤ M^{1-p}/(p-1)`. -/
theorem pseries_tsum_le (p : ℝ) (hp : 1 < p) (M : ℕ) (hM : 1 ≤ M) :
    ∑' k : ℕ, ((M + k + 1 : ℕ) : ℝ) ^ (-p) ≤ (M : ℝ) ^ (1 - p) / (p - 1) :=
  Real.tsum_le_of_sum_range_le (fun k => Real.rpow_nonneg (by positivity) _)
    (fun n => pseries_partial_le p hp M n hM)

/-- Exponential-tail bound, partial sums: for `k ≥ 2`, with `p = 3/4 + k/4`,
`∑_{j<n} m_j^{-3/4} e^{-c m_j^{1/4}} ≤ (k!/c^k) · M^{1-p}/(p-1)`, where `m_j = M+j+1`. -/
theorem tail_exp_partial (c : ℝ) (hc : 0 < c) (k : ℕ) (hk : 2 ≤ k) (M n : ℕ) (hM : 1 ≤ M) :
    ∑ j ∈ Finset.range n, ((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ))
        * Real.exp (-(c * ((M + j + 1 : ℕ) : ℝ) ^ (1 / 4 : ℝ)))
      ≤ (k.factorial : ℝ) / c ^ k
          * ((M : ℝ) ^ (1 - (3 / 4 + (k : ℝ) / 4)) / ((3 / 4 + (k : ℝ) / 4) - 1)) := by
  have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
  have hp : 1 < (3 / 4 + (k : ℝ) / 4) := by linarith
  calc ∑ j ∈ Finset.range n, ((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ))
        * Real.exp (-(c * ((M + j + 1 : ℕ) : ℝ) ^ (1 / 4 : ℝ)))
      ≤ ∑ j ∈ Finset.range n, (k.factorial : ℝ) / c ^ k
          * ((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 + (k : ℝ) / 4)) := by
        apply Finset.sum_le_sum
        intro j _
        have h := term_le c hc k ((M + j + 1 : ℕ) : ℝ) (by positivity)
        rwa [show -(3 / 4 : ℝ) - (k : ℝ) / 4 = -(3 / 4 + (k : ℝ) / 4) by ring] at h
    _ = (k.factorial : ℝ) / c ^ k
          * ∑ j ∈ Finset.range n, ((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 + (k : ℝ) / 4)) :=
        (Finset.mul_sum _ _ _).symm
    _ ≤ (k.factorial : ℝ) / c ^ k
          * ((M : ℝ) ^ (1 - (3 / 4 + (k : ℝ) / 4)) / ((3 / 4 + (k : ℝ) / 4) - 1)) :=
        mul_le_mul_of_nonneg_left (pseries_partial_le _ hp M n hM) (by positivity)

end DirichletDivisor

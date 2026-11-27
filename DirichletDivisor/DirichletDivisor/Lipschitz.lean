import DirichletDivisor.DivisorSum

open Finset

namespace DirichletDivisor

/-- ∑_{a ≤ M} 1/a is the harmonic number. -/
theorem sum_Icc_inv_eq_harmonic (M : ℕ) :
    ∑ a ∈ Icc 1 M, (1 : ℝ) / a = (harmonic M : ℝ) := by
  induction M with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_Icc_succ_top (by omega), ih, harmonic_succ]
    push_cast <;> ring

/-- ∑_{m ≤ M} d(m) = #{(a,b) : ab ≤ M}. -/
theorem sum_d_eq_card (M : ℕ) :
    ∑ m ∈ Icc 1 M, (d m : ℝ)
      = ((Icc 1 M ×ˢ Icc 1 M).filter (fun p => p.1 * p.2 ≤ M)).card := by
  have h := sum_d_mul_eq_sum_pairs M (fun _ => (1 : ℝ))
  simp only [mul_one, Finset.sum_const, nsmul_eq_mul] at h
  exact h

/-- #{(a,b) : ab ≤ M} = ∑_{a ≤ M} ⌊M/a⌋. -/
theorem card_pairs_eq (M : ℕ) :
    ((Icc 1 M ×ˢ Icc 1 M).filter (fun p => p.1 * p.2 ≤ M)).card
      = ∑ a ∈ Icc 1 M, M / a := by
  rw [Finset.card_filter, Finset.sum_product]
  refine Finset.sum_congr rfl fun a ha => ?_
  show ∑ b ∈ Icc 1 M, (if a * b ≤ M then 1 else 0) = M / a
  rw [← Finset.card_filter (fun b => a * b ≤ M) (Icc 1 M)]
  have ha1 : 0 < a := (Finset.mem_Icc.mp ha).1
  have hset : (Icc 1 M).filter (fun b => a * b ≤ M) = Icc 1 (M / a) := by
    ext b
    simp only [Finset.mem_filter, Finset.mem_Icc]
    constructor
    · rintro ⟨⟨hb1, _⟩, hab⟩
      exact ⟨hb1, (Nat.le_div_iff_mul_le ha1).2 (by rw [mul_comm]; exact hab)⟩
    · rintro ⟨hb1, hb2⟩
      have h1 : b * a ≤ M := (Nat.le_div_iff_mul_le ha1).1 hb2
      refine ⟨⟨hb1, ?_⟩, by rw [mul_comm]; exact h1⟩
      calc b ≤ b * a := Nat.le_mul_of_pos_right b ha1
        _ ≤ M := h1
  rw [hset]
  simp

/-- ∑_{k ≤ M} d(k) ≤ M (1 + log M). -/
theorem sum_d_le (M : ℕ) :
    ∑ m ∈ Icc 1 M, (d m : ℝ) ≤ M * (1 + Real.log M) := by
  rw [sum_d_eq_card, card_pairs_eq, Nat.cast_sum]
  calc ∑ a ∈ Icc 1 M, ((M / a : ℕ) : ℝ)
      ≤ ∑ a ∈ Icc 1 M, (M : ℝ) * (1 / a) := by
        refine Finset.sum_le_sum fun a _ => ?_
        rw [mul_one_div]
        exact Nat.cast_div_le
    _ = M * (harmonic M : ℝ) := by
        rw [← Finset.mul_sum, sum_Icc_inv_eq_harmonic]
    _ ≤ M * (1 + Real.log M) :=
        mul_le_mul_of_nonneg_left (harmonic_le_one_add_log M) (Nat.cast_nonneg M)

/-- R(x) = ∑_{k ≤ K} d(k) max(x-k, 0) for any K ≥ ⌊x⌋. -/
theorem R_eq_sum_max (K : ℕ) (x : ℝ) (hK : ⌊x⌋₊ ≤ K) :
    R x = ∑ k ∈ Icc 1 K, (d k : ℝ) * max (x - k) 0 := by
  unfold R
  have hsub : Icc 1 ⌊x⌋₊ ⊆ Icc 1 K := Finset.Icc_subset_Icc le_rfl hK
  calc ∑ k ∈ Icc 1 ⌊x⌋₊, (d k : ℝ) * (x - k)
      = ∑ k ∈ Icc 1 ⌊x⌋₊, (d k : ℝ) * max (x - k) 0 := by
        refine Finset.sum_congr rfl fun k hk => ?_
        have hk1 : 1 ≤ k := (Finset.mem_Icc.mp hk).1
        have hk2 : k ≤ ⌊x⌋₊ := (Finset.mem_Icc.mp hk).2
        have hkx : (k : ℝ) ≤ x := (Nat.le_floor_iff' (by omega)).1 hk2
        rw [max_eq_left (by linarith)]
    _ = ∑ k ∈ Icc 1 K, (d k : ℝ) * max (x - k) 0 := by
        refine Finset.sum_subset hsub ?_
        intro k hkK hkn
        have hk1 : 1 ≤ k := (Finset.mem_Icc.mp hkK).1
        have hk2 : ¬ (k ≤ ⌊x⌋₊) := fun h => hkn (Finset.mem_Icc.mpr ⟨hk1, h⟩)
        have hkx : x < (k : ℝ) := (Nat.floor_lt' (by omega)).1 (by omega)
        rw [max_eq_right (by linarith), mul_zero]

/-- Lipschitz bound with constant ∑_{k ≤ K} d(k). -/
theorem R_lipschitz_aux (K : ℕ) (x y : ℝ) (hx : ⌊x⌋₊ ≤ K) (hy : ⌊y⌋₊ ≤ K) :
    |R x - R y| ≤ |x - y| * ∑ k ∈ Icc 1 K, (d k : ℝ) := by
  rw [R_eq_sum_max K x hx, R_eq_sum_max K y hy, ← Finset.sum_sub_distrib, Finset.mul_sum]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k _ => ?_)
  have h1 : |max (x - k) 0 - max (y - k) 0| ≤ |x - y| := by
    calc |max (x - k) 0 - max (y - k) 0| ≤ |(x - k) - (y - k)| :=
          abs_max_sub_max_le_abs _ _ _
      _ = |x - y| := by rw [sub_sub_sub_cancel_right]
  rw [← mul_sub, abs_mul, abs_of_nonneg (Nat.cast_nonneg _), mul_comm (|x - y|)]
  exact mul_le_mul_of_nonneg_left h1 (Nat.cast_nonneg _)

/-- Eq. (25)'s Lipschitz constant: |R x - R y| ≤ |x-y| · 2N(1 + log 2N) for x, y ≤ 2N. -/
theorem R_lipschitz (N : ℕ) (x y : ℝ) (hx : x ≤ 2 * N) (hy : y ≤ 2 * N) :
    |R x - R y| ≤ |x - y| * (2 * N * (1 + Real.log (2 * N))) := by
  have hx' : ⌊x⌋₊ ≤ 2 * N := Nat.floor_le_of_le (by exact_mod_cast hx)
  have hy' : ⌊y⌋₊ ≤ 2 * N := Nat.floor_le_of_le (by exact_mod_cast hy)
  have h := R_lipschitz_aux (2 * N) x y hx' hy'
  have hs := sum_d_le (2 * N)
  push_cast at hs
  calc |R x - R y| ≤ |x - y| * ∑ k ∈ Icc 1 (2 * N), (d k : ℝ) := h
    _ ≤ |x - y| * (2 * N * (1 + Real.log (2 * N))) :=
        mul_le_mul_of_nonneg_left hs (abs_nonneg _)

end DirichletDivisor

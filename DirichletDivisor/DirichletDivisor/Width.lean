import DirichletDivisor.Moments

set_option linter.style.whitespace false

open Finset

namespace DirichletDivisor

/-- Integers strictly inside a real interval [a, b], 0 ≤ a ≤ b: at most b - a + 1. -/
theorem card_Ioo_real (a b : ℝ) (ha : 0 ≤ a) (hab : a ≤ b) (K : ℕ) :
    (((Icc 1 K).filter (fun k : ℕ => a < k ∧ (k : ℝ) < b)).card : ℝ) ≤ b - a + 1 := by
  have hsub : (Icc 1 K).filter (fun k : ℕ => a < k ∧ (k : ℝ) < b)
      ⊆ Finset.Ioo ⌊a⌋₊ ⌈b⌉₊ := by
    intro k hk
    rw [Finset.mem_filter] at hk
    rw [Finset.mem_Ioo]
    exact ⟨(Nat.floor_lt ha).2 hk.2.1, Nat.lt_ceil.2 hk.2.2⟩
  have hc := Finset.card_le_card hsub
  rw [Nat.card_Ioo] at hc
  have hf : a - 1 < ⌊a⌋₊ := Nat.sub_one_lt_floor a
  have hce : (⌈b⌉₊ : ℝ) < b + 1 := Nat.ceil_lt_add_one (ha.trans hab)
  have hnat : ∀ c x y : ℕ, c ≤ x - y - 1 → c = 0 ∨ c + y + 1 ≤ x := by
    intro c x y h
    omega
  rcases hnat _ _ _ hc with h0 | h1
  · rw [h0]
    push_cast
    linarith
  · have h1' : (((Icc 1 K).filter (fun k : ℕ => a < k ∧ (k : ℝ) < b)).card : ℝ)
        + ⌊a⌋₊ + 1 ≤ ⌈b⌉₊ := by exact_mod_cast h1
    linarith

/-- Locality: every k in the window is within 2sL + L² of s². -/
theorem window_dist (s L k : ℝ) (h1 : (s - L) ^ 2 < k) (h2 : k < (s + L) ^ 2) :
    |k - s ^ 2| ≤ 2 * s * L + L ^ 2 := by
  rw [abs_le]
  constructor <;> nlinarith [sq_nonneg L]

/-- The window (s-L)² < k < (s+L)² contains at most 4sL + 1 integers. -/
theorem window_card (s L : ℝ) (hs : 0 ≤ s) (hL : 0 ≤ L) (K : ℕ) :
    (((Icc 1 K).filter
        (fun k : ℕ => (s - L) ^ 2 < k ∧ (k : ℝ) < (s + L) ^ 2)).card : ℝ)
      ≤ 4 * s * L + 1 := by
  have hab : (s - L) ^ 2 ≤ (s + L) ^ 2 := by nlinarith [mul_nonneg hs hL]
  have h := card_Ioo_real ((s - L) ^ 2) ((s + L) ^ 2) (sq_nonneg _) hab K
  have e : (s + L) ^ 2 - (s - L) ^ 2 = 4 * s * L := by ring
  linarith

/-- With L = H/(4s) the window has at most H + 1 integers and every k in it
satisfies |k - s²| ≤ H/2 + L². -/
theorem window_card_H (s H : ℝ) (hs : 0 < s) (hH : 0 ≤ H) (K : ℕ) :
    (((Icc 1 K).filter
        (fun k : ℕ => (s - H / (4 * s)) ^ 2 < k ∧ (k : ℝ) < (s + H / (4 * s)) ^ 2)).card : ℝ)
      ≤ H + 1 := by
  have hL : 0 ≤ H / (4 * s) := by positivity
  have h := window_card s (H / (4 * s)) hs.le hL K
  have e : 4 * s * (H / (4 * s)) = H := by field_simp
  rw [e] at h
  exact h

theorem window_dist_H (s H k : ℝ) (hs : 0 < s)
    (h1 : (s - H / (4 * s)) ^ 2 < k) (h2 : k < (s + H / (4 * s)) ^ 2) :
    |k - s ^ 2| ≤ H / 2 + (H / (4 * s)) ^ 2 := by
  have h := window_dist s (H / (4 * s)) k h1 h2
  have e : 2 * s * (H / (4 * s)) = H / 2 := by field_simp; ring
  rw [e] at h
  exact h

end DirichletDivisor

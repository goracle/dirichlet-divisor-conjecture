import DirichletDivisor.Bump

set_option linter.style.whitespace false

open Finset MeasureTheory

namespace DirichletDivisor

/-- Smoothing identity with a localized correction, for any continuous curve x
with a ≤ x ≤ b on [-1,1] and K ≥ b. -/
theorem R_window_gen (x : ℝ → ℝ) (hx : Continuous x) (a b : ℝ)
    (hr : ∀ u ∈ Set.Icc (-1:ℝ) 1, a ≤ x u ∧ x u ≤ b) (K : ℕ) (hK : b ≤ K) :
    R (∫ u in (-1:ℝ)..1, eta u * x u)
      = (∫ u in (-1:ℝ)..1, eta u * R (x u))
        + ∑ k ∈ (Icc 1 K).filter (fun k : ℕ => a < k ∧ (k : ℝ) < b),
            (d k : ℝ) * corr eta x (∫ u in (-1:ℝ)..1, eta u * x u) k := by
  have hm : ∫ u in (-1:ℝ)..1, eta u * x u ≤ b := by
    have h1 : ∫ u in (-1:ℝ)..1, eta u * x u ≤ ∫ u in (-1:ℝ)..1, eta u * b :=
      intervalIntegral.integral_mono_on (by norm_num)
        ((eta_continuous.mul hx).intervalIntegrable _ _)
        ((eta_continuous.mul continuous_const).intervalIntegrable _ _)
        (fun u hu => mul_le_mul_of_nonneg_left (hr u hu).2 (eta_nonneg u))
    have h2 : ∫ u in (-1:ℝ)..1, eta u * b = b := by
      rw [intervalIntegral.integral_mul_const, eta_integral, one_mul]
    linarith
  have hlow : ∀ k : ℝ, k ≤ a →
      corr eta x (∫ u in (-1:ℝ)..1, eta u * x u) k = 0 := fun k hk =>
    corr_eq_zero_of_le eta x eta_continuous hx (fun u _ => eta_nonneg u)
      eta_integral a (fun u hu => (hr u hu).1) k hk
  have hhigh : ∀ k : ℝ, b ≤ k →
      corr eta x (∫ u in (-1:ℝ)..1, eta u * x u) k = 0 := fun k hk =>
    corr_eq_zero_of_ge eta x eta_continuous hx (fun u _ => eta_nonneg u)
      eta_integral b (fun u hu => (hr u hu).2) k hk
  have hfilt : ∀ k ∈ Icc 1 K,
      (d k : ℝ) * corr eta x (∫ u in (-1:ℝ)..1, eta u * x u) (k : ℝ) ≠ 0 →
      (a < k ∧ (k : ℝ) < b) := by
    intro k _ hne
    by_contra hcon
    apply hne
    rcases not_and_or.mp hcon with h | h
    · rw [hlow k (not_lt.mp h), mul_zero]
    · rw [hhigh k (not_lt.mp h), mul_zero]
  have hid := R_smoothing_identity eta x eta_continuous hx K
    (fun u hu => Nat.floor_le_of_le ((hr u hu).2.trans hK))
    (∫ u in (-1:ℝ)..1, eta u * x u) (Nat.floor_le_of_le (hm.trans hK))
  rw [hid]
  congr 1
  exact (Finset.sum_filter_of_ne hfilt).symm

/-- The square-root smoothing curve x(u) = (s + L u)², 0 ≤ L ≤ s. The correction
sum runs only over the window (s-L)² < k < (s+L)². -/
theorem R_window_sqrt (s L : ℝ) (hL0 : 0 ≤ L) (hLs : L ≤ s) (K : ℕ)
    (hK : (s + L) ^ 2 ≤ K) :
    R (∫ u in (-1:ℝ)..1, eta u * (s + L * u) ^ 2)
      = (∫ u in (-1:ℝ)..1, eta u * R ((s + L * u) ^ 2))
        + ∑ k ∈ (Icc 1 K).filter
              (fun k : ℕ => (s - L) ^ 2 < k ∧ (k : ℝ) < (s + L) ^ 2),
            (d k : ℝ) * corr eta (fun u => (s + L * u) ^ 2)
              (∫ u in (-1:ℝ)..1, eta u * (s + L * u) ^ 2) k := by
  refine R_window_gen (fun u => (s + L * u) ^ 2)
    (by fun_prop) ((s - L) ^ 2) ((s + L) ^ 2) ?_ K hK
  intro u hu
  have hu1 : -1 ≤ u := hu.1
  have hu2 : u ≤ 1 := hu.2
  have h1 : 0 ≤ L * (u + 1) := mul_nonneg hL0 (by linarith)
  have h2 : 0 ≤ 2 * s + L * u - L := by nlinarith
  have h3 : 0 ≤ L * (1 - u) := mul_nonneg hL0 (by linarith)
  have h4 : 0 ≤ 2 * s + L + L * u := by nlinarith
  constructor
  · show (s - L) ^ 2 ≤ (s + L * u) ^ 2
    nlinarith [mul_nonneg h1 h2]
  · show (s + L * u) ^ 2 ≤ (s + L) ^ 2
    nlinarith [mul_nonneg h3 h4]

end DirichletDivisor

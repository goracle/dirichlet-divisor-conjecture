import DirichletDivisor.HankelSum

set_option linter.style.whitespace false

namespace DirichletDivisor

open Finset

/-! ## Summed `K₂` contribution (§2/§6 of the note)

For `N/2 ≤ x ≤ 2N`, `N ≥ 128`: `∑ d(m) (x/m) |K₂(4π√(mx))| / π² ≤ 2 N^{3/4} e^{-2π√(N/2)}`,
which is exactly the `hK` hypothesis shape consumed by `Ledger`/`Assembly`. -/

/-- The `m`-th `K₂` term of the Voronoï series: `d(m) (x/m) K₂(Z)/π²`, `Z = 4π√(mx)`. -/
noncomputable def k2Term (ci : CitedInputs) (x : ℝ) (m : ℕ) : ℝ :=
  (d m : ℝ) * ((x / m) * (ci.K2 (4 * Real.pi * Real.sqrt (m * x)) / Real.pi ^ 2))

/-- `s^6 e^{-a s} ≤ e^{-a}` for `a ≥ 6`, `s ≥ 1`. -/
theorem exp_sqrt_decay (a s : ℝ) (ha : 6 ≤ a) (hs : 1 ≤ s) :
    s ^ 6 * Real.exp (-(a * s)) ≤ Real.exp (-a) := by
  have h1 : s ≤ Real.exp (s - 1) := by
    have := Real.add_one_le_exp (s - 1)
    linarith
  have h0 : 0 ≤ s := by linarith
  have h2 : s ^ 6 ≤ Real.exp (a * (s - 1)) := by
    calc s ^ 6 ≤ (Real.exp (s - 1)) ^ 6 := pow_le_pow_left₀ h0 h1 6
      _ = Real.exp (6 * (s - 1)) := by
          rw [← Real.exp_nat_mul]; norm_num
      _ ≤ Real.exp (a * (s - 1)) := by
          apply Real.exp_le_exp.2
          nlinarith [mul_nonneg (sub_nonneg.2 ha) (sub_nonneg.2 hs)]
  calc s ^ 6 * Real.exp (-(a * s))
      ≤ Real.exp (a * (s - 1)) * Real.exp (-(a * s)) :=
        mul_le_mul_of_nonneg_right h2 (Real.exp_pos _).le
    _ = Real.exp (-a) := by
        rw [← Real.exp_add]; congr 1; ring

/-- Partial sums of `1/(j+1)²` are below `2 - 2/(n+1)`. -/
theorem inv_sq_partial_le (n : ℕ) :
    ∑ j ∈ Finset.range n, 1 / (((j : ℝ) + 1) ^ 2) ≤ 2 - 2 / ((n : ℝ) + 1) := by
  induction n with
  | zero => norm_num
  | succ n ih =>
    rw [Finset.sum_range_succ]
    have hn : (0 : ℝ) ≤ n := Nat.cast_nonneg n
    have hn1 : (n : ℝ) + 1 ≠ 0 := by positivity
    have hn2 : (n : ℝ) + 1 + 1 ≠ 0 := by positivity
    have hn3 : (n : ℝ) + 2 ≠ 0 := by positivity
    have key : 2 / ((n : ℝ) + 1) - 2 / (((n + 1 : ℕ) : ℝ) + 1) - 1 / (((n : ℝ) + 1) ^ 2)
        = (n : ℝ) / (((n : ℝ) + 1) ^ 2 * ((n : ℝ) + 2)) := by
      push_cast
      field_simp
      ring
    have hpos : 0 ≤ (n : ℝ) / (((n : ℝ) + 1) ^ 2 * ((n : ℝ) + 2)) := by positivity
    linarith

/-- Per-term bound: for `x ≥ 64`, `|k2Term| ≤ x e^{-4π√x} / m²`. -/
theorem k2Term_le (ci : CitedInputs) (x : ℝ) (hx : 64 ≤ x) (m : ℕ) (hm : 1 ≤ m) :
    |k2Term ci x m| ≤
      x * Real.exp (-(4 * Real.pi * Real.sqrt x)) * (1 / ((m : ℝ) ^ 2)) := by
  have hx0 : 0 < x := by linarith
  have hmR : (1 : ℝ) ≤ m := by exact_mod_cast hm
  have hm0 : (0 : ℝ) < m := by linarith
  have hpi : (3.14 : ℝ) < Real.pi := Real.pi_gt_d2
  have hpi4 : Real.pi ≤ 4 := Real.pi_le_four
  have hs8 : Real.sqrt 64 = 8 := by
    rw [show (64 : ℝ) = 8 ^ 2 by norm_num]
    exact Real.sqrt_sq (by norm_num)
  have hsx : 8 ≤ Real.sqrt x := by
    have := Real.sqrt_le_sqrt hx
    rw [hs8] at this
    exact this
  have hsm : 1 ≤ Real.sqrt (m : ℝ) := by
    have := Real.sqrt_le_sqrt hmR
    rw [Real.sqrt_one] at this
    exact this
  obtain ⟨a, ha⟩ : ∃ a : ℝ, a = 4 * Real.pi * Real.sqrt x := ⟨_, rfl⟩
  have ha100 : 100 ≤ a := by rw [ha]; nlinarith
  have hZ : 4 * Real.pi * Real.sqrt ((m : ℝ) * x) = a * Real.sqrt (m : ℝ) := by
    rw [Real.sqrt_mul hm0.le, ha]; ring
  have hZ100 : 100 ≤ a * Real.sqrt (m : ℝ) := by
    have := mul_le_mul_of_nonneg_left hsm (by linarith : 0 ≤ a)
    linarith
  have hZpos : 0 < 2 * (a * Real.sqrt (m : ℝ)) := by linarith
  rw [← ha]
  unfold k2Term
  rw [hZ]
  obtain ⟨E, hEdef⟩ : ∃ E : ℝ, E = Real.exp (-(a * Real.sqrt (m : ℝ))) := ⟨_, rfl⟩
  have hE0 : 0 ≤ E := by rw [hEdef]; exact (Real.exp_pos _).le
  -- |K₂| ≤ 1.02 E
  have hsq : Real.sqrt (Real.pi / (2 * (a * Real.sqrt (m : ℝ)))) ≤ 1 := by
    have h : Real.pi / (2 * (a * Real.sqrt (m : ℝ))) ≤ 1 := by
      rw [div_le_one hZpos]; linarith
    calc Real.sqrt (Real.pi / (2 * (a * Real.sqrt (m : ℝ)))) ≤ Real.sqrt 1 :=
          Real.sqrt_le_sqrt h
      _ = 1 := Real.sqrt_one
  have hKb : |ci.K2 (a * Real.sqrt (m : ℝ))| ≤ 1.02 * E := by
    have h := ci.bound_K2 (a * Real.sqrt (m : ℝ)) hZ100
    rw [← hEdef] at h
    have h2 : 1.02 * Real.sqrt (Real.pi / (2 * (a * Real.sqrt (m : ℝ)))) * E ≤ 1.02 * 1 * E :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hsq (by norm_num)) hE0
    linarith
  -- E ≤ e^{-a} / m²
  have hdec := exp_sqrt_decay a (Real.sqrt (m : ℝ)) (by linarith) hsm
  have hsq6 : Real.sqrt (m : ℝ) ^ 6 = (m : ℝ) ^ 3 := by
    have h2 : Real.sqrt (m : ℝ) ^ 2 = m := Real.sq_sqrt hm0.le
    calc Real.sqrt (m : ℝ) ^ 6 = (Real.sqrt (m : ℝ) ^ 2) ^ 3 := by ring
      _ = (m : ℝ) ^ 3 := by rw [h2]
  rw [hsq6, ← hEdef] at hdec
  have hm23 : (m : ℝ) ^ 2 ≤ (m : ℝ) ^ 3 := by nlinarith
  have hE2 : E * (m : ℝ) ^ 2 ≤ Real.exp (-a) := by
    have := mul_le_mul_of_nonneg_right hm23 hE0
    nlinarith
  have hEle : E ≤ Real.exp (-a) / (m : ℝ) ^ 2 :=
    (le_div_iff₀ (by positivity)).2 hE2
  -- assemble
  have hd : (d m : ℝ) ≤ m := by
    have : d m ≤ m := Nat.card_divisors_le_self m
    exact_mod_cast this
  have habs : |(d m : ℝ) * ((x / m) * (ci.K2 (a * Real.sqrt (m : ℝ)) / Real.pi ^ 2))|
      = (d m : ℝ) * ((x / m) * (|ci.K2 (a * Real.sqrt (m : ℝ))| / Real.pi ^ 2)) := by
    rw [abs_mul, abs_mul, abs_of_nonneg (Nat.cast_nonneg (d m) : (0 : ℝ) ≤ (d m : ℝ)),
      abs_of_nonneg (div_nonneg hx0.le hm0.le), abs_div,
      abs_of_nonneg (sq_nonneg Real.pi)]
  have hY : (x / m) * (|ci.K2 (a * Real.sqrt (m : ℝ))| / Real.pi ^ 2)
      ≤ (x / m) * (1.02 * E / Real.pi ^ 2) := by
    have h1 : |ci.K2 (a * Real.sqrt (m : ℝ))| / Real.pi ^ 2 ≤ 1.02 * E / Real.pi ^ 2 := by
      gcongr
    exact mul_le_mul_of_nonneg_left h1 (div_nonneg hx0.le hm0.le)
  have hY0 : 0 ≤ (x / m) * (|ci.K2 (a * Real.sqrt (m : ℝ))| / Real.pi ^ 2) :=
    mul_nonneg (div_nonneg hx0.le hm0.le) (div_nonneg (abs_nonneg _) (sq_nonneg _))
  have hmx : (m : ℝ) * (x / m) = x := by field_simp
  have hW : 1.02 * E / Real.pi ^ 2 ≤ E := by
    have hp2 : (1.02 : ℝ) ≤ Real.pi ^ 2 := by nlinarith
    have hpp : 0 < Real.pi ^ 2 := by positivity
    rw [div_le_iff₀ hpp]
    nlinarith [mul_le_mul_of_nonneg_right hp2 hE0]
  calc |(d m : ℝ) * ((x / m) * (ci.K2 (a * Real.sqrt (m : ℝ)) / Real.pi ^ 2))|
      = (d m : ℝ) * ((x / m) * (|ci.K2 (a * Real.sqrt (m : ℝ))| / Real.pi ^ 2)) := habs
    _ ≤ (m : ℝ) * ((x / m) * (1.02 * E / Real.pi ^ 2)) :=
        mul_le_mul hd hY hY0 hm0.le
    _ = x * (1.02 * E / Real.pi ^ 2) := by
        calc (m : ℝ) * ((x / m) * (1.02 * E / Real.pi ^ 2))
            = ((m : ℝ) * (x / m)) * (1.02 * E / Real.pi ^ 2) := by ring
          _ = x * (1.02 * E / Real.pi ^ 2) := by rw [hmx]
    _ ≤ x * E := mul_le_mul_of_nonneg_left hW hx0.le
    _ ≤ x * (Real.exp (-a) / (m : ℝ) ^ 2) := mul_le_mul_of_nonneg_left hEle hx0.le
    _ = x * Real.exp (-a) * (1 / (m : ℝ) ^ 2) := by ring

/-- **Summed `K₂` terms**, pointwise in `x ≥ 64`. -/
theorem k2_remainder_sum (ci : CitedInputs) (x : ℝ) (hx : 64 ≤ x) :
    Summable (fun j : ℕ => |k2Term ci x (j + 1)|) ∧
      ∑' j : ℕ, |k2Term ci x (j + 1)|
        ≤ 2 * (x * Real.exp (-(4 * Real.pi * Real.sqrt x))) := by
  have hx0 : 0 < x := by linarith
  have hC : 0 ≤ x * Real.exp (-(4 * Real.pi * Real.sqrt x)) :=
    mul_nonneg hx0.le (Real.exp_pos _).le
  have hpart : ∀ n : ℕ, ∑ j ∈ Finset.range n, |k2Term ci x (j + 1)|
      ≤ 2 * (x * Real.exp (-(4 * Real.pi * Real.sqrt x))) := by
    intro n
    calc ∑ j ∈ Finset.range n, |k2Term ci x (j + 1)|
        ≤ ∑ j ∈ Finset.range n,
            (x * Real.exp (-(4 * Real.pi * Real.sqrt x))) * (1 / (((j : ℝ) + 1) ^ 2)) := by
          apply Finset.sum_le_sum
          intro j _
          have := k2Term_le ci x hx (j + 1) (by omega)
          push_cast at this
          linarith
      _ = (x * Real.exp (-(4 * Real.pi * Real.sqrt x))) *
            ∑ j ∈ Finset.range n, 1 / (((j : ℝ) + 1) ^ 2) := by rw [Finset.mul_sum]
      _ ≤ (x * Real.exp (-(4 * Real.pi * Real.sqrt x))) * 2 := by
          apply mul_le_mul_of_nonneg_left _ hC
          have h1 := inv_sq_partial_le n
          have h2 : 0 < 2 / ((n : ℝ) + 1) := by positivity
          linarith
      _ = 2 * (x * Real.exp (-(4 * Real.pi * Real.sqrt x))) := by ring
  exact ⟨summable_of_sum_range_le (fun j => abs_nonneg _) hpart,
    Real.tsum_le_of_sum_range_le (fun j => abs_nonneg _) hpart⟩

/-- The note's `E_{K₂} ≤ 2 N^{3/4} e^{-2π√(N/2)}`, for `N/2 ≤ x ≤ 2N`, `N ≥ 128`. -/
theorem k2_remainder_sum_N (ci : CitedInputs) (N x : ℝ) (hN : 128 ≤ N) (hx : N / 2 ≤ x)
    (hx2 : x ≤ 2 * N) :
    Summable (fun j : ℕ => |k2Term ci x (j + 1)|) ∧
      ∑' j : ℕ, |k2Term ci x (j + 1)|
        ≤ 2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * Real.sqrt (N / 2))) := by
  have hx64 : 64 ≤ x := by linarith
  have hx0 : 0 < x := by linarith
  obtain ⟨hs, ht⟩ := k2_remainder_sum ci x hx64
  refine ⟨hs, ht.trans ?_⟩
  obtain ⟨s, hs'⟩ : ∃ s : ℝ, s = Real.sqrt (N / 2) := ⟨_, rfl⟩
  rw [← hs']
  have hs2 : s ^ 2 = N / 2 := by rw [hs']; exact Real.sq_sqrt (by linarith)
  have hs0 : 0 ≤ s := by rw [hs']; exact Real.sqrt_nonneg _
  have hsx : s ≤ Real.sqrt x := by rw [hs']; exact Real.sqrt_le_sqrt hx
  have hpi : (3 : ℝ) < Real.pi := Real.pi_gt_three
  have hx4 : x ≤ 4 * s ^ 2 := by rw [hs2]; linarith
  have hE1 : Real.exp (-(4 * Real.pi * Real.sqrt x))
      ≤ Real.exp (-(2 * Real.pi * s)) * Real.exp (-(2 * Real.pi * s)) := by
    rw [← Real.exp_add]
    apply Real.exp_le_exp.2
    nlinarith [mul_le_mul_of_nonneg_left hsx Real.pi_pos.le]
  have hexp : 4 * s ^ 2 * Real.exp (-(2 * Real.pi * s)) ≤ 1 := by
    have hq := Real.quadratic_le_exp_of_nonneg (by positivity : 0 ≤ 2 * Real.pi * s)
    have h4 : 4 * s ^ 2 ≤ Real.exp (2 * Real.pi * s) := by
      have hp2 : (9 : ℝ) ≤ Real.pi ^ 2 := by nlinarith
      have hps : 9 * s ^ 2 ≤ Real.pi ^ 2 * s ^ 2 :=
        mul_le_mul_of_nonneg_right hp2 (sq_nonneg s)
      have hlin : 0 ≤ 2 * Real.pi * s := by positivity
      have hexpand : (2 * Real.pi * s) ^ 2 / 2 = 2 * (Real.pi ^ 2 * s ^ 2) := by ring
      have hs2' : 0 ≤ s ^ 2 := sq_nonneg s
      rw [hexpand] at hq
      linarith
    calc 4 * s ^ 2 * Real.exp (-(2 * Real.pi * s))
        ≤ Real.exp (2 * Real.pi * s) * Real.exp (-(2 * Real.pi * s)) :=
          mul_le_mul_of_nonneg_right h4 (Real.exp_pos _).le
      _ = 1 := by rw [← Real.exp_add]; simp
  have hxe : x * Real.exp (-(4 * Real.pi * Real.sqrt x)) ≤ Real.exp (-(2 * Real.pi * s)) := by
    calc x * Real.exp (-(4 * Real.pi * Real.sqrt x))
        ≤ 4 * s ^ 2 * (Real.exp (-(2 * Real.pi * s)) * Real.exp (-(2 * Real.pi * s))) :=
          mul_le_mul hx4 hE1 (Real.exp_pos _).le (by positivity)
      _ = (4 * s ^ 2 * Real.exp (-(2 * Real.pi * s))) * Real.exp (-(2 * Real.pi * s)) := by ring
      _ ≤ 1 * Real.exp (-(2 * Real.pi * s)) :=
          mul_le_mul_of_nonneg_right hexp (Real.exp_pos _).le
      _ = Real.exp (-(2 * Real.pi * s)) := one_mul _
  have h34 : 1 ≤ N ^ (3 / 4 : ℝ) := Real.one_le_rpow (by linarith) (by norm_num)
  have hep := (Real.exp_pos (-(2 * Real.pi * s))).le
  nlinarith [mul_le_mul_of_nonneg_right h34 hep]

end DirichletDivisor

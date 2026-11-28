import DirichletDivisor.Cited
import DirichletDivisor.TailSeries

set_option linter.style.whitespace false

namespace DirichletDivisor

open Finset

/-! ## Summed Hankel remainder (§2 of the note)

We prove `∑ d(m)·m^{-9/4} ≤ (1.462)²` (a rigorous upper bound for `ζ(9/4)²`), and from the
cited Hankel remainder bound for `Y₂` deduce that the summed remainder is at most
`4.03e-4 · x^{-1/4}` for `x ≥ 64`, hence `≤ 4.8e-4 · N^{-1/4}` when `x ≥ N/2`, `N ≥ 128`. -/

/-- `k^{-9/4}`. -/
noncomputable def pm94 (k : ℕ) : ℝ := (k : ℝ) ^ (-(9 / 4 : ℝ))

theorem pm94_nonneg (k : ℕ) : 0 ≤ pm94 k := Real.rpow_nonneg (Nat.cast_nonneg k) _

theorem rpow_quarter_pow4 (r : ℝ) (hr : 0 ≤ r) : (r ^ 4) ^ (1 / 4 : ℝ) = r := by
  rw [← Real.rpow_natCast r 4, ← Real.rpow_mul hr]
  norm_num

/-- `k^{-9/4} ≤ 1/(k² r)` whenever `r^4 ≤ k`. -/
theorem pm94_le (k : ℕ) (r : ℝ) (hk : 0 < k) (hr : 0 < r) (hr4 : r ^ 4 ≤ (k : ℝ)) :
    pm94 k ≤ 1 / ((k : ℝ) ^ 2 * r) := by
  have hkpos : (0 : ℝ) < k := by exact_mod_cast hk
  have hrk : r ≤ (k : ℝ) ^ (1 / 4 : ℝ) := by
    calc r = (r ^ 4) ^ (1 / 4 : ℝ) := (rpow_quarter_pow4 r hr.le).symm
      _ ≤ (k : ℝ) ^ (1 / 4 : ℝ) := Real.rpow_le_rpow (by positivity) hr4 (by norm_num)
  have hid : (k : ℝ) ^ (-(9 / 4 : ℝ)) = ((k : ℝ) ^ 2 * (k : ℝ) ^ (1 / 4 : ℝ))⁻¹ := by
    rw [← Real.rpow_natCast (k : ℝ) 2, ← Real.rpow_add hkpos, ← Real.rpow_neg hkpos.le]
    norm_num
  unfold pm94
  rw [hid, inv_eq_one_div]
  exact one_div_le_one_div_of_le (by positivity) (mul_le_mul_of_nonneg_left hrk (by positivity))

theorem sixteen_tail : ((16 : ℕ) : ℝ) ^ (1 - (9 / 4 : ℝ)) / ((9 / 4 : ℝ) - 1) = 1 / 40 := by
  have h5 : ((16 : ℕ) : ℝ) ^ (1 - (9 / 4 : ℝ)) = 1 / 32 := by
    rw [show ((16 : ℕ) : ℝ) = (2 : ℝ) ^ (4 : ℝ) by
        rw [show (4 : ℝ) = ((4 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
        norm_num,
      ← Real.rpow_mul (by norm_num),
      show (4 : ℝ) * (1 - 9 / 4) = ((-5 : ℤ) : ℝ) by norm_num, Real.rpow_intCast]
    norm_num
  rw [h5]
  norm_num

theorem pm94_sum16 : ∑ k ∈ Finset.range 16, pm94 (k + 1) ≤ 1.4363 := by
  have e : ∑ k ∈ Finset.range 16, pm94 (k + 1) = pm94 1 + pm94 2 + pm94 3 + pm94 4 + pm94 5
      + pm94 6 + pm94 7 + pm94 8 + pm94 9 + pm94 10 + pm94 11 + pm94 12 + pm94 13
      + pm94 14 + pm94 15 + pm94 16 := by
    simp [Finset.sum_range_succ]
  have h1 := pm94_le 1 (1000/1000) (by norm_num) (by norm_num) (by norm_num)
  have h2 := pm94_le 2 (1189/1000) (by norm_num) (by norm_num) (by norm_num)
  have h3 := pm94_le 3 (1316/1000) (by norm_num) (by norm_num) (by norm_num)
  have h4 := pm94_le 4 (1414/1000) (by norm_num) (by norm_num) (by norm_num)
  have h5 := pm94_le 5 (1495/1000) (by norm_num) (by norm_num) (by norm_num)
  have h6 := pm94_le 6 (1565/1000) (by norm_num) (by norm_num) (by norm_num)
  have h7 := pm94_le 7 (1626/1000) (by norm_num) (by norm_num) (by norm_num)
  have h8 := pm94_le 8 (1681/1000) (by norm_num) (by norm_num) (by norm_num)
  have h9 := pm94_le 9 (1732/1000) (by norm_num) (by norm_num) (by norm_num)
  have h10 := pm94_le 10 (1778/1000) (by norm_num) (by norm_num) (by norm_num)
  have h11 := pm94_le 11 (1821/1000) (by norm_num) (by norm_num) (by norm_num)
  have h12 := pm94_le 12 (1861/1000) (by norm_num) (by norm_num) (by norm_num)
  have h13 := pm94_le 13 (1898/1000) (by norm_num) (by norm_num) (by norm_num)
  have h14 := pm94_le 14 (1934/1000) (by norm_num) (by norm_num) (by norm_num)
  have h15 := pm94_le 15 (1967/1000) (by norm_num) (by norm_num) (by norm_num)
  have h16 := pm94_le 16 (2000/1000) (by norm_num) (by norm_num) (by norm_num)
  rw [e]
  linarith [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16]

/-- Rigorous partial-sum bound for `ζ(9/4)`: `∑_{k ≤ n} k^{-9/4} ≤ 1.462` (true value ≈ 1.4602). -/
theorem zeta94_partial_le (n : ℕ) : ∑ k ∈ Finset.range n, pm94 (k + 1) ≤ 1.462 := by
  have hmono : ∑ k ∈ Finset.range n, pm94 (k + 1) ≤ ∑ k ∈ Finset.range (16 + n), pm94 (k + 1) :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono (by omega))
      (fun k _ _ => pm94_nonneg _)
  rw [Finset.sum_range_add] at hmono
  have ht : ∑ x ∈ Finset.range n, pm94 (16 + x + 1) ≤ 1 / 40 := by
    have h := pseries_partial_le (9 / 4) (by norm_num) 16 n (by norm_num)
    exact h.trans (le_of_eq sixteen_tail)
  have h16 := pm94_sum16
  linarith

/-- `∑_{k ∈ [1,n]} f k = ∑_{k<n} f (k+1)`. -/
theorem sum_Icc_eq_range (f : ℕ → ℝ) (n : ℕ) :
    ∑ k ∈ Finset.Icc 1 n, f k = ∑ k ∈ Finset.range n, f (k + 1) := by
  induction n with
  | zero => simp
  | succ n ih => rw [Finset.sum_Icc_succ_top (by omega), ih, Finset.sum_range_succ]

/-- Dirichlet-convolution square bound: for completely multiplicative nonnegative `g`,
`∑_{m ≤ n} d(m) g(m) ≤ (∑_{k ≤ n} g(k))²`. -/
theorem dsum_le_sq (g : ℕ → ℝ) (hg : ∀ k, 0 ≤ g k) (hmul : ∀ a b, g (a * b) = g a * g b)
    (n : ℕ) :
    ∑ m ∈ Finset.Icc 1 n, (d m : ℝ) * g m ≤ (∑ k ∈ Finset.Icc 1 n, g k) ^ 2 := by
  have h1 : ∀ m ∈ Finset.Icc 1 n,
      (d m : ℝ) * g m = ∑ p ∈ Nat.divisorsAntidiagonal m, g p.1 * g p.2 := by
    intro m _
    have hc : ∑ p ∈ Nat.divisorsAntidiagonal m, g p.1 * g p.2
        = ∑ p ∈ Nat.divisorsAntidiagonal m, g m := by
      apply Finset.sum_congr rfl
      intro p hp
      rw [← hmul, (Nat.mem_divisorsAntidiagonal.1 hp).1]
    have hcard : (Nat.divisorsAntidiagonal m).card = d m := by
      unfold d
      rw [← Nat.map_div_right_divisors, Finset.card_map]
    rw [hc, Finset.sum_const, nsmul_eq_mul, hcard]
  rw [Finset.sum_congr rfl h1]
  have hdisj : ∀ m ∈ Finset.Icc 1 n, ∀ m' ∈ Finset.Icc 1 n, m ≠ m' →
      Disjoint (Nat.divisorsAntidiagonal m) (Nat.divisorsAntidiagonal m') := by
    intro m _ m' _ hne
    refine Finset.disjoint_left.2 ?_
    intro p hp hp'
    exact hne ((Nat.mem_divisorsAntidiagonal.1 hp).1.symm.trans
      (Nat.mem_divisorsAntidiagonal.1 hp').1)
  rw [← Finset.sum_biUnion (fun m hm m' hm' hne => hdisj m hm m' hm' hne)]
  have hsub : (Finset.Icc 1 n).biUnion Nat.divisorsAntidiagonal
      ⊆ Finset.Icc 1 n ×ˢ Finset.Icc 1 n := by
    intro p hp
    rw [Finset.mem_biUnion] at hp
    obtain ⟨m, hm, hp⟩ := hp
    rw [Nat.mem_divisorsAntidiagonal] at hp
    obtain ⟨hpm, hm0⟩ := hp
    have hm' := Finset.mem_Icc.1 hm
    have ha : 0 < p.1 := Nat.pos_of_ne_zero (fun h => hm0 (by rw [← hpm, h]; simp))
    have hb : 0 < p.2 := Nat.pos_of_ne_zero (fun h => hm0 (by rw [← hpm, h]; simp))
    have ha' : p.1 ≤ m := by rw [← hpm]; exact Nat.le_mul_of_pos_right _ hb
    have hb' : p.2 ≤ m := by rw [← hpm]; exact Nat.le_mul_of_pos_left _ ha
    exact Finset.mem_product.2
      ⟨Finset.mem_Icc.2 ⟨ha, by omega⟩, Finset.mem_Icc.2 ⟨hb, by omega⟩⟩
  calc ∑ p ∈ (Finset.Icc 1 n).biUnion Nat.divisorsAntidiagonal, g p.1 * g p.2
      ≤ ∑ p ∈ Finset.Icc 1 n ×ˢ Finset.Icc 1 n, g p.1 * g p.2 :=
        Finset.sum_le_sum_of_subset_of_nonneg hsub
          (fun p _ _ => mul_nonneg (hg _) (hg _))
    _ = (∑ k ∈ Finset.Icc 1 n, g k) ^ 2 := by
        rw [Finset.sum_product, sq, Finset.sum_mul_sum]

theorem pm94_mul (a b : ℕ) : pm94 (a * b) = pm94 a * pm94 b := by
  unfold pm94
  push_cast
  exact Real.mul_rpow (Nat.cast_nonneg a) (Nat.cast_nonneg b)

/-- Partial sums of `d(m) m^{-9/4}` are bounded by `1.462²` (a bound for `ζ(9/4)²`). -/
theorem dsum94_partial_le (n : ℕ) :
    ∑ j ∈ Finset.range n, (d (j + 1) : ℝ) * pm94 (j + 1) ≤ 1.462 ^ 2 := by
  have h := dsum_le_sq pm94 pm94_nonneg pm94_mul n
  rw [sum_Icc_eq_range (fun m => (d m : ℝ) * pm94 m), sum_Icc_eq_range pm94] at h
  have hP := zeta94_partial_le n
  have hP0 : 0 ≤ ∑ k ∈ Finset.range n, pm94 (k + 1) :=
    Finset.sum_nonneg (fun k _ => pm94_nonneg _)
  exact h.trans (pow_le_pow_left₀ hP0 hP 2)

/-! ### The Hankel remainder terms -/

noncomputable def hankelMain (z : ℝ) : ℝ :=
  Real.sqrt (2 / (Real.pi * z)) *
    (Real.sin (z - 5 * Real.pi / 4) + 15 / (8 * z) * Real.cos (z - 5 * Real.pi / 4))

/-- The `m`-th Hankel-remainder term `d(m) (x/m) (Y₂(Z) - main(Z)) / (2π)`, `Z = 4π√(mx)`. -/
noncomputable def hankelRem (ci : CitedInputs) (x : ℝ) (m : ℕ) : ℝ :=
  (d m : ℝ) * ((x / m) * ((ci.Y2 (4 * Real.pi * Real.sqrt (m * x))
      - hankelMain (4 * Real.pi * Real.sqrt (m * x))) / (2 * Real.pi)))

/-- `C₁ = 0.83 √2 / (64 π⁴) ≈ 1.883e-4`. -/
noncomputable def hankelC1 : ℝ := 83 * Real.sqrt 2 / (6400 * Real.pi ^ 4)

theorem hankelC1_num : hankelC1 * 1.462 ^ 2 ≤ 4.03e-4 := by
  have hs2 : Real.sqrt 2 ≤ 1.4143 := Real.sqrt_le_iff.2 ⟨by norm_num, by norm_num⟩
  have hpi4 : (97.39 : ℝ) ≤ Real.pi ^ 4 := by
    calc (97.39 : ℝ) ≤ 3.1415 ^ 4 := by norm_num
      _ ≤ Real.pi ^ 4 := pow_le_pow_left₀ (by norm_num) Real.pi_gt_d4.le 4
  have hpp : 0 < Real.pi ^ 4 := by positivity
  unfold hankelC1
  rw [div_mul_eq_mul_div, div_le_iff₀ (by positivity)]
  nlinarith [hs2, hpi4]

theorem hankelRem_le (ci : CitedInputs) (x : ℝ) (hx : 64 ≤ x) (m : ℕ) (hm : 1 ≤ m) :
    |hankelRem ci x m| ≤ hankelC1 * x ^ (-(1 / 4 : ℝ)) * ((d m : ℝ) * pm94 m) := by
  have hx0 : 0 < x := by linarith
  have hmr : (1 : ℝ) ≤ m := by exact_mod_cast hm
  have hmpos : (0 : ℝ) < m := by linarith
  have hP : 64 ≤ (m : ℝ) * x := by nlinarith
  have hpi : Real.pi ≠ 0 := Real.pi_ne_zero
  obtain ⟨t, ht⟩ : ∃ t : ℝ, t = ((m : ℝ) * x) ^ (1 / 4 : ℝ) := ⟨_, rfl⟩
  have htpos : 0 < t := by rw [ht]; positivity
  have ht4 : t ^ 4 = (m : ℝ) * x := by
    rw [ht, ← Real.rpow_natCast (((m : ℝ) * x) ^ (1 / 4 : ℝ)) 4, ← Real.rpow_mul (by positivity)]
    norm_num
  have hsq : Real.sqrt ((m : ℝ) * x) = t ^ 2 := by
    rw [← ht4, show t ^ 4 = (t ^ 2) ^ 2 by ring]
    exact Real.sqrt_sq (by positivity)
  have ht2 : 8 ≤ t ^ 2 := by
    by_contra h
    have h' := not_le.mp h
    nlinarith [mul_pos (sub_pos.2 h') (by positivity : (0 : ℝ) < 8 + t ^ 2), ht4, hP]
  have hZ100 : 100 ≤ 4 * Real.pi * t ^ 2 := by
    have := Real.pi_gt_d4
    nlinarith [ht2, this]
  have h := ci.hankel_Y2 (4 * Real.pi * t ^ 2) hZ100
  have h083 : (0.83 : ℝ) = 83 / 100 := by norm_num
  rw [h083] at h
  have h' : |ci.Y2 (4 * Real.pi * t ^ 2) - hankelMain (4 * Real.pi * t ^ 2)|
      ≤ Real.sqrt (2 / (Real.pi * (4 * Real.pi * t ^ 2))) * (83 / 100 / (4 * Real.pi * t ^ 2) ^ 2) :=
    h
  have hr : Real.sqrt (2 / (Real.pi * (4 * Real.pi * t ^ 2))) = Real.sqrt 2 / (2 * Real.pi * t) := by
    have h2 := Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num)
    have hh : 2 / (Real.pi * (4 * Real.pi * t ^ 2)) = (Real.sqrt 2 / (2 * Real.pi * t)) ^ 2 := by
      rw [div_pow, h2]
      first | ring1 | (field_simp; ring1) | field_simp
    rw [hh]
    exact Real.sqrt_sq (by positivity)
  have h5 : t ^ 5 = (m : ℝ) ^ (5 / 4 : ℝ) * x ^ (5 / 4 : ℝ) := by
    rw [ht, ← Real.rpow_natCast (((m : ℝ) * x) ^ (1 / 4 : ℝ)) 5, ← Real.rpow_mul (by positivity),
      Real.mul_rpow (by positivity) (by positivity)]
    norm_num
  have hxm : x / m = x ^ (-(1 / 4 : ℝ)) * (m : ℝ) ^ (-(9 / 4 : ℝ)) * t ^ 5 := by
    have e1 : x ^ (-(1 / 4 : ℝ)) * x ^ (5 / 4 : ℝ) = x := by
      rw [← Real.rpow_add hx0]
      norm_num
    have e2 : (m : ℝ) ^ (-(9 / 4 : ℝ)) * (m : ℝ) ^ (5 / 4 : ℝ) = (m : ℝ)⁻¹ := by
      rw [← Real.rpow_add hmpos, ← Real.rpow_neg_one]
      norm_num
    rw [h5]
    calc x / m = x * (m : ℝ)⁻¹ := by rw [div_eq_mul_inv]
      _ = (x ^ (-(1 / 4 : ℝ)) * x ^ (5 / 4 : ℝ)) * ((m : ℝ) ^ (-(9 / 4 : ℝ)) * (m : ℝ) ^ (5 / 4 : ℝ)) := by
          rw [e1, e2]
      _ = _ := by ring
  have habs : |hankelRem ci x m| = (d m : ℝ) * ((x / m) *
      (|ci.Y2 (4 * Real.pi * t ^ 2) - hankelMain (4 * Real.pi * t ^ 2)| / (2 * Real.pi))) := by
    unfold hankelRem
    rw [hsq, abs_mul, abs_mul, abs_of_nonneg (Nat.cast_nonneg (d m)),
      abs_of_nonneg (show (0 : ℝ) ≤ x / m by positivity), abs_div,
      abs_of_pos (show 0 < 2 * Real.pi by positivity)]
  rw [habs]
  calc (d m : ℝ) * ((x / m) *
        (|ci.Y2 (4 * Real.pi * t ^ 2) - hankelMain (4 * Real.pi * t ^ 2)| / (2 * Real.pi)))
      ≤ (d m : ℝ) * ((x / m) *
        ((Real.sqrt (2 / (Real.pi * (4 * Real.pi * t ^ 2))) *
          (83 / 100 / (4 * Real.pi * t ^ 2) ^ 2)) / (2 * Real.pi))) := by
        gcongr
    _ = hankelC1 * x ^ (-(1 / 4 : ℝ)) * ((d m : ℝ) * pm94 m) := by
        unfold hankelC1 pm94
        rw [hr, hxm]
        have ht0 : t ≠ 0 := htpos.ne'
        field_simp
        ring

/-- **Summed Hankel remainder** (pointwise in `x ≥ 64`): the series of absolute values
converges and is at most `4.03e-4 · x^{-1/4}`. -/
theorem hankel_remainder_sum (ci : CitedInputs) (x : ℝ) (hx : 64 ≤ x) :
    Summable (fun j : ℕ => |hankelRem ci x (j + 1)|) ∧
      ∑' j : ℕ, |hankelRem ci x (j + 1)| ≤ 4.03e-4 * x ^ (-(1 / 4 : ℝ)) := by
  have hx0 : 0 < x := by linarith
  have hC1 : 0 ≤ hankelC1 := by unfold hankelC1; positivity
  have hxr : 0 ≤ x ^ (-(1 / 4 : ℝ)) := Real.rpow_nonneg hx0.le _
  have hpart : ∀ n : ℕ, ∑ j ∈ Finset.range n, |hankelRem ci x (j + 1)|
      ≤ 4.03e-4 * x ^ (-(1 / 4 : ℝ)) := by
    intro n
    calc ∑ j ∈ Finset.range n, |hankelRem ci x (j + 1)|
        ≤ ∑ j ∈ Finset.range n, hankelC1 * x ^ (-(1 / 4 : ℝ)) * ((d (j + 1) : ℝ) * pm94 (j + 1)) :=
          Finset.sum_le_sum (fun j _ => hankelRem_le ci x hx (j + 1) (by omega))
      _ = hankelC1 * x ^ (-(1 / 4 : ℝ)) *
            ∑ j ∈ Finset.range n, (d (j + 1) : ℝ) * pm94 (j + 1) := by rw [Finset.mul_sum]
      _ ≤ hankelC1 * x ^ (-(1 / 4 : ℝ)) * 1.462 ^ 2 :=
          mul_le_mul_of_nonneg_left (dsum94_partial_le n) (mul_nonneg hC1 hxr)
      _ = (hankelC1 * 1.462 ^ 2) * x ^ (-(1 / 4 : ℝ)) := by ring
      _ ≤ 4.03e-4 * x ^ (-(1 / 4 : ℝ)) := mul_le_mul_of_nonneg_right hankelC1_num hxr
  exact ⟨summable_of_sum_range_le (fun j => abs_nonneg _) hpart,
    Real.tsum_le_of_sum_range_le (fun j => abs_nonneg _) hpart⟩

/-- `2^{1/4} ≤ 1.1893`. -/
theorem two_rpow_quarter_le : (2 : ℝ) ^ (1 / 4 : ℝ) ≤ 1.1893 := by
  calc (2 : ℝ) ^ (1 / 4 : ℝ) ≤ ((1.1893 : ℝ) ^ 4) ^ (1 / 4 : ℝ) :=
        Real.rpow_le_rpow (by norm_num) (by norm_num) (by norm_num)
    _ = 1.1893 := rpow_quarter_pow4 _ (by norm_num)

/-- The note's §2 bound: for `x ≥ N/2`, `N ≥ 128`, the summed Hankel remainder is at most
`4.8e-4 · N^{-1/4}`. -/
theorem hankel_remainder_sum_N (ci : CitedInputs) (N x : ℝ) (hN : 128 ≤ N) (hx : N / 2 ≤ x) :
    Summable (fun j : ℕ => |hankelRem ci x (j + 1)|) ∧
      ∑' j : ℕ, |hankelRem ci x (j + 1)| ≤ 4.8e-4 * N ^ (-(1 / 4 : ℝ)) := by
  have hx64 : 64 ≤ x := by linarith
  have hx0 : 0 < x := by linarith
  have hN0 : 0 < N := by linarith
  obtain ⟨hs, ht⟩ := hankel_remainder_sum ci x hx64
  refine ⟨hs, ht.trans ?_⟩
  have h1 : (2 * x) ^ (-(1 / 4 : ℝ)) ≤ N ^ (-(1 / 4 : ℝ)) :=
    Real.rpow_le_rpow_of_nonpos hN0 (by linarith) (by norm_num)
  rw [Real.mul_rpow (by norm_num) hx0.le, Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 2)] at h1
  have hw : 0 < (2 : ℝ) ^ (1 / 4 : ℝ) := by positivity
  have h2 : x ^ (-(1 / 4 : ℝ)) ≤ (2 : ℝ) ^ (1 / 4 : ℝ) * N ^ (-(1 / 4 : ℝ)) := by
    have := mul_le_mul_of_nonneg_left h1 hw.le
    rwa [← mul_assoc, mul_inv_cancel₀ hw.ne', one_mul] at this
  have hNr : 0 ≤ N ^ (-(1 / 4 : ℝ)) := Real.rpow_nonneg hN0.le _
  have h3 : (2 : ℝ) ^ (1 / 4 : ℝ) * N ^ (-(1 / 4 : ℝ)) ≤ 1.1893 * N ^ (-(1 / 4 : ℝ)) :=
    mul_le_mul_of_nonneg_right two_rpow_quarter_le hNr
  nlinarith [h2, h3]

end DirichletDivisor

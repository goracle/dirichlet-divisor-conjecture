import DirichletDivisor.DualOracle

set_option linter.style.whitespace false

namespace DirichletDivisor

open MeasureTheory

/-! # Discharging `DualOracle` from a quadrature oracle for `F_{s,δ}` and the phase factor

The abstract `DualOracle` of `DualOracle.lean` asked for per-term values with per-term errors.
Here the numerical content is reduced to three elementary primitives, each a pointwise
(per-`m`) hypothesis:

* `Q3 j`, `Q1 j`: quadrature values for `F_{3/2,δ}(ξ_m)` and `F_{1/2,δ}(ξ_m)` with error `≤ εQ`
  (this is eq. (8) of the note, `εQ = N^{-Aq-2}`);
* `U j`: the computed phase factor `e^{iθ_m}` with error `≤ εφ` (this is `Ephase`).

The computed term is `d(m)(K₁·Im(U Q₃) + K₂·Re(U Q₁))`, and the exact error propagation
(`cmul_err`, `abs_combo`), the `Σ d(m)(K₁+K₂)` bound (via `dsum54_partial_le`) and the
budget arithmetic are proved here.
-/

/-! ## Error propagation -/

theorem cmul_err (U e Q F : ℂ) (εQ εφ : ℝ) (he : ‖e‖ = 1) (hF : ‖F‖ ≤ 2)
    (hQ : ‖Q - F‖ ≤ εQ) (hU : ‖U - e‖ ≤ εφ) :
    ‖U * Q - e * F‖ ≤ (1 + εφ) * εQ + 2 * εφ := by
  have h1 : U * Q - e * F = U * (Q - F) + (U - e) * F := by ring
  have hεφ0 : 0 ≤ εφ := (norm_nonneg _).trans hU
  have hUn : ‖U‖ ≤ 1 + εφ :=
    calc ‖U‖ = ‖(U - e) + e‖ := by rw [sub_add_cancel]
      _ ≤ ‖U - e‖ + ‖e‖ := norm_add_le _ _
      _ ≤ εφ + 1 := add_le_add hU he.le
      _ = 1 + εφ := add_comm _ _
  rw [h1]
  calc ‖U * (Q - F) + (U - e) * F‖ ≤ ‖U * (Q - F)‖ + ‖(U - e) * F‖ := norm_add_le _ _
    _ = ‖U‖ * ‖Q - F‖ + ‖U - e‖ * ‖F‖ := by rw [norm_mul, norm_mul]
    _ ≤ (1 + εφ) * εQ + εφ * 2 :=
        add_le_add (mul_le_mul hUn hQ (norm_nonneg _) (by linarith))
          (mul_le_mul hU hF (norm_nonneg _) hεφ0)
    _ = (1 + εφ) * εQ + 2 * εφ := by ring

theorem abs_combo (D a b : ℝ) (z w : ℂ) (E : ℝ) (hD : 0 ≤ D) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hz : ‖z‖ ≤ E) (hw : ‖w‖ ≤ E) :
    |D * (a * z.im + b * w.re)| ≤ D * (a + b) * E := by
  have hzi0 : |z.im| ≤ ‖z‖ := by
    first | exact Complex.abs_im_le_norm z | exact Complex.abs_im_le_abs z
  have hwr0 : |w.re| ≤ ‖w‖ := by
    first | exact Complex.abs_re_le_norm w | exact Complex.abs_re_le_abs w
  have hzi : |z.im| ≤ E := hzi0.trans hz
  have hwr : |w.re| ≤ E := hwr0.trans hw
  have h1 : |a * z.im + b * w.re| ≤ a * E + b * E :=
    calc |a * z.im + b * w.re| ≤ |a * z.im| + |b * w.re| := abs_add_le _ _
      _ = a * |z.im| + b * |w.re| := by
          rw [abs_mul, abs_mul, abs_of_nonneg ha, abs_of_nonneg hb]
      _ ≤ a * E + b * E :=
          add_le_add (mul_le_mul_of_nonneg_left hzi ha) (mul_le_mul_of_nonneg_left hwr hb)
  rw [abs_mul, abs_of_nonneg hD]
  calc D * |a * z.im + b * w.re| ≤ D * (a * E + b * E) := mul_le_mul_of_nonneg_left h1 hD
    _ = D * (a + b) * E := by ring

/-! ## The quadrature/phase oracle -/

/-- **[Oracle]** Quadrature values for `F_{3/2,δ}`, `F_{1/2,δ}` at `ξ_m` (error `εQ`) and the
computed phase factor `e^{iθ_m}` (error `εφ`), for `m = j + 1`, `j < M`. -/
structure FsQuadOracle (s L : ℝ) (M : ℕ) (εQ εφ : ℝ) where
  Q3 : ℕ → ℂ
  Q1 : ℕ → ℂ
  U : ℕ → ℂ
  hQ3 : ∀ j : ℕ, j < M → ‖Q3 j - Fs (3 / 2) (L / s) (dualFreq L (j + 1))‖ ≤ εQ
  hQ1 : ∀ j : ℕ, j < M → ‖Q1 j - Fs (1 / 2) (L / s) (dualFreq L (j + 1))‖ ≤ εQ
  hU : ∀ j : ℕ, j < M →
    ‖U j - Complex.exp ((dualPhase s (j + 1) : ℂ) * Complex.I)‖ ≤ εφ

/-- The algorithm's `j`-th dual term. -/
noncomputable def FsQuadOracle.comp {s L : ℝ} {M : ℕ} {εQ εφ : ℝ}
    (O : FsQuadOracle s L M εQ εφ) (j : ℕ) : ℝ :=
  (d (j + 1) : ℝ) * (dualK1 s (j + 1) * (O.U j * O.Q3 j).im
    + dualK2 s (j + 1) * (O.U j * O.Q1 j).re)

/-- The per-term error bound. -/
noncomputable def quadErr (s εQ εφ : ℝ) (j : ℕ) : ℝ :=
  (d (j + 1) : ℝ) * (dualK1 s (j + 1) + dualK2 s (j + 1)) * ((1 + εφ) * εQ + 2 * εφ)

theorem FsQuadOracle.hcomp' {s L : ℝ} {M : ℕ} {εQ εφ : ℝ} (O : FsQuadOracle s L M εQ εφ)
    (hs : 0 < s) (hL0 : 0 ≤ L) (hδ : L / s ≤ 1 / 2) (j : ℕ) (hj : j < M) :
    |O.comp j - dualTermExact s L (j + 1)| ≤ quadErr s εQ εφ j := by
  have hδ0 : 0 ≤ L / s := div_nonneg hL0 hs.le
  have hF3 : ‖Fs (3 / 2) (L / s) (dualFreq L (j + 1))‖ ≤ 2 :=
    (Fs_norm_lt_two (3 / 2) (L / s) (dualFreq L (j + 1)) (by norm_num) (by norm_num)
      hδ0 hδ).le
  have hF1 : ‖Fs (1 / 2) (L / s) (dualFreq L (j + 1))‖ ≤ 2 :=
    (Fs_norm_lt_two (1 / 2) (L / s) (dualFreq L (j + 1)) (by norm_num) (by norm_num)
      hδ0 hδ).le
  have he : ‖Complex.exp ((dualPhase s (j + 1) : ℂ) * Complex.I)‖ = 1 := by
    first
      | exact Complex.norm_exp_ofReal_mul_I _
      | simpa using Complex.abs_exp_ofReal_mul_I (dualPhase s (j + 1))
  have e3 := cmul_err (O.U j) _ (O.Q3 j) _ εQ εφ he hF3 (O.hQ3 j hj) (O.hU j hj)
  have e1 := cmul_err (O.U j) _ (O.Q1 j) _ εQ εφ he hF1 (O.hQ1 j hj) (O.hU j hj)
  have hd0 : (0 : ℝ) ≤ d (j + 1) := Nat.cast_nonneg _
  have key : O.comp j - dualTermExact s L (j + 1)
      = (d (j + 1) : ℝ) * (dualK1 s (j + 1) *
            (O.U j * O.Q3 j - Complex.exp ((dualPhase s (j + 1) : ℂ) * Complex.I)
              * Fs (3 / 2) (L / s) (dualFreq L (j + 1))).im
          + dualK2 s (j + 1) *
            (O.U j * O.Q1 j - Complex.exp ((dualPhase s (j + 1) : ℂ) * Complex.I)
              * Fs (1 / 2) (L / s) (dualFreq L (j + 1))).re) := by
    unfold FsQuadOracle.comp dualTermExact
    simp only [Complex.sub_im, Complex.sub_re]
    ring
  rw [key]
  exact abs_combo _ _ _ _ _ _ hd0 (dualK1_nonneg s (j + 1)) (dualK2_nonneg s (j + 1)) e3 e1

/-- Turn a quadrature oracle into a `DualOracle`. -/
noncomputable def FsQuadOracle.toDualOracle {s L : ℝ} {M : ℕ} {εQ εφ : ℝ}
    (O : FsQuadOracle s L M εQ εφ) (hs : 0 < s) (hL0 : 0 ≤ L) (hδ : L / s ≤ 1 / 2) :
    DualOracle s L M where
  comp := O.comp
  err := quadErr s εQ εφ
  hcomp := fun j hj => O.hcomp' hs hL0 hδ j hj

/-! ## Summing the errors -/

theorem dualC_le_cube (s : ℝ) (hs1 : 1 ≤ s) :
    1 / (2 * Real.sqrt 2 * Real.pi ^ 2) * (Real.sqrt s) ^ 3
      + 15 / (64 * Real.sqrt 2 * Real.pi ^ 3) * Real.sqrt s ≤ (Real.sqrt s) ^ 3 := by
  have ht : 1 ≤ Real.sqrt s := by simpa using Real.sqrt_le_sqrt hs1
  have h2 : 1 ≤ Real.sqrt 2 := by simpa using Real.sqrt_le_sqrt (by norm_num : (1 : ℝ) ≤ 2)
  have hπ : 3 ≤ Real.pi := Real.pi_gt_three.le
  have hπ2 : 9 ≤ Real.pi ^ 2 := by nlinarith
  have hπ3 : 27 ≤ Real.pi ^ 3 := by nlinarith [hπ2]
  have hc1 : 1 / (2 * Real.sqrt 2 * Real.pi ^ 2) ≤ 1 / 2 := by
    rw [div_le_iff₀ (by positivity)]
    nlinarith [mul_le_mul h2 hπ2 (by norm_num) (by linarith)]
  have hc2 : 15 / (64 * Real.sqrt 2 * Real.pi ^ 3) ≤ 1 / 2 := by
    rw [div_le_iff₀ (by positivity)]
    nlinarith [mul_le_mul h2 hπ3 (by norm_num) (by linarith)]
  have ht0 : 0 ≤ Real.sqrt s := by linarith
  have ht2 : 1 ≤ (Real.sqrt s) ^ 2 := one_le_pow₀ ht
  have h3 : Real.sqrt s ≤ (Real.sqrt s) ^ 3 :=
    calc Real.sqrt s = Real.sqrt s * 1 := (mul_one _).symm
      _ ≤ Real.sqrt s * (Real.sqrt s) ^ 2 := mul_le_mul_of_nonneg_left ht2 ht0
      _ = (Real.sqrt s) ^ 3 := by ring
  have a1 := mul_le_mul_of_nonneg_right hc1 (pow_nonneg ht0 3)
  have a2 := mul_le_mul_of_nonneg_right hc2 ht0
  linarith

theorem sum_dK_le (s E : ℝ) (hs1 : 1 ≤ s) (hE : 0 ≤ E) (M : ℕ) :
    ∑ j ∈ Finset.range M, (d (j + 1) : ℝ) * (dualK1 s (j + 1) + dualK2 s (j + 1)) * E
      ≤ 25 * (Real.sqrt s) ^ 3 * E := by
  have hC := dualC_le_cube s hs1
  have hC3 : 0 ≤ (Real.sqrt s) ^ 3 * E := mul_nonneg (pow_nonneg (Real.sqrt_nonneg _) 3) hE
  calc ∑ j ∈ Finset.range M, (d (j + 1) : ℝ) * (dualK1 s (j + 1) + dualK2 s (j + 1)) * E
      ≤ ∑ j ∈ Finset.range M, ((Real.sqrt s) ^ 3 * E) * ((d (j + 1) : ℝ) * pm54 (j + 1)) := by
        refine Finset.sum_le_sum (fun j _ => ?_)
        have hd0 : (0 : ℝ) ≤ d (j + 1) := Nat.cast_nonneg _
        have h := dualK_le_54 s (j + 1) (by omega)
        have h' : dualK1 s (j + 1) + dualK2 s (j + 1) ≤ (Real.sqrt s) ^ 3 * pm54 (j + 1) :=
          h.trans (mul_le_mul_of_nonneg_right hC (pm54_nonneg _))
        calc (d (j + 1) : ℝ) * (dualK1 s (j + 1) + dualK2 s (j + 1)) * E
            ≤ (d (j + 1) : ℝ) * ((Real.sqrt s) ^ 3 * pm54 (j + 1)) * E :=
              mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left h' hd0) hE
          _ = ((Real.sqrt s) ^ 3 * E) * ((d (j + 1) : ℝ) * pm54 (j + 1)) := by ring
    _ = ((Real.sqrt s) ^ 3 * E) * ∑ j ∈ Finset.range M, (d (j + 1) : ℝ) * pm54 (j + 1) := by
        rw [Finset.mul_sum]
    _ ≤ ((Real.sqrt s) ^ 3 * E) * 25 := mul_le_mul_of_nonneg_left (dsum54_partial_le M) hC3
    _ = 25 * (Real.sqrt s) ^ 3 * E := by ring

theorem quadErr_sum_le (s εQ εφ : ℝ) (hs1 : 1 ≤ s) (hE : 0 ≤ (1 + εφ) * εQ + 2 * εφ) (M : ℕ) :
    ∑ j ∈ Finset.range M, quadErr s εQ εφ j
      ≤ 25 * (Real.sqrt s) ^ 3 * ((1 + εφ) * εQ + 2 * εφ) := by
  unfold quadErr
  exact sum_dK_le s _ hs1 hE M

theorem sqrt_cube_le (s N : ℝ) (hs1 : 1 ≤ s) (hsN : s ^ 2 ≤ N) : (Real.sqrt s) ^ 3 ≤ N := by
  have ht : 1 ≤ Real.sqrt s := by simpa using Real.sqrt_le_sqrt hs1
  have ht2 : (Real.sqrt s) ^ 2 = s := Real.sq_sqrt (by linarith)
  have h4 : (Real.sqrt s) ^ 3 ≤ (Real.sqrt s) ^ 4 := pow_le_pow_right₀ ht (by norm_num)
  have e : (Real.sqrt s) ^ 4 = s ^ 2 :=
    calc (Real.sqrt s) ^ 4 = ((Real.sqrt s) ^ 2) ^ 2 := by ring
      _ = s ^ 2 := by rw [ht2]
  linarith

theorem quad_budget (N c E : ℝ) (hN0 : 0 < N) (hc : c ≤ N) (hE0 : 0 ≤ E)
    (hE : E ≤ (N ^ 5)⁻¹) : 25 * c * E ≤ 2 ^ 58 * (N ^ 4)⁻¹ := by
  have h1 : 25 * c * E ≤ 25 * N * E := by
    have := mul_le_mul_of_nonneg_right hc hE0
    linarith
  have h2 : 25 * N * E ≤ 25 * N * (N ^ 5)⁻¹ :=
    mul_le_mul_of_nonneg_left hE (by positivity)
  have e5 : N * (N ^ 5)⁻¹ = (N ^ 4)⁻¹ := by
    field_simp <;> ring
  have hq : 0 ≤ (N ^ 4)⁻¹ := by positivity
  calc 25 * c * E ≤ 25 * N * (N ^ 5)⁻¹ := h1.trans h2
    _ = 25 * (N * (N ^ 5)⁻¹) := by ring
    _ = 25 * (N ^ 4)⁻¹ := by rw [e5]
    _ ≤ 2 ^ 58 * (N ^ 4)⁻¹ := mul_le_mul_of_nonneg_right (by norm_num) hq

/-! ## Parameter facts at `H = N^{9/28}` -/

theorem tail_one_le_s (N s σ : ℝ) (hN2 : 2 ≤ N) (hσ0 : 0 ≤ σ) (hσ1 : σ ≤ 1)
    (hs0 : 0 ≤ s) (hs : s ^ 2 = N - tailL N ^ 2 * σ) : 1 ≤ s := by
  have hN1 : 1 ≤ N := by linarith
  have hN0 : 0 < N := by linarith
  have hL0 : 0 ≤ tailL N := (tailL_pos N hN0).le
  have hL4 : tailL N ≤ 1 / 4 := tailL_le N hN1
  have hL2 : tailL N ^ 2 ≤ 1 / 16 :=
    calc tailL N ^ 2 ≤ (1 / 4) ^ 2 := pow_le_pow_left₀ hL0 hL4 2
      _ = 1 / 16 := by norm_num
  have hLσ : tailL N ^ 2 * σ ≤ 1 / 16 :=
    calc tailL N ^ 2 * σ ≤ tailL N ^ 2 * 1 := mul_le_mul_of_nonneg_left hσ1 (sq_nonneg _)
      _ = tailL N ^ 2 := mul_one _
      _ ≤ 1 / 16 := hL2
  have hs2 : 1 ≤ s ^ 2 := by rw [hs]; linarith
  by_contra h
  have h' := not_le.mp h
  nlinarith [hs2, hs0, h']

theorem param_facts (n : ℕ) (H sigh : ℝ) (hH : H = ((n : ℝ) + 1) ^ (9 / 28 : ℝ))
    (hN : 2 ^ 65536 ≤ n + 1) (hsig0 : 0 ≤ sigh) (hsig1 : sigh ≤ 1) :
    1 ≤ parS (n + 1) H sigh ∧ 0 ≤ parL (n + 1) H ∧
      parL (n + 1) H / parS (n + 1) H sigh ≤ 1 / 2 ∧
      parS (n + 1) H sigh ^ 2 ≤ (n : ℝ) + 1 := by
  have hN' : (2 : ℝ) ^ 65536 ≤ (n : ℝ) + 1 := by exact_mod_cast hN
  have hN2 : (2 : ℝ) ≤ (n : ℝ) + 1 := le_trans two_le_two_pow_65536 hN'
  have hN1 : (1 : ℝ) ≤ (n : ℝ) + 1 := le_trans (by norm_num) hN2
  have hN0 : (0 : ℝ) < (n : ℝ) + 1 := lt_of_lt_of_le one_pos hN1
  have hH0 : 0 ≤ H := by rw [hH]; exact Real.rpow_nonneg hN0.le _
  have hHN : H ≤ (n : ℝ) + 1 := by
    rw [hH]
    calc ((n : ℝ) + 1) ^ (9 / 28 : ℝ) ≤ ((n : ℝ) + 1) ^ (1 : ℝ) :=
          Real.rpow_le_rpow_of_exponent_le hN1 (by norm_num)
      _ = (n : ℝ) + 1 := Real.rpow_one _
  have hNpos : (0 : ℝ) < ((n + 1 : ℕ) : ℝ) := by positivity
  have hHN' : H ≤ ((n + 1 : ℕ) : ℝ) := by push_cast; exact hHN
  have hs := parS_sq (n + 1) H sigh hNpos hH0 hHN' hsig0 hsig1
  push_cast at hs
  rw [parL_eq_tailL n H hH] at hs
  obtain ⟨_, hsN, _, hδ⟩ := tail_side_conditions ((n : ℝ) + 1)
    (parS (n + 1) H sigh) sigh hN2 hsig0 hsig1 (Real.sqrt_nonneg _) hs
  have h1 := tail_one_le_s ((n : ℝ) + 1) (parS (n + 1) H sigh) sigh hN2 hsig0 hsig1
    (Real.sqrt_nonneg _) hs
  refine ⟨h1, parL_nonneg (n + 1) H hH0, ?_, hsN⟩
  rw [parL_eq_tailL n H hH]
  exact hδ

/-! ## Final theorem -/

/-- **Correctness of rounding, modulo `CitedInputs` and a quadrature/phase oracle.**
The oracle supplies quadrature values for `F_{s,δ}(ξ_m)` (error `εQ`, eq. (8) of the note) and the
computed phase factors (error `εφ`). The only numerical conditions are
`(1+εφ)εQ + 2εφ ≤ N^{-5}` and the `σ̂²` budget `hσbud`. -/
theorem hoyingSum_eq_round_quad (ci : CitedInputs) (n : ℕ) (H sigh ε εQ εφ : ℝ) (M : ℕ)
    (hH : H = ((n : ℝ) + 1) ^ (9 / 28 : ℝ))
    (hN : 2 ^ 65536 ≤ n + 1)
    (hsig0 : 0 ≤ sigh) (hsig1 : sigh ≤ 1)
    (hsigε : |etaVar - sigh| ≤ ε)
    (hM1 : tailM0 ((n : ℝ) + 1) ≤ (M : ℝ)) (hM2 : (M : ℝ) ≤ tailM0 ((n : ℝ) + 1) + 1)
    (hεQ0 : 0 ≤ εQ) (hεφ0 : 0 ≤ εφ)
    (hquad : (1 + εφ) * εQ + 2 * εφ ≤ (((n : ℝ) + 1) ^ 5)⁻¹)
    (O : FsQuadOracle (parS (n + 1) H sigh) (parL (n + 1) H) M εQ εφ)
    (hσbud : H ^ 2 * ε * (1 + Real.log (2 * ((n : ℝ) + 1))) / 8
      ≤ 2 ^ 58 * (((n : ℝ) + 1) ^ 4)⁻¹) :
    (hoyingSum n : ℤ)
      = round (idealAhat (parS (n + 1) H sigh) (parL (n + 1) H) (parK (n + 1) H sigh)
          (∑ j ∈ Finset.range M, O.comp j)) := by
  obtain ⟨h1, hL0, hδ, hsN⟩ := param_facts n H sigh hH hN hsig0 hsig1
  have hs_pos : 0 < parS (n + 1) H sigh := lt_of_lt_of_le one_pos h1
  have hN' : (2 : ℝ) ^ 65536 ≤ (n : ℝ) + 1 := by exact_mod_cast hN
  have hN2 : (2 : ℝ) ≤ (n : ℝ) + 1 := le_trans two_le_two_pow_65536 hN'
  have hN0 : (0 : ℝ) < (n : ℝ) + 1 :=
    lt_of_lt_of_le one_pos (le_trans (by norm_num) hN2)
  have hE0 : 0 ≤ (1 + εφ) * εQ + 2 * εφ :=
    add_nonneg (mul_nonneg (by linarith) hεQ0) (by linarith)
  have hsum : ∑ j ∈ Finset.range M, (O.toDualOracle hs_pos hL0 hδ).err j
      ≤ 2 ^ 58 * (((n : ℝ) + 1) ^ 4)⁻¹ := by
    show ∑ j ∈ Finset.range M, quadErr (parS (n + 1) H sigh) εQ εφ j ≤ _
    exact (quadErr_sum_le _ εQ εφ h1 hE0 M).trans
      (quad_budget ((n : ℝ) + 1) _ _ hN0
        (sqrt_cube_le _ _ h1 hsN) hE0 hquad)
  exact hoyingSum_eq_round_oracle ci n H sigh ε M hH hN hsig0 hsig1 hsigε hM1 hM2
    (O.toDualOracle hs_pos hL0 hδ) hsum hσbud

end DirichletDivisor

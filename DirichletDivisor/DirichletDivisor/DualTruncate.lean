import DirichletDivisor.DualFourier

set_option linter.style.whitespace false

namespace DirichletDivisor

open MeasureTheory

/-! # Truncating the dual integral

`∫ η·dualMain(x(u)) du = ∑_{m ≤ M} ∫ η·mainTerm_m + tail`, with `|tail| ≤ N^{-6}`.

1. `Σ d(m) m^{-5/4}` converges (partial sums ≤ 25, via `dsum_le_sq`).
2. Each `η·mainTerm_m` is a continuous function on `[-1,1]` bounded by `2 d(m)(K₁+K₂) η`,
   so `integral_tsum_of_summable_integral_norm` swaps `∑'` and `∫`.
3. Split off the first `M` terms; bound the rest by `dual_term_bound` and `tail_param_le`.
-/

/-! ## Summability of `d(m) m^{-5/4}` -/

noncomputable def pm54 (k : ℕ) : ℝ := (k : ℝ) ^ (-(5 / 4 : ℝ))

theorem pm54_nonneg (k : ℕ) : 0 ≤ pm54 k := Real.rpow_nonneg (Nat.cast_nonneg k) _

theorem pm54_mul (a b : ℕ) : pm54 (a * b) = pm54 a * pm54 b := by
  unfold pm54
  push_cast
  exact Real.mul_rpow (Nat.cast_nonneg a) (Nat.cast_nonneg b)

theorem pm54_sum_le (n : ℕ) : ∑ k ∈ Finset.range n, pm54 (k + 1) ≤ 5 := by
  cases n with
  | zero => simp
  | succ n =>
    rw [Finset.sum_range_succ']
    have h := pseries_partial_le (5 / 4) (by norm_num) 1 n le_rfl
    have e : ∑ i ∈ Finset.range n, pm54 (i + 1 + 1)
        = ∑ k ∈ Finset.range n, (((1 + k + 1 : ℕ) : ℝ)) ^ (-(5 / 4 : ℝ)) := by
      refine Finset.sum_congr rfl (fun i _ => ?_)
      unfold pm54
      rw [show i + 1 + 1 = 1 + i + 1 by omega]
    have h1 : pm54 (0 + 1) = 1 := by simp [pm54]
    have h4 : (((1 : ℕ) : ℝ)) ^ (1 - (5 / 4 : ℝ)) / ((5 / 4 : ℝ) - 1) = 4 := by norm_num
    rw [e, h1]
    linarith [h, h4]

theorem dsum54_partial_le (n : ℕ) :
    ∑ j ∈ Finset.range n, (d (j + 1) : ℝ) * pm54 (j + 1) ≤ 25 := by
  have h := dsum_le_sq pm54 pm54_nonneg pm54_mul n
  rw [sum_Icc_eq_range (fun m => (d m : ℝ) * pm54 m), sum_Icc_eq_range pm54] at h
  have hP := pm54_sum_le n
  have hP0 : 0 ≤ ∑ k ∈ Finset.range n, pm54 (k + 1) :=
    Finset.sum_nonneg (fun k _ => pm54_nonneg _)
  have := pow_le_pow_left₀ hP0 hP 2
  linarith

theorem summable_dsum54 : Summable (fun j : ℕ => (d (j + 1) : ℝ) * pm54 (j + 1)) :=
  summable_of_sum_range_le (fun j => mul_nonneg (Nat.cast_nonneg _) (pm54_nonneg _))
    dsum54_partial_le

/-- `K₁ + K₂ ≲ m^{-5/4}` (the summable envelope). -/
theorem dualK_le_54 (s : ℝ) (m : ℕ) (hm : 1 ≤ m) :
    dualK1 s m + dualK2 s m ≤
      (1 / (2 * Real.sqrt 2 * Real.pi ^ 2) * (Real.sqrt s) ^ 3
        + 15 / (64 * Real.sqrt 2 * Real.pi ^ 3) * Real.sqrt s) * pm54 m := by
  have hm0 : (0 : ℝ) < m := by exact_mod_cast hm
  have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast hm
  have hb1 : 1 ≤ Real.sqrt (Real.sqrt (m : ℝ)) := one_le_sqrt_sqrt _ hm1
  have h54 : pm54 m = ((Real.sqrt (Real.sqrt (m : ℝ))) ^ 5)⁻¹ := by
    unfold pm54
    rw [Real.rpow_neg hm0.le, show (5 / 4 : ℝ) = ((5 : ℕ) : ℝ) / 4 by norm_num,
      ← pow_sqrt_sqrt _ hm0.le]
  rw [h54]
  unfold dualK1 dualK2
  generalize Real.sqrt (Real.sqrt (m : ℝ)) = b at hb1 ⊢
  have hb0 : 0 < b := by linarith
  have e7 : (b ^ 7)⁻¹ ≤ (b ^ 5)⁻¹ :=
    inv_anti₀ (by positivity) (pow_le_pow_right₀ hb1 (by norm_num))
  have hC2 : 0 ≤ 15 / (64 * Real.sqrt 2 * Real.pi ^ 3) := by positivity
  have hr : 0 ≤ Real.sqrt s := Real.sqrt_nonneg _
  have h1 := mul_le_mul_of_nonneg_left e7 hC2
  nlinarith [mul_le_mul_of_nonneg_right h1 hr]

/-! ## Continuous representatives with an `η`-envelope -/

theorem abs_coef_trig_le (c e a w : ℝ) (hc : 0 ≤ c) (he : 0 ≤ e) (ha : 0 ≤ a) (hw : |w| ≤ 1) :
    |c * (e * a * w)| ≤ c * (e * a) := by
  rw [abs_mul, abs_of_nonneg hc, abs_mul, abs_of_nonneg (mul_nonneg he ha)]
  exact mul_le_mul_of_nonneg_left (mul_le_of_le_one_right (mul_nonneg he ha) hw) hc

theorem eta_mainTerm_rep (s L : ℝ) (m : ℕ) (hm : 1 ≤ m) (hs : 0 < s) (hL0 : 0 ≤ L)
    (hLs : L < s) (hδ : L / s ≤ 1 / 2) :
    ∃ G : ℝ → ℝ, Continuous G ∧
      (∀ u ∈ Set.Icc (-1 : ℝ) 1, eta u * mainTerm ((s + L * u) ^ 2) m = G u) ∧
      (∀ u ∈ Set.Icc (-1 : ℝ) 1,
        |G u| ≤ 2 * ((d m : ℝ) * (dualK1 s m + dualK2 s m)) * eta u) := by
  obtain ⟨θ, hθ⟩ : ∃ θ : ℝ, θ = 4 * Real.pi * Real.sqrt (m : ℝ) * s - 5 * Real.pi / 4 :=
    ⟨_, rfl⟩
  obtain ⟨ξ, hξ⟩ : ∃ ξ : ℝ, ξ = 2 * L * Real.sqrt (m : ℝ) := ⟨_, rfl⟩
  refine ⟨fun u => (d m : ℝ) * (dualK1 s m * (eta u * (1 + (L / s) * u) ^ (3 / 2 : ℝ)
            * Real.sin (θ + 2 * Real.pi * ξ * u))
          + dualK2 s m * (eta u * (1 + (L / s) * u) ^ (1 / 2 : ℝ)
            * Real.cos (θ + 2 * Real.pi * ξ * u))), ?_, ?_, ?_⟩
  · exact continuous_const.mul
      ((continuous_const.mul (continuous_eta_pow_sin (3 / 2) (L / s) θ ξ (by norm_num))).add
        (continuous_const.mul (continuous_eta_pow_cos (1 / 2) (L / s) θ ξ (by norm_num))))
  · intro u hu
    exact eta_mainTerm_pt s L u m hm hs hL0 hLs hu θ ξ hθ hξ
  · intro u hu
    have hdiv0 : 0 ≤ L / s := div_nonneg hL0 hs.le
    have h1 : (L / s) * u ≤ L / s := by
      have := mul_le_mul_of_nonneg_left hu.2 hdiv0
      linarith
    have h2 : -(L / s) ≤ (L / s) * u := by
      have := mul_le_mul_of_nonneg_left hu.1 hdiv0
      linarith
    have hx0 : 0 ≤ 1 + (L / s) * u := by linarith
    have hx1 : 1 + (L / s) * u ≤ 3 / 2 := by linarith
    have ha : (1 + (L / s) * u) ^ (3 / 2 : ℝ) ≤ 2 :=
      (Real.rpow_le_rpow hx0 hx1 (by norm_num)).trans three_halves_rpow_lt_two.le
    have hb : (1 + (L / s) * u) ^ (1 / 2 : ℝ) ≤ 2 := by
      rw [← Real.sqrt_eq_rpow]
      exact Real.sqrt_le_iff.mpr ⟨by norm_num, by nlinarith⟩
    have ha0 : 0 ≤ (1 + (L / s) * u) ^ (3 / 2 : ℝ) := Real.rpow_nonneg hx0 _
    have hb0 : 0 ≤ (1 + (L / s) * u) ^ (1 / 2 : ℝ) := Real.rpow_nonneg hx0 _
    have he := eta_nonneg u
    have hK1 := dualK1_nonneg s m
    have hK2 := dualK2_nonneg s m
    have hd0 : (0 : ℝ) ≤ d m := Nat.cast_nonneg _
    have t1 := abs_coef_trig_le (dualK1 s m) (eta u) ((1 + (L / s) * u) ^ (3 / 2 : ℝ))
      (Real.sin (θ + 2 * Real.pi * ξ * u)) hK1 he ha0 (Real.abs_sin_le_one _)
    have t2 := abs_coef_trig_le (dualK2 s m) (eta u) ((1 + (L / s) * u) ^ (1 / 2 : ℝ))
      (Real.cos (θ + 2 * Real.pi * ξ * u)) hK2 he hb0 (Real.abs_cos_le_one _)
    have u1 : dualK1 s m * (eta u * (1 + (L / s) * u) ^ (3 / 2 : ℝ))
        ≤ dualK1 s m * (eta u * 2) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left ha he) hK1
    have u2 : dualK2 s m * (eta u * (1 + (L / s) * u) ^ (1 / 2 : ℝ))
        ≤ dualK2 s m * (eta u * 2) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hb he) hK2
    show |(d m : ℝ) * (dualK1 s m * (eta u * (1 + (L / s) * u) ^ (3 / 2 : ℝ)
            * Real.sin (θ + 2 * Real.pi * ξ * u))
          + dualK2 s m * (eta u * (1 + (L / s) * u) ^ (1 / 2 : ℝ)
            * Real.cos (θ + 2 * Real.pi * ξ * u)))| ≤ _
    rw [abs_mul, abs_of_nonneg hd0]
    calc (d m : ℝ) * |dualK1 s m * (eta u * (1 + (L / s) * u) ^ (3 / 2 : ℝ)
            * Real.sin (θ + 2 * Real.pi * ξ * u))
          + dualK2 s m * (eta u * (1 + (L / s) * u) ^ (1 / 2 : ℝ)
            * Real.cos (θ + 2 * Real.pi * ξ * u))|
        ≤ (d m : ℝ) * (dualK1 s m * (eta u * 2) + dualK2 s m * (eta u * 2)) := by
          refine mul_le_mul_of_nonneg_left ?_ hd0
          calc _ ≤ _ := abs_add_le _ _
            _ ≤ dualK1 s m * (eta u * (1 + (L / s) * u) ^ (3 / 2 : ℝ))
                + dualK2 s m * (eta u * (1 + (L / s) * u) ^ (1 / 2 : ℝ)) := add_le_add t1 t2
            _ ≤ _ := add_le_add u1 u2
      _ = _ := by ring

/-! ## The sum-integral exchange -/

noncomputable def Fterm (s L : ℝ) (j : ℕ) (u : ℝ) : ℝ :=
  eta u * mainTerm ((s + L * u) ^ 2) (j + 1)

theorem dual_integral_tsum (s L : ℝ) (hs : 0 < s) (hL0 : 0 ≤ L) (hLs : L < s)
    (hδ : L / s ≤ 1 / 2) :
    Summable (fun j : ℕ => |∫ u in (-1 : ℝ)..1, Fterm s L j u|) ∧
    (∫ u in (-1 : ℝ)..1, eta u * dualMain ((s + L * u) ^ 2))
      = ∑' j : ℕ, ∫ u in (-1 : ℝ)..1, Fterm s L j u := by
  have rep : ∀ j : ℕ, ∃ G : ℝ → ℝ, Continuous G ∧
      (∀ u ∈ Set.Icc (-1 : ℝ) 1, eta u * mainTerm ((s + L * u) ^ 2) (j + 1) = G u) ∧
      (∀ u ∈ Set.Icc (-1 : ℝ) 1,
        |G u| ≤ 2 * ((d (j + 1) : ℝ) * (dualK1 s (j + 1) + dualK2 s (j + 1))) * eta u) :=
    fun j => eta_mainTerm_rep s L (j + 1) (by omega) hs hL0 hLs hδ
  choose G hGc hGeq hGb using rep
  have hFint : ∀ j : ℕ, Integrable (Fterm s L j) (volume.restrict (Set.Ioc (-1 : ℝ) 1)) := by
    intro j
    have h1 : IntegrableOn (G j) (Set.Ioc (-1 : ℝ) 1) volume :=
      ((hGc j).integrableOn_Icc (a := -1) (b := 1)).mono_set Set.Ioc_subset_Icc_self
    exact h1.congr_fun
      (fun u hu => (hGeq j u (Set.Ioc_subset_Icc_self hu)).symm) measurableSet_Ioc
  have hetaI : ∫ u in Set.Ioc (-1 : ℝ) 1, eta u = 1 := by
    rw [← intervalIntegral.integral_of_le (by norm_num : (-1 : ℝ) ≤ 1)]
    exact eta_integral
  set Cs : ℝ := 1 / (2 * Real.sqrt 2 * Real.pi ^ 2) * (Real.sqrt s) ^ 3
        + 15 / (64 * Real.sqrt 2 * Real.pi ^ 3) * Real.sqrt s with hCs
  have hnorm : ∀ j : ℕ, ∫ u in Set.Ioc (-1 : ℝ) 1, ‖Fterm s L j u‖
      ≤ (2 * Cs) * ((d (j + 1) : ℝ) * pm54 (j + 1)) := by
    intro j
    have hint2 : IntegrableOn (fun u => 2 * ((d (j + 1) : ℝ) * (dualK1 s (j + 1)
        + dualK2 s (j + 1))) * eta u) (Set.Ioc (-1 : ℝ) 1) volume :=
      ((continuous_const.mul eta_continuous).integrableOn_Icc (a := -1) (b := 1)).mono_set
        Set.Ioc_subset_Icc_self
    calc ∫ u in Set.Ioc (-1 : ℝ) 1, ‖Fterm s L j u‖
        ≤ ∫ u in Set.Ioc (-1 : ℝ) 1, 2 * ((d (j + 1) : ℝ) * (dualK1 s (j + 1)
            + dualK2 s (j + 1))) * eta u := by
          refine setIntegral_mono_on (hFint j).norm hint2 measurableSet_Ioc (fun u hu => ?_)
          have hu' := Set.Ioc_subset_Icc_self hu
          rw [Real.norm_eq_abs]
          have e : Fterm s L j u = G j u := hGeq j u hu'
          rw [e]
          exact hGb j u hu'
      _ = 2 * ((d (j + 1) : ℝ) * (dualK1 s (j + 1) + dualK2 s (j + 1))) := by
          rw [integral_const_mul, hetaI, mul_one]
      _ ≤ (2 * Cs) * ((d (j + 1) : ℝ) * pm54 (j + 1)) := by
          have hk := dualK_le_54 s (j + 1) (by omega)
          have hd0 : (0 : ℝ) ≤ d (j + 1) := Nat.cast_nonneg _
          have := mul_le_mul_of_nonneg_left hk hd0
          nlinarith [this]
  have hsum : Summable (fun j : ℕ => ∫ u in Set.Ioc (-1 : ℝ) 1, ‖Fterm s L j u‖) :=
    Summable.of_nonneg_of_le (fun j => integral_nonneg (fun u => norm_nonneg _)) hnorm
      (summable_dsum54.mul_left (2 * Cs))
  have e1 : ∀ j : ℕ, ∫ u in (-1 : ℝ)..1, Fterm s L j u
      = ∫ u in Set.Ioc (-1 : ℝ) 1, Fterm s L j u :=
    fun j => intervalIntegral.integral_of_le (by norm_num)
  refine ⟨?_, ?_⟩
  · refine Summable.of_nonneg_of_le (fun _ => abs_nonneg _) (fun j => ?_) hsum
    rw [e1 j]
    have := norm_integral_le_integral_norm (μ := volume.restrict (Set.Ioc (-1 : ℝ) 1))
      (Fterm s L j)
    rwa [Real.norm_eq_abs] at this
  · have hint := integral_tsum_of_summable_integral_norm
      (μ := volume.restrict (Set.Ioc (-1 : ℝ) 1)) (F := Fterm s L) hFint hsum
    have e2 : ∀ u : ℝ, eta u * dualMain ((s + L * u) ^ 2) = ∑' j : ℕ, Fterm s L j u := by
      intro u
      unfold dualMain Fterm
      exact (tsum_mul_left).symm
    calc (∫ u in (-1 : ℝ)..1, eta u * dualMain ((s + L * u) ^ 2))
        = ∫ u in Set.Ioc (-1 : ℝ) 1, eta u * dualMain ((s + L * u) ^ 2) :=
          intervalIntegral.integral_of_le (by norm_num)
      _ = ∫ u in Set.Ioc (-1 : ℝ) 1, ∑' j : ℕ, Fterm s L j u := by
          congr 1; funext u; exact e2 u
      _ = ∑' j : ℕ, ∫ u in Set.Ioc (-1 : ℝ) 1, Fterm s L j u := hint.symm
      _ = ∑' j : ℕ, ∫ u in (-1 : ℝ)..1, Fterm s L j u := tsum_congr (fun j => (e1 j).symm)

/-! ## Truncation at `M` -/

theorem dual_truncation (N s : ℝ) (M : ℕ) (hN : (2 : ℝ) ^ 65536 ≤ N)
    (hM1 : tailM0 N ≤ (M : ℝ)) (hM2 : (M : ℝ) ≤ tailM0 N + 1)
    (hs : 0 < s) (hsN : s ^ 2 ≤ N) (hLs : tailL N < s) (hδ : tailL N / s ≤ 1 / 2) :
    |(∫ u in (-1 : ℝ)..1, eta u * dualMain ((s + tailL N * u) ^ 2))
      - ∑ j ∈ Finset.range M, ∫ u in (-1 : ℝ)..1, Fterm s (tailL N) j u|
      ≤ N ^ (-6 : ℝ) := by
  have hN1 : (1 : ℝ) ≤ N := le_trans (one_le_pow₀ (by norm_num)) hN
  have hN0 : 0 < N := lt_of_lt_of_le one_pos hN1
  have hL0 : 0 ≤ tailL N := (tailL_pos N hN0).le
  obtain ⟨hab, heq⟩ := dual_integral_tsum s (tailL N) hs hL0 hLs hδ
  obtain ⟨f, hfdef⟩ : ∃ f : ℕ → ℝ, ∀ j, f j = ∫ u in (-1 : ℝ)..1, Fterm s (tailL N) j u :=
    ⟨_, fun j => rfl⟩
  have hab' : Summable (fun j => |f j|) := hab.congr (fun j => by rw [hfdef j])
  have hsf : Summable f := summable_abs_iff.mp hab'
  have hsplit : ∑ i ∈ Finset.range M, f i + ∑' i, f (i + M) = ∑' i, f i :=
    hsf.sum_add_tsum_nat_add M
  have heq' : (∫ u in (-1 : ℝ)..1, eta u * dualMain ((s + tailL N * u) ^ 2)) = ∑' j, f j := by
    rw [heq]; exact tsum_congr (fun j => (hfdef j).symm)
  have hsum_eq : ∑ j ∈ Finset.range M, ∫ u in (-1 : ℝ)..1, Fterm s (tailL N) j u
      = ∑ j ∈ Finset.range M, f j := Finset.sum_congr rfl (fun j _ => (hfdef j).symm)
  rw [heq', hsum_eq, ← hsplit, add_sub_cancel_left]
  have hshift : Summable (fun i => |f (i + M)|) := (summable_nat_add_iff M).2 hab'
  have hshift' : Summable (fun i => ‖f (i + M)‖) := by
    simpa only [Real.norm_eq_abs] using hshift
  calc |∑' i, f (i + M)| = ‖∑' i, f (i + M)‖ := (Real.norm_eq_abs _).symm
    _ ≤ ∑' i, ‖f (i + M)‖ := norm_tsum_le_tsum_norm hshift'
    _ ≤ N ^ (-6 : ℝ) := by
      refine Real.tsum_le_of_sum_range_le (fun i => norm_nonneg _) (fun n => ?_)
      calc ∑ i ∈ Finset.range n, ‖f (i + M)‖
          ≤ ∑ i ∈ Finset.range n, 2 ^ 16 * N ^ (3 / 4 : ℝ) *
              ((d (M + i + 1) : ℝ) * (((M + i + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ))
                * Real.exp (-(tailC N * ((M + i + 1 : ℕ) : ℝ) ^ (1 / 4 : ℝ))))) := by
            refine Finset.sum_le_sum (fun i _ => ?_)
            have e : i + M + 1 = M + i + 1 := by omega
            have h := dual_term_bound N s (i + M + 1) hN1 (by omega) hs hsN hLs hδ
            rw [e] at h
            rw [Real.norm_eq_abs, hfdef (i + M)]
            show |∫ u in (-1 : ℝ)..1, eta u * mainTerm ((s + tailL N * u) ^ 2) (i + M + 1)| ≤ _
            rw [e]
            exact h
        _ = 2 ^ 16 * N ^ (3 / 4 : ℝ) *
              ∑ i ∈ Finset.range n, (d (M + i + 1) : ℝ) * (((M + i + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ))
                * Real.exp (-(tailC N * ((M + i + 1 : ℕ) : ℝ) ^ (1 / 4 : ℝ)))) :=
            (Finset.mul_sum _ _ _).symm
        _ ≤ N ^ (-6 : ℝ) := tail_param_le N hN M n hM1 hM2

end DirichletDivisor

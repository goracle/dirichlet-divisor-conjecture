import DirichletDivisor.Idealized

set_option linter.style.whitespace false

namespace DirichletDivisor

open Finset MeasureTheory Filter Topology

/-! ## Discharging `hD`: integrability of `u ↦ η(u) · dualMain((s+Lu)²)`

No continuity of `Y₂` is needed. `mainTerm` is elementary, hence measurable; `dualMain` is the
pointwise limit of partial sums on the window (by `voronoi_pointwise`), hence a.e.-measurable;
and `η·dualMain = (ηP - ηR) + η·(R - (P - dualMain))` is bounded by an integrable function
via the pointwise Voronoï bound. -/

theorem mainTerm_measurable (s L : ℝ) (m : ℕ) :
    Measurable fun u : ℝ => mainTerm ((s + L * u) ^ 2) m := by
  unfold mainTerm hankelMain
  first | fun_prop | measurability

theorem intervalIntegrable_eta_dualMain (ci : CitedInputs) (N s L σ : ℝ) (K : ℕ)
    (hN128 : 128 ≤ N) (hs0 : 0 ≤ s) (hL0 : 0 ≤ L) (hL : L ^ 2 ≤ N / 16)
    (hs : s ^ 2 = N - L ^ 2 * σ) (hσ0 : 0 ≤ σ) (hσ1 : σ ≤ 1) (hK : (s + L) ^ 2 ≤ K) :
    IntervalIntegrable (fun u => eta u * dualMain ((s + L * u) ^ 2)) volume (-1) 1 := by
  obtain ⟨h0, hlo, hhi⟩ := window_bounds N s L σ (by linarith) hs0 hL0 hL hs hσ0 hσ1
  have hxc : Continuous (fun u : ℝ => (s + L * u) ^ 2) :=
    (continuous_const.add (continuous_const.mul continuous_id)).pow 2
  have hlo' : ∀ u ∈ Set.Icc (-1 : ℝ) 1, N / 2 ≤ (s + L * u) ^ 2 := fun u hu => by
    have := (xu_bounds s L u h0 hL0 hu).1
    linarith
  have hhi' : ∀ u ∈ Set.Icc (-1 : ℝ) 1, (s + L * u) ^ 2 ≤ 2 * N := fun u hu => by
    have := (xu_bounds s L u h0 hL0 hu).2
    linarith
  have hK' : ∀ u ∈ Set.Icc (-1 : ℝ) 1, ⌊(s + L * u) ^ 2⌋₊ ≤ K := fun u hu =>
    Nat.floor_le_of_le (le_trans (xu_bounds s L u h0 hL0 hu).2 hK)
  have hpos : ∀ u ∈ Set.Icc (-1 : ℝ) 1, 0 < (s + L * u) ^ 2 := fun u hu => by
    have := hlo' u hu
    linarith
  have hRint := (intervalIntegrable_iff_integrableOn_Ioc_of_le (by norm_num : (-1 : ℝ) ≤ 1)).1
    (intervalIntegrable_eta_R (fun u : ℝ => (s + L * u) ^ 2) hxc K hK')
  have hPint := (intervalIntegrable_iff_integrableOn_Ioc_of_le (by norm_num : (-1 : ℝ) ≤ 1)).1
    (intervalIntegrable_eta_P (fun u : ℝ => (s + L * u) ^ 2) hxc hpos)
  have hcint : IntegrableOn
      (fun u : ℝ => eta u * (4.8e-4 * N ^ (-(1 / 4 : ℝ))
        + 2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * Real.sqrt (N / 2)))))
      (Set.Ioc (-1 : ℝ) 1) volume :=
    (intervalIntegrable_iff_integrableOn_Ioc_of_le (by norm_num : (-1 : ℝ) ≤ 1)).1
      ((eta_continuous.mul continuous_const).intervalIntegrable (-1) 1)
  have hmeas : AEStronglyMeasurable (fun u => eta u * dualMain ((s + L * u) ^ 2))
      (volume.restrict (Set.Ioc (-1 : ℝ) 1)) := by
    refine aestronglyMeasurable_of_tendsto_ae atTop
      (f := fun (n : ℕ) (u : ℝ) =>
        eta u * ∑ j ∈ Finset.range n, mainTerm ((s + L * u) ^ 2) (j + 1))
      (fun n => ?_) ?_
    · apply Measurable.aestronglyMeasurable
      exact eta_continuous.measurable.mul
        (Finset.measurable_sum _ (fun j _ => mainTerm_measurable s L (j + 1)))
    · refine (ae_restrict_iff' measurableSet_Ioc).2 (ae_of_all _ (fun u hu => ?_))
      have hu' : u ∈ Set.Icc (-1 : ℝ) 1 := ⟨hu.1.le, hu.2⟩
      obtain ⟨hsum, _⟩ :=
        voronoi_pointwise ci N ((s + L * u) ^ 2) hN128 (hlo' u hu') (hhi' u hu')
      exact (hsum.hasSum.tendsto_sum_nat).const_mul (eta u)
  have hg : IntegrableOn
      (fun u : ℝ => |eta u * voronoiP ((s + L * u) ^ 2) - eta u * R ((s + L * u) ^ 2)|
        + eta u * (4.8e-4 * N ^ (-(1 / 4 : ℝ))
          + 2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * Real.sqrt (N / 2)))))
      (Set.Ioc (-1 : ℝ) 1) volume :=
    (Integrable.abs (hPint.sub hRint)).add hcint
  have hbound : ∀ᵐ u ∂(volume.restrict (Set.Ioc (-1 : ℝ) 1)),
      ‖eta u * dualMain ((s + L * u) ^ 2)‖
        ≤ |eta u * voronoiP ((s + L * u) ^ 2) - eta u * R ((s + L * u) ^ 2)|
          + eta u * (4.8e-4 * N ^ (-(1 / 4 : ℝ))
            + 2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * Real.sqrt (N / 2)))) := by
    refine (ae_restrict_iff' measurableSet_Ioc).2 (ae_of_all _ (fun u hu => ?_))
    have hu' : u ∈ Set.Icc (-1 : ℝ) 1 := ⟨hu.1.le, hu.2⟩
    have hp := (voronoi_pointwise ci N ((s + L * u) ^ 2) hN128 (hlo' u hu') (hhi' u hu')).2
    have h1 : eta u * dualMain ((s + L * u) ^ 2)
        = (eta u * voronoiP ((s + L * u) ^ 2) - eta u * R ((s + L * u) ^ 2))
          + eta u * (R ((s + L * u) ^ 2)
            - (voronoiP ((s + L * u) ^ 2) - dualMain ((s + L * u) ^ 2))) := by ring
    rw [Real.norm_eq_abs, h1]
    refine (abs_add_le _ _).trans (add_le_add le_rfl ?_)
    rw [abs_mul, abs_of_nonneg (eta_nonneg u)]
    exact mul_le_mul_of_nonneg_left hp (eta_nonneg u)
  have hI : IntegrableOn (fun u => eta u * dualMain ((s + L * u) ^ 2))
      (Set.Ioc (-1 : ℝ) 1) volume := Integrable.mono' hg hmeas hbound
  exact (intervalIntegrable_iff_integrableOn_Ioc_of_le (by norm_num : (-1 : ℝ) ≤ 1)).2 hI

/-- `hoyingSum_eq_round_ideal` without the integrability hypothesis `hD`. -/
theorem hoyingSum_eq_round_ideal' (ci : CitedInputs) (n : ℕ)
    (H sigh ε Edual dualApprox Enon : ℝ)
    (hN : 2 ^ 65536 ≤ n + 1)
    (hH0 : 0 ≤ H) (hHN : H ≤ (n : ℝ) + 1)
    (hsig0 : 0 ≤ sigh) (hsig1 : sigh ≤ 1)
    (hsigε : |etaVar - sigh| ≤ ε)
    (hdual : |dualApprox - ∫ u in (-1 : ℝ)..1,
      eta u * dualMain ((parS (n + 1) H sigh + parL (n + 1) H * u) ^ 2)| ≤ Edual)
    (hE : Edual + H ^ 2 * ε * (1 + Real.log (2 * ((n : ℝ) + 1))) / 8 ≤ Enon)
    (hn : Enon ≤ (2 : ℝ) ^ 60 * (((n : ℝ) + 1) ^ 4)⁻¹) :
    (hoyingSum n : ℤ)
      = round (idealAhat (parS (n + 1) H sigh) (parL (n + 1) H) (parK (n + 1) H sigh)
          dualApprox) := by
  have hN' : (2 : ℝ) ^ 65536 ≤ (n : ℝ) + 1 := by exact_mod_cast hN
  have h7 : (128 : ℝ) ≤ 2 ^ 65536 := by
    calc (128 : ℝ) = 2 ^ 7 := by norm_num
      _ ≤ 2 ^ 65536 := pow_le_pow_right₀ (by norm_num) (by norm_num)
  have h128 : (128 : ℝ) ≤ (n : ℝ) + 1 := le_trans h7 hN'
  have hNpos : (0 : ℝ) < ((n + 1 : ℕ) : ℝ) := by positivity
  have hHN' : H ≤ ((n + 1 : ℕ) : ℝ) := by push_cast; exact hHN
  have hs := parS_sq (n + 1) H sigh hNpos hH0 hHN' hsig0 hsig1
  have hL := parL_sq_le (n + 1) H hNpos hH0 hHN'
  have hK := par_K (n + 1) H sigh
  push_cast at hs hL
  have hD := intervalIntegrable_eta_dualMain ci ((n : ℝ) + 1) (parS (n + 1) H sigh)
    (parL (n + 1) H) sigh (parK (n + 1) H sigh) h128 (Real.sqrt_nonneg _)
    (parL_nonneg (n + 1) H hH0) hL hs hsig0 hsig1 hK
  exact hoyingSum_eq_round_ideal ci n H sigh ε Edual dualApprox Enon hN hH0 hHN hsig0 hsig1
    hsigε hD hdual hE hn

end DirichletDivisor

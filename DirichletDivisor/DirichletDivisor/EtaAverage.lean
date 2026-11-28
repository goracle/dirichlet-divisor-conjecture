import DirichletDivisor.VoronoiSplit
import DirichletDivisor.Bump
import DirichletDivisor.Lipschitz

set_option linter.style.whitespace false

namespace DirichletDivisor

open Finset MeasureTheory

/-! ## η-average of the pointwise Voronoï bound

Turns `voronoi_pointwise` (pointwise in `x`) into a bound on the smoothed integral
`∫ η(u) R(x(u)) du` against `∫ η (P - dualMain)(x(u)) du`. The only integrability input not
derived here is that of `u ↦ η(u) dualMain(x(u))` (it needs continuity of `Y₂`, which is abstract
in `CitedInputs`); it is taken as hypothesis `hD`. -/

theorem continuousOn_voronoiP_comp {s : Set ℝ} {f : ℝ → ℝ} (hf : ContinuousOn f s)
    (hpos : ∀ u ∈ s, 0 < f u) : ContinuousOn (fun u => voronoiP (f u)) s := by
  unfold voronoiP
  have hlog : ContinuousOn (fun u => Real.log (f u)) s :=
    hf.log (fun u hu => (hpos u hu).ne')
  exact ((((hf.pow 2).div_const 2).mul
    ((hlog.add continuousOn_const).sub continuousOn_const)).add
      (hf.div_const 4)).sub continuousOn_const

theorem intervalIntegrable_eta_R (xf : ℝ → ℝ) (hxc : Continuous xf) (K : ℕ)
    (hK : ∀ u ∈ Set.Icc (-1 : ℝ) 1, ⌊xf u⌋₊ ≤ K) :
    IntervalIntegrable (fun u => eta u * R (xf u)) volume (-1) 1 := by
  have hc : ContinuousOn (fun u => eta u * R (xf u)) (Set.uIcc (-1 : ℝ) 1) := by
    rw [Set.uIcc_of_le (by norm_num)]
    have h0 : Continuous fun u => eta u * ∑ k ∈ Icc 1 K, (d k : ℝ) * max (xf u - k) 0 :=
      eta_continuous.mul (continuous_finsetSum _
        (fun k _ => continuous_const.mul ((hxc.sub continuous_const).max continuous_const)))
    refine h0.continuousOn.congr ?_
    intro u hu
    simp only [R_eq_sum_max K (xf u) (hK u hu)]
  exact hc.intervalIntegrable

theorem intervalIntegrable_eta_P (xf : ℝ → ℝ) (hxc : Continuous xf)
    (hpos : ∀ u ∈ Set.Icc (-1 : ℝ) 1, 0 < xf u) :
    IntervalIntegrable (fun u => eta u * voronoiP (xf u)) volume (-1) 1 := by
  have hc : ContinuousOn (fun u => eta u * voronoiP (xf u)) (Set.uIcc (-1 : ℝ) 1) := by
    rw [Set.uIcc_of_le (by norm_num)]
    exact eta_continuous.continuousOn.mul
      (continuousOn_voronoiP_comp hxc.continuousOn hpos)
  exact hc.intervalIntegrable

/-- **η-averaged Voronoï error.** -/
theorem eta_average_voronoi (ci : CitedInputs) (N : ℝ) (xf : ℝ → ℝ) (hxc : Continuous xf)
    (K : ℕ) (hK : ∀ u ∈ Set.Icc (-1 : ℝ) 1, ⌊xf u⌋₊ ≤ K) (hN : 128 ≤ N)
    (hlo : ∀ u ∈ Set.Icc (-1 : ℝ) 1, N / 2 ≤ xf u)
    (hhi : ∀ u ∈ Set.Icc (-1 : ℝ) 1, xf u ≤ 2 * N)
    (hD : IntervalIntegrable (fun u => eta u * dualMain (xf u)) volume (-1) 1) :
    |(∫ u in (-1 : ℝ)..1, eta u * R (xf u))
        - ((∫ u in (-1 : ℝ)..1, eta u * voronoiP (xf u))
            - ∫ u in (-1 : ℝ)..1, eta u * dualMain (xf u))|
      ≤ 4.8e-4 * N ^ (-(1 / 4 : ℝ))
          + 2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * Real.sqrt (N / 2))) := by
  have hpos : ∀ u ∈ Set.Icc (-1 : ℝ) 1, 0 < xf u := fun u hu => by
    have := hlo u hu
    linarith
  have hR := intervalIntegrable_eta_R xf hxc K hK
  have hP := intervalIntegrable_eta_P xf hxc hpos
  have hPD : IntervalIntegrable
      (fun u => eta u * voronoiP (xf u) - eta u * dualMain (xf u)) volume (-1) 1 :=
    hP.sub hD
  have e : (fun u => eta u * (R (xf u) - (voronoiP (xf u) - dualMain (xf u))))
      = fun u => eta u * R (xf u)
          - (eta u * voronoiP (xf u) - eta u * dualMain (xf u)) := by
    funext u
    ring
  have h1 : (∫ u in (-1 : ℝ)..1, eta u * (R (xf u) - (voronoiP (xf u) - dualMain (xf u))))
      = (∫ u in (-1 : ℝ)..1, eta u * R (xf u))
        - ((∫ u in (-1 : ℝ)..1, eta u * voronoiP (xf u))
            - ∫ u in (-1 : ℝ)..1, eta u * dualMain (xf u)) := by
    rw [e, intervalIntegral.integral_sub hR hPD, intervalIntegral.integral_sub hP hD]
  rw [← h1]
  have hE : (∫ u in (-1 : ℝ)..1,
      eta u * (4.8e-4 * N ^ (-(1 / 4 : ℝ))
        + 2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * Real.sqrt (N / 2)))))
      = 4.8e-4 * N ^ (-(1 / 4 : ℝ))
        + 2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * Real.sqrt (N / 2))) := by
    rw [intervalIntegral.integral_mul_const, eta_integral, one_mul]
  have hgint : IntervalIntegrable
      (fun u => eta u * (4.8e-4 * N ^ (-(1 / 4 : ℝ))
        + 2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * Real.sqrt (N / 2)))))
      volume (-1) 1 :=
    (eta_continuous.mul continuous_const).intervalIntegrable _ _
  have hle := intervalIntegral.norm_integral_le_of_norm_le (by norm_num : (-1 : ℝ) ≤ 1)
    (f := fun u => eta u * (R (xf u) - (voronoiP (xf u) - dualMain (xf u))))
    (g := fun u => eta u * (4.8e-4 * N ^ (-(1 / 4 : ℝ))
        + 2 * N ^ (3 / 4 : ℝ) * Real.exp (-(2 * Real.pi * Real.sqrt (N / 2)))))
    (μ := volume)
    (ae_of_all _ (fun t ht => by
      have ht' : t ∈ Set.Icc (-1 : ℝ) 1 := ⟨ht.1.le, ht.2⟩
      have hp := (voronoi_pointwise ci N (xf t) hN (hlo t ht') (hhi t ht')).2
      rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (eta_nonneg t)]
      exact mul_le_mul_of_nonneg_left hp (eta_nonneg t)))
    hgint
  rw [hE, Real.norm_eq_abs] at hle
  exact hle

end DirichletDivisor

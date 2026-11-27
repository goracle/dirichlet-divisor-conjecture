import DirichletDivisor.Lipschitz

open Finset MeasureTheory

namespace DirichletDivisor

/-- Smoothed positive part: V(k) = ∫_{-1}^{1} w(u) max(x(u) - k, 0) du. -/
noncomputable def smoothPos (w x : ℝ → ℝ) (k : ℝ) : ℝ :=
  ∫ u in (-1:ℝ)..1, w u * max (x u - k) 0

/-- Local correction c(k) = max(m - k, 0) - V(k). -/
noncomputable def corr (w x : ℝ → ℝ) (m : ℝ) (k : ℝ) : ℝ :=
  max (m - k) 0 - smoothPos w x k
  

/-- Finite sum commutes with the interval integral (continuous summands). -/
theorem integral_finset_sum_cont {ι : Type*} (f : ι → ℝ → ℝ) (s : Finset ι) :
    (∀ i ∈ s, Continuous (f i)) →
    ∫ u in (-1:ℝ)..1, ∑ i ∈ s, f i u = ∑ i ∈ s, ∫ u in (-1:ℝ)..1, f i u := by
  classical
  refine Finset.induction_on s ?_ ?_
  · intro _
    simp
  · intro a s ha ih h
    have hs : ∀ i ∈ s, Continuous (f i) := fun i hi => h i (Finset.mem_insert_of_mem hi)
    have ha' : Continuous (f a) := h a (Finset.mem_insert_self a s)
    have hcs : ∀ t : Finset ι, (∀ i ∈ t, Continuous (f i)) →
        Continuous fun u => ∑ i ∈ t, f i u := by
      intro t
      refine Finset.induction_on t ?_ ?_
      · intro _
        simp only [Finset.sum_empty]
        exact continuous_const
      · intro b t hb ihb hbt
        simp only [Finset.sum_insert hb]
        exact (hbt b (Finset.mem_insert_self b t)).add
          (ihb fun i hi => hbt i (Finset.mem_insert_of_mem hi))
    have hsum : Continuous fun u => ∑ i ∈ s, f i u := hcs s hs
    simp only [Finset.sum_insert ha]
    rw [intervalIntegral.integral_add (ha'.intervalIntegrable _ _)
        (hsum.intervalIntegrable _ _), ih hs]



/-- Interchange of the finite sum and the integral. -/
theorem smooth_R_eq (w x : ℝ → ℝ) (hw : Continuous w) (hx : Continuous x) (K : ℕ)
    (hK : ∀ u ∈ Set.Icc (-1:ℝ) 1, ⌊x u⌋₊ ≤ K) :
    ∫ u in (-1:ℝ)..1, w u * R (x u)
      = ∑ k ∈ Icc 1 K, (d k : ℝ) * smoothPos w x k := by
  have h1 : ∫ u in (-1:ℝ)..1, w u * R (x u)
      = ∫ u in (-1:ℝ)..1, ∑ k ∈ Icc 1 K, (d k : ℝ) * (w u * max (x u - k) 0) := by
    refine intervalIntegral.integral_congr (fun u hu => ?_)
    have hu' : u ∈ Set.Icc (-1:ℝ) 1 := by
      rwa [Set.uIcc_of_le (by norm_num)] at hu
    show w u * R (x u) = ∑ k ∈ Icc 1 K, (d k : ℝ) * (w u * max (x u - k) 0)
    rw [R_eq_sum_max K (x u) (hK u hu'), Finset.mul_sum]
    exact Finset.sum_congr rfl fun k _ => by ring
  have hcont : ∀ k ∈ Icc 1 K,
      Continuous fun u => (d k : ℝ) * (w u * max (x u - k) 0) := fun k _ =>
    continuous_const.mul (hw.mul ((hx.sub continuous_const).max continuous_const))
  rw [h1, integral_finset_sum_cont (fun k u => (d k : ℝ) * (w u * max (x u - k) 0))
    (Icc 1 K) hcont]
  refine Finset.sum_congr rfl fun k _ => ?_
  unfold smoothPos
  rw [intervalIntegral.integral_const_mul]


/-- c(k) = 0 for k below the range of x. -/
theorem corr_eq_zero_of_le (w x : ℝ → ℝ) (hw : Continuous w) (hx : Continuous x)
    (hw0 : ∀ u ∈ Set.Icc (-1:ℝ) 1, 0 ≤ w u) (hint : ∫ u in (-1:ℝ)..1, w u = 1)
    (a : ℝ) (ha : ∀ u ∈ Set.Icc (-1:ℝ) 1, a ≤ x u) (k : ℝ) (hk : k ≤ a) :
    corr w x (∫ u in (-1:ℝ)..1, w u * x u) k = 0 := by
  have h1 : IntervalIntegrable (fun u => w u * x u) volume (-1) 1 :=
    (hw.mul hx).intervalIntegrable _ _
  have h2 : IntervalIntegrable (fun u => k * w u) volume (-1) 1 :=
    (continuous_const.mul hw).intervalIntegrable _ _
  have hsp : smoothPos w x k = (∫ u in (-1:ℝ)..1, w u * x u) - k := by
    unfold smoothPos
    have : ∀ u ∈ Set.uIcc (-1:ℝ) 1,
        w u * max (x u - k) 0 = w u * x u - k * w u := by
      intro u hu
      rw [Set.uIcc_of_le (by norm_num)] at hu
      rw [max_eq_left (by linarith [ha u hu])]
      ring
    rw [intervalIntegral.integral_congr this, intervalIntegral.integral_sub h1 h2,
      intervalIntegral.integral_const_mul, hint]
    ring
  have hnn : 0 ≤ smoothPos w x k :=
    intervalIntegral.integral_nonneg (by norm_num)
      (fun u hu => mul_nonneg (hw0 u hu) (le_max_right _ _))
  rw [hsp] at hnn
  unfold corr
  rw [hsp, max_eq_left hnn]
  ring

/-- c(k) = 0 for k above the range of x. -/
theorem corr_eq_zero_of_ge (w x : ℝ → ℝ) (hw : Continuous w) (hx : Continuous x)
    (hw0 : ∀ u ∈ Set.Icc (-1:ℝ) 1, 0 ≤ w u) (hint : ∫ u in (-1:ℝ)..1, w u = 1)
    (b : ℝ) (hb : ∀ u ∈ Set.Icc (-1:ℝ) 1, x u ≤ b) (k : ℝ) (hk : b ≤ k) :
    corr w x (∫ u in (-1:ℝ)..1, w u * x u) k = 0 := by
  have h1 : IntervalIntegrable (fun u => k * w u) volume (-1) 1 :=
    (continuous_const.mul hw).intervalIntegrable _ _
  have h2 : IntervalIntegrable (fun u => w u * x u) volume (-1) 1 :=
    (hw.mul hx).intervalIntegrable _ _
  have hsp : smoothPos w x k = 0 := by
    unfold smoothPos
    have : ∀ u ∈ Set.uIcc (-1:ℝ) 1, w u * max (x u - k) 0 = 0 := by
      intro u hu
      rw [Set.uIcc_of_le (by norm_num)] at hu
      rw [max_eq_right (by linarith [hb u hu]), mul_zero]
    rw [intervalIntegral.integral_congr this]
    simp
  have hm : (∫ u in (-1:ℝ)..1, w u * x u) ≤ k := by
    have hnn : 0 ≤ ∫ u in (-1:ℝ)..1, w u * (k - x u) :=
      intervalIntegral.integral_nonneg (by norm_num)
        (fun u hu => mul_nonneg (hw0 u hu) (by linarith [hb u hu]))
    have heq : ∫ u in (-1:ℝ)..1, w u * (k - x u)
        = k - ∫ u in (-1:ℝ)..1, w u * x u := by
      have : ∀ u ∈ Set.uIcc (-1:ℝ) 1, w u * (k - x u) = k * w u - w u * x u := by
        intro u _
        ring
      rw [intervalIntegral.integral_congr this, intervalIntegral.integral_sub h1 h2,
        intervalIntegral.integral_const_mul, hint]
      ring
    linarith
  unfold corr
  rw [hsp, max_eq_right (by linarith)]
  ring

/-- The exact smoothing identity:
R(m) = ∫ w R(x) + ∑_{k ≤ K} d(k) c(k). -/
theorem R_smoothing_identity (w x : ℝ → ℝ) (hw : Continuous w) (hx : Continuous x)
    (K : ℕ) (hK : ∀ u ∈ Set.Icc (-1:ℝ) 1, ⌊x u⌋₊ ≤ K)
    (m : ℝ) (hm : ⌊m⌋₊ ≤ K) :
    R m = (∫ u in (-1:ℝ)..1, w u * R (x u))
          + ∑ k ∈ Icc 1 K, (d k : ℝ) * corr w x m k := by
  rw [smooth_R_eq w x hw hx K hK]
  simp only [corr, mul_sub, Finset.sum_sub_distrib]
  rw [← R_eq_sum_max K m hm]
  ring

end DirichletDivisor

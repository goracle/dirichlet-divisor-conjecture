import DirichletDivisor.Gevrey
import DirichletDivisor.Decay

set_option linter.style.whitespace false

open MeasureTheory

namespace DirichletDivisor

/-! ## A linear-in-`d` decay bound for `e^{-c/d} / d^r` -/

theorem exp_neg_div_pow_le_linear (c d : ℝ) (hc : 0 < c) (hd : 0 < d) (r : ℕ) :
    Real.exp (-(c / d)) / d ^ r ≤ (((r + 1).factorial : ℝ) / c ^ (r + 1)) * d := by
  have hcne : c ≠ 0 := hc.ne'
  have hdne : d ≠ 0 := hd.ne'
  have hF : (0 : ℝ) < ((r + 1).factorial : ℝ) := Nat.cast_pos.mpr (Nat.factorial_pos _)
  have h := Real.pow_div_factorial_le_exp (c / d) (by positivity) (r + 1)
  rw [div_le_iff₀ hF] at h
  have key : (c / d) ^ (r + 1) * Real.exp (-(c / d)) ≤ ((r + 1).factorial : ℝ) := by
    calc (c / d) ^ (r + 1) * Real.exp (-(c / d))
        ≤ (Real.exp (c / d) * ((r + 1).factorial : ℝ)) * Real.exp (-(c / d)) :=
          mul_le_mul_of_nonneg_right h (Real.exp_pos _).le
      _ = ((r + 1).factorial : ℝ) := by
          rw [mul_right_comm, ← Real.exp_add]
          simp
  rw [div_pow, div_mul_eq_mul_div, div_le_iff₀ (pow_pos hd _)] at key
  have e : (((r + 1).factorial : ℝ) / c ^ (r + 1) * d) * d ^ r
      = ((r + 1).factorial : ℝ) * d ^ (r + 1) / c ^ (r + 1) := by
    rw [pow_succ]
    field_simp
    ring
  rw [div_le_iff₀ (pow_pos hd r), e, le_div_iff₀ (pow_pos hc _)]
  nlinarith [key]

/-! ## Elementary facts about the bump -/

theorem bump_eq_exp (u : ℝ) (h : u ^ 2 < 1) : bump u = Real.exp (-(1 / (1 - u ^ 2))) := by
  have h1 : 0 < 1 - u ^ 2 := by linarith
  unfold bump expNegInvGlue
  rw [if_neg (not_le.mpr h1), one_div]

theorem eta_eq_zero (t : ℝ) (h : 1 ≤ |t|) : eta t = 0 := by
  have h1 := sq_abs t
  have h2 : 1 - t ^ 2 ≤ 0 := by nlinarith [sq_nonneg (|t| - 1)]
  unfold eta bump expNegInvGlue
  rw [if_pos h2]
  simp

theorem exp_four_thirds_le : Real.exp (4 / 3) ≤ 4 := by
  have h1 : Real.exp (4 / 3) ^ 3 = Real.exp 1 ^ 4 := by
    rw [exp_pow', exp_pow']
    norm_num
  have he1 : Real.exp 1 < 2.72 := by
    have := Real.exp_one_lt_d9
    linarith
  have h2 : Real.exp 1 ^ 4 ≤ (2.72 : ℝ) ^ 4 :=
    pow_le_pow_left₀ (Real.exp_pos 1).le he1.le 4
  by_contra h
  have h4 : (4 : ℝ) < Real.exp (4 / 3) := not_le.mp h
  have h3 : (4 : ℝ) ^ 3 < Real.exp (4 / 3) ^ 3 :=
    pow_lt_pow_left₀ h4 (by norm_num) (by norm_num)
  have h5 : (2.72 : ℝ) ^ 4 < 4 ^ 3 := by norm_num
  linarith

/-- `bumpMass ≥ 1/4` (restrict to `|u| ≤ 1/2`, where `bump ≥ e^{-4/3}`). -/
theorem bumpMass_ge : (1 / 4 : ℝ) ≤ bumpMass := by
  have h43 := exp_four_thirds_le
  have hlow : ∀ u ∈ Set.Icc (-(1 / 2) : ℝ) (1 / 2), Real.exp (-(4 / 3)) ≤ bump u := by
    intro u hu
    have hq : 0 ≤ (1 / 2 - u) * (1 / 2 + u) :=
      mul_nonneg (by linarith [hu.2]) (by linarith [hu.1])
    have h3 : 3 / 4 ≤ 1 - u ^ 2 := by nlinarith [hq]
    have hu2 : u ^ 2 < 1 := by linarith
    rw [bump_eq_exp u hu2]
    apply Real.exp_le_exp.2
    have h5 : 1 / (1 - u ^ 2) ≤ 4 / 3 := by
      rw [div_le_iff₀ (by linarith)]
      nlinarith [h3]
    linarith
  calc (1 / 4 : ℝ) ≤ Real.exp (-(4 / 3)) := by
        rw [Real.exp_neg, ← one_div]
        exact one_div_le_one_div_of_le (Real.exp_pos _) h43
    _ = ∫ _u in (-(1 / 2) : ℝ)..(1 / 2), Real.exp (-(4 / 3)) := by
        rw [intervalIntegral.integral_const, smul_eq_mul]
        ring
    _ ≤ ∫ u in (-(1 / 2) : ℝ)..(1 / 2), bump u :=
        intervalIntegral.integral_mono_on (by norm_num)
          (continuous_const.intervalIntegrable _ _) (bump_continuous.intervalIntegrable _ _) hlow
    _ ≤ bumpMass := by
        unfold bumpMass
        exact intervalIntegral.integral_mono_interval (by norm_num) (by norm_num) (by norm_num)
          (Filter.Eventually.of_forall bump_nonneg) (bump_continuous.intervalIntegrable _ _)

theorem gevC_le_eight : gevC ≤ 8 := by
  unfold gevC
  have h1 : bumpMass⁻¹ ≤ 4 := by
    rw [← one_div, div_le_iff₀ bumpMass_pos]
    linarith [bumpMass_ge]
  have h2 := three_halves_rpow_lt_two.le
  have h3 : (0 : ℝ) ≤ (3 / 2 : ℝ) ^ (3 / 2 : ℝ) := Real.rpow_nonneg (by norm_num) _
  nlinarith [mul_le_mul h1 h2 h3 (by norm_num : (0 : ℝ) ≤ 4)]

/-! ## The truncated derivatives `hk` and the phase `Ek` -/

/-- `k`-th derivative of `fc` on `(-1,1)`, extended by `0` elsewhere. -/
noncomputable def hk (s δ : ℝ) (k : ℕ) (t : ℝ) : ℂ :=
  if |t| < 1 then iteratedDeriv k (fc s δ) t else 0

/-- The constant in the linear decay of `hk` near the endpoints. -/
noncomputable def hkM (k : ℕ) : ℝ :=
  (k.factorial : ℝ) * gevC * (((k + 1).factorial : ℝ) / (9 / 38 : ℝ) ^ (k + 1)) * (9 / 10)

/-- The phase `e(a t) = exp(2πi · (a t) / (2π))`, i.e. `exp(i a t)`. -/
noncomputable def Ek (a t : ℝ) : ℂ := Complex.exp (((a * t : ℝ) : ℂ) * Complex.I)

theorem hk_of_lt (s δ : ℝ) (k : ℕ) (t : ℝ) (h : |t| < 1) :
    hk s δ k t = iteratedDeriv k (fc s δ) t := by
  unfold hk
  rw [if_pos h]

theorem hk_of_not_lt (s δ : ℝ) (k : ℕ) (t : ℝ) (h : ¬ |t| < 1) : hk s δ k t = 0 := by
  unfold hk
  rw [if_neg h]

theorem hk_one (s δ : ℝ) (k : ℕ) : hk s δ k 1 = 0 :=
  hk_of_not_lt s δ k 1 (by simp)

theorem hk_neg_one (s δ : ℝ) (k : ℕ) : hk s δ k (-1) = 0 :=
  hk_of_not_lt s δ k (-1) (by simp)

theorem hk_eventuallyEq (s δ : ℝ) (k : ℕ) (x : ℝ) (hx : |x| < 1) :
    hk s δ k =ᶠ[nhds x] iteratedDeriv k (fc s δ) := by
  have hnhds : {y : ℝ | |y| < 1} ∈ nhds x :=
    (isOpen_lt continuous_abs continuous_const).mem_nhds hx
  filter_upwards [hnhds] with y hy using hk_of_lt s δ k y hy

theorem fc_hasDerivAt (s δ : ℝ) (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2) (k : ℕ) (x : ℝ)
    (hx : |x| < 1) :
    HasDerivAt (iteratedDeriv k (fc s δ)) (iteratedDeriv (k + 1) (fc s δ) x) x := by
  have hnhds : {y : ℝ | |y| < 1} ∈ nhds x :=
    (isOpen_lt continuous_abs continuous_const).mem_nhds hx
  have hev : iteratedDeriv k (fc s δ) =ᶠ[nhds x]
      fun y : ℝ => iteratedDeriv k (Gc s δ) (y : ℂ) := by
    filter_upwards [hnhds] with y hy using fc_iteratedDeriv s δ hδ0 hδ k y hy
  have hmem : (x : ℂ) ∈ Metric.ball (0 : ℂ) 1 := by
    rw [mem_ball_zero_iff, Complex.norm_real, Real.norm_eq_abs]
    exact hx
  have hda : DifferentiableAt ℂ (iteratedDeriv k (Gc s δ)) (x : ℂ) :=
    (Gc_iteratedDeriv_diffOn s δ hδ0 hδ k).differentiableAt (Metric.isOpen_ball.mem_nhds hmem)
  have hd : HasDerivAt (fun y : ℝ => iteratedDeriv k (Gc s δ) (y : ℂ))
      (deriv (iteratedDeriv k (Gc s δ)) (x : ℂ)) x := hda.hasDerivAt.comp_ofReal
  have hval : iteratedDeriv (k + 1) (fc s δ) x = deriv (iteratedDeriv k (Gc s δ)) (x : ℂ) := by
    rw [fc_iteratedDeriv s δ hδ0 hδ (k + 1) x hx, iteratedDeriv_succ]
  rw [hval]
  exact hd.congr_of_eventuallyEq hev

theorem hk_hasDerivAt (s δ : ℝ) (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2) (k : ℕ) (x : ℝ)
    (hx : |x| < 1) : HasDerivAt (hk s δ k) (hk s δ (k + 1) x) x := by
  rw [hk_of_lt s δ (k + 1) x hx]
  exact (fc_hasDerivAt s δ hδ0 hδ k x hx).congr_of_eventuallyEq (hk_eventuallyEq s δ k x hx)

/-- Cauchy estimate on the disk of radius `(9/10)(1-|t|)` around `t` (Section 3 of the note).
This is the intermediate step of `iteratedDeriv_fc_bound`, valid for every order `k`. -/
theorem iteratedDeriv_fc_cauchy (s δ : ℝ) (hs0 : 0 ≤ s) (hs : s ≤ 3 / 2)
    (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2) (k : ℕ) (t : ℝ) (ht : |t| < 1) :
    ‖iteratedDeriv k (fc s δ) t‖ ≤ (k.factorial : ℝ) * gevC *
      (Real.exp (-((9 / 38 : ℝ) / (9 / 10 * (1 - |t|)))) / (9 / 10 * (1 - |t|)) ^ k) := by
  obtain ⟨d, hd⟩ : ∃ d : ℝ, d = 1 - |t| := ⟨_, rfl⟩
  have hdpos : 0 < d := by rw [hd]; linarith
  have hR : 0 < 9 / 10 * d := by linarith
  have hsub : Metric.closedBall (t : ℂ) (9 / 10 * d) ⊆ Metric.ball (0 : ℂ) 1 := by
    intro w hw
    rw [Metric.mem_closedBall, dist_eq_norm] at hw
    rw [mem_ball_zero_iff]
    exact norm_lt_one_of_near t d hd ht w hw
  have hdc : DiffContOnCl ℂ (Gc s δ) (Metric.ball (t : ℂ) (9 / 10 * d)) := by
    apply DifferentiableOn.diffContOnCl
    rw [closure_ball _ hR.ne']
    exact (Gc_diffOn s δ hδ0 hδ).mono hsub
  have hC : ∀ z ∈ Metric.sphere (t : ℂ) (9 / 10 * d),
      ‖Gc s δ z‖ ≤ gevC * Real.exp (-(5 / 19 / d)) := fun z hz =>
    Gc_sphere_bound s δ hs0 hs hδ0 hδ t d hd ht z (mem_sphere_iff_norm.mp hz)
  have hcau := Complex.norm_iteratedDeriv_le_of_forall_mem_sphere_norm_le k hR hdc hC
  rw [fc_iteratedDeriv s δ hδ0 hδ k t ht]
  have e7 : (9 / 38 : ℝ) / (9 / 10 * d) = 5 / 19 / d := by
    have : d ≠ 0 := hdpos.ne'
    field_simp <;> ring
  refine hcau.trans (le_of_eq ?_)
  rw [← hd, e7]
  ring

theorem hk_bound (s δ : ℝ) (hs0 : 0 ≤ s) (hs : s ≤ 3 / 2) (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2)
    (k : ℕ) (t : ℝ) (ht : |t| ≤ 1) : ‖hk s δ k t‖ ≤ hkM k * (1 - |t|) := by
  by_cases hlt : |t| < 1
  · rw [hk_of_lt s δ k t hlt]
    have hdpos : 0 < 1 - |t| := by linarith
    have hR : 0 < 9 / 10 * (1 - |t|) := by linarith
    have hc := iteratedDeriv_fc_cauchy s δ hs0 hs hδ0 hδ k t hlt
    have hlin := exp_neg_div_pow_le_linear (9 / 38) (9 / 10 * (1 - |t|)) (by norm_num) hR k
    have hnn : 0 ≤ (k.factorial : ℝ) * gevC := mul_nonneg (Nat.cast_nonneg _) gevC_nonneg
    calc ‖iteratedDeriv k (fc s δ) t‖
        ≤ (k.factorial : ℝ) * gevC
          * (Real.exp (-((9 / 38 : ℝ) / (9 / 10 * (1 - |t|)))) / (9 / 10 * (1 - |t|)) ^ k) := hc
      _ ≤ (k.factorial : ℝ) * gevC
          * ((((k + 1).factorial : ℝ) / (9 / 38 : ℝ) ^ (k + 1)) * (9 / 10 * (1 - |t|))) :=
          mul_le_mul_of_nonneg_left hlin hnn
      _ = hkM k * (1 - |t|) := by unfold hkM; ring
  · have h1 : |t| = 1 := le_antisymm ht (not_lt.1 hlt)
    rw [hk_of_not_lt s δ k t hlt, norm_zero, h1]
    simp

theorem hk_const_bound (s δ : ℝ) (hs0 : 0 ≤ s) (hs : s ≤ 3 / 2) (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2)
    (r : ℕ) (hr : 1 ≤ r) (t : ℝ) :
    ‖hk s δ r t‖ ≤ gevC * (38 / 9 : ℝ) ^ r * (r.factorial : ℝ) ^ 2 := by
  by_cases hlt : |t| < 1
  · rw [hk_of_lt s δ r t hlt]
    exact iteratedDeriv_fc_bound s δ hs0 hs hδ0 hδ r hr t hlt
  · rw [hk_of_not_lt s δ r t hlt, norm_zero]
    exact mul_nonneg (mul_nonneg gevC_nonneg (pow_nonneg (by norm_num) _)) (sq_nonneg _)

theorem hk_continuousOn (s δ : ℝ) (hs0 : 0 ≤ s) (hs : s ≤ 3 / 2) (hδ0 : 0 ≤ δ)
    (hδ : δ ≤ 1 / 2) (k : ℕ) : ContinuousOn (hk s δ k) (Set.Icc (-1 : ℝ) 1) := by
  intro t ht
  by_cases hlt : |t| < 1
  · have h1 : ContinuousAt (iteratedDeriv k (fc s δ)) t :=
      (fc_hasDerivAt s δ hδ0 hδ k t hlt).continuousAt
    exact (h1.congr (hk_eventuallyEq s δ k t hlt).symm).continuousWithinAt
  · have habs : |t| = 1 := le_antisymm (abs_le.2 ⟨ht.1, ht.2⟩) (not_lt.1 hlt)
    have hk0 : hk s δ k t = 0 := hk_of_not_lt s δ k t hlt
    have hc : Continuous fun y : ℝ => hkM k * (1 - |y|) :=
      continuous_const.mul (continuous_const.sub continuous_abs)
    have h0 : hkM k * (1 - |t|) = 0 := by rw [habs]; ring
    have htend : Filter.Tendsto (fun y : ℝ => hkM k * (1 - |y|))
        (nhdsWithin t (Set.Icc (-1 : ℝ) 1)) (nhds 0) := by
      have := (hc.tendsto t).mono_left
        (nhdsWithin_le_nhds (s := Set.Icc (-1 : ℝ) 1))
      rwa [h0] at this
    have hev : ∀ᶠ y in nhdsWithin t (Set.Icc (-1 : ℝ) 1), ‖hk s δ k y‖ ≤ hkM k * (1 - |y|) :=
      eventually_nhdsWithin_of_forall (fun y hy =>
        hk_bound s δ hs0 hs hδ0 hδ k y (abs_le.2 ⟨hy.1, hy.2⟩))
    have hsq := squeeze_zero_norm' hev htend
    show Filter.Tendsto (hk s δ k) (nhdsWithin t (Set.Icc (-1 : ℝ) 1)) (nhds (hk s δ k t))
    rw [hk0]
    exact hsq

theorem Ek_hasDerivAt (a x : ℝ) : HasDerivAt (Ek a) (Ek a x * ((a : ℂ) * Complex.I)) x := by
  have h0 : HasDerivAt (fun y : ℝ => a * y) a x := by
    simpa using (hasDerivAt_id x).const_mul a
  have h1 : HasDerivAt (fun y : ℝ => ((a * y : ℝ) : ℂ)) (a : ℂ) x := h0.ofReal_comp
  have h2 : HasDerivAt (fun y : ℝ => ((a * y : ℝ) : ℂ) * Complex.I) ((a : ℂ) * Complex.I) x :=
    h1.mul_const Complex.I
  exact h2.cexp

theorem continuous_Ek (a : ℝ) : Continuous (Ek a) := by
  unfold Ek
  fun_prop

theorem norm_Ek (a t : ℝ) : ‖Ek a t‖ = 1 := by
  unfold Ek
  exact Complex.norm_exp_ofReal_mul_I _

/-! ## Integration by parts -/

theorem hk_ibp (s δ : ℝ) (hs0 : 0 ≤ s) (hs : s ≤ 3 / 2) (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2)
    (a : ℝ) (k : ℕ) :
    (∫ t in (-1 : ℝ)..1, hk s δ (k + 1) t * Ek a t)
      = -(((a : ℂ) * Complex.I) * ∫ t in (-1 : ℝ)..1, hk s δ k t * Ek a t) := by
  have hck := hk_continuousOn s δ hs0 hs hδ0 hδ k
  have hck1 := hk_continuousOn s δ hs0 hs hδ0 hδ (k + 1)
  have hE : ContinuousOn (Ek a) (Set.Icc (-1 : ℝ) 1) := (continuous_Ek a).continuousOn
  have i1 : IntervalIntegrable (fun t => hk s δ (k + 1) t * Ek a t) volume (-1) 1 :=
    (hck1.mul hE).intervalIntegrable_of_Icc (by norm_num)
  have i2 : IntervalIntegrable
      (fun t => hk s δ k t * (Ek a t * ((a : ℂ) * Complex.I))) volume (-1) 1 :=
    (hck.mul (hE.mul continuousOn_const)).intervalIntegrable_of_Icc (by norm_num)
  have hderiv : ∀ x ∈ Set.Ioo (-1 : ℝ) 1,
      HasDerivAt (fun t => hk s δ k t * Ek a t)
        (hk s δ (k + 1) x * Ek a x + hk s δ k x * (Ek a x * ((a : ℂ) * Complex.I))) x :=
    fun x hx => (hk_hasDerivAt s δ hδ0 hδ k x (abs_lt.2 ⟨hx.1, hx.2⟩)).mul (Ek_hasDerivAt a x)
  have hftc := intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le (by norm_num : (-1 : ℝ) ≤ 1)
    (hck.mul hE) hderiv (i1.add i2)
  have h0 : (hk s δ k 1 * Ek a 1 - hk s δ k (-1) * Ek a (-1)) = 0 := by
    rw [hk_one, hk_neg_one]
    ring
  have hsplit : (∫ t in (-1 : ℝ)..1,
        (hk s δ (k + 1) t * Ek a t + hk s δ k t * (Ek a t * ((a : ℂ) * Complex.I))))
      = (∫ t in (-1 : ℝ)..1, hk s δ (k + 1) t * Ek a t)
        + ((a : ℂ) * Complex.I) * ∫ t in (-1 : ℝ)..1, hk s δ k t * Ek a t := by
    rw [intervalIntegral.integral_add i1 i2, ← intervalIntegral.integral_const_mul]
    congr 1
    apply intervalIntegral.integral_congr
    intro t _
    simp only
    ring
  have hzero : (∫ t in (-1 : ℝ)..1, hk s δ (k + 1) t * Ek a t)
      + ((a : ℂ) * Complex.I) * ∫ t in (-1 : ℝ)..1, hk s δ k t * Ek a t = 0 :=
    hsplit.symm.trans (hftc.trans h0)
  exact eq_neg_of_add_eq_zero_left hzero

theorem hk_ibp_norm (s δ : ℝ) (hs0 : 0 ≤ s) (hs : s ≤ 3 / 2) (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2)
    (a : ℝ) (r : ℕ) :
    ‖∫ t in (-1 : ℝ)..1, hk s δ r t * Ek a t‖
      = |a| ^ r * ‖∫ t in (-1 : ℝ)..1, hk s δ 0 t * Ek a t‖ := by
  induction r with
  | zero => simp
  | succ n ih =>
    rw [hk_ibp s δ hs0 hs hδ0 hδ a n, norm_neg, norm_mul, ih]
    have : ‖(a : ℂ) * Complex.I‖ = |a| := by
      rw [norm_mul, Complex.norm_real, Complex.norm_I, mul_one, Real.norm_eq_abs]
    rw [this, pow_succ]
    ring

theorem Fs_eq_hk0 (s δ ξ : ℝ) :
    Fs s δ ξ = ∫ t in (-1 : ℝ)..1, hk s δ 0 t * Ek (2 * Real.pi * ξ) t := by
  unfold Fs
  apply intervalIntegral.integral_congr
  intro t ht
  rw [Set.uIcc_of_le (by norm_num)] at ht
  dsimp only
  by_cases hlt : |t| < 1
  · rw [hk_of_lt s δ 0 t hlt, iteratedDeriv_zero]
    rfl
  · have he : eta t = 0 := eta_eq_zero t (not_lt.1 hlt)
    have hk0 : hk s δ 0 t = 0 := hk_of_not_lt s δ 0 t hlt
    rw [hk0, he]
    simp

/-- Integration by parts `r` times: `‖Fs(ξ)‖ ≤ 2 K / (2π|ξ|)^r`, `K = C* B^r (r!)²`. -/
theorem Fs_norm_le_poly (s δ ξ : ℝ) (hs0 : 0 ≤ s) (hs : s ≤ 3 / 2) (hδ0 : 0 ≤ δ)
    (hδ : δ ≤ 1 / 2) (hξ : ξ ≠ 0) (r : ℕ) (hr : 1 ≤ r) :
    ‖Fs s δ ξ‖ ≤ 2 * (gevC * (38 / 9 : ℝ) ^ r * (r.factorial : ℝ) ^ 2)
      / (2 * Real.pi * |ξ|) ^ r := by
  have ha : (2 * Real.pi * ξ) ≠ 0 :=
    mul_ne_zero (mul_ne_zero two_ne_zero Real.pi_ne_zero) hξ
  have habs : |2 * Real.pi * ξ| = 2 * Real.pi * |ξ| := by
    rw [abs_mul, abs_mul, abs_of_pos Real.pi_pos]
    norm_num
  have hchain := hk_ibp_norm s δ hs0 hs hδ0 hδ (2 * Real.pi * ξ) r
  rw [← Fs_eq_hk0 s δ ξ] at hchain
  have hck := hk_continuousOn s δ hs0 hs hδ0 hδ r
  have hE : ContinuousOn (Ek (2 * Real.pi * ξ)) (Set.Icc (-1 : ℝ) 1) :=
    (continuous_Ek _).continuousOn
  have hI : ‖∫ t in (-1 : ℝ)..1, hk s δ r t * Ek (2 * Real.pi * ξ) t‖
      ≤ 2 * (gevC * (38 / 9 : ℝ) ^ r * (r.factorial : ℝ) ^ 2) := by
    calc ‖∫ t in (-1 : ℝ)..1, hk s δ r t * Ek (2 * Real.pi * ξ) t‖
        ≤ ∫ t in (-1 : ℝ)..1, ‖hk s δ r t * Ek (2 * Real.pi * ξ) t‖ :=
          intervalIntegral.norm_integral_le_integral_norm (by norm_num)
      _ ≤ ∫ _t in (-1 : ℝ)..1, gevC * (38 / 9 : ℝ) ^ r * (r.factorial : ℝ) ^ 2 := by
          apply intervalIntegral.integral_mono_on (by norm_num)
          · exact ((hck.mul hE).norm).intervalIntegrable_of_Icc (by norm_num)
          · exact continuous_const.intervalIntegrable _ _
          · intro t _
            rw [norm_mul, norm_Ek, mul_one]
            exact hk_const_bound s δ hs0 hs hδ0 hδ r hr t
      _ = 2 * (gevC * (38 / 9 : ℝ) ^ r * (r.factorial : ℝ) ^ 2) := by
          rw [intervalIntegral.integral_const, smul_eq_mul]
          norm_num
  have hpos : 0 < |2 * Real.pi * ξ| ^ r := pow_pos (abs_pos.2 ha) r
  rw [← habs, le_div_iff₀ hpos]
  rw [hchain] at hI
  linarith [mul_comm (|2 * Real.pi * ξ| ^ r) ‖Fs s δ ξ‖]

/-- The Fourier-decay estimate of the note: `‖Fs(ξ)‖ ≤ 2^16 e^{-2√|ξ|}`. -/
theorem Fs_norm_le_exp (s δ ξ : ℝ) (hs0 : 0 ≤ s) (hs : s ≤ 3 / 2) (hδ0 : 0 ≤ δ)
    (hδ : δ ≤ 1 / 2) :
    ‖Fs s δ ξ‖ ≤ 2 ^ 16 * Real.exp (-(2 * Real.sqrt |ξ|)) := by
  have htriv : ‖Fs s δ ξ‖ ≤ 2 := (Fs_norm_lt_two s δ ξ hs0 hs hδ0 hδ).le
  rcases eq_or_ne ξ 0 with hξ0 | hξ
  · subst hξ0
    have h0 : (2 : ℝ) ^ 16 * Real.exp (-(2 * Real.sqrt |(0 : ℝ)|)) = 2 ^ 16 := by simp
    rw [h0]
    norm_num
    linarith
  · have hξpos : 0 < |ξ| := abs_pos.2 hξ
    have hP : 0 < 2 * Real.pi * |ξ| := by
      have := Real.pi_pos
      positivity
    have hX0 : 0 ≤ 2 * Real.pi * |ξ| / (38 / 9) := by positivity
    have hX : 0 < Real.sqrt (2 * Real.pi * |ξ| / (38 / 9)) :=
      Real.sqrt_pos.2 (by positivity)
    have hy : 2 * Real.sqrt |ξ| ≤ 1.64 * Real.sqrt (2 * Real.pi * |ξ| / (38 / 9)) :=
      sqrt_le_scale |ξ| (abs_nonneg ξ)
    have hr : ∀ r : ℕ, ‖Fs s δ ξ‖ ≤
        16 * ((r.factorial : ℝ)) ^ 2 / (Real.sqrt (2 * Real.pi * |ξ| / (38 / 9))) ^ (2 * r) := by
      intro r
      rcases Nat.eq_zero_or_pos r with h0 | hpos
      · subst h0
        norm_num
        linarith
      · have hb := Fs_norm_le_poly s δ ξ hs0 hs hδ0 hδ hξ r hpos
        have hB : (0 : ℝ) < (38 / 9 : ℝ) ^ r := by positivity
        have hF : (0 : ℝ) ≤ ((r.factorial : ℝ)) ^ 2 := sq_nonneg _
        have hPr : (0 : ℝ) < (2 * Real.pi * |ξ|) ^ r := pow_pos hP r
        have e : (Real.sqrt (2 * Real.pi * |ξ| / (38 / 9))) ^ (2 * r)
            = (2 * Real.pi * |ξ|) ^ r / (38 / 9 : ℝ) ^ r := by
          rw [pow_mul, Real.sq_sqrt hX0, div_pow]
        have hg := gevC_le_eight
        calc ‖Fs s δ ξ‖
            ≤ 2 * (gevC * (38 / 9 : ℝ) ^ r * (r.factorial : ℝ) ^ 2)
                / (2 * Real.pi * |ξ|) ^ r := hb
          _ ≤ 16 * ((r.factorial : ℝ)) ^ 2 * (38 / 9 : ℝ) ^ r / (2 * Real.pi * |ξ|) ^ r := by
              apply (div_le_div_iff_of_pos_right hPr).2
              nlinarith [mul_nonneg (mul_nonneg hB.le hF) (sub_nonneg.2 hg)]
          _ = 16 * ((r.factorial : ℝ)) ^ 2
                / (Real.sqrt (2 * Real.pi * |ξ| / (38 / 9))) ^ (2 * r) := by
              rw [e, div_div_eq_mul_div]
    exact decay_optimise ‖Fs s δ ξ‖ (Real.sqrt (2 * Real.pi * |ξ| / (38 / 9)))
      (2 * Real.sqrt |ξ|) hX (by positivity) hy (norm_nonneg _) htriv hr

end DirichletDivisor

import DirichletDivisor.Fourier

set_option linter.style.whitespace false

open MeasureTheory

namespace DirichletDivisor

/-! ## Elementary helpers: exp powers, `u^r e^{-u}`, Stirling-type lower bound -/

theorem exp_pow' (y : ℝ) (r : ℕ) : Real.exp y ^ r = Real.exp ((r : ℝ) * y) := by
  induction r with
  | zero => simp
  | succ n ih =>
    rw [pow_succ, ih, ← Real.exp_add]
    congr 1
    push_cast
    ring

/-- `u^r e^{-u} ≤ (r/e)^r` for `u ≥ 0`, `r ≥ 1`. -/
theorem pow_mul_exp_neg_le (u : ℝ) (hu : 0 ≤ u) (r : ℕ) (hr : 1 ≤ r) :
    u ^ r * Real.exp (-u) ≤ ((r : ℝ) / Real.exp 1) ^ r := by
  have hr0 : (0 : ℝ) < r := Nat.cast_pos.mpr (by omega)
  have hrne : (r : ℝ) ≠ 0 := hr0.ne'
  have h1 : u / r ≤ Real.exp (u / r - 1) := by
    have := Real.add_one_le_exp (u / r - 1)
    linarith
  have h2 : (u / r) ^ r ≤ Real.exp (u / r - 1) ^ r :=
    pow_le_pow_left₀ (div_nonneg hu hr0.le) h1 r
  have h3 : Real.exp (u / r - 1) ^ r = Real.exp (u - r) := by
    rw [exp_pow']
    congr 1
    field_simp <;> ring
  have e4 : u ^ r = (r : ℝ) ^ r * (u / r) ^ r := by
    rw [div_pow]
    field_simp
  have h5 : Real.exp (u - r) * Real.exp (-u) = Real.exp (-(r : ℝ)) := by
    rw [← Real.exp_add]
    congr 1
    ring
  calc u ^ r * Real.exp (-u) = (r : ℝ) ^ r * ((u / r) ^ r * Real.exp (-u)) := by
        rw [e4]; ring
    _ ≤ (r : ℝ) ^ r * (Real.exp (u - r) * Real.exp (-u)) := by
        apply mul_le_mul_of_nonneg_left _ (pow_nonneg hr0.le r)
        exact mul_le_mul_of_nonneg_right (h2.trans h3.le) (Real.exp_pos _).le
    _ = ((r : ℝ) / Real.exp 1) ^ r := by
        rw [h5, div_pow, exp_pow', mul_one, Real.exp_neg]
        ring

/-- `d^{-r} e^{-c/d} ≤ (r/(e c))^r`. -/
theorem exp_neg_div_pow_le (c d : ℝ) (hc : 0 < c) (hd : 0 < d) (r : ℕ) (hr : 1 ≤ r) :
    Real.exp (-(c / d)) / d ^ r ≤ ((r : ℝ) / (Real.exp 1 * c)) ^ r := by
  have key := pow_mul_exp_neg_le (c / d) (by positivity) r hr
  have hcne : c ≠ 0 := hc.ne'
  have hdne : d ≠ 0 := hd.ne'
  have e5 : Real.exp (-(c / d)) / d ^ r = (c / d) ^ r * Real.exp (-(c / d)) / c ^ r := by
    rw [div_pow]
    field_simp
  have e6 : ((r : ℝ) / (Real.exp 1 * c)) ^ r = ((r : ℝ) / Real.exp 1) ^ r / c ^ r := by
    rw [← div_div, div_pow]
  rw [e5, e6]
  gcongr

/-- `(r/e)^r ≤ r!`. -/
theorem div_exp_pow_le_factorial (r : ℕ) : ((r : ℝ) / Real.exp 1) ^ r ≤ (r.factorial : ℝ) := by
  have h := Real.pow_div_factorial_le_exp (r : ℝ) (Nat.cast_nonneg r) r
  have hf : (0 : ℝ) < (r.factorial : ℝ) := Nat.cast_pos.mpr (Nat.factorial_pos r)
  rw [div_le_iff₀ hf] at h
  calc ((r : ℝ) / Real.exp 1) ^ r = (r : ℝ) ^ r / Real.exp r := by
        rw [div_pow, exp_pow', mul_one]
    _ ≤ (r.factorial : ℝ) := by
        rw [div_le_iff₀ (Real.exp_pos _), mul_comm]
        exact h

/-! ## The key real-part estimate for `1/(1-z²)` on a disc -/

theorem re_one_div_one_sub (z : ℂ) :
    (1 / (1 - z)).re = (1 - z.re) / ((1 - z.re) ^ 2 + z.im ^ 2) := by
  rw [one_div, Complex.inv_re, Complex.normSq_apply]
  simp only [Complex.sub_re, Complex.sub_im, Complex.one_re, Complex.one_im]
  congr 1 <;> ring

theorem re_one_div_one_add (z : ℂ) :
    (1 / (1 + z)).re = (1 + z.re) / ((1 + z.re) ^ 2 + z.im ^ 2) := by
  rw [one_div, Complex.inv_re, Complex.normSq_apply]
  simp only [Complex.add_re, Complex.add_im, Complex.one_re, Complex.one_im]
  congr 1 <;> ring

theorem re_inv_one_sub_sq_nonneg (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t < 1) (z : ℂ)
    (hz : ‖z - (t : ℂ)‖ = 9 / 10 * (1 - t)) :
    5 / 19 / (1 - t) ≤ (1 / (1 - z ^ 2)).re := by
  have hd : 0 < 1 - t := by linarith
  have hsq : (z.re - t) ^ 2 + z.im ^ 2 = (9 / 10 * (1 - t)) ^ 2 := by
    have h := congrArg (fun r : ℝ => r ^ 2) hz
    simp only [Complex.sq_norm, Complex.normSq_apply, Complex.sub_re, Complex.sub_im,
      Complex.ofReal_re, Complex.ofReal_im] at h
    nlinarith [h]
  have ha : -(9 / 10 * (1 - t)) ≤ z.re - t ∧ z.re - t ≤ 9 / 10 * (1 - t) := by
    refine abs_le_of_sq_le_sq' ?_ (by linarith)
    nlinarith [sq_nonneg z.im, hsq]
  obtain ⟨hlo, hhi⟩ := ha
  have hx1 : 0 < 1 - z.re := by linarith
  have hx2 : 0 < 1 + z.re := by linarith
  have h1z : (1 : ℂ) - z ≠ 0 := by
    intro h
    have := congrArg Complex.re h
    simp at this
    linarith
  have h2z : (1 : ℂ) + z ≠ 0 := by
    intro h
    have := congrArg Complex.re h
    simp at this
    linarith
  have h3 : (1 : ℂ) - z ^ 2 ≠ 0 := by
    have : (1 : ℂ) - z ^ 2 = (1 - z) * (1 + z) := by ring
    rw [this]
    exact mul_ne_zero h1z h2z
  have hre : (1 / (1 - z ^ 2)).re
      = 1 / 2 * ((1 / (1 - z)).re + (1 / (1 + z)).re) := by
    have hid : 1 / (1 - z ^ 2) = ((1 / 2 : ℝ) : ℂ) * (1 / (1 - z) + 1 / (1 + z)) := by
      push_cast
      field_simp
      ring
    rw [hid, Complex.re_ofReal_mul, Complex.add_re]
  rw [hre, re_one_div_one_sub, re_one_div_one_add]
  have hQ : 0 < (1 - z.re) ^ 2 + z.im ^ 2 :=
    add_pos_of_pos_of_nonneg (pow_pos hx1 2) (sq_nonneg _)
  have hA : 1 / (19 / 10 * (1 - t)) ≤ (1 - z.re) / ((1 - z.re) ^ 2 + z.im ^ 2) := by
    first
      | rw [div_le_div_iff₀ (by linarith) hQ]
      | rw [div_le_div_iff (by linarith) hQ]
    nlinarith [mul_nonneg hd.le (by linarith : 0 ≤ 9 / 10 * (1 - t) + (z.re - t)), hsq]
  have hB : 0 ≤ (1 + z.re) / ((1 + z.re) ^ 2 + z.im ^ 2) :=
    div_nonneg hx2.le (add_nonneg (sq_nonneg _) (sq_nonneg _))
  have hid2 : 5 / 19 / (1 - t) = 1 / 2 * (1 / (19 / 10 * (1 - t))) := by
    have : (1 - t) ≠ 0 := hd.ne'
    field_simp <;> ring
  rw [hid2]
  linarith

theorem re_inv_one_sub_sq_ge (t : ℝ) (ht : |t| < 1) (z : ℂ)
    (hz : ‖z - (t : ℂ)‖ = 9 / 10 * (1 - |t|)) :
    5 / 19 / (1 - |t|) ≤ (1 / (1 - z ^ 2)).re := by
  have ht' := abs_lt.mp ht
  rcases le_or_gt 0 t with h0 | h0
  · rw [abs_of_nonneg h0] at hz ⊢
    exact re_inv_one_sub_sq_nonneg t h0 ht'.2 z hz
  · have hz' : ‖-z - ((-t : ℝ) : ℂ)‖ = 9 / 10 * (1 - (-t)) := by
      rw [show -z - ((-t : ℝ) : ℂ) = -(z - (t : ℂ)) by push_cast; ring, norm_neg, hz,
        abs_of_neg h0]
    have h := re_inv_one_sub_sq_nonneg (-t) (by linarith) (by linarith) (-z) hz'
    rw [neg_sq] at h
    rw [abs_of_neg h0]
    exact h

/-! ## Complex extension of the smoothing weight -/

/-- `exp(-1/(1-z²)) (1+δz)^s`, the complex extension of the unnormalized weight. -/
noncomputable def gc (s δ : ℝ) (z : ℂ) : ℂ :=
  Complex.exp (-(1 / (1 - z ^ 2))) * (1 + (δ : ℂ) * z) ^ (s : ℂ)

/-- The normalized complex extension `η(z)(1+δz)^s`. -/
noncomputable def Gc (s δ : ℝ) (z : ℂ) : ℂ :=
  ((bumpMass⁻¹ : ℝ) : ℂ) * gc s δ z

/-- The weight as a real-variable, complex-valued function (the integrand of `Fs`). -/
noncomputable def fc (s δ : ℝ) (y : ℝ) : ℂ := ((eta y * (1 + δ * y) ^ s : ℝ) : ℂ)

/-- The constant `C* = C_η (3/2)^{3/2}`. -/
noncomputable def gevC : ℝ := bumpMass⁻¹ * (3 / 2 : ℝ) ^ (3 / 2 : ℝ)

theorem gevC_nonneg : 0 ≤ gevC := by
  unfold gevC
  have := bumpMass_pos
  positivity

theorem eta_mul_eq_Gc (s δ t : ℝ) (ht : |t| < 1) (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2) :
    ((eta t * (1 + δ * t) ^ s : ℝ) : ℂ) = Gc s δ (t : ℂ) := by
  have ht' := abs_lt.mp ht
  have h1 : 0 < 1 - t ^ 2 := by
    nlinarith [mul_pos (by linarith : 0 < 1 - t) (by linarith : 0 < 1 + t)]
  have h2 : 0 < 1 + δ * t := by
    nlinarith [mul_nonneg hδ0 (by linarith [ht'.1] : 0 ≤ t + 1)]
  have hb : bump t = Real.exp (-(1 / (1 - t ^ 2))) := by
    unfold bump expNegInvGlue
    rw [if_neg (not_le.mpr h1), one_div]
  have e1 : ((Real.exp (-(1 / (1 - t ^ 2))) : ℝ) : ℂ)
      = Complex.exp (-(1 / (1 - (t : ℂ) ^ 2))) := by
    rw [Complex.ofReal_exp]
    congr 1
    push_cast
    try ring
  have e2 : (((1 + δ * t) ^ s : ℝ) : ℂ) = ((1 : ℂ) + (δ : ℂ) * (t : ℂ)) ^ (s : ℂ) := by
    rw [Complex.ofReal_cpow h2.le]
    push_cast
    try rfl
  have e3 : eta t * (1 + δ * t) ^ s = bumpMass⁻¹ * (bump t * (1 + δ * t) ^ s) := by
    unfold eta
    ring
  rw [e3, hb, Complex.ofReal_mul, Complex.ofReal_mul, e1, e2]
  rfl

theorem gc_diffAt (s δ : ℝ) (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2) (z : ℂ) (hz : ‖z‖ < 1) :
    DifferentiableAt ℂ (gc s δ) z := by
  have hz2 : ‖z ^ 2‖ < 1 := by
    rw [norm_pow]
    nlinarith [norm_nonneg z]
  have h1 : (1 : ℂ) - z ^ 2 ≠ 0 := by
    intro h
    have h' : (1 : ℂ) = z ^ 2 := sub_eq_zero.mp h
    rw [← h'] at hz2
    simp at hz2
  have hre : 0 < ((1 : ℂ) + (δ : ℂ) * z).re := by
    have e : ((1 : ℂ) + (δ : ℂ) * z).re = 1 + δ * z.re := by simp
    rw [e]
    obtain ⟨hl, hr⟩ := abs_lt.mp (lt_of_le_of_lt (Complex.abs_re_le_norm z) hz)
    nlinarith [mul_nonneg hδ0 (by linarith : 0 ≤ z.re + 1)]
  have h2 : ((1 : ℂ) + (δ : ℂ) * z) ∈ Complex.slitPlane :=
    Complex.mem_slitPlane_iff.2 (Or.inl hre)
  have hd1 : DifferentiableAt ℂ (fun w : ℂ => (1 : ℂ) - w ^ 2) z := by fun_prop
  have hd2 : DifferentiableAt ℂ (fun w : ℂ => Complex.exp (-(1 / (1 - w ^ 2)))) z :=
    (((differentiableAt_const (1 : ℂ)).div hd1 h1).neg).cexp
  have hd3 : DifferentiableAt ℂ (fun w : ℂ => ((1 : ℂ) + (δ : ℂ) * w) ^ (s : ℂ)) z :=
    (show DifferentiableAt ℂ (fun w : ℂ => (1 : ℂ) + (δ : ℂ) * w) z by fun_prop).cpow
      (differentiableAt_const _) h2
  exact hd2.mul hd3

theorem Gc_diffOn (s δ : ℝ) (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2) :
    DifferentiableOn ℂ (Gc s δ) (Metric.ball (0 : ℂ) 1) := by
  intro z hz
  have := gc_diffAt s δ hδ0 hδ z (mem_ball_zero_iff.mp hz)
  exact ((differentiableAt_const _).mul this).differentiableWithinAt

theorem Gc_iteratedDeriv_diffOn (s δ : ℝ) (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2) (r : ℕ) :
    DifferentiableOn ℂ (iteratedDeriv r (Gc s δ)) (Metric.ball (0 : ℂ) 1) := by
  induction r with
  | zero =>
    rw [iteratedDeriv_zero]
    exact Gc_diffOn s δ hδ0 hδ
  | succ r ih =>
    rw [iteratedDeriv_succ]
    exact ih.deriv Metric.isOpen_ball

/-- Real-variable iterated derivatives of `fc` agree with the complex ones of `Gc` on `(-1,1)`. -/
theorem fc_iteratedDeriv (s δ : ℝ) (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2) (r : ℕ) :
    ∀ x : ℝ, |x| < 1 → iteratedDeriv r (fc s δ) x = iteratedDeriv r (Gc s δ) (x : ℂ) := by
  induction r with
  | zero =>
    intro x hx
    rw [iteratedDeriv_zero, iteratedDeriv_zero]
    exact eta_mul_eq_Gc s δ x hx hδ0 hδ
  | succ r ih =>
    intro x hx
    have hnhds : {y : ℝ | |y| < 1} ∈ nhds x :=
      (isOpen_lt continuous_abs continuous_const).mem_nhds hx
    have hev : iteratedDeriv r (fc s δ) =ᶠ[nhds x]
        fun y : ℝ => iteratedDeriv r (Gc s δ) (y : ℂ) := by
      filter_upwards [hnhds] with y hy using ih y hy
    have hmem : (x : ℂ) ∈ Metric.ball (0 : ℂ) 1 := by
      rw [mem_ball_zero_iff, Complex.norm_real, Real.norm_eq_abs]
      exact hx
    have hda : DifferentiableAt ℂ (iteratedDeriv r (Gc s δ)) (x : ℂ) :=
      (Gc_iteratedDeriv_diffOn s δ hδ0 hδ r).differentiableAt
        (Metric.isOpen_ball.mem_nhds hmem)
    rw [iteratedDeriv_succ, iteratedDeriv_succ, hev.deriv_eq]
    exact (hda.hasDerivAt.comp_ofReal).deriv

/-! ## Cauchy estimate on a disc around `t` -/

theorem norm_lt_one_of_near (t d : ℝ) (hd : d = 1 - |t|) (ht : |t| < 1) (w : ℂ)
    (hw : ‖w - (t : ℂ)‖ ≤ 9 / 10 * d) : ‖w‖ < 1 := by
  have h := norm_sub_norm_le w (t : ℂ)
  rw [Complex.norm_real, Real.norm_eq_abs] at h
  subst hd
  linarith

theorem Gc_sphere_bound (s δ : ℝ) (hs0 : 0 ≤ s) (hs : s ≤ 3 / 2) (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 2)
    (t d : ℝ) (hd : d = 1 - |t|) (ht : |t| < 1) (z : ℂ) (hz : ‖z - (t : ℂ)‖ = 9 / 10 * d) :
    ‖Gc s δ z‖ ≤ gevC * Real.exp (-(5 / 19 / d)) := by
  have hz1 : ‖z‖ < 1 := norm_lt_one_of_near t d hd ht z hz.le
  have hre : 5 / 19 / d ≤ (1 / (1 - z ^ 2)).re := by
    subst hd
    exact re_inv_one_sub_sq_ge t ht z hz
  have hn : ‖(1 : ℂ) + (δ : ℂ) * z‖ ≤ 3 / 2 := by
    calc ‖(1 : ℂ) + (δ : ℂ) * z‖ ≤ ‖(1 : ℂ)‖ + ‖(δ : ℂ) * z‖ := norm_add_le _ _
      _ = 1 + δ * ‖z‖ := by
          rw [norm_one, norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg hδ0]
      _ ≤ 3 / 2 := by nlinarith [norm_nonneg z]
  have hpw : ‖(1 : ℂ) + (δ : ℂ) * z‖ ^ s ≤ (3 / 2 : ℝ) ^ (3 / 2 : ℝ) :=
    calc ‖(1 : ℂ) + (δ : ℂ) * z‖ ^ s ≤ (3 / 2 : ℝ) ^ s :=
          Real.rpow_le_rpow (norm_nonneg _) hn hs0
      _ ≤ (3 / 2 : ℝ) ^ (3 / 2 : ℝ) := Real.rpow_le_rpow_of_exponent_le (by norm_num) hs
  have hexp : Real.exp (-(1 / (1 - z ^ 2))).re ≤ Real.exp (-(5 / 19 / d)) := by
    apply Real.exp_le_exp.2
    rw [Complex.neg_re]
    linarith
  unfold Gc gc gevC
  rw [norm_mul, Complex.norm_real, norm_mul, Real.norm_eq_abs,
    abs_of_pos (inv_pos.mpr bumpMass_pos), Complex.norm_exp, Complex.norm_cpow_real]
  have hm := mul_le_mul hexp hpw (Real.rpow_nonneg (norm_nonneg _) _) (Real.exp_pos _).le
  have := mul_le_mul_of_nonneg_left hm (inv_pos.mpr bumpMass_pos).le
  nlinarith [this]

theorem iteratedDeriv_fc_bound (s δ : ℝ) (hs0 : 0 ≤ s) (hs : s ≤ 3 / 2) (hδ0 : 0 ≤ δ)
    (hδ : δ ≤ 1 / 2) (r : ℕ) (hr : 1 ≤ r) (t : ℝ) (ht : |t| < 1) :
    ‖iteratedDeriv r (fc s δ) t‖ ≤ gevC * (38 / 9 : ℝ) ^ r * (r.factorial : ℝ) ^ 2 := by
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
  have hcau := Complex.norm_iteratedDeriv_le_of_forall_mem_sphere_norm_le r hR hdc hC
  rw [fc_iteratedDeriv s δ hδ0 hδ r t ht]
  have e7 : (9 / 38 : ℝ) / (9 / 10 * d) = 5 / 19 / d := by
    have : d ≠ 0 := hdpos.ne'
    field_simp <;> ring
  have he : Real.exp 1 ≠ 0 := (Real.exp_pos 1).ne'
  have e8 : ((r : ℝ) / (Real.exp 1 * (9 / 38))) ^ r
      = (38 / 9 : ℝ) ^ r * ((r : ℝ) / Real.exp 1) ^ r := by
    rw [← mul_pow]
    congr 1
    field_simp <;> ring
  have hnn : 0 ≤ (r.factorial : ℝ) * gevC := mul_nonneg (Nat.cast_nonneg _) gevC_nonneg
  calc ‖iteratedDeriv r (Gc s δ) (t : ℂ)‖
      ≤ (r.factorial : ℝ) * (gevC * Real.exp (-(5 / 19 / d))) / (9 / 10 * d) ^ r := hcau
    _ = (r.factorial : ℝ) * gevC
          * (Real.exp (-((9 / 38 : ℝ) / (9 / 10 * d))) / (9 / 10 * d) ^ r) := by
        rw [e7]; ring
    _ ≤ (r.factorial : ℝ) * gevC * (((r : ℝ) / (Real.exp 1 * (9 / 38))) ^ r) :=
        mul_le_mul_of_nonneg_left
          (exp_neg_div_pow_le (9 / 38) (9 / 10 * d) (by norm_num) hR r hr) hnn
    _ = (r.factorial : ℝ) * gevC * ((38 / 9 : ℝ) ^ r * ((r : ℝ) / Real.exp 1) ^ r) := by
        rw [e8]
    _ ≤ (r.factorial : ℝ) * gevC * ((38 / 9 : ℝ) ^ r * (r.factorial : ℝ)) :=
        mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_left (div_exp_pow_le_factorial r) (pow_nonneg (by norm_num) r))
          hnn
    _ = gevC * (38 / 9 : ℝ) ^ r * (r.factorial : ℝ) ^ 2 := by ring

/-- The note's Gevrey estimate: `‖f^{(r)}‖_{L¹(-1,1)} ≤ 2 C* B^r (r!)²` with `B = 38/9`. -/
theorem fc_iteratedDeriv_L1_le (s δ : ℝ) (hs0 : 0 ≤ s) (hs : s ≤ 3 / 2) (hδ0 : 0 ≤ δ)
    (hδ : δ ≤ 1 / 2) (r : ℕ) (hr : 1 ≤ r) :
    ∫ t in Set.Ioo (-1 : ℝ) 1, ‖iteratedDeriv r (fc s δ) t‖
      ≤ 2 * (gevC * (38 / 9 : ℝ) ^ r * (r.factorial : ℝ) ^ 2) := by
  have hpt : ∀ t ∈ Set.Ioo (-1 : ℝ) 1,
      ‖iteratedDeriv r (fc s δ) t‖ ≤ gevC * (38 / 9 : ℝ) ^ r * (r.factorial : ℝ) ^ 2 :=
    fun t ht => iteratedDeriv_fc_bound s δ hs0 hs hδ0 hδ r hr t (abs_lt.mpr ⟨ht.1, ht.2⟩)
  calc ∫ t in Set.Ioo (-1 : ℝ) 1, ‖iteratedDeriv r (fc s δ) t‖
      ≤ ∫ _t in Set.Ioo (-1 : ℝ) 1, gevC * (38 / 9 : ℝ) ^ r * (r.factorial : ℝ) ^ 2 := by
        apply MeasureTheory.integral_mono_of_nonneg
        · exact MeasureTheory.ae_restrict_of_forall_mem measurableSet_Ioo
            (fun t _ => norm_nonneg _)
        · exact MeasureTheory.integrable_const _
        · exact MeasureTheory.ae_restrict_of_forall_mem measurableSet_Ioo hpt
    _ = 2 * (gevC * (38 / 9 : ℝ) ^ r * (r.factorial : ℝ) ^ 2) := by
        rw [MeasureTheory.setIntegral_const, MeasureTheory.measureReal_def, Real.volume_Ioo,
          ENNReal.toReal_ofReal (by norm_num), smul_eq_mul]
        norm_num

end DirichletDivisor

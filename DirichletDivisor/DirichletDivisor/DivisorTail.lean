import DirichletDivisor.Defs
import DirichletDivisor.TailSeries

set_option linter.style.whitespace false

namespace DirichletDivisor

/-! ## The divisor bound `d(m) ≤ 2√m` and the weighted tail estimate -/

/-- `d(m) ≤ 2 √m`: divisors `a` with `a² ≤ m` number at most `⌊√m⌋`, and divisors with
`a² > m` inject into them via `a ↦ m / a`. -/
theorem d_le_two_sqrt (m : ℕ) : (d m : ℝ) ≤ 2 * Real.sqrt m := by
  rcases Nat.eq_zero_or_pos m with h0 | hm
  · subst h0
    simp [d]
  have hsplit : d m = ((Nat.divisors m).filter (fun a => a * a ≤ m)).card
      + ((Nat.divisors m).filter (fun a => ¬ a * a ≤ m)).card := by
    unfold d
    simp only [Finset.card_eq_sum_ones]
    exact (Finset.sum_filter_add_sum_filter_not _ _ _).symm
  have hS : ((Nat.divisors m).filter (fun a => a * a ≤ m)).card ≤ Nat.sqrt m := by
    calc ((Nat.divisors m).filter (fun a => a * a ≤ m)).card
        ≤ (Finset.Icc 1 (Nat.sqrt m)).card := by
          apply Finset.card_le_card
          intro a ha
          rw [Finset.mem_filter, Nat.mem_divisors] at ha
          obtain ⟨⟨hdv, _⟩, hle⟩ := ha
          exact Finset.mem_Icc.2 ⟨Nat.pos_of_dvd_of_pos hdv hm, Nat.le_sqrt.2 hle⟩
      _ = Nat.sqrt m := by simp
  have hT : ((Nat.divisors m).filter (fun a => ¬ a * a ≤ m)).card
      ≤ ((Nat.divisors m).filter (fun a => a * a ≤ m)).card := by
    apply Finset.card_le_card_of_injOn (fun a => m / a)
    · intro a ha
      rw [Finset.mem_coe, Finset.mem_filter, Nat.mem_divisors] at ha
      obtain ⟨⟨hdv, hm0⟩, hlt⟩ := ha
      have ha0 : 0 < a := Nat.pos_of_dvd_of_pos hdv hm
      obtain ⟨c, hc⟩ := hdv
      have hdiv : m / a = c := by
        rw [hc]
        exact Nat.mul_div_cancel_left c ha0
      have hca : c < a := by
        by_contra hh
        push_neg at hh
        exact hlt (calc a * a ≤ a * c := Nat.mul_le_mul_left a hh
          _ = m := hc.symm)
      have hcc : c * c ≤ m := calc c * c ≤ a * c := Nat.mul_le_mul_right c hca.le
        _ = m := hc.symm
      have hmem : m / a ∈ (Nat.divisors m).filter (fun a => a * a ≤ m) :=
        Finset.mem_filter.2
          ⟨Nat.mem_divisors.2 ⟨⟨a, by rw [hdiv, hc]; ring⟩, hm0⟩, by rw [hdiv]; exact hcc⟩
      exact Finset.mem_coe.2 hmem
    · intro a ha b hb hab
      have ha' := (Finset.mem_filter.1 (Finset.mem_coe.1 ha)).1
      have hb' := (Finset.mem_filter.1 (Finset.mem_coe.1 hb)).1
      have h1 := Nat.div_div_self (Nat.dvd_of_mem_divisors ha') hm.ne'
      have h2 := Nat.div_div_self (Nat.dvd_of_mem_divisors hb') hm.ne'
      change m / a = m / b at hab
      calc a = m / (m / a) := h1.symm
        _ = m / (m / b) := by rw [hab]
        _ = b := h2
  have hnat : d m ≤ 2 * Nat.sqrt m := by omega
  calc (d m : ℝ) ≤ 2 * (Nat.sqrt m : ℝ) := by exact_mod_cast hnat
    _ ≤ 2 * Real.sqrt m :=
        mul_le_mul_of_nonneg_left Real.nat_sqrt_le_real_sqrt (by norm_num)

/-- Weighted exponential tail, partial sums.  For `k ≥ 4` and `p = 1/4 + k/4`,
`∑_{j<n} d(m_j) · m_j^{-3/4} e^{-c m_j^{1/4}} ≤ 2 (k!/c^k) · M^{1-p}/(p-1)`,
where `m_j = M + j + 1`. -/
theorem tail_weighted_partial (c : ℝ) (hc : 0 < c) (k : ℕ) (hk : 4 ≤ k) (M n : ℕ)
    (hM : 1 ≤ M) :
    ∑ j ∈ Finset.range n, (d (M + j + 1) : ℝ)
        * (((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ))
            * Real.exp (-(c * ((M + j + 1 : ℕ) : ℝ) ^ (1 / 4 : ℝ))))
      ≤ 2 * ((k.factorial : ℝ) / c ^ k)
          * ((M : ℝ) ^ (1 - (1 / 4 + (k : ℝ) / 4)) / ((1 / 4 + (k : ℝ) / 4) - 1)) := by
  have hk' : (4 : ℝ) ≤ k := by exact_mod_cast hk
  have hp : 1 < (1 / 4 + (k : ℝ) / 4) := by linarith
  calc ∑ j ∈ Finset.range n, (d (M + j + 1) : ℝ)
        * (((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ))
            * Real.exp (-(c * ((M + j + 1 : ℕ) : ℝ) ^ (1 / 4 : ℝ))))
      ≤ ∑ j ∈ Finset.range n, 2 * ((k.factorial : ℝ) / c ^ k)
          * ((M + j + 1 : ℕ) : ℝ) ^ (-(1 / 4 + (k : ℝ) / 4)) := by
        apply Finset.sum_le_sum
        intro j _
        have hm : (0 : ℝ) < ((M + j + 1 : ℕ) : ℝ) := by positivity
        have h := term_le c hc k ((M + j + 1 : ℕ) : ℝ) hm
        have hd := d_le_two_sqrt (M + j + 1)
        have hT : 0 ≤ ((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ))
            * Real.exp (-(c * ((M + j + 1 : ℕ) : ℝ) ^ (1 / 4 : ℝ))) :=
          mul_nonneg (Real.rpow_nonneg hm.le _) (Real.exp_pos _).le
        calc (d (M + j + 1) : ℝ)
              * (((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ))
                  * Real.exp (-(c * ((M + j + 1 : ℕ) : ℝ) ^ (1 / 4 : ℝ))))
            ≤ (2 * Real.sqrt ((M + j + 1 : ℕ) : ℝ))
                * ((k.factorial : ℝ) / c ^ k
                    * ((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ) - (k : ℝ) / 4)) :=
              mul_le_mul hd h hT (by positivity)
          _ = 2 * ((k.factorial : ℝ) / c ^ k)
                * (Real.sqrt ((M + j + 1 : ℕ) : ℝ)
                    * ((M + j + 1 : ℕ) : ℝ) ^ (-(3 / 4 : ℝ) - (k : ℝ) / 4)) := by ring
          _ = 2 * ((k.factorial : ℝ) / c ^ k)
                * ((M + j + 1 : ℕ) : ℝ) ^ (-(1 / 4 + (k : ℝ) / 4)) := by
              rw [Real.sqrt_eq_rpow, ← Real.rpow_add hm,
                show (1 / 2 : ℝ) + (-(3 / 4 : ℝ) - (k : ℝ) / 4) = -(1 / 4 + (k : ℝ) / 4) by ring]
    _ = 2 * ((k.factorial : ℝ) / c ^ k)
          * ∑ j ∈ Finset.range n, ((M + j + 1 : ℕ) : ℝ) ^ (-(1 / 4 + (k : ℝ) / 4)) :=
        (Finset.mul_sum _ _ _).symm
    _ ≤ 2 * ((k.factorial : ℝ) / c ^ k)
          * ((M : ℝ) ^ (1 - (1 / 4 + (k : ℝ) / 4)) / ((1 / 4 + (k : ℝ) / 4) - 1)) :=
        mul_le_mul_of_nonneg_left (pseries_partial_le _ hp M n hM) (by positivity)

end DirichletDivisor

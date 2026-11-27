import DirichletDivisor.Defs

open Finset

namespace DirichletDivisor

/-- Step 1: ∑_{m ≤ M} d(m) f(m) = ∑ over ordered pairs (a,b), ab ≤ M, of f(ab). -/
theorem sum_d_mul_eq_sum_pairs (M : ℕ) (f : ℕ → ℝ) :
    ∑ m ∈ Icc 1 M, (d m : ℝ) * f m
      = ∑ p ∈ (Icc 1 M ×ˢ Icc 1 M).filter (fun p => p.1 * p.2 ≤ M), f (p.1 * p.2) := by
  have h1 : ∀ m ∈ Icc 1 M, (d m : ℝ) * f m
      = ∑ p ∈ Nat.divisorsAntidiagonal m, f (p.1 * p.2) := by
    intro m _
    rw [Nat.sum_divisorsAntidiagonal (fun a b => f (a * b))]
    have h : ∀ i ∈ m.divisors, f (i * (m / i)) = f m := fun i hi => by
      rw [Nat.mul_div_cancel' (Nat.dvd_of_mem_divisors hi)]
    rw [Finset.sum_congr rfl h]
    simp [d]
  rw [Finset.sum_congr rfl h1, Finset.sum_sigma']
  refine Finset.sum_bij' (fun x _ => x.2) (fun p _ => ⟨p.1 * p.2, p⟩) ?_ ?_ ?_ ?_ ?_
  · intro x hx
    simp only [Finset.mem_sigma, Nat.mem_divisorsAntidiagonal, Finset.mem_Icc] at hx
    obtain ⟨⟨hm1, hmM⟩, hab, hm0⟩ := hx
    simp only [Finset.mem_filter, Finset.mem_product, Finset.mem_Icc]
    have ha : 0 < x.2.1 := Nat.pos_of_ne_zero (fun h => by simp [h] at hab; omega)
    have hb : 0 < x.2.2 := Nat.pos_of_ne_zero (fun h => by simp [h] at hab; omega)
    refine ⟨⟨⟨ha, ?_⟩, ⟨hb, ?_⟩⟩, by rw [hab]; exact hmM⟩
    · calc x.2.1 ≤ x.2.1 * x.2.2 := Nat.le_mul_of_pos_right _ hb
        _ = x.1 := hab
        _ ≤ M := hmM
    · calc x.2.2 ≤ x.2.1 * x.2.2 := Nat.le_mul_of_pos_left _ ha
        _ = x.1 := hab
        _ ≤ M := hmM
  · intro p hp
    simp only [Finset.mem_filter, Finset.mem_product, Finset.mem_Icc] at hp
    obtain ⟨⟨⟨ha, _⟩, ⟨hb, _⟩⟩, hle⟩ := hp
    have hpos : 0 < p.1 * p.2 := Nat.mul_pos ha hb
    simp only [Finset.mem_sigma, Nat.mem_divisorsAntidiagonal, Finset.mem_Icc]
    exact ⟨⟨hpos, hle⟩, by simp [hpos.ne']⟩
  · intro x hx
    simp only [Finset.mem_sigma, Nat.mem_divisorsAntidiagonal] at hx
    obtain ⟨m, p⟩ := x
    simp only at hx ⊢
    obtain ⟨_, hab, _⟩ := hx
    subst hab
    try rfl
  · intro p _
    rfl
  · intro x hx
    simp only [Finset.mem_sigma, Nat.mem_divisorsAntidiagonal] at hx
    obtain ⟨m, p⟩ := x
    simp only at hx ⊢

/-- Step 2 (the note's §3 identity):
∑_{m ≤ M} d(m) f(m) = 2 ∑_{a ≤ b, ab ≤ M} f(ab) - ∑_{a² ≤ M} f(a²). -/
theorem sum_d_mul_symm (M : ℕ) (f : ℕ → ℝ) :
    ∑ m ∈ Icc 1 M, (d m : ℝ) * f m
      = 2 * ∑ p ∈ (Icc 1 M ×ˢ Icc 1 M).filter (fun p => p.1 ≤ p.2 ∧ p.1 * p.2 ≤ M),
              f (p.1 * p.2)
        - ∑ a ∈ (Icc 1 M).filter (fun a => a * a ≤ M), f (a * a) := by
  rw [sum_d_mul_eq_sum_pairs]
  set S := (Icc 1 M ×ˢ Icc 1 M).filter (fun p => p.1 * p.2 ≤ M) with hS
  set T := (Icc 1 M ×ˢ Icc 1 M).filter (fun p => p.1 ≤ p.2 ∧ p.1 * p.2 ≤ M) with hTdef
  -- membership in S
  have hmem : ∀ p : ℕ × ℕ,
      p ∈ S ↔ (p.1 ∈ Icc 1 M ∧ p.2 ∈ Icc 1 M) ∧ p.1 * p.2 ≤ M := by
    intro p
    rw [hS, Finset.mem_filter, Finset.mem_product]
  -- S is closed under swap
  have hswapmem : ∀ p : ℕ × ℕ, p ∈ S ↔ p.swap ∈ S := by
    intro p
    rw [hmem p, hmem p.swap]
    constructor
    · rintro ⟨⟨h1, h2⟩, h3⟩
      exact ⟨⟨h2, h1⟩, by simpa [mul_comm] using h3⟩
    · rintro ⟨⟨h1, h2⟩, h3⟩
      exact ⟨⟨h2, h1⟩, by simpa [mul_comm] using h3⟩
  -- T = S ∩ {a ≤ b}
  have hT : T = S.filter (fun p => p.1 ≤ p.2) := by
    rw [hTdef, hS]
    ext p
    simp only [Finset.mem_filter, Finset.mem_product]
    tauto
  -- split S into a<b, a=b, a>b
  have hsplit : ∑ p ∈ S, f (p.1 * p.2)
      = ∑ p ∈ S.filter (fun p => p.1 < p.2), f (p.1 * p.2)
        + ∑ p ∈ S.filter (fun p => p.1 = p.2), f (p.1 * p.2)
        + ∑ p ∈ S.filter (fun p => p.2 < p.1), f (p.1 * p.2) := by
    simp only [Finset.sum_filter, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun p _ => ?_
    rcases lt_trichotomy p.1 p.2 with h | h | h
    · simp [h, h.ne, not_lt.mpr h.le]
    · simp [h]
    · simp [h, h.ne', not_lt.mpr h.le]
  -- swap symmetry: the a>b part equals the a<b part
  have hswap : ∑ p ∈ S.filter (fun p => p.2 < p.1), f (p.1 * p.2)
      = ∑ p ∈ S.filter (fun p => p.1 < p.2), f (p.1 * p.2) := by
    refine Finset.sum_nbij' Prod.swap Prod.swap ?_ ?_ ?_ ?_ ?_
    · intro p hp
      rw [Finset.mem_filter] at hp ⊢
      exact ⟨(hswapmem p).1 hp.1, by simpa using hp.2⟩
    · intro p hp
      rw [Finset.mem_filter] at hp ⊢
      exact ⟨(hswapmem p).1 hp.1, by simpa using hp.2⟩
    · intro p _
      exact Prod.swap_swap p
    · intro p _
      exact Prod.swap_swap p
    · intro p _
      simp [mul_comm]
  -- diagonal
  have hdiag : ∑ p ∈ S.filter (fun p => p.1 = p.2), f (p.1 * p.2)
      = ∑ a ∈ (Icc 1 M).filter (fun a => a * a ≤ M), f (a * a) := by
    refine Finset.sum_nbij' (fun p => p.1) (fun a => (a, a)) ?_ ?_ ?_ ?_ ?_
    · intro p hp
      rw [Finset.mem_filter, hmem] at hp
      obtain ⟨⟨⟨h1, _⟩, h3⟩, heq⟩ := hp
      have e : p.1 * p.1 = p.1 * p.2 := by rw [heq]
      rw [Finset.mem_filter]
      exact ⟨h1, e.trans_le h3⟩
    · intro a ha
      rw [Finset.mem_filter] at ha
      rw [Finset.mem_filter, hmem]
      exact ⟨⟨⟨ha.1, ha.1⟩, ha.2⟩, rfl⟩
    · intro p hp
      rw [Finset.mem_filter] at hp
      exact Prod.ext rfl hp.2
    · intro a _
      rfl
    · intro p hp
      rw [Finset.mem_filter] at hp
      obtain ⟨_, heq⟩ := hp
      show f (p.1 * p.2) = f (p.1 * p.1)
      rw [heq]
  -- the a ≤ b part is (a<b) ∪ (a=b)
  have hle : ∑ p ∈ T, f (p.1 * p.2)
      = ∑ p ∈ S.filter (fun p => p.1 < p.2), f (p.1 * p.2)
        + ∑ p ∈ S.filter (fun p => p.1 = p.2), f (p.1 * p.2) := by
    rw [hT]
    simp only [Finset.sum_filter, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun p _ => ?_
    rcases lt_trichotomy p.1 p.2 with h | h | h
    · simp [h, h.le, h.ne]
    · simp [h]
    · simp [not_le.mpr h, not_lt.mpr h.le, h.ne']
  linarith

end DirichletDivisor

#!/usr/bin/env python3
"""
b1_identity.py

Verifies and analyzes the non-square Gaussian integer identity:

    b1(N) = N * A(N-1) - Sigma1(N-1)

where:
    r2(k)       = #{(a,b) in Z^2 : a^2 + b^2 = k}  (signed, all integers)
    A(N)        = sum_{k=1}^{N} r2(k)               (Gauss circle counting fn)
    Sigma1(N)   = sum_{k=1}^{N} k * r2(k)           (weighted sum)

    b1(N)       = #{ (alpha, j) : alpha in Z[i], alpha != 0, j >= 1, N(alpha) + j <= N }
                = sum_{k=1}^{N-1} (N - k) * r2(k)   (combinatorial definition)

Geometric picture:
    b1(N) counts lattice points (alpha, j) in the region
        { N(alpha) + j <= N, N(alpha) >= 1, j >= 1 }
    which is the integer points under a "staircase" built from the
    Gauss circle: for each shell of norm k, there are r2(k) Gaussian
    integers, each contributing (N - k) choices of j.  The identity
    is then Abel summation / summation by parts.

Asymptotic:
    A(N) ~ pi * N  (the Gauss circle theorem)
    b1(N) ~ (pi/2) * N^2  (leading term)
    Error: E(N) = b1(N) - (pi/2) * N^2
           expected to fluctuate as O(N^(1/2+eps)), tied to the
           Gaussian circle problem error term.
"""

import math
from collections import defaultdict


# =============================================================================
# Core arithmetic
# =============================================================================

def chi_m4(n):
    """Dirichlet character chi_{-4}(n)."""
    if n % 2 == 0:
        return 0
    return 1 if n % 4 == 1 else -1


def r2(n):
    """
    r2(n) = #{(a,b) in Z^2 : a^2 + b^2 = n}, via the divisor formula
    r2(n) = 4 * sum_{d|n} chi_{-4}(d).
    """
    s = 0
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            s += chi_m4(d)
            if d * d != n:
                s += chi_m4(n // d)
    return 4 * s


def r2_direct(n):
    """Brute-force count of (a,b) with a^2+b^2 = n. For verification only."""
    cnt = 0
    R = int(math.isqrt(n))
    for a in range(-R, R + 1):
        b2 = n - a * a
        if b2 < 0:
            continue
        b = int(math.isqrt(b2))
        if b * b == b2:
            cnt += 1 if b == 0 else 2
    return cnt


# =============================================================================
# Sieve r2 up to limit -- O(N sqrt(N)) but fine for our range
# =============================================================================

def sieve_r2(limit):
    """Return list r2v where r2v[k] = r2(k) for k = 0..limit."""
    r2v = [0] * (limit + 1)
    for k in range(1, limit + 1):
        r2v[k] = r2(k)
    return r2v


# =============================================================================
# Direct (brute-force) b1
#
# b1_direct(N) = #{(alpha, j) : alpha nonzero, j >= 1, N(alpha) + j <= N}
#              = sum over nonzero alpha with N(alpha) <= N-1 of (N - N(alpha))
# =============================================================================

def b1_direct(N):
    """Brute-force b1(N) by iterating over all nonzero Gaussian integers."""
    if N <= 1:
        return 0
    target = N - 1
    total = 0
    R = int(math.isqrt(target))
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            na = a * a + b * b
            if 1 <= na <= target:
                total += N - na
    return total


# =============================================================================
# Formula-based b1
#
# b1(N) = N * A(N-1) - Sigma1(N-1)
# where we maintain running sums A and Sigma1.
# =============================================================================

def b1_formula(N, A_prev, Sigma1_prev):
    """
    Given cumulative A(N-1) and Sigma1(N-1), return b1(N).
    (Caller updates the accumulators.)
    """
    return N * A_prev - Sigma1_prev


# =============================================================================
# Phase 1: Sanity-check r2
# =============================================================================

def test_r2(limit=300):
    print(f"--- Phase 1: r2 sanity check (up to {limit}) ---")
    known = {1: 4, 2: 4, 3: 0, 4: 4, 5: 8, 9: 4, 10: 8, 25: 12}
    for k, v in known.items():
        assert r2(k) == v, f"r2({k}) expected {v}, got {r2(k)}"
    for n in range(1, limit + 1):
        if r2(n) != r2_direct(n):
            print(f"  MISMATCH at n={n}: formula={r2(n)}, direct={r2_direct(n)}")
            return
    print(f"  [OK] r2 formula matches brute force for all n in [1, {limit}]\n")


# =============================================================================
# Phase 2: Cross-verify b1_formula vs b1_direct
# =============================================================================

def test_b1_identity(limit=500):
    print(f"--- Phase 2: b1 identity verification (N = 2..{limit}) ---")
    print(f"{'N':>6}  {'b1_direct':>16}  {'b1_formula':>16}  {'match':>6}")
    print(f"{'------':>6}  {'----------------':>16}  {'----------------':>16}  {'------':>6}")

    r2v = sieve_r2(limit)

    A      = 0   # A(N-1)      = sum_{k=1}^{N-1} r2(k)
    Sigma1 = 0   # Sigma1(N-1) = sum_{k=1}^{N-1} k * r2(k)

    all_match = True
    for N in range(2, limit + 1):
        # update accumulators: add the k = N-1 term
        k = N - 1
        A      += r2v[k]
        Sigma1 += k * r2v[k]

        b_form = N * A - Sigma1
        b_dir  = b1_direct(N)
        match  = (b_form == b_dir)
        if not match:
            all_match = False

        if N <= 30 or N % 50 == 0 or not match:
            flag = "OK" if match else "MISMATCH <<<"
            print(f"{N:>6}  {b_dir:>16}  {b_form:>16}  {flag:>6}")

    print("  ...")
    status = "OK" if all_match else "FAIL -- mismatches found!"
    print(f"\n  [{status}] All N in [2, {limit}] {'verified.' if all_match else 'FAILED.'}\n")
    return all_match


# =============================================================================
# Phase 3: Asymptotic and error term
#
# Leading asymptotic: b1(N) ~ (pi/2) * N^2
#
# Derivation sketch:
#   b1(N) = sum_{k=1}^{N-1} (N-k) * r2(k)
#          = N * A(N-1) - Sigma1(N-1)
#   A(N) ~ pi*N  =>  N * A(N-1) ~ pi * N^2
#   Sigma1(N) ~ sum_{k<=N} k * r2(k) ~ (pi/2) * N^2  (Abel summation of A)
#   =>  b1(N) ~ pi*N^2 - (pi/2)*N^2 = (pi/2)*N^2
#
# Error term:
#   E(N) = b1(N) - (pi/2) * N^2
#
# The Gauss circle problem says A(N) = pi*N + O(N^theta), theta ~ 1/3 currently,
# conjectured 1/4.  The error in b1 inherits this but with an extra integration:
#   E(N) = N * (A(N-1) - pi*(N-1)) - (Sigma1(N-1) - (pi/2)*(N-1)^2) + lower order
# so E(N) ~ N * Delta(N-1)  where Delta(N) = A(N) - pi*N is the circle error.
# Expect E(N) = O(N^(1+theta)) ~ O(N^(4/3)), fluctuating, with sign changes.
# =============================================================================

def analyze_asymptotics(N_max=5000):
    print(f"--- Phase 3: Asymptotic and error term analysis (up to N={N_max}) ---")
    print(f"\n  Leading term: b1(N) ~ (pi/2) * N^2,  pi/2 = {math.pi/2:.8f}")
    print(f"\n{'N':>8}  {'b1(N)':>18}  {'(pi/2)N^2':>18}  {'E(N)=b1-(pi/2)N^2':>20}  {'E(N)/N^(4/3)':>14}  {'ratio b1/(pi/2 N^2)':>20}")
    print("-" * 110)

    r2v = sieve_r2(N_max)
    A      = 0
    Sigma1 = 0

    checkpoints = set([10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000])

    for N in range(2, N_max + 1):
        k = N - 1
        A      += r2v[k]
        Sigma1 += k * r2v[k]

        if N in checkpoints:
            b1     = N * A - Sigma1
            asym   = (math.pi / 2) * N * N
            E      = b1 - asym
            scale  = N ** (4/3)
            ratio  = b1 / asym
            print(f"{N:>8}  {b1:>18}  {asym:>18.2f}  {E:>20.2f}  {E/scale:>14.6f}  {ratio:>20.10f}")

    print()


# =============================================================================
# Phase 4: Gauss circle error Delta(N) = A(N) - pi*N
# and its relationship to b1 error E(N)
# =============================================================================

def analyze_circle_error(N_max=2000):
    print(f"--- Phase 4: Circle error Delta(N) = A(N) - pi*N vs b1 error ---")
    print(f"\n{'N':>8}  {'A(N)':>10}  {'pi*N':>12}  {'Delta(N)':>12}  {'Delta/sqrt(N)':>14}  {'E(N)/N':>12}")
    print("-" * 80)

    r2v = sieve_r2(N_max)
    A      = 0
    Sigma1 = 0

    checkpoints = set([10, 50, 100, 200, 500, 1000, 2000])

    for N in range(2, N_max + 1):
        k = N - 1
        A      += r2v[k]
        Sigma1 += k * r2v[k]

        if N in checkpoints:
            b1      = N * A - Sigma1
            Delta   = A - math.pi * N          # circle error at A(N)
            E       = b1 - (math.pi / 2) * N * N
            print(f"{N:>8}  {A:>10}  {math.pi*N:>12.2f}  {Delta:>12.4f}  {Delta/math.sqrt(N):>14.6f}  {E/N:>12.4f}")

    print()


# =============================================================================
# Phase 5: Sign changes and oscillation of E(N)
# Detect zero crossings in the error term -- evidence of genuine oscillation
# rather than a bias.
# =============================================================================

def analyze_sign_changes(N_max=2000):
    print(f"--- Phase 5: Sign changes in E(N) = b1(N) - (pi/2)*N^2 ---")

    r2v    = sieve_r2(N_max)
    A      = 0
    Sigma1 = 0
    prev_sign = 0
    crossings = []

    for N in range(2, N_max + 1):
        k = N - 1
        A      += r2v[k]
        Sigma1 += k * r2v[k]
        b1      = N * A - Sigma1
        E       = b1 - (math.pi / 2) * N * N
        sign    = 1 if E >= 0 else -1
        if prev_sign != 0 and sign != prev_sign:
            crossings.append(N)
        prev_sign = sign

    print(f"  Found {len(crossings)} sign changes in E(N) for N in [2, {N_max}]")
    if crossings:
        print(f"  First few crossings: {crossings[:20]}")
    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" b1 Identity Verification and Error Analysis")
    print(" b1(N) = N * A(N-1) - Sigma1(N-1)")
    print("=" * 70)
    print()

    test_r2(limit=300)
    test_b1_identity(limit=500)
    analyze_asymptotics(N_max=5000)
    analyze_circle_error(N_max=2000)
    analyze_sign_changes(N_max=2000)

    print("=" * 70)
    print(" Done. Ready for deeper error-term analysis.")
    print("=" * 70)

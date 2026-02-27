#!/usr/bin/env python3
"""
b1_hyperbola.py

Applies the Dirichlet hyperbola / blocking trick to b1(N), giving an
O(sqrt(N)) exact formula.  Then uses the speed gain to study the error term
at large N.

=============================================================================
DERIVATION
=============================================================================

Start from:

    b1(N) = sum_{k=1}^{N-1} (N - k) * r2(k)

Expand r2(k) = 4 * sum_{d | k} chi_{-4}(d) and swap summation order
(set k = d*m):

    b1(N) = 4 * sum_{d=1}^{N-1} chi(d) * sum_{m=1}^{M(d)} (N - d*m)
          = 4 * sum_{d=1}^{N-1} chi(d) * [ N*M(d)  -  d * M(d)*(M(d)+1)/2 ]

where M(d) = floor((N-1)/d).

Now apply the standard hyperbola blocking: group all d that share the same
value v = floor((N-1)/d).  For each block [d_lo, d_hi] with constant v:

    contribution = N*v * sum_{d in block} chi(d)
                 - v*(v+1)/2 * sum_{d in block} d * chi(d)

Both inner sums are computable in O(1) via the prefix sums C(x) and W(x):

    C(x) = sum_{n=1}^{x} chi_{-4}(n)        (period-4 prefix sum)
    W(x) = sum_{n=1}^{x} n * chi_{-4}(n)    (weighted period-4 prefix sum)

The number of distinct values of floor((N-1)/d) is O(sqrt(N)), giving the
overall O(sqrt(N)) complexity.

=============================================================================
ERROR TERM
=============================================================================

b1(N) ~ (pi/2) * N^2

Define  E(N) = b1(N) - (pi/2) * N^2.

Since b1(N) = N * A(N-1) - Sigma1(N-1)  and  A(N) = pi*N + Delta(N),
the error inherits the Gauss circle error Delta(N) = A(N) - pi*N:

    E(N) ~ N * Delta(N-1)  +  lower order

The Gauss circle conjecture is Delta(N) = O(N^{1/2 + eps}).
We therefore expect E(N) = O(N^{3/2 + eps}).

We study E(N) / N^{3/2} for stability, and track sign changes.
"""

import math


# =============================================================================
# Prefix sums for chi_{-4}
# =============================================================================

def C(x):
    """
    C(x) = sum_{n=1}^{x} chi_{-4}(n).
    chi_{-4} has period 4: (1, 0, -1, 0).
    Full period sums to 0, so C(x) in {0, 1} depending on x mod 4.
    """
    if x <= 0:
        return 0
    return (x + 3) // 4 - (x + 1) // 4


def W(x):
    """
    W(x) = sum_{n=1}^{x} n * chi_{-4}(n).
    chi_{-4} period: 1*1 + 2*0 + 3*(-1) + 4*0 = -2 per full period of 4.
    """
    if x <= 0:
        return 0
    q, r = divmod(x, 4)
    total = q * (-2)
    # partial period: n = 4q+1 .. 4q+r, with chi values (1,0,-1,0)
    if r >= 1: total += (4 * q + 1)   # chi = +1
    if r >= 3: total -= (4 * q + 3)   # chi = -1
    return total


# =============================================================================
# O(sqrt(N)) exact formula for b1(N)
# =============================================================================

def b1_sqrt(N):
    """
    Exact O(sqrt(N)) computation of b1(N) via hyperbola blocking.

    b1(N) = 4 * sum_{d=1}^{N-1} chi(d) * [N*M - d*M*(M+1)/2]
    where M = floor((N-1)/d).

    Blocked over d-ranges where floor((N-1)/d) is constant.
    """
    if N <= 1:
        return 0

    P = N - 1   # we need floor(P/d) for d = 1..P
    total = 0
    d = 1
    while d <= P:
        v    = P // d           # v = floor(P/d), constant in this block
        d_hi = P // v           # largest d with floor(P/d) == v
        d_lo = d

        block_C = C(d_hi) - C(d_lo - 1)
        block_W = W(d_hi) - W(d_lo - 1)

        total += N * v * block_C - (v * (v + 1) // 2) * block_W

        d = d_hi + 1

    return 4 * total


# =============================================================================
# Verification helpers (slow -- only used for small N)
# =============================================================================

def chi_m4(n):
    if n % 2 == 0:
        return 0
    return 1 if n % 4 == 1 else -1


def r2(n):
    s = 0
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            s += chi_m4(d)
            if d * d != n:
                s += chi_m4(n // d)
    return 4 * s


def b1_direct(N):
    """Brute-force O(N) via iterating over Gaussian integers."""
    if N <= 1:
        return 0
    target = N - 1
    total  = 0
    R = int(math.isqrt(target))
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            na = a * a + b * b
            if 1 <= na <= target:
                total += N - na
    return total


# =============================================================================
# Phase 1: Verify prefix sums C and W
# =============================================================================

def test_prefix_sums(limit=100):
    print("--- Phase 1: Prefix sum sanity checks ---")
    for x in range(1, limit + 1):
        C_naive = sum(chi_m4(n) for n in range(1, x + 1))
        W_naive = sum(n * chi_m4(n) for n in range(1, x + 1))
        assert C(x) == C_naive, f"C mismatch at x={x}: {C(x)} vs {C_naive}"
        assert W(x) == W_naive, f"W mismatch at x={x}: {W(x)} vs {W_naive}"
    print(f"  [OK] C(x) and W(x) match naive sums for x in [1, {limit}]\n")


# =============================================================================
# Phase 2: Verify b1_sqrt against brute force
# =============================================================================

def test_b1_sqrt(limit=500):
    print(f"--- Phase 2: b1_sqrt vs b1_direct (N = 2..{limit}) ---")
    all_match = True
    for N in range(2, limit + 1):
        bd = b1_direct(N)
        bs = b1_sqrt(N)
        if bd != bs:
            print(f"  MISMATCH at N={N}: direct={bd}, sqrt={bs}")
            all_match = False
    status = "OK" if all_match else "FAIL"
    print(f"  [{status}] All N in [2, {limit}] {'verified' if all_match else 'FAILED'}.\n")
    return all_match


# =============================================================================
# Phase 3: Error term E(N) = b1(N) - (pi/2)*N^2 at large N
# =============================================================================

def analyze_error(N_values):
    print("--- Phase 3: Error term E(N) = b1(N) - (pi/2)*N^2 ---")
    print(f"\n  Expected scaling: E(N) ~ N * Delta(N),  Delta = Gauss circle error")
    print(f"  Gauss circle conjecture => E(N) = O(N^(3/2 + eps))\n")
    print(f"{'N':>12}  {'b1(N)':>22}  {'E(N)':>18}  {'E(N)/N^(3/2)':>16}  {'E(N)/N^(4/3)':>16}")
    print("-" * 92)

    for N in N_values:
        b1  = b1_sqrt(N)
        asym = (math.pi / 2) * N * N
        E   = b1 - asym
        print(f"{N:>12}  {b1:>22}  {E:>18.2f}  {E / N**1.5:>16.8f}  {E / N**(4/3):>16.6f}")

    print()


# =============================================================================
# Phase 4: Sign changes -- scan densely using the fast formula
# =============================================================================

def find_sign_changes(N_max):
    print(f"--- Phase 4: Sign changes in E(N) for N up to {N_max} ---")
    crossings = []
    prev_sign = 0
    for N in range(2, N_max + 1):
        b1   = b1_sqrt(N)
        E    = b1 - (math.pi / 2) * N * N
        sign = 1 if E >= 0 else -1
        if prev_sign != 0 and sign != prev_sign:
            crossings.append((N, E))
        prev_sign = sign

    print(f"  Found {len(crossings)} sign changes.")
    if crossings:
        print(f"  First crossings (N, E):")
        for N, E in crossings[:20]:
            print(f"    N={N:>8}  E={E:.4f}")
    else:
        print("  No sign changes found -- error is persistently one-signed.")
        print("  Consider pushing N_max higher or investigating secondary asymptotic.")
    print()


# =============================================================================
# Phase 5: Ratio E(N) / (N * Delta(N)) -- check the inheritance claim
#
# Delta(N) = A(N) - pi*N where A(N) = sum_{k<=N} r2(k)
# We compute A(N) from r2 sieve for moderate N.
# =============================================================================

def sieve_r2_vals(limit):
    return [0] + [r2(k) for k in range(1, limit + 1)]


def analyze_inheritance(N_values):
    print("--- Phase 5: E(N) / (N * Delta(N-1)) -- testing the inheritance claim ---")
    print(f"\n  If E(N) ~ N * Delta(N-1), ratio should approach a constant.\n")
    print(f"{'N':>10}  {'Delta(N-1)':>14}  {'N*Delta':>14}  {'E(N)':>14}  {'ratio':>10}")
    print("-" * 68)

    N_max = max(N_values)
    r2v   = sieve_r2_vals(N_max)

    for N in N_values:
        A_prev  = sum(r2v[1:N])   # A(N-1)
        Delta   = A_prev - math.pi * (N - 1)
        b1      = b1_sqrt(N)
        E       = b1 - (math.pi / 2) * N * N
        N_Delta = N * Delta
        ratio   = E / N_Delta if N_Delta != 0 else float('inf')
        print(f"{N:>10}  {Delta:>14.4f}  {N_Delta:>14.4f}  {E:>14.4f}  {ratio:>10.6f}")

    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" b1 Hyperbola Formula: O(sqrt(N)) exact computation")
    print(" b1(N) = 4 * sum_{d<N} chi(d) * [N*M - d*M*(M+1)/2],  M=floor((N-1)/d)")
    print("=" * 70)
    print()

    test_prefix_sums(limit=200)
    test_b1_sqrt(limit=500)

    large_N = [1_000, 5_000, 10_000, 50_000, 100_000,
               500_000, 1_000_000, 5_000_000, 10_000_000]
    analyze_error(large_N)

    find_sign_changes(N_max=50_000)

    moderate_N = [100, 200, 500, 1000, 2000, 5000]
    analyze_inheritance(moderate_N)

    print("=" * 70)
    print(" Done.")
    print("=" * 70)

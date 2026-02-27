#!/usr/bin/env python3
"""
verify_identity.py
==================
Verifies the finite-difference identity:

    A(N) = b1(N+1) - b1(N)

where:
  - A(N) = #{(x,y) in Z^2 : x^2 + y^2 <= N, (x,y) != (0,0)}
           (lattice points in disk of radius sqrt(N), excluding origin)
  - b1(N) = sum_{k=1}^{N-1} (N-k) * r2(k)
           (convolution sum; r2(k) = #{(x,y) : x^2+y^2 = k})

This is the "golden ticket" from the Gemini analysis: your O(sqrt N) hyperbola
code implicitly computes the cumulative sum of A(m), so its finite difference
recovers A(N) exactly.  Stripping pi*N then gives the Gauss circle error Delta(N)
directly, with NO +1/2 stationary-phase correction needed for OLS.

Run:
    python3 verify_identity.py [Nmax]

Default Nmax = 5000.  All arithmetic is exact (Python arbitrary precision).
"""

import sys
import math
from fractions import Fraction

# -----------------------------------------------------------------------
# Exact integer arithmetic implementations
# -----------------------------------------------------------------------

def r2(n: int) -> int:
    """Number of ways to write n = x^2 + y^2  (all integer pairs, including 0)."""
    if n < 0:
        return 0
    count = 0
    x = 0
    while x * x <= n:
        rem = n - x * x
        sq = int(math.isqrt(rem))
        if sq * sq == rem:
            count += 2 if sq > 0 else 1  # (x, sq) and (x, -sq) if sq>0
            if x > 0:
                count *= 1  # already counted by symmetry below
        x += 1
    # Faster: use the chi_{-4} Dirichlet series formula r2(n) = 4*(d1(n) - d3(n))
    # where d1 = # divisors ≡1 mod 4, d3 = # divisors ≡3 mod 4.
    # Use that formula for speed.
    count = 0
    for d in range(1, n + 1):
        if n % d == 0:
            if d % 4 == 1:
                count += 1
            elif d % 4 == 3:
                count -= 1
    return 4 * count if n > 0 else 1  # r2(0)=1 for the origin

def r2_fast(n: int) -> int:
    """r2(n) via 4*(d1-d3), fast version."""
    if n == 0:
        return 1
    count = 0
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            q = n // d
            # count both d and q (handle perfect squares once)
            for divisor in ([d, q] if d != q else [d]):
                if divisor % 4 == 1:
                    count += 1
                elif divisor % 4 == 3:
                    count -= 1
    return 4 * count

def b1_direct(N: int) -> int:
    """
    b1(N) = sum_{k=1}^{N-1} (N - k) * r2(k)   [exact, O(N*sqrt(N))]
    Only feasible for small N (used for verification).
    """
    total = 0
    for k in range(1, N):
        total += (N - k) * r2_fast(k)
    return total

def A_direct(N: int) -> int:
    """
    A(N) = #{(x,y) in Z^2 : x^2+y^2 <= N, (x,y)!=(0,0)}
         = sum_{k=1}^{N} r2(k)   [exact brute force, O(N*sqrt(N))]
    """
    return sum(r2_fast(k) for k in range(1, N + 1))

def A_geometry(N: int) -> int:
    """
    A(N) via direct geometric count: iterate x, count valid y.
    O(sqrt(N)) and exact.  The ground truth.
    """
    R = int(math.isqrt(N))
    count = 0
    for x in range(-R, R + 1):
        rem = N - x * x
        if rem < 0:
            continue
        y_max = int(math.isqrt(rem))
        # y in [-y_max, y_max], but exclude (0,0)
        n_y = 2 * y_max + 1
        if x == 0:
            n_y -= 1  # exclude origin
        count += n_y
    return count

# -----------------------------------------------------------------------
# Main verification
# -----------------------------------------------------------------------

def main():
    Nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    print(f"Verifying A(N) = b1(N+1) - b1(N)  for N in [1, {Nmax}]")
    print(f"Also checking A(N) against direct geometric count.")
    print(f"All arithmetic is exact (Python arbitrary precision integers).")
    print()

    # Pre-compute b1 values b1[0..Nmax+1] incrementally.
    # b1(N) = sum_{k=1}^{N-1} (N-k)*r2(k)
    # b1(N+1) - b1(N) = sum_{k=1}^{N} r2(k) = A(N)
    # We build b1 incrementally: b1[N+1] = b1[N] + A_cumsum[N]
    # where A_cumsum[N] = sum_{k=1}^{N} r2(k) grows by r2(N) each step.

    print("Building b1 table incrementally (O(Nmax * sqrt(Nmax)))...")
    b1 = [0] * (Nmax + 2)   # b1[N] = b1(N)
    A_cumsum = 0             # sum_{k=1}^{N} r2(k) = A(N)

    errors_identity = 0
    errors_geometry = 0
    checks = 0

    for N in range(1, Nmax + 1):
        A_cumsum += r2_fast(N)          # A_cumsum is now A(N)
        b1[N + 1] = b1[N] + A_cumsum   # incremental: b1(N+1) = b1(N) + A(N)

    print("Verifying identity and geometry for each N...")
    print()
    print(f"  {'N':>8}  {'A_geom':>12}  {'b1(N+1)-b1(N)':>15}  {'match?':>8}  {'Delta(N)':>12}")
    print("  " + "-" * 65)

    sample_step = max(1, Nmax // 50)  # print ~50 rows
    for N in range(1, Nmax + 1):
        A_fd   = b1[N + 1] - b1[N]   # finite difference
        A_geom = A_geometry(N)        # direct geometric count

        identity_ok = (A_fd == A_geom)
        if not identity_ok:
            errors_identity += 1
        if A_fd != A_geom:
            errors_geometry += 1
        checks += 1

        if N % sample_step == 0 or N <= 20 or not identity_ok:
            delta = A_geom - round(math.pi * N)
            marker = "" if identity_ok else " <-- MISMATCH"
            print(f"  {N:>8}  {A_geom:>12}  {A_fd:>15}  {'YES' if identity_ok else 'NO':>8}  {delta:>12}{marker}")

    print()
    print("=" * 65)
    print(f"Checked {checks} values of N in [1, {Nmax}].")
    if errors_identity == 0:
        print("IDENTITY VERIFIED:  A(N) = b1(N+1) - b1(N)  holds for ALL N tested.")
    else:
        print(f"IDENTITY FAILED for {errors_identity} values!")

    print()
    # Also verify the b1 formula from first principles for a few values
    print("Cross-checking b1 against direct formula for small N:")
    print(f"  {'N':>6}  {'b1_incr':>18}  {'b1_direct':>18}  {'match?':>8}")
    print("  " + "-" * 55)
    for N in [1, 2, 3, 5, 10, 20, 50]:
        if N > Nmax:
            break
        b1_d = b1_direct(N)
        b1_i = b1[N]
        ok = (b1_d == b1_i)
        print(f"  {N:>6}  {b1_i:>18}  {b1_d:>18}  {'YES' if ok else 'NO  <-- MISMATCH':>8}")

    print()
    print("Delta(N) = A(N) - pi*N  (Gauss circle error, OLS gives theta directly):")
    print(f"  {'N':>10}  {'Delta(N)':>14}  {'Delta/N^0.25':>14}  {'Delta/N^0.315':>15}")
    print("  " + "-" * 58)
    import random
    random.seed(42)
    # Sample log-spaced points
    sample_Ns = sorted(set(
        [int(Nmax * (i / 20) ** 2) for i in range(1, 21)] +
        [Nmax]
    ))
    sample_Ns = [n for n in sample_Ns if 1 <= n <= Nmax]
    for N in sample_Ns:
        A_fd = b1[N + 1] - b1[N]
        delta = A_fd - math.pi * N   # float approximation for display
        d025  = delta / (N ** 0.25)
        d0315 = delta / (N ** 0.31490)
        print(f"  {N:>10}  {delta:>14.4f}  {d025:>14.6f}  {d0315:>15.6f}")

    print()
    print("Interpretation:")
    print("  - Delta(N)/N^0.25 bounded => theta_Gauss <= 1/4  (Hardy-Landau conjecture)")
    print("  - Delta(N)/N^0.315 bounded => theta_Gauss <= 131/416  (Huxley proved)")
    print("  - Use ./delta <Nmax> <samples> <cutoff> --delta for large-N OLS on Delta(N)")

if __name__ == "__main__":
    main()

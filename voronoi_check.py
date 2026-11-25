#!/usr/bin/env python3
"""
voronoi_check.py -- validate  err_osc(n) = int_0^n Delta(t) dt  against Voronoi.

Facts being checked
-------------------
  a(n) = n*D(n-1) - sigma_tau(n-1) = sum_{k<n} (n-k) tau(k)          (exact)
  err_osc(n) := a(n) - [n^2/2 log n + (g - 3/4) n^2 + n/4]
             = int_0^n Delta(t) dt                                   (exact)
  int_0^x Delta ~ x^{3/4}/(2 sqrt2 pi^2) * sum_k d(k) k^{-5/4} sin(4 pi sqrt(kx) - pi/4)

The Voronoi series converges absolutely, so err_osc(n) = O(n^{3/4}) with no
hypothesis (this is why the OLS exponent in main.cpp is ~0.25 regardless of RH
or the divisor conjecture).

Usage:  python3 voronoi_check.py [K [n1 n2 ...]]
  K   number of Voronoi terms (default 2_000_000)
  n_i test points (default 1e6+3, 1e8+7, 1e10+1, 1e12+39); exact a(n) costs
      O(sqrt n) pure-Python steps, so keep n <= ~1e13.
Needs: numpy, mpmath.  Exits nonzero if the relative agreement is worse than
the tolerance expected from truncating at K terms.
"""
import sys
import math
import numpy as np
from mpmath import mp, mpf, euler, log


def D_and_sigma_tau(n):
    """D(n)=sum_{m<=n} tau(m) and sigma_tau(n)=sum_{m<=n} m*tau(m), exact, O(sqrt n)."""
    r = math.isqrt(n)
    s = 0
    for k in range(1, r + 1):
        s += n // k
    D = 2 * s - r * r
    ST = 0
    d = 1
    while d <= n:
        q = n // d
        hi = n // q
        ST += (q * (q + 1) // 2) * ((d + hi) * (hi - d + 1) // 2)
        d = hi + 1
    return D, ST


def a_exact(n):
    D, ST = D_and_sigma_tau(n - 1)
    return n * D - ST


def err_osc(n):
    mp.dps = 2 * len(str(n)) + 30
    x = mpf(n)
    asym = x * x / 2 * log(x) + (euler - mpf(3) / 4) * x * x + x / 4
    return a_exact(n) - asym


def tau_sieve(K):
    tau = np.zeros(K + 1, dtype=np.float64)
    for d in range(1, K + 1):
        tau[d::d] += 1.0
    return tau


def voronoi(n, tau, K):
    k = np.arange(1, K + 1, dtype=np.float64)
    phase = 4.0 * np.pi * np.sqrt(k * float(n)) - np.pi / 4.0
    # phase reaches ~1e10..1e11 for n ~ 1e12..1e14: float64 abs error ~1e-5, fine.
    s = float(np.sum(tau[1:] * k ** -1.25 * np.sin(phase)))
    return n ** 0.75 / (2.0 * math.sqrt(2.0) * math.pi ** 2) * s


def main():
    args = sys.argv[1:]
    K = int(float(args[0])) if args else 2_000_000
    ns = [int(float(a)) for a in args[1:]] or [10**6 + 3, 10**8 + 7, 10**10 + 1, 10**12 + 39]
    for n in ns:
        if n > 3 * 10**13:
            sys.exit("n too large for exact pure-Python evaluation (use n <= 3e13)")
    print(f"sieving tau up to K={K} ...", flush=True)
    tau = tau_sieve(K)
    # truncation error of the Voronoi sum is ~ log(K) K^{-1/4} times a random-phase factor
    amp = 1.0 / (2.0 * math.sqrt(2.0) * math.pi ** 2)
    tol = 5.0 * amp * math.log(K) * K ** -0.25 * 0.3   # in units of n^{3/4}
    print(f"{'n':>16} {'err_osc/n^.75':>15} {'voronoi/n^.75':>15} {'diff/n^.75':>12}   tol")
    bad = 0
    for n in ns:
        e = float(err_osc(n))
        v = voronoi(n, tau, K)
        diff = (e - v) / n ** 0.75
        flag = "" if abs(diff) <= tol else "   <-- OUT OF TOLERANCE"
        if flag:
            bad += 1
        print(f"{n:>16} {e / n ** 0.75:>15.6f} {v / n ** 0.75:>15.6f} {diff:>12.6f}   {tol:.4f}{flag}")
    if bad:
        sys.exit(f"{bad} point(s) outside tolerance: identity or sieve is wrong")
    print("OK: err_osc(n) matches the Voronoi series for the integrated divisor error.")


if __name__ == "__main__":
    main()

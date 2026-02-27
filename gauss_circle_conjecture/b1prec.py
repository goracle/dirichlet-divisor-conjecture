#!/usr/bin/env python3
"""
b1_hyperbola_highprec.py

Same hyperbola b1_sqrt() as before, but compute E(N) relative to
(pi/2)*N*(N-1) using high precision arithmetic (mpmath) and compare
E(N) with S(N) = sum_{m=1}^{N-1} Delta(m).
"""

import math
import sys
from collections import namedtuple

# high-precision floats
try:
    import mpmath as mp
except ImportError:
    mp = None
    print("Warning: mpmath not found; install 'mpmath' for high-precision arithmetic.", file=sys.stderr)

# Set precision (decimal places)
MP_DPS = 80
if mp:
    mp.mp.dps = MP_DPS

# ---------------------------------------------------------------------
# Prefix sums for chi_{-4}
# ---------------------------------------------------------------------
def C(x):
    if x <= 0:
        return 0
    return (x + 3) // 4 - (x + 1) // 4

def W(x):
    if x <= 0:
        return 0
    q, r = divmod(x, 4)
    total = q * (-2)
    if r >= 1:
        total += (4 * q + 1)
    if r >= 3:
        total -= (4 * q + 3)
    return total

# ---------------------------------------------------------------------
# b1 via hyperbola blocking (exact integer)
# ---------------------------------------------------------------------
def b1_sqrt(N):
    if N <= 1:
        return 0
    P = N - 1
    total = 0
    d = 1
    while d <= P:
        v = P // d
        d_hi = P // v
        block_C = C(d_hi) - C(d - 1)
        block_W = W(d_hi) - W(d - 1)
        total += N * v * block_C - (v * (v + 1) // 2) * block_W
        d = d_hi + 1
    return 4 * total

# ---------------------------------------------------------------------
# A(N) via divisor-sum using the same prefix-sums (O(sqrt N))
# A(N) = 4 * sum_{d<=N} chi(d) * floor(N/d)
# ---------------------------------------------------------------------
def A_via_divisor_sum(N):
    if N <= 0:
        return 0
    total = 0
    d = 1
    while d <= N:
        t = N // d
        d_hi = N // t
        block_C = C(d_hi) - C(d - 1)
        total += 4 * block_C * t
        d = d_hi + 1
    return total

# -----------------------
# high-precision asymptotic and E computation
# -----------------------
def asym_det_mp(N):
    # asymptotic deterministic two-term: (pi/2) * N * (N-1)
    if mp is None:
        return float(math.pi/2.0 * N * (N-1))
    return (mp.pi/2) * mp.mpf(N) * (mp.mpf(N) - 1)

def compute_E_highprec(N, b1_int):
    """
    Return E = b1 - (pi/2)*N*(N-1) computed in high precision (mpmath).
    We compute integer part floor(asym) using mp and subtract via integers
    to avoid catastrophic cancellation.
    """
    if mp is None:
        # fallback, lower precision
        asym = math.pi/2.0 * N * (N-1)
        return b1_int - asym
    asym = asym_det_mp(N)
    ai = mp.floor(asym)   # integer part as mp
    afrac = asym - ai     # fractional part
    # idiff = b1 - ai  (exact integer difference)
    idiff = mp.mpf(b1_int - int(ai))
    E = idiff - afrac
    return E

# -----------------------
# Sum S(N) = sum_{m=1}^{N-1} Delta(m)
# Delta(m) = A(m) - pi*m  (use mp for pi*m)
# -----------------------
def S_via_A_prefix(N):
    # compute S(N) = sum_{m=1}^{N-1} (A(m) - pi*m)
    # caution: this is O(N sqrt N) if naive; use for moderate N only.
    s = mp.mpf('0') if mp else 0.0
    for m in range(1, N):
        A_m = A_via_divisor_sum(m)
        if mp:
            s += mp.mpf(A_m) - mp.pi * mp.mpf(m)
        else:
            s += float(A_m) - math.pi * m
    return s

# -----------------------
# OLS fit log|E| = alpha log N + C
# -----------------------
def ols_log_fit(points):
    # points: list of (N, E) with E as mp or float
    xs = []
    ys = []
    for N, E in points:
        if E == 0:
            continue
        absE = abs(E)
        if mp:
            if absE <= mp.mpf('0'):
                continue
            xs.append(float(mp.log(mp.mpf(N))))
            ys.append(float(mp.log(absE)))
        else:
            xs.append(math.log(N))
            ys.append(math.log(absE))
    if len(xs) < 3:
        return None, None
    import numpy as np
    X = np.vstack([np.ones(len(xs)), xs]).T
    y = np.array(ys)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    C = beta[0]; alpha = beta[1]
    theta = alpha - 1.0
    return alpha, theta

# -----------------------
# Driver: compute grid, show comparisons
# -----------------------
def main():
    # sample grid (edit as needed)
    Ns = [10**3, 5*10**3, 10**4, 5*10**4, 10**5, 5*10**5, 10**6]
    print("mpmath available:", mp is not None)
    if mp:
        print("mp.dps =", mp.mp.dps)

    rows = []
    for N in Ns:
        b1 = b1_sqrt(N)
        E = compute_E_highprec(N, b1)
        # compute S(N) by summation of Delta (only moderate N)
        if N <= 2000000 and mp:   # only if we can afford it
            S = S_via_A_prefix(N)
        else:
            S = None

        # scaled versions
        if mp:
            s125 = E / (mp.mpf(N) ** mp.mpf(1.25))
            s_theta_hux = E / (mp.mpf(N) ** (1.0 + mp.mpf(131) / 416.0))
        else:
            s125 = E / (N ** 1.25)
            s_theta_hux = E / (N ** (1.0 + 131.0/416.0))

        rows.append((N, b1, E, S, s125, s_theta_hux))

    # Print
    print()
    hdr = ("N", "E (b1 - pi/2*N*(N-1))", "E/N^1.25", "E/N^(1+131/416)", "S(N) [cum Delta]")
    print("{:>12s}  {:>24s}  {:>12s}  {:>12s}  {:>16s}".format(*hdr))
    for (N, b1, E, S, s125, sH) in rows:
        if mp:
            Estr = mp.nstr(E, 15)
            s125s = mp.nstr(s125, 10)
            sHs = mp.nstr(sH, 10)
            Sstr = mp.nstr(S, 12) if S is not None else "-"
        else:
            Estr = f"{E:.6g}"
            s125s = f"{s125:.6g}"
            sHs = f"{sH:.6g}"
            Sstr = "-" if S is None else f"{S:.6g}"
        print(f"{N:12d}  {Estr:>24s}  {s125s:>12s}  {sHs:>12s}  {Sstr:>16s}")

    # OLS fit
    points = [(N, rows[i][2]) for i, N in enumerate(Ns)]
    try:
        alpha, theta = ols_log_fit(points)
        print("\nOLS slope alpha, theta = alpha - 1:")
        print("  alpha = ", alpha, " theta = ", theta)
    except Exception as e:
        print("OLS failed:", e)

if __name__ == "__main__":
    main()

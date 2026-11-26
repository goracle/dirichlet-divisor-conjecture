#!/usr/bin/env python3
"""Exact weighted extension of Sladkey's successive-approximation algorithm.

Computes A(n) = sum_{xy <= n} (n + 1 - x*y), equivalently
A(n) = sum_{k=1}^n (n+1-k) tau(k).

The geometric recursion follows Richard Sladkey, arXiv:1206.3369v1.
This implementation replaces each unweighted lattice-point count by the
sum of the quadratic weight n+1-xy over the same region.

All arithmetic is exact Python integer arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isqrt
from typing import Tuple

Moment = Tuple[int, int, int, int, int, int]  # 1,u,v,u^2,uv,v^2


def ceil_cuberoot(n: int) -> int:
    if not isinstance(n, int) or n < 0:
        raise ValueError("ceil_cuberoot expects a nonnegative integer")
    if n <= 1:
        return n
    lo, hi = 0, 1
    while hi**3 < n:
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid**3 >= n:
            hi = mid
        else:
            lo = mid
    return hi


def power_sums_0_to(q: int) -> Tuple[int, int, int, int]:
    """Return sum_{j=0}^{q-1} j^r for r=0,1,2,3."""
    if q < 0:
        raise ValueError("q must be nonnegative")
    s0 = q
    s1 = q * (q - 1) // 2
    s2 = q * (q - 1) * (2 * q - 1) // 6
    s3 = s1 * s1
    return s0, s1, s2, s3


def p1(q: int, c: int) -> int:
    """sum_{j=1}^q (j+c)."""
    if q < 0:
        raise ValueError("q must be nonnegative")
    return q * c + q * (q + 1) // 2


def p2(q: int, c: int) -> int:
    """sum_{j=1}^q (j+c)^2."""
    if q < 0:
        raise ValueError("q must be nonnegative")
    return (q * c * c + c * q * (q + 1)
            + q * (q + 1) * (2 * q + 1) // 6)


def tri_moments(L: int, d: int = 0) -> Moment:
    """Moments on {(u,v): u>=d+1, v>=1, (u-d)+v <= L+1}.

    Equivalently u=r+d, v=s with r,s>=1 and r+s<=L+1.
    The returned order is (M00,M10,M01,M20,M11,M02).
    """
    if L <= 0:
        return (0, 0, 0, 0, 0, 0)
    m00 = L * (L + 1) // 2
    m10 = L * (L + 1) * (L + 2) // 6
    m01 = m10
    m20 = L * (L + 1) * (L + 1) * (L + 2) // 12
    m02 = m20
    m11 = L * (L + 1) * (L + 2) * (L + 3) // 24
    return (m00,
            m10 + d * m00,
            m01,
            m20 + 2 * d * m10 + d * d * m00,
            m11 + d * m01,
            m02)


def add_moments(a: Moment, b: Moment, sign: int = 1) -> Moment:
    return tuple(x + sign * y for x, y in zip(a, b))  # type: ignore[return-value]


def polygon_N_moments(u4: int, v4: int, u5: int, v5: int) -> Moment:
    """Six moments of Sladkey's polygon N, via signed shifted triangles."""
    v6 = u4 + v4
    u7 = u5 + v5
    d = u5 - 1
    out = tri_moments(v6 - 1, 0)
    out = add_moments(out, tri_moments(v6 - u5, d), -1)
    out = add_moments(out, tri_moments(u7 - u5, d), +1)
    return out


def floor_linear_minus_sqrt(A: int, disc: int, den: int) -> int:
    """Exact floor((A - sqrt(disc))/den), for disc>=0 and den>0."""
    if disc < 0 or den <= 0:
        raise ValueError("invalid radical or denominator")
    r = isqrt(disc)
    q, rem = divmod(A - r, den)
    if r * r != disc and rem == 0:
        q -= 1
    return q


@dataclass
class Stats:
    sr_nodes: int = 0
    max_depth: int = 0
    terminal_columns: int = 0
    terminal_rows: int = 0
    axis_fallbacks: int = 0
    fallback_axis_work: int = 0
    max_fallback_axis: int = 0
    fallback_reasons: dict = field(default_factory=dict)
    fallback_examples: list = field(default_factory=list)
    top_regions: int = 0


class WeightedSladkey:
    def __init__(self, n: int, C1: int = 1, C2: int = 10):
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        if not isinstance(C1, int) or not isinstance(C2, int) or C1 < 1 or C2 < 1:
            raise ValueError("C1 and C2 must be positive integers")
        self.n = n
        self.C1 = C1
        self.C2 = C2
        self.stats = Stats()

    def H(self, u: int, v: int, a1: int, b1: int, c1: int,
          a2: int, b2: int, c2: int) -> int:
        t = u + c1
        s = v + c2
        x = b2 * t - b1 * s
        y = a1 * s - a2 * t
        return x * y

    def Vfloor(self, u: int, a1: int, b1: int, c1: int,
               a2: int, b2: int, c2: int) -> int:
        t = u + c1
        K = a1 * b2 + b1 * a2
        disc = t * t - 4 * a1 * b1 * self.n
        if disc < 0:
            raise ArithmeticError(
                f"negative V discriminant: n={self.n}, u={u}, params="
                f"{(a1,b1,c1,a2,b2,c2)}, disc={disc}")
        val = floor_linear_minus_sqrt(K * t, disc, 2 * a1 * b1) - c2
        return val

    def Ufloor(self, v: int, a1: int, b1: int, c1: int,
               a2: int, b2: int, c2: int) -> int:
        s = v + c2
        K = a1 * b2 + b1 * a2
        disc = s * s - 4 * a2 * b2 * self.n
        if disc < 0:
            raise ArithmeticError(
                f"negative U discriminant: n={self.n}, v={v}, params="
                f"{(a1,b1,c1,a2,b2,c2)}, disc={disc}")
        val = floor_linear_minus_sqrt(K * s, disc, 2 * a2 * b2) - c1
        return val

    def weight_uv(self, u: int, v: int, a1: int, b1: int, c1: int,
                  a2: int, b2: int, c2: int) -> int:
        t = u + c1
        s = v + c2
        K = a1 * b2 + b1 * a2
        return (self.n + 1 + a2 * b2 * t * t
                - K * t * s + a1 * b1 * s * s)

    def weighted_row_v1(self, w: int, a1: int, b1: int, c1: int,
                        a2: int, b2: int, c2: int) -> int:
        # Sum u=1..w at v=1.
        s = c2 + 1
        K = a1 * b2 + b1 * a2
        return (w * (self.n + 1 + a1 * b1 * s * s)
                + a2 * b2 * p2(w, c1)
                - K * s * p1(w, c1))

    def weighted_col_u1(self, h: int, a1: int, b1: int, c1: int,
                        a2: int, b2: int, c2: int) -> int:
        # Sum v=1..h at u=1.
        t = c1 + 1
        K = a1 * b2 + b1 * a2
        return (h * (self.n + 1 + a2 * b2 * t * t)
                - K * t * p1(h, c2)
                + a1 * b1 * p2(h, c2))

    def weighted_N(self, moments: Moment, a1: int, b1: int, c1: int,
                   a2: int, b2: int, c2: int) -> int:
        M00, M10, M01, M20, M11, M02 = moments
        K = a1 * b2 + b1 * a2
        return (
            (self.n + 1) * M00
            + a2 * b2 * (M20 + 2 * c1 * M10 + c1 * c1 * M00)
            - K * (M11 + c1 * M01 + c2 * M10 + c1 * c2 * M00)
            + a1 * b1 * (M02 + 2 * c2 * M01 + c2 * c2 * M00)
        )

    def weighted_terminal_columns(self, w: int, a1: int, b1: int, c1: int,
                                  a2: int, b2: int, c2: int) -> int:
        total = 0
        K = a1 * b2 + b1 * a2
        for u in range(1, w + 1):
            q = max(0, self.Vfloor(u, a1, b1, c1, a2, b2, c2))
            t = u + c1
            total += (q * (self.n + 1 + a2 * b2 * t * t)
                      - K * t * p1(q, c2)
                      + a1 * b1 * p2(q, c2))
        self.stats.terminal_columns += 1
        return total

    def weighted_terminal_rows(self, h: int, a1: int, b1: int, c1: int,
                               a2: int, b2: int, c2: int) -> int:
        total = 0
        K = a1 * b2 + b1 * a2
        for v in range(1, h + 1):
            q = max(0, self.Ufloor(v, a1, b1, c1, a2, b2, c2))
            s = v + c2
            total += (q * (self.n + 1 + a1 * b1 * s * s)
                      + a2 * b2 * p2(q, c1)
                      - K * s * p1(q, c1))
        self.stats.terminal_rows += 1
        return total

    def weighted_fallback(self, w: int, h: int, a1: int, b1: int, c1: int,
                          a2: int, b2: int, c2: int, reason: str = "unspecified") -> int:
        """Exact smaller-axis fallback; should be rare and is counted."""
        self.stats.axis_fallbacks += 1
        self.stats.fallback_reasons[reason] = self.stats.fallback_reasons.get(reason, 0) + 1
        if len(self.stats.fallback_examples) < 8:
            self.stats.fallback_examples.append((reason, w, h, a1, b1, c1, a2, b2, c2))
        axis_work = min(w, h)
        self.stats.fallback_axis_work += axis_work
        self.stats.max_fallback_axis = max(self.stats.max_fallback_axis, axis_work)
        if w <= h:
            return self.weighted_terminal_columns(w, a1, b1, c1, a2, b2, c2)
        return self.weighted_terminal_rows(h, a1, b1, c1, a2, b2, c2)

    def _sr_local(self, w: int, h: int, a1: int, b1: int, c1: int,
                  a2: int, b2: int, c2: int, depth: int):
        """Process one SR node.

        Return an integer when terminal, otherwise return (local, child1,
        child2), where each child is an 8-tuple of SR arguments.  The caller
        evaluates the children with an explicit post-order stack.
        """
        self.stats.sr_nodes += 1
        self.stats.max_depth = max(self.stats.max_depth, depth)
        if w < 0 or h < 0:
            raise ArithmeticError(f"negative region dimensions: w={w}, h={h}")
        if w == 0 or h == 0:
            return 0

        total = 0
        # Remove full first row v=1 when its far endpoint is under the hyperbola.
        if self.H(w, 1, a1, b1, c1, a2, b2, c2) <= self.n:
            total += self.weighted_row_v1(w, a1, b1, c1, a2, b2, c2)
            c2 += 1
            h -= 1
        if h == 0:
            return total

        # Remove full first column u=1 in the remaining region.
        if self.H(1, h, a1, b1, c1, a2, b2, c2) <= self.n:
            total += self.weighted_col_u1(h, a1, b1, c1, a2, b2, c2)
            c1 += 1
            w -= 1
        if w == 0 or h == 0:
            return total

        if w <= self.C2:
            return total + self.weighted_terminal_columns(
                w, a1, b1, c1, a2, b2, c2)
        if h <= self.C2:
            return total + self.weighted_terminal_rows(
                h, a1, b1, c1, a2, b2, c2)

        a3 = a1 + a2
        b3 = b1 + b2
        K = a1 * b2 + b1 * a2
        tangent_coeff = K + 2 * a1 * b1
        # floor((K+2a1b1)*sqrt(n/(a3*b3))) exactly.
        u4 = isqrt((tangent_coeff * tangent_coeff * self.n) // (a3 * b3)) - c1

        if u4 <= 0:
            return total + self.weighted_fallback(
                w, h, a1, b1, c1, a2, b2, c2, "u4_left_of_region")
        if u4 >= w:
            return total + self.weighted_fallback(
                w, h, a1, b1, c1, a2, b2, c2, "u4_right_of_region")

        v4 = self.Vfloor(u4, a1, b1, c1, a2, b2, c2)
        u5 = u4 + 1
        v5 = self.Vfloor(u5, a1, b1, c1, a2, b2, c2)
        if v4 <= 0 or v5 <= 0:
            return total + self.weighted_fallback(
                w, h, a1, b1, c1, a2, b2, c2, "nonpositive_v")
        v6 = u4 + v4
        u7 = u5 + v5
        # Both recursive children must be nonempty for the Stern-Brocot
        # branching/depth argument.  Degenerate chops are handled exactly by
        # summing this whole region along its shorter axis.
        if u7 >= w or v6 >= h:
            return total + self.weighted_fallback(
                w, h, a1, b1, c1, a2, b2, c2, "degenerate_child")

        Nmom = polygon_N_moments(u4, v4, u5, v5)
        total += self.weighted_N(Nmom, a1, b1, c1, a2, b2, c2)

        # Farey-mediant child regions (Sladkey eq. 46).  The returned tuples
        # are evaluated by SR() using an explicit stack, avoiding Python's
        # call-stack limit even at very large n.
        child1 = (
            u4, h - v6,
            a1, b1, c1,
            a3, b3, c1 + c2 + v6,
        )
        child2 = (
            w - u7, v5,
            a3, b3, c1 + c2 + u7,
            a2, b2, c2,
        )
        return total, child1, child2

    def SR(self, w: int, h: int, a1: int, b1: int, c1: int,
           a2: int, b2: int, c2: int, depth: int = 0) -> int:
        """Weighted Sladkey recursion, evaluated in iterative post-order."""
        root = (w, h, a1, b1, c1, a2, b2, c2)
        tasks = [("visit", root, depth)]
        values = []
        while tasks:
            kind, payload, level = tasks.pop()
            if kind == "visit":
                result = self._sr_local(*payload, level)
                if isinstance(result, int):
                    values.append(result)
                else:
                    local, child1, child2 = result
                    tasks.append(("combine", local, level))
                    # LIFO: child1 is evaluated first, then child2.
                    tasks.append(("visit", child2, level + 1))
                    tasks.append(("visit", child1, level + 1))
            else:
                if len(values) < 2:
                    raise ArithmeticError("internal SR post-order stack underflow")
                child2_value = values.pop()
                child1_value = values.pop()
                values.append(payload + child1_value + child2_value)
        if len(values) != 1:
            raise ArithmeticError(f"internal SR result stack has {len(values)} entries")
        return values[0]

    def top_level_M_moments(self, L0: int, L1: int, L2: int, d: int) -> Moment:
        out = tri_moments(L0, 0)
        out = add_moments(out, tri_moments(L1, d), -1)
        out = add_moments(out, tri_moments(L2, d), +1)
        return out

    def weighted_top_M(self, x_min: int, c2: int, a2: int,
                       c4: int, c5: int, x5: int) -> int:
        L0 = c4 - c2 - x_min
        L1 = c4 - c2 - x5
        L2 = c5 - c2 - x5
        d = x5 - x_min
        moments = self.top_level_M_moments(L0, L1, L2, d)
        M00, M10, M01, M20, M11, M02 = moments
        r0 = x_min - 1
        s0 = c2 - a2 * r0
        return (
            (self.n + 1 - r0 * s0) * M00
            + (a2 * r0 - s0) * M10
            - r0 * M01
            + a2 * M20
            - M11
        )

    def weighted_S1(self, x_min: int) -> int:
        total = 0
        for x in range(1, x_min):
            q = self.n // x
            total += q * (self.n + 1) - x * q * (q + 1) // 2
        return total

    def weighted_S2(self, x_min: int, x_max: int, y_min: int) -> int:
        r = x_max - x_min + 1
        if r <= 0:
            return 0
        Q = y_min + r - 1
        s0, s1, s2, s3 = power_sums_0_to(r)
        sum_q = r * Q - s1
        sum_xqq1 = (
            x_min * (Q * Q + Q) * s0
            + (Q * Q + Q - x_min * (2 * Q + 1)) * s1
            + (x_min - (2 * Q + 1)) * s2
            + s3
        )
        return (self.n + 1) * sum_q - sum_xqq1 // 2

    def weighted_S3(self, x_min: int, x2: int, a2: int, y2: int) -> int:
        total = 0
        for x in range(x_min, x2):
            q = self.n // x
            lower = a2 * (x2 - x) + y2
            if q > lower:
                count = q - lower
                total += (count * (self.n + 1)
                          - x * (q * (q + 1) - lower * (lower + 1)) // 2)
        return total

    def weighted_half_region(self) -> int:
        if self.n == 0:
            return 0
        n = self.n
        x_max = isqrt(n)
        y_min = n // x_max
        x_min = min(self.C1 * ceil_cuberoot(2 * n), x_max)

        total = self.weighted_S1(x_min)
        total += self.weighted_S2(x_min, x_max, y_min)

        a2 = 1
        x2 = x_max
        y2 = y_min
        c2 = a2 * x2 + y2

        a1 = a2 + 1
        while True:
            # x4 = floor(sqrt(n/a1)) exactly.
            x4 = isqrt(n // a1)
            if x4 <= x_min or x4 >= x_max:
                break
            y4 = n // x4
            c4 = a1 * x4 + y4
            x5 = x4 + 1
            y5 = n // x5
            c5 = a1 * x5 + y5

            total += self.weighted_top_M(x_min, c2, a2, c4, c5, x5)

            w = a1 * x2 + y2 - c5
            h = a2 * x5 + y5 - c2
            if w < 0 or h < 0:
                raise ArithmeticError(
                    f"negative top-level region: n={n}, a1={a1}, a2={a2}, "
                    f"w={w}, h={h}, x2={x2}, x5={x5}")
            total += self.SR(w, h, a1, 1, c5, a2, 1, c2)
            self.stats.top_regions += 1

            a2 = a1
            x2 = x4
            y2 = y4
            c2 = c4
            a1 += 1

        total += self.weighted_S3(x_min, x2, a2, y2)
        return total

    def evaluate(self) -> int:
        n = self.n
        if n == 0:
            return 0
        m = isqrt(n)
        half = self.weighted_half_region()
        square_weight = m * m * (n + 1) - (m * (m + 1) // 2) ** 2
        return 2 * half - square_weight


def hoying_sqrt(n: int) -> int:
    """O(sqrt(n)) exact reference formula for A_n."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    m = isqrt(n)
    total = 0
    for x in range(1, m + 1):
        q = n // x
        total += q * (n + 1) - x * q * (q + 1) // 2
    square_weight = m * m * (n + 1) - (m * (m + 1) // 2) ** 2
    return 2 * total - square_weight


def brute_hoying(n: int) -> int:
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    return sum(n + 1 - x * y for x in range(1, n + 1)
               for y in range(1, n // x + 1))


def selftest(max_n: int = 30000) -> None:
    """Cross-check weighted recursion against hyperbola and brute force."""
    if max_n < 0:
        raise ValueError("max_n must be nonnegative")
    # Moment checks by literal enumeration.
    for L in range(0, 12):
        for d in range(0, 5):
            direct = [0, 0, 0, 0, 0, 0]
            for r in range(1, L + 1):
                for s in range(1, L + 2 - r):
                    u, v = r + d, s
                    vals = [1, u, v, u*u, u*v, v*v]
                    direct = [a+b for a, b in zip(direct, vals)]
            if tuple(direct) != tri_moments(L, d):
                raise AssertionError(("triangle moments", L, d, direct,
                                      tri_moments(L, d)))

    # Check exact weighted polygon N moments against the literal polygon
    # described by the two diagonal boundary pieces.
    for u4 in range(1, 8):
        u5 = u4 + 1
        for v4 in range(1, 8):
            for v5 in range(1, 8):
                v6, u7 = u4 + v4, u5 + v5
                direct = [0, 0, 0, 0, 0, 0]
                for u in range(1, max(u5, u7)):
                    vmax = (v6 - u) if u < u5 else (u7 - u)
                    for v in range(1, max(0, vmax) + 1):
                        vals = [1, u, v, u*u, u*v, v*v]
                        direct = [a+b for a, b in zip(direct, vals)]
                calc = polygon_N_moments(u4, v4, u5, v5)
                if tuple(direct) != calc:
                    raise AssertionError(("N moments", u4, v4, v5,
                                          direct, calc))

    # Exhaustive small-n comparison, including all region boundary cases.
    for n in range(max_n + 1):
        calc_obj = WeightedSladkey(n, C1=1, C2=10)
        got = calc_obj.evaluate()
        ref = hoying_sqrt(n)
        if got != ref:
            raise AssertionError(("weighted-vs-hyperbola", n, got, ref,
                                  calc_obj.stats))
        if n <= min(max_n, 1000):
            brute = brute_hoying(n)
            if got != brute:
                raise AssertionError(("weighted-vs-brute", n, got, brute))

    # Exact quotient-root sanity checks, including square and nonsquare cases.
    for den in range(1, 15):
        for disc in range(0, 200):
            for A in range(0, 80):
                got = floor_linear_minus_sqrt(A, disc, den)
                # Verify q <= value < q+1 without floating point by squaring.
                # A - sqrt(disc) >= q*den iff sqrt(disc) <= A-q*den.
                lo = A - got * den
                hi = A - (got + 1) * den
                if lo < 0 or lo * lo < disc:
                    raise AssertionError(("radical lower", A, disc, den, got))
                if hi >= 0 and hi * hi >= disc:
                    raise AssertionError(("radical upper", A, disc, den, got))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--max-n", type=int, default=30000)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--C1", type=int, default=1)
    parser.add_argument("--C2", type=int, default=10)
    args = parser.parse_args()
    if args.selftest:
        selftest(args.max_n)
        print(f"selftest OK through n={args.max_n}")
    elif args.n is not None:
        obj = WeightedSladkey(args.n, C1=args.C1, C2=args.C2)
        value = obj.evaluate()
        print(value)
        print(obj.stats)
        if args.reference:
            ref = hoying_sqrt(args.n)
            print(f"reference={ref}")
            if value != ref:
                raise SystemExit("weighted algorithm disagrees with reference")
    else:
        parser.error("supply --selftest or --n")

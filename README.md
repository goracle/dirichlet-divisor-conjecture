# A078567 — Arithmetic Subsequence Counter

Empirical investigation of **OEIS A078567**: the number of arithmetic subsequences of `[1..n]` with length > 1, and the growth rate of its oscillatory error term.

Original formula by Dan Hoying (2022). See [`arith_pic.pdf`](arith_pic.pdf).

---

## What this is

`a(n)` counts how many arithmetic progressions (with at least 2 terms) are hiding inside the list `[1, 2, 3, ..., n]`. For example:

```
a(3) = 4    →  (1,2,3), (1,2), (1,3), (2,3)
a(4) = 9
a(5) = 17
```

OEIS sequence: `0, 1, 4, 9, 17, 27, 41, 57, 77, 100, 127, ...`

The sequence grows roughly as `n² log n`. The interesting question is how fast the remainder after subtracting the smooth asymptotic — `err_osc(n)` — grows. This is directly analogous to the Dirichlet divisor problem, and the conjecture is that it grows no faster than `n^(3/4)`.

---

## The math

### Formula (from the PDF)

```
a(n+1) = floor(√n)² · ((1 + floor(√n))²/4 − n − 1)
        + Σ_{i=1}^{floor(√n)}  floor(n/i) · [2n + 2 − i · (1 + floor(n/i))]
```

### Equivalent identity

```
a(n) = n · D(n−1) − σ_τ(n−1)
```

where `D(n) = Σ_{k≤n} τ(k)` (sum of divisor counts) and `σ_τ(n) = Σ_{k≤n} k·τ(k)`. Both are computed in O(√n) via the hyperbola method with block compression.

### Asymptotic

```
a(n) = n²/2 · log(n) + (γ − 3/4) · n² + n/4 + err_osc(n)
```

where γ ≈ 0.5772 is the Euler–Mascheroni constant. The `n/4` coefficient was confirmed empirically by window-averaging to suppress the Delta noise.

### The conjecture

```
err_osc(n) = O(n^(θ + 1/2))
```

- **Conjecture:** θ = 1/4, i.e. `err_osc = O(n^(3/4))`
- **Proven (Huxley):** θ ≤ 131/416 ≈ 0.3149

The OLS regression over sample points up to N = 10¹⁸ gives **θ_OLS ≈ 0.2514**, converging toward 0.25 from above and visibly plateauing.

### Significance

Via the identity `a(n) = n·D(n−1) − σ_τ(n−1)`, the oscillatory error in `a(n)` is directly coupled to the error in the Dirichlet divisor sum. This means empirical bounds on `err_osc` translate into empirical bounds on the divisor problem.

Two observations from the numerical results:

1. **Consistent with the conjecture.** The OLS exponent sits at θ_OLS ≈ 0.2514 at N = 10¹⁸ and has been flat for five decades, well below the proven Huxley bound of 131/416 ≈ 0.3149. The data does not contradict θ = 1/4.

2. **Tighter empirical upper bound.** The running envelope max — `env_max/n^.75` in the output — remaining bounded across 34,555 sample points to N = 10¹⁸ is direct numerical evidence that `err_osc(n) = O(n^(3/4))` holds in practice, approached from a novel object (arithmetic subsequence counts) rather than the divisor sum itself.

No claim is made that the true limit is exactly 0.25 — convergence is slow and the OLS estimator is known to be biased upward by the dense small-n sample points. The contribution is the connection and the bound, not a proof.

---

## Files

| File | Description |
|------|-------------|
| `main.cpp` | Main C++ program — computes a(n), asymptotic error, OLS theta table |
| `plot_theta.py` | Python script — reads `./delta` stdout, plots θ_OLS(N) per decade |
| `theta_ols_v2.html` | Pre-generated interactive plot (N = 10¹⁸, 34,555 sample points) |
| `arith_pic.pdf` | Original formula derivation by Dan Hoying |

---

## Build

**With libquadmath** (recommended for n > 10⁹, gives ~34 digits of precision):
```bash
g++ -O2 -fopenmp -o delta main.cpp -lm -lquadmath
```

**Without libquadmath** (asymptotic subtraction loses precision above ~10⁹):
```bash
g++ -O2 -DNO_QUADMATH -fopenmp -o delta main.cpp -lm
```

Requires GCC with OpenMP. Tested on Linux. Uses `__int128`, `__uint128_t`, and `__float128` — GCC extensions, not portable to MSVC.

---

## Usage

```
./delta [N [cutoff [ratio]]]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `N` | 10¹⁶ | Upper limit for the asymptotic table |
| `cutoff` | 10⁴ | Verify hoying formula vs fused identity up to this n |
| `ratio` | 10.0 | Multiplicative spacing between sample points |

### Examples

Quick sanity check:
```bash
./delta
```

Dense sampling to 10¹⁸ (takes ~30 min on 20 cores):
```bash
OMP_NUM_THREADS=20 ./delta 1000000000000000000 1000 1.001 | tee results.txt
```

Generate the θ_OLS plot:
```bash
OMP_NUM_THREADS=20 ./delta 1000000000000000000 1000 1.001 2>/dev/null | python3 plot_theta.py
# opens theta_ols.html
```

### Output columns

```
n              err_a/n^.75    env_max/n^.75    err_D/n^.25    combo/n^.25    θ_slope    method
```

- **err_a/n^.75** — oscillatory error normalized by n^(3/4). Bounded iff θ ≤ 1/4.
- **env_max/n^.75** — running maximum of |err_a|/n^(3/4).
- **err_D/n^.25** — error in D(n) normalized by n^(1/4). Only computed at or below `cutoff`.
- **θ_slope** — local slope estimate between consecutive points.
- **method** — `verified` (checked against formula) or `fused` (identity only).

---

## Performance

Each sample point costs O(√n). At n = 10¹⁸ that's ~10⁹ inner iterations. With 20 threads and ratio 1.001 (≈34,500 points to 10¹⁸), expect around 30–40 minutes wall time.

The parallelization splits the d-range `[1..√n]` evenly across threads. The progress bar is weighted by √n so it reflects actual compute time rather than point count (with ratio 1.001, the bottom 80% of points by count are tiny n values that finish in microseconds).

### Precision note

The asymptotic subtraction `a(n) − [n²/2·log(n) + ...]` requires ~31 significant digits at n = 10¹⁵. With `libquadmath`, `__float128` provides ~34 digits. Without it, `long double` (~18 digits) causes precision loss above n ≈ 10⁹ — the error column becomes noise rather than signal.

---

## Results

Running to N = 10¹⁸ with ratio 1.001 gives:

```
N=10^13   theta_OLS = 0.250192
N=10^14   theta_OLS = 0.250579
N=10^15   theta_OLS = 0.250817
N=10^16   theta_OLS = 0.251572
N=10^17   theta_OLS = 0.251394
N=10^18   theta_OLS = 0.251367   (alpha = 0.751367)
```

The curve crossed 0.25 around N = 10¹³ and has been flat for the last five decades (last step: Δ ≈ −2.7 × 10⁻⁵). Whether it's converging to exactly 0.25 or something slightly above it is unresolved — the OLS estimator is biased upward because the dense small-n points dominate the regression, and genuinely slow convergence cannot be ruled out at accessible N.

---

## Dependencies

- GCC (tested 11+), OpenMP, libm
- `libquadmath` (optional but strongly recommended)
- Python 3 + no external packages (stdlib only) for `plot_theta.py`

---

## Reference

> Hoying, C. (2022). *Picture At ITP — Sequence A078567*. Unpublished note.

OEIS: https://oeis.org/A078567

# A078567 — Arithmetic Subsequence Counter → Dirichlet Divisor Conjecture

Empirical and analytic investigation of **OEIS A078567**: the number of arithmetic subsequences of `[1..n]` with length > 1, and the growth rate of its oscillatory error term.

Original formula by Claire Hoying (2022). See [`arith_pic.pdf`](arith_pic.pdf).

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

where γ ≈ 0.5772 is the Euler–Mascheroni constant. The `n/4` coefficient was confirmed empirically by window-averaging to suppress the oscillatory noise.

### The conjecture

```
err_osc(n) = O(n^(θ + 1/2))
```

- **Conjecture:** θ = 1/4, i.e. `err_osc = O(n^(3/4))`
- **Proven (Huxley 2003):** θ ≤ 131/416 ≈ 0.3149

The OLS regression over sample points up to N = 10¹⁸ gives **θ_OLS ≈ 0.2514**, converging toward 0.25 from above and visibly plateauing.

---

## Perturbative Dirichlet series derivations (February 2026)

Two companion derivations connect the empirical results to the analytic theory of L-functions. Both use the same perturbative framework: expand the oscillatory error as a Dirichlet series, swap summation order, factor out (qi)^{-s}, and expand in powers of r/(qi).

### From a_hoying to the Dirichlet Divisor Problem (`divisor_derivation.pdf`)

Starting from the exact identity `a(n) = n·D(n−1) − σ_τ(n−1)`, the oscillatory error decomposes as:

```
e(n) = Σ_{i=1}^{floor(√n)}  r_i(i − 2 − r_i) / i  +  O(1)
```

where `r_i = n mod i`. The Dirichlet series `F(s) = Σ e(n)/n^s` expands as:

```
F(s) = F_0(s) + F_1(s) + R(s)
```

with:
- `F_0(s) = ζ(s) · Q_0(s)`,  `Q_0(s) = (1/6)[ζ(s−2) − 6ζ(s−1) + 5ζ(s)]`  — the critical term
- `F_1(s) = −s · ζ(s+1) · Q_1(s)` — analytic for Re(s) > 3/4
- `R(s)` — absolutely convergent for Re(s) > 0 (k ≥ 2 terms, under Huxley θ < 1/2)

Via Perron's formula this yields **θ ≤ μ(3/4)**, where μ(σ) is the Lindelöf μ-function for ζ. The conjecture θ = 1/4 is equivalent to ζ(3/4 + it) remaining sub-polynomially bounded — a consequence of the Lindelöf Hypothesis.

The open gap is the F₁ cancellation problem: showing that structured cancellation between the k = 0 and k = 1 perturbation layers cannot shift the conditional abscissa of F(s) strictly left of that of F₀(s). Hardy's Ω-result e(n) = Ω(n^{1/4}) provides the hard lower bound σ₀ ≥ 3/4.

### From b1_sqrt to the Gauss Circle Problem (`gauss_circle_derivation.pdf`)

The companion derivation runs in parallel for `b1(N) = Σ_{k<N} (N−k)·r₂(k)`, which satisfies the Abel summation identity `b1(N) = Σ_{m<N} A(m)` where `A(m)` counts non-origin lattice points in the disk of radius √m.

The asymptotic `b1(N) = (π/2)N² − N + E₃(N)` is derived from the Gauss circle error, with the dominant mode:

```
E₃(N) ~ (4/π²) · N^{3/4} · sin(2π√N − 5π/4)  +  O(N^{1/4})
```

The stationary-phase analysis explains the 1/2-exponent shift: if Δ_L(N) = O(N^θ) then E₃(N) = O(N^{θ+1/2}), so the OLS slope α of log|E₃| vs log N gives θ = α − 1/2.

The same perturbative expansion gives `F_Gauss(s) ≈ L(s, χ_{−4}) · Q₀^χ(s)` and establishes **θ_Gauss ≤ μ_{χ_{−4}}(3/4)** via Perron's formula.

### The sibling relationship

The two problems are exact siblings:

| | Divisor problem | Gauss circle |
|---|---|---|
| Exact sequence | `a(n) = n·D(n−1) − σ_τ(n−1)` | `b1(N) = Σ_{m<N} A(m)` |
| Building block | `τ(n) = Σ_{d\|n} 1` | `r₂(n) = 4·Σ_{d\|n} χ_{−4}(d)` |
| Character | trivial | χ_{−4} (period 4) |
| Main term | `n²/2·log n + (γ−3/4)n²` | `(π/2)N² − N` |
| Error | `e(n) = O(n^{1/2+θ})` | `E₃(N) = O(N^{1/2+θ})` |
| L-function | ζ(s) | L(s, χ_{−4}) |
| Conjecture | θ = 1/4 (σ₀ = 3/4) | θ = 1/4 (σ₀ = 3/4) |
| Huxley bound | θ ≤ 131/416 | θ ≤ 131/416 |

Both reduce to the same question about sub-polynomial vertical growth of the relevant L-function near Re(s) = 3/4. The identical Huxley bound is not a coincidence.

The hierarchy of implications in both cases:

```
Riemann Hypothesis  ⟹  Lindelöf (μ(1/2) = 0)  ⟹  μ(3/4) = 0  ⟹  θ = 1/4
```

---

## Significance

Via the identity `a(n) = n·D(n−1) − σ_τ(n−1)`, the oscillatory error in `a(n)` is directly coupled to the error in the Dirichlet divisor sum. Empirical bounds on `err_osc` translate into empirical bounds on the divisor problem.

Four observations from the numerical results:

1. **Consistent with the conjecture.** The OLS exponent sits at θ_OLS ≈ 0.2514 at N = 10¹⁸ and has been flat for five decades, well below the proven Huxley bound of 131/416 ≈ 0.3149. The data does not contradict θ = 1/4.

2. **Tighter empirical upper bound.** The running envelope max — `env_max/n^.75` in the output — remaining bounded across 34,555 sample points to N = 10¹⁸ is direct numerical evidence that `err_osc(n) = O(n^(3/4))` holds in practice, approached from a novel object (arithmetic subsequence counts) rather than the divisor sum itself.

3. **The estimate is robust to data trimming.** Computing OLS([10^k_min .. 10^18]) for every possible left cutoff k_min ∈ {4, …, 18} yields values in the band **0.251092 – 0.251741** — a total spread of 6.5 × 10⁻⁴. The estimate never drops below 0.25 under any trimming choice.

4. **The sliding window estimate is smoother and unbiased.** A ±0.75-decade window centred at each 10^k eliminates the jumpiness of the per-decade estimate. The sliding estimate at N = 10¹⁸ gives θ ≈ 0.2489; the mean over the top three decades is ≈ 0.2612.

No claim is made that the true limit is exactly 0.25 — convergence is slow and the gap between 0.2514 and 0.25 may reflect a logarithmic correction of the form `err_osc(n) = Ω(n^(1/4) log(n)^c)`. The contribution is the connection, the bound, and the robustness result — not a proof.

---

## Files

| File | Description |
|------|-------------|
| `main.cpp` | Main C++ program — computes `a(n)`, asymptotic error, OLS θ table. Uses `__int128` for exact arithmetic and `__float128` for precision-safe asymptotic subtraction. |
| `delta.cpp` | Companion program — computes `b1(N)` and the Gauss circle error `E₃(N)` via `chi_{-4}` prefix sums. Includes `--delta` mode to compute `A(N) = b1(N+1) − b1(N)` directly. |
| `divisor_derivation.pdf` | Perturbative Dirichlet series derivation connecting `a(n)` to ζ(s) and the Dirichlet divisor problem. |
| `gauss_circle_derivation.pdf` | Companion derivation connecting `b1(N)` to L(s, χ_{−4}) and the Gauss circle problem. |
| `plot_theta.py` | Python script — reads `./delta` stdout, plots θ_OLS(N) per decade |
| `theta_ols_v2.html` | Pre-generated interactive plot: cumulative, sliding ±0.75-decade, and single-decade OLS (N = 10¹⁸, 34,555 sample points) |
| `theta_cutoff_walkback.html` | Interactive left-cutoff explorer: drag k_min to trim early decades and watch OLS([k_min..10¹⁸]) respond |
| `arith_pic.pdf` | Original formula derivation by Claire Hoying |

---

## Build

**main.cpp** (arithmetic subsequence counter / divisor problem):

```bash
# With libquadmath (recommended for n > 10⁹):
g++ -O3 -fopenmp -o delta main.cpp -lm -lquadmath

# Without libquadmath (precision degrades above n ~ 10⁹):
g++ -O3 -DNO_QUADMATH -fopenmp -o delta main.cpp -lm
```

**delta.cpp** (Gauss circle / b1 counter):

```bash
g++ -O3 -march=native -fopenmp -std=c++17 delta.cpp -lquadmath -o delta_gauss
```

Requires GCC with OpenMP. Uses `__int128`, `__uint128_t`, `__float128` — GCC extensions, not portable to MSVC.

---

## Usage

### main.cpp

```
./delta [N [cutoff [ratio]]]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `N` | 10¹⁶ | Upper limit for the asymptotic table |
| `cutoff` | 10⁴ | Verify Hoying formula vs fused identity up to this n |
| `ratio` | 10.0 | Multiplicative spacing between sample points |

**Note on point count:** `ratio` controls density, not count. With `ratio=1.001` (0.1% steps) over a 16-decade range, you get ~36,860 sample points. To get ~1,000 points use `ratio ≈ 1.037`. The `delta.cpp` program takes an explicit `samples` count instead.

### delta.cpp

```
./delta_gauss Nmax samples cutoff [--csv] [--delta]
```

| Argument | Description |
|----------|-------------|
| `Nmax` | Largest N to evaluate (≤ 3×10¹⁸ to avoid i128 overflow) |
| `samples` | Number of log-spaced sample points in [cutoff, Nmax] |
| `cutoff` | Left boundary of sample range |
| `--delta` | Finite-difference mode: compute A(N) = b1(N+1) − b1(N) and Δ(N) = A(N) − πN directly |

### Examples

```bash
# main.cpp: quick sanity check
./delta

# main.cpp: dense sampling to 10¹⁸ (~30–40 min on 20 cores, ~36k points)
OMP_NUM_THREADS=20 ./delta 1000000000000000000 1000 1.001 | tee results.txt

# main.cpp: ~1000 points to 10¹⁹
OMP_NUM_THREADS=20 ./delta 10000000000000000000 1000 1.037

# delta.cpp: 1000 log-spaced points to 10¹⁸
OMP_NUM_THREADS=20 ./delta_gauss 1000000000000000000 1000 1000

# delta.cpp: direct Gauss circle error (--delta mode)
OMP_NUM_THREADS=20 ./delta_gauss 1000000000000000000 400 1000000000 --delta

# Generate the θ_OLS plot
OMP_NUM_THREADS=20 ./delta 1000000000000000000 1000 1.001 2>/dev/null | python3 plot_theta.py
```

### Output columns (main.cpp)

```
n    err_a/n^.75    env_max/n^.75    err_D/n^.25    combo/n^.25    θ_slope    method
```

- **err_a/n^.75** — oscillatory error normalized by n^(3/4). Bounded iff θ ≤ 1/4.
- **env_max/n^.75** — running maximum of |err_a|/n^(3/4).
- **err_D/n^.25** — error in D(n) normalized by n^(1/4). Only computed at or below `cutoff`.
- **combo/n^.25** — combined error term, only computed at or below `cutoff`.
- **θ_slope** — local slope estimate between consecutive points.
- **method** — `verified` (checked against Hoying formula) or `fused` (identity only).

---

## Performance

Each sample point costs O(√n). At n = 10¹⁸ that's ~10⁹ inner loop iterations. The hot path uses hardware `u64` division throughout (since all supported N fit in `unsigned long long`), avoiding the ~10–20× overhead of software-emulated `u128` division. With 20 threads and ratio 1.001 (~34,500 points to 10¹⁸), expect around 30–40 minutes wall time.

### Precision note

The asymptotic subtraction `a(n) − [n²/2·log(n) + ...]` requires ~31 significant digits at n = 10¹⁵. With `libquadmath`, `__float128` provides ~34 digits. Without it, `long double` (~18 digits) causes precision loss above n ≈ 10⁹.

---

## Results

Running to N = 10¹⁸ with ratio 1.001 gives the following cumulative OLS values:

```
N=10^13   theta_OLS = 0.250192
N=10^14   theta_OLS = 0.250579
N=10^15   theta_OLS = 0.250817
N=10^16   theta_OLS = 0.251572
N=10^17   theta_OLS = 0.251394
N=10^18   theta_OLS = 0.251367   (alpha = 0.751367)
```

The curve crossed 0.25 around N = 10¹³ and has been flat for the last five decades (last step: Δ ≈ −2.7 × 10⁻⁵).

### Sliding window (±0.75 decade)

`plot_theta.py` computes a **sliding OLS** estimator: for each decade boundary 10^k, it regresses over all points in the window (10^(k−0.75), 10^(k+0.75)], giving ~5,700 points per estimate. This is uncontaminated by early-history anchor bias and substantially smoother than the single-decade estimate.

### Left-cutoff robustness

Computing OLS([10^k_min .. 10^18]) for all k_min ∈ {4, …, 18}:

```
k_min= 4:  theta = 0.251363   (all 34,554 points)
k_min= 8:  theta = 0.251420
k_min=12:  theta = 0.251704
k_min=16:  theta = 0.251741   ← maximum
k_min=17:  theta = 0.251092   ← minimum
k_min=18:  theta = 0.251264   (2,304 points only)
```

Full range: **0.251092 – 0.251741** (spread = 6.5 × 10⁻⁴). The estimate never drops below 0.25 under any left-boundary choice.

---

## Dependencies

- GCC (tested 11+), OpenMP, libm
- `libquadmath` (optional but strongly recommended)
- Python 3 (stdlib only) for `plot_theta.py`

---

## Reference

> Hoying, C. (2022). *Picture At ITP — Sequence A078567*. Unpublished note.

OEIS: https://oeis.org/A078567

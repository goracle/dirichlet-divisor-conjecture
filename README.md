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

The sequence grows roughly as `n² log n`. The remainder after subtracting the smooth asymptotic, `err_osc(n)`, turns out to be *exactly* the integral of the Dirichlet divisor error, `err_osc(n) = ∫₀ⁿ Δ(t) dt` (see below). It therefore grows like `n^(3/4)` unconditionally (Voronoi), and by itself says nothing about the Dirichlet divisor conjecture, which concerns `Δ(x)` pointwise. `divisor_delta.cpp` measures `Δ` directly.

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

### Exact structure of `err_osc` (corrected November 2026)

Unrolling the identity gives

```
a(n) = Σ_{k<n} (n − k)·τ(k) = D(n−1) + Σ_{m≤n−2} D(m)
```

an order-1 Riesz mean of the divisor function. Write `D(x) = x(log x + 2γ − 1) + 1/4 + Δ(x)` (the `1/4` makes `Δ` mean-zero). Then, exactly,

```
err_osc(n) = ∫₀ⁿ Δ(t) dt
```

Voronoi's expansion integrates term by term to

```
∫₀ˣ Δ ≈ x^(3/4) / (2√2 π²) · Σ_k d(k) k^(−5/4) · sin(4π√(kx) − π/4)
```

whose coefficients are absolutely summable, so `err_osc(n) = O(n^(3/4))` **unconditionally**, and it is not `o(n^(3/4))`. `voronoi_check.py` confirms the agreement numerically.

Consequences:

- The empirical exponent `α ≈ 0.75` (`θ_OLS ≈ 0.25`) is the exponent of an order-1 Riesz mean. It is a theorem; it neither supports nor refutes the divisor conjecture.
- Pointwise `Δ` enters `err_osc` only at relative size `≲ n^(−5/12)` (`n^(−1/2)` if the conjecture holds, i.e. ~10⁻⁹ at n = 10¹⁸). The 10¹⁸ run cannot distinguish `α = 1/4` from `α = 1/3`.

### The conjecture this does *not* test

**Dirichlet divisor conjecture:** `Δ(x) = O(x^(1/4+ε))`. Known: `Δ = Ω(x^(1/4))` (Hardy), the Cramér mean-square asymptotic `∫₁^X Δ² ~ c·X^(3/2)` with `c = ζ(3/2)⁴ / (6π² ζ(3)) ≈ 0.6543`, and `Δ = O(x^(131/416+ε))` (Huxley). The *generalized* conjecture `α_k = (k−1)/(2k)` for all `k ≥ 2` implies the Lindelöf hypothesis (Titchmarsh, ch. XII); I found no implication Lindelöf ⟹ `α₂ = 1/4`.

---

## Perturbative Dirichlet series derivations (February 2026)

Two companion derivations connect the empirical results to the analytic theory of L-functions. Both use the same perturbative framework: expand the oscillatory error as a Dirichlet series, swap summation order, factor out (qi)^{-s}, and expand in powers of r/(qi).

### From a_hoying to the Dirichlet Divisor Problem (`divisor_derivation.pdf`)

Starting from the exact identity `a(n) = n·D(n−1) − σ_τ(n−1)`, the oscillatory error decomposes as:

```
e(n) = Σ_{i=1}^{s}  r_i(i − 2 − r_i) / i  −  m/12  +  √m · P({√m})  +  C  +  o(1),
       m = n − 1,  s = floor(√m),  r_i = m mod i,
       P(u) = 1 − (2/3)u + 2u² − (4/3)u³
```

> **Correction (November 2026).** An earlier version of this section stated the sum with an `O(1)` error. That is false: the sum alone is off by `≈ −m/12`, a smooth term, plus `√m·P({√m})`, which oscillates with the fractional part of `√m`. Both come from replacing `Σ_{i≤s} 1/i` and the `s`-dependent polynomial terms in the Hoying formula by their expansions in `√m` (`s = √m − {√m}`). The coefficients of `P` were found by least-squares fit to exact values (`check_decomposition.py`, fit `1.0000, −0.6667, 2.0002, −1.3338`) and not yet derived symbolically; `C ≈ −8.7` is also a fit, not derived. The `Σ r_i(i−2−r_i)/i` part has mean `≈ m/12` (each term averages `i/6`), so the `−m/12` re-centres it; the remaining correction is **not** `O(1)`. This also means the perturbative Dirichlet-series framework below (`F = F₀ + F₁ + R`), which was built on the `O(1)` form, is missing these terms; it is kept as structure only.

The Dirichlet series `F(s) = Σ e(n)/n^s` expands as:

```
F(s) = F_0(s) + F_1(s) + R(s)
```

with:
- `F_0(s) = ζ(s) · Q_0(s)`,  `Q_0(s) = (1/6)[ζ(s−2) − 6ζ(s−1) + 5ζ(s)]`  — the critical term
- `F_1(s) = −s · ζ(s+1) · Q_1(s)` — analytic for Re(s) > 3/4
- `R(s)` — absolutely convergent for Re(s) > 0 (k ≥ 2 terms, under Huxley θ < 1/2)

> **Status note (November 2026).** The paragraphs that stood here (the claim `θ ≤ μ(3/4)`, the identification of the conjecture with sub-polynomial growth of `ζ(3/4+it)`, and the "F₁ cancellation gap") are **withdrawn**. `F(s) = Σ e(n) n^(−s)` is the Dirichlet series of the Riesz-mean error `err_osc = ∫Δ`, which is `Θ(n^(3/4))` unconditionally, so its abscissa is not the divisor-problem exponent; and abscissa/Perron arguments control means, not the pointwise bound on `Δ`. The algebraic decomposition `F = F₀ + F₁ + R` is kept in `divisor_derivation.pdf` as structure only. See "Exact structure of `err_osc`" above.

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

Both problems are order-1 Riesz means of an arithmetic function (`τ`, resp. `r₂`), so both "Error" rows above are Θ(N^(3/4)) **unconditionally** (Voronoi / Hardy expansions). The identical Huxley bound applies to the *pointwise* errors `Δ` and `Δ_L`, which is where the actual conjectures live. In `delta.cpp`, `--delta` mode measures `Δ_L` directly; `divisor_delta.cpp` is the analogue for the divisor problem.

The implication chain previously shown here (`RH ⟹ Lindelöf ⟹ μ(3/4) = 0 ⟹ θ = 1/4`) is **withdrawn**: its last arrow is not known. What is known is the other direction for the *generalized* conjecture (all `k`) ⟹ Lindelöf.

---

## Significance (revised November 2026)

1. **The formula and evaluator are sound.** Hoying's formula is an O(√n) evaluator of `a(n) = Σ_{k<n}(n−k)τ(k)`, cross-checked against brute force and the fused hyperbola identity.
2. **The numerics confirm Voronoi, not the divisor conjecture.** `err_osc = ∫Δ` is `Θ(n^(3/4))` by theorem. The observed `θ_OLS ≈ 0.2514`, the bounded `env_max/n^.75`, and the left-cutoff and sliding-window results below all measure this known exponent. The small persistent offset above 0.25 is most likely OLS sampling noise on a bounded oscillation rather than a logarithmic correction (untested).
3. **Pointwise `Δ` is nearly invisible in `err_osc`** (see above), so `err_osc` bounds on `Δ` are vacuous.
4. **What does test the conjecture:** `divisor_delta.cpp` evaluates `Δ(x)` at half-integers `x = n + 1/2` out to 10¹⁸ and reports per-decade mean, mean-square (compare Cramér: `E[(Δ/x^(1/4))²] → 0.98143` for log-uniform sampling) and extremes. A sampled maximum is a weak sup-probe; the mean-square and distribution are the sharper statistics.
5. **Riemann hypothesis status.** On 20 November 2026 Levent Alpöge announced, with Claude, a Lean-formalized proof of RH and of GRH for Dirichlet L-functions; community checking of the Lean development is ongoing and a human-readable proof is still in progress. Even granting it, `α = 1/4` is not known to follow from RH/Lindelöf, so nothing here is "solved" by it, and nothing in this repo depended on GRH being open. The Gauss-circle sibling uses `L(s, χ₋₄)`; `ζ_{Q(i)} = ζ · L(s, χ₋₄)`, so it is covered by RH + GRH for Dirichlet L-functions, with the same conclusion.

---

## Files

| File | Description |
|------|-------------|
| `main.cpp` | Main C++ program — computes `a(n)`, asymptotic error, OLS θ table. Uses `__int128` for exact arithmetic and `__float128` for precision-safe asymptotic subtraction. |
| `divisor_delta.cpp` | **Direct** divisor-problem probe: `Δ(x)` at `x = n+1/2` to 10¹⁸, per-decade mean / mean-square / extremes of `Δ/x^(1/4)`, optional CSV. Self-checks against a sieve and aborts on any mismatch. |
| `voronoi_check.py` | Verifies `err_osc(n) = ∫₀ⁿΔ` against the Voronoi series (exact big-int `a(n)`, `n ≲ 10¹³`). |
| `gauss_circle_conjecture/delta.cpp` | Companion program — computes `b1(N)` and the Gauss circle error `E₃(N)` via `chi_{-4}` prefix sums. Includes `--delta` mode to compute `A(N) = b1(N+1) − b1(N)` directly. |
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

**divisor_delta.cpp** (direct Δ probe):

```bash
g++ -O3 -march=native -fopenmp -std=gnu++17 divisor_delta.cpp -lquadmath -o divisor_delta
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

### divisor_delta.cpp

```
./divisor_delta [Nmax [ratio [nmin]]] [--csv=path]     # defaults 1e18, 1.001, 1e4
```

Quick check: `./divisor_delta 100000000 1.01`. Full run: `OMP_NUM_THREADS=$(nproc) ./divisor_delta 1000000000000000000 1.001 --csv=delta_direct.csv`. Per point cost is O(√n) with hardware u64 division, so expect roughly 20–60 min on 20 cores for ~34k points to 10¹⁸. `python3 voronoi_check.py` validates the `err_osc = ∫Δ` identity.

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

> **Reading note (November 2026):** the exponents below are those of the Riesz-mean error `err_osc = ∫Δ`, which is `Θ(n^(3/4))` by theorem, so they confirm Voronoi's expansion rather than bear on the Dirichlet divisor conjecture. See "Exact structure of `err_osc`".

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

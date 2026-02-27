// Compile: g++ -O2 -o delta main.cpp -lm -lquadmath
// (use -std=gnu++14 if needed; -std=gnu11 is C-only and will warn)
// Without libquadmath: g++ -O2 -DNO_QUADMATH -o delta main.cpp -lm
// (asymptotic error display inaccurate for n > ~10^9 without it)
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <vector>
#include <cassert>
#include <atomic>
#include <cstring>
#include <omp.h>

#ifndef NO_QUADMATH
#include <quadmath.h>
#endif

typedef __int128 i128;
typedef __uint128_t u128;
typedef unsigned long long ull;

/* ---- Progress bar: weighted by sqrt(n) so bar reflects actual compute time.
   Small-n points are cheap (microseconds) and numerous; large-n points are
   expensive (seconds).  Counting points equally makes the bar sprint to ~80%
   instantly then crawl -- weighting by sqrt(n) matches the O(sqrt(n)) cost.
   Thread-safe: atomic CAS accumulator, mutex-throttled rendering (~1/sec).  */
struct Progress {
    double                 total_weight;
    std::atomic<long long> done_weight_bits;  // double accumulated as raw bits via CAS
    double                 t_start;
    FILE                  *tty;
    omp_lock_t             lock;
    double                 last_render;

    void init(const std::vector<u128> &ns) {
        total_weight = 0.0;
        for (auto n : ns) total_weight += sqrt((double)n);
        done_weight_bits.store(0);
        t_start     = omp_get_wtime();
        last_render = t_start - 1.0;
        omp_init_lock(&lock);
        tty = fopen("/dev/tty", "w");
        if (!tty) tty = stderr;
        render_locked();
    }

    void tick(double w) {
        // Accumulate w into a double atomically via CAS on its bit pattern.
        while (true) {
            long long old_bits = done_weight_bits.load(std::memory_order_relaxed);
            double old_val; memcpy(&old_val, &old_bits, 8);
            double new_val = old_val + w;
            long long new_bits; memcpy(&new_bits, &new_val, 8);
            if (done_weight_bits.compare_exchange_weak(old_bits, new_bits,
                    std::memory_order_relaxed)) break;
        }
        double now = omp_get_wtime();
        if (now - last_render < 1.0) return;
        if (omp_test_lock(&lock)) {
            if (now - last_render >= 1.0) { last_render = now; render_nolock(); }
            omp_unset_lock(&lock);
        }
    }

    void finish() {
        omp_set_lock(&lock);
        double elapsed = omp_get_wtime() - t_start;
        fprintf(tty, "\r%-80s\r", "");
        fprintf(tty, "  done in %.1fs\n", elapsed);
        fflush(tty);
        omp_unset_lock(&lock);
        omp_destroy_lock(&lock);
        if (tty != stderr) fclose(tty);
        tty = nullptr;
    }

private:
    void render_locked() { omp_set_lock(&lock); render_nolock(); omp_unset_lock(&lock); }

    void render_nolock() {
        long long bits = done_weight_bits.load(std::memory_order_relaxed);
        double done_w; memcpy(&done_w, &bits, 8);

        double elapsed = omp_get_wtime() - t_start;
        double frac    = (total_weight > 0.0) ? done_w / total_weight : 0.0;
        if (frac > 1.0) frac = 1.0;

        char eta_buf[32] = "  --:--";
        if (frac > 0.001) {
            double remain = elapsed / frac * (1.0 - frac);
            if (remain < 3600)
                snprintf(eta_buf, sizeof(eta_buf), " eta %2d:%02d",
                         (int)(remain/60), (int)remain % 60);
            else
                snprintf(eta_buf, sizeof(eta_buf), " eta %dh%02dm",
                         (int)(remain/3600), ((int)(remain/60)) % 60);
        }

        const int W = 20;
        int filled = (int)(frac * W + 0.5);
        char bar[W+3]; bar[0]='[';
        for (int i = 0; i < W; i++) bar[1+i] = (i < filled) ? '#' : '-';
        bar[W+1]=']'; bar[W+2]='\0';

        fprintf(tty, "\r  %s %.1f%%  %.0fs elapsed%s   ",
                bar, frac * 100.0, elapsed, eta_buf);
        fflush(tty);
    }
};
static Progress g_progress;

/* --- Utility: i128 to String --- */
std::string i128_to_str(i128 n) {
    if (n == 0) return "0";
    std::string s = "";
    bool neg = false;
    if (n < 0) { neg = true; n = -n; }
    while (n > 0) {
        s += (char)('0' + (n % 10));
        n /= 10;
    }
    if (neg) s += '-';
    std::reverse(s.begin(), s.end());
    return s;
}

std::string u128_to_str(u128 n) {
    if (n == 0) return "0";
    std::string s = "";
    while (n > 0) { s += (char)('0' + (n % 10)); n /= 10; }
    std::reverse(s.begin(), s.end());
    return s;
}

/* Parse a decimal string into u128 (handles values beyond strtoull range) */
u128 parse_u128(const char *s) {
    u128 v = 0;
    for (; *s >= '0' && *s <= '9'; s++)
        v = v * 10 + (*s - '0');
    return v;
}

/* Integer square root for u128: returns floor(sqrt(n)) exactly */
u128 isqrt_u128(u128 n) {
    if (n == 0) return 0;
    // Start with float128 approximation then correct
    u128 r = (u128)sqrtl((long double)n);
    // sqrtl loses precision for n > ~2^106; nudge up/down to find exact floor
    while ((r+1)*(r+1) <= n) r++;
    while (r*r > n) r--;
    return r;
}

/* --- D(n) = sum_{m=1}^{n} tau(m)  [O(sqrt(n))]
   Hyperbola method:  sum_{m<=n} tau(m) = 2 * sum_{k=1}^{r} floor(n/k) - r^2
   where r = floor(sqrt(n)).                                              */
i128 D(u128 n) {
    if (n == 0) return 0;
    u128 r = isqrt_u128(n);
    i128 s = 0;
    for (u128 k = 1; k <= r; k++)
        s += (i128)(n / k);
    return 2*s - (i128)r*(i128)r;
}

/* --- sigma_tau(n) = sum_{m=1}^{n} m*tau(m)  [O(sqrt(n))]
   Via: sum_{m<=n} m*tau(m) = sum_{d=1}^{n} d * T(floor(n/d))
   where T(k)=k(k+1)/2, since tau(m) = #{d|m} => each d contributes d*1
   to every multiple m=d*k<=n, and the inner sum over k of d*k = d*T(floor(n/d)).
   Block-compress over distinct values of floor(n/d).                     */
i128 sigma_tau(u128 n) {
    if (n == 0) return 0;
    auto T = [](i128 k) -> i128 { return k*(k+1)/2; };
    i128 s = 0;
    u128 d = 1;
    while (d <= n) {
        u128 q  = n / d;
        u128 d2 = n / q;
        i128 lo = (i128)d, hi = (i128)d2;
        i128 sum_d = (lo + hi) * (hi - lo + 1) / 2;
        s += T((i128)q) * sum_d;
        d = d2 + 1;
    }
    return s;
}

/* --- Original O(sqrt(n)) formula from PDF --- */
/* Sum over i=1..r is a pure reduction -- no dependencies, trivially parallel. */
i128 a_hoying(u128 n) {
    if (n == 0) return 0;
    u128 r_val = isqrt_u128(n);
    i128 r     = (i128)r_val;
    i128 n_128 = (i128)n;
    i128 term1 = (r * r) * ((1 + r) * (1 + r) - 4 * (n_128 + 1)) / 4;

    // OpenMP reduction on i128 not natively supported; accumulate per-thread
    // partial sums in a vector then reduce serially.
    int nt = omp_in_parallel() ? 1 : omp_get_max_threads();
    std::vector<i128> parts(nt, (i128)0);

    #pragma omp parallel num_threads(nt)
    {
        int t = omp_get_thread_num();
        i128 local = 0;
        #pragma omp for schedule(static) nowait
        for (u128 i = 1; i <= r_val; i++) {
            i128 q     = (i128)(n / i);
            i128 i_128 = (i128)i;
            local += q * (2 * n_128 + 2 - i_128 * (1 + q));
        }
        parts[t] = local;
    }

    i128 sum_part = 0;
    for (int t = 0; t < nt; t++) sum_part += parts[t];
    return term1 + sum_part;
}

/* --- Joint computation of D(n) and sigma_tau(n).
   PARALLELIZED: split d-range [1..n] into per-thread chunks, pure reduction.

   The block-compression loop has ~2*sqrt(n) iterations (one per distinct
   floor(n/d) value).  Each iteration is independent, so we split the d-range
   into nt contiguous chunks and run each on its own core.

   Chunk boundary subtlety: a block [d, n/(n/d)] may straddle a chunk boundary.
   We handle this by clamping hi = min(n/q, d_hi) inside the kernel, which
   splits one logical block into two partial blocks — each is still a valid
   arithmetic sum, just over a sub-range of d.

   Load balance: block density is ~1/sqrt(d), so small-d chunks have more blocks
   than large-d chunks.  Splitting [1..n] equally by d-value gives ~sqrt(n)/nt
   blocks per chunk at small d vs ~1 block at large d -- mildly uneven, but the
   large-d region (d > r = sqrt(n)) contributes negligibly to wall time since
   each block there spans a huge range of d values.  In practice this gives
   near-linear speedup up to ~8 cores; beyond that consider splitting [1..r]
   more carefully.                                                               */

/* Kernel: accumulate partial D and sigma_tau sums for d in [d_lo, d_hi].       */
static void D_ST_chunk(u128 n, u128 r, u128 d_lo, u128 d_hi,
                       i128 &part_D, i128 &part_ST) {
    part_D  = 0;
    part_ST = 0;
    if (d_lo > d_hi || d_lo > n) return;

    u128 d = d_lo;
    while (d <= d_hi) {
        u128 q  = n / d;
        u128 hi = n / q;
        if (hi > d_hi) hi = d_hi;   // clamp block to chunk boundary

        u128 blen  = hi - d + 1;
        u128 bslo  = d + hi;
        i128 bsumd = (bslo & 1) ? (i128)bslo  * (i128)(blen / 2)
                                 : (i128)(bslo / 2) * (i128)blen;
        i128 tq = (i128)q * ((i128)q + 1) / 2;
        part_ST += tq * bsumd;

        if (d <= r) {
            u128 hic = (hi <= r) ? hi : r;
            part_D += (i128)q * (i128)(hic - d + 1);
        }

        d = hi + 1;
    }
}

void D_and_sigma_tau(u128 n, i128 &out_D, i128 &out_ST) {
    if (n == 0) { out_D = 0; out_ST = 0; return; }

    u128 r  = isqrt_u128(n);
    int  nt = omp_get_max_threads();

    // KEY: split [1..r] evenly, not [1..n].
    // All ~sqrt(n) non-trivial blocks live in d in [1..r].
    // Splitting [1..n] gave thread 0 all the work (density ~ 1/sqrt(d)).
    // [r+1..n] has at most r trivial blocks; handle serially as a tail.
    std::vector<i128> pD(nt, 0), pST(nt, 0);

    // If we're already inside a parallel region (points-level parallelism),
    // run single-threaded to avoid nested OMP. Otherwise use all cores.
    bool nested = (omp_in_parallel() != 0);
    int active  = nested ? 1 : nt;

    #pragma omp parallel for schedule(static) num_threads(active)
    for (int t = 0; t < active; t++) {
        u128 chunk = (r + (u128)active - 1) / (u128)active;
        u128 d_lo  = (u128)t * chunk + 1;
        u128 d_hi  = d_lo + chunk - 1;
        if (d_hi > r) d_hi = r;
        D_ST_chunk(n, r, d_lo, d_hi, pD[t], pST[t]);
    }

    // Serial tail: d in [r+1..n] (~r trivial blocks, negligible time)
    i128 tail_D = 0, tail_ST = 0;
    D_ST_chunk(n, r, r + 1, n, tail_D, tail_ST);

    i128 sum_D = tail_D, sum_ST = tail_ST;
    for (int t = 0; t < nt; t++) { sum_D += pD[t]; sum_ST += pST[t]; }

    out_D  = 2 * sum_D - (i128)r * (i128)r;
    out_ST = sum_ST;
}

i128 a_identity_fused(u128 n) {
    if (n <= 1) return 0;
    i128 dn, st;
    D_and_sigma_tau(n-1, dn, st);
    return (i128)n * dn - st;
}

i128 a_identity(u128 n) {
    if (n <= 1) return 0;
    return (i128)n * D(n-1) - sigma_tau(n-1);
}

i128 a_brute_force(u128 n) {
    if (n < 2) return 0;
    i128 count = 0;
    for (u128 d = 1; d < n; ++d)
        for (u128 s = 1; s <= n; ++s) {
            u128 max_k = (n - s) / d + 1;
            if (max_k >= 2) count += (i128)(max_k - 1);
        }
    return count;
}

/* --- Asymptotic for a(n):
   Via a(n) = n*D(n-1) - sigma_tau(n-1) and Dirichlet expansions:
     D(x)         = x*log(x) + (2g-1)*x + Delta(x),    Delta(x) = O(x^{1/3})
     sigma_tau(x) = x^2/2*log(x) + (g-1/4)*x^2 + x*Delta(x) + O(x^{4/3})
   Exact coefficient n/4 empirically confirmed by window-averaging to kill Delta noise.

     a(n) = n^2/2 * log(n) + (g-3/4)*n^2 + n/4 + O(n^{2/3})

   The O(n^{2/3}) remainder comes from the x*Delta(x) = O(x^{4/3}) term in sigma_tau,
   which after cancellation in a(n) = n*D(n-1) - sigma_tau(n-1) leaves O(n^{2/3}).

   PRECISION: n^2/2*log(n) ~ n^2 * 1.15*log10(n) requires ~2*log10(n)+1 significant digits.
   At n=10^15 that is ~31 digits. We use __float128 (113-bit mantissa, ~34 digits) with
   logq() from libquadmath for a correctly-rounded __float128 logarithm.              */
__float128 a_asymptotic(u128 n) {
    const __float128 gamma_em = 0.57721566490153286060651209008240Q;
    __float128 x = (__float128)n;
#ifdef NO_QUADMATH
    __float128 logx = (__float128)logl((long double)x);
#else
    __float128 logx = logq(x);
#endif
    return x*x/2.0Q * logx + (gamma_em - 0.75Q) * x*x + x/4.0Q;
}

/* ======== Tests ======== */

void run_sequence_test() {
    uint64_t expected[] = {0, 1, 4, 9, 17, 27, 41, 57, 77, 100, 127};
    printf("--- A078567 sequence check ---\n");
    for (u128 i = 0; i <= 10; ++i) {
        i128 r = a_hoying(i);
        printf("a(%lu): %s (%lu)\n", (unsigned long)(i+1),
            (uint64_t)r == expected[(int)i] ? "PASS" : "FAIL", (uint64_t)r);
    }
}

void run_identity_test(u128 cutoff) {
    u128 limit = (cutoff < 300) ? cutoff : (u128)300;
    printf("\n--- Identity check (all methods vs brute force, n <= %s) ---\n", u128_to_str(limit).c_str());
    bool ok = true;
    for (u128 n = 2; n <= limit; n++) {
        i128 h  = a_hoying(n-1);
        i128 id = a_identity(n);
        i128 fu = a_identity_fused(n);
        i128 bf = a_brute_force(n);
        if (h != id || h != bf || h != fu) {
            printf("MISMATCH n=%s: hoying=%s identity=%s fused=%s brute=%s\n",
                u128_to_str(n).c_str(), i128_to_str(h).c_str(), i128_to_str(id).c_str(),
                i128_to_str(fu).c_str(), i128_to_str(bf).c_str());
            ok = false;
        }
    }
    if (ok) printf("All n in [2,%s]: PASS\n", u128_to_str(limit).c_str());
}

long double exact_err(u128 n) {
    // Use a_hoying(n-1) directly -- simpler inner loop (no multiply) than
    // a_identity_fused, and already verified correct vs brute force.
    // i128 max ~ 1.7e38; caller must ensure n <= I128_SAFE_LIMIT.
    i128       an        = a_hoying(n - 1);
    __float128 asym128   = a_asymptotic(n);
    i128       asym_int  = (i128)asym128;
    __float128 asym_frac = asym128 - (__float128)asym_int;
    return (long double)(an - asym_int) + (long double)asym_frac;
}

// Max n where a(n) fits in i128: a(n) ~ n^2/2*log(n) < 1.7e38 => n <~ 3e18
// Use 2e18 as safe limit to keep intermediate arithmetic well clear of overflow.
static const u128 I128_SAFE_LIMIT = (u128)2000000000000000000ULL;

/* --- Exact error for sigma_tau(n):
   sigma_tau(n) ~ n^2/2*log(n) + (g-1/4)*n^2  +  err_sigma
   (the constant differs from a(n) by exactly +1/2 * n^2)             */
long double exact_err_sigma(u128 n) {
    const __float128 gamma_em = 0.57721566490153286060651209008240Q;
    __float128 x = (__float128)n;
#ifdef NO_QUADMATH
    __float128 logx = (__float128)logl((long double)x);
#else
    __float128 logx = logq(x);
#endif
    __float128 asym = x*x/2.0Q * logx + (gamma_em - 0.25Q) * x*x;
    i128 st       = sigma_tau(n);
    i128 asym_int = (i128)asym;
    return (long double)(st - asym_int) + (long double)(asym - (__float128)asym_int);
}

/* --- Exact error for "recovered D" = (sigma_tau(n) - a(n+1)) / n
   By the identity a(n+1) = (n+1)*D(n) - sigma_tau(n), we have:
     D(n) = (a(n+1) + sigma_tau(n)) / (n+1)  ... uses integer division, messy.
   Cleaner: just look at err_sigma - err_a as a combined oscillation,
   divided by n, and compare its magnitude to err_D directly.
   err_recovered_D(n) = (err_sigma(n) - err_a(n+1)) / n
   vs err_D(n) = D(n) - [n*log(n) + (2g-1)*n]                          */
long double exact_err_D(u128 n) {
    const __float128 gamma_em = 0.57721566490153286060651209008240Q;
    __float128 x = (__float128)n;
#ifdef NO_QUADMATH
    __float128 logx = (__float128)logl((long double)x);
#else
    __float128 logx = logq(x);
#endif
    __float128 asym = x * logx + (2.0Q*gamma_em - 1.0Q) * x;
    i128 d        = D(n);
    i128 asym_int = (i128)asym;
    return (long double)(d - asym_int) + (long double)(asym - (__float128)asym_int);
}

/* --- Combined: (sigma_tau(n) - a(n+1)) / n  recovers D(n) exactly.
   Its error relative to D's asymptotic:
     err_combo(n) = (err_sigma(n) - err_a(n+1)) / n
   We compute this as exact_err_sigma(n)/n - exact_err(n+1)/n.          */
long double exact_err_combo(u128 n) {
    long double es = exact_err_sigma(n);
    long double ea = exact_err(n + 1);
    return (es - ea) / (long double)n;
}


/* --- Empirical theta estimate ---
   From err_a = O(n^{theta + 1/2}), we estimate theta via:
     theta_point(n) = log|err_a| / log(n) - 1/2
     theta_slope    = slope of log|err_a| vs log(n) between consecutive pts - 1/2
   Both are noisy due to sign oscillation in err_a.

   Better estimate: track the running envelope max|err_a| over [n0..n],
   regress log(env) ~ (theta+1/2)*log(n) + C on all data, and report
   the OLS theta.  Also report the running-max column directly.

   Theory:
     Divisor conjecture (theta=1/4):   err_a = O(n^{3/4})
     Huxley proven    (theta=131/416): err_a = O(n^{339/416}) ~ O(n^{0.815}) */

/* Generate the set of sample points from n_start up to max_n,
   multiplicatively spaced by ratio (a double, e.g. 10.0 or 2.0).
   We always start at 1000 and include max_n exactly.               */
static std::vector<u128> make_sample_ns(u128 max_n, double ratio) {
    if (ratio < 1.001) ratio = 1.001;
    std::vector<u128> ns;
    // Walk in log space: n_{k+1} = round(n_k * ratio), but use double accumulator
    double acc = 1000.0;
    u128   prev = 0;
    while (true) {
        u128 n = (u128)acc;
        if (n < 1000) n = 1000;
        if (n != prev) { ns.push_back(n); prev = n; }
        if (n >= max_n) break;
        acc *= ratio;
        if ((u128)acc >= max_n) {
            if (max_n != prev) ns.push_back(max_n);
            break;
        }
    }
    return ns;
}

/* Row result from computing one sample point -- filled in parallel, printed in order */
struct RowResult {
    u128        n;
    bool        skip;          // overflows i128
    bool        mismatch;      // hoying vs fused disagree
    std::string mismatch_msg;
    bool        verified;
    long double err_a;
    long double err_d;     // NAN if not computed (above cutoff)
    long double err_combo; // NAN if not computed (above cutoff)
    long double logn;
};

void run_asymptotic_table(u128 cutoff, u128 max_n, double ratio) {
    printf("\n--- Asymptotic error and empirical theta ---\n");
    printf("a(n) = n^2/2*log(n) + (g-3/4)*n^2 + n/4  +  err_osc\n");
    printf("err_osc = O(n^{theta+1/2}).  ratio=%.4f  cutoff=%s\n",
           ratio, u128_to_str(cutoff).c_str());
    printf("Below cutoff: fused verified vs hoying.  Above: fused only (trusted).\n");

    std::vector<u128> ns = make_sample_ns(max_n, ratio);
    int npts = (int)ns.size();

    printf("Computing %d sample points on %d threads...\n\n", npts, omp_get_max_threads());

    // Without this, stdout is fully buffered when piped (e.g. tee), so rows
    // accumulate silently and dump all at once at exit.  Line-buffering means
    // each completed row flushes immediately.
    setvbuf(stdout, nullptr, _IOLBF, 0);

    printf("%20s  %12s  %12s  %12s  %12s  %12s  %s\n",
           "n", "err_a/n^.75", "env_max/n^.75", "err_D/n^.25", "combo/n^.25", "θ_slope", "method");
    fflush(stdout);

    // Parallelize across points. D_and_sigma_tau detects omp_in_parallel()
    // and runs serially inside, so no nested-OMP conflict.
    std::vector<RowResult> rows(npts);

    // Progress during compute: atomic counter incremented by each thread as it
    // finishes a point.  A dedicated render call after each increment keeps the
    // bar live during the slow parallel phase rather than only after it.
    g_progress.init(ns);

    #pragma omp parallel for schedule(dynamic,1) num_threads(omp_get_max_threads())
    for (int i = 0; i < npts; i++) {
        u128 n = ns[i];
        RowResult &r = rows[i];
        r.n        = n;
        r.skip     = false;
        r.mismatch = false;
        r.verified = (n <= cutoff);

        if (n > I128_SAFE_LIMIT) {
            r.skip = true;
            g_progress.tick(sqrt((double)n));
            continue;
        }

        if (r.verified) {
            i128 hoy = a_hoying(n-1);
            i128 fus = a_identity_fused(n);
            if (hoy != fus) {
                r.mismatch = true;
                r.mismatch_msg = "MISMATCH hoying=" + i128_to_str(hoy)
                               + " fused=" + i128_to_str(fus);
                g_progress.tick(sqrt((double)n));
                continue;
            }
        }

        r.err_a     = exact_err(n);
        // Only compute D/combo columns at or below cutoff -- they each
        // require a full extra O(sqrt(n)) pass and aren't needed for theta.
        if (r.verified) {
            r.err_d     = exact_err_D(n);
            r.err_combo = exact_err_combo(n);
        } else {
            r.err_d     = NAN;
            r.err_combo = NAN;
        }
        r.logn      = logl((long double)n);
        g_progress.tick(sqrt((double)n));
    }

    g_progress.finish();

    // Print in order (serial), accumulate running stats
    long double prev_err  = 0.0L, prev_logn = 0.0L;
    long double env_max   = 0.0L;

    long double sum1  = 0.0L, sumx = 0.0L, sumy  = 0.0L;
    long double sumxx = 0.0L, sumxy = 0.0L;
    int reg_cnt = 0;

    for (int i = 0; i < npts; i++) {
        const RowResult &r = rows[i];
        u128 n = r.n;

        if (r.skip) {
            printf("%20s  [skipped: overflows i128]\n", u128_to_str(n).c_str());
            continue;
        }
        if (r.mismatch) {
            printf("%20s  %s\n", u128_to_str(n).c_str(), r.mismatch_msg.c_str());
            prev_err = 0.0L; prev_logn = 0.0L;
            continue;
        }

        long double x    = (long double)n;
        long double n75  = powl(x, 0.75L);
        long double n25  = powl(x, 0.25L);
        long double logn = r.logn;

        long double err_a     = r.err_a;
        long double err_d     = r.err_d;
        long double err_combo = r.err_combo;
        long double abs_err   = fabsl(err_a);

        if (abs_err > env_max) env_max = abs_err;

        long double theta_sl = NAN;
        if (prev_err != 0.0L && err_a != 0.0L && prev_logn > 0.0L)
            theta_sl = (logl(abs_err) - logl(fabsl(prev_err)))
                       / (logn - prev_logn) - 0.5L;

        // Outlier filter: skip points where |err_a|/n^0.75 > 10 (overflow artifact)
        if (abs_err > 0.0L && fabsl(err_a / n75) < 10.0L) {
            long double loge = logl(abs_err);
            sum1  += 1.0L; sumx  += logn;  sumy  += loge;
            sumxx += logn * logn;           sumxy += logn * loge;
            reg_cnt++;
        }

        const char *method = r.verified ? "verified" : "fused";
        char col_d[16], col_combo[16];
        if (isnan(err_d))     snprintf(col_d,     sizeof(col_d),     "%12s", "-");
        else                  snprintf(col_d,     sizeof(col_d),     "%12.5Lf", (long double)(err_d / n25));
        if (isnan(err_combo)) snprintf(col_combo, sizeof(col_combo), "%12s", "-");
        else                  snprintf(col_combo, sizeof(col_combo), "%12.5Lf", (long double)(err_combo / n25));
        printf("%20s  %12.5Lf  %12.5Lf  %s  %s",
            u128_to_str(n).c_str(),
            err_a / n75, env_max / n75,
            col_d, col_combo);
        if (!isnan(theta_sl))
            printf("  %+12.4Lf", theta_sl);
        else
            printf("  %12s", "     -");
        printf("  %s\n", method);

        prev_err  = err_a;
        prev_logn = logn;
    }

    printf("\n  Columns normalized so bounded value => theta <= 1/4 for a (n^3/4)\n");
    printf("  and theta <= 1/4 for D, combo (n^1/4, since err_D = O(n^{theta-1/2})*n)\n");
    printf("  theta=131/416~0.315 (Huxley bound)\n");

    if (reg_cnt >= 3) {
        long double denom = sum1 * sumxx - sumx * sumx;
        if (fabsl(denom) > 0.0L) {
            long double alpha     = (sum1 * sumxy - sumx * sumy) / denom;
            long double theta_ols = alpha - 0.5L;
            printf("\n  OLS regression log|err_a| ~ alpha*log(n) + C over %d pts:\n", reg_cnt);
            printf("    alpha = %.6Lf  =>  theta_OLS = alpha - 0.5 = %.6Lf\n", alpha, theta_ols);
            printf("    (compare: conjecture theta=0.25, Huxley theta<=0.315)\n");
        }
    }
}

int main(int argc, char **argv) {
    // Usage: ./delta [N [cutoff [ratio]]]
    // N:      max n for asymptotic table (u128); default 10^16
    // cutoff: verify hoying vs fused up to this n; default 10^4
    // ratio:  multiplicative step between sample points; default 10.0
    //         (use 2.0 for ~3x more points/decade, sqrt(10)~3.162 for 2x)
    u128   N      = 0;
    u128   cutoff = (u128)10000ULL;
    double ratio  = 10.0;

    if (argc >= 2) N      = parse_u128(argv[1]);
    if (argc >= 3) cutoff = parse_u128(argv[2]);
    if (argc >= 4) ratio  = atof(argv[3]);
    if (ratio < 1.001) {
        fprintf(stderr, "ratio must be > 1.001\n");
        return 1;
    }

    u128 max_table_n = N ? N : (u128)10000000000000000ULL;

    run_sequence_test();
    run_identity_test(cutoff);
    run_asymptotic_table(cutoff, max_table_n, ratio);

    if (!N) return 0;

    if (N > I128_SAFE_LIMIT) {
        printf("\nNote: N=%s exceeds i128 safe limit (~3e18). a(N) overflows; upgrade accumulator to go higher.\n",
            u128_to_str(N).c_str());
        return 1;
    }
    long double x    = (long double)N;
    long double logN = logl(x);
    long double err  = exact_err(N);

    long double theta_pt = (err != 0.0L) ? logl(fabsl(err)) / logN - 0.5L : 0.0L;

    printf("\n--- N = %s ---\n", u128_to_str(N).c_str());
    printf("err_osc                = %.4Lf\n", err);
    printf("err / n^(3/4)          = %.8Lf   [bounded iff theta <= 1/4]\n",
           err / powl(x, 0.75L));
    printf("err / n^(0.815)        = %.8Lf   [bounded iff theta <= 131/416]\n",
           err / powl(x, 0.815L));
    printf("theta_point            = %.6Lf\n", theta_pt);

    if (N >= 10000) {
        u128 N2 = N / 10;
        long double err2  = exact_err(N2);
        long double logN2 = logl((long double)N2);
        long double theta_sl = (err != 0.0L && err2 != 0.0L)
            ? (logl(fabsl(err)) - logl(fabsl(err2))) / (logN - logN2) - 0.5L
            : NAN;
        printf("theta_slope (vs N/10)  = %.6Lf\n", theta_sl);
    }

    return 0;
}

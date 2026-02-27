// delta.cpp
//
// Computes b1(N) = sum_{k=1}^{N-1} (N-k) * r2(k)
// via the O(sqrt N) hyperbola-blocking formula, then studies the error term.
//
// ============================================================
// ASYMPTOTIC (verified analytically and against data to N=10^18)
// ============================================================
//
// Key identity: b1(N) = sum_{m=1}^{N-1} A(m)
// (from discrete Abel summation; A(m) = sum_{k<=m} r2(k))
//
// A(m) = pi*m + Delta_A(m) where Delta_A(m) = A(m) - pi*m.
// A excludes the origin, so Delta_A = Delta_L - 1 where Delta_L is the
// standard Gauss circle error.  Summing the constant -1 per term:
//
//   b1(N) = pi*(N-1)*N/2 + sum_{m=1}^{N-1} Delta_A(m)
//         = pi*(N-1)*N/2 + [sum Delta_L(m)] - (N-1)
//
// Empirically (confirmed to 5 significant figures at N = 10^18):
//   sum_{m=1}^{N-1} Delta_A(m) ~ (pi/2 - 1)*N + E3(N)
//   (pi/2 - 1 = 0.570796...; data gives 0.570785... at N=10^18, delta < 1.1e-5)
//
// Therefore the FULL deterministic asymptotic has THREE terms:
//
//   b1(N) = N * ((pi/2)*N - 1)  +  E3(N)
//         = (pi/2)*N^2 - N      +  E3(N)
//
// The oscillatory residual E3 is governed by the Hardy-Voronoi expansion of
// the Gauss circle error.  The dominant mode of Delta_L(m) is:
//   Delta_L(m) ~ (4/pi) * m^{1/4} * cos(2*pi*sqrt(m) - pi/4)
//
// Summing by stationary phase (sub u=sqrt(m), integrate by parts):
//   E3(N) ~ (4/pi^2) * N^{3/4} * sin(2*pi*sqrt(N) - 5*pi/4)
//            + O(N^{1/4})   (from next integration-by-parts term)
//            + O(N^{3/4}/2^{3/4}) * ...  (from j=2,3,... harmonics)
//
// Therefore E3 oscillates with amplitude ~ N^{3/4}:
//   E3 / N^{0.75} is O(1) -- confirmed empirically to N=10^18.
//   Theoretical amplitude: (4/pi^2)*N^{3/4} ~ 1.28e13 at N=10^18.
//   Observed max |E3|: 1.93e13 at N=9.33e17 (multi-harmonic peak). 
//
// Stationary phase means the OLS alpha of log|E3| vs log(N) relates to the
// Gauss circle exponent theta by:
//   alpha = theta_Gauss + 1/2         (NOT alpha = theta_Gauss + 1)
//   theta_Gauss = alpha - 1/2
//
// Conjecture (Hardy-Landau): theta_Gauss = 1/4  =>  alpha -> 3/4 = 0.750
// Huxley proven bound:        theta_Gauss <= 131/416  =>  alpha <= 131/416 + 1/2 ~ 0.815
// Empirical at N=10^18:       alpha ~ 0.763  =>  theta_Gauss ~ 0.263 (near 1/4)
//
// Display columns:
//   E3 / N^{0.75}         -- bounded iff theta_Gauss <= 1/4 (conjecture); confirmed O(1)
//   E3 / N^{0.815}        -- bounded iff theta_Gauss <= 131/416 (Huxley)
//
// OLS: log|E3| ~ alpha*log(N); theta_Gauss = alpha - 0.5.
//
// ============================================================
// PRECISION
// ============================================================
//
// b1(N) is an exact integer stored in __int128.
// The asymptotic (pi/2)*N*(N-1) is ~pi/2 * N^2.
// At N = 10^18 that is ~1.57e36, requiring ~37 significant digits.
// long double has only ~18-19 digits => catastrophic cancellation.
//
// Fix: use __float128 (113-bit, ~34 decimal digits) for the asymptotic,
// then compute E = (b1 - floor(asym)) + frac(asym) with integer subtraction
// for the large part and floating-point for the small fractional correction.
// This gives E to full long double precision regardless of N.
//
// __float128 arithmetic requires -lquadmath; use -DNO_QUADMATH to fall back
// to long double (accurate only for N <~ 10^9).
//
// ============================================================
// OVERFLOW GUARD
// ============================================================
//
// b1(N) ~ (pi/2)*N^2.  __int128 max ~ 1.7e38.
// Safe limit: (pi/2)*N^2 < 1.7e38  =>  N < ~3.28e18.
// We use N_SAFE_MAX = 3e18 with a hard check.
//
// W_pref uses i128 internally and takes a long long argument.
// C_pref and W_pref are correct for all d <= N-1 <= 3e18 < LLONG_MAX ~ 9.2e18.
//
// ============================================================
// COMPILE
// ============================================================
//
//  g++ -O3 -march=native -fopenmp -std=c++17 delta.cpp -lquadmath -o delta
//  g++ -O3 -march=native -fopenmp -std=c++17 -DNO_QUADMATH delta.cpp -o delta
//
// USAGE
//   ./delta Nmax samples cutoff [--csv]
//
//   Nmax    : largest N to evaluate  (u64, <= 3e18)
//   samples : number of log-spaced sample points in [cutoff, Nmax]
//   cutoff  : left boundary of sample range
//   --csv   : output CSV instead of aligned table
//
// EXAMPLE
//   OMP_NUM_THREADS=$(nproc) ./delta 1000000000000000000 300 1000000000000

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

using u64  = unsigned long long;
using i128 = __int128_t;
using ld   = long double;

#ifndef NO_QUADMATH
#  include <quadmath.h>
   using f128 = __float128;
#  define LOG128(x)  logq(x)
// __float128 literals require -fext-numeric-literals (not always available).
// Use strtoflt128 instead, which always works.
   static inline f128 f128_from_str(const char *s) {
       return strtoflt128(s, nullptr);
   }
#  define PI_HALF_VAL  (f128_from_str("1.57079632679489661923132169163975144"))
#  define ONE_F128     ((__float128)1.0L)
#else
   using f128 = long double;
#  define LOG128(x)  logl(x)
#  define PI_HALF_VAL  (1.57079632679489661923132169163975144L)
#  define ONE_F128     (1.0L)
#endif

// -----------------------------------------------------------------------
// Safe limit: b1(N) ~ (pi/2)*N^2 must fit in i128 (~1.7e38).
// (pi/2)*N^2 < 1.7e38  =>  N < sqrt(2*1.7e38/pi) ~ 3.28e18.
// We use 3e18 to keep intermediate sums well clear of overflow.
// -----------------------------------------------------------------------
static constexpr u64 N_SAFE_MAX = 3000000000000000000ULL;

// -----------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------

static string i128_to_string(i128 x) {
    if (x == 0) return "0";
    bool neg = (x < 0);
    if (neg) x = -x;
    string s;
    while (x) { s += char('0' + (int)(x % 10)); x /= 10; }
    if (neg) s += '-';
    reverse(s.begin(), s.end());
    return s;
}

// Convert i128 to long double without losing high bits.
// Splits into two 64-bit chunks to stay within ld mantissa precision.
static ld i128_to_ld(i128 x) {
    if (x == 0) return 0.0L;
    bool neg = (x < 0);
    if (neg) x = -x;
    // split: x = hi * 2^64 + lo
    u64 lo = (u64)(x);
    u64 hi = (u64)((unsigned __int128)x >> 64);
    ld result = (ld)hi * (ld)(1ULL << 32) * (ld)(1ULL << 32) + (ld)lo;
    return neg ? -result : result;
}

// -----------------------------------------------------------------------
// chi_{-4} prefix sums  (period-4 character: 1,0,-1,0,...)
// -----------------------------------------------------------------------

// C(x) = sum_{n=1}^{x} chi_{-4}(n)  in {0, 1}
static inline long long C_pref(long long x) {
    if (x <= 0) return 0LL;
    return ((x + 3) / 4) - ((x + 1) / 4);
}

// W(x) = sum_{n=1}^{x} n * chi_{-4}(n)
// Full period contributes -2 per 4 terms; handle remainder explicitly.
static inline i128 W_pref(long long x) {
    if (x <= 0) return (i128)0;
    i128 q = x / 4;
    int  r = x % 4;
    i128 total = q * (i128)(-2);
    if (r >= 1) total += (i128)(4 * q + 1);
    if (r >= 3) total -= (i128)(4 * q + 3);
    return total;
}

// -----------------------------------------------------------------------
// b1(N) via hyperbola blocking  [O(sqrt N), exact, result in i128]
//
//   b1(N) = 4 * sum_{d=1}^{N-1} chi(d) * [N*M - d*M*(M+1)/2]
//   where M = floor((N-1)/d).
//
// Blocked over d-ranges sharing the same value of floor((N-1)/d).
// -----------------------------------------------------------------------
static i128 b1_sqrt(u64 N) {
    if (N <= 1) return (i128)0;

    // Overflow guard: caller should check, but defend here too.
    if (N > N_SAFE_MAX) {
        fprintf(stderr, "b1_sqrt: N=%llu exceeds N_SAFE_MAX=%llu, result would overflow i128\n",
                N, N_SAFE_MAX);
        return (i128)0;
    }

    u64  P     = N - 1;   // floor((N-1)/d) for d in [1..P]
    i128 total = 0;
    u64  d     = 1;

    while (d <= P) {
        u64 v    = P / d;
        u64 d_hi = P / v;

        // d_hi <= P < 2^64, safe cast to long long (P <= N-1 <= 3e18 < LLONG_MAX)
        long long lo = (long long)d;
        long long hi = (long long)d_hi;

        long long block_C = C_pref(hi) - C_pref(lo - 1);
        i128      block_W = W_pref(hi) - W_pref(lo - 1);

        // term1 = N * v * block_C.  Max: 3e18 * 3e18 * 1 ~ 9e36 < 1.7e38. Safe.
        i128 term1 = (i128)N * (i128)v * (i128)block_C;
        // term2 = v*(v+1)/2 * block_W.  v*(v+1)/2 <= (3e18)^2/2 ~ 4.5e36,
        // block_W <= O(v^2/4) so the product is safe as well.
        i128 vv    = (i128)v * (i128)(v + 1) / 2;
        i128 term2 = vv * block_W;

        total += term1 - term2;
        d = d_hi + 1;
    }

    return (i128)4 * total;
}

// -----------------------------------------------------------------------
// High-precision asymptotic and error computation
//
//   asym(N) = (pi/2) * N * (N-1)      [the correct two-term deterministic part]
//
// E(N) = b1(N) - asym(N)  computed as:
//   integer part: b1 - floor(asym)   [exact, using i128]
//   fractional correction: asym - floor(asym)  [using f128]
// This gives E to full ld precision regardless of magnitude.
// -----------------------------------------------------------------------

static f128 asym_f128(u64 N) {
    // N * ((pi/2)*N - 1)  =  (pi/2)*N^2 - N
    // Full three-term deterministic asymptotic. The -N term accounts for the
    // -1 per step from origin exclusion (Delta_A = Delta_L - 1) and the
    // mean of Delta_A converging to (pi/2-1).
    // Verified: E_corr/N -> pi/2-1=0.570796... at N=10^18 (observed 0.570785).
    f128 x    = (f128)N;
    f128 half = PI_HALF_VAL;
    return x * (half * x - ONE_F128);
}

// Returns E3(N) = b1(N) - N*((pi/2)*N - 1) to full ld precision.
// E3 is the oscillatory residual after removing all deterministic terms.
// b1_val must be the exact i128 result from b1_sqrt(N).
static ld compute_E(u64 N, i128 b1_val) {
    f128 asym   = asym_f128(N);
    i128 ai     = (i128)asym;          // floor(asym), exact truncation
    f128 afrac  = asym - (f128)ai;     // fractional part, in [0,1)
    // E = (b1 - ai) + (-afrac)  -- integer part exact, float part tiny
    i128 idiff  = b1_val - ai;
    return (ld)idiff - (ld)afrac;
}

// -----------------------------------------------------------------------
// Finite-difference mode helpers
//
// Identity (Abel summation):
//   b1(N) = sum_{m=1}^{N-1} A(m)
// Therefore:
//   A(N) = b1(N+1) - b1(N)   [exact integer arithmetic]
//
// Gauss circle error:
//   Delta(N) = A(N) - pi*N
//
// OLS on log|Delta(N)| vs log(N) gives theta_Gauss DIRECTLY
// (no +1/2 stationary-phase shift needed, since we are no longer
// looking at the integral of Delta but at Delta itself).
// Conjecture: theta_Gauss = 1/4.
// -----------------------------------------------------------------------

// Returns A(N) = number of lattice points in disk x^2+y^2 <= N (excluding origin).
// Computed exactly via b1_sqrt(N+1) - b1_sqrt(N) using i128 arithmetic.
// N+1 must also be <= N_SAFE_MAX.
static i128 A_exact(u64 N) {
    if (N + 1 > N_SAFE_MAX) {
        fprintf(stderr, "A_exact: N+1=%llu exceeds N_SAFE_MAX\n", N + 1);
        return (i128)0;
    }
    return b1_sqrt(N + 1) - b1_sqrt(N);
}

// Returns Delta(N) = A(N) - pi*N  (Gauss circle error, excluding origin).
// Uses f128 for pi*N to avoid cancellation.
static ld compute_Delta(u64 N, i128 A_val) {
    // pi * N  -- need f128 precision since N can be ~10^18
    // f128 has ~34 decimal digits; pi*N ~ 3.14e18, well within range.
    static const f128 PI_VAL = f128_from_str("3.14159265358979323846264338327950288");
    f128 piN    = PI_VAL * (f128)N;
    i128 piN_i  = (i128)piN;           // floor(pi*N)
    f128 piN_fr = piN - (f128)piN_i;   // fractional part
    i128 idiff  = A_val - piN_i;
    return (ld)idiff - (ld)piN_fr;
}

// -----------------------------------------------------------------------
// Progress bar (weighted by sqrt(N) to match O(sqrt N) compute cost)
// -----------------------------------------------------------------------
struct Progress {
    double total_w;
    double done_w;      // protected by lock
    double t_start;
    double last_render;
    omp_lock_t lock;
    FILE *out;

    void init(const vector<u64> &ns) {
        total_w = 0.0;
        for (u64 n : ns) total_w += sqrt((double)n);
        done_w = 0.0;
        t_start = last_render = omp_get_wtime();
        omp_init_lock(&lock);
        out = fopen("/dev/tty", "w");
        if (!out) out = stderr;
        render();
    }

    void tick(double w) {
        bool should_render = false;
        omp_set_lock(&lock);
        done_w += w;
        double now = omp_get_wtime();
        if (now - last_render >= 0.5) { last_render = now; should_render = true; }
        omp_unset_lock(&lock);
        if (should_render) render();
    }

    void finish() {
        omp_set_lock(&lock);
        double elapsed = omp_get_wtime() - t_start;
        fprintf(out, "\r%-90s\r  done in %.1fs\n", "", elapsed);
        fflush(out);
        omp_unset_lock(&lock);
        omp_destroy_lock(&lock);
        if (out != stderr) fclose(out);
        out = nullptr;
    }

private:
    void render() {
        omp_set_lock(&lock);
        double frac    = (total_w > 0.0) ? min(1.0, done_w / total_w) : 0.0;
        double elapsed = omp_get_wtime() - t_start;

        char eta[32] = "  --:--";
        if (frac > 0.002) {
            double rem = elapsed / frac * (1.0 - frac);
            if (rem < 3600.0)
                snprintf(eta, sizeof(eta), " eta %2d:%02d", (int)(rem/60), (int)rem%60);
            else
                snprintf(eta, sizeof(eta), " eta %dh%02dm", (int)(rem/3600), ((int)(rem/60))%60);
        }

        const int W = 30;
        int filled = (int)(frac * W + 0.5);
        char bar[W+3]; bar[0]='[';
        for (int i=0; i<W; i++) bar[1+i] = (i < filled) ? '#' : '-';
        bar[W+1]=']'; bar[W+2]='\0';

        fprintf(out, "\r  %s %.1f%%  %.0fs%s   ", bar, frac*100.0, elapsed, eta);
        fflush(out);
        omp_unset_lock(&lock);
    }
} g_progress;

// -----------------------------------------------------------------------
// OLS accumulator  (log|E3| ~ alpha*log(N) + C)
// Via stationary phase: alpha = theta_Gauss + 1/2.
// Conjecture: alpha -> 3/4 = 0.75.  Empirical: ~0.763 at N=10^18.
// Protected by a mutex; updates are infrequent (once per point, serial phase).
// -----------------------------------------------------------------------
struct OLS {
    long double sum1=0, sumx=0, sumy=0, sumxx=0, sumxy=0;
    int n=0;

    void add(u64 N, ld E) {
        if (E == 0.0L) return;
        ld logn = logl((ld)N);
        ld loge = logl(fabsl(E));
        sum1  += 1;   sumx  += logn;  sumy  += loge;
        sumxx += logn*logn;           sumxy += logn*loge;
        n++;
    }

    // Returns alpha (OLS slope of log|E3| vs log(N)).
    // Returns false if not enough data.
    bool result(ld &alpha_out) const {
        if (n < 3) return false;
        ld denom = sum1*sumxx - sumx*sumx;
        if (fabsl(denom) == 0.0L) return false;
        alpha_out = (sum1*sumxy - sumx*sumy) / denom;
        return true;
    }
};

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 4) {
        fprintf(stderr,
            "Usage: ./delta Nmax samples cutoff [--csv] [--delta]\n"
            "\n"
            "  Nmax    : largest N (<= %llu)\n"
            "  samples : number of log-spaced sample points in [cutoff, Nmax]\n"
            "  cutoff  : left boundary of sample range\n"
            "  --csv   : emit CSV instead of table\n"
            "  --delta : finite-difference mode: compute A(N)=b1(N+1)-b1(N) and\n"
            "            Delta(N)=A(N)-pi*N directly.  OLS gives theta_Gauss directly\n"
            "            (no +1/2 stationary-phase shift).  Conjecture: theta -> 0.25.\n"
            "\n"
            "Example:\n"
            "  OMP_NUM_THREADS=$(nproc) ./delta 1000000000000000000 300 1000000000000\n"
            "  OMP_NUM_THREADS=$(nproc) ./delta 1000000000000000000 400 1000000000 --delta\n",
            N_SAFE_MAX);
        return 1;
    }

    u64  Nmax     = strtoull(argv[1], nullptr, 10);
    int  samples  = max(1, atoi(argv[2]));
    u64  cutoff   = strtoull(argv[3], nullptr, 10);
    bool csv_out  = false;
    bool delta_mode = false;
    for (int a = 4; a < argc; ++a) {
        if (string(argv[a]) == "--csv")   csv_out    = true;
        if (string(argv[a]) == "--delta") delta_mode = true;
    }

    // ---- Validation ----
    // In delta mode we evaluate b1(N+1), so Nmax+1 must fit in N_SAFE_MAX.
    u64 safe_limit = delta_mode ? N_SAFE_MAX - 1 : N_SAFE_MAX;
    if (Nmax > safe_limit) {
        fprintf(stderr, "ERROR: Nmax=%llu exceeds safe limit %llu%s\n",
                Nmax, safe_limit,
                delta_mode ? " (delta mode: need N+1 <= N_SAFE_MAX)" : " (b1 would overflow i128)");
        return 1;
    }
    if (cutoff < 2) cutoff = 2;
    if (cutoff > Nmax) {
        fprintf(stderr, "ERROR: cutoff=%llu > Nmax=%llu\n", cutoff, Nmax);
        return 1;
    }

    // ---- Generate log-spaced sample points in [cutoff, Nmax] ----
    vector<u64> Ns;
    Ns.reserve(samples);

    if (samples == 1) {
        Ns.push_back(Nmax);
    } else {
        ld log_lo = logl((ld)cutoff);
        ld log_hi = logl((ld)Nmax);
        u64 prev  = 0;
        for (int i = 0; i < samples; ++i) {
            ld t  = (ld)i / (ld)(samples - 1);
            u64 v = (u64)expl(log_lo + t * (log_hi - log_lo));
            v = max(v, cutoff);
            v = min(v, Nmax);
            if (v > prev) { Ns.push_back(v); prev = v; }
        }
        if (Ns.empty() || Ns.back() != Nmax) Ns.push_back(Nmax);
    }

    Ns.erase(unique(Ns.begin(), Ns.end()), Ns.end());
    size_t total = Ns.size();
    if (total == 0) { fprintf(stderr, "No samples generated.\n"); return 1; }

    // ---- Allocate result storage ----
    struct Row {
        u64  N;
        i128 b1;
        ld   E;         // E3(N)   [integral mode]   or Delta(N) [delta mode]
        i128 A;         // A(N)    [delta mode only]
        bool overflow;
    };
    vector<Row> rows(total);

    // ---- Parallel compute ----
    g_progress.init(Ns);
    omp_set_dynamic(0);

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < total; ++i) {
        u64  n        = Ns[i];
        Row &r        = rows[i];
        r.N           = n;
        r.overflow    = (n > safe_limit);

        if (!r.overflow) {
            if (delta_mode) {
                // Two b1 evaluations: O(sqrt(N+1)) + O(sqrt(N)) ~ 2*O(sqrt N)
                i128 b1_n1 = b1_sqrt(n + 1);
                i128 b1_n  = b1_sqrt(n);
                r.b1 = b1_n;
                r.A  = b1_n1 - b1_n;           // A(N) exact
                r.E  = compute_Delta(n, r.A);   // Delta(N) = A(N) - pi*N
            } else {
                r.b1 = b1_sqrt(n);
                r.A  = 0;
                r.E  = compute_E(n, r.b1);
            }
        } else {
            r.b1 = 0; r.A = 0; r.E = 0.0L;
        }

        g_progress.tick(sqrt((double)n));
    }

    g_progress.finish();

    // ---- OLS regression ----
    // Integral mode:  log|E3|    ~ alpha*log(N);  theta = alpha - 0.5
    // Delta mode:     log|Delta| ~ theta*log(N);  theta read directly
    OLS ols;
    for (size_t i = 0; i < total; ++i) {
        if (!rows[i].overflow && rows[i].E != 0.0L)
            ols.add(rows[i].N, rows[i].E);
    }

    // ---- Print header ----
    if (csv_out) {
        if (delta_mode)
            printf("N,A_N,Delta_N,Delta_over_N0.25,Delta_over_N0.31490,theta_local\n");
        else
            printf("N,b1,E3,E3_over_N0.75,E3_over_N0.815,alpha_local\n");
    } else {
        printf("\n");
        if (delta_mode) {
            printf("  FINITE DIFFERENCE MODE: A(N) = b1(N+1) - b1(N),  Delta(N) = A(N) - pi*N\n");
            printf("  Identity verified: A(N) is exact count of lattice points in disk x^2+y^2<=N (excl. origin).\n");
            printf("  OLS on log|Delta(N)| gives theta_Gauss DIRECTLY (no +1/2 stationary-phase shift).\n");
            printf("  Conjecture (Hardy-Landau): theta -> 1/4 = 0.25000\n");
            printf("  Huxley proven bound:       theta <= 131/416 ~ 0.31490\n");
            printf("  Dominant mode: Delta ~ (4/pi)*N^{1/4}*cos(2*pi*sqrt(N)-pi/4)\n");
        } else {
            printf("  b1(N) = N*((pi/2)*N - 1) + E3(N)  =  (pi/2)*N^2 - N + E3(N)\n");
            printf("  E3(N) ~ (4/pi^2)*N^{3/4}*sin(2*pi*sqrt(N)+phi) + higher harmonics\n");
            printf("  Via stationary phase: alpha_OLS = theta_Gauss + 1/2.\n");
            printf("  Conjecture (Hardy-Landau): theta=1/4  =>  alpha->3/4=0.75, E3/N^0.75 bounded.\n");
            printf("  Huxley proven bound:       theta<=131/416  =>  alpha<=0.8149, E3/N^0.815 bounded.\n");
            printf("  Empirical at N=10^18:     alpha~0.763, theta~0.263 (near 1/4).\n");
        }
        printf("\n");
#ifdef NO_QUADMATH
        printf("  [WARNING] Compiled without quadmath; precision degraded for N > ~1e9\n\n");
#endif
        if (delta_mode) {
            printf("  %18s  %28s  %18s  %14s  %14s  %10s\n",
                   "N", "A(N)", "Delta(N)", "D/N^0.25", "D/N^0.3149", "theta_sl");
        } else {
            printf("  %18s  %28s  %18s  %14s  %14s  %10s\n",
                   "N", "b1(N)", "E3(N)", "E3/N^0.75", "E3/N^0.815", "alpha_sl");
        }
        printf("  %s\n", string(110, '-').c_str());
    }

    // ---- Print rows, compute running slope ----
    ld prev_logE = 0.0L, prev_logN = 0.0L;

    for (size_t i = 0; i < total; ++i) {
        const Row &r = rows[i];
        if (r.overflow) {
            if (!csv_out)
                printf("  %18llu  [SKIPPED: N > safe_limit]\n", r.N);
            continue;
        }

        ld   absE  = fabsl(r.E);
        ld   logN  = logl((ld)r.N);

        ld   slope_local = NAN;
        if (prev_logE != 0.0L && absE > 0.0L && prev_logN > 0.0L) {
            ld logE_now = logl(absE);
            slope_local = (logE_now - prev_logE) / (logN - prev_logN);
        }
        if (absE > 0.0L) { prev_logE = logl(absE); prev_logN = logN; }

        if (delta_mode) {
            // Delta mode: exponents are theta_Gauss directly
            ld d025  = powl((ld)r.N, 0.25L);
            ld d0315 = powl((ld)r.N, 0.31490L);
            ld s025  = r.E / d025;
            ld s0315 = r.E / d0315;
            string Astr = i128_to_string(r.A);
            if (csv_out) {
                printf("%llu,%s,%.10Lg,%.12Lg,%.12Lg",
                       r.N, Astr.c_str(), r.E, s025, s0315);
                if (!isnanl(slope_local)) printf(",%.8Lg\n", slope_local);
                else                      printf(",\n");
            } else {
                printf("  %18llu  %28s  %18.4Lf  %14.8Lf  %14.8Lf",
                       r.N, Astr.c_str(), r.E, s025, s0315);
                if (!isnanl(slope_local)) printf("  %+10.5Lf\n", slope_local);
                else                      printf("  %10s\n", "    -");
            }
        } else {
            // Integral mode: exponents are alpha = theta + 1/2
            ld n075  = powl((ld)r.N, 0.75L);
            ld n0815 = powl((ld)r.N, 0.8149L);
            ld s075  = r.E / n075;
            ld s0815 = r.E / n0815;
            string b1str = i128_to_string(r.b1);
            if (csv_out) {
                printf("%llu,%s,%.10Lg,%.12Lg,%.12Lg",
                       r.N, b1str.c_str(), r.E, s075, s0815);
                if (!isnanl(slope_local)) printf(",%.8Lg\n", slope_local);
                else                      printf(",\n");
            } else {
                printf("  %18llu  %28s  %18.4Lf  %14.8Lf  %14.8Lf",
                       r.N, b1str.c_str(), r.E, s075, s0815);
                if (!isnanl(slope_local)) printf("  %+10.5Lf\n", slope_local);
                else                      printf("  %10s\n", "    -");
            }
        }
    }

    // ---- OLS summary ----
    ld alpha_ols;
    if (!csv_out) {
        printf("\n  %s\n", string(110, '-').c_str());
        if (ols.result(alpha_ols)) {
            if (delta_mode) {
                printf("\n  OLS regression: log|Delta(N)| ~ theta*log(N) + C  over %d points\n", ols.n);
                printf("    theta_Gauss  = %.8Lf  (DIRECT -- no stationary-phase correction needed)\n", alpha_ols);
                printf("    Conjecture (Hardy-Landau): theta -> 0.25000\n");
                printf("    Huxley proven bound:       theta <= 0.31490  (131/416)\n");
                printf("    Dominant mode: Delta ~ (4/pi)*N^{1/4}*cos(2*pi*sqrt(N)-pi/4)\n");
            } else {
                printf("\n  OLS regression: log|E3(N)| ~ alpha*log(N) + C  over %d points\n", ols.n);
                printf("    alpha     = %.8Lf\n", alpha_ols);
                printf("    theta_Gauss = alpha - 1/2 = %.8Lf  (via stationary phase)\n", alpha_ols - 0.5L);
                printf("    Conjecture (Hardy-Landau): alpha -> 0.75000  (theta -> 0.25000)\n");
                printf("    Huxley proven bound:       alpha <= 0.81490  (theta <= 0.31490)\n");
                printf("    Dominant mode: E3 ~ (4/pi^2)*N^{3/4}*sin(2*pi*sqrt(N)+phi)\n");
            }
        } else {
            printf("\n  OLS: not enough valid points (%d)\n", ols.n);
        }
        const Row &last = rows.back();
        ld nfinal = (ld)last.N;
        if (delta_mode) {
            printf("\n  Final Delta(N) / N^0.250 = %.10Lf  (N = %llu)\n",
                   last.E / powl(nfinal, 0.25L), last.N);
            printf("  Final Delta(N) / N^0.315 = %.10Lf  (Huxley column)\n",
                   last.E / powl(nfinal, 0.31490L));
            printf("  Theoretical amplitude at this N: (4/pi)*N^0.25 = %.4Le\n",
                   (4.0L / (long double)M_PIl) * powl(nfinal, 0.25L));
        } else {
            printf("\n  Final E3(N) / N^0.750 = %.10Lf  (N = %llu)\n",
                   last.E / powl(nfinal, 0.75L), last.N);
            printf("  Final E3(N) / N^0.815 = %.10Lf  (Huxley column)\n",
                   last.E / powl(nfinal, 0.8149L));
            printf("  Theoretical amplitude at this N: (4/pi^2)*N^0.75 = %.4Le\n",
                   (long double)(4.0L/((long double)M_PIl*(long double)M_PIl)) * powl(nfinal, 0.75L));
        }
    } else {
        if (ols.result(alpha_ols)) {
            if (delta_mode)
                printf("# OLS: theta_Gauss=%.10Lg  n_pts=%d\n", alpha_ols, ols.n);
            else
                printf("# OLS: alpha=%.10Lg  theta_if_linear_removed=%.10Lg  n_pts=%d\n",
                       alpha_ols, alpha_ols - 1.0L, ols.n);
        }
    }

    return 0;
}

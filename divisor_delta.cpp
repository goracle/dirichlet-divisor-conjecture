// divisor_delta.cpp -- direct probe of the Dirichlet divisor problem.
//
// Why this exists
// ---------------
// main.cpp measures err_osc(n) = a(n) - [n^2/2 log n + (g-3/4) n^2 + n/4].
// That quantity is EXACTLY  int_0^n Delta(t) dt  (a(n) = sum_{k<n} (n-k) tau(k)),
// which is Theta(n^{3/4}) unconditionally (Voronoi).  So the OLS exponent
// ~0.25 in main.cpp is a theorem about an order-1 Riesz mean and says nothing
// about the Dirichlet divisor conjecture Delta(x) = O(x^{1/4+eps}).
// This program measures Delta(x) itself.
//
// Definitions
// -----------
//   D(x)     = sum_{m<=x} tau(m)
//   Delta(x) = D(x) - x(log x + 2g - 1) - 1/4        (mean-zero convention)
// Samples are taken at HALF-INTEGERS x = n + 1/2.  At integers D jumps by
// tau(n), so D(n) is biased by ~tau(n)/2 ~ (log n)/2 relative to the mean-zero
// Delta; at n+1/2 there is no jump and the bias vanishes.
// (The usual Delta without the 1/4 differs by a constant; irrelevant for
// the conjecture, but the 1/4 makes the sample mean -> 0, a free sanity check.)
//
// Output per decade of x:  mean, mean-square, min, max of  v = Delta/x^{1/4}.
//   Expected (Cramer / Ingham mean square, log-uniform sampling):
//       E[v^2] -> (3/2) * zeta(3/2)^4 / (6 pi^2 zeta(3)) = 0.98143
//       E[v]   -> 0
//   Conjecture alpha = 1/4  <=>  v stays O(x^eps): decade maxima grow at most
//   very slowly (Hafner-type Omega results force a slow log-power growth of
//   the true sup, which sparse sampling will NOT reach).
//   If alpha > 1/4, decade maxima would grow like x^{alpha-1/4}.
//
// Build:
//   g++ -O3 -march=native -fopenmp -std=gnu++17 divisor_delta.cpp -lquadmath -o divisor_delta
// Run:
//   OMP_NUM_THREADS=$(nproc) ./divisor_delta [Nmax [ratio [nmin]]] [--csv=path]
//   defaults: Nmax=1e18  ratio=1.001  nmin=1e4
//   cost ~ O(sqrt(n)) per point with plain 64-bit division; Nmax <= 1.8e19.
//
// Errors are never silent: any failed self-check or bad argument aborts with
// a nonzero exit status.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <atomic>
#include <omp.h>
#include <quadmath.h>

typedef unsigned long long ull;
typedef unsigned __int128 u128;

static const __float128 GAMMA_Q = 0.577215664901532860606512090082402431042Q;
static const long double MS_EXPECTED = 0.98142596626326840562L;  // (3/2) c, c = zeta(3/2)^4/(6 pi^2 zeta(3))

[[noreturn]] static void die(const char *msg) {
    fprintf(stderr, "divisor_delta: FATAL: %s\n", msg);
    exit(1);
}

static ull isqrt_ull(ull n) {
    ull r = (ull)sqrtl((long double)n);
    while ((u128)(r + 1) * (r + 1) <= n) r++;
    while ((u128)r * r > n) r--;
    return r;
}

// D(n) = 2 * sum_{k<=r} floor(n/k) - r^2,  r = floor(sqrt n).  Hardware u64 division.
static u128 D_hyperbola(ull n) {
    if (n == 0) return 0;
    ull r = isqrt_ull(n);
    u128 s = 0;
    for (ull k = 1; k <= r; k++) s += n / k;
    return 2 * s - (u128)r * r;
}

// Delta(n + 1/2) given D(n).  __float128 subtraction: D ~ 4e19 at n = 1e18,
// 113-bit mantissa leaves ~1e-14 absolute error, vs |Delta| ~ 1e4..1e5.
static long double delta_half(ull n, u128 D) {
    __float128 x = (__float128)n + 0.5Q;
    __float128 mainterm = x * (logq(x) + 2 * GAMMA_Q - 1) + 0.25Q;
    return (long double)((__float128)D - mainterm);
}

static ull parse_count(const char *s, const char *what) {
    char *end = nullptr;
    long double v = strtold(s, &end);
    if (end == s || !(v >= 0) || v > 1.8e19L) {
        fprintf(stderr, "divisor_delta: bad %s: '%s' (need 0 <= value <= 1.8e19)\n", what, s);
        exit(1);
    }
    return (ull)v;
}

// Self-check hyperbola D(n) against a sieve of tau, including every n <= 300,
// all perfect squares and neighbours up to 2e6, and a strided sample.
static void self_check() {
    const ull M = 2000000;
    std::vector<unsigned> tau(M + 1, 0);
    for (ull d = 1; d <= M; d++)
        for (ull m = d; m <= M; m += d) tau[m]++;
    std::vector<u128> D(M + 1, 0);
    for (ull m = 1; m <= M; m++) D[m] = D[m - 1] + tau[m];
    auto chk = [&](ull n) {
        if (D_hyperbola(n) != D[n]) {
            fprintf(stderr, "divisor_delta: FATAL: D mismatch at n=%llu\n", n);
            exit(1);
        }
    };
    for (ull n = 1; n <= 300; n++) chk(n);
    for (ull r = 1; r * r <= M; r++) {
        for (ull t = r * r - 1; t <= r * r + 1; t++) if (t >= 1 && t <= M) chk(t);
    }
    for (ull n = 301; n <= M; n += 997) chk(n);
    // known value: D(10^6) = 13970034
    if (D_hyperbola(1000000ULL) != (u128)13970034ULL) die("D(10^6) != 13970034");
    // mean of Delta over half-integers must be ~0 (validates the 1/4 and 2g-1 constants)
    long double acc = 0; ull cnt = 0;
    for (ull n = 1000; n <= M - 1; n++) { acc += delta_half(n, D[n]); cnt++; }
    long double mean = acc / (long double)cnt;
    if (fabsl(mean) > 0.05L) {
        fprintf(stderr, "divisor_delta: FATAL: mean Delta over [1e3,2e6] = %.4Lf (expected ~0)\n", mean);
        exit(1);
    }
    fprintf(stderr, "self-check ok (hyperbola D vs sieve; mean Delta = %.5Lf)\n", mean);
}

int main(int argc, char **argv) {
    ull Nmax = 1000000000000000000ULL;
    double ratio = 1.001;
    ull nmin = 10000;
    const char *csv_path = nullptr;

    int pos = 0;
    for (int a = 1; a < argc; a++) {
        if (!strncmp(argv[a], "--csv=", 6)) { csv_path = argv[a] + 6; continue; }
        if (!strcmp(argv[a], "-h") || !strcmp(argv[a], "--help")) {
            printf("Usage: ./divisor_delta [Nmax [ratio [nmin]]] [--csv=path]\n");
            return 0;
        }
        if (pos == 0) Nmax = parse_count(argv[a], "Nmax");
        else if (pos == 1) { ratio = atof(argv[a]); if (!(ratio > 1.0)) die("ratio must be > 1"); }
        else if (pos == 2) nmin = parse_count(argv[a], "nmin");
        else die("too many positional arguments");
        pos++;
    }
    if (nmin < 100 || nmin >= Nmax) die("need 100 <= nmin < Nmax");

    self_check();

    // geometric grid, strictly increasing, always includes Nmax
    std::vector<ull> ns;
    {
        ull n = nmin;
        while (n < Nmax) {
            ns.push_back(n);
            ull nx = (ull)((long double)n * (long double)ratio);
            n = (nx > n) ? nx : n + 1;
        }
        ns.push_back(Nmax);
    }
    const long P = (long)ns.size();
    fprintf(stderr, "%ld sample points in [%llu, %llu], ratio %.5f, %d threads\n",
            P, nmin, Nmax, ratio, omp_get_max_threads());

    long double total_work = 0;
    for (ull n : ns) total_work += sqrtl((long double)n);

    std::vector<long double> delta(P, 0.0L);
    std::atomic<ull> done_work{0};
    std::atomic<int> last_pct{-1};

    // largest n first for load balance
    #pragma omp parallel for schedule(dynamic, 1)
    for (long j = 0; j < P; j++) {
        long i = P - 1 - j;
        u128 D = D_hyperbola(ns[i]);
        delta[i] = delta_half(ns[i], D);
        ull w = done_work.fetch_add((ull)sqrtl((long double)ns[i])) + (ull)sqrtl((long double)ns[i]);
        int pct = (int)(100.0L * (long double)w / total_work);
        int prev = last_pct.load();
        if (pct > prev && last_pct.compare_exchange_strong(prev, pct)) {
            #pragma omp critical
            { fprintf(stderr, "\r  %3d%%", pct); fflush(stderr); }
        }
    }
    fprintf(stderr, "\n");

    for (long i = 0; i < P; i++)
        if (!std::isfinite((double)delta[i])) die("non-finite Delta produced");

    if (csv_path) {
        FILE *f = fopen(csv_path, "w");
        if (!f) die("cannot open csv path for writing");
        fprintf(f, "n,delta,delta_over_x14\n");
        for (long i = 0; i < P; i++) {
            long double x = (long double)ns[i] + 0.5L;
            fprintf(f, "%llu,%.6Lf,%.9Lf\n", ns[i], delta[i], delta[i] / powl(x, 0.25L));
        }
        if (fclose(f) != 0) die("error closing csv file");
    }

    // per-decade statistics of v = Delta / x^{1/4}
    struct Dec { long cnt = 0; long double s1 = 0, s2 = 0, vmin = 1e30L, vmax = -1e30L; ull nmax_at = 0, nmin_at = 0; };
    std::vector<Dec> dec(21);
    for (long i = 0; i < P; i++) {
        long double x = (long double)ns[i] + 0.5L;
        long double v = delta[i] / powl(x, 0.25L);
        int k = (int)floorl(log10l((long double)ns[i]));
        if (k < 0 || k > 19) die("decade index out of range");
        Dec &d = dec[k];
        d.cnt++; d.s1 += v; d.s2 += v * v;
        if (v > d.vmax) { d.vmax = v; d.nmax_at = ns[i]; }
        if (v < d.vmin) { d.vmin = v; d.nmin_at = ns[i]; }
    }

    printf("v = Delta(x)/x^{1/4} at x = n+1/2.   Expected: mean -> 0, mean-square -> %.5Lf\n", MS_EXPECTED);
    printf("%-8s %7s %10s %10s %10s %10s\n", "decade", "points", "mean", "mean_sq", "min", "max");
    std::vector<double> fx, fy;
    for (int k = 0; k <= 19; k++) {
        const Dec &d = dec[k];
        if (d.cnt == 0) continue;
        long double amax = fmaxl(fabsl(d.vmax), fabsl(d.vmin));
        printf("1e%-6d %7ld %10.5Lf %10.5Lf %10.4Lf %10.4Lf\n", k, d.cnt, d.s1 / d.cnt, d.s2 / d.cnt, d.vmin, d.vmax);
        if (k >= 8 && d.cnt >= 200) { fx.push_back((k + 0.5) * log(10.0)); fy.push_back((double)logl(amax)); }
    }

    // slope of log(decade max |v|) vs log x over decades >= 1e8.
    // alpha = 1/4 predicts ~0 (plus slow sampling drift); alpha > 1/4 predicts >= alpha-1/4.
    if (fx.size() >= 3) {
        double mx = 0, my = 0;
        for (size_t i = 0; i < fx.size(); i++) { mx += fx[i]; my += fy[i]; }
        mx /= fx.size(); my /= fx.size();
        double sxy = 0, sxx = 0;
        for (size_t i = 0; i < fx.size(); i++) { sxy += (fx[i] - mx) * (fy[i] - my); sxx += (fx[i] - mx) * (fx[i] - mx); }
        printf("\nslope of log(decade max|v|) vs log x, decades >= 1e8: %.5f  (alpha=1/4 predicts ~0)\n", sxy / sxx);
        printf("NOTE: sampled decade maxima are a weak sup-probe (points are ~random phase; Delta peaks are rare).\n"
               "      The mean-square column is the sharper statistic; compare it with %.5Lf.\n", MS_EXPECTED);
    }
    return 0;
}

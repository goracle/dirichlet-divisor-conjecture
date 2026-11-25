/* divisor_peaks.cpp -- hunt for LARGE values of the Dirichlet divisor error
 *
 *     Delta(x) = D(x) - x(log x + 2g - 1) - 1/4,     D(x) = sum_{n<=x} tau(n)
 *
 * Why: random sampling of Delta (divisor_delta.cpp) sees the bulk of the
 * distribution but almost never the rare peaks, and the divisor conjecture
 * Delta = O(x^{1/4+eps}) is a statement about the peaks.
 *
 * Method (two stages, per decade of x):
 *   1. PROXY SCAN.  Voronoi:  Delta(x)/x^{1/4} ~ (1/(pi*sqrt2)) sum_k tau(k) k^{-3/4} cos(4 pi sqrt(k x) - pi/4).
 *      Truncate at K terms, put t = sqrt(x), and scan t on a fine grid (phasor
 *      recurrence, float128 phase seeds).  Local maxima of the truncated sum are
 *      the places where the low-frequency phases line up -- candidate peaks.
 *      (The truncated series is NOT accurate pointwise at this x; it is only a
 *      way to choose where to look.  Same idea as the resonance method.)
 *   2. EXACT REFINEMENT.  Around each candidate x_c take a window of L integers,
 *      get tau(n) for the whole window with a segmented sieve, get D(a-1) once
 *      by the O(sqrt n) hyperbola method, and walk the window exactly.
 *      Delta only goes UP at integers (jump tau(n)) and DOWN between them, so
 *      the true local extrema are  Delta+(n) = D(n)-M(n)  and  Delta-(n) = D(n-1)-M(n).
 *   3. CONTROL.  The same exact-window machinery at uniformly random x.
 *      Peak-hunt beating control is the evidence that the hunt works.
 *
 * Build:  g++ -O3 -march=native -fopenmp -std=gnu++17 divisor_peaks.cpp -lquadmath -o divisor_peaks
 * Run:    OMP_NUM_THREADS=$(nproc) ./divisor_peaks --dec 10 16 --csv=peaks.csv | tee peaks.txt
 *
 * Self-checks run first (hyperbola D vs sieve, window tau vs sieve, window walk vs
 * independent hyperbola); any mismatch aborts with exit code 2.  The same
 * end-of-window hyperbola cross-check runs on the first two windows of every
 * decade.  Valid for x up to ~4e18 (u64 arithmetic).
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <omp.h>
#include <quadmath.h>

typedef uint64_t u64;
typedef unsigned __int128 u128;
typedef __int128 i128;
typedef __float128 f128;

static const f128 GAMMA_Q = 0.57721566490153286060651209008240243104215933593992Q;
static const long double GAMMA_L = 0.57721566490153286060651209008240243104215933593992L;

/* ---------- integer helpers ---------- */
static u64 isqrt64(u64 n) {
    u64 r = (u64)sqrtl((long double)n);
    while (r * r > n) r--;
    while ((r + 1) * (r + 1) <= n) r++;
    return r;
}

/* D(n) = sum_{m<=n} tau(m) = 2*sum_{k<=r} floor(n/k) - r^2 */
static i128 D_hyp(u64 n) {
    if (n == 0) return 0;
    u64 r = isqrt64(n);
    u128 s = 0;
    for (u64 k = 1; k <= r; k++) s += n / k;
    return (i128)(2 * s) - (i128)((u128)r * r);
}

/* M(x) = x(log x + 2g - 1) + 1/4, in float128 */
static f128 M_q(u64 n) {
    f128 x = (f128)n;
    return x * (logq(x) + 2 * GAMMA_Q - 1) + 0.25Q;
}

/* primes <= limit (segmented) */
static std::vector<uint32_t> gen_primes(u64 limit) {
    std::vector<uint32_t> pr;
    u64 sq = isqrt64(limit) + 1;
    std::vector<char> small(sq + 1, 1);
    std::vector<uint32_t> base;
    for (u64 i = 2; i <= sq; i++) {
        if (small[i]) {
            base.push_back((uint32_t)i);
            for (u64 j = i * i; j <= sq; j += i) small[j] = 0;
        }
    }
    for (uint32_t p : base) if ((u64)p <= limit) pr.push_back(p);
    const u64 S = (u64)1 << 24;
    for (u64 lo = sq + 1; lo <= limit; lo += S) {
        u64 hi = std::min(limit, lo + S - 1);
        std::vector<char> seg(hi - lo + 1, 1);
        for (uint32_t p : base) {
            if ((u64)p * p > hi) break;
            u64 start = std::max((u64)p * p, ((lo + p - 1) / p) * p);
            for (u64 m = start; m <= hi; m += p) seg[m - lo] = 0;
        }
        for (u64 i = 0; i < seg.size(); i++) if (seg[i]) pr.push_back((uint32_t)(lo + i));
    }
    return pr;
}

/* tau(n) for n in [a, a+L) via segmented sieve; needs all primes <= sqrt(a+L-1) */
static void window_tau(u64 a, u64 L, const std::vector<uint32_t> &pr,
                       std::vector<u64> &rem, std::vector<uint32_t> &dc) {
    rem.resize(L);
    dc.assign(L, 1);
    for (u64 i = 0; i < L; i++) rem[i] = a + i;
    u64 b = a + L - 1;
    u64 sq = isqrt64(b);
    for (uint32_t p : pr) {
        if ((u64)p > sq) break;
        u64 m = ((a + p - 1) / p) * p;
        for (; m <= b; m += p) {
            u64 &r = rem[m - a];
            uint32_t e = 0;
            while (r % p == 0) { r /= p; e++; }
            dc[m - a] *= (e + 1);
        }
    }
    for (u64 i = 0; i < L; i++) if (rem[i] > 1) dc[i] *= 2;
}

/* ---------- exact window walk ---------- */
struct WinRes {
    u64 a;
    long double maxp, minm;   /* raw Delta+ max and Delta- min over the window */
    u64 xp, xm;               /* where */
    long double vp, vm;       /* normalised by x^{1/4} */
    double proxy;             /* proxy value that selected this window (0 for control) */
};

/* Walk n = a .. a+L-1.  Delta-(n) = D(n-1)-M(n);  Delta+(n) = Delta-(n)+tau(n);
   Delta-(n+1) = Delta+(n) - [M(n+1)-M(n)],  M(n+1)-M(n) = log(n+1) + n*log1p(1/n) + 2g - 1. */
static bool scan_window(u64 a, u64 L, const std::vector<uint32_t> &pr,
                        std::vector<u64> &rem, std::vector<uint32_t> &dc,
                        bool verify, WinRes &out) {
    i128 Dm1 = D_hyp(a - 1);
    window_tau(a, L, pr, rem, dc);
    long double dlm = (long double)((f128)Dm1 - M_q(a));
    long double maxp = -1e300L, minm = 1e300L, dlp = 0;
    u64 xp = a, xm = a;
    for (u64 n = a; n < a + L; n++) {
        if (dlm < minm) { minm = dlm; xm = n; }
        dlp = dlm + (long double)dc[n - a];
        if (dlp > maxp) { maxp = dlp; xp = n; }
        long double nn = (long double)n;
        long double step = logl(nn + 1.0L) + nn * log1pl(1.0L / nn) + 2.0L * GAMMA_L - 1.0L;
        dlm = dlp - step;
    }
    bool ok = true;
    if (verify) {
        u64 e = a + L - 1;
        long double ref = (long double)((f128)D_hyp(e) - M_q(e));
        if (fabsl(ref - dlp) > 1e-3L) {
            fprintf(stderr, "VERIFY FAIL: window a=%llu L=%llu  walk=%.9Lf  hyperbola=%.9Lf\n",
                    (unsigned long long)a, (unsigned long long)L, dlp, ref);
            ok = false;
        }
    }
    out.a = a; out.maxp = maxp; out.minm = minm; out.xp = xp; out.xm = xm;
    out.vp = maxp / powl((long double)xp, 0.25L);
    out.vm = minm / powl((long double)xm, 0.25L);
    out.proxy = 0;
    return ok;
}

/* ---------- proxy scan ---------- */
struct Cand { double v; long double t; };

static void proxy_scan(f128 t0, double W, int K, int Q, double thr, int per_sign,
                       std::vector<Cand> &pos, std::vector<Cand> &neg) {
    const int B = 4096;
    u64 S = (u64)(W * Q);
    u64 nblocks = (S + B - 1) / B;
    /* tau(k), k<=K */
    std::vector<int> tau(K + 1, 0);
    for (int d = 1; d <= K; d++) for (int m = d; m <= K; m += d) tau[m]++;
    std::vector<f128> sk(K + 1);
    std::vector<double> wk(K + 1), cd(K + 1), sd(K + 1);
    for (int k = 1; k <= K; k++) {
        sk[k] = sqrtq((f128)k);
        wk[k] = (double)tau[k] / pow((double)k, 0.75);
        long double dl = 4.0L * M_PIl * sqrtl((long double)k) / (long double)Q;
        cd[k] = (double)cosl(dl);
        sd[k] = (double)sinl(dl);
    }
    const double inv = 1.0 / (M_PI * sqrt(2.0));
    std::vector<Cand> allpos, allneg;
#pragma omp parallel
    {
        std::vector<double> c(K + 1), s(K + 1), val(B + 2);
        std::vector<Cand> lp, ln;
#pragma omp for schedule(dynamic, 1)
        for (long long b = 0; b < (long long)nblocks; b++) {
            long long i0 = b * (long long)B;
            long long is = i0 - 1;
            f128 tb = t0 + (f128)is / (f128)Q;
            for (int k = 1; k <= K; k++) {
                f128 cyc = 2 * sk[k] * tb;
                f128 fr = cyc - floorq(cyc);
                long double th = 2.0L * M_PIl * (long double)fr - M_PIl / 4.0L;
                c[k] = (double)cosl(th);
                s[k] = (double)sinl(th);
            }
            double *cp = c.data() + 1, *sp = s.data() + 1;
            const double *wp = wk.data() + 1, *cdp = cd.data() + 1, *sdp = sd.data() + 1;
            for (int j = 0; j < B + 2; j++) {
                double acc = 0;
#pragma omp simd reduction(+ : acc)
                for (int k = 0; k < K; k++) acc += wp[k] * cp[k];
                val[j] = acc * inv;
#pragma omp simd
                for (int k = 0; k < K; k++) {
                    double c0 = cp[k];
                    cp[k] = c0 * cdp[k] - sp[k] * sdp[k];
                    sp[k] = sp[k] * cdp[k] + c0 * sdp[k];
                }
            }
            for (int j = 1; j <= B; j++) {
                long long i = is + j;
                if (i < 0 || i >= (long long)S) continue;
                long double t = (long double)t0 + (long double)i / (long double)Q;
                if (val[j] > thr && val[j] > val[j - 1] && val[j] >= val[j + 1]) lp.push_back({val[j], t});
                if (-val[j] > thr && val[j] < val[j - 1] && val[j] <= val[j + 1]) ln.push_back({val[j], t});
            }
        }
#pragma omp critical
        {
            allpos.insert(allpos.end(), lp.begin(), lp.end());
            allneg.insert(allneg.end(), ln.begin(), ln.end());
        }
    }
    std::sort(allpos.begin(), allpos.end(), [](const Cand &x, const Cand &y) { return x.v > y.v; });
    std::sort(allneg.begin(), allneg.end(), [](const Cand &x, const Cand &y) { return x.v < y.v; });
    if ((int)allpos.size() > per_sign) allpos.resize(per_sign);
    if ((int)allneg.size() > per_sign) allneg.resize(per_sign);
    pos = allpos;
    neg = allneg;
}

/* ---------- self-test ---------- */
static void selftest(const std::vector<uint32_t> &pr) {
    const u64 N = 3000000;
    std::vector<uint32_t> tau(N + 1, 0);
    for (u64 d = 1; d <= N; d++) for (u64 m = d; m <= N; m += d) tau[m]++;
    std::vector<u64> cum(N + 1, 0);
    for (u64 n = 1; n <= N; n++) cum[n] = cum[n - 1] + tau[n];
    u64 tests[] = {1, 2, 3, 10, 99, 1000, 123456, 999999, 2000000, 3000000};
    for (u64 n : tests)
        if (D_hyp(n) != (i128)cum[n]) { fprintf(stderr, "SELFTEST FAIL: D_hyp(%llu)\n", (unsigned long long)n); exit(2); }
    std::vector<u64> rem; std::vector<uint32_t> dc;
    u64 a = 2000000, L = 5000;
    window_tau(a, L, pr, rem, dc);
    for (u64 i = 0; i < L; i++)
        if (dc[i] != tau[a + i]) { fprintf(stderr, "SELFTEST FAIL: window tau at %llu\n", (unsigned long long)(a + i)); exit(2); }
    WinRes r;
    if (!scan_window(a, L, pr, rem, dc, true, r)) { fprintf(stderr, "SELFTEST FAIL: window walk\n"); exit(2); }
    printf("self-check ok (hyperbola D, window tau, window walk)\n");
}

/* ---------- main ---------- */
int main(int argc, char **argv) {
    int dlo = 10, dhi = 16, K = 1024, cands = 100;
    double W = 2000.0, thr = 1.5;
    u64 L = 200000, seed = 1;
    std::string csv;
    bool control = true;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if (s == "--dec" && i + 2 < argc) { dlo = atoi(argv[++i]); dhi = atoi(argv[++i]); }
        else if (s == "--K" && i + 1 < argc) K = atoi(argv[++i]);
        else if (s == "--W" && i + 1 < argc) W = atof(argv[++i]);
        else if (s == "--cands" && i + 1 < argc) cands = atoi(argv[++i]);
        else if (s == "--L" && i + 1 < argc) L = strtoull(argv[++i], 0, 10);
        else if (s == "--thr" && i + 1 < argc) thr = atof(argv[++i]);
        else if (s == "--seed" && i + 1 < argc) seed = strtoull(argv[++i], 0, 10);
        else if (s == "--nocontrol") control = false;
        else if (s.rfind("--csv=", 0) == 0) csv = s.substr(6);
        else { fprintf(stderr, "unknown/incomplete argument: %s\n"
               "usage: divisor_peaks [--dec lo hi] [--K 1024] [--W 2000] [--cands 100] [--L 200000]\n"
               "                     [--thr 1.5] [--seed 1] [--nocontrol] [--csv=file]\n", s.c_str()); return 1; }
    }
    if (dlo < 8 || dhi > 18 || dlo > dhi) { fprintf(stderr, "--dec must satisfy 8 <= lo <= hi <= 18\n"); return 1; }
    if (K < 16 || K > 8192 || cands < 1 || L < 1000 || W <= 0) { fprintf(stderr, "bad parameter\n"); return 1; }

    int Q = (int)ceil(16.0 * sqrt((double)K));
    f128 xhi_c = 1; for (int i = 0; i < dhi; i++) xhi_c *= 10;
    f128 thi = sqrtq(xhi_c) + (f128)W + 2;
    u64 xmax = (u64)(thi * thi) + L + 10;
    if (xmax > 4000000000000000000ULL) { fprintf(stderr, "x range exceeds 4e18\n"); return 1; }
    u64 pmax = isqrt64(xmax) + 2;
    printf("primes up to %llu ...\n", (unsigned long long)pmax); fflush(stdout);
    std::vector<uint32_t> pr = gen_primes(pmax);
    selftest(pr);
    printf("K=%d  grid=1/%d in t=sqrt(x)  W=%.0f  cands/sign=%d  L=%llu  thr=%.2f  threads=%d\n",
           K, Q, W, cands, (unsigned long long)L, thr, omp_get_max_threads());
    printf("v = Delta/x^{1/4}.  v+ = max of Delta(n) (just after a jump), v- = min of Delta(n-) (just before).\n\n");

    FILE *fc = 0;
    if (!csv.empty()) {
        fc = fopen(csv.c_str(), "w");
        if (!fc) { fprintf(stderr, "cannot open %s\n", csv.c_str()); return 1; }
        fprintf(fc, "mode,decade,x_center,proxy_v,v_plus,x_plus,v_minus,x_minus\n");
    }
    const char *mname[3] = {"hunt+", "hunt-", "ctrl "};
    long double gbest_p[2] = {-1e9L, -1e9L}, gbest_m[2] = {1e9L, 1e9L};
    u64 gxp[2] = {0, 0}, gxm[2] = {0, 0};

    printf("%-6s %-6s %7s | %10s %9s %20s | %10s %9s %20s\n",
           "decade", "mode", "windows", "max v+", "mean v+", "x at max v+", "min v-", "mean v-", "x at min v-");
    for (int e = dlo; e <= dhi; e++) {
        f128 xc = 1; for (int i = 0; i < e; i++) xc *= 10;
        f128 t0 = sqrtq(xc);
        std::vector<Cand> pos, neg;
        proxy_scan(t0, W, K, Q, thr, cands, pos, neg);

        size_t np = pos.size(), nn = neg.size();
        std::vector<long double> tc; std::vector<int> md; std::vector<double> pv;
        for (auto &c : pos) { tc.push_back(c.t); md.push_back(0); pv.push_back(c.v); }
        for (auto &c : neg) { tc.push_back(c.t); md.push_back(1); pv.push_back(c.v); }
        if (control) {
            std::mt19937_64 rng(seed * 1000003ULL + (u64)e);
            std::uniform_real_distribution<double> U(0.0, 1.0);
            for (size_t i = 0; i < np + nn; i++) { tc.push_back((long double)t0 + (long double)(U(rng) * W)); md.push_back(2); pv.push_back(0); }
        }
        size_t nw = tc.size();
        std::vector<WinRes> res(nw);
        int bad = 0;
#pragma omp parallel
        {
            std::vector<u64> rem; std::vector<uint32_t> dc;
#pragma omp for schedule(dynamic, 1)
            for (long long i = 0; i < (long long)nw; i++) {
                u64 x = (u64)(tc[i] * tc[i]);
                u64 a = x > L / 2 + 2 ? x - L / 2 : 2;
                bool verify = (i % 40 < 2) || (i < 2);
                WinRes r;
                bool ok = scan_window(a, L, pr, rem, dc, verify, r);
                r.proxy = pv[i];
                res[i] = r;
                if (!ok) {
#pragma omp atomic write
                    bad = 1;
                }
            }
        }
        if (bad) { fprintf(stderr, "aborting: verification failed in decade 1e%d\n", e); return 2; }

        for (int mode = 0; mode < 3; mode++) {
            if (mode == 2 && !control) continue;
            long double mxp = -1e9L, sump = 0, mnm = 1e9L, summ = 0; u64 xpm = 0, xmm = 0; size_t cnt = 0;
            for (size_t i = 0; i < nw; i++) {
                if (md[i] != mode) continue;
                cnt++;
                bool usep = (mode == 0 || mode == 2), usem = (mode == 1 || mode == 2);
                if (usep) { sump += res[i].vp; if (res[i].vp > mxp) { mxp = res[i].vp; xpm = res[i].xp; } }
                if (usem) { summ += res[i].vm; if (res[i].vm < mnm) { mnm = res[i].vm; xmm = res[i].xm; } }
                if (fc) fprintf(fc, "%s,%d,%llu,%.4f,%.6Lf,%llu,%.6Lf,%llu\n", mname[mode], e,
                                (unsigned long long)(res[i].a + L / 2), res[i].proxy, res[i].vp,
                                (unsigned long long)res[i].xp, res[i].vm, (unsigned long long)res[i].xm);
            }
            if (cnt == 0) continue;
            size_t cp = (mode == 1) ? 0 : cnt, cm = (mode == 0) ? 0 : cnt;
            printf("1e%-4d %-6s %7zu | ", e, mname[mode], cnt);
            if (cp) printf("%10.4Lf %9.4Lf %20llu | ", mxp, sump / cp, (unsigned long long)xpm);
            else    printf("%10s %9s %20s | ", "-", "-", "-");
            if (cm) printf("%10.4Lf %9.4Lf %20llu\n", mnm, summ / cm, (unsigned long long)xmm);
            else    printf("%10s %9s %20s\n", "-", "-", "-");
            int g = (mode == 2) ? 1 : 0;
            if (cp && mxp > gbest_p[g]) { gbest_p[g] = mxp; gxp[g] = xpm; }
            if (cm && mnm < gbest_m[g]) { gbest_m[g] = mnm; gxm[g] = xmm; }
        }
        printf("\n"); fflush(stdout);
    }
    printf("OVERALL  hunt: max v+ = %.4Lf at x=%llu,  min v- = %.4Lf at x=%llu\n",
           gbest_p[0], (unsigned long long)gxp[0], gbest_m[0], (unsigned long long)gxm[0]);
    if (control)
        printf("OVERALL  ctrl: max v+ = %.4Lf at x=%llu,  min v- = %.4Lf at x=%llu\n",
               gbest_p[1], (unsigned long long)gxp[1], gbest_m[1], (unsigned long long)gxm[1]);
    printf("NOTE: only |v| clearly above the ctrl rows, and slowly growing with the decade, is informative.\n"
           "      An x^{1/4+c} violation would make max v+ grow like x^c; the conjecture allows slow growth\n"
           "      (Omega-results: sup |v| is unbounded but only like exp(c (loglog x)^{1/4}...)).\n");
    if (fc) fclose(fc);
    return 0;
}

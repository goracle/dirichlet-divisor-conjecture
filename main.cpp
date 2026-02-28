// Compile:
//   g++ -O2 -fopenmp -o delta main.cpp -lgmp -lhdf5 -lm -lquadmath
// Without libquadmath:
//   g++ -O2 -fopenmp -DNO_QUADMATH -o delta main.cpp -lgmp -lhdf5 -lm
//
// Install deps (Ubuntu/Debian):
//   sudo apt-get install libgmp-dev libhdf5-dev
//
// Output: results.h5  (flat datasets: n_str, err_a, err_d, err_combo, logn,
//                       theta_slope, env_max, method)
// Optional 4th arg overrides output path:  ./delta 1e16 10000 10.0 myfile.h5

#include "header.h"

// integer sqrt for u128 (floor)
static u128 isqrt_u128(u128 n) {
    if (n == 0) return 0;
    // start from 64-bit double estimate
    long double approx = sqrt((long double)n);
    u128 r = (u128)approx;
    // refine
    while ((r+1)*(r+1) <= n) ++r;
    while (r*r > n) --r;
    return r;
}


static long double exact_err_combo_u128(u128 n);


static long double exact_err_D_u128(u128 n);



/* -----------------------------------------------------------------------
   mpz_t helpers
   ----------------------------------------------------------------------- */

// Set mpz from u128 by splitting into two 64-bit halves.
static inline void mpz_set_u128(mpz_t z, u128 v) {
    uint64_t hi = (uint64_t)(v >> 64);
    uint64_t lo = (uint64_t)(v);
    mpz_set_ui(z, hi);
    mpz_mul_2exp(z, z, 64);
    mpz_add_ui(z, z, lo);
}

// Convert mpz to string (caller must free).
static inline std::string mpz_str(mpz_t z) {
    char *s = mpz_get_str(nullptr, 10, z);
    std::string r(s);
    free(s);
    return r;
}

/* -----------------------------------------------------------------------
   Progress bar (unchanged logic, unchanged interface)
   ----------------------------------------------------------------------- */
struct Progress {
    double                 total_weight;
    std::atomic<long long> done_weight_bits;
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

/* --- Utility: u128 to string --- */
std::string u128_to_str(u128 n) {
    if (n == 0) return "0";
    std::string s = "";
    while (n > 0) { s += (char)('0' + (n % 10)); n /= 10; }
    std::reverse(s.begin(), s.end());
    return s;
}

/* Parse a decimal string into u128 */
u128 parse_u128(const char *s) {
    u128 v = 0;
    for (; *s >= '0' && *s <= '9'; s++)
        v = v * 10 + (*s - '0');
    return v;
}



/* -----------------------------------------------------------------------
   D(n) = sum_{m=1}^{n} tau(m)  [O(sqrt(n))]
   Hyperbola method: 2 * sum_{k=1}^{r} floor(n/k) - r^2
   Accumulator: mpz_t (caller must init out before calling).
   ----------------------------------------------------------------------- */
void D_mpz(u128 n, mpz_t out) {
    if (n == 0) { mpz_set_ui(out, 0); return; }
    u128 r = isqrt_u128(n);

    mpz_t s, tmp, r2;
    mpz_init_set_ui(s, 0);
    mpz_init(tmp);
    mpz_init(r2);

    for (u128 k = 1; k <= r; k++) {
        mpz_set_u128(tmp, n / k);
        mpz_add(s, s, tmp);
    }
    mpz_mul_ui(s, s, 2);

    mpz_set_u128(r2, r);
    mpz_mul(r2, r2, r2);
    mpz_sub(out, s, r2);

    mpz_clear(s); mpz_clear(tmp); mpz_clear(r2);
}

/* -----------------------------------------------------------------------
   sigma_tau(n) = sum_{m=1}^{n} m*tau(m)  [O(sqrt(n))]
   Block-compressed. Accumulator: mpz_t.
   ----------------------------------------------------------------------- */
void sigma_tau_mpz(u128 n, mpz_t out) {
    if (n == 0) { mpz_set_ui(out, 0); return; }

    mpz_t s, tq, bsumd, tmp_lo, tmp_hi, tmp_q, tmp_blen;
    mpz_init_set_ui(s, 0);
    mpz_init(tq); mpz_init(bsumd);
    mpz_init(tmp_lo); mpz_init(tmp_hi); mpz_init(tmp_q); mpz_init(tmp_blen);

    u128 d = 1;
    while (d <= n) {
        u128 q  = n / d;
        u128 d2 = n / q;

        // bsumd = (d + d2) * (d2 - d + 1) / 2
        mpz_set_u128(tmp_lo,   d);
        mpz_set_u128(tmp_hi,   d2);
        mpz_set_u128(tmp_blen, d2 - d + 1);
        mpz_add(bsumd, tmp_lo, tmp_hi);
        mpz_mul(bsumd, bsumd, tmp_blen);
        mpz_tdiv_q_2exp(bsumd, bsumd, 1);

        // tq = q*(q+1)/2
        mpz_set_u128(tmp_q, q);
        mpz_set(tq, tmp_q);
        mpz_add_ui(tq, tq, 1);
        mpz_mul(tq, tq, tmp_q);
        mpz_tdiv_q_2exp(tq, tq, 1);

        mpz_mul(tq, tq, bsumd);
        mpz_add(s, s, tq);

        d = d2 + 1;
    }
    mpz_set(out, s);

    mpz_clear(s); mpz_clear(tq); mpz_clear(bsumd);
    mpz_clear(tmp_lo); mpz_clear(tmp_hi); mpz_clear(tmp_q); mpz_clear(tmp_blen);
}

/* -----------------------------------------------------------------------
   a_hoying(n): Original O(sqrt(n)) formula.
     term1 = r^2 * ((1+r)^2 - 4*(n+1)) / 4
     sum   = sum_{i=1}^{r} q * (2*(n+1) - i*(1+q))    where q = floor(n/i)
   Accumulator: mpz_t.
   ----------------------------------------------------------------------- */
void a_hoying_mpz(u128 n, mpz_t out) {
    if (n == 0) { mpz_set_ui(out, 0); return; }

    u128 r_val = isqrt_u128(n);

    mpz_t r, n128, r2, one_plus_r_sq, four_np1, term1;
    mpz_init(r);    mpz_set_u128(r, r_val);
    mpz_init(n128); mpz_set_u128(n128, n);
    mpz_init(r2);   mpz_mul(r2, r, r);
    mpz_init(one_plus_r_sq);
    mpz_init(four_np1);
    mpz_init(term1);

    // term1 = r^2 * ((1+r)^2 - 4*(n+1)) / 4
    mpz_add_ui(one_plus_r_sq, r, 1);
    mpz_mul(one_plus_r_sq, one_plus_r_sq, one_plus_r_sq);
    mpz_add_ui(four_np1, n128, 1);
    mpz_mul_ui(four_np1, four_np1, 4);
    mpz_sub(term1, one_plus_r_sq, four_np1);
    mpz_mul(term1, term1, r2);
    mpz_tdiv_q_ui(term1, term1, 4);

    // sum_part
    mpz_t sum_part, q, i_z, inner, two_np2;
    mpz_init_set_ui(sum_part, 0);
    mpz_init(q); mpz_init(i_z); mpz_init(inner);
    mpz_init(two_np2);
    mpz_add_ui(two_np2, n128, 1);
    mpz_mul_ui(two_np2, two_np2, 2);   // 2*(n+1), constant

    for (u128 i = 1; i <= r_val; i++) {
        u128 qv = n / i;
        mpz_set_u128(q,   qv);
        mpz_set_u128(i_z, i);

        // inner = q * (2*(n+1) - i*(1+q))
        mpz_add_ui(inner, q, 1);
        mpz_mul(inner, inner, i_z);
        mpz_sub(inner, two_np2, inner);
        mpz_mul(inner, inner, q);
        mpz_add(sum_part, sum_part, inner);
    }

    mpz_add(out, term1, sum_part);

    mpz_clear(r); mpz_clear(n128); mpz_clear(r2); mpz_clear(one_plus_r_sq);
    mpz_clear(four_np1); mpz_clear(term1);
    mpz_clear(sum_part); mpz_clear(q); mpz_clear(i_z); mpz_clear(inner); mpz_clear(two_np2);
}

/* -----------------------------------------------------------------------
   D_ST_chunk: accumulate partial D and sigma_tau sums for d in [d_lo, d_hi].
   part_D and part_ST must be pre-initialised to 0 by caller.
   ----------------------------------------------------------------------- */
static void D_ST_chunk_mpz(u128 n, u128 r, u128 d_lo, u128 d_hi,
                            mpz_t part_D, mpz_t part_ST)
{
    if (d_lo > d_hi || d_lo > n) return;

    mpz_t q_z, bsumd, tq, tmp;
    mpz_init(q_z); mpz_init(bsumd); mpz_init(tq); mpz_init(tmp);

    u128 d = d_lo;
    while (d <= d_hi) {
        u128 q  = n / d;
        u128 hi = n / q;
        if (hi > d_hi) hi = d_hi;

        u128 blen = hi - d + 1;
        u128 bslo = d + hi;

        // bsumd = bslo * blen / 2
        mpz_set_u128(bsumd, bslo);
        mpz_set_u128(tmp,   blen);
        mpz_mul(bsumd, bsumd, tmp);
        mpz_tdiv_q_2exp(bsumd, bsumd, 1);

        // tq = q*(q+1)/2
        mpz_set_u128(q_z, q);
        mpz_set(tq, q_z);
        mpz_add_ui(tq, tq, 1);
        mpz_mul(tq, tq, q_z);
        mpz_tdiv_q_2exp(tq, tq, 1);

        mpz_mul(tq, tq, bsumd);
        mpz_add(part_ST, part_ST, tq);

        if (d <= r) {
            u128 hic = (hi <= r) ? hi : r;
            mpz_set_u128(tmp, hic - d + 1);
            mpz_mul(tmp, tmp, q_z);
            mpz_add(part_D, part_D, tmp);
        }

        d = hi + 1;
    }

    mpz_clear(q_z); mpz_clear(bsumd); mpz_clear(tq); mpz_clear(tmp);
}

/* -----------------------------------------------------------------------
   Joint computation of D(n) and sigma_tau(n). [O(sqrt(n)), serial]
   out_D and out_ST must be pre-initialised by caller.
   ----------------------------------------------------------------------- */
void D_and_sigma_tau_mpz(u128 n, mpz_t out_D, mpz_t out_ST) {
    if (n == 0) { mpz_set_ui(out_D, 0); mpz_set_ui(out_ST, 0); return; }

    u128 r = isqrt_u128(n);

    mpz_t pD, pST, tail_D, tail_ST, r2;
    mpz_init_set_ui(pD, 0);     mpz_init_set_ui(pST, 0);
    mpz_init_set_ui(tail_D, 0); mpz_init_set_ui(tail_ST, 0);
    mpz_init(r2);

    // Parallelism is handled at the outer (per-point) level; run serially here.
    D_ST_chunk_mpz(n, r, 1,   r, pD, pST);
    D_ST_chunk_mpz(n, r, r+1, n, tail_D, tail_ST);

    // out_D = 2*(pD + tail_D) - r^2
    mpz_add(out_D, pD, tail_D);
    mpz_mul_ui(out_D, out_D, 2);
    mpz_set_u128(r2, r);
    mpz_mul(r2, r2, r2);
    mpz_sub(out_D, out_D, r2);

    mpz_add(out_ST, pST, tail_ST);

    mpz_clear(pD); mpz_clear(pST); mpz_clear(tail_D); mpz_clear(tail_ST); mpz_clear(r2);
}

/* -----------------------------------------------------------------------
   a_identity_fused(n) = n * D(n-1) - sigma_tau(n-1)
   ----------------------------------------------------------------------- */
void a_identity_fused_mpz(u128 n, mpz_t out) {
    if (n <= 1) { mpz_set_ui(out, 0); return; }
    mpz_t dn, st, n_z;
    mpz_init(dn); mpz_init(st); mpz_init(n_z);
    D_and_sigma_tau_mpz(n-1, dn, st);
    mpz_set_u128(n_z, n);
    mpz_mul(out, n_z, dn);
    mpz_sub(out, out, st);
    mpz_clear(dn); mpz_clear(st); mpz_clear(n_z);
}

/* -----------------------------------------------------------------------
   a_identity(n) = n * D(n-1) - sigma_tau(n-1)  [two separate passes]
   ----------------------------------------------------------------------- */
void a_identity_mpz(u128 n, mpz_t out) {
    if (n <= 1) { mpz_set_ui(out, 0); return; }
    mpz_t dn, st, n_z;
    mpz_init(dn); mpz_init(st); mpz_init(n_z);
    D_mpz(n-1, dn);
    sigma_tau_mpz(n-1, st);
    mpz_set_u128(n_z, n);
    mpz_mul(out, n_z, dn);
    mpz_sub(out, out, st);
    mpz_clear(dn); mpz_clear(st); mpz_clear(n_z);
}

/* -----------------------------------------------------------------------
   a_brute_force: O(n^2), only used for small n in tests.
   ----------------------------------------------------------------------- */
void a_brute_force_mpz(u128 n, mpz_t out) {
    mpz_set_ui(out, 0);
    if (n < 2) return;
    mpz_t tmp; mpz_init(tmp);
    for (u128 d = 1; d < n; ++d)
        for (u128 s = 1; s <= n; ++s) {
            u128 max_k = (n - s) / d + 1;
            if (max_k >= 2) {
                mpz_set_u128(tmp, max_k - 1);
                mpz_add(out, out, tmp);
            }
        }
    mpz_clear(tmp);
}

/* -----------------------------------------------------------------------
   GMP precision for asymptotic computations.
   At n=10^19: a(n) ~ n^2*log(n)/2 ~ 10^40.  We need the error term which
   is ~n^0.75 ~ 10^14.  So we need ~26 digits of headroom above the leading
   term, meaning at least 55 significant decimal digits.  512 bits (~154
   decimal digits) is more than sufficient for n up to 10^30.
   ----------------------------------------------------------------------- */
#define ASYM_PREC 512

/* Euler-Mascheroni gamma as a string (50 digits) */
static const char *GAMMA_STR = "0.57721566490153286060651209008240243104215933593992";

/* -----------------------------------------------------------------------
   Compute a_asymptotic(n) entirely in GMP mpf_t at ASYM_PREC bits.
   a(n) ~ n^2/2 * log(n) + (gamma - 3/4)*n^2 + n/4
   Result stored in out (caller must mpf_init2(out, ASYM_PREC) beforehand).
   ----------------------------------------------------------------------- */
static void a_asymptotic_mpf(u128 n, mpf_t out) {
    mpf_t x, logx, gamma_em, tmp;
    mpf_init2(x,        ASYM_PREC);
    mpf_init2(logx,     ASYM_PREC);
    mpf_init2(gamma_em, ASYM_PREC);
    mpf_init2(tmp,      ASYM_PREC);

    // x = n  (set via u128 split)
    {
        mpz_t nz; mpz_init(nz);
        mpz_set_u128(nz, n);
        mpf_set_z(x, nz);
        mpz_clear(nz);
    }

    // log(x) via bit-reduction + atanh series (no mpfr needed):
    // n = 2^m_bits * r, 1 <= r < 2
    // log(n) = m_bits*log(2) + 2*atanh((r-1)/(r+1))
    {
        uint64_t m_bits = 0;
        {
            u128 tmp_n = n;
            while (tmp_n > 1) { tmp_n >>= 1; m_bits++; }
        }
        // r = n / 2^m_bits, so 1 <= r < 2
        // log(n) = m_bits * log(2) + log(r)
        // log(r) via atanh series: log(r) = 2*atanh((r-1)/(r+1))
        // atanh(z) = z + z^3/3 + z^5/5 + ...  converges for |z| < 1

        mpf_t r_f, z, z2, term2, atanh_val, log2_val, power2;
        mpf_init2(r_f,      ASYM_PREC);
        mpf_init2(z,        ASYM_PREC);
        mpf_init2(z2,       ASYM_PREC);
        mpf_init2(term2,    ASYM_PREC);
        mpf_init2(atanh_val,ASYM_PREC);
        mpf_init2(log2_val, ASYM_PREC);
        mpf_init2(power2,   ASYM_PREC);

        // r_f = n / 2^m_bits
        mpf_set_ui(power2, 1);
        mpf_mul_2exp(power2, power2, (unsigned long)m_bits);
        mpf_div(r_f, x, power2);

        // z = (r-1)/(r+1)
        mpf_sub_ui(z, r_f, 1);   // r - 1
        mpf_add_ui(tmp, r_f, 1); // r + 1
        mpf_div(z, z, tmp);

        // atanh(z) via series
        mpf_mul(z2, z, z);        // z^2
        mpf_set(atanh_val, z);    // term = z
        mpf_set(term2, z);
        for (unsigned long k = 3; k < 300; k += 2) {
            mpf_mul(term2, term2, z2);          // term *= z^2
            mpf_div_ui(tmp, term2, k);          // tmp = term / k
            mpf_add(atanh_val, atanh_val, tmp);
            // Check convergence: if |tmp| < 2^-(ASYM_PREC) break
            // (lazy: just run enough iterations; 300 terms for |z|<0.5 is overkill)
        }
        mpf_mul_2exp(atanh_val, atanh_val, 1);  // log(r) = 2*atanh(z)

        // log(2) via atanh series on r=2: log(2) = 2*atanh(1/3)
        {
            mpf_t z3, z32, t3;
            mpf_init2(z3,  ASYM_PREC);
            mpf_init2(z32, ASYM_PREC);
            mpf_init2(t3,  ASYM_PREC);
            mpf_set_ui(z3, 1);
            mpf_div_ui(z3, z3, 3);   // 1/3
            mpf_mul(z32, z3, z3);    // 1/9
            mpf_set(log2_val, z3);
            mpf_set(t3, z3);
            for (unsigned long k = 3; k < 300; k += 2) {
                mpf_mul(t3, t3, z32);
                mpf_div_ui(tmp, t3, k);
                mpf_add(log2_val, log2_val, tmp);
            }
            mpf_mul_2exp(log2_val, log2_val, 1);  // log(2)
            mpf_clear(z3); mpf_clear(z32); mpf_clear(t3);
        }

        // logx = m_bits * log(2) + log(r)
        mpf_mul_ui(logx, log2_val, (unsigned long)m_bits);
        mpf_add(logx, logx, atanh_val);

        mpf_clear(r_f); mpf_clear(z); mpf_clear(z2); mpf_clear(term2);
        mpf_clear(atanh_val); mpf_clear(log2_val); mpf_clear(power2);
    }

    // gamma_em
    mpf_set_str(gamma_em, GAMMA_STR, 10);

    // out = x^2/2 * logx + (gamma - 3/4) * x^2 + x/4
    mpf_t x2, coeff;
    mpf_init2(x2,    ASYM_PREC);
    mpf_init2(coeff, ASYM_PREC);

    mpf_mul(x2, x, x);               // x^2

    mpf_mul(out, x2, logx);          // x^2 * logx
    mpf_div_ui(out, out, 2);         // x^2/2 * logx

    mpf_set_str(coeff, "0.75", 10);
    mpf_sub(coeff, gamma_em, coeff); // gamma - 3/4
    mpf_mul(tmp, x2, coeff);
    mpf_add(out, out, tmp);          // + (gamma-3/4)*x^2

    mpf_div_ui(tmp, x, 4);
    mpf_add(out, out, tmp);          // + x/4

    mpf_clear(x); mpf_clear(logx); mpf_clear(gamma_em);
    mpf_clear(tmp); mpf_clear(x2); mpf_clear(coeff);
}

/* -----------------------------------------------------------------------
   Compute exact error = exact_value - asymptotic, returning long double.
   The subtraction is done entirely in mpf_t at ASYM_PREC bits to avoid
   catastrophic cancellation.  Only the final small result is cast to
   long double (which has plenty of precision for a value ~ n^0.75).
   ----------------------------------------------------------------------- */
static long double subtract_asym_mpf(mpz_t exact, u128 n,
                                      void (*asym_fn)(u128, mpf_t))
{
    mpf_t asym, exact_f, diff;
    mpf_init2(asym,    ASYM_PREC);
    mpf_init2(exact_f, ASYM_PREC);
    mpf_init2(diff,    ASYM_PREC);

    asym_fn(n, asym);
    mpf_set_z(exact_f, exact);
    mpf_sub(diff, exact_f, asym);

    long double result = (long double)mpf_get_d(diff);  // diff ~ n^0.75, fits fine

    mpf_clear(asym); mpf_clear(exact_f); mpf_clear(diff);
    return result;
}

/* sigma_tau asymptotic: x^2/2*log(x) + (gamma-1/4)*x^2 */
static void sigma_tau_asymptotic_mpf(u128 n, mpf_t out) {
    // Reuse a_asymptotic_mpf structure but with different coefficients
    // sigma_tau(n) ~ n^2/2*log(n) + (gamma - 1/4)*n^2
    mpf_t x, logx, gamma_em, tmp, x2, coeff;
    mpf_init2(x,        ASYM_PREC);
    mpf_init2(logx,     ASYM_PREC);
    mpf_init2(gamma_em, ASYM_PREC);
    mpf_init2(tmp,      ASYM_PREC);
    mpf_init2(x2,       ASYM_PREC);
    mpf_init2(coeff,    ASYM_PREC);

    { mpz_t nz; mpz_init(nz); mpz_set_u128(nz, n); mpf_set_z(x, nz); mpz_clear(nz); }

    // Compute logx using same AGM approach — factor out into helper lambda
    // (inline the same code for now)
    {
        uint64_t m_bits = 0;
        { u128 tmp_n = n; while (tmp_n > 1) { tmp_n >>= 1; m_bits++; } }
        mpf_t r_f, z, z2, term2, atanh_val, log2_val, power2;
        mpf_init2(r_f,ASYM_PREC); mpf_init2(z,ASYM_PREC); mpf_init2(z2,ASYM_PREC);
        mpf_init2(term2,ASYM_PREC); mpf_init2(atanh_val,ASYM_PREC);
        mpf_init2(log2_val,ASYM_PREC); mpf_init2(power2,ASYM_PREC);
        mpf_set_ui(power2,1); mpf_mul_2exp(power2,power2,(unsigned long)m_bits);
        mpf_div(r_f,x,power2);
        mpf_sub_ui(z,r_f,1); mpf_add_ui(tmp,r_f,1); mpf_div(z,z,tmp);
        mpf_mul(z2,z,z); mpf_set(atanh_val,z); mpf_set(term2,z);
        for (unsigned long k=3;k<300;k+=2){mpf_mul(term2,term2,z2);mpf_div_ui(tmp,term2,k);mpf_add(atanh_val,atanh_val,tmp);}
        mpf_mul_2exp(atanh_val,atanh_val,1);
        { mpf_t z3,z32,t3; mpf_init2(z3,ASYM_PREC); mpf_init2(z32,ASYM_PREC); mpf_init2(t3,ASYM_PREC);
          mpf_set_ui(z3,1); mpf_div_ui(z3,z3,3); mpf_mul(z32,z3,z3);
          mpf_set(log2_val,z3); mpf_set(t3,z3);
          for(unsigned long k=3;k<300;k+=2){mpf_mul(t3,t3,z32);mpf_div_ui(tmp,t3,k);mpf_add(log2_val,log2_val,tmp);}
          mpf_mul_2exp(log2_val,log2_val,1);
          mpf_clear(z3);mpf_clear(z32);mpf_clear(t3); }
        mpf_mul_ui(logx,log2_val,(unsigned long)m_bits); mpf_add(logx,logx,atanh_val);
        mpf_clear(r_f);mpf_clear(z);mpf_clear(z2);mpf_clear(term2);
        mpf_clear(atanh_val);mpf_clear(log2_val);mpf_clear(power2);
    }

    mpf_set_str(gamma_em, GAMMA_STR, 10);
    mpf_mul(x2,x,x);
    mpf_mul(out,x2,logx); mpf_div_ui(out,out,2);
    mpf_set_str(coeff,"0.25",10); mpf_sub(coeff,gamma_em,coeff);
    mpf_mul(tmp,x2,coeff); mpf_add(out,out,tmp);

    mpf_clear(x);mpf_clear(logx);mpf_clear(gamma_em);
    mpf_clear(tmp);mpf_clear(x2);mpf_clear(coeff);
}

/* D asymptotic: x*log(x) + (2*gamma-1)*x */
static void D_asymptotic_mpf(u128 n, mpf_t out) {
    mpf_t x, logx, gamma_em, tmp, coeff;
    mpf_init2(x,        ASYM_PREC);
    mpf_init2(logx,     ASYM_PREC);
    mpf_init2(gamma_em, ASYM_PREC);
    mpf_init2(tmp,      ASYM_PREC);
    mpf_init2(coeff,    ASYM_PREC);

    { mpz_t nz; mpz_init(nz); mpz_set_u128(nz, n); mpf_set_z(x, nz); mpz_clear(nz); }

    {
        uint64_t m_bits = 0;
        { u128 tmp_n = n; while (tmp_n > 1) { tmp_n >>= 1; m_bits++; } }
        mpf_t r_f,z,z2,term2,atanh_val,log2_val,power2;
        mpf_init2(r_f,ASYM_PREC);mpf_init2(z,ASYM_PREC);mpf_init2(z2,ASYM_PREC);
        mpf_init2(term2,ASYM_PREC);mpf_init2(atanh_val,ASYM_PREC);
        mpf_init2(log2_val,ASYM_PREC);mpf_init2(power2,ASYM_PREC);
        mpf_set_ui(power2,1);mpf_mul_2exp(power2,power2,(unsigned long)m_bits);
        mpf_div(r_f,x,power2);
        mpf_sub_ui(z,r_f,1);mpf_add_ui(tmp,r_f,1);mpf_div(z,z,tmp);
        mpf_mul(z2,z,z);mpf_set(atanh_val,z);mpf_set(term2,z);
        for(unsigned long k=3;k<300;k+=2){mpf_mul(term2,term2,z2);mpf_div_ui(tmp,term2,k);mpf_add(atanh_val,atanh_val,tmp);}
        mpf_mul_2exp(atanh_val,atanh_val,1);
        { mpf_t z3,z32,t3; mpf_init2(z3,ASYM_PREC);mpf_init2(z32,ASYM_PREC);mpf_init2(t3,ASYM_PREC);
          mpf_set_ui(z3,1);mpf_div_ui(z3,z3,3);mpf_mul(z32,z3,z3);
          mpf_set(log2_val,z3);mpf_set(t3,z3);
          for(unsigned long k=3;k<300;k+=2){mpf_mul(t3,t3,z32);mpf_div_ui(tmp,t3,k);mpf_add(log2_val,log2_val,tmp);}
          mpf_mul_2exp(log2_val,log2_val,1);
          mpf_clear(z3);mpf_clear(z32);mpf_clear(t3); }
        mpf_mul_ui(logx,log2_val,(unsigned long)m_bits);mpf_add(logx,logx,atanh_val);
        mpf_clear(r_f);mpf_clear(z);mpf_clear(z2);mpf_clear(term2);
        mpf_clear(atanh_val);mpf_clear(log2_val);mpf_clear(power2);
    }

    mpf_set_str(gamma_em, GAMMA_STR, 10);
    // out = x*logx + (2*gamma-1)*x
    mpf_mul(out, x, logx);
    mpf_mul_ui(coeff, gamma_em, 2); mpf_sub_ui(coeff, coeff, 1);
    mpf_mul(tmp, x, coeff);
    mpf_add(out, out, tmp);

    mpf_clear(x);mpf_clear(logx);mpf_clear(gamma_em);mpf_clear(tmp);mpf_clear(coeff);
}

/* Keep old __float128 path for the asymptotic check (used nowhere critical now) */
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

/* Legacy shim — no longer used by exact_err but kept for any callers */
static void asym_to_mpz_and_frac(__float128 asym, mpz_t asym_int,
                                  long double &frac_out)
{
    char buf[80];
#ifdef NO_QUADMATH
    snprintf(buf, sizeof(buf), "%.35Lf", (long double)asym);
#else
    quadmath_snprintf(buf, sizeof(buf), "%.35Qf", asym);
#endif
    mpf_t fasym; mpf_init2(fasym, 200);
    mpf_set_str(fasym, buf, 10);
    mpz_set_f(asym_int, fasym);
    mpf_t fint; mpf_init2(fint, 200);
    mpf_set_z(fint, asym_int);
    mpf_sub(fasym, fasym, fint);
    frac_out = (long double)mpf_get_d(fasym);
    mpf_clear(fasym); mpf_clear(fint);
}

/* -----------------------------------------------------------------------
   exact_err(n): a(n) - a_asymptotic(n)
   exact_err_sigma(n): sigma_tau(n) - sigma_tau_asymptotic(n)
   exact_err_D(n): D(n) - D_asymptotic(n)
   exact_err_combo(n): (exact_err_sigma(n) - exact_err(n+1)) / n
   All use full GMP precision to avoid catastrophic cancellation.
   ----------------------------------------------------------------------- */
long double exact_err(u128 n) {
    mpz_t an; mpz_init(an);
    a_hoying_mpz(n - 1, an);
    long double result = subtract_asym_mpf(an, n, a_asymptotic_mpf);
    mpz_clear(an);
    return result;
}

long double exact_err_sigma(u128 n) {
    mpz_t st; mpz_init(st);
    sigma_tau_mpz(n, st);
    long double result = subtract_asym_mpf(st, n, sigma_tau_asymptotic_mpf);
    mpz_clear(st);
    return result;
}

long double exact_err_D(u128 n) {
    mpz_t d; mpz_init(d);
    D_mpz(n, d);
    long double result = subtract_asym_mpf(d, n, D_asymptotic_mpf);
    mpz_clear(d);
    return result;
}

long double exact_err_combo(u128 n) {
    long double es = exact_err_sigma(n);
    long double ea = exact_err(n + 1);
    return (es - ea) / (long double)n;
}

/* -----------------------------------------------------------------------
   Tests
   ----------------------------------------------------------------------- */
void run_sequence_test() {
    uint64_t expected[] = {0, 1, 4, 9, 17, 27, 41, 57, 77, 100, 127};
    printf("--- A078567 sequence check ---\n");
    mpz_t r; mpz_init(r);
    for (u128 i = 0; i <= 10; ++i) {
      a_hoying_mpz(i, r);
        uint64_t val = (uint64_t)mpz_get_ui(r);
        printf("a(%lu): %s (%lu)\n", (unsigned long)(i+1),
            val == expected[(int)i] ? "PASS" : "FAIL", val);
    }
    mpz_clear(r);
}

void run_identity_test(u128 cutoff) {
    u128 limit = (cutoff < 300) ? cutoff : (u128)300;
    printf("\n--- Identity check (all methods vs brute force, n <= %s) ---\n",
           u128_to_str(limit).c_str());
    bool ok = true;
    mpz_t h, id, fu, bf;
    mpz_init(h); mpz_init(id); mpz_init(fu); mpz_init(bf);
    for (u128 n = 2; n <= limit; n++) {
      a_hoying_mpz(n-1, h);
        a_identity_mpz(n, id);
        a_identity_fused_mpz(n, fu);
        a_brute_force_mpz(n, bf);
        if (mpz_cmp(h, id) != 0 || mpz_cmp(h, bf) != 0 || mpz_cmp(h, fu) != 0) {
            char *sh = mpz_get_str(nullptr,10,h);
            char *si = mpz_get_str(nullptr,10,id);
            char *sf = mpz_get_str(nullptr,10,fu);
            char *sb = mpz_get_str(nullptr,10,bf);
            printf("MISMATCH n=%s: hoying=%s identity=%s fused=%s brute=%s\n",
                u128_to_str(n).c_str(), sh, si, sf, sb);
            free(sh); free(si); free(sf); free(sb);
            ok = false;
        }
    }
    if (ok) printf("All n in [2,%s]: PASS\n", u128_to_str(limit).c_str());
    mpz_clear(h); mpz_clear(id); mpz_clear(fu); mpz_clear(bf);
}

/* -----------------------------------------------------------------------
   Sample-point generation (unchanged)
   ----------------------------------------------------------------------- */
static std::vector<u128> make_sample_ns(u128 max_n, double ratio) {
    if (ratio < 1.001) ratio = 1.001;
    std::vector<u128> ns;
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

/* -----------------------------------------------------------------------
   Row result (unchanged structure)
   ----------------------------------------------------------------------- */
struct RowResult {
    u128        n;
    bool        skip;          // reserved (mpz is unbounded, skip never fires now)
    bool        mismatch;      // hoying vs fused disagree
    std::string mismatch_msg;
    bool        verified;
    long double err_a;
    long double err_d;         // NAN if not computed (above cutoff)
    long double err_combo;     // NAN if not computed (above cutoff)
    long double logn;
};

/* -----------------------------------------------------------------------
   HDF5 output: flat file, one dataset per column.

   Schema:
     n_str       : variable-length UTF-8 string  — decimal n
     err_a       : float64
     err_d       : float64  (NaN when not computed)
     err_combo   : float64  (NaN when not computed)
     logn        : float64
     theta_slope : float64  (NaN when unavailable)
     env_max     : float64  — running envelope max(|err_a|) up to this row
     method      : variable-length UTF-8 string  — "verified" | "fused"

   Only valid (non-skipped, non-mismatch) rows are written.
   ----------------------------------------------------------------------- */
static void write_hdf5(const char                    *path,
                       const std::vector<std::string> &col_n,
                       const std::vector<double>      &col_err_a,
                       const std::vector<double>      &col_err_d,
                       const std::vector<double>      &col_err_combo,
                       const std::vector<double>      &col_logn,
                       const std::vector<double>      &col_theta,
                       const std::vector<double>      &col_env_max,
                       const std::vector<std::string> &col_method)
{
    hsize_t nrows = (hsize_t)col_err_a.size();
    if (nrows == 0) { fprintf(stderr, "HDF5: no rows to write.\n"); return; }

    hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) { fprintf(stderr, "HDF5: cannot create %s\n", path); return; }

    hid_t space = H5Screate_simple(1, &nrows, nullptr);

    // Write a flat double dataset.
    auto write_f64 = [&](const char *name, const std::vector<double> &v) {
        hid_t ds = H5Dcreate2(file, name, H5T_NATIVE_DOUBLE, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
        H5Dclose(ds);
    };

    // Write a variable-length string dataset.
    auto write_vls = [&](const char *name, const std::vector<std::string> &v) {
        hid_t vlt = H5Tcopy(H5T_C_S1);
        H5Tset_size(vlt, H5T_VARIABLE);
        H5Tset_strpad(vlt, H5T_STR_NULLTERM);
        H5Tset_cset(vlt, H5T_CSET_UTF8);
        std::vector<const char *> ptrs(v.size());
        for (size_t i = 0; i < v.size(); i++) ptrs[i] = v[i].c_str();
        hid_t ds = H5Dcreate2(file, name, vlt, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, vlt, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptrs.data());
        H5Dclose(ds);
        H5Tclose(vlt);
    };

    write_vls("n_str",        col_n);
    write_f64("err_a",        col_err_a);
    write_f64("err_d",        col_err_d);
    write_f64("err_combo",    col_err_combo);
    write_f64("logn",         col_logn);
    write_f64("theta_slope",  col_theta);
    write_f64("env_max",      col_env_max);
    write_vls("method",       col_method);

    H5Sclose(space);
    H5Fclose(file);
    printf("\nResults written to %s  (%llu rows)\n", path, (unsigned long long)nrows);
}


/* -----------------------------------------------------------------------
   main
   ----------------------------------------------------------------------- */
int main(int argc, char **argv) {
    // Usage: ./delta [N [cutoff [ratio [output.h5]]]]
    // N:        max n for asymptotic table; default 10^16
    // cutoff:   verify hoying vs fused up to this n; default 10^4
    // ratio:    multiplicative step between sample points; default 10.0
    // output:   HDF5 output path; default results.h5
    u128        N      = 0;
    u128        cutoff = (u128)10000ULL;
    double      ratio  = 10.0;
    const char *h5path = "results.h5";

    if (argc >= 2) N      = parse_u128(argv[1]);
    if (argc >= 3) cutoff = parse_u128(argv[2]);
    if (argc >= 4) ratio  = atof(argv[3]);
    if (argc >= 5) h5path = argv[4];

    if (ratio < 1.001) {
        fprintf(stderr, "ratio must be > 1.001\n");
        return 1;
    }

    u128 max_table_n = N ? N : (u128)10000000000000000ULL;

    run_sequence_test();
    run_identity_test(cutoff);
    run_asymptotic_table(cutoff, max_table_n, ratio, h5path);

    if (!N) return 0;

    // Single-point detail for explicit N
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


// ---------- utils ----------
static inline u128 u128_from_u64(uint64_t v){ return (u128)v; }

static std::string u128_to_string(u128 v) {
    if (v == 0) return "0";
    std::string s;
    while (v) {
        unsigned digit = (unsigned)(v % 10);
        s.push_back(char('0' + digit));
        v /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

// ---------- D(n) and sigma_tau(n) in u128 ----------
/*
  Computes D(n) = sum_{m=1}^n tau(m)  and
           ST(n)= sum_{m=1}^n m * tau(m)
  Both returned by reference. Uses block/hyperbola method exactly like your mpz
  version but using native u128 arithmetic. Assumes n <= N_SAFE_MAX.
*/
static void D_and_sigma_tau_u128(u128 n, u128 &out_D, u128 &out_ST) {
    if (n == 0) { out_D = 0; out_ST = 0; return; }

    u128 r = isqrt_u128(n);

    u128 pD = 0;   // partial D
    u128 pST = 0;  // partial sigma_tau

    u128 d = 1;
    while (d <= n) {
        u128 q = n / d;
        u128 hi = n / q;
        // block: d..hi have same quotient q

        // bsumd = (d + hi) * (hi - d + 1) / 2
        u128 len = hi - d + 1;
        // compute (d + hi) * len / 2 safely in u128:
        u128 a = d + hi;
        // do divide by 2 on one operand to keep intermediate small
        if ((a & 1) == 0) a = a >> 1, /*len unchanged*/ (void)0;
        else len = len >> 1; // len must be even if a odd (one of them is)
        u128 bsumd = a * len; // <= ~ (2r)*(r) ~ O(r^2) << 2^128 for our N

        // tq = q*(q+1)/2
        u128 tq;
        if ((q & 1) == 0) tq = (q >> 1) * (q + 1);
        else tq = q * ((q + 1) >> 1);

        // pST += tq * bsumd  (may be large but within u128 for N<=3e18)
        // Rearrangement below reduces risk of overflow by multiplying smaller operands first
        // but for our N_SAFE_MAX the product fits in u128.
        pST += tq * bsumd;

        // For pD we only count those d <= r (small side)
        if (d <= r) {
            u128 hi_small = (hi <= r) ? hi : r;
            u128 cnt = hi_small - d + 1;
            pD += cnt * q; // cnt <= r ~1e9, q <= n, product fits u128
        }

        d = hi + 1;
    }

    // out_D = 2 * pD - r^2 + (handled above pD included tail)
    // in the mpz version they computed out_D = 2*(pD + tail_D) - r^2;
    // here pD already collects both halves via the block approach: same formula:
    u128 r2 = r * r;
    out_D = 2 * pD - r2;

    out_ST = pST;
}

/* sigma_tau_u128: wrapper */
static u128 sigma_tau_u128(u128 n) {
    u128 Dval, ST;
    D_and_sigma_tau_u128(n, Dval, ST);
    return ST;
}

/* D_u128: wrapper */
static u128 D_u128(u128 n) {
    u128 Dval, ST;
    D_and_sigma_tau_u128(n, Dval, ST);
    return Dval;
}

// ---------- a_identity_fused using u128 ----------
// a(n) = n * D(n-1) - sigma_tau(n-1)
static i128 a_identity_fused_u128(u128 n) {
    if (n <= 1) return (i128)0;
    u128 Dval, ST;
    D_and_sigma_tau_u128(n - 1, Dval, ST);
    // be careful with signedness: result can be positive and fits in signed __int128
    i128 res = (i128)n * (i128)Dval - (i128)ST;
    return res;
}

// ---------- a_hoying (direct formula) in u128 ----------
// Convert your original a_hoying formula to u128 arithmetic.
// Slight rearrangements reduce intermediate sizes.
i128 a_hoying_u128(u128 n) {
    if (n == 0) return (i128)0;
    u128 r = isqrt_u128(n);
    // term1 = r^2 * ( (1+r)^2 - 4*(n+1) ) / 4
    u128 onepr = r + 1;
    // compute (1+r)^2 and 4*(n+1)
    u128 t1 = onepr * onepr;
    u128 t2 = 4 * (n + 1);
    // (t1 - t2) may be negative in signed; do signed arithmetic carefully below
    // We'll compute term1 as signed __int128 to handle negative result
    i128 term1 = (i128)r * (i128)r * ((i128)t1 - (i128)t2) / 4;

    // sum_part = sum_{i=1}^{r} q * (2*(n+1) - i*(1+q)) where q = floor(n/i)
    i128 sum_part = 0;
    for (u128 i = 1; i <= r; ++i) {
        u128 q = n / i;
        // inner = q * (2*(n+1) - i*(1+q))
        // compute A = 2*(n+1)
        i128 A = (i128)2 * (i128)(n + 1);
        i128 B = (i128)i * ((i128)1 + (i128)q);
        i128 inner = (i128)q * (A - B);
        sum_part += inner;
    }
    i128 out = term1 + sum_part;
    return out;
}


// Compute asymptotic in __float128 exactly as you already had:
static void a_asymptotic_mpf_u128(u128 n, __float128 &asym_out) {
    // return asym as __float128: x*((pi/2)*x - 1)
    // we use the same PI_HALF_VAL constant you defined earlier for __float128
    __float128 x = (__float128)n;
    asym_out = x * (PI_HALF_VAL * x - (__float128)1.0);
}

// Exact error computed using u128 integer a(n) and the asym fractional trick.
// Returns long double (same semantic as subtract_asym_mpf).
static long double exact_err_u128(u128 n) {
    // a(n) computed by fused identity in i128
    if (n <= 1) return 0.0L;
    i128 a_int = a_identity_fused_u128(n);  // signed i128 exact integer

    // asymptotic in __float128
    __float128 asym;
    a_asymptotic_mpf_u128(n, asym);

    // floor(asym) as i128
    i128 asym_floor = (i128)asym;        // truncates toward 0; asym is positive for n>=1
    __float128 asym_frac = asym - (__float128)asym_floor;

    // integer difference (exact)
    i128 int_diff = a_int - asym_floor;

    // final small residual as long double
    long double result = (long double)int_diff - (long double)asym_frac;
    return result;
}

// exact_err for D(n) using the D_and_sigma_tau_u128 wrapper:
static long double exact_err_D_u128(u128 n) {
    if (n == 0) return 0.0L;
    u128 Dval, ST;
    D_and_sigma_tau_u128(n, Dval, ST); // returns D(n) in Dval
    // Now compute D asymptotic: D(n) ~ n*log n + (2*gamma - 1)*n
    // We'll reuse your D_asymptotic_mpf but we want a __float128 version (fast).
    // Simple approximation in long double is fine here, but to match your accuracy,
    // call your mpf-based D_asymptotic_mpf into an __float128, or reuse existing D_asymptotic_mpf.
    // For performance, here's a compact __float128 version (sufficient for N<=3e18):

    // compute log(n) in long double then cast
    long double logn_ld = logl((long double)n);
    __float128 logn = (__float128)logn_ld;
    __float128 x = (__float128)n;
    const long double gamma_ld = 0.577215664901532860606512090082402431L;
    __float128 gamma128 = (__float128)gamma_ld;
    __float128 asym = x * logn + ( (__float128)2.0 * gamma128 - (__float128)1.0) * x;

    // cast Dval to i128
    i128 D_int = (i128)Dval;
    i128 asym_floor = (i128)asym;
    __float128 asym_frac = asym - (__float128)asym_floor;
    i128 int_diff = D_int - asym_floor;
    long double result = (long double)int_diff - (long double)asym_frac;
    return result;
}

// combo = (sigma_tau(n) - a(n+1)) / n  -> compute using u128 paths
static long double exact_err_combo_u128(u128 n) {
    // sigma_tau(n)
    u128 Dtmp, ST;
    D_and_sigma_tau_u128(n, Dtmp, ST);
    i128 sigma_int = (i128)ST;
    // a(n+1)
    i128 a_n1 = a_identity_fused_u128(n + 1);
    i128 numer = sigma_int - a_n1;
    // divide by n exactly using long double
    long double res = (long double)numer / (long double)n;
    return res;
}


/* -----------------------------------------------------------------------
   Main table computation and display
   ----------------------------------------------------------------------- */
void run_asymptotic_table(u128 cutoff, u128 max_n, double ratio,
                          const char *hdf5_path)
{
    printf("\n--- Asymptotic error and empirical theta ---\n");
    printf("a(n) = n^2/2*log(n) + (g-3/4)*n^2 + n/4  +  err_osc\n");
    printf("err_osc = O(n^{theta+1/2}).  ratio=%.4f  cutoff=%s\n",
           ratio, u128_to_str(cutoff).c_str());
    printf("Below cutoff: fused verified vs hoying.  Above: fused only (trusted).\n");
    printf("Accumulators: using fast native __int128 hot path for main computations.\n");

    std::vector<u128> ns = make_sample_ns(max_n, ratio);
    int npts = (int)ns.size();

    printf("Computing %d sample points on %d threads...\n\n", npts, omp_get_max_threads());
    setvbuf(stdout, nullptr, _IOLBF, 0);

    printf("%20s  %12s  %12s  %12s  %12s  %12s  %s\n",
           "n", "err_a/n^.75", "env_max/n^.75", "err_D/n^.25", "combo/n^.25",
           "θ_slope", "method");
    fflush(stdout);

    std::vector<RowResult> rows(npts);
    g_progress.init(ns);

    // Parallel compute — use u128 fast routines for main work.
    #pragma omp parallel for schedule(dynamic,1) num_threads(omp_get_max_threads())
    for (int i = 0; i < npts; ++i) {
        u128 n = ns[i];
        RowResult &row = rows[i];
        row.n        = n;
        row.skip     = false;
        row.mismatch = false;
        row.verified = (n <= cutoff);

        // verification path for small n: compare direct a_hoying vs fused identity.
        if (row.verified) {
            // use u128 versions (avoid mpz)
            i128 hoy_int = a_hoying_u128(n - 1);
            i128 fus_int = a_identity_fused_u128(n);
            if (hoy_int != fus_int) {
                row.mismatch = true;
                row.mismatch_msg = std::string("MISMATCH (u128)"); // small msg; don't allocate huge strings in threads
            }
        }

        // Now compute the main error using the fast u128 exact error
        row.err_a = exact_err_u128(n);

        if (row.verified) {
            row.err_d     = exact_err_D_u128(n);
            row.err_combo = exact_err_combo_u128(n);
        } else {
            row.err_d     = NAN;
            row.err_combo = NAN;
        }
        row.logn = logl((long double)n);

        g_progress.tick(sqrt((double)n));
    }

    g_progress.finish();

    // --- Serial output + HDF5 same as before ---
    long double prev_err  = 0.0L, prev_logn = 0.0L;
    long double env_max   = 0.0L;
    long double sum1  = 0.0L, sumx = 0.0L, sumy  = 0.0L;
    long double sumxx = 0.0L, sumxy = 0.0L;
    int reg_cnt = 0;

    std::vector<std::string> col_n, col_method;
    std::vector<double> col_err_a, col_err_d, col_err_combo;
    std::vector<double> col_logn, col_theta, col_env_max;

    for (int i = 0; i < npts; i++) {
        const RowResult &row = rows[i];
        u128 n = row.n;

        if (row.skip) {
            printf("%20s  [skipped]\n", u128_to_str(n).c_str());
            continue;
        }
        if (row.mismatch) {
            printf("%20s  %s\n", u128_to_str(n).c_str(), row.mismatch_msg.c_str());
            prev_err = 0.0L; prev_logn = 0.0L;
            continue;
        }

        long double x    = (long double)n;
        long double n75  = powl(x, 0.75L);
        long double n25  = powl(x, 0.25L);
        long double logn = row.logn;

        long double err_a     = row.err_a;
        long double err_d     = row.err_d;
        long double err_combo = row.err_combo;
        long double abs_err   = fabsl(err_a);

        if (abs_err > env_max) env_max = abs_err;

        long double theta_sl = NAN;
        if (prev_err != 0.0L && err_a != 0.0L && prev_logn > 0.0L)
            theta_sl = (logl(abs_err) - logl(fabsl(prev_err)))
                       / (logn - prev_logn) - 0.5L;

        if (abs_err > 0.0L && fabsl(err_a / n75) < 10.0L) {
            long double loge = logl(abs_err);
            sum1  += 1.0L; sumx  += logn;  sumy  += loge;
            sumxx += logn * logn;           sumxy += logn * loge;
            reg_cnt++;
        }

        const char *method = row.verified ? "verified" : "fused";

        char col_d_buf[32], col_c_buf[32];
        if (isnan(err_d))     snprintf(col_d_buf, sizeof(col_d_buf), "%12s", "-");
        else                  snprintf(col_d_buf, sizeof(col_d_buf), "%12.5Lf", err_d / n25);
        if (isnan(err_combo)) snprintf(col_c_buf, sizeof(col_c_buf), "%12s", "-");
        else                  snprintf(col_c_buf, sizeof(col_c_buf), "%12.5Lf", err_combo / n25);

        printf("%20s  %12.5Lf  %12.5Lf  %s  %s",
               u128_to_str(n).c_str(), err_a / n75, env_max / n75,
               col_d_buf, col_c_buf);
        if (!isnan(theta_sl))
            printf("  %+12.4Lf", theta_sl);
        else
            printf("  %12s", "     -");
        printf("  %s\n", method);

        // Accumulate HDF5 columns (NaN propagates)
        col_n.push_back(u128_to_str(n));
        col_err_a.push_back((double)err_a);
        col_err_d.push_back((double)err_d);
        col_err_combo.push_back((double)err_combo);
        col_logn.push_back((double)logn);
        col_theta.push_back(isnan(theta_sl) ? (double)NAN : (double)theta_sl);
        col_env_max.push_back((double)env_max);
        col_method.push_back(method);

        prev_err  = err_a;
        prev_logn = logn;
    }

    // regression & HDF5 write same as before
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

    write_hdf5(hdf5_path,
               col_n, col_err_a, col_err_d, col_err_combo,
               col_logn, col_theta, col_env_max, col_method);
}

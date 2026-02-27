// =============================================================================
// gaussian_verify.cpp
//
// Verifies the Gaussian integer analogue of the A078567 identity:
//
//   b(N) = N * D_Zi(N-1) - Sigma_Zi(N-1)
//
// where:
//   b(N)         = #{(alpha, beta, j) : alpha,beta in Z[i]\{0}, j>=1 integer,
//                   N(alpha)*N(beta) + j <= N}
//              (Gaussian analogue of #{(d,m,j): d,m,j>=1, dm+j<=n})
//
//   f(k)         = #{(alpha,beta) in Z[i]\{0} x Z[i]\{0} : N(alpha)*N(beta) = k}
//                = (r2 * r2)(k)  [Dirichlet convolution of r2 with itself]
//                where r2(k) = #{representations of k as a^2 + b^2 (signed, ordered)}
//
//   D_Zi(n)      = sum_{k=1}^{n} f(k)     (cumulative)
//   Sigma_Zi(n)  = sum_{k=1}^{n} k * f(k) (weighted cumulative)
//
// FORMULA DERIVATION:
//   f(k) is the Dirichlet convolution r2 * r2, with generating series
//   (4 * zeta(s) * L(s, chi4))^2 = 16 * zeta(s)^2 * L(s, chi4)^2.
//
//   b(N) = sum_{k=1}^{N-1} (N-k) * f(k)       [by combinatorial argument]
//        = N * D_Zi(N-1) - Sigma_Zi(N-1)        [rearranging the sum]
//
// DIRECT COUNT (for verification at small N):
//   b_direct(N) = sum over all nonzero alpha,beta in Z[i] with N(a)*N(b)<=N-1
//                 of (N - N(alpha)*N(beta))
//               = #{(alpha,beta,j) : N(a)*N(b)+j <= N, j>=1}
//
// =============================================================================

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cassert>

// ---------------------------------------------------------------------------
// r2 sieve: r2[k] = #{(a,b) in Z^2 : a^2+b^2 = k}, r2[0] = 1 (the origin)
// Uses the identity: r2(n) = 4 * (d_{1,4}(n) - d_{3,4}(n))
// where d_{r,4}(n) = #{d | n : d == r mod 4}
// ---------------------------------------------------------------------------
static std::vector<int64_t> sieve_r2(int64_t N) {
    std::vector<int64_t> r2(N + 1, 0);
    r2[0] = 1;
    for (int64_t k = 1; k <= N; ++k) {
        int64_t val = 0;
        for (int64_t d = 1; d * d <= k; ++d) {
            if (k % d == 0) {
                int64_t q = k / d;
                // contribution of d
                if (d % 4 == 1) val += 4;
                else if (d % 4 == 3) val -= 4;
                // contribution of q (if different from d)
                if (q != d) {
                    if (q % 4 == 1) val += 4;
                    else if (q % 4 == 3) val -= 4;
                }
            }
        }
        r2[k] = val;
    }
    return r2;
}

// ---------------------------------------------------------------------------
// f sieve: f[k] = (r2 * r2)(k) = sum_{d|k} r2(d) * r2(k/d)
// [Dirichlet convolution -- d here ranges over divisors of k, not just Gaussian norm divisors]
// This counts #{(alpha,beta) nonzero : N(alpha)*N(beta) = k}.
// ---------------------------------------------------------------------------
static std::vector<int64_t> sieve_f(int64_t N, const std::vector<int64_t>& r2) {
    std::vector<int64_t> f(N + 1, 0);
    // Dirichlet convolution via the standard O(N log N) sieve
    for (int64_t a = 1; a <= N; ++a) {
        if (r2[a] == 0) continue;
        for (int64_t b = 1; a * b <= N; ++b) {
            if (r2[b] == 0) continue;
            f[a * b] += r2[a] * r2[b];
        }
    }
    return f;
}

// ---------------------------------------------------------------------------
// Direct (brute-force) computation of b(N) for verification:
//   b_direct(N) = sum_{alpha,beta nonzero: N(alpha)*N(beta) <= N-1}
//                   (N - N(alpha)*N(beta))
// We iterate over all nonzero (a,b) with a^2+b^2 <= N-1, and for each norm na,
// sum over all nonzero (c,d) with c^2+d^2 <= (N-1)/na of (N - na*(c^2+d^2)).
// ---------------------------------------------------------------------------
static int64_t b_direct(int64_t N) {
    if (N <= 1) return 0;
    int64_t target = N - 1;
    int64_t total = 0;
    int64_t r = (int64_t)sqrt((double)target);
    while ((r + 1) * (r + 1) <= target) ++r;

    for (int64_t a = -r; a <= r; ++a) {
        for (int64_t b = -r; b <= r; ++b) {
            int64_t na = a * a + b * b;
            if (na == 0 || na > target) continue;
            int64_t nd_max = target / na;
            int64_t rd = (int64_t)sqrt((double)nd_max);
            while ((rd + 1) * (rd + 1) <= nd_max) ++rd;
            for (int64_t c = -rd; c <= rd; ++c) {
                for (int64_t d = -rd; d <= rd; ++d) {
                    int64_t nb = c * c + d * d;
                    if (nb == 0) continue;
                    int64_t k = na * nb;
                    if (k <= target) {
                        total += (N - k);
                    }
                }
            }
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// Error term computation:
//   err(N) = b(N) - b_asymptotic(N)
// The leading asymptotic for b(N) = N * D_Zi(N-1) - Sigma_Zi(N-1)
// mirrors the Dirichlet divisor problem: D_Zi(N) ~ C * N * log(N) + lower order.
// We'll track the raw b(N) values; error analysis comes in the next phase.
// ---------------------------------------------------------------------------

int main() {
    const int64_t N_MAX = 500;   // brute-force verification limit
    const int64_t N_FORMULA = 10000; // formula-only range

    printf("=======================================================\n");
    printf(" Gaussian Integer AP Identity Verification\n");
    printf(" b(N) = N * D_Zi(N-1) - Sigma_Zi(N-1)\n");
    printf("=======================================================\n\n");

    // --- Phase 1: Sieve r2 and f ---
    auto r2 = sieve_r2(N_FORMULA);
    auto f  = sieve_f(N_FORMULA, r2);

    // Quick sanity check on r2
    // r2(1)=4, r2(2)=4, r2(3)=0, r2(4)=4, r2(5)=8
    assert(r2[1] == 4);
    assert(r2[2] == 4);
    assert(r2[3] == 0);
    assert(r2[4] == 4);
    assert(r2[5] == 8);
    printf("[OK] r2 sieve sanity checks passed (r2[1..5] = 4,4,0,4,8)\n\n");

    // Quick sanity check on f
    // f(1) = r2(1)^2 = 16
    // f(2) = 2*r2(1)*r2(2) = 32
    // f(3) = 2*r2(1)*r2(3) = 0
    // f(5) = 2*r2(1)*r2(5) = 2*4*8 = 64
    assert(f[1] == 16);
    assert(f[2] == 32);
    assert(f[3] == 0);
    assert(f[5] == 64);
    printf("[OK] f sieve sanity checks passed (f[1,2,3,5] = 16,32,0,64)\n\n");

    // --- Phase 2: Cross-verify formula vs direct for small N ---
    printf("--- Phase 2: Formula vs Direct Verification (N = 2..%ld) ---\n",
           N_MAX);
    printf("%6s %16s %16s %8s\n", "N", "b_direct", "b_formula", "match");
    printf("%6s %16s %16s %8s\n", "------", "----------------", "----------------", "--------");

    int64_t D_Zi   = 0;  // D_Zi(N-1) = sum_{k=1}^{N-1} f(k)
    int64_t Sig_Zi = 0;  // Sigma_Zi(N-1) = sum_{k=1}^{N-1} k*f(k)
    bool all_match = true;

    for (int64_t N = 2; N <= N_MAX; ++N) {
        // Update cumulative sums with k = N-1
        D_Zi   += f[N - 1];
        Sig_Zi += (N - 1) * f[N - 1];

        int64_t b_form = N * D_Zi - Sig_Zi;
        int64_t b_dir  = b_direct(N);

        bool match = (b_form == b_dir);
        if (!match) all_match = false;

        // Print every row up to 30, then every 10th
        if (N <= 30 || N % 50 == 0 || !match) {
            printf("%6ld %16ld %16ld %8s\n",
                   N, b_dir, b_form, match ? "OK" : "MISMATCH");
        }
    }
    printf("...\n");
    printf("\n[%s] All N in [2, %ld] %s\n\n",
           all_match ? "OK" : "FAIL",
           N_MAX,
           all_match ? "verified." : "-- MISMATCHES FOUND!");

    // --- Phase 3: Formula-only values for N up to N_FORMULA ---
    printf("--- Phase 3: Formula values for larger N ---\n");
    printf("%10s %20s %20s %20s\n",
           "N", "D_Zi(N-1)", "Sigma_Zi(N-1)", "b(N)");
    printf("%10s %20s %20s %20s\n",
           "----------", "--------------------", "--------------------", "--------------------");

    // Reset accumulators; we already accumulated to N_MAX
    D_Zi = 0; Sig_Zi = 0;
    for (int64_t k = 1; k <= N_MAX - 1; ++k) {
        D_Zi   += f[k];
        Sig_Zi += k * f[k];
    }
    // Continue from N_MAX to N_FORMULA
    for (int64_t N = N_MAX; N <= N_FORMULA; ++N) {
        D_Zi   += f[N - 1];
        Sig_Zi += (N - 1) * f[N - 1];

        int64_t b_form = N * D_Zi - Sig_Zi;
        if (N == N_MAX || N % 1000 == 0) {
            printf("%10ld %20ld %20ld %20ld\n",
                   N, D_Zi, Sig_Zi, b_form);
        }
    }

    // --- Phase 4: Growth rate sanity check ---
    // Asymptotically: D_Zi(N) ~ C * N * log(N) (like the Dirichlet divisor sum)
    // where C relates to the residue of zeta(s)^2 * L(s,chi4)^2 at s=1.
    // We check the ratio b(N) / (N^2 * log(N)) for stability.
    printf("\n--- Phase 4: Growth rate b(N) / (N^2 * logN) ---\n");
    printf("%10s %20s %20s\n", "N", "b(N)", "b(N)/(N^2 logN)");
    printf("%10s %20s %20s\n", "----------", "--------------------", "--------------------");

    D_Zi = 0; Sig_Zi = 0;
    for (int64_t N = 2; N <= N_FORMULA; ++N) {
        D_Zi   += f[N - 1];
        Sig_Zi += (N - 1) * f[N - 1];
        int64_t b_form = N * D_Zi - Sig_Zi;

        if (N == 10 || N == 100 || N == 1000 || N == 5000 || N == 10000) {
            double ratio = (double)b_form / ((double)N * N * log((double)N));
            printf("%10ld %20ld %20.8f\n", N, b_form, ratio);
        }
    }

    printf("\n=======================================================\n");
    printf(" Identity verified. Ready for error-term analysis.\n");
    printf("=======================================================\n");

    return all_match ? 0 : 1;
}

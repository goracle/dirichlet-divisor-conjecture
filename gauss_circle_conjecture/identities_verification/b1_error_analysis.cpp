// b1_error_analysis.cpp
// Compile: g++ -O3 -std=c++17 b1_error_analysis.cpp -lm -o b1_analysis
//
// Purpose: compute b1(N) via O(sqrt(N)) hyperbola blocking, analyze
// E(N) = b1(N) - (pi/2) N^2, print diagnostics, find sign changes,
// and test inheritance ratio E(N) / (N * Delta(N-1)) where
// Delta(M) = A(M) - pi*M, A(M) = sum_{k<=M} r2(k).
//
// Uses __int128 for exact integer accumulation. Outputs human-readable
// columns and CSV-like rows; redirect to file for plotting.

#include <bits/stdc++.h>
using namespace std;

using i64 = long long;
using i128 = __int128_t;
const long double PI_LD = acosl(-1.0L);

// --- helpers to convert/print __int128 safely ---
string i128_to_string(i128 x) {
    if (x == 0) return "0";
    bool neg = false;
    if (x < 0) { neg = true; x = -x; }
    string s;
    while (x != 0) {
        int digit = (int)(x % 10);
        s.push_back('0' + digit);
        x /= 10;
    }
    if (neg) s.push_back('-');
    reverse(s.begin(), s.end());
    return s;
}

// safe cast to long double for moderately large integers
long double i128_to_ld(i128 x) {
    bool neg = false;
    if (x < 0) { neg = true; x = -x; }
    // convert in chunks to avoid precision loss for extremely big values
    const long double LIM = (long double)1e18L;
    long double acc = 0.0L;
    while (x > 0) {
        i128 chunk = x % (i128)1000000000000000000ULL; // 1e18
        acc = acc + (long double)( (long double)chunk );
        x /= (i128)1000000000000000000ULL;
        if (x) acc *= LIM;
    }
    return neg ? -acc : acc;
}

// --- arithmetic functions for chi_{-4} prefix sums ---
// C(x) = sum_{n=1}^x chi_{-4}(n) ; values in {0,1}
inline i64 C_pref(i64 x) {
    if (x <= 0) return 0;
    // formula: (x+3)//4 - (x+1)//4
    return ( (x + 3) / 4 ) - ( (x + 1) / 4 );
}

// W(x) = sum_{n=1}^x n * chi_{-4}(n)
inline i128 W_pref(i64 x) {
    if (x <= 0) return (i128)0;
    i128 q = x / 4;
    int r = x % 4;
    i128 total = q * (i128)(-2); // each full period contributes -2
    if (r >= 1) total += (i128)(4 * q + 1); // chi = +1 at 4q+1
    if (r >= 3) total -= (i128)(4 * q + 3); // chi = -1 at 4q+3
    return total;
}

// --- b1 via hyperbola blocking (O(sqrt(N))) ---
// returns exact integer b1(N)
i128 b1_sqrt(i64 N) {
    if (N <= 1) return (i128)0;
    i64 P = N - 1;
    i128 total = 0;
    i64 d = 1;
    while (d <= P) {
        i64 v = P / d;
        i64 d_hi = P / v;
        i64 d_lo = d;
        i64 c_hi = C_pref(d_hi);
        i64 c_lo_1 = C_pref(d_lo - 1);
        i64 block_C = c_hi - c_lo_1;
        i128 block_W = W_pref(d_hi) - W_pref(d_lo - 1);

        // total += N * v * block_C - (v * (v + 1) / 2) * block_W
        i128 term1 = (i128)N * (i128)v * (i128)block_C;
        i128 vv = (i128)v * (i128)(v + 1) / 2;
        i128 term2 = vv * block_W;
        total += term1 - term2;

        d = d_hi + 1;
    }
    return (i128)4 * total;
}

// --- A(N) via divisor-sum / Dirichlet convolution ---
// A(N) = sum_{k<=N} r2(k) = 4 * sum_{d<=N} chi_{-4}(d) * floor(N/d)
// computed in O(sqrt(N)) by grouping equal floor(N/d)
i128 A_via_divisor_sum(i64 N) {
    if (N <= 0) return (i128)0;
    i128 total = 0;
    i64 d = 1;
    while (d <= N) {
        i64 t = N / d;
        i64 d_hi = N / t;
        i64 block_C = C_pref(d_hi) - C_pref(d - 1);
        // contribution = 4 * block_C * t
        total += (i128)4 * (i128)block_C * (i128)t;
        d = d_hi + 1;
    }
    return total;
}

// --- compute r2(n) directly for small n (used only optionally) ---
i64 chi_m4(i64 n) {
    if ((n & 1) == 0) return 0;
    return (n % 4 == 1) ? 1 : -1;
}
i64 r2_direct(i64 n) {
    i64 s = 0;
    for (i64 d = 1; d * d <= n; ++d) {
        if (n % d == 0) {
            s += chi_m4(d);
            if (d * d != n) s += chi_m4(n / d);
        }
    }
    return 4 * s;
}

// --- main experiment driver ---
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // default N list (same idea as your Python): you can edit or pass custom args
    vector<i64> Ns = {1000LL, 5000LL, 10000LL, 50000LL, 100000LL,
                      500000LL, 1000000LL, 5000000LL, 10000000LL};

    // If user supplied arguments (space-separated integers), use them as explicit N list
    if (argc > 1) {
        Ns.clear();
        for (int i = 1; i < argc; ++i) {
            long long v = atoll(argv[i]);
            if (v > 1) Ns.push_back((i64)v);
        }
        if (Ns.empty()) {
            cerr << "No valid N provided on command line; using defaults.\n";
            Ns = {1000LL, 5000LL, 10000LL, 50000LL, 100000LL};
        }
    }

    cout << "=== b1 hyperbola O(sqrt(N)) error analysis ===\n";
    cout << "Columns: N, b1(N) (integer), E(N) = b1 - (pi/2)N^2, E/N^1.5, E/N^(4/3)\n\n";
    cout << setw(12) << "N"
         << setw(26) << "b1(N)"
         << setw(18) << "E(N)"
         << setw(16) << "E/N^1.5"
         << setw(16) << "E/N^1.3333"
         << "\n";
    cout << string(88, '-') << "\n";

    for (i64 N : Ns) {
        i128 b1 = b1_sqrt(N);
        long double b1_ld = i128_to_ld(b1);
        long double asym = (PI_LD / 2.0L) * (long double)N * (long double)N;
        long double E = b1_ld - asym;
        long double s13 = expl(1.0L * logl((long double)N) * 1.3333333333333333L); // N^(4/3)
        // safer: compute N^p directly:
        long double N_1p5 = pow((long double)N, 1.5L);
        long double N_4_3 = pow((long double)N, 4.0L/3.0L);

        cout << setw(12) << N
             << setw(26) << i128_to_string(b1)
             << setw(18) << fixed << setprecision(6) << E
             << setw(16) << scientific << setprecision(8) << (E / N_1p5)
             << setw(16) << scientific << setprecision(6) << (E / N_4_3)
             << "\n";
        // reset format
        cout << dec << defaultfloat;
    }
    cout << "\n";

    // --- find sign changes up to a max N (dense scan) ---
    i64 Nmax = 50000; // default same as your Python for sign changes
    {
        cout << "--- Sign changes (dense scan) up to N = " << Nmax << " ---\n";
        vector<pair<i64,long double>> crossings;
        int prev_sign = 0;
        for (i64 N = 2; N <= Nmax; ++N) {
            i128 b1 = b1_sqrt(N);
            long double b1_ld = i128_to_ld(b1);
            long double E = b1_ld - (PI_LD / 2.0L) * (long double)N * (long double)N;
            int sign = (E >= 0.0L) ? 1 : -1;
            if (prev_sign != 0 && sign != prev_sign) {
                crossings.emplace_back(N, E);
            }
            prev_sign = sign;
        }
        cout << "Found " << crossings.size() << " sign changes.\n";
        if (!crossings.empty()) {
            cout << "First 20 crossings (N, E):\n";
            for (size_t i = 0; i < crossings.size() && i < 20; ++i) {
                cout << "  N=" << setw(8) << crossings[i].first
                     << "  E=" << fixed << setprecision(6) << crossings[i].second << "\n";
            }
        } else {
            cout << "  No sign changes found in this range.\n";
        }
        cout << "\n";
    }

    // --- inheritance test: compare E(N) to N * Delta(N-1) for moderate N ---
    vector<i64> moderate_N = {100LL, 200LL, 500LL, 1000LL, 2000LL, 5000LL};
    cout << "--- Inheritance test: E(N) / (N * Delta(N-1)) ---\n";
    cout << setw(10) << "N"
         << setw(14) << "Delta(N-1)"
         << setw(14) << "N*Delta"
         << setw(14) << "E(N)"
         << setw(12) << "ratio"
         << "\n";
    cout << string(64, '-') << "\n";
    for (i64 N : moderate_N) {
        i128 Aprev_i128 = A_via_divisor_sum(N - 1);
        long double Aprev_ld = i128_to_ld(Aprev_i128);
        long double Delta = Aprev_ld - PI_LD * (long double)(N - 1);
        i128 b1 = b1_sqrt(N);
        long double b1_ld = i128_to_ld(b1);
        long double E = b1_ld - (PI_LD / 2.0L) * (long double)N * (long double)N;
        long double NDelta = (long double)N * Delta;
        long double ratio = (fabsl(NDelta) > 0.0L) ? (E / NDelta) : numeric_limits<long double>::infinity();

        cout << setw(10) << N
             << setw(14) << fixed << setprecision(4) << Delta
             << setw(14) << fixed << setprecision(4) << NDelta
             << setw(14) << fixed << setprecision(4) << E
             << setw(12) << fixed << setprecision(6) << ratio
             << "\n";
        cout << defaultfloat;
    }
    cout << "\n";

    cout << "=== done ===\n";
    return 0;
}

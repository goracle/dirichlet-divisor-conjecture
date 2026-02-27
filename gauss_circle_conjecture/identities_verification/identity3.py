#!/usr/bin/env python3

# ----- chi_{-4} -----
def chi_m4(n):
    if n % 2 == 0:
        return 0
    return 1 if n % 4 == 1 else -1


# ----- prefix sum C(x) = sum_{n<=x} chi_{-4}(n) -----
def C(x):
    if x <= 0:
        return 0
    return (x + 3)//4 - (x + 1)//4


# ----- O(N) reference implementation -----
def S_rhs_naive(N):
    total = 0
    for d in range(1, N+1):
        v = N // d
        total += 4 * chi_m4(d) * v * v
    return total


# ----- O(sqrt(N)) hyperbola implementation -----
def S_sqrt(N):
    total = 0
    d = 1
    while d <= N:
        v = N // d
        dmax = N // v

        block_sum = C(dmax) - C(d-1)

        total += 4 * v * v * block_sum

        d = dmax + 1

    return total


# ----- verification -----
if __name__ == "__main__":
    for N in [10, 50, 100, 500, 1000]:
        a = S_rhs_naive(N)
        b = S_sqrt(N)
        print(f"N={N}  naive={a}  sqrt={b}  equal? {a==b}")

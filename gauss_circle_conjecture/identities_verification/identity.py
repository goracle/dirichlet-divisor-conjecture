import math
from collections import defaultdict

# ---- chi_{-4} ----
def chi_m4(d):
    if d % 2 == 0:
        return 0
    r = d % 4
    if r == 1:
        return 1
    if r == 3:
        return -1
    return 0

# ---- r2 via divisor formula ----
def r2(n):
    s = 0
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            s += chi_m4(d)
            if d*d != n:
                s += chi_m4(n//d)
    return 4*s

# ---- direct geometric r2 check ----
def r2_direct(n):
    cnt = 0
    R = int(math.isqrt(n))
    for a in range(-R, R+1):
        b2 = n - a*a
        if b2 < 0:
            continue
        b = int(math.isqrt(b2))
        if b*b == b2:
            if b == 0:
                cnt += 1
            else:
                cnt += 2
    return cnt

# ---- verify r2 identity ----
def test_r2(limit=200):
    for n in range(1, limit+1):
        if r2(n) != r2_direct(n):
            print("Mismatch at", n)
            return
    print("r2 identity verified up to", limit)

# ---- convolution g = r2 * r2 ----
def g_conv(n):
    s = 0
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            s += r2(d)*r2(n//d)
            if d*d != n:
                s += r2(n//d)*r2(d)
    return s

# ---- brute Gaussian factor count by norm ----
def gaussian_norms(limit):
    # map norm -> count of gaussian integers
    norms = defaultdict(int)
    R = int(math.isqrt(limit))
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            n = a*a + b*b
            if 1 <= n <= limit:
                norms[n] += 1
    return norms

# ---- verify convolution identity ----
def test_convolution(limit=200):
    norms = gaussian_norms(limit)
    for n in range(1, limit+1):
        if g_conv(n) != sum(norms[d]*norms[n//d]
                            for d in range(1, n+1)
                            if n%d==0):
            print("Convolution mismatch at", n)
            return
    print("Gaussian convolution verified up to", limit)

# ---- summatory D_G ----
def D_G(N):
    return sum(g_conv(n) for n in range(1, N+1))

# ---- quick asymptotic sanity ----
def test_summatory(limit=500):
    for N in [50,100,200,400]:
        val = D_G(N)
        approx = N*math.log(N)
        print(N, val/approx)

def gaussian_divisors_bounded(Y, X):
    # LHS
    lhs = 0
    Rb = int(math.isqrt(Y))
    Ra = int(math.isqrt(X))

    betas = []
    for a in range(-Rb, Rb+1):
        for b in range(-Rb, Rb+1):
            if 0 < a*a+b*b <= Y:
                betas.append((a,b))

    alphas = []
    for a in range(-Ra, Ra+1):
        for b in range(-Ra, Ra+1):
            if 0 < a*a+b*b <= X:
                alphas.append((a,b))

    # count divisors
    for (ba,bb) in betas:
        for (aa,ab) in alphas:
            # check divisibility
            denom = aa*aa + ab*ab
            if denom == 0:
                continue
            # compute gamma = beta / alpha
            num_re = ba*aa + bb*ab
            num_im = bb*aa - ba*ab
            if num_re % denom == 0 and num_im % denom == 0:
                lhs += 1

    # RHS
    rhs = 0
    for (aa,ab) in alphas:
        for (ba,bb) in betas:
            # compute gamma = beta / alpha
            denom = aa*aa + ab*ab
            if denom == 0:
                continue
            num_re = ba*aa + bb*ab
            num_im = bb*aa - ba*ab
            if num_re % denom == 0 and num_im % denom == 0:
                rhs += 1

    return lhs, rhs

if __name__ == "__main__":
    test_r2()
    test_convolution()
    test_summatory()

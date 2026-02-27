import math

# ---------------------------
# Generate Gaussian integers
# ---------------------------

def gaussian_with_norm_leq(N):
    """Return list of (a,b) with 0 < a^2+b^2 <= N"""
    R = int(math.isqrt(N))
    pts = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            n = a*a + b*b
            if 0 < n <= N:
                pts.append((a,b))
    return pts


# ---------------------------
# Divisibility test in Z[i]
# ---------------------------

def divides(alpha, beta):
    """Return True if alpha | beta in Z[i]"""
    aa, ab = alpha
    ba, bb = beta

    denom = aa*aa + ab*ab
    if denom == 0:
        return False

    # beta / alpha = (beta * conjugate(alpha)) / N(alpha)
    num_re = ba*aa + bb*ab
    num_im = bb*aa - ba*ab

    return (num_re % denom == 0) and (num_im % denom == 0)


# ---------------------------
# Verify hardcore identity
# ---------------------------

def verify_hardcore(X, Y):
    alphas = gaussian_with_norm_leq(X)
    betas  = gaussian_with_norm_leq(Y)

    # ---- LHS ----
    lhs = 0
    for beta in betas:
        for alpha in alphas:
            if divides(alpha, beta):
                lhs += 1

    # ---- RHS ----
    rhs = 0
    for alpha in alphas:
        Na = alpha[0]**2 + alpha[1]**2
        max_gamma_norm = Y // Na
        if max_gamma_norm <= 0:
            continue
        gammas = gaussian_with_norm_leq(max_gamma_norm)
        rhs += len(gammas)

    return lhs, rhs


# ---------------------------
# Run small tests
# ---------------------------

if __name__ == "__main__":
    for X,Y in [(5,20),(10,50),(20,100)]:
        lhs, rhs = verify_hardcore(X,Y)
        print(f"X={X}, Y={Y}  LHS={lhs}  RHS={rhs}  equal? {lhs==rhs}")

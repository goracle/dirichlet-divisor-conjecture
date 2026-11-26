#!/usr/bin/env python3
"""Verify: err_osc(m+1) = sum_{i<=s} r_i(i-2-r_i)/i - m/12 + sqrt(m)*P({sqrt m}) + C + o(1)
with r_i = m mod i, s=floor(sqrt m), P(u)=1-2u/3+2u^2-4u^3/3.  Fits P's coefficients by least squares."""
import numpy as np, random
from math import isqrt
from mpmath import mp, mpf, log as mlog, sqrt as msqrt
mp.dps=50
g=mpf('0.5772156649015328606065120900824024310421593359399')
def D(m):
    s=isqrt(m); return 2*sum(m//i for i in range(1,s+1))-s*s
def sig(m):
    s=isqrt(m); T=lambda q:q*(q+1)//2
    return 2*sum(i*T(m//i) for i in range(1,s+1))-T(s)**2
def err_at(M):
    a=M*D(M-1)-sig(M-1); N=mpf(M)
    return mpf(a)-(N**2/2*mlog(N)+(g-mpf(3)/4)*N**2+N/4)
def R(m):
    s=isqrt(m); return sum(mpf(m%i)*(i-2-(m%i))/i for i in range(1,s+1))
random.seed(3); rows=[]
for _ in range(60):
    s0=random.choice([3000,5000,8000]); m=s0*s0+random.randrange(0,2*s0+1)
    t=msqrt(m); u=t-isqrt(m); rows.append((float(t),float(u),float(err_at(m+1)-R(m)+mpf(m)/12)))
t,u,E=np.array(rows).T
X=np.vstack([t,t*u,t*u**2,t*u**3,np.ones_like(t)]).T
c=np.linalg.lstsq(X,E,rcond=None)[0]
print("fit coefficients [1, -2/3, 2, -4/3, C]:",np.round(c,4)," max resid",np.abs(X@c-E).max())
Pfix=1-2*u/3+2*u**2-4*u**3/3
C=(E-t*Pfix).mean()
print("with exact rational P: C=%.3f, max |E - t*P(u) - C| = %.3f (t~3000-8000)"%(C,np.abs(E-t*Pfix-C).max()))

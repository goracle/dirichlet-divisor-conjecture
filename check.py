import numpy as np, math, random
from math import isqrt, log, pi, sqrt
g=0.5772156649015328606065120900824024
def D(m):
    s=isqrt(m); return 2*sum(m//i for i in range(1,s+1))-s*s
def sig(m):
    s=isqrt(m); T=lambda q:q*(q+1)//2
    return 2*sum(i*T(m//i) for i in range(1,s+1))-T(s)**2
def a(n): return n*D(n-1)-sig(n-1)
# brute check
tau=[0]*60
for i in range(1,60):
    for j in range(i,60,i): tau[j]+=1
for n in range(2,40):
    assert a(n)==sum((n-k)*tau[k] for k in range(1,n)),n
print("identity ok")
from mpmath import mp, mpf, log as mlog
mp.dps=40
def err(n):
    n_=mpf(n)
    return mpf(a(n))-(n_**2/2*mlog(n_)+(mpf(g)-mpf(3)/4)*n_**2+n_/4)
K=400000
d=np.zeros(K+1,dtype=np.int32)
for i in range(1,K+1): d[i::i]+=1
k=np.arange(1,K+1,dtype=np.float64)
w=d[1:]*k**-1.25/(2*sqrt(2)*pi**2)
random.seed(1)
for n in [10**5+3, 3*10**5+17, 10**6+1, 7*10**6+5, 10**7+123]:
    e=float(err(n))/n**0.75
    ph=4*pi*np.sqrt(k*n)-pi/4
    s=float(np.dot(w,np.sin(ph)))
    print(n, e, s, e-s)

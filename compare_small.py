import json,random,numpy as np,math
from math import isqrt
from mpmath import mp,mpf,log as mlog
mp.dps=40
g=mpf('0.5772156649015328606065120900824024')
def D(m):
    s=isqrt(m); return 2*sum(m//i for i in range(1,s+1))-s*s
def sig(m):
    s=isqrt(m); T=lambda q:q*(q+1)//2
    return 2*sum(i*T(m//i) for i in range(1,s+1))-T(s)**2
def f(n):
    a=n*D(n-1)-sig(n-1); N=mpf(n)
    return float((mpf(a)-(N**2/2*mlog(N)+(g-mpf(3)/4)*N**2+N/4))/N**mpf('0.75'))
random.seed(7)
xs=[f(int(math.exp(random.uniform(math.log(2e8),math.log(1e9))))) for _ in range(1500)]
xs=np.array(xs)
m=json.load(open('model_stats.json'))['quantiles']
print("exact n~2e8..1e9 (1500 draws) vs model")
print("std",xs.std(),"model 0.04478")
for q in [0.01,0.05,0.25,0.5,0.75,0.95,0.99]:
    print(q,round(float(np.quantile(xs,q)),4),round(m[str(q)],4))
print("min/max",xs.min(),xs.max())

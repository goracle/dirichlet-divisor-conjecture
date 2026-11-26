#!/usr/bin/env python3
"""
Heath-Brown-style 'dice model' for the Riesz-mean error of tau:
    f(x) = err_osc(x)/x^(3/4) = c * sum_k d(k) k^(-5/4) sin(4 pi sqrt(k x) - pi/4),  c = 1/(2 sqrt2 pi^2)
Write k = m^2 j (j squarefree).  Phases theta_j = 4 pi sqrt(j x) are modelled as
independent uniform; then f = sum_j g_j(theta_j) with
    g_j(t) = c j^(-5/4) sum_m d(m^2 j) m^(-5/2) sin(m t - pi/4).
Outputs: model_stats.json (mean, var, quantiles, exact limsup/liminf constants with tail
extrapolation) and model_quantiles.csv.
Usage: python3 dice_model.py [--draws 2000000] [--J 2000] [--Jsup 20000]
"""
import numpy as np, json, argparse, math
from math import pi, sqrt
ap=argparse.ArgumentParser()
ap.add_argument('--draws',type=int,default=2_000_000)
ap.add_argument('--J',type=int,default=2000)       # squarefree j kept exactly in sampling
ap.add_argument('--Jsup',type=int,default=20000)   # j range for sup/inf sums
ap.add_argument('--M',type=int,default=64)         # harmonics per j
ap.add_argument('--G',type=int,default=8192)       # theta grid
ap.add_argument('--seed',type=int,default=12345)
A=ap.parse_args()
c=1/(2*sqrt(2)*pi**2)

def factor(n):
    f={};p=2
    while p*p<=n:
        while n%p==0: f[p]=f.get(p,0)+1; n//=p
        p+=1
    if n>1: f[n]=f.get(n,0)+1
    return f
def sqfree_list(N):
    return [j for j in range(1,N+1) if all(e==1 for e in factor(j).values())]
def d_m2j(m,jf):
    fm=factor(m); r=1
    for p in set(fm)|set(jf):
        r*= 2*fm.get(p,0)+(1 if p in jf else 0)+1
    return r
dm_cache={}
def coeffs(j):
    jf=factor(j)
    return np.array([c*j**-1.25*d_m2j(m,jf)*m**-2.5 for m in range(1,A.M+1)])

G=A.G; th=2*pi*np.arange(G)/G
S=np.sin(np.outer(th,np.arange(1,A.M+1))*1.0-pi/4*0)  # placeholder, rebuilt below
# sin(m t - pi/4) table
ms=np.arange(1,A.M+1)
S=np.sin(np.outer(th,ms)-pi/4)       # G x M

sq=sqfree_list(A.Jsup)
print("squarefree j up to",A.Jsup,":",len(sq))
# per-j max/min and tables
tabs=[];mx=[];mn=[];var=[]
for j in sq:
    cf=coeffs(j); g=S@cf
    mx.append(g.max()); mn.append(g.min()); var.append(0.5*float(np.dot(cf,cf)))
    if j<=A.J: tabs.append(g.astype(np.float32))
mx=np.array(mx);mn=np.array(mn);var=np.array(var);sqa=np.array(sq)
# --- sup/inf constants with tail extrapolation S(J)=Sinf - a J^(-1/4)
cm=np.cumsum(mx);cn=np.cumsum(mn)
def extrap(cs):
    Js=np.array([A.Jsup//8,A.Jsup//4,A.Jsup//2,A.Jsup]); idx=[np.searchsorted(sqa,J,side='right')-1 for J in Js]
    y=np.array([cs[i] for i in idx]); X=np.vstack([np.ones(4),sqa[idx].astype(float)**-0.25]).T
    coef,*_=np.linalg.lstsq(X,y,rcond=None); return coef[0],coef[1]
sup_inf,sup_a=extrap(cm); inf_inf,inf_a=extrap(cn)
# --- tail std for j>J (used as Gaussian tail in sampling)
tail_var=var[sqa>A.J].sum()
# remainder beyond Jsup: var_j ~ 0.5 c^2 j^-2.5 * (sum d^2 m^-5 ~ 4*1.04) ; integral over squarefree density 6/pi^2
tail_var+= (6/pi**2)*0.5*c*c*4.2*(A.Jsup**-1.5)/1.5
print("tail std beyond J:",math.sqrt(tail_var))
rng=np.random.default_rng(A.seed)
T=np.stack(tabs)            # nj x G float32
nj=T.shape[0]
out=np.empty(A.draws,dtype=np.float64)
CH=200_000
for s in range(0,A.draws,CH):
    n=min(CH,A.draws-s); acc=np.zeros(n)
    for i in range(nj):
        idx=rng.integers(0,G,size=n); acc+=T[i][idx]
    acc+=rng.normal(0,math.sqrt(tail_var),size=n)
    out[s:s+n]=acc
qs=[0,1e-6,1e-5,1e-4,1e-3,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,0.999,0.9999,0.99999,1-1e-6,1]
quant={str(q):float(np.quantile(out,q)) for q in qs}
stats=dict(draws=A.draws,J=A.J,Jsup=A.Jsup,mean=float(out.mean()),std=float(out.std()),
  var_exact_partial=float(var.sum()),sample_min=float(out.min()),sample_max=float(out.max()),
  limsup_const_extrap=float(sup_inf),liminf_const_extrap=float(inf_inf),
  limsup_partial=float(cm[-1]),liminf_partial=float(cn[-1]),
  tail_fit_coeff_sup=float(sup_a),tail_fit_coeff_inf=float(inf_a),
  abs_ceiling_zeta=float(1.0*(c*float(__import__('mpmath').zeta(1.25))**2)),quantiles=quant)
json.dump(stats,open('model_stats.json','w'),indent=1)
with open('model_quantiles.csv','w') as f:
    f.write('q,value\n'); [f.write(f'{q},{v}\n') for q,v in quant.items()]
np.save('model_samples_head.npy',out[:200000])
print(json.dumps({k:v for k,v in stats.items() if k!='quantiles'},indent=1))

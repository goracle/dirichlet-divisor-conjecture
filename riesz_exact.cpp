// riesz_exact.cpp -- exact Riesz-mean error of tau (OEIS A078567) at huge n, plus a phase-aligned extreme hunt.
//   a(n)   = sum_{k<n} (n-k) tau(k) = n*D(n-1) - sigma_tau(n-1)        (O(sqrt n), exact, mod 2^128)
//   err(n) = a(n) - [ n^2/2 log n + (gamma-3/4) n^2 + n/4 ]  = int_0^n Delta
//   f(n)   = err(n)/n^(3/4)    (dice-model: bounded, skewed, std ~0.0448)
// Modes (per decade p: n in [10^p, min(10^(p+1), 4e18)]):
//   rand : R log-uniform random n, exact f(n)             -> rows "rand"
//   hunt : multi-stage proxy search for max and min of f  -> rows "hunt+" / "hunt-"
//   Proxy = Voronoi series  f(x) ~ c sum_k d(k) k^-5/4 sin(4 pi sqrt(kx) - pi/4), c=1/(2 sqrt2 pi^2)
//   (checked vs exact to ~1e-5 at n~1e6; every hunted point is then re-verified EXACTLY).
// Build: g++ -O3 -march=native -fopenmp -std=gnu++17 riesz_exact.cpp -lquadmath -o riesz_exact
// Run:   ./riesz_exact --selftest
//        OMP_NUM_THREADS=$(nproc) ./riesz_exact --dec 12 17 --R 200 --csv=riesz.csv
#include <bits/stdc++.h>
#include <quadmath.h>
#include <omp.h>
using namespace std;
typedef unsigned long long u64; typedef unsigned __int128 u128; typedef __float128 f128;
static const f128 GAMMA = 0.57721566490153286060651209008240243104215933593992Q;
static const long double PI_L = 3.14159265358979323846264338327950288L;
static const double C_VOR = 1.0/(2.0*sqrt(2.0)*M_PI*M_PI);
static const u64 NMAX = 4000000000000000000ULL;

static u64 isqrt64(u64 m){ u64 s=(u64)sqrtl((long double)m); while((u128)s*s>m) --s; while((u128)(s+1)*(s+1)<=m) ++s; return s; }

// a(n) as unsigned 128-bit, valid while true a(n) < 2^128 (n <= ~4.5e18); intermediates may wrap.
static u128 a_exact(u64 n){
    u64 m=n-1, s=isqrt64(m);
    u128 Dsum=0, sg=0;
    for(u64 i=1;i<=s;++i){ u64 q=m/i; Dsum+=q; u128 T=(u128)q*(q+1)/2; sg+=(u128)i*T; }
    u128 D=2*Dsum-(u128)s*s;
    u128 Ts=(u128)s*(s+1)/2;
    u128 sig=2*sg-Ts*Ts;
    return (u128)n*D-sig;
}
static double f_exact(u64 n){
    f128 N=(f128)n, A=(f128)a_exact(n);
    f128 asym=N*N/2*logq(N)+(GAMMA-0.75Q)*N*N+N/4;
    f128 n34=sqrtq(N)*sqrtq(sqrtq(N));
    return (double)((A-asym)/n34);
}
// ---------- Voronoi proxy ----------
static vector<double> W; static vector<long double> SK; static int KMAX=0;
static void init_proxy(int K){
    KMAX=K; vector<int> d(K+1,0);
    for(int i=1;i<=K;++i) for(int j=i;j<=K;j+=i) ++d[j];
    W.assign(K+1,0); SK.assign(K+1,0);
    for(int k=1;k<=K;++k){ W[k]=C_VOR*d[k]*pow((double)k,-1.25); SK[k]=sqrtl((long double)k); }
}
static inline double proxy(long double t,int K){          // f at x=t^2
    double s=0;
    for(int k=1;k<=K;++k){ long double r=2.0L*t*SK[k]; long double u=r-floorl(r);
        s+=W[k]*sin(2.0*M_PI*(double)u-M_PI/4); }
    return s;
}
// ---------- helpers ----------
struct Row{ string kind; int dec; u64 n; double proxy; double f; };
static vector<Row> rows;
static int selftest(){
    // 1) a_exact vs brute force
    vector<u64> tau(4001,0); for(int i=1;i<=4000;++i) for(int j=i;j<=4000;j+=i) ++tau[j];
    for(u64 n=2;n<=4000;++n){ u128 b=0; for(u64 k=1;k<n;++k) b+=(u128)(n-k)*tau[k];
        if(b!=a_exact(n)){ fprintf(stderr,"a_exact mismatch n=%llu\n",n); return 2; } }
    puts("selftest: a_exact == brute force for n<=4000");
    // 2) f_exact vs Voronoi proxy
    init_proxy(200000);
    double worst=0; for(u64 n: {100003ULL,300017ULL,1000001ULL,7000005ULL,10000123ULL,123456789ULL,987654321ULL}){
        double fe=f_exact(n), fp=proxy(sqrtl((long double)n),200000); worst=max(worst,fabs(fe-fp));
        printf("  n=%llu  exact=% .6f  proxy=% .6f  diff=% .2e\n",n,fe,fp,fe-fp); }
    if(worst>2e-3){ fprintf(stderr,"proxy/exact disagree (%.3g)\n",worst); return 2; }
    puts("selftest OK (python cross-check values: n=100003 0.064611, n=1000001 -0.036850, n=10000123 0.048713)");
    return 0;
}
int main(int argc,char**argv){
    int dlo=10,dhi=16,R=100,Win=0; double Wt=3e5; int K1=12,K2=3000,K3=100000,C1=40,top2=3; string csv="riesz.csv";
    u64 seed=20261125; bool dorand=true,dohunt=true;
    for(int i=1;i<argc;++i){ string a=argv[i];
        auto nxt=[&](){return atof(argv[++i]);};
        if(a=="--selftest") return selftest();
        else if(a=="--dec"){ dlo=(int)nxt(); dhi=(int)nxt(); }
        else if(a=="--R") R=(int)nxt(); else if(a=="--win") Win=(int)nxt(); else if(a=="--W") Wt=nxt();
        else if(a=="--K1") K1=(int)nxt(); else if(a=="--K2") K2=(int)nxt(); else if(a=="--K3") K3=(int)nxt();
        else if(a=="--cands") C1=(int)nxt(); else if(a=="--seed") seed=(u64)nxt();
        else if(a=="--norand") dorand=false; else if(a=="--nohunt") dohunt=false;
        else if(a.rfind("--csv=",0)==0) csv=a.substr(6); }
    if(dhi>18) dhi=18; int nth=omp_get_max_threads(); if(Win<=0) Win=nth;
    if(selftest()) return 2;
    init_proxy(max(K3,max(K2,K1)));
    for(int p=dlo;p<=dhi;++p){
        double lo=pow(10.0,p), hi=min(pow(10.0,p+1),(double)NMAX);
        if(dorand){
            vector<Row> loc(R);
            #pragma omp parallel for schedule(dynamic,1)
            for(int r=0;r<R;++r){ mt19937_64 g(seed*1000003ULL+p*7919ULL+r); uniform_real_distribution<double> U(log(lo),log(hi));
                u64 n=(u64)exp(U(g)); loc[r]={"rand",p,n,NAN,f_exact(n)}; }
            for(auto&x:loc) rows.push_back(x);
        }
        if(dohunt){
            for(int sgn=+1;sgn>=-1;sgn-=2){
                vector<Row> loc(Win);
                #pragma omp parallel for schedule(dynamic,1)
                for(int w=0;w<Win;++w){
                    mt19937_64 g(seed*7777ULL+p*104729ULL+w*13+(sgn>0)); 
                    long double tlo=sqrtl(lo),thi=sqrtl(hi); uniform_real_distribution<double> U(0,1);
                    long double t0=tlo+(thi-tlo-Wt)*U(g);
                    double step=1.0/(16.0*sqrt((double)K1));                // 8 pts per period of the K1-th term
                    long long NP=(long long)(Wt/step);
                    // stage 1: local extrema of K1-proxy, keep top C1
                    vector<pair<double,long double>> cand; double a=0,b=0,c;
                    auto val=[&](long long i){ return sgn*proxy(t0+(long double)i*step,K1); };
                    a=val(0); b=val(1);
                    priority_queue<pair<double,long double>,vector<pair<double,long double>>,greater<>> pq;
                    for(long long i=2;i<NP;++i){ c=val(i);
                        if(b>a&&b>=c){ pq.push({b,t0+(long double)(i-1)*step}); if((int)pq.size()>C1) pq.pop(); }
                        a=b; b=c; }
                    // stage 2: refine each with K2 terms over +-0.5/sqrt(K1)... (neglected terms move peak by <~ period of K1-th term)
                    double span=0.6/ sqrt((double)K1); double st2=1.0/(16.0*sqrt((double)K2));
                    vector<pair<double,long double>> s2;
                    while(!pq.empty()){ long double tc=pq.top().second; pq.pop(); double best=-1e9; long double tb=tc;
                        for(double d=-span;d<=span;d+=st2){ double v=sgn*proxy(tc+d,K2); if(v>best){best=v;tb=tc+d;} }
                        s2.push_back({best,tb}); }
                    sort(s2.begin(),s2.end(),greater<>()); if((int)s2.size()>top2) s2.resize(top2);
                    // stage 3: K3 terms, local, then exact
                    double bestExact=-1e9; Row br{"",p,0,0,0};
                    for(auto&pr:s2){ long double tc=pr.second; double best=-1e9; long double tb=tc; double st3=1.0/(16.0*sqrt((double)K3));
                        for(double d=-3*st2;d<=3*st2;d+=st3){ double v=sgn*proxy(tc+d,K3); if(v>best){best=v;tb=tc+d;} }
                        u64 n=(u64)llroundl(tb*tb); if(n>NMAX) n=NMAX; double fe=f_exact(n);
                        if(sgn*fe>bestExact){ bestExact=sgn*fe; br={sgn>0?"hunt+":"hunt-",p,n,sgn*best,fe}; } }
                    loc[w]=br;
                }
                for(auto&x:loc) rows.push_back(x);
            }
        }
        // per-decade summary
        double mxr=-1e9,mnr=1e9,mxh=-1e9,mnh=1e9,s=0,s2=0; int nr=0;
        for(auto&x:rows) if(x.dec==p){ if(x.kind=="rand"){ mxr=max(mxr,x.f); mnr=min(mnr,x.f); s+=x.f; s2+=x.f*x.f; ++nr; }
            else if(x.kind=="hunt+") mxh=max(mxh,x.f); else if(x.kind=="hunt-") mnh=min(mnh,x.f); }
        printf("dec 1e%d  rand: n=%d mean=% .4f std=%.4f min=% .4f max=% .4f | hunt: max=% .4f min=% .4f\n",p,nr,nr?s/nr:0.0,nr?sqrt(max(0.0,s2/nr-(s/nr)*(s/nr))):0.0,mnr,mxr,mxh,mnh);
        fflush(stdout);
    }
    FILE*F=fopen(csv.c_str(),"w"); fprintf(F,"kind,dec,n,proxy,f\n");
    for(auto&x:rows) fprintf(F,"%s,%d,%llu,%.8f,%.8f\n",x.kind.c_str(),x.dec,x.n,x.proxy,x.f); fclose(F);
    return 0;
}

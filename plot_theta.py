#!/usr/bin/env python3
"""
plot_theta.py — read ./delta output from stdin, plot theta_OLS(N).

Usage:
    ./delta 1000000000000000000 1000 1.001 2>/dev/null | python3 plot_theta.py

For each decade boundary N = 10^k computes two OLS estimates:

  cumulative  — regression over ALL points with n <= N.
                Biased upward at large N because the dense small-n points
                (where theta was still climbing from ~0.22) anchor the fit.

  windowed    — regression over only the points in the single decade
                10^(k-1) < n <= 10^k.
                Noisier per-estimate but uncontaminated by early history;
                reflects the local scaling exponent at that scale.

Reference lines:
  theta = 0.25        (divisor conjecture)
  theta = 131/416     (Huxley proven upper bound, ~0.3149)
"""

import sys, re, math, json, os

pat = re.compile(r'^\s+(\d+)\s+([-+]?\d+\.\d+)')
points, skipped = [], 0

print("Reading from stdin...", file=sys.stderr)
for line in sys.stdin:
    m = pat.match(line)
    if not m: continue
    n, err_norm = int(m.group(1)), float(m.group(2))
    if err_norm == 0.0 or abs(err_norm) >= 10.0:
        skipped += 1; continue
    points.append((math.log(n), math.log(abs(err_norm)) + 0.75*math.log(n)))

print(f"  parsed {len(points)} points, skipped {skipped}", file=sys.stderr)
if len(points) < 10:
    print("Not enough data.", file=sys.stderr); sys.exit(1)

points.sort()
log10 = math.log(10)
max_k = int(points[-1][0] / log10) + 1

def ols_theta(pts):
    n = len(pts)
    if n < 5: return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    sx, sy  = sum(xs), sum(ys)
    sxx     = sum(x*x for x in xs)
    sxy     = sum(x*y for x,y in zip(xs,ys))
    d = n*sxx - sx*sx
    return (n*sxy - sx*sy)/d - 0.5 if abs(d) > 1e-12 else None

cum_results, win_results = [], []

print(f"\n{'k':>4}  {'cum_pts':>8}  {'cum_theta':>10}  {'win_pts':>8}  {'win_theta':>10}", file=sys.stderr)
print("-"*50, file=sys.stderr)

for k in range(3, max_k+1):
    lo, hi = (k-1)*log10, k*log10
    cum_pts = [p for p in points if p[0] <= hi+1e-9]
    win_pts = [p for p in points if lo-1e-9 < p[0] <= hi+1e-9]
    ct, wt  = ols_theta(cum_pts), ols_theta(win_pts)
    cs = f"{ct:.6f}" if ct is not None else "    n/a"
    ws = f"{wt:.6f}" if wt is not None else "    n/a"
    print(f"  10^{k:<2d}  {len(cum_pts):>8d}  {cs:>10}  {len(win_pts):>8d}  {ws:>10}", file=sys.stderr)
    if ct is not None: cum_results.append((k, ct, len(cum_pts)))
    if wt is not None: win_results.append((k, wt, len(win_pts)))

data = {
    'cum_ks':     [r[0] for r in cum_results],
    'cum_thetas': [r[1] for r in cum_results],
    'cum_npts':   [r[2] for r in cum_results],
    'win_ks':     [r[0] for r in win_results],
    'win_thetas': [r[1] for r in win_results],
    'win_npts':   [r[2] for r in win_results],
    'max_k': max_k, 'total_pts': len(points),
}

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>theta_OLS vs N — A078567</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Oxanium:wght@300;400;600;700&family=Source+Code+Pro:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{background:#05080f;color:#c4d4e8;font-family:'Oxanium',sans-serif;
       min-height:100vh;display:flex;flex-direction:column;padding:32px 36px 48px;gap:24px}
  body::before{content:'';position:fixed;inset:0;
    background-image:linear-gradient(rgba(0,200,255,0.022) 1px,transparent 1px),
                     linear-gradient(90deg,rgba(0,200,255,0.022) 1px,transparent 1px);
    background-size:40px 40px;pointer-events:none;z-index:0}
  .header{position:relative;z-index:1;border-left:3px solid #00c8ff;padding-left:18px}
  .header h1{font-size:1.45rem;font-weight:700;letter-spacing:.04em;color:#fff;text-transform:uppercase}
  .header h1 em{color:#00c8ff;font-style:normal}
  .header .sub{font-size:.75rem;font-family:'Source Code Pro',monospace;color:#5a7a9a;margin-top:5px}
  .stat-row{position:relative;z-index:1;display:flex;gap:14px;flex-wrap:wrap}
  .stat{background:rgba(0,200,255,.04);border:1px solid rgba(0,200,255,.1);border-radius:4px;
        padding:8px 16px;display:flex;flex-direction:column;gap:2px}
  .stat .lbl{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:#5a7a9a}
  .stat .val{font-size:1rem;font-weight:600;color:#e0eeff;font-family:'Source Code Pro',monospace}
  .chart-card{position:relative;z-index:1;background:rgba(8,14,26,.85);
              border:1px solid rgba(0,200,255,.1);border-radius:8px;padding:24px 24px 16px}
  .chart-title{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:#5a7a9a;margin-bottom:14px}
  .chart-wrap{position:relative;height:460px}
  .legend{display:flex;gap:20px;flex-wrap:wrap;margin-top:14px;
          font-size:11px;font-family:'Source Code Pro',monospace;color:#5a7a9a}
  .legend-item{display:flex;align-items:center;gap:7px}
  .ld{width:24px;height:2px;display:inline-block}
  .obs-grid{position:relative;z-index:1;display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:750px){.obs-grid{grid-template-columns:1fr}}
  .obs-card{background:rgba(0,200,255,.03);border:1px solid rgba(0,200,255,.1);
            border-radius:6px;padding:14px 18px;
            font-family:'Source Code Pro',monospace;font-size:11.5px;line-height:1.85;color:#7a9ab8}
  .obs-card strong{color:#c4d4e8}
  .hl{color:#00c8ff}.hl2{color:#ffaa44}.hl3{color:#ff6688}
  code{background:rgba(0,200,255,.07);padding:1px 5px;border-radius:2px;color:#8ab8d8}
</style>
</head>
<body>
<div class="header">
  <h1><em>theta</em><sub>OLS</sub>(N) — A078567</h1>
  <div class="sub">log|err_osc| ~ α·log(n) + C &nbsp;·&nbsp; theta_OLS = α − 0.5 &nbsp;·&nbsp; cumulative vs single-decade sliding window</div>
</div>
<div class="stat-row" id="statRow"></div>
<div class="chart-card">
  <div class="chart-title">cumulative OLS (all n ≤ N) vs windowed OLS (single decade only)</div>
  <div class="chart-wrap"><canvas id="mainChart"></canvas></div>
  <div class="legend">
    <div class="legend-item"><span class="ld" style="background:#00c8ff"></span>cumulative OLS</div>
    <div class="legend-item"><span class="ld" style="background:#c084fc"></span>windowed OLS (one decade)</div>
    <div class="legend-item"><span class="ld" style="background:rgba(255,68,102,.7)"></span>θ = 0.25 (conjecture)</div>
    <div class="legend-item"><span class="ld" style="background:rgba(255,170,68,.7)"></span>θ = 131/416 (Huxley)</div>
  </div>
</div>
<div class="obs-grid">
  <div class="obs-card">
    <strong>Cumulative OLS</strong> (blue)<br>
    Regresses over every sample point up to N. Biased because the early
    decades (where theta was still climbing from ~0.22) anchor the fit.
    Converges slowly as those early points get diluted. Stable but always
    lags the true local exponent.
  </div>
  <div class="obs-card">
    <strong>Windowed OLS</strong> (purple)<br>
    Regresses over only the ~2300 points inside the current decade.
    Noisier — one decade of log-spaced data is a narrow lever arm —
    but uncontaminated by early history. Where it stabilises is the
    unbiased local estimate of theta at that scale.
  </div>
  <div class="obs-card" id="obsLeft"></div>
  <div class="obs-card" id="obsRight"></div>
</div>
<script>
const D=""" + json.dumps(data) + """;
const CONJ=0.25, HUXLEY=131/416;
const cum_final=D.cum_thetas[D.cum_thetas.length-1];
const win_last3=D.win_thetas.slice(-3);
const win_mean3=win_last3.reduce((a,b)=>a+b,0)/win_last3.length;
const win_final=D.win_thetas[D.win_thetas.length-1];

[['Total pts',D.total_pts.toLocaleString()],['Max N','10^'+D.max_k],
 ['Cumulative theta',cum_final.toFixed(6)],['Windowed theta',win_final.toFixed(6)],
 ['Win mean top-3',win_mean3.toFixed(6)]
].forEach(([l,v])=>{
  document.getElementById('statRow').innerHTML+=
    `<div class="stat"><span class="lbl">${l}</span><span class="val">${v}</span></div>`;
});

const allKs=[...new Set([...D.cum_ks,...D.win_ks])].sort((a,b)=>a-b);
const TT={backgroundColor:'#0a1220',borderColor:'rgba(0,200,255,.25)',borderWidth:1,
          titleColor:'#7ab8d8',bodyColor:'#c4d4e8',
          titleFont:{family:"'Source Code Pro'",size:11},bodyFont:{family:"'Source Code Pro'",size:11}};

new Chart(document.getElementById('mainChart').getContext('2d'),{
  type:'line',
  data:{datasets:[
    {label:'cumulative',data:D.cum_ks.map((k,i)=>({x:k,y:D.cum_thetas[i]})),
     borderColor:'#00c8ff',backgroundColor:'rgba(0,200,255,.06)',borderWidth:2,
     pointRadius:3.5,pointBackgroundColor:'#00c8ff',pointBorderColor:'#05080f',
     pointBorderWidth:1.5,fill:false,tension:.2,order:1},
    {label:'windowed',data:D.win_ks.map((k,i)=>({x:k,y:D.win_thetas[i]})),
     borderColor:'#c084fc',backgroundColor:'rgba(192,132,252,.06)',borderWidth:2,
     pointRadius:3.5,pointBackgroundColor:'#c084fc',pointBorderColor:'#05080f',
     pointBorderWidth:1.5,fill:false,tension:.2,order:2},
    {label:'conj',data:allKs.map(k=>({x:k,y:CONJ})),borderColor:'rgba(255,68,102,.65)',
     borderWidth:1.5,borderDash:[6,4],pointRadius:0,fill:false,order:3},
    {label:'huxley',data:allKs.map(k=>({x:k,y:HUXLEY})),borderColor:'rgba(255,170,68,.65)',
     borderWidth:1.5,borderDash:[3,4],pointRadius:0,fill:false,order:4},
  ]},
  options:{
    responsive:true,maintainAspectRatio:false,parsing:false,
    animation:{duration:500,easing:'easeOutQuart'},
    plugins:{legend:{display:false},tooltip:{...TT,callbacks:{
      title:c=>`N = 10^${c[0].parsed.x}`,
      label:c=>{
        const labs=['cumulative','windowed','θ=0.25','Huxley'];
        if(c.datasetIndex>1) return `${labs[c.datasetIndex]}: ${c.parsed.y.toFixed(4)}`;
        const npts=c.datasetIndex===0?D.cum_npts[c.dataIndex]:D.win_npts[c.dataIndex];
        return[`${labs[c.datasetIndex]}: theta = ${c.parsed.y.toFixed(6)}`,
               `alpha = ${(c.parsed.y+0.5).toFixed(6)}`,`pts = ${npts.toLocaleString()}`,
               `vs 0.25: ${(c.parsed.y-0.25>=0?'+':'')}${(c.parsed.y-0.25).toFixed(6)}`];
      }
    }}},
    scales:{
      x:{type:'linear',ticks:{color:'#5a7a9a',font:{family:"'Source Code Pro'",size:10},
           callback:v=>`10^${v}`,stepSize:1},grid:{color:'rgba(0,200,255,.06)'},
         title:{display:true,text:'log₁₀(N)',color:'#5a7a9a',font:{family:"'Source Code Pro'",size:10}}},
      y:{ticks:{color:'#5a7a9a',font:{family:"'Source Code Pro'",size:10},callback:v=>v.toFixed(3)},
         grid:{color:'rgba(0,200,255,.06)'},
         title:{display:true,text:'theta_OLS = alpha − 0.5',color:'#5a7a9a',font:{family:"'Source Code Pro'",size:10}}}
    }
  }
});

const win_top_str=D.win_thetas.slice(-5).map((t,i)=>
  `10^${D.win_ks[D.win_ks.length-5+i]}: ${t.toFixed(5)}`).join('<br>');
document.getElementById('obsLeft').innerHTML=`
  <strong>Windowed — last 5 decades</strong><br>${win_top_str}<br><br>
  Mean of top 3: <span class="hl">${win_mean3.toFixed(6)}</span>
  &nbsp;(vs conjecture: ${(win_mean3-0.25>=0?'+':'')}${(win_mean3-0.25).toFixed(6)})`;

const gap=cum_final-win_mean3;
document.getElementById('obsRight').innerHTML=`
  <strong>Bias estimate</strong><br>
  cumulative (biased):&nbsp;&nbsp; <span class="hl">${cum_final.toFixed(6)}</span><br>
  windowed mean top-3: <span class="hl">${win_mean3.toFixed(6)}</span><br>
  difference:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <span class="${gap>0?'hl3':'hl'}">${(gap>=0?'+':'')}${gap.toFixed(6)}</span><br><br>
  The gap between the two estimators is the empirical measure of how much
  the early-decade anchor is still pulling the cumulative fit upward.`;
</script>
</body>
</html>"""

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'theta_ols.html')
with open(out_path, 'w') as f:
    f.write(html)

print(f"\nWrote {out_path}", file=sys.stderr)
print(f"Open it in a browser.", file=sys.stderr)

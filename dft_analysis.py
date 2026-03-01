#!/usr/bin/env python3
"""
DFT analysis of err_a signal from results.h5
Cuts off data at the precision cliff (~4.855e18)
Compares spectral peaks to imaginary parts of Riemann zeta zeros.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress

PRECISION_CLIFF = 2.80e18
HDF5_PATH = "results.h5"

# ── Load data ──────────────────────────────────────────────────────────────────
with h5py.File(HDF5_PATH, "r") as f:
    n_str   = f["n_str"][:]
    err_a   = f["err_a"][:]
    logn    = f["logn"][:]

# n_str is bytes; convert to float
n_vals = np.array([float(s.decode() if isinstance(s, bytes) else s) for s in n_str])

# ── OLS regression ─────────────────────────────────────────────────────────────
# Fit log|err_a| ~ alpha*log(n) + C on clean data only:
#   n > 1e6  : skip transient / small-n regime
#   n < cliff: skip precision-cliff contamination (frozen/zero residuals)
#   err_a != 0 and finite: skip any exact zeros or NaNs from cliff artefacts
# theta = alpha - 0.5  (since err_osc = O(n^{theta + 0.5}))
ols_mask = (
    (n_vals > 1e6) &
    (n_vals < PRECISION_CLIFF) &
    (err_a != 0.0) &
    np.isfinite(err_a)
)
_logn_ols = np.log(n_vals[ols_mask])
_loge_ols = np.log(np.abs(err_a[ols_mask]))
print(f"OLS on {ols_mask.sum()} points  (1e6 < n < {PRECISION_CLIFF:.2e})")
if ols_mask.sum() >= 3:
    _slope, _intercept, _r, _, _se = linregress(_logn_ols, _loge_ols)
    _theta = _slope - 0.5
    print(f"  log|err_a| ~ {_slope:.6f}*log(n) + {_intercept:.4f}")
    print(f"  alpha={_slope:.6f}  =>  theta_OLS = alpha-0.5 = {_theta:.6f}")
    print(f"  R²={_r**2:.6f}  std_err(alpha)={_se:.6f}")
    print(f"  (conjecture: theta=0.25 | Huxley: theta<=0.315)")
else:
    _slope = _intercept = _theta = _r = _se = None

# Cut at precision cliff
mask   = n_vals < PRECISION_CLIFF
n_vals = n_vals[mask]
err_a  = err_a[mask]
err_normalized = err_a / np.power(n_vals, 0.75)
err_a = err_normalized
logn   = logn[mask]

print(f"Using {len(n_vals)} points  (n < {PRECISION_CLIFF:.3e})")
print(f"n range: {n_vals[0]:.3e} – {n_vals[-1]:.3e}")

# ── The signal is sampled on a geometric (log-uniform) grid, not a uniform one.
# Interpolate onto a uniform grid in log(n) before DFT.
# ──────────────────────────────────────────────────────────────────────────────
N_uniform = len(n_vals)   # keep same number of points
logn_uniform = np.linspace(logn[0], logn[-1], N_uniform)
err_uniform  = np.interp(logn_uniform, logn, err_a)

# ── Hann window to reduce spectral leakage ─────────────────────────────────────
window     = np.hanning(N_uniform)
err_w      = err_uniform * window

# ── DFT ───────────────────────────────────────────────────────────────────────
fft_vals   = np.fft.rfft(err_w)
freqs      = np.fft.rfftfreq(N_uniform)   # cycles per (log-uniform) sample
magnitudes = np.abs(fft_vals)

# ── Find peaks ────────────────────────────────────────────────────────────────
# Minimum prominence = 1% of global max
peak_idx, props = find_peaks(magnitudes, prominence=magnitudes.max() * 0.01)
peak_idx = peak_idx[np.argsort(magnitudes[peak_idx])[::-1]]  # sort by amplitude

print(f"\nTop 20 spectral peaks (cycles per log-sample):")
print(f"{'rank':>4}  {'freq':>10}  {'period':>10}  {'|DFT|':>14}")
print("-" * 44)
for rank, idx in enumerate(peak_idx[:20], 1):
    f  = freqs[idx]
    T  = 1.0 / f if f > 0 else np.inf
    print(f"{rank:>4}  {f:>10.6f}  {T:>10.2f}  {magnitudes[idx]:>14.1f}")

# ── Check if peak periods are close to small integers ─────────────────────────
print(f"\nPeak period vs nearest integer:")
print(f"{'freq':>10}  {'period':>10}  {'nearest_int':>12}  {'residual':>10}")
print("-" * 48)
for idx in peak_idx[:20]:
    f = freqs[idx]
    if f == 0:
        continue
    T     = 1.0 / f
    near  = round(T)
    resid = T - near
    print(f"{f:>10.6f}  {T:>10.3f}  {near:>12}  {resid:>10.4f}")

# ── Riemann zeta zeros (imaginary parts γ_k) ──────────────────────────────────
# The divisor sum error term has contributions ~ n^(1/2+iγ), so in log(n) space
# the oscillation frequency is γ / (2π) cycles per unit of log(n).
# Our DFT frequency is in cycles per log-sample, where each sample step is
# delta_logn = (logn[-1] - logn[0]) / N_uniform.
# So the expected DFT bin for zeta zero γ is:
#   f_expected = (γ / (2π)) * delta_logn
ZETA_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446247, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 92.491899,
    94.651344, 95.870634, 98.831194, 101.317851, 103.725538,
]

delta_logn = (logn_uniform[-1] - logn_uniform[0]) / N_uniform
print("delta_logn", delta_logn)
zeta_freqs = np.array([(g / (2 * np.pi)) * delta_logn for g in ZETA_ZEROS])

print(f"\ndelta_logn per sample = {delta_logn:.6f}")
print(f"\nRiemann zeta zero comparison (top 40 DFT peaks vs nearest zero):")
print(f"{'rank':>4}  {'f_peak':>10}  {'gamma_peak':>12}  {'nearest_zero':>13}  {'delta_gamma':>12}  {'|DFT|':>12}")
print("-" * 72)

# For each peak, compute implied gamma and find nearest zeta zero
for rank, idx in enumerate(peak_idx[:40], 1):
    f = freqs[idx]
    if f == 0:
        continue
    gamma_implied = f * (2 * np.pi) / delta_logn
    # find nearest known zero
    diffs = np.abs(np.array(ZETA_ZEROS) - gamma_implied)
    nearest_idx = np.argmin(diffs)
    nearest_gamma = ZETA_ZEROS[nearest_idx]
    delta = gamma_implied - nearest_gamma
    print(f"{rank:>4}  {f:>10.6f}  {gamma_implied:>12.4f}  {nearest_gamma:>13.6f}  {delta:>+12.4f}  {magnitudes[idx]:>12.1f}")

# ── Also: for each known zeta zero, find the closest DFT peak ─────────────────
print(f"\nZeta zeros → closest DFT peak:")
print(f"{'zero_γ':>12}  {'f_expected':>12}  {'f_nearest_peak':>15}  {'delta_γ':>10}  {'|DFT|':>12}")
print("-" * 66)
for g, fz in zip(ZETA_ZEROS[:20], zeta_freqs[:20]):
    # find nearest peak
    if len(peak_idx) == 0:
        continue
    peak_freqs_arr = freqs[peak_idx]
    nearest = peak_freqs_arr[np.argmin(np.abs(peak_freqs_arr - fz))]
    nearest_pidx = peak_idx[np.argmin(np.abs(peak_freqs_arr - fz))]
    gamma_nearest = nearest * (2 * np.pi) / delta_logn
    delta_g = gamma_nearest - g
    print(f"{g:>12.6f}  {fz:>12.6f}  {nearest:>15.6f}  {delta_g:>+10.4f}  {magnitudes[nearest_pidx]:>12.1f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 16))
fig.patch.set_facecolor("#080b10")
for ax in axes:
    ax.set_facecolor("#0d1219")
    ax.tick_params(colors="#5a7a96")
    ax.xaxis.label.set_color("#5a7a96")
    ax.yaxis.label.set_color("#5a7a96")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d3d")

# Panel 1: time series (err_a on log-n axis)
axes[0].plot(logn, err_a, color="#00d4ff", lw=0.6)
axes[0].set_xlabel("log(n)")
axes[0].set_ylabel("err_a / n^0.75")
axes[0].set_title("err_osc / n^0.75  (n < 2.8e18)", color="#c8d8e8")

# Panel 2: OLS scatter + fit
if _slope is not None:
    axes[1].scatter(_logn_ols, _loge_ols, s=2, color="#00d4ff", alpha=0.25, rasterized=True)
    _fit = _slope * _logn_ols + _intercept
    axes[1].plot(_logn_ols, _fit, color="#ff6b35", lw=1.5,
                 label=f"OLS: α={_slope:.4f}, θ={_theta:.4f}, R²={_r**2:.4f}")
    # Reference slopes anchored to data centroid
    _mx, _my = _logn_ols.mean(), _loge_ols.mean()
    for _tref, _lbl, _col in [(0.25, "θ=0.25 (conjecture)", "#d966ff"),
                               (0.315, "θ=0.315 (Huxley)",   "#ffcc00")]:
        _s = _tref + 0.5   # alpha = theta + 0.5
        axes[1].plot(_logn_ols, _s*(_logn_ols - _mx) + _my,
                     '--', color=_col, lw=1.0, alpha=0.8, label=_lbl)
    axes[1].set_xlabel("log(n)")
    axes[1].set_ylabel("log|err_a|")
    axes[1].set_title(f"OLS  [1e6 < n < 2.8e18]  →  θ = {_theta:.6f}", color="#c8d8e8")
    axes[1].legend(facecolor="#0d1219", edgecolor="#1e2d3d", labelcolor="#c8d8e8", fontsize=8)

# Panel 3: full DFT spectrum
axes[2].set_xlim(0, 0.02)
axes[3].set_xlim(0, 0.02)
axes[2].plot(freqs, magnitudes, color="#00d4ff", lw=0.7)
axes[2].plot(freqs[peak_idx[:20]], magnitudes[peak_idx[:20]],
             "x", color="#d966ff", ms=6, lw=1.5, label="top 20 peaks")
for fz, g in zip(zeta_freqs[:20], ZETA_ZEROS[:20]):
    axes[2].axvline(fz, color="#ff6b35", lw=0.6, alpha=0.5)
axes[2].axvline(zeta_freqs[0], color="#ff6b35", lw=0.6, alpha=0.5, label="zeta zeros γ_k")
axes[2].set_xlabel("frequency (cycles per log-sample)")
axes[2].set_ylabel("|DFT|")
axes[2].set_title("DFT magnitude spectrum — err_a (Hann windowed, log-uniform grid)", color="#c8d8e8")
axes[2].legend(facecolor="#0d1219", edgecolor="#1e2d3d", labelcolor="#c8d8e8")

# Panel 4: log-scale DFT to see high-freq structure
axes[3].semilogy(freqs[1:], magnitudes[1:], color="#00d4ff", lw=0.7)
axes[3].semilogy(freqs[peak_idx[:20]], magnitudes[peak_idx[:20]],
                 "x", color="#d966ff", ms=6, lw=1.5)
for fz in zeta_freqs[:20]:
    axes[3].axvline(fz, color="#ff6b35", lw=0.6, alpha=0.5)
axes[3].set_xlabel("frequency (cycles per log-sample)")
axes[3].set_ylabel("log |DFT|")
axes[3].set_title("DFT — log scale (orange lines = zeta zeros)", color="#c8d8e8")

plt.tight_layout(pad=2.0)
plt.savefig("dft_err_a.png", dpi=150, facecolor="#080b10")
print("\nSaved dft_err_a.png")
plt.show()

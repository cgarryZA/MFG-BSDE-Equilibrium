#!/usr/bin/env python -u
"""Adverse selection at extreme alpha (0.9, 0.95, 0.99) — does linearity break?

Quick test: is the linear spread(α) relationship sustained all the way to
informed-dominated markets (α close to 1), or does the system break down?

CPU, ~2 min.
"""

import sys, os, json
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.adverse_selection_deep import adverse_fp

print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

# Full range including extreme
alpha_grid = np.concatenate([
    np.linspace(0.0, 0.8, 9),
    np.array([0.85, 0.9, 0.95, 0.98, 0.99]),
])
theta = 0.3

print(f"\nAlpha sweep to extreme values (theta={theta}):")
print(f"{'alpha':>6s}  {'spread':>8s}  {'V(0)':>8s}  {'dspread/dalpha':>16s}")
results = []
prev_spread = None
for a in alpha_grid:
    r = adverse_fp(alpha_a=a, alpha_b=a, theta_a=theta, theta_b=theta)
    s = r['spread_q0']
    v = r['V_q0']
    deriv = "" if prev_spread is None else f"{(s - prev_spread)/(alpha_grid[list(alpha_grid).index(a)] - alpha_grid[list(alpha_grid).index(a)-1]):.4f}"
    print(f"  {a:6.3f}  {s:8.4f}  {v:8.4f}  {deriv:>16s}", flush=True)
    results.append({"alpha": float(a), "spread": float(s), "V_q0": float(v), "theta": theta})
    prev_spread = s

# Fit linear to low-alpha region, extrapolate to extreme
alpha_arr = np.array([r['alpha'] for r in results])
spread_arr = np.array([r['spread'] for r in results])
V_arr = np.array([r['V_q0'] for r in results])

mask_low = alpha_arr <= 0.5
slope_low, int_low = np.polyfit(alpha_arr[mask_low], spread_arr[mask_low], 1)
predicted = slope_low * alpha_arr + int_low
residuals = spread_arr - predicted

print(f"\nLinear fit on low-alpha (<=0.5): spread = {slope_low:.4f}*alpha + {int_low:.4f}")
print(f"\n{'alpha':>6s}  {'observed':>10s}  {'linear pred':>12s}  {'deviation':>10s}")
for a, s, p in zip(alpha_arr, spread_arr, predicted):
    dev = s - p
    marker = " <-- deviation > 0.02" if abs(dev) > 0.02 else ""
    print(f"  {a:6.3f}  {s:10.4f}  {p:12.4f}  {dev:+10.4f}{marker}")

# Check for value collapse (V becomes negative?)
any_negative = any(v < 0 for v in V_arr)
print(f"\nDealer value ever negative: {any_negative}")
print(f"V(alpha=max) = {V_arr[-1]:.4f} (baseline V(0) = {V_arr[0]:.4f})")
print(f"Value lost at alpha={alpha_arr[-1]:.2f}: {(V_arr[0]-V_arr[-1])/V_arr[0]*100:.1f}%")

with open("results_final/as_extreme_alpha.json", "w") as f:
    json.dump({
        "results": results,
        "linear_slope_low_alpha": float(slope_low),
        "linear_intercept": float(int_low),
        "max_deviation": float(max(abs(r) for r in residuals)),
        "value_lost_pct_at_max": float((V_arr[0]-V_arr[-1])/V_arr[0]*100),
    }, f, indent=2, default=float)
print(f"\nSaved to results_final/as_extreme_alpha.json", flush=True)
print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

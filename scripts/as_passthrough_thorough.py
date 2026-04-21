#!/usr/bin/env python -u
"""Thorough pass-through robustness: dense N, lambda, Q grid.

Previous test found mild variation (std 0.021) across (N, lambda) — not
quite structural. Test if:
  - Is variation monotonic in N and lambda (systematic) or random?
  - Does Q matter?
  - What's the right claim: "~85% with some variation" or "exactly structural"?

For each (N, lambda, Q, theta), compute pass-through ratio.
CPU, ~1h.
"""

import sys, os, json, time
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.as_robustness_N_lambda import adverse_fp_param, measure_pass_through

print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def measure_ratio(N, lam, Q, theta, alpha_grid=None):
    """Pass-through ratio with custom Q."""
    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 0.5, 6)
    spreads = []
    for a in alpha_grid:
        s, _ = adverse_fp_param(N=N, lam=lam, alpha_a=a, alpha_b=a,
                                 theta_a=theta, theta_b=theta, Q=Q)
        spreads.append(s)
    spreads = np.array(spreads)
    slope = np.polyfit(alpha_grid, spreads, 1)[0]
    return float(slope / (2 * theta)), float(slope), float(spreads[0])


# Dense grid
N_vals = [2, 3, 5, 7, 10]
lam_vals = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
Q_vals = [5, 7, 10]
theta = 0.3

print(f"\n{'='*70}")
print(f"THOROUGH PASS-THROUGH TEST: {len(N_vals)}*{len(lam_vals)}*{len(Q_vals)} = "
      f"{len(N_vals)*len(lam_vals)*len(Q_vals)} configs")
print(f"{'='*70}", flush=True)

print(f"\n{'N':>4s}  {'lambda':>6s}  {'Q':>3s}  {'nash':>8s}  {'slope':>8s}  {'ratio':>8s}")
results = []
for Q in Q_vals:
    for N in N_vals:
        for lam in lam_vals:
            try:
                ratio, slope, nash = measure_ratio(N, lam, Q, theta)
                print(f"  {N:4d}  {lam:6.1f}  {Q:3d}  {nash:8.4f}  {slope:8.4f}  {ratio:7.4f}",
                      flush=True)
                results.append({
                    "N": N, "lambda": lam, "Q": Q,
                    "nash": nash, "slope": slope, "ratio": ratio,
                })
            except Exception as e:
                print(f"  {N}, {lam}, {Q}: FAILED ({e})", flush=True)

# Save
with open("results_final/as_passthrough_thorough.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

# Analysis: is the variation systematic?
ratios = np.array([r['ratio'] for r in results])
Ns = np.array([r['N'] for r in results])
lams = np.array([r['lambda'] for r in results])
Qs = np.array([r['Q'] for r in results])

print(f"\n{'='*70}")
print(f"STATISTICAL ANALYSIS")
print(f"{'='*70}")
print(f"Total configs:  {len(results)}")
print(f"Mean ratio:     {np.mean(ratios):.4f}")
print(f"Std ratio:      {np.std(ratios, ddof=1):.4f}")
print(f"Range:          [{np.min(ratios):.4f}, {np.max(ratios):.4f}]")
print(f"IQR:            [{np.percentile(ratios, 25):.4f}, {np.percentile(ratios, 75):.4f}]")

# Check correlations
print(f"\nCorrelation analysis:")
print(f"  ratio vs N:        {np.corrcoef(Ns, ratios)[0,1]:+.4f}")
print(f"  ratio vs lambda:   {np.corrcoef(lams, ratios)[0,1]:+.4f}")
print(f"  ratio vs Q:        {np.corrcoef(Qs, ratios)[0,1]:+.4f}")

# By N
print(f"\nBy N (averaged across lambda, Q):")
for N in N_vals:
    sub = ratios[Ns == N]
    print(f"  N={N}: mean={np.mean(sub):.4f}, std={np.std(sub, ddof=1):.4f}, n={len(sub)}")

# By lambda
print(f"\nBy lambda (averaged across N, Q):")
for lam in lam_vals:
    sub = ratios[lams == lam]
    print(f"  lambda={lam}: mean={np.mean(sub):.4f}, std={np.std(sub, ddof=1):.4f}, n={len(sub)}")

# By Q
print(f"\nBy Q:")
for Q in Q_vals:
    sub = ratios[Qs == Q]
    print(f"  Q={Q}: mean={np.mean(sub):.4f}, std={np.std(sub, ddof=1):.4f}, n={len(sub)}")

# Save summary
summary = {
    "n_configs": len(results),
    "mean_ratio": float(np.mean(ratios)),
    "std_ratio": float(np.std(ratios, ddof=1)),
    "range": [float(np.min(ratios)), float(np.max(ratios))],
    "corr_N": float(np.corrcoef(Ns, ratios)[0, 1]),
    "corr_lambda": float(np.corrcoef(lams, ratios)[0, 1]),
    "corr_Q": float(np.corrcoef(Qs, ratios)[0, 1]),
}
summary["all_results"] = results

with open("results_final/as_passthrough_thorough.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)

# Verdict
print(f"\n{'='*70}")
print(f"VERDICT")
print(f"{'='*70}")
mean_r = float(np.mean(ratios)); std_r = float(np.std(ratios, ddof=1))
if std_r < 0.015:
    print(f"  Pass-through is a tight structural constant: {mean_r:.3f} +- {std_r:.3f}")
elif std_r < 0.03:
    print(f"  Pass-through ~ {mean_r:.3f}+-{std_r:.3f} with systematic variation")
    # Describe dominant trend
    trends = []
    if abs(np.corrcoef(Ns, ratios)[0,1]) > 0.3: trends.append(f"N (corr {np.corrcoef(Ns, ratios)[0,1]:+.2f})")
    if abs(np.corrcoef(lams, ratios)[0,1]) > 0.3: trends.append(f"lambda (corr {np.corrcoef(lams, ratios)[0,1]:+.2f})")
    if abs(np.corrcoef(Qs, ratios)[0,1]) > 0.3: trends.append(f"Q (corr {np.corrcoef(Qs, ratios)[0,1]:+.2f})")
    if trends:
        print(f"  Main drivers of variation: {', '.join(trends)}")
else:
    print(f"  Pass-through varies substantially ({mean_r:.3f}+-{std_r:.3f}); not structural")

print(f"\nSaved to results_final/as_passthrough_thorough.json")
print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

#!/usr/bin/env python -u
"""Robustness check: is the 85% pass-through coefficient structural
or a theta=0.3 artefact?

Runs the alpha sweep at multiple theta values. Fits slope each time.
If pass-through ratio is stable across theta → structural invariant.
"""

import sys, os, json
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.adverse_selection_deep import adverse_fp

print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

print(f"\n{'='*60}")
print("Robustness: is 85% pass-through structural or theta-specific?")
print(f"{'='*60}", flush=True)

alpha_grid = np.linspace(0.0, 0.6, 7)
theta_grid = [0.1, 0.2, 0.3, 0.5, 0.7]

results = []
print(f"\n{'theta':>6s}  {'slope':>8s}  {'naive':>8s}  {'ratio':>8s}  {'nash sprd':>10s}")
for theta in theta_grid:
    spreads = []
    for a in alpha_grid:
        r = adverse_fp(alpha_a=a, alpha_b=a, theta_a=theta, theta_b=theta)
        spreads.append(r['spread_q0'])
    spreads = np.array(spreads)

    # Linear fit
    slope, intercept = np.polyfit(alpha_grid, spreads, 1)
    naive_slope = 2 * theta
    ratio = slope / naive_slope
    print(f"{theta:6.2f}  {slope:8.4f}  {naive_slope:8.4f}  {ratio:7.4f}  {spreads[0]:10.4f}")

    results.append({
        "theta": float(theta), "alpha_grid": alpha_grid.tolist(),
        "spreads": spreads.tolist(), "slope": float(slope),
        "naive_slope": float(naive_slope), "ratio": float(ratio),
    })

# Check if ratio is stable
ratios = [r['ratio'] for r in results]
mean_ratio = np.mean(ratios)
std_ratio = np.std(ratios, ddof=1)
range_ratio = max(ratios) - min(ratios)

print(f"\n{'='*60}")
print("STRUCTURAL INVARIANT CHECK")
print(f"{'='*60}")
print(f"Pass-through ratios across theta: {[f'{r:.4f}' for r in ratios]}")
print(f"Mean: {mean_ratio:.4f}")
print(f"Std:  {std_ratio:.4f}")
print(f"Range: {range_ratio:.4f}")
print()
if std_ratio < 0.02:
    print(f"CONCLUSION: Ratio is STABLE (std={std_ratio:.4f} < 0.02).")
    print(f"Pass-through beta ~ {mean_ratio:.3f} is a STRUCTURAL invariant of the model.")
    print(f"Economic interpretation: the equilibrium execution-probability feedback")
    print(f"  prevents dealers from fully internalising adverse costs, regardless")
    print(f"  of the magnitude of the information asymmetry.")
else:
    print(f"CONCLUSION: Ratio VARIES with theta (std={std_ratio:.4f}).")
    print(f"Pass-through is NOT structural — depends on adverse cost level.")

save = {
    "theta_grid": theta_grid,
    "alpha_grid": alpha_grid.tolist(),
    "results_per_theta": results,
    "mean_ratio": float(mean_ratio),
    "std_ratio": float(std_ratio),
    "range_ratio": float(range_ratio),
    "is_structural": bool(std_ratio < 0.02),
}
with open("results_final/adverse_selection_robustness.json", "w") as f:
    json.dump(save, f, indent=2, default=float)
print(f"\nSaved to results_final/adverse_selection_robustness.json", flush=True)
print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

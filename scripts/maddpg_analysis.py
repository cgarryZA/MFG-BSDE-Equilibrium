#!/usr/bin/env python -u
"""
Post-hoc analysis of MADDPG results.

1. Learning curves — which seeds converge fastest?
2. Final spread distribution — histogram, outliers
3. Above-Nash statistics — tightened CIs with non-parametric methods
"""

import sys, os, json
import numpy as np
sys.stdout.reconfigure(line_buffering=True)

from scipy import stats

# Load all 20 MADDPG N=2 seeds
data = []
for i in range(20):
    try:
        d = json.load(open(f'results_cluster/maddpg_seed{i}.json'))
        data.append(d)
    except FileNotFoundError:
        continue

print(f"Loaded {len(data)} MADDPG N=2 seeds")

spreads = [d['final_spread'] for d in data]
nash = data[0]['nash_spread']
print(f"\nNash spread: {nash:.4f}")

# Summary
print(f"\n=== Final spread distribution ===")
print(f"  Mean:   {np.mean(spreads):.4f}")
print(f"  Median: {np.median(spreads):.4f}")
print(f"  Std:    {np.std(spreads, ddof=1):.4f}")
print(f"  Min:    {np.min(spreads):.4f}")
print(f"  Max:    {np.max(spreads):.4f}")
print(f"  25%:    {np.percentile(spreads, 25):.4f}")
print(f"  75%:    {np.percentile(spreads, 75):.4f}")

# Above-Nash
above = [s > nash for s in spreads]
print(f"\n=== Above Nash ===")
print(f"  Count: {sum(above)}/{len(above)} ({100*sum(above)/len(above):.0f}%)")
print(f"  Sign test p-value: {stats.binom.sf(sum(above)-1, len(above), 0.5):.6f}")

# t-test
t, p = stats.ttest_1samp(spreads, nash)
p_one = p / 2 if t > 0 else 1 - p/2
print(f"\n=== Tests (one-sided, H1: mean > nash) ===")
print(f"  t-test p = {p_one:.6f}")
w_stat, w_p = stats.wilcoxon(np.array(spreads) - nash, alternative='greater')
print(f"  Wilcoxon p = {w_p:.6f}")

# CIs
ci_t = stats.t.interval(0.95, len(spreads)-1, loc=np.mean(spreads), scale=stats.sem(spreads))
print(f"\n=== 95% CIs ===")
print(f"  t-distribution: [{ci_t[0]:.4f}, {ci_t[1]:.4f}]")
# Bootstrap CI
rng = np.random.default_rng(42)
n_boot = 10000
boot_means = np.array([np.mean(rng.choice(spreads, size=len(spreads), replace=True))
                       for _ in range(n_boot)])
ci_boot = (np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5))
print(f"  Bootstrap:      [{ci_boot[0]:.4f}, {ci_boot[1]:.4f}]")

# Learning curves (skip if history not in data)
print(f"\n=== Learning trajectories ===")
checkpoints = [100, 250, 499]
for cp in checkpoints:
    vals = []
    for d in data:
        if 'history' not in d:
            continue
        for h in d['history']:
            if h.get('episode') == cp:
                vals.append(h['avg_spread'])
                break
    if vals:
        n_above = sum(1 for v in vals if v > nash)
        print(f"  Ep {cp}: mean={np.mean(vals):.3f}, std={np.std(vals, ddof=1):.3f}, above Nash: {n_above}/{len(vals)}")
    else:
        print(f"  Ep {cp}: no history data available in local files")

# Save analysis
analysis = {
    "n_seeds": len(data),
    "nash_spread": float(nash),
    "mean_spread": float(np.mean(spreads)),
    "median_spread": float(np.median(spreads)),
    "std_spread": float(np.std(spreads, ddof=1)),
    "above_nash": int(sum(above)),
    "t_p_value": float(p_one),
    "wilcoxon_p_value": float(w_p),
    "sign_test_p": float(stats.binom.sf(sum(above)-1, len(above), 0.5)),
    "ci_t_95": [float(ci_t[0]), float(ci_t[1])],
    "ci_bootstrap_95": [float(ci_boot[0]), float(ci_boot[1])],
    "final_spreads": [float(s) for s in spreads],
}

with open("results_final/maddpg_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)
print(f"\nSaved to results_final/maddpg_analysis.json")

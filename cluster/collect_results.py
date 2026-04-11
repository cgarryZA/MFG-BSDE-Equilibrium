#!/usr/bin/env python
"""
Collect all MADDPG results from cluster array job.
Run after all jobs complete: python cluster/collect_results.py
"""

import json
import glob
import numpy as np
from scipy import stats
import os

def main():
    files = sorted(glob.glob("results_cluster/maddpg_seed*.json"))
    if not files:
        print("No results found in results_cluster/")
        return

    final_spreads = []
    avg_spreads = []
    nash_spread = None

    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        final_spreads.append(r["final_spread"])
        avg_spreads.append(r["avg_spread"])
        if nash_spread is None:
            nash_spread = r["nash_spread"]
        print(f"  {os.path.basename(f)}: final={r['final_spread']:.4f}, "
              f"avg={r['avg_spread']:.4f}, above_nash={r['above_nash']}")

    n = len(final_spreads)
    mean_f = np.mean(final_spreads)
    std_f = np.std(final_spreads)
    mean_a = np.mean(avg_spreads)
    std_a = np.std(avg_spreads)

    ci = stats.t.interval(0.95, df=n-1, loc=mean_a, scale=stats.sem(avg_spreads))
    n_above = sum(1 for s in final_spreads if s > nash_spread)

    print(f"\n{'='*50}")
    print(f"  Rounds:              {n}")
    print(f"  Nash spread:         {nash_spread:.4f}")
    print(f"  Final spread:        {mean_f:.4f} +/- {std_f:.4f}")
    print(f"  Avg spread:          {mean_a:.4f} +/- {std_a:.4f}")
    print(f"  95% CI (avg):        [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  Above Nash (final):  {n_above}/{n}")
    print(f"  CI lower > Nash?     {'YES — SIGNIFICANT' if ci[0] > nash_spread else 'NO'}")

    with open("results_cluster/summary.json", "w") as f:
        json.dump({
            "n_rounds": n, "nash_spread": nash_spread,
            "final_spreads": final_spreads, "avg_spreads": avg_spreads,
            "mean_final": mean_f, "std_final": std_f,
            "mean_avg": mean_a, "std_avg": std_a,
            "ci_95": list(ci), "n_above_nash": n_above,
            "significant": bool(ci[0] > nash_spread),
        }, f, indent=2)
    print(f"  Saved results_cluster/summary.json")


if __name__ == "__main__":
    main()

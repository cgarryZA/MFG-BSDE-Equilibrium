#!/usr/bin/env python -u
"""Seed variance for learning-by-doing.

Re-train LBD solver at each kappa with 5 different seeds. Measure mean and
std of the 3 core findings:
  1. Spread at (q=0, a=0) — the loss-leading value
  2. Franchise premium V(a=0.7) - V(a=0)
  3. Inventory std in simulation

Gives error bars for the paper-quality section.

CPU, ~2h (5 seeds x 3 kappa x ~8min each).
"""

import sys, os, json, time, gc
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.learning_by_doing_deep import train_adaptive, extract_policy, simulate_inventory

print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

device = torch.device("cpu")
N_SEEDS = 5
KAPPAS = [0.0, 0.25, 0.5, 0.75]

all_results = {}

for kappa in KAPPAS:
    print(f"\n{'='*60}")
    print(f"kappa = {kappa} across {N_SEEDS} seeds")
    print(f"{'='*60}", flush=True)

    seed_results = []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        gc.collect()

        t0 = time.time()
        net = train_adaptive(kappa=kappa, n_iter=1500)
        policies = extract_policy(net)
        dyn = simulate_inventory(net, kappa=kappa, n_paths=200, T=150, seed=seed)
        elapsed = time.time() - t0

        p = policies["a=0.0"]
        p_hi = policies["a=0.7"]
        mid = 5
        spread_at_a0 = p["spread"][mid]
        V_a0 = p["V"][mid]
        V_a_hi = p_hi["V"][mid]
        franchise = V_a_hi - V_a0

        seed_results.append({
            "seed": seed,
            "spread_at_a0": float(spread_at_a0),
            "V_a0": float(V_a0),
            "V_a_hi": float(V_a_hi),
            "franchise_premium": float(franchise),
            "inv_std": float(dyn["final_inv_std"]),
            "elapsed": float(elapsed),
        })
        print(f"  seed {seed}: spread(a=0)={spread_at_a0:.4f}, V(a=0)={V_a0:.4f}, "
              f"franchise={franchise:+.4f}, inv_std={dyn['final_inv_std']:.3f} [{elapsed:.0f}s]",
              flush=True)

    # Stats across seeds
    spreads = [r["spread_at_a0"] for r in seed_results]
    Vs = [r["V_a0"] for r in seed_results]
    frs = [r["franchise_premium"] for r in seed_results]
    invs = [r["inv_std"] for r in seed_results]

    all_results[f"kappa={kappa}"] = {
        "kappa": kappa,
        "seeds": seed_results,
        "spread_at_a0": {"mean": float(np.mean(spreads)), "std": float(np.std(spreads, ddof=1))},
        "V_a0": {"mean": float(np.mean(Vs)), "std": float(np.std(Vs, ddof=1))},
        "franchise_premium": {"mean": float(np.mean(frs)), "std": float(np.std(frs, ddof=1))},
        "inv_std": {"mean": float(np.mean(invs)), "std": float(np.std(invs, ddof=1))},
    }

    with open("results_final/lbd_seed_variance.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"  -> Saved kappa={kappa} incrementally", flush=True)

# Summary
print(f"\n{'='*60}")
print("SEED-VARIANCE SUMMARY")
print(f"{'='*60}")
print(f"{'kappa':>6s}  {'spread(a=0)':>14s}  {'franchise':>14s}  {'inv_std':>12s}")
for key, r in all_results.items():
    k = r["kappa"]
    s = r["spread_at_a0"]
    f = r["franchise_premium"]
    i = r["inv_std"]
    print(f"{k:6.2f}  {s['mean']:8.4f}+-{s['std']:.4f}  "
          f"{f['mean']:+8.4f}+-{f['std']:.4f}  {i['mean']:6.3f}+-{i['std']:.3f}")

print(f"\nFinished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

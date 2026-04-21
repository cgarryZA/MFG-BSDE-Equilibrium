#!/usr/bin/env python -u
"""Test BSDEJ with correct equilibrium avg_comp (0.637 instead of 0.75).

BSDEJ uses avg_comp = 0.75 (monopolist) as a constant. The true equilibrium
avg is 0.637. Check if fixing this reduces the 2.6% spread error.

Run on GPU: ~40 min (with warm-start + shared weights).
"""

import sys, os, json, time
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_bsdej_shared import CXBSDEJShared
from utils import EarlyStopping

device = torch.device("cpu")  # GPU unstable on this setup
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def run_with_avg(avg_comp, n_iter=10000):
    print(f"\n{'='*60}")
    print(f"BSDEJ with avg_comp = {avg_comp}")
    print(f"{'='*60}", flush=True)

    solver = CXBSDEJShared(
        N=2, Q=5, Delta=1, T=10.0, M=50,
        lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
        device=device, lr=5e-4, n_iter=n_iter, batch_size=128,  # smaller for CPU
        hidden=128, n_layers=3,
    )
    solver.avg_comp = avg_comp  # override default 0.75

    solver.warmstart_from_bellman(n_pretrain=2000)
    print(flush=True)

    result = solver.train(early_stopping=True, es_patience=1000, es_warmup=2000)

    spread = result["U_profile"][5]["spread"]
    nash = 1.5153
    error = abs(spread - nash) / nash * 100
    print(f"\n  avg_comp={avg_comp}: spread={spread:.4f}, error={error:.2f}%, loss={result['best_loss']:.2e}",
          flush=True)

    return {
        "avg_comp": avg_comp,
        "spread_q0": float(spread),
        "error_pct": float(error),
        "best_loss": float(result['best_loss']),
        "U_profile": result["U_profile"],
        "elapsed": result['elapsed'],
    }


if __name__ == "__main__":
    results = []

    # Test three values:
    # 0.75 = old default (monopolist-ish)
    # 0.637 = exact equilibrium average (from exact solver)
    # Also self-consistent (re-compute after training)
    for avg in [0.75, 0.637]:
        r = run_with_avg(avg, n_iter=8000)
        results.append(r)
        with open("results_final/bsdej_avg_comp_fix.json", "w") as f:
            json.dump(results, f, indent=2, default=float)

    print(f"\n{'='*60}")
    print("BSDEJ avg_comp SENSITIVITY SUMMARY")
    print(f"{'='*60}")
    print(f"{'avg_comp':>10s}  {'spread(0)':>10s}  {'error':>8s}  {'loss':>10s}")
    for r in results:
        print(f"{r['avg_comp']:10.3f}  {r['spread_q0']:10.4f}  {r['error_pct']:7.2f}%  {r['best_loss']:10.2e}")

    print(f"\nFinished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

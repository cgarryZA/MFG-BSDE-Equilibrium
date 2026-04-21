#!/usr/bin/env python -u
"""
N-scaling for the BSDEJ warm-started jump solver.

Tests whether the deep BSDE (jump) method captures the mean-field
convergence toward spread(N) = 2.247 - 2.343/sqrt(N), similar to
the neural Bellman solver (which we showed tracks exact to 1.9%).

Run: python -u scripts/bsdej_n_scaling.py
"""

import sys, os, json, time, gc
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_bsdej_shared import CXBSDEJShared
from scripts.cont_xiong_exact import fictitious_play

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def run_bsdej_N(N):
    print(f"\n{'='*60}")
    print(f"BSDEJ (shared + warmstart): N={N}")
    print(f"{'='*60}", flush=True)

    gpu_reset()

    exact = fictitious_play(N=N, Q=5, Delta=1, max_iter=200)
    mid = len(exact['V']) // 2
    exact_spread = exact['delta_a'][mid] + exact['delta_b'][mid]
    print(f"  Exact spread: {exact_spread:.4f}", flush=True)

    solver = CXBSDEJShared(
        N=N, Q=5, Delta=1, T=10.0, M=50,
        lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
        device=device, lr=5e-4, n_iter=10000,
        batch_size=512, hidden=128, n_layers=3,
    )
    solver.warmstart_from_bellman(n_pretrain=2000)
    print(flush=True)
    result = solver.train()

    spread = result['U_profile'][5]['spread']
    error = abs(spread - exact_spread) / exact_spread * 100
    print(f"\n  N={N}: bsdej={spread:.4f}, exact={exact_spread:.4f}, error={error:.2f}%", flush=True)

    del solver; gpu_reset()
    return {
        "N": N, "exact_spread": float(exact_spread),
        "bsdej_spread": float(spread), "error_pct": float(error),
        "best_loss": float(result['best_loss']),
        "elapsed": result['elapsed'],
    }


if __name__ == "__main__":
    results = []
    for N in [2, 5, 10, 20, 50]:
        try:
            r = run_bsdej_N(N)
            results.append(r)
        except Exception as e:
            print(f"  N={N} FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print("SUMMARY: BSDEJ N-scaling")
    print(f"{'='*60}")
    print(f"{'N':>4s}  {'Exact':>8s}  {'BSDEJ':>8s}  {'Error':>7s}")
    for r in results:
        print(f"{r['N']:4d}  {r['exact_spread']:8.4f}  {r['bsdej_spread']:8.4f}  {r['error_pct']:6.2f}%", flush=True)

    with open("results_final/bsdej_n_scaling.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved", flush=True)

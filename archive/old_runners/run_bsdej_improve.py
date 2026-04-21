#!/usr/bin/env python -u
"""
Try to push the BSDEJ shared+warmstart error below 2.6%.
Three experiments in order of likelihood to work.

Run after current GPU job finishes:
  python -u run_bsdej_improve.py
"""

import sys, os, json, time, gc
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

from solver_cx_bsdej_shared import CXBSDEJShared

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def run_config(label, **kwargs):
    print(f"\n{'='*60}")
    print(f"CONFIG: {label}")
    print(f"{'='*60}", flush=True)

    gpu_reset()
    solver = CXBSDEJShared(
        N=2, Q=5, Delta=1,
        lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
        device=device,
        **kwargs,
    )
    solver.warmstart_from_bellman(n_pretrain=2000)
    print(flush=True)
    result = solver.train()

    spread = result["U_profile"][5]["spread"]
    error = abs(spread - 1.5153) / 1.5153 * 100
    print(f"\n  Spread: {spread:.4f}, error: {error:.2f}%, loss: {result['best_loss']:.2e}", flush=True)

    with open(f"results_final/bsdej_improve_{label}.json", "w") as f:
        json.dump(result, f, indent=2, default=float)

    del solver; gpu_reset()
    return spread, error, result['best_loss']


results = []

# Baseline: what we had (for reference)
# spread=1.476, error=2.6%, loss=1.9e-3

# Experiment 1: More iterations (20k instead of 10k)
try:
    s, e, l = run_config("20k_iter",
                         T=10.0, M=50, lr=5e-4, n_iter=20000,
                         batch_size=512, hidden=128, n_layers=3)
    results.append({"label": "20k iters", "spread": float(s), "error_pct": float(e), "loss": float(l)})
except Exception as err:
    print(f"FAILED: {err}", flush=True)

# Experiment 2: Larger batch (1024 instead of 512)
try:
    s, e, l = run_config("batch1024",
                         T=10.0, M=50, lr=5e-4, n_iter=15000,
                         batch_size=1024, hidden=128, n_layers=3)
    results.append({"label": "batch 1024", "spread": float(s), "error_pct": float(e), "loss": float(l)})
except Exception as err:
    print(f"FAILED: {err}", flush=True)

# Experiment 3: Lower LR, more iters (slower convergence but more stable)
try:
    s, e, l = run_config("slow_lr",
                         T=10.0, M=50, lr=2e-4, n_iter=25000,
                         batch_size=1024, hidden=128, n_layers=3)
    results.append({"label": "slow LR + big batch", "spread": float(s), "error_pct": float(e), "loss": float(l)})
except Exception as err:
    print(f"FAILED: {err}", flush=True)

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Config':<25s}  {'Spread':>8s}  {'Error':>7s}  {'Loss':>10s}")
print("-" * 60)
print(f"{'baseline (10k, batch 512)':<25s}  {'1.4760':>8s}  {'2.60%':>7s}  {'1.90e-03':>10s}")
for r in results:
    print(f"{r['label']:<25s}  {r['spread']:8.4f}  {r['error_pct']:6.2f}%  {r['loss']:10.2e}")

with open("results_final/bsdej_improvement_summary.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

print(f"\nFinished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

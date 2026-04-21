#!/usr/bin/env python -u
"""
Hyperparameter sensitivity sweep for the BSDEJ shared-weight solver.

Tests lr × hidden × n_layers grid to show the key result (spread ≈ 1.52)
is robust, not a lucky hyperparameter hit.

Run after run_all_remaining.py finishes:
  python -u scripts/hyperparam_sweep.py

Results go to results_final/hyperparam_sweep.json
"""

import sys, os, json, time, gc
import numpy as np
import torch
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results_final"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60, flush=True)


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_one(lr, hidden, n_layers, n_iter=5000):
    """Run one BSDEJ config. Returns spread at q=0 and loss."""
    from solver_cx_bsdej_shared import CXBSDEJShared

    gpu_reset()
    solver = CXBSDEJShared(
        N=2, Q=5, Delta=1,
        T=10.0, M=50,
        lambda_a=2.0, lambda_b=2.0,
        r=0.01, phi=0.005,
        device=device,
        lr=lr,
        n_iter=n_iter,
        batch_size=512,
        hidden=hidden,
        n_layers=n_layers,
    )
    solver.warmstart_from_bellman(n_pretrain=1500)
    result = solver.train()

    spread = result["U_profile"][5]["spread"]
    loss = result["best_loss"]
    elapsed = result["elapsed"]

    del solver
    gpu_reset()
    return spread, loss, elapsed


if __name__ == "__main__":
    nash = 1.5153

    # Grid
    lrs = [1e-4, 5e-4, 1e-3]
    hiddens = [64, 128, 256]
    n_layers_list = [2, 3]

    configs = []
    for lr in lrs:
        for h in hiddens:
            for nl in n_layers_list:
                configs.append({"lr": lr, "hidden": h, "n_layers": nl})

    print(f"Total configs: {len(configs)}")
    print(f"Estimated time: {len(configs) * 20 / 60:.1f} hours")
    print(f"{'='*70}", flush=True)
    print(f"{'lr':>8s}  {'hidden':>6s}  {'layers':>6s}  {'spread':>8s}  "
          f"{'error%':>7s}  {'loss':>10s}  {'time':>6s}", flush=True)
    print("-" * 70, flush=True)

    results = []
    for i, cfg in enumerate(configs):
        lr, h, nl = cfg["lr"], cfg["hidden"], cfg["n_layers"]
        try:
            spread, loss, elapsed = run_one(lr, h, nl)
            error = abs(spread - nash) / nash * 100
            print(f"{lr:8.0e}  {h:6d}  {nl:6d}  {spread:8.4f}  "
                  f"{error:6.1f}%  {loss:10.2e}  {elapsed/60:5.1f}m", flush=True)
            results.append({
                "lr": lr, "hidden": h, "n_layers": nl,
                "spread_q0": float(spread),
                "error_pct": float(error),
                "best_loss": float(loss),
                "elapsed": float(elapsed),
            })
        except Exception as e:
            print(f"{lr:8.0e}  {h:6d}  {nl:6d}  FAILED: {e}", flush=True)
            results.append({
                "lr": lr, "hidden": h, "n_layers": nl,
                "error": str(e),
            })

        # Save after each run (in case of crash)
        with open(os.path.join(RESULTS_DIR, "hyperparam_sweep.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)

    # Summary
    print(f"\n{'='*70}")
    print("SWEEP COMPLETE", flush=True)

    valid = [r for r in results if "spread_q0" in r]
    if valid:
        errors = [r["error_pct"] for r in valid]
        spreads = [r["spread_q0"] for r in valid]
        best = min(valid, key=lambda r: r["error_pct"])
        worst = max(valid, key=lambda r: r["error_pct"])

        print(f"\n  Configs tested: {len(valid)}/{len(configs)}")
        print(f"  Best:  lr={best['lr']:.0e}, h={best['hidden']}, "
              f"nl={best['n_layers']} → spread={best['spread_q0']:.4f} "
              f"({best['error_pct']:.1f}%)")
        print(f"  Worst: lr={worst['lr']:.0e}, h={worst['hidden']}, "
              f"nl={worst['n_layers']} → spread={worst['spread_q0']:.4f} "
              f"({worst['error_pct']:.1f}%)")
        print(f"  Mean error: {np.mean(errors):.1f}%")
        print(f"  Spread range: [{min(spreads):.4f}, {max(spreads):.4f}]")
        print(f"  Configs within 5% of Nash: "
              f"{sum(1 for e in errors if e < 5)}/{len(errors)}")
        print(f"  Configs within 10% of Nash: "
              f"{sum(1 for e in errors if e < 10)}/{len(errors)}", flush=True)

    total = sum(r.get("elapsed", 0) for r in results)
    print(f"\n  Total GPU time: {total/3600:.1f} hours")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

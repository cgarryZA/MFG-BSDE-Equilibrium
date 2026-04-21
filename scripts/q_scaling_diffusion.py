#!/usr/bin/env python -u
"""
Q-scaling for the continuous-inventory diffusion solver.

Tests whether the diffusion BSDE solver scales better than the discrete
NN Bellman solver, since the continuous target is genuinely a smooth
function over [-Q, Q] — matching the NN's inductive bias.

Run: python -u scripts/q_scaling_diffusion.py
"""

import sys, os, json, time, gc
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_bsde_diffusion import CXBSDEDiffusion
from scripts.cont_xiong_exact import fictitious_play

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_diffusion_Q(Q):
    print(f"\n{'='*60}")
    print(f"Continuous-inventory diffusion BSDE: Q={Q}")
    print(f"{'='*60}", flush=True)

    gpu_reset()

    # Exact reference (discrete)
    exact = fictitious_play(N=2, Q=Q, Delta=1, max_iter=200)
    mid = len(exact['V']) // 2
    exact_spread = exact['delta_a'][mid] + exact['delta_b'][mid]
    print(f"  Exact discrete Nash spread: {exact_spread:.4f}", flush=True)

    solver = CXBSDEDiffusion(
        N=2, Q=Q, Delta=1,
        T=10.0, M=50,
        lambda_a=2.0, lambda_b=2.0,
        r=0.01, phi=0.005,
        device=device,
        lr=5e-4,
        n_iter=10000,
        batch_size=512,
        hidden=128, n_layers=3,
    )

    solver.warmstart_from_bellman(n_pretrain=2000)
    print(flush=True)
    result = solver.train()

    # Find the Z_profile entry at q=0
    mid_idx = next(i for i, item in enumerate(result['Z_profile']) if item['q'] == 0.0)
    spread = result['Z_profile'][mid_idx]['spread']
    Z_q0 = result['Z_profile'][mid_idx]['Z']
    error = abs(spread - exact_spread) / exact_spread * 100

    print(f"\n  Q={Q}: spread={spread:.4f}, exact={exact_spread:.4f}, error={error:.2f}%", flush=True)
    print(f"  Z(q=0) = {Z_q0:.6f}", flush=True)
    print(f"  Best loss: {result['best_loss']:.2e}", flush=True)

    del solver; gpu_reset()
    return {
        "Q": Q,
        "exact_spread": float(exact_spread),
        "neural_spread": float(spread),
        "error_pct": float(error),
        "Z_q0": float(Z_q0),
        "best_loss": float(result['best_loss']),
        "elapsed": result['elapsed'],
    }


if __name__ == "__main__":
    results = []
    for Q in [5, 10, 20]:  # skip Q=50 — too long at M=50, T=10
        try:
            r = run_diffusion_Q(Q)
            results.append(r)
        except Exception as e:
            print(f"  Q={Q} FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()
            results.append({"Q": Q, "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Continuous-inventory Diffusion Solver Q-scaling")
    print(f"{'='*60}")
    print(f"{'Q':>4s}  {'Exact':>8s}  {'Diffusion':>10s}  {'Error':>8s}  {'Loss':>10s}")
    print("-" * 55)
    for r in results:
        if "neural_spread" in r:
            print(f"{r['Q']:4d}  {r['exact_spread']:8.4f}  {r['neural_spread']:10.4f}  "
                  f"{r['error_pct']:7.2f}%  {r['best_loss']:10.2e}", flush=True)

    print(f"\nComparison with discrete NN:")
    print(f"{'Q':>4s}  {'Discrete NN':>12s}  {'Continuous NN':>14s}")
    nn_errors = {5: 0.59, 10: 0.32, 20: 8.17}
    for r in results:
        if "error_pct" in r:
            Q = r['Q']
            disc = nn_errors.get(Q, 0)
            cont = r['error_pct']
            winner = "continuous" if cont < disc else "discrete"
            print(f"{Q:4d}  {disc:11.2f}%  {cont:13.2f}%  [{winner} better]", flush=True)

    with open("results_final/q_scaling_diffusion.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to results_final/q_scaling_diffusion.json", flush=True)
    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

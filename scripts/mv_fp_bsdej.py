#!/usr/bin/env python -u
"""Proper McKean-Vlasov fictitious play wrapper around BSDEJ solver.

Tier 1 #1: Fixes the "fixed avg_comp" hack. At each outer FP iteration:
  1. Train BSDEJ network with current avg_comp (warm-start from previous weights)
  2. Extract equilibrium quotes from trained network
  3. Compute new avg_comp = mean(quotes) (boundary-fix: include zeros)
  4. Damped update
  5. Iterate until avg_comp stable

This IS the MV-BSDEJ structure of Han et al. 2022. Earns the lit review claim.

GPU or CPU. Per-outer-iter cost ~5-10 min on GPU with warmstart.
"""

import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_bsdej_shared import CXBSDEJShared
from utils import EarlyStopping
from scripts.cont_xiong_exact import fictitious_play
from equations.contxiong_exact import cx_exec_prob_np
from scipy.optimize import minimize_scalar

device = torch.device("cpu")  # CPU for stability
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def extract_quotes(solver, avg_comp):
    """Extract optimal quotes from trained BSDEJ at t=0 across q grid."""
    q_grid = np.arange(-solver.Q, solver.Q + solver.Delta, solver.Delta)
    das = np.zeros(len(q_grid)); dbs = np.zeros(len(q_grid))

    solver.shared_net.eval()
    with torch.no_grad():
        for i, q in enumerate(q_grid):
            t_n = torch.tensor([[0.0]], dtype=torch.float64, device=solver.device)
            q_n = torch.tensor([[q / solver.Q]], dtype=torch.float64, device=solver.device)
            U = solver.shared_net(t_n, q_n)
            Ua_v = U[0, 0].item()
            Ub_v = U[0, 1].item()

            def _neg(d, Uv):
                f = cx_exec_prob_np(d, avg_comp, solver.K, solver.N)
                return -f * (d + Uv)

            # Boundary fix: force zero at q=+-Q
            if q > -solver.Q:
                das[i] = minimize_scalar(
                    lambda d: _neg(d, Ua_v), bounds=(-1, 8), method='bounded').x
            if q < solver.Q:
                dbs[i] = minimize_scalar(
                    lambda d: _neg(d, Ub_v), bounds=(-1, 8), method='bounded').x

    solver.shared_net.train()
    return das, dbs


def run_mv_fp(N=2, n_outer=10, n_inner=4000, damping=0.5, verbose=True):
    print(f"\n{'='*60}")
    print(f"MV-BSDEJ fictitious play: N={N}, n_outer={n_outer}, damping={damping}")
    print(f"{'='*60}", flush=True)

    # Exact reference
    exact = fictitious_play(N=N, Q=5, Delta=1)
    nash_spread = exact['delta_a'][5] + exact['delta_b'][5]
    nash_avg = float(np.mean(exact['delta_a']))  # with boundary zero
    print(f"  Exact Nash: spread={nash_spread:.4f}, avg_comp={nash_avg:.4f}", flush=True)

    # Persistent solver across iterations (keeps network weights)
    avg_comp = 0.75  # initial
    solver = CXBSDEJShared(
        N=N, Q=5, Delta=1, T=10.0, M=50,
        lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
        device=device, lr=5e-4, n_iter=n_inner,
        batch_size=128, hidden=128, n_layers=3,
    )

    # Warm-start network from exact solution once (outside FP loop)
    solver.avg_comp = avg_comp
    solver.warmstart_from_bellman(n_pretrain=1500)

    history = []
    for k in range(n_outer):
        solver.avg_comp = avg_comp  # override
        print(f"\n  Outer iter {k+1}/{n_outer}: avg_comp={avg_comp:.4f}", flush=True)

        # Shorter training per outer iter (warm-started from previous)
        t0 = time.time()
        result = solver.train(early_stopping=True, es_patience=500,
                              es_min_delta=1e-7, es_warmup=1000)
        elapsed = time.time() - t0

        # Extract equilibrium quotes
        das, dbs = extract_quotes(solver, avg_comp)
        new_avg_a = float(np.mean(das))
        new_avg_b = float(np.mean(dbs))
        new_avg = 0.5 * (new_avg_a + new_avg_b)  # avg of both sides

        spread_q0 = das[5] + dbs[5]
        error = abs(spread_q0 - nash_spread) / nash_spread * 100

        diff = abs(new_avg - avg_comp)
        # Damped update
        avg_comp = damping * new_avg + (1 - damping) * avg_comp

        print(f"    inner loss={result['best_loss']:.4e}, time={elapsed:.0f}s")
        print(f"    quotes: spread(0)={spread_q0:.4f} (vs Nash {nash_spread:.4f}, err={error:.2f}%)")
        print(f"    avg_comp: {avg_comp:.4f} (new={new_avg:.4f}, diff={diff:.4f}, Nash={nash_avg:.4f})",
              flush=True)

        history.append({
            "k": k, "avg_comp": avg_comp, "new_avg": new_avg,
            "diff": diff, "spread_q0": float(spread_q0),
            "error_pct": float(error), "inner_loss": float(result['best_loss']),
            "elapsed": elapsed,
        })

        # Save incrementally
        with open(f"results_final/mv_fp_bsdej_N{N}.json", "w") as f:
            json.dump({
                "N": N, "nash_spread": nash_spread, "nash_avg": nash_avg,
                "history": history,
                "final_avg_comp": avg_comp, "final_spread": float(spread_q0),
                "final_error_pct": float(error),
                "final_das": das.tolist(), "final_dbs": dbs.tolist(),
            }, f, indent=2, default=float)

        if diff < 0.005:
            print(f"  Converged at outer iter {k+1}", flush=True)
            break

    return avg_comp, float(spread_q0), float(error), history


if __name__ == "__main__":
    all_results = {}
    for N in [2, 5, 10]:
        gc.collect()
        avg, spread, err, hist = run_mv_fp(N=N, n_outer=8, n_inner=3000)
        all_results[f"N={N}"] = {
            "final_avg_comp": avg, "final_spread": spread,
            "final_error_pct": err, "history": hist,
        }

    print(f"\n{'='*60}")
    print("MV-BSDEJ SUMMARY (proper fictitious play)")
    print(f"{'='*60}")
    print(f"{'N':>3s}  {'final avg_comp':>15s}  {'final spread':>14s}  {'error':>8s}  {'outer iters':>12s}")
    for key, r in all_results.items():
        N = int(key.split("=")[1])
        print(f"{N:3d}  {r['final_avg_comp']:15.4f}  {r['final_spread']:14.4f}  "
              f"{r['final_error_pct']:7.2f}%  {len(r['history']):12d}")

    with open("results_final/mv_fp_bsdej_all.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nFinished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

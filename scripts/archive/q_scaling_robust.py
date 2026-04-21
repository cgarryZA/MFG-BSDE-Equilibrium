#!/usr/bin/env python -u
"""
Robust Q-scaling: more FP iterations, track convergence per iteration.
Compares neural vs exact at each FP step to diagnose where it goes wrong.

Run: python -u scripts/q_scaling_robust.py
"""

import sys, os, json, time, gc
import numpy as np
import torch
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_q_scaling(Q_val, n_inner, n_outer, lr):
    """Run neural FP with convergence tracking per iteration."""
    print(f"\n{'='*60}")
    print(f"Q={Q_val}: inner={n_inner}, outer={n_outer}, lr={lr}")
    print(f"{'='*60}", flush=True)

    gpu_reset()

    from types import SimpleNamespace
    from equations.contxiong_exact import ContXiongExact
    from solver_cx import CXFictitiousPlay
    from scripts.cont_xiong_exact import fictitious_play as exact_fp

    # Get exact solution first
    print("  Running exact Algorithm 1...", flush=True)
    exact = exact_fp(N=2, Q=Q_val, Delta=1, max_iter=100)
    mid = len(exact["V"]) // 2
    exact_spread = exact["delta_a"][mid] + exact["delta_b"][mid]
    exact_V = np.array(exact["V"])
    print(f"  Exact: spread(0)={exact_spread:.6f}, converged in {len(exact.get('history',[]))} iters", flush=True)

    # Neural FP
    config = SimpleNamespace(
        lambda_a=2.0, lambda_b=2.0, discount_rate=0.01,
        Delta_q=1.0, q_max=Q_val, phi=0.005, N_agents=2,
    )
    eqn = ContXiongExact(config)
    nq = eqn.nq
    hidden = max(64, nq * 4)
    n_layers = 2 if nq <= 21 else 3
    print(f"  Grid: {nq} levels, hidden={hidden}, layers={n_layers}", flush=True)

    fp = CXFictitiousPlay(
        eqn, device=device,
        outer_iter=n_outer,
        inner_iter=n_inner,
        lr=lr,
        damping=0.5,
    )

    t0 = time.time()
    result = fp.train()
    elapsed = time.time() - t0

    da = np.array(result["final_delta_a"])
    db = np.array(result["final_delta_b"])
    neural_spread = da[mid] + db[mid]
    error = abs(neural_spread - exact_spread) / exact_spread * 100

    # V error across all q
    neural_V = np.array(result["final_V"])
    v_rmse = np.sqrt(np.mean((neural_V - exact_V)**2))
    v_max_err = np.max(np.abs(neural_V - exact_V))

    print(f"\n  Neural: spread(0)={neural_spread:.6f}")
    print(f"  Exact:  spread(0)={exact_spread:.6f}")
    print(f"  Spread error: {error:.2f}%")
    print(f"  V RMSE: {v_rmse:.4f}, max|V_nn - V_exact|: {v_max_err:.4f}")
    print(f"  Time: {elapsed:.0f}s", flush=True)

    # Track FP convergence from history
    if "history" in result:
        print(f"\n  FP convergence history:")
        for h in result["history"]:
            if "spread_q0" in h:
                print(f"    iter {h.get('iteration', '?')}: spread(0)={h['spread_q0']:.4f}", flush=True)

    save = {
        "Q": Q_val, "nq": nq,
        "exact_spread": float(exact_spread),
        "neural_spread": float(neural_spread),
        "error_pct": float(error),
        "v_rmse": float(v_rmse),
        "v_max_err": float(v_max_err),
        "n_inner": n_inner, "n_outer": n_outer, "lr": lr,
        "hidden": hidden, "n_layers": n_layers,
        "elapsed": elapsed,
        "delta_a": da.tolist(),
        "delta_b": db.tolist(),
        "V_neural": neural_V.tolist(),
        "V_exact": exact_V.tolist(),
    }

    del fp, eqn
    gpu_reset()
    return save


if __name__ == "__main__":
    results = []

    # Q=5 baseline (should be ~0.6%)
    r = run_q_scaling(Q_val=5, n_inner=5000, n_outer=20, lr=1e-3)
    results.append(r)

    # Q=10
    r = run_q_scaling(Q_val=10, n_inner=5000, n_outer=20, lr=1e-3)
    results.append(r)

    # Q=20 — more iterations than before
    r = run_q_scaling(Q_val=20, n_inner=8000, n_outer=25, lr=5e-4)
    results.append(r)

    # Q=20 with even more
    r = run_q_scaling(Q_val=20, n_inner=10000, n_outer=30, lr=3e-4)
    results.append(r)

    # Q=50 — much more iterations
    r = run_q_scaling(Q_val=50, n_inner=12000, n_outer=40, lr=2e-4)
    results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("Q-SCALING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Q':>5s}  {'inner':>6s}  {'outer':>6s}  {'lr':>8s}  "
          f"{'exact':>8s}  {'neural':>8s}  {'error':>7s}  {'time':>6s}", flush=True)
    print("-" * 70, flush=True)
    for r in results:
        print(f"{r['Q']:5d}  {r['n_inner']:6d}  {r['n_outer']:6d}  {r['lr']:8.0e}  "
              f"{r['exact_spread']:8.4f}  {r['neural_spread']:8.4f}  "
              f"{r['error_pct']:6.2f}%  {r['elapsed']/60:5.0f}m", flush=True)

    out = "results_final/q_scaling_robust.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out}", flush=True)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

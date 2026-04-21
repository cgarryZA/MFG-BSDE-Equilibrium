"""
Overnight GPU run: everything we need for the dissertation.

Run with: python run_overnight.py

Estimated time: 6-8 hours total on a single GPU.

Jobs:
  1. BSDEJ solver — 15k iters, T=20 (close the 6% gap)          ~2.5 hrs
  2. BSDEJ solver — 15k iters, T=50 (even closer to stationary)  ~3.0 hrs
  3. Q=20 neural Bellman with FP (scalability)                    ~1.0 hr
  4. Q=50 neural Bellman with FP (scalability)                    ~1.5 hrs

Results saved to results_overnight/ with timestamps.
"""

import os
import sys
import json
import time
import gc
import numpy as np
import torch
from datetime import datetime

RESULTS_DIR = "results_overnight"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =====================================================================
# JOB 1: BSDEJ solver, T=20, 15k iterations
# Target: close the 6% gap between BSDEJ spread and Nash 1.515
# =====================================================================

def run_bsdej(T, M, n_iter, label):
    print(f"\n{'='*60}")
    print(f"JOB: BSDEJ solver T={T}, M={M}, {n_iter} iters [{label}]")
    print(f"{'='*60}")

    gpu_reset()
    from solver_cx_bsdej import CXBSDEJSolver

    solver = CXBSDEJSolver(
        N=2, Q=5, Delta=1,
        T=T, M=M,
        lambda_a=2.0, lambda_b=2.0,
        r=0.01, phi=0.005,
        device=device,
        lr=3e-4,
        n_iter=n_iter,
        batch_size=512,
        hidden=64,
    )

    result = solver.train()

    spread_q0 = result["U_profile"][5]["spread"]
    nash = 1.5153
    error = abs(spread_q0 - nash) / nash
    print(f"\n  Spread at q=0: {spread_q0:.4f} (Nash: {nash:.4f}, error: {error:.1%})")
    print(f"  Best loss: {result['best_loss']:.4e}")
    print(f"  Time: {result['elapsed']:.0f}s")

    out_path = os.path.join(RESULTS_DIR, f"bsdej_{label}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"  Saved to {out_path}")

    # Cleanup
    del solver
    gpu_reset()

    return result


# =====================================================================
# JOB 2: Q-scaling with neural Bellman + FP
# Target: validate solver works for Q=20, Q=50
# =====================================================================

def run_q_scaling(Q_val, n_inner, n_outer):
    print(f"\n{'='*60}")
    print(f"JOB: Q={Q_val} neural Bellman + Fictitious Play")
    print(f"{'='*60}")

    gpu_reset()

    from types import SimpleNamespace
    from equations.contxiong_exact import ContXiongExact
    from solver_cx import CXFictitiousPlay

    config = SimpleNamespace(
        lambda_a=2.0, lambda_b=2.0,
        discount_rate=0.01,
        Delta_q=1.0,
        q_max=Q_val,
        phi=0.005,
        N_agents=2,
    )
    eqn = ContXiongExact(config)
    print(f"  Grid: {eqn.nq} inventory levels, K={eqn.K}")

    fp = CXFictitiousPlay(
        eqn, device=device,
        outer_iter=n_outer,
        inner_iter=n_inner,
        lr=5e-4,
        damping=0.5,
    )

    start = time.time()
    result = fp.train()
    elapsed = time.time() - start

    # Extract spread at q=0
    mid = eqn.mid
    spread_q0 = result["delta_a"][mid] + result["delta_b"][mid]
    print(f"\n  Q={Q_val}: spread(0) = {spread_q0:.4f}")
    print(f"  Time: {elapsed:.0f}s")

    # Save
    save_data = {
        "Q": Q_val,
        "nq": eqn.nq,
        "spread_q0": spread_q0,
        "delta_a": [float(x) for x in result["delta_a"]],
        "delta_b": [float(x) for x in result["delta_b"]],
        "V": [float(x) for x in result["V"]],
        "elapsed": elapsed,
        "outer_iterations": len(result.get("history", [])),
    }

    out_path = os.path.join(RESULTS_DIR, f"q_scaling_Q{Q_val}.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved to {out_path}")

    del fp, eqn
    gpu_reset()

    return save_data


# =====================================================================
# MAIN: run everything sequentially
# =====================================================================

if __name__ == "__main__":
    all_results = {}
    total_start = time.time()

    # --- BSDEJ T=20 ---
    try:
        r = run_bsdej(T=20.0, M=100, n_iter=15000, label="T20")
        all_results["bsdej_T20"] = {
            "spread_q0": r["U_profile"][5]["spread"],
            "best_loss": r["best_loss"],
            "elapsed": r["elapsed"],
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["bsdej_T20"] = {"error": str(e)}

    # --- BSDEJ T=50 ---
    try:
        r = run_bsdej(T=50.0, M=200, n_iter=15000, label="T50")
        all_results["bsdej_T50"] = {
            "spread_q0": r["U_profile"][5]["spread"],
            "best_loss": r["best_loss"],
            "elapsed": r["elapsed"],
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["bsdej_T50"] = {"error": str(e)}

    # --- Q=20 ---
    try:
        r = run_q_scaling(Q_val=20, n_inner=5000, n_outer=15)
        all_results["q20"] = {
            "spread_q0": r["spread_q0"],
            "elapsed": r["elapsed"],
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["q20"] = {"error": str(e)}

    # --- Q=50 ---
    try:
        r = run_q_scaling(Q_val=50, n_inner=8000, n_outer=15)
        all_results["q50"] = {
            "spread_q0": r["spread_q0"],
            "elapsed": r["elapsed"],
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["q50"] = {"error": str(e)}

    # --- Summary ---
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL JOBS COMPLETE")
    print(f"Total time: {total_elapsed/3600:.1f} hours")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    print("\n--- Summary ---")
    nash = 1.5153
    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name}: FAILED — {res['error']}")
        elif "spread_q0" in res:
            s = res["spread_q0"]
            err = abs(s - nash) / nash * 100
            print(f"  {name}: spread(0) = {s:.4f}  ({err:.1f}% from Nash)  [{res['elapsed']/60:.0f} min]")

    # Save summary
    all_results["total_elapsed"] = total_elapsed
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSummary saved to {summary_path}")

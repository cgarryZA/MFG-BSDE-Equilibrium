#!/usr/bin/env python
"""
Q scaling test: push inventory limit from Q=5 to Q=50.

Q=5:  11 grid points (current, validated)
Q=10: 21 grid points (Algorithm 1 still fast)
Q=20: 41 grid points (Algorithm 1 slower)
Q=50: 101 grid points (Algorithm 1 slow, NN should be same speed)

For each Q: run exact Algorithm 1 + neural solver, compare.
"""

import gc, json, os, sys, time, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equations.contxiong_exact import ContXiongExact
from solver_cx import CXSolver
from scripts.cont_xiong_exact import fictitious_play as exact_fp

OUT = "results_cx_q_scaling"
os.makedirs(OUT, exist_ok=True)

def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def convert(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert(v) for v in obj]
    return obj


class CXConfig:
    lambda_a = 2.0; lambda_b = 2.0; discount_rate = 0.01
    Delta_q = 1.0; q_max = 5.0; phi = 0.005; N_agents = 2


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    results = {}

    for Q in [5, 10, 20, 50]:
        nq = int(2 * Q + 1)
        print(f"\n{'='*60}")
        print(f"Q={Q} ({nq} grid points)")
        print(f"{'='*60}")

        # Exact Algorithm 1
        print(f"  Running exact Algorithm 1...")
        t0 = time.time()
        try:
            ex = exact_fp(N=2, Q=Q, max_iter=100)
            mid = len(ex["q_grid"]) // 2
            ex_spread = ex["delta_a"][mid] + ex["delta_b"][mid]
            ex_v = ex["V"][mid]
            ex_time = time.time() - t0
            print(f"  Exact: spread(0)={ex_spread:.4f}, V(0)={ex_v:.4f} ({ex_time:.1f}s)")
        except Exception as e:
            print(f"  Exact FAILED: {e}")
            ex_spread = None; ex_v = None; ex_time = None

        # Neural solver
        gpu_reset()
        print(f"  Running neural solver...")
        t0 = time.time()
        cfg = CXConfig(); cfg.q_max = float(Q); cfg.N_agents = 2
        eqn = ContXiongExact(cfg)
        # More iterations for larger Q (harder problem)
        n_iter = 5000 if Q <= 10 else 8000 if Q <= 20 else 12000
        solver = CXSolver(eqn, device=device, lr=1e-3, n_iter=n_iter, verbose=False)
        r = solver.train()
        nn_time = time.time() - t0
        nn_spread = r["delta_a"][eqn.mid] + r["delta_b"][eqn.mid]
        nn_v = r["V"][eqn.mid]
        print(f"  Neural: spread(0)={nn_spread:.4f}, V(0)={nn_v:.4f} ({nn_time:.1f}s)")

        if ex_spread is not None:
            err = abs(nn_spread - ex_spread)
            print(f"  Error: {err:.4f} ({err/ex_spread*100:.1f}%)")

        results[f"Q={Q}"] = {
            "Q": Q, "nq": nq,
            "exact_spread": ex_spread, "exact_V": ex_v, "exact_time": ex_time,
            "nn_spread": nn_spread, "nn_V": nn_v, "nn_time": nn_time,
            "nn_delta_a": r["delta_a"], "nn_delta_b": r["delta_b"], "nn_V_all": r["V"],
        }
        if ex_spread is not None:
            results[f"Q={Q}"]["exact_delta_a"] = ex["delta_a"]
            results[f"Q={Q}"]["exact_delta_b"] = ex["delta_b"]

        with open(os.path.join(OUT, "q_scaling.json"), "w") as f:
            json.dump(convert(results), f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Q':>4} {'nq':>5} {'Ex_spr':>8} {'NN_spr':>8} {'Err':>8} {'Ex_time':>8} {'NN_time':>8}")
    for k in sorted(results.keys(), key=lambda x: results[x]["Q"]):
        r = results[k]
        ex_s = f"{r['exact_spread']:.4f}" if r['exact_spread'] else "---"
        err = f"{abs(r['nn_spread']-r['exact_spread']):.4f}" if r['exact_spread'] else "---"
        ex_t = f"{r['exact_time']:.1f}s" if r['exact_time'] else "---"
        print(f"  {r['Q']:4d} {r['nq']:5d} {ex_s:>8} {r['nn_spread']:8.4f} {err:>8} {ex_t:>8} {r['nn_time']:.1f}s")


if __name__ == "__main__":
    main()

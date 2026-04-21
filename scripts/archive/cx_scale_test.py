#!/usr/bin/env python
"""
Scale test: N=1,2,5,10,50 dealers on the exact CX model.

For N=1,2: compare neural solver against exact Algorithm 1.
For N=5+: Algorithm 1 becomes slow/intractable; neural solver is the only option.

This demonstrates the curse of dimensionality that motivates the deep learning approach.
"""

import sys
import os
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equations.contxiong_exact import ContXiongExact
from solver_cx import CXSolver
from scripts.cont_xiong_exact import fictitious_play as exact_fp


class CXConfig:
    lambda_a = 2.0
    lambda_b = 2.0
    discount_rate = 0.01
    Delta_q = 1.0
    q_max = 5.0
    phi = 0.005
    N_agents = 2  # overridden per test


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("results_cx_scale", exist_ok=True)
    results = {}

    for N in [1, 2, 5, 10, 50]:
        print(f"\n{'='*60}")
        print(f"N = {N} dealers")
        print(f"{'='*60}")

        # Exact Algorithm 1 (only feasible for small N)
        exact_spread = None
        exact_v = None
        if N <= 5:
            print(f"  Running exact Algorithm 1...")
            t0 = time.time()
            try:
                ex = exact_fp(N=N, max_iter=50)
                mid = len(ex["q_grid"]) // 2
                exact_spread = ex["delta_a"][mid] + ex["delta_b"][mid]
                exact_v = ex["V"][mid]
                exact_time = time.time() - t0
                print(f"  Exact: spread(0)={exact_spread:.4f}, V(0)={exact_v:.4f} ({exact_time:.1f}s)")
            except Exception as e:
                print(f"  Exact FAILED: {e}")
                exact_time = None
        else:
            print(f"  Exact Algorithm 1: SKIPPED (intractable for N={N})")

        # Neural solver
        print(f"  Running neural solver...")
        config = CXConfig()
        config.N_agents = N
        eqn = ContXiongExact(config)
        print(f"    K={eqn.K} competitor levels, gamma={eqn.gamma:.4f}")

        t0 = time.time()
        solver = CXSolver(eqn, device=device, lr=1e-3, n_iter=5000, verbose=False)
        r = solver.train()
        nn_time = time.time() - t0
        nn_spread = r["delta_a"][eqn.mid] + r["delta_b"][eqn.mid]
        nn_v = r["V"][eqn.mid]
        print(f"  Neural: spread(0)={nn_spread:.4f}, V(0)={nn_v:.4f} ({nn_time:.1f}s)")

        if exact_spread is not None:
            err = abs(nn_spread - exact_spread)
            print(f"  Error: {err:.4f} ({err/exact_spread*100:.1f}%)")

        results[f"N={N}"] = {
            "N": N, "K": eqn.K,
            "nn_spread": nn_spread, "nn_V": nn_v, "nn_time": nn_time,
            "exact_spread": exact_spread, "exact_V": exact_v,
            "nn_delta_a": r["delta_a"], "nn_delta_b": r["delta_b"],
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'N':>4} {'K':>6} {'NN_spread':>10} {'Ex_spread':>10} {'Error':>8} {'NN_time':>8}")
    for key in sorted(results.keys(), key=lambda x: results[x]["N"]):
        r = results[key]
        ex_s = f"{r['exact_spread']:.4f}" if r['exact_spread'] else "---"
        err = f"{abs(r['nn_spread']-r['exact_spread']):.4f}" if r['exact_spread'] else "---"
        print(f"  {r['N']:4d} {r['K']:6d} {r['nn_spread']:10.4f} {ex_s:>10} {err:>8} {r['nn_time']:7.1f}s")

    with open("results_cx_scale/scale_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved results_cx_scale/scale_results.json")


if __name__ == "__main__":
    main()

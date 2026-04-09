#!/usr/bin/env python
"""
Test the CX solver against the exact Algorithm 1 ground truth.

Ground truth (from cont_xiong_exact.py):
  N=2 Nash: spread(0) = 1.4779, V(0) = 33.4081

The neural solver should reproduce these numbers.
"""

import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from equations.contxiong_exact import ContXiongExact
from solver_cx import CXSolver


class SimpleConfig:
    """Minimal config for CX model."""
    lambda_a = 2.0
    lambda_b = 2.0
    discount_rate = 0.01
    Delta_q = 1.0
    q_max = 5.0
    phi = 0.005
    N_agents = 2


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Ground truth
    print("\n=== Ground Truth (Algorithm 1) ===")
    print("  N=2: spread(0)=1.4779, V(0)=33.4081")

    # Neural solver
    print("\n=== Neural CX Solver ===")
    config = SimpleConfig()
    eqn = ContXiongExact(config)
    print(f"  Inventory grid: {eqn.q_grid}")
    print(f"  nq={eqn.nq}, K={eqn.K}, gamma={eqn.gamma:.4f}")

    solver = CXSolver(eqn, device=device, lr=1e-3, n_iter=5000)
    result = solver.train()

    # Compare
    print("\n=== Comparison ===")
    mid = eqn.mid
    nn_spread = result["delta_a"][mid] + result["delta_b"][mid]
    nn_v = result["V"][mid]
    print(f"  Ground truth: spread(0)=1.4779, V(0)=33.4081")
    print(f"  Neural solver: spread(0)={nn_spread:.4f}, V(0)={nn_v:.4f}")
    print(f"  Spread error: {abs(nn_spread - 1.4779):.4f}")
    print(f"  Value error: {abs(nn_v - 33.4081):.4f}")

    # Full quote profile
    print(f"\n  Quote profile:")
    print(f"  {'q':>4} {'NN_da':>8} {'NN_db':>8} {'NN_spr':>8} {'CX_da':>8} {'CX_db':>8} {'CX_spr':>8}")

    # Load CX ground truth if available
    cx_path = "results_cx_exact/nash_N2.json"
    if os.path.exists(cx_path):
        with open(cx_path) as f:
            cx = json.load(f)
        for j, q in enumerate(eqn.q_grid):
            nn_da = result["delta_a"][j]
            nn_db = result["delta_b"][j]
            cx_da = cx["delta_a"][j]
            cx_db = cx["delta_b"][j]
            print(f"  {q:4.0f} {nn_da:8.4f} {nn_db:8.4f} {nn_da+nn_db:8.4f} "
                  f"{cx_da:8.4f} {cx_db:8.4f} {cx_da+cx_db:8.4f}")
    else:
        for j, q in enumerate(eqn.q_grid):
            nn_da = result["delta_a"][j]
            nn_db = result["delta_b"][j]
            print(f"  {q:4.0f} {nn_da:8.4f} {nn_db:8.4f} {nn_da+nn_db:8.4f}")

    # Save
    os.makedirs("results_cx_solver", exist_ok=True)
    with open("results_cx_solver/test_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved results_cx_solver/test_result.json")


if __name__ == "__main__":
    main()

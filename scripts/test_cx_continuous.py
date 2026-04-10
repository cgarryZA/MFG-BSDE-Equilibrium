#!/usr/bin/env python
"""
Test continuous-inventory CX solver.
Compare against discrete Algorithm 1 at grid points.
Show smooth interpolation between grid points.
"""

import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_continuous import CXContinuousSolver
from scripts.cont_xiong_exact import fictitious_play as exact_fp


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("results_cx_continuous", exist_ok=True)

    # Ground truth
    print("\n=== Exact Algorithm 1 (N=2) ===")
    exact = exact_fp(N=2, max_iter=50)
    mid = len(exact["q_grid"]) // 2
    print(f"  spread(0)={exact['delta_a'][mid]+exact['delta_b'][mid]:.4f}")

    # Continuous solver
    print("\n=== Continuous CX solver (N=2) ===")
    solver = CXContinuousSolver(N=2, device=device, n_iter=15000, batch_size=64, lr=5e-4)
    result = solver.train()

    # Compare at grid points
    print(f"\n=== Comparison at grid points ===")
    print(f"  {'q':>4} {'Ex_da':>7} {'NN_da':>7} {'Ex_db':>7} {'NN_db':>7} {'Ex_spr':>7} {'NN_spr':>7}")
    for j, q in enumerate(exact["q_grid"]):
        ex_da = exact["delta_a"][j]; ex_db = exact["delta_b"][j]
        nn_da = result["delta_a"][j]; nn_db = result["delta_b"][j]
        print(f"  {q:4.0f} {ex_da:7.4f} {nn_da:7.4f} {ex_db:7.4f} {nn_db:7.4f} "
              f"{ex_da+ex_db:7.4f} {nn_da+nn_db:7.4f}")

    # Show interpolation at non-grid points
    print(f"\n=== Interpolation (non-grid points) ===")
    print(f"  {'q':>6} {'V(q)':>8} {'da':>7} {'db':>7} {'spread':>7}")
    q_fine = result["q_fine"]
    V_fine = result["V_fine"]
    for i in range(0, len(q_fine), 5):
        q = q_fine[i]
        v = V_fine[i]
        # Only show V, quotes require FOC solve
        print(f"  {q:6.2f} {v:8.4f}")

    with open("results_cx_continuous/continuous_result.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\n  Saved results_cx_continuous/continuous_result.json")


if __name__ == "__main__":
    main()

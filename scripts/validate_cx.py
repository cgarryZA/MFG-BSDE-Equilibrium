#!/usr/bin/env python
"""
Full validation: Neural CX solver vs exact Algorithm 1.

1. Run exact Algorithm 1 (ground truth)
2. Run neural single-pass solver
3. Run neural fictitious play
4. Compare all three
"""

import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equations.contxiong_exact import ContXiongExact
from solver_cx import CXSolver, CXFictitiousPlay
from scripts.cont_xiong_exact import fictitious_play as exact_fp, psi_func


class SimpleConfig:
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
    os.makedirs("results_cx_validation", exist_ok=True)

    # =========================================================
    # 1. Exact Algorithm 1 (ground truth)
    # =========================================================
    print("\n" + "=" * 60)
    print("1. Exact Cont-Xiong Algorithm 1 (N=2)")
    print("=" * 60)
    exact = exact_fp(N=2, max_iter=50)
    mid = len(exact["q_grid"]) // 2
    print(f"  spread(0)={exact['delta_a'][mid]+exact['delta_b'][mid]:.4f}")
    print(f"  V(0)={exact['V'][mid]:.4f}")

    # =========================================================
    # 2. Neural single-pass solver
    # =========================================================
    print("\n" + "=" * 60)
    print("2. Neural CX solver (single-pass, self-consistent)")
    print("=" * 60)
    config = SimpleConfig()
    eqn = ContXiongExact(config)
    solver = CXSolver(eqn, device=device, lr=1e-3, n_iter=8000)
    r_single = solver.train()

    # =========================================================
    # 3. Neural fictitious play
    # =========================================================
    print("\n" + "=" * 60)
    print("3. Neural fictitious play (outer loop)")
    print("=" * 60)
    fp = CXFictitiousPlay(eqn, device=device, outer_iter=15, inner_iter=3000,
                           lr=1e-3, damping=0.5)
    r_fp = fp.train()

    # =========================================================
    # 4. Comparison
    # =========================================================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    exact_spread = exact["delta_a"][mid] + exact["delta_b"][mid]
    exact_v = exact["V"][mid]
    single_spread = r_single["delta_a"][mid] + r_single["delta_b"][mid]
    single_v = r_single["V"][mid]
    fp_spread = r_fp["final_delta_a"][mid] + r_fp["final_delta_b"][mid]
    fp_v = r_fp["final_V"][mid]

    print(f"\n  {'Method':>25} {'spread(0)':>10} {'V(0)':>10} {'spr_err':>10} {'V_err':>10}")
    print(f"  {'Exact Algorithm 1':>25} {exact_spread:10.4f} {exact_v:10.4f} {'---':>10} {'---':>10}")
    print(f"  {'Neural single-pass':>25} {single_spread:10.4f} {single_v:10.4f} "
          f"{abs(single_spread-exact_spread):10.4f} {abs(single_v-exact_v):10.4f}")
    print(f"  {'Neural FP':>25} {fp_spread:10.4f} {fp_v:10.4f} "
          f"{abs(fp_spread-exact_spread):10.4f} {abs(fp_v-exact_v):10.4f}")

    # Full quote comparison
    print(f"\n  Quote profile (N=2, at equilibrium):")
    print(f"  {'q':>4} {'Ex_da':>7} {'Ex_db':>7} {'Ex_spr':>7} | {'NN_da':>7} {'NN_db':>7} {'NN_spr':>7} | {'FP_da':>7} {'FP_db':>7} {'FP_spr':>7}")
    for j, q in enumerate(exact["q_grid"]):
        ex_da = exact["delta_a"][j]; ex_db = exact["delta_b"][j]
        nn_da = r_single["delta_a"][j]; nn_db = r_single["delta_b"][j]
        fp_da = r_fp["final_delta_a"][j]; fp_db = r_fp["final_delta_b"][j]
        print(f"  {q:4.0f} {ex_da:7.4f} {ex_db:7.4f} {ex_da+ex_db:7.4f} | "
              f"{nn_da:7.4f} {nn_db:7.4f} {nn_da+nn_db:7.4f} | "
              f"{fp_da:7.4f} {fp_db:7.4f} {fp_da+fp_db:7.4f}")

    # Save
    results = {
        "exact": {"spread_q0": exact_spread, "V_q0": exact_v,
                  "delta_a": exact["delta_a"], "delta_b": exact["delta_b"], "V": exact["V"]},
        "neural_single": {"spread_q0": single_spread, "V_q0": single_v,
                          "delta_a": r_single["delta_a"], "delta_b": r_single["delta_b"]},
        "neural_fp": {"spread_q0": fp_spread, "V_q0": fp_v,
                      "delta_a": r_fp["final_delta_a"], "delta_b": r_fp["final_delta_b"],
                      "history": r_fp["history"]},
    }
    with open("results_cx_validation/validation.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Saved results_cx_validation/validation.json")


if __name__ == "__main__":
    main()

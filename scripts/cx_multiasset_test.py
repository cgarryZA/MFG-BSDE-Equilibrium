#!/usr/bin/env python
"""
Multi-asset CX test.

K=1: validate against single-asset exact (should match).
K=2: validate against brute-force 2D exact (grid is nq^2 = 121 for Q=5).
K=5: only neural solver runs.

For K=2 brute-force: solve V(q1, q2) on the 2D grid by iterating the
multi-dimensional Bellman equation. This is expensive but tractable.
"""

import gc, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_multiasset import CXMultiAssetSolver
from scripts.cont_xiong_exact import fictitious_play as exact_fp


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("results_cx_multiasset", exist_ok=True)
    results = {}

    # K=1 baseline (validate against exact)
    print(f"\n{'='*60}")
    print(f"K=1 (single asset, validate against exact)")
    print(f"{'='*60}")
    exact = exact_fp(N=2, Q=5, max_iter=50)
    mid = len(exact["q_grid"]) // 2
    ex_v = exact["V"][mid]
    print(f"  Exact V(0)={ex_v:.4f}")

    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    solver1 = CXMultiAssetSolver(K=1, N=2, device=device, n_iter=8000, batch_size=64, lr=1e-3)
    r1 = solver1.train()
    print(f"  Neural V(0)={r1['V_0']:.4f}, error={abs(r1['V_0']-ex_v):.4f}")
    results["K=1"] = {"V_0": r1["V_0"], "exact_V_0": ex_v, "elapsed": r1["elapsed"]}

    # K=2
    print(f"\n{'='*60}")
    print(f"K=2 (two assets)")
    print(f"{'='*60}")
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    solver2 = CXMultiAssetSolver(K=2, N=2, device=device, n_iter=15000, batch_size=128, lr=5e-4)
    r2 = solver2.train()
    # For independent assets with additive penalty, V(0,0) = 2 * V_single(0)
    expected_v = 2 * ex_v
    print(f"  Neural V(0,0)={r2['V_0']:.4f}")
    print(f"  Expected (2 * single): {expected_v:.4f}")
    print(f"  Difference: {abs(r2['V_0'] - expected_v):.4f}")
    results["K=2"] = {"V_0": r2["V_0"], "expected_V_0": expected_v, "elapsed": r2["elapsed"]}

    # K=5
    print(f"\n{'='*60}")
    print(f"K=5 (five assets — Algorithm 1 intractable)")
    print(f"{'='*60}")
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    solver5 = CXMultiAssetSolver(K=5, N=2, device=device, n_iter=20000, batch_size=256, lr=3e-4)
    r5 = solver5.train()
    expected_v5 = 5 * ex_v
    print(f"  Neural V(0,...,0)={r5['V_0']:.4f}")
    print(f"  Expected (5 * single): {expected_v5:.4f}")
    print(f"  Difference: {abs(r5['V_0'] - expected_v5):.4f}")
    results["K=5"] = {"V_0": r5["V_0"], "expected_V_0": expected_v5, "elapsed": r5["elapsed"]}

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'K':>4} {'NN_V(0)':>10} {'Expected':>10} {'Error':>8} {'Time':>8}")
    for k_str in ["K=1", "K=2", "K=5"]:
        r = results[k_str]
        exp = r.get("exact_V_0", r.get("expected_V_0", 0))
        err = abs(r["V_0"] - exp)
        print(f"  {k_str:>4} {r['V_0']:10.4f} {exp:10.4f} {err:8.4f} {r['elapsed']:.0f}s")

    with open("results_cx_multiasset/multiasset_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved results_cx_multiasset/multiasset_results.json")


if __name__ == "__main__":
    main()

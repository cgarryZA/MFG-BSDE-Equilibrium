#!/usr/bin/env python -u
"""Multi-asset K=5: genuine curse of dimensionality regime.

Tier 1 #2. Exact grid at K=5 is 11^5 = 161,051 points (painful but tractable).
At K=10 it's 2.6e10 (truly intractable). We do K=5 as the real CoD test.

GPU, ~1h. Saves incrementally. Graceful on failure.
"""

import sys, os, json, time, gc, traceback
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

OUT = "results_final/multiasset_K5_CoD.json"


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache(); torch.cuda.synchronize()
        except Exception:
            pass


def run_K(K, n_iter=2500, batch=32):
    """Train multi-asset solver at given K.

    Returns dict or None on failure.
    """
    try:
        print(f"\n{'='*60}")
        print(f"K = {K} (grid size {11**K:,})")
        print(f"{'='*60}", flush=True)
        gpu_reset()

        from solver_cx_multiasset import CXMultiAssetSolver
        solver = CXMultiAssetSolver(
            K=K, N=2, Q=5, device=device,
            n_iter=n_iter, batch_size=batch, lr=1e-3,
        )
        t0 = time.time()
        r = solver.train()
        elapsed = time.time() - t0

        out = {
            "K": K, "grid_size_brute_force": 11**K,
            "V_0": r["V_0"],
            "spreads_per_asset": r.get("spreads_per_asset", []),
            "avg_da": r.get("avg_da", [0.75]*K),
            "elapsed": elapsed,
        }
        print(f"\n  K={K}: V(0)={r['V_0']:.4f}, elapsed={elapsed:.0f}s", flush=True)

        # Independence benchmark: at K assets fully independent, V(0,...,0) = K * V(K=1)
        # If sub-additive: slight competitive coupling
        del solver; gpu_reset()
        return out
    except Exception as e:
        print(f"\n  K={K} FAILED: {e}", flush=True)
        traceback.print_exc()
        gpu_reset()
        return None


def main():
    all_results = []
    # Always save incrementally
    def save():
        with open(OUT, "w") as f:
            json.dump(all_results, f, indent=2, default=float)

    # Get K=1 baseline for sub-additivity comparison (should be fast)
    for K in [1, 2, 3, 5]:
        r = run_K(K, n_iter=2500 if K<=3 else 3500, batch=64 if K<=2 else (48 if K==3 else 32))
        if r is not None:
            all_results.append(r)
            save()
        else:
            all_results.append({"K": K, "error": "failed; see logs"})
            save()

    # Summary: sub-additivity check
    v_K1 = next((r["V_0"] for r in all_results if r.get("K") == 1), None)
    print(f"\n{'='*60}")
    print("MULTI-ASSET CoD SUMMARY")
    print(f"{'='*60}")
    print(f"{'K':>3s}  {'grid':>10s}  {'V(0)':>10s}  {'K*V(K=1)':>10s}  {'ratio':>7s}  {'time':>6s}")
    for r in all_results:
        K = r.get("K")
        if "V_0" in r and v_K1:
            expected = K * v_K1
            ratio = r["V_0"] / expected if expected > 0 else 0.0
            print(f"{K:3d}  {r['grid_size_brute_force']:>10,d}  {r['V_0']:10.4f}  "
                  f"{expected:10.4f}  {ratio:7.4f}  {r['elapsed']:5.0f}s")
        else:
            print(f"{K}: FAILED")

    save()
    print(f"\nSaved to {OUT}")
    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()

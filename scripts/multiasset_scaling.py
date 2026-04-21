#!/usr/bin/env python -u
"""Multi-asset scaling: K=1, K=2 with neural solver.

K=1: compare vs exact (single-asset Algorithm 1)
K=2: compare vs exact brute force on 121-point grid (if feasible)

Shows how neural solver scales with state dimension.

Run: python -u scripts/multiasset_scaling.py
"""

import sys, os, json, time, gc
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_multiasset import CXMultiAssetSolver
from scripts.cont_xiong_exact import fictitious_play

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

results = []

# ======== K=1 ========
print(f"\n{'='*60}")
print("K=1 (validate against exact Algorithm 1)")
print(f"{'='*60}", flush=True)
exact = fictitious_play(N=2, Q=5, Delta=1)
mid = len(exact['V']) // 2
exact_V0 = exact['V'][mid]
exact_spread = exact['delta_a'][mid] + exact['delta_b'][mid]
print(f"  Exact V(0)={exact_V0:.4f}, spread(0)={exact_spread:.4f}", flush=True)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

t0 = time.time()
solver = CXMultiAssetSolver(K=1, N=2, Q=5, device=device,
                            n_iter=2500, batch_size=64, lr=1e-3)
r = solver.train()
t_K1 = time.time() - t0

V0_K1 = r["V_0"]
spread_K1 = r["spreads_per_asset"][0]
V_err = abs(V0_K1 - exact_V0) / exact_V0 * 100
s_err = abs(spread_K1 - exact_spread) / exact_spread * 100

print(f"\n  K=1: V(0)={V0_K1:.4f} ({V_err:.2f}% error), "
      f"spread(0)={spread_K1:.4f} ({s_err:.2f}% error), "
      f"time={t_K1:.0f}s", flush=True)

results.append({
    "K": 1, "V_0": V0_K1, "exact_V0": exact_V0,
    "spread_q0": spread_K1, "exact_spread": exact_spread,
    "V_error_pct": V_err, "spread_error_pct": s_err,
    "elapsed": t_K1,
})

del solver
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ======== K=2 ========
print(f"\n{'='*60}")
print("K=2 (curse of dimensionality begins)")
print(f"{'='*60}", flush=True)

# No exact K=2 reference available here — just run and report
# But we can compare: at q=(0,0), V should be approximately 2*V_exact(0)
# because the two assets are independent given symmetric FP avg quotes.
# Actually it's NOT 2x because quotes adapt to competitive pressure.
# Just report the numerical result.

t0 = time.time()
solver = CXMultiAssetSolver(K=2, N=2, Q=5, device=device,
                            n_iter=2000, batch_size=64, lr=1e-3)
r = solver.train()
t_K2 = time.time() - t0

V0_K2 = r["V_0"]
spreads_K2 = r["spreads_per_asset"]

print(f"\n  K=2: V(0,0)={V0_K2:.4f}, spreads=[{spreads_K2[0]:.4f}, {spreads_K2[1]:.4f}], "
      f"time={t_K2:.0f}s", flush=True)
print(f"  (Expected V(0,0) ~ 2*V_K1(0) ={2*V0_K1:.4f} if assets truly independent)", flush=True)

results.append({
    "K": 2, "V_00": V0_K2, "spreads": spreads_K2,
    "V_K1_ref": V0_K1, "expected_if_independent": 2 * V0_K1,
    "elapsed": t_K2,
})

# Summary
print(f"\n{'='*60}")
print("MULTI-ASSET SCALING SUMMARY")
print(f"{'='*60}")
for r in results:
    K = r["K"]
    if K == 1:
        print(f"  K={K}: V={r['V_0']:.4f}, spread={r['spread_q0']:.4f}, V err={r['V_error_pct']:.2f}%, time={r['elapsed']:.0f}s")
    else:
        print(f"  K={K}: V={r['V_00']:.4f} (vs {r['expected_if_independent']:.4f} if indep), spreads={[f'{s:.4f}' for s in r['spreads']]}, time={r['elapsed']:.0f}s")

with open("results_final/multiasset_scaling.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved to results_final/multiasset_scaling.json", flush=True)
print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

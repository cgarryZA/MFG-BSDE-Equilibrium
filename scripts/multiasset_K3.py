#!/usr/bin/env python -u
"""Multi-asset K=3 — genuine curse of dimensionality regime.

At K=3 the exact brute-force grid is nq^3 = 11^3 = 1331 points — still
tractable but getting hard. At K=5 it's 161,051 and essentially intractable.

This run shows the neural solver handles K=3 where traditional methods
start to strain.

Run: python -u scripts/multiasset_K3.py
"""

import sys, os, json, time, gc
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_multiasset import CXMultiAssetSolver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

# Load K=1 reference from previous run
K1_result = None
try:
    prev = json.load(open('results_final/multiasset_scaling.json'))
    K1_result = prev[0]  # K=1 entry
    print(f"  K=1 reference: V(0)={K1_result['V_0']:.4f}", flush=True)
except Exception as e:
    print(f"  No K=1 reference available: {e}", flush=True)

print(f"\n{'='*60}")
print("K=3 multi-asset (grid size 1331)")
print(f"{'='*60}", flush=True)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

t0 = time.time()
solver = CXMultiAssetSolver(
    K=3, N=2, Q=5, device=device,
    n_iter=2000, batch_size=64, lr=1e-3,
)
r = solver.train()
t_K3 = time.time() - t0

V0 = r["V_0"]
spreads = r["spreads_per_asset"]
expected = 3 * K1_result['V_0'] if K1_result else None

print(f"\n  K=3: V(0,0,0)={V0:.4f}")
print(f"  spreads=[{', '.join(f'{s:.4f}' for s in spreads)}]")
if expected:
    print(f"  Reference 3*V_K1(0) = {expected:.4f} (independence upper bound)")
    print(f"  Ratio V_K3 / (3*V_K1) = {V0/expected:.4f}")
print(f"  Time: {t_K3:.0f}s", flush=True)

# Save
result = {
    "K": 3, "V_000": float(V0),
    "spreads": [float(s) for s in spreads],
    "V_K1_ref": K1_result['V_0'] if K1_result else None,
    "expected_if_independent": expected,
    "elapsed": t_K3,
}

# Append to multiasset results
try:
    prev = json.load(open('results_final/multiasset_scaling.json'))
except:
    prev = []
prev.append(result)
with open("results_final/multiasset_scaling.json", "w") as f:
    json.dump(prev, f, indent=2, default=float)
print(f"  Appended to results_final/multiasset_scaling.json", flush=True)

# Summary
print(f"\n{'='*60}")
print("MULTI-ASSET FULL SCALING SUMMARY")
print(f"{'='*60}")
for r in prev:
    K = r["K"]
    if K == 1:
        print(f"  K={K}: V(0)={r.get('V_0', 0):.4f}, spread={r.get('spread_q0', 0):.4f} (err {r.get('spread_error_pct', 0):.2f}%), time={r['elapsed']:.0f}s")
    elif K == 2:
        print(f"  K={K}: V(0,0)={r.get('V_00', 0):.4f}, spreads={r.get('spreads', [])}, time={r['elapsed']:.0f}s")
    elif K == 3:
        print(f"  K={K}: V(0,0,0)={r.get('V_000', 0):.4f}, spreads={r.get('spreads', [])}, time={r['elapsed']:.0f}s")

print(f"\nFinished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

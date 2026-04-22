#!/usr/bin/env python -u
"""Smoke test for the continuous-inventory diffusion BSDE solver.

Runs: warmstart (2000 pretraining steps) + 200-step training burst at N=2.
Target: network produces finite Z, loss doesn't diverge, extracted
spread is in a sensible range.

Matches the existing solver_cx_bsde_diffusion.CXBSDEDiffusion API:
  - attribute: solver.z_net (not solver.shared_net)
  - warmstart_from_bellman(n_pretrain=N) -- no verbose kwarg
  - train() returns a dict with "history" (list of logged step dicts)
    and "best_loss", etc.

Run: python -u scripts/test_diffusion_smoke.py
"""

import sys, os, traceback
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_bsde_diffusion import CXBSDEDiffusion
from scripts.cont_xiong_exact import fictitious_play

device = torch.device("cpu")
print("Diffusion BSDE smoke test")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}",
      flush=True)

try:
    nash = fictitious_play(N=2, Q=5, Delta=1)
    nash_spread = nash["delta_a"][5] + nash["delta_b"][5]
    nash_avg = float(np.mean(nash["delta_a"]))
    print(f"\n  Nash reference: spread(q=0) = {nash_spread:.4f}", flush=True)

    # Short run: 200 training steps, 500 pretrain
    solver = CXBSDEDiffusion(
        N=2, Q=5, Delta=1, T=10.0, M=50,
        lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
        device=device, lr=5e-4, n_iter=200,
        batch_size=64, hidden=128, n_layers=3,
    )
    solver.avg_comp = nash_avg

    print(f"\n  Z-net param count: "
          f"{sum(p.numel() for p in solver.z_net.parameters())}",
          flush=True)

    solver.warmstart_from_bellman(n_pretrain=500)

    # Pre-training smoke forward pass
    print(f"\n  Smoke forward pass (post-warmstart, pre-BSDE-training):", flush=True)
    with torch.no_grad():
        Y_T, q_T = solver.forward(batch_size=16)
        g_T = solver.terminal_condition(q_T)
        initial_loss = torch.mean((Y_T - g_T) ** 2).item()
        print(f"    Y_T range:  [{Y_T.min().item():.4f}, "
              f"{Y_T.max().item():.4f}]")
        print(f"    q_T range:  [{q_T.min().item():.4f}, "
              f"{q_T.max().item():.4f}]")
        print(f"    |Y_T - g(q_T)|^2 = {initial_loss:.4e}", flush=True)

    # Short training burst
    print(f"\n  Running 200-step training burst:", flush=True)
    result = solver.train(early_stopping=False)

    best = result["best_loss"]
    hist = result["history"]
    print(f"\n  Best loss: {best:.4e}", flush=True)
    if len(hist) >= 2:
        print(f"  Start loss: {hist[0]['loss']:.4e}")
        print(f"  End   loss: {hist[-1]['loss']:.4e}", flush=True)

    # Spread at q=0 from the result's Z_profile (index 5 is q=0)
    q0_entry = result["Z_profile"][5]
    spread_q0 = q0_entry["spread"]
    Z_q0 = q0_entry["Z"]
    err = abs(spread_q0 - nash_spread) / nash_spread * 100
    print(f"\n  Extracted spread(q=0) = {spread_q0:.4f}  "
          f"(Nash {nash_spread:.4f}, err {err:.2f}%)")
    print(f"  Z(q=0) = {Z_q0:+.6f} (should be near 0 by symmetry)",
          flush=True)

    # Full quote profile
    das = [e["da"] for e in result["Z_profile"]]
    dbs = [e["db"] for e in result["Z_profile"]]
    print(f"  das = {[round(x, 3) for x in das]}")
    print(f"  dbs = {[round(x, 3) for x in dbs]}", flush=True)

    # Validity checks
    checks_passed = 0; checks_total = 4

    if not np.isnan(best) and best < 100.0:
        print(f"\n  [PASS] loss is finite ({best:.4e})"); checks_passed += 1
    else:
        print(f"\n  [FAIL] loss is NaN or huge: {best}")

    if 0.1 < abs(spread_q0) < 10.0:
        print(f"  [PASS] extracted spread is finite and non-degenerate")
        checks_passed += 1
    else:
        print(f"  [FAIL] extracted spread is degenerate: {spread_q0}")

    if len(hist) >= 2 and hist[-1]["loss"] <= hist[0]["loss"] * 3.0:
        print(f"  [PASS] loss did not explode")
        checks_passed += 1
    else:
        print(f"  [FAIL] loss exploded or history too short")

    # Z antisymmetry sanity check
    Z_pos = result["Z_profile"][10]["Z"]   # q = +Q
    Z_neg = result["Z_profile"][0]["Z"]    # q = -Q
    if Z_pos * Z_neg < 0 or (abs(Z_pos) < 0.1 and abs(Z_neg) < 0.1):
        print(f"  [PASS] Z shows antisymmetry: Z(+Q)={Z_pos:+.4f}, "
              f"Z(-Q)={Z_neg:+.4f}"); checks_passed += 1
    else:
        print(f"  [WARN] Z antisymmetry not established: "
              f"Z(+Q)={Z_pos:+.4f}, Z(-Q)={Z_neg:+.4f}")

    print(f"\n  Smoke test: {checks_passed}/{checks_total} checks passed",
          flush=True)

except Exception as e:
    print(f"\n  SMOKE TEST FAILED: {e}", flush=True)
    traceback.print_exc()

print(f"\nFinished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}",
      flush=True)

#!/usr/bin/env python
"""
Phase transition mapper: where does non-linear price impact break the solver?

Sweeps:
1. kappa (impact strength): 0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0
2. impact_type: "linear", "sqrt", "quadratic"
3. Records: Y0, loss, max|Z|, whether training diverged

This documents the stability boundary when monotonicity breaks,
as promised in the lit review.

Usage:
    python scripts/run_impact_phase_transition.py --device cuda
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from registry import EQUATION_REGISTRY
import equations
from solver import ContXiongLOBMVSolver


def convert(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(v) for v in obj]
    return obj


def run_single(kappa, impact_type, n_iters, device):
    """Train one model, return diagnostics."""
    config = Config.from_json("configs/lob_d2_mv_impact.json")
    config.eqn.law_encoder_type = "moments"
    config.eqn.kappa = kappa
    config.eqn.impact_type = impact_type
    config.eqn.phi = 0.1
    config.net.opt_config1.num_iterations = n_iters
    config.net.logging_frequency = n_iters
    config.net.verbose = False

    bsde = EQUATION_REGISTRY["contxiong_lob_impact"](config.eqn)
    solver = ContXiongLOBMVSolver(config, bsde, device=device)

    try:
        result = solver.train()
        y0 = result["y0"]
        loss = result["final_loss"]
        z_max = solver.model._last_z_max_overall if hasattr(solver.model, "_last_z_max_overall") else 0.0

        # Check for divergence signs
        diverged = (
            np.isnan(y0) or np.isnan(loss) or
            abs(y0) > 10 or loss > 10 or z_max > 100
        )

        # Evaluate h at narrow/wide
        solver.model.eval()
        h_narrow, h_wide = None, None
        try:
            pop_n = np.stack([np.full(256, 100.0), np.clip(np.random.normal(0, 0.1, 256), -10, 10)], axis=1)
            pop_w = np.stack([np.full(256, 100.0), np.clip(np.random.normal(0, 3.0, 256), -10, 10)], axis=1)
            with torch.no_grad():
                le_n = solver.model.law_encoder.encode(torch.tensor(pop_n, dtype=torch.float64, device=device))
                le_w = solver.model.law_encoder.encode(torch.tensor(pop_w, dtype=torch.float64, device=device))
                h_narrow = bsde.compute_competitive_factor(le_n).item()
                h_wide = bsde.compute_competitive_factor(le_w).item()
        except:
            pass

        return {
            "y0": y0, "loss": loss, "z_max": z_max, "diverged": diverged,
            "h_narrow": h_narrow, "h_wide": h_wide,
        }
    except Exception as e:
        return {"y0": None, "loss": None, "z_max": None, "diverged": True, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_iters", type=int, default=2000)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    print(f"Device: {device}, iterations: {args.n_iters}")

    os.makedirs("results_impact", exist_ok=True)
    all_results = {}
    start = time.time()

    kappas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    impact_types = ["linear", "sqrt", "quadratic"]

    # ================================================================
    # Sweep: kappa × impact_type
    # ================================================================
    for impact_type in impact_types:
        print(f"\n{'='*60}")
        print(f"Impact type: {impact_type}")
        print(f"{'='*60}")
        for kappa in kappas:
            torch.manual_seed(42); np.random.seed(42)
            print(f"  kappa={kappa:.2f}...", end=" ", flush=True)
            t0 = time.time()
            r = run_single(kappa, impact_type, args.n_iters, device)
            elapsed = time.time() - t0
            status = "DIVERGED" if r["diverged"] else "OK"
            y0_str = f"{r['y0']:.4f}" if r['y0'] is not None else "NaN"
            loss_str = f"{r['loss']:.4e}" if r['loss'] is not None else "NaN"
            print(f"Y0={y0_str}, loss={loss_str}, [{status}] ({elapsed:.0f}s)")

            key = f"{impact_type}_kappa={kappa}"
            all_results[key] = {**r, "kappa": kappa, "impact_type": impact_type}

    # ================================================================
    # Summary: phase diagram
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE DIAGRAM")
    print(f"{'='*60}")
    print(f"{'kappa':>8}", end="")
    for it in impact_types:
        print(f"  {it:>12}", end="")
    print()
    print("-" * 50)
    for kappa in kappas:
        print(f"{kappa:8.2f}", end="")
        for it in impact_types:
            key = f"{it}_kappa={kappa}"
            r = all_results[key]
            if r["diverged"]:
                print(f"  {'DIVERGED':>12}", end="")
            else:
                print(f"  {r['y0']:12.4f}", end="")
        print()

    # Find boundary
    print(f"\nStability boundary:")
    for it in impact_types:
        last_stable = 0.0
        for kappa in kappas:
            key = f"{it}_kappa={kappa}"
            if not all_results[key]["diverged"]:
                last_stable = kappa
        print(f"  {it}: stable up to kappa={last_stable}")

    # ================================================================
    # Save
    # ================================================================
    elapsed = (time.time() - start) / 60
    all_results["metadata"] = {
        "n_iters": args.n_iters, "elapsed_minutes": elapsed,
        "device": str(device), "kappas": kappas, "impact_types": impact_types,
    }
    out = "results_impact/phase_transition.json"
    with open(out, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\nDone in {elapsed:.1f} min. Saved: {out}")


if __name__ == "__main__":
    main()

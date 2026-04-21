#!/usr/bin/env python
"""
Phase 3: Stability frontier mapping.

Systematic sweep across multiple axes to build a phase diagram
of solver behaviour: stable / marginal / diverged.

Axes:
  - phi (inventory penalty): 0.001 → 5.0
  - eta (adverse selection strength): 0 → 2.0
  - gamma (exponential penalty parameter): 0.5 → 10.0
  - T (horizon): 0.5 → 5.0
  - coupling type: none / moments / deepsets

Diagnostics per run:
  - Y0 (value at t=0)
  - final loss
  - max|Z_t| (gradient explosion)
  - convergence (loss < threshold and not NaN)
  - W2 residual (for MV runs)

Output: JSON with full results + phase diagram data.

Usage:
    python scripts/stability_frontier.py --quick --device cuda
    python scripts/stability_frontier.py --device cuda
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
from solver import ContXiongLOBSolver, ContXiongLOBMVSolver


def run_single(eqn_name, config_path, overrides, device, n_iters=500):
    """Run one experiment, return diagnostics."""
    config = Config.from_json(config_path)
    for key, val in overrides.items():
        parts = key.split(".")
        obj = config
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)

    config.net.opt_config1.num_iterations = n_iters
    config.net.logging_frequency = n_iters
    config.net.verbose = False

    try:
        bsde = EQUATION_REGISTRY[eqn_name](config.eqn)
        if "mv" in eqn_name:
            solver = ContXiongLOBMVSolver(config, bsde, device=device)
        else:
            solver = ContXiongLOBSolver(config, bsde, device=device)
        result = solver.train()

        y0 = result["y0"]
        loss = result["final_loss"]
        z_max = result["history"][-1, 3] if result["history"].shape[1] > 3 else 0.0
        converged = not (np.isnan(loss) or loss > 1.0)
        w2 = result.get("w2_history", [{}])[-1].get("w2", 0.0) if result.get("w2_history") else 0.0

        return {
            "y0": float(y0), "loss": float(loss), "z_max": float(z_max),
            "converged": converged, "w2": float(w2),
        }
    except Exception as e:
        return {
            "y0": float("nan"), "loss": float("inf"), "z_max": float("inf"),
            "converged": False, "w2": 0.0, "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_dir", default="results_stability")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    n_iters = 300 if args.quick else 1500
    all_results = {}
    start = time.time()

    # ================================================================
    # Sweep 1: phi × penalty_type (base model)
    # ================================================================
    print(f"\n{'='*60}")
    print("Sweep 1: phi × penalty type")
    print(f"{'='*60}")
    phi_sweep = []
    phis = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0] if not args.quick else [0.01, 0.1, 1.0, 5.0]
    for penalty in ["quadratic", "exponential"]:
        for phi in phis:
            overrides = {
                "eqn.phi": phi, "eqn.penalty_type": penalty, "eqn.type": 1,
                "net.opt_config1.freq_update_drift": 9999999,
            }
            if penalty == "exponential":
                overrides["eqn.gamma"] = 1.0
            label = f"{penalty}_phi={phi}"
            r = run_single("contxiong_lob", "configs/lob_d2.json", overrides, device, n_iters)
            r["phi"] = phi
            r["penalty"] = penalty
            phi_sweep.append(r)
            status = "OK" if r["converged"] else "DIVERGED"
            print(f"  {label}: Y0={r['y0']:.4f}, z_max={r['z_max']:.2f}, {status}")
    all_results["phi_penalty_sweep"] = phi_sweep

    # ================================================================
    # Sweep 2: eta (adverse selection strength)
    # ================================================================
    print(f"\n{'='*60}")
    print("Sweep 2: eta (adverse selection)")
    print(f"{'='*60}")
    eta_sweep = []
    etas = [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0] if not args.quick else [0.0, 0.5, 1.0, 2.0]
    for eta in etas:
        r = run_single("contxiong_lob_adverse", "configs/lob_d3_adverse.json",
                       {"eqn.eta": eta}, device, n_iters)
        r["eta"] = eta
        eta_sweep.append(r)
        status = "OK" if r["converged"] else "DIVERGED"
        print(f"  eta={eta}: Y0={r['y0']:.4f}, z_max={r['z_max']:.2f}, {status}")
    all_results["eta_sweep"] = eta_sweep

    # ================================================================
    # Sweep 3: T (horizon) with adverse selection
    # ================================================================
    print(f"\n{'='*60}")
    print("Sweep 3: T (horizon, adverse selection)")
    print(f"{'='*60}")
    T_sweep = []
    Ts = [0.5, 1.0, 2.0, 5.0] if not args.quick else [0.5, 1.0, 2.0]
    for T in Ts:
        r = run_single("contxiong_lob_adverse", "configs/lob_d3_adverse.json",
                       {"eqn.total_time": T}, device, n_iters)
        r["T"] = T
        T_sweep.append(r)
        status = "OK" if r["converged"] else "DIVERGED"
        print(f"  T={T}: Y0={r['y0']:.4f}, z_max={r['z_max']:.2f}, {status}")
    all_results["T_sweep_adverse"] = T_sweep

    # ================================================================
    # Sweep 4: phi × eta (2D phase diagram)
    # ================================================================
    print(f"\n{'='*60}")
    print("Sweep 4: phi × eta phase diagram")
    print(f"{'='*60}")
    phase_diagram = []
    phis_pd = [0.01, 0.05, 0.1, 0.5, 1.0] if not args.quick else [0.01, 0.1, 1.0]
    etas_pd = [0.0, 0.25, 0.5, 1.0] if not args.quick else [0.0, 0.5, 1.0]
    for phi in phis_pd:
        for eta in etas_pd:
            r = run_single("contxiong_lob_adverse", "configs/lob_d3_adverse.json",
                           {"eqn.phi": phi, "eqn.eta": eta}, device, n_iters)
            r["phi"] = phi
            r["eta"] = eta
            phase_diagram.append(r)
            status = "OK" if r["converged"] else "DIV"
            print(f"  phi={phi}, eta={eta}: z_max={r['z_max']:.2f}, {status}")
    all_results["phase_diagram_phi_eta"] = phase_diagram

    # ================================================================
    # Sweep 5: coupling type comparison (at stronger phi)
    # ================================================================
    print(f"\n{'='*60}")
    print("Sweep 5: coupling type at phi=0.1")
    print(f"{'='*60}")
    coupling_sweep = []
    for coupling in ["none", "moments", "deepsets"]:
        if coupling == "none":
            r = run_single("contxiong_lob_adverse", "configs/lob_d3_adverse.json",
                           {"eqn.phi": 0.1, "eqn.type": 1,
                            "net.opt_config1.freq_update_drift": 9999999},
                           device, n_iters)
        else:
            r = run_single("contxiong_lob_mv_adverse", "configs/lob_d3_mv_adverse.json",
                           {"eqn.phi": 0.1, "eqn.law_encoder_type": coupling},
                           device, n_iters)
        r["coupling"] = coupling
        coupling_sweep.append(r)
        status = "OK" if r["converged"] else "DIVERGED"
        print(f"  {coupling}: Y0={r['y0']:.4f}, loss={r['loss']:.4e}, {status}")
    all_results["coupling_sweep_phi01"] = coupling_sweep

    # ================================================================
    # Save
    # ================================================================
    elapsed = time.time() - start
    all_results["metadata"] = {
        "device": str(device), "n_iters": n_iters,
        "elapsed_seconds": elapsed, "quick": args.quick,
    }

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

    out_path = os.path.join(args.out_dir, "stability_frontier.json")
    with open(out_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print(f"All sweeps complete in {elapsed/60:.1f} min")
    print(f"Saved: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

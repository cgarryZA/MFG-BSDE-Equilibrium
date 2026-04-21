#!/usr/bin/env python
"""
McKean-Vlasov experiment suite.

Experiments:
1. Encoder ablation: moments vs quantiles vs histogram vs DeepSets
2. Law sensitivity: fix mean(q)=0, vary population shape
3. Comparison: MV solver vs moment-proxy vs no-coupling
4. Fictitious play convergence: W2 across iterations

Usage:
    python scripts/run_mv_experiments.py --quick --device cuda
    python scripts/run_mv_experiments.py --device cuda
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
from solver import ContXiongLOBMVSolver, ContXiongLOBSolver


def run_mv(config_path, overrides, device, label=""):
    """Run a single MV experiment."""
    config = Config.from_json(config_path)
    for key, val in overrides.items():
        parts = key.split(".")
        obj = config
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)

    config.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_mv"](config.eqn)
    solver = ContXiongLOBMVSolver(config, bsde, device=device)
    result = solver.train()
    y0 = result["y0"]
    loss = result["final_loss"]
    w2 = result["w2_history"][-1]["w2"] if result["w2_history"] else 0.0
    print(f"  {label}: Y0={y0:.4f}, loss={loss:.4e}, W2={w2:.6f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_dir", default="results_mv")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    n_iters = 500 if args.quick else 3000
    base = "configs/lob_d2_mv.json"
    all_results = {}
    start = time.time()

    # ================================================================
    # 1. Encoder ablation
    # ================================================================
    print(f"\n{'='*60}")
    print(f"1. Encoder ablation ({n_iters} iter each)")
    print(f"{'='*60}")
    for enc in ["moments", "quantiles", "histogram", "deepsets"]:
        r = run_mv(base, {
            "eqn.law_encoder_type": enc,
            "net.opt_config1.num_iterations": n_iters,
            "net.logging_frequency": n_iters,
        }, device, label=enc)
        all_results[f"encoder_{enc}"] = {
            "y0": r["y0"], "loss": r["final_loss"],
            "w2_history": r["w2_history"],
        }

    # ================================================================
    # 2. Comparison: MV vs moment-proxy vs no-coupling
    # ================================================================
    print(f"\n{'='*60}")
    print(f"2. MV vs baselines")
    print(f"{'='*60}")

    # MV with DeepSets
    r = run_mv(base, {
        "eqn.law_encoder_type": "deepsets",
        "net.opt_config1.num_iterations": n_iters,
        "net.logging_frequency": n_iters,
    }, device, label="MV-DeepSets")
    all_results["mv_deepsets"] = {"y0": r["y0"], "loss": r["final_loss"]}

    # Moment proxy (old solver, Type 3)
    config = Config.from_json("configs/lob_d2.json")
    config.eqn.type = 3
    config.net.opt_config1.num_iterations = n_iters
    config.net.opt_config1.freq_update_drift = 100
    config.net.logging_frequency = n_iters
    config.net.verbose = False
    bsde_old = EQUATION_REGISTRY["contxiong_lob"](config.eqn)
    solver_old = ContXiongLOBSolver(config, bsde_old, device=device)
    r_old = solver_old.train()
    print(f"  moment-proxy: Y0={r_old['y0']:.4f}, loss={r_old['final_loss']:.4e}")
    all_results["moment_proxy"] = {"y0": r_old["y0"], "loss": r_old["final_loss"]}

    # No coupling (Type 1)
    config2 = Config.from_json("configs/lob_d2.json")
    config2.eqn.type = 1
    config2.net.opt_config1.num_iterations = n_iters
    config2.net.opt_config1.freq_update_drift = 9999999
    config2.net.logging_frequency = n_iters
    config2.net.verbose = False
    bsde_nc = EQUATION_REGISTRY["contxiong_lob"](config2.eqn)
    solver_nc = ContXiongLOBSolver(config2, bsde_nc, device=device)
    r_nc = solver_nc.train()
    print(f"  no-coupling:  Y0={r_nc['y0']:.4f}, loss={r_nc['final_loss']:.4e}")
    all_results["no_coupling"] = {"y0": r_nc["y0"], "loss": r_nc["final_loss"]}

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

    out_path = os.path.join(args.out_dir, "mv_experiments.json")
    with open(out_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\nAll experiments complete in {elapsed/60:.1f} min")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

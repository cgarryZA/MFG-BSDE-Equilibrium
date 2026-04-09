#!/usr/bin/env python
"""
Fictitious play convergence experiments.

Tests whether the outer FP loop converges (W2 -> 0), which would
establish a Nash equilibrium candidate.

Usage:
    python scripts/run_fictitious_play.py --device cuda
    python scripts/run_fictitious_play.py --device cuda --quick
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
from solver import FictitiousPlaySolver


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--subnet_type", default="two_stream")
    parser.add_argument("--outer", type=int, default=None)
    parser.add_argument("--inner", type=int, default=None)
    parser.add_argument("--damping", type=float, default=None)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")

    outer = args.outer or (5 if args.quick else 15)
    inner = args.inner or (500 if args.quick else 2000)
    print(f"Device: {device}, outer={outer}, inner={inner}, subnet={args.subnet_type}")

    os.makedirs("results_fp", exist_ok=True)
    all_results = {}
    start = time.time()

    # ================================================================
    # 1. Main FP convergence test
    # ================================================================
    print(f"\n{'='*60}")
    print(f"1. FP convergence ({args.subnet_type}, damping=0.5)")
    print(f"{'='*60}")

    torch.manual_seed(42); np.random.seed(42)
    config = Config.from_json("configs/lob_d2_mv.json")
    config.eqn.law_encoder_type = "moments"
    config.eqn.subnet_type = args.subnet_type
    config.eqn.phi = 0.1
    config.net.opt_config1.num_iterations = inner
    config.net.logging_frequency = inner
    config.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_mv"](config.eqn)

    fp = FictitiousPlaySolver(
        config, bsde, device=device,
        outer_iterations=outer,
        inner_iterations=inner,
        w2_threshold=0.01,
        damping_alpha=args.damping or 0.5,
        n_sim_agents=256,
        warm_start=True,
    )
    result = fp.train()
    all_results["main"] = result

    # ================================================================
    # 2. Damping sweep (if not quick)
    # ================================================================
    if not args.quick:
        print(f"\n{'='*60}")
        print("2. Damping sweep")
        print(f"{'='*60}")

        for alpha in [0.1, 0.3, 0.7, 1.0]:
            print(f"\n--- damping={alpha} ---")
            torch.manual_seed(42); np.random.seed(42)
            bsde2 = EQUATION_REGISTRY["contxiong_lob_mv"](config.eqn)
            fp2 = FictitiousPlaySolver(
                config, bsde2, device=device,
                outer_iterations=outer,
                inner_iterations=inner,
                w2_threshold=0.01,
                damping_alpha=alpha,
                n_sim_agents=256,
                warm_start=True,
            )
            r2 = fp2.train()
            all_results[f"damping_{alpha}"] = r2

    # ================================================================
    # Save
    # ================================================================
    elapsed = time.time() - start
    all_results["metadata"] = {
        "outer": outer, "inner": inner,
        "subnet_type": args.subnet_type,
        "elapsed_seconds": elapsed,
        "device": str(device),
    }

    out = f"results_fp/fp_{args.subnet_type}.json"
    with open(out, "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min. Saved: {out}")

    # Summary
    if result["history"]:
        w2s = [h["w2"] for h in result["history"]]
        print(f"W2 trajectory: {' -> '.join(f'{w:.4f}' for w in w2s)}")
        print(f"Converged: {result['converged']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

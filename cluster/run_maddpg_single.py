#!/usr/bin/env python
"""
Single MADDPG round. Called by SLURM array job.
Usage: python run_maddpg_single.py --seed 0 --outdir results_cluster
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_multiagent import MADDPGTrainer
from scripts.cont_xiong_exact import fictitious_play


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outdir", default="results_cluster")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.outdir, exist_ok=True)

    torch.manual_seed(args.seed * 7 + 13)
    np.random.seed(args.seed * 7 + 13)

    print(f"Seed={args.seed}, device={device}")

    # Nash reference
    nash = fictitious_play(N=2, Q=5, max_iter=50)
    mid = len(nash["q_grid"]) // 2
    nash_spread = nash["delta_a"][mid] + nash["delta_b"][mid]

    # Train
    trainer = MADDPGTrainer(
        N=2, Q=5, device=device,
        lr_actor=1e-4, lr_critic=1e-3, tau=0.001,
        n_episodes=args.episodes, steps_per_episode=args.steps,
        batch_size=32,
    )
    r = trainer.train()

    # Episode-averaged spread (last 5 logged points)
    last_spreads = [h["avg_spread"] for h in r["history"][-5:]]
    avg_spread = float(np.mean(last_spreads)) if last_spreads else r["avg_final_spread"]

    result = {
        "seed": args.seed,
        "final_spread": r["avg_final_spread"],
        "avg_spread": avg_spread,
        "nash_spread": nash_spread,
        "above_nash": r["avg_final_spread"] > nash_spread,
        "history": r["history"],
        "elapsed": r["elapsed"],
    }

    outfile = os.path.join(args.outdir, f"maddpg_seed{args.seed}.json")
    with open(outfile, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"Saved {outfile}")
    print(f"Final spread: {r['avg_final_spread']:.4f}, Nash: {nash_spread:.4f}, "
          f"Above: {r['avg_final_spread'] > nash_spread}")


if __name__ == "__main__":
    main()

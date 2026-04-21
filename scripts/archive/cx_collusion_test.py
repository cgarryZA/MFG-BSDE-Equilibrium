#!/usr/bin/env python
"""
Tacit collusion test: multi-agent decentralised learning.

1. Compute Nash equilibrium (ground truth)
2. Train N=2 agents with full market simulation
3. Compare learned spreads to Nash
4. If spread > Nash → tacit collusion detected

Reproduces the main finding of CX Section 6.
"""

import gc, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_multiagent import MADDPGTrainer
from scripts.cont_xiong_exact import fictitious_play as exact_fp


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("results_cx_collusion", exist_ok=True)

    # Ground truth
    print("=== Nash Equilibrium (ground truth) ===")
    exact = exact_fp(N=2, Q=5, max_iter=50)
    mid = len(exact["q_grid"]) // 2
    nash_spread = exact["delta_a"][mid] + exact["delta_b"][mid]
    print(f"  Nash spread(0) = {nash_spread:.4f}")

    # Monopolist benchmark
    mono = exact_fp(N=1, Q=5, max_iter=50)
    mono_mid = len(mono["q_grid"]) // 2
    mono_spread = mono["delta_a"][mono_mid] + mono["delta_b"][mono_mid]
    print(f"  Monopolist spread(0) = {mono_spread:.4f}")

    # Multi-agent training (multiple rounds for statistics)
    print("\n=== Multi-agent decentralised training ===")
    all_spreads = []
    n_rounds = 5

    for round_idx in range(n_rounds):
        print(f"\n--- Round {round_idx+1}/{n_rounds} ---")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        torch.manual_seed(round_idx)
        np.random.seed(round_idx)

        trainer = MADDPGTrainer(
            N=2, Q=5, device=device,
            lr_actor=1e-3, lr_critic=1e-3, tau=0.01,
            n_episodes=500, steps_per_episode=500,
            batch_size=32, buffer_size=10000,
        )
        result = trainer.train()
        avg_spread = result["avg_final_spread"]
        all_spreads.append(avg_spread)
        print(f"  Final avg spread: {avg_spread:.4f}")

    mean_spread = np.mean(all_spreads)
    std_spread = np.std(all_spreads)
    collusion_gap = mean_spread - nash_spread

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Nash equilibrium spread:     {nash_spread:.4f}")
    print(f"  Monopolist spread:           {mono_spread:.4f}")
    print(f"  Learned spread (mean±std):   {mean_spread:.4f} ± {std_spread:.4f}")
    print(f"  Collusion gap:               {collusion_gap:+.4f}")
    print(f"  Collusion detected:          {'YES' if collusion_gap > 0.01 else 'NO'}")
    print(f"  Individual rounds:           {[f'{s:.4f}' for s in all_spreads]}")

    results = {
        "nash_spread": nash_spread,
        "monopolist_spread": mono_spread,
        "learned_spreads": all_spreads,
        "mean_learned_spread": mean_spread,
        "std_learned_spread": std_spread,
        "collusion_gap": collusion_gap,
        "n_rounds": n_rounds,
    }
    with open("results_cx_collusion/collusion_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved results_cx_collusion/collusion_results.json")


if __name__ == "__main__":
    main()

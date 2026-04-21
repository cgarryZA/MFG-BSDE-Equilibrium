#!/usr/bin/env python
"""
Law sensitivity test: does the policy change when the population
distribution shape changes at fixed mean?

This is the critical test for genuine distribution dependence.
If the policy is identical for narrow vs wide inventory distributions
(same mean), the law encoder is not doing real work.

Method:
1. Train a MV model with DeepSets encoder
2. Fix mean(q) = 0
3. Vary the population's inventory variance from 0.1 to 5.0
4. Evaluate the trained model's Z output at each distribution
5. If Z changes → law dependence is real

Usage:
    python scripts/law_sensitivity_test.py --weights logs/mv_model.pt
    python scripts/law_sensitivity_test.py --train --device cuda
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from registry import EQUATION_REGISTRY
import equations
from solver import ContXiongLOBMVModel, ContXiongLOBMVSolver


def generate_population(n_agents, mean_q, std_q, s_init=100.0):
    """Generate a synthetic population with specified inventory distribution."""
    S = np.full(n_agents, s_init)
    q = np.random.normal(mean_q, std_q, n_agents)
    q = np.clip(q, -10, 10)
    return np.stack([S, q], axis=1)  # [n_agents, 2]


def evaluate_policy_at_distribution(model, bsde, mean_q, std_q, n_agents=256, device="cpu"):
    """Evaluate the trained policy at a population with given distribution.

    Returns Z values for a representative agent at q=0.
    """
    model.eval()
    device = next(model.parameters()).device

    # Generate population
    pop = generate_population(n_agents, mean_q, std_q)
    particles = torch.tensor(pop, dtype=torch.float64, device=device)

    # Compute law embedding
    law_embed = model.law_encoder.encode(particles)  # [embed_dim]
    law_embed_batch = law_embed.unsqueeze(0)  # [1, embed_dim]

    # Evaluate subnet for a single agent at q=0
    agent_state = torch.tensor([[100.0, 0.0]], dtype=torch.float64, device=device)  # [1, 2]
    subnet_input = torch.cat([agent_state, law_embed_batch], dim=1)  # [1, 2+embed_dim]

    with torch.no_grad():
        z = model.subnet[0](subnet_input) / bsde.dim

    return {
        "z_s": z[0, 0].item(),
        "z_q": z[0, 1].item(),
        "law_embed": law_embed.detach().cpu().numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lob_d2_mv.json")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--train", action="store_true", help="Train a model first")
    parser.add_argument("--n_iters", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_dir", default="plots/law_sensitivity")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")

    config = Config.from_json(args.config)
    config.eqn.law_encoder_type = "deepsets"
    bsde = EQUATION_REGISTRY["contxiong_lob_mv"](config.eqn)

    if args.train:
        print(f"Training MV model ({args.n_iters} iter)...")
        config.net.opt_config1.num_iterations = args.n_iters
        config.net.logging_frequency = args.n_iters // 5
        solver = ContXiongLOBMVSolver(config, bsde, device=device)
        result = solver.train()
        model = solver.model
        print(f"Y0={result['y0']:.4f}, loss={result['final_loss']:.4e}")
    elif args.weights and os.path.exists(args.weights):
        model = ContXiongLOBMVModel(config, bsde, device=device)
        model.to(device)
        ckpt = torch.load(args.weights, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded: {args.weights}")
    else:
        print("No weights. Use --train or --weights. Training with defaults...")
        config.net.opt_config1.num_iterations = 1000
        config.net.logging_frequency = 500
        solver = ContXiongLOBMVSolver(config, bsde, device=device)
        result = solver.train()
        model = solver.model

    # Sweep over population variance at fixed mean=0
    print("\n=== Law Sensitivity Test ===")
    print("Fixed mean(q) = 0, varying std(q)")
    print(f"{'std_q':>8} {'Z^S':>10} {'Z^q':>10} {'|embed|':>10}")
    print("-" * 42)

    stds = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    results = []
    for std_q in stds:
        r = evaluate_policy_at_distribution(model, bsde, mean_q=0.0, std_q=std_q, device=device)
        embed_norm = np.linalg.norm(r["law_embed"])
        print(f"{std_q:8.1f} {r['z_s']:10.6f} {r['z_q']:10.6f} {embed_norm:10.4f}")
        results.append({"std_q": std_q, **r, "embed_norm": embed_norm})

    # Also test: same variance, different mean
    print("\n=== Mean Sensitivity (control) ===")
    print("Fixed std(q) = 1.0, varying mean(q)")
    print(f"{'mean_q':>8} {'Z^S':>10} {'Z^q':>10}")
    print("-" * 30)

    mean_results = []
    for mean_q in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        r = evaluate_policy_at_distribution(model, bsde, mean_q=mean_q, std_q=1.0, device=device)
        print(f"{mean_q:8.1f} {r['z_s']:10.6f} {r['z_q']:10.6f}")
        mean_results.append({"mean_q": mean_q, **r})

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Z^q vs std(q) at fixed mean=0
    z_qs = [r["z_q"] for r in results]
    axes[0].plot(stds, z_qs, "o-", color="#2196F3", linewidth=2, markersize=8)
    axes[0].set_xlabel("Population std($q$)", fontsize=11)
    axes[0].set_ylabel("$Z^q$ at $q=0$", fontsize=11)
    axes[0].set_title("Z response to distribution width\n(fixed mean=0)", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Embedding norm vs std(q)
    norms = [r["embed_norm"] for r in results]
    axes[1].plot(stds, norms, "o-", color="#FF9800", linewidth=2, markersize=8)
    axes[1].set_xlabel("Population std($q$)", fontsize=11)
    axes[1].set_ylabel("$\\|\\Phi(\\mu)\\|$", fontsize=11)
    axes[1].set_title("Law embedding norm vs width", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Z^q vs mean(q) at fixed std=1
    means = [r["mean_q"] for r in mean_results]
    z_qs_mean = [r["z_q"] for r in mean_results]
    axes[2].plot(means, z_qs_mean, "o-", color="#4CAF50", linewidth=2, markersize=8)
    axes[2].set_xlabel("Population mean($q$)", fontsize=11)
    axes[2].set_ylabel("$Z^q$ at $q=0$", fontsize=11)
    axes[2].set_title("Z response to population mean\n(fixed std=1)", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "law_sensitivity.png"), dpi=150)
    plt.close()
    print(f"\nSaved {args.out_dir}/law_sensitivity.png")

    # Verdict
    z_range = max(z_qs) - min(z_qs)
    print(f"\nZ^q range across std sweep: {z_range:.6f}")
    if z_range > 0.001:
        print("=> LAW SENSITIVITY DETECTED: Z changes with distribution shape")
    else:
        print("=> NO LAW SENSITIVITY: Z is insensitive to distribution shape")


if __name__ == "__main__":
    main()

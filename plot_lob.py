#!/usr/bin/env python
"""
Visualization and diagnostics for the Cont-Xiong LOB solver.

Usage:
    python plot_lob.py --config configs/lob_d2.json [--weights logs/lob_d2_model.pt]

Generates:
    1. Training convergence: loss vs iterations
    2. Optimal quoting strategy: spread vs inventory
    3. Sample forward paths: price and inventory trajectories
    4. Value function surface: V(q) at t=0
    5. Mean-field diagnostics: mu_t evolution across fictitious play iterations
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from registry import EQUATION_REGISTRY
import equations
from solver import ContXiongLOBModel


def plot_training_history(result_path, out_dir):
    """Plot loss convergence from training result file."""
    data = np.loadtxt(result_path, delimiter=",", skiprows=1)
    steps, loss, y0, elapsed = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].semilogy(steps, loss, "b-", linewidth=0.8)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Training Convergence")
    axes[0].grid(True, alpha=0.3)

    # Y0 evolution
    axes[1].plot(steps, y0, "r-", linewidth=0.8)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("$Y_0$ (value function at $t=0$)")
    axes[1].set_title("Initial Value Function")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence.png"), dpi=150)
    plt.close()
    print("Saved convergence.png")


def plot_quoting_strategy(config, bsde, model, out_dir):
    """Plot optimal bid-ask spread as a function of inventory.

    At q=0, spread should be 2/alpha (Avellaneda-Stoikov).
    At large |q|, spread should widen (risk aversion from inventory penalty).
    """
    model.eval()
    q_range = np.linspace(-5, 5, 200)
    s_val = bsde.s_init

    spreads = []
    delta_a_list = []
    delta_b_list = []

    for q in q_range:
        x = torch.tensor([[s_val, q]], dtype=torch.float64)
        # Get Z from the initial z_init (t=0 approximation)
        z = model.z_init.detach()  # [1, 2]
        z_q = z[0, 1].item()

        # In a trained model, Z depends on (t, x). Use z_init as t=0 proxy.
        # Better: use subnet[0] to predict Z at t=0+ given state
        if len(model.subnet) > 0:
            model.subnet[0].eval()
            with torch.no_grad():
                z_pred = model.subnet[0](x) / bsde.dim
            z_q = z_pred[0, 1].item()

        delta_a = 1.0 / bsde.alpha + z_q
        delta_b = 1.0 / bsde.alpha - z_q
        delta_a_list.append(delta_a)
        delta_b_list.append(delta_b)
        spreads.append(delta_a + delta_b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Total spread vs inventory
    equilibrium = 2.0 / bsde.alpha
    axes[0].plot(q_range, spreads, "b-", linewidth=1.5, label="Learned spread")
    axes[0].axhline(y=equilibrium, color="r", linestyle="--", alpha=0.7,
                    label=f"A-S equilibrium $2/\\alpha = {equilibrium:.3f}$")
    axes[0].set_xlabel("Inventory $q$")
    axes[0].set_ylabel("Bid-ask spread $\\delta^a + \\delta^b$")
    axes[0].set_title("Optimal Spread vs Inventory")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Individual quotes
    axes[1].plot(q_range, delta_a_list, "g-", linewidth=1.2, label="Ask half-spread $\\delta^a$")
    axes[1].plot(q_range, delta_b_list, "r-", linewidth=1.2, label="Bid half-spread $\\delta^b$")
    axes[1].axhline(y=1.0 / bsde.alpha, color="gray", linestyle=":", alpha=0.5,
                    label=f"$1/\\alpha = {1.0/bsde.alpha:.3f}$")
    axes[1].set_xlabel("Inventory $q$")
    axes[1].set_ylabel("Half-spread")
    axes[1].set_title("Optimal Quotes: Ask and Bid")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "quoting_strategy.png"), dpi=150)
    plt.close()
    print("Saved quoting_strategy.png")


def plot_sample_paths(bsde, out_dir, num_paths=20):
    """Plot sample forward paths of price and inventory."""
    dw, x = bsde.sample(num_paths)
    t_grid = bsde.t_grid

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Price paths
    for i in range(num_paths):
        axes[0].plot(t_grid, x[i, 0, :], alpha=0.4, linewidth=0.5)
    axes[0].set_xlabel("Time $t$")
    axes[0].set_ylabel("Mid-price $S_t$")
    axes[0].set_title(f"Price Paths ($\\sigma = {bsde.sigma_s}$)")
    axes[0].grid(True, alpha=0.3)

    # Inventory paths
    for i in range(num_paths):
        axes[1].plot(t_grid, x[i, 1, :], alpha=0.4, linewidth=0.5)
    axes[1].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[1].set_xlabel("Time $t$")
    axes[1].set_ylabel("Inventory $q_t$")
    axes[1].set_title(f"Inventory Paths ($\\phi = {bsde.phi}$)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sample_paths.png"), dpi=150)
    plt.close()
    print("Saved sample_paths.png")


def plot_inventory_distribution(bsde, out_dir, num_sample=2000):
    """Plot terminal inventory distribution — should be centered near 0."""
    dw, x = bsde.sample(num_sample)
    q_terminal = x[:, 1, -1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(q_terminal, bins=50, density=True, alpha=0.7, color="steelblue",
            edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="r", linestyle="--", alpha=0.7, label="$q=0$")
    ax.set_xlabel("Terminal inventory $q_T$")
    ax.set_ylabel("Density")
    ax.set_title(f"Terminal Inventory Distribution ($N={num_sample}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "inventory_distribution.png"), dpi=150)
    plt.close()
    print("Saved inventory_distribution.png")
    print(f"  Mean q_T = {np.mean(q_terminal):.4f}, Std = {np.std(q_terminal):.4f}")


def plot_value_function(config, bsde, model, out_dir):
    """Plot V(q) at t=0 for a range of inventories."""
    model.eval()
    q_range = np.linspace(-5, 5, 100)
    values = []

    for q in q_range:
        # Approximate V(q) using the terminal condition and Y_0
        # In a fully trained model, Y_0 + integral of generator gives V(q)
        # Simple proxy: -phi * q^2 (the terminal penalty shape)
        values.append(-bsde.phi * q ** 2)

    # If model is trained, Y_0 gives V at q=0
    y0 = model.y_init.item()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(q_range, values, "b-", linewidth=1.5, label="Terminal penalty $-\\phi q^2$")
    ax.plot(0, y0, "ro", markersize=8, label=f"Learned $Y_0 = {y0:.4f}$")
    ax.set_xlabel("Inventory $q$")
    ax.set_ylabel("Value $V(q)$")
    ax.set_title("Value Function Shape")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "value_function.png"), dpi=150)
    plt.close()
    print("Saved value_function.png")


def main():
    parser = argparse.ArgumentParser(description="LOB solver visualization")
    parser.add_argument("--config", type=str, default="configs/lob_d2.json")
    parser.add_argument("--result", type=str, default=None,
                        help="Path to training result .txt file")
    parser.add_argument("--out_dir", type=str, default="./plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = Config.from_json(args.config)
    torch.set_default_dtype(torch.float64)

    bsde = EQUATION_REGISTRY["contxiong_lob"](config.eqn)
    model = ContXiongLOBModel(config, bsde)

    # Plot training history if result file exists
    if args.result and os.path.exists(args.result):
        plot_training_history(args.result, args.out_dir)

    # Plot sample paths and distributions (no trained model needed)
    plot_sample_paths(bsde, args.out_dir)
    plot_inventory_distribution(bsde, args.out_dir)

    # Plot quoting strategy and value function (uses model, even untrained)
    plot_quoting_strategy(config, bsde, model, args.out_dir)
    plot_value_function(config, bsde, model, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

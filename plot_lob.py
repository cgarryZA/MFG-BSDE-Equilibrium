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


def plot_spread_heatmap(config, bsde, model, out_dir):
    """Heatmap: optimal spread as function of (time, inventory).

    Uses trained subnets at each time step to evaluate Z^q(t, S, q),
    then derives spread = 2/alpha + 2*Z^q (approximately, for small Z^q).
    Requires trained model weights.
    """
    model.eval()
    n_t = min(bsde.num_time_interval - 1, len(model.subnet))
    n_q = 80
    t_indices = np.linspace(0, n_t - 1, min(n_t, 40), dtype=int)
    q_range = np.linspace(-4, 4, n_q)

    spread_grid = np.zeros((len(t_indices), n_q))

    for i, t_idx in enumerate(t_indices):
        for j, q in enumerate(q_range):
            x = torch.tensor([[bsde.s_init, q]], dtype=torch.float64)
            with torch.no_grad():
                z = model.subnet[t_idx](x) / bsde.dim
            z_q = z[0, 1].item()
            delta_a = 1.0 / bsde.alpha + z_q
            delta_b = 1.0 / bsde.alpha - z_q
            spread_grid[i, j] = delta_a + delta_b

    t_vals = np.array([bsde.t_grid[idx] for idx in t_indices])

    fig, ax = plt.subplots(figsize=(8, 5))
    # Use tight vmin/vmax centered on equilibrium to show actual variation
    equilibrium = 2.0 / bsde.alpha
    spread_min = np.min(spread_grid)
    spread_max = np.max(spread_grid)
    # Symmetric deviation from equilibrium for colorbar
    dev = max(abs(spread_max - equilibrium), abs(spread_min - equilibrium), 0.01)
    im = ax.imshow(
        spread_grid.T, aspect="auto", origin="lower",
        extent=[t_vals[0], t_vals[-1], q_range[0], q_range[-1]],
        cmap="RdYlBu_r", vmin=equilibrium - dev, vmax=equilibrium + dev,
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Bid-ask spread")
    ax.axhline(y=0, color="white", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Inventory $q$")
    ax.set_title("Optimal Spread Surface (trained model)")
    equilibrium = 2.0 / bsde.alpha
    ax.set_title(f"Optimal Spread Surface (equilibrium = {equilibrium:.3f})")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spread_heatmap.png"), dpi=150)
    plt.close()
    print("Saved spread_heatmap.png")


def plot_value_function(config, bsde, model, out_dir):
    """Plot V(q) at t=0 by running the full BSDE forward pass at each q.

    For each initial inventory q, we set Y_0 = model.y_init (same for all q),
    simulate the forward SDE, and propagate Y through the BSDE. The resulting
    Y_T should match g(X_T). The quality of match tells us V(q).

    Simpler approach: use the subnet at t=0 to get Z(0, S, q), then compute
    the implied value via the HJB: rV = profits(q) - psi(q).
    """
    model.eval()
    q_range = np.linspace(-5, 5, 100)
    y0 = model.y_init.item()

    # Method: at stationarity, rV(q) = profits(q) - psi(q)
    # profits(q) depends on optimal quotes which depend on Z^q(q)
    values_hjb = []
    values_terminal = []

    for q in q_range:
        x = torch.tensor([[bsde.s_init, q]], dtype=torch.float64)

        # Get Z from first subnet (t ≈ 0)
        if len(model.subnet) > 0:
            with torch.no_grad():
                z = model.subnet[0](x) / bsde.dim
            z_q = z[0, 1].item()
        else:
            z_q = 0.0

        # Optimal quotes
        delta_a = 1.0 / bsde.alpha + z_q
        delta_b = 1.0 / bsde.alpha - z_q

        # Execution rates and profits
        f_a = np.exp(-bsde.alpha * delta_a) * bsde.lambda_a
        f_b = np.exp(-bsde.alpha * delta_b) * bsde.lambda_b
        profits = f_a * delta_a + f_b * delta_b
        psi = bsde.phi * q ** 2

        # HJB: rV = profits - psi → V = (profits - psi) / r
        v_hjb = (profits - psi) / bsde.discount_rate
        values_hjb.append(v_hjb)
        values_terminal.append(-psi)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(q_range, values_hjb, "b-", linewidth=2.0,
            label="$V(q) = (\\mathrm{profits} - \\psi) / r$ (HJB)")
    ax.plot(q_range, values_terminal, "k--", linewidth=1.0, alpha=0.5,
            label="Terminal penalty $-\\phi q^2$")
    ax.plot(0, y0, "ro", markersize=10, zorder=5,
            label=f"Learned $Y_0 = {y0:.4f}$")

    # Mark the theoretical V(0) = profits(0) / r
    f_eq = np.exp(-1.0) * bsde.lambda_a  # f(1/alpha) at q=0
    v0_theory = 2 * f_eq * (1.0 / bsde.alpha) / bsde.discount_rate
    ax.plot(0, v0_theory, "g^", markersize=10, zorder=5,
            label=f"$V(0)$ theory $= {v0_theory:.4f}$")

    ax.set_xlabel("Inventory $q$", fontsize=12)
    ax.set_ylabel("Value $V(q)$", fontsize=12)
    ax.set_title("Value Function: Learned vs HJB", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "value_function.png"), dpi=150)
    plt.close()
    print("Saved value_function.png")


def plot_value_surface_3d(config, bsde, model, out_dir):
    """3D surface: Value function V(t, q) over (time, inventory).

    Uses subnets to propagate Y from Y_0 through the BSDE dynamics
    at different inventory levels. Requires trained weights.
    """
    from mpl_toolkits.mplot3d import Axes3D

    model.eval()
    n_t = min(bsde.num_time_interval - 1, len(model.subnet))
    t_indices = np.linspace(0, n_t - 1, min(n_t, 30), dtype=int)
    q_range = np.linspace(-4, 4, 60)

    # For each (t, q), evaluate the subnet Z and compute the implied value
    # V(t,q) ≈ Y_0 - sum of f*dt from 0 to t (rough approximation)
    # Better: use the terminal condition and Z to infer the value
    value_grid = np.zeros((len(t_indices), len(q_range)))

    for i, t_idx in enumerate(t_indices):
        for j, q in enumerate(q_range):
            x = torch.tensor([[bsde.s_init, q]], dtype=torch.float64)
            with torch.no_grad():
                z = model.subnet[t_idx](x) / bsde.dim
            z_q = z[0, 1].item()
            # Value proxy: terminal penalty discounted back + spread revenue
            t_remain = bsde.total_time - bsde.t_grid[t_idx]
            terminal_penalty = -bsde.phi * q ** 2
            # Revenue from optimal quoting at this inventory
            delta_a = 1.0 / bsde.alpha + z_q
            delta_b = 1.0 / bsde.alpha - z_q
            f_a = np.exp(-bsde.alpha * delta_a) * bsde.lambda_a
            f_b = np.exp(-bsde.alpha * delta_b) * bsde.lambda_b
            revenue_rate = f_a * delta_a + f_b * delta_b - bsde.phi * q ** 2
            value_grid[i, j] = terminal_penalty * np.exp(-bsde.discount_rate * t_remain) + \
                               revenue_rate * t_remain

    t_vals = np.array([bsde.t_grid[idx] for idx in t_indices])
    T, Q = np.meshgrid(t_vals, q_range)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(T, Q, value_grid.T, cmap="coolwarm", alpha=0.85,
                    edgecolor="none", antialiased=True)
    ax.set_xlabel("Time $t$", fontsize=11)
    ax.set_ylabel("Inventory $q$", fontsize=11)
    ax.set_zlabel("Value $V(t, q)$", fontsize=11)
    ax.set_title("Value Function Surface", fontsize=13)
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "value_surface_3d.png"), dpi=150)
    plt.close()
    print("Saved value_surface_3d.png")


def plot_z_gradient_surface_3d(config, bsde, model, out_dir):
    """3D surface: Z^q(t, q) — the inventory gradient driving optimal quotes.

    This is THE key object: Z^q determines δ^a = 1/α + Z^q, δ^b = 1/α - Z^q.
    Should be roughly linear in q (from quadratic penalty), with deviations
    showing the neural network's learned correction.
    """
    from mpl_toolkits.mplot3d import Axes3D

    model.eval()
    n_t = min(bsde.num_time_interval - 1, len(model.subnet))
    t_indices = np.linspace(0, n_t - 1, min(n_t, 30), dtype=int)
    q_range = np.linspace(-4, 4, 60)

    z_grid = np.zeros((len(t_indices), len(q_range)))

    for i, t_idx in enumerate(t_indices):
        for j, q in enumerate(q_range):
            x = torch.tensor([[bsde.s_init, q]], dtype=torch.float64)
            with torch.no_grad():
                z = model.subnet[t_idx](x) / bsde.dim
            z_grid[i, j] = z[0, 1].item()  # Z^q component

    t_vals = np.array([bsde.t_grid[idx] for idx in t_indices])
    T, Q = np.meshgrid(t_vals, q_range)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(T, Q, z_grid.T, cmap="RdBu", alpha=0.85,
                    edgecolor="none", antialiased=True)
    ax.set_xlabel("Time $t$", fontsize=11)
    ax.set_ylabel("Inventory $q$", fontsize=11)
    ax.set_zlabel("$Z_t^q$ (inventory gradient)", fontsize=11)
    ax.set_title("Learned Gradient Surface $Z_t^q(t, q)$", fontsize=13)
    ax.view_init(elev=25, azim=-60)

    # Add the zero plane for reference
    ax.plot_surface(T, Q, np.zeros_like(z_grid.T), alpha=0.1, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "z_gradient_surface_3d.png"), dpi=150)
    plt.close()
    print("Saved z_gradient_surface_3d.png")


def plot_z_max_evolution(result_path, out_dir):
    """Plot max|Z_t| over training — the Lipschitz stability diagnostic."""
    data = np.loadtxt(result_path, delimiter=",", skiprows=1)
    if data.shape[1] < 5:
        print("No z_max data in result file (old format), skipping")
        return
    steps, z_max = data[:, 0], data[:, 3]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, z_max, "r-", linewidth=1.0)
    ax.set_xlabel("Training step")
    ax.set_ylabel("max $|Z_t|$")
    ax.set_title("Gradient Stability: max $|Z_t|$ During Training")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "z_max_evolution.png"), dpi=150)
    plt.close()
    print("Saved z_max_evolution.png")


def main():
    parser = argparse.ArgumentParser(description="LOB solver visualization")
    parser.add_argument("--config", type=str, default="configs/lob_d2.json")
    parser.add_argument("--result", type=str, default=None,
                        help="Path to training result .txt file")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to saved model .pt file")
    parser.add_argument("--out_dir", type=str, default="./plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = Config.from_json(args.config)
    torch.set_default_dtype(torch.float64)

    bsde = EQUATION_REGISTRY["contxiong_lob"](config.eqn)
    model = ContXiongLOBModel(config, bsde)

    # Load trained weights if available
    if args.weights and os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        print(f"Loaded weights from {args.weights}")
        print(f"  Y0 = {checkpoint['y0']:.6f}, loss = {checkpoint['final_loss']:.6e}")
    elif args.weights:
        print(f"Warning: weights file {args.weights} not found, using random init")

    # Plot training history if result file exists
    if args.result and os.path.exists(args.result):
        plot_training_history(args.result, args.out_dir)

    # Plot sample paths and distributions (no trained model needed)
    plot_sample_paths(bsde, args.out_dir)
    plot_inventory_distribution(bsde, args.out_dir)

    # Plot quoting strategy and value function (uses model, even untrained)
    plot_quoting_strategy(config, bsde, model, args.out_dir)
    plot_value_function(config, bsde, model, args.out_dir)

    # Plots requiring trained weights
    if args.weights and os.path.exists(args.weights):
        plot_spread_heatmap(config, bsde, model, args.out_dir)
        plot_value_surface_3d(config, bsde, model, args.out_dir)
        plot_z_gradient_surface_3d(config, bsde, model, args.out_dir)

    # Z_t evolution (from training history)
    if args.result and os.path.exists(args.result):
        plot_z_max_evolution(args.result, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

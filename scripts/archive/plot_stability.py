#!/usr/bin/env python
"""
Plot stability frontier results as phase diagrams.

Usage:
    python scripts/plot_stability.py
    python scripts/plot_stability.py --results results_stability/stability_frontier.json
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_phi_penalty(results, out_dir):
    """1D sweep: loss and z_max vs phi for each penalty type."""
    data = results.get("phi_penalty_sweep", [])
    if not data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for penalty in ["quadratic", "exponential"]:
        pts = [d for d in data if d["penalty"] == penalty]
        if not pts:
            continue
        phis = [p["phi"] for p in pts]
        z_maxs = [p["z_max"] for p in pts]
        losses = [p["loss"] for p in pts]
        converged = [p["converged"] for p in pts]

        color = "#4CAF50" if penalty == "quadratic" else "#F44336"
        axes[0].semilogy(phis, losses, "o-", color=color, label=penalty, linewidth=1.5)
        for phi, loss, ok in zip(phis, losses, converged):
            if not ok:
                axes[0].plot(phi, loss, "x", color="black", markersize=10, zorder=5)

        axes[1].semilogy(phis, z_maxs, "o-", color=color, label=penalty, linewidth=1.5)
        for phi, zm, ok in zip(phis, z_maxs, converged):
            if not ok:
                axes[1].plot(phi, zm, "x", color="black", markersize=10, zorder=5)

    axes[0].set_xlabel("$\\phi$ (penalty coefficient)")
    axes[0].set_ylabel("Loss (log)")
    axes[0].set_title("Loss vs $\\phi$")
    axes[0].set_xscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("$\\phi$ (penalty coefficient)")
    axes[1].set_ylabel("max $|Z_t|$ (log)")
    axes[1].set_title("Gradient magnitude vs $\\phi$")
    axes[1].set_xscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stability_phi_penalty.png"), dpi=150)
    plt.close()
    print("Saved stability_phi_penalty.png")


def plot_phase_diagram(results, out_dir):
    """2D phase diagram: phi × eta, colored by convergence/z_max."""
    data = results.get("phase_diagram_phi_eta", [])
    if not data:
        return

    phis = sorted(set(d["phi"] for d in data))
    etas = sorted(set(d["eta"] for d in data))

    z_grid = np.full((len(phis), len(etas)), np.nan)
    conv_grid = np.full((len(phis), len(etas)), 0)

    for d in data:
        i = phis.index(d["phi"])
        j = etas.index(d["eta"])
        z_grid[i, j] = d["z_max"]
        conv_grid[i, j] = 1 if d["converged"] else 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Z_max heatmap
    im0 = axes[0].imshow(
        np.log10(z_grid.T + 1e-6), aspect="auto", origin="lower",
        extent=[np.log10(phis[0]), np.log10(phis[-1]), etas[0], etas[-1]],
        cmap="YlOrRd",
    )
    plt.colorbar(im0, ax=axes[0], label="$\\log_{10}(\\max|Z_t|)$")
    axes[0].set_xlabel("$\\log_{10}(\\phi)$")
    axes[0].set_ylabel("$\\eta$ (adverse selection)")
    axes[0].set_title("Gradient Magnitude Phase Diagram")

    # Convergence map
    im1 = axes[1].imshow(
        conv_grid.T, aspect="auto", origin="lower",
        extent=[np.log10(phis[0]), np.log10(phis[-1]), etas[0], etas[-1]],
        cmap="RdYlGn", vmin=0, vmax=1,
    )
    plt.colorbar(im1, ax=axes[1], label="Converged (1) / Diverged (0)")
    axes[1].set_xlabel("$\\log_{10}(\\phi)$")
    axes[1].set_ylabel("$\\eta$ (adverse selection)")
    axes[1].set_title("Convergence Phase Diagram")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase_diagram.png"), dpi=150)
    plt.close()
    print("Saved phase_diagram.png")


def plot_eta_sweep(results, out_dir):
    """Adverse selection strength sweep."""
    data = results.get("eta_sweep", [])
    if not data:
        return

    etas = [d["eta"] for d in data]
    y0s = [d["y0"] for d in data]
    z_maxs = [d["z_max"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(etas, y0s, "o-", color="#2196F3", linewidth=1.5)
    axes[0].set_xlabel("$\\eta$ (adverse selection)")
    axes[0].set_ylabel("$Y_0$")
    axes[0].set_title("Value vs Adverse Selection Strength")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(etas, z_maxs, "o-", color="#F44336", linewidth=1.5)
    axes[1].set_xlabel("$\\eta$ (adverse selection)")
    axes[1].set_ylabel("max $|Z_t|$ (log)")
    axes[1].set_title("Gradient Magnitude vs $\\eta$")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stability_eta.png"), dpi=150)
    plt.close()
    print("Saved stability_eta.png")


def plot_coupling_comparison(results, out_dir):
    """Coupling type comparison bar chart."""
    data = results.get("coupling_sweep_phi01", [])
    if not data:
        return

    labels = [d["coupling"] for d in data]
    losses = [d["loss"] for d in data]
    y0s = [d["y0"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#9E9E9E", "#FF9800", "#4CAF50"]

    axes[0].bar(labels, y0s, color=colors[:len(labels)], edgecolor="white")
    axes[0].set_ylabel("$Y_0$")
    axes[0].set_title("Value by Coupling Type ($\\phi=0.1$)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, losses, color=colors[:len(labels)], edgecolor="white")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss by Coupling Type ($\\phi=0.1$)")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "coupling_comparison.png"), dpi=150)
    plt.close()
    print("Saved coupling_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results_stability/stability_frontier.json")
    parser.add_argument("--out_dir", default="plots/stability")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.results):
        print(f"Results not found: {args.results}")
        print("Run: python scripts/stability_frontier.py")
        return

    results = load_results(args.results)
    print(f"Loaded: {args.results}")

    plot_phi_penalty(results, args.out_dir)
    plot_phase_diagram(results, args.out_dir)
    plot_eta_sweep(results, args.out_dir)
    plot_coupling_comparison(results, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

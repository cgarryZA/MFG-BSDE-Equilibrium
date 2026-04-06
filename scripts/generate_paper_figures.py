#!/usr/bin/env python
"""
Generate all paper figures from saved results + model weights.

Figures:
1. mv_control_sensitivity.png — quotes, skew, intensities vs inventory (3 panels)
2. encoder_comparison.png — h by encoder and population shape (bar chart)
3. regime_transition.png — h-gap and policy-gap vs phi (2 panels)
4. model_hierarchy.png — Y0 bar chart across model variants

Usage:
    python scripts/generate_paper_figures.py
"""

import json
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
from solver import ContXiongLOBMVModel

OUT_DIR = "plots"


def load_model(config_path, weights_path, device):
    """Load trained MV model from weights."""
    config = Config.from_json(config_path)
    config.eqn.law_encoder_type = "moments"
    bsde = EQUATION_REGISTRY["contxiong_lob_mv"](config.eqn)
    model = ContXiongLOBMVModel(config, bsde, device=device)
    model.to(device)
    ckpt = torch.load(weights_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if ckpt.get("competitive_factor_state"):
        bsde.competitive_factor_net.load_state_dict(ckpt["competitive_factor_state"])
    bsde.competitive_factor_net.to(device)
    model.eval()
    return model, bsde


def generate_population(n_agents, mean_q, std_q, s_init=100.0):
    S = np.full(n_agents, s_init)
    q = np.random.normal(mean_q, std_q, n_agents)
    q = np.clip(q, -10, 10)
    return np.stack([S, q], axis=1)


def evaluate_at_q(model, bsde, law_embed, q_val, device):
    """Evaluate quotes and intensities at a specific inventory q."""
    alpha = bsde.alpha
    with torch.no_grad():
        h = bsde.compute_competitive_factor(law_embed).item()
        agent = torch.tensor([[100.0, q_val]], dtype=torch.float64, device=device)
        leb = law_embed.unsqueeze(0)
        si = torch.cat([agent, leb], dim=1)
        z = model.subnet[0](si) / bsde.dim
        zq = z[:, 1:2]
        sig = bsde._sigma_q_equilibrium()
        p = zq / sig
        da = (1.0 / alpha + p).item()
        db = (1.0 / alpha - p).item()
        # Clip for execution rate calculation
        da_c = max(0.01, min(da, 10.0 / alpha))
        db_c = max(0.01, min(db, 10.0 / alpha))
        nu_a = np.exp(-alpha * da_c) * bsde.lambda_a * h
        nu_b = np.exp(-alpha * db_c) * bsde.lambda_b * h
    return da, db, nu_a, nu_b, h


# ================================================================
# Figure 1: mv_control_sensitivity (3-panel)
# ================================================================
def fig_control_sensitivity(model, bsde, device):
    print("Generating mv_control_sensitivity.png...")
    q_range = np.linspace(-5, 5, 50)
    pops = [
        ("Narrow (std=0.1)", 0.1, "#2196F3"),
        ("Medium (std=1.0)", 1.0, "#FF9800"),
        ("Wide (std=3.0)", 3.0, "#F44336"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for label, std_q, color in pops:
        pop = generate_population(256, 0.0, std_q)
        particles = torch.tensor(pop, dtype=torch.float64, device=device)
        law_embed = model.law_encoder.encode(particles)

        das, dbs, nuas, nubs = [], [], [], []
        for q in q_range:
            da, db, nua, nub, h = evaluate_at_q(model, bsde, law_embed, q, device)
            das.append(da)
            dbs.append(db)
            nuas.append(nua)
            nubs.append(nub)

        # Panel 1: Quotes
        axes[0].plot(q_range, das, "-", color=color, label=f"{label} ask", linewidth=1.5)
        axes[0].plot(q_range, dbs, "--", color=color, label=f"{label} bid", linewidth=1.5)

        # Panel 2: Skew
        skew = [a - b for a, b in zip(das, dbs)]
        axes[1].plot(q_range, skew, "-", color=color, label=label, linewidth=2)

        # Panel 3: Intensities
        axes[2].plot(q_range, nuas, "-", color=color, label=f"{label} ask", linewidth=1.5)
        axes[2].plot(q_range, nubs, "--", color=color, label=f"{label} bid", linewidth=1.5)

    axes[0].set_title("Optimal Quotes vs Inventory", fontsize=12)
    axes[0].set_xlabel("Inventory $q$")
    axes[0].set_ylabel("Quote")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Quote Skew vs Inventory", fontsize=12)
    axes[1].set_xlabel("Inventory $q$")
    axes[1].set_ylabel("Skew (ask $-$ bid)")
    axes[1].axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Intensities vs Inventory", fontsize=12)
    axes[2].set_xlabel("Inventory $q$")
    axes[2].set_ylabel("Execution rate")
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mv_control_sensitivity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ================================================================
# Figure 2: encoder_comparison (bar chart)
# ================================================================
def fig_encoder_comparison():
    print("Generating encoder_comparison.png...")
    # Data from experiment output
    encoders = ["MomentEnc", "QuantileEnc", "HistogramEnc", "DeepSets"]
    h_narrow = [0.417, 0.406, 0.314, 0.194]
    h_medium = [0.252, 0.204, 0.324, 0.195]
    h_wide   = [0.043, 0.037, 0.344, 0.195]

    x = np.arange(len(encoders))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars1 = ax.bar(x - width, h_narrow, width, label="Narrow (std=0.1)", color="#2196F3")
    bars2 = ax.bar(x, h_medium, width, label="Medium (std=1.0)", color="#FF9800")
    bars3 = ax.bar(x + width, h_wide, width, label="Wide (std=3.0)", color="#F44336")

    ax.set_ylabel("Competitive Factor $h$", fontsize=12)
    ax.set_xlabel("Law Encoder", fontsize=12)
    ax.set_title("Distribution Sensitivity by Encoder Type", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(encoders, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.55)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "encoder_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ================================================================
# Figure 3: regime_transition (2-panel: h-gap and policy-gap vs phi)
# ================================================================
def fig_regime_transition():
    print("Generating regime_transition.png...")
    with open("results_paper_final/remaining_results.json") as f:
        R = json.load(f)
    ps = R["penalty_sweep"]

    phis = [0.01, 0.05, 0.1, 0.5]
    h_gaps = [ps[f"phi={p}"]["h_gap"] for p in phis]
    pol_gaps = [ps[f"phi={p}"]["policy_gap"] for p in phis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(phis, h_gaps, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax1.set_xscale("log")
    ax1.set_xlabel("Penalty coefficient ($\\phi$)", fontsize=11)
    ax1.set_ylabel("$h$-gap (narrow $-$ wide)", fontsize=11)
    ax1.set_title("Law Sensitivity vs Penalty", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(phis, pol_gaps, "o-", color="#F44336", linewidth=2, markersize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Penalty coefficient ($\\phi$)", fontsize=11)
    ax2.set_ylabel("PolicyGap ($|\\delta^a_{narrow} - \\delta^a_{wide}|$)", fontsize=11)
    ax2.set_title("Policy Sensitivity vs Penalty", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "regime_transition.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ================================================================
# Figure 4: model_hierarchy (bar chart)
# ================================================================
def fig_model_hierarchy():
    print("Generating model_hierarchy.png...")
    models = ["Base\n(2D)", "MV only\n(2D)", "Adverse\n(3D)", "Adverse+MV\n(3D)"]
    y0s = [0.456, 0.142, 0.380, 0.086]
    colors = ["#9E9E9E", "#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, y0s, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, y0s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("$Y_0$ (value at $q=0$)", fontsize=12)
    ax.set_title("Model Hierarchy: Effect of MV Coupling and Adverse Selection", fontsize=13)
    ax.set_ylim(0, 0.55)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_hierarchy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ================================================================
# Figure 5: deepsets_ablation (unchanged — no new data needed)
# ================================================================
# The DeepSets ablation (sum vs mean pooling, q^2 features) was a
# standalone encoder-level test independent of h floor. Keep existing figure.


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load model for Figure 1
    model, bsde = load_model(
        "configs/lob_d2_mv.json",
        "results_paper_final/main_model.pt",
        device,
    )

    fig_control_sensitivity(model, bsde, device)
    fig_encoder_comparison()
    fig_regime_transition()
    fig_model_hierarchy()

    print("\nAll figures saved to plots/")


if __name__ == "__main__":
    main()

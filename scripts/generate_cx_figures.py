#!/usr/bin/env python
"""
Generate all dissertation figures for the CX model results.

1. Nash vs Pareto quote profiles (the competition bounds)
2. Scale test: exact vs neural across Q values
3. N-scaling: spread vs number of dealers
4. Fictitious play convergence animation (Algorithm 1)
5. Value function: discrete vs continuous neural solver
6. Spread convergence during multi-agent training (collusion detection)
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "plots_cx"
os.makedirs(OUT, exist_ok=True)


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ================================================================
# Figure 1: Nash vs Pareto quote profiles
# ================================================================
def fig_nash_vs_pareto():
    print("Generating nash_vs_pareto.png...")
    nash = load_json("results_cx_exact/nash_N2.json")
    pareto = load_json("results_cx_exact/pareto_N2.json")
    mono = load_json("results_cx_exact/monopolist.json")

    if not nash or not pareto:
        print("  SKIP: missing data")
        return

    q = nash["q_grid"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Ask quotes
    ax1.plot(q, nash["delta_a"], "b-o", label="Nash (competitive)", linewidth=2, markersize=5)
    ax1.plot(q, pareto["delta_a"], "r-s", label="Pareto (collusion)", linewidth=2, markersize=5)
    if mono:
        ax1.plot(q, mono["delta_a"], "k--", label="Monopolist", linewidth=1.5, alpha=0.5)
    ax1.set_xlabel("Inventory $q$", fontsize=12)
    ax1.set_ylabel("Ask quote $\\delta^a(q)$", fontsize=12)
    ax1.set_title("Ask Quotes", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Spreads
    nash_spread = [a + b for a, b in zip(nash["delta_a"], nash["delta_b"])]
    pareto_spread = pareto["spread"]
    ax2.plot(q, nash_spread, "b-o", label="Nash spread", linewidth=2, markersize=5)
    ax2.plot(q, pareto_spread, "r-s", label="Pareto spread", linewidth=2, markersize=5)
    if mono:
        mono_spread = [a + b for a, b in zip(mono["delta_a"], mono["delta_b"])]
        ax2.plot(q, mono_spread, "k--", label="Monopolist", linewidth=1.5, alpha=0.5)

    # Shade collusion zone
    ax2.fill_between(q, nash_spread, pareto_spread, alpha=0.15, color="red",
                     label="Collusion zone")
    ax2.set_xlabel("Inventory $q$", fontsize=12)
    ax2.set_ylabel("Total spread $\\delta^a + \\delta^b$", fontsize=12)
    ax2.set_title("Bid-Ask Spread", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "nash_vs_pareto.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ================================================================
# Figure 2: Neural solver validation (exact vs neural at each q)
# ================================================================
def fig_validation():
    print("Generating validation.png...")
    val = load_json("results_cx_validation/validation.json")
    if not val:
        val = load_json("results_cx_overnight/validation.json")
    if not val:
        print("  SKIP: missing data")
        return

    nash = load_json("results_cx_exact/nash_N2.json")
    if not nash:
        print("  SKIP: missing nash data")
        return

    q = nash["q_grid"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Ask quotes comparison
    ax1.plot(q, nash["delta_a"], "b-o", label="Exact (Algorithm 1)", linewidth=2, markersize=6)
    if "single_da" in val:
        ax1.plot(q, val["single_da"], "g-^", label="Neural single-pass", linewidth=2, markersize=5)
    if "fp_da" in val:
        ax1.plot(q, val["fp_da"], "r-s", label="Neural FP", linewidth=2, markersize=5)
    ax1.set_xlabel("Inventory $q$", fontsize=12)
    ax1.set_ylabel("Ask quote $\\delta^a(q)$", fontsize=12)
    ax1.set_title("Ask Quotes: Exact vs Neural", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Spread comparison
    nash_spread = [a + b for a, b in zip(nash["delta_a"], nash["delta_b"])]
    ax2.plot(q, nash_spread, "b-o", label="Exact", linewidth=2, markersize=6)
    ax2.axhline(y=nash_spread[len(q)//2], color="blue", linestyle=":", alpha=0.3)
    ax2.set_xlabel("Inventory $q$", fontsize=12)
    ax2.set_ylabel("Spread", fontsize=12)
    ax2.set_title(f"Spread at $q=0$: Exact={nash_spread[len(q)//2]:.4f}", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "validation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ================================================================
# Figure 3: N-scaling (spread vs number of dealers)
# ================================================================
def fig_n_scaling():
    print("Generating n_scaling.png...")
    scale = load_json("results_cx_overnight/scale_test.json")
    if not scale:
        print("  SKIP: missing data")
        return

    Ns = []
    exact_spreads = []
    nn_spreads = []
    for k in sorted(scale.keys(), key=lambda x: scale[x]["N"]):
        d = scale[k]
        Ns.append(d["N"])
        exact_spreads.append(d.get("exact_spread"))
        nn_spreads.append(d["nn_spread"])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(Ns, exact_spreads, "b-o", label="Exact (Algorithm 1)", linewidth=2, markersize=8)
    ax.plot(Ns, nn_spreads, "r-s", label="Neural solver", linewidth=2, markersize=8)
    ax.set_xlabel("Number of dealers $N$", fontsize=12)
    ax.set_ylabel("Spread at $q=0$", fontsize=12)
    ax.set_title("Nash Equilibrium Spread vs Number of Dealers", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(Ns)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "n_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ================================================================
# Figure 4: Q-scaling (neural solver across inventory limits)
# ================================================================
def fig_q_scaling():
    print("Generating q_scaling.png...")
    # Merge all Q scaling data
    data = {}
    for path in ["results_cx_q_scaling/q_scaling.json", "results_cx_q_scaling/q_scaling_extra.json"]:
        d = load_json(path)
        if d:
            data.update(d)

    if not data:
        print("  SKIP: missing data")
        return

    Qs = []
    exact_spreads = []
    nn_spreads = []
    nn_times = []
    for k in sorted(data.keys(), key=lambda x: data[x]["Q"]):
        d = data[k]
        Qs.append(d["Q"])
        exact_spreads.append(d.get("exact_spread"))
        nn_spreads.append(d["nn_spread"])
        nn_times.append(d.get("nn_time", 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Spread accuracy
    ax1.plot(Qs, exact_spreads, "b-o", label="Exact", linewidth=2, markersize=8)
    ax1.plot(Qs, nn_spreads, "r-s", label="Neural", linewidth=2, markersize=8)
    ax1.set_xlabel("Inventory limit $Q$", fontsize=12)
    ax1.set_ylabel("Spread at $q=0$", fontsize=12)
    ax1.set_title("Spread Accuracy vs Inventory Limit", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right: Training time
    ax2.bar(range(len(Qs)), nn_times, tick_label=[str(Q) for Q in Qs],
            color="#2196F3", alpha=0.7)
    ax2.set_xlabel("Inventory limit $Q$", fontsize=12)
    ax2.set_ylabel("Neural solver time (seconds)", fontsize=12)
    ax2.set_title("Training Time vs Problem Size", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "q_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ================================================================
# Figure 5: Fictitious play convergence
# ================================================================
def fig_fp_convergence():
    print("Generating fp_convergence.png...")
    nash = load_json("results_cx_exact/nash_N2.json")
    if not nash or "history" not in nash:
        print("  SKIP: missing data")
        return

    history = nash["history"]
    iters = [h["iter"] for h in history]
    spreads = [h["spread_q0"] for h in history]
    values = [h["V_q0"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.plot(iters, spreads, "b-o", linewidth=2, markersize=6)
    ax1.axhline(y=spreads[-1], color="blue", linestyle=":", alpha=0.5,
                label=f"Converged: {spreads[-1]:.4f}")
    ax1.set_xlabel("FP Iteration", fontsize=12)
    ax1.set_ylabel("Spread at $q=0$", fontsize=12)
    ax1.set_title("Fictitious Play: Spread Convergence", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, values, "r-o", linewidth=2, markersize=6)
    ax2.axhline(y=values[-1], color="red", linestyle=":", alpha=0.5,
                label=f"Converged: {values[-1]:.4f}")
    ax2.set_xlabel("FP Iteration", fontsize=12)
    ax2.set_ylabel("$V(0)$", fontsize=12)
    ax2.set_title("Fictitious Play: Value Convergence", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fp_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ================================================================
# Figure 6: Collusion detection (learned spread vs Nash vs Pareto)
# ================================================================
def fig_collusion():
    print("Generating collusion.png...")
    collusion = load_json("results_cx_collusion/collusion_results.json")
    overnight = load_json("results_cx_overnight/collusion.json")

    nash_spread = None
    pareto_spread = None
    mono_spread = None

    # Get reference spreads
    pareto = load_json("results_cx_exact/pareto_N2.json")
    nash = load_json("results_cx_exact/nash_N2.json")
    mono = load_json("results_cx_exact/monopolist.json")
    if nash:
        mid = len(nash["q_grid"]) // 2
        nash_spread = nash["delta_a"][mid] + nash["delta_b"][mid]
    if pareto:
        pareto_spread = pareto["spread_q0"]
    if mono:
        mid = len(mono["q_grid"]) // 2
        mono_spread = mono["delta_a"][mid] + mono["delta_b"][mid]

    if not nash_spread:
        print("  SKIP: missing reference data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Reference lines
    ax.axhline(y=nash_spread, color="blue", linewidth=2, label=f"Nash equilibrium ({nash_spread:.4f})")
    if pareto_spread:
        ax.axhline(y=pareto_spread, color="red", linewidth=2, linestyle="--",
                    label=f"Pareto optimum ({pareto_spread:.4f})")
    if mono_spread:
        ax.axhline(y=mono_spread, color="gray", linewidth=1.5, linestyle=":",
                    label=f"Monopolist ({mono_spread:.4f})")

    # Shade collusion zone
    if pareto_spread:
        ax.axhspan(nash_spread, pareto_spread, alpha=0.1, color="red")
        ax.text(0.02, (nash_spread + pareto_spread) / 2, "Collusion zone",
                transform=ax.get_yaxis_transform(), fontsize=9, color="red", alpha=0.7)

    # Overnight collusion results (our simple test)
    if overnight:
        labels = ["Full info", "Partial info", "No info"]
        values = [overnight.get("full_info"), overnight.get("partial_info"),
                  overnight.get("no_info_mono")]
        colors = ["#4CAF50", "#FF9800", "#F44336"]
        for i, (label, val, col) in enumerate(zip(labels, values, colors)):
            if val:
                ax.scatter([i + 1], [val], s=200, color=col, zorder=5, edgecolors="black")
                ax.annotate(f"{val:.4f}", (i + 1, val), textcoords="offset points",
                            xytext=(15, 0), fontsize=10)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, fontsize=11)

    # Multi-agent collusion results
    if collusion and "learned_spreads" in collusion:
        x_offset = 4
        for i, s in enumerate(collusion["learned_spreads"]):
            ax.scatter([x_offset + i * 0.3], [s], s=100, color="#9C27B0", zorder=5,
                       marker="D", edgecolors="black")
        mean_s = collusion["mean_learned_spread"]
        ax.axhline(y=mean_s, color="#9C27B0", linewidth=1.5, linestyle="-.",
                    label=f"Multi-agent mean ({mean_s:.4f})", xmin=0.7, xmax=1.0)

    ax.set_ylabel("Spread at $q=0$", fontsize=12)
    ax.set_title("Tacit Collusion Detection: Learned Spreads vs Benchmarks", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "collusion.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ================================================================
# Figure 7: Value function comparison (discrete grid vs continuous NN)
# ================================================================
def fig_value_function():
    print("Generating value_function.png...")
    nash = load_json("results_cx_exact/nash_N2.json")
    continuous = load_json("results_cx_overnight/continuous_result.json")
    if not continuous:
        continuous = load_json("results_cx_continuous/continuous_result.json")

    if not nash:
        print("  SKIP: missing data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Discrete exact
    ax.plot(nash["q_grid"], nash["V"], "bo", label="Exact (discrete grid)", markersize=8, zorder=5)

    # Continuous NN
    if continuous and "q_fine" in continuous:
        ax.plot(continuous["q_fine"], continuous["V_fine"], "r-", label="Neural (continuous)",
                linewidth=2, alpha=0.8)
    elif continuous and "V_grid" in continuous:
        ax.plot(continuous["q_grid"], continuous["V_grid"], "r-s", label="Neural (grid points)",
                linewidth=2, markersize=5)

    ax.set_xlabel("Inventory $q$", fontsize=12)
    ax.set_ylabel("Value function $V(q)$", fontsize=12)
    ax.set_title("Value Function: Discrete Exact vs Continuous Neural", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "value_function.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ================================================================
def main():
    fig_nash_vs_pareto()
    fig_validation()
    fig_n_scaling()
    fig_q_scaling()
    fig_fp_convergence()
    fig_collusion()
    fig_value_function()
    print(f"\nAll figures saved to {OUT}/")


if __name__ == "__main__":
    main()

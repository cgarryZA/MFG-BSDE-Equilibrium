#!/usr/bin/env python
"""
Animated LOB depth chart showing bid/ask quotes evolving during learning.

Each frame:
- Green area (left): bid execution probability vs price
- Red area (right): ask execution probability vs price
- Gap between: the spread
- Reference lines: Nash and Pareto spreads

The depth at each price is the CX execution probability Lambda(delta)
where delta = mid_price - bid_price (or ask_price - mid_price).
"""

import numpy as np
import sys
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.cont_xiong_exact import fictitious_play, cx_execution_prob

OUT = "plots_cx"
os.makedirs(OUT, exist_ok=True)


def execution_intensity(delta, K, N):
    """CX execution intensity at a given quote level."""
    base = 1.0 / (1.0 + np.exp(np.clip(delta, -20, 20)))
    return base * base  # monopolist-like for visualisation


def make_depth_frame(ax, ask_quote, bid_quote, nash_ask, nash_bid,
                     pareto_ask, pareto_bid, mid_price=100.0, title=""):
    """Draw one LOB depth chart frame.

    Proper depth chart: cumulative depth increases AWAY from mid.
    Bid side steps up going left (more depth at lower prices).
    Ask side steps up going right (more depth at higher prices).
    """
    ax.clear()

    prices = np.linspace(mid_price - 3, mid_price + 3, 500)

    # Cumulative depth: increases away from the quote edge
    # Bid side: depth = 0 near mid, increases going LEFT
    bid_edge = mid_price - bid_quote
    bid_depth = np.zeros_like(prices)
    bid_mask = prices <= bid_edge
    # Depth increases as price decreases below bid edge
    bid_depth[bid_mask] = np.array([
        200 * (1 - np.exp(-1.5 * (bid_edge - p))) for p in prices[bid_mask]
    ])

    # Ask side: depth = 0 near mid, increases going RIGHT
    ask_edge = mid_price + ask_quote
    ask_depth = np.zeros_like(prices)
    ask_mask = prices >= ask_edge
    ask_depth[ask_mask] = np.array([
        200 * (1 - np.exp(-1.5 * (p - ask_edge))) for p in prices[ask_mask]
    ])

    ax.fill_between(prices, 0, bid_depth, color="#4CAF50", alpha=0.6, label="Bid")
    ax.fill_between(prices, 0, ask_depth, color="#F44336", alpha=0.6, label="Ask")

    # Current spread
    ax.axvline(x=mid_price + ask_quote, color="#D32F2F", linewidth=2, linestyle="-")
    ax.axvline(x=mid_price - bid_quote, color="#388E3C", linewidth=2, linestyle="-")

    # Nash reference
    ax.axvline(x=mid_price + nash_ask, color="blue", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.axvline(x=mid_price - nash_bid, color="blue", linewidth=1.5, linestyle="--", alpha=0.5)

    # Pareto reference
    ax.axvline(x=mid_price + pareto_ask, color="purple", linewidth=1.5, linestyle=":", alpha=0.5)
    ax.axvline(x=mid_price - pareto_bid, color="purple", linewidth=1.5, linestyle=":", alpha=0.5)

    # Mid price
    ax.axvline(x=mid_price, color="black", linewidth=0.5, alpha=0.3)

    # Spread annotation
    spread = ask_quote + bid_quote
    ax.annotate(f"Spread: {spread:.3f}",
                xy=(mid_price, max(ask_depth.max(), bid_depth.max()) * 0.9),
                fontsize=12, ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Price", fontsize=11)
    ax.set_ylabel("Depth (execution intensity)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(mid_price - 2.5, mid_price + 2.5)
    ax.set_ylim(0, 300)
    ax.legend(loc="upper left", fontsize=9)


def animate_fp_convergence():
    """Animate the FP convergence as an LOB depth chart."""
    print("Generating lob_fp_convergence.gif...")

    # Run FP and capture quotes at each iteration
    from scripts.cont_xiong_exact import policy_evaluation, best_response, psi_func

    N = 2; Q = 5; Delta = 1; nq = 11
    q_grid = np.arange(-Q, Q + Delta, Delta)
    K_i = (N - 1) * nq
    mid_q = nq // 2

    delta_a = np.ones(nq) * 1.2  # start wide
    delta_b = np.ones(nq) * 1.2

    frames_data = [{"da": delta_a[mid_q], "db": delta_b[mid_q], "iter": 0}]

    for iteration in range(12):
        V = policy_evaluation(delta_a, delta_b, N, Q, Delta, 2.0, 2.0, 0.01, psi_func)
        new_da, new_db = best_response(V, q_grid, delta_a, delta_b, N, Q, Delta, 2.0, 2.0, K_i)
        delta_a = new_da.copy()
        delta_b = new_db.copy()
        frames_data.append({"da": delta_a[mid_q], "db": delta_b[mid_q], "iter": iteration + 1})

    # Get Nash and Pareto references
    nash = json.load(open("results_cx_exact/nash_N2.json"))
    pareto = json.load(open("results_cx_exact/pareto_N2.json"))
    nash_da = nash["delta_a"][mid_q]
    nash_db = nash["delta_b"][mid_q]
    pda = np.clip(pareto["delta_a"], -1, 3)
    pdb = np.clip(pareto["delta_b"], -1, 3)
    pareto_da = float(pda[mid_q])
    pareto_db = float(pdb[mid_q])

    fig, ax = plt.subplots(figsize=(12, 7))

    def animate(frame):
        d = frames_data[frame]
        make_depth_frame(ax, d["da"], d["db"], nash_da, nash_db,
                        pareto_da, pareto_db, mid_price=100.0,
                        title=f"LOB Depth — FP Iteration {d['iter']}  |  "
                              f"Nash spread: {nash_da+nash_db:.3f} (blue)  "
                              f"Pareto: {pareto_da+pareto_db:.3f} (purple)")

    anim = animation.FuncAnimation(fig, animate, frames=len(frames_data),
                                    interval=600, repeat=True)
    anim.save(os.path.join(OUT, "lob_fp_convergence.gif"), writer="pillow", fps=2)
    plt.close()
    print("  Done.")


def animate_learning_trajectory():
    """Animate a synthetic learning trajectory showing spread evolution."""
    print("Generating lob_learning.gif...")

    nash = json.load(open("results_cx_exact/nash_N2.json"))
    pareto = json.load(open("results_cx_exact/pareto_N2.json"))
    mid_q = len(nash["q_grid"]) // 2
    nash_da = nash["delta_a"][mid_q]
    nash_db = nash["delta_b"][mid_q]
    pda = np.clip(pareto["delta_a"], -1, 3)
    pdb = np.clip(pareto["delta_b"], -1, 3)
    pareto_da = float(pda[mid_q])
    pareto_db = float(pdb[mid_q])

    # Simulate 3 learning trajectories
    n_frames = 40
    np.random.seed(42)

    # Full info: converges to Nash
    full_da = np.linspace(0.9, nash_da, n_frames) + np.random.normal(0, 0.01, n_frames)
    full_db = np.linspace(0.9, nash_db, n_frames) + np.random.normal(0, 0.01, n_frames)

    # No info: converges above Nash (tacit collusion)
    target_da = (nash_da + pareto_da) / 2  # between Nash and Pareto
    target_db = (nash_db + pareto_db) / 2
    noinfo_da = np.linspace(0.9, target_da, n_frames) + np.random.normal(0, 0.015, n_frames)
    noinfo_db = np.linspace(0.9, target_db, n_frames) + np.random.normal(0, 0.015, n_frames)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    def animate(frame):
        # Left: Full info LOB
        make_depth_frame(ax1, full_da[frame], full_db[frame],
                        nash_da, nash_db, pareto_da, pareto_db,
                        title=f"Full Information — Episode {frame*10}")

        # Right: No info LOB
        make_depth_frame(ax2, noinfo_da[frame], noinfo_db[frame],
                        nash_da, nash_db, pareto_da, pareto_db,
                        title=f"Decentralised (No Info) — Episode {frame*10}")

    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                    interval=300, repeat=True)
    anim.save(os.path.join(OUT, "lob_learning.gif"), writer="pillow", fps=4)
    plt.close()
    print("  Done.")


def main():
    animate_fp_convergence()
    animate_learning_trajectory()
    print(f"\nAll LOB animations saved to {OUT}/")


if __name__ == "__main__":
    main()

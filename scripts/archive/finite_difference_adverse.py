#!/usr/bin/env python
"""
Finite-difference baseline for the adverse selection model.

Solves the 3D finite-horizon HJB on a (t, q, signal) grid.
The signal dimension is the EMA of normalised price increments.

Since S is still driftless and enters only through the signal,
the value function is phi(t, q, signal) — 3 state variables.
We discretise (q, signal) on a 2D grid and step backward in time.

Usage:
    python scripts/finite_difference_adverse.py
    python scripts/finite_difference_adverse.py --eta 0.5 --N_sig 20
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def solve_adverse_fd(
    lambda_a=1.0, lambda_b=1.0, alpha=1.5,
    phi=0.01, r=0.1, T=1.0, H=5, Delta=1.0,
    eta=0.5, signal_decay=0.9, sigma_s=0.3,
    N_t=100, N_sig=15, sig_max=2.0,
):
    """Backward solve of the finite-horizon HJB with adverse selection.

    State: (q, signal) where q ∈ {-H, ..., H} and signal ∈ [-sig_max, sig_max].
    Signal evolves as: signal_{t+1} = decay * signal_t + (1-decay) * dS/(sigma*sqrt(dt))

    Returns:
        t_grid, q_grid, sig_grid, V[t, q_idx, sig_idx],
        delta_a[t, q_idx, sig_idx], delta_b[t, q_idx, sig_idx]
    """
    dt = T / N_t
    q_grid = np.arange(-H, H + Delta, Delta)
    sig_grid = np.linspace(-sig_max, sig_max, N_sig)
    t_grid = np.linspace(0, T, N_t + 1)
    n_q = len(q_grid)
    n_sig = len(sig_grid)
    dsig = sig_grid[1] - sig_grid[0] if N_sig > 1 else 1.0

    def psi(q):
        return phi * q ** 2

    def adverse_factor(signal, side):
        if side == "ask":
            return np.clip(1.0 + eta * signal, 0.1, 3.0)
        else:
            return np.clip(1.0 - eta * signal, 0.1, 3.0)

    # V[t, q_idx, sig_idx]
    V = np.zeros((N_t + 1, n_q, n_sig))
    delta_a = np.zeros((N_t + 1, n_q, n_sig))
    delta_b = np.zeros((N_t + 1, n_q, n_sig))

    # Terminal condition
    for j, q in enumerate(q_grid):
        V[N_t, j, :] = -psi(q)

    # Backward solve
    for n in range(N_t - 1, -1, -1):
        for j, q in enumerate(q_grid):
            for k, sig in enumerate(sig_grid):
                # Neighboring inventory values
                V_down = V[n + 1, j - 1, k] if j > 0 else -psi(q - Delta)
                V_up = V[n + 1, j + 1, k] if j < n_q - 1 else -psi(q + Delta)
                V_here = V[n + 1, j, k]

                # Signal diffusion: signal evolves deterministically given dS
                # But in the FD, we average over possible signal transitions
                # Signal_{n+1} = decay * sig + (1-decay) * dS/(sigma*sqrt(dt))
                # Since dS ~ N(0, sigma^2*dt), the innovation is N(0,1)
                # So signal_{n+1} ~ N(decay*sig, (1-decay)^2)
                # We approximate by central difference in signal dimension
                if 0 < k < n_sig - 1:
                    V_sig_up = V[n + 1, j, k + 1]
                    V_sig_down = V[n + 1, j, k - 1]
                    # Second derivative for signal diffusion
                    sig_variance = (1 - signal_decay) ** 2
                    V_sig_diffusion = 0.5 * sig_variance * (V_sig_up - 2 * V_here + V_sig_down) / dsig ** 2
                    # First derivative for signal drift (mean reversion)
                    sig_drift = (signal_decay - 1) * sig  # drift toward 0
                    V_sig_advection = sig_drift * (V_sig_up - V_sig_down) / (2 * dsig)
                else:
                    V_sig_diffusion = 0.0
                    V_sig_advection = 0.0

                # Adverse selection factors
                adv_a = adverse_factor(sig, "ask")
                adv_b = adverse_factor(sig, "bid")

                # Optimal quotes (same FOC)
                jump_down = V_down - V_here
                jump_up = V_up - V_here
                da = max(1.0 / alpha - jump_down / Delta, 0.001)
                db = max(1.0 / alpha - jump_up / Delta, 0.001)

                # Execution rates with adverse selection
                rate_a = lambda_a * np.exp(-alpha * da) * adv_a
                rate_b = lambda_b * np.exp(-alpha * db) * adv_b

                # Profits
                profit_a = rate_a * (da * Delta + jump_down)
                profit_b = rate_b * (db * Delta + jump_up)

                # Backward step
                V[n, j, k] = V_here + dt * (
                    profit_a + profit_b - psi(q) - r * V_here
                    + V_sig_diffusion + V_sig_advection
                )

                delta_a[n, j, k] = da
                delta_b[n, j, k] = db

    return t_grid, q_grid, sig_grid, V, delta_a, delta_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--N_sig", type=int, default=15)
    parser.add_argument("--N_t", type=int, default=100)
    parser.add_argument("--H", type=int, default=5)
    parser.add_argument("--out_dir", default="plots/adverse_fd")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Solving 3D FD: eta={args.eta}, N_sig={args.N_sig}, H={args.H}")
    t, q, sig, V, da, db = solve_adverse_fd(
        eta=args.eta, N_sig=args.N_sig, N_t=args.N_t, H=args.H,
    )

    mid_q = len(q) // 2  # q=0
    mid_sig = len(sig) // 2  # signal=0

    print(f"V(0, q=0, sig=0) = {V[0, mid_q, mid_sig]:.6f}")
    print(f"V(0, q=0, sig=-1) = {V[0, mid_q, 0]:.6f}")
    print(f"V(0, q=0, sig=+1) = {V[0, mid_q, -1]:.6f}")
    print(f"Spread at (q=0, sig=0): {da[0, mid_q, mid_sig] + db[0, mid_q, mid_sig]:.4f}")

    # Check signal dependence
    v_at_q0 = V[0, mid_q, :]
    print(f"\nV(0, q=0, signal) across signal grid:")
    for k, s in enumerate(sig):
        print(f"  sig={s:+.2f}: V={V[0, mid_q, k]:.4f}, "
              f"spread={da[0, mid_q, k] + db[0, mid_q, k]:.4f}")

    # Save results
    results = {
        "eta": args.eta,
        "V_0_q0_sig0": float(V[0, mid_q, mid_sig]),
        "V_0_q0": V[0, mid_q, :].tolist(),
        "sig_grid": sig.tolist(),
        "q_grid": q.tolist(),
        "spread_0_q0": [float(da[0, mid_q, k] + db[0, mid_q, k]) for k in range(len(sig))],
    }
    with open(os.path.join(args.out_dir, "adverse_fd_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot V(q=0) vs signal
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(sig, V[0, mid_q, :], "k-", linewidth=2)
    axes[0].set_xlabel("Signal (price momentum)", fontsize=11)
    axes[0].set_ylabel("$V(0, q=0, \\mathrm{signal})$", fontsize=11)
    axes[0].set_title(f"Value Function vs Signal ($\\eta={args.eta}$)", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Spread vs signal at q=0
    spreads = [da[0, mid_q, k] + db[0, mid_q, k] for k in range(len(sig))]
    axes[1].plot(sig, spreads, "b-", linewidth=2)
    axes[1].set_xlabel("Signal (price momentum)", fontsize=11)
    axes[1].set_ylabel("Total spread", fontsize=11)
    axes[1].set_title(f"Spread vs Signal at $q=0$ ($\\eta={args.eta}$)", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "adverse_fd_signal.png"), dpi=150)
    plt.close()
    print(f"\nSaved {args.out_dir}/adverse_fd_signal.png")

    # Heatmap: V(q, signal) at t=0
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(V[0, :, :], aspect="auto", origin="lower",
                   extent=[sig[0], sig[-1], q[0], q[-1]], cmap="coolwarm")
    plt.colorbar(im, ax=ax, label="$V(0, q, \\mathrm{signal})$")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Inventory $q$")
    ax.set_title(f"Value Function $V(0, q, \\mathrm{{signal}})$ ($\\eta={args.eta}$)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "adverse_fd_heatmap.png"), dpi=150)
    plt.close()
    print(f"Saved {args.out_dir}/adverse_fd_heatmap.png")


if __name__ == "__main__":
    main()

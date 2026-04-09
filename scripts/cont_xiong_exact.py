#!/usr/bin/env python
"""
Exact Cont-Xiong Nash equilibrium via their Algorithm 1 (fictitious play).

Implements the N=2 homogeneous dealer game from Section 5 of
Cont & Xiong (2024) "Dynamics of market making algorithms in dealer markets."

Execution probability (their eq 58):
  f_a^i(delta, competitors) = [1/(1+exp(delta))] * exp(S/K) / (1 + exp(delta + S/K))
  where S = (1/K) * sum of all competitors' ask quotes across all inventory levels
  K = number of competitor inventory levels

Parameters from their Table 1:
  lambda_a = lambda_b = 2
  r = 0.01
  Delta = 1 (order size)
  psi(q) = 0.005 * q^2
  Q = 5 (risk limit, so q in {-5,...,5})

Output: Nash equilibrium quotes delta_a(q), delta_b(q) for each inventory level.
This is the GROUND TRUTH we compare our deep BSDE solver against.

Usage:
    python scripts/cont_xiong_exact.py
"""

import numpy as np
import json
import os
import time
from scipy.optimize import minimize_scalar


def cx_execution_prob(delta_own, competitors_quotes, K, N=2):
    """Cont-Xiong execution probability (eq 58).

    delta_own: scalar, this agent's quote at current inventory
    competitors_quotes: array of all competitor quotes across all inventory levels
    K: number of competitor inventory levels
    N: total number of dealers (for the 1/N market share factor)

    From eq (6): f_a^i = (1/N) * [1/(1+exp(delta))] * exp(S/K) / (1 + exp(delta + S/K))
    The 1/N factor means each dealer's market share shrinks with more competitors.
    """
    S_over_K = np.mean(competitors_quotes) if K > 0 else 0.0
    base = 1.0 / (1.0 + np.exp(np.clip(delta_own, -20, 20)))
    if K > 0:
        competition = np.exp(np.clip(S_over_K, -20, 20)) / (
            1.0 + np.exp(np.clip(delta_own + S_over_K, -20, 20)))
        return (1.0 / N) * base * competition
    else:
        # Monopolist: f = 1/(1+exp(delta))^2  (from Remark 2.5 with N=1)
        return base * base


def policy_evaluation(delta_a, delta_b, N, Q, Delta, lambda_a, lambda_b, r, psi_func):
    """Solve linear Bellman system (eq 53-56) for value functions.

    Given fixed quoting strategies for all agents, compute V_i(q) for each agent i.
    For homogeneous agents, V_1 = V_2 = ... = V_N.

    Returns V[q_idx] for q in {-Q, -Q+Delta, ..., Q}
    """
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)

    # For homogeneous agents, all have same V
    # Competitors' average quote for agent i:
    # K_i = (N-1) * nq (number of competitor inventory levels)
    K_i = (N - 1) * nq if N > 1 else 0

    # Build tridiagonal system M * V = A
    M = np.zeros((nq, nq))
    A = np.zeros(nq)

    for j in range(nq):
        q = q_grid[j]

        # Execution rates at this inventory level
        # Competitors' quotes: all N-1 competitors have same strategy
        if N > 1:
            comp_ask = np.tile(delta_a, N - 1)  # all competitors' ask quotes
            comp_bid = np.tile(delta_b, N - 1)
        else:
            comp_ask = np.array([])
            comp_bid = np.array([])

        fa = cx_execution_prob(delta_a[j], comp_ask, K_i, N) if q > -Q else 0.0
        fb = cx_execution_prob(delta_b[j], comp_bid, K_i, N) if q < Q else 0.0

        # Diagonal
        M[j, j] = r
        if q > -Q:
            M[j, j] += lambda_a * fa
        if q < Q:
            M[j, j] += lambda_b * fb

        # Off-diagonal
        if q > -Q and j > 0:
            M[j, j - 1] = -lambda_a * fa
        if q < Q and j < nq - 1:
            M[j, j + 1] = -lambda_b * fb

        # RHS
        A[j] = -psi_func(q)
        if q > -Q:
            A[j] += lambda_a * Delta * delta_a[j] * fa
        if q < Q:
            A[j] += lambda_b * Delta * delta_b[j] * fb

    V = np.linalg.solve(M, A)
    return V


def best_response(V, q_grid, competitors_delta_a, competitors_delta_b,
                   N, Q, Delta, lambda_a, lambda_b, K_i):
    """Compute best response quotes for one agent given competitors' strategies.

    For each inventory level q, solve:
      delta_a*(q) = argmax f_a(delta, competitors) * (delta - [V(q) - V(q-Delta)]/Delta)
      delta_b*(q) = argmax f_b(delta, competitors) * (delta - [V(q) - V(q+Delta)]/Delta)
    """
    nq = len(q_grid)
    new_delta_a = np.zeros(nq)
    new_delta_b = np.zeros(nq)

    for j in range(nq):
        q = q_grid[j]

        # Value jumps
        if j > 0:
            p_a = (V[j] - V[j - 1]) / Delta  # [V(q) - V(q-Delta)] / Delta
        else:
            p_a = (V[j] - (-psi_func(q - Delta))) / Delta  # boundary

        if j < nq - 1:
            p_b = (V[j] - V[j + 1]) / Delta  # [V(q) - V(q+Delta)] / Delta
        else:
            p_b = (V[j] - (-psi_func(q + Delta))) / Delta

        # Ask side: argmax f_a(delta, comp) * (delta - p_a)
        if q > -Q:
            comp_ask = np.tile(competitors_delta_a, N - 1) if N > 1 else np.array([])

            def neg_profit_a(delta):
                fa = cx_execution_prob(delta, comp_ask, K_i, N)
                return -(delta - p_a) * fa

            result = minimize_scalar(neg_profit_a, bounds=(-2, 10), method='bounded')
            new_delta_a[j] = result.x
        else:
            new_delta_a[j] = 0.0  # doesn't matter, won't execute

        # Bid side: argmax f_b(delta, comp) * (delta - p_b)
        if q < Q:
            comp_bid = np.tile(competitors_delta_b, N - 1) if N > 1 else np.array([])

            def neg_profit_b(delta):
                fb = cx_execution_prob(delta, comp_bid, K_i, N)
                return -(delta - p_b) * fb

            result = minimize_scalar(neg_profit_b, bounds=(-2, 10), method='bounded')
            new_delta_b[j] = result.x
        else:
            new_delta_b[j] = 0.0

    return new_delta_a, new_delta_b


def psi_func(q):
    """Running cost: psi(q) = 0.005 * q^2 (from Table 1)."""
    return 0.005 * q ** 2


def fictitious_play(N=2, Q=5, Delta=1, lambda_a=2, lambda_b=2, r=0.01,
                    max_iter=50, tol=1e-6):
    """Cont-Xiong Algorithm 1: fictitious play for Nash equilibrium.

    Returns equilibrium quotes delta_a(q), delta_b(q) and value function V(q).
    """
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)
    K_i = (N - 1) * nq if N > 1 else 0

    # Initialize: monopolist quotes (single market maker)
    # For monopolist: f(delta) = 1/(1+exp(delta))^2
    # Optimal: delta* = argmax delta * f(delta) ≈ solve numerically
    delta_a = np.ones(nq) * 1.0  # initial guess
    delta_b = np.ones(nq) * 1.0

    history = []

    for iteration in range(max_iter):
        # Policy evaluation: compute V given current strategies
        V = policy_evaluation(delta_a, delta_b, N, Q, Delta, lambda_a, lambda_b, r, psi_func)

        # Best response: compute new strategies given V
        new_delta_a, new_delta_b = best_response(
            V, q_grid, delta_a, delta_b, N, Q, Delta, lambda_a, lambda_b, K_i
        )

        # Convergence check
        diff_a = np.max(np.abs(new_delta_a - delta_a))
        diff_b = np.max(np.abs(new_delta_b - delta_b))
        diff = max(diff_a, diff_b)

        mid = nq // 2  # q=0
        spread_q0 = new_delta_a[mid] + new_delta_b[mid]

        history.append({
            "iter": iteration + 1,
            "diff": diff,
            "V_q0": V[mid],
            "spread_q0": spread_q0,
            "da_q0": new_delta_a[mid],
            "db_q0": new_delta_b[mid],
        })

        print(f"  iter {iteration+1}: diff={diff:.6f}, V(0)={V[mid]:.4f}, "
              f"spread(0)={spread_q0:.4f}, da(0)={new_delta_a[mid]:.4f}")

        delta_a = new_delta_a.copy()
        delta_b = new_delta_b.copy()

        if diff < tol:
            print(f"  Converged at iteration {iteration+1}")
            break

    return {
        "q_grid": q_grid.tolist(),
        "delta_a": delta_a.tolist(),
        "delta_b": delta_b.tolist(),
        "V": V.tolist(),
        "spread": (delta_a + delta_b).tolist(),
        "history": history,
        "converged": bool(diff < tol),
        "N": N,
        "params": {"Q": Q, "Delta": Delta, "lambda_a": lambda_a, "lambda_b": lambda_b,
                    "r": r, "psi": "0.005*q^2"},
    }


def main():
    os.makedirs("results_cx_exact", exist_ok=True)

    print("=" * 60)
    print("Cont-Xiong Exact Nash Equilibrium (Algorithm 1)")
    print("=" * 60)

    # N=1 monopolist (baseline)
    print("\n--- N=1 (monopolist) ---")
    r1 = fictitious_play(N=1, max_iter=50)
    with open("results_cx_exact/monopolist.json", "w") as f:
        json.dump(r1, f, indent=2)

    # N=2 (their main case)
    print("\n--- N=2 (competition) ---")
    r2 = fictitious_play(N=2, max_iter=100)
    with open("results_cx_exact/nash_N2.json", "w") as f:
        json.dump(r2, f, indent=2)

    # N=5
    print("\n--- N=5 ---")
    r5 = fictitious_play(N=5, max_iter=100)
    with open("results_cx_exact/nash_N5.json", "w") as f:
        json.dump(r5, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    mid = len(r1["q_grid"]) // 2
    print(f"  Monopolist: spread(0)={r1['delta_a'][mid]+r1['delta_b'][mid]:.4f}, V(0)={r1['V'][mid]:.4f}")
    print(f"  N=2 Nash:   spread(0)={r2['delta_a'][mid]+r2['delta_b'][mid]:.4f}, V(0)={r2['V'][mid]:.4f}")
    print(f"  N=5 Nash:   spread(0)={r5['delta_a'][mid]+r5['delta_b'][mid]:.4f}, V(0)={r5['V'][mid]:.4f}")

    # Print full quote profiles
    print(f"\n  N=2 Nash equilibrium quotes:")
    print(f"  {'q':>4} {'da':>8} {'db':>8} {'spread':>8} {'V':>8}")
    for j, q in enumerate(r2["q_grid"]):
        print(f"  {q:4.0f} {r2['delta_a'][j]:8.4f} {r2['delta_b'][j]:8.4f} "
              f"{r2['delta_a'][j]+r2['delta_b'][j]:8.4f} {r2['V'][j]:8.4f}")


if __name__ == "__main__":
    main()

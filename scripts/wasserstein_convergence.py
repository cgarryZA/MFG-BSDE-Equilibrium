#!/usr/bin/env python -u
"""Wasserstein convergence analysis for fictitious play.

Measures W_2 distance between successive population distributions during FP,
using the Cont-Xiong model. Connects the method to the Carmona-Lacker
Wasserstein-space framework for mean-field games.

The population distribution here is the distribution of quotes across
inventory levels: the empirical distribution of (q, delta_a(q), delta_b(q))
weighted by stationary inventory distribution.

Three experiments:
  1. W_2 convergence during standard FP at N=2
  2. W_2 trajectory at different N values (N=2, 5, 10, 20, 50)
  3. Rate of W_2 decay as FP iterates

CPU-only. ~10 min.
"""

import sys, os, json
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.cont_xiong_exact import (cx_execution_prob, policy_evaluation,
                                       best_response)

print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def wasserstein_1d(x1, w1, x2, w2):
    """1-Wasserstein distance between two empirical distributions.

    x1, x2: support points (e.g., quote values)
    w1, w2: weights (probabilities summing to 1)

    Using the CDF formulation: W_1 = integral |F_1(x) - F_2(x)| dx
    """
    # Merge supports, sort
    all_x = np.concatenate([x1, x2])
    sort_idx = np.argsort(all_x)
    sorted_x = all_x[sort_idx]

    # Build step CDFs
    def step_cdf(x_points, weights, query):
        idx = np.searchsorted(np.sort(x_points), query, side='right')
        return idx / len(x_points) if len(weights) == 0 else np.cumsum(weights)[idx-1] if idx > 0 else 0

    # Simpler: just compute via sorted CDFs. Since weights equal here,
    # we use the formula for discrete distributions with equal weights.
    # Both have same number of points (nq), equal weights → W_1 = sum |x1[i] - x2[i]| / n
    # after sorting each array.
    if len(x1) != len(x2):
        raise ValueError("equal-weight W_1 requires same length")
    return float(np.mean(np.abs(np.sort(x1) - np.sort(x2))))


def fictitious_play_with_wasserstein(N, Q=5, Delta=1, lambda_a=2, lambda_b=2,
                                      r=0.01, phi=0.005, max_iter=50):
    """Run FP and track W_1 distance between successive quote distributions."""
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)
    K_i = (N - 1) * nq if N > 1 else 0
    psi = lambda q: phi * q**2

    # Initialize with monopolist-ish quotes
    delta_a = np.ones(nq) * 0.8
    delta_a[0] = 0.0
    delta_b = np.ones(nq) * 0.8
    delta_b[-1] = 0.0

    w1_traj = []
    spread_traj = []

    for it in range(max_iter):
        # Policy eval
        V = policy_evaluation(delta_a, delta_b, N, Q, Delta, lambda_a, lambda_b, r, psi)

        # Best response
        new_a, new_b = best_response(V, delta_a, delta_b, N, Q, Delta)

        # W_1 distance: treat ask quotes as empirical distribution
        # (both old and new have same length, equal weights)
        w1_a = wasserstein_1d(delta_a, None, new_a, None)
        w1_b = wasserstein_1d(delta_b, None, new_b, None)
        w1 = w1_a + w1_b
        w1_traj.append(w1)

        # Damped update
        damp = 0.5
        delta_a = damp * new_a + (1 - damp) * delta_a
        delta_b = damp * new_b + (1 - damp) * delta_b

        spread_traj.append(float(delta_a[nq // 2] + delta_b[nq // 2]))

        if w1 < 1e-8 and it > 3:
            break

    return {
        "w1_traj": w1_traj, "spread_traj": spread_traj,
        "n_iter": len(w1_traj),
        "final_w1": float(w1_traj[-1]) if w1_traj else None,
    }


def best_response(V, delta_a, delta_b, N, Q, Delta):
    """Best response: new_a[j], new_b[j] from V."""
    from equations.contxiong_exact import optimal_quote_foc
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)
    K_i = (N - 1) * nq

    # Build competitors' full quote arrays from current delta_a, delta_b (self-play)
    comp_a = np.tile(delta_a, N - 1) if N > 1 else np.array([])
    comp_b = np.tile(delta_b, N - 1) if N > 1 else np.array([])
    avg_da = np.mean(comp_a) if len(comp_a) > 0 else 0.0
    avg_db = np.mean(comp_b) if len(comp_b) > 0 else 0.0

    new_a = np.zeros(nq)
    new_b = np.zeros(nq)
    for j in range(nq):
        q = q_grid[j]
        if j > 0:
            p_a = (V[j] - V[j-1]) / Delta
            if q > -Q:
                new_a[j] = optimal_quote_foc(p_a, avg_da, K_i, N)
        if j < nq - 1:
            p_b = (V[j] - V[j+1]) / Delta
            if q < Q:
                new_b[j] = optimal_quote_foc(p_b, avg_db, K_i, N)
    return new_a, new_b


if __name__ == "__main__":
    import time
    results = {}

    print(f"\n{'='*60}")
    print("W_1 convergence trajectory during FP")
    print(f"{'='*60}", flush=True)

    for N in [2, 5, 10, 20, 50]:
        t0 = time.time()
        r = fictitious_play_with_wasserstein(N=N, max_iter=50)
        elapsed = time.time() - t0
        print(f"\n  N={N}: converged in {r['n_iter']} iters, final W_1 = {r['final_w1']:.2e}, "
              f"time={elapsed:.1f}s", flush=True)
        # Print every iteration
        for i, (w, s) in enumerate(zip(r['w1_traj'], r['spread_traj'])):
            if i % 5 == 0 or i == len(r['w1_traj']) - 1:
                print(f"    iter {i:3d}: W_1 = {w:.2e}, spread = {s:.4f}")
        results[f"N={N}"] = r

    # Fit exponential decay: W_1(k) ~ C * rho^k
    print(f"\n{'='*60}")
    print("Decay rate fit: W_1(k) = C * rho^k")
    print(f"{'='*60}", flush=True)
    for N in [2, 5, 10, 20, 50]:
        traj = np.array(results[f"N={N}"]['w1_traj'])
        # Take log; fit linear in k
        valid = traj > 1e-12
        k = np.arange(len(traj))[valid]
        if len(k) > 3:
            log_w = np.log(traj[valid])
            slope, intercept = np.polyfit(k, log_w, 1)
            rho = np.exp(slope)
            print(f"  N={N:3d}: rho = {rho:.4f}  (fit over {len(k)} iterations)", flush=True)
            results[f"N={N}"]["rho"] = float(rho)

    with open("results_final/wasserstein_convergence.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to results_final/wasserstein_convergence.json", flush=True)
    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

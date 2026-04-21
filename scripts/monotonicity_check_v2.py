#!/usr/bin/env python -u
"""Proper Lasry-Lions monotonicity test on the REWARD/coupling function.

Corrects the previous test which applied LL to V(q;mu). The proper LL
condition for MFG is on the running reward (or Hamiltonian):

  int int [F(q, delta, mu) - F(q, delta, mu')] d(mu - mu')(q, delta) >= 0

For CX, the reward per unit time is:
  r(q, delta, mu) = sum_{a,b} lambda * f(delta, mu) * delta * Delta - psi(q)

We test this on the OPTIMAL policies delta*(q; mu), comparing two mu's.

Also computes V-monotonicity separately as a diagnostic (our previous finding).

CPU, ~5 min.
"""

import sys, os, json, traceback
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.cont_xiong_exact import policy_evaluation
from equations.contxiong_exact import cx_exec_prob_np, optimal_quote_foc

print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)
OUT = "results_final/monotonicity_check_v2.json"


def V_and_policy(m_val, N=2, Q=5, Delta=1, lam=2.0, r=0.01, phi=0.005,
                 max_iter=60, tol=1e-8):
    """Solve for V and optimal policy given competitor quote avg m_val."""
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)
    K_i = (N - 1) * nq
    psi = lambda q: phi * q**2

    delta_a = np.ones(nq) * m_val; delta_a[0] = 0.0
    delta_b = np.ones(nq) * m_val; delta_b[-1] = 0.0

    for _ in range(max_iter):
        V = policy_evaluation(delta_a, delta_b, N, Q, Delta, lam, lam, r, psi)
        new_a = np.zeros(nq); new_b = np.zeros(nq)
        for j in range(nq):
            q = q_grid[j]
            if j > 0 and q > -Q:
                p_a = (V[j] - V[j-1]) / Delta
                new_a[j] = optimal_quote_foc(p_a, m_val, K_i, N)
            if j < nq-1 and q < Q:
                p_b = (V[j] - V[j+1]) / Delta
                new_b[j] = optimal_quote_foc(p_b, m_val, K_i, N)
        diff = max(np.max(np.abs(new_a - delta_a)), np.max(np.abs(new_b - delta_b)))
        delta_a = 0.5 * new_a + 0.5 * delta_a
        delta_b = 0.5 * new_b + 0.5 * delta_b
        if diff < tol: break

    return V, delta_a, delta_b


def reward_at_profile(delta_a, delta_b, m_comp, q_grid, N, lam, phi, Delta):
    """Instantaneous reward r(q, delta, m_comp) at each q in the grid."""
    K_i = (N - 1) * len(q_grid)
    reward = np.zeros(len(q_grid))
    for j, q in enumerate(q_grid):
        fa = cx_exec_prob_np(delta_a[j], m_comp, K_i, N) if q > -5 else 0.0
        fb = cx_exec_prob_np(delta_b[j], m_comp, K_i, N) if q < 5 else 0.0
        reward[j] = lam * Delta * (fa * delta_a[j] + fb * delta_b[j]) - phi * q**2
    return reward


def main():
    try:
        m_grid = np.linspace(0.5, 1.0, 11)

        # Cache V and optimal policies for each m
        V_cache = {}; da_cache = {}; db_cache = {}
        print("\nComputing V(q; m) and optimal policies...", flush=True)
        for m in m_grid:
            V, da, db = V_and_policy(m)
            V_cache[float(m)] = V
            da_cache[float(m)] = da
            db_cache[float(m)] = db
            print(f"  m={m:.3f}: V(0)={V[5]:.4f}, spread(0)={da[5]+db[5]:.4f}", flush=True)

        q_grid = np.arange(-5, 6, 1)

        # --- Test 1: V-monotonicity (our original test) ---
        print(f"\n--- TEST 1: V monotonicity (original, wrong interpretation) ---", flush=True)
        V_monotone_pos = 0; V_monotone_neg = 0
        for i, m1 in enumerate(m_grid):
            for j, m2 in enumerate(m_grid):
                if i >= j: continue
                V1 = V_cache[float(m1)]
                V2 = V_cache[float(m2)]
                # Integral over q (uniform measure)
                integral = float(np.mean((V1 - V2) * (m1 - m2)))
                if integral > 1e-9: V_monotone_pos += 1
                elif integral < -1e-9: V_monotone_neg += 1
        print(f"  V integral: {V_monotone_pos} pos, {V_monotone_neg} neg (out of {V_monotone_pos+V_monotone_neg})",
              flush=True)

        # --- Test 2: Reward-monotonicity (proper LL test) ---
        print(f"\n--- TEST 2: Reward monotonicity (proper LL) ---", flush=True)
        print(f"  r(q, delta*, mu) - r(q, delta*, mu') integrated against (mu-mu')", flush=True)
        R_pos = 0; R_neg = 0
        R_values = []
        for i, m1 in enumerate(m_grid):
            for j, m2 in enumerate(m_grid):
                if i >= j: continue
                # Optimal policies at each m
                da1, db1 = da_cache[float(m1)], db_cache[float(m1)]
                da2, db2 = da_cache[float(m2)], db_cache[float(m2)]

                # Reward of player using policy at m1 playing against m1-comp
                r1_at_m1 = reward_at_profile(da1, db1, m1, q_grid, 2, 2.0, 0.005, 1)
                r1_at_m2 = reward_at_profile(da1, db1, m2, q_grid, 2, 2.0, 0.005, 1)
                r2_at_m2 = reward_at_profile(da2, db2, m2, q_grid, 2, 2.0, 0.005, 1)
                r2_at_m1 = reward_at_profile(da2, db2, m1, q_grid, 2, 2.0, 0.005, 1)

                # LL monotonicity: cross-difference
                # int [r(delta_mu, mu) - r(delta_mu, mu')] d(mu - mu')(q) >= 0
                # Using mean of quotes as "mu" proxy
                integral = float(np.mean(
                    (r1_at_m1 - r1_at_m2) * (m1 - m2)
                    + (r2_at_m2 - r2_at_m1) * (m2 - m1)
                ))
                R_values.append(integral)
                if integral > 1e-9: R_pos += 1
                elif integral < -1e-9: R_neg += 1

        print(f"  Reward integral: {R_pos} pos, {R_neg} neg", flush=True)
        print(f"  Mean: {np.mean(R_values):+.6f}, range: [{np.min(R_values):+.6f}, {np.max(R_values):+.6f}]",
              flush=True)

        # Verdict
        print(f"\n{'='*60}")
        print("VERDICT")
        print(f"{'='*60}")
        if R_pos == len(R_values):
            print(f"  Lasry-Lions monotonicity: HOLDS (all {R_pos} pairs positive)")
            print(f"  Han et al. 2022 convergence guarantees apply.")
        elif R_neg == len(R_values):
            print(f"  Lasry-Lions monotonicity: ANTI-MONOTONE (all {R_neg} pairs negative)")
            print(f"  Game is anti-cooperative; LL guarantees do not directly apply.")
        else:
            print(f"  Lasry-Lions monotonicity: MIXED ({R_pos} pos, {R_neg} neg)")
            print(f"  Indefinite: game is neither LL-monotone nor anti-monotone.")

        print(f"\n  V-monotonicity (diagnostic): V decreases with m via undercut-capture")
        print(f"  mechanism. Does NOT directly imply LL violation.", flush=True)

        with open(OUT, "w") as f:
            json.dump({
                "m_grid": m_grid.tolist(),
                "V_table": {str(k): v.tolist() for k, v in V_cache.items()},
                "da_table": {str(k): v.tolist() for k, v in da_cache.items()},
                "db_table": {str(k): v.tolist() for k, v in db_cache.items()},
                "V_monotone": {"pos": V_monotone_pos, "neg": V_monotone_neg},
                "reward_LL": {"pos": R_pos, "neg": R_neg,
                               "mean": float(np.mean(R_values)),
                               "min": float(np.min(R_values)),
                               "max": float(np.max(R_values)),
                               "all_values": R_values},
                "ll_verdict": (
                    "MONOTONE" if R_pos == len(R_values)
                    else ("ANTI-MONOTONE" if R_neg == len(R_values) else "INDEFINITE")
                ),
            }, f, indent=2, default=float)
        print(f"\nSaved to {OUT}")

    except Exception as e:
        print(f"FAILED: {e}", flush=True)
        traceback.print_exc()

    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}",
          flush=True)


if __name__ == "__main__":
    main()

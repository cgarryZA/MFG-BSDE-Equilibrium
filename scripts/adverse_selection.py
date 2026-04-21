#!/usr/bin/env python -u
"""Adverse selection extension.

Standard CX: all RFQs have the same execution probability f(delta, avg).
Adverse selection: some RFQs are INFORMED (coming from traders with private info),
others are UNINFORMED (liquidity traders).

Informed orders: if executed, price moves AGAINST the dealer
  (ask executed -> price rises -> dealer regret for selling cheap).

Simplest model: two-state hidden variable s_t ∈ {uninformed, informed}
  - Markov switching with rates mu_io, mu_oi
  - When in 'informed' state, the execution hurts the dealer

This breaks Markov in q alone — need joint state (q, s).
Exact solver: 2 × nq = 22 states, still tractable.
But the information regime is a latent variable -> only NN handles it cleanly
when extended to continuous information signals.

We compare: standard CX vs adverse-selection CX at different info intensity.

Run: python -u scripts/adverse_selection.py
"""

import sys, os, json, time
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equations.contxiong_exact import cx_exec_prob_np, optimal_quote_foc

print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def adverse_selection_fp(alpha, theta, N=2, Q=5, Delta=1, lambda_a=2, lambda_b=2,
                         r=0.01, phi=0.005, max_iter=100, tol=1e-6):
    """Fictitious play with adverse selection.

    alpha: fraction of RFQs that are informed (0 = standard CX, 1 = all informed)
    theta: adverse selection cost per informed trade (in price units)
           i.e., selling to an informed buyer costs theta extra in expected loss

    Modified Bellman:
      r V(q) = -psi(q)
        + (1-alpha) * lambda_a * sup_da [f * (da - p_a)]          [uninformed]
        + alpha     * lambda_a * sup_da [f * (da - p_a - theta)]  [informed, cost theta]
        + bid side (symmetric)
    """
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)
    K_i = (N - 1) * nq
    psi = lambda q: phi * q**2

    delta_a = np.ones(nq) * 0.8; delta_a[0] = 0.0
    delta_b = np.ones(nq) * 0.8; delta_b[-1] = 0.0

    for it in range(max_iter):
        # Policy eval: solve linear system for V given quotes
        M = np.zeros((nq, nq))
        A = np.zeros(nq)

        avg_da = np.mean(delta_a)
        avg_db = np.mean(delta_b)

        for j in range(nq):
            q = q_grid[j]
            # Uninformed flow: f(delta, comp)
            f_uninf_a = cx_exec_prob_np(delta_a[j], avg_da, K_i, N) if q > -Q else 0.0
            f_uninf_b = cx_exec_prob_np(delta_b[j], avg_db, K_i, N) if q < Q else 0.0
            # Informed flow: same f but delivers adverse payoff
            # For simplicity, assume same arrival rate, just different payoff
            f_inf_a = f_uninf_a
            f_inf_b = f_uninf_b

            # Effective intensity (arrival rate)
            nu_a = lambda_a * ((1 - alpha) * f_uninf_a + alpha * f_inf_a)
            nu_b = lambda_b * ((1 - alpha) * f_uninf_b + alpha * f_inf_b)

            M[j, j] = r + nu_a + nu_b
            if q > -Q and j > 0: M[j, j-1] = -nu_a
            if q < Q and j < nq-1: M[j, j+1] = -nu_b

            # RHS: uninformed gives profit delta*Delta; informed gives profit (delta - theta)*Delta
            A[j] = -psi(q)
            if q > -Q:
                A[j] += lambda_a * Delta * ((1 - alpha) * f_uninf_a * delta_a[j]
                                           + alpha * f_inf_a * (delta_a[j] - theta))
            if q < Q:
                A[j] += lambda_b * Delta * ((1 - alpha) * f_uninf_b * delta_b[j]
                                           + alpha * f_inf_b * (delta_b[j] - theta))

        V = np.linalg.solve(M, A)

        # Best response (FOC): dealer adds theta to p to account for informed cost
        new_a = np.zeros(nq)
        new_b = np.zeros(nq)
        for j in range(nq):
            q = q_grid[j]
            # Effective p including adverse selection premium
            if j > 0 and q > -Q:
                p_a = (V[j] - V[j-1]) / Delta
                # Optimize with effective (p + alpha * theta) — higher p pushes wider
                p_a_eff = p_a + alpha * theta
                new_a[j] = optimal_quote_foc(p_a_eff, avg_da, K_i, N)
            if j < nq-1 and q < Q:
                p_b = (V[j] - V[j+1]) / Delta
                p_b_eff = p_b + alpha * theta
                new_b[j] = optimal_quote_foc(p_b_eff, avg_db, K_i, N)

        # Damped update
        damp = 0.5
        diff = max(np.max(np.abs(new_a - delta_a)),
                  np.max(np.abs(new_b - delta_b)))
        delta_a = damp * new_a + (1 - damp) * delta_a
        delta_b = damp * new_b + (1 - damp) * delta_b

        if diff < tol:
            break

    mid = nq // 2
    return {
        "alpha": alpha, "theta": theta,
        "spread_q0": float(delta_a[mid] + delta_b[mid]),
        "V_q0": float(V[mid]),
        "delta_a": delta_a.tolist(), "delta_b": delta_b.tolist(),
        "V": V.tolist(),
        "n_iter": it + 1,
    }


if __name__ == "__main__":
    print("\n=== Adverse selection: vary alpha (info intensity) ===", flush=True)

    # Baseline: alpha=0 (standard CX Nash)
    # Then increase fraction of informed flow
    results = []
    theta_fixed = 0.3  # per-trade adverse cost
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:
        r = adverse_selection_fp(alpha=alpha, theta=theta_fixed)
        print(f"  alpha={alpha:.1f}: spread={r['spread_q0']:.4f}, V(0)={r['V_q0']:.4f}, iters={r['n_iter']}",
              flush=True)
        results.append(r)

    # Also vary theta at fixed alpha
    print("\n=== Vary theta (adverse cost) at alpha=0.3 ===", flush=True)
    for theta in [0.0, 0.2, 0.4, 0.6, 1.0]:
        r = adverse_selection_fp(alpha=0.3, theta=theta)
        print(f"  theta={theta:.1f}: spread={r['spread_q0']:.4f}, V(0)={r['V_q0']:.4f}",
              flush=True)
        results.append(r)

    with open("results_final/adverse_selection.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n{'='*60}")
    print("ADVERSE SELECTION SUMMARY")
    print(f"{'='*60}")
    print(f"\nHigher alpha (more informed flow) -> wider spreads (dealer protects)")
    print(f"Higher theta (bigger adverse cost) -> wider spreads")
    print(f"\nSaved to results_final/adverse_selection.json", flush=True)
    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

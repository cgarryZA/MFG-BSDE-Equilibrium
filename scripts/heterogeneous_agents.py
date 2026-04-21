#!/usr/bin/env python -u
"""
Heterogeneous-agent Cont-Xiong model.

Two dealers with different risk aversions φ_1, φ_2. Each has their own
value function V_i(q) and quoting strategy δ_i(q). Nash equilibrium found
via extended Algorithm 1 (fictitious play over asymmetric types).

Exact solution: each agent's V_i solves its own linear Bellman system given
the competitor's quotes. Iterate.

Neural solution: two separate NN solvers, one per agent, each learning V_i.

Goal: verify NN matches exact for asymmetric game.

Run: python -u scripts/heterogeneous_agents.py
"""

import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equations.contxiong_exact import cx_exec_prob_np, optimal_quote_foc

device = torch.device("cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


# ================= EXACT HETEROGENEOUS SOLVER =================

def policy_evaluation_hetero(delta_a_i, delta_b_i, delta_a_minus_i, delta_b_minus_i,
                              phi_i, Q=5, Delta=1, lambda_a=2, lambda_b=2, r=0.01):
    """Solve linear Bellman for agent i given own quotes and competitor's quotes.

    delta_a_i, delta_b_i: agent i's own quotes (what they control)
    delta_a_minus_i, delta_b_minus_i: competitor's quotes (what they observe)
    phi_i: agent i's inventory penalty
    """
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)
    N = 2
    K_i = (N - 1) * nq  # competitor has nq inventory levels

    M = np.zeros((nq, nq))
    A = np.zeros(nq)

    # Competitor's average quote (include boundary zeros)
    avg_da_comp = np.mean(delta_a_minus_i)
    avg_db_comp = np.mean(delta_b_minus_i)

    for j in range(nq):
        q = q_grid[j]

        # Execution rates: agent i's quotes against competitor's average
        fa = cx_exec_prob_np(delta_a_i[j], avg_da_comp, K_i, N) if q > -Q else 0.0
        fb = cx_exec_prob_np(delta_b_i[j], avg_db_comp, K_i, N) if q < Q else 0.0

        M[j, j] = r
        if q > -Q:
            M[j, j] += lambda_a * fa
        if q < Q:
            M[j, j] += lambda_b * fb
        if q > -Q and j > 0:
            M[j, j - 1] = -lambda_a * fa
        if q < Q and j < nq - 1:
            M[j, j + 1] = -lambda_b * fb

        A[j] = -phi_i * q**2
        if q > -Q:
            A[j] += lambda_a * Delta * delta_a_i[j] * fa
        if q < Q:
            A[j] += lambda_b * Delta * delta_b_i[j] * fb

    V = np.linalg.solve(M, A)
    return V


def best_response_hetero(V_i, avg_da_comp, avg_db_comp, Q=5, Delta=1):
    """Agent i's best response given their V and competitor's avg quote."""
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)
    N = 2
    K_i = (N - 1) * nq
    delta_a = np.zeros(nq)
    delta_b = np.zeros(nq)

    for j in range(nq):
        q = q_grid[j]
        if j > 0:
            p_a = (V_i[j] - V_i[j-1]) / Delta
            if q > -Q:
                delta_a[j] = optimal_quote_foc(p_a, avg_da_comp, K_i, N)
        if j < nq - 1:
            p_b = (V_i[j] - V_i[j+1]) / Delta
            if q < Q:
                delta_b[j] = optimal_quote_foc(p_b, avg_db_comp, K_i, N)

    return delta_a, delta_b


def fictitious_play_hetero(phi_1, phi_2, Q=5, Delta=1, max_iter=100, tol=1e-6):
    """FP for 2 agents with different phi.

    Each iteration:
    1. Agent 1: given agent 2's quotes, solve for V_1 and best-respond
    2. Agent 2: given agent 1's quotes, solve for V_2 and best-respond
    3. Check convergence
    """
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)

    # Initialize — both agents with monopolist quotes
    delta_a_1 = np.ones(nq) * 0.8; delta_a_1[0] = 0.0
    delta_b_1 = np.ones(nq) * 0.8; delta_b_1[-1] = 0.0
    delta_a_2 = delta_a_1.copy(); delta_b_2 = delta_b_1.copy()

    history = []
    for it in range(max_iter):
        # Agent 1 best-responds to agent 2
        V_1 = policy_evaluation_hetero(delta_a_1, delta_b_1, delta_a_2, delta_b_2, phi_1, Q, Delta)
        avg_a_2 = np.mean(delta_a_2); avg_b_2 = np.mean(delta_b_2)
        new_da_1, new_db_1 = best_response_hetero(V_1, avg_a_2, avg_b_2, Q, Delta)

        # Agent 2 best-responds to agent 1
        V_2 = policy_evaluation_hetero(delta_a_2, delta_b_2, delta_a_1, delta_b_1, phi_2, Q, Delta)
        avg_a_1 = np.mean(delta_a_1); avg_b_1 = np.mean(delta_b_1)
        new_da_2, new_db_2 = best_response_hetero(V_2, avg_a_1, avg_b_1, Q, Delta)

        # Damped update
        damping = 0.5
        diff = 0
        for arr_old, arr_new in [(delta_a_1, new_da_1), (delta_b_1, new_db_1),
                                  (delta_a_2, new_da_2), (delta_b_2, new_db_2)]:
            diff = max(diff, np.max(np.abs(arr_new - arr_old)))
        delta_a_1 = damping * new_da_1 + (1 - damping) * delta_a_1
        delta_b_1 = damping * new_db_1 + (1 - damping) * delta_b_1
        delta_a_2 = damping * new_da_2 + (1 - damping) * delta_a_2
        delta_b_2 = damping * new_db_2 + (1 - damping) * delta_b_2

        mid = nq // 2
        spread_1 = delta_a_1[mid] + delta_b_1[mid]
        spread_2 = delta_a_2[mid] + delta_b_2[mid]
        history.append({"iter": it, "spread_1": float(spread_1), "spread_2": float(spread_2),
                       "V_1_0": float(V_1[mid]), "V_2_0": float(V_2[mid]), "diff": float(diff)})
        if it % 5 == 0 or diff < tol:
            print(f"  iter {it:3d}: spread_1={spread_1:.4f}, spread_2={spread_2:.4f}, "
                  f"V_1(0)={V_1[mid]:.4f}, V_2(0)={V_2[mid]:.4f}, diff={diff:.2e}", flush=True)
        if diff < tol:
            print(f"  Converged at iter {it+1}", flush=True)
            break

    return {
        "phi_1": float(phi_1), "phi_2": float(phi_2),
        "V_1": V_1.tolist(), "V_2": V_2.tolist(),
        "delta_a_1": delta_a_1.tolist(), "delta_b_1": delta_b_1.tolist(),
        "delta_a_2": delta_a_2.tolist(), "delta_b_2": delta_b_2.tolist(),
        "spread_1_q0": float(delta_a_1[nq//2] + delta_b_1[nq//2]),
        "spread_2_q0": float(delta_a_2[nq//2] + delta_b_2[nq//2]),
        "V_1_q0": float(V_1[nq//2]), "V_2_q0": float(V_2[nq//2]),
        "history": history,
    }


# ================= NEURAL HETEROGENEOUS SOLVER =================

def train_direct_V_hetero(phi_1, phi_2, Q=5, Delta=1, lambda_a=2, lambda_b=2, r=0.01,
                          n_fp=40, n_lbfgs=5):
    """Direct-V solver for heterogeneous 2-agent game.

    Learn V_1, V_2 as two separate tabular vectors.
    """
    nq = int(2 * Q / Delta + 1)
    q_grid = np.arange(-Q, Q + Delta, Delta)
    q_tensor = torch.tensor(q_grid, dtype=torch.float64)
    K = nq  # competitor has nq quote levels

    V_1 = nn.Parameter(torch.tensor(-phi_1 * q_grid**2 / r + 16.0, dtype=torch.float64))
    V_2 = nn.Parameter(torch.tensor(-phi_2 * q_grid**2 / r + 16.0, dtype=torch.float64))

    # Initial quotes
    da_1 = np.ones(nq) * 0.8; da_1[0] = 0.0
    db_1 = np.ones(nq) * 0.8; db_1[-1] = 0.0
    da_2 = da_1.copy(); db_2 = db_1.copy()

    def compute_quotes(V_np, avg_a_comp, avg_b_comp):
        da = np.zeros(nq); db = np.zeros(nq)
        for i in range(nq):
            if i > 0:
                p_a = (V_np[i] - V_np[i-1]) / Delta
                da[i] = optimal_quote_foc(p_a, avg_a_comp, K, 2)
            if i < nq - 1:
                p_b = (V_np[i] - V_np[i+1]) / Delta
                db[i] = optimal_quote_foc(p_b, avg_b_comp, K, 2)
        return da, db

    def bellman_residual(V, da_self, db_self, avg_a_comp, avg_b_comp, phi):
        da_t = torch.tensor(da_self, dtype=torch.float64)
        db_t = torch.tensor(db_self, dtype=torch.float64)
        U_a = torch.zeros(nq, dtype=torch.float64)
        U_b = torch.zeros(nq, dtype=torch.float64)
        U_a[1:] = V[:-1] - V[1:]
        U_b[:-1] = V[1:] - V[:-1]

        fa_vals = np.array([cx_exec_prob_np(float(da_self[i]), avg_a_comp, K, 2)
                            for i in range(nq)]) * lambda_a
        fb_vals = np.array([cx_exec_prob_np(float(db_self[i]), avg_b_comp, K, 2)
                            for i in range(nq)]) * lambda_b
        fa = torch.tensor(fa_vals, dtype=torch.float64)
        fb = torch.tensor(fb_vals, dtype=torch.float64)

        can_sell = (q_tensor > -Q).double()
        can_buy = (q_tensor < Q).double()
        profit_a = can_sell * fa * (da_t * Delta + U_a)
        profit_b = can_buy * fb * (db_t * Delta + U_b)
        psi_q = phi * q_tensor**2
        return r * V + psi_q - profit_a - profit_b

    best_spreads = (None, None)
    best_V = (None, None)
    best_diff = float('inf')

    for fp_iter in range(n_fp):
        # Update V_1 given competitor's quotes (agent 2)
        avg_a_2 = float(np.mean(da_2)); avg_b_2 = float(np.mean(db_2))
        for _ in range(n_lbfgs):
            V_np = V_1.detach().numpy()
            da_1, db_1 = compute_quotes(V_np, avg_a_2, avg_b_2)
            opt = torch.optim.LBFGS([V_1], lr=1.0, max_iter=200,
                                    tolerance_grad=1e-12, line_search_fn="strong_wolfe")
            def c1():
                opt.zero_grad()
                res = bellman_residual(V_1, da_1, db_1, avg_a_2, avg_b_2, phi_1)
                loss = torch.sum(res**2)
                loss.backward()
                return loss
            opt.step(c1)

        # Update V_2 given competitor's quotes (agent 1)
        avg_a_1 = float(np.mean(da_1)); avg_b_1 = float(np.mean(db_1))
        for _ in range(n_lbfgs):
            V_np = V_2.detach().numpy()
            da_2, db_2 = compute_quotes(V_np, avg_a_1, avg_b_1)
            opt = torch.optim.LBFGS([V_2], lr=1.0, max_iter=200,
                                    tolerance_grad=1e-12, line_search_fn="strong_wolfe")
            def c2():
                opt.zero_grad()
                res = bellman_residual(V_2, da_2, db_2, avg_a_1, avg_b_1, phi_2)
                loss = torch.sum(res**2)
                loss.backward()
                return loss
            opt.step(c2)

        mid = nq // 2
        s1 = da_1[mid] + db_1[mid]
        s2 = da_2[mid] + db_2[mid]
        if fp_iter % 5 == 0:
            print(f"  FP {fp_iter}: s1={s1:.4f}, s2={s2:.4f}, V_1(0)={V_1[mid].item():.4f}, V_2(0)={V_2[mid].item():.4f}", flush=True)

        if fp_iter > n_fp // 2:
            # Track best
            total_diff = abs(s1 + s2)
            if total_diff < best_diff:
                best_diff = total_diff
                best_spreads = (float(s1), float(s2))
                best_V = (V_1.detach().numpy().copy(), V_2.detach().numpy().copy())

    return {
        "phi_1": phi_1, "phi_2": phi_2,
        "spread_1_q0": best_spreads[0] if best_spreads[0] is not None else float(da_1[nq//2]+db_1[nq//2]),
        "spread_2_q0": best_spreads[1] if best_spreads[1] is not None else float(da_2[nq//2]+db_2[nq//2]),
        "V_1_q0": float(V_1[nq//2].item()),
        "V_2_q0": float(V_2[nq//2].item()),
    }


# ================= MAIN =================

if __name__ == "__main__":
    # Three experiments:
    # 1. Symmetric (phi_1 = phi_2 = 0.005) — should match standard Nash
    # 2. Mild asymmetry (phi_1 = 0.003, phi_2 = 0.007)
    # 3. Strong asymmetry (phi_1 = 0.001, phi_2 = 0.02)

    configs = [
        {"name": "symmetric", "phi_1": 0.005, "phi_2": 0.005},
        {"name": "mild", "phi_1": 0.003, "phi_2": 0.007},
        {"name": "strong", "phi_1": 0.001, "phi_2": 0.020},
    ]

    all_results = []
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"CONFIG: {cfg['name']} (phi_1={cfg['phi_1']}, phi_2={cfg['phi_2']})")
        print(f"{'='*60}", flush=True)

        # Exact
        print("\n--- Exact ---", flush=True)
        exact = fictitious_play_hetero(cfg['phi_1'], cfg['phi_2'])
        print(f"  Exact: s1={exact['spread_1_q0']:.4f}, s2={exact['spread_2_q0']:.4f}")
        print(f"  Exact: V_1={exact['V_1_q0']:.4f}, V_2={exact['V_2_q0']:.4f}")

        # Neural (direct V)
        print("\n--- Neural (direct V) ---", flush=True)
        neural = train_direct_V_hetero(cfg['phi_1'], cfg['phi_2'], n_fp=30, n_lbfgs=3)
        print(f"  Neural: s1={neural['spread_1_q0']:.4f}, s2={neural['spread_2_q0']:.4f}")

        # Comparison
        s1_err = abs(neural['spread_1_q0'] - exact['spread_1_q0']) / exact['spread_1_q0'] * 100
        s2_err = abs(neural['spread_2_q0'] - exact['spread_2_q0']) / exact['spread_2_q0'] * 100
        print(f"\n  Spread 1 error: {s1_err:.3f}%")
        print(f"  Spread 2 error: {s2_err:.3f}%")

        all_results.append({
            "name": cfg['name'], "phi_1": cfg['phi_1'], "phi_2": cfg['phi_2'],
            "exact": {"s1": exact['spread_1_q0'], "s2": exact['spread_2_q0'],
                     "V_1": exact['V_1_q0'], "V_2": exact['V_2_q0']},
            "neural": {"s1": neural['spread_1_q0'], "s2": neural['spread_2_q0']},
            "s1_error_pct": float(s1_err), "s2_error_pct": float(s2_err),
        })

    # Summary
    print(f"\n{'='*60}")
    print("HETEROGENEOUS AGENTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<12s}  {'phi_1':>6s}  {'phi_2':>6s}  {'s1 exact':>9s}  {'s1 neural':>10s}  {'err1':>6s}  {'s2 exact':>9s}  {'s2 neural':>10s}  {'err2':>6s}")
    for r in all_results:
        print(f"{r['name']:<12s}  {r['phi_1']:6.3f}  {r['phi_2']:6.3f}  "
              f"{r['exact']['s1']:9.4f}  {r['neural']['s1']:10.4f}  {r['s1_error_pct']:5.2f}%  "
              f"{r['exact']['s2']:9.4f}  {r['neural']['s2']:10.4f}  {r['s2_error_pct']:5.2f}%")

    with open("results_final/heterogeneous_agents.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved to results_final/heterogeneous_agents.json", flush=True)
    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

#!/usr/bin/env python -u
"""
Short CPU experiments.

1. Boundary fix ablation at Q=5, 10, 20 (direct-V with/without fix)
2. Extended Q scan for direct-V at Q in {3, 5, 7, 10, 15, 20, 30, 50}
"""

import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the working standalone functions
from scripts.q_scaling_direct_v import train_direct_V_lbfgs
from types import SimpleNamespace
from equations.contxiong_exact import ContXiongExact, optimal_quote_foc, cx_exec_prob_np
from scripts.cont_xiong_exact import fictitious_play

print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def run_fp_no_boundary_fix(Q, n_outer, n_lbfgs, damping):
    """Run FP where quotes are computed WITHOUT the boundary fix
    (i.e., boundary quotes computed via FOC like all others).
    """
    config = SimpleNamespace(lambda_a=2.0, lambda_b=2.0, discount_rate=0.01,
                             Delta_q=1.0, q_max=Q, phi=0.005, N_agents=2)
    eqn = ContXiongExact(config)
    nq = eqn.nq

    exact = fictitious_play(N=2, Q=Q, Delta=1, max_iter=200)
    mid = len(exact['V']) // 2
    exact_spread = exact['delta_a'][mid] + exact['delta_b'][mid]

    avg_da, avg_db = 0.75, 0.75
    best_err = float('inf')
    best_result = None

    # Same structure as train_direct_V_lbfgs but with modified compute_quotes
    for outer in range(n_outer):
        q_grid_np = eqn.q_grid
        V_param = nn.Parameter(torch.zeros(nq, dtype=torch.float64))
        with torch.no_grad():
            mono = fictitious_play(N=1, Q=Q, Delta=1, max_iter=50)
            V_init_level = mono['V'][len(mono['V'])//2]
            V_param.copy_(torch.tensor(-eqn.phi * q_grid_np**2 / eqn.r + V_init_level,
                                        dtype=torch.float64))

        # Modified compute_quotes: NO boundary fix
        def compute_quotes(V_np):
            da = np.zeros(nq); db = np.zeros(nq)
            for i in range(nq):
                # Always compute — use p=0 at boundaries (no neighbour to compare)
                p_a = (V_np[i] - V_np[max(0, i-1)]) / eqn.Delta
                p_b = (V_np[i] - V_np[min(nq-1, i+1)]) / eqn.Delta
                da[i] = optimal_quote_foc(p_a, avg_da, eqn.K, eqn.N_agents)
                db[i] = optimal_quote_foc(p_b, avg_db, eqn.K, eqn.N_agents)
            return da, db

        for inner in range(n_lbfgs):
            V_np = V_param.detach().numpy()
            da, db = compute_quotes(V_np)

            def bellman_residual(V):
                q_grid = torch.tensor(q_grid_np, dtype=torch.float64)
                da_t = torch.tensor(da, dtype=torch.float64)
                db_t = torch.tensor(db, dtype=torch.float64)
                U_a = torch.zeros(nq, dtype=torch.float64)
                U_b = torch.zeros(nq, dtype=torch.float64)
                U_a[1:] = V[:-1] - V[1:]
                U_b[:-1] = V[1:] - V[:-1]
                fa_vals = np.array([cx_exec_prob_np(float(da[i]), avg_da, eqn.K, eqn.N_agents)
                                    for i in range(nq)]) * eqn.lambda_a
                fb_vals = np.array([cx_exec_prob_np(float(db[i]), avg_db, eqn.K, eqn.N_agents)
                                    for i in range(nq)]) * eqn.lambda_b
                fa = torch.tensor(fa_vals, dtype=torch.float64)
                fb = torch.tensor(fb_vals, dtype=torch.float64)
                can_sell = (q_grid > -eqn.Q).double()
                can_buy = (q_grid < eqn.Q).double()
                profit_a = can_sell * fa * (da_t * eqn.Delta + U_a)
                profit_b = can_buy * fb * (db_t * eqn.Delta + U_b)
                psi_q = eqn.phi * q_grid**2
                return eqn.r * V + psi_q - profit_a - profit_b

            opt = torch.optim.LBFGS([V_param], lr=1.0, max_iter=200,
                                    tolerance_grad=1e-12, tolerance_change=1e-14,
                                    history_size=50, line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad()
                res = bellman_residual(V_param)
                loss = torch.sum(res**2)
                loss.backward()
                return loss
            loss = opt.step(closure)
            if loss.item() > 1.0:
                break

        V_np = V_param.detach().numpy()
        da, db = compute_quotes(V_np)
        new_da = float(np.mean(da)); new_db = float(np.mean(db))
        avg_da = damping * new_da + (1 - damping) * avg_da
        avg_db = damping * new_db + (1 - damping) * avg_db

        s = da[mid] + db[mid]
        err = abs(s - exact_spread) / exact_spread * 100
        if err < best_err:
            best_err = err
            best_result = {"da": da.copy(), "db": db.copy(), "V": V_np.copy()}

    return {"error_pct": float(best_err), "exact_spread": float(exact_spread),
            "neural_spread": float(best_result['da'][mid] + best_result['db'][mid])}


# ================= EXPERIMENT 1: Boundary fix ablation =================

def exp1_boundary_ablation():
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: Boundary fix ablation")
    print(f"{'='*60}", flush=True)

    # Import the standalone function (which HAS the fix)
    from scripts.q_scaling_direct_v import run_fp_direct_V

    results = []
    for Q in [5, 10, 20]:
        print(f"\n  Q={Q}", flush=True)

        # With fix (standalone implementation)
        r_fix = run_fp_direct_V(Q, n_outer=20 if Q<=10 else 50, n_lbfgs_outer=20 if Q<=10 else 5)

        # Without fix (our local implementation)
        damping = 0.5 if Q <= 10 else 0.05
        n_outer = 20 if Q <= 10 else 50
        n_lbfgs = 20 if Q <= 10 else 5
        r_nofix = run_fp_no_boundary_fix(Q, n_outer, n_lbfgs, damping)

        print(f"  Q={Q} WITH    boundary fix: error={r_fix['error_pct']:.4f}%", flush=True)
        print(f"  Q={Q} WITHOUT boundary fix: error={r_nofix['error_pct']:.4f}%", flush=True)
        ratio = r_nofix['error_pct'] / max(r_fix['error_pct'], 1e-4)
        print(f"  Ratio: {ratio:.0f}x", flush=True)

        results.append({"Q": Q,
                        "with_fix_error": r_fix['error_pct'],
                        "without_fix_error": r_nofix['error_pct']})

    with open("results_final/exp1_boundary_ablation.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Saved to results_final/exp1_boundary_ablation.json", flush=True)
    return results


# ================= EXPERIMENT 2: Extended Q scan =================

def exp2_extended_q():
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Extended Q scan (with boundary fix)")
    print(f"{'='*60}", flush=True)

    from scripts.q_scaling_direct_v import run_fp_direct_V

    Q_values = [3, 5, 7, 10, 15, 20, 30, 50]
    results = []
    for Q in Q_values:
        n_outer = 20 if Q <= 10 else (50 if Q <= 30 else 80)
        n_lbfgs = 20 if Q <= 10 else 5

        print(f"\n  Q={Q}...", flush=True)
        t0 = time.time()
        r = run_fp_direct_V(Q, n_outer=n_outer, n_lbfgs_outer=n_lbfgs)
        elapsed = time.time() - t0
        print(f"  Q={Q:3d}: error={r['error_pct']:.4f}%, time={elapsed:.0f}s", flush=True)
        results.append({"Q": Q, **r, "elapsed": elapsed})

    with open("results_final/exp2_extended_q_scan.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Saved to results_final/exp2_extended_q_scan.json", flush=True)
    return results


# ================= MAIN =================

if __name__ == "__main__":
    try:
        exp1_boundary_ablation()
    except Exception as e:
        print(f"EXP1 FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()

    try:
        exp2_extended_q()
    except Exception as e:
        print(f"EXP2 FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()

    print(f"\nAll done: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

#!/usr/bin/env python -u
"""MV-FP BSDEJ diagnostic — fixed-avg_comp sweep.

Isolate whether the inner BSDE correctly computes best-response at
a fixed avg_comp, by running it at several avg_comp values without
the outer MV-FP loop.  For each avg_comp, compare the extracted
spread against the Bellman-best-response spread computed by direct
policy iteration.

Key question: is the inner BSDE correct at all?  If yes, drift is
MV-FP problem.  If no, inner BSDE is buggy.

Run: python -u scripts/mv_fp_bsdej_vdiag.py
"""

import sys, os, json, time, traceback
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_bsdej_shared import CXBSDEJShared
from scripts.cont_xiong_exact import fictitious_play, policy_evaluation
from equations.contxiong_exact import cx_exec_prob_np, optimal_quote_foc
from scipy.optimize import minimize_scalar

device = torch.device("cpu")
print(f"vdiag started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}",
      flush=True)


def bellman_best_response(avg_comp, N=2, Q=5, Delta=1,
                          lam_a=2.0, lam_b=2.0, r=0.01, phi=0.005,
                          max_iter=60, tol=1e-7):
    """Ground truth: Bellman best-response quotes given competitors quote
    avg_comp (fixed).  Iterate V <- policy_evaluation, quotes <- FOC."""
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)
    K_i = (N - 1) * nq
    psi = lambda q: phi * q ** 2

    delta_a = np.ones(nq) * avg_comp; delta_a[0] = 0.0
    delta_b = np.ones(nq) * avg_comp; delta_b[-1] = 0.0

    for _ in range(max_iter):
        V = policy_evaluation(delta_a, delta_b, N, Q, Delta, lam_a, lam_b, r, psi)
        new_a = np.zeros(nq); new_b = np.zeros(nq)
        for j, q in enumerate(q_grid):
            if j > 0 and q > -Q:
                new_a[j] = optimal_quote_foc((V[j] - V[j-1]) / Delta,
                                              avg_comp, K_i, N)
            if j < nq - 1 and q < Q:
                new_b[j] = optimal_quote_foc((V[j] - V[j+1]) / Delta,
                                              avg_comp, K_i, N)
        diff = max(np.max(np.abs(new_a - delta_a)), np.max(np.abs(new_b - delta_b)))
        delta_a = 0.5 * new_a + 0.5 * delta_a
        delta_b = 0.5 * new_b + 0.5 * delta_b
        if diff < tol: break

    return delta_a, delta_b, V


def extract_quotes(solver, avg_comp):
    q_grid = np.arange(-solver.Q, solver.Q + solver.Delta, solver.Delta)
    das = np.zeros(len(q_grid)); dbs = np.zeros(len(q_grid))
    solver.shared_net.eval()
    with torch.no_grad():
        for i, q in enumerate(q_grid):
            t_n = torch.tensor([[0.0]], dtype=torch.float64, device=solver.device)
            q_n = torch.tensor([[q / solver.Q]], dtype=torch.float64, device=solver.device)
            U = solver.shared_net(t_n, q_n)
            Ua_v, Ub_v = U[0, 0].item(), U[0, 1].item()
            if q > -solver.Q:
                das[i] = minimize_scalar(
                    lambda d: -cx_exec_prob_np(d, avg_comp, solver.K, solver.N) * (d + Ua_v),
                    bounds=(-1, 8), method='bounded').x
            if q < solver.Q:
                dbs[i] = minimize_scalar(
                    lambda d: -cx_exec_prob_np(d, avg_comp, solver.K, solver.N) * (d + Ub_v),
                    bounds=(-1, 8), method='bounded').x
    solver.shared_net.train()
    return das, dbs


def train_inner(solver, n_steps=3000, lr=5e-4):
    all_params = [solver.Y0] + list(solver.shared_net.parameters())
    solver.optimizer = torch.optim.Adam(all_params, lr=lr)
    solver.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        solver.optimizer, T_max=solver.n_iter, eta_min=lr * 0.01)
    best = float('inf')
    for step in range(n_steps):
        q_paths, ea, eb = solver.sample_paths(solver.batch_size)
        Y_T = solver.forward(q_paths, ea, eb)
        q_T = torch.tensor(q_paths[:, -1].reshape(-1, 1),
                           dtype=torch.float64, device=solver.device)
        g_T = solver.terminal_condition(q_T)
        loss = torch.mean((Y_T - g_T) ** 2)
        solver.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(
            list(solver.shared_net.parameters()) + [solver.Y0], max_norm=5.0)
        solver.optimizer.step(); solver.scheduler.step()
        if loss.item() < best: best = loss.item()
        if step % 500 == 0:
            print(f"      step {step:5d}: loss={loss.item():.4e}, best={best:.4e}",
                  flush=True)
    return best


def main():
    # Exact Nash reference
    nash = fictitious_play(N=2, Q=5, Delta=1)
    nash_spread = nash['delta_a'][5] + nash['delta_b'][5]
    nash_avg = float(np.mean(nash['delta_a']))
    print(f"\n  Nash: spread={nash_spread:.4f}, avg_comp={nash_avg:.4f}", flush=True)

    # Sweep avg_comp values
    avg_comps = [0.45, 0.55, 0.60, nash_avg, 0.68, 0.70, 0.75]
    results = []

    for avg_comp in avg_comps:
        print(f"\n{'='*60}\n  avg_comp = {avg_comp:.4f}\n{'='*60}", flush=True)

        # Bellman best-response
        ba_da, ba_db, _ = bellman_best_response(avg_comp)
        ba_spread = ba_da[5] + ba_db[5]
        ba_mean = 0.5 * (np.mean(ba_da) + np.mean(ba_db))
        print(f"  Bellman best-response: spread(0) = {ba_spread:.4f}, "
              f"mean quote = {ba_mean:.4f}", flush=True)

        # Neural BSDE (fresh init, NO warmstart, fixed avg_comp, 3000 steps)
        solver = CXBSDEJShared(
            N=2, Q=5, Delta=1, T=10.0, M=50,
            lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
            device=device, lr=5e-4, n_iter=3000,
            batch_size=128, hidden=128, n_layers=3,
        )
        solver.avg_comp = avg_comp
        # NO warmstart — we want to test whether the inner BSDE alone
        # can find best-response, without Nash leakage.

        t0 = time.time()
        best_loss = train_inner(solver, n_steps=3000, lr=5e-4)
        elapsed = time.time() - t0

        das, dbs = extract_quotes(solver, avg_comp)
        nn_spread = das[5] + dbs[5]
        nn_mean = 0.5 * (np.mean(das) + np.mean(dbs))

        spread_err = abs(nn_spread - ba_spread) / ba_spread * 100
        print(f"  Neural:                spread(0) = {nn_spread:.4f}, "
              f"mean quote = {nn_mean:.4f}")
        print(f"  Spread mismatch vs Bellman: {spread_err:.2f}%, "
              f"BSDE loss = {best_loss:.4e}, time = {elapsed:.0f}s",
              flush=True)

        results.append({
            "avg_comp": float(avg_comp),
            "bellman_spread": float(ba_spread),
            "bellman_mean": float(ba_mean),
            "bellman_da": ba_da.tolist(),
            "bellman_db": ba_db.tolist(),
            "nn_spread": float(nn_spread),
            "nn_mean": float(nn_mean),
            "nn_da": das.tolist(),
            "nn_db": dbs.tolist(),
            "spread_mismatch_pct": float(spread_err),
            "bsde_loss": float(best_loss),
            "time": float(elapsed),
        })

        os.makedirs("results_final", exist_ok=True)
        with open("results_final/mv_fp_bsdej_vdiag_N2.json", "w") as f:
            json.dump({
                "nash_spread": nash_spread, "nash_avg": nash_avg,
                "sweep": results,
            }, f, indent=2, default=float)

    # Summary
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"  {'avg_comp':>8s}  {'Bellman BR':>12s}  {'Neural':>12s}  "
          f"{'err %':>8s}  {'loss':>10s}")
    for r in results:
        print(f"  {r['avg_comp']:>8.4f}  {r['bellman_spread']:>12.4f}  "
              f"{r['nn_spread']:>12.4f}  {r['spread_mismatch_pct']:>8.2f}  "
              f"{r['bsde_loss']:>10.2e}")
    print("", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {e}", flush=True); traceback.print_exc()
    print(f"\nvdiag finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}",
          flush=True)

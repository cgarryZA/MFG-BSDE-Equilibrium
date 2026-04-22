#!/usr/bin/env python -u
"""MV-FP BSDEJ v7 — v2 config with short inner (1500) + low damping (0.15).

Hypothesis: v2's iter 2 was correct; drift to spurious FP came after.
Shorter inner (1500 vs 3000) + slower damping (0.15 vs 0.2) may
preserve the good iter-2 solution without letting Adam find the
spurious minimum.  Cheapest of the variants.

Run: python -u scripts/mv_fp_bsdej_v7.py
"""

import sys, os, json, time, gc, traceback
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_bsdej_shared import CXBSDEJShared
from scripts.cont_xiong_exact import fictitious_play
from equations.contxiong_exact import cx_exec_prob_np
from scipy.optimize import minimize_scalar

device = torch.device("cpu")
print(f"v7 started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}",
      flush=True)


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


def train_inner(solver, n_steps=1500, lr=5e-4):
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
        nn.utils.clip_grad_norm_(list(solver.shared_net.parameters()) + [solver.Y0], max_norm=5.0)
        solver.optimizer.step(); solver.scheduler.step()
        if loss.item() < best: best = loss.item()
        if step % 300 == 0:
            print(f"      inner {step:5d}: loss={loss.item():.4e}, best={best:.4e}",
                  flush=True)
    return best


def run_mv_fp_v7(N=2, n_outer=8, n_inner=1500, damping=0.15, lr=5e-4):
    print(f"\n{'='*60}\nv7: N={N}, n_outer={n_outer}, n_inner={n_inner}, damping={damping}")
    print(f"{'='*60}", flush=True)

    exact = fictitious_play(N=N, Q=5, Delta=1)
    nash_spread = exact['delta_a'][5] + exact['delta_b'][5]
    nash_avg = float(np.mean(exact['delta_a']))
    print(f"  Nash: spread={nash_spread:.4f}, avg_comp={nash_avg:.4f}", flush=True)

    avg_comp = 0.75
    solver = CXBSDEJShared(
        N=N, Q=5, Delta=1, T=10.0, M=50,
        lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
        device=device, lr=lr, n_iter=n_inner,
        batch_size=128, hidden=128, n_layers=3,
    )
    solver.avg_comp = avg_comp
    solver.warmstart_from_bellman(n_pretrain=1500)

    history = []
    for k in range(n_outer):
        solver.avg_comp = avg_comp
        print(f"\n  === Outer {k+1}/{n_outer}: avg_comp = {avg_comp:.4f} ===", flush=True)
        t0 = time.time()
        best_loss = train_inner(solver, n_steps=n_inner, lr=lr)
        inner_time = time.time() - t0

        das, dbs = extract_quotes(solver, avg_comp)
        new_avg = 0.5 * (float(np.mean(das)) + float(np.mean(dbs)))
        spread_q0 = das[5] + dbs[5]
        error = abs(spread_q0 - nash_spread) / nash_spread * 100
        diff = abs(new_avg - avg_comp)
        avg_comp = damping * new_avg + (1 - damping) * avg_comp

        print(f"    bsde_loss={best_loss:.4e}, time={inner_time:.0f}s")
        print(f"    spread(0)={spread_q0:.4f} (Nash {nash_spread:.4f}, err {error:.2f}%)")
        print(f"    avg: new={new_avg:.4f}, damped={avg_comp:.4f}", flush=True)

        history.append({"k": k, "avg_comp_used": float(solver.avg_comp),
                        "new_avg_extracted": float(new_avg),
                        "avg_comp_after_damping": float(avg_comp),
                        "diff": float(diff), "spread_q0": float(spread_q0),
                        "error_pct": float(error), "inner_loss": float(best_loss),
                        "inner_time": float(inner_time)})
        os.makedirs("results_final", exist_ok=True)
        out = {"variant": "v7_short_inner_low_damping",
               "N": N, "nash_spread": nash_spread, "nash_avg": nash_avg,
               "history": history, "final_avg_comp": avg_comp,
               "final_spread": float(spread_q0), "final_error_pct": float(error),
               "final_das": das.tolist(), "final_dbs": dbs.tolist()}
        with open(f"results_final/mv_fp_bsdej_v7_N{N}.json", "w") as f:
            json.dump(out, f, indent=2, default=float)
        if diff < 0.003:
            print(f"  Converged at outer {k+1}", flush=True); break
    return avg_comp, float(spread_q0), float(error), history


if __name__ == "__main__":
    try:
        gc.collect()
        run_mv_fp_v7(N=2, n_outer=8, n_inner=1500, damping=0.15)
    except Exception as e:
        print(f"FAILED: {e}", flush=True); traceback.print_exc()
    print(f"v7 finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}",
          flush=True)

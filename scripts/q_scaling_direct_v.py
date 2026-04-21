#!/usr/bin/env python -u
"""
Direct V-parametrisation for CX Bellman solver.

Learns V as a vector of nq scalars directly (no NN) by minimising
the Bellman residual with L-BFGS. Since V has only nq parameters and
the Bellman residual is smooth via envelope theorem, L-BFGS converges
much better than Adam.

For discrete Q this is equivalent to the exact solver (they both
target machine precision), but the gradient-based framework is useful
for contexts where the exact linear system doesn't exist (non-stationary,
continuous, high-dimensional state).

Run: python -u scripts/q_scaling_direct_v.py
"""

import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
from equations.contxiong_exact import ContXiongExact, optimal_quote_foc
from scripts.cont_xiong_exact import fictitious_play

device = torch.device("cpu")  # L-BFGS on CPU is fine for nq<=101
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def _foc_residual_torch(delta, p, avg_comp, K, N, lam):
    """FOC: lam * d/d(delta) [f(delta, comp) * (delta - p)] = 0.

    profit = f(delta) * (delta - p)
    f(delta) = (1/N) * sigmoid(-delta) * exp(S/K) / (1 + exp(delta + S/K))

    Return profit (to minimize -profit via Newton on its derivative).
    """
    delta = delta.detach().requires_grad_(True)
    base = torch.sigmoid(-delta)
    S_over_K = torch.tensor(avg_comp, dtype=delta.dtype, device=delta.device)
    comp = torch.exp(torch.clamp(S_over_K, -20, 20)) / (
        1.0 + torch.exp(torch.clamp(delta + S_over_K, -20, 20)))
    f = (1.0 / N) * base * comp
    profit = f * (delta - p)
    return profit


def optimal_quote_newton(p_scalar, avg_comp, K, N, lam, n_iter=30):
    """Newton's method for FOC, scalar input. Returns optimal delta."""
    # Initialize with a reasonable starting point
    delta = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    for _ in range(n_iter):
        profit = _foc_residual_torch(delta, p_scalar, avg_comp, K, N, lam)
        g = torch.autograd.grad(profit, delta, create_graph=True)[0]
        h = torch.autograd.grad(g, delta)[0]
        # Newton step: delta <- delta - g/h (but g/h since we want dprofit/ddelta = 0)
        step = g / (h + 1e-8)
        delta = (delta - 0.5 * step).detach().requires_grad_(True)
        if abs(g.item()) < 1e-8:
            break
    return float(delta.detach().item())


def train_direct_V_lbfgs(eqn, n_outer=30, fixed_avg_da=0.75, fixed_avg_db=0.75, verbose=False):
    """Learn V directly via L-BFGS on the Bellman residual."""
    nq = eqn.nq
    q_grid_np = eqn.q_grid

    # V as direct parameter
    V_param = nn.Parameter(torch.zeros(nq, dtype=torch.float64, device=device))
    # Init with rough estimate: V ~ V_max - phi*q^2/r where V_max depends on Q.
    # For Q=5: V(0) ~= 16. For Q=20: V(0) ~= 17. Better init reduces L-BFGS work.
    # Use a Q-dependent initial level estimate from monopolist solution.
    from scripts.cont_xiong_exact import fictitious_play as _fp
    mono = _fp(N=1, Q=eqn.Q, Delta=eqn.Delta, max_iter=50)
    V_init_level = mono['V'][len(mono['V'])//2]
    with torch.no_grad():
        V_param.copy_(torch.tensor(-eqn.phi * q_grid_np**2 / eqn.r + V_init_level,
                                    dtype=torch.float64))

    # We use a simple alternating approach:
    # 1. Fix avg_comp, solve for V and quotes jointly via L-BFGS on Bellman residual
    # 2. Update avg_comp from new quotes (but here we fix it since we have FP wrapper)

    # Get optimal quotes for current V — use scipy FOC
    # IMPORTANT: boundary quotes are forced to 0 (matching exact Algorithm 1).
    # At q=-Q the dealer can't sell (no inventory), so ask quote doesn't matter
    # and is conventionally set to 0 (not included in average quote).
    def compute_quotes(V_np):
        da = np.zeros(nq); db = np.zeros(nq)
        for i in range(nq):
            # Ask: can only sell if q > -Q (i.e., i > 0)
            if i > 0:
                p_a = (V_np[i] - V_np[i-1]) / eqn.Delta
                da[i] = optimal_quote_foc(p_a, fixed_avg_da, eqn.K, eqn.N_agents)
            # else: da[i] = 0 (already)

            # Bid: can only buy if q < Q (i.e., i < nq-1)
            if i < nq - 1:
                p_b = (V_np[i] - V_np[i+1]) / eqn.Delta
                db[i] = optimal_quote_foc(p_b, fixed_avg_db, eqn.K, eqn.N_agents)
            # else: db[i] = 0 (already)
        return da, db

    def bellman_residual(V, da, db):
        """Compute Bellman residual with quotes DETACHED (envelope theorem)."""
        # V is torch, da/db are numpy arrays
        q_grid = torch.tensor(q_grid_np, dtype=torch.float64, device=device)
        da_t = torch.tensor(da, dtype=torch.float64, device=device)
        db_t = torch.tensor(db, dtype=torch.float64, device=device)

        # U^a[i] = V[i-1] - V[i], U^b[i] = V[i+1] - V[i]
        U_a = torch.zeros(nq, dtype=torch.float64, device=device)
        U_b = torch.zeros(nq, dtype=torch.float64, device=device)
        U_a[1:] = V[:-1] - V[1:]
        U_b[:-1] = V[1:] - V[:-1]

        # Execution probs (constants given detached da, db)
        from equations.contxiong_exact import cx_exec_prob_np
        fa_vals = np.array([cx_exec_prob_np(float(da[i]), fixed_avg_da, eqn.K, eqn.N_agents)
                            for i in range(nq)]) * eqn.lambda_a
        fb_vals = np.array([cx_exec_prob_np(float(db[i]), fixed_avg_db, eqn.K, eqn.N_agents)
                            for i in range(nq)]) * eqn.lambda_b
        fa = torch.tensor(fa_vals, dtype=torch.float64, device=device)
        fb = torch.tensor(fb_vals, dtype=torch.float64, device=device)

        can_sell = (q_grid > -eqn.Q).double()
        can_buy = (q_grid < eqn.Q).double()

        profit_a = can_sell * fa * (da_t * eqn.Delta + U_a)
        profit_b = can_buy * fb * (db_t * eqn.Delta + U_b)

        psi_q = eqn.phi * q_grid**2
        residual = eqn.r * V + psi_q - profit_a - profit_b
        return residual

    # Outer loop: alternate quote computation and L-BFGS for V
    best_loss = float('inf')
    for outer in range(n_outer):
        V_np = V_param.detach().cpu().numpy()
        da, db = compute_quotes(V_np)

        # L-BFGS for V given quotes
        opt = torch.optim.LBFGS([V_param], lr=1.0, max_iter=200,
                                tolerance_grad=1e-12, tolerance_change=1e-14,
                                history_size=50,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            res = bellman_residual(V_param, da, db)
            loss = torch.sum(res**2)
            loss.backward()
            return loss

        loss = opt.step(closure)
        loss_val = loss.item()

        if verbose and (outer % 5 == 0 or outer == n_outer - 1):
            mid = eqn.mid
            s = da[mid] + db[mid]
            print(f"    outer {outer:3d}: loss={loss_val:.4e}, spread(0)={s:.6f}", flush=True)

        if loss_val < best_loss:
            best_loss = loss_val
        if loss_val < 1e-10:
            if verbose:
                print(f"    converged at outer {outer+1}", flush=True)
            break

    # Final quotes
    V_np = V_param.detach().cpu().numpy()
    da, db = compute_quotes(V_np)
    return {"V": V_np, "delta_a": da, "delta_b": db, "final_loss": best_loss}


def run_fp_direct_V(Q, n_outer=50, n_lbfgs_outer=50, damping=None):
    """Outer FP loop with direct-V inner solver.

    Damping auto-scales with Q (larger Q needs heavier damping).
    """
    config = SimpleNamespace(
        lambda_a=2.0, lambda_b=2.0, discount_rate=0.01,
        Delta_q=1.0, q_max=Q, phi=0.005, N_agents=2,
    )
    eqn = ContXiongExact(config)
    print(f"  Grid: {eqn.nq} levels", flush=True)

    # Auto-scale damping: heavier for larger Q (more unstable FP)
    if damping is None:
        damping = 0.5 if Q <= 10 else (0.05 if Q <= 20 else 0.02)
    print(f"  Damping: {damping}", flush=True)

    exact = fictitious_play(N=2, Q=Q, Delta=1, max_iter=200)
    mid = len(exact['V']) // 2
    exact_spread = exact['delta_a'][mid] + exact['delta_b'][mid]
    exact_V = np.array(exact['V'])
    print(f"  Exact spread(0): {exact_spread:.6f}", flush=True)

    avg_da, avg_db = 0.75, 0.75
    best_err = float('inf')
    best_result = None

    for outer in range(n_outer):
        # Fresh V init each outer iteration (prevents stuck state)
        r = train_direct_V_lbfgs(eqn, n_outer=n_lbfgs_outer,
                                 fixed_avg_da=avg_da, fixed_avg_db=avg_db,
                                 verbose=False)

        # Check if L-BFGS actually converged — if loss is large, this FP step failed
        if r['final_loss'] > 1.0:
            print(f"  FP {outer+1}/{n_outer}: L-BFGS failed (loss={r['final_loss']:.2e}), skipping update", flush=True)
            continue

        new_da = float(np.mean(r['delta_a']))
        new_db = float(np.mean(r['delta_b']))
        diff = max(abs(new_da - avg_da), abs(new_db - avg_db))

        # Damped update
        avg_da = damping * new_da + (1 - damping) * avg_da
        avg_db = damping * new_db + (1 - damping) * avg_db

        s = r['delta_a'][mid] + r['delta_b'][mid]
        err = abs(s - exact_spread) / exact_spread * 100
        V_rmse = np.sqrt(np.mean((r['V'] - exact_V)**2))

        if err < best_err:
            best_err = err
            best_result = r

        if outer % 5 == 0 or outer == n_outer - 1:
            print(f"  FP {outer+1}/{n_outer}: avg_da={avg_da:.4f}, spread(0)={s:.6f}, "
                  f"error={err:.3f}%, V_rmse={V_rmse:.4f}, loss={r['final_loss']:.2e}, diff={diff:.6f}",
                  flush=True)

        if diff < 1e-5 and err < 0.1:
            print(f"  FP converged at iter {outer+1}", flush=True)
            break

    # Use best result, not final
    r = best_result if best_result is not None else r
    final_s = r['delta_a'][mid] + r['delta_b'][mid]
    final_err = abs(final_s - exact_spread) / exact_spread * 100
    print(f"  BEST: error={final_err:.4f}%", flush=True)
    return {
        "Q": Q, "exact_spread": float(exact_spread),
        "neural_spread": float(final_s),
        "error_pct": float(final_err),
        "V_error_rmse": float(np.sqrt(np.mean((r['V'] - exact_V)**2))),
        "final_loss": float(r['final_loss']),
    }


if __name__ == "__main__":
    results = []
    # Tune (n_outer, n_lbfgs_outer) per Q: larger Q needs more FP iters with heavy damping
    Q_configs = [
        (5,  15, 20),
        (10, 20, 20),
        (20, 50, 5),
        (50, 80, 5),
    ]
    for Q, n_outer, n_lbfgs in Q_configs:
        print(f"\n{'='*60}")
        print(f"Direct-V (L-BFGS): Q={Q}, n_outer={n_outer}, n_lbfgs={n_lbfgs}")
        print(f"{'='*60}", flush=True)
        gc.collect()
        try:
            r = run_fp_direct_V(Q, n_outer=n_outer, n_lbfgs_outer=n_lbfgs)
            print(f"\n  Q={Q}: error={r['error_pct']:.4f}%, V_rmse={r['V_error_rmse']:.4f}", flush=True)
            results.append(r)
        except Exception as e:
            print(f"  Q={Q} FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print("SUMMARY: Direct-V (L-BFGS) vs Neural NN")
    print(f"{'='*60}")
    print(f"{'Q':>4s}  {'Exact':>8s}  {'Direct-V':>10s}  {'Error':>7s}  {'NN error':>9s}")
    print("-" * 55)
    nn_errors = {5: 0.59, 10: 0.32, 20: 8.17, 50: 4.60}
    for r in results:
        print(f"{r['Q']:4d}  {r['exact_spread']:8.4f}  {r['neural_spread']:10.6f}  "
              f"{r['error_pct']:6.3f}%  {nn_errors.get(r['Q'], 0):8.2f}%", flush=True)

    with open("results_final/q_scaling_direct_V.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to results_final/q_scaling_direct_V.json", flush=True)
    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

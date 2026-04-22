#!/usr/bin/env python -u
"""T3 MV-FP diffusion BSDE — fixes bugs 2 & 3 from the audit.

Bug 2 fix: wraps the continuous-inventory diffusion BSDE solver
(solver_cx_bsde_diffusion.CXBSDEDiffusion) in an outer McKean--Vlasov
fictitious-play loop, so we find the Nash fixed point rather than a
best-response at an arbitrary avg_comp.

Bug 3 fix: patches the Z-network to enforce architectural
antisymmetry,
    Z_anti(t, q) = (1/2)(Z_theta(t, q) - Z_theta(t, -q)),
which is exact by symmetry of V(t, q) = V(t, -q) under the
symmetric inventory penalty phi q^2.

Run: python -u scripts/mv_fp_bsde_diffusion_v1.py
"""

import sys, os, json, time, gc, traceback, copy
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_bsde_diffusion import CXBSDEDiffusion, SharedDiffusionNet
from equations.contxiong_exact import cx_exec_prob_np
from scripts.cont_xiong_exact import fictitious_play
from scipy.optimize import minimize_scalar

device = torch.device("cpu")
print(f"diffusion v1 started: "
      f"{__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


# =====================================================================
# Antisymmetric Z wrapper
# =====================================================================

class AntisymmetricZNet(nn.Module):
    """Enforces Z(t, q) = -Z(t, -q) architecturally.

    Wraps a standard SharedDiffusionNet, and on forward computes
        Z(t, q) = (1/2) (phi(t, q) - phi(t, -q)),
    so Z(t, 0) = 0 by construction and antisymmetry is exact.
    """

    def __init__(self, inner_net):
        super().__init__()
        self.inner = inner_net

    def forward(self, t_norm, q_norm):
        z_pos = self.inner(t_norm, q_norm)
        z_neg = self.inner(t_norm, -q_norm)
        return 0.5 * (z_pos - z_neg)


def extract_quotes_diffusion(solver, avg_comp, t_query=0.0):
    """Extract optimal quotes on the inventory grid from the learned Z.

    Mirrors what CXBSDEDiffusion.train() does at the end internally:
      dVdq = Z / sigma_q_init  (constant-quote sigma proxy)
      delta = argmax f(d, avg_comp, K, N) * (d*Delta + U_eff)
    where U_eff_a = -Delta*dVdq, U_eff_b = +Delta*dVdq.
    """
    q_grid = np.arange(-solver.Q, solver.Q + solver.Delta, solver.Delta,
                        dtype=np.float64)
    das = np.zeros(len(q_grid))
    dbs = np.zeros(len(q_grid))
    solver.z_net.eval()
    with torch.no_grad():
        for i, q in enumerate(q_grid):
            t_n = torch.tensor([[t_query / solver.T]], dtype=torch.float64,
                                device=solver.device)
            q_n = torch.tensor([[q / solver.Q]], dtype=torch.float64,
                                device=solver.device)
            Z_val = solver.z_net(t_n, q_n).item()
            dVdq = Z_val / max(solver.sigma_q_init, 1e-4)
            U_eff_a = -solver.Delta * dVdq
            U_eff_b = solver.Delta * dVdq

            def _neg(d, U_v):
                f = cx_exec_prob_np(d, avg_comp, solver.K, solver.N)
                return -f * (d * solver.Delta + U_v)

            if q > -solver.Q:
                das[i] = minimize_scalar(
                    lambda d: _neg(d, U_eff_a),
                    bounds=(-1, 8), method='bounded').x
            if q < solver.Q:
                dbs[i] = minimize_scalar(
                    lambda d: _neg(d, U_eff_b),
                    bounds=(-1, 8), method='bounded').x
    solver.z_net.train()
    return das, dbs


def patch_solver_with_antisym(solver):
    """Replace solver.z_net with an antisymmetric wrapper that shares
    the underlying parameters.  Optimizer must be rebuilt afterwards."""
    inner = solver.z_net
    solver.z_net = AntisymmetricZNet(inner).to(solver.device)
    # Rebuild optimizer so it sees the (same) parameters through the wrapper
    all_params = [solver.Y0] + list(solver.z_net.parameters())
    solver.optimizer = torch.optim.Adam(all_params, lr=5e-4)
    solver.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        solver.optimizer, T_max=solver.n_iter, eta_min=5e-6)


# =====================================================================
# MV-FP outer loop
# =====================================================================

def run_mv_fp_diffusion(N=2, n_outer=6, n_inner=3000, damping=0.2,
                         lr=5e-4, batch_size=256,
                         persist_optimizer=True):
    print(f"\n{'='*60}")
    print(f"T3 MV-FP diffusion: N={N}, n_outer={n_outer}, n_inner={n_inner}")
    print(f"{'='*60}", flush=True)

    exact = fictitious_play(N=N, Q=5, Delta=1)
    nash_spread = exact['delta_a'][5] + exact['delta_b'][5]
    nash_avg = float(np.mean(exact['delta_a']))
    print(f"  Nash: spread={nash_spread:.4f}, avg_comp={nash_avg:.4f}",
          flush=True)

    avg_comp = 0.75
    solver = CXBSDEDiffusion(
        N=N, Q=5, Delta=1, T=10.0, M=50,
        lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
        device=device, lr=lr, n_iter=n_inner,
        batch_size=batch_size, hidden=128, n_layers=3,
    )
    solver.avg_comp = avg_comp

    # Warmstart the inner network BEFORE patching; patching wraps
    # an already-pretrained inner.  Warmstart uses Nash V but the
    # outer MV-FP will carry the quotes to the correct branch if
    # the inner is correct.
    solver.warmstart_from_bellman(n_pretrain=1500)

    # Apply architectural antisymmetry
    patch_solver_with_antisym(solver)
    print(f"  Z-net antisymmetrised; param count = "
          f"{sum(p.numel() for p in solver.z_net.parameters())}", flush=True)

    history = []
    # Initial optimizer / scheduler (built once if persisting)
    if persist_optimizer:
        all_params = [solver.Y0] + list(solver.z_net.parameters())
        solver.optimizer = torch.optim.Adam(all_params, lr=lr)
        total_steps = n_outer * n_inner
        solver.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            solver.optimizer, T_max=total_steps, eta_min=lr * 0.01)

    for k in range(n_outer):
        solver.avg_comp = avg_comp
        print(f"\n  === Outer iter {k+1}/{n_outer}: avg_comp = {avg_comp:.4f} ===",
              flush=True)

        if not persist_optimizer:
            all_params = [solver.Y0] + list(solver.z_net.parameters())
            solver.optimizer = torch.optim.Adam(all_params, lr=lr)
            solver.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                solver.optimizer, T_max=n_inner, eta_min=lr * 0.01)

        t0 = time.time()
        # Custom inner loop (don't use solver.train which has its own loop)
        best_loss = float('inf')
        for step in range(n_inner):
            Y_T, q_T = solver.forward(batch_size=batch_size)
            g_T = solver.terminal_condition(q_T)
            loss = torch.mean((Y_T - g_T) ** 2)
            solver.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(solver.z_net.parameters()) + [solver.Y0], max_norm=5.0)
            solver.optimizer.step()
            solver.scheduler.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
            if step % 500 == 0:
                print(f"      step {step:5d}: loss={loss.item():.4e}, "
                      f"best={best_loss:.4e}, Y0(q=0)={solver.Y0[solver.Q].item():.4f}",
                      flush=True)
        inner_time = time.time() - t0

        das, dbs = extract_quotes_diffusion(solver, avg_comp, t_query=0.0)
        new_avg = 0.5 * (float(np.mean(das)) + float(np.mean(dbs)))
        spread_q0 = das[5] + dbs[5]
        error = abs(spread_q0 - nash_spread) / nash_spread * 100
        diff = abs(new_avg - avg_comp)
        avg_comp = damping * new_avg + (1 - damping) * avg_comp

        # Also report antisymmetry and stationarity diagnostics
        with torch.no_grad():
            t0_t = torch.zeros(1, 1, dtype=torch.float64, device=device)
            q_plus = torch.tensor([[1.0]], dtype=torch.float64, device=device)
            q_minus = torch.tensor([[-1.0]], dtype=torch.float64, device=device)
            Z_pos = solver.z_net(t0_t, q_plus).item()
            Z_neg = solver.z_net(t0_t, q_minus).item()
            Z_zero = solver.z_net(t0_t, torch.zeros(1, 1, dtype=torch.float64)).item()

        print(f"    bsde_loss={best_loss:.4e}, time={inner_time:.0f}s")
        print(f"    spread(0)={spread_q0:.4f} (Nash {nash_spread:.4f}, err {error:.2f}%)")
        print(f"    Z(+Q)={Z_pos:+.4f}, Z(-Q)={Z_neg:+.4f}, "
              f"Z(0)={Z_zero:+.4e}  (antisym sum: {Z_pos + Z_neg:+.4e})")
        print(f"    avg: new={new_avg:.4f}, damped={avg_comp:.4f}, "
              f"diff={diff:.4f} (target {nash_avg:.4f})", flush=True)

        history.append({
            "k": k, "avg_comp_used": float(solver.avg_comp),
            "new_avg_extracted": float(new_avg),
            "avg_comp_after_damping": float(avg_comp),
            "diff": float(diff), "spread_q0": float(spread_q0),
            "error_pct": float(error), "inner_loss": float(best_loss),
            "inner_time": float(inner_time),
            "Z_plus_Q": float(Z_pos), "Z_minus_Q": float(Z_neg),
            "Z_zero": float(Z_zero),
        })
        os.makedirs("results_final", exist_ok=True)
        out = {
            "variant": "diffusion_MVFP_antisym_v1",
            "N": N, "nash_spread": nash_spread, "nash_avg": nash_avg,
            "history": history, "final_avg_comp": avg_comp,
            "final_spread": float(spread_q0), "final_error_pct": float(error),
            "final_das": das.tolist(), "final_dbs": dbs.tolist(),
        }
        with open(f"results_final/mv_fp_bsde_diffusion_v1_N{N}.json", "w") as f:
            json.dump(out, f, indent=2, default=float)
        if diff < 0.003:
            print(f"  Converged at outer iter {k+1}", flush=True); break

    return avg_comp, float(spread_q0), float(error), history


if __name__ == "__main__":
    try:
        gc.collect()
        run_mv_fp_diffusion(N=2, n_outer=6, n_inner=3000, damping=0.2)
    except Exception as e:
        print(f"FAILED: {e}", flush=True); traceback.print_exc()
    print(f"\ndiffusion v1 finished: "
          f"{__import__('datetime').datetime.now().strftime('%H:%M:%S')}",
          flush=True)

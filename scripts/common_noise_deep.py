#!/usr/bin/env python -u
"""Deep common noise analysis.

Longer training (15k iters), 5-value S grid, 3 kappa values for sensitivity.
Extracts the Z_S surface and how it varies with intensity coupling.

Key questions:
  1. Does Z_S change sign across (q, S)? What's its shape?
  2. How does kappa (intensity coupling strength) affect Z_S magnitude?
  3. Are spreads S-sensitive or approximately invariant?
  4. At zero kappa (standard CX), does Z_S -> 0 as expected?

GPU, ~3h.
"""

import sys, os, json, time, gc
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common_noise import CXCommonNoiseSolver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def run_kappa(kappa, n_iter=15000):
    """Train common-noise BSDEJ at given kappa (intensity-modulation strength)."""
    print(f"\n{'='*60}")
    print(f"Training common noise BSDEJ at kappa = {kappa}")
    print(f"{'='*60}", flush=True)

    gpu_reset()
    solver = CXCommonNoiseSolver(
        N=2, Q=5, Delta=1,
        T=5.0, M=30,
        sigma_S=0.3, kappa=kappa, S_0=1.0, S_scale=1.0,
        device=device, lr=5e-4, n_iter=n_iter, batch_size=256,
        hidden=128, n_layers=3,
    )

    # Patch train to use the new early stopping interface
    from utils import EarlyStopping
    start = time.time()
    history = []
    best_loss = float('inf')
    es = EarlyStopping(patience=1000, min_delta=1e-7, warmup=3000)

    for step in range(solver.n_iter):
        qp, Sp, ea, eb, dws = solver.sample_paths(solver.batch_size)
        Y_T = solver.forward(qp, Sp, ea, eb, dws)
        q_T = torch.tensor(qp[:, -1].reshape(-1, 1), dtype=torch.float64, device=device)
        S_T = torch.tensor(Sp[:, -1].reshape(-1, 1), dtype=torch.float64, device=device)
        g_T = solver.terminal_condition(q_T, S_T)
        loss = torch.mean((Y_T - g_T) ** 2)

        solver.optimizer.zero_grad()
        loss.backward()
        import torch.nn as nn
        nn.utils.clip_grad_norm_(
            list(solver.net.parameters()) + [solver.Y0], max_norm=5.0)
        solver.optimizer.step()
        solver.scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if step % 500 == 0 or step == solver.n_iter - 1:
            y0_q0 = solver.Y0[solver.Q].item()
            print(f"  step {step:5d}: loss={loss.item():.4e}, best={best_loss:.4e}, "
                  f"Y0(q=0)={y0_q0:.4f}", flush=True)
            history.append({"step": step, "loss": loss.item(),
                            "best_loss": best_loss, "Y0_q0": y0_q0})

        if es(loss.item()):
            print(f"  Early stopped at step {step} (best={es.best_loss:.4e})", flush=True)
            break

    elapsed = time.time() - start

    # Extract Z_S surface across a denser S grid
    from scipy.optimize import minimize_scalar
    from equations.contxiong_exact import cx_exec_prob_np

    q_grid = np.arange(-solver.Q, solver.Q + solver.Delta, solver.Delta)
    S_vals = [0.4, 0.7, 1.0, 1.3, 1.6]  # 5-point grid (vs 3 before)

    profiles = {}
    solver.net.eval()
    with torch.no_grad():
        for S_val in S_vals:
            profile = []
            for q in q_grid:
                t_n = torch.tensor([[0.0]], dtype=torch.float64, device=device)
                q_n = torch.tensor([[q/solver.Q]], dtype=torch.float64, device=device)
                S_n = torch.tensor([[(S_val-solver.S_0)/solver.S_scale]],
                                   dtype=torch.float64, device=device)
                out = solver.net(t_n, q_n, S_n)
                Ua_v = out[0, 0].item(); Ub_v = out[0, 1].item(); Z_v = out[0, 2].item()

                def _neg(d, Uv):
                    f = cx_exec_prob_np(d, solver.avg_comp, solver.K, solver.N)
                    return -f * (d + Uv)
                da = minimize_scalar(lambda d: _neg(d, Ua_v), bounds=(-1, 8), method='bounded').x
                db = minimize_scalar(lambda d: _neg(d, Ub_v), bounds=(-1, 8), method='bounded').x
                profile.append({"q": float(q), "Ua": Ua_v, "Ub": Ub_v, "Z_S": Z_v,
                                "da": da, "db": db, "spread": da + db})
            profiles[f"S={S_val:.2f}"] = profile

    result = {
        "kappa": kappa,
        "profiles_by_S": profiles,
        "S_values": S_vals,
        "history": history,
        "elapsed": elapsed,
        "best_loss": best_loss,
        "sigma_S": solver.sigma_S,
    }

    del solver; gpu_reset()
    return result


if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print("COMMON NOISE DEEP DIVE")
    print(f"{'#'*60}", flush=True)

    all_results = {}
    for kappa in [0.0, 0.3, 0.6]:  # 0 = standard CX (Z_S should be ~0), then increasing coupling
        r = run_kappa(kappa, n_iter=15000)
        all_results[f"kappa={kappa}"] = r
        # Save incrementally
        with open("results_final/common_noise_deep.json", "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"  Saved kappa={kappa} incrementally.", flush=True)

    # Summary
    print(f"\n{'='*60}")
    print("COMMON NOISE DEEP SUMMARY")
    print(f"{'='*60}", flush=True)
    print(f"{'kappa':>6s}  {'loss':>10s}  {'Z_S(0,S=1)':>12s}  {'spread(0,S=1)':>14s}  {'time':>7s}")
    for key, r in all_results.items():
        profile_S1 = r["profiles_by_S"].get("S=1.00", [])
        if profile_S1:
            Z_q0 = profile_S1[5]["Z_S"]
            spread_q0 = profile_S1[5]["spread"]
            print(f"{r['kappa']:6.2f}  {r['best_loss']:10.2e}  {Z_q0:+12.4f}  "
                  f"{spread_q0:14.4f}  {r['elapsed']/60:6.1f}m")

    print(f"\nFinished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

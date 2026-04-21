#!/usr/bin/env python -u
"""Non-stationary BSDEJ with alternative phi(t) profiles.

Current: phi rises linearly (terminal liquidation effect).
Tests:
  1. Decreasing phi(t): dealer faces LOW risk aversion at terminal
  2. Constant phi(t): sanity check — should recover stationary
  3. V-shape phi(t): low midday, high at open + close (end-of-day effect)

Each run: ~40 min GPU. Total ~2h.
"""

import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.nonstationary_phi import CXBSDEJNonStationaryPhi
from utils import EarlyStopping

# Force CPU — GPU consistently crashes on this setup (CUDA memory issue)
device = torch.device("cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


class CustomPhiSolver(CXBSDEJNonStationaryPhi):
    """Variant that accepts a custom phi_t function."""
    def __init__(self, phi_fn, phi_end_override=None, **kwargs):
        # Base initialization with phi_0 (won't use phi_end for profile)
        super().__init__(phi_0=0.005, phi_end=phi_end_override or 0.025, **kwargs)
        self.phi_fn = phi_fn

    def phi_t(self, t_frac):
        return self.phi_fn(t_frac)


def make_profile(name, phi_fn, phi_end_terminal, n_iter=8000):
    print(f"\n{'='*60}")
    print(f"Profile: {name}")
    print(f"{'='*60}", flush=True)
    gpu_reset()

    solver = CustomPhiSolver(
        phi_fn=phi_fn, phi_end_override=phi_end_terminal,
        N=2, Q=5, Delta=1, T=10.0, M=50,
        lambda_a=2.0, lambda_b=2.0, r=0.01,
        device=device, lr=5e-4, n_iter=n_iter, batch_size=128,  # smaller for CPU
        hidden=128, n_layers=3,
    )
    solver.warmstart_from_bellman(n_pretrain=1500)

    start = time.time()
    best = float('inf')
    es = EarlyStopping(patience=800, min_delta=1e-7, warmup=2000)

    for step in range(solver.n_iter):
        q_paths, ea, eb = solver.sample_paths(solver.batch_size)
        Y_T = solver.forward(q_paths, ea, eb)
        q_T = torch.tensor(q_paths[:, -1].reshape(-1, 1), dtype=torch.float64, device=device)
        g_T = solver.terminal_condition(q_T)
        loss = torch.mean((Y_T - g_T) ** 2)

        solver.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(solver.shared_net.parameters()) + [solver.Y0], max_norm=5.0)
        solver.optimizer.step()
        solver.scheduler.step()

        if loss.item() < best: best = loss.item()
        if step % 500 == 0:
            print(f"  step {step:5d}: loss={loss.item():.4e}, best={best:.4e}", flush=True)
        if es(loss.item()):
            print(f"  Early stopped at step {step}", flush=True)
            break

    elapsed = time.time() - start

    # Extract quote profile at t=0, 0.25, 0.5, 0.75, 0.99
    from scipy.optimize import minimize_scalar
    from equations.contxiong_exact import cx_exec_prob_np

    q_grid = np.arange(-solver.Q, solver.Q + solver.Delta, solver.Delta)
    time_profiles = {}
    solver.shared_net.eval()

    for t_frac in [0.0, 0.25, 0.5, 0.75, 0.99]:
        phi_m = phi_fn(t_frac)
        profile = []
        with torch.no_grad():
            for q in q_grid:
                t_n = torch.tensor([[t_frac]], dtype=torch.float64, device=device)
                q_n = torch.tensor([[q / solver.Q]], dtype=torch.float64, device=device)
                U = solver.shared_net(t_n, q_n)
                Ua_v = U[0, 0].item()
                Ub_v = U[0, 1].item()

                def _neg(d, Uv):
                    f = cx_exec_prob_np(d, solver.avg_comp, solver.K, solver.N)
                    return -f * (d + Uv)
                da = minimize_scalar(lambda d: _neg(d, Ua_v), bounds=(-1, 8),
                                     method='bounded').x
                db = minimize_scalar(lambda d: _neg(d, Ub_v), bounds=(-1, 8),
                                     method='bounded').x
                profile.append({"q": float(q), "da": float(da), "db": float(db),
                               "spread": float(da + db)})
        mid = len(q_grid) // 2
        s0 = profile[mid]["spread"]
        print(f"  t/T={t_frac:.2f}, phi={phi_m:.4f}: spread(q=0)={s0:.4f}", flush=True)
        time_profiles[f"t={t_frac:.2f}"] = {"phi": float(phi_m), "profile": profile}

    del solver; gpu_reset()
    return {
        "name": name,
        "phi_end": float(phi_end_terminal),
        "time_profiles": time_profiles,
        "best_loss": float(best),
        "elapsed": float(elapsed),
    }


if __name__ == "__main__":
    all_results = {}

    # Profile 1: Decreasing phi (risk aversion FALLS over time)
    phi_decreasing = lambda t: 0.025 - 0.020 * t  # 0.025 -> 0.005
    r1 = make_profile("decreasing", phi_decreasing, phi_end_terminal=0.005, n_iter=4000)
    all_results["decreasing"] = r1
    with open("results_final/nonstationary_alt_profiles.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"  Saved profile 1 incrementally", flush=True)

    # Profile 2: Constant phi (sanity check)
    phi_constant = lambda t: 0.005
    r2 = make_profile("constant", phi_constant, phi_end_terminal=0.005, n_iter=4000)
    all_results["constant"] = r2
    with open("results_final/nonstationary_alt_profiles.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"  Saved profile 2 incrementally", flush=True)

    # Profile 3: V-shape (high at open + close, low midday)
    def phi_vshape(t):
        # min at t=0.5, max at t=0 and t=1
        return 0.005 + 0.020 * abs(2 * t - 1)
    r3 = make_profile("vshape", phi_vshape, phi_end_terminal=0.025, n_iter=4000)
    all_results["vshape"] = r3
    with open("results_final/nonstationary_alt_profiles.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"  Saved profile 3 incrementally", flush=True)

    # Summary
    print(f"\n{'='*60}")
    print("ALT PROFILES SUMMARY: spread(q=0) at each t/T")
    print(f"{'='*60}")
    print(f"{'profile':>12s}  {'t=0':>8s}  {'t=0.25':>8s}  {'t=0.5':>8s}  {'t=0.75':>8s}  {'t=0.99':>8s}")
    for name, r in all_results.items():
        vals = []
        for t in ["t=0.00", "t=0.25", "t=0.50", "t=0.75", "t=0.99"]:
            profile = r['time_profiles'][t]['profile']
            mid = len(profile) // 2
            vals.append(profile[mid]['spread'])
        print(f"{name:>12s}  {vals[0]:8.4f}  {vals[1]:8.4f}  {vals[2]:8.4f}  {vals[3]:8.4f}  {vals[4]:8.4f}")

    print(f"\nFinished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

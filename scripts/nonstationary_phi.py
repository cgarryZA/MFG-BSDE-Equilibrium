#!/usr/bin/env python -u
"""
Non-stationary BSDEJ: time-varying inventory penalty phi(t).

Models intraday risk aversion: dealers become more risk-averse near
close-of-trading (phi rises), forcing them to unload inventory.

Previous lambda(t) test showed quotes barely varied. This tests
whether phi(t) produces meaningful time-dependent quoting behaviour.

Run: python -u scripts/nonstationary_phi.py
"""

import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_bsdej_shared import CXBSDEJShared, SharedJumpNet
from solver_cx_bsdej import _exec_prob_torch_vec, optimal_quotes_vectorised
from equations.contxiong_exact import cx_exec_prob_np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


class CXBSDEJNonStationaryPhi(CXBSDEJShared):
    """Time-varying inventory penalty: phi(t) = phi_0 * (1 + k*t/T).

    Near t=T (end of day), penalty is larger — dealers reduce exposure.
    """

    def __init__(self, phi_0=0.005, phi_end=0.025, **kwargs):
        # Pass base phi to parent
        super().__init__(phi=phi_0, **kwargs)
        self.phi_0 = phi_0
        self.phi_end = phi_end

    def phi_t(self, t_frac):
        """Inventory penalty at time fraction t/T in [0, 1]."""
        return self.phi_0 + (self.phi_end - self.phi_0) * t_frac

    def forward(self, q_paths, exec_a_all, exec_b_all):
        batch = q_paths.shape[0]
        dtype = torch.float64
        dev = self.device

        q0 = torch.tensor(q_paths[:, 0], dtype=dtype, device=dev).long()
        q0_idx = (q0 + self.Q).clamp(0, self.nq - 1)
        Y = self.Y0[q0_idx].unsqueeze(1)

        for m in range(self.M):
            t_frac = m / self.M
            phi_m = self.phi_t(t_frac)  # time-varying penalty

            t_norm = torch.full((batch, 1), t_frac, dtype=dtype, device=dev)
            q_m_raw = torch.tensor(q_paths[:, m], dtype=dtype, device=dev)
            q_m_norm = (q_m_raw / self.Q).unsqueeze(1)

            U = self.shared_net(t_norm, q_m_norm)
            Ua = U[:, 0:1]; Ub = U[:, 1:2]

            da_t = optimal_quotes_vectorised(Ua, self.avg_comp, self.K, self.N)
            db_t = optimal_quotes_vectorised(Ub, self.avg_comp, self.K, self.N)

            fa = _exec_prob_torch_vec(da_t, self.avg_comp, self.K, self.N) * self.lambda_a
            fb = _exec_prob_torch_vec(db_t, self.avg_comp, self.K, self.N) * self.lambda_b

            can_sell = (q_m_raw > -self.Q).float().unsqueeze(1)
            can_buy = (q_m_raw < self.Q).float().unsqueeze(1)

            profit_a = can_sell * fa * (da_t * self.Delta + Ua)
            profit_b = can_buy * fb * (db_t * self.Delta + Ub)

            # Time-varying psi
            psi_q = phi_m * q_m_raw.unsqueeze(1) ** 2

            f_val = self.r * Y + psi_q - profit_a - profit_b

            dN_a = torch.tensor(exec_a_all[:, m].reshape(-1, 1), dtype=dtype, device=dev)
            dN_b = torch.tensor(exec_b_all[:, m].reshape(-1, 1), dtype=dtype, device=dev)
            nu_a = can_sell * fa
            nu_b = can_buy * fb
            jump_a = can_sell * Ua * (dN_a - nu_a * self.dt)
            jump_b = can_buy * Ub * (dN_b - nu_b * self.dt)

            Y = Y - f_val * self.dt + jump_a + jump_b

        return Y

    def terminal_condition(self, q):
        """g(q_T) = -phi_T * q_T² (end-of-day penalty)."""
        return -self.phi_end * q ** 2


if __name__ == "__main__":
    print("=" * 60)
    print("Non-stationary BSDEJ: time-varying phi(t)")
    print("phi(t) = 0.005 + 0.02 * t/T  (risk aversion 5x at close)")
    print("=" * 60, flush=True)

    solver = CXBSDEJNonStationaryPhi(
        phi_0=0.005, phi_end=0.025,
        N=2, Q=5, Delta=1, T=10.0, M=50,
        lambda_a=2.0, lambda_b=2.0, r=0.01,
        device=device, lr=5e-4, n_iter=10000,
        batch_size=512, hidden=128, n_layers=3,
    )

    solver.warmstart_from_bellman(n_pretrain=2000)
    print(flush=True)
    result = solver.train()

    # Extract quote profiles at multiple time points
    from scipy.optimize import minimize_scalar
    q_grid = np.arange(-solver.Q, solver.Q + solver.Delta, solver.Delta)
    time_profiles = {}

    solver.shared_net.eval()
    for t_frac in [0.0, 0.25, 0.5, 0.75, 0.99]:
        phi_m = solver.phi_t(t_frac)
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

                da = minimize_scalar(lambda d: _neg(d, Ua_v), bounds=(-1, 8), method='bounded').x
                db = minimize_scalar(lambda d: _neg(d, Ub_v), bounds=(-1, 8), method='bounded').x
                profile.append({"q": float(q), "da": float(da), "db": float(db), "spread": float(da + db)})

        mid = len(q_grid) // 2
        s0 = profile[mid]["spread"]
        print(f"  t/T={t_frac:.2f}, phi={phi_m:.4f}: spread(q=0)={s0:.4f}", flush=True)
        time_profiles[f"t={t_frac:.2f}"] = {"phi": phi_m, "profile": profile}

    result["time_profiles"] = time_profiles
    with open("results_final/nonstationary_phi.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved to results_final/nonstationary_phi.json", flush=True)

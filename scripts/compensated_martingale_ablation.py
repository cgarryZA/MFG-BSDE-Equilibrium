#!/usr/bin/env python -u
"""
Ablation: compensated jump martingale (Wang et al. 2023 fix).

Demonstrates the methodological contribution: including the compensated
jump martingale term in the BSDEJ forward propagation is essential.
Without it, the solver converges to the wrong answer.

Runs shared-weight BSDEJ solver in two modes:
  (a) CORRECT: Y_{m+1} = Y_m - f*dt + U*(dN - nu*dt)   [Wang et al. 2023]
  (b) BUGGY:   Y_{m+1} = Y_m - f*dt                     [drops jump terms]

Both with warm-start from Bellman, same architecture, same training.

Run: python -u scripts/compensated_martingale_ablation.py
"""

import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_bsdej_shared import CXBSDEJShared
from equations.contxiong_exact import cx_exec_prob_np
from solver_cx_bsdej import _exec_prob_torch_vec, optimal_quotes_vectorised

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


class CXBSDEJBuggy(CXBSDEJShared):
    """BSDEJ solver that DROPS the compensated jump martingale.

    This is the wrong version — as our diagnostic showed, dropping
    U*(dN - nu*dt) because 'it has zero expectation' defeats the
    deep BSDE method.
    """

    def forward(self, q_paths, exec_a_all, exec_b_all):
        """Forward without jump martingale terms."""
        batch = q_paths.shape[0]
        dtype = torch.float64
        dev = self.device

        q0 = torch.tensor(q_paths[:, 0], dtype=dtype, device=dev).long()
        q0_idx = (q0 + self.Q).clamp(0, self.nq - 1)
        Y = self.Y0[q0_idx].unsqueeze(1)

        for m in range(self.M):
            t_norm = torch.full((batch, 1), m / self.M, dtype=dtype, device=dev)
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
            psi_q = self.phi * q_m_raw.unsqueeze(1) ** 2

            f_val = self.r * Y + psi_q - profit_a - profit_b

            # *** NO JUMP MARTINGALE *** (this is the bug)
            Y = Y - f_val * self.dt

        return Y


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def run_config(solver_cls, label, n_iter=10000):
    print(f"\n{'='*60}")
    print(f"CONFIG: {label}")
    print(f"{'='*60}", flush=True)

    gpu_reset()
    solver = solver_cls(
        N=2, Q=5, Delta=1, T=10.0, M=50,
        lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
        device=device, lr=5e-4, n_iter=n_iter,
        batch_size=512, hidden=128, n_layers=3,
    )
    solver.warmstart_from_bellman(n_pretrain=2000)
    print(flush=True)
    result = solver.train()

    spread = result["U_profile"][5]["spread"]
    error = abs(spread - 1.5153) / 1.5153 * 100
    print(f"\n  {label}: spread={spread:.4f}, error={error:.2f}%, loss={result['best_loss']:.2e}", flush=True)

    with open(f"results_final/ablation_martingale_{label}.json", "w") as f:
        json.dump(result, f, indent=2, default=float)

    del solver; gpu_reset()
    return spread, error, result['best_loss']


results = {}

# Correct version (Wang et al. 2023)
s, e, l = run_config(CXBSDEJShared, "CORRECT")
results["correct"] = {"spread": float(s), "error_pct": float(e), "loss": float(l)}

# Buggy version (drops jump martingale)
s, e, l = run_config(CXBSDEJBuggy, "BUGGY")
results["buggy"] = {"spread": float(s), "error_pct": float(e), "loss": float(l)}

# Summary
print(f"\n{'='*60}")
print("ABLATION SUMMARY")
print(f"{'='*60}")
print(f"  Nash: 1.5153")
print(f"  CORRECT (Wang et al.): spread={results['correct']['spread']:.4f}, error={results['correct']['error_pct']:.2f}%")
print(f"  BUGGY (no martingale): spread={results['buggy']['spread']:.4f}, error={results['buggy']['error_pct']:.2f}%")
print(f"  Difference: {abs(results['correct']['error_pct'] - results['buggy']['error_pct']):.2f} percentage points")

with open("results_final/ablation_martingale_summary.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nFinished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

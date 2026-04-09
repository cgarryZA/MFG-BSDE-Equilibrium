"""
Deep learning solver for the exact Cont-Xiong dealer market model.

Two modes:
1. CXSolver: learns V(q) with self-consistent population (single-pass)
2. CXFictitiousPlay: outer FP loop with neural inner solver

Validates against Algorithm 1 ground truth: spread(0) ≈ 1.478 for N=2.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import gc

from equations.contxiong_exact import (
    ContXiongExact, cx_exec_prob_np, optimal_quote_foc
)


class ValueNet(nn.Module):
    """Small network to approximate V(q) on discrete inventory grid."""

    def __init__(self, hidden=64, dtype=torch.float64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=dtype),
        )

    def forward(self, q_normalised):
        return self.net(q_normalised)


class CXSolver:
    """Neural solver for CX model. Learns V(q) and derives quotes from FOC.

    If fixed_avg_da/db are provided, uses those (for FP outer loop).
    Otherwise, self-consistently updates population averages.
    """

    def __init__(self, eqn, device=None, lr=1e-3, n_iter=5000,
                 fixed_avg_da=None, fixed_avg_db=None, verbose=True):
        self.eqn = eqn
        self.device = device or torch.device("cpu")
        self.n_iter = n_iter
        self.verbose = verbose

        self.value_net = ValueNet(hidden=64).to(self.device)
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)

        self.q_norm = torch.tensor(
            eqn.q_grid / eqn.Q, dtype=torch.float64, device=self.device
        ).unsqueeze(1)

        self.fixed_avg_da = fixed_avg_da
        self.fixed_avg_db = fixed_avg_db

    def get_V(self):
        return self.value_net(self.q_norm).squeeze(1)

    def train(self):
        eqn = self.eqn
        start = time.time()
        history = []

        # Population averages
        if self.fixed_avg_da is not None:
            avg_da = self.fixed_avg_da
            avg_db = self.fixed_avg_db
            update_avg = False
        else:
            avg_da = 0.75
            avg_db = 0.75
            update_avg = True

        for step in range(self.n_iter):
            self.value_net.train()
            V = self.get_V()
            V_np = V.detach().cpu().numpy()

            # Optimal quotes from FOC
            delta_a_np, delta_b_np = eqn.compute_optimal_quotes(V_np, avg_da, avg_db)

            # Update population averages (only if not fixed by outer loop)
            if update_avg:
                avg_da = float(np.mean(delta_a_np[1:]))
                avg_db = float(np.mean(delta_b_np[:-1]))

            delta_a = torch.tensor(delta_a_np, dtype=torch.float64, device=self.device)
            delta_b = torch.tensor(delta_b_np, dtype=torch.float64, device=self.device)

            residuals = eqn.bellman_residual(V, delta_a, delta_b,
                                             torch.tensor(avg_da), torch.tensor(avg_db))
            loss = torch.sum(residuals ** 2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.verbose and (step % 500 == 0 or step == self.n_iter - 1):
                spread_q0 = delta_a_np[eqn.mid] + delta_b_np[eqn.mid]
                v_q0 = V_np[eqn.mid]
                print(f"  step {step}: loss={loss.item():.4e}, V(0)={v_q0:.4f}, "
                      f"spread(0)={spread_q0:.4f}")
                history.append({"step": step, "loss": loss.item(), "V_q0": v_q0,
                                "spread_q0": spread_q0})

        self.value_net.eval()
        with torch.no_grad():
            V_final = self.get_V().cpu().numpy()
        delta_a_final, delta_b_final = eqn.compute_optimal_quotes(V_final, avg_da, avg_db)

        elapsed = time.time() - start

        return {
            "V": V_final.tolist(),
            "delta_a": delta_a_final.tolist(),
            "delta_b": delta_b_final.tolist(),
            "spread": (delta_a_final + delta_b_final).tolist(),
            "q_grid": eqn.q_grid.tolist(),
            "history": history,
            "elapsed": elapsed,
            "avg_da": avg_da,
            "avg_db": avg_db,
        }


class CXFictitiousPlay:
    """Fictitious play for CX model with neural inner solver.

    This mirrors Algorithm 1 from Cont-Xiong but uses a neural network
    instead of direct linear system solve for the inner loop.

    Outer loop:
    1. Given current population average quotes (avg_da, avg_db)
    2. Train neural V(q) with these fixed averages
    3. Compute new quotes from learned V
    4. Update population averages (damped)
    5. Check convergence against exact Algorithm 1 result
    """

    def __init__(self, eqn, device=None, outer_iter=20, inner_iter=3000,
                 lr=1e-3, damping=0.5):
        self.eqn = eqn
        self.device = device or torch.device("cpu")
        self.outer_iter = outer_iter
        self.inner_iter = inner_iter
        self.lr = lr
        self.damping = damping

    def train(self):
        eqn = self.eqn
        history = []
        start = time.time()

        # Initial population: monopolist-level quotes
        avg_da = 0.80
        avg_db = 0.80

        for k in range(self.outer_iter):
            print(f"\n--- FP iteration {k+1}/{self.outer_iter} ---")

            # Clear GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Inner solve: learn V given FIXED population averages
            solver = CXSolver(
                eqn, device=self.device, lr=self.lr, n_iter=self.inner_iter,
                fixed_avg_da=avg_da, fixed_avg_db=avg_db, verbose=False,
            )
            result = solver.train()

            # New quotes from learned V
            new_da = np.array(result["delta_a"])
            new_db = np.array(result["delta_b"])

            # New population averages from learned quotes
            new_avg_da = float(np.mean(new_da[1:]))   # exclude q=-Q
            new_avg_db = float(np.mean(new_db[:-1]))   # exclude q=Q

            # Damped update
            old_avg_da = avg_da
            old_avg_db = avg_db
            avg_da = self.damping * new_avg_da + (1 - self.damping) * avg_da
            avg_db = self.damping * new_avg_db + (1 - self.damping) * avg_db

            # Convergence metric
            diff = max(abs(avg_da - old_avg_da), abs(avg_db - old_avg_db))

            V = np.array(result["V"])
            spread_q0 = new_da[eqn.mid] + new_db[eqn.mid]
            v_q0 = V[eqn.mid]

            entry = {
                "iteration": k + 1,
                "avg_da": avg_da, "avg_db": avg_db,
                "spread_q0": spread_q0, "V_q0": v_q0,
                "diff": diff, "loss": result["history"][-1]["loss"] if result["history"] else 0,
            }
            history.append(entry)
            print(f"  avg_da={avg_da:.4f}, avg_db={avg_db:.4f}, spread(0)={spread_q0:.4f}, "
                  f"V(0)={v_q0:.4f}, diff={diff:.6f}")

            if diff < 1e-4:
                print(f"  Converged at iteration {k+1}")
                break

        elapsed = time.time() - start
        print(f"\n  FP complete in {elapsed:.1f}s")

        return {
            "history": history,
            "final_V": result["V"],
            "final_delta_a": result["delta_a"],
            "final_delta_b": result["delta_b"],
            "final_spread": (np.array(result["delta_a"]) + np.array(result["delta_b"])).tolist(),
            "final_avg_da": avg_da,
            "final_avg_db": avg_db,
            "elapsed": elapsed,
        }

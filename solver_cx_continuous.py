"""
Continuous-inventory CX solver.

Instead of q ∈ {-5,...,5} (11 discrete levels), q ∈ [-5, 5] continuous.

The value function V(q) is learned as a smooth neural network.
Quotes are computed from the FOC at any q, not just grid points.

This is where the deep learning approach adds genuine value:
Algorithm 1 needs a discrete grid (and its cost scales with grid size).
The neural solver works on continuous q with no grid.

The Bellman equation at continuous q:
  r*V(q) + psi(q) = lambda_a * max_delta_a [f_a(da, comp) * (da*Delta - (V(q) - V(q-Delta)))]
                    + lambda_b * max_delta_b [f_b(db, comp) * (db*Delta - (V(q) - V(q+Delta)))]

where the value jumps V(q) - V(q±Delta) are evaluated on the continuous V network.

For validation: compare against Algorithm 1 at the grid points.
The NN should interpolate smoothly between grid points.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import gc

from equations.contxiong_exact import cx_exec_prob_np, cx_exec_prob_torch
from scipy.optimize import minimize_scalar


class ContinuousValueNet(nn.Module):
    """Deeper network for continuous V(q).

    Input: q (normalised to [-1, 1])
    Output: V(q)
    """
    def __init__(self, hidden=128, n_layers=3, dtype=torch.float64):
        super().__init__()
        layers = [nn.Linear(1, hidden, dtype=dtype), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden, dtype=dtype), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, q_norm):
        return self.net(q_norm)


class CXContinuousSolver:
    """CX solver with continuous inventory.

    Samples random q values in [-Q, Q] and minimises the Bellman residual.
    Quotes computed from FOC at each sampled q.
    """

    def __init__(self, N=2, Q=5.0, Delta=1.0, lambda_a=2.0, lambda_b=2.0,
                 r=0.01, phi=0.005, device=None, lr=5e-4, n_iter=10000,
                 batch_size=64):
        self.N = N
        self.Q = Q
        self.Delta = Delta
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.r = r
        self.phi = phi
        self.device = device or torch.device("cpu")
        self.n_iter = n_iter
        self.batch_size = batch_size

        self.K = (N - 1) * 11 if N > 1 else 0  # approx competitor levels

        self.value_net = ContinuousValueNet(hidden=128, n_layers=3).to(self.device)
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_iter, eta_min=1e-5
        )

    def V(self, q_tensor):
        """Evaluate V at arbitrary q values. q_tensor: [batch, 1] in [-Q, Q]."""
        q_norm = q_tensor / self.Q  # normalise to [-1, 1]
        return self.value_net(q_norm)

    def psi(self, q):
        return self.phi * q ** 2

    def optimal_quote_at_q(self, q_val, V_q, V_q_minus, V_q_plus, avg_comp):
        """Compute optimal ask and bid quotes at a single q value.

        Returns (delta_a, delta_b) from FOC maximisation.
        """
        Delta = self.Delta
        p_a = (V_q - V_q_minus) / Delta  # value jump on ask execution
        p_b = (V_q - V_q_plus) / Delta   # value jump on bid execution

        # Ask side
        if q_val > -self.Q:
            def neg_prof_a(delta):
                f = cx_exec_prob_np(delta, avg_comp, self.K, self.N)
                return -(delta - p_a) * f
            res = minimize_scalar(neg_prof_a, bounds=(-2, 10), method='bounded')
            da = res.x
        else:
            da = 0.0

        # Bid side
        if q_val < self.Q:
            def neg_prof_b(delta):
                f = cx_exec_prob_np(delta, avg_comp, self.K, self.N)
                return -(delta - p_b) * f
            res = minimize_scalar(neg_prof_b, bounds=(-2, 10), method='bounded')
            db = res.x
        else:
            db = 0.0

        return da, db

    def bellman_loss(self, q_batch, avg_da, avg_db):
        """Compute Bellman residual at a batch of q values.

        q_batch: [batch, 1] tensor of inventory values
        """
        Delta = self.Delta
        V_q = self.V(q_batch)  # [batch, 1]
        V_q_minus = self.V(torch.clamp(q_batch - Delta, -self.Q, self.Q))
        V_q_plus = self.V(torch.clamp(q_batch + Delta, -self.Q, self.Q))

        # Compute optimal quotes for each q (numpy, via FOC)
        q_np = q_batch.detach().cpu().numpy().flatten()
        V_q_np = V_q.detach().cpu().numpy().flatten()
        V_qm_np = V_q_minus.detach().cpu().numpy().flatten()
        V_qp_np = V_q_plus.detach().cpu().numpy().flatten()

        das = np.zeros(len(q_np))
        dbs = np.zeros(len(q_np))
        for i in range(len(q_np)):
            das[i], dbs[i] = self.optimal_quote_at_q(
                q_np[i], V_q_np[i], V_qm_np[i], V_qp_np[i], avg_da
            )

        da_t = torch.tensor(das, dtype=torch.float64, device=self.device).unsqueeze(1)
        db_t = torch.tensor(dbs, dtype=torch.float64, device=self.device).unsqueeze(1)

        # Execution probabilities (torch, for gradient flow)
        avg_da_t = torch.tensor(avg_da, dtype=torch.float64, device=self.device)
        avg_db_t = torch.tensor(avg_db, dtype=torch.float64, device=self.device)
        fa = cx_exec_prob_torch(da_t, avg_da_t, self.K, self.N)
        fb = cx_exec_prob_torch(db_t, avg_db_t, self.K, self.N)

        # Mask for inventory limits
        can_sell = (q_batch > -self.Q).float()
        can_buy = (q_batch < self.Q).float()

        # Bellman residual: r*V + psi - profits = 0
        psi_q = self.phi * q_batch ** 2
        profit_a = can_sell * self.lambda_a * Delta * fa * (da_t - (V_q - V_q_minus) / Delta)
        profit_b = can_buy * self.lambda_b * Delta * fb * (db_t - (V_q - V_q_plus) / Delta)

        residual = self.r * V_q + psi_q - profit_a - profit_b
        return torch.mean(residual ** 2), das, dbs

    def train(self):
        start = time.time()
        history = []
        avg_da = 0.75
        avg_db = 0.75

        for step in range(self.n_iter):
            self.value_net.train()

            # Sample random q values (mix of uniform + grid points for stability)
            q_uniform = torch.rand(self.batch_size - 11, 1, dtype=torch.float64,
                                   device=self.device) * 2 * self.Q - self.Q
            q_grid = torch.tensor(np.arange(-self.Q, self.Q + 1, 1.0),
                                  dtype=torch.float64, device=self.device).unsqueeze(1)
            q_batch = torch.cat([q_grid, q_uniform], dim=0)

            loss, das, dbs = self.bellman_loss(q_batch, avg_da, avg_db)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Update population averages from grid-point quotes
            avg_da = float(np.mean(das[1:11]))   # grid points excluding q=-Q
            avg_db = float(np.mean(das[:10]))     # grid points excluding q=Q

            if step % 1000 == 0 or step == self.n_iter - 1:
                # Evaluate at q=0
                with torch.no_grad():
                    q0 = torch.tensor([[0.0]], dtype=torch.float64, device=self.device)
                    v0 = self.V(q0).item()
                # Find spread at q=0 (index 5 in the grid points)
                spread_0 = das[5] + dbs[5] if len(das) > 5 else 0
                print(f"  step {step}: loss={loss.item():.4e}, V(0)={v0:.4f}, "
                      f"spread(0)={spread_0:.4f}")
                history.append({"step": step, "loss": loss.item(), "V_q0": v0,
                                "spread_q0": spread_0})

        # Final evaluation on fine grid
        self.value_net.eval()
        q_fine = np.linspace(-self.Q, self.Q, 101)
        V_fine = []
        with torch.no_grad():
            for q in q_fine:
                qt = torch.tensor([[q]], dtype=torch.float64, device=self.device)
                V_fine.append(self.V(qt).item())
        V_fine = np.array(V_fine)

        # Quotes on standard grid
        q_grid = np.arange(-self.Q, self.Q + 1, 1.0)
        da_grid = np.zeros(len(q_grid))
        db_grid = np.zeros(len(q_grid))
        V_grid = np.zeros(len(q_grid))
        with torch.no_grad():
            for j, q in enumerate(q_grid):
                qt = torch.tensor([[q]], dtype=torch.float64, device=self.device)
                V_grid[j] = self.V(qt).item()
                qm = torch.tensor([[max(q - 1, -self.Q)]], dtype=torch.float64, device=self.device)
                qp = torch.tensor([[min(q + 1, self.Q)]], dtype=torch.float64, device=self.device)
                vm = self.V(qm).item()
                vp = self.V(qp).item()
                da_grid[j], db_grid[j] = self.optimal_quote_at_q(q, V_grid[j], vm, vp, avg_da)

        elapsed = time.time() - start
        return {
            "q_grid": q_grid.tolist(),
            "V_grid": V_grid.tolist(),
            "delta_a": da_grid.tolist(),
            "delta_b": db_grid.tolist(),
            "spread": (da_grid + db_grid).tolist(),
            "q_fine": q_fine.tolist(),
            "V_fine": V_fine.tolist(),
            "history": history,
            "elapsed": elapsed,
            "avg_da": avg_da, "avg_db": avg_db,
        }

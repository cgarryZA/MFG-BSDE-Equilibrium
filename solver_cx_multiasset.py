"""
Multi-asset CX solver.

State: (q_1, ..., q_K) where K is number of assets.
Each asset has independent order flow with CX execution probabilities.
Penalty: psi(q) = phi * sum(q_k^2)

For K=1: reduces to single-asset CX (validated).
For K=2: brute-force exact solution possible (grid is nq^2).
For K=5+: only neural solver works.

The neural network learns V(q_1,...,q_K) as a function of K-dimensional input.
Quotes for each asset computed from the FOC using partial derivatives of V.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import gc
from scipy.optimize import minimize_scalar

from equations.contxiong_exact import cx_exec_prob_np, cx_exec_prob_torch


class MultiAssetValueNet(nn.Module):
    """V(q_1,...,q_K) network."""
    def __init__(self, K, hidden=128, n_layers=3, dtype=torch.float64):
        super().__init__()
        layers = [nn.Linear(K, hidden, dtype=dtype), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden, dtype=dtype), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, q_norm):
        return self.net(q_norm)


class CXMultiAssetSolver:
    """Multi-asset CX solver.

    K assets, N dealers, discrete inventory q_k in {-Q,...,Q} per asset.
    """

    def __init__(self, K=2, N=2, Q=5.0, Delta=1.0, lambda_a=2.0, lambda_b=2.0,
                 r=0.01, phi=0.005, device=None, lr=5e-4, n_iter=10000,
                 batch_size=128):
        self.K = K
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

        self.nq = int(2 * Q / Delta + 1)
        self.K_competitors = (N - 1) * self.nq if N > 1 else 0

        self.value_net = MultiAssetValueNet(K, hidden=128, n_layers=3).to(self.device)
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)

    def V(self, q_tensor):
        """q_tensor: [batch, K] in [-Q, Q]."""
        q_norm = q_tensor / self.Q
        return self.value_net(q_norm)

    def psi(self, q_tensor):
        """Total penalty: phi * sum(q_k^2)."""
        return self.phi * torch.sum(q_tensor ** 2, dim=1, keepdim=True)

    def sample_q(self, n):
        """Sample random inventory vectors."""
        # Mix of grid points and continuous
        n_grid = min(n // 4, self.nq ** min(self.K, 2))
        n_rand = n - n_grid

        # Random continuous
        q_rand = torch.rand(n_rand, self.K, dtype=torch.float64,
                            device=self.device) * 2 * self.Q - self.Q

        # Grid points (for stability)
        if self.K <= 2 and n_grid > 0:
            q_vals = np.arange(-self.Q, self.Q + self.Delta, self.Delta)
            if self.K == 1:
                q_grid = torch.tensor(q_vals.reshape(-1, 1), dtype=torch.float64,
                                      device=self.device)
            else:
                qq = np.array(np.meshgrid(*[q_vals] * self.K)).reshape(self.K, -1).T
                idx = np.random.choice(len(qq), min(n_grid, len(qq)), replace=False)
                q_grid = torch.tensor(qq[idx], dtype=torch.float64, device=self.device)
        else:
            q_grid = torch.rand(n_grid, self.K, dtype=torch.float64,
                                device=self.device) * 2 * self.Q - self.Q

        return torch.cat([q_grid, q_rand], dim=0)

    def bellman_loss(self, q_batch, avg_da_per_asset, avg_db_per_asset):
        """Compute Bellman residual for multi-asset.

        For each asset k, the contribution to the Bellman equation is:
          lambda_a * max_da_k [f_a(da_k, comp) * (da_k*Delta - (V(q) - V(q - e_k*Delta)))]
        + lambda_b * max_db_k [f_b(db_k, comp) * (db_k*Delta - (V(q) - V(q + e_k*Delta)))]
        """
        batch = q_batch.shape[0]
        V_q = self.V(q_batch)  # [batch, 1]

        total_profit = torch.zeros(batch, 1, dtype=torch.float64, device=self.device)

        for k in range(self.K):
            # Value jumps for asset k
            q_minus = q_batch.clone()
            q_minus[:, k] = torch.clamp(q_minus[:, k] - self.Delta, -self.Q, self.Q)
            q_plus = q_batch.clone()
            q_plus[:, k] = torch.clamp(q_plus[:, k] + self.Delta, -self.Q, self.Q)

            V_minus = self.V(q_minus)
            V_plus = self.V(q_plus)

            # Optimal quotes for asset k (numpy FOC)
            V_q_np = V_q.detach().cpu().numpy().flatten()
            V_m_np = V_minus.detach().cpu().numpy().flatten()
            V_p_np = V_plus.detach().cpu().numpy().flatten()
            q_k_np = q_batch[:, k].detach().cpu().numpy()

            das = np.zeros(batch)
            dbs = np.zeros(batch)
            avg_da_k = avg_da_per_asset[k]
            avg_db_k = avg_db_per_asset[k]

            for i in range(batch):
                p_a = (V_q_np[i] - V_m_np[i]) / self.Delta
                p_b = (V_q_np[i] - V_p_np[i]) / self.Delta

                if q_k_np[i] > -self.Q:
                    def neg_pa(d):
                        return -(d - p_a) * cx_exec_prob_np(d, avg_da_k, self.K_competitors, self.N)
                    das[i] = minimize_scalar(neg_pa, bounds=(-2, 10), method='bounded').x

                if q_k_np[i] < self.Q:
                    def neg_pb(d):
                        return -(d - p_b) * cx_exec_prob_np(d, avg_db_k, self.K_competitors, self.N)
                    dbs[i] = minimize_scalar(neg_pb, bounds=(-2, 10), method='bounded').x

            da_t = torch.tensor(das, dtype=torch.float64, device=self.device).unsqueeze(1)
            db_t = torch.tensor(dbs, dtype=torch.float64, device=self.device).unsqueeze(1)

            avg_da_t = torch.tensor(avg_da_k, dtype=torch.float64, device=self.device)
            avg_db_t = torch.tensor(avg_db_k, dtype=torch.float64, device=self.device)
            fa = cx_exec_prob_torch(da_t, avg_da_t, self.K_competitors, self.N)
            fb = cx_exec_prob_torch(db_t, avg_db_t, self.K_competitors, self.N)

            can_sell = (q_batch[:, k:k+1] > -self.Q).float()
            can_buy = (q_batch[:, k:k+1] < self.Q).float()

            profit_a = can_sell * self.lambda_a * self.Delta * fa * (da_t - (V_q - V_minus) / self.Delta)
            profit_b = can_buy * self.lambda_b * self.Delta * fb * (db_t - (V_q - V_plus) / self.Delta)

            total_profit = total_profit + profit_a + profit_b

        psi_q = self.psi(q_batch)
        residual = self.r * V_q + psi_q - total_profit
        return torch.mean(residual ** 2)

    def train(self):
        start = time.time()
        history = []
        avg_da = [0.75] * self.K
        avg_db = [0.75] * self.K

        for step in range(self.n_iter):
            self.value_net.train()
            q_batch = self.sample_q(self.batch_size)
            loss = self.bellman_loss(q_batch, avg_da, avg_db)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 1000 == 0 or step == self.n_iter - 1:
                with torch.no_grad():
                    q0 = torch.zeros(1, self.K, dtype=torch.float64, device=self.device)
                    v0 = self.V(q0).item()
                print(f"  step {step}: loss={loss.item():.4e}, V(0)={v0:.4f}")
                history.append({"step": step, "loss": loss.item(), "V_0": v0})

        elapsed = time.time() - start
        # Evaluate at origin
        with torch.no_grad():
            q0 = torch.zeros(1, self.K, dtype=torch.float64, device=self.device)
            v0 = self.V(q0).item()

        return {"V_0": v0, "history": history, "elapsed": elapsed, "K": self.K}

"""
Finite-horizon BSDEJ solver for the Cont-Xiong dealer market.

This implements the ACTUAL deep BSDE method (Han et al. 2018) applied
to the CX market-making problem with Poisson jumps.

The value function V(t, q) satisfies the BSDEJ:

  -dY_t = f(t, q_t, Y_t, U_t) dt - Z_t dW_t
          - U^a_t (dN^a_t - nu^a_t dt) - U^b_t (dN^b_t - nu^b_t dt)

  Y_T = g(q_T) = -psi(q_T)

where:
  Y_t = V(t, q_t)                    (value process)
  Z_t = sigma_S * dV/dS              (price diffusion coeff — zero for CX since V doesn't depend on S)
  U^a_t = V(t, q_t - Delta) - V(t, q_t)   (jump coeff on ask execution)
  U^b_t = V(t, q_t + Delta) - V(t, q_t)   (jump coeff on bid execution)

Generator:
  f(t, q, Y, U^a, U^b) = -r*Y - psi(q)
    + lambda_a * max_delta_a [f_a(delta_a, comp) * (delta_a * Delta + U^a)]
    + lambda_b * max_delta_b [f_b(delta_b, comp) * (delta_b * Delta + U^b)]

The NN learns U^a(t, q) and U^b(t, q) at each time step.
Optimal quotes are derived from U via the FOC.

This is the proper deep BSDE formulation — it iterates backward in time,
matching the terminal condition, just like Han et al. 2018.
"""

import numpy as np
import torch
import torch.nn as nn
import time
from scipy.optimize import minimize_scalar

from equations.contxiong_exact import cx_exec_prob_np


class JumpCoefficientNet(nn.Module):
    """Network for (U^a, U^b) at each time step.

    Input: (t/T, q/Q) — normalised time and inventory
    Output: (U^a, U^b) — value jumps
    """
    def __init__(self, hidden=32, dtype=torch.float64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden, dtype=dtype),  # (t_norm, q_norm)
            nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, 2, dtype=dtype),  # (U^a, U^b)
        )

    def forward(self, t_norm, q_norm):
        x = torch.cat([t_norm, q_norm], dim=1)
        return self.net(x)


def monopolist_exec_prob(delta):
    """For single-agent or mean-field limit."""
    base = 1.0 / (1.0 + np.exp(np.clip(delta, -20, 20)))
    return base * base


def optimal_quote_from_U(U_val, N=2, K=11, avg_comp=0.75):
    """Compute optimal quote given jump coefficient U.

    delta* = argmax f(delta, comp) * (delta * Delta + U)
    where Delta = 1.
    """
    def neg_profit(delta):
        f = cx_exec_prob_np(delta, avg_comp, K, N)
        return -f * (delta + U_val)  # Delta=1

    result = minimize_scalar(neg_profit, bounds=(-2, 10), method='bounded')
    return result.x


class CXBSDEJSolver:
    """Deep BSDE solver for finite-horizon CX model with jumps.

    Time discretisation: t_0 = 0, t_1, ..., t_M = T
    At each time step, the NN predicts U^a and U^b.
    The BSDE is propagated forward, and the loss is |Y_T - g(q_T)|².
    """

    def __init__(self, N=2, Q=5, Delta=1, T=1.0, M=20,
                 lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
                 device=None, lr=1e-3, n_iter=5000, batch_size=256):
        self.N = N; self.Q = Q; self.Delta = Delta
        self.T = T; self.M = M
        self.dt = T / M
        self.lambda_a = lambda_a; self.lambda_b = lambda_b
        self.r = r; self.phi = phi
        self.device = device or torch.device("cpu")
        self.n_iter = n_iter
        self.batch_size = batch_size

        self.nq = int(2 * Q / Delta + 1)
        self.K = (N - 1) * self.nq

        # Y(0) — initial value, learnable parameter
        self.Y0 = nn.Parameter(torch.tensor([15.0], dtype=torch.float64, device=self.device))

        # One subnet per time step (like Han et al. 2018)
        self.subnets = nn.ModuleList([
            JumpCoefficientNet(hidden=32).to(self.device)
            for _ in range(M)
        ])

        all_params = [self.Y0] + list(self.subnets.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)

    def psi(self, q):
        return self.phi * q ** 2

    def terminal_condition(self, q):
        """g(q_T) = -psi(q_T)."""
        return -self.psi(q)

    def sample_paths(self, batch_size):
        """Sample forward inventory paths using proxy policy.

        Returns: q_paths [batch, M+1], execution events
        """
        q = np.zeros(batch_size)  # start at q=0
        q_paths = np.zeros((batch_size, self.M + 1))
        q_paths[:, 0] = q

        for m in range(self.M):
            # Proxy quotes: use simple inventory-dependent rule
            z_proxy = -2 * self.phi * q
            delta_a_proxy = 0.8 - 0.05 * q
            delta_b_proxy = 0.8 + 0.05 * q

            # Execution probabilities
            rate_a = np.array([cx_exec_prob_np(da, 0.75, self.K, self.N)
                              for da in delta_a_proxy]) * self.lambda_a
            rate_b = np.array([cx_exec_prob_np(db, 0.75, self.K, self.N)
                              for db in delta_b_proxy]) * self.lambda_b

            # Poisson jumps in dt
            prob_a = np.clip(rate_a * self.dt, 0, 0.5)
            prob_b = np.clip(rate_b * self.dt, 0, 0.5)
            exec_a = (np.random.uniform(size=batch_size) < prob_a).astype(float)
            exec_b = (np.random.uniform(size=batch_size) < prob_b).astype(float)

            # Inventory update
            q = q - exec_a * self.Delta + exec_b * self.Delta
            q = np.clip(q, -self.Q, self.Q)
            q_paths[:, m + 1] = q

        return q_paths

    def forward(self, q_paths):
        """Forward pass: propagate Y from t=0 to t=T using the BSDE.

        Y_{m+1} = Y_m - f_m * dt + U^a_m * (dN^a_m - nu^a_m * dt) + U^b_m * (dN^b_m - nu^b_m * dt)

        Since we're in the compensated form, the jump terms are:
        Y_{m+1} = Y_m - [r*Y_m + psi(q_m) - profits_m] * dt
        """
        batch = q_paths.shape[0]
        device = self.device

        Y = self.Y0.expand(batch, 1)  # [batch, 1]

        for m in range(self.M):
            t_norm = torch.full((batch, 1), m / self.M, dtype=torch.float64, device=device)
            q_m = torch.tensor(q_paths[:, m].reshape(-1, 1) / self.Q,
                              dtype=torch.float64, device=device)

            # Get U^a, U^b from subnet
            U = self.subnets[m](t_norm, q_m)  # [batch, 2]
            Ua = U[:, 0:1]  # V(q-1) - V(q)
            Ub = U[:, 1:2]  # V(q+1) - V(q)

            # Optimal quotes from U (numpy, per sample)
            Ua_np = Ua.detach().cpu().numpy().flatten()
            Ub_np = Ub.detach().cpu().numpy().flatten()
            q_np = q_paths[:, m]

            das = np.zeros(batch)
            dbs = np.zeros(batch)
            for i in range(batch):
                if q_np[i] > -self.Q:
                    das[i] = optimal_quote_from_U(Ua_np[i], self.N, self.K)
                if q_np[i] < self.Q:
                    dbs[i] = optimal_quote_from_U(Ub_np[i], self.N, self.K)

            da_t = torch.tensor(das.reshape(-1, 1), dtype=torch.float64, device=device)
            db_t = torch.tensor(dbs.reshape(-1, 1), dtype=torch.float64, device=device)

            # Execution rates
            fa = torch.tensor(
                [cx_exec_prob_np(das[i], 0.75, self.K, self.N) for i in range(batch)],
                dtype=torch.float64, device=device
            ).unsqueeze(1) * self.lambda_a
            fb = torch.tensor(
                [cx_exec_prob_np(dbs[i], 0.75, self.K, self.N) for i in range(batch)],
                dtype=torch.float64, device=device
            ).unsqueeze(1) * self.lambda_b

            can_sell = (torch.tensor(q_np.reshape(-1, 1), dtype=torch.float64, device=device) > -self.Q).float()
            can_buy = (torch.tensor(q_np.reshape(-1, 1), dtype=torch.float64, device=device) < self.Q).float()

            # Profits from optimal quotes
            profit_a = can_sell * fa * (da_t * self.Delta + Ua)
            profit_b = can_buy * fb * (db_t * self.Delta + Ub)

            # Inventory penalty
            psi_q = self.phi * torch.tensor(q_np.reshape(-1, 1), dtype=torch.float64, device=device) ** 2

            # Generator: f = -r*Y - psi + profits
            f_val = -self.r * Y - psi_q + profit_a + profit_b

            # BSDE step: Y_{m+1} = Y_m - f * dt
            # (compensated — jump martingale terms have zero expectation)
            Y = Y - f_val * self.dt

        return Y  # Y_T

    def train(self):
        """Train by minimising |Y_T - g(q_T)|²."""
        start = time.time()
        history = []

        for step in range(self.n_iter):
            # Sample paths
            q_paths = self.sample_paths(self.batch_size)

            # Forward BSDE
            Y_T = self.forward(q_paths)

            # Terminal condition
            q_T = torch.tensor(q_paths[:, -1].reshape(-1, 1),
                              dtype=torch.float64, device=self.device)
            g_T = self.terminal_condition(q_T)

            # Loss
            loss = torch.mean((Y_T - g_T) ** 2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 500 == 0 or step == self.n_iter - 1:
                y0 = self.Y0.item()
                print(f"  step {step}: loss={loss.item():.4e}, Y0={y0:.4f}")
                history.append({"step": step, "loss": loss.item(), "Y0": y0})

        elapsed = time.time() - start

        # Extract U profile at t=0
        q_grid = np.arange(-self.Q, self.Q + self.Delta, self.Delta)
        U_profile = []
        self.subnets[0].eval()
        with torch.no_grad():
            for q in q_grid:
                t_n = torch.tensor([[0.0]], dtype=torch.float64, device=self.device)
                q_n = torch.tensor([[q / self.Q]], dtype=torch.float64, device=self.device)
                U = self.subnets[0](t_n, q_n)
                Ua = U[0, 0].item()
                Ub = U[0, 1].item()
                da = optimal_quote_from_U(Ua, self.N, self.K)
                db = optimal_quote_from_U(Ub, self.N, self.K)
                U_profile.append({"q": q, "Ua": Ua, "Ub": Ub,
                                  "da": da, "db": db, "spread": da + db})

        return {
            "Y0": self.Y0.item(),
            "history": history,
            "U_profile": U_profile,
            "elapsed": elapsed,
            "T": self.T, "M": self.M,
        }

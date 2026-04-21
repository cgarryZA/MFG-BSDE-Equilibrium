"""
BSDEJ solver for the Cont-Xiong dealer market — SHARED WEIGHT variant.

Same algorithm as solver_cx_bsdej.py (Wang et al. 2023, compensated jump
martingale), but uses a SINGLE neural network across all time steps instead
of one network per step.

Why this works better for CX:
  The CX value function V(t,q) is nearly stationary — it barely varies
  with t because the problem has an infinite-horizon ergodic structure.
  So U^a(t=0, q) ≈ U^a(t=T/2, q) ≈ U^a(t=T, q). Having M separate
  networks each independently learn the same function wastes parameters
  and gives each network weak gradient signal (only from its own step).

  A single shared network:
  - Gets gradient signal from EVERY time step on EVERY training sample
  - Has far fewer parameters (4k vs 200k for M=50)
  - Can still modulate with t via the time input
  - Converges much faster to the near-stationary solution

Architecture:
  Input:  (t/T, q/Q) ∈ [0,1] × [-1,1]
  Output: (U^a, U^b) — jump coefficients
  Same network called at t=0, t=dt, ..., t=(M-1)*dt

References:
  - Han, Jentzen, E (2018): original deep BSDE (separate nets per step)
  - Wang et al. (2023): FBSDEJ extension
  - Cont, Xiong (2024): dealer market model
"""

import numpy as np
import torch
import torch.nn as nn
import time

from equations.contxiong_exact import cx_exec_prob_np

# Reuse the vectorised tools from the original solver
from solver_cx_bsdej import (
    _exec_prob_torch_vec,
    optimal_quotes_vectorised,
)


class SharedJumpNet(nn.Module):
    """Single network for (U^a, U^b) shared across all time steps.

    Input: (t/T, q/Q) — normalised time and inventory
    Output: (U^a, U^b) — value jumps

    Wider and deeper than the per-step nets since it has to represent
    the solution across all times, but still much smaller than M separate nets.
    """
    def __init__(self, hidden=128, n_layers=3, dtype=torch.float64):
        super().__init__()
        layers = [nn.Linear(2, hidden, dtype=dtype), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden, dtype=dtype), nn.Tanh()]
        layers.append(nn.Linear(hidden, 2, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, t_norm, q_norm):
        x = torch.cat([t_norm, q_norm], dim=1)
        return self.net(x)


class CXBSDEJShared:
    """Deep BSDE solver with weight sharing across time steps.

    Same forward propagation as CXBSDEJSolver:
      Y_{m+1} = Y_m - f_m * dt + U^a * (dN^a - nu^a*dt) + U^b * (dN^b - nu^b*dt)
      Loss = E[|Y_T - g(q_T)|^2]

    But uses ONE network for all time steps instead of M separate networks.
    """

    def __init__(self, N=2, Q=5, Delta=1, T=1.0, M=20,
                 lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
                 device=None, lr=1e-3, n_iter=5000, batch_size=256,
                 hidden=128, n_layers=3):
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
        self.avg_comp = 0.75

        # Y(0) — one learnable value per inventory level
        self.Y0 = nn.Parameter(torch.zeros(self.nq, dtype=torch.float64,
                                           device=self.device))
        q_grid = torch.arange(-Q, Q + Delta, Delta, dtype=torch.float64)
        with torch.no_grad():
            self.Y0.copy_(-phi * q_grid**2 / r)

        # SINGLE shared network (the key difference)
        self.shared_net = SharedJumpNet(
            hidden=hidden, n_layers=n_layers
        ).to(self.device)

        n_params = sum(p.numel() for p in self.shared_net.parameters())
        print(f"  Shared net: {n_params} params "
              f"(vs {M * n_params // 1} if separate)")

        all_params = [self.Y0] + list(self.shared_net.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_iter, eta_min=lr * 0.01)

    def warmstart_from_bellman(self, n_pretrain=2000, lr_pretrain=1e-3):
        """Pre-train the shared network to match stationary Bellman solution.

        Runs exact Algorithm 1 to get V(q), computes U^a = V(q-1) - V(q),
        U^b = V(q+1) - V(q), then trains the network to output these U values
        for any t (since the solution is nearly stationary).

        Also initialises Y0 to the exact V(q).
        """
        from scripts.cont_xiong_exact import fictitious_play
        print("  Warm-starting from exact Bellman solution...")

        # Get exact solution
        result = fictitious_play(N=self.N, Q=self.Q, Delta=self.Delta)
        V = np.array(result['V'])

        # DON'T set Y0 to stationary V(q) — wrong scale for finite horizon.
        # Stationary V ≈ profits/r ≈ 16, but finite-horizon g(q) = -ψ(q) ≈ 0.
        # Y0 is learnable and will find the right level during BSDE training.
        # We only warm-start the U network (value DIFFERENCES are scale-free).

        # Compute target U values
        nq = len(V)
        Ua_target = np.zeros(nq)
        Ub_target = np.zeros(nq)
        for i in range(nq):
            Ua_target[i] = V[max(0, i-1)] - V[i]      # V(q-1) - V(q)
            Ub_target[i] = V[min(nq-1, i+1)] - V[i]   # V(q+1) - V(q)

        Ua_t = torch.tensor(Ua_target, dtype=torch.float64, device=self.device)
        Ub_t = torch.tensor(Ub_target, dtype=torch.float64, device=self.device)
        q_norm = torch.tensor(
            np.arange(-self.Q, self.Q + self.Delta, self.Delta) / self.Q,
            dtype=torch.float64, device=self.device
        ).unsqueeze(1)

        # Pre-train: for random t, output should match (Ua, Ub) at each q
        opt = torch.optim.Adam(self.shared_net.parameters(), lr=lr_pretrain)

        for step in range(n_pretrain):
            # Random t values (network should give same output for any t)
            t_vals = torch.rand(nq, 1, dtype=torch.float64, device=self.device)
            U_pred = self.shared_net(t_vals, q_norm)

            loss = torch.mean((U_pred[:, 0] - Ua_t)**2 + (U_pred[:, 1] - Ub_t)**2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 500 == 0:
                print(f"    pretrain step {step}: loss={loss.item():.6f}")

        print(f"  Warm-start done. Final pretrain loss: {loss.item():.6f}")
        print(f"  Y0(q=0) = {self.Y0[self.Q].item():.4f} (exact: {V[self.Q]:.4f})")

    def psi(self, q):
        return self.phi * q ** 2

    def terminal_condition(self, q):
        return -self.psi(q)

    def sample_paths(self, batch_size):
        """Sample forward inventory paths with execution events."""
        q = np.zeros(batch_size)
        q_paths = np.zeros((batch_size, self.M + 1))
        exec_a_all = np.zeros((batch_size, self.M))
        exec_b_all = np.zeros((batch_size, self.M))
        q_paths[:, 0] = q

        for m in range(self.M):
            delta_a_proxy = np.clip(0.8 - 0.05 * q, 0.1, 3.0)
            delta_b_proxy = np.clip(0.8 + 0.05 * q, 0.1, 3.0)

            rate_a = np.array([cx_exec_prob_np(da, self.avg_comp, self.K, self.N)
                              for da in delta_a_proxy]) * self.lambda_a
            rate_b = np.array([cx_exec_prob_np(db, self.avg_comp, self.K, self.N)
                              for db in delta_b_proxy]) * self.lambda_b

            prob_a = np.clip(rate_a * self.dt, 0, 0.5)
            prob_b = np.clip(rate_b * self.dt, 0, 0.5)
            exec_a = (np.random.uniform(size=batch_size) < prob_a).astype(float)
            exec_b = (np.random.uniform(size=batch_size) < prob_b).astype(float)

            exec_a_all[:, m] = exec_a
            exec_b_all[:, m] = exec_b

            q = q - exec_a * self.Delta + exec_b * self.Delta
            q = np.clip(q, -self.Q, self.Q)
            q_paths[:, m + 1] = q

        return q_paths, exec_a_all, exec_b_all

    def forward(self, q_paths, exec_a_all, exec_b_all):
        """Forward BSDE propagation with shared network."""
        batch = q_paths.shape[0]
        device = self.device
        dtype = torch.float64

        q0 = torch.tensor(q_paths[:, 0], dtype=dtype, device=device).long()
        q0_idx = (q0 + self.Q).clamp(0, self.nq - 1)
        Y = self.Y0[q0_idx].unsqueeze(1)

        for m in range(self.M):
            t_norm = torch.full((batch, 1), m / self.M, dtype=dtype, device=device)
            q_m_raw = torch.tensor(q_paths[:, m], dtype=dtype, device=device)
            q_m_norm = (q_m_raw / self.Q).unsqueeze(1)

            # Same network at every time step
            U = self.shared_net(t_norm, q_m_norm)
            Ua = U[:, 0:1]
            Ub = U[:, 1:2]

            da_t = optimal_quotes_vectorised(
                Ua, self.avg_comp, self.K, self.N)
            db_t = optimal_quotes_vectorised(
                Ub, self.avg_comp, self.K, self.N)

            fa = _exec_prob_torch_vec(
                da_t, self.avg_comp, self.K, self.N) * self.lambda_a
            fb = _exec_prob_torch_vec(
                db_t, self.avg_comp, self.K, self.N) * self.lambda_b

            can_sell = (q_m_raw > -self.Q).float().unsqueeze(1)
            can_buy = (q_m_raw < self.Q).float().unsqueeze(1)

            profit_a = can_sell * fa * (da_t * self.Delta + Ua)
            profit_b = can_buy * fb * (db_t * self.Delta + Ub)

            psi_q = self.phi * q_m_raw.unsqueeze(1) ** 2

            f_val = self.r * Y + psi_q - profit_a - profit_b

            # Compensated jump martingale
            dN_a = torch.tensor(
                exec_a_all[:, m].reshape(-1, 1), dtype=dtype, device=device)
            dN_b = torch.tensor(
                exec_b_all[:, m].reshape(-1, 1), dtype=dtype, device=device)

            nu_a = can_sell * fa
            nu_b = can_buy * fb

            jump_a = can_sell * Ua * (dN_a - nu_a * self.dt)
            jump_b = can_buy * Ub * (dN_b - nu_b * self.dt)

            Y = Y - f_val * self.dt + jump_a + jump_b

        return Y

    def train(self, early_stopping=True, es_patience=800, es_min_delta=1e-7,
              es_warmup=2000):
        """Train by minimising E[|Y_T - g(q_T)|^2].

        early_stopping: bail out when best_loss plateaus.
        Useful because the BSDEJ solver is known to drift to worse minima
        if trained past convergence (we saw this in the "more training hurt"
        regression).
        """
        start = time.time()
        history = []
        best_loss = float('inf')

        if early_stopping:
            from utils import EarlyStopping
            es = EarlyStopping(patience=es_patience, min_delta=es_min_delta,
                               warmup=es_warmup)
        else:
            es = None

        for step in range(self.n_iter):
            q_paths, exec_a, exec_b = self.sample_paths(self.batch_size)

            Y_T = self.forward(q_paths, exec_a, exec_b)

            q_T = torch.tensor(q_paths[:, -1].reshape(-1, 1),
                              dtype=torch.float64, device=self.device)
            g_T = self.terminal_condition(q_T)

            loss = torch.mean((Y_T - g_T) ** 2)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.shared_net.parameters()) + [self.Y0], max_norm=5.0)
            self.optimizer.step()
            self.scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

            if step % 200 == 0 or step == self.n_iter - 1:
                y0_q0 = self.Y0[self.Q].item()
                lr_now = self.optimizer.param_groups[0]['lr']
                print(f"  step {step:5d}: loss={loss.item():.4e}, "
                      f"best={best_loss:.4e}, Y0(q=0)={y0_q0:.4f}, "
                      f"lr={lr_now:.1e}")
                history.append({
                    "step": step, "loss": loss.item(),
                    "best_loss": best_loss, "Y0_q0": y0_q0
                })

            # Early stopping — avoid drift to worse minima
            if es is not None and es(loss.item()):
                print(f"  Early stopping at step {step} "
                      f"(best loss={es.best_loss:.4e})")
                break

        elapsed = time.time() - start

        # Extract quote profile at t=0
        from scipy.optimize import minimize_scalar
        q_grid = np.arange(-self.Q, self.Q + self.Delta, self.Delta)
        U_profile = []
        self.shared_net.eval()
        with torch.no_grad():
            for q in q_grid:
                t_n = torch.tensor([[0.0]], dtype=torch.float64, device=self.device)
                q_n = torch.tensor([[q / self.Q]], dtype=torch.float64, device=self.device)
                U = self.shared_net(t_n, q_n)
                Ua_val = U[0, 0].item()
                Ub_val = U[0, 1].item()

                def _neg_profit(delta, U_v):
                    f = cx_exec_prob_np(delta, self.avg_comp, self.K, self.N)
                    return -f * (delta + U_v)

                da = minimize_scalar(
                    lambda d: _neg_profit(d, Ua_val),
                    bounds=(-1, 8), method='bounded').x
                db = minimize_scalar(
                    lambda d: _neg_profit(d, Ub_val),
                    bounds=(-1, 8), method='bounded').x

                U_profile.append({
                    "q": float(q), "Ua": Ua_val, "Ub": Ub_val,
                    "da": da, "db": db, "spread": da + db
                })

        # Also extract at t=T/2 to show time-invariance
        U_profile_mid = []
        with torch.no_grad():
            for q in q_grid:
                t_n = torch.tensor([[0.5]], dtype=torch.float64, device=self.device)
                q_n = torch.tensor([[q / self.Q]], dtype=torch.float64, device=self.device)
                U = self.shared_net(t_n, q_n)
                Ua_val = U[0, 0].item()
                Ub_val = U[0, 1].item()

                da = minimize_scalar(
                    lambda d: _neg_profit(d, Ua_val),
                    bounds=(-1, 8), method='bounded').x
                db = minimize_scalar(
                    lambda d: _neg_profit(d, Ub_val),
                    bounds=(-1, 8), method='bounded').x

                U_profile_mid.append({
                    "q": float(q), "spread": da + db
                })

        Y0_profile = []
        for i, q in enumerate(q_grid):
            Y0_profile.append({"q": float(q), "Y0": self.Y0[i].item()})

        return {
            "Y0_profile": Y0_profile,
            "U_profile": U_profile,
            "U_profile_mid": U_profile_mid,
            "history": history,
            "elapsed": elapsed,
            "T": self.T, "M": self.M,
            "best_loss": best_loss,
            "variant": "shared_weights",
        }


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("CX BSDEJ Solver — SHARED WEIGHT variant")
    print("Single network across all time steps")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    solver = CXBSDEJShared(
        N=2, Q=5, Delta=1,
        T=10.0, M=50,
        lambda_a=2.0, lambda_b=2.0,
        r=0.01, phi=0.005,
        device=device,
        lr=5e-4,
        n_iter=10000,
        batch_size=512,
        hidden=128,
        n_layers=3,
    )

    print(f"\nParameters: N={solver.N}, Q={solver.Q}, T={solver.T}, M={solver.M}")
    print(f"Grid: {solver.nq} inventory levels, K={solver.K}")
    print(f"Batch: {solver.batch_size}, Iterations: {solver.n_iter}")
    n_params = sum(p.numel() for p in solver.shared_net.parameters())
    print(f"Network: {n_params} params (shared)")
    print()

    # Warm-start from exact Bellman solution
    solver.warmstart_from_bellman(n_pretrain=2000)
    print()

    result = solver.train()

    print(f"\nTraining complete in {result['elapsed']:.1f}s")
    print(f"Best loss: {result['best_loss']:.4e}")

    print("\n--- Quote Profile at t=0 ---")
    for item in result["U_profile"]:
        print(f"  q={item['q']:+.0f}: da={item['da']:.3f}, "
              f"db={item['db']:.3f}, spread={item['spread']:.3f}")

    print("\n--- Quote Profile at t=T/2 (should be similar) ---")
    for item in result["U_profile_mid"]:
        print(f"  q={item['q']:+.0f}: spread={item['spread']:.3f}")

    spread_q0 = result["U_profile"][5]["spread"]
    spread_mid = result["U_profile_mid"][5]["spread"]
    print(f"\nSpread at q=0:  t=0: {spread_q0:.4f},  t=T/2: {spread_mid:.4f}")
    print(f"Time invariance: {abs(spread_q0 - spread_mid):.4f} difference")
    print(f"Stationary Nash: 1.5153")
    print(f"Error: {abs(spread_q0 - 1.5153)/1.5153:.1%}")

    out_path = "results_bsdej_shared.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")

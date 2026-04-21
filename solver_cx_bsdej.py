"""
Finite-horizon BSDEJ solver for the Cont-Xiong dealer market.

This implements the deep BSDE method (Han et al. 2018, Wang et al. 2023)
applied to the CX market-making problem with Poisson jumps.

The value function V(t, q) satisfies the BSDEJ:

  -dY_t = f(t, q_t, Y_t, U_t) dt - Z_t dW_t
          - U^a_t (dN^a_t - nu^a_t dt) - U^b_t (dN^b_t - nu^b_t dt)

  Y_T = g(q_T) = -psi(q_T)

where:
  Y_t = V(t, q_t)                    (value process)
  Z_t = 0                            (no price dependence in CX)
  U^a_t = V(t, q_t - Delta) - V(t, q_t)   (jump coeff on ask execution)
  U^b_t = V(t, q_t + Delta) - V(t, q_t)   (jump coeff on bid execution)

Generator:
  f(t, q, Y, U^a, U^b) = r*Y + psi(q)
    - lambda_a * max_delta_a [f_a(delta_a, comp) * (delta_a * Delta + U^a)]
    - lambda_b * max_delta_b [f_b(delta_b, comp) * (delta_b * Delta + U^b)]

Forward propagation (Wang et al. 2023, eq 2.16):
  Y_{m+1} = Y_m - f_m * dt + U^a_m * (dN^a_m - nu^a_m * dt)
                             + U^b_m * (dN^b_m - nu^b_m * dt)

The NN learns U^a(t, q) and U^b(t, q) at each time step.
Optimal quotes are derived from U via the FOC.
Loss = E[|Y_T - g(q_T)|^2].

Key references:
  - Han, Jentzen, E (2018): Deep BSDE method (diffusion case)
  - Wang, Wang, Li, Gao, Fu (2023): Extension to FBSDEJs with jumps
  - Cont, Xiong (2024): Dealer market model with exact execution probabilities
"""

import numpy as np
import torch
import torch.nn as nn
import time

from equations.contxiong_exact import cx_exec_prob_np, cx_exec_prob_torch


class JumpCoefficientNet(nn.Module):
    """Network for (U^a, U^b) at each time step.

    Input: (t/T, q/Q) — normalised time and inventory
    Output: (U^a, U^b) — value jumps
    """
    def __init__(self, hidden=64, dtype=torch.float64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, 2, dtype=dtype),
        )

    def forward(self, t_norm, q_norm):
        x = torch.cat([t_norm, q_norm], dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Vectorised FOC solver (replaces per-sample scipy.optimize.minimize_scalar)
# ---------------------------------------------------------------------------

def _exec_prob_torch_vec(delta, avg_comp, K, N):
    """Vectorised CX execution probability in torch (no loops)."""
    base = torch.sigmoid(-delta)
    if K > 0:
        S_over_K = torch.tensor(avg_comp, dtype=delta.dtype, device=delta.device) \
            if not isinstance(avg_comp, torch.Tensor) else avg_comp
        comp = torch.exp(torch.clamp(S_over_K, -20, 20)) / (
            1.0 + torch.exp(torch.clamp(delta + S_over_K, -20, 20)))
        return (1.0 / N) * base * comp
    else:
        return base * base


# Pre-compute a fine grid of delta values for vectorised argmax
_DELTA_GRID = None
_DELTA_GRID_DEVICE = None
_GRID_SIZE = 200


def optimal_quotes_vectorised(U_vals, avg_comp, K, N):
    """Vectorised FOC solver using grid search in PyTorch.

    Evaluates profit = f(delta) * (delta + U) on a fine grid and
    takes the argmax. Fully vectorised, no autograd, very fast.

    Args:
        U_vals: [batch, 1] tensor of jump coefficients
        avg_comp: scalar, average competitor quote
        K: int, number of competitor levels
        N: int, number of dealers

    Returns:
        delta_star: [batch, 1] tensor of optimal quotes
    """
    global _DELTA_GRID, _DELTA_GRID_DEVICE

    device = U_vals.device
    dtype = U_vals.dtype

    # Lazily create grid on correct device
    if _DELTA_GRID is None or _DELTA_GRID_DEVICE != device:
        _DELTA_GRID = torch.linspace(-0.5, 5.0, _GRID_SIZE,
                                     dtype=dtype, device=device)
        _DELTA_GRID_DEVICE = device

    grid = _DELTA_GRID  # [G]

    # Broadcast: U_vals is [B, 1], grid is [G]
    # delta_grid: [B, G]
    delta_grid = grid.unsqueeze(0).expand(U_vals.shape[0], -1)  # [B, G]

    # Exec prob at each grid point
    f_grid = _exec_prob_torch_vec(delta_grid, avg_comp, K, N)  # [B, G]

    # Profit at each grid point: f(delta) * (delta + U)
    # U_vals is [B, 1], broadcast to [B, G]
    profit_grid = f_grid * (delta_grid + U_vals)  # [B, G]

    # Argmax over grid
    best_idx = profit_grid.argmax(dim=1, keepdim=True)  # [B, 1]
    delta_star = grid[best_idx.squeeze(1)].unsqueeze(1)  # [B, 1]

    return delta_star.detach()


class CXBSDEJSolver:
    """Deep BSDE solver for finite-horizon CX model with jumps.

    Follows Wang et al. (2023) eq 2.16:
      Y_{n+1} = Y_n - f_n * dt + U^a_n * (dN^a - nu^a*dt)
                                + U^b_n * (dN^b - nu^b*dt)
      Loss = E[|Y_T - g(q_T)|^2]

    Key difference from our initial (incorrect) version:
    The compensated jump martingale terms U*(dN - nu*dt) are now INCLUDED
    in the Y propagation, matching Wang et al.'s algorithm exactly.
    Previously we dropped them claiming "zero expectation" — that defeats
    the purpose of the deep BSDE method.
    """

    def __init__(self, N=2, Q=5, Delta=1, T=1.0, M=20,
                 lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
                 device=None, lr=1e-3, n_iter=5000, batch_size=256,
                 hidden=64):
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
        self.avg_comp = 0.75  # mean-field average competitor quote

        # Y(0) — initial value, learnable parameter (one per inventory level)
        self.Y0 = nn.Parameter(torch.zeros(self.nq, dtype=torch.float64,
                                           device=self.device))
        # Initialise Y0 ~ -phi * q^2 / r as rough guess
        q_grid = torch.arange(-Q, Q + Delta, Delta, dtype=torch.float64)
        with torch.no_grad():
            self.Y0.copy_(-phi * q_grid**2 / r)

        # One subnet per time step (Wang et al. / Han et al. architecture)
        self.subnets = nn.ModuleList([
            JumpCoefficientNet(hidden=hidden).to(self.device)
            for _ in range(M)
        ])

        all_params = [self.Y0] + list(self.subnets.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)

    def psi(self, q):
        return self.phi * q ** 2

    def terminal_condition(self, q):
        """g(q_T) = -psi(q_T)."""
        return -self.psi(q)

    def q_to_index(self, q):
        """Convert inventory q to index into Y0 array."""
        return (q + self.Q).long()

    def sample_paths(self, batch_size):
        """Sample forward inventory paths WITH execution events.

        Returns:
            q_paths:    [batch, M+1] inventory at each time step
            exec_a_all: [batch, M]   ask execution indicators (0 or 1)
            exec_b_all: [batch, M]   bid execution indicators (0 or 1)
        """
        q = np.zeros(batch_size)
        q_paths = np.zeros((batch_size, self.M + 1))
        exec_a_all = np.zeros((batch_size, self.M))
        exec_b_all = np.zeros((batch_size, self.M))
        q_paths[:, 0] = q

        for m in range(self.M):
            # Proxy quotes: simple inventory-dependent rule
            delta_a_proxy = np.clip(0.8 - 0.05 * q, 0.1, 3.0)
            delta_b_proxy = np.clip(0.8 + 0.05 * q, 0.1, 3.0)

            # Execution rates (vectorised over batch)
            rate_a = np.array([cx_exec_prob_np(da, self.avg_comp, self.K, self.N)
                              for da in delta_a_proxy]) * self.lambda_a
            rate_b = np.array([cx_exec_prob_np(db, self.avg_comp, self.K, self.N)
                              for db in delta_b_proxy]) * self.lambda_b

            # Poisson jumps in dt (thin approximation for small dt)
            prob_a = np.clip(rate_a * self.dt, 0, 0.5)
            prob_b = np.clip(rate_b * self.dt, 0, 0.5)
            exec_a = (np.random.uniform(size=batch_size) < prob_a).astype(float)
            exec_b = (np.random.uniform(size=batch_size) < prob_b).astype(float)

            # Store execution events
            exec_a_all[:, m] = exec_a
            exec_b_all[:, m] = exec_b

            # Inventory update: sell on ask exec, buy on bid exec
            q = q - exec_a * self.Delta + exec_b * self.Delta
            q = np.clip(q, -self.Q, self.Q)
            q_paths[:, m + 1] = q

        return q_paths, exec_a_all, exec_b_all

    def forward(self, q_paths, exec_a_all, exec_b_all):
        """Forward BSDE propagation following Wang et al. (2023) eq 2.16.

        Y_{m+1} = Y_m - f_m * dt
                  + U^a_m * (dN^a_m - nu^a_m * dt)   ← ask jump martingale
                  + U^b_m * (dN^b_m - nu^b_m * dt)   ← bid jump martingale

        The compensated jump martingale terms are ESSENTIAL for the deep BSDE
        method to work. They provide the stochastic signal that forces the
        neural network to learn the correct jump coefficients U^a, U^b.
        """
        batch = q_paths.shape[0]
        device = self.device
        dtype = torch.float64

        # Initial Y: look up Y0 by inventory index
        q0 = torch.tensor(q_paths[:, 0], dtype=dtype, device=device).long()
        q0_idx = (q0 + self.Q).clamp(0, self.nq - 1)
        Y = self.Y0[q0_idx].unsqueeze(1)  # [batch, 1]

        for m in range(self.M):
            t_norm = torch.full((batch, 1), m / self.M, dtype=dtype, device=device)
            q_m_raw = torch.tensor(q_paths[:, m], dtype=dtype, device=device)
            q_m_norm = (q_m_raw / self.Q).unsqueeze(1)  # normalised for NN

            # Get U^a, U^b from subnet
            U = self.subnets[m](t_norm, q_m_norm)  # [batch, 2]
            Ua = U[:, 0:1]  # V(q-1) - V(q) — typically negative
            Ub = U[:, 1:2]  # V(q+1) - V(q)

            # Optimal quotes from U (vectorised Newton, no scipy)
            da_t = optimal_quotes_vectorised(
                Ua, self.avg_comp, self.K, self.N)
            db_t = optimal_quotes_vectorised(
                Ub, self.avg_comp, self.K, self.N)

            # Execution probabilities at optimal quotes (vectorised)
            fa = _exec_prob_torch_vec(
                da_t, self.avg_comp, self.K, self.N) * self.lambda_a
            fb = _exec_prob_torch_vec(
                db_t, self.avg_comp, self.K, self.N) * self.lambda_b

            # Boundary masks
            can_sell = (q_m_raw > -self.Q).float().unsqueeze(1)
            can_buy = (q_m_raw < self.Q).float().unsqueeze(1)

            # Profits from optimal quotes (part of generator f)
            profit_a = can_sell * fa * (da_t * self.Delta + Ua)
            profit_b = can_buy * fb * (db_t * self.Delta + Ub)

            # Inventory penalty
            psi_q = self.phi * q_m_raw.unsqueeze(1) ** 2

            # Generator: f = r*Y + psi - profits
            # (note sign: BSDE is -dY = f dt - martingale terms)
            f_val = self.r * Y + psi_q - profit_a - profit_b

            # === Compensated jump martingale (Wang et al. eq 2.16) ===
            # This is the critical term we were previously missing.
            dN_a = torch.tensor(
                exec_a_all[:, m].reshape(-1, 1), dtype=dtype, device=device)
            dN_b = torch.tensor(
                exec_b_all[:, m].reshape(-1, 1), dtype=dtype, device=device)

            # nu = execution rate (intensity), already computed as fa, fb
            nu_a = can_sell * fa  # zero if can't sell
            nu_b = can_buy * fb   # zero if can't buy

            # Compensated Poisson: dN - nu*dt
            # When ask executes (dN_a=1): Y jumps by +U^a (value goes V(q)->V(q-1))
            # When no execution (dN_a=0): Y drifts by -U^a*nu_a*dt (compensator)
            jump_a = can_sell * Ua * (dN_a - nu_a * self.dt)
            jump_b = can_buy * Ub * (dN_b - nu_b * self.dt)

            # Full BSDE step: Y_{m+1} = Y_m - f*dt + jump martingale
            Y = Y - f_val * self.dt + jump_a + jump_b

        return Y  # Y_T

    def train(self):
        """Train by minimising E[|Y_T - g(q_T)|^2]."""
        start = time.time()
        history = []
        best_loss = float('inf')

        for step in range(self.n_iter):
            # Sample paths WITH execution events
            q_paths, exec_a, exec_b = self.sample_paths(self.batch_size)

            # Forward BSDE (with jump martingale)
            Y_T = self.forward(q_paths, exec_a, exec_b)

            # Terminal condition
            q_T = torch.tensor(q_paths[:, -1].reshape(-1, 1),
                              dtype=torch.float64, device=self.device)
            g_T = self.terminal_condition(q_T)

            # Loss
            loss = torch.mean((Y_T - g_T) ** 2)

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(
                list(self.subnets.parameters()) + [self.Y0], max_norm=5.0)
            self.optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

            if step % 200 == 0 or step == self.n_iter - 1:
                # Report Y0 at q=0
                y0_q0 = self.Y0[self.Q].item()
                print(f"  step {step:5d}: loss={loss.item():.4e}, "
                      f"best={best_loss:.4e}, Y0(q=0)={y0_q0:.4f}")
                history.append({
                    "step": step, "loss": loss.item(),
                    "best_loss": best_loss, "Y0_q0": y0_q0
                })

        elapsed = time.time() - start

        # Extract U and quote profile at t=0
        # Use scipy for extraction (autograd not available in no_grad context)
        from scipy.optimize import minimize_scalar
        q_grid = np.arange(-self.Q, self.Q + self.Delta, self.Delta)
        U_profile = []
        self.subnets[0].eval()
        with torch.no_grad():
            for q in q_grid:
                t_n = torch.tensor([[0.0]], dtype=torch.float64, device=self.device)
                q_n = torch.tensor([[q / self.Q]], dtype=torch.float64, device=self.device)
                U = self.subnets[0](t_n, q_n)
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
                    "q": float(q),
                    "Ua": Ua_val, "Ub": Ub_val,
                    "da": da, "db": db,
                    "spread": da + db
                })

        # Y0 profile
        Y0_profile = []
        for i, q in enumerate(q_grid):
            Y0_profile.append({"q": float(q), "Y0": self.Y0[i].item()})

        return {
            "Y0_profile": Y0_profile,
            "U_profile": U_profile,
            "history": history,
            "elapsed": elapsed,
            "T": self.T, "M": self.M,
            "best_loss": best_loss,
        }


# ---------------------------------------------------------------------------
# Comparison with stationary neural Bellman solver
# ---------------------------------------------------------------------------

def compare_with_bellman(bsdej_result, bellman_V, Q=5):
    """Compare finite-horizon BSDEJ Y0 with stationary Bellman V(q).

    As T -> inf, the BSDEJ value Y(0, q) should converge to the
    stationary V(q) from the Bellman solver.
    """
    print("\n=== BSDEJ vs Stationary Bellman ===")
    print(f"{'q':>4s}  {'BSDEJ Y0':>10s}  {'Bellman V':>10s}  {'Error':>8s}")
    print("-" * 40)

    for item in bsdej_result["Y0_profile"]:
        q = int(item["q"])
        y0 = item["Y0"]
        idx = q + Q
        if 0 <= idx < len(bellman_V):
            v = bellman_V[idx]
            err = abs(y0 - v) / (abs(v) + 1e-8)
            print(f"{q:4d}  {y0:10.4f}  {v:10.4f}  {err:7.2%}")

    print("\nQuote profile comparison:")
    print(f"{'q':>4s}  {'BSDEJ spread':>13s}")
    print("-" * 20)
    for item in bsdej_result["U_profile"]:
        q = int(item["q"])
        s = item["spread"]
        print(f"{q:4d}  {s:13.4f}")


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("CX BSDEJ Solver (Wang et al. 2023 algorithm)")
    print("With compensated jump martingale terms")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    solver = CXBSDEJSolver(
        N=2, Q=5, Delta=1,
        T=5.0,       # longer horizon -> closer to stationary
        M=50,        # 50 time steps
        lambda_a=2.0, lambda_b=2.0,
        r=0.01, phi=0.005,
        device=device,
        lr=5e-4,
        n_iter=5000,
        batch_size=512,
        hidden=64,
    )

    print(f"\nParameters: N={solver.N}, Q={solver.Q}, T={solver.T}, M={solver.M}")
    print(f"Grid: {solver.nq} inventory levels, K={solver.K}")
    print(f"Batch: {solver.batch_size}, Iterations: {solver.n_iter}")
    print(f"Network: {sum(p.numel() for p in solver.subnets.parameters())} params")
    print()

    result = solver.train()

    print(f"\nTraining complete in {result['elapsed']:.1f}s")
    print(f"Best loss: {result['best_loss']:.4e}")

    print("\n--- Y0 Profile ---")
    for item in result["Y0_profile"]:
        print(f"  q={item['q']:+.0f}: Y0={item['Y0']:.4f}")

    print("\n--- Quote Profile at t=0 ---")
    for item in result["U_profile"]:
        print(f"  q={item['q']:+.0f}: da={item['da']:.3f}, "
              f"db={item['db']:.3f}, spread={item['spread']:.3f}")

    # Compare with known stationary Nash spread
    spread_q0 = result["U_profile"][5]["spread"]  # q=0
    print(f"\nSpread at q=0: {spread_q0:.4f}")
    print(f"Stationary Nash: 1.5153")
    print(f"Error: {abs(spread_q0 - 1.5153)/1.5153:.1%}")

    # Save results
    out_path = "results_bsdej.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")

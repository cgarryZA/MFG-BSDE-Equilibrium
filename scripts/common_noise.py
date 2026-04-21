#!/usr/bin/env python -u
"""
Cont-Xiong model with common noise (systematic price shock).

Adds a Brownian motion dW_S affecting all dealers simultaneously:
  dS_t = σ_S dW_t

The value function now depends on both inventory q and price S:
  V(q, S). But by translation invariance, V(q, S) = S*q + W(q) where
  W only depends on inventory — UNLESS the model has non-linear dependence
  on S via the execution probability (which in our setup it doesn't, since
  f depends only on relative quote delta, not absolute price).

To break this and introduce genuine common noise, we modify the exec
probability to depend on S via a stochastic intensity:
  λ(t) = λ_0 * (1 + σ_S * S_t / S_0)

This models market-wide "flow" shocks — when S rises (favorable conditions),
all dealers see more flow. This creates conditional McKean-Vlasov structure
because the population average is now S-dependent.

Forward SDE:
  dq_t = jump process (per CX)
  dS_t = σ_S dW_t  (common Brownian motion)

Backward BSDE:
  -dY_t = f(t, q_t, S_t, Y_t, Z_S_t, U^a_t, U^b_t) dt
          - Z_S_t dW_t - U^a(dN^a - ν dt) - U^b(dN^b - ν dt)
  Y_T = g(q_T, S_T) = -φ q_T² (no S dependence at terminal)

Where Z_S_t = σ_S · ∂V/∂S.

This is a genuine 2D state BSDE — Z_S ≠ 0 is learned.

Run: python -u scripts/common_noise.py
"""

import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equations.contxiong_exact import cx_exec_prob_np
from solver_cx_bsdej import _exec_prob_torch_vec, optimal_quotes_vectorised

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


class CommonNoiseNet(nn.Module):
    """Network for V-related quantities with (t, q, S) input.

    Outputs:
      U^a, U^b: jump coefficients (as in pure-jump BSDEJ)
      Z_S:      diffusion coefficient w.r.t. price (common noise)

    Note the 3-dim input (t/T, q/Q, S/S_scale).
    """
    def __init__(self, hidden=128, n_layers=3, dtype=torch.float64):
        super().__init__()
        layers = [nn.Linear(3, hidden, dtype=dtype), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden, dtype=dtype), nn.Tanh()]
        layers.append(nn.Linear(hidden, 3, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, t_norm, q_norm, S_norm):
        x = torch.cat([t_norm, q_norm, S_norm], dim=1)
        return self.net(x)


class CXCommonNoiseSolver:
    """Deep BSDE solver for CX with common price noise.

    State: (t, q, S). Forward:
      q: Poisson jumps (as in BSDEJ solver)
      S: continuous Brownian, dS = σ_S dW

    Intensity modulation: λ(S) = λ_0 (1 + κ · (S - S_0)/S_0)

    This makes the population distribution S-dependent, so
    Z_S = σ_S * ∂V/∂S ≠ 0.
    """

    def __init__(self, N=2, Q=5, Delta=1, T=10.0, M=50,
                 lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
                 sigma_S=0.3, kappa=0.2, S_0=1.0, S_scale=1.0,
                 device=None, lr=5e-4, n_iter=10000, batch_size=512,
                 hidden=128, n_layers=3):
        self.N = N; self.Q = Q; self.Delta = Delta
        self.T = T; self.M = M
        self.dt = T / M
        self.lambda_a = lambda_a; self.lambda_b = lambda_b
        self.r = r; self.phi = phi
        self.sigma_S = sigma_S  # price volatility
        self.kappa = kappa      # intensity modulation strength
        self.S_0 = S_0          # initial price / reference
        self.S_scale = S_scale  # normaliser for network input
        self.device = device or torch.device("cpu")
        self.n_iter = n_iter
        self.batch_size = batch_size

        self.nq = int(2 * Q / Delta + 1)
        self.K = (N - 1) * self.nq
        self.avg_comp = 0.75

        # Y(0) — learnable per inventory level (assuming S_0 fixed start)
        self.Y0 = nn.Parameter(torch.zeros(self.nq, dtype=torch.float64,
                                           device=self.device))
        q_grid = torch.arange(-Q, Q + Delta, Delta, dtype=torch.float64)
        with torch.no_grad():
            self.Y0.copy_(-phi * q_grid**2 / r)

        self.net = CommonNoiseNet(hidden=hidden, n_layers=n_layers).to(self.device)

        n_params = sum(p.numel() for p in self.net.parameters())
        print(f"  Net: {n_params} params", flush=True)

        all_params = [self.Y0] + list(self.net.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_iter, eta_min=lr * 0.01)

    def lambda_mod(self, S):
        """Intensity modulation: λ(S) = λ_0 (1 + κ (S-S_0)/S_0).

        Returns scalar multiplier (tensor).
        """
        return 1.0 + self.kappa * (S - self.S_0) / self.S_0

    def terminal_condition(self, q, S):
        """g(q_T, S_T) = -φ q_T²."""
        return -self.phi * q**2

    def sample_paths(self, batch_size):
        """Sample forward paths for (q, S) with common Brownian driving S."""
        q = np.zeros(batch_size)
        S = np.ones(batch_size) * self.S_0

        q_paths = np.zeros((batch_size, self.M + 1))
        S_paths = np.zeros((batch_size, self.M + 1))
        exec_a = np.zeros((batch_size, self.M))
        exec_b = np.zeros((batch_size, self.M))
        dW_S = np.zeros((batch_size, self.M))

        q_paths[:, 0] = q
        S_paths[:, 0] = S
        sqrt_dt = np.sqrt(self.dt)

        for m in range(self.M):
            # S follows Brownian motion
            dW = np.random.randn(batch_size) * sqrt_dt
            dW_S[:, m] = dW
            S_new = S + self.sigma_S * dW

            # Intensity at this S (clipped)
            lam_mult = 1.0 + self.kappa * (S - self.S_0) / self.S_0
            lam_mult = np.clip(lam_mult, 0.1, 3.0)

            # Proxy quotes
            delta_a_proxy = np.clip(0.8 - 0.05 * q, 0.1, 3.0)
            delta_b_proxy = np.clip(0.8 + 0.05 * q, 0.1, 3.0)

            rate_a = np.array([cx_exec_prob_np(da, self.avg_comp, self.K, self.N)
                              for da in delta_a_proxy]) * self.lambda_a * lam_mult
            rate_b = np.array([cx_exec_prob_np(db, self.avg_comp, self.K, self.N)
                              for db in delta_b_proxy]) * self.lambda_b * lam_mult

            prob_a = np.clip(rate_a * self.dt, 0, 0.5)
            prob_b = np.clip(rate_b * self.dt, 0, 0.5)
            ea = (np.random.uniform(size=batch_size) < prob_a).astype(float)
            eb = (np.random.uniform(size=batch_size) < prob_b).astype(float)
            exec_a[:, m] = ea
            exec_b[:, m] = eb

            q = q - ea * self.Delta + eb * self.Delta
            q = np.clip(q, -self.Q, self.Q)
            S = S_new

            q_paths[:, m + 1] = q
            S_paths[:, m + 1] = S

        return q_paths, S_paths, exec_a, exec_b, dW_S

    def forward(self, q_paths, S_paths, exec_a_all, exec_b_all, dW_S_all):
        """Joint forward BSDE propagation."""
        batch = q_paths.shape[0]
        dtype = torch.float64
        dev = self.device

        q0 = torch.tensor(q_paths[:, 0], dtype=dtype, device=dev).long()
        q0_idx = (q0 + self.Q).clamp(0, self.nq - 1)
        Y = self.Y0[q0_idx].unsqueeze(1)

        for m in range(self.M):
            t_norm = torch.full((batch, 1), m / self.M, dtype=dtype, device=dev)
            q_m = torch.tensor(q_paths[:, m], dtype=dtype, device=dev).unsqueeze(1)
            S_m = torch.tensor(S_paths[:, m], dtype=dtype, device=dev).unsqueeze(1)
            q_m_norm = q_m / self.Q
            S_m_norm = (S_m - self.S_0) / self.S_scale

            out = self.net(t_norm, q_m_norm, S_m_norm)
            Ua = out[:, 0:1]
            Ub = out[:, 1:2]
            Z_S = out[:, 2:3]

            # Intensity modulation
            lam_mult = torch.clamp(1.0 + self.kappa * (S_m - self.S_0) / self.S_0, 0.1, 3.0)

            da = optimal_quotes_vectorised(Ua, self.avg_comp, self.K, self.N)
            db = optimal_quotes_vectorised(Ub, self.avg_comp, self.K, self.N)

            fa = _exec_prob_torch_vec(da, self.avg_comp, self.K, self.N) * self.lambda_a * lam_mult
            fb = _exec_prob_torch_vec(db, self.avg_comp, self.K, self.N) * self.lambda_b * lam_mult

            can_sell = (q_m > -self.Q).float()
            can_buy = (q_m < self.Q).float()

            profit_a = can_sell * fa * (da * self.Delta + Ua)
            profit_b = can_buy * fb * (db * self.Delta + Ub)
            psi_q = self.phi * q_m**2

            f_val = self.r * Y + psi_q - profit_a - profit_b

            dN_a = torch.tensor(exec_a_all[:, m].reshape(-1, 1), dtype=dtype, device=dev)
            dN_b = torch.tensor(exec_b_all[:, m].reshape(-1, 1), dtype=dtype, device=dev)
            dW = torch.tensor(dW_S_all[:, m].reshape(-1, 1), dtype=dtype, device=dev)

            nu_a = can_sell * fa
            nu_b = can_buy * fb
            jump_a = can_sell * Ua * (dN_a - nu_a * self.dt)
            jump_b = can_buy * Ub * (dN_b - nu_b * self.dt)

            # Full BSDE: drift + jump martingale + diffusion martingale
            Y = Y - f_val * self.dt + jump_a + jump_b + Z_S * dW

        return Y

    def train(self):
        start = time.time()
        history = []
        best_loss = float('inf')

        for step in range(self.n_iter):
            qp, Sp, ea, eb, dws = self.sample_paths(self.batch_size)
            Y_T = self.forward(qp, Sp, ea, eb, dws)

            q_T = torch.tensor(qp[:, -1].reshape(-1, 1), dtype=torch.float64, device=self.device)
            S_T = torch.tensor(Sp[:, -1].reshape(-1, 1), dtype=torch.float64, device=self.device)
            g_T = self.terminal_condition(q_T, S_T)

            loss = torch.mean((Y_T - g_T) ** 2)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.net.parameters()) + [self.Y0], max_norm=5.0)
            self.optimizer.step()
            self.scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

            if step % 200 == 0 or step == self.n_iter - 1:
                y0_q0 = self.Y0[self.Q].item()
                print(f"  step {step:5d}: loss={loss.item():.4e}, best={best_loss:.4e}, "
                      f"Y0(q=0)={y0_q0:.4f}", flush=True)
                history.append({"step": step, "loss": loss.item(), "best_loss": best_loss,
                                "Y0_q0": y0_q0})

        elapsed = time.time() - start

        # Extract Z_S profile at t=0 for different S values
        from scipy.optimize import minimize_scalar
        q_grid = np.arange(-self.Q, self.Q + self.Delta, self.Delta)
        S_vals = [self.S_0 - 2*self.sigma_S, self.S_0, self.S_0 + 2*self.sigma_S]

        profiles = {}
        self.net.eval()
        with torch.no_grad():
            for S_val in S_vals:
                profile = []
                for q in q_grid:
                    t_n = torch.tensor([[0.0]], dtype=torch.float64, device=self.device)
                    q_n = torch.tensor([[q/self.Q]], dtype=torch.float64, device=self.device)
                    S_n = torch.tensor([[(S_val-self.S_0)/self.S_scale]], dtype=torch.float64, device=self.device)
                    out = self.net(t_n, q_n, S_n)
                    Ua_v = out[0, 0].item(); Ub_v = out[0, 1].item(); Z_v = out[0, 2].item()

                    def _neg(d, Uv):
                        f = cx_exec_prob_np(d, self.avg_comp, self.K, self.N)
                        return -f * (d + Uv)
                    da = minimize_scalar(lambda d: _neg(d, Ua_v), bounds=(-1, 8), method='bounded').x
                    db = minimize_scalar(lambda d: _neg(d, Ub_v), bounds=(-1, 8), method='bounded').x
                    profile.append({"q": float(q), "Ua": Ua_v, "Ub": Ub_v, "Z_S": Z_v,
                                    "da": da, "db": db, "spread": da + db})
                profiles[f"S={S_val:.2f}"] = profile

        return {
            "profiles_by_S": profiles,
            "history": history,
            "elapsed": elapsed,
            "best_loss": best_loss,
            "sigma_S": self.sigma_S, "kappa": self.kappa,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("CX with common noise (dW_S) — conditional MV-BSDE")
    print("=" * 60, flush=True)

    solver = CXCommonNoiseSolver(
        N=2, Q=5, Delta=1, T=5.0, M=30,
        lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
        sigma_S=0.3, kappa=0.3, S_0=1.0, S_scale=1.0,
        device=device, lr=5e-4, n_iter=5000, batch_size=256,
        hidden=128, n_layers=3,
    )

    result = solver.train()

    print("\n--- Quote profiles at different S ---")
    for key, prof in result["profiles_by_S"].items():
        print(f"\n  {key}:")
        for item in prof:
            print(f"    q={item['q']:+.0f}: Z_S={item['Z_S']:+.3f}, spread={item['spread']:.3f}")

    with open("results_final/common_noise.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved to results_final/common_noise.json", flush=True)
    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

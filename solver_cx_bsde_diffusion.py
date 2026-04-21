#!/usr/bin/env python -u
"""
Continuous inventory BSDE solver for the Cont-Xiong dealer market.

This is the GENUINE deep BSDE method — the network learns Z_t = σ_q · ∂V/∂q,
the diffusion coefficient of the value process. Unlike the jump solver
(solver_cx_bsdej_shared.py) where Z=0 and only jump coefficients U are learned,
here Z does all the work.

The discrete Poisson execution events are replaced by their diffusion
approximation (Poisson CLT):
  dq_t = μ_q dt + σ_q dW_t
  μ_q = (λ_b·f_b - λ_a·f_a)·Δ        [net flow]
  σ_q = √((λ_a·f_a + λ_b·f_b)·Δ²)   [Poisson variance]

The BSDE:
  -dY_t = f(t, q_t, Y_t, Z_t) dt - Z_t dW_t
  Y_T = g(q_T) = -φ·q_T²

where Z_t = σ_q · ∂V/∂q, and the generator f encodes the HJB with
optimal quotes derived from Z via the FOC.

Key difference from jump solver:
  - Network outputs Z (1D), not (U^a, U^b) (2D)
  - Martingale term is Z·dW, not U·(dN - ν·dt)
  - Forward SDE and backward SDE share the SAME Brownian motion
  - This is what Han et al. (2018) actually designed

References:
  - Han, Jentzen, E (2018): Deep BSDE method
  - Wang et al. (2023): Extension to FBSDEJs with jumps
  - Cont, Xiong (2024): Dealer market model
"""

import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

from equations.contxiong_exact import cx_exec_prob_np
from solver_cx_bsdej import _exec_prob_torch_vec, optimal_quotes_vectorised


class SharedDiffusionNet(nn.Module):
    """Network for Z_t = σ_q · ∂V/∂q, shared across all time steps.

    Input: (t/T, q/Q) — normalised time and inventory
    Output: Z_t — scalar diffusion coefficient

    Z should be:
      - Antisymmetric in q: Z(t, q) = -Z(t, -q) (since V is symmetric)
      - Near zero at q=0 (∂V/∂q = 0 at the symmetric point)
      - Nearly independent of t (stationary problem)
    """
    def __init__(self, hidden=128, n_layers=3, dtype=torch.float64):
        super().__init__()
        layers = [nn.Linear(2, hidden, dtype=dtype), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden, dtype=dtype), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, t_norm, q_norm):
        """Returns Z_t given normalised (t, q)."""
        x = torch.cat([t_norm, q_norm], dim=1)
        return self.net(x)


class CXBSDEDiffusion:
    """Deep BSDE solver with continuous inventory (Z ≠ 0).

    The genuine deep BSDE method: learns Z_t = σ_q · ∂V/∂q.
    Quotes are derived from Z via the FOC, not learned directly.

    Forward propagation (coupled SDE + BSDE):
      q_{m+1} = q_m + μ_q·dt + σ_q·dW_m        [inventory SDE]
      Y_{m+1} = Y_m - f_m·dt + Z_m·dW_m         [value BSDE]
      Loss = E[|Y_T - g(q_T)|²]

    Same Brownian motion dW drives both — this is the coupling.
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
        self.avg_comp = 0.75  # mean-field average competitor quote

        # Y(0) — learnable per inventory level
        self.Y0 = nn.Parameter(torch.zeros(self.nq, dtype=torch.float64,
                                           device=self.device))
        q_grid = torch.arange(-Q, Q + Delta, Delta, dtype=torch.float64)
        with torch.no_grad():
            self.Y0.copy_(-phi * q_grid**2 / r)

        # Z network (the star of the show)
        self.z_net = SharedDiffusionNet(
            hidden=hidden, n_layers=n_layers
        ).to(self.device)

        n_params = sum(p.numel() for p in self.z_net.parameters())
        print(f"  Z-net: {n_params} params (shared across {M} time steps)", flush=True)

        all_params = [self.Y0] + list(self.z_net.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_iter, eta_min=lr * 0.01)

        # Initial sigma_q estimate (from monopolist equilibrium quotes ~0.8)
        f_init = cx_exec_prob_np(0.8, self.avg_comp, self.K, self.N)
        self.sigma_q_init = float(np.sqrt(
            (self.lambda_a * f_init + self.lambda_b * f_init) * Delta**2))

    def psi(self, q):
        return self.phi * q ** 2

    def terminal_condition(self, q):
        return -self.psi(q)

    def forward(self, batch_size=None):
        """Joint forward propagation of inventory SDE + value BSDE.

        No separate sample_paths — the same dW drives both q and Y.
        """
        batch = batch_size or self.batch_size
        device = self.device
        dtype = torch.float64

        # Pre-sample all Brownian increments
        sqrt_dt = np.sqrt(self.dt)
        dW_all = torch.randn(batch, self.M, dtype=dtype, device=device) * sqrt_dt

        # Initial conditions
        q = torch.zeros(batch, 1, dtype=dtype, device=device)  # start at q=0
        q0_idx = torch.full((batch,), self.Q, dtype=torch.long, device=device)
        Y = self.Y0[q0_idx].unsqueeze(1)

        # Lagged sigma_q (from initial estimate)
        sigma_q_lag = torch.full((batch, 1), self.sigma_q_init,
                                 dtype=dtype, device=device)

        for m in range(self.M):
            t_norm = torch.full((batch, 1), m / self.M, dtype=dtype, device=device)
            q_norm = q / self.Q  # normalise for network

            # Network predicts Z
            Z = self.z_net(t_norm, q_norm)  # [batch, 1]

            # ∂V/∂q = Z / σ_q (with safety floor)
            sigma_q_safe = torch.clamp(sigma_q_lag, min=1e-4)
            dVdq = torch.clamp(Z / sigma_q_safe, -20.0, 20.0)

            # Optimal quotes from ∂V/∂q via FOC
            # Ask: argmax f_a(δ) · (δ·Δ + U_eff_a) where U_eff_a = -Δ·∂V/∂q
            # Bid: argmax f_b(δ) · (δ·Δ + U_eff_b) where U_eff_b = +Δ·∂V/∂q
            U_eff_a = -self.Delta * dVdq   # selling reduces value if V slopes up
            U_eff_b = self.Delta * dVdq    # buying increases value if V slopes up

            da = optimal_quotes_vectorised(U_eff_a, self.avg_comp, self.K, self.N)
            db = optimal_quotes_vectorised(U_eff_b, self.avg_comp, self.K, self.N)

            # Execution rates
            fa = _exec_prob_torch_vec(da, self.avg_comp, self.K, self.N) * self.lambda_a
            fb = _exec_prob_torch_vec(db, self.avg_comp, self.K, self.N) * self.lambda_b

            # Update sigma_q and mu_q for this step
            sigma_q = torch.sqrt(torch.clamp(
                (fa + fb) * self.Delta**2, min=1e-8))
            mu_q = (fb - fa) * self.Delta

            # Boundary masks (soft — continuous inventory can approach but not exceed Q)
            can_sell = torch.sigmoid(10.0 * (q + self.Q))   # ~1 when q > -Q
            can_buy = torch.sigmoid(10.0 * (self.Q - q))    # ~1 when q < Q

            # Profits from optimal quotes
            profit_a = can_sell * fa * (da * self.Delta + U_eff_a)
            profit_b = can_buy * fb * (db * self.Delta + U_eff_b)

            # Generator: f = r·Y + ψ(q) - profits
            psi_q = self.phi * q ** 2
            f_val = self.r * Y + psi_q - profit_a - profit_b

            # Get this step's Brownian increment
            dW = dW_all[:, m:m+1]  # [batch, 1]

            # === THE REAL BSDE STEP: Y_{m+1} = Y_m - f·dt + Z·dW ===
            Y = Y - f_val * self.dt + Z * dW

            # === Forward SDE: q_{m+1} = q_m + μ·dt + σ·dW ===
            q = q + mu_q * self.dt + sigma_q * dW
            q = torch.clamp(q, -self.Q, self.Q)

            # Update lagged sigma for next step
            sigma_q_lag = sigma_q.detach()

        return Y, q  # Y_T and q_T

    def warmstart_from_bellman(self, n_pretrain=2000):
        """Pre-train Z network from exact stationary solution.

        Computes Z_target = σ_q · ∂V/∂q via central differences on V(q)
        from exact Algorithm 1, then trains the network to match.
        """
        from scripts.cont_xiong_exact import fictitious_play
        print("  Warm-starting Z network from Bellman solution...", flush=True)

        result = fictitious_play(N=self.N, Q=self.Q, Delta=self.Delta)
        V = np.array(result['V'])
        da_eq = np.array(result['delta_a'])
        db_eq = np.array(result['delta_b'])
        q_grid = np.arange(-self.Q, self.Q + self.Delta, self.Delta)
        nq = len(V)

        # Central differences for ∂V/∂q
        dVdq = np.zeros(nq)
        for j in range(nq):
            if j == 0:
                dVdq[j] = (V[1] - V[0]) / self.Delta
            elif j == nq - 1:
                dVdq[j] = (V[-1] - V[-2]) / self.Delta
            else:
                dVdq[j] = (V[j+1] - V[j-1]) / (2 * self.Delta)

        # σ_q at each grid point from equilibrium quotes
        sigma_q_grid = np.zeros(nq)
        for j in range(nq):
            fa_j = cx_exec_prob_np(da_eq[j], self.avg_comp, self.K, self.N) * self.lambda_a
            fb_j = cx_exec_prob_np(db_eq[j], self.avg_comp, self.K, self.N) * self.lambda_b
            sigma_q_grid[j] = np.sqrt(max((fa_j + fb_j) * self.Delta**2, 1e-8))

        # Z_target = σ_q · ∂V/∂q
        Z_target = sigma_q_grid * dVdq

        print(f"  Z targets: min={Z_target.min():.4f}, max={Z_target.max():.4f}", flush=True)
        print(f"  Z(q=0) = {Z_target[nq//2]:.6f} (should be ~0)", flush=True)

        # Tensors
        Z_t = torch.tensor(Z_target, dtype=torch.float64, device=self.device)
        q_norm = torch.tensor(q_grid / self.Q, dtype=torch.float64,
                              device=self.device).unsqueeze(1)

        # Pre-train
        opt = torch.optim.Adam(self.z_net.parameters(), lr=1e-3)
        for step in range(n_pretrain):
            # Random t (network should give same output for any t — stationarity)
            t_vals = torch.rand(nq, 1, dtype=torch.float64, device=self.device)
            Z_pred = self.z_net(t_vals, q_norm).squeeze(1)
            loss = torch.mean((Z_pred - Z_t)**2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 500 == 0:
                print(f"    pretrain step {step}: loss={loss.item():.6f}", flush=True)

        print(f"  Warm-start done. Final loss: {loss.item():.6f}", flush=True)

    def train(self, early_stopping=True, es_patience=800, es_min_delta=1e-7,
              es_warmup=2000):
        """Train by minimising E[|Y_T - g(q_T)|²]. Early stopping by default."""
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
            Y_T, q_T = self.forward()

            g_T = self.terminal_condition(q_T)
            loss = torch.mean((Y_T - g_T) ** 2)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.z_net.parameters()) + [self.Y0], max_norm=5.0)
            self.optimizer.step()
            self.scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

            if step % 200 == 0 or step == self.n_iter - 1:
                y0_q0 = self.Y0[self.Q].item()
                lr_now = self.optimizer.param_groups[0]['lr']
                print(f"  step {step:5d}: loss={loss.item():.4e}, "
                      f"best={best_loss:.4e}, Y0(q=0)={y0_q0:.4f}, "
                      f"lr={lr_now:.1e}", flush=True)
                history.append({
                    "step": step, "loss": loss.item(),
                    "best_loss": best_loss, "Y0_q0": y0_q0
                })

            if es is not None and es(loss.item()):
                print(f"  Early stopping at step {step} (best={es.best_loss:.4e})",
                      flush=True)
                break

        elapsed = time.time() - start

        # Extract Z profile and quotes at t=0
        from scipy.optimize import minimize_scalar
        q_grid = np.arange(-self.Q, self.Q + self.Delta, self.Delta)
        Z_profile = []
        self.z_net.eval()
        with torch.no_grad():
            for q in q_grid:
                t_n = torch.tensor([[0.0]], dtype=torch.float64, device=self.device)
                q_n = torch.tensor([[q / self.Q]], dtype=torch.float64, device=self.device)
                Z_val = self.z_net(t_n, q_n).item()

                # Compute sigma_q and dVdq
                # Use equilibrium sigma_q estimate
                sigma_q_est = self.sigma_q_init
                dVdq = Z_val / max(sigma_q_est, 1e-4)

                # Quotes from FOC
                U_eff_a = -self.Delta * dVdq
                U_eff_b = self.Delta * dVdq

                def _neg(d, U_v):
                    f = cx_exec_prob_np(d, self.avg_comp, self.K, self.N)
                    return -f * (d * self.Delta + U_v)

                da = minimize_scalar(lambda d: _neg(d, U_eff_a),
                                     bounds=(-1, 8), method='bounded').x
                db = minimize_scalar(lambda d: _neg(d, U_eff_b),
                                     bounds=(-1, 8), method='bounded').x

                Z_profile.append({
                    "q": float(q), "Z": Z_val, "dVdq": dVdq,
                    "da": da, "db": db, "spread": da + db
                })

        # Z profile at t=T/2 for stationarity check
        Z_profile_mid = []
        with torch.no_grad():
            for q in q_grid:
                t_n = torch.tensor([[0.5]], dtype=torch.float64, device=self.device)
                q_n = torch.tensor([[q / self.Q]], dtype=torch.float64, device=self.device)
                Z_val = self.z_net(t_n, q_n).item()
                Z_profile_mid.append({"q": float(q), "Z": Z_val})

        # Y0 profile
        Y0_profile = []
        for i, q in enumerate(q_grid):
            Y0_profile.append({"q": float(q), "Y0": self.Y0[i].item()})

        return {
            "Y0_profile": Y0_profile,
            "Z_profile": Z_profile,
            "Z_profile_mid": Z_profile_mid,
            "history": history,
            "elapsed": elapsed,
            "T": self.T, "M": self.M,
            "best_loss": best_loss,
            "variant": "diffusion_Z",
        }


if __name__ == "__main__":
    print("=" * 60)
    print("CX BSDE Solver -- CONTINUOUS INVENTORY (Z != 0)")
    print("Genuine deep BSDE: network learns Z = sigma_q * dV/dq")
    print("=" * 60, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    solver = CXBSDEDiffusion(
        N=2, Q=5, Delta=1,
        T=10.0, M=50,
        lambda_a=2.0, lambda_b=2.0,
        r=0.01, phi=0.005,
        device=device,
        lr=5e-4,
        n_iter=10000,
        batch_size=512,
        hidden=128, n_layers=3,
    )

    print(f"\nParams: N={solver.N}, Q={solver.Q}, T={solver.T}, M={solver.M}")
    print(f"Grid: {solver.nq} levels, K={solver.K}")
    print(f"sigma_q init: {solver.sigma_q_init:.4f}", flush=True)

    solver.warmstart_from_bellman(n_pretrain=2000)
    print(flush=True)

    result = solver.train()

    print(f"\nTraining complete in {result['elapsed']:.1f}s")
    print(f"Best loss: {result['best_loss']:.4e}", flush=True)

    # Z profile
    print("\n--- Z Profile at t=0 ---")
    for item in result["Z_profile"]:
        print(f"  q={item['q']:+.0f}: Z={item['Z']:+.4f}, "
              f"dV/dq={item['dVdq']:+.4f}, spread={item['spread']:.3f}", flush=True)

    # Stationarity check
    print("\n--- Z at t=0 vs t=T/2 ---")
    for z0, zm in zip(result["Z_profile"], result["Z_profile_mid"]):
        diff = abs(z0["Z"] - zm["Z"])
        print(f"  q={z0['q']:+.0f}: Z(0)={z0['Z']:+.4f}, "
              f"Z(T/2)={zm['Z']:+.4f}, diff={diff:.4f}", flush=True)

    # Key result
    spread_q0 = result["Z_profile"][5]["spread"]  # q=0 is index 5
    Z_q0 = result["Z_profile"][5]["Z"]
    print(f"\nSpread at q=0: {spread_q0:.4f}")
    print(f"Z at q=0: {Z_q0:.6f} (should be ~0)")
    print(f"Stationary Nash: 1.5153")
    print(f"Error: {abs(spread_q0 - 1.5153)/1.5153:.1%}", flush=True)

    # Antisymmetry check
    print("\n--- Antisymmetry check: Z(q) + Z(-q) ~ 0 ---")
    for i in range(solver.nq // 2):
        j = solver.nq - 1 - i
        z_pos = result["Z_profile"][j]["Z"]
        z_neg = result["Z_profile"][i]["Z"]
        asym = z_pos + z_neg
        print(f"  q=+/-{int(result['Z_profile'][j]['q'])}: "
              f"Z({int(result['Z_profile'][j]['q'])})={z_pos:+.4f}, "
              f"Z({int(result['Z_profile'][i]['q'])})={z_neg:+.4f}, "
              f"sum={asym:+.4f}", flush=True)

    out_path = "results_bsde_diffusion.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved to {out_path}", flush=True)

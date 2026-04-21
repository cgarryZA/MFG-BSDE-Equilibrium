#!/usr/bin/env python -u
"""Non-local jump kernel BSDEJ.

CX assumes jumps of fixed size ±Δ (one unit of inventory per RFQ).
In reality, RFQs can be for any size. Modelled as a jump kernel:

  dN^a has mark distribution ξ ~ ρ(ξ) on [Δ_min, Δ_max]
  dq = -ξ dN^a (ask execution removes ξ units of inventory)

The generator of the BSDE involves an integral over the kernel:

  integral over ξ of f_a(δ, avg, ξ) * [δ*ξ + V(q-ξ) - V(q)] * ρ(ξ) dξ

The exact solver would need to discretise both q AND ξ — grid blows up.
The NN handles ξ as a parameter input naturally.

This IS what Wang et al. (2023) were designed for — jumps with a continuous
mark space driven by compensated Poisson random measures.

Run: python -u scripts/nonlocal_jump_kernel.py
"""

import sys, os, json, time
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equations.contxiong_exact import cx_exec_prob_np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


class NonlocalValueNet(nn.Module):
    """V(q) network (standard), returns scalar."""
    def __init__(self, hidden=64, n_layers=2, dtype=torch.float64):
        super().__init__()
        layers = [nn.Linear(1, hidden, dtype=dtype), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden, dtype=dtype), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, q_norm):
        return self.net(q_norm).squeeze(-1)


class NonlocalJumpSolver:
    """CX with non-local jump kernel.

    Jump size distribution ρ(ξ) — we use truncated geometric:
      ρ(ξ=k) = (1-p)*p^{k-1} for k=1,2,...,ξ_max

    Each execution event removes/adds ξ units (randomly drawn).
    Expected jump size: 1/(1-p).
    """

    def __init__(self, N=2, Q=10, lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
                 geometric_p=0.5, xi_max=4,
                 device=None, lr=1e-3, n_iter=5000):
        self.N = N; self.Q = Q
        self.lambda_a = lambda_a; self.lambda_b = lambda_b
        self.r = r; self.phi = phi
        self.geometric_p = geometric_p
        self.xi_max = xi_max
        self.device = device or torch.device("cpu")
        self.n_iter = n_iter

        # Kernel: rho(xi=k) for k=1..xi_max
        self.xi_values = np.arange(1, xi_max + 1)
        raw = (1 - geometric_p) * geometric_p ** (self.xi_values - 1)
        self.rho = raw / raw.sum()  # normalise
        self.expected_xi = float(np.sum(self.xi_values * self.rho))
        print(f"  Jump kernel: xi in {list(self.xi_values)}, "
              f"rho = {[f'{r:.3f}' for r in self.rho]}, E[xi]={self.expected_xi:.2f}", flush=True)

        # Inventory grid for FOC etc
        self.q_grid = np.arange(-Q, Q + 1, 1, dtype=float)
        self.nq = len(self.q_grid)
        self.K = (N - 1) * self.nq
        self.avg_comp = 0.75

        self.net = NonlocalValueNet(hidden=128, n_layers=3).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.q_norm = torch.tensor(self.q_grid / Q, dtype=torch.float64,
                                  device=self.device).unsqueeze(1)

    def psi(self, q):
        return self.phi * q**2

    def V_q(self, q):
        """Evaluate V at arbitrary q (scalar or array)."""
        q_arr = np.atleast_1d(q)
        q_norm = torch.tensor(q_arr / self.Q, dtype=torch.float64,
                             device=self.device).unsqueeze(1)
        return self.net(q_norm).detach().cpu().numpy()

    def bellman_residual(self):
        """Non-local Bellman:
        r V(q) + psi(q) = sum over xi of rho(xi) * [
          lambda_a * max_delta f_a(delta)(delta*xi + V(q-xi) - V(q))
          + lambda_b * max_delta f_b(delta)(delta*xi + V(q+xi) - V(q))
        ]

        For simplicity at each q, we use the same optimal delta FOC as CX,
        but integrate across xi.
        """
        V = self.net(self.q_norm)  # [nq]

        # For each xi in kernel and each q, compute V(q - xi) and V(q + xi)
        # Out of range (|q ± xi| > Q) → boundary: use -psi(q ± xi) as terminal-like
        residuals = torch.zeros(self.nq, dtype=torch.float64, device=self.device)
        from equations.contxiong_exact import optimal_quote_foc

        # First compute V_np for numpy FOC
        V_np = V.detach().cpu().numpy()

        for j in range(self.nq):
            q = self.q_grid[j]

            # Integrate over xi
            total_profit_a = 0.0
            total_profit_b = 0.0
            for xi_idx, xi in enumerate(self.xi_values):
                rho_xi = self.rho[xi_idx]

                # V(q - xi): clip to boundary penalty if outside grid
                if q - xi >= -self.Q:
                    V_minus_np = V_np[int(q - xi + self.Q)]
                else:
                    V_minus_np = -self.psi(q - xi)  # approximate terminal cost

                if q + xi <= self.Q:
                    V_plus_np = V_np[int(q + xi + self.Q)]
                else:
                    V_plus_np = -self.psi(q + xi)

                # Optimal quotes (FOC with xi-specific value difference)
                if q > -self.Q:
                    p_a = (V_np[j] - V_minus_np) / xi
                    delta_a_star = optimal_quote_foc(p_a, self.avg_comp, self.K, self.N)
                    f_a = cx_exec_prob_np(delta_a_star, self.avg_comp, self.K, self.N)
                    total_profit_a += rho_xi * self.lambda_a * f_a * (
                        delta_a_star * xi + V_minus_np - V_np[j])

                if q < self.Q:
                    p_b = (V_np[j] - V_plus_np) / xi
                    delta_b_star = optimal_quote_foc(p_b, self.avg_comp, self.K, self.N)
                    f_b = cx_exec_prob_np(delta_b_star, self.avg_comp, self.K, self.N)
                    total_profit_b += rho_xi * self.lambda_b * f_b * (
                        delta_b_star * xi + V_plus_np - V_np[j])

            # Residual (torch)
            residuals[j] = self.r * V[j] + self.phi * q**2 - total_profit_a - total_profit_b

        return residuals

    def train(self):
        from utils import EarlyStopping
        es = EarlyStopping(patience=500, min_delta=1e-8, warmup=1000)
        history = []
        for step in range(self.n_iter):
            res = self.bellman_residual()
            loss = torch.sum(res**2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 500 == 0 or step == self.n_iter - 1:
                V_q0 = self.net(torch.tensor([[0.0]], dtype=torch.float64,
                                             device=self.device)).item()
                print(f"  step {step:5d}: loss={loss.item():.4e}, V(0)={V_q0:.4f}",
                      flush=True)
                history.append({"step": step, "loss": float(loss.item()),
                               "V_q0": float(V_q0)})

            if es(loss.item()):
                print(f"  Early stopped at step {step}", flush=True)
                break

        return history


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Non-local jump kernel BSDEJ (Wang et al. natural territory)")
    print(f"{'='*60}", flush=True)

    # Run with geometric jump size distribution
    solver = NonlocalJumpSolver(
        N=2, Q=10, lambda_a=2.0, lambda_b=2.0,
        geometric_p=0.5, xi_max=4,
        device=device, lr=1e-3, n_iter=5000,
    )
    history = solver.train()

    # Evaluate final V and quotes
    V_np = solver.V_q(solver.q_grid)
    print(f"\n  Final V:")
    for q, v in zip(solver.q_grid, V_np):
        print(f"    q={q:+.0f}: V={v:.4f}")

    result = {
        "xi_values": solver.xi_values.tolist(),
        "rho": solver.rho.tolist(),
        "expected_xi": solver.expected_xi,
        "V_final": V_np.tolist(),
        "q_grid": solver.q_grid.tolist(),
        "history": history,
    }
    with open("results_final/nonlocal_jump_kernel.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved to results_final/nonlocal_jump_kernel.json", flush=True)
    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

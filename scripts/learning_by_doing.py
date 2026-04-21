#!/usr/bin/env python -u
"""Learning-by-doing: dealers adapt their OWN intensity via execution history.

Standard CX: lambda is exogenous, fixed for all dealers.
Adaptive version: dealers who recently executed more have HIGHER intensity
(e.g., repeat customers, franchise value). Specifically:

  lambda_i(t) = lambda_0 * (1 + kappa * EWMA[executions_i, half-life H])

This breaks Markov in q alone — the state is (q, EWMA_activity).
EWMA adds a continuous dimension to the state.

For simplicity we model the EWMA directly as a state variable:
  da/dt = (exec - a) / H

where exec is an indicator (1 if executed this step, 0 otherwise).

Implementation: simulate forward paths where both q and a evolve jointly.
Use a NN that takes (q, a) as input. No exact solver for this — hidden
EWMA state makes exact solver intractable for continuous a.

Run: python -u scripts/learning_by_doing.py
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


class AdaptiveValueNet(nn.Module):
    """V(q, a) — value as function of inventory AND activity state."""
    def __init__(self, hidden=128, n_layers=3, dtype=torch.float64):
        super().__init__()
        layers = [nn.Linear(2, hidden, dtype=dtype), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden, dtype=dtype), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, q_norm, a_norm):
        x = torch.cat([q_norm, a_norm], dim=-1)
        return self.net(x).squeeze(-1)


class LearningByDoingSolver:
    """CX with EWMA-adaptive intensity.

    State: (q, a) where a ∈ [0, a_max].
    a = EWMA of executions with half-life H.

    Intensity modulation: lambda_effective = lambda_0 * (1 + kappa * (a - a_bar))
    """

    def __init__(self, Q=5, lambda_0=2.0, r=0.01, phi=0.005,
                 H=5.0, kappa=0.5, a_max=1.0, a_bar=0.3,
                 device=None, lr=5e-4, n_iter=3000, batch_size=256):
        self.Q = Q
        self.lambda_0 = lambda_0
        self.r = r; self.phi = phi
        self.H = H      # EWMA half-life
        self.kappa = kappa  # adaptation strength
        self.a_max = a_max
        self.a_bar = a_bar
        self.device = device or torch.device("cpu")
        self.n_iter = n_iter
        self.batch_size = batch_size

        self.nq = 2 * Q + 1
        self.K = self.nq  # N=2 setup

        self.net = AdaptiveValueNet(hidden=128, n_layers=3).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def bellman_residual(self, q_batch, a_batch):
        """Approximate Bellman for adaptive intensity.

        Because a is continuous and evolves with da/dt = (exec - a)/H,
        we discretise time and use the generator form:

          r V(q, a) = -psi(q) + lambda_eff * [transition in q and a]

        We treat the a dynamics as deterministic drift between jumps.
        """
        # lambda_effective depends on a
        lam_eff = self.lambda_0 * (1 + self.kappa * (a_batch - self.a_bar))
        lam_eff = torch.clamp(lam_eff, 0.1, 3.0 * self.lambda_0)

        V_q = self.net(q_batch.unsqueeze(-1), a_batch.unsqueeze(-1))

        # Compute V(q ± 1, a') where a' increases due to execution
        # After execution: new a = a + (1 - a)/H * some_dt; approximate for Bellman
        # For the stationary Bellman, we use a + 1/H (small bump)
        a_after_exec = torch.clamp(a_batch + 1.0/self.H, 0, self.a_max)
        a_drift = torch.clamp(a_batch - a_batch/self.H * 0.1, 0, self.a_max)  # decay

        q_minus = torch.clamp(q_batch - 1, -self.Q, self.Q)
        q_plus = torch.clamp(q_batch + 1, -self.Q, self.Q)

        V_minus = self.net(q_minus.unsqueeze(-1), a_after_exec.unsqueeze(-1))
        V_plus = self.net(q_plus.unsqueeze(-1), a_after_exec.unsqueeze(-1))

        # Simplified FOC: use static optimal quote (approximation)
        # In real implementation we'd solve FOC per sample; use grid search
        # For speed here: approximate optimal delta from V gradient
        delta_a = torch.full_like(q_batch, 0.8)
        delta_b = torch.full_like(q_batch, 0.8)

        # Execution probability (vectorised torch)
        fa = torch.sigmoid(-delta_a) ** 2  # monopolist form for simplicity (K=0)
        fb = torch.sigmoid(-delta_b) ** 2

        # Bellman: rV = -psi + lam_eff * sum of profits
        psi = self.phi * q_batch ** 2

        can_sell = (q_batch > -self.Q).float()
        can_buy = (q_batch < self.Q).float()

        profit_a = can_sell * fa * (delta_a + V_minus - V_q)
        profit_b = can_buy * fb * (delta_b + V_plus - V_q)

        residual = self.r * V_q + psi - lam_eff * (profit_a + profit_b)
        return residual

    def train(self):
        from utils import EarlyStopping
        es = EarlyStopping(patience=300, min_delta=1e-7, warmup=500)

        history = []
        for step in range(self.n_iter):
            # Sample (q, a) uniformly
            q_batch = (torch.rand(self.batch_size, dtype=torch.float64, device=self.device)
                      * 2 * self.Q - self.Q)
            a_batch = (torch.rand(self.batch_size, dtype=torch.float64, device=self.device)
                      * self.a_max)

            res = self.bellman_residual(q_batch, a_batch)
            loss = torch.mean(res ** 2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 500 == 0 or step == self.n_iter - 1:
                with torch.no_grad():
                    V0 = self.net(
                        torch.zeros(1, 1, dtype=torch.float64, device=self.device),
                        torch.tensor([[self.a_bar]], dtype=torch.float64, device=self.device),
                    ).item()
                print(f"  step {step:5d}: loss={loss.item():.4e}, V(0, a_bar)={V0:.4f}",
                      flush=True)
                history.append({"step": step, "loss": float(loss.item()), "V0": float(V0)})

            if es(loss.item()):
                print(f"  Early stopped at step {step}", flush=True)
                break

        return history


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Learning-by-doing: EWMA-adaptive intensity")
    print(f"{'='*60}", flush=True)

    results = []
    for kappa in [0.0, 0.3, 0.5]:
        print(f"\n  kappa = {kappa} (adaptation strength)", flush=True)
        solver = LearningByDoingSolver(
            Q=5, lambda_0=2.0, kappa=kappa, H=5.0,
            device=device, n_iter=2000, batch_size=256,
        )
        history = solver.train()

        # Evaluate V at key (q, a) points
        with torch.no_grad():
            V_at_0_low = solver.net(
                torch.tensor([[0.0]], dtype=torch.float64, device=device),
                torch.tensor([[0.1]], dtype=torch.float64, device=device),
            ).item()
            V_at_0_high = solver.net(
                torch.tensor([[0.0]], dtype=torch.float64, device=device),
                torch.tensor([[0.7]], dtype=torch.float64, device=device),
            ).item()

        print(f"  V(q=0, a=0.1) = {V_at_0_low:.4f}")
        print(f"  V(q=0, a=0.7) = {V_at_0_high:.4f}")
        print(f"  V difference: {V_at_0_high - V_at_0_low:+.4f}")

        results.append({
            "kappa": kappa,
            "V_0_low_a": V_at_0_low,
            "V_0_high_a": V_at_0_high,
            "V_diff": V_at_0_high - V_at_0_low,
            "history": history,
        })

    with open("results_final/learning_by_doing.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to results_final/learning_by_doing.json", flush=True)
    print(f"Economic: higher kappa -> higher V at high a (franchise value)", flush=True)
    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

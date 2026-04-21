#!/usr/bin/env python -u
"""Multi-asset CX with correlated order flow.

Standard K-asset model: each asset has independent Poisson RFQ flow.
Correlated extension: flows are coupled via a Hawkes-like structure —
an ask RFQ on asset k raises the intensity of bid RFQs on asset k' (for k' != k)
with some copula/correlation parameter eta.

Specifically:
  lambda_a^k(t) = lambda_0 * (1 + eta * sum_{k' != k} recent_bid_activity[k'])

This is a proper curse-of-dimensionality regime: exact needs to track
joint activity state, nq^K * (activity states). NN handles it natively
with state (q, recent_activity).

Simplified version for testing: instead of time-dependent activity, we use
a static correlation: execution on asset k changes avg_da for asset k' via
a correlation matrix.

Run: python -u scripts/multiasset_correlated.py
"""

import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_cx_multiasset import CXMultiAssetSolver, MultiAssetValueNet
from equations.contxiong_exact import cx_exec_prob_np, cx_exec_prob_torch
from scipy.optimize import minimize_scalar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


class CXCorrelatedSolver(CXMultiAssetSolver):
    """K-asset CX with correlated execution intensities.

    Correlation: flow on asset k scales with (1 + eta * sum_{k'!=k} quote_pressure[k']),
    where quote_pressure is captured via the cross-asset average quote spread.

    This couples the assets: tighter quotes on asset k' -> more flow on asset k
    (business stealing or complementarity, depending on sign of eta).
    """

    def __init__(self, K=2, eta=0.3, **kwargs):
        super().__init__(K=K, **kwargs)
        self.eta = eta

    def bellman_loss(self, q_batch, avg_da_per_asset, avg_db_per_asset):
        """Modified Bellman with cross-asset correlation in lambdas."""
        batch = q_batch.shape[0]
        V_q = self.V(q_batch)

        # Cross-asset pressure: average spread on other assets
        avg_spreads = [avg_da_per_asset[k] + avg_db_per_asset[k]
                      for k in range(self.K)]
        mean_other_pressure = []  # for each k, avg spread of other assets
        for k in range(self.K):
            others = [avg_spreads[kk] for kk in range(self.K) if kk != k]
            mean_other_pressure.append(np.mean(others) if others else 0.0)

        total_profit = torch.zeros(batch, 1, dtype=torch.float64, device=self.device)

        for k in range(self.K):
            q_minus = q_batch.clone()
            q_minus[:, k] = torch.clamp(q_minus[:, k] - self.Delta, -self.Q, self.Q)
            q_plus = q_batch.clone()
            q_plus[:, k] = torch.clamp(q_plus[:, k] + self.Delta, -self.Q, self.Q)
            V_minus = self.V(q_minus)
            V_plus = self.V(q_plus)

            # Correlated intensity: lambda_k scaled by (1 + eta * other pressure)
            # When other assets have tighter spreads, our flow increases.
            scale = 1.0 + self.eta * (mean_other_pressure[k] - 1.5)  # centred on Nash
            scale = max(0.3, min(2.0, scale))  # clip
            lam_a_k = self.lambda_a * scale
            lam_b_k = self.lambda_b * scale

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

            # NOTE: scaled lam_a_k, lam_b_k
            profit_a = can_sell * lam_a_k * self.Delta * fa * (da_t - (V_q - V_minus) / self.Delta)
            profit_b = can_buy * lam_b_k * self.Delta * fb * (db_t - (V_q - V_plus) / self.Delta)
            total_profit = total_profit + profit_a + profit_b

        psi_q = self.psi(q_batch)
        residual = self.r * V_q + psi_q - total_profit
        return torch.mean(residual ** 2)


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Multi-asset K=2 with correlated order flow")
    print(f"{'='*60}", flush=True)

    # Run at multiple eta values
    results = []
    for eta in [0.0, 0.3, -0.3]:  # 0: uncorrelated, 0.3: complementary, -0.3: substitutive
        print(f"\n  eta = {eta:+.2f}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        solver = CXCorrelatedSolver(
            K=2, N=2, Q=5, eta=eta,
            device=device, n_iter=2000, batch_size=64, lr=1e-3,
        )
        t0 = time.time()
        r = solver.train()
        elapsed = time.time() - t0

        print(f"  eta={eta:+.2f}: V(0,0)={r['V_0']:.4f}, spreads={r['spreads_per_asset']}, time={elapsed:.0f}s",
              flush=True)
        results.append({
            "eta": eta, "V_00": r['V_0'],
            "spreads": r['spreads_per_asset'],
            "elapsed": elapsed,
        })

    with open("results_final/multiasset_correlated.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n{'='*60}")
    print("CORRELATED MULTI-ASSET SUMMARY")
    print(f"{'='*60}")
    print(f"{'eta':>6s}  {'V(0,0)':>10s}  {'spread 1':>10s}  {'spread 2':>10s}")
    for r in results:
        s = r['spreads']
        print(f"{r['eta']:+6.2f}  {r['V_00']:10.4f}  {s[0]:10.4f}  {s[1]:10.4f}")
    print(f"\nEconomic interpretation:")
    print(f"  eta>0: tighter other asset -> more flow on this asset (complementary)")
    print(f"  eta<0: tighter other asset -> less flow on this asset (substitution)")
    print(f"\nSaved to results_final/multiasset_correlated.json", flush=True)
    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

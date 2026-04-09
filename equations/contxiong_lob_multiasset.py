"""
Multi-asset Cont-Xiong LOB with MV coupling.

State: (S_1, ..., S_K, q_1, ..., q_K) — K assets, dim = 2K
Each asset has its own price and inventory.
Cross-asset interaction through:
  - Shared competitive factor h(mu) from joint population distribution
  - Inventory penalty across all positions: psi(q) = phi * sum(q_k^2)

This tests the curse of dimensionality: at K=5 (d=10), FD is intractable
but deep BSDE should still work.

Parameters:
  n_assets: number of assets (K)
  sigma_s: price volatility (same for all assets, or list)
  correlation: cross-asset price correlation (0 = independent)
"""

import numpy as np
import torch
import time

from registry import register_equation
from equations.contxiong_lob_mv import ContXiongLOBMV, CompetitiveFactorNet


@register_equation("contxiong_lob_multiasset")
class ContXiongLOBMultiAsset(ContXiongLOBMV):
    """Multi-asset MV market-making.

    Extends ContXiongLOBMV to K assets. State dim = 2K.
    Inherits law encoder and competitive factor machinery.
    """

    def __init__(self, eqn_config):
        self.n_assets = getattr(eqn_config, "n_assets", 1)
        K = self.n_assets
        super().__init__(eqn_config)

        # Override dim (base class hardcodes 2)
        self.dim = 2 * K

        # Override law encoder for multi-asset state
        from equations.law_encoders import create_law_encoder
        encoder_type = getattr(eqn_config, "law_encoder_type", "moments")
        self.law_encoder = create_law_encoder(
            encoder_type,
            state_dim=2 * K,
            embed_dim=getattr(eqn_config, "law_embed_dim", 16),
            n_bins=getattr(eqn_config, "n_bins", 20),
            q_max=self.q_max,
        )
        self.law_embed_dim = self.law_encoder.embed_dim

        # Override competitive factor net for new embed dim
        self.competitive_factor_net = CompetitiveFactorNet(
            self.law_embed_dim, dtype=torch.float64
        )

        self.correlation = getattr(eqn_config, "correlation", 0.0)

        # Per-asset parameters (same for all for simplicity)
        self.sigma_s_arr = np.full(self.n_assets, self.sigma_s)
        self.lambda_a_arr = np.full(self.n_assets, self.lambda_a)
        self.lambda_b_arr = np.full(self.n_assets, self.lambda_b)

    def sample(self, num_sample, withtime=False, seed=None):
        """Sample multi-asset paths."""
        K = self.n_assets
        d = 2 * K
        if seed:
            np.random.seed(seed)

        # Generate correlated Brownian increments
        # Correlation matrix: block diagonal (prices correlated, inventories independent)
        dw_raw = np.random.normal(size=[num_sample, d, self.num_time_interval])
        if self.correlation > 0:
            # Apply correlation to price components (first K dimensions)
            L = np.eye(K) * np.sqrt(1 - self.correlation) + np.full((K, K), np.sqrt(self.correlation / K))
            for t in range(self.num_time_interval):
                dw_raw[:, :K, t] = dw_raw[:, :K, t] @ L.T
        dw_sample = dw_raw * self.sqrt_delta_t

        if seed:
            np.random.seed(seed=int(time.time()))

        dt = self.delta_t
        x_sample = np.zeros([num_sample, self.num_time_interval + 1, d])

        # Initial conditions
        for k in range(K):
            x_sample[:, 0, k] = self.s_init  # prices
            x_sample[:, 0, K + k] = 0.0  # inventories start at 0

        for i in range(self.num_time_interval):
            h_factor = self._drift_predict_mc(i)

            for k in range(K):
                s_k = x_sample[:, i, k]
                q_k = x_sample[:, i, K + k]

                # Proxy Z for sampling
                z_q_proxy = -2 * self.phi * q_k
                sigma_q = self._sigma_q_equilibrium()
                p = z_q_proxy / sigma_q
                delta_a = 1.0 / self.alpha + p
                delta_b = 1.0 / self.alpha - p

                f_a = self._exec_prob_np(delta_a, h_factor) * self.lambda_a_arr[k]
                f_b = self._exec_prob_np(delta_b, h_factor) * self.lambda_b_arr[k]

                # Price
                x_sample[:, i + 1, k] = s_k + self.sigma_s_arr[k] * dw_sample[:, k, i]

                # Inventory
                inv_drift = (f_b - f_a) * dt
                inv_diff = np.sqrt(np.maximum(f_b + f_a, 1e-8)) * dw_sample[:, K + k, i]
                x_sample[:, i + 1, K + k] = np.clip(
                    q_k + inv_drift + inv_diff, -self.q_max, self.q_max
                )

        x_sample = x_sample.transpose((0, 2, 1))
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        """Generator for multi-asset model.

        Sum of per-asset profits minus joint inventory penalty.
        """
        K = self.n_assets
        d = 2 * K

        # Competitive factor from law embedding
        if self._current_law_embed is not None:
            h_factor = self.compute_competitive_factor(self._current_law_embed)
        else:
            t_val = t.item() if isinstance(t, torch.Tensor) else float(t)
            t_idx = min(int(round(t_val / self.delta_t)), self.num_time_interval)
            h_factor = self._drift_predict_mc(t_idx)

        total_profit = torch.zeros_like(y)
        total_penalty = torch.zeros_like(y)

        for k in range(K):
            q_k = x[:, K + k: K + k + 1]  # inventory for asset k
            z_q_k = z[:, K + k: K + k + 1]  # Z for inventory k

            delta_a, delta_b = self._optimal_quotes_tf(z_q_k)
            f_a = self._exec_prob_tf(delta_a, h_factor) * self.lambda_a_arr[k]
            f_b = self._exec_prob_tf(delta_b, h_factor) * self.lambda_b_arr[k]

            total_profit = total_profit + f_a * delta_a + f_b * delta_b
            total_penalty = total_penalty + self.phi * q_k ** 2

        return -self.discount_rate * y - total_penalty + total_profit

    def g_tf(self, t, x):
        """Terminal condition: sum of per-asset penalties."""
        K = self.n_assets
        penalty = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        for k in range(K):
            q_k = x[:, K + k: K + k + 1]
            penalty = penalty + self.phi * q_k ** 2
        return -penalty

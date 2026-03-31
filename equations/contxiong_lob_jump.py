# equations/contxiong_lob_jump.py
#
# Cont-Xiong (2024) LOB model — Option B: exact jump-diffusion BSDE.
# Keeps discrete Poisson inventory (no continuous relaxation).
#
# State: X_t = (S_t, q_t) where S = mid-price (Brownian), q = inventory (jumps)
# Forward SDE:
#   dS_t = sigma_s * dW_t
#   q jumps by +Delta on buy execution (Poisson, rate lambda_b * f_b)
#   q jumps by -Delta on sell execution (Poisson, rate lambda_a * f_a)
#
# BSDE with jumps (FBSDEJ):
#   dY_t = -f(t,X,Y,Z,U) dt + Z_t dW_t + U^+_t dN~^b_t + U^-_t dN~^a_t
#
# where N~^a, N~^b are compensated Poisson processes and:
#   Z_t: gradient of V w.r.t. price (ℝ^1)
#   U^+_t: V(S, q+Δ) - V(S, q) — value change on buy
#   U^-_t: V(S, q-Δ) - V(S, q) — value change on sell
#
# Reference: Cont & Xiong (2024), Mathematical Finance.

import time
import numpy as np
import torch

from registry import register_equation
from .base import Equation


@register_equation("contxiong_lob_jump")
class ContXiongLOBJump(Equation):
    """Cont-Xiong LOB with exact Poisson jumps for inventory (FBSDEJ)."""

    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.dim = 2  # (S, q) — but q is discrete

        # Market parameters
        self.sigma_s = getattr(eqn_config, "sigma_s", 0.3)
        self.lambda_a = getattr(eqn_config, "lambda_a", 1.0)
        self.lambda_b = getattr(eqn_config, "lambda_b", 1.0)
        self.alpha = getattr(eqn_config, "alpha", 1.5)
        self.phi = getattr(eqn_config, "phi", 0.01)
        self.discount_rate = getattr(eqn_config, "discount_rate", 0.1)
        self.delta = getattr(eqn_config, "order_size", 1.0)  # Δ: order size
        self.h_max = getattr(eqn_config, "h_max", 10)  # inventory limit ±H

        self.s_init = getattr(eqn_config, "s_init", 100.0)
        self.q_init = 0.0
        self.y_init = None

        # Mean-field (same as continuous version)
        self.mean_spread_estimate = np.ones(self.num_time_interval + 1) * (2.0 / self.alpha)

    # ------------------------------------------------------------------
    # Execution probabilities (same as continuous version)
    # ------------------------------------------------------------------

    def _exec_prob_np(self, delta_quote):
        """exp(-alpha * delta)"""
        delta_clamped = np.clip(delta_quote, -5.0 / self.alpha, 10.0 / self.alpha)
        return np.exp(-self.alpha * delta_clamped)

    def _exec_prob_tf(self, delta_quote):
        delta_clamped = torch.clamp(delta_quote, -5.0 / self.alpha, 10.0 / self.alpha)
        return torch.exp(-self.alpha * delta_clamped)

    def _optimal_quotes_tf(self, z_price, u_plus, u_minus):
        """Optimal quotes from HJB first-order condition with jumps.

        For jump model, optimal ask quote satisfies:
            delta_a* = argmax_delta [f_a(delta) * (delta*Delta + u_minus)]
        With f_a = exp(-alpha*delta), the FOC gives:
            delta_a* = 1/alpha - u_minus/Delta
            delta_b* = 1/alpha + u_plus/Delta

        Note: u_minus < 0 (selling inventory reduces value when q>0),
        so -u_minus/Delta > 0, widening the ask.
        """
        delta_a = 1.0 / self.alpha - u_minus / self.delta
        delta_b = 1.0 / self.alpha + u_plus / self.delta
        return delta_a, delta_b

    # ------------------------------------------------------------------
    # Forward SDE sampling with Poisson jumps
    # ------------------------------------------------------------------

    def sample(self, num_sample, withtime=False, seed=None):
        """Sample forward paths with Brownian price and Poisson inventory.

        Returns:
            dw_sample: [num_sample, 1, num_time_interval] — Brownian increments for price only
            x_sample: [num_sample, 2, num_time_interval+1] — (S, q) paths
            jump_data: dict with Poisson jump counts per timestep
        """
        if seed:
            np.random.seed(seed)

        dt = self.delta_t

        # Brownian increments for price (dim=1, not 2)
        dw_sample = (
            np.random.normal(size=[num_sample, 1, self.num_time_interval])
            * self.sqrt_delta_t
        )

        # State paths
        x_sample = np.zeros([num_sample, self.num_time_interval + 1, 2])
        x_sample[:, 0, 0] = self.s_init
        x_sample[:, 0, 1] = self.q_init

        # Poisson jump counts: N^a_n, N^b_n per timestep
        n_ask = np.zeros([num_sample, self.num_time_interval], dtype=np.int32)
        n_bid = np.zeros([num_sample, self.num_time_interval], dtype=np.int32)

        for i in range(self.num_time_interval):
            s = x_sample[:, i, 0]
            q = x_sample[:, i, 1]

            # Proxy optimal quotes (using inventory penalty gradient as Z proxy)
            z_q_proxy = -2 * self.phi * q
            delta_a_proxy = 1.0 / self.alpha + z_q_proxy
            delta_b_proxy = 1.0 / self.alpha - z_q_proxy

            # Execution rates
            rate_a = self.lambda_a * self._exec_prob_np(delta_a_proxy)
            rate_b = self.lambda_b * self._exec_prob_np(delta_b_proxy)

            # Draw Poisson jumps
            n_ask[:, i] = np.random.poisson(rate_a * dt)
            n_bid[:, i] = np.random.poisson(rate_b * dt)

            # Price update (Brownian)
            x_sample[:, i + 1, 0] = s + self.sigma_s * dw_sample[:, 0, i]

            # Inventory update (discrete jumps)
            q_new = q + self.delta * (n_bid[:, i] - n_ask[:, i])
            x_sample[:, i + 1, 1] = np.clip(q_new, -self.h_max, self.h_max)

        if seed:
            np.random.seed(seed=int(time.time()))

        # Transpose to [batch, features, time]
        x_sample = x_sample.transpose((0, 2, 1))

        return dw_sample, x_sample, {"n_ask": n_ask, "n_bid": n_bid}

    # ------------------------------------------------------------------
    # BSDE generator with jumps
    # ------------------------------------------------------------------

    def f_tf(self, t, x, y, z, u_plus, u_minus):
        """Generator for jump-diffusion BSDE.

        f(t, x, y, z, u+, u-) includes:
        - Discount: -r*y
        - Inventory penalty: phi * q^2
        - Expected execution profit (compensator terms)

        Args:
            t: scalar time
            x: [batch, 2] (S, q)
            y: [batch, 1]
            z: [batch, 1] — price gradient only
            u_plus: [batch, 1] — jump up value V(q+Δ)-V(q)
            u_minus: [batch, 1] — jump down value V(q-Δ)-V(q)
        """
        q = x[:, 1:2]

        # Optimal quotes
        delta_a, delta_b = self._optimal_quotes_tf(z, u_plus, u_minus)

        # Execution rates
        rate_a = self.lambda_a * self._exec_prob_tf(delta_a)
        rate_b = self.lambda_b * self._exec_prob_tf(delta_b)

        # Inventory penalty
        psi = self.phi * q ** 2

        # Compensator terms: rate * (profit per execution + jump in value)
        # For ask execution: profit = delta_a * Delta, value changes by u_minus
        # For bid execution: profit = delta_b * Delta, value changes by u_plus
        comp_a = rate_a * (delta_a * self.delta + u_minus)
        comp_b = rate_b * (delta_b * self.delta + u_plus)

        return -self.discount_rate * y + psi - comp_a - comp_b

    def g_tf(self, t, x):
        """Terminal condition: -phi * q_T^2"""
        q = x[:, 1:2]
        return -self.phi * q ** 2

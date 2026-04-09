"""
Cont-Xiong LOB with proper jump-diffusion for inventory.

The subnet outputs 4 values: (Z_s, Z_q, U_a, U_b) where:
  Z_s, Z_q: diffusion coefficients for the BSDE step
  U_a = V(q-1) - V(q): value jump on ask execution (sell)
  U_b = V(q+1) - V(q): value jump on bid execution (buy)

Optimal quotes from the discrete FOC:
  delta_a = 1/alpha - U_a    (NOT 1/alpha + Z_q/sigma_q)
  delta_b = 1/alpha - U_b

Spread = 2/alpha - U_a - U_b = 2/alpha - (V(q-1) + V(q+1) - 2V(q))
       > 2/alpha when V is convex (which it is due to inventory penalty)

This matches the FD ground truth spread (~1.43) rather than the
diffusion approximation spread (1.333 = 2/alpha).
"""

import numpy as np
import torch
import time

from registry import register_equation
from equations.contxiong_lob_mv import ContXiongLOBMV


@register_equation("contxiong_lob_jump")
class ContXiongLOBJump(ContXiongLOBMV):
    """MV LOB with discrete inventory jumps.

    Key difference from base: dim_output = 4 (Z_s, Z_q, U_a, U_b)
    instead of 2 (Z_s, Z_q). The extra outputs are value jump
    coefficients used for quote computation.
    """

    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.Delta_q = getattr(eqn_config, "Delta_q", 1.0)
        # Output dimension for subnets: 4 instead of 2
        self.subnet_output_dim = 4  # (Z_s, Z_q, U_a, U_b)
        # Storage for jump coefficients (set by model forward)
        self._current_U = None

    def set_current_jump_coeffs(self, U):
        """Store U_a, U_b for use by f_tf. Called by model forward."""
        self._current_U = U

    def f_tf(self, t, x, y, z):
        """Generator with learned jump coefficients.

        Uses U_a, U_b directly from the subnet output (not approximated
        from Z_q). The quotes are:
          delta_a = 1/alpha - U_a
          delta_b = 1/alpha - U_b
        """
        q = x[:, 1:2]

        # Get jump coefficients
        if self._current_U is not None:
            U_a = self._current_U[:, 0:1]  # V(q-1) - V(q)
            U_b = self._current_U[:, 1:2]  # V(q+1) - V(q)
        else:
            # Fallback: symmetric approximation from Z_q
            z_q = z[:, 1:2]
            sigma_q = self._sigma_q_equilibrium()
            U_a = -z_q * self.Delta_q / sigma_q
            U_b = z_q * self.Delta_q / sigma_q

        # Optimal quotes from discrete FOC
        delta_a = 1.0 / self.alpha - U_a
        delta_b = 1.0 / self.alpha - U_b

        # Competitive factor
        if self._current_law_embed is not None:
            h_factor = self.compute_competitive_factor(self._current_law_embed)
        else:
            t_val = t.item() if isinstance(t, torch.Tensor) else float(t)
            t_idx = min(int(round(t_val / self.delta_t)), self.num_time_interval)
            h_factor = self._drift_predict_mc(t_idx)

        # Clamp quotes for numerical stability
        delta_a = torch.clamp(delta_a, 0.001, 10.0 / self.alpha)
        delta_b = torch.clamp(delta_b, 0.001, 10.0 / self.alpha)

        # Execution rates
        f_a = self._exec_prob_tf(delta_a, h_factor) * self.lambda_a
        f_b = self._exec_prob_tf(delta_b, h_factor) * self.lambda_b

        # Profits: rate × (spread revenue + value change on execution)
        profit_a = f_a * (delta_a + U_a)
        profit_b = f_b * (delta_b + U_b)

        psi = self._penalty_tf(q)

        return -self.discount_rate * y - psi + profit_a + profit_b

    def sample(self, num_sample, withtime=False, seed=None):
        """Sample paths with Poisson inventory jumps."""
        if seed:
            np.random.seed(seed)
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
            * self.sqrt_delta_t
        )
        if seed:
            np.random.seed(seed=int(time.time()))

        dt = self.delta_t
        x_sample = np.zeros([num_sample, self.num_time_interval + 1, self.dim])
        x_sample[:, 0, 0] = self.s_init
        x_sample[:, 0, 1] = 0.0

        for i in range(self.num_time_interval):
            s = x_sample[:, i, 0]
            q = x_sample[:, i, 1]

            h_factor = self._drift_predict_mc(i)
            # Proxy quotes for sampling (use simple formula)
            z_q_proxy = -2 * self.phi * q
            sigma_q = self._sigma_q_equilibrium()
            p = z_q_proxy / sigma_q
            delta_a = 1.0 / self.alpha + p
            delta_b = 1.0 / self.alpha - p

            rate_a = self._exec_prob_np(delta_a, h_factor) * self.lambda_a
            rate_b = self._exec_prob_np(delta_b, h_factor) * self.lambda_b

            # Poisson jumps
            prob_a = np.clip(rate_a * dt, 0, 0.5)
            prob_b = np.clip(rate_b * dt, 0, 0.5)
            exec_a = (np.random.uniform(size=num_sample) < prob_a).astype(float)
            exec_b = (np.random.uniform(size=num_sample) < prob_b).astype(float)

            # Price: pure diffusion
            x_sample[:, i + 1, 0] = s + self.sigma_s * dw_sample[:, 0, i]

            # Inventory: discrete jumps
            x_sample[:, i + 1, 1] = np.clip(
                q - exec_a * self.Delta_q + exec_b * self.Delta_q,
                -self.q_max, self.q_max
            )

        x_sample = x_sample.transpose((0, 2, 1))
        return dw_sample, x_sample

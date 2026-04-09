"""
Cont-Xiong LOB with common noise.

All agents share a systematic market shock:
  dS_i = sigma_s * dW^S_i + sigma_common * dW^common

where dW^common is the same Brownian increment for all agents.
This means all agents' prices are correlated, which breaks the
conditional independence assumption of the standard mean-field limit.

Under common noise, the mean-field limit involves conditional
distributions (conditioned on the common noise path), and the
master equation becomes an SPDE on Wasserstein space (Carmona 2018a).

Parameters:
  sigma_common: common noise volatility (0 = no common noise = base model)
"""

import numpy as np
import torch
import time

from registry import register_equation
from equations.contxiong_lob_mv import ContXiongLOBMV


@register_equation("contxiong_lob_common_noise")
class ContXiongLOBCommonNoise(ContXiongLOBMV):
    """MV LOB with common noise affecting all agents' prices."""

    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.sigma_common = getattr(eqn_config, "sigma_common", 0.0)

    def sample(self, num_sample, withtime=False, seed=None):
        """Sample paths with common noise.

        The common noise dW^common is shared across all agents at each timestep.
        This creates correlation in price movements.
        """
        if seed:
            np.random.seed(seed)

        # Individual Brownian increments
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
            * self.sqrt_delta_t
        )

        # Common noise: same for all agents at each timestep
        dw_common = (
            np.random.normal(size=[1, 1, self.num_time_interval])
            * self.sqrt_delta_t
        )  # [1, 1, T] — broadcast to all agents

        if seed:
            np.random.seed(seed=int(time.time()))

        dt = self.delta_t
        x_sample = np.zeros([num_sample, self.num_time_interval + 1, self.dim])
        x_sample[:, 0, 0] = self.s_init
        x_sample[:, 0, 1] = self.q_init

        for i in range(self.num_time_interval):
            s = x_sample[:, i, 0]
            q = x_sample[:, i, 1]

            h_factor = self._drift_predict_mc(i)
            z_q_proxy = -2 * self.phi * q
            delta_a, delta_b = self._optimal_quotes_np(z_q_proxy)

            f_a = self._exec_prob_np(delta_a, h_factor) * self.lambda_a
            f_b = self._exec_prob_np(delta_b, h_factor) * self.lambda_b

            # Price: idiosyncratic + common noise
            x_sample[:, i + 1, 0] = (
                s
                + self.sigma_s * dw_sample[:, 0, i]
                + self.sigma_common * dw_common[0, 0, i]  # shared across batch
            )

            # Inventory: same as base
            inv_drift = (f_b - f_a) * dt
            inv_diff = np.sqrt(np.maximum(f_b + f_a, 1e-8)) * dw_sample[:, 1, i]
            x_sample[:, i + 1, 1] = np.clip(
                q + inv_drift + inv_diff, -self.q_max, self.q_max
            )

        x_sample = x_sample.transpose((0, 2, 1))
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        """Generator — same as MV base.

        Common noise affects the SDE dynamics (price paths are correlated)
        but the generator structure is unchanged. The effect shows up through
        the population distribution being correlated, not through the f function.
        """
        return super().f_tf(t, x, y, z)

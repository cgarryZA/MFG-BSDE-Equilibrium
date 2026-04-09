"""
Cont-Xiong LOB with permanent price impact.

Extends the base model with Almgren-Chriss style permanent impact:
when a trade executes, the mid-price shifts proportionally.

  dS_t = sigma_s * dW^S_t + kappa * (dN^b_t - dN^a_t)

where kappa is the permanent impact coefficient and dN^a, dN^b are
the execution counting processes.

This breaks the dimensional reduction V = x + qs + phi(t,q) because
the price now depends on the agent's actions, making S dynamically
relevant. The state is genuinely 2D: (S, q).

Key parameters:
  kappa: permanent impact strength (0 = no impact = base model)
  impact_nonlinearity: "linear" | "sqrt" | "quadratic"
    - linear: impact = kappa * (dN^b - dN^a)
    - sqrt: impact = kappa * sign(flow) * sqrt(|flow|) (concave, more realistic)
    - quadratic: impact = kappa * flow^2 * sign(flow) (convex, severely breaks monotonicity)

Under sqrt or quadratic impact, the Lasry-Lions monotonicity condition
is violated, which is exactly what the lit review promises to test.
"""

import numpy as np
import torch

from registry import register_equation
from equations.contxiong_lob_mv import ContXiongLOBMV


@register_equation("contxiong_lob_impact")
class ContXiongLOBImpact(ContXiongLOBMV):
    """MV LOB with permanent price impact.

    Inherits all MV machinery (law encoders, competitive factor, etc.)
    and adds price impact to the SDE dynamics.
    """

    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.kappa = getattr(eqn_config, "kappa", 0.0)
        self.impact_type = getattr(eqn_config, "impact_type", "linear")

    def _impact_fn_np(self, net_flow):
        """Compute price impact from net order flow (numpy)."""
        if self.impact_type == "linear":
            return self.kappa * net_flow
        elif self.impact_type == "sqrt":
            return self.kappa * np.sign(net_flow) * np.sqrt(np.abs(net_flow))
        elif self.impact_type == "quadratic":
            return self.kappa * net_flow * np.abs(net_flow)
        else:
            return self.kappa * net_flow

    def _impact_fn_tf(self, net_flow):
        """Compute price impact from net order flow (torch)."""
        if self.impact_type == "linear":
            return self.kappa * net_flow
        elif self.impact_type == "sqrt":
            return self.kappa * torch.sign(net_flow) * torch.sqrt(torch.abs(net_flow) + 1e-8)
        elif self.impact_type == "quadratic":
            return self.kappa * net_flow * torch.abs(net_flow)
        else:
            return self.kappa * net_flow

    def f_tf(self, t, x, y, z):
        """Generator with price impact + MV coupling.

        The key difference from base: profit calculation accounts for
        price impact. When agent sells (ask executed), price drops by
        impact; when buys (bid executed), price rises.

        Mark-to-market effect: inventory value changes due to price impact.
        """
        s = x[:, 0:1]
        q = x[:, 1:2]
        z_s = z[:, 0:1]
        z_q = z[:, 1:2]

        delta_a, delta_b = self._optimal_quotes_tf(z_q)

        # Competitive factor from law embedding
        if self._current_law_embed is not None:
            h_factor = self.compute_competitive_factor(self._current_law_embed)
        else:
            t_val = t.item() if isinstance(t, torch.Tensor) else float(t)
            t_idx = min(int(round(t_val / self.delta_t)), self.num_time_interval)
            h_factor = self._drift_predict_mc(t_idx)

        f_a = self._exec_prob_tf(delta_a, h_factor) * self.lambda_a
        f_b = self._exec_prob_tf(delta_b, h_factor) * self.lambda_b

        # Net order flow (expected): buys - sells
        net_flow = f_b - f_a  # positive = net buying

        # Price impact on value: q * d(price impact)
        # When the agent is long (q > 0) and net buying pushes price up, that's good
        # Impact enters as: q * kappa * net_flow (mark-to-market from impact)
        impact_value = q * self._impact_fn_tf(net_flow)

        psi = self._penalty_tf(q)
        profit_a = f_a * delta_a
        profit_b = f_b * delta_b

        return -self.discount_rate * y - psi + profit_a + profit_b + impact_value

    def sample(self, num_sample, withtime=False, seed=None):
        """Sample paths with price impact in the SDE."""
        if seed:
            np.random.seed(seed)
        dw_sample = (
            np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
            * self.sqrt_delta_t
        )
        if seed:
            np.random.seed(seed=int(__import__("time").time()))

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

            # Net order flow and price impact
            net_flow = f_b - f_a
            impact = self._impact_fn_np(net_flow)

            # Price: dS = sigma_s * dW^S + impact * dt
            x_sample[:, i + 1, 0] = s + self.sigma_s * dw_sample[:, 0, i] + impact * dt

            # Inventory: same diffusion approximation
            inv_drift = (f_b - f_a) * dt
            inv_diff = np.sqrt(np.maximum(f_b + f_a, 1e-8)) * dw_sample[:, 1, i]
            x_sample[:, i + 1, 1] = q + inv_drift + inv_diff
            x_sample[:, i + 1, 1] = np.clip(x_sample[:, i + 1, 1], -self.q_max, self.q_max)

        x_sample = x_sample.transpose((0, 2, 1))
        return dw_sample, x_sample

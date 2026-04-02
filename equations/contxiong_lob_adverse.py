# equations/contxiong_lob_adverse.py
#
# Cont-Xiong LOB model with adverse selection.
# Execution probability depends on recent price moves, breaking
# the V = x + qs + phi(t,q) reduction and making the problem
# genuinely 2-dimensional.
#
# Adverse selection model:
#   f_a(delta, dS) = exp(-alpha * delta) * (1 + eta * dS / sigma)
#   f_b(delta, dS) = exp(-alpha * delta) * (1 - eta * dS / sigma)
#
# When price rises (dS > 0):
#   - Ask fills more likely (informed buyers hit you)
#   - Bid fills less likely
# This creates inventory risk correlated with price, breaking the
# independence that allowed the 1D reduction.
#
# The state is now genuinely (S, q) or (dS_recent, q) with the
# value function depending on both dimensions.

import time
import numpy as np
import torch

from registry import register_equation
from .contxiong_lob import ContXiongLOB


@register_equation("contxiong_lob_adverse")
class ContXiongLOBAdverse(ContXiongLOB):
    """Cont-Xiong LOB with adverse selection (price-dependent execution).

    The adverse selection parameter eta controls how much recent price
    moves affect fill probabilities. When eta=0, reduces to the base model.
    When eta>0, the 1D reduction breaks and V genuinely depends on price state.

    We track price momentum via an exponentially-weighted moving average
    of recent price increments, creating a continuous "signal" state.
    """

    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.eta = getattr(eqn_config, "eta", 0.5)  # adverse selection strength
        self.signal_decay = getattr(eqn_config, "signal_decay", 0.9)  # EMA decay for price signal
        # State is now 3D: (S, q, signal) where signal = EMA of dS/sigma
        self.dim = 3

    def _adverse_factor_np(self, signal, side="ask"):
        """Adverse selection factor: how price signal affects fill rate.

        For ask (selling): factor > 1 when signal > 0 (price rising, adverse)
        For bid (buying):  factor > 1 when signal < 0 (price falling, adverse)

        Clamped to (0.1, 3.0) for numerical stability.
        """
        if side == "ask":
            factor = 1.0 + self.eta * signal
        else:
            factor = 1.0 - self.eta * signal
        return np.clip(factor, 0.1, 3.0)

    def _adverse_factor_tf(self, signal, side="ask"):
        """PyTorch version."""
        if side == "ask":
            factor = 1.0 + self.eta * signal
        else:
            factor = 1.0 - self.eta * signal
        return torch.clamp(factor, 0.1, 3.0)

    def sample(self, num_sample, withtime=False, seed=None):
        """Sample forward paths with price signal tracking.

        State: X_t = (S_t, q_t, signal_t) where signal is EMA of dS/sigma.
        Returns dw[batch, 3, T] and x[batch, 3, T+1].
        """
        if seed:
            np.random.seed(seed)

        dt = self.delta_t
        # 3 Brownian motions: price, inventory diffusion, (signal is deterministic given dS)
        # But signal is derived from price, so we only need 2 independent BMs
        dw_sample = (
            np.random.normal(size=[num_sample, 2, self.num_time_interval])
            * self.sqrt_delta_t
        )
        # Pad to dim=3 for the model (signal channel is zero noise)
        dw_padded = np.zeros([num_sample, 3, self.num_time_interval])
        dw_padded[:, :2, :] = dw_sample

        # State: [batch, T+1, 3] = (S, q, signal)
        x_sample = np.zeros([num_sample, self.num_time_interval + 1, 3])
        x_sample[:, 0, 0] = self.s_init  # price
        x_sample[:, 0, 1] = 0.0  # inventory
        x_sample[:, 0, 2] = 0.0  # signal (no momentum initially)

        for i in range(self.num_time_interval):
            s = x_sample[:, i, 0]
            q = x_sample[:, i, 1]
            signal = x_sample[:, i, 2]

            # Adverse selection factors
            adv_a = self._adverse_factor_np(signal, "ask")
            adv_b = self._adverse_factor_np(signal, "bid")

            # Proxy quotes (inventory penalty gradient)
            z_q_proxy = -2 * self.phi * q
            delta_a, delta_b = self._optimal_quotes_np(z_q_proxy)

            # Execution probabilities with adverse selection
            f_a = np.exp(-self.alpha * np.clip(delta_a, 0.01, 10.0)) * adv_a * self.lambda_a
            f_b = np.exp(-self.alpha * np.clip(delta_b, 0.01, 10.0)) * adv_b * self.lambda_b

            # Price: dS = sigma * dW^S
            dS = self.sigma_s * dw_sample[:, 0, i]
            x_sample[:, i + 1, 0] = s + dS

            # Inventory: diffusion approximation
            inv_drift = (f_b - f_a) * dt
            inv_diff = np.sqrt(np.maximum(f_b + f_a, 1e-8)) * dw_sample[:, 1, i]
            x_sample[:, i + 1, 1] = np.clip(q + inv_drift + inv_diff, -self.q_max, self.q_max)

            # Signal: EMA of normalised price increments
            x_sample[:, i + 1, 2] = self.signal_decay * signal + (1 - self.signal_decay) * dS / (self.sigma_s * self.sqrt_delta_t)

        if seed:
            np.random.seed(seed=int(time.time()))

        x_sample = x_sample.transpose((0, 2, 1))  # [batch, 3, T+1]
        return dw_padded, x_sample

    def f_tf(self, t, x, y, z):
        """Generator with adverse selection.

        x: [batch, 3] = (S, q, signal)
        z: [batch, 3] = (Z^S, Z^q, Z^signal)
        """
        q = x[:, 1:2]
        signal = x[:, 2:3]
        z_q = z[:, 1:2]

        # Optimal quotes from Z^q (same as base, via inventory gradient)
        delta_a, delta_b = self._optimal_quotes_tf(z_q)

        # Adverse selection factors
        adv_a = self._adverse_factor_tf(signal, "ask")
        adv_b = self._adverse_factor_tf(signal, "bid")

        # Execution probabilities with adverse selection
        f_a = self._exec_prob_tf(delta_a) * adv_a * self.lambda_a
        f_b = self._exec_prob_tf(delta_b) * adv_b * self.lambda_b

        # Penalty
        psi = self._penalty_tf(q)

        # Profits
        profit_a = f_a * delta_a
        profit_b = f_b * delta_b

        return -self.discount_rate * y - psi + profit_a + profit_b

    def g_tf(self, t, x):
        """Terminal condition: -psi(q_T)."""
        q = x[:, 1:2]
        return -self._penalty_tf(q)

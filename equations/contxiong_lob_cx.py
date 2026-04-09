"""
Cont-Xiong LOB with PROPER mean-field execution probabilities.

Uses equation (6) from Cont & Xiong (2024) in the mean-field limit:

In the N-player game, execution probability for dealer i on the ask side:
  f_a^i(delta_i, delta^{-i}) = (1/N) * Lambda(delta_i) * softmax_term

In the mean-field limit (N -> inf, identical agents):
  f_a(delta, mu) = Lambda(delta) * C(delta, mu)

where:
  Lambda(delta) = 1 / (1 + exp(a*delta + b))    [base execution prob]
  C(delta, mu) = competition factor from population

The competition factor captures: if your quote is tighter than the
population average, you get more executions; if wider, you get fewer.

Parameters a, b in Lambda correspond to price sensitivity of order flow.
The mean-field coupling enters through the population average quotes,
NOT through a learned scalar h. This is the correct Cont-Xiong structure.

Key difference from our previous implementation:
- OLD: f_a = exp(-alpha*delta) * h(Phi(mu))  [h is learned, arbitrary]
- NEW: f_a = Lambda(delta) * C(delta, mu)    [C is computed from mu]

The competitive factor C depends on the population's average quotes,
which in turn depend on the population's inventory distribution. So the
chain is: mu(q) -> average quotes -> competitive factor -> execution rate.
This is the proper mean-field mechanism, not a learned proxy.
"""

import numpy as np
import torch
import time

from registry import register_equation
from equations.contxiong_lob_mv import ContXiongLOBMV
from equations.law_encoders import create_law_encoder


@register_equation("contxiong_lob_cx")
class ContXiongLOBCX(ContXiongLOBMV):
    """Cont-Xiong model with proper mean-field execution probabilities.

    Instead of a learned h(Phi(mu)), execution probabilities depend on
    how the agent's quote compares to the population average quote.
    """

    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        # Cont-Xiong execution probability parameters
        # Lambda(delta) = 1 / (1 + exp(a*delta + b))
        self.cx_a = getattr(eqn_config, "cx_a", 1.5)  # price sensitivity (= alpha in our notation)
        self.cx_b = getattr(eqn_config, "cx_b", 0.0)   # baseline offset

        # Store current population average quotes for use in f_tf
        self._pop_avg_delta_a = None
        self._pop_avg_delta_b = None

    def set_population_quotes(self, avg_delta_a, avg_delta_b):
        """Store population average quotes for use by f_tf.
        Called by the solver's forward pass at each timestep."""
        self._pop_avg_delta_a = avg_delta_a
        self._pop_avg_delta_b = avg_delta_b

    def _cx_exec_prob_tf(self, delta, pop_avg_delta, side="ask"):
        """Cont-Xiong execution probability (mean-field limit).

        From eq (6), in the mean-field limit:
          f(delta, mu) = Lambda(delta) * competition(delta, avg_delta_mu)

        Competition factor: if your quote (delta) is tighter than the
        population average (pop_avg_delta), you execute more often.

        We use a softmax-inspired form:
          competition = exp(-cx_a * delta) / (exp(-cx_a * delta) + exp(-cx_a * pop_avg))
                      = sigmoid(cx_a * (pop_avg - delta))

        When delta < pop_avg (tighter): competition > 0.5 (more executions)
        When delta > pop_avg (wider): competition < 0.5 (fewer executions)
        When delta = pop_avg: competition = 0.5 (fair share)
        """
        # Base execution probability
        Lambda = torch.sigmoid(-(self.cx_a * delta + self.cx_b))

        # Competition factor: your share of executions given population
        if pop_avg_delta is not None:
            competition = torch.sigmoid(self.cx_a * (pop_avg_delta - delta))
        else:
            competition = 0.5  # no competition info -> fair share

        return Lambda * competition * 2.0  # *2 so that at fair share, rate = Lambda

    def _cx_exec_prob_np(self, delta, pop_avg_delta):
        """Numpy version for sampling."""
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

        Lambda = sigmoid(-(self.cx_a * delta + self.cx_b))
        if pop_avg_delta is not None:
            competition = sigmoid(self.cx_a * (pop_avg_delta - delta))
        else:
            competition = 0.5
        return Lambda * competition * 2.0

    def f_tf(self, t, x, y, z):
        """Generator with Cont-Xiong mean-field execution probabilities.

        The execution rate depends on:
        1. The agent's own quote (delta_a, delta_b)
        2. The population's average quotes (from stored population stats)

        This replaces the learned h(Phi(mu)) with the proper CX mechanism.
        """
        q = x[:, 1:2]
        z_q = z[:, 1:2]

        delta_a, delta_b = self._optimal_quotes_tf(z_q)

        # Cont-Xiong execution probabilities (population-dependent)
        f_a = self._cx_exec_prob_tf(delta_a, self._pop_avg_delta_a, "ask") * self.lambda_a
        f_b = self._cx_exec_prob_tf(delta_b, self._pop_avg_delta_b, "bid") * self.lambda_b

        psi = self._penalty_tf(q)
        profit_a = f_a * delta_a
        profit_b = f_b * delta_b

        return -self.discount_rate * y - psi + profit_a + profit_b

    def sample(self, num_sample, withtime=False, seed=None):
        """Sample paths with CX execution probabilities."""
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

            # Proxy quotes for sampling
            z_q_proxy = -2 * self.phi * q
            sigma_q = self._sigma_q_equilibrium()
            p = z_q_proxy / sigma_q
            delta_a = 1.0 / self.alpha + p
            delta_b = 1.0 / self.alpha - p

            # Population average quotes (from the batch itself)
            avg_da = np.mean(delta_a)
            avg_db = np.mean(delta_b)

            # CX execution probabilities
            f_a = self._cx_exec_prob_np(delta_a, avg_da) * self.lambda_a
            f_b = self._cx_exec_prob_np(delta_b, avg_db) * self.lambda_b

            # Price
            x_sample[:, i + 1, 0] = s + self.sigma_s * dw_sample[:, 0, i]

            # Inventory (diffusion approx)
            inv_drift = (f_b - f_a) * dt
            inv_diff = np.sqrt(np.maximum(f_b + f_a, 1e-8)) * dw_sample[:, 1, i]
            x_sample[:, i + 1, 1] = np.clip(
                q + inv_drift + inv_diff, -self.q_max, self.q_max
            )

        x_sample = x_sample.transpose((0, 2, 1))
        return dw_sample, x_sample

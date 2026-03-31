# equations/contxiong_lob.py
#
# Cont-Xiong (2024) limit order book model with continuous relaxation.
# Market makers quote bid/ask around mid-price; inventory evolves as
# diffusion approximation of Poisson order arrivals.
#
# State: X_t = (S_t, q_t)  where S = mid-price, q = inventory
# Forward SDE:
#   dS_t = sigma_s * dW^S_t
#   dq_t = (lambda_b * f_b - lambda_a * f_a) dt + sqrt(lambda_b*f_b + lambda_a*f_a) dW^q_t
#
# Optimal quotes (Avellaneda-Stoikov / HJB first-order condition):
#   delta_a = 1/alpha + z_q       (ask half-spread + inventory gradient)
#   delta_b = 1/alpha - z_q       (bid half-spread - inventory gradient)
#
# Execution probability: f(delta) = exp(-alpha * delta) * competitive_factor(mu_t)
#
# Reference: Cont & Xiong (2024), "Dynamics of market making algorithms
# in dealer markets: Learning and tacit collusion", Mathematical Finance.

import logging
import time
import numpy as np
import torch
import torch.nn as nn

from registry import register_equation
from .base import Equation


class _MeanFieldDriftNet(nn.Module):
    """Small NN to approximate the mean-field competitive factor h(mu_t).

    Takes (t, mean_q, mean_spread) as input, outputs a scalar competitive
    adjustment factor applied to execution probabilities.
    """

    def __init__(self, num_hiddens, dtype=torch.float64):
        super().__init__()
        layers = []
        in_dim = 3  # (t, mean_inventory, mean_spread)
        for h in num_hiddens:
            layers.extend([nn.Linear(in_dim, h, dtype=dtype), nn.Softplus()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1, dtype=dtype))
        layers.append(nn.Sigmoid())  # output in (0, 1) — competitive dampening
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@register_equation("contxiong_lob")
class ContXiongLOB(Equation):
    """Cont-Xiong LOB model with continuous inventory relaxation.

    The mean-field coupling enters through the execution probabilities:
    each market maker's fill rate depends on the population distribution
    of quotes mu_t. In the simplest form, we track the mean spread as
    a sufficient statistic of mu_t.
    """

    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.dim = 2  # (S, q)

        # Market parameters
        self.sigma_s = getattr(eqn_config, "sigma_s", 0.3)
        self.lambda_a = getattr(eqn_config, "lambda_a", 1.0)
        self.lambda_b = getattr(eqn_config, "lambda_b", 1.0)
        self.alpha = getattr(eqn_config, "alpha", 1.5)
        self.phi = getattr(eqn_config, "phi", 0.01)
        self.discount_rate = getattr(eqn_config, "discount_rate", 0.1)
        self.q_max = getattr(eqn_config, "q_max", 10.0)
        self.penalty_type = getattr(eqn_config, "penalty_type", "quadratic")
        self.gamma = getattr(eqn_config, "gamma", 1.0)  # for exponential penalty

        # Initial state: price at 100, inventory at 0
        self.s_init = getattr(eqn_config, "s_init", 100.0)
        self.q_init = 0.0

        # Mean-field state: track mean spread over population at each time step
        # mean_spread_estimate[i] = E[delta_a + delta_b] at time t_i
        self.mean_spread_estimate = np.ones(self.num_time_interval + 1) * (2.0 / self.alpha)
        self.mean_q_estimate = np.zeros(self.num_time_interval + 1)

        # Drift NN for mean-field competitive factor
        drift_hiddens = getattr(eqn_config, "num_hiddens", None) or [24, 24]
        if eqn_config.drift_approx == "nn":
            self.drift_model = _MeanFieldDriftNet(drift_hiddens)
            self.drift_model.eval()
            self.drift_predict = self._drift_predict_nn
            self.update_drift = self._update_drift_nn
        else:
            # MC fallback: no drift NN, just use stored statistics directly
            self.drift_predict = self._drift_predict_mc
            self.update_drift = self._update_drift_mc

        self.y_init = None

    # ------------------------------------------------------------------
    # Execution probabilities and optimal quotes
    # ------------------------------------------------------------------

    def _exec_prob_np(self, delta, competitive_factor=1.0):
        """Execution probability: Lambda(delta) = exp(-alpha * delta) * h(mu).
        Clamp delta to avoid numerical issues."""
        delta_clamped = np.clip(delta, -5.0 / self.alpha, 10.0 / self.alpha)
        return np.exp(-self.alpha * delta_clamped) * competitive_factor

    def _exec_prob_tf(self, delta, competitive_factor=1.0):
        """PyTorch version of execution probability."""
        delta_clamped = torch.clamp(delta, -5.0 / self.alpha, 10.0 / self.alpha)
        return torch.exp(-self.alpha * delta_clamped) * competitive_factor

    def _optimal_quotes_np(self, z_q):
        """Avellaneda-Stoikov optimal quotes from Z^q (inventory gradient).
        delta_a = 1/alpha + z_q,  delta_b = 1/alpha - z_q
        """
        base_spread = 1.0 / self.alpha
        delta_a = base_spread + z_q
        delta_b = base_spread - z_q
        return delta_a, delta_b

    def _optimal_quotes_tf(self, z_q):
        """PyTorch version."""
        base_spread = 1.0 / self.alpha
        delta_a = base_spread + z_q
        delta_b = base_spread - z_q
        return delta_a, delta_b

    # ------------------------------------------------------------------
    # Drift NN for mean-field competitive factor
    # ------------------------------------------------------------------

    def _drift_predict_nn(self, t_idx, verbose=0):
        """Predict competitive factor h(mu_t) at time index t_idx."""
        t = self.t_grid[t_idx] if t_idx < len(self.t_grid) else self.total_time
        inp = np.array([[t, self.mean_q_estimate[t_idx], self.mean_spread_estimate[t_idx]]])
        with torch.no_grad():
            x_t = torch.tensor(inp, dtype=torch.float64)
            return self.drift_model(x_t).item()

    def _drift_predict_mc(self, t_idx, verbose=0):
        """MC fallback: competitive factor from stored statistics.
        Simple model: more competition (tighter spreads) reduces fill probability.
        h(mu) = exp(-beta * mean_spread)  where beta is a sensitivity param.
        """
        mean_spread = self.mean_spread_estimate[t_idx]
        # When mean spread is tight, competition is fierce, harder to get filled
        # When mean spread is wide, easy to get filled
        # Normalise: at the equilibrium spread 2/alpha, h = 1
        equilibrium_spread = 2.0 / self.alpha
        return np.exp(-0.5 * (mean_spread - equilibrium_spread))

    def _update_drift_nn(self):
        """Retrain drift NN on Monte Carlo estimates of competitive factor."""
        N_simu = getattr(self.eqn_config, "N_simu", 500)

        # Generate training data: simulate paths with current mean-field estimate
        # and compute what the competitive factor should be
        t_data = []
        h_data = []
        for t_idx in range(self.num_time_interval + 1):
            t = self.t_grid[t_idx] if t_idx < len(self.t_grid) else self.total_time
            h_mc = self._drift_predict_mc(t_idx)
            for _ in range(10):  # augment with noise
                t_data.append([t, self.mean_q_estimate[t_idx] + np.random.normal(0, 0.1),
                               self.mean_spread_estimate[t_idx] + np.random.normal(0, 0.05)])
                h_data.append([h_mc])

        X_t = torch.tensor(np.array(t_data), dtype=torch.float64)
        Y_t = torch.tensor(np.array(h_data), dtype=torch.float64)
        dataset = torch.utils.data.TensorDataset(X_t, Y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.Adam(self.drift_model.parameters(), lr=1e-3)
        self.drift_model.train()
        for epoch in range(50):
            for xb, yb in loader:
                pred = self.drift_model(xb)
                loss = nn.functional.mse_loss(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.drift_model.eval()

    def _update_drift_mc(self):
        """MC fallback: nothing to retrain, statistics are used directly."""
        pass

    # ------------------------------------------------------------------
    # Mean-field coupling
    # ------------------------------------------------------------------

    def update_mean_y_estimate(self, mean_y_estimate):
        """Called by the solver to pass back mean Y (value function) across batch.
        We use this to update our estimate of the mean spread and inventory."""
        self.mean_y_estimate = mean_y_estimate.copy()

    def update_mean_field(self, mean_spreads, mean_inventories):
        """Update stored mean-field statistics from solver's forward pass."""
        n = min(len(mean_spreads), len(self.mean_spread_estimate))
        self.mean_spread_estimate[:n] = mean_spreads[:n]
        self.mean_q_estimate[:n] = mean_inventories[:n]

    # ------------------------------------------------------------------
    # Forward SDE sampling
    # ------------------------------------------------------------------

    def sample(self, num_sample, withtime=False, seed=None):
        if seed:
            np.random.seed(seed)
            dw_sample = (
                np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
                * self.sqrt_delta_t
            )
            np.random.seed(seed=int(time.time()))
        else:
            dw_sample = (
                np.random.normal(size=[num_sample, self.dim, self.num_time_interval])
                * self.sqrt_delta_t
            )

        dt = self.delta_t
        # x_sample: [num_sample, num_time_interval+1, dim]
        # x[:, :, 0] = S (price), x[:, :, 1] = q (inventory)
        x_sample = np.zeros([num_sample, self.num_time_interval + 1, self.dim])
        x_sample[:, 0, 0] = self.s_init  # initial price
        x_sample[:, 0, 1] = self.q_init  # initial inventory

        for i in range(self.num_time_interval):
            s = x_sample[:, i, 0]  # [num_sample]
            q = x_sample[:, i, 1]  # [num_sample]

            # Get competitive factor from mean-field
            h_factor = self._drift_predict_mc(i)

            # Use a proxy for Z^q: the inventory penalty gradient = -2*phi*q
            # This is a rough estimate for path simulation only; the actual Z
            # is learned by the neural network during training
            z_q_proxy = -2 * self.phi * q
            delta_a, delta_b = self._optimal_quotes_np(z_q_proxy)

            # Execution probabilities
            f_a = self._exec_prob_np(delta_a, h_factor) * self.lambda_a
            f_b = self._exec_prob_np(delta_b, h_factor) * self.lambda_b

            # Price: dS = sigma_s * dW^S
            x_sample[:, i + 1, 0] = s + self.sigma_s * dw_sample[:, 0, i]

            # Inventory: diffusion approximation
            # drift = lambda_b * f_b - lambda_a * f_a
            # diffusion = sqrt(lambda_b * f_b + lambda_a * f_a)
            inv_drift = (f_b - f_a) * dt
            inv_diff = np.sqrt(np.maximum(f_b + f_a, 1e-8)) * dw_sample[:, 1, i]
            x_sample[:, i + 1, 1] = q + inv_drift + inv_diff

            # Soft clamp inventory to prevent explosion
            x_sample[:, i + 1, 1] = np.clip(
                x_sample[:, i + 1, 1], -self.q_max, self.q_max
            )

        # Transpose to [batch, features, time] for consistency with dw
        x_sample = x_sample.transpose((0, 2, 1))
        return dw_sample, x_sample

    # ------------------------------------------------------------------
    # PDE coefficients (PyTorch, for use in the solver's forward pass)
    # ------------------------------------------------------------------

    def f_tf(self, t, x, y, z):
        """Generator: f(t, x, y, z) for the BSDE.

        The value function satisfies (in continuous form):
        rV + psi(q) - lambda_a * f_a(delta_a*) * delta_a*
                     - lambda_b * f_b(delta_b*) * delta_b* = 0

        In BSDE form with Y = V, Z = grad V:
        f(t,x,y,z) = -r*y + psi(q) - execution_profits

        Args:
            t: scalar time
            x: [batch, 2]  (S, q)
            y: [batch, 1]  value function
            z: [batch, 2]  gradient (Z^S, Z^q)
        """
        q = x[:, 1:2]       # [batch, 1]
        z_q = z[:, 1:2]     # [batch, 1] — inventory gradient

        # Optimal quotes from Z^q
        delta_a, delta_b = self._optimal_quotes_tf(z_q)

        # Execution probabilities (no competitive factor in generator — it's in the sample)
        f_a = self._exec_prob_tf(delta_a) * self.lambda_a
        f_b = self._exec_prob_tf(delta_b) * self.lambda_b

        # Running cost: inventory penalty (configurable type)
        psi = self._penalty_tf(q)

        # Execution profits: f_a * delta_a + f_b * delta_b
        profit_a = f_a * delta_a
        profit_b = f_b * delta_b

        # Generator: f = -r*y + psi - profits
        # Note: in BSDE, dY = -f dt + Z dW, so positive f means Y decreases
        return -self.discount_rate * y + psi - profit_a - profit_b

    def _penalty_tf(self, q):
        """Inventory penalty psi(q). Configurable via penalty_type.

        Types:
            quadratic: phi * q^2  (standard Avellaneda-Stoikov)
            cubic:     phi * q^2 + phi * |q|^3 / 3  (non-linear, breaks monotonicity)
            exponential: phi * (exp(gamma * |q|) - 1)  (severe, blows up at large |q|)
        """
        if self.penalty_type == "quadratic":
            return self.phi * q ** 2
        elif self.penalty_type == "cubic":
            return self.phi * q ** 2 + self.phi * torch.abs(q) ** 3 / 3.0
        elif self.penalty_type == "exponential":
            return self.phi * (torch.exp(self.gamma * torch.abs(q)) - 1.0)
        else:
            return self.phi * q ** 2

    def g_tf(self, t, x):
        """Terminal condition: g(T, x) = -psi(q_T) (liquidation penalty)."""
        q = x[:, 1:2]  # [batch, 1]
        return -self._penalty_tf(q)

"""
Exact Cont-Xiong dealer market model for deep BSDE solver.

Uses the ACTUAL execution probabilities from eq (58) of Cont & Xiong (2024),
not our previous approximation. Parameters from their Table 1.

State: q_i in {-Q, ..., Q} (discrete inventory, 2Q/Delta + 1 levels)
Control: (delta_a, delta_b) -- continuous centered ask/bid quotes
Dynamics: dq = Delta * (dN^b - dN^a) where N^a, N^b are Poisson with
  intensity nu_a = lambda_a * f_a(delta_a, competitors)
  intensity nu_b = lambda_b * f_b(delta_b, competitors)

Execution probability (eq 58):
  f_a(delta, comp) = [1/(1+exp(delta))] * exp(S/K) / (1 + exp(delta + S/K))
  where S = sum of all competitors' ask quotes, K = number of competitor levels

For the mean-field limit (N -> inf, identical agents):
  S/K -> mean quote across population = E_mu[delta_a(q)]
  So f_a(delta, mu) depends on the population's average ask quote.

The BSDE formulation: since inventory is discrete, we use the Bellman equation
(their eq 27) as the loss function. The NN learns V(q) for each q, and quotes
are computed from the FOC of the Bellman equation.

Parameters (Table 1):
  lambda_a = lambda_b = 2
  r = 0.01
  Delta = 1
  psi(q) = 0.005 * q^2
  Q = 5 (risk limit)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar

from registry import register_equation


def cx_exec_prob_np(delta, avg_competitor_quote, K, N=2):
    """CX execution probability (numpy). Eq 58 in mean-field limit.

    delta: own quote
    avg_competitor_quote: mean quote across population
    K: number of competitor inventory levels
    N: total number of dealers (for 1/N market share factor)
    """
    base = 1.0 / (1.0 + np.exp(np.clip(delta, -20, 20)))
    if K > 0:
        S_over_K = avg_competitor_quote
        comp = np.exp(np.clip(S_over_K, -20, 20)) / (
            1.0 + np.exp(np.clip(delta + S_over_K, -20, 20)))
        return (1.0 / N) * base * comp
    else:
        return base * base  # monopolist


def cx_exec_prob_torch(delta, avg_competitor_quote, K, N=2):
    """CX execution probability (torch). Eq 58 in mean-field limit."""
    base = torch.sigmoid(-delta)
    if K > 0 and avg_competitor_quote is not None:
        S_over_K = avg_competitor_quote
        comp = torch.exp(torch.clamp(S_over_K, -20, 20)) / (
            1.0 + torch.exp(torch.clamp(delta + S_over_K, -20, 20)))
        return (1.0 / N) * base * comp
    else:
        return base * base


def optimal_quote_foc(p, avg_competitor_quote, K, N=2):
    """Solve FOC for optimal quote given value jump p.

    delta* = argmax f(delta, comp) * (delta - p)
    where p = [V(q) - V(q-Delta)] / Delta for ask side.

    Returns optimal delta.
    """
    def neg_profit(delta):
        f = cx_exec_prob_np(delta, avg_competitor_quote, K, N)
        return -(delta - p) * f

    result = minimize_scalar(neg_profit, bounds=(-3, 10), method='bounded')
    return result.x


@register_equation("contxiong_exact")
class ContXiongExact:
    """Exact Cont-Xiong model with proper execution probabilities.

    This is NOT derived from our previous equation classes. It implements
    the model directly from the paper.
    """

    def __init__(self, eqn_config):
        # CX parameters (Table 1)
        self.lambda_a = getattr(eqn_config, 'lambda_a', 2.0)
        self.lambda_b = getattr(eqn_config, 'lambda_b', 2.0)
        self.r = getattr(eqn_config, 'discount_rate', 0.01)
        self.Delta = getattr(eqn_config, 'Delta_q', 1.0)  # order size
        self.Q = getattr(eqn_config, 'q_max', 5.0)  # risk limit
        self.phi = getattr(eqn_config, 'phi', 0.005)  # psi(q) = phi * q^2
        self.N_agents = getattr(eqn_config, 'N_agents', 2)  # number of dealers

        # Inventory grid
        self.q_grid = np.arange(-self.Q, self.Q + self.Delta, self.Delta)
        self.nq = len(self.q_grid)
        self.mid = self.nq // 2  # index of q=0

        # Mean-field: K = (N-1) * nq competitor inventory levels
        self.K = (self.N_agents - 1) * self.nq if self.N_agents > 1 else 0

        # Discount factor for discrete-time RL formulation (their eq 72)
        self.gamma = (self.lambda_a + self.lambda_b) / (
            self.r + self.lambda_a + self.lambda_b
        )

        # For compatibility with solver
        self.dim = self.nq  # "dimension" = number of inventory levels
        self.total_time = 1.0  # not really used (infinite horizon)
        self.num_time_interval = 1  # single-step Bellman

    def psi(self, q):
        """Running cost (inventory penalty)."""
        return self.phi * q ** 2

    def bellman_residual(self, V, delta_a, delta_b, avg_da, avg_db):
        """Compute Bellman equation residual for all inventory levels.

        This is the loss function for the neural network.
        V: value function at each inventory level [nq]
        delta_a, delta_b: quotes at each inventory level [nq]
        avg_da, avg_db: population average quotes (scalars)

        Returns residual at each q (should be zero at equilibrium).
        """
        residuals = torch.zeros(self.nq)

        for j in range(self.nq):
            q = self.q_grid[j]

            # Execution probabilities
            if q > -self.Q:
                fa = cx_exec_prob_torch(delta_a[j], avg_da, self.K, self.N_agents)
            else:
                fa = torch.tensor(0.0)
            if q < self.Q:
                fb = cx_exec_prob_torch(delta_b[j], avg_db, self.K, self.N_agents)
            else:
                fb = torch.tensor(0.0)

            # Value jumps
            V_down = V[j - 1] if j > 0 else torch.tensor(-self.psi(q - self.Delta))
            V_up = V[j + 1] if j < self.nq - 1 else torch.tensor(-self.psi(q + self.Delta))

            # Bellman equation (their eq 27)
            lhs = self.r * V[j] + self.psi(q)

            profit_a = self.lambda_a * self.Delta * fa * (
                delta_a[j] - (V[j] - V_down) / self.Delta
            ) if q > -self.Q else 0.0

            profit_b = self.lambda_b * self.Delta * fb * (
                delta_b[j] - (V[j] - V_up) / self.Delta
            ) if q < self.Q else 0.0

            residuals[j] = lhs - profit_a - profit_b

        return residuals

    def compute_optimal_quotes(self, V_np, avg_da, avg_db):
        """Given value function V, compute optimal quotes at each q.

        Uses the FOC from eq (29): delta* = argmax f(delta, comp) * (delta - p)
        """
        delta_a = np.zeros(self.nq)
        delta_b = np.zeros(self.nq)

        for j in range(self.nq):
            q = self.q_grid[j]

            # Value jumps
            if j > 0:
                p_a = (V_np[j] - V_np[j - 1]) / self.Delta
            else:
                p_a = (V_np[j] - (-self.psi(q - self.Delta))) / self.Delta

            if j < self.nq - 1:
                p_b = (V_np[j] - V_np[j + 1]) / self.Delta
            else:
                p_b = (V_np[j] - (-self.psi(q + self.Delta))) / self.Delta

            # Optimal ask quote
            if q > -self.Q:
                delta_a[j] = optimal_quote_foc(p_a, avg_da, self.K, self.N_agents)
            # Optimal bid quote
            if q < self.Q:
                delta_b[j] = optimal_quote_foc(p_b, avg_db, self.K, self.N_agents)

        return delta_a, delta_b

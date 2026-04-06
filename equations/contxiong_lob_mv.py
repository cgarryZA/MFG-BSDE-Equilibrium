# equations/contxiong_lob_mv.py
#
# McKean-Vlasov extension of the Cont-Xiong LOB model.
# Distribution-dependent mean-field coupling via law encoders.
#
# CRITICAL: The law embedding Phi(mu_t) must enter BOTH:
# 1. The subnet inputs (so Z depends on population)
# 2. The generator f_tf (so execution probabilities depend on population)
#
# Without (2), the BSDE dynamics don't depend on Phi and the network
# learns to ignore it. This was a bug in the initial implementation.

import time
import numpy as np
import torch
import torch.nn as nn

from registry import register_equation
from .contxiong_lob import ContXiongLOB
from .law_encoders import create_law_encoder


class CompetitiveFactorNet(nn.Module):
    """Learned competitive factor h(Phi(mu_t)) -> scalar in (0, 1].

    Maps the law embedding to a scalar that modulates execution
    probabilities. This is the key coupling mechanism: when the
    population quotes tightly, each agent's fill rate drops.
    """

    def __init__(self, embed_dim, hidden_dim=16, dtype=torch.float64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, dtype=dtype),
            nn.Sigmoid(),  # output in (0, 1)
        )

    def forward(self, law_embed):
        """law_embed: [embed_dim] -> scalar h in (0.01, 1]."""
        # Small epsilon to avoid exact zero; no artificial floor
        return 0.01 + 0.99 * self.net(law_embed)


@register_equation("contxiong_lob_mv")
class ContXiongLOBMV(ContXiongLOB):
    """McKean-Vlasov Cont-Xiong LOB with distribution-dependent coupling.

    The law embedding enters BOTH the subnet inputs AND the generator:
    - Subnets: (S_i, q_i, Phi(mu_t)) -> (Z^S, Z^q)
    - Generator: h(Phi(mu_t)) modulates execution probabilities

    This ensures the BSDE dynamics actually depend on the population
    distribution, not just the control prediction.
    """

    def __init__(self, eqn_config):
        super().__init__(eqn_config)

        encoder_type = getattr(eqn_config, "law_encoder_type", "moments")
        encoder_kwargs = {
            "state_dim": 2,  # (S, q)
            "embed_dim": getattr(eqn_config, "law_embed_dim", 16),
            "n_bins": getattr(eqn_config, "n_bins", 20),
            "q_max": self.q_max,
        }
        self.law_encoder = create_law_encoder(encoder_type, **encoder_kwargs)
        self.law_embed_dim = self.law_encoder.embed_dim

        # Learned competitive factor: Phi(mu_t) -> h in (0, 1]
        self.competitive_factor_net = CompetitiveFactorNet(
            self.law_embed_dim, dtype=torch.float64
        )

        # Store current law embedding for use in f_tf
        self._current_law_embed = None

        # Wasserstein tracking
        self._prev_particle_snapshot = None
        self._w2_history = []

    def compute_law_embedding(self, particles):
        """particles: [batch, state_dim] -> embedding: [embed_dim]."""
        return self.law_encoder.encode(particles)

    def compute_competitive_factor(self, law_embed):
        """Compute h(Phi(mu_t)) from law embedding. Returns scalar tensor."""
        return self.competitive_factor_net(law_embed)

    def set_current_law_embed(self, law_embed):
        """Store the current law embedding for use by f_tf.
        Called by the solver's forward pass at each timestep."""
        self._current_law_embed = law_embed

    def f_tf(self, t, x, y, z):
        """Generator with distribution-dependent competitive factor.

        Uses h(Phi(mu_t)) from the stored law embedding, NOT the old
        scalar moment proxy. This is the critical fix that makes the
        BSDE dynamics actually depend on the population distribution.
        """
        q = x[:, 1:2]
        z_q = z[:, 1:2]

        delta_a, delta_b = self._optimal_quotes_tf(z_q)

        # USE THE LAW EMBEDDING for competitive factor
        if self._current_law_embed is not None:
            h_factor = self.compute_competitive_factor(self._current_law_embed)
        else:
            # Fallback to MC proxy if no embedding available (e.g. during sampling)
            t_val = t.item() if isinstance(t, torch.Tensor) else float(t)
            t_idx = min(int(round(t_val / self.delta_t)), self.num_time_interval)
            h_factor = self._drift_predict_mc(t_idx)

        f_a = self._exec_prob_tf(delta_a, h_factor) * self.lambda_a
        f_b = self._exec_prob_tf(delta_b, h_factor) * self.lambda_b

        psi = self._penalty_tf(q)
        profit_a = f_a * delta_a
        profit_b = f_b * delta_b

        return -self.discount_rate * y - psi + profit_a + profit_b

    def compute_w2_distance(self, particles_new):
        if self._prev_particle_snapshot is None:
            self._prev_particle_snapshot = particles_new.detach().cpu().numpy().copy()
            return 0.0
        q_new = np.sort(particles_new[:, 1].detach().cpu().numpy())
        q_old = np.sort(self._prev_particle_snapshot[:, 1])
        n = min(len(q_new), len(q_old))
        w2 = np.sqrt(np.mean((q_new[:n] - q_old[:n]) ** 2))
        self._prev_particle_snapshot = particles_new.detach().cpu().numpy().copy()
        return float(w2)

    def update_mean_field_mv(self, particles):
        particles_t = torch.tensor(particles, dtype=torch.float64)
        w2 = self.compute_w2_distance(particles_t)
        self._w2_history.append(w2)
        q = particles[:, 1]
        self.mean_q_estimate[:] = np.mean(q)
        self.mean_spread_estimate[:] = 2.0 / self.alpha

    def get_w2_history(self):
        return self._w2_history.copy()

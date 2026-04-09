# equations/law_encoders.py
#
# Law encoders for McKean-Vlasov distribution-dependent coupling.
# Each encoder maps a population of particle states to a fixed-size
# embedding that summarises the empirical distribution mu_t.
#
# All encoders share the interface:
#   encode(particles) -> embedding
# where particles: [batch, state_dim] and embedding: [embed_dim]
# (broadcast to all agents in the batch).

import torch
import torch.nn as nn
import numpy as np


class MomentEncoder(nn.Module):
    """Encodes mu_t via low-order moments of inventory q.

    For single-asset (state_dim=2): uses col 1 (q). Output dim: 6.
    For multi-asset (state_dim=2K): uses cols K..2K-1. Output dim: 6*K.
    """

    def __init__(self, state_dim=2, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        n_assets = max(state_dim // 2, 1)
        self.embed_dim = 6 * n_assets
        self.n_assets = n_assets

    def _moments_1d(self, q):
        """6 moments of a 1D inventory vector."""
        n = q.shape[0]
        mean_q = torch.mean(q)
        var_q = torch.var(q) if n > 1 else torch.tensor(0.0, dtype=q.dtype, device=q.device)
        std_q = torch.sqrt(var_q + 1e-8)
        skew_q = torch.mean(((q - mean_q) / (std_q + 1e-8)) ** 3) if n > 2 else torch.tensor(0.0, dtype=q.dtype, device=q.device)
        mean_abs_q = torch.mean(torch.abs(q))
        max_abs_q = torch.max(torch.abs(q))
        return torch.stack([mean_q, var_q, skew_q, mean_abs_q, max_abs_q, std_q])

    def encode(self, particles):
        """particles: [batch, state_dim]. Returns [embed_dim]."""
        K = self.n_assets
        feats = []
        for k in range(K):
            q_k = particles[:, K + k] if particles.shape[1] > 2 else particles[:, 1]
            feats.append(self._moments_1d(q_k))
        return torch.cat(feats)


class QuantileEncoder(nn.Module):
    """Encodes mu_t via mean + quantiles of inventory distribution.

    Features: [mean(q), q10, q25, q50, q75, q90]
    Output dim: 6
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.embed_dim = 6
        self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    def encode(self, particles):
        """particles: [batch, 2]. Returns [embed_dim]."""
        q = particles[:, 1]
        mean_q = torch.mean(q)
        quants = torch.quantile(q, torch.tensor(self.quantiles, dtype=q.dtype, device=q.device))
        return torch.cat([mean_q.unsqueeze(0), quants])


class HistogramEncoder(nn.Module):
    """Encodes mu_t via a soft histogram of inventory.

    Bins inventory into n_bins equally-spaced bins over [-q_max, q_max].
    Each bin count is normalised by batch size → density estimate.
    Output dim: n_bins
    """

    def __init__(self, n_bins=20, q_max=10.0, **kwargs):
        super().__init__()
        self.n_bins = n_bins
        self.q_max = q_max
        self.embed_dim = n_bins
        # Bin edges (not learnable)
        self.register_buffer(
            "bin_centres",
            torch.linspace(-q_max, q_max, n_bins)
        )
        self.bin_width = 2 * q_max / n_bins

    def encode(self, particles):
        """particles: [batch, 2]. Returns [n_bins]."""
        q = particles[:, 1]  # [batch]
        # Soft histogram: Gaussian kernel around each bin centre
        # [batch, n_bins]
        diffs = q.unsqueeze(1) - self.bin_centres.unsqueeze(0)
        weights = torch.exp(-0.5 * (diffs / (self.bin_width * 0.5)) ** 2)
        histogram = torch.mean(weights, dim=0)  # [n_bins]
        # Normalise to sum to 1
        histogram = histogram / (histogram.sum() + 1e-8)
        return histogram


class DeepSetsEncoder(nn.Module):
    """Encodes mu_t via a learned permutation-invariant embedding.

    Architecture (DeepSets):
        psi: per-particle encoder
            Linear(state_dim → hidden) → ReLU → Linear(hidden → hidden)
        mean pool over particles
        rho: summary decoder
            Linear(hidden → hidden) → ReLU → Linear(hidden → embed_dim)

    Output dim: embed_dim (configurable, default 16)
    """

    def __init__(self, state_dim=2, hidden_dim=32, embed_dim=16, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        dtype = torch.float64

        # Per-particle encoder psi
        self.psi = nn.Sequential(
            nn.Linear(state_dim, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
        )

        # Summary decoder rho
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim, dtype=dtype),
        )

    def encode(self, particles):
        """particles: [batch, state_dim]. Returns [embed_dim]."""
        # Per-particle features
        h = self.psi(particles)      # [batch, hidden_dim]
        # Mean pool (permutation invariant)
        pooled = torch.mean(h, dim=0)  # [hidden_dim]
        # Decode to embedding
        return self.rho(pooled)        # [embed_dim]


# Registry for easy construction from config
LAW_ENCODER_REGISTRY = {
    "moments": MomentEncoder,
    "quantiles": QuantileEncoder,
    "histogram": HistogramEncoder,
    "deepsets": DeepSetsEncoder,
}


def create_law_encoder(encoder_type, **kwargs):
    """Create a law encoder by name."""
    if encoder_type not in LAW_ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown law encoder: {encoder_type}. "
            f"Available: {list(LAW_ENCODER_REGISTRY.keys())}"
        )
    return LAW_ENCODER_REGISTRY[encoder_type](**kwargs)

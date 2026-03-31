# equations/base.py

import numpy as np
import torch


class Equation:
    """Base class for defining PDE related functions."""

    def __init__(self, config):
        self.eqn_config = config
        self.dim = config.dim
        self.total_time = config.total_time
        self.num_time_interval = config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.t_grid = np.linspace(0, self.total_time, self.num_time_interval + 1)
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError

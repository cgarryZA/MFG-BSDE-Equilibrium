# equations/sinebm.py
#
# Sine-of-Brownian-Motion McKean-Vlasov BSDE.
# Explicit solution: mean_y(t) = sin(t) * exp(-t/2)
# Reference: Han, Hu, Long (2022) — "Convergence of Deep Fictitious Play for
# Stochastic Differential Games"

import logging
import time
import numpy as np
import torch
import torch.nn as nn

from registry import register_equation
from .base import Equation


class _DriftNet(nn.Module):
    """Small NN to approximate the mean-field drift kernel."""

    def __init__(self, dim, num_hiddens, dtype=torch.float64):
        super().__init__()
        layers = []
        in_dim = dim + 1  # (t, x)
        for h in num_hiddens:
            layers.extend([nn.Linear(in_dim, h, dtype=dtype), nn.Softplus()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1, dtype=dtype))
        layers.append(nn.Softplus())  # output is positive (it's an expectation of exp)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@register_equation("sinebm")
class SineBM(Equation):
    """Sine of BM: the benchmark MV-BSDE with explicit solution."""

    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.mean_y = np.sin(self.t_grid) * np.exp(-self.t_grid / 2)
        self.mean_y_estimate = self.mean_y * 0

        if self.eqn_config.drift_approx == "nn":
            self.drift_model = None
            self.drift_predict = None
            self._create_model()
            self.update_drift = self._update_drift_nn
        elif self.eqn_config.drift_approx == "mc":
            self.saved_xs = np.zeros(
                shape=[self.eqn_config.N_simu, self.num_time_interval + 1, self.dim]
            )
            self.drift_predict = self._drift_predict_mc
            self.update_drift = self._update_drift_mc
        else:
            raise ValueError(f"Invalid drift_approx: {self.eqn_config.drift_approx}")

        if eqn_config.type != 3:
            self.update_drift()

    # ------------------------------------------------------------------
    # Drift NN creation and training
    # ------------------------------------------------------------------

    def _create_model(self):
        num_hiddens = self.eqn_config.num_hiddens
        self.drift_model = _DriftNet(self.dim, num_hiddens)
        self.drift_predict = self._drift_predict_nn

    def _drift_predict_nn(self, x, verbose=0):
        """x: [N, dim+1] where x[:, 0] = t, x[:, 1:] = spatial."""
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float64)
            return self.drift_model(x_t).numpy()

    def _update_drift_nn(self):
        N_simu = self.eqn_config.N_simu
        N_learn = self.eqn_config.N_learn
        N_iter = 3
        dim = self.dim
        Nt = self.num_time_interval
        dt = self.delta_t

        for _ in range(N_iter):
            x_path = np.zeros(shape=[N_simu, Nt + 1, dim])

            for i, t in enumerate(self.t_grid):
                drift_true = (
                    np.exp(-np.sum(x_path[:, i] ** 2, axis=-1, keepdims=True) / (dim + 2 * t))
                    * (dim / (dim + 2 * t)) ** (dim / 2)
                )
                t_tmp = np.zeros(shape=[N_simu, 1]) + t
                x_tmp = np.concatenate([t_tmp, x_path[:, i]], axis=-1)
                drift_nn = self.drift_predict(x_tmp, verbose=0)
                if i < Nt:
                    x_path[:, i + 1, :] = (
                        x_path[:, i, :]
                        + np.sin(drift_nn - drift_true) * dt
                        + np.random.normal(scale=np.sqrt(dt), size=(N_simu, dim))
                    )
                    if self.eqn_config.type == 3:
                        x_path[:, i + 1, :] += (
                            self.eqn_config.couple_coeff
                            * (self.mean_y_estimate[i] - self.mean_y[i])
                            * dt
                        )

            # Build training data: kernel approximation vs true drift
            term_approx = np.zeros(shape=[N_learn, self.t_grid.shape[0]])
            path_idx = np.random.choice(N_simu, N_learn, replace=False)
            for i, t in enumerate(self.t_grid):
                diff_x = x_path[path_idx, None, i, :] - x_path[None, path_idx, i, :]
                norm = np.sum(diff_x ** 2, axis=-1)
                term_approx[:, i] = np.average(np.exp(-norm / dim), axis=1)

            # Retrain drift model
            self._create_model()
            t_tmp = self.t_grid[None, :, None] + np.zeros(shape=[N_learn, Nt + 1, 1])
            X = np.concatenate([t_tmp, x_path[path_idx]], axis=-1)
            X = X.reshape([-1, dim + 1])
            Y = term_approx.reshape([-1, 1])

            X_t = torch.tensor(X, dtype=torch.float64)
            Y_t = torch.tensor(Y, dtype=torch.float64)
            dataset = torch.utils.data.TensorDataset(X_t, Y_t)
            loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

            optimizer = torch.optim.Adam(self.drift_model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

            self.drift_model.train()
            for epoch in range(80):
                for xb, yb in loader:
                    pred = self.drift_model(xb)
                    loss = nn.functional.mse_loss(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if (epoch + 1) % 10 == 0:
                    scheduler.step()
            self.drift_model.eval()

            # Log R^2
            with torch.no_grad():
                Y_pred = self.drift_model(X_t).numpy()
            # Compute true values for R^2
            term_true = np.zeros(shape=[N_learn, self.t_grid.shape[0]])
            for i, t in enumerate(self.t_grid):
                term_true[:, i] = (
                    np.exp(-np.sum(x_path[path_idx, i] ** 2, axis=-1) / (dim + 2 * t))
                    * (dim / (dim + 2 * t)) ** (dim / 2)
                )
            Y_true = term_true.reshape([-1, 1])
            r2 = np.sum((Y_pred - Y_true) ** 2) / np.sum((Y_true - np.mean(Y_true)) ** 2)
            logging.debug("Relative err for learning drift: {}".format(r2))

    # ------------------------------------------------------------------
    # MC drift
    # ------------------------------------------------------------------

    def _drift_predict_mc(self, x, verbose=0):
        t_idx = int(x[0, 0] / self.delta_t)
        diff_x = x[:, None, 1:] - self.saved_xs[None, :, t_idx, :]
        norm = np.sum(diff_x ** 2, axis=-1)
        drift_approx = np.mean(np.exp(-norm / self.dim), axis=1, keepdims=True)
        return drift_approx

    def _update_drift_mc(self):
        N_simu = self.eqn_config.N_simu
        N_iter = 3
        dim = self.dim
        Nt = self.num_time_interval
        dt = self.delta_t

        for _ in range(N_iter):
            x_path = np.zeros(self.saved_xs.shape)

            for i, t in enumerate(self.t_grid):
                drift_true = (
                    np.exp(-np.sum(x_path[:, i] ** 2, axis=-1, keepdims=True) / (dim + 2 * t))
                    * (dim / (dim + 2 * t)) ** (dim / 2)
                )
                t_tmp = np.zeros(shape=[N_simu, 1]) + t
                x_tmp = np.concatenate([t_tmp, x_path[:, i]], axis=-1)
                drift_mc = self.drift_predict(x_tmp, verbose=0)
                if i < Nt:
                    x_path[:, i + 1, :] = (
                        x_path[:, i, :]
                        + np.sin(drift_mc - drift_true) * dt
                        + np.random.normal(scale=np.sqrt(dt), size=(N_simu, dim))
                    )
                    if self.eqn_config.type == 3:
                        x_path[:, i + 1, :] += (
                            self.eqn_config.couple_coeff
                            * (self.mean_y_estimate[i] - self.mean_y[i])
                            * dt
                        )

            r2_res = 0
            r2_tot = 0
            for i, t in enumerate(self.t_grid):
                drift_true = (
                    np.exp(-np.sum(x_path[:, i] ** 2, axis=-1, keepdims=True) / (dim + 2 * t))
                    * (dim / (dim + 2 * t)) ** (dim / 2)
                )
                t_tmp = np.zeros(shape=[N_simu, 1]) + t
                x_tmp = np.concatenate([t_tmp, x_path[:, i]], axis=-1)
                drift_mc = self.drift_predict(x_tmp, verbose=0)
                r2_tot += np.sum(drift_true ** 2)
                r2_res += np.sum((drift_true - drift_mc) ** 2)

            r2 = r2_res / r2_tot
            logging.debug("Relative err for learning drift: {}".format(r2))
            self.saved_xs = x_path

    # ------------------------------------------------------------------
    # Mean-field coupling
    # ------------------------------------------------------------------

    def update_mean_y_estimate(self, mean_y_estimate):
        self.mean_y_estimate = mean_y_estimate.copy()

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

        dim = self.dim
        dt = self.delta_t
        x_sample = np.zeros([num_sample, self.num_time_interval + 1, self.dim])
        t_tmp = np.zeros(shape=[num_sample, 1])

        for i, t in enumerate(self.t_grid):
            drift_true = (
                np.exp(-np.sum(x_sample[:, i] ** 2, axis=-1, keepdims=True) / (dim + 2 * t))
                * (dim / (dim + 2 * t)) ** (dim / 2)
            )
            x_tmp = np.concatenate([t_tmp, x_sample[:, i]], axis=-1)
            drift_nn = self.drift_predict(x_tmp, verbose=0)
            t_tmp += self.delta_t
            if i < self.num_time_interval:
                x_sample[:, i + 1, :] = (
                    x_sample[:, i, :]
                    + np.sin(drift_nn - drift_true) * dt
                    + dw_sample[:, :, i]
                )
                if self.eqn_config.type == 3:
                    x_sample[:, i + 1, :] += (
                        self.eqn_config.couple_coeff
                        * (self.mean_y_estimate[i] - self.mean_y[i])
                        * dt
                    )

        if withtime:
            t_data = np.zeros([num_sample, self.num_time_interval + 1, 1])
            for i, t in enumerate(self.t_grid):
                t_data[:, i, :] = t
            x_sample = np.concatenate([t_data, x_sample], axis=-1)

        # Transpose to [batch, features, time] for consistency with dw
        x_sample = x_sample.transpose((0, 2, 1))
        return dw_sample, x_sample

    # ------------------------------------------------------------------
    # PDE coefficients (PyTorch, for use in the solver's forward pass)
    # ------------------------------------------------------------------

    def f_tf(self, t, x, y, z):
        """Generator: f(t, x, y, z) = -(sum(z)/sqrt(d) - y/2) - (sqrt(1 + y^2 + |z|^2) - sqrt(2))"""
        term1 = torch.sum(z, dim=1, keepdim=True) / np.sqrt(self.dim) - y / 2
        term2 = torch.sqrt(1 + y ** 2 + torch.sum(z ** 2, dim=1, keepdim=True)) - np.sqrt(2)
        return -term1 - term2

    def g_tf(self, t, x):
        """Terminal: g(T, x) = sin(T + sum(x)/sqrt(d))"""
        return torch.sin(t + torch.sum(x, dim=1, keepdim=True) / np.sqrt(self.dim))

    def z_true(self, t, x):
        """True gradient (for validation)."""
        return torch.cos(t + torch.sum(x, dim=1, keepdim=True) / np.sqrt(self.dim)) / np.sqrt(
            self.dim
        )

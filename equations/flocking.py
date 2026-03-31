# equations/flocking.py
#
# Cucker-Smale mean-field game flocking model.
# Ground truth via Riccati ODE for eta(t), xi(t).
# Reference: Han, Hu, Long (2022)

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

from registry import register_equation
from .base import Equation


class _YDriftNet(nn.Module):
    """NN to approximate the mean-field drift terms (y1_drift, y2_drift)."""

    def __init__(self, dim, net_width):
        super().__init__()
        layers = []
        in_dim = 2 * dim + 1  # (t, x, v)
        for w in net_width:
            layers.extend([nn.Linear(in_dim, w, dtype=torch.float64), nn.ReLU()])
            in_dim = w
        layers.append(nn.Linear(in_dim, 2 * dim, dtype=torch.float64))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@register_equation("flocking")
class Flocking(Equation):
    """Cucker-Smale flocking MFG."""

    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.x_init_mean = 0.0
        self.x_init_sigma = 1.0
        self.v_init_mean = 1.0
        self.v_init_sigma = 1.0
        self.R, self.Q, self.C = 0.5, 1.0, 0.1
        self.eta, self.xi = self._riccati_solu()
        self.y2_init_true_fn = lambda v: v @ self.eta[0].T + self.xi[0][None, :]
        self.y_drift_model = _YDriftNet(self.dim, [48, 48])

    def _riccati_solu(self):
        """Solve the Riccati ODE backward in time for ground truth."""
        n = self.dim

        def riccati(t, y):
            eta = np.reshape(y[: n ** 2], (n, n))
            xi = y[n ** 2 :]
            deta = 2 * self.Q * np.identity(n) - eta @ eta / self.R / 2
            dxi = -2 * self.Q * self.v_init_mean - eta @ xi / self.R / 2
            dy = np.concatenate([deta.reshape([-1]), dxi])
            return dy

        y_init = np.zeros([n ** 2 + n])
        sol = solve_ivp(
            riccati,
            [0, self.total_time],
            y_init,
            t_eval=np.linspace(0, self.total_time, self.num_time_interval + 1),
        )
        sol_path = np.flip(sol.y, axis=-1)
        eta_path = np.reshape(
            sol_path[: n ** 2].transpose(), (self.num_time_interval + 1, n, n)
        )
        xi_path = sol_path[n ** 2 :].transpose()
        return eta_path, xi_path

    def sample(self, num_sample):
        dw_sample = np.random.normal(
            scale=self.sqrt_delta_t, size=[num_sample, self.dim, self.num_time_interval]
        )
        x_init = np.random.normal(
            loc=self.x_init_mean, scale=self.x_init_sigma, size=[num_sample, self.dim]
        )
        v_init = np.random.normal(
            loc=self.v_init_mean, scale=self.v_init_sigma, size=[num_sample, self.dim]
        )
        return {"dw": dw_sample, "x_init": x_init, "v_init": v_init}

    def update_drift(self, path_data):
        """Retrain the drift NN on MC-estimated mean-field interaction terms."""
        optimizer = torch.optim.Adam(self.y_drift_model.parameters(), lr=1e-3)
        X = path_data["input"]
        Y = path_data["y_drift"]

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        self.y_drift_model.train()
        for epoch in range(3):
            for xb, yb in loader:
                pred = self.y_drift_model(xb)
                loss = nn.functional.mse_loss(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.y_drift_model.eval()

    def y_drift_nn(self, t, x, v):
        """NN approximation of mean-field drift."""
        inp = torch.cat([t, x, v], dim=1)
        y_drift = self.y_drift_model(inp)
        y1_drift = y_drift[:, : self.dim]
        y2_drift = y_drift[:, self.dim :]
        return y1_drift, y2_drift

    def y_drift_mc(self, t, x, v):
        """Monte Carlo kernel approximation of mean-field drift."""
        beta = self.eqn_config.beta
        # delta_x[i,j] = x[i] - x[j], shape [N, N, dim]
        delta_x = x.unsqueeze(1) - x
        delta_v = v.unsqueeze(1) - v
        x_norm2 = torch.sum(delta_x ** 2, dim=2, keepdim=True)  # [N, N, 1]
        weight = 1.0 / torch.pow(1 + x_norm2, beta)  # [N, N, 1]
        weight_mean = torch.mean(weight, dim=1)  # [N, 1]

        # partial_weight: derivative of weight w.r.t. x_i
        partial_weight = (
            -2 * beta * x.unsqueeze(1) / torch.pow(1 + x_norm2, beta + 1)
        )  # [N, N, dim]
        partial_weight_v = partial_weight.unsqueeze(-1) @ (-delta_v).unsqueeze(
            2
        )  # [N, N, dim, dim]
        partial_weight_v_mean = torch.mean(partial_weight_v, dim=1)  # [N, dim, dim]

        weight_v_mean = torch.mean(weight * (-delta_v), dim=1)  # [N, dim]
        weight_v_mean_expand = weight_v_mean.unsqueeze(-1)  # [N, dim, 1]

        y1_drift = (partial_weight_v_mean @ weight_v_mean_expand)[..., 0]  # [N, dim]
        y1_drift = 2 * self.Q * y1_drift
        y2_drift = -2 * self.Q * weight_mean * weight_v_mean  # [N, dim]
        return y1_drift, y2_drift

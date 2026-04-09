# solver.py
#
# PyTorch reimplementation of the Deep MV-BSDE solver.
# Contains:
#   - FeedForwardSubNet (with batch norm)
#   - FeedForwardNoBNSubNet (without batch norm, for DBDP-iterative)
#   - SineBMNonsharedModel (DeepBSDE global loss)
#   - SineBMNonsharedModelDBDPSingle (DBDP with intermediate losses)
#   - SineBMSolver (trains DeepBSDE or DBDP-single models)
#   - SineBMDBDPSolver (DBDP iterative backward sweep)
#   - FlockNonsharedModel
#   - FlockSolver
#
# Reference: Han, Hu, Long (2022) — frankhan91/DeepMVBSDE

import copy
import logging
import time
import numpy as np
import torch
import torch.nn as nn

from config import DELTA_CLIP


# =====================================================================
# Neural network building blocks
# =====================================================================


class MeanFieldSubNet(nn.Module):
    """Two-stream subnet for MV models: separate state and law processing.

    The law embedding Phi(mu_t) is broadcast (identical across batch),
    so ANY BatchNorm layer will zero it out (zero cross-batch variance).

    Solution: two separate streams merged at the output:
    - State stream: BN-Dense-BN-ReLU (standard, handles per-agent state)
    - Law stream: Dense-ReLU-Dense (NO BN, handles broadcast law embedding)
    - Output: state_output + law_output (additive combination)

    This ensures the law signal survives all normalisation layers.
    """

    def __init__(self, num_hiddens, state_dim, law_dim, dim_out, dtype=torch.float64):
        super().__init__()
        self.state_dim = state_dim
        self.law_dim = law_dim
        hidden = num_hiddens[0] if num_hiddens else 32

        # State stream (with BN, standard architecture)
        self.state_bn_in = nn.BatchNorm1d(state_dim, momentum=0.01, eps=1e-6, dtype=dtype)
        nn.init.normal_(self.state_bn_in.bias, mean=0.0, std=0.1)
        nn.init.uniform_(self.state_bn_in.weight, 0.1, 0.5)
        self.state_dense1 = nn.Linear(state_dim, hidden, bias=False, dtype=dtype)
        self.state_bn1 = nn.BatchNorm1d(hidden, momentum=0.01, eps=1e-6, dtype=dtype)
        nn.init.normal_(self.state_bn1.bias, mean=0.0, std=0.1)
        nn.init.uniform_(self.state_bn1.weight, 0.1, 0.5)
        self.state_dense2 = nn.Linear(hidden, dim_out, dtype=dtype)

        # Law stream (NO BN — broadcast features must not be normalised)
        self.law_dense1 = nn.Linear(law_dim, hidden, dtype=dtype)
        self.law_dense2 = nn.Linear(hidden, dim_out, dtype=dtype)

    def forward(self, x):
        state = x[:, :self.state_dim]
        law = x[:, self.state_dim:]

        # State stream: BN -> Dense -> BN -> ReLU -> Dense
        s = self.state_bn_in(state)
        s = self.state_dense1(s)
        s = self.state_bn1(s)
        s = torch.relu(s)
        s = self.state_dense2(s)

        # Law stream: Dense -> ReLU -> Dense (no BN)
        l = self.law_dense1(law)
        l = torch.relu(l)
        l = self.law_dense2(l)

        # Additive combination
        return s + l


class FiLMSubNet(nn.Module):
    """FiLM-conditioned subnet: law embedding modulates state activations.

    Instead of Z = f(state) + g(law) (additive, no interaction),
    FiLM computes Z = Dense(ReLU(gamma(law) * BN(Dense(state)) + beta(law))).

    This allows the law embedding to change HOW state maps to Z,
    not just shift the output. The cross-partial d2Z/(dq * d_law)
    can be nonzero, unlike the additive two-stream architecture.
    """

    def __init__(self, num_hiddens, state_dim, law_dim, dim_out, dtype=torch.float64):
        super().__init__()
        self.state_dim = state_dim
        self.law_dim = law_dim
        hidden = num_hiddens[0] if num_hiddens else 32

        # State stream (with BN, same as MeanFieldSubNet)
        self.state_bn_in = nn.BatchNorm1d(state_dim, momentum=0.01, eps=1e-6, dtype=dtype)
        nn.init.normal_(self.state_bn_in.bias, mean=0.0, std=0.1)
        nn.init.uniform_(self.state_bn_in.weight, 0.1, 0.5)
        self.state_dense1 = nn.Linear(state_dim, hidden, bias=False, dtype=dtype)
        self.state_bn1 = nn.BatchNorm1d(hidden, momentum=0.01, eps=1e-6, dtype=dtype)
        nn.init.normal_(self.state_bn1.bias, mean=0.0, std=0.1)
        nn.init.uniform_(self.state_bn1.weight, 0.1, 0.5)

        # FiLM generator: law_embed -> (gamma, beta) for hidden layer
        self.film_net = nn.Sequential(
            nn.Linear(law_dim, hidden, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden, 2 * hidden, dtype=dtype),  # first half = gamma, second = beta
        )
        # Initialize so FiLM starts as identity: gamma=1, beta=0
        with torch.no_grad():
            self.film_net[-1].weight.zero_()
            self.film_net[-1].bias[:hidden] = 1.0   # gamma = 1
            self.film_net[-1].bias[hidden:] = 0.0   # beta = 0

        # Output layer
        self.state_dense2 = nn.Linear(hidden, dim_out, dtype=dtype)

    def forward(self, x):
        state = x[:, :self.state_dim]
        law = x[:, self.state_dim:]
        hidden = self.state_dense2.in_features

        # State stream up to BN1
        s = self.state_bn_in(state)
        s = self.state_dense1(s)
        s = self.state_bn1(s)

        # FiLM modulation
        film_params = self.film_net(law)
        gamma = film_params[:, :hidden]
        beta = film_params[:, hidden:]
        s = gamma * s + beta

        # ReLU + output
        s = torch.relu(s)
        return self.state_dense2(s)


class FiLMPlusAdditiveSubNet(nn.Module):
    """FiLM modulation + additive law stream (for ablation).

    Output = FiLM_path(state, law) + additive_path(law)
    This tests whether additive and multiplicative channels carry
    complementary information.
    """

    def __init__(self, num_hiddens, state_dim, law_dim, dim_out, dtype=torch.float64):
        super().__init__()
        self.film = FiLMSubNet(num_hiddens, state_dim, law_dim, dim_out, dtype=dtype)
        hidden = num_hiddens[0] if num_hiddens else 32
        self.additive = nn.Sequential(
            nn.Linear(law_dim, hidden, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden, dim_out, dtype=dtype),
        )

    def forward(self, x):
        return self.film(x) + self.additive(x[:, self.film.state_dim:])


# Subnet factory for MV models
SUBNET_REGISTRY = {
    "two_stream": MeanFieldSubNet,
    "film": FiLMSubNet,
    "film_additive": FiLMPlusAdditiveSubNet,
}


def create_mv_subnet(subnet_type, num_hiddens, state_dim, law_dim, dim_out, dtype=torch.float64):
    """Create a subnet for MV models by type name."""
    if subnet_type not in SUBNET_REGISTRY:
        raise ValueError(f"Unknown subnet_type '{subnet_type}'. Options: {list(SUBNET_REGISTRY.keys())}")
    return SUBNET_REGISTRY[subnet_type](num_hiddens, state_dim, law_dim, dim_out, dtype=dtype)


class FeedForwardSubNet(nn.Module):
    """BN -> (Dense(no bias) -> BN -> ReLU) x L -> Dense -> BN

    Exactly matches the reference architecture. Dense hidden layers have
    no bias because the following BN absorbs it.
    """

    def __init__(self, num_hiddens, dim_in, dim_out, dtype=torch.float64):
        super().__init__()
        dims = [dim_in] + list(num_hiddens) + [dim_out]

        self.bn_layers = nn.ModuleList()
        for d in dims:
            bn = nn.BatchNorm1d(d, momentum=0.01, eps=1e-6, dtype=dtype)
            nn.init.normal_(bn.bias, mean=0.0, std=0.1)
            nn.init.uniform_(bn.weight, 0.1, 0.5)
            self.bn_layers.append(bn)

        self.dense_layers = nn.ModuleList()
        for i in range(len(num_hiddens)):
            self.dense_layers.append(
                nn.Linear(dims[i], dims[i + 1], bias=False, dtype=dtype)
            )
        self.dense_layers.append(
            nn.Linear(dims[-2], dims[-1], dtype=dtype)
        )

    def forward(self, x):
        x = self.bn_layers[0](x)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i + 1](x)
            x = torch.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x)
        return x


class FeedForwardNoBNSubNet(nn.Module):
    """(Dense(no bias) -> ReLU) x L -> Dense. Used in DBDP-iterative."""

    def __init__(self, num_hiddens, dim_in, dim_out, dtype=torch.float64):
        super().__init__()
        layers = []
        in_d = dim_in
        for h in num_hiddens:
            layers.extend([nn.Linear(in_d, h, bias=False, dtype=dtype), nn.ReLU()])
            in_d = h
        layers.append(nn.Linear(in_d, dim_out, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =====================================================================
# Utility: piecewise constant LR schedule
# =====================================================================


def make_piecewise_lr_scheduler(optimizer, lr_boundaries, lr_values):
    """Mimics tf.keras.optimizers.schedules.PiecewiseConstantDecay."""

    def lr_lambda(step):
        lr = lr_values[0]
        for boundary, value in zip(lr_boundaries, lr_values[1:]):
            if step >= boundary:
                lr = value
        return lr / lr_values[0]  # relative to initial lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =====================================================================
# SineBM Models
# =====================================================================


class SineBMNonsharedModel(nn.Module):
    """DeepBSDE global loss model for SineBM.

    Learnable y_init, z_init. One subnet per time step for Z.
    Forward Euler: y_{t+1} = y_t - dt * f(t,x,y,z) + z . dW
    """

    def __init__(self, config, bsde):
        super().__init__()
        self.eqn_config = config.eqn
        self.net_config = config.net
        self.bsde = bsde
        dtype = torch.float64

        self.y_init = nn.Parameter(
            torch.tensor(
                np.random.uniform(
                    low=self.net_config.y_init_range[0],
                    high=self.net_config.y_init_range[1],
                    size=[1],
                ),
                dtype=dtype,
            )
        )
        self.z_init = nn.Parameter(
            torch.tensor(
                np.random.uniform(low=-0.1, high=0.1, size=[1, self.eqn_config.dim]),
                dtype=dtype,
            )
        )
        self.subnet = nn.ModuleList(
            [
                FeedForwardSubNet(
                    self.net_config.num_hiddens, self.eqn_config.dim,
                    self.eqn_config.dim, dtype=dtype,
                )
                for _ in range(self.bsde.num_time_interval - 1)
            ]
        )

    def forward(self, inputs):
        dw, x, mean_y_input = inputs
        dw = torch.as_tensor(dw, dtype=torch.float64)
        x = torch.as_tensor(x, dtype=torch.float64)

        loss_inter = torch.tensor(0.0, dtype=torch.float64)
        mean_y = []
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        batch_size = dw.shape[0]

        all_one = torch.ones(batch_size, 1, dtype=torch.float64, device=dw.device)
        y = all_one * self.y_init
        z = all_one @ self.z_init  # [batch, dim]
        mean_y.append(torch.mean(y))

        for t in range(self.bsde.num_time_interval - 1):
            y = (
                y
                - self.bsde.delta_t * self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
                + torch.sum(z * dw[:, :, t], dim=1, keepdim=True)
            )
            mean_y.append(torch.mean(y))
            if self.eqn_config.type == 2:
                y = y + (mean_y_input[t] - self.bsde.mean_y[t]) * self.bsde.delta_t
            z = self.subnet[t](x[:, :, t + 1]) / self.bsde.dim

        # Terminal step
        y = (
            y
            - self.bsde.delta_t
            * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z)
            + torch.sum(z * dw[:, :, -1], dim=1, keepdim=True)
        )
        if self.eqn_config.type == 2:
            y = y + (mean_y_input[-2] - self.bsde.mean_y[-2]) * self.bsde.delta_t

        return y, mean_y, loss_inter


class SineBMNonsharedModelDBDPSingle(nn.Module):
    """DBDP single-loss model: separate Y and Z subnets per step.

    Adds intermediate loss |y_predicted - stop_gradient(y_next_net)|^2.
    """

    def __init__(self, config, bsde):
        super().__init__()
        self.eqn_config = config.eqn
        self.net_config = config.net
        self.bsde = bsde
        dtype = torch.float64

        self.y_init = nn.Parameter(
            torch.tensor(
                np.random.uniform(
                    low=self.net_config.y_init_range[0],
                    high=self.net_config.y_init_range[1],
                    size=[1],
                ),
                dtype=dtype,
            )
        )
        self.z_init = nn.Parameter(
            torch.tensor(
                np.random.uniform(low=-0.1, high=0.1, size=[1, self.eqn_config.dim]),
                dtype=dtype,
            )
        )
        self.subnetz = nn.ModuleList(
            [
                FeedForwardSubNet(
                    self.net_config.num_hiddens, self.eqn_config.dim,
                    self.eqn_config.dim, dtype=dtype,
                )
                for _ in range(self.bsde.num_time_interval - 1)
            ]
        )
        self.subnety = nn.ModuleList(
            [
                FeedForwardSubNet(
                    self.net_config.num_hiddens, self.eqn_config.dim, 1, dtype=dtype,
                )
                for _ in range(self.bsde.num_time_interval - 1)
            ]
        )

    def forward(self, inputs):
        dw, x, mean_y_input = inputs
        dw = torch.as_tensor(dw, dtype=torch.float64)
        x = torch.as_tensor(x, dtype=torch.float64)

        loss_inter = torch.tensor(0.0, dtype=torch.float64, device=dw.device)
        mean_y = []
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        batch_size = dw.shape[0]

        all_one = torch.ones(batch_size, 1, dtype=torch.float64, device=dw.device)
        y_now = all_one * self.y_init
        z = all_one @ self.z_init
        mean_y.append(torch.mean(y_now))

        for t in range(self.bsde.num_time_interval - 1):
            y_predict = (
                y_now
                - self.bsde.delta_t
                * self.bsde.f_tf(time_stamp[t], x[:, :, t], y_now, z)
                + torch.sum(z * dw[:, :, t], dim=1, keepdim=True)
            )
            if self.eqn_config.type == 2:
                y_predict = (
                    y_predict
                    + (mean_y_input[t] - self.bsde.mean_y[t]) * self.bsde.delta_t
                )
            y_next = self.subnety[t](x[:, :, t + 1])
            loss_inter = loss_inter + torch.mean(
                (y_predict - y_next.detach()) ** 2
            )
            mean_y.append(torch.mean(y_next))
            y_now = y_next
            z = self.subnetz[t](x[:, :, t + 1]) / self.bsde.dim

        # Terminal step
        y_terminal = (
            y_now
            - self.bsde.delta_t
            * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y_now, z)
            + torch.sum(z * dw[:, :, -1], dim=1, keepdim=True)
        )
        if self.eqn_config.type == 2:
            y_terminal = (
                y_terminal
                + (mean_y_input[-2] - self.bsde.mean_y[-2]) * self.bsde.delta_t
            )

        return y_terminal, mean_y, loss_inter


# =====================================================================
# SineBM Solvers
# =====================================================================


class SineBMSolver:
    """Trains SineBMNonsharedModel or SineBMNonsharedModelDBDPSingle."""

    def __init__(self, config, bsde):
        self.eqn_config = config.eqn
        self.net_config = config.net
        self.bsde = bsde

        if self.net_config.loss_type == "DeepBSDE":
            self.model = SineBMNonsharedModel(config, bsde)
            self.opt_config = self.net_config.opt_config1
        elif self.net_config.loss_type == "DBDPsingle":
            self.model = SineBMNonsharedModelDBDPSingle(config, bsde)
            self.opt_config = self.net_config.opt_config2
        else:
            raise ValueError(f"SineBMSolver: unknown loss_type {self.net_config.loss_type}")

        self.y_init = self.model.y_init
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.opt_config.lr_values[0], eps=1e-8
        )
        self.scheduler = make_piecewise_lr_scheduler(
            self.optimizer, self.opt_config.lr_boundaries, self.opt_config.lr_values
        )

    def loss_fn(self, inputs):
        y_terminal, mean_y, loss_inter = self.model(inputs)
        dw, x, mean_y_input = inputs
        x = torch.as_tensor(x, dtype=torch.float64)
        y_target = self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        delta = y_terminal - y_target
        mean_y.append(torch.mean(y_target))
        # Clipped Huber-like loss
        loss = loss_inter + torch.mean(
            torch.where(
                torch.abs(delta) < DELTA_CLIP,
                delta ** 2,
                2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2,
            )
        )
        return loss, mean_y

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)
        mean_y_train = self.bsde.mean_y * 0

        for step in range(self.opt_config.num_iterations + 1):
            # Update drift for type 3 coupling
            if (
                self.eqn_config.type == 3
                and step % self.opt_config.freq_update_drift == 0
            ):
                self.bsde.update_mean_y_estimate(mean_y_train)
                self.bsde.update_drift()
                valid_data = self.bsde.sample(self.net_config.valid_size, seed=1)

            # Resample training paths
            if step % self.opt_config.freq_resample == 0:
                train_data = self.bsde.sample(self.net_config.batch_size)
                train_data = (train_data[0], train_data[1], mean_y_train)

            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            loss, mean_y = self.loss_fn(train_data)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Extract mean_y for coupling
            mean_y_train = np.array([m.item() for m in mean_y])

            # Logging
            if step % self.net_config.logging_frequency == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss, val_mean_y = self.loss_fn(
                        (valid_data[0], valid_data[1], mean_y_train)
                    )
                val_loss = val_loss.item()
                y_init_val = self.y_init.item()
                mean_y_valid = np.array([m.item() for m in val_mean_y])
                err_mean_y = np.mean((mean_y_valid - self.bsde.mean_y) ** 2)
                elapsed = time.time() - start_time
                training_history.append(
                    [step, val_loss, y_init_val, err_mean_y, elapsed]
                )
                if self.net_config.verbose:
                    logging.info(
                        "step: %5u,    loss: %.4e, Y0: %.4e,   err_mean_y: %.4e,    elapsed time: %3u"
                        % (step, val_loss, y_init_val, err_mean_y, elapsed)
                    )

        # Final validation
        valid_data = self.bsde.sample(self.net_config.valid_size * 20)
        self.model.eval()
        with torch.no_grad():
            _, mean_y_valid = self.loss_fn(
                (valid_data[0], valid_data[1], mean_y_train)
            )
        mean_y_valid = np.array([m.item() for m in mean_y_valid])
        print("Estimated mean_y:")
        print(mean_y_valid)
        print("Error of mean_y:")
        print(mean_y_valid - self.bsde.mean_y)
        print("Average squared error of mean_y:")
        print(np.mean((mean_y_valid - self.bsde.mean_y) ** 2))

        return {
            "history": np.array(training_history),
            "true_mean_y": self.bsde.mean_y,
            "estimated_mean_y": mean_y_valid,
            "err_mean_y": np.mean((mean_y_valid - self.bsde.mean_y) ** 2),
        }


# =====================================================================
# SineBM DBDP Iterative Solver
# =====================================================================


class SineBMDBDPSolver:
    """DBDP iterative: backward sweep, one time step at a time."""

    def __init__(self, config, bsde):
        self.eqn_config = config.eqn
        self.net_config = config.net
        self.bsde = bsde
        self.opt_config = self.net_config.opt_config3
        dtype = torch.float64
        dim_in = bsde.dim + 1  # (t, x) for withtime=True

        self.nety_weights = [None] * bsde.num_time_interval
        self.netz_weights = [None] * bsde.num_time_interval

        self.y_init = nn.Parameter(
            torch.tensor(
                np.random.uniform(
                    low=self.net_config.y_init_range[0],
                    high=self.net_config.y_init_range[1],
                    size=[1],
                ),
                dtype=dtype,
            )
        )
        self.z_init = nn.Parameter(
            torch.tensor(
                np.random.uniform(low=-0.1, high=0.1, size=[1, self.eqn_config.dim]),
                dtype=dtype,
            )
        )

        self.nety = FeedForwardNoBNSubNet(
            self.net_config.num_hiddens, dim_in, 1, dtype=dtype
        )
        self.netz = FeedForwardNoBNSubNet(
            self.net_config.num_hiddens, dim_in, self.eqn_config.dim, dtype=dtype
        )
        self.nety_target = FeedForwardNoBNSubNet(
            self.net_config.num_hiddens, dim_in, 1, dtype=dtype
        )

        self.init_variables = [self.y_init, self.z_init]
        self.net_variables = (
            list(self.nety.parameters()) + list(self.netz.parameters())
        )
        self.mean_y_train = self.bsde.mean_y * 0

    def _save_weights(self, net):
        return copy.deepcopy(net.state_dict())

    def _load_weights(self, net, state):
        if state is not None:
            net.load_state_dict(state)

    def local_loss_fn(self, inputs, t):
        dw, x, mean_y_input = inputs
        dw = torch.as_tensor(dw, dtype=torch.float64)
        x = torch.as_tensor(x, dtype=torch.float64)
        batch_size = dw.shape[0]

        if t == self.bsde.num_time_interval - 1:
            y_target = self.bsde.g_tf(self.bsde.total_time, x[:, 1:, -1])
            self.nety_target(x[:, :, t])  # dummy call
        else:
            y_target = self.nety_target(x[:, :, t + 1])

        if t == 0:
            all_one = torch.ones(batch_size, 1, dtype=torch.float64)
            y = all_one * self.y_init
            z = all_one @ self.z_init
        else:
            y = self.nety(x[:, :, t])
            z = self.netz(x[:, :, t]) / self.bsde.dim

        mean_y_estimate = [torch.mean(y), torch.mean(y_target)]

        y_next = (
            y
            - self.bsde.delta_t
            * self.bsde.f_tf(self.bsde.delta_t * t, x[:, :, t], y, z)
            + torch.sum(z * dw[:, :, t], dim=1, keepdim=True)
        )
        if self.eqn_config.type == 2:
            y_next = y_next + (mean_y_input[t] - self.bsde.mean_y[t]) * self.bsde.delta_t

        delta = y_next - y_target
        loss = torch.mean(
            torch.where(
                torch.abs(delta) < DELTA_CLIP,
                delta ** 2,
                2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2,
            )
        )
        return loss, mean_y_estimate

    def total_loss_fn(self, inputs):
        total_loss = 0
        mean_y = [None] * (self.bsde.num_time_interval + 1)
        for t in range(self.bsde.num_time_interval - 1, -1, -1):
            if t > 0:
                self._load_weights(self.nety, self.nety_weights[t])
                self._load_weights(self.netz, self.netz_weights[t])
            if t < self.bsde.num_time_interval - 1:
                self._load_weights(self.nety_target, self.nety_weights[t + 1])
            with torch.no_grad():
                loss_tmp, mean_y_tmp = self.local_loss_fn(inputs, t)
            mean_y[t] = mean_y_tmp[0]
            if t == self.bsde.num_time_interval - 1:
                mean_y[self.bsde.num_time_interval] = mean_y_tmp[1]
            total_loss += loss_tmp.item()
        return total_loss, mean_y

    def train_one_sweep(self, train_data):
        if self.nety_weights[-1] is not None:
            self._load_weights(self.nety, self.nety_weights[-1])
            self._load_weights(self.netz, self.netz_weights[-1])

        for t in range(self.bsde.num_time_interval - 1, -1, -1):
            # Fresh optimizer per step (matches reference resetting optimizer state)
            if t == 0:
                optimizer = torch.optim.Adam(
                    self.init_variables, lr=self.opt_config.lr_values[0], eps=1e-8
                )
            else:
                optimizer = torch.optim.Adam(
                    self.net_variables, lr=self.opt_config.lr_values[0], eps=1e-8
                )

            for _ in range(self.opt_config.num_iterations_perstep):
                optimizer.zero_grad()
                loss, mean_y_local = self.local_loss_fn(train_data, t)
                loss.backward()
                optimizer.step()

            self.mean_y_train[t] = mean_y_local[0].item()
            if t == self.bsde.num_time_interval - 1:
                self.mean_y_train[t + 1] = mean_y_local[1].item()

            if t > 0:
                self.nety_weights[t] = self._save_weights(self.nety)
                self.netz_weights[t] = self._save_weights(self.netz)
                self._load_weights(self.nety_target, self.nety_weights[t])

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size, withtime=True)

        for step in range(self.opt_config.num_sweep):
            if self.eqn_config.type == 3:
                self.bsde.update_mean_y_estimate(self.mean_y_train)
                self.bsde.update_drift()
                valid_data = self.bsde.sample(
                    self.net_config.valid_size, withtime=True, seed=1
                )

            train_data = self.bsde.sample(self.net_config.batch_size, withtime=True)
            train_data = (train_data[0], train_data[1], self.mean_y_train)
            self.train_one_sweep(train_data)

            loss, mean_y_valid = self.total_loss_fn(
                (valid_data[0], valid_data[1], self.mean_y_train)
            )
            y_init_val = self.y_init.item()
            mean_y_valid_np = np.array([m.item() for m in mean_y_valid])
            err_mean_y = np.mean((mean_y_valid_np - self.bsde.mean_y) ** 2)
            elapsed = time.time() - start_time
            training_history.append([step, loss, y_init_val, err_mean_y, elapsed])
            if self.net_config.verbose:
                logging.info(
                    "step: %5u,    loss: %.4e, Y0: %.4e,   err_mean_y: %.4e,    elapsed time: %3u"
                    % (step, loss, y_init_val, err_mean_y, elapsed)
                )

        # Final validation
        valid_data = self.bsde.sample(self.net_config.valid_size * 20, withtime=True)
        _, mean_y_valid = self.total_loss_fn(
            (valid_data[0], valid_data[1], self.mean_y_train)
        )
        mean_y_valid_np = np.array([m.item() for m in mean_y_valid])
        print("Estimated mean_y:")
        print(mean_y_valid_np)
        print("Error of mean_y:")
        print(mean_y_valid_np - self.bsde.mean_y)
        print("Average squared error of mean_y:")
        print(np.mean((mean_y_valid_np - self.bsde.mean_y) ** 2))

        return {
            "history": np.array(training_history),
            "true_mean_y": self.bsde.mean_y,
            "estimated_mean_y": mean_y_valid_np,
            "err_mean_y": np.mean((mean_y_valid_np - self.bsde.mean_y) ** 2),
        }


# =====================================================================
# Flocking Model and Solver
# =====================================================================


class FlockNonsharedModel(nn.Module):
    """Non-shared model for Cucker-Smale flocking MFG.

    y_init_net: maps (x, v) -> (y1, y2) of dim 2*d
    z_subnet[t]: maps v -> z of dim d^2 * 2
    """

    def __init__(self, config, bsde):
        super().__init__()
        self.eqn_config = config.eqn
        self.net_config = config.net
        self.bsde = bsde
        dtype = torch.float64
        dim = self.eqn_config.dim

        self.y_init_net = FeedForwardSubNet(
            self.net_config.num_hiddens, 2 * dim, 2 * dim, dtype=dtype,
        )
        self.z_subnet = nn.ModuleList(
            [
                FeedForwardSubNet(
                    self.net_config.num_hiddens, dim, dim ** 2 * 2, dtype=dtype,
                )
                for _ in range(bsde.num_time_interval)
            ]
        )

    def simulate_abstract(self, inputs, drift_type="NN"):
        dtype = torch.float64
        dw = torch.as_tensor(inputs["dw"], dtype=dtype)
        x = torch.as_tensor(inputs["x_init"], dtype=dtype)
        v = torch.as_tensor(inputs["v_init"], dtype=dtype)
        batch_size = dw.shape[0]
        dim = self.eqn_config.dim

        v_std = [torch.std(v[:, 0])]
        y = self.y_init_net(torch.cat([x, v], dim=1))
        y1, y2 = y[:, :dim], y[:, dim:]

        z = self.z_subnet[0](v) / self.bsde.dim
        z = z.reshape(-1, 2 * dim, dim)

        drift_input = torch.zeros(0, 2 * dim + 1, dtype=dtype)
        y_drift_label = torch.zeros(0, 2 * dim, dtype=dtype)

        for t in range(self.bsde.num_time_interval):
            all_one = torch.ones(batch_size, 1, dtype=dtype)
            t_input = t * self.bsde.delta_t * all_one

            if drift_type == "NN":
                y1_drift, y2_drift = self.bsde.y_drift_nn(t_input, x, v)
            elif drift_type == "MC":
                y1_drift_mc, y2_drift_mc = self.bsde.y_drift_mc(t_input, x, v)
                y_drift_mc = torch.cat([y1_drift_mc, y2_drift_mc], dim=1)
                y_drift_label = torch.cat([y_drift_label, y_drift_mc], dim=0)
                drift_input = torch.cat(
                    [drift_input, torch.cat([t_input, x, v], dim=1)], dim=0
                )
                y1_drift, y2_drift = self.bsde.y_drift_nn(t_input, x, v)

            # State dynamics
            x = x + v * self.bsde.delta_t
            v = v - y2 / self.bsde.R / 2 * self.bsde.delta_t + self.bsde.C * dw[:, :, t]

            # BSDE dynamics
            diffusion = (z @ dw[:, :, t : t + 1])[..., 0]  # [batch, 2*dim]
            y1 = y1 - y1_drift * self.bsde.delta_t + diffusion[:, :dim]
            y2 = y2 - (y1 + y2_drift) * self.bsde.delta_t + diffusion[:, dim:]

            v_std.append(torch.std(v[:, 0]))
            if t < self.bsde.num_time_interval - 1:
                z = self.z_subnet[t + 1](v) / self.bsde.dim
                z = z.reshape(-1, 2 * dim, dim)

        v_std = torch.stack(v_std, dim=0)
        final_state = torch.cat([x, v], dim=1)
        path_data = {
            "input": drift_input,
            "y_drift": y_drift_label,
            "v_std": v_std,
            "final_state": final_state,
        }
        y = torch.cat([y1, y2], dim=1)
        return y, path_data

    def y2_init_predict(self, inputs):
        x = torch.as_tensor(inputs["x_init"], dtype=torch.float64)
        v = torch.as_tensor(inputs["v_init"], dtype=torch.float64)
        with torch.no_grad():
            y = self.y_init_net(torch.cat([x, v], dim=1))
        return y[:, self.eqn_config.dim :]

    def forward(self, inputs):
        return self.simulate_abstract(inputs, "NN")


class FlockSolver:
    """Solver for the Cucker-Smale flocking MFG."""

    def __init__(self, config, bsde):
        self.eqn_config = config.eqn
        self.net_config = config.net
        self.bsde = bsde

        self.model = FlockNonsharedModel(config, bsde)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.net_config.lr_values[0],
            eps=1e-8,
        )
        self.scheduler = make_piecewise_lr_scheduler(
            self.optimizer,
            self.net_config.lr_boundaries,
            self.net_config.lr_values,
        )

    def loss_fn(self, inputs, drift_type="NN"):
        y_terminal, path_data = self.model.simulate_abstract(inputs, drift_type)
        delta = y_terminal
        loss = torch.mean(
            torch.where(
                torch.abs(delta) < DELTA_CLIP,
                delta ** 2,
                2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2,
            )
        )
        return loss, path_data

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)
        valid_y2_init_true = self.bsde.y2_init_true_fn(valid_data["v_init"])

        for step in range(self.net_config.num_iterations + 1):
            # Periodically retrain drift NN on MC estimates
            if step % 50 == 0:
                simul_data = self.bsde.sample(self.net_config.simul_size)
                self.model.eval()
                with torch.no_grad():
                    _, path_data = self.model.simulate_abstract(simul_data, "MC")
                self.bsde.update_drift(path_data)

            train_data = self.bsde.sample(self.net_config.batch_size)
            self.model.train()
            self.optimizer.zero_grad()
            loss, _ = self.loss_fn(train_data)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if step % self.net_config.logging_frequency == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss, _ = self.loss_fn(valid_data)
                val_loss = val_loss.item()
                elapsed = time.time() - start_time
                y2_init = self.model.y2_init_predict(valid_data).numpy()
                err_y2_init = np.mean((y2_init - valid_y2_init_true) ** 2)
                logging.info(
                    "step: %5u,    loss: %.4e, err_Y2_init: %.4e,    elapsed time: %3u"
                    % (step, val_loss, err_y2_init, elapsed)
                )
                training_history.append([step, val_loss, err_y2_init, elapsed])

        # Final validation
        np.random.seed(self.eqn_config.simul_seed)
        valid_data = self.bsde.sample(self.net_config.simul_size * 10)
        valid_y2_init_true = self.bsde.y2_init_true_fn(valid_data["v_init"])
        y2_init = self.model.y2_init_predict(valid_data).numpy()
        print("Y2_true", valid_y2_init_true[:3])
        self.model.eval()
        with torch.no_grad():
            _, path_data = self.model.simulate_abstract(valid_data, "MC")
        print("Y2_approx", y2_init[:3])
        print("Std of v_terminal", path_data["v_std"].numpy()[-1])
        y2_err = np.mean((y2_init - valid_y2_init_true) ** 2)
        y2_square = np.mean(y2_init ** 2)

        return {
            "history": np.array(training_history),
            "y2_err": y2_err,
            "R2": 1 - y2_err / y2_square,
            "v_std": path_data["v_std"].numpy(),
        }


# =====================================================================
# Cont-Xiong LOB Model and Solver
# =====================================================================


class ContXiongLOBModel(nn.Module):
    """DeepBSDE model for Cont-Xiong LOB market-making.

    State dim = 2 (price S, inventory q).
    Learnable y_init (value function at t=0), z_init (gradient at t=0).
    One FeedForwardSubNet per time step outputs Z_t = (Z^S, Z^q).

    The inventory component Z^q directly gives optimal quotes via
    the Avellaneda-Stoikov first-order condition:
        delta_a = 1/alpha + Z^q,   delta_b = 1/alpha - Z^q
    """

    def __init__(self, config, bsde, device=None):
        super().__init__()
        self.eqn_config = config.eqn
        self.net_config = config.net
        self.bsde = bsde
        self.device = device or torch.device("cpu")
        dtype = torch.float64
        dim = bsde.dim  # 2 for base, 3 for adverse selection

        self.y_init = nn.Parameter(
            torch.tensor(
                np.random.uniform(
                    low=self.net_config.y_init_range[0],
                    high=self.net_config.y_init_range[1],
                    size=[1],
                ),
                dtype=dtype,
            )
        )
        self.z_init = nn.Parameter(
            torch.tensor(
                np.random.uniform(low=-0.1, high=0.1, size=[1, dim]),
                dtype=dtype,
            )
        )
        self.subnet = nn.ModuleList(
            [
                FeedForwardSubNet(
                    self.net_config.num_hiddens, dim, dim, dtype=dtype,
                )
                for _ in range(bsde.num_time_interval - 1)
            ]
        )

    def forward(self, inputs):
        dw, x, mean_y_input = inputs
        dw = torch.as_tensor(dw, dtype=torch.float64, device=self.device)
        x = torch.as_tensor(x, dtype=torch.float64, device=self.device)

        loss_inter = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        mean_y = []
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        batch_size = dw.shape[0]

        all_one = torch.ones(batch_size, 1, dtype=torch.float64, device=self.device)
        y = all_one * self.y_init
        z = all_one @ self.z_init  # [batch, 2]
        mean_y.append(torch.mean(y))

        # Also collect mean spread and inventory for mean-field update
        mean_spreads = []
        mean_inventories = []
        # Z_t diagnostics: track max |Z| and Lipschitz estimate per timestep
        z_max_history = []

        for t in range(self.bsde.num_time_interval - 1):
            # Track mean-field statistics from Z
            z_q = z[:, 1:2]  # inventory gradient
            delta_a, delta_b = self.bsde._optimal_quotes_tf(z_q)
            mean_spreads.append((torch.mean(delta_a) + torch.mean(delta_b)).item())
            mean_inventories.append(torch.mean(x[:, 1, t]).item())
            z_max_history.append(torch.max(torch.abs(z)).item())

            # BSDE step: Y_{t+1} = Y_t - dt * f(t, X_t, Y_t, Z_t) + Z_t . dW_t
            y = (
                y
                - self.bsde.delta_t * self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
                + torch.sum(z * dw[:, :, t], dim=1, keepdim=True)
            )
            mean_y.append(torch.mean(y))

            # Predict next Z from state at t+1
            z = self.subnet[t](x[:, :, t + 1]) / self.bsde.dim

        # Terminal step
        z_q = z[:, 1:2]
        delta_a, delta_b = self.bsde._optimal_quotes_tf(z_q)
        mean_spreads.append((torch.mean(delta_a) + torch.mean(delta_b)).item())
        mean_inventories.append(torch.mean(x[:, 1, -2]).item())

        y = (
            y
            - self.bsde.delta_t
            * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z)
            + torch.sum(z * dw[:, :, -1], dim=1, keepdim=True)
        )

        z_max_history.append(torch.max(torch.abs(z)).item())

        # Store diagnostics for solver
        self._last_mean_spreads = mean_spreads
        self._last_mean_inventories = mean_inventories
        self._last_z_max = z_max_history
        self._last_z_max_overall = max(z_max_history) if z_max_history else 0.0

        return y, mean_y, loss_inter


class ContXiongLOBSolver:
    """Trains ContXiongLOBModel with Type 3 mean-field coupling.

    Follows SineBMSolver pattern:
    - Periodic drift retraining via update_mean_y_estimate + update_drift
    - Clipped Huber loss on terminal mismatch
    - Piecewise LR scheduler
    - Supports CPU and CUDA devices
    """

    def __init__(self, config, bsde, device=None):
        self.eqn_config = config.eqn
        self.net_config = config.net
        self.bsde = bsde
        self.device = device or torch.device("cpu")

        self.model = ContXiongLOBModel(config, bsde, device=self.device)
        self.model.to(self.device)
        self.opt_config = self.net_config.opt_config1
        self.y_init = self.model.y_init

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.opt_config.lr_values[0], eps=1e-8
        )
        self.scheduler = make_piecewise_lr_scheduler(
            self.optimizer, self.opt_config.lr_boundaries, self.opt_config.lr_values
        )

    def loss_fn(self, inputs):
        y_terminal, mean_y, loss_inter = self.model(inputs)
        dw, x, mean_y_input = inputs
        x = torch.as_tensor(x, dtype=torch.float64, device=self.device)
        y_target = self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        delta = y_terminal - y_target
        mean_y.append(torch.mean(y_target))
        loss = loss_inter + torch.mean(
            torch.where(
                torch.abs(delta) < DELTA_CLIP,
                delta ** 2,
                2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2,
            )
        )
        return loss, mean_y

    def train(self):
        start_time = time.time()
        training_history = []
        mean_field_history = []  # Track mu_t evolution across fictitious play iterations
        valid_data = self.bsde.sample(self.net_config.valid_size)
        mean_y_train = np.zeros(self.bsde.num_time_interval + 2)

        for step in range(self.opt_config.num_iterations + 1):
            # Update mean-field for type 3 coupling
            if (
                self.eqn_config.type == 3
                and step % self.opt_config.freq_update_drift == 0
                and step > 0
            ):
                # Pass mean-field statistics back to equation
                if hasattr(self.model, "_last_mean_spreads"):
                    self.bsde.update_mean_field(
                        self.model._last_mean_spreads,
                        self.model._last_mean_inventories,
                    )
                    # Log fictitious play iteration
                    mean_field_history.append({
                        "step": step,
                        "mean_spreads": list(self.model._last_mean_spreads),
                        "mean_inventories": list(self.model._last_mean_inventories),
                        "avg_spread": np.mean(self.model._last_mean_spreads),
                        "avg_abs_inventory": np.mean(np.abs(self.model._last_mean_inventories)),
                    })
                    logging.info(
                        "  [MF update] step %d: avg_spread=%.4f, avg_|q|=%.4f"
                        % (step,
                           mean_field_history[-1]["avg_spread"],
                           mean_field_history[-1]["avg_abs_inventory"])
                    )
                self.bsde.update_drift()
                valid_data = self.bsde.sample(self.net_config.valid_size, seed=1)

            # Resample training paths
            if step % self.opt_config.freq_resample == 0:
                train_data = self.bsde.sample(self.net_config.batch_size)
                train_data = (train_data[0], train_data[1], mean_y_train)

            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            loss, mean_y = self.loss_fn(train_data)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Progress indicator (lightweight, always shown)
            if step > 0 and step % 100 == 0:
                import sys as _sys
                total = self.opt_config.num_iterations
                pct = step * 100 // total
                _sys.stderr.write(f"\r  [{pct:3d}%] step {step}/{total}")
                _sys.stderr.flush()

            # Extract mean_y for logging
            mean_y_train = np.array([m.item() for m in mean_y])

            # Logging
            if step % self.net_config.logging_frequency == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss, val_mean_y = self.loss_fn(
                        (valid_data[0], valid_data[1], mean_y_train)
                    )
                val_loss = val_loss.item()
                y_init_val = self.y_init.item()
                elapsed = time.time() - start_time

                # Extract diagnostics
                diag_info = ""
                z_max_val = 0.0
                if hasattr(self.model, "_last_mean_spreads") and self.model._last_mean_spreads:
                    avg_spread = np.mean(self.model._last_mean_spreads)
                    z_max_val = self.model._last_z_max_overall
                    diag_info = "  spread: %.4f  max|Z|: %.4f" % (avg_spread, z_max_val)

                training_history.append(
                    [step, val_loss, y_init_val, z_max_val, elapsed]
                )
                if self.net_config.verbose:
                    logging.info(
                        "step: %5u,    loss: %.4e, Y0: %.4e,%s    elapsed time: %3u"
                        % (step, val_loss, y_init_val, diag_info, elapsed)
                    )

        # Final validation
        valid_data = self.bsde.sample(self.net_config.valid_size * 10)
        self.model.eval()
        with torch.no_grad():
            val_loss, _ = self.loss_fn(
                (valid_data[0], valid_data[1], mean_y_train)
            )

        print("\n=== Cont-Xiong LOB Results ===")
        print("Final Y0 (value function at t=0): %.6f" % self.y_init.item())
        print("Final loss: %.6e" % val_loss.item())
        if hasattr(self.model, "_last_mean_spreads") and self.model._last_mean_spreads:
            print("Mean bid-ask spread: %.4f" % np.mean(self.model._last_mean_spreads))
            print("Equilibrium spread (2/alpha): %.4f" % (2.0 / self.bsde.alpha))

        # Save model weights for reloading without retraining
        if hasattr(self, '_save_path'):
            save_data = {
                "model_state": self.model.state_dict(),
                "config_eqn": {
                    "sigma_s": self.bsde.sigma_s,
                    "lambda_a": self.bsde.lambda_a,
                    "lambda_b": self.bsde.lambda_b,
                    "alpha": self.bsde.alpha,
                    "phi": self.bsde.phi,
                    "discount_rate": self.bsde.discount_rate,
                },
                "y0": self.y_init.item(),
                "final_loss": val_loss.item(),
                "mean_field_history": mean_field_history,
            }
            torch.save(save_data, self._save_path)
            logging.info("Model saved to %s" % self._save_path)

        return {
            "history": np.array(training_history),
            "y0": self.y_init.item(),
            "final_loss": val_loss.item(),
            "mean_field_history": mean_field_history,
        }


# =====================================================================
# Cont-Xiong LOB Jump-Diffusion Model and Solver (Option B: FBSDEJ)
# =====================================================================


class ContXiongLOBJumpModel(nn.Module):
    """DeepBSDE model for jump-diffusion LOB (FBSDEJ).

    Key difference from continuous version: subnets output 3 values
    per timestep — (Z, U+, U-) instead of just (Z^S, Z^q).

    Z: gradient w.r.t. price (Brownian component)
    U+: value jump on buy execution V(q+Δ) - V(q)
    U-: value jump on sell execution V(q-Δ) - V(q)
    """

    def __init__(self, config, bsde, device=None):
        super().__init__()
        self.eqn_config = config.eqn
        self.net_config = config.net
        self.bsde = bsde
        self.device = device or torch.device("cpu")
        dtype = torch.float64

        # Y_0: trainable scalar (value at t=0, q=0)
        self.y_init = nn.Parameter(
            torch.tensor(
                np.random.uniform(
                    low=self.net_config.y_init_range[0],
                    high=self.net_config.y_init_range[1],
                    size=[1],
                ),
                dtype=dtype,
            )
        )
        # Z_0, U+_0, U-_0: trainable initial values
        self.z_init = nn.Parameter(
            torch.tensor(np.random.uniform(-0.1, 0.1, size=[1, 1]), dtype=dtype)
        )
        self.u_plus_init = nn.Parameter(
            torch.tensor(np.random.uniform(-0.1, 0.1, size=[1, 1]), dtype=dtype)
        )
        self.u_minus_init = nn.Parameter(
            torch.tensor(np.random.uniform(-0.1, 0.1, size=[1, 1]), dtype=dtype)
        )

        # Subnets: input = (S, q) dim=2, output = (Z, U+, U-) dim=3
        self.subnet = nn.ModuleList(
            [
                FeedForwardSubNet(
                    self.net_config.num_hiddens, 2, 3, dtype=dtype,
                )
                for _ in range(bsde.num_time_interval - 1)
            ]
        )

    def forward(self, inputs):
        dw, x, jump_data = inputs
        dw = torch.as_tensor(dw, dtype=torch.float64, device=self.device)
        x = torch.as_tensor(x, dtype=torch.float64, device=self.device)
        n_ask = torch.as_tensor(jump_data["n_ask"], dtype=torch.float64, device=self.device)
        n_bid = torch.as_tensor(jump_data["n_bid"], dtype=torch.float64, device=self.device)

        dt = self.bsde.delta_t
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * dt
        batch_size = dw.shape[0]

        all_one = torch.ones(batch_size, 1, dtype=torch.float64, device=self.device)
        y = all_one * self.y_init
        z = all_one @ self.z_init          # [batch, 1]
        u_plus = all_one @ self.u_plus_init   # [batch, 1]
        u_minus = all_one @ self.u_minus_init  # [batch, 1]

        mean_y = [torch.mean(y)]
        loss_inter = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        for t in range(self.bsde.num_time_interval - 1):
            # Compute execution rates for compensated Poisson
            delta_a, delta_b = self.bsde._optimal_quotes_tf(z, u_plus, u_minus)
            rate_a = self.bsde.lambda_a * self.bsde._exec_prob_tf(delta_a)
            rate_b = self.bsde.lambda_b * self.bsde._exec_prob_tf(delta_b)

            # Compensated Poisson increments: dÑ = N_n - rate*dt
            dn_ask = n_ask[:, t:t+1] - rate_a * dt  # [batch, 1]
            dn_bid = n_bid[:, t:t+1] - rate_b * dt  # [batch, 1]

            # BSDE step with jumps:
            # Y_{n+1} = Y_n - f*dt + Z*dW + U-*dÑ^a + U+*dÑ^b
            y = (
                y
                - dt * self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z, u_plus, u_minus)
                + z * dw[:, 0:1, t]      # Brownian (price only, dim=1)
                + u_minus * dn_ask        # sell execution jump
                + u_plus * dn_bid         # buy execution jump
            )
            mean_y.append(torch.mean(y))

            # Predict (Z, U+, U-) at next timestep
            out = self.subnet[t](x[:, :, t + 1])
            z = out[:, 0:1]       # price gradient
            u_plus = out[:, 1:2]  # jump up value
            u_minus = out[:, 2:3]  # jump down value

        # Terminal step
        delta_a, delta_b = self.bsde._optimal_quotes_tf(z, u_plus, u_minus)
        rate_a = self.bsde.lambda_a * self.bsde._exec_prob_tf(delta_a)
        rate_b = self.bsde.lambda_b * self.bsde._exec_prob_tf(delta_b)
        dn_ask = n_ask[:, -1:] - rate_a * dt
        dn_bid = n_bid[:, -1:] - rate_b * dt

        y = (
            y
            - dt * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z, u_plus, u_minus)
            + z * dw[:, 0:1, -1]
            + u_minus * dn_ask
            + u_plus * dn_bid
        )

        return y, mean_y, loss_inter


class ContXiongLOBJumpSolver:
    """Trains ContXiongLOBJumpModel (FBSDEJ with Poisson jumps)."""

    def __init__(self, config, bsde, device=None):
        self.eqn_config = config.eqn
        self.net_config = config.net
        self.bsde = bsde
        self.device = device or torch.device("cpu")

        self.model = ContXiongLOBJumpModel(config, bsde, device=self.device)
        self.model.to(self.device)
        self.opt_config = self.net_config.opt_config1
        self.y_init = self.model.y_init

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.opt_config.lr_values[0], eps=1e-8
        )
        self.scheduler = make_piecewise_lr_scheduler(
            self.optimizer, self.opt_config.lr_boundaries, self.opt_config.lr_values
        )

    def loss_fn(self, inputs):
        y_terminal, mean_y, loss_inter = self.model(inputs)
        dw, x, jump_data = inputs
        x = torch.as_tensor(x, dtype=torch.float64, device=self.device)
        y_target = self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        delta = y_terminal - y_target
        mean_y.append(torch.mean(y_target))
        loss = loss_inter + torch.mean(
            torch.where(
                torch.abs(delta) < DELTA_CLIP,
                delta ** 2,
                2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2,
            )
        )
        return loss, mean_y

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)

        for step in range(self.opt_config.num_iterations + 1):
            if step % self.opt_config.freq_resample == 0:
                train_data = self.bsde.sample(self.net_config.batch_size)

            self.model.train()
            self.optimizer.zero_grad()
            loss, mean_y = self.loss_fn(train_data)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if step % self.net_config.logging_frequency == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss, _ = self.loss_fn(valid_data)
                val_loss = val_loss.item()
                y_init_val = self.y_init.item()
                elapsed = time.time() - start_time
                training_history.append([step, val_loss, y_init_val, elapsed])
                if self.net_config.verbose:
                    logging.info(
                        "step: %5u,    loss: %.4e, Y0: %.4e,    elapsed time: %3u"
                        % (step, val_loss, y_init_val, elapsed)
                    )

        # Final validation
        valid_data = self.bsde.sample(self.net_config.valid_size * 10)
        self.model.eval()
        with torch.no_grad():
            val_loss, _ = self.loss_fn(valid_data)

        print("\n=== Cont-Xiong LOB Jump Results ===")
        print("Final Y0: %.6f" % self.y_init.item())
        print("Final loss: %.6e" % val_loss.item())

        if hasattr(self, '_save_path'):
            torch.save({
                "model_state": self.model.state_dict(),
                "y0": self.y_init.item(),
                "final_loss": val_loss.item(),
            }, self._save_path)
            logging.info("Model saved to %s" % self._save_path)

        return {
            "history": np.array(training_history),
            "y0": self.y_init.item(),
            "final_loss": val_loss.item(),
        }


# =====================================================================
# McKean-Vlasov LOB Model and Solver
# =====================================================================


class ContXiongLOBMVModel(nn.Module):
    """Deep BSDE model with distribution-dependent mean-field coupling.

    Key difference from ContXiongLOBModel: subnets take
    (S_i, q_i, Phi(mu_t)) as input, where Phi is a law embedding
    computed from all particles at the current timestep.

    The law encoder is part of the model and its gradients flow
    through the BSDE loss (for DeepSets encoder).
    """

    def __init__(self, config, bsde, device=None):
        super().__init__()
        self.eqn_config = config.eqn
        self.net_config = config.net
        self.bsde = bsde
        self.device = device or torch.device("cpu")
        dtype = torch.float64

        # Law encoder (from the equation)
        self.law_encoder = bsde.law_encoder
        law_dim = bsde.law_embed_dim

        # Subnet input: own state (dim) + law embedding (law_dim)
        state_dim = bsde.dim  # 2 for base, 3 for adverse selection
        subnet_in = state_dim + law_dim
        # Jump models output extra U_a, U_b coefficients
        subnet_out = getattr(bsde, 'subnet_output_dim', state_dim)
        self._is_jump = hasattr(bsde, 'subnet_output_dim') and bsde.subnet_output_dim > state_dim

        self.y_init = nn.Parameter(
            torch.tensor(
                np.random.uniform(
                    low=self.net_config.y_init_range[0],
                    high=self.net_config.y_init_range[1],
                    size=[1],
                ),
                dtype=dtype,
            )
        )
        self.z_init = nn.Parameter(
            torch.tensor(
                np.random.uniform(low=-0.1, high=0.1, size=[1, subnet_out]),
                dtype=dtype,
            )
        )
        # Select subnet architecture based on config
        subnet_type = getattr(self.eqn_config, 'subnet_type', 'two_stream')
        self.subnet = nn.ModuleList(
            [
                create_mv_subnet(
                    subnet_type, self.net_config.num_hiddens,
                    state_dim, law_dim, subnet_out, dtype=dtype,
                )
                for _ in range(bsde.num_time_interval - 1)
            ]
        )

    def forward(self, inputs):
        dw, x, mean_y_input = inputs
        dw = torch.as_tensor(dw, dtype=torch.float64, device=self.device)
        x = torch.as_tensor(x, dtype=torch.float64, device=self.device)

        loss_inter = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        mean_y = []
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        batch_size = dw.shape[0]

        all_one = torch.ones(batch_size, 1, dtype=torch.float64, device=self.device)
        y = all_one * self.y_init
        z = all_one @ self.z_init  # [batch, 2]
        mean_y.append(torch.mean(y))

        # Diagnostics
        mean_spreads = []
        mean_inventories = []
        z_max_history = []
        law_embeddings = []

        for t in range(self.bsde.num_time_interval - 1):
            # Track statistics
            z_q = z[:, 1:2]
            delta_a, delta_b = self.bsde._optimal_quotes_tf(z_q)
            mean_spreads.append((torch.mean(delta_a) + torch.mean(delta_b)).item())
            mean_inventories.append(torch.mean(x[:, 1, t]).item())
            z_max_history.append(torch.max(torch.abs(z)).item())

            # === MV COUPLING: compute law embedding from batch particles ===
            particles_t = x[:, :, t]  # [batch, 2] at time t
            law_embed = self.law_encoder.encode(particles_t)  # [law_dim]
            # Broadcast to all agents: [batch, law_dim]
            law_embed_batch = law_embed.unsqueeze(0).expand(batch_size, -1)
            # Skip storing law embeddings to avoid CUDA memory issues in long runs

            # SET LAW EMBEDDING so f_tf uses it for competitive factor
            if hasattr(self.bsde, 'set_current_law_embed'):
                self.bsde.set_current_law_embed(law_embed)

            # For CX model: compute population average quotes from current Z
            if hasattr(self.bsde, 'set_population_quotes'):
                z_q_batch = z[:, 1:2]  # current Z_q for all agents
                da_batch, db_batch = self.bsde._optimal_quotes_tf(z_q_batch)
                self.bsde.set_population_quotes(
                    torch.mean(da_batch).detach(),
                    torch.mean(db_batch).detach(),
                )

            # For jump models: split z into diffusion part (Z) and jump part (U)
            if self._is_jump:
                z_diffusion = z[:, :self.bsde.dim]  # (Z_s, Z_q)
                z_jump = z[:, self.bsde.dim:]        # (U_a, U_b)
                if hasattr(self.bsde, 'set_current_jump_coeffs'):
                    self.bsde.set_current_jump_coeffs(z_jump)
            else:
                z_diffusion = z

            # BSDE step (f_tf now uses h(Phi(mu_t)) not old scalar proxy)
            y = (
                y
                - self.bsde.delta_t * self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z_diffusion)
                + torch.sum(z_diffusion * dw[:, :, t], dim=1, keepdim=True)
            )
            mean_y.append(torch.mean(y))

            # === Subnet input: own state + law embedding ===
            own_state = x[:, :, t + 1]  # [batch, 2]
            if getattr(self, 'h_only_mode', False):
                # H-only control: law enters generator (via set_current_law_embed)
                # but NOT the subnet — zero out law features
                zero_law = torch.zeros_like(law_embed_batch)
                subnet_input = torch.cat([own_state, zero_law], dim=1)
            else:
                subnet_input = torch.cat([own_state, law_embed_batch], dim=1)  # [batch, 2+law_dim]
            z = self.subnet[t](subnet_input) / self.bsde.dim

        # Terminal step — set law embedding for final f_tf call
        if self._is_jump:
            z_diffusion = z[:, :self.bsde.dim]
            z_jump = z[:, self.bsde.dim:]
            if hasattr(self.bsde, 'set_current_jump_coeffs'):
                self.bsde.set_current_jump_coeffs(z_jump)
        else:
            z_diffusion = z
        z_q = z_diffusion[:, 1:2]
        delta_a, delta_b = self.bsde._optimal_quotes_tf(z_q)
        mean_spreads.append((torch.mean(delta_a) + torch.mean(delta_b)).item())
        mean_inventories.append(torch.mean(x[:, 1, -2]).item())
        z_max_history.append(torch.max(torch.abs(z)).item())

        if hasattr(self.bsde, 'set_current_law_embed'):
            particles_terminal = x[:, :, -2]
            law_embed_terminal = self.law_encoder.encode(particles_terminal)
            self.bsde.set_current_law_embed(law_embed_terminal)

        y = (
            y
            - self.bsde.delta_t
            * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z_diffusion)
            + torch.sum(z_diffusion * dw[:, :, -1], dim=1, keepdim=True)
        )

        self._last_mean_spreads = mean_spreads
        self._last_mean_inventories = mean_inventories
        self._last_z_max = z_max_history
        self._last_z_max_overall = max(z_max_history) if z_max_history else 0.0
        self._last_law_embeddings = law_embeddings

        return y, mean_y, loss_inter


class ContXiongLOBMVSolver:
    """Trains ContXiongLOBMVModel with distribution-dependent coupling.

    Differences from ContXiongLOBSolver:
    - Law encoder parameters are part of the optimizer
    - Wasserstein distance tracked across fictitious play iterations
    - Full particle snapshots stored for diagnostics
    """

    def __init__(self, config, bsde, device=None):
        self.eqn_config = config.eqn
        self.net_config = config.net
        self.bsde = bsde
        self.device = device or torch.device("cpu")

        self.model = ContXiongLOBMVModel(config, bsde, device=self.device)
        self.model.to(self.device)
        self.opt_config = self.net_config.opt_config1
        self.y_init = self.model.y_init

        # Include competitive factor net parameters in optimizer
        all_params = list(self.model.parameters())
        if hasattr(bsde, 'competitive_factor_net'):
            bsde.competitive_factor_net.to(self.device)
            all_params += list(bsde.competitive_factor_net.parameters())

        self.optimizer = torch.optim.Adam(
            all_params, lr=self.opt_config.lr_values[0], eps=1e-8
        )
        self.scheduler = make_piecewise_lr_scheduler(
            self.optimizer, self.opt_config.lr_boundaries, self.opt_config.lr_values
        )

    def loss_fn(self, inputs):
        y_terminal, mean_y, loss_inter = self.model(inputs)
        dw, x, mean_y_input = inputs
        x = torch.as_tensor(x, dtype=torch.float64, device=self.device)
        y_target = self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        delta = y_terminal - y_target
        mean_y.append(torch.mean(y_target))
        loss = loss_inter + torch.mean(
            torch.where(
                torch.abs(delta) < DELTA_CLIP,
                delta ** 2,
                2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2,
            )
        )
        return loss, mean_y

    def train(self):
        start_time = time.time()
        training_history = []
        w2_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)
        mean_y_train = np.zeros(self.bsde.num_time_interval + 2)

        for step in range(self.opt_config.num_iterations + 1):
            # Mean-field update with Wasserstein tracking
            if (
                self.eqn_config.type == 3
                and step % self.opt_config.freq_update_drift == 0
                and step > 0
            ):
                if hasattr(self.model, "_last_mean_spreads"):
                    self.bsde.update_mean_field(
                        self.model._last_mean_spreads,
                        self.model._last_mean_inventories,
                    )

                    # W2 tracking on particle snapshots
                    last_x = valid_data[1]  # [batch, 2, T+1]
                    particles = last_x[:, :, -1].T if isinstance(last_x, np.ndarray) else last_x[:, :, -1].cpu().numpy()
                    if particles.shape[1] == 2:
                        self.bsde.update_mean_field_mv(particles)
                        w2 = self.bsde._w2_history[-1] if self.bsde._w2_history else 0.0
                        w2_history.append({"step": step, "w2": w2})
                        logging.info("  [MV update] step %d: W2=%.6f" % (step, w2))

                self.bsde.update_drift()
                valid_data = self.bsde.sample(self.net_config.valid_size, seed=1)

            # Resample
            if step % self.opt_config.freq_resample == 0:
                train_data = self.bsde.sample(self.net_config.batch_size)
                train_data = (train_data[0], train_data[1], mean_y_train)

            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            loss, mean_y = self.loss_fn(train_data)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Progress indicator
            if step > 0 and step % 100 == 0:
                import sys as _sys
                total = self.opt_config.num_iterations
                pct = step * 100 // total
                _sys.stderr.write(f"\r  [{pct:3d}%] step {step}/{total}")
                _sys.stderr.flush()

            mean_y_train = np.array([m.item() for m in mean_y])

            # Logging
            if step % self.net_config.logging_frequency == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss, val_mean_y = self.loss_fn(
                        (valid_data[0], valid_data[1], mean_y_train)
                    )
                val_loss = val_loss.item()
                y_init_val = self.y_init.item()
                elapsed = time.time() - start_time

                diag_info = ""
                z_max_val = 0.0
                if hasattr(self.model, "_last_mean_spreads") and self.model._last_mean_spreads:
                    avg_spread = np.mean(self.model._last_mean_spreads)
                    z_max_val = self.model._last_z_max_overall
                    diag_info = "  spread: %.4f  max|Z|: %.4f" % (avg_spread, z_max_val)

                training_history.append(
                    [step, val_loss, y_init_val, z_max_val, elapsed]
                )
                if self.net_config.verbose:
                    logging.info(
                        "step: %5u,    loss: %.4e, Y0: %.4e,%s    elapsed time: %3u"
                        % (step, val_loss, y_init_val, diag_info, elapsed)
                    )

        # Final validation
        valid_data = self.bsde.sample(self.net_config.valid_size * 10)
        self.model.eval()
        with torch.no_grad():
            val_loss, _ = self.loss_fn(
                (valid_data[0], valid_data[1], mean_y_train)
            )

        print("\n=== Cont-Xiong LOB MV Results ===")
        print("Final Y0: %.6f" % self.y_init.item())
        print("Final loss: %.6e" % val_loss.item())
        print("Law encoder: %s (embed_dim=%d)" % (
            self.eqn_config.law_encoder_type, self.bsde.law_embed_dim))
        if w2_history:
            print("Final W2 residual: %.6f" % w2_history[-1]["w2"])
        if hasattr(self.model, "_last_mean_spreads") and self.model._last_mean_spreads:
            print("Mean bid-ask spread: %.4f" % np.mean(self.model._last_mean_spreads))

        if hasattr(self, '_save_path'):
            torch.save({
                "model_state": self.model.state_dict(),
                "y0": self.y_init.item(),
                "final_loss": val_loss.item(),
                "w2_history": w2_history,
                "law_encoder_type": self.eqn_config.law_encoder_type,
            }, self._save_path)
            logging.info("Model saved to %s" % self._save_path)

        return {
            "history": np.array(training_history),
            "y0": self.y_init.item(),
            "final_loss": val_loss.item(),
            "w2_history": w2_history,
        }

    def compute_diagnostics(self):
        """Compute stability diagnostics on the trained model.

        Returns:
            lipschitz_z: max |dZ/dq| over q grid (Lipschitz constant of Z)
            z_profile: Z_q values at each q point
            path_stats: terminal state variance from sampled paths
            grad_norm: total gradient norm of model parameters
        """
        self.model.eval()
        device = self.device
        bsde = self.bsde

        # 1. Lipschitz(Z) — evaluate Z_q on a q grid
        q_vals = np.linspace(-4, 4, 50)
        pop = np.stack([np.full(256, 100.0),
                        np.clip(np.random.normal(0, 1.0, 256), -10, 10)], axis=1)
        particles = torch.tensor(pop, dtype=torch.float64, device=device)
        law_embed = self.model.law_encoder.encode(particles)
        leb = law_embed.unsqueeze(0)

        zqs = []
        with torch.no_grad():
            for q in q_vals:
                agent = torch.tensor([[100.0, q]], dtype=torch.float64, device=device)
                si = torch.cat([agent, leb], dim=1)
                z = self.model.subnet[0](si) / bsde.dim
                zqs.append(z[:, 1].item())
        zqs = np.array(zqs)
        dz = np.abs(np.diff(zqs))
        dq = np.abs(np.diff(q_vals))
        lip_z = float(np.max(dz / dq)) if len(dq) > 0 else 0.0

        # 2. Path variance
        dw, x = bsde.sample(512)
        var_q = float(np.var(x[:, 1, -1]))
        var_s = float(np.var(x[:, 0, -1]))
        max_q = float(np.max(np.abs(x[:, 1, -1])))

        # 3. Gradient norm (from last training step)
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5

        return {
            "lipschitz_z": lip_z,
            "z_max": float(np.max(np.abs(zqs))),
            "var_q_T": var_q,
            "var_s_T": var_s,
            "max_abs_q_T": max_q,
            "grad_norm": grad_norm,
            "z_profile": zqs.tolist(),
            "q_grid": q_vals.tolist(),
        }


# =====================================================================
# Fictitious Play Solver
# =====================================================================


class FictitiousPlaySolver:
    """Outer fictitious play loop for mean-field equilibrium.

    Following Han, Hu, Long (2022):
    1. Initialize population distribution (from bsde.sample() proxy)
    2. Train inner BSDE solver given current population
    3. Simulate population under learned policy
    4. Update population (damped)
    5. Check W2 convergence
    """

    def __init__(self, config, bsde, device=None,
                 outer_iterations=20, inner_iterations=2000,
                 w2_threshold=0.01, damping_alpha=0.5,
                 n_sim_agents=256, warm_start=True):
        self.config = config
        self.bsde = bsde
        self.device = device or torch.device("cpu")
        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
        self.w2_threshold = w2_threshold
        self.damping_alpha = damping_alpha
        self.n_sim_agents = n_sim_agents
        self.warm_start = warm_start

        # Current population: terminal inventories from proxy simulation
        dw, x = bsde.sample(n_sim_agents)
        # x is [batch, dim, T+1] — terminal inventory
        self.current_q = x[:, 1, -1].copy()  # [n_sim_agents]

    def _simulate_population(self, model, bsde):
        """Forward-simulate SDE using trained subnet Z predictions.

        Unlike bsde.sample() which uses proxy z_q = -2*phi*q,
        this uses the actual learned Z from the neural network.
        """
        model.eval()
        device = self.device
        n = self.n_sim_agents
        dt = bsde.delta_t
        n_steps = bsde.num_time_interval

        # Initialize
        S = np.full(n, bsde.s_init)
        q = self.current_q.copy()  # Start from current population

        with torch.no_grad():
            for t in range(n_steps):
                # Build particles tensor
                particles = np.stack([S, q], axis=1)  # [n, 2]
                particles_t = torch.tensor(particles, dtype=torch.float64, device=device)

                # Law embedding
                law_embed = model.law_encoder.encode(particles_t)
                law_batch = law_embed.unsqueeze(0).expand(n, -1)

                # Set law embed for h computation
                if hasattr(bsde, 'set_current_law_embed'):
                    bsde.set_current_law_embed(law_embed)
                h = bsde.compute_competitive_factor(law_embed).item()

                # Subnet input: [n, state_dim + law_dim]
                state_t = torch.tensor(particles, dtype=torch.float64, device=device)
                if getattr(model, 'h_only_mode', False):
                    zero_law = torch.zeros(n, law_batch.shape[1], dtype=torch.float64, device=device)
                    si = torch.cat([state_t, zero_law], dim=1)
                else:
                    si = torch.cat([state_t, law_batch], dim=1)

                # Get Z from subnet (use first subnet — they share architecture)
                z = model.subnet[min(t, len(model.subnet) - 1)](si) / bsde.dim
                z_q = z[:, 1].cpu().numpy()  # [n]

                # Optimal quotes
                sigma_q = bsde._sigma_q_equilibrium()
                p = z_q / sigma_q
                delta_a = 1.0 / bsde.alpha + p
                delta_b = 1.0 / bsde.alpha - p

                # Execution rates
                f_a = bsde._exec_prob_np(delta_a, h) * bsde.lambda_a
                f_b = bsde._exec_prob_np(delta_b, h) * bsde.lambda_b

                # SDE step
                dW_S = np.random.normal(0, np.sqrt(dt), n)
                dW_q = np.random.normal(0, np.sqrt(dt), n)

                S = S + bsde.sigma_s * dW_S
                inv_drift = (f_b - f_a) * dt
                inv_diff = np.sqrt(np.maximum(f_b + f_a, 1e-8)) * dW_q
                q = q + inv_drift + inv_diff
                q = np.clip(q, -bsde.q_max, bsde.q_max)

        return q  # Terminal inventories

    def _compute_w2(self, q_old, q_new):
        """Sorted 1D Wasserstein-2 distance."""
        q1 = np.sort(q_old)
        q2 = np.sort(q_new)
        n = min(len(q1), len(q2))
        return float(np.sqrt(np.mean((q1[:n] - q2[:n]) ** 2)))

    def train(self):
        """Run the full fictitious play loop."""
        history = []
        solver = None

        for k in range(self.outer_iterations):
            print(f"\n--- FP outer iteration {k+1}/{self.outer_iterations} ---")

            # Create or warm-start inner solver
            if solver is None or not self.warm_start:
                solver = ContXiongLOBMVSolver(self.config, self.bsde, device=self.device)
            # Set inner iteration count
            solver.opt_config.num_iterations = self.inner_iterations

            # Train inner solver
            result = solver.train()
            y0 = result["y0"]
            loss = result["final_loss"]

            # Evaluate h at current population
            model = solver.model
            model.eval()
            pop = np.stack([np.full(self.n_sim_agents, self.bsde.s_init), self.current_q], axis=1)
            particles = torch.tensor(pop, dtype=torch.float64, device=self.device)
            law_embed = model.law_encoder.encode(particles)
            with torch.no_grad():
                h = self.bsde.compute_competitive_factor(law_embed).item()

            # Simulate population under learned policy
            q_simulated = self._simulate_population(model, self.bsde)

            # Damped update
            q_new = self.damping_alpha * q_simulated + (1.0 - self.damping_alpha) * self.current_q

            # W2 distance
            w2 = self._compute_w2(self.current_q, q_new)

            # Store
            entry = {
                "iteration": k + 1,
                "y0": y0, "loss": loss, "h": h,
                "w2": w2,
                "q_mean": float(np.mean(q_new)),
                "q_std": float(np.std(q_new)),
            }
            history.append(entry)
            print(f"  Y0={y0:.4f}, h={h:.4f}, W2={w2:.6f}, q_std={np.std(q_new):.4f}")

            # Update population
            self.current_q = q_new

            # Check convergence
            if w2 < self.w2_threshold:
                print(f"\n  CONVERGED at iteration {k+1} (W2={w2:.6f} < {self.w2_threshold})")
                break

        return {
            "history": history,
            "converged": history[-1]["w2"] < self.w2_threshold if history else False,
            "final_y0": history[-1]["y0"] if history else None,
            "final_w2": history[-1]["w2"] if history else None,
        }

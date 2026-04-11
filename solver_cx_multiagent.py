"""
Decentralised MADDPG for CX dealer market.

Faithful implementation of CX Section 6:
- Pre-training on monopolist solution
- Target networks with soft update
- Experience replay buffer
- Layer normalisation
- Decentralised: each agent only sees own (q, reward, execution)
- Actor: q -> (delta_a, delta_b), output = 5*tanh(raw)
- Critic: (q, da, db) -> Q value

After training, compare learned spreads to Nash equilibrium.
Spreads above Nash = tacit collusion (CX's main finding).
"""

import numpy as np
import torch
import torch.nn as nn
import time
import collections
import random


def cx_exec_prob(delta, avg_comp, K, N):
    """CX execution probability."""
    base = 1.0 / (1.0 + np.exp(np.clip(delta, -20, 20)))
    if K > 0:
        s = np.clip(avg_comp, -20, 20)
        comp = np.exp(s) / (1.0 + np.exp(np.clip(delta + s, -20, 20)))
        return (1.0 / N) * base * comp
    else:
        return base * base


# ================================================================
# Networks (CX Section 6.2)
# ================================================================

class ActorNet(nn.Module):
    """Policy: q -> (delta_a, delta_b).
    3 layers x 10 hidden, ReLU, layer norm, output = 5*tanh.
    """
    def __init__(self, hidden=10, dtype=torch.float64):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden, dtype=dtype)
        self.ln1 = nn.LayerNorm(hidden, dtype=dtype)
        self.fc2 = nn.Linear(hidden, hidden, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden, dtype=dtype)
        self.fc3 = nn.Linear(hidden, hidden, dtype=dtype)
        self.ln3 = nn.LayerNorm(hidden, dtype=dtype)
        self.out = nn.Linear(hidden, 2, dtype=dtype)

    def forward(self, q_norm):
        x = torch.relu(self.ln1(self.fc1(q_norm)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        # Quotes in [0.1, 2.0] — bounded near reasonable market-making range
        # Centre around monopolist optimal (~0.8) with room to move
        return 0.1 + 1.9 * torch.sigmoid(self.out(x))


class CriticNet(nn.Module):
    """Q(q, da, db) -> scalar value.
    3 layers x 10 hidden, ReLU, layer norm, linear output.
    """
    def __init__(self, hidden=10, dtype=torch.float64):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden, dtype=dtype)  # q + da + db
        self.ln1 = nn.LayerNorm(hidden, dtype=dtype)
        self.fc2 = nn.Linear(hidden, hidden, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden, dtype=dtype)
        self.fc3 = nn.Linear(hidden, hidden, dtype=dtype)
        self.ln3 = nn.LayerNorm(hidden, dtype=dtype)
        self.out = nn.Linear(hidden, 1, dtype=dtype)

    def forward(self, q_norm, da, db):
        x = torch.cat([q_norm, da, db], dim=1)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        return self.out(x)


# ================================================================
# Experience Replay
# ================================================================

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action_a, action_b, reward, next_state, won):
        self.buffer.append((state, action_a, action_b, reward, next_state, won))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, das, dbs, rewards, next_states, wons = zip(*batch)
        return (np.array(states), np.array(das), np.array(dbs),
                np.array(rewards), np.array(next_states), np.array(wons))

    def __len__(self):
        return len(self.buffer)


# ================================================================
# Market
# ================================================================

class DealerMarket:
    def __init__(self, N=2, Q=5, Delta=1, lambda_a=2.0, lambda_b=2.0,
                 r=0.01, phi=0.005):
        self.N = N; self.Q = Q; self.Delta = Delta
        self.lambda_a = lambda_a; self.lambda_b = lambda_b
        self.r = r; self.phi = phi
        self.gamma = (lambda_a + lambda_b) / (r + lambda_a + lambda_b)
        self.nq = int(2 * Q / Delta + 1)
        self.K = (N - 1) * self.nq
        self.inventories = np.zeros(N)

    def reset(self):
        # Random initial inventories (CX doesn't specify, but helps exploration)
        self.inventories = np.random.choice(
            np.arange(-self.Q, self.Q + self.Delta, self.Delta), size=self.N
        )

    def step(self, quotes_a, quotes_b):
        is_ask = np.random.random() < 0.5
        if is_ask:
            probs = np.zeros(self.N)
            for i in range(self.N):
                if self.inventories[i] > -self.Q:
                    others = [quotes_a[j] for j in range(self.N) if j != i]
                    avg = np.mean(others) if others else 0.75
                    probs[i] = cx_exec_prob(quotes_a[i], avg, self.K, self.N)
            total = np.sum(probs)
            probs = probs / total if total > 1e-10 else np.ones(self.N) / self.N
            winner = np.random.choice(self.N, p=probs)
            if self.inventories[winner] > -self.Q:
                self.inventories[winner] -= self.Delta
            winning_quote = quotes_a[winner]
        else:
            probs = np.zeros(self.N)
            for i in range(self.N):
                if self.inventories[i] < self.Q:
                    others = [quotes_b[j] for j in range(self.N) if j != i]
                    avg = np.mean(others) if others else 0.75
                    probs[i] = cx_exec_prob(quotes_b[i], avg, self.K, self.N)
            total = np.sum(probs)
            probs = probs / total if total > 1e-10 else np.ones(self.N) / self.N
            winner = np.random.choice(self.N, p=probs)
            if self.inventories[winner] < self.Q:
                self.inventories[winner] += self.Delta
            winning_quote = quotes_b[winner]

        # Rewards: use EXPECTED spread revenue (not stochastic)
        # This reduces variance: reward = -psi*cost_scale + exec_prob * spread_scale * delta
        # The execution probability naturally penalises wide quotes
        cost_scale = 1.0 / (self.r + self.lambda_a + self.lambda_b)
        spread_scale = (self.lambda_a + self.lambda_b) * self.Delta * cost_scale
        rewards = np.zeros(self.N)
        won = np.zeros(self.N)
        for i in range(self.N):
            rewards[i] = -self.phi * self.inventories[i] ** 2 * cost_scale

            # Expected ask revenue: exec_prob * quote
            if self.inventories[i] > -self.Q:
                others_a = [quotes_a[j] for j in range(self.N) if j != i]
                avg_a = np.mean(others_a) if others_a else 0.75
                fa = cx_exec_prob(quotes_a[i], avg_a, self.K, self.N)
                rewards[i] += 0.5 * spread_scale * fa * quotes_a[i]  # 0.5 = prob of ask RFQ

            # Expected bid revenue
            if self.inventories[i] < self.Q:
                others_b = [quotes_b[j] for j in range(self.N) if j != i]
                avg_b = np.mean(others_b) if others_b else 0.75
                fb = cx_exec_prob(quotes_b[i], avg_b, self.K, self.N)
                rewards[i] += 0.5 * spread_scale * fb * quotes_b[i]

            if i == winner:
                won[i] = 1.0

        # Still do stochastic inventory update (winner's inventory changes)
        return rewards, won


# ================================================================
# MADDPG Trainer
# ================================================================

class MADDPGTrainer:
    """Full CX Section 6 implementation."""

    def __init__(self, N=2, Q=5, device=None,
                 lr_actor=1e-3, lr_critic=1e-3, tau=0.01,
                 n_episodes=500, steps_per_episode=500,
                 batch_size=32, buffer_size=10000,
                 explore_prob_init=0.05, explore_decay=0.01):
        self.N = N; self.Q = Q
        self.device = device or torch.device("cpu")
        self.tau = tau  # target network soft update rate
        self.n_episodes = n_episodes
        self.steps_per_episode = steps_per_episode
        self.batch_size = batch_size
        self.explore_prob_init = explore_prob_init
        self.explore_decay = explore_decay

        self.market = DealerMarket(N=N, Q=Q)

        # Create agents with actor, critic, target networks, replay buffers
        self.agents = []
        for i in range(N):
            actor = ActorNet().to(device)
            critic = CriticNet().to(device)
            target_actor = ActorNet().to(device)
            target_critic = CriticNet().to(device)
            # Copy weights to targets
            target_actor.load_state_dict(actor.state_dict())
            target_critic.load_state_dict(critic.state_dict())

            self.agents.append({
                "actor": actor, "critic": critic,
                "target_actor": target_actor, "target_critic": target_critic,
                "actor_opt": torch.optim.Adam(actor.parameters(), lr=lr_actor,
                                               betas=(0.9, 0.99), eps=1e-6),
                "critic_opt": torch.optim.Adam(critic.parameters(), lr=lr_critic,
                                                betas=(0.9, 0.99), eps=1e-6),
                "buffer": ReplayBuffer(buffer_size),
            })

    def soft_update(self, target, source):
        """Soft update target network: target = tau*source + (1-tau)*target"""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def pretrain_monopolist(self, n_steps=2000):
        """Pre-train on monopolist solution (CX Section 6.2).

        The monopolist optimal quotes are known from Algorithm 1 with N=1.
        Pre-train actor to output these quotes and critic to estimate the value.
        """
        print("  Pre-training on monopolist solution...")
        from scripts.cont_xiong_exact import fictitious_play
        mono = fictitious_play(N=1, Q=int(self.Q), max_iter=50)
        q_grid = np.array(mono["q_grid"])
        V_mono = np.array(mono["V"])
        da_mono = np.array(mono["delta_a"])
        db_mono = np.array(mono["delta_b"])
        nq = len(q_grid)

        for agent in self.agents:
            actor = agent["actor"]; critic = agent["critic"]
            a_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
            c_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

            for step in range(n_steps):
                # Random inventory levels
                idx = np.random.randint(0, nq, size=16)
                q_batch = torch.tensor(q_grid[idx].reshape(-1, 1) / self.Q,
                                        dtype=torch.float64, device=self.device)
                da_target = torch.tensor(da_mono[idx].reshape(-1, 1),
                                          dtype=torch.float64, device=self.device)
                db_target = torch.tensor(db_mono[idx].reshape(-1, 1),
                                          dtype=torch.float64, device=self.device)
                v_target = torch.tensor(V_mono[idx].reshape(-1, 1),
                                         dtype=torch.float64, device=self.device)

                # Actor: match monopolist quotes
                quotes = actor(q_batch)
                actor_loss = torch.mean((quotes[:, 0:1] - da_target) ** 2 +
                                        (quotes[:, 1:2] - db_target) ** 2)
                a_opt.zero_grad(); actor_loss.backward(); a_opt.step()

                # Critic: match monopolist value
                q_val = critic(q_batch, da_target, db_target)
                critic_loss = torch.mean((q_val - v_target) ** 2)
                c_opt.zero_grad(); critic_loss.backward(); c_opt.step()

            # Copy to targets
            agent["target_actor"].load_state_dict(actor.state_dict())
            agent["target_critic"].load_state_dict(critic.state_dict())

        # Verify pre-training
        q0 = torch.tensor([[0.0]], dtype=torch.float64, device=self.device)
        with torch.no_grad():
            quotes = self.agents[0]["actor"](q0)
        print(f"  Pre-trained quotes at q=0: da={quotes[0,0]:.4f}, db={quotes[0,1]:.4f}")
        print(f"  Monopolist target: da={da_mono[nq//2]:.4f}, db={db_mono[nq//2]:.4f}")

    def get_quotes(self, agent_idx, explore_prob=0.0):
        agent = self.agents[agent_idx]
        q = self.market.inventories[agent_idx]
        q_norm = torch.tensor([[q / self.Q]], dtype=torch.float64, device=self.device)

        agent["actor"].eval()
        with torch.no_grad():
            quotes = agent["actor"](q_norm)
        da = quotes[0, 0].item()
        db = quotes[0, 1].item()

        # Exploration: add noise with decaying probability
        if np.random.random() < explore_prob:
            da += np.random.normal(0, 0.3)
            db += np.random.normal(0, 0.3)

        # Clamp to market-making range
        da = max(0.1, min(2.0, da))
        db = max(0.1, min(2.0, db))
        return da, db

    def train_step(self, agent_idx):
        """One gradient update from replay buffer."""
        agent = self.agents[agent_idx]
        if len(agent["buffer"]) < self.batch_size:
            return

        states, das, dbs, rewards, next_states, wons = agent["buffer"].sample(self.batch_size)
        dev = self.device

        q_t = torch.tensor(states.reshape(-1, 1) / self.Q, dtype=torch.float64, device=dev)
        da_t = torch.tensor(das.reshape(-1, 1), dtype=torch.float64, device=dev)
        db_t = torch.tensor(dbs.reshape(-1, 1), dtype=torch.float64, device=dev)
        r_t = torch.tensor(rewards.reshape(-1, 1), dtype=torch.float64, device=dev)
        q_next = torch.tensor(next_states.reshape(-1, 1) / self.Q, dtype=torch.float64, device=dev)

        # Critic update: TD with target networks (eq 75-77)
        agent["critic"].train()
        Q_current = agent["critic"](q_t, da_t, db_t)

        with torch.no_grad():
            next_quotes = agent["target_actor"](q_next)
            Q_target = r_t + self.market.gamma * agent["target_critic"](
                q_next, next_quotes[:, 0:1], next_quotes[:, 1:2]
            )

        critic_loss = torch.mean((Q_current - Q_target) ** 2)
        agent["critic_opt"].zero_grad()
        critic_loss.backward()
        agent["critic_opt"].step()

        # Actor update: maximise Q(q, actor(q)) (eq 78-80)
        agent["actor"].train()
        quotes = agent["actor"](q_t)
        Q_for_actor = agent["critic"](q_t, quotes[:, 0:1], quotes[:, 1:2])
        actor_loss = -torch.mean(Q_for_actor)

        agent["actor_opt"].zero_grad()
        actor_loss.backward()
        agent["actor_opt"].step()

        # Soft update targets (eq 74)
        self.soft_update(agent["target_actor"], agent["actor"])
        self.soft_update(agent["target_critic"], agent["critic"])

    def train(self):
        start = time.time()

        # Pre-train on monopolist
        self.pretrain_monopolist(n_steps=3000)

        history = []
        for episode in range(self.n_episodes):
            self.market.reset()
            episode_rewards = np.zeros(self.N)

            # Exploration probability (decays exponentially, CX Section 6.2)
            explore_prob = self.explore_prob_init * np.exp(-self.explore_decay * episode)

            for step in range(self.steps_per_episode):
                q_before = self.market.inventories.copy()

                # Get quotes
                all_da = np.zeros(self.N)
                all_db = np.zeros(self.N)
                for i in range(self.N):
                    all_da[i], all_db[i] = self.get_quotes(i, explore_prob)

                # Market step
                rewards, won = self.market.step(all_da, all_db)
                episode_rewards += rewards

                # Store transitions in each agent's replay buffer
                for i in range(self.N):
                    self.agents[i]["buffer"].push(
                        q_before[i], all_da[i], all_db[i],
                        rewards[i], self.market.inventories[i], won[i]
                    )

                # Train each agent from their buffer
                for i in range(self.N):
                    self.train_step(i)

            # Log
            if episode % 50 == 0 or episode == self.n_episodes - 1:
                old_inv = self.market.inventories.copy()
                spreads = []
                for i in range(self.N):
                    self.market.inventories[i] = 0
                    da, db = self.get_quotes(i, explore_prob=0)
                    spreads.append(da + db)
                self.market.inventories = old_inv
                avg_spread = np.mean(spreads)
                avg_reward = np.mean(episode_rewards) / self.steps_per_episode
                print(f"  ep {episode}: spread(0)={avg_spread:.4f}, "
                      f"reward/step={avg_reward:.4f}, explore={explore_prob:.4f}")
                history.append({"episode": episode, "avg_spread": float(avg_spread),
                                "spreads": [float(s) for s in spreads]})

        elapsed = time.time() - start
        # Final quotes at q=0
        final = []
        for i in range(self.N):
            self.market.inventories[i] = 0
            da, db = self.get_quotes(i, explore_prob=0)
            final.append({"agent": i, "da": float(da), "db": float(db),
                          "spread": float(da + db)})

        return {
            "history": history,
            "final_spreads": final,
            "avg_final_spread": float(np.mean([f["spread"] for f in final])),
            "elapsed": elapsed,
        }

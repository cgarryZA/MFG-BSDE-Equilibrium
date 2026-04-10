"""
Multi-agent decentralised learning for CX model.

Reproduces CX Section 6: N agents each learn independently.
Each agent has a POLICY network that directly outputs quotes given inventory.
Learning is through policy gradient: the agent adjusts quotes to maximise
its own discounted reward, without knowing competitors' strategies.

Tacit collusion emerges because:
- When both agents happen to quote wide, both get high profit per trade
- The reward signal reinforces "wide = profitable" for both simultaneously
- They settle above Nash without any coordination

This is closer to CX's MADDPG approach than our previous Bellman-residual solver.
"""

import numpy as np
import torch
import torch.nn as nn
import time


class PolicyNet(nn.Module):
    """Agent's policy: maps inventory q → (delta_a, delta_b).

    Output is softplus to ensure positive quotes, shifted to allow reasonable range.
    """
    def __init__(self, hidden=32, dtype=torch.float64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, 2, dtype=dtype),  # (da, db)
        )

    def forward(self, q_norm):
        raw = self.net(q_norm)
        # Softplus + offset to get quotes in reasonable range [0.1, ~3]
        return 0.1 + torch.nn.functional.softplus(raw)


class CriticNet(nn.Module):
    """Critic: estimates V(q) for TD learning."""
    def __init__(self, hidden=32, dtype=torch.float64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=dtype),
        )

    def forward(self, q_norm):
        return self.net(q_norm)


def cx_exec_prob(delta, avg_comp_delta, K, N):
    """CX execution probability (numpy)."""
    base = 1.0 / (1.0 + np.exp(np.clip(delta, -20, 20)))
    if K > 0:
        s = np.clip(avg_comp_delta, -20, 20)
        comp = np.exp(s) / (1.0 + np.exp(np.clip(delta + s, -20, 20)))
        return (1.0 / N) * base * comp
    else:
        return base * base


class DealerMarket:
    """Simulates CX dealer market."""

    def __init__(self, N=2, Q=5, Delta=1, lambda_a=2.0, lambda_b=2.0,
                 r=0.01, phi=0.005):
        self.N = N
        self.Q = Q
        self.Delta = Delta
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.r = r
        self.phi = phi
        self.gamma = (lambda_a + lambda_b) / (r + lambda_a + lambda_b)
        self.nq = int(2 * Q / Delta + 1)
        self.K = (N - 1) * self.nq
        self.inventories = np.zeros(N)

    def reset(self):
        self.inventories = np.zeros(self.N)

    def step(self, quotes_a, quotes_b):
        """One market step. Returns rewards, side, winner."""
        is_ask = np.random.random() < 0.5  # equal ask/bid probability

        if is_ask:
            probs = np.zeros(self.N)
            for i in range(self.N):
                if self.inventories[i] > -self.Q:
                    others = [quotes_a[j] for j in range(self.N) if j != i]
                    avg_comp = np.mean(others) if others else 0.75
                    probs[i] = cx_exec_prob(quotes_a[i], avg_comp, self.K, self.N)
            total = np.sum(probs)
            if total > 1e-10:
                probs /= total
            else:
                probs = np.ones(self.N) / self.N
            winner = np.random.choice(self.N, p=probs)
            if self.inventories[winner] > -self.Q:
                self.inventories[winner] -= self.Delta
            winning_quote = quotes_a[winner]
        else:
            probs = np.zeros(self.N)
            for i in range(self.N):
                if self.inventories[i] < self.Q:
                    others = [quotes_b[j] for j in range(self.N) if j != i]
                    avg_comp = np.mean(others) if others else 0.75
                    probs[i] = cx_exec_prob(quotes_b[i], avg_comp, self.K, self.N)
            total = np.sum(probs)
            if total > 1e-10:
                probs /= total
            else:
                probs = np.ones(self.N) / self.N
            winner = np.random.choice(self.N, p=probs)
            if self.inventories[winner] < self.Q:
                self.inventories[winner] += self.Delta
            winning_quote = quotes_b[winner]

        # Rewards: only winner gets spread revenue, all pay inventory cost
        cost_scale = 1.0 / (self.r + self.lambda_a + self.lambda_b)
        spread_scale = (self.lambda_a + self.lambda_b) * self.Delta * cost_scale
        rewards = np.zeros(self.N)
        for i in range(self.N):
            rewards[i] = -self.phi * self.inventories[i] ** 2 * cost_scale
            if i == winner:
                rewards[i] += spread_scale * winning_quote

        return rewards, winner


class MultiAgentTrainer:
    """Train N agents with actor-critic in the CX market.

    Each agent has:
    - PolicyNet (actor): inventory → (delta_a, delta_b)
    - CriticNet: inventory → V(q)

    Updated via:
    - Critic: TD learning from rewards
    - Actor: policy gradient to maximise expected reward
    """

    def __init__(self, N=2, Q=5, device=None, lr_actor=5e-4, lr_critic=1e-3,
                 n_episodes=500, steps_per_episode=500, exploration_std=0.2):
        self.N = N
        self.Q = Q
        self.device = device or torch.device("cpu")
        self.n_episodes = n_episodes
        self.steps_per_episode = steps_per_episode
        self.exploration_std = exploration_std

        self.market = DealerMarket(N=N, Q=Q)

        self.agents = []
        for i in range(N):
            actor = PolicyNet(hidden=32).to(self.device)
            critic = CriticNet(hidden=32).to(self.device)
            self.agents.append({
                "actor": actor,
                "critic": critic,
                "actor_opt": torch.optim.Adam(actor.parameters(), lr=lr_actor),
                "critic_opt": torch.optim.Adam(critic.parameters(), lr=lr_critic),
            })

    def get_quotes(self, agent_idx, explore=True):
        """Get quotes from agent's policy network."""
        agent = self.agents[agent_idx]
        q = self.market.inventories[agent_idx]
        q_norm = torch.tensor([[q / self.Q]], dtype=torch.float64, device=self.device)

        agent["actor"].eval()
        with torch.no_grad():
            quotes = agent["actor"](q_norm)  # [1, 2] = (da, db)
            da = quotes[0, 0].item()
            db = quotes[0, 1].item()

        if explore:
            # Add exploration noise (decays over episodes)
            da += np.random.normal(0, self.exploration_std)
            db += np.random.normal(0, self.exploration_std)
            da = max(0.01, da)
            db = max(0.01, db)

        return da, db

    def update_agent(self, agent_idx, q_before, reward, q_after):
        """Update critic (TD) and actor (policy gradient) for one agent."""
        agent = self.agents[agent_idx]
        device = self.device

        q_norm = torch.tensor([[q_before / self.Q]], dtype=torch.float64, device=device)
        q_norm_after = torch.tensor([[q_after / self.Q]], dtype=torch.float64, device=device)

        # Critic update (TD learning)
        agent["critic"].train()
        V = agent["critic"](q_norm)
        with torch.no_grad():
            V_next = agent["critic"](q_norm_after)
        td_target = reward + self.market.gamma * V_next
        critic_loss = (V - td_target) ** 2

        agent["critic_opt"].zero_grad()
        critic_loss.backward()
        agent["critic_opt"].step()

        # Actor update (policy gradient)
        # The actor should output quotes that maximise the critic's V
        agent["actor"].train()
        quotes = agent["actor"](q_norm)  # [1, 2]
        # Use critic value at current state as the objective
        V_for_actor = agent["critic"](q_norm)
        actor_loss = -V_for_actor  # maximise V

        agent["actor_opt"].zero_grad()
        actor_loss.backward()
        agent["actor_opt"].step()

    def train(self):
        start = time.time()
        history = []

        for episode in range(self.n_episodes):
            self.market.reset()
            episode_rewards = np.zeros(self.N)

            # Decay exploration
            explore_std = self.exploration_std * np.exp(-episode / (self.n_episodes / 3))
            self.exploration_std = explore_std

            for step in range(self.steps_per_episode):
                # Each agent quotes
                all_da = np.zeros(self.N)
                all_db = np.zeros(self.N)
                q_before = self.market.inventories.copy()

                for i in range(self.N):
                    all_da[i], all_db[i] = self.get_quotes(i, explore=(episode < self.n_episodes * 0.8))

                # Market step
                rewards, winner = self.market.step(all_da, all_db)
                episode_rewards += rewards

                # Each agent updates from own reward
                for i in range(self.N):
                    self.update_agent(i, q_before[i], rewards[i],
                                     self.market.inventories[i])

            if episode % 50 == 0 or episode == self.n_episodes - 1:
                # Evaluate: get quotes at q=0 (no exploration)
                old_inv = self.market.inventories.copy()
                spreads = []
                for i in range(self.N):
                    self.market.inventories[i] = 0
                    da, db = self.get_quotes(i, explore=False)
                    spreads.append(da + db)
                self.market.inventories = old_inv

                avg_spread = np.mean(spreads)
                avg_reward = np.mean(episode_rewards) / self.steps_per_episode
                print(f"  ep {episode}: spread(0)={avg_spread:.4f}, "
                      f"reward/step={avg_reward:.4f}, explore={explore_std:.4f}")
                history.append({
                    "episode": episode, "avg_spread": float(avg_spread),
                    "spreads": [float(s) for s in spreads],
                    "avg_reward": float(avg_reward),
                })

        elapsed = time.time() - start

        # Final evaluation
        self.market.inventories = np.zeros(self.N)
        final_spreads = []
        for i in range(self.N):
            self.market.inventories[i] = 0
            da, db = self.get_quotes(i, explore=False)
            final_spreads.append({"agent": i, "da": float(da), "db": float(db),
                                  "spread": float(da + db)})

        return {
            "history": history,
            "final_spreads": final_spreads,
            "avg_final_spread": float(np.mean([s["spread"] for s in final_spreads])),
            "elapsed": elapsed,
            "N": self.N,
        }

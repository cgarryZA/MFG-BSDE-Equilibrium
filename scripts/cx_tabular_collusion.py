#!/usr/bin/env python
"""
Tabular Q-learning for CX tacit collusion detection.

Each agent maintains Q(q, action_idx) where:
- q in {-5,...,5} (11 states)
- action = (da, db) discretised to a grid of ~20 quote levels each

No neural networks. Pure Q-learning with epsilon-greedy exploration.
This is simpler than CX's MADDPG but tests the same hypothesis:
do decentralised learners converge above Nash?
"""

import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver_cx_multiagent import cx_exec_prob


def main():
    N = 2; Q = 5; Delta = 1
    lambda_a = 2.0; lambda_b = 2.0; r = 0.01; phi = 0.005
    gamma = (lambda_a + lambda_b) / (r + lambda_a + lambda_b)

    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)
    K = (N - 1) * nq

    # Discretise quote levels
    quote_levels = np.linspace(0.1, 2.0, 20)  # 20 possible quote values
    n_quotes = len(quote_levels)
    n_actions = n_quotes * n_quotes  # (da, db) pairs

    def action_to_quotes(action_idx):
        da_idx = action_idx // n_quotes
        db_idx = action_idx % n_quotes
        return quote_levels[da_idx], quote_levels[db_idx]

    # Q-tables: one per agent, Q[q_idx, action_idx]
    # Initialise optimistically to encourage exploration
    Q_tables = [np.ones((nq, n_actions)) * 5.0 for _ in range(N)]

    # Training parameters
    alpha = 0.1  # learning rate
    epsilon_init = 0.3
    epsilon_min = 0.01
    n_episodes = 1000
    steps_per_episode = 500

    # Market state
    inventories = np.zeros(N)

    history = []
    start = time.time()

    for episode in range(n_episodes):
        # Reset inventories
        inventories = np.random.choice(q_grid, size=N)
        episode_rewards = np.zeros(N)

        epsilon = max(epsilon_min, epsilon_init * (1 - episode / n_episodes))

        for step in range(steps_per_episode):
            q_indices = [int(inventories[i] + Q) for i in range(N)]

            # Each agent selects action (epsilon-greedy)
            actions = []
            all_da = np.zeros(N)
            all_db = np.zeros(N)
            for i in range(N):
                if np.random.random() < epsilon:
                    a = np.random.randint(n_actions)
                else:
                    a = np.argmax(Q_tables[i][q_indices[i]])
                actions.append(a)
                all_da[i], all_db[i] = action_to_quotes(a)

            # Determine RFQ side and winner
            is_ask = np.random.random() < 0.5
            if is_ask:
                probs = np.zeros(N)
                for i in range(N):
                    if inventories[i] > -Q:
                        others = [all_da[j] for j in range(N) if j != i]
                        avg = np.mean(others) if others else 0.75
                        probs[i] = cx_exec_prob(all_da[i], avg, K, N)
                total = np.sum(probs)
                probs = probs / total if total > 1e-10 else np.ones(N) / N
                winner = np.random.choice(N, p=probs)
                if inventories[winner] > -Q:
                    inventories[winner] -= Delta
            else:
                probs = np.zeros(N)
                for i in range(N):
                    if inventories[i] < Q:
                        others = [all_db[j] for j in range(N) if j != i]
                        avg = np.mean(others) if others else 0.75
                        probs[i] = cx_exec_prob(all_db[i], avg, K, N)
                total = np.sum(probs)
                probs = probs / total if total > 1e-10 else np.ones(N) / N
                winner = np.random.choice(N, p=probs)
                if inventories[winner] < Q:
                    inventories[winner] += Delta

            # Rewards
            cost_scale = 1.0 / (r + lambda_a + lambda_b)
            spread_scale = (lambda_a + lambda_b) * Delta * cost_scale
            rewards = np.zeros(N)
            for i in range(N):
                rewards[i] = -phi * inventories[i] ** 2 * cost_scale
                if i == winner:
                    if is_ask:
                        rewards[i] += spread_scale * all_da[i]
                    else:
                        rewards[i] += spread_scale * all_db[i]
            episode_rewards += rewards

            # Q-learning update for each agent
            new_q_indices = [int(np.clip(inventories[i], -Q, Q) + Q) for i in range(N)]
            for i in range(N):
                old_q = Q_tables[i][q_indices[i], actions[i]]
                next_max = np.max(Q_tables[i][new_q_indices[i]])
                td_target = rewards[i] + gamma * next_max
                Q_tables[i][q_indices[i], actions[i]] = (
                    old_q + alpha * (td_target - old_q)
                )

        # Log: extract greedy quotes at q=0
        if episode % 100 == 0 or episode == n_episodes - 1:
            spreads = []
            for i in range(N):
                best_a = np.argmax(Q_tables[i][Q])  # q=0 -> index Q
                da, db = action_to_quotes(best_a)
                spreads.append(da + db)
            avg_spread = np.mean(spreads)
            avg_reward = np.mean(episode_rewards) / steps_per_episode
            print(f"  ep {episode}: spread(0)={avg_spread:.4f}, "
                  f"eps={epsilon:.3f}, reward/step={avg_reward:.4f}")
            history.append({"episode": episode, "avg_spread": float(avg_spread),
                            "spreads": [float(s) for s in spreads],
                            "epsilon": epsilon})

    elapsed = time.time() - start

    # Final greedy quotes for each agent at each q
    print(f"\nFinal quote profiles (greedy):")
    print(f"  {'q':>4} {'A1_da':>7} {'A1_db':>7} {'A1_spr':>7} {'A2_da':>7} {'A2_db':>7} {'A2_spr':>7}")
    for j, q in enumerate(q_grid):
        quotes = []
        for i in range(N):
            best_a = np.argmax(Q_tables[i][j])
            da, db = action_to_quotes(best_a)
            quotes.append((da, db))
        print(f"  {q:4.0f} {quotes[0][0]:7.4f} {quotes[0][1]:7.4f} {sum(quotes[0]):7.4f} "
              f"{quotes[1][0]:7.4f} {quotes[1][1]:7.4f} {sum(quotes[1]):7.4f}")

    # Compare to benchmarks
    from scripts.cont_xiong_exact import fictitious_play
    nash = fictitious_play(N=2, Q=Q, max_iter=50)
    mid = nq // 2
    nash_spread = nash["delta_a"][mid] + nash["delta_b"][mid]

    final_spreads = []
    for i in range(N):
        best_a = np.argmax(Q_tables[i][Q])
        da, db = action_to_quotes(best_a)
        final_spreads.append(da + db)
    learned_spread = np.mean(final_spreads)

    print(f"\n{'='*50}")
    print(f"  Nash spread(0):     {nash_spread:.4f}")
    print(f"  Monopolist spread:  1.5933")
    print(f"  Pareto spread:      1.6361")
    print(f"  Learned spread(0):  {learned_spread:.4f}")
    print(f"  Above Nash?         {'YES — TACIT COLLUSION' if learned_spread > nash_spread else 'NO'}")
    print(f"  Elapsed: {elapsed:.0f}s")

    os.makedirs("results_cx_collusion", exist_ok=True)
    results = {
        "nash_spread": nash_spread, "learned_spread": learned_spread,
        "history": history, "elapsed": elapsed,
        "final_spreads": final_spreads,
    }
    with open("results_cx_collusion/tabular_collusion.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  Saved results_cx_collusion/tabular_collusion.json")


if __name__ == "__main__":
    main()

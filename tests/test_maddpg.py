"""Tests for the MADDPG multi-agent trainer."""
import pytest
import numpy as np
import torch


@pytest.fixture
def small_trainer(cpu_device):
    from solver_cx_multiagent import MADDPGTrainer
    return MADDPGTrainer(
        N=2, Q=5, device=cpu_device,
        n_episodes=2, steps_per_episode=5,
        batch_size=4,
    )


def test_maddpg_constructs(small_trainer):
    assert small_trainer.N == 2
    assert len(small_trainer.agents) == 2


def test_maddpg_actor_output_bounded(small_trainer):
    """Actor output should be in [0.1, 2.0] (bounded sigmoid)."""
    actor = small_trainer.agents[0]["actor"]
    q = torch.tensor([[0.0]], dtype=torch.float64)
    out = actor(q)
    assert out.shape == (1, 2)
    assert (out >= 0.1 - 1e-6).all()
    assert (out <= 2.0 + 1e-6).all()


def test_maddpg_market_step(small_trainer):
    """Market.step returns rewards and won indicators."""
    quotes_a = np.array([0.8, 0.9])
    quotes_b = np.array([0.8, 0.9])
    small_trainer.market.reset()
    rewards, won = small_trainer.market.step(quotes_a, quotes_b)
    assert rewards.shape == (2,)
    assert won.shape == (2,)
    assert won.sum() == 1  # exactly one winner per step

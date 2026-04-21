"""Shared pytest fixtures.

Provides CX parameters and exact reference values for validation.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np
from types import SimpleNamespace


@pytest.fixture(scope="session")
def cpu_device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def cx_config():
    """Standard CX parameters (Table 1 of Cont-Xiong 2024)."""
    return SimpleNamespace(
        lambda_a=2.0, lambda_b=2.0,
        discount_rate=0.01,
        Delta_q=1.0,
        q_max=5.0,
        phi=0.005,
        N_agents=2,
    )


@pytest.fixture(scope="session")
def exact_N2_Q5():
    """Exact Algorithm 1 result for N=2, Q=5. Reference for all tests."""
    from scripts.cont_xiong_exact import fictitious_play
    r = fictitious_play(N=2, Q=5, Delta=1, max_iter=200)
    return {
        "V": np.array(r["V"]),
        "delta_a": np.array(r["delta_a"]),
        "delta_b": np.array(r["delta_b"]),
        "spread_q0": r["delta_a"][5] + r["delta_b"][5],
        "V_q0": r["V"][5],
    }


@pytest.fixture(scope="session")
def exact_monopolist():
    """Exact monopolist (N=1) reference."""
    from scripts.cont_xiong_exact import fictitious_play
    r = fictitious_play(N=1, Q=5, Delta=1, max_iter=50)
    return {
        "V": np.array(r["V"]),
        "delta_a": np.array(r["delta_a"]),
        "delta_b": np.array(r["delta_b"]),
        "spread_q0": r["delta_a"][5] + r["delta_b"][5],
        "V_q0": r["V"][5],
    }

"""Tests for the neural Bellman solver (solver_cx.py)."""
import pytest
import numpy as np
from types import SimpleNamespace


@pytest.fixture(scope="module")
def small_solver_result(cpu_device, cx_config):
    """Train a small neural Bellman solver for testing.

    Short training — accuracy isn't the point, just that it runs
    and returns sensible structure.
    """
    from equations.contxiong_exact import ContXiongExact
    from solver_cx import CXSolver

    eqn = ContXiongExact(cx_config)
    solver = CXSolver(eqn, device=cpu_device, lr=1e-3, n_iter=500, verbose=False)
    return solver.train()


def test_cx_solver_returns_correct_shape(small_solver_result):
    assert len(small_solver_result["V"]) == 11  # nq = 2*5/1 + 1
    assert len(small_solver_result["delta_a"]) == 11
    assert len(small_solver_result["delta_b"]) == 11


def test_cx_solver_boundary_quotes(small_solver_result):
    """Boundary quotes set to zero by compute_optimal_quotes."""
    assert abs(small_solver_result["delta_a"][0]) < 1e-10
    assert abs(small_solver_result["delta_b"][-1]) < 1e-10


def test_cx_solver_V_symmetric_approx(small_solver_result):
    """V should be approximately symmetric (even with short training)."""
    V = np.array(small_solver_result["V"])
    V_reversed = V[::-1]
    # Symmetry up to 10% because training is short
    assert np.max(np.abs(V - V_reversed) / (np.abs(V) + 1e-6)) < 0.1


@pytest.mark.slow
def test_cx_solver_full_training_accuracy(cpu_device, cx_config, exact_N2_Q5):
    """Full-length training should achieve sub-1% spread error.

    Marked as slow — skip with `pytest -m 'not slow'`.
    """
    from equations.contxiong_exact import ContXiongExact
    from solver_cx import CXSolver

    eqn = ContXiongExact(cx_config)
    solver = CXSolver(eqn, device=cpu_device, lr=1e-3, n_iter=5000, verbose=False)
    r = solver.train()
    nn_spread = r["delta_a"][5] + r["delta_b"][5]
    err = abs(nn_spread - exact_N2_Q5["spread_q0"]) / exact_N2_Q5["spread_q0"]
    assert err < 0.01, f"NN spread error {err:.4%} exceeds 1%"


def test_cx_solver_V_is_negative_parabolic_shape(small_solver_result):
    """V(q) should be maximal around q=0 and decrease with |q|."""
    V = np.array(small_solver_result["V"])
    mid = 5
    # V(0) should be larger than V(±Q) (inventory penalty)
    assert V[mid] > V[0] - 0.5  # allow slack for short training
    assert V[mid] > V[-1] - 0.5

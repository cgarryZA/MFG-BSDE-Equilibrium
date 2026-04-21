"""Tests for the CX execution probability formula (eq 58).

This is foundational — if this is wrong, everything else is too.
"""
import numpy as np
import torch
import pytest
from equations.contxiong_exact import (
    cx_exec_prob_np, cx_exec_prob_torch, optimal_quote_foc,
)


def test_exec_prob_monopolist_formula():
    """Monopolist (K=0): f = sigmoid(-delta)^2."""
    for delta in [-1.0, 0.0, 0.5, 1.0, 2.0]:
        f = cx_exec_prob_np(delta, 0.0, K=0, N=1)
        expected = (1.0 / (1.0 + np.exp(delta))) ** 2
        assert abs(f - expected) < 1e-10, f"At delta={delta}: {f} vs {expected}"


def test_exec_prob_decreases_with_delta():
    """f should decrease as delta (own quote) increases (wider quote → less execution)."""
    prev = 1.0
    for delta in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        f = cx_exec_prob_np(delta, 0.75, K=11, N=2)
        assert f < prev, f"f not decreasing at delta={delta}"
        prev = f


def test_exec_prob_torch_matches_numpy():
    """Torch and numpy versions should agree."""
    delta = 0.8
    avg = 0.75
    f_np = cx_exec_prob_np(delta, avg, K=11, N=2)
    f_t = cx_exec_prob_torch(
        torch.tensor(delta, dtype=torch.float64),
        torch.tensor(avg, dtype=torch.float64),
        K=11, N=2
    )
    assert abs(f_np - f_t.item()) < 1e-10


def test_exec_prob_1_over_N_scaling():
    """1/N market share factor: doubling N should halve execution probability
    at the same delta and avg_comp."""
    delta, avg = 0.8, 0.75
    f_N2 = cx_exec_prob_np(delta, avg, K=11, N=2)
    f_N4 = cx_exec_prob_np(delta, avg, K=11, N=4)
    ratio = f_N2 / f_N4
    # f = (1/N) * base * comp — base and comp don't depend on N directly,
    # so doubling N halves f.
    assert abs(ratio - 2.0) < 0.01


def test_optimal_quote_foc_basic():
    """FOC should give a finite positive quote for standard values."""
    delta_star = optimal_quote_foc(p=0.0, avg_competitor_quote=0.75, K=11, N=2)
    assert 0.3 < delta_star < 1.5


def test_optimal_quote_increases_with_p():
    """Higher p (value of inventory) → higher optimal quote."""
    prev = optimal_quote_foc(p=-1.0, avg_competitor_quote=0.75, K=11, N=2)
    for p in [0.0, 0.5, 1.0, 1.5]:
        d = optimal_quote_foc(p, 0.75, K=11, N=2)
        assert d > prev, f"quote not monotone at p={p}"
        prev = d

"""Tests for exact Algorithm 1 (ground truth)."""
import pytest
import numpy as np


def test_exact_N2_converges(exact_N2_Q5):
    """Exact solver should return standard CX Nash spread at q=0."""
    assert abs(exact_N2_Q5["spread_q0"] - 1.5153) < 1e-3


def test_exact_N2_symmetry(exact_N2_Q5):
    """V should be symmetric: V(q) = V(-q)."""
    V = exact_N2_Q5["V"]
    for i in range(len(V) // 2):
        assert abs(V[i] - V[-(i+1)]) < 1e-6, f"V not symmetric at index {i}"


def test_exact_N2_boundary_quotes_zero(exact_N2_Q5):
    """Boundary quotes must be zero: da[0] = 0, db[-1] = 0."""
    assert abs(exact_N2_Q5["delta_a"][0]) < 1e-10
    assert abs(exact_N2_Q5["delta_b"][-1]) < 1e-10


def test_exact_monopolist_spread(exact_monopolist):
    """Monopolist spread ~ 1.59 at q=0 (from CX eq 58 monopolist case)."""
    assert 1.55 < exact_monopolist["spread_q0"] < 1.65


def test_exact_spread_ordering():
    """Nash (N=2) < Monopolist: more competition tightens spreads."""
    from scripts.cont_xiong_exact import fictitious_play
    nash = fictitious_play(N=2, Q=5, Delta=1)
    mono = fictitious_play(N=1, Q=5, Delta=1)
    mid = 5
    s_nash = nash["delta_a"][mid] + nash["delta_b"][mid]
    s_mono = mono["delta_a"][mid] + mono["delta_b"][mid]
    assert s_nash < s_mono, f"Expected Nash={s_nash} < Monopolist={s_mono}"


@pytest.mark.parametrize("N,expected_approx", [
    (2, 1.515),
    (5, 1.587),
    (10, 1.661),
])
def test_exact_N_scaling(N, expected_approx):
    """Spread should match known values as N varies."""
    from scripts.cont_xiong_exact import fictitious_play
    r = fictitious_play(N=N, Q=5, Delta=1)
    s = r["delta_a"][5] + r["delta_b"][5]
    assert abs(s - expected_approx) < 0.01


def test_exact_N_monotone():
    """Spreads should increase monotonically with N (more dealers → wider spreads)."""
    from scripts.cont_xiong_exact import fictitious_play
    spreads = []
    for N in [2, 3, 5, 10, 20]:
        r = fictitious_play(N=N, Q=5, Delta=1)
        spreads.append(r["delta_a"][5] + r["delta_b"][5])
    for i in range(1, len(spreads)):
        assert spreads[i] > spreads[i-1], f"Spread not monotone at N={[2,3,5,10,20][i]}"

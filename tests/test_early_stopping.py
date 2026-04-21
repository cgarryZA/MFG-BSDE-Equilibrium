"""Tests for EarlyStopping utility."""
import pytest
from utils import EarlyStopping


def test_stops_after_patience():
    es = EarlyStopping(patience=10, min_delta=1e-6, warmup=0)
    # First call establishes baseline
    for step in range(5):
        assert not es(1.0)
    # No improvement for 10 more steps → stop
    for step in range(10):
        triggered = es(1.0)
    assert es.stopped
    assert triggered


def test_continues_while_improving():
    es = EarlyStopping(patience=5, min_delta=1e-6, warmup=0)
    # Monotonically decreasing loss — should never stop
    for step in range(50):
        stop = es(1.0 - step * 0.01)
        assert not stop
    assert not es.stopped


def test_warmup_prevents_early_stop():
    es = EarlyStopping(patience=5, min_delta=1e-6, warmup=100)
    # Constant loss from the start — should not stop during warmup
    for step in range(99):
        assert not es(1.0)
    # After warmup + patience, should stop
    stop = False
    for step in range(100):
        stop = es(1.0)
        if stop:
            break
    assert stop


def test_tracks_best_loss():
    es = EarlyStopping(patience=100, warmup=0)
    es(1.0)
    es(0.5)
    es(0.7)
    es(0.3)
    es(0.4)
    assert abs(es.best_loss - 0.3) < 1e-9


def test_mode_max():
    """max mode for metrics we want to maximise."""
    es = EarlyStopping(patience=5, min_delta=1e-6, warmup=0, mode="max")
    for step in range(10):
        # Increasing value → improvement
        stop = es(step * 0.1)
        assert not stop
    # Stop improving
    for step in range(10):
        stop = es(0.5)  # flat
    assert es.stopped

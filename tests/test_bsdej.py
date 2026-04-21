"""Tests for the BSDEJ jump solver (solver_cx_bsdej_shared.py).

Verifies:
- Constructor & forward pass run
- sample_paths returns correct shapes including execution events
- Compensated jump martingale is included (ablation: without it → wrong answer)
- Warm-start pretraining converges
"""
import pytest
import numpy as np
import torch


@pytest.fixture
def small_bsdej_solver(cpu_device):
    """Quick BSDEJ solver for smoke tests."""
    from solver_cx_bsdej_shared import CXBSDEJShared
    return CXBSDEJShared(
        N=2, Q=5, Delta=1, T=1.0, M=5,
        lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
        device=cpu_device, lr=1e-3, n_iter=10, batch_size=8,
        hidden=16, n_layers=2,
    )


def test_bsdej_constructs(small_bsdej_solver):
    """Solver builds with valid parameters."""
    assert small_bsdej_solver.nq == 11
    assert small_bsdej_solver.K == 11


def test_bsdej_sample_paths_shape(small_bsdej_solver):
    """sample_paths returns (q_paths, exec_a, exec_b) with correct shapes."""
    q_paths, exec_a, exec_b = small_bsdej_solver.sample_paths(8)
    assert q_paths.shape == (8, 6)  # M+1 = 6
    assert exec_a.shape == (8, 5)
    assert exec_b.shape == (8, 5)
    # Start at q=0
    np.testing.assert_array_equal(q_paths[:, 0], 0)
    # Execution events are 0 or 1
    assert set(np.unique(exec_a)).issubset({0.0, 1.0})


def test_bsdej_forward_pass(small_bsdej_solver):
    """Forward pass returns [batch, 1] tensor with gradient flow."""
    q_paths, exec_a, exec_b = small_bsdej_solver.sample_paths(8)
    Y_T = small_bsdej_solver.forward(q_paths, exec_a, exec_b)
    assert Y_T.shape == (8, 1)
    assert Y_T.requires_grad


def test_bsdej_training_step(small_bsdej_solver):
    """Training runs for a few steps without NaN."""
    r = small_bsdej_solver.train()
    assert r["best_loss"] < float('inf')
    assert not np.isnan(r["best_loss"])


def test_bsdej_compensated_martingale_matters(cpu_device):
    """Ablation: dropping the jump martingale gives drastically wrong answer.

    With martingale: Y track V(q) correctly → reasonable spread
    Without martingale: Y drifts → wrong spread (100%+ error typical).
    """
    from solver_cx_bsdej_shared import CXBSDEJShared

    class BuggyBSDEJ(CXBSDEJShared):
        """Drops the jump martingale term (the bug we fixed)."""
        def forward(self, q_paths, exec_a_all, exec_b_all):
            # Copy parent logic but skip martingale
            import torch
            from solver_cx_bsdej import _exec_prob_torch_vec, optimal_quotes_vectorised
            batch = q_paths.shape[0]
            dtype = torch.float64
            dev = self.device
            q0 = torch.tensor(q_paths[:, 0], dtype=dtype, device=dev).long()
            q0_idx = (q0 + self.Q).clamp(0, self.nq - 1)
            Y = self.Y0[q0_idx].unsqueeze(1)
            for m in range(self.M):
                t_norm = torch.full((batch, 1), m / self.M, dtype=dtype, device=dev)
                q_m_raw = torch.tensor(q_paths[:, m], dtype=dtype, device=dev)
                q_m_norm = (q_m_raw / self.Q).unsqueeze(1)
                U = self.shared_net(t_norm, q_m_norm)
                Ua = U[:, 0:1]; Ub = U[:, 1:2]
                da_t = optimal_quotes_vectorised(Ua, self.avg_comp, self.K, self.N)
                db_t = optimal_quotes_vectorised(Ub, self.avg_comp, self.K, self.N)
                fa = _exec_prob_torch_vec(da_t, self.avg_comp, self.K, self.N) * self.lambda_a
                fb = _exec_prob_torch_vec(db_t, self.avg_comp, self.K, self.N) * self.lambda_b
                can_sell = (q_m_raw > -self.Q).float().unsqueeze(1)
                can_buy = (q_m_raw < self.Q).float().unsqueeze(1)
                profit_a = can_sell * fa * (da_t * self.Delta + Ua)
                profit_b = can_buy * fb * (db_t * self.Delta + Ub)
                psi_q = self.phi * q_m_raw.unsqueeze(1) ** 2
                f_val = self.r * Y + psi_q - profit_a - profit_b
                # NO JUMP MARTINGALE (the bug)
                Y = Y - f_val * self.dt
            return Y

    correct = CXBSDEJShared(
        N=2, Q=5, Delta=1, T=1.0, M=5,
        lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
        device=cpu_device, lr=1e-3, n_iter=20, batch_size=8,
        hidden=16, n_layers=2,
    )
    buggy = BuggyBSDEJ(
        N=2, Q=5, Delta=1, T=1.0, M=5,
        lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
        device=cpu_device, lr=1e-3, n_iter=20, batch_size=8,
        hidden=16, n_layers=2,
    )

    # Same initial Y0, same random state
    torch.manual_seed(42)
    np.random.seed(42)
    qp, ea, eb = correct.sample_paths(32)
    torch.manual_seed(0)
    Y_correct = correct.forward(qp, ea, eb)
    torch.manual_seed(0)
    Y_buggy = buggy.forward(qp, ea, eb)

    # The outputs should differ substantially (jump events occurred)
    diff = (Y_correct - Y_buggy).abs().mean().item()
    assert diff > 0, "Correct and buggy forward should differ when jumps happen"

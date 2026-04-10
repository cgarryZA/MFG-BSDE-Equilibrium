#!/usr/bin/env python
"""
Cont-Xiong Pareto optimum (collusion).

Under explicit collusion, N dealers form a cartel. The cartel controls
all quotes and maximises the JOINT value. Since the cartel has no
internal competition, the execution probability is the MONOPOLIST rate,
not the competitive per-dealer rate.

For N=2 homogeneous dealers colluding:
- Execution prob = monopolist rate Lambda(delta) = 1/(1+exp(delta))^2
- Quotes depend on joint inventory (q1, q2)
- Result: spreads WIDER than Nash (cartel extracts monopoly rents)

This gives: Pareto spread > Nash spread > 0, which is the standard
collusion result.
"""

import numpy as np
import sys
import os
from scipy.optimize import minimize_scalar
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def monopolist_exec_prob(delta):
    """Monopolist execution probability: Lambda(delta) = 1/(1+exp(delta))^2.

    From CX Remark 2.2 / Remark 2.5: when N=1, f = Lambda(delta).
    Under collusion, the cartel acts as a monopolist.
    """
    base = 1.0 / (1.0 + np.exp(np.clip(delta, -20, 20)))
    return base * base


def pareto_optimum(N=2, Q=5, Delta=1, lambda_a=2.0, lambda_b=2.0, r=0.01,
                   phi=0.005, max_iter=50, tol=1e-5):
    """Pareto optimum via policy iteration on joint state space.

    The cartel uses monopolist execution probabilities since there's
    no internal competition. Quotes are per-dealer but the execution
    decision is centralised.
    """
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)
    n_states = nq * nq

    def psi(q):
        return phi * q ** 2

    def idx(i1, i2):
        return i1 * nq + i2

    # Initialise quotes with inventory skew
    # Under collusion: each agent's quote at (q1,q2)
    da1 = np.zeros((nq, nq))  # agent 1 ask
    db1 = np.zeros((nq, nq))
    da2 = np.zeros((nq, nq))
    db2 = np.zeros((nq, nq))
    for i1 in range(nq):
        for i2 in range(nq):
            q1 = q_grid[i1]; q2 = q_grid[i2]
            da1[i1, i2] = 0.9 - 0.05 * q1
            db1[i1, i2] = 0.9 + 0.05 * q1
            da2[i1, i2] = 0.9 - 0.05 * q2
            db2[i1, i2] = 0.9 + 0.05 * q2

    history = []

    for iteration in range(max_iter):
        # Policy evaluation: solve M * W = A
        M = np.zeros((n_states, n_states))
        A_vec = np.zeros(n_states)

        for i1 in range(nq):
            for i2 in range(nq):
                q1 = q_grid[i1]; q2 = q_grid[i2]
                s = idx(i1, i2)

                # Under collusion: monopolist execution rates
                # The cartel chooses which dealer executes, but total rate
                # is the monopolist rate applied to the BEST quote
                # For symmetric cartel: both quote same, split executions
                fa1 = monopolist_exec_prob(da1[i1, i2]) / N if q1 > -Q else 0
                fb1 = monopolist_exec_prob(db1[i1, i2]) / N if q1 < Q else 0
                fa2 = monopolist_exec_prob(da2[i1, i2]) / N if q2 > -Q else 0
                fb2 = monopolist_exec_prob(db2[i1, i2]) / N if q2 < Q else 0

                # Diagonal
                M[s, s] = r + lambda_a * (fa1 + fa2) + lambda_b * (fb1 + fb2)

                # Transitions
                if q1 > -Q and i1 > 0:
                    M[s, idx(i1-1, i2)] -= lambda_a * fa1
                if q1 < Q and i1 < nq-1:
                    M[s, idx(i1+1, i2)] -= lambda_b * fb1
                if q2 > -Q and i2 > 0:
                    M[s, idx(i1, i2-1)] -= lambda_a * fa2
                if q2 < Q and i2 < nq-1:
                    M[s, idx(i1, i2+1)] -= lambda_b * fb2

                # RHS
                A_vec[s] = -(psi(q1) + psi(q2))
                A_vec[s] += lambda_a * Delta * (fa1 * da1[i1, i2] + fa2 * da2[i1, i2])
                A_vec[s] += lambda_b * Delta * (fb1 * db1[i1, i2] + fb2 * db2[i1, i2])

        W = np.linalg.solve(M, A_vec)
        W_grid = W.reshape(nq, nq)

        # Policy improvement
        new_da1 = np.zeros((nq, nq))
        new_db1 = np.zeros((nq, nq))
        new_da2 = np.zeros((nq, nq))
        new_db2 = np.zeros((nq, nq))

        for i1 in range(nq):
            for i2 in range(nq):
                q1 = q_grid[i1]; q2 = q_grid[i2]
                W_here = W_grid[i1, i2]

                W_a1d = W_grid[i1-1, i2] if i1 > 0 else -(psi(q1-Delta) + psi(q2))
                W_a1u = W_grid[i1+1, i2] if i1 < nq-1 else -(psi(q1+Delta) + psi(q2))
                W_a2d = W_grid[i1, i2-1] if i2 > 0 else -(psi(q1) + psi(q2-Delta))
                W_a2u = W_grid[i1, i2+1] if i2 < nq-1 else -(psi(q1) + psi(q2+Delta))

                p_a1 = (W_here - W_a1d) / Delta
                p_b1 = (W_here - W_a1u) / Delta
                p_a2 = (W_here - W_a2d) / Delta
                p_b2 = (W_here - W_a2u) / Delta

                # FOC with monopolist execution probability / N
                # argmax (1/N) * Lambda(delta) * (delta - p)
                # Same FOC as monopolist since 1/N is constant
                if q1 > -Q:
                    def neg_pa1(d):
                        return -monopolist_exec_prob(d) * (d - p_a1)
                    new_da1[i1, i2] = minimize_scalar(neg_pa1, bounds=(-2, 10), method='bounded').x
                if q1 < Q:
                    def neg_pb1(d):
                        return -monopolist_exec_prob(d) * (d - p_b1)
                    new_db1[i1, i2] = minimize_scalar(neg_pb1, bounds=(-2, 10), method='bounded').x
                if q2 > -Q:
                    def neg_pa2(d):
                        return -monopolist_exec_prob(d) * (d - p_a2)
                    new_da2[i1, i2] = minimize_scalar(neg_pa2, bounds=(-2, 10), method='bounded').x
                if q2 < Q:
                    def neg_pb2(d):
                        return -monopolist_exec_prob(d) * (d - p_b2)
                    new_db2[i1, i2] = minimize_scalar(neg_pb2, bounds=(-2, 10), method='bounded').x

        diff = max(np.max(np.abs(new_da1 - da1)), np.max(np.abs(new_db1 - db1)),
                   np.max(np.abs(new_da2 - da2)), np.max(np.abs(new_db2 - db2)))

        mid = nq // 2
        spread_q0 = new_da1[mid, mid] + new_db1[mid, mid]
        W_q0 = W_grid[mid, mid]

        history.append({"iter": iteration+1, "diff": diff, "spread_q0": spread_q0, "W_q0": W_q0})
        print(f"  iter {iteration+1}: diff={diff:.6f}, spread(0,0)={spread_q0:.4f}, W(0,0)={W_q0:.4f}")

        da1, db1, da2, db2 = new_da1, new_db1, new_da2, new_db2

        if diff < tol:
            print(f"  Converged at iteration {iteration+1}")
            break

    mid2 = nq // 2
    return {
        "q_grid": q_grid.tolist(),
        "delta_a": da1[:, mid2].tolist(),
        "delta_b": db1[:, mid2].tolist(),
        "spread": [da1[i, mid2] + db1[i, mid2] for i in range(nq)],
        "W_q0": W_q0,
        "spread_q0": spread_q0,
        "history": history,
        "N": N,
    }


def main():
    os.makedirs("results_cx_exact", exist_ok=True)
    print("=" * 60)
    print("Cont-Xiong Pareto Optimum (Collusion, N=2)")
    print("=" * 60)

    result = pareto_optimum(N=2, max_iter=30)
    if result:
        print(f"\n  Pareto spread(0) = {result['spread_q0']:.4f}")

        from scripts.cont_xiong_exact import fictitious_play
        nash = fictitious_play(N=2, max_iter=50)
        mid = len(nash["q_grid"]) // 2
        nash_spread = nash["delta_a"][mid] + nash["delta_b"][mid]
        print(f"  Nash spread(0) = {nash_spread:.4f}")
        print(f"  Collusion premium = {result['spread_q0'] - nash_spread:.4f}")

        result["nash_spread_q0"] = nash_spread
        with open("results_cx_exact/pareto_N2.json", "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"\n  Saved results_cx_exact/pareto_N2.json")


if __name__ == "__main__":
    main()

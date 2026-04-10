#!/usr/bin/env python
"""
Cont-Xiong Pareto optimum (collusion) via Algorithm 0 (policy iteration).

For N=2 homogeneous dealers, the collusive strategy maximises the JOINT
value W(q1, q2) = sum of both agents' values.

Under collusion, quotes depend on BOTH inventories: delta_a(q1, q2).
Approximated by linear form (eq 50):
  delta_a ≈ xi_0^a + xi_1^a * q_1 + xi_2^a * q_2
  delta_b ≈ xi_0^b + xi_1^b * q_1 + xi_2^b * q_2

Algorithm 0:
1. Initialise quoting strategies
2. Policy evaluation: solve linear system (eq 51) for W(q) given strategies
3. Policy improvement: find xi parameters that maximise W
4. Repeat until convergence

Output: Pareto optimal quotes — the upper bound on spreads.
Nash < learned < Pareto means tacit collusion.
"""

import numpy as np
import sys
import os
from scipy.optimize import minimize
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.cont_xiong_exact import cx_execution_prob


def pareto_optimum(N=2, Q=5, Delta=1, lambda_a=2.0, lambda_b=2.0, r=0.01,
                   phi=0.005, max_iter=50, tol=1e-5):
    """Compute Pareto optimum for N homogeneous dealers.

    For N=2, the joint state is (q1, q2) in Q x Q.
    Joint value W(q1, q2) satisfies eq (51).

    We use the linear approximation (eq 50) for quotes:
      delta_a_i(q) = xi_0^a + sum_k xi_k^a * q_k
      delta_b_i(q) = xi_0^b + sum_k xi_k^b * q_k

    For homogeneous dealers, all share the same xi parameters.
    """
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)

    # For N=2: joint state (q1, q2), each in q_grid
    # Total states: nq^2
    n_states = nq ** N if N <= 2 else nq  # only do joint for N<=2

    if N > 2:
        print(f"  Pareto for N>{2} not implemented (state space too large)")
        return None

    # Map (q1, q2) to flat index
    def state_idx(q1_idx, q2_idx):
        return q1_idx * nq + q2_idx

    def psi(q):
        return phi * q ** 2

    # K_i for execution probability
    K_i = (N - 1) * nq

    # Linear quote parameters: xi_0, xi_1, xi_2 for ask and bid
    # delta_a_i(q1,q2) = xi_0a + xi_1a * q_i + xi_2a * q_j (where j != i)
    # For homogeneous: xi_1a is own-inventory sensitivity, xi_2a is other's
    # Initialise near monopolist quotes
    xi_a = np.array([0.8, -0.05, 0.0])  # [xi_0, xi_own, xi_other]
    xi_b = np.array([0.8, 0.05, 0.0])

    history = []

    for iteration in range(max_iter):
        # Compute quotes from linear parameters for all joint states
        # For agent i at joint state (q1, q2):
        #   delta_a_1(q1,q2) = xi_0a + xi_1a*q1 + xi_2a*q2
        #   delta_a_2(q1,q2) = xi_0a + xi_1a*q2 + xi_2a*q1  (symmetric)

        # Policy evaluation: solve M * W = A for W(q1, q2)
        M = np.zeros((n_states, n_states))
        A = np.zeros(n_states)

        for i1 in range(nq):
            for i2 in range(nq):
                q1 = q_grid[i1]
                q2 = q_grid[i2]
                idx = state_idx(i1, i2)

                # Quotes for both agents at this joint state
                da1 = xi_a[0] + xi_a[1] * q1 + xi_a[2] * q2
                db1 = xi_b[0] + xi_b[1] * q1 + xi_b[2] * q2
                da2 = xi_a[0] + xi_a[1] * q2 + xi_a[2] * q1  # symmetric
                db2 = xi_b[0] + xi_b[1] * q2 + xi_b[2] * q1

                # Competitors' quotes for agent 1: agent 2's quotes
                # Agent 2 has nq possible inventory levels, but at this state q2 is fixed
                # For the execution prob, we need the avg of competitor quotes across inventory levels
                # Under collusion, competitor quotes ARE known (shared info)
                # So we use agent 2's actual quote at current q2
                comp_da1 = np.array([da2])  # agent 2's ask quote
                comp_db1 = np.array([db2])

                # Execution probabilities (sum over both agents)
                fa1 = cx_execution_prob(da1, comp_da1, K_i, N) if q1 > -Q else 0
                fb1 = cx_execution_prob(db1, comp_db1, K_i, N) if q1 < Q else 0
                fa2 = cx_execution_prob(da2, np.array([da1]), K_i, N) if q2 > -Q else 0
                fb2 = cx_execution_prob(db2, np.array([db1]), K_i, N) if q2 < Q else 0

                # Diagonal
                M[idx, idx] = r + lambda_a * (fa1 + fa2) + lambda_b * (fb1 + fb2)

                # Off-diagonal: transitions
                # Agent 1 ask execution: q1 -> q1-1
                if q1 > -Q and i1 > 0:
                    M[idx, state_idx(i1 - 1, i2)] -= lambda_a * fa1
                # Agent 1 bid execution: q1 -> q1+1
                if q1 < Q and i1 < nq - 1:
                    M[idx, state_idx(i1 + 1, i2)] -= lambda_b * fb1
                # Agent 2 ask execution: q2 -> q2-1
                if q2 > -Q and i2 > 0:
                    M[idx, state_idx(i1, i2 - 1)] -= lambda_a * fa2
                # Agent 2 bid execution: q2 -> q2+1
                if q2 < Q and i2 < nq - 1:
                    M[idx, state_idx(i1, i2 + 1)] -= lambda_b * fb2

                # RHS: running costs + spread revenues
                A[idx] = -(psi(q1) + psi(q2))
                A[idx] += lambda_a * Delta * (fa1 * da1 + fa2 * da2)
                A[idx] += lambda_b * Delta * (fb1 * db1 + fb2 * db2)

        # Solve for W
        W = np.linalg.solve(M, A)

        # Policy improvement: find xi that maximises W
        # We optimise xi_a and xi_b to maximise total W at the (0,0) state
        # (or equivalently, sum of W across all states)
        mid1 = nq // 2
        mid2 = nq // 2

        def neg_total_value(params):
            """Negative of W(0,0) given xi parameters."""
            xa = params[:3]
            xb = params[3:]

            # Recompute quotes and value at (0,0)
            total = 0
            for i1 in range(nq):
                for i2 in range(nq):
                    q1 = q_grid[i1]; q2 = q_grid[i2]
                    da1 = xa[0] + xa[1]*q1 + xa[2]*q2
                    db1 = xb[0] + xb[1]*q1 + xb[2]*q2
                    da2 = xa[0] + xa[1]*q2 + xa[2]*q1
                    db2 = xb[0] + xb[1]*q2 + xb[2]*q1

                    comp1 = np.array([da2]); comp2 = np.array([da1])
                    fa1 = cx_execution_prob(da1, comp1, K_i, N) if q1 > -Q else 0
                    fb1 = cx_execution_prob(db1, np.array([db2]), K_i, N) if q1 < Q else 0
                    fa2 = cx_execution_prob(da2, comp2, K_i, N) if q2 > -Q else 0
                    fb2 = cx_execution_prob(db2, np.array([db1]), K_i, N) if q2 < Q else 0

                    profit = lambda_a * Delta * (fa1*da1 + fa2*da2) + lambda_b * Delta * (fb1*db1 + fb2*db2)
                    cost = psi(q1) + psi(q2)
                    total += profit - cost
            return -total

        # Optimise xi parameters
        result = minimize(neg_total_value, np.concatenate([xi_a, xi_b]),
                          method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6})
        new_xi_a = result.x[:3]
        new_xi_b = result.x[3:]

        diff = max(np.max(np.abs(new_xi_a - xi_a)), np.max(np.abs(new_xi_b - xi_b)))

        # Collusive spread at q=0
        da_q0 = new_xi_a[0]  # xi_0a (since q1=q2=0)
        db_q0 = new_xi_b[0]
        spread_q0 = da_q0 + db_q0
        W_q0 = W[state_idx(mid1, mid2)]

        history.append({
            "iter": iteration + 1, "diff": diff,
            "spread_q0": spread_q0, "W_q0": W_q0,
            "xi_a": new_xi_a.tolist(), "xi_b": new_xi_b.tolist(),
        })
        print(f"  iter {iteration+1}: diff={diff:.6f}, spread(0)={spread_q0:.4f}, "
              f"W(0,0)={W_q0:.4f}")

        xi_a = new_xi_a
        xi_b = new_xi_b

        if diff < tol:
            print(f"  Converged at iteration {iteration+1}")
            break

    # Final quotes at each inventory level (for agent 1 when agent 2 is at q=0)
    delta_a_profile = [xi_a[0] + xi_a[1] * q + xi_a[2] * 0 for q in q_grid]
    delta_b_profile = [xi_b[0] + xi_b[1] * q + xi_b[2] * 0 for q in q_grid]

    return {
        "q_grid": q_grid.tolist(),
        "delta_a": delta_a_profile,
        "delta_b": delta_b_profile,
        "spread": [a + b for a, b in zip(delta_a_profile, delta_b_profile)],
        "xi_a": xi_a.tolist(),
        "xi_b": xi_b.tolist(),
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
        mid = len(result["q_grid"]) // 2
        print(f"\n  Pareto spread(0) = {result['spread_q0']:.4f}")
        print(f"  W(0,0) = {result['W_q0']:.4f}")
        print(f"  xi_a = {result['xi_a']}")
        print(f"  xi_b = {result['xi_b']}")

        # Compare to Nash
        from scripts.cont_xiong_exact import fictitious_play
        nash = fictitious_play(N=2, max_iter=50)
        nash_spread = nash["delta_a"][mid] + nash["delta_b"][mid]
        print(f"\n  Nash spread(0) = {nash_spread:.4f}")
        print(f"  Pareto spread(0) = {result['spread_q0']:.4f}")
        print(f"  Collusion premium = {result['spread_q0'] - nash_spread:.4f}")

        result["nash_spread_q0"] = nash_spread
        with open("results_cx_exact/pareto_N2.json", "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"\n  Saved results_cx_exact/pareto_N2.json")


if __name__ == "__main__":
    main()

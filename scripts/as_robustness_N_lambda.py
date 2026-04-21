#!/usr/bin/env python -u
"""Test whether 85% pass-through is structural across N and lambda.

If beta stays ~0.85 across:
  N in {2, 5, 10}
  lambda in {1, 2, 4}

then it's a true structural invariant of the CX equilibrium.
If it varies, it's specific to (N=2, lambda=2).

CPU, ~30 min.
"""

import sys, os, json
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equations.contxiong_exact import cx_exec_prob_np, optimal_quote_foc

print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def adverse_fp_param(N, lam, alpha_a, alpha_b, theta_a, theta_b,
                     Q=5, Delta=1, r=0.01, phi=0.005, max_iter=100, tol=1e-7):
    """Adverse-selection FP at arbitrary N and lambda."""
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)
    K_i = (N - 1) * nq if N > 1 else 0
    psi = lambda q: phi * q**2

    delta_a = np.ones(nq) * 0.8; delta_a[0] = 0.0
    delta_b = np.ones(nq) * 0.8; delta_b[-1] = 0.0

    for it in range(max_iter):
        M = np.zeros((nq, nq)); A = np.zeros(nq)
        avg_da = float(np.mean(delta_a)); avg_db = float(np.mean(delta_b))

        for j in range(nq):
            q = q_grid[j]
            fa = cx_exec_prob_np(delta_a[j], avg_da, K_i, N) if q > -Q else 0.0
            fb = cx_exec_prob_np(delta_b[j], avg_db, K_i, N) if q < Q else 0.0
            nu_a = lam * fa
            nu_b = lam * fb

            M[j, j] = r + nu_a + nu_b
            if q > -Q and j > 0: M[j, j-1] = -nu_a
            if q < Q and j < nq-1: M[j, j+1] = -nu_b

            A[j] = -psi(q)
            if q > -Q:
                A[j] += lam * Delta * fa * (delta_a[j] - alpha_a * theta_a)
            if q < Q:
                A[j] += lam * Delta * fb * (delta_b[j] - alpha_b * theta_b)

        V = np.linalg.solve(M, A)

        new_a = np.zeros(nq); new_b = np.zeros(nq)
        for j in range(nq):
            q = q_grid[j]
            if j > 0 and q > -Q:
                p_a = (V[j] - V[j-1]) / Delta + alpha_a * theta_a
                new_a[j] = optimal_quote_foc(p_a, avg_da, K_i, N)
            if j < nq-1 and q < Q:
                p_b = (V[j] - V[j+1]) / Delta + alpha_b * theta_b
                new_b[j] = optimal_quote_foc(p_b, avg_db, K_i, N)

        damp = 0.5
        diff = max(np.max(np.abs(new_a - delta_a)), np.max(np.abs(new_b - delta_b)))
        delta_a = damp * new_a + (1 - damp) * delta_a
        delta_b = damp * new_b + (1 - damp) * delta_b
        if diff < tol: break

    mid = nq // 2
    return float(delta_a[mid] + delta_b[mid]), float(V[mid])


def measure_pass_through(N, lam, theta=0.3):
    """Measure pass-through coefficient at given N, lambda."""
    alpha_grid = np.linspace(0.0, 0.5, 6)
    spreads = []
    for a in alpha_grid:
        s, _ = adverse_fp_param(N=N, lam=lam, alpha_a=a, alpha_b=a,
                                 theta_a=theta, theta_b=theta)
        spreads.append(s)
    spreads = np.array(spreads)
    slope, _ = np.polyfit(alpha_grid, spreads, 1)
    naive = 2 * theta
    ratio = slope / naive
    return float(ratio), float(slope), float(spreads[0])


print(f"\n{'='*60}")
print("AS robustness: pass-through across N and lambda")
print(f"{'='*60}", flush=True)

configs = [(N, lam) for N in [2, 3, 5, 10] for lam in [1.0, 2.0, 4.0]]

print(f"\n{'N':>4s}  {'lambda':>6s}  {'nash sprd':>10s}  {'slope':>8s}  {'naive':>8s}  {'ratio':>8s}")
results = []
for N, lam in configs:
    ratio, slope, nash = measure_pass_through(N, lam)
    print(f"  {N:4d}  {lam:6.1f}  {nash:10.4f}  {slope:8.4f}  {2*0.3:8.4f}  {ratio:7.4f}",
          flush=True)
    results.append({"N": N, "lambda": lam, "nash": nash,
                   "slope": slope, "ratio": ratio})

# Stability check
ratios = [r['ratio'] for r in results]
mean_r = np.mean(ratios)
std_r = np.std(ratios, ddof=1)
print(f"\n{'='*60}")
print(f"Pass-through ratio across all (N, lambda) configs:")
print(f"  Mean:  {mean_r:.4f}")
print(f"  Std:   {std_r:.4f}")
print(f"  Range: [{min(ratios):.4f}, {max(ratios):.4f}]")
if std_r < 0.02:
    print(f"\n  VERDICT: Pass-through ~ {mean_r:.3f} is a TRUE structural invariant")
    print(f"  (stable across 4 N values and 3 lambda values)")
else:
    print(f"\n  VERDICT: Pass-through varies with (N, lambda); not fully structural")

with open("results_final/as_robustness_N_lambda.json", "w") as f:
    json.dump({
        "results": results,
        "mean_ratio": float(mean_r),
        "std_ratio": float(std_r),
        "is_structural": bool(std_r < 0.02),
    }, f, indent=2, default=float)
print(f"\nSaved to results_final/as_robustness_N_lambda.json", flush=True)
print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

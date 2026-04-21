#!/usr/bin/env python -u
"""Deep analysis of adverse selection in the CX dealer market.

Goes beyond the basic "spread widens with alpha" curve to answer:

  1. Is spread(alpha) linear? quadratic? Fit both and compare.
  2. Does the observed premium match the naive formula (alpha * theta)?
  3. Does the quote profile vs inventory change shape with alpha?
  4. Does bid-ask asymmetry emerge if we make informed flow one-sided?
  5. Is there a regime threshold (alpha too high → dealer effectively exits)?

CPU only. ~5 min.
"""

import sys, os, json, time
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equations.contxiong_exact import cx_exec_prob_np, optimal_quote_foc

print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def adverse_fp(alpha_a, alpha_b, theta_a, theta_b, N=2, Q=5, Delta=1,
               lambda_a=2, lambda_b=2, r=0.01, phi=0.005, max_iter=100):
    """FP with asymmetric adverse selection: informed on ask side (alpha_a, theta_a)
    and bid side (alpha_b, theta_b) independently.
    """
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)
    K_i = (N - 1) * nq
    psi = lambda q: phi * q**2

    delta_a = np.ones(nq) * 0.8; delta_a[0] = 0.0
    delta_b = np.ones(nq) * 0.8; delta_b[-1] = 0.0

    for it in range(max_iter):
        M = np.zeros((nq, nq))
        A = np.zeros(nq)
        avg_da = float(np.mean(delta_a)); avg_db = float(np.mean(delta_b))

        for j in range(nq):
            q = q_grid[j]
            fa = cx_exec_prob_np(delta_a[j], avg_da, K_i, N) if q > -Q else 0.0
            fb = cx_exec_prob_np(delta_b[j], avg_db, K_i, N) if q < Q else 0.0
            nu_a = lambda_a * fa
            nu_b = lambda_b * fb

            M[j, j] = r + nu_a + nu_b
            if q > -Q and j > 0: M[j, j-1] = -nu_a
            if q < Q and j < nq-1: M[j, j+1] = -nu_b

            A[j] = -psi(q)
            if q > -Q:
                # Ask revenue: (1-alpha_a) * delta + alpha_a * (delta - theta_a)
                eff_delta_a = delta_a[j] - alpha_a * theta_a
                A[j] += lambda_a * Delta * fa * eff_delta_a
            if q < Q:
                eff_delta_b = delta_b[j] - alpha_b * theta_b
                A[j] += lambda_b * Delta * fb * eff_delta_b

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
        if diff < 1e-8: break

    mid = nq // 2
    return {
        "alpha_a": alpha_a, "alpha_b": alpha_b, "theta_a": theta_a, "theta_b": theta_b,
        "V": V.tolist(), "delta_a": delta_a.tolist(), "delta_b": delta_b.tolist(),
        "spread_q0": float(delta_a[mid] + delta_b[mid]),
        "ask_q0": float(delta_a[mid]), "bid_q0": float(delta_b[mid]),
        "V_q0": float(V[mid]), "n_iter": it + 1,
    }


# =========================================================================
# EXPERIMENT 1: Dense alpha sweep — is it linear in alpha?
# =========================================================================
print(f"\n{'='*60}")
print("EXP 1: Dense alpha sweep (symmetric informed flow)")
print(f"{'='*60}", flush=True)

theta_fixed = 0.3
alpha_grid = np.linspace(0.0, 0.8, 17)  # 17 points
sym_results = []
for a in alpha_grid:
    r = adverse_fp(alpha_a=a, alpha_b=a, theta_a=theta_fixed, theta_b=theta_fixed)
    sym_results.append(r)
    print(f"  alpha={a:.3f}: spread={r['spread_q0']:.4f}, V(0)={r['V_q0']:.4f}", flush=True)

spreads = np.array([r['spread_q0'] for r in sym_results])
Vs = np.array([r['V_q0'] for r in sym_results])
nash_spread = spreads[0]
nash_V = Vs[0]

# Fit: linear vs quadratic
from numpy.polynomial import polynomial as P
coeffs_lin = np.polyfit(alpha_grid, spreads, 1)
coeffs_quad = np.polyfit(alpha_grid, spreads, 2)
residual_lin = np.sum((spreads - np.polyval(coeffs_lin, alpha_grid))**2)
residual_quad = np.sum((spreads - np.polyval(coeffs_quad, alpha_grid))**2)
print(f"\n  Linear fit:    spread = {coeffs_lin[0]:.4f}*alpha + {coeffs_lin[1]:.4f}")
print(f"  Quadratic fit: spread = {coeffs_quad[0]:.4f}*alpha^2 + {coeffs_quad[1]:.4f}*alpha + {coeffs_quad[2]:.4f}")
print(f"  SSE linear:    {residual_lin:.2e}")
print(f"  SSE quadratic: {residual_quad:.2e}")
print(f"  -> {'Linear' if residual_lin < 10 * residual_quad else 'Quadratic'} fit is acceptable")

# Naive premium check: dealer adds alpha*theta to their effective p. Does spread
# increase by 2 * alpha * theta (once per side)? Compare observed vs predicted.
naive_prediction = nash_spread + 2 * alpha_grid * theta_fixed
observed_excess = spreads - nash_spread
naive_excess = 2 * alpha_grid * theta_fixed
print(f"\n  Observed vs naive premium:")
print(f"  {'alpha':>6s}  {'observed':>10s}  {'naive 2*a*th':>14s}  {'ratio':>8s}")
for i in range(0, len(alpha_grid), 4):
    if alpha_grid[i] > 0:
        ratio = observed_excess[i] / naive_excess[i]
        print(f"  {alpha_grid[i]:6.3f}  {observed_excess[i]:10.4f}  {naive_excess[i]:14.4f}  {ratio:7.3f}")

# =========================================================================
# EXPERIMENT 2: Asymmetric informed flow — only ask side is informed
# =========================================================================
print(f"\n{'='*60}")
print("EXP 2: Asymmetric informed flow (only ask side, alpha_b=0)")
print(f"{'='*60}", flush=True)

asym_results = []
for a in [0.0, 0.2, 0.4, 0.6, 0.8]:
    r = adverse_fp(alpha_a=a, alpha_b=0.0, theta_a=theta_fixed, theta_b=theta_fixed)
    asym_results.append(r)
    print(f"  alpha_a={a:.2f}, alpha_b=0: ask(q=0)={r['ask_q0']:.4f}, bid(q=0)={r['bid_q0']:.4f}, "
          f"spread={r['spread_q0']:.4f}", flush=True)

# =========================================================================
# EXPERIMENT 3: Quote profile across inventory at different alpha
# =========================================================================
print(f"\n{'='*60}")
print("EXP 3: Quote profile vs inventory at different alpha")
print(f"{'='*60}", flush=True)

q_grid = np.arange(-5, 6, 1)
for target_alpha in [0.0, 0.3, 0.6]:
    # Find closest alpha in sym_results
    idx = np.argmin(np.abs(alpha_grid - target_alpha))
    r = sym_results[idx]
    print(f"\n  alpha = {alpha_grid[idx]:.3f}:")
    print(f"    {'q':>4s}  {'delta_a':>8s}  {'delta_b':>8s}  {'spread':>8s}")
    for j in range(len(q_grid)):
        s = r['delta_a'][j] + r['delta_b'][j]
        print(f"    {q_grid[j]:+4.0f}  {r['delta_a'][j]:8.4f}  {r['delta_b'][j]:8.4f}  {s:8.4f}")

# =========================================================================
# EXPERIMENT 4: Inventory sensitivity — does dealer become more risk-averse?
# =========================================================================
print(f"\n{'='*60}")
print("EXP 4: Inventory sensitivity: slope of spread vs |q|")
print(f"{'='*60}", flush=True)

# At each alpha, compute slope of spread vs |q| (how much wider at edges vs middle)
print(f"\n  {'alpha':>6s}  {'spread at q=0':>14s}  {'spread at |q|=3':>16s}  {'ratio':>8s}")
for idx in [0, 4, 8, 12, 16]:
    r = sym_results[idx]
    mid = 5
    s_mid = r['delta_a'][mid] + r['delta_b'][mid]
    s_edge = (r['delta_a'][mid-3] + r['delta_b'][mid-3] +
              r['delta_a'][mid+3] + r['delta_b'][mid+3]) / 2
    ratio = s_edge / s_mid
    print(f"  {alpha_grid[idx]:6.3f}  {s_mid:14.4f}  {s_edge:16.4f}  {ratio:7.4f}")

# =========================================================================
# EXPERIMENT 5: Dealer value degradation
# =========================================================================
print(f"\n{'='*60}")
print("EXP 5: Value degradation and welfare")
print(f"{'='*60}", flush=True)

V_coeffs_lin = np.polyfit(alpha_grid, Vs, 1)
V_coeffs_quad = np.polyfit(alpha_grid, Vs, 2)
print(f"\n  V(0) linear fit:    V = {V_coeffs_lin[0]:.4f}*alpha + {V_coeffs_lin[1]:.4f}")
print(f"  V(0) quadratic fit: V = {V_coeffs_quad[0]:.4f}*alpha^2 + {V_coeffs_quad[1]:.4f}*alpha + {V_coeffs_quad[2]:.4f}")
print(f"  Dealer loses {(Vs[0] - Vs[-1])/Vs[0]*100:.1f}% of value going from alpha=0 to alpha=0.8")

# =========================================================================
# Save everything
# =========================================================================
save_data = {
    "experiment_1_symmetric_alpha_sweep": {
        "alpha_grid": alpha_grid.tolist(),
        "spreads": spreads.tolist(),
        "V_q0": Vs.tolist(),
        "linear_fit_coeffs": coeffs_lin.tolist(),
        "quadratic_fit_coeffs": coeffs_quad.tolist(),
        "linear_sse": float(residual_lin),
        "quadratic_sse": float(residual_quad),
        "naive_premium_slope": 2 * theta_fixed,
        "observed_premium_slope": float(coeffs_lin[0]),
        "results": sym_results,
    },
    "experiment_2_asymmetric": {
        "description": "alpha_a varies, alpha_b=0 (informed only on ask)",
        "results": asym_results,
    },
    "theta": theta_fixed,
}

with open("results_final/adverse_selection_deep.json", "w") as f:
    json.dump(save_data, f, indent=2, default=float)
print(f"\nSaved to results_final/adverse_selection_deep.json", flush=True)
print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

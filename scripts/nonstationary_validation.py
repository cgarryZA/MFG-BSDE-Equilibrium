#!/usr/bin/env python -u
"""Validation: compare non-stationary phi(t) BSDEJ against stationary Nash at phi(t).

The non-stationary BSDEJ solver produced smooth spread(t) trajectory going from
1.22 at t=0 (phi=0.005) to 1.52 at t=T (phi=0.025). We need a sanity check:
at each t, the stationary Nash spread for the instantaneous phi(t) gives an
"adiabatic reference." The NN should:
  - At t=T: converge to stationary Nash at terminal phi (1.52 for phi=0.025 ~ 1.51)
  - At t=0: quote tighter than stationary (forward-looking)

If both hold, the BSDE's backward-induction structure is working correctly.

CPU, ~30 seconds.
"""

import sys, os, json
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.cont_xiong_exact import fictitious_play, policy_evaluation
from equations.contxiong_exact import cx_exec_prob_np, optimal_quote_foc

print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

# Load the NN trajectory
with open("results_final/nonstationary_phi.json") as f:
    nn = json.load(f)

tp = nn["time_profiles"]
print(f"NN trajectory (from non-stationary BSDEJ):")
for key, info in tp.items():
    phi = info["phi"]
    spread = info["profile"][5]["spread"]
    print(f"  {key}, phi={phi:.4f}: NN spread(q=0)={spread:.4f}", flush=True)

def stationary_nash_at_phi(phi_val, N=2, Q=5, Delta=1, lam=2.0, r=0.01,
                            max_iter=200, tol=1e-8):
    """Exact FP with a custom phi value."""
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)
    K_i = (N - 1) * nq
    psi = lambda q: phi_val * q ** 2
    delta_a = np.ones(nq); delta_a[0] = 0.0
    delta_b = np.ones(nq); delta_b[-1] = 0.0

    for it in range(max_iter):
        V = policy_evaluation(delta_a, delta_b, N, Q, Delta, lam, lam, r, psi)
        avg_a = float(np.mean(delta_a)); avg_b = float(np.mean(delta_b))
        new_a = np.zeros(nq); new_b = np.zeros(nq)
        for j in range(nq):
            q = q_grid[j]
            if j > 0 and q > -Q:
                p_a = (V[j] - V[j-1]) / Delta
                new_a[j] = optimal_quote_foc(p_a, avg_a, K_i, N)
            if j < nq-1 and q < Q:
                p_b = (V[j] - V[j+1]) / Delta
                new_b[j] = optimal_quote_foc(p_b, avg_b, K_i, N)
        damp = 0.5
        diff = max(np.max(np.abs(new_a - delta_a)), np.max(np.abs(new_b - delta_b)))
        delta_a = damp * new_a + (1 - damp) * delta_a
        delta_b = damp * new_b + (1 - damp) * delta_b
        if diff < tol: break
    mid = nq // 2
    return float(delta_a[mid] + delta_b[mid]), float(V[mid])


# Compute stationary Nash at each phi
print(f"\nStationary Nash reference (exact Algorithm 1 at each phi):")
nash_refs = []
for key, info in tp.items():
    phi = info["phi"]
    nash, V0 = stationary_nash_at_phi(phi)
    nash_refs.append({"t_key": key, "phi": phi, "nash_spread": nash, "V_q0": V0,
                     "nn_spread": info["profile"][5]["spread"]})
    print(f"  {key}, phi={phi:.4f}: stationary Nash={nash:.4f}, V(0)={V0:.4f}", flush=True)

# Compare
print(f"\n{'='*60}")
print("COMPARISON: NN trajectory vs stationary reference at each t")
print(f"{'='*60}")
print(f"{'t/T':>6s}  {'phi':>7s}  {'NN':>8s}  {'stat Nash':>10s}  {'NN-Nash':>9s}  {'interpretation':>25s}")
for r in nash_refs:
    t_key = r["t_key"]
    phi = r["phi"]
    nn_s = r["nn_spread"]
    nash = r["nash_spread"]
    diff = nn_s - nash
    if abs(diff) < 0.02:
        interp = "converges to stat Nash"
    elif diff < -0.02:
        interp = "tighter (forward-looking)"
    else:
        interp = "wider (terminal cost)"
    print(f"  {t_key:>6s}  {phi:7.4f}  {nn_s:8.4f}  {nash:10.4f}  {diff:+9.4f}  {interp:>25s}")

# Save
with open("results_final/nonstationary_validation.json", "w") as f:
    json.dump({"comparison": nash_refs}, f, indent=2, default=float)

# Interpretation
nn_t0 = nash_refs[0]["nn_spread"]
nash_t0 = nash_refs[0]["nash_spread"]
nn_tT = nash_refs[-1]["nn_spread"]
nash_tT = nash_refs[-1]["nash_spread"]

print(f"\n{'='*60}")
print("VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"  At t=0: NN quotes {nn_t0:.4f} vs stationary {nash_t0:.4f} "
      f"(NN {(nn_t0-nash_t0)/nash_t0*100:+.1f}%)")
print(f"  At t=T: NN quotes {nn_tT:.4f} vs stationary {nash_tT:.4f} "
      f"(NN {(nn_tT-nash_tT)/nash_tT*100:+.1f}%)")

if abs(nn_tT - nash_tT) < 0.02 and nn_t0 < nash_t0 - 0.05:
    print(f"\n  PASS: NN trajectory is consistent with backward induction.")
    print(f"  - terminal spread matches stationary Nash at phi(T)")
    print(f"  - t=0 quotes are tighter than stationary, indicating")
    print(f"    forward-looking accumulation before phi rises.")
else:
    print(f"\n  MIXED: Validation not fully consistent; may need more training.")

print(f"\nSaved to results_final/nonstationary_validation.json", flush=True)
print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

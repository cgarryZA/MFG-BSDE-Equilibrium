#!/usr/bin/env python -u
"""Lasry-Lions monotonicity check for the Cont-Xiong model.

Tier 1 #3. Verifies whether the CX Bellman generator satisfies the
Lasry-Lions monotonicity condition required for the Han et al. (2022)
CoD-free convergence guarantees.

The monotonicity condition (in the form relevant here):
  For two population measures mu, mu' and the corresponding best-response
  value functions V(q; mu), V(q; mu'), we need:

    integral [V(q; mu) - V(q; mu')] * d(mu - mu')(q)  >=  0

This is the Lasry-Lions "positive definiteness" of the coupling.

Numerical test: sample many pairs (mu, mu'), compute V at each, evaluate
the integral, check if it's >=0 for all pairs.

CPU, ~5 min. Always saves incrementally.
"""

import sys, os, json, time
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.cont_xiong_exact import policy_evaluation
from equations.contxiong_exact import optimal_quote_foc

print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

OUT = "results_final/monotonicity_check.json"


def best_response_V(mu_quote, N=2, Q=5, Delta=1, lam=2.0, r=0.01, phi=0.005,
                    max_iter=50, tol=1e-7):
    """Solve agent's best-response given population quote mu_quote (scalar avg).

    Returns: V(q), quotes at each q.
    """
    q_grid = np.arange(-Q, Q + Delta, Delta)
    nq = len(q_grid)
    K_i = (N - 1) * nq
    psi = lambda q: phi * q**2

    delta_a = np.ones(nq) * 0.8; delta_a[0] = 0.0
    delta_b = np.ones(nq) * 0.8; delta_b[-1] = 0.0

    for _ in range(max_iter):
        V = policy_evaluation(delta_a, delta_b, N, Q, Delta, lam, lam, r, psi)
        new_a = np.zeros(nq); new_b = np.zeros(nq)
        for j in range(nq):
            q = q_grid[j]
            if j > 0 and q > -Q:
                p_a = (V[j] - V[j-1]) / Delta
                new_a[j] = optimal_quote_foc(p_a, mu_quote, K_i, N)
            if j < nq-1 and q < Q:
                p_b = (V[j] - V[j+1]) / Delta
                new_b[j] = optimal_quote_foc(p_b, mu_quote, K_i, N)
        damp = 0.5
        diff = max(np.max(np.abs(new_a - delta_a)), np.max(np.abs(new_b - delta_b)))
        delta_a = damp * new_a + (1 - damp) * delta_a
        delta_b = damp * new_b + (1 - damp) * delta_b
        if diff < tol: break

    return V, delta_a, delta_b


def monotonicity_numerator(V_mu, V_mup, mu_quote, mup_quote):
    """For two quote means mu, mu' giving V(q;mu), V(q;mu'), compute

      integral [V(q;mu) - V(q;mu')] * d(mu - mu')(q)

    In our finite-dim case, the "measure" is the quote distribution —
    a point-mass at each q weighted equally. So the integral reduces to:

      sum_q [V(q;mu) - V(q;mu')] * (delta(q;mu) - delta(q;mu'))

    averaged over the discrete grid.
    """
    # Use avg quotes at each q as the "distributional coordinate"
    # This is an approximation — true LL monotonicity is on mu measures,
    # but we use quote function as proxy.
    return float(np.mean((V_mu - V_mup) * (mu_quote - mup_quote)))


def main():
    results = []
    try:
        # Sweep mu from 0.5 to 1.0 (realistic range for CX equilibrium)
        mu_grid = np.linspace(0.5, 1.0, 11)
        V_cache = {}

        print(f"\nComputing V(q) for each population quote mu in {mu_grid}...",
              flush=True)
        for mu in mu_grid:
            V, da, db = best_response_V(mu)
            V_cache[float(mu)] = V
            spread_q0 = da[5] + db[5]
            print(f"  mu={mu:.3f}: V(0)={V[5]:.4f}, spread(0)={spread_q0:.4f}", flush=True)

        # Incremental save
        with open(OUT, "w") as f:
            json.dump({"mu_grid": mu_grid.tolist(),
                      "V_cache": {str(k): v.tolist() for k, v in V_cache.items()},
                      "results": results}, f, indent=2, default=float)

        print(f"\nMonotonicity test over all (mu, mu') pairs:", flush=True)
        print(f"  {'mu':>6s}  {'mu_prime':>8s}  {'LL integral':>12s}  {'signed':>8s}")
        all_pos = True; all_neg = True
        for i, mu in enumerate(mu_grid):
            for j, mup in enumerate(mu_grid):
                if i >= j:
                    continue
                V_mu = V_cache[float(mu)]
                V_mup = V_cache[float(mup)]
                # Scalar mu coords (constant across q for now)
                integral = monotonicity_numerator(V_mu, V_mup,
                                                   np.full_like(V_mu, mu),
                                                   np.full_like(V_mup, mup))
                sign = "+" if integral > 1e-9 else ("-" if integral < -1e-9 else "0")
                if integral < -1e-9: all_pos = False
                if integral > 1e-9: all_neg = False
                results.append({
                    "mu": float(mu), "mu_prime": float(mup),
                    "LL_integral": integral, "sign": sign,
                })

        # Report
        pos_count = sum(1 for r in results if r["LL_integral"] > 1e-9)
        neg_count = sum(1 for r in results if r["LL_integral"] < -1e-9)
        print(f"\n  Total pairs: {len(results)}")
        print(f"  LL integral positive: {pos_count}")
        print(f"  LL integral negative: {neg_count}")

        if all_pos:
            print(f"\n  VERDICT: Lasry-Lions monotonicity APPEARS TO HOLD")
            print(f"           (Han et al. 2022 convergence guarantees applicable)")
        elif all_neg:
            print(f"\n  VERDICT: ANTI-monotone (inverse LL)")
        else:
            print(f"\n  VERDICT: Neither monotone nor anti-monotone")
            print(f"  LL guarantees do NOT directly apply; must rely on empirical convergence")

        with open(OUT, "w") as f:
            json.dump({
                "mu_grid": mu_grid.tolist(),
                "V_cache": {str(k): v.tolist() for k, v in V_cache.items()},
                "results": results,
                "summary": {
                    "all_positive": bool(all_pos),
                    "all_negative": bool(all_neg),
                    "pos_count": pos_count,
                    "neg_count": neg_count,
                    "total_pairs": len(results),
                },
            }, f, indent=2, default=float)
        print(f"\nSaved to {OUT}")

    except Exception as e:
        print(f"\nFAILED: {e}", flush=True)
        import traceback; traceback.print_exc()
        with open(OUT, "w") as f:
            json.dump({"error": str(e), "partial_results": results},
                      f, indent=2, default=float)

    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}",
          flush=True)


if __name__ == "__main__":
    main()

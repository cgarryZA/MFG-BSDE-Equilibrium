"""
Two key experiments for the dissertation:

1. BSDEJ convergence rate: vary M (time steps), show error ~ h^{1/2}
   Verifies Wang et al. (2023) Theorem 3.1 on the CX model.

2. Mean-field convergence: vary N, show spread(0) → limit as N → ∞
   Both exact solver AND neural solver, proving the NN captures the limit.

Run: python scripts/convergence_experiments.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os, json, time, gc
import numpy as np
import torch
from datetime import datetime

RESULTS_DIR = "results_convergence"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =====================================================================
# EXPERIMENT 1: Mean-field convergence (exact + neural)
# N = 2, 3, 5, 7, 10, 15, 20, 30, 50, 100
# =====================================================================

def experiment_meanfield():
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: Mean-field convergence")
    print("Exact solver + Neural solver across N")
    print(f"{'='*60}")

    from scripts.cont_xiong_exact import fictitious_play
    from types import SimpleNamespace
    from equations.contxiong_exact import ContXiongExact
    from solver_cx import CXSolver

    N_values = [2, 3, 5, 7, 10, 15, 20, 30, 50, 100]

    # --- Exact solver (fast, seconds each) ---
    print("\n--- Exact Algorithm 1 ---")
    exact_results = []
    for N in N_values:
        t0 = time.time()
        result = fictitious_play(N=N, Q=5, Delta=1)
        elapsed = time.time() - t0
        mid = len(result['V']) // 2
        spread = result['delta_a'][mid] + result['delta_b'][mid]
        print(f"  N={N:3d}: spread(0) = {spread:.6f}  [{elapsed:.1f}s]")
        sys.stdout.flush()
        exact_results.append({
            "N": N, "spread_q0": spread,
            "V_q0": result['V'][mid],
            "delta_a_q0": result['delta_a'][mid],
            "delta_b_q0": result['delta_b'][mid],
            "elapsed": elapsed,
        })

    # Mean-field limit estimate (extrapolate from large N)
    large_N_spreads = [r["spread_q0"] for r in exact_results if r["N"] >= 30]
    mf_limit = np.mean(large_N_spreads)
    print(f"\n  Estimated MF limit (avg of N>=30): {mf_limit:.6f}")

    # --- Neural Bellman solver (slower, minutes each) ---
    print("\n--- Neural Bellman Solver ---")
    N_neural = [2, 5, 10, 20, 50]  # subset for speed
    neural_results = []

    for N in N_neural:
        print(f"\n  Training N={N}...")
        sys.stdout.flush()
        gpu_reset()

        config = SimpleNamespace(
            lambda_a=2.0, lambda_b=2.0, discount_rate=0.01,
            Delta_q=1.0, q_max=5.0, phi=0.005, N_agents=N,
        )
        eqn = ContXiongExact(config)

        # Use CXSolver with self-consistent population
        solver = CXSolver(eqn, device=device, lr=1e-3, n_iter=5000, verbose=False)
        t0 = time.time()
        result = solver.train()
        elapsed = time.time() - t0

        mid = eqn.mid
        spread = result['delta_a'][mid] + result['delta_b'][mid]
        V_q0 = result['V'][mid]

        # Get exact for comparison
        exact_spread = [r["spread_q0"] for r in exact_results if r["N"] == N][0]
        error = abs(spread - exact_spread) / exact_spread * 100

        print(f"  N={N:3d}: neural={spread:.4f}, exact={exact_spread:.4f}, "
              f"error={error:.1f}%  [{elapsed:.0f}s]")
        sys.stdout.flush()

        neural_results.append({
            "N": N, "spread_q0_neural": spread,
            "spread_q0_exact": exact_spread,
            "error_pct": error,
            "V_q0": V_q0,
            "elapsed": elapsed,
        })

        del solver; gpu_reset()

    # Save
    save_data = {
        "exact": exact_results,
        "neural": neural_results,
        "mf_limit_estimate": mf_limit,
        "N_values_exact": N_values,
        "N_values_neural": N_neural,
    }
    out_path = os.path.join(RESULTS_DIR, "meanfield_convergence.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=float)
    print(f"\n  Saved to {out_path}")

    return save_data


# =====================================================================
# EXPERIMENT 2: BSDEJ convergence rate
# Fix T=10, vary M = 10, 20, 30, 50, 75, 100
# Measure error vs h = T/M, expect ~ h^{1/2}
# =====================================================================

def experiment_convergence_rate():
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: BSDEJ convergence rate")
    print("Vary M (time steps), measure error vs h = T/M")
    print(f"{'='*60}")

    from solver_cx_bsdej_shared import CXBSDEJShared

    T = 10.0
    M_values = [10, 20, 30, 50, 75]
    nash_spread = 1.5153  # ground truth from exact solver
    n_iter = 8000  # enough for convergence at each M

    results = []
    for M in M_values:
        h = T / M
        print(f"\n  M={M}, h={h:.3f}, n_iter={n_iter}")
        sys.stdout.flush()
        gpu_reset()

        solver = CXBSDEJShared(
            N=2, Q=5, Delta=1,
            T=T, M=M,
            lambda_a=2.0, lambda_b=2.0,
            r=0.01, phi=0.005,
            device=device,
            lr=5e-4,
            n_iter=n_iter,
            batch_size=512,
            hidden=128, n_layers=3,
        )

        # Warm-start from Bellman
        solver.warmstart_from_bellman(n_pretrain=1500)
        result = solver.train()

        spread = result["U_profile"][5]["spread"]
        error = abs(spread - nash_spread)
        rel_error = error / nash_spread

        print(f"  M={M}: spread={spread:.4f}, error={error:.4f}, "
              f"rel={rel_error:.3f}, loss={result['best_loss']:.2e}")
        sys.stdout.flush()

        results.append({
            "M": M, "h": h, "T": T,
            "spread_q0": spread,
            "abs_error": error,
            "rel_error": rel_error,
            "best_loss": result["best_loss"],
            "elapsed": result["elapsed"],
        })

        del solver; gpu_reset()

    # Fit log-log: error ~ C * h^alpha
    h_vals = np.array([r["h"] for r in results])
    errors = np.array([r["abs_error"] for r in results])
    # Only fit where error > 0
    mask = errors > 0
    if mask.sum() >= 2:
        log_h = np.log(h_vals[mask])
        log_e = np.log(errors[mask])
        alpha, log_C = np.polyfit(log_h, log_e, 1)
        C = np.exp(log_C)
        print(f"\n  Fitted: error ≈ {C:.4f} * h^{alpha:.3f}")
        print(f"  Wang et al. predicts: h^{{(1-ε)/2}} ≈ h^0.5")
        print(f"  Observed exponent: {alpha:.3f}")
    else:
        alpha = None
        print("\n  Not enough data points for fit")

    save_data = {
        "results": results,
        "fitted_exponent": alpha,
        "T": T,
        "nash_spread": nash_spread,
    }
    out_path = os.path.join(RESULTS_DIR, "convergence_rate.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=float)
    print(f"  Saved to {out_path}")

    return save_data


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    total_start = time.time()

    # Experiment 1: Mean-field convergence (exact is fast, neural ~30min)
    mf = experiment_meanfield()

    # Experiment 2: BSDEJ convergence rate (~2h for 5 M values)
    cr = experiment_convergence_rate()

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE in {total/3600:.1f} hours")
    print(f"{'='*60}")

    # Print summary table
    print("\n--- Mean-field convergence ---")
    print(f"{'N':>5s}  {'Exact':>8s}  {'Neural':>8s}  {'Error':>7s}")
    for nr in mf["neural"]:
        print(f"{nr['N']:5d}  {nr['spread_q0_exact']:8.4f}  "
              f"{nr['spread_q0_neural']:8.4f}  {nr['error_pct']:6.1f}%")

    print(f"\n--- BSDEJ convergence rate ---")
    print(f"{'M':>5s}  {'h':>6s}  {'Spread':>8s}  {'Error':>8s}")
    for r in cr["results"]:
        print(f"{r['M']:5d}  {r['h']:6.3f}  {r['spread_q0']:8.4f}  {r['abs_error']:8.4f}")
    if cr["fitted_exponent"]:
        print(f"\n  Fitted exponent: {cr['fitted_exponent']:.3f} "
              f"(theory: 0.5)")

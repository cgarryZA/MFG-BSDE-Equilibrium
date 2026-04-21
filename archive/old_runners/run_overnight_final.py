#!/usr/bin/env python -u
"""
Final overnight run: diffusion solver + Q-scaling robust.
Queue after run_everything.py finishes.

Run: python -u run_overnight_final.py
"""

import sys, os, json, time, gc
import numpy as np
import torch
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60, flush=True)


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# =====================================================================
# JOB 1: Diffusion BSDE solver (Z ≠ 0) — the big new extension
# =====================================================================
def job_diffusion():
    print(f"\n{'='*60}")
    print("JOB 1: Continuous Inventory BSDE (Z != 0)")
    print(f"{'='*60}", flush=True)

    gpu_reset()
    from solver_cx_bsde_diffusion import CXBSDEDiffusion

    solver = CXBSDEDiffusion(
        N=2, Q=5, Delta=1,
        T=10.0, M=50,
        lambda_a=2.0, lambda_b=2.0,
        r=0.01, phi=0.005,
        device=device,
        lr=5e-4,
        n_iter=10000,
        batch_size=512,
        hidden=128, n_layers=3,
    )

    solver.warmstart_from_bellman(n_pretrain=2000)
    print(flush=True)
    result = solver.train()

    spread = result["Z_profile"][5]["spread"]
    Z_q0 = result["Z_profile"][5]["Z"]
    print(f"\n  Spread at q=0: {spread:.4f} (Nash: 1.5153, error: {abs(spread-1.5153)/1.5153*100:.1f}%)")
    print(f"  Z at q=0: {Z_q0:.6f} (should be ~0)")

    with open("results_final/bsde_diffusion.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"  Saved to results_final/bsde_diffusion.json", flush=True)

    del solver; gpu_reset()
    return result


# =====================================================================
# JOB 2: Q-scaling robust (more FP iters, track convergence)
# =====================================================================
def job_q_robust():
    print(f"\n{'='*60}")
    print("JOB 2: Q-scaling robust (more iterations)")
    print(f"{'='*60}", flush=True)

    from scripts.cont_xiong_exact import fictitious_play
    from types import SimpleNamespace
    from equations.contxiong_exact import ContXiongExact
    from solver_cx import CXFictitiousPlay

    configs = [
        {"Q": 5,  "inner": 5000,  "outer": 20, "lr": 1e-3},
        {"Q": 10, "inner": 5000,  "outer": 20, "lr": 1e-3},
        {"Q": 20, "inner": 8000,  "outer": 25, "lr": 5e-4},
        {"Q": 20, "inner": 10000, "outer": 30, "lr": 3e-4},
        {"Q": 50, "inner": 12000, "outer": 40, "lr": 2e-4},
    ]

    results = []
    for cfg in configs:
        Q = cfg["Q"]
        print(f"\n  Q={Q}, inner={cfg['inner']}, outer={cfg['outer']}...", flush=True)
        gpu_reset()

        # Exact reference
        exact = fictitious_play(N=2, Q=Q, Delta=1, max_iter=100)
        mid = len(exact["V"]) // 2
        exact_spread = exact["delta_a"][mid] + exact["delta_b"][mid]

        # Neural
        config = SimpleNamespace(lambda_a=2.0, lambda_b=2.0, discount_rate=0.01,
                                 Delta_q=1.0, q_max=Q, phi=0.005, N_agents=2)
        eqn = ContXiongExact(config)

        try:
            fp = CXFictitiousPlay(eqn, device=device,
                                  outer_iter=cfg["outer"], inner_iter=cfg["inner"],
                                  lr=cfg["lr"], damping=0.5)
            t0 = time.time()
            result = fp.train()
            elapsed = time.time() - t0

            da = np.array(result["final_delta_a"])
            db = np.array(result["final_delta_b"])
            neural_spread = float(da[eqn.mid] + db[eqn.mid])
            error = abs(neural_spread - exact_spread) / exact_spread * 100

            print(f"  Q={Q}: exact={exact_spread:.4f}, neural={neural_spread:.4f}, "
                  f"error={error:.1f}%, time={elapsed/60:.0f}m", flush=True)

            results.append({
                "Q": Q, "exact_spread": float(exact_spread),
                "neural_spread": neural_spread, "error_pct": float(error),
                "inner": cfg["inner"], "outer": cfg["outer"], "lr": cfg["lr"],
                "elapsed": elapsed,
            })
            del fp
        except Exception as e:
            print(f"  Q={Q} FAILED: {e}", flush=True)
            results.append({"Q": Q, "error": str(e), **cfg})

        del eqn; gpu_reset()

    with open("results_final/q_scaling_robust.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Saved to results_final/q_scaling_robust.json", flush=True)

    # Summary
    print(f"\n  {'Q':>5s}  {'inner':>6s}  {'outer':>6s}  {'exact':>8s}  {'neural':>8s}  {'error':>7s}")
    for r in results:
        if "neural_spread" in r:
            print(f"  {r['Q']:5d}  {r['inner']:6d}  {r['outer']:6d}  "
                  f"{r['exact_spread']:8.4f}  {r['neural_spread']:8.4f}  {r['error_pct']:6.1f}%", flush=True)

    return results


# =====================================================================
# JOB 3: N-convergence rate fit (quick, uses existing data)
# =====================================================================
def job_n_rate():
    print(f"\n{'='*60}")
    print("JOB 3: N->inf convergence rate fit")
    print(f"{'='*60}", flush=True)

    from scipy.optimize import curve_fit

    mf = json.load(open("results_final/mf_exact.json"))
    N = np.array([r["N"] for r in mf["results"]])
    spreads = np.array([r["spread_q0"] for r in mf["results"]])

    # Fit: spread(N) = a + b/sqrt(N) (Buckdahn et al. prediction)
    def model_sqrt(N, a, b):
        return a + b / np.sqrt(N)

    def model_inv(N, a, b):
        return a + b / N

    def model_pow(N, a, b, alpha):
        return a + b / N**alpha

    popt_sqrt, _ = curve_fit(model_sqrt, N, spreads)
    popt_inv, _ = curve_fit(model_inv, N, spreads)
    popt_pow, _ = curve_fit(model_pow, N, spreads, p0=[2.0, -1.0, 0.5])

    rmse_sqrt = np.sqrt(np.mean((spreads - model_sqrt(N, *popt_sqrt))**2))
    rmse_inv = np.sqrt(np.mean((spreads - model_inv(N, *popt_inv))**2))
    rmse_pow = np.sqrt(np.mean((spreads - model_pow(N, *popt_pow))**2))

    print(f"  1/sqrt(N) fit: a={popt_sqrt[0]:.4f}, b={popt_sqrt[1]:.4f}, RMSE={rmse_sqrt:.6f}")
    print(f"  1/N fit:   a={popt_inv[0]:.4f}, b={popt_inv[1]:.4f}, RMSE={rmse_inv:.6f}")
    print(f"  1/N^alpha fit: a={popt_pow[0]:.4f}, b={popt_pow[1]:.4f}, alpha={popt_pow[2]:.4f}, RMSE={rmse_pow:.6f}")
    print(f"  Best fit: 1/N^{popt_pow[2]:.3f}", flush=True)

    result = {
        "sqrt_N": {"a": float(popt_sqrt[0]), "b": float(popt_sqrt[1]), "rmse": float(rmse_sqrt)},
        "inv_N": {"a": float(popt_inv[0]), "b": float(popt_inv[1]), "rmse": float(rmse_inv)},
        "pow_N": {"a": float(popt_pow[0]), "b": float(popt_pow[1]),
                  "alpha": float(popt_pow[2]), "rmse": float(rmse_pow)},
        "mf_limit": float(popt_pow[0]),
    }

    with open("results_final/n_convergence_rate.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to results_final/n_convergence_rate.json", flush=True)
    return result


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    os.makedirs("results_final", exist_ok=True)
    total_start = time.time()

    # Job 3 is fast (no GPU), do it first
    try:
        job_n_rate()
    except Exception as e:
        print(f"  N-rate FAILED: {e}", flush=True)

    # Job 1: Diffusion solver (~2.5h)
    try:
        job_diffusion()
    except Exception as e:
        print(f"  Diffusion FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()

    # Job 2: Q-scaling (~4h)
    try:
        job_q_robust()
    except Exception as e:
        print(f"  Q-scaling FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total/3600:.1f} hours")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}", flush=True)

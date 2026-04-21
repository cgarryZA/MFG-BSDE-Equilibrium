"""
Overnight GPU run v2.

Jobs:
  1. BSDEJ shared-weight solver (T=10, M=50, 10k iters)    ~40 min
  2. BSDEJ shared-weight solver (T=20, M=50, 10k iters)    ~40 min
  3. BSDEJ shared-weight solver (T=10, M=20, 15k iters)    ~30 min
  4. Q=20 neural Bellman + FP                               ~1 hr
  5. Q=50 neural Bellman + FP                               ~1.5 hr

Run: python run_overnight_v2.py
"""

import os, json, time, gc
import numpy as np
import torch
from datetime import datetime

RESULTS_DIR = "results_overnight"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_bsdej_shared(T, M, n_iter, hidden, n_layers, lr, label):
    print(f"\n{'='*60}")
    print(f"JOB: BSDEJ shared-weight T={T}, M={M}, {n_iter} iters [{label}]")
    print(f"{'='*60}")

    gpu_reset()
    from solver_cx_bsdej_shared import CXBSDEJShared

    solver = CXBSDEJShared(
        N=2, Q=5, Delta=1,
        T=T, M=M,
        lambda_a=2.0, lambda_b=2.0,
        r=0.01, phi=0.005,
        device=device,
        lr=lr,
        n_iter=n_iter,
        batch_size=512,
        hidden=hidden,
        n_layers=n_layers,
    )

    result = solver.train()

    spread_q0 = result["U_profile"][5]["spread"]
    spread_mid = result["U_profile_mid"][5]["spread"]
    nash = 1.5153
    error = abs(spread_q0 - nash) / nash
    print(f"\n  Spread(q=0): t=0: {spread_q0:.4f}, t=T/2: {spread_mid:.4f}")
    print(f"  Nash: {nash:.4f}, error: {error:.1%}")
    print(f"  Time invariance: {abs(spread_q0 - spread_mid):.4f}")
    print(f"  Best loss: {result['best_loss']:.4e}")
    print(f"  Time: {result['elapsed']:.0f}s")

    out_path = os.path.join(RESULTS_DIR, f"bsdej_shared_{label}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"  Saved to {out_path}")

    del solver
    gpu_reset()
    return result


def run_q_scaling(Q_val, n_inner, n_outer):
    print(f"\n{'='*60}")
    print(f"JOB: Q={Q_val} neural Bellman + Fictitious Play")
    print(f"{'='*60}")

    gpu_reset()
    from types import SimpleNamespace
    from equations.contxiong_exact import ContXiongExact
    from solver_cx import CXFictitiousPlay

    config = SimpleNamespace(
        lambda_a=2.0, lambda_b=2.0,
        discount_rate=0.01, Delta_q=1.0,
        q_max=Q_val, phi=0.005, N_agents=2,
    )
    eqn = ContXiongExact(config)
    print(f"  Grid: {eqn.nq} inventory levels, K={eqn.K}")

    fp = CXFictitiousPlay(
        eqn, device=device,
        outer_iter=n_outer,
        inner_iter=n_inner,
        lr=5e-4,
        damping=0.5,
    )

    start = time.time()
    result = fp.train()
    elapsed = time.time() - start

    mid = eqn.mid
    spread_q0 = result["delta_a"][mid] + result["delta_b"][mid]
    print(f"\n  Q={Q_val}: spread(0) = {spread_q0:.4f}")
    print(f"  Time: {elapsed:.0f}s")

    save_data = {
        "Q": Q_val, "nq": eqn.nq, "spread_q0": spread_q0,
        "delta_a": [float(x) for x in result["delta_a"]],
        "delta_b": [float(x) for x in result["delta_b"]],
        "V": [float(x) for x in result["V"]],
        "elapsed": elapsed,
    }

    out_path = os.path.join(RESULTS_DIR, f"q_scaling_Q{Q_val}.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved to {out_path}")

    del fp, eqn
    gpu_reset()
    return save_data


if __name__ == "__main__":
    all_results = {}
    total_start = time.time()

    # --- BSDEJ shared: T=10, M=50, 10k iters ---
    try:
        r = run_bsdej_shared(T=10, M=50, n_iter=10000,
                             hidden=128, n_layers=3, lr=1e-3,
                             label="T10_M50")
        all_results["bsdej_shared_T10_M50"] = {
            "spread_q0": r["U_profile"][5]["spread"],
            "best_loss": r["best_loss"],
            "elapsed": r["elapsed"],
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["bsdej_shared_T10_M50"] = {"error": str(e)}

    # --- BSDEJ shared: T=20, M=50, 10k iters ---
    try:
        r = run_bsdej_shared(T=20, M=50, n_iter=10000,
                             hidden=128, n_layers=3, lr=1e-3,
                             label="T20_M50")
        all_results["bsdej_shared_T20_M50"] = {
            "spread_q0": r["U_profile"][5]["spread"],
            "best_loss": r["best_loss"],
            "elapsed": r["elapsed"],
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["bsdej_shared_T20_M50"] = {"error": str(e)}

    # --- BSDEJ shared: T=10, M=20, 15k iters (fewer steps, more training) ---
    try:
        r = run_bsdej_shared(T=10, M=20, n_iter=15000,
                             hidden=128, n_layers=3, lr=1e-3,
                             label="T10_M20")
        all_results["bsdej_shared_T10_M20"] = {
            "spread_q0": r["U_profile"][5]["spread"],
            "best_loss": r["best_loss"],
            "elapsed": r["elapsed"],
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["bsdej_shared_T10_M20"] = {"error": str(e)}

    # --- Q=20 Bellman + FP ---
    try:
        r = run_q_scaling(Q_val=20, n_inner=5000, n_outer=15)
        all_results["q20"] = {
            "spread_q0": r["spread_q0"],
            "elapsed": r["elapsed"],
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["q20"] = {"error": str(e)}

    # --- Q=50 Bellman + FP ---
    try:
        r = run_q_scaling(Q_val=50, n_inner=8000, n_outer=15)
        all_results["q50"] = {
            "spread_q0": r["spread_q0"],
            "elapsed": r["elapsed"],
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["q50"] = {"error": str(e)}

    # --- Summary ---
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL JOBS COMPLETE")
    print(f"Total time: {total_elapsed/3600:.1f} hours")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    nash = 1.5153
    print("\n--- Summary ---")
    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name}: FAILED — {res['error']}")
        elif "spread_q0" in res:
            s = res["spread_q0"]
            err = abs(s - nash) / nash * 100
            t = res['elapsed']
            print(f"  {name}: spread(0)={s:.4f} ({err:.1f}% from Nash) [{t/60:.0f} min]")

    all_results["total_elapsed"] = total_elapsed
    summary_path = os.path.join(RESULTS_DIR, "summary_v2.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSummary saved to {summary_path}")

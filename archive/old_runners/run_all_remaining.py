#!/usr/bin/env python -u
"""
All remaining experiments. Unbuffered output throughout.

Jobs:
  1. N=5 MADDPG collusion (5 seeds, correct hyperparams)   ~2.5h
  2. Q=20 neural Bellman + FP                               ~1h
  3. Q=50 neural Bellman + FP                               ~1.5h
  4. Mean-field convergence (exact N=2..100 + neural N=2..50) ~30min
  5. BSDEJ convergence rate (M=10,20,30,50,75)              ~2.5h

Total: ~8 hours

Run: python -u run_all_remaining.py
"""

import sys, os, json, time, gc
import numpy as np
import torch
from datetime import datetime

# Force unbuffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS_DIR = "results_final"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60, flush=True)


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =====================================================================
# JOB 1: N=5 MADDPG (correct hyperparams matching cluster script)
# =====================================================================

def job_maddpg_n5():
    print(f"\n{'='*60}")
    print("JOB 1: N=5 MADDPG collusion (5 seeds)")
    print(f"{'='*60}", flush=True)

    from solver_cx_multiagent import MADDPGTrainer
    from scripts.cont_xiong_exact import fictitious_play

    # Get N=5 Nash reference (NOT N=2)
    nash_result = fictitious_play(N=5, Q=5, max_iter=50)
    mid = len(nash_result["q_grid"]) // 2
    nash_spread = nash_result["delta_a"][mid] + nash_result["delta_b"][mid]
    print(f"  N=5 Nash spread: {nash_spread:.4f}", flush=True)

    results = []
    for seed in range(5):
        print(f"\n  --- Seed {seed} ---", flush=True)
        gpu_reset()

        torch.manual_seed(seed * 7 + 13)
        np.random.seed(seed * 7 + 13)

        # Match cluster hyperparams: lower lr_actor, lower tau
        trainer = MADDPGTrainer(
            N=5, Q=5, device=device,
            lr_actor=1e-4, lr_critic=1e-3, tau=0.001,
            n_episodes=500, steps_per_episode=500,
            batch_size=32,
        )
        r = trainer.train()

        final_spread = r["avg_final_spread"]
        above = final_spread > nash_spread

        print(f"  Seed {seed}: spread={final_spread:.3f}, "
              f"nash={nash_spread:.3f}, {'ABOVE' if above else 'BELOW'}", flush=True)

        results.append({
            "seed": seed,
            "final_spread": final_spread,
            "nash_spread": nash_spread,
            "above_nash": above,
            "history": r["history"],
            "elapsed": r["elapsed"],
        })

        del trainer; gpu_reset()

    spreads = [r["final_spread"] for r in results]
    n_above = sum(1 for r in results if r["above_nash"])
    print(f"\n  N=5 Summary: {n_above}/{len(results)} above Nash", flush=True)
    print(f"  Mean spread: {np.mean(spreads):.4f} (Nash: {nash_spread:.4f})", flush=True)

    path = os.path.join(RESULTS_DIR, "maddpg_N5.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  Saved to {path}", flush=True)
    return results


# =====================================================================
# JOB 2-3: Q-scaling (neural Bellman + FP)
# =====================================================================

def job_q_scaling(Q_val, n_inner, n_outer):
    print(f"\n{'='*60}")
    print(f"JOB: Q={Q_val} neural Bellman + FP")
    print(f"{'='*60}", flush=True)

    gpu_reset()
    from types import SimpleNamespace
    from equations.contxiong_exact import ContXiongExact
    from solver_cx import CXFictitiousPlay

    config = SimpleNamespace(
        lambda_a=2.0, lambda_b=2.0, discount_rate=0.01,
        Delta_q=1.0, q_max=Q_val, phi=0.005, N_agents=2,
    )
    eqn = ContXiongExact(config)
    print(f"  Grid: {eqn.nq} levels, K={eqn.K}", flush=True)

    fp = CXFictitiousPlay(
        eqn, device=device,
        outer_iter=n_outer, inner_iter=n_inner,
        lr=5e-4, damping=0.5,
    )
    t0 = time.time()
    result = fp.train()
    elapsed = time.time() - t0

    da = np.array(result["final_delta_a"])
    db = np.array(result["final_delta_b"])
    spread_q0 = da[eqn.mid] + db[eqn.mid]
    print(f"  Q={Q_val}: spread(0)={spread_q0:.4f}, time={elapsed:.0f}s", flush=True)

    save = {
        "Q": Q_val, "nq": eqn.nq, "spread_q0": float(spread_q0),
        "delta_a": da.tolist(), "delta_b": db.tolist(),
        "V": [float(x) for x in result["final_V"]],
        "elapsed": elapsed,
    }
    path = os.path.join(RESULTS_DIR, f"q_scaling_Q{Q_val}.json")
    with open(path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"  Saved to {path}", flush=True)

    del fp, eqn; gpu_reset()
    return save


# =====================================================================
# JOB 4: Mean-field convergence (exact + neural)
# =====================================================================

def job_meanfield():
    print(f"\n{'='*60}")
    print("JOB 4: Mean-field convergence (exact + neural)")
    print(f"{'='*60}", flush=True)

    from scripts.cont_xiong_exact import fictitious_play
    from types import SimpleNamespace
    from equations.contxiong_exact import ContXiongExact
    from solver_cx import CXSolver

    N_exact = [2, 3, 5, 7, 10, 15, 20, 30, 50, 100]
    N_neural = [2, 5, 10, 20, 50]

    # Exact
    print("\n  --- Exact Algorithm 1 ---", flush=True)
    exact = []
    for N in N_exact:
        t0 = time.time()
        r = fictitious_play(N=N, Q=5, Delta=1)
        mid = len(r["V"]) // 2
        spread = r["delta_a"][mid] + r["delta_b"][mid]
        print(f"  N={N:3d}: spread(0)={spread:.6f} [{time.time()-t0:.1f}s]", flush=True)
        exact.append({"N": N, "spread_q0": spread})

    mf_limit = np.mean([e["spread_q0"] for e in exact if e["N"] >= 30])
    print(f"  MF limit estimate (N>=30 avg): {mf_limit:.6f}", flush=True)

    # Neural
    print("\n  --- Neural Bellman Solver ---", flush=True)
    neural = []
    for N in N_neural:
        print(f"\n  Training N={N}...", flush=True)
        gpu_reset()

        config = SimpleNamespace(
            lambda_a=2.0, lambda_b=2.0, discount_rate=0.01,
            Delta_q=1.0, q_max=5.0, phi=0.005, N_agents=N,
        )
        eqn = ContXiongExact(config)
        solver = CXSolver(eqn, device=device, lr=1e-3, n_iter=5000, verbose=False)
        t0 = time.time()
        result = solver.train()
        elapsed = time.time() - t0

        mid = eqn.mid
        spread_nn = result["delta_a"][mid] + result["delta_b"][mid]
        spread_exact = [e["spread_q0"] for e in exact if e["N"] == N][0]
        error = abs(spread_nn - spread_exact) / spread_exact * 100

        print(f"  N={N:3d}: neural={spread_nn:.4f}, exact={spread_exact:.4f}, "
              f"error={error:.1f}% [{elapsed:.0f}s]", flush=True)

        neural.append({
            "N": N, "spread_nn": float(spread_nn),
            "spread_exact": float(spread_exact),
            "error_pct": float(error), "elapsed": elapsed,
        })
        del solver; gpu_reset()

    save = {"exact": exact, "neural": neural, "mf_limit": float(mf_limit)}
    path = os.path.join(RESULTS_DIR, "meanfield_convergence.json")
    with open(path, "w") as f:
        json.dump(save, f, indent=2, default=float)
    print(f"  Saved to {path}", flush=True)
    return save


# =====================================================================
# JOB 5: BSDEJ convergence rate
# =====================================================================

def job_convergence_rate():
    print(f"\n{'='*60}")
    print("JOB 5: BSDEJ convergence rate (vary M)")
    print(f"{'='*60}", flush=True)

    from solver_cx_bsdej_shared import CXBSDEJShared

    T = 10.0
    M_values = [10, 20, 30, 50, 75]
    nash = 1.5153
    n_iter = 8000

    results = []
    for M in M_values:
        h = T / M
        print(f"\n  M={M}, h={h:.3f}", flush=True)
        gpu_reset()

        solver = CXBSDEJShared(
            N=2, Q=5, Delta=1, T=T, M=M,
            lambda_a=2.0, lambda_b=2.0, r=0.01, phi=0.005,
            device=device, lr=5e-4, n_iter=n_iter,
            batch_size=512, hidden=128, n_layers=3,
        )
        solver.warmstart_from_bellman(n_pretrain=1500)
        result = solver.train()

        spread = result["U_profile"][5]["spread"]
        error = abs(spread - nash)
        print(f"  M={M}: spread={spread:.4f}, abs_error={error:.4f}, "
              f"loss={result['best_loss']:.2e}", flush=True)

        results.append({
            "M": M, "h": h, "spread_q0": float(spread),
            "abs_error": float(error), "rel_error": float(error / nash),
            "best_loss": float(result["best_loss"]),
            "elapsed": result["elapsed"],
        })
        del solver; gpu_reset()

    # Fit error ~ C * h^alpha
    h_arr = np.array([r["h"] for r in results])
    err_arr = np.array([r["abs_error"] for r in results])
    mask = err_arr > 1e-6
    if mask.sum() >= 2:
        alpha, logC = np.polyfit(np.log(h_arr[mask]), np.log(err_arr[mask]), 1)
        print(f"\n  Fitted: error ~ h^{alpha:.3f} (Wang et al. predicts ~h^0.5)", flush=True)
    else:
        alpha = None

    save = {"results": results, "fitted_exponent": alpha, "T": T, "nash": nash}
    path = os.path.join(RESULTS_DIR, "convergence_rate.json")
    with open(path, "w") as f:
        json.dump(save, f, indent=2, default=float)
    print(f"  Saved to {path}", flush=True)
    return save


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    total_start = time.time()
    summary = {}

    for name, fn in [
        ("maddpg_N5", job_maddpg_n5),
        ("q20", lambda: job_q_scaling(20, 5000, 15)),
        ("q50", lambda: job_q_scaling(50, 8000, 15)),
        ("meanfield", job_meanfield),
        ("convergence_rate", job_convergence_rate),
    ]:
        try:
            r = fn()
            summary[name] = "done"
        except Exception as e:
            print(f"\n  {name} FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()
            summary[name] = {"error": str(e)}

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total/3600:.1f} hours")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(json.dumps(summary, indent=2))
    print(f"{'='*60}", flush=True)

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

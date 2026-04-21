"""
Tonight's GPU run:
  1. Non-stationary BSDEJ (time-varying lambda) — Paper 1 extension
  2. N=5 MADDPG collusion — Paper 2 extension
  3. Warm-started BSDEJ T=20 (better than T=10?)

Run: python run_tonight.py
Estimated: ~6-8 hours
"""

import os, json, time, gc, sys
import numpy as np
import torch
from datetime import datetime

# Force unbuffered output so we can monitor progress
sys.stdout.reconfigure(line_buffering=True)

RESULTS_DIR = "results_tonight"
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


# =====================================================================
# JOB 1: Warm-started BSDEJ T=20 (does longer horizon help?)
# =====================================================================

def run_warmstart_bsdej(T, M, n_iter, label):
    print(f"\n{'='*60}")
    print(f"JOB 1: Warm-started BSDEJ T={T}, M={M}, {n_iter} iters")
    print(f"{'='*60}")
    sys.stdout.flush()

    gpu_reset()
    from solver_cx_bsdej_shared import CXBSDEJShared

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
    solver.warmstart_from_bellman(n_pretrain=2000)
    result = solver.train()

    spread = result["U_profile"][5]["spread"]
    print(f"\n  Spread(q=0): {spread:.4f} (Nash: 1.5153, error: {abs(spread-1.5153)/1.5153*100:.1f}%)")
    sys.stdout.flush()

    out_path = os.path.join(RESULTS_DIR, f"bsdej_warmstart_{label}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)

    del solver; gpu_reset()
    return result


# =====================================================================
# JOB 2: Non-stationary BSDEJ (time-varying lambda)
# This is the Paper 1 contribution — Bellman solver CAN'T do this.
# lambda_a(t) = lambda_0 * (1 + amplitude * sin(2*pi*t/T))
# Models intraday activity: high at open/close, low at midday.
# =====================================================================

def run_nonstationary_bsdej():
    print(f"\n{'='*60}")
    print(f"JOB 2: Non-stationary BSDEJ (time-varying lambda)")
    print(f"{'='*60}")
    sys.stdout.flush()

    gpu_reset()

    # We need to modify the shared solver to accept time-varying lambda.
    # The key change: in the forward pass, lambda_a and lambda_b are
    # functions of t instead of constants.

    from solver_cx_bsdej_shared import CXBSDEJShared, SharedJumpNet
    from solver_cx_bsdej import _exec_prob_torch_vec, optimal_quotes_vectorised
    from equations.contxiong_exact import cx_exec_prob_np
    import torch.nn as nn

    class CXBSDEJNonStationary(CXBSDEJShared):
        """BSDEJ solver with time-varying arrival rates.

        lambda_a(t) = lambda_0 * (1 + amplitude * sin(2*pi*t/T))
        lambda_b(t) = lambda_0 * (1 + amplitude * sin(2*pi*t/T))

        This models intraday patterns in RFQ arrival intensity.
        The stationary Bellman solver CANNOT handle this.
        """

        def __init__(self, amplitude=0.5, **kwargs):
            super().__init__(**kwargs)
            self.amplitude = amplitude
            self.lambda_0_a = self.lambda_a
            self.lambda_0_b = self.lambda_b

        def lambda_t(self, t_frac):
            """Time-varying arrival rate. t_frac in [0, 1]."""
            modulation = 1.0 + self.amplitude * np.sin(2 * np.pi * t_frac)
            return self.lambda_0_a * modulation

        def sample_paths(self, batch_size):
            """Sample paths with time-varying arrival rates."""
            q = np.zeros(batch_size)
            q_paths = np.zeros((batch_size, self.M + 1))
            exec_a_all = np.zeros((batch_size, self.M))
            exec_b_all = np.zeros((batch_size, self.M))
            q_paths[:, 0] = q

            for m in range(self.M):
                t_frac = m / self.M
                lam = self.lambda_t(t_frac)

                delta_a_proxy = np.clip(0.8 - 0.05 * q, 0.1, 3.0)
                delta_b_proxy = np.clip(0.8 + 0.05 * q, 0.1, 3.0)

                rate_a = np.array([cx_exec_prob_np(da, self.avg_comp, self.K, self.N)
                                  for da in delta_a_proxy]) * lam
                rate_b = np.array([cx_exec_prob_np(db, self.avg_comp, self.K, self.N)
                                  for db in delta_b_proxy]) * lam

                prob_a = np.clip(rate_a * self.dt, 0, 0.5)
                prob_b = np.clip(rate_b * self.dt, 0, 0.5)
                exec_a = (np.random.uniform(size=batch_size) < prob_a).astype(float)
                exec_b = (np.random.uniform(size=batch_size) < prob_b).astype(float)

                exec_a_all[:, m] = exec_a
                exec_b_all[:, m] = exec_b

                q = q - exec_a * self.Delta + exec_b * self.Delta
                q = np.clip(q, -self.Q, self.Q)
                q_paths[:, m + 1] = q

            return q_paths, exec_a_all, exec_b_all

        def forward(self, q_paths, exec_a_all, exec_b_all):
            """Forward pass with time-varying lambda."""
            batch = q_paths.shape[0]
            dev = self.device
            dtype = torch.float64

            q0 = torch.tensor(q_paths[:, 0], dtype=dtype, device=dev).long()
            q0_idx = (q0 + self.Q).clamp(0, self.nq - 1)
            Y = self.Y0[q0_idx].unsqueeze(1)

            for m in range(self.M):
                t_frac = m / self.M
                lam = self.lambda_t(t_frac)

                t_norm = torch.full((batch, 1), t_frac, dtype=dtype, device=dev)
                q_m_raw = torch.tensor(q_paths[:, m], dtype=dtype, device=dev)
                q_m_norm = (q_m_raw / self.Q).unsqueeze(1)

                U = self.shared_net(t_norm, q_m_norm)
                Ua = U[:, 0:1]
                Ub = U[:, 1:2]

                da_t = optimal_quotes_vectorised(Ua, self.avg_comp, self.K, self.N)
                db_t = optimal_quotes_vectorised(Ub, self.avg_comp, self.K, self.N)

                # Time-varying rates
                fa = _exec_prob_torch_vec(da_t, self.avg_comp, self.K, self.N) * lam
                fb = _exec_prob_torch_vec(db_t, self.avg_comp, self.K, self.N) * lam

                can_sell = (q_m_raw > -self.Q).float().unsqueeze(1)
                can_buy = (q_m_raw < self.Q).float().unsqueeze(1)

                profit_a = can_sell * fa * (da_t * self.Delta + Ua)
                profit_b = can_buy * fb * (db_t * self.Delta + Ub)
                psi_q = self.phi * q_m_raw.unsqueeze(1) ** 2

                f_val = self.r * Y + psi_q - profit_a - profit_b

                dN_a = torch.tensor(exec_a_all[:, m].reshape(-1, 1), dtype=dtype, device=dev)
                dN_b = torch.tensor(exec_b_all[:, m].reshape(-1, 1), dtype=dtype, device=dev)

                nu_a = can_sell * fa
                nu_b = can_buy * fb

                jump_a = can_sell * Ua * (dN_a - nu_a * self.dt)
                jump_b = can_buy * Ub * (dN_b - nu_b * self.dt)

                Y = Y - f_val * self.dt + jump_a + jump_b

            return Y

    # Run with amplitude = 0.5 (lambda varies from 1.0 to 3.0)
    solver = CXBSDEJNonStationary(
        amplitude=0.5,
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

    # Warm-start from stationary solution
    solver.warmstart_from_bellman(n_pretrain=2000)
    print()

    result = solver.train()

    # Extract quotes at multiple time points to show time variation
    from scipy.optimize import minimize_scalar
    q_grid = np.arange(-solver.Q, solver.Q + solver.Delta, solver.Delta)
    time_profiles = {}

    solver.shared_net.eval()
    for t_frac in [0.0, 0.25, 0.5, 0.75]:
        lam = solver.lambda_t(t_frac)
        profile = []
        with torch.no_grad():
            for q in q_grid:
                t_n = torch.tensor([[t_frac]], dtype=torch.float64, device=device)
                q_n = torch.tensor([[q / solver.Q]], dtype=torch.float64, device=device)
                U = solver.shared_net(t_n, q_n)
                Ua_v = U[0, 0].item()
                Ub_v = U[0, 1].item()

                def _neg(d, Uv):
                    f = cx_exec_prob_np(d, solver.avg_comp, solver.K, solver.N)
                    return -f * (d + Uv)

                da = minimize_scalar(lambda d: _neg(d, Ua_v), bounds=(-1, 8), method='bounded').x
                db = minimize_scalar(lambda d: _neg(d, Ub_v), bounds=(-1, 8), method='bounded').x
                profile.append({"q": float(q), "da": da, "db": db, "spread": da + db})

        time_profiles[f"t={t_frac:.2f}_lam={lam:.2f}"] = profile
        s0 = profile[5]["spread"]
        print(f"  t/T={t_frac:.2f}, lambda={lam:.2f}: spread(q=0)={s0:.4f}")

    result["time_profiles"] = time_profiles
    result["amplitude"] = solver.amplitude

    out_path = os.path.join(RESULTS_DIR, "bsdej_nonstationary.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"  Saved to {out_path}")
    sys.stdout.flush()

    del solver; gpu_reset()
    return result


# =====================================================================
# JOB 3: N=5 MADDPG collusion
# Does tacit collusion survive with more competitors?
# =====================================================================

def run_maddpg_n5(n_seeds=5):
    print(f"\n{'='*60}")
    print(f"JOB 3: N=5 MADDPG collusion ({n_seeds} seeds)")
    print(f"{'='*60}")
    sys.stdout.flush()

    gpu_reset()
    from solver_cx_multiagent import MADDPGTrainer

    results = []
    for seed in range(n_seeds):
        print(f"\n  --- Seed {seed} ---")
        sys.stdout.flush()

        gpu_reset()
        torch.manual_seed(seed)
        np.random.seed(seed)

        trainer = MADDPGTrainer(
            N=5, Q=5, device=device,
        )

        result = trainer.train()
        results.append(result)

        spread = result["final_spread"]
        nash = result["nash_spread"]
        above = "ABOVE" if result["above_nash"] else "BELOW"
        print(f"  Seed {seed}: spread={spread:.3f}, nash={nash:.3f}, {above}")
        sys.stdout.flush()

        del trainer; gpu_reset()

    # Summary
    spreads = [r["final_spread"] for r in results]
    nash = results[0]["nash_spread"]
    above = sum(1 for r in results if r["above_nash"])
    print(f"\n  N=5 Summary: {above}/{len(results)} above Nash")
    print(f"  Mean spread: {np.mean(spreads):.4f} (Nash: {nash:.4f})")

    out_path = os.path.join(RESULTS_DIR, "maddpg_N5.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  Saved to {out_path}")
    sys.stdout.flush()

    return results


# =====================================================================
# JOB 4: Q-scaling (neural Bellman + FP)
# =====================================================================

def run_q_scaling(Q_val, n_inner, n_outer):
    print(f"\n{'='*60}")
    print(f"JOB 4: Q={Q_val} neural Bellman + Fictitious Play")
    print(f"{'='*60}")
    sys.stdout.flush()

    gpu_reset()
    from types import SimpleNamespace
    from equations.contxiong_exact import ContXiongExact
    from solver_cx import CXFictitiousPlay

    config = SimpleNamespace(
        lambda_a=2.0, lambda_b=2.0, discount_rate=0.01,
        Delta_q=1.0, q_max=Q_val, phi=0.005, N_agents=2,
    )
    eqn = ContXiongExact(config)
    print(f"  Grid: {eqn.nq} inventory levels, K={eqn.K}")
    sys.stdout.flush()

    fp = CXFictitiousPlay(
        eqn, device=device,
        outer_iter=n_outer, inner_iter=n_inner,
        lr=5e-4, damping=0.5,
    )

    start = time.time()
    result = fp.train()
    elapsed = time.time() - start

    mid = eqn.mid
    da = np.array(result["final_delta_a"])
    db = np.array(result["final_delta_b"])
    spread_q0 = da[mid] + db[mid]
    print(f"\n  Q={Q_val}: spread(0) = {spread_q0:.4f}, time = {elapsed:.0f}s")
    sys.stdout.flush()

    save_data = {
        "Q": Q_val, "nq": eqn.nq, "spread_q0": spread_q0,
        "delta_a": [float(x) for x in result["final_delta_a"]],
        "delta_b": [float(x) for x in result["final_delta_b"]],
        "V": [float(x) for x in result["final_V"]],
        "elapsed": elapsed,
    }
    out_path = os.path.join(RESULTS_DIR, f"q_scaling_Q{Q_val}.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved to {out_path}")

    del fp, eqn; gpu_reset()
    return save_data


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    total_start = time.time()
    summary = {}

    # Job 1: Warm-started BSDEJ T=20
    try:
        r = run_warmstart_bsdej(T=20, M=50, n_iter=10000, label="T20")
        summary["bsdej_warmstart_T20"] = {
            "spread_q0": r["U_profile"][5]["spread"],
            "best_loss": r["best_loss"],
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        summary["bsdej_warmstart_T20"] = {"error": str(e)}

    # Job 2: Non-stationary BSDEJ
    try:
        r = run_nonstationary_bsdej()
        summary["bsdej_nonstationary"] = "done"
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        summary["bsdej_nonstationary"] = {"error": str(e)}

    # Job 3: N=5 MADDPG
    try:
        r = run_maddpg_n5(n_seeds=5)
        summary["maddpg_N5"] = {
            "above_nash": sum(1 for x in r if x["above_nash"]),
            "total": len(r),
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        summary["maddpg_N5"] = {"error": str(e)}

    # Job 4a: Q=20
    try:
        r = run_q_scaling(Q_val=20, n_inner=5000, n_outer=15)
        summary["q20"] = {"spread_q0": r["spread_q0"], "elapsed": r["elapsed"]}
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        summary["q20"] = {"error": str(e)}

    # Job 4b: Q=50
    try:
        r = run_q_scaling(Q_val=50, n_inner=8000, n_outer=15)
        summary["q50"] = {"spread_q0": r["spread_q0"], "elapsed": r["elapsed"]}
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        summary["q50"] = {"error": str(e)}

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total/3600:.1f} hours")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(json.dumps(summary, indent=2, default=float))

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

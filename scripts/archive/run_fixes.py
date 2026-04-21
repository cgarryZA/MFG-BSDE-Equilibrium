#!/usr/bin/env python
"""
Targeted fixes for the three weak results:

1. FD spread mismatch: test FD with finer grid (Delta=0.1) to check
   if spread converges toward 2/alpha=1.333 (proving NN is correct)
2. Impact phase transition: push much harder (kappa=5,10,20, T=5,10)
3. FP convergence: more simulation agents (1000, 2000) + lower threshold

~3-4 hours total.
"""

import gc
import json
import os
import sys
import time
import traceback
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from registry import EQUATION_REGISTRY
import equations
from solver import FictitiousPlaySolver, ContXiongLOBMVSolver

OUT = "results_fixes"
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "log.txt")
with open(LOG, "w") as f:
    f.write("")


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def convert(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(v) for v in obj]
    return obj


def save(data, name):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        json.dump(convert(data), f, indent=2)
    log(f"  Saved {p}")


def run_experiment(name, func):
    log(f"\n{'='*60}")
    log(f"{name}")
    log(f"{'='*60}")
    gpu_reset()
    t0 = time.time()
    try:
        result = func()
        log(f"  Completed in {(time.time()-t0)/60:.0f} min")
        return result
    except Exception:
        log(f"  CRASHED after {(time.time()-t0)/60:.0f} min")
        log(traceback.format_exc())
        gpu_reset()
        return None


def solve_fd_2d(h_val, phi=0.1, alpha=1.5, r=0.1, T=1.0, H=5, Delta=1.0, N_t=200):
    dt = T / N_t
    q = np.arange(-H, H + Delta, Delta)
    nq = len(q); mid = nq // 2
    V = np.zeros((N_t + 1, nq))
    da = np.zeros((N_t + 1, nq))
    db = np.zeros((N_t + 1, nq))
    for j in range(nq):
        V[N_t, j] = -phi * q[j] ** 2
    for n in range(N_t - 1, -1, -1):
        for j in range(nq):
            Vh = V[n+1, j]
            Vd = V[n+1, j-1] if j > 0 else -phi * (q[j] - Delta) ** 2
            Vu = V[n+1, j+1] if j < nq-1 else -phi * (q[j] + Delta) ** 2
            d_a = max(1/alpha - (Vd - Vh)/Delta, 0.001)
            d_b = max(1/alpha - (Vu - Vh)/Delta, 0.001)
            ra = h_val * np.exp(-alpha * d_a)
            rb = h_val * np.exp(-alpha * d_b)
            V[n, j] = Vh + dt * (ra*(d_a*Delta + Vd - Vh) + rb*(d_b*Delta + Vu - Vh) - phi*q[j]**2 - r*Vh)
            da[n, j] = d_a; db[n, j] = d_b
    return {"V": V[0, mid], "da": da[0, mid], "db": db[0, mid],
            "spread": da[0, mid] + db[0, mid], "V_all": V[0, :].tolist()}


def generate_pop(n, std_q):
    S = np.full(n, 100.0)
    q = np.clip(np.random.normal(0, std_q, n), -10, 10)
    return np.stack([S, q], axis=1)


def run_impact(kappa, impact_type, n_iters, device):
    torch.manual_seed(42); np.random.seed(42)
    config = Config.from_json("configs/lob_d2_mv_impact.json")
    config.eqn.kappa = kappa
    config.eqn.impact_type = impact_type
    config.net.opt_config1.num_iterations = n_iters
    config.net.logging_frequency = n_iters
    config.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_impact"](config.eqn)
    solver = ContXiongLOBMVSolver(config, bsde, device=device)
    try:
        r = solver.train()
        y0 = r["y0"]; loss = r["final_loss"]
        zm = solver.model._last_z_max_overall if hasattr(solver.model, "_last_z_max_overall") else 0
        diverged = np.isnan(y0) or np.isnan(loss) or abs(y0) > 10 or loss > 10 or zm > 100
        return {"y0": y0, "loss": loss, "z_max": zm, "diverged": diverged}
    except Exception as e:
        return {"y0": None, "loss": None, "z_max": None, "diverged": True, "error": str(e)}


def run_impact_long_horizon(kappa, impact_type, T, n_iters, device):
    """Impact + long horizon combined stress test."""
    torch.manual_seed(42); np.random.seed(42)
    config = Config.from_json("configs/lob_d2_mv_impact.json")
    config.eqn.kappa = kappa
    config.eqn.impact_type = impact_type
    config.eqn.total_time = T
    config.net.opt_config1.num_iterations = n_iters
    config.net.logging_frequency = n_iters
    config.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_impact"](config.eqn)
    solver = ContXiongLOBMVSolver(config, bsde, device=device)
    try:
        r = solver.train()
        y0 = r["y0"]; loss = r["final_loss"]
        zm = solver.model._last_z_max_overall if hasattr(solver.model, "_last_z_max_overall") else 0
        diverged = np.isnan(y0) or np.isnan(loss) or abs(y0) > 10 or loss > 10 or zm > 100
        return {"y0": y0, "loss": loss, "z_max": zm, "diverged": diverged, "T": T, "kappa": kappa}
    except Exception as e:
        return {"y0": None, "loss": None, "diverged": True, "error": str(e), "T": T, "kappa": kappa}


def run_fp(subnet_type, outer, inner, damping, n_agents, seed=42, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    config = Config.from_json("configs/lob_d2_mv.json")
    config.eqn.law_encoder_type = "moments"
    config.eqn.subnet_type = subnet_type
    config.eqn.phi = 0.1
    config.net.opt_config1.num_iterations = inner
    config.net.logging_frequency = inner
    config.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_mv"](config.eqn)
    fp = FictitiousPlaySolver(
        config, bsde, device=device,
        outer_iterations=outer, inner_iterations=inner,
        w2_threshold=0.005, damping_alpha=damping,
        n_sim_agents=n_agents, warm_start=True,
    )
    return fp.train()


# ================================================================
# FIX 1: FD grid convergence (proves NN spread is correct)
# ================================================================
def fix_fd_grid_convergence():
    log("  Testing if FD spread converges to 2/alpha as grid refines...")
    results = {}
    for Delta in [2.0, 1.0, 0.5, 0.25, 0.1]:
        H = 5
        fd = solve_fd_2d(h_val=0.4, phi=0.1, Delta=Delta, H=H, N_t=400)
        results[f"Delta={Delta}"] = {"spread": fd["spread"], "V_q0": fd["V"],
                                      "da": fd["da"], "db": fd["db"]}
        log(f"  Delta={Delta:.2f}: spread={fd['spread']:.6f}, V(0)={fd['V']:.6f}")

    log(f"\n  2/alpha = {2/1.5:.6f}")
    log(f"  Does FD spread -> 2/alpha as Delta -> 0?")
    spreads = [results[f"Delta={d}"]["spread"] for d in [2.0, 1.0, 0.5, 0.25, 0.1]]
    log(f"  Spreads: {' -> '.join(f'{s:.4f}' for s in spreads)}")
    converging = spreads[-1] < spreads[0]  # getting smaller (toward 1.333)?
    log(f"  Converging toward 1.333? {converging}")

    # Also test at multiple h values
    log(f"\n  FD spread at Delta=0.1 for various h:")
    for h in [0.05, 0.1, 0.25, 0.4, 1.0]:
        fd = solve_fd_2d(h_val=h, phi=0.1, Delta=0.1, H=5, N_t=400)
        results[f"fine_h={h}"] = {"spread": fd["spread"], "V_q0": fd["V"]}
        log(f"    h={h:.2f}: spread={fd['spread']:.6f}")

    save(results, "fd_grid_convergence.json")
    return results


# ================================================================
# FIX 2: Harder impact stress test (find the phase transition)
# ================================================================
def fix_impact_harder(device):
    results = {}

    # Extreme kappa values
    log("  Extreme kappa sweep (linear impact):")
    for kappa in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        gpu_reset()
        r = run_impact(kappa, "linear", 2000, device)
        results[f"extreme_linear_{kappa}"] = r
        s = "DIV" if r["diverged"] else "OK "
        y = f"{r['y0']:.4f}" if r["y0"] is not None else "NaN"
        zm = f"{r.get('z_max', 0):.2f}" if r.get("z_max") else "---"
        log(f"    kappa={kappa:5.1f}: [{s}] Y0={y}, max|Z|={zm}")

    # Combined stress: impact + long horizon
    log("\n  Combined stress: impact × horizon:")
    log(f"  {'T':>5} {'kappa':>7} {'status':>8} {'Y0':>8} {'max|Z|':>8}")
    for T in [1.0, 2.0, 5.0, 10.0]:
        for kappa in [0.0, 0.5, 2.0, 10.0]:
            gpu_reset()
            r = run_impact_long_horizon(kappa, "sqrt", T, 2000, device)
            key = f"stress_T={T}_kappa={kappa}"
            results[key] = r
            s = "DIV" if r["diverged"] else "OK "
            y = f"{r['y0']:.4f}" if r["y0"] is not None else "NaN"
            zm = f"{r.get('z_max', 0):.2f}" if r.get("z_max") else "---"
            log(f"  {T:5.1f} {kappa:7.1f} [{s:>6}] {y:>8} {zm:>8}")

    # Find boundary
    log("\n  Stability boundary:")
    for T in [1.0, 2.0, 5.0, 10.0]:
        last_stable = 0.0
        for kappa in [0.0, 0.5, 2.0, 10.0]:
            if not results[f"stress_T={T}_kappa={kappa}"]["diverged"]:
                last_stable = kappa
        log(f"    T={T}: stable up to kappa={last_stable}")

    save(results, "impact_harder.json")
    return results


# ================================================================
# FIX 3: FP with more agents (reduce sampling noise)
# ================================================================
def fix_fp_more_agents(device):
    results = {}
    for n_agents in [256, 512, 1024, 2048]:
        gpu_reset()
        log(f"\n  --- n_agents={n_agents} ---")
        r = run_fp("two_stream", 15, 1500, 0.1, n_agents=n_agents, device=device)
        w2s = [h["w2"] for h in r["history"]]
        results[f"n={n_agents}"] = {
            "final_w2": w2s[-1],
            "min_w2": min(w2s),
            "w2s": w2s,
            "converged": r["converged"],
        }
        log(f"  final W2={w2s[-1]:.4f}, min W2={min(w2s):.4f}, converged={r['converged']}")
        save(results, "fp_agent_sweep.json")

    log(f"\n  Agent count vs W2:")
    for k, v in results.items():
        log(f"    {k}: final={v['final_w2']:.4f}, min={v['min_w2']:.4f}")
    return results


# ================================================================
# FIX 4: Jump BSDE (proper discrete inventory, match FD spread)
# ================================================================
def fix_jump_bsde(device):
    results = {}

    # Train jump BSDE model
    log("  Training jump BSDE (5000 iter)...")
    torch.manual_seed(42); np.random.seed(42)
    config = Config.from_json("configs/lob_d2_mv_jump.json")
    config.net.opt_config1.num_iterations = 5000
    config.net.logging_frequency = 5000
    config.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_jump"](config.eqn)
    solver = ContXiongLOBMVSolver(config, bsde, device=device)
    r = solver.train()
    solver.model.eval()
    log(f"  Y0={r['y0']:.4f}, loss={r['final_loss']:.4e}")

    # Evaluate spread at q=0 for different populations
    log(f"\n  Jump BSDE spread comparison:")
    log(f"  {'pop':>6} {'h':>6} {'spread':>8} {'FD_spread':>10}")
    for std_q in [0.1, 1.0, 3.0]:
        pop = generate_pop(256, std_q)
        particles = torch.tensor(pop, dtype=torch.float64, device=device)
        le = solver.model.law_encoder.encode(particles)
        with torch.no_grad():
            h = bsde.compute_competitive_factor(le).item()
            agent = torch.tensor([[100.0, 0.0]], dtype=torch.float64, device=device)
            si = torch.cat([agent, le.unsqueeze(0)], dim=1)
            z = solver.model.subnet[0](si) / bsde.dim
            zq = z[:, 1:2]
            sig = bsde._sigma_q_equilibrium()
            p = zq * bsde.Delta_q / sig
            da = (1.0 / bsde.alpha + p).item()
            db = (1.0 / bsde.alpha - p).item()
        # FD at same h
        fd = solve_fd_2d(h, phi=0.1, Delta=1.0)
        results[f"std={std_q}"] = {
            "h": h, "nn_spread": da + db, "nn_da": da, "nn_db": db,
            "fd_spread": fd["spread"], "fd_da": fd["da"], "fd_V": fd["V"],
        }
        log(f"  {std_q:6.1f} {h:6.3f} {da+db:8.4f} {fd['spread']:10.4f}")

    # Also train diffusion version for direct comparison
    log("\n  Training diffusion BSDE (5000 iter) for comparison...")
    gpu_reset()
    torch.manual_seed(42); np.random.seed(42)
    config2 = Config.from_json("configs/lob_d2_mv.json")
    config2.eqn.law_encoder_type = "moments"
    config2.net.opt_config1.num_iterations = 5000
    config2.net.logging_frequency = 5000
    config2.net.verbose = False
    bsde2 = EQUATION_REGISTRY["contxiong_lob_mv"](config2.eqn)
    solver2 = ContXiongLOBMVSolver(config2, bsde2, device=device)
    r2 = solver2.train()
    solver2.model.eval()

    pop = generate_pop(256, 1.0)
    particles = torch.tensor(pop, dtype=torch.float64, device=device)
    le2 = solver2.model.law_encoder.encode(particles)
    with torch.no_grad():
        agent = torch.tensor([[100.0, 0.0]], dtype=torch.float64, device=device)
        si2 = torch.cat([agent, le2.unsqueeze(0)], dim=1)
        z2 = solver2.model.subnet[0](si2) / bsde2.dim
        sig2 = bsde2._sigma_q_equilibrium()
        da2 = (1.0 / bsde2.alpha + z2[:, 1:2] / sig2).item()
        db2 = (1.0 / bsde2.alpha - z2[:, 1:2] / sig2).item()
    results["diffusion_comparison"] = {
        "diffusion_spread": da2 + db2,
        "jump_spread_std1": results.get("std=1.0", {}).get("nn_spread"),
        "fd_spread_std1": results.get("std=1.0", {}).get("fd_spread"),
    }
    log(f"\n  Spread comparison at std=1.0:")
    log(f"    Diffusion BSDE: {da2+db2:.4f}")
    log(f"    Jump BSDE:      {results.get('std=1.0', {}).get('nn_spread', 'N/A')}")
    log(f"    FD (Delta=1):   {results.get('std=1.0', {}).get('fd_spread', 'N/A')}")
    log(f"    Target (2/alpha): {2/1.5:.4f}")

    save(results, "jump_bsde.json")
    return results


# ================================================================
# MAIN
# ================================================================
def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()

    r1 = run_experiment("FIX 1: FD grid convergence (is NN spread correct?)",
                        lambda: fix_fd_grid_convergence())

    r2 = run_experiment("FIX 2: Harder impact stress test (find phase transition)",
                        lambda: fix_impact_harder(device))

    r3 = run_experiment("FIX 3: FP with more agents (reduce W2 noise floor)",
                        lambda: fix_fp_more_agents(device))

    r4 = run_experiment("FIX 4: Jump BSDE (proper FBSDEJ, should match FD spread)",
                        lambda: fix_jump_bsde(device))

    elapsed = (time.time() - start) / 60
    log(f"\n{'='*60}")
    log(f"ALL FIXES DONE in {elapsed:.0f} min")
    log(f"{'='*60}")
    for n, r in zip(["FD_grid", "impact_harder", "FP_agents", "jump_bsde"], [r1, r2, r3, r4]):
        log(f"  {n}: {'OK' if r else 'CRASHED'}")
    save({"completed": elapsed}, "summary.json")


if __name__ == "__main__":
    main()

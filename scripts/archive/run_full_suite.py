#!/usr/bin/env python
"""
FULL EXPERIMENT SUITE — crash-resilient version.
=================================================
Each experiment runs in isolation with GPU cache clearing.
If one crashes, the rest still run. ~10 hours total.

Experiments:
 1. FP two_stream (15 outer × 1500 inner, alpha=0.1)
 2. FP film (15 outer × 1500 inner, alpha=0.1)
 3. Multi-seed FP (5 seeds, alpha=0.1)
 4. FD quantitative matching
 5. Impact phase transition (24 runs + diagnostics)
 6. Equilibrium comparison + linearity test
 7. FD fictitious play (ground truth)
 8. Scale test (K=1,2,5)
 9. FiLM + FP comparison
10. Common noise sweep
11. Time horizon sweep
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

OUT = "results_full_suite"
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
    """Aggressively free GPU memory between experiments."""
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


def generate_pop(n, std_q):
    S = np.full(n, 100.0)
    q = np.clip(np.random.normal(0, std_q, n), -10, 10)
    return np.stack([S, q], axis=1)


def run_experiment(name, func):
    """Run an experiment with crash protection."""
    log(f"\n{'='*60}")
    log(f"{name}")
    log(f"{'='*60}")
    gpu_reset()
    t0 = time.time()
    try:
        result = func()
        elapsed = (time.time() - t0) / 60
        log(f"  Completed in {elapsed:.0f} min")
        return result
    except Exception:
        elapsed = (time.time() - t0) / 60
        log(f"  CRASHED after {elapsed:.0f} min")
        log(traceback.format_exc())
        gpu_reset()
        return None


# ================================================================
# Helper functions
# ================================================================

def run_fp(subnet_type, outer, inner, damping, seed=42, device="cpu"):
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
        n_sim_agents=256, warm_start=True,
    )
    return fp.train()


def eval_h(model, bsde, std_q, device):
    pop = generate_pop(256, std_q)
    p = torch.tensor(pop, dtype=torch.float64, device=device)
    le = model.law_encoder.encode(p)
    with torch.no_grad():
        return bsde.compute_competitive_factor(le).item()


def eval_quotes(model, bsde, std_q, device):
    pop = generate_pop(256, std_q)
    p = torch.tensor(pop, dtype=torch.float64, device=device)
    le = model.law_encoder.encode(p)
    with torch.no_grad():
        a = torch.tensor([[100.0, 0.0]], dtype=torch.float64, device=device)
        si = torch.cat([a, le.unsqueeze(0)], dim=1)
        z = model.subnet[0](si) / bsde.dim
        sig = bsde._sigma_q_equilibrium()
        da = (1.0 / bsde.alpha + z[:, 1:2] / sig).item()
        db = (1.0 / bsde.alpha - z[:, 1:2] / sig).item()
    return {"da": da, "db": db, "spread": da + db}


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
            "spread": da[0, mid] + db[0, mid], "q_grid": q.tolist(),
            "V_all": V[0, :].tolist(), "da_all": da[0, :].tolist(), "db_all": db[0, :].tolist()}


def solve_fd_full(h_val, phi=0.1, alpha=1.5, lambda_a=1.0, lambda_b=1.0,
                  r=0.1, T=1.0, H=5, Delta=1.0, N_t=200):
    dt = T / N_t
    q_grid = np.arange(-H, H + Delta, Delta)
    nq = len(q_grid)
    V = np.zeros((N_t + 1, nq))
    da = np.zeros((N_t + 1, nq))
    db = np.zeros((N_t + 1, nq))
    for j in range(nq):
        V[N_t, j] = -phi * q_grid[j] ** 2
    for n in range(N_t - 1, -1, -1):
        for j in range(nq):
            Vh = V[n+1, j]
            Vd = V[n+1, j-1] if j > 0 else -phi * (q_grid[j] - Delta) ** 2
            Vu = V[n+1, j+1] if j < nq-1 else -phi * (q_grid[j] + Delta) ** 2
            d_a = max(1/alpha - (Vd - Vh)/Delta, 0.001)
            d_b = max(1/alpha - (Vu - Vh)/Delta, 0.001)
            ra = h_val * lambda_a * np.exp(-alpha * d_a)
            rb = h_val * lambda_b * np.exp(-alpha * d_b)
            V[n, j] = Vh + dt * (ra*(d_a*Delta + Vd - Vh) + rb*(d_b*Delta + Vu - Vh) - phi*q_grid[j]**2 - r*Vh)
            da[n, j] = d_a; db[n, j] = d_b
    return q_grid, V, da, db


def fd_fictitious_play(phi=0.1, alpha=1.5, lambda_a=1.0, lambda_b=1.0,
                       r=0.1, T=1.0, sigma_s=0.3, H=5, Delta=1.0, N_t=200,
                       n_agents=1000, outer_iterations=20, damping=0.3):
    h = 1.0
    beta = 0.5
    history = []
    for k in range(outer_iterations):
        q_grid, V, da_g, db_g = solve_fd_full(h, phi, alpha, lambda_a, lambda_b, r, T, H, Delta, N_t)
        # Simulate population
        dt_sim = T / N_t
        q_sim = np.zeros(n_agents)
        for t in range(N_t):
            q_idx = np.clip(np.round((q_sim - q_grid[0]) / Delta).astype(int), 0, len(q_grid) - 1)
            d_a = da_g[t, q_idx]; d_b = db_g[t, q_idx]
            f_a = lambda_a * h * np.exp(-alpha * np.clip(d_a, 0.001, 10.0))
            f_b = lambda_b * h * np.exp(-alpha * np.clip(d_b, 0.001, 10.0))
            q_sim = np.clip(q_sim + (f_b - f_a) * dt_sim
                            + np.sqrt(np.maximum(f_b + f_a, 1e-8)) * np.random.normal(0, np.sqrt(dt_sim), n_agents),
                            -10, 10)
        h_implied = max(0.01, min(float(np.exp(-beta * np.std(q_sim))), 1.0))
        h_new = damping * h_implied + (1 - damping) * h
        mid = len(q_grid) // 2
        history.append({"iter": k+1, "h": h, "h_implied": h_implied, "h_new": h_new,
                        "q_std": float(np.std(q_sim)), "V_q0": float(V[0, mid]),
                        "spread": float(da_g[0, mid] + db_g[0, mid])})
        h = h_new
    return {"history": history, "final_h": h, "final_V": float(V[0, mid]),
            "final_spread": float(da_g[0, mid] + db_g[0, mid])}


def compute_path_variance(bsde, n_sample=512):
    dw, x = bsde.sample(n_sample)
    return {"var_q_T": float(np.var(x[:, 1, -1])), "var_s_T": float(np.var(x[:, 0, -1])),
            "max_abs_q_T": float(np.max(np.abs(x[:, 1, -1])))}


def compute_lipschitz_z(model, bsde, device):
    model.eval()
    q_vals = np.linspace(-4, 4, 50)
    pop = generate_pop(256, 1.0)
    particles = torch.tensor(pop, dtype=torch.float64, device=device)
    law_embed = model.law_encoder.encode(particles)
    leb = law_embed.unsqueeze(0)
    zqs = []
    with torch.no_grad():
        for qv in q_vals:
            a = torch.tensor([[100.0, qv]], dtype=torch.float64, device=device)
            si = torch.cat([a, leb], dim=1)
            z = model.subnet[0](si) / bsde.dim
            zqs.append(z[:, 1].item())
    zqs = np.array(zqs)
    dz = np.abs(np.diff(zqs))
    dq = np.abs(np.diff(q_vals))
    return float(np.max(dz / dq)) if len(dq) > 0 else 0.0


def run_impact(kappa, impact_type, n_iters, device):
    torch.manual_seed(42); np.random.seed(42)
    config = Config.from_json("configs/lob_d2_mv_impact.json")
    config.eqn.kappa = kappa
    config.eqn.impact_type = impact_type
    config.net.opt_config1.num_iterations = n_iters
    config.net.logging_frequency = n_iters
    config.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_impact"](config.eqn)
    path_var = compute_path_variance(bsde)
    solver = ContXiongLOBMVSolver(config, bsde, device=device)
    try:
        r = solver.train()
        y0 = r["y0"]; loss = r["final_loss"]
        zm = solver.model._last_z_max_overall if hasattr(solver.model, "_last_z_max_overall") else 0
        diverged = np.isnan(y0) or np.isnan(loss) or abs(y0) > 10 or loss > 10 or zm > 100
        lip_z = compute_lipschitz_z(solver.model, bsde, device) if not diverged else None
        return {"y0": y0, "loss": loss, "z_max": zm, "diverged": diverged,
                "lipschitz_z": lip_z, "path_var_pre": path_var}
    except Exception as e:
        return {"y0": None, "loss": None, "z_max": None, "diverged": True,
                "error": str(e), "path_var_pre": path_var}


def train_mv(subnet_type, n_iters, device):
    torch.manual_seed(42); np.random.seed(42)
    config = Config.from_json("configs/lob_d2_mv.json")
    config.eqn.law_encoder_type = "moments"
    config.eqn.subnet_type = subnet_type
    config.eqn.phi = 0.1
    config.net.opt_config1.num_iterations = n_iters
    config.net.logging_frequency = n_iters
    config.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_mv"](config.eqn)
    solver = ContXiongLOBMVSolver(config, bsde, device=device)
    r = solver.train()
    return r, solver.model, bsde


# ================================================================
# EXPERIMENTS
# ================================================================

def exp_fp_two_stream(device):
    r = run_fp("two_stream", 15, 1500, 0.1, device=device)
    w2s = [h["w2"] for h in r["history"]]
    log(f"  W2: {' '.join(f'{x:.4f}' for x in w2s)}")
    save(r, "fp_two_stream.json")
    return r

def exp_fp_film(device):
    r = run_fp("film", 15, 1500, 0.1, device=device)
    w2s = [h["w2"] for h in r["history"]]
    log(f"  W2: {' '.join(f'{x:.4f}' for x in w2s)}")
    save(r, "fp_film.json")
    return r

def exp_multi_seed(device):
    seeds = []
    for s in range(5):
        gpu_reset()
        log(f"  --- seed={s} ---")
        r = run_fp("two_stream", 15, 1500, 0.1, seed=s, device=device)
        entry = {"seed": s, "final_w2": r["history"][-1]["w2"],
                 "final_y0": r["history"][-1]["y0"],
                 "final_h": r["history"][-1]["h"],
                 "w2s": [h["w2"] for h in r["history"]]}
        seeds.append(entry)
        log(f"  W2={entry['final_w2']:.4f} Y0={entry['final_y0']:.4f}")
        save(seeds, "multi_seed.json")  # save after each
    w2s = [s["final_w2"] for s in seeds]
    log(f"  W2 mean={np.mean(w2s):.4f} +/- {np.std(w2s):.4f}")
    return seeds

def exp_fd_matching(device):
    _, model, bsde = train_mv("two_stream", 3000, device)
    model.eval()
    result = {}
    log(f"  {'pop':>6} {'h':>6} {'NN_da':>7} {'FD_da':>7} {'NN_spr':>7} {'FD_spr':>7}")
    for std in [0.1, 1.0, 3.0]:
        h = eval_h(model, bsde, std, device)
        nn = eval_quotes(model, bsde, std, device)
        fd = solve_fd_2d(h)
        result[f"std={std}"] = {"h": h, "nn": nn, "fd": {"V": fd["V"], "da": fd["da"], "spread": fd["spread"]}}
        log(f"  {std:6.1f} {h:6.3f} {nn['da']:7.4f} {fd['da']:7.4f} {nn['spread']:7.4f} {fd['spread']:7.4f}")
    # Sensitivity
    hs = [result[f"std={s}"]["h"] for s in [0.1, 3.0]]
    dh = hs[0] - hs[1]
    nn_s = [result[f"std={s}"]["nn"]["spread"] for s in [0.1, 3.0]]
    fd_s = [result[f"std={s}"]["fd"]["spread"] for s in [0.1, 3.0]]
    nn_sens = (nn_s[0] - nn_s[1]) / max(dh, 1e-6)
    fd_sens = (fd_s[0] - fd_s[1]) / max(dh, 1e-6)
    result["sensitivity"] = {"nn": nn_sens, "fd": fd_sens}
    log(f"  d(spread)/d(h): NN={nn_sens:.4f}  FD={fd_sens:.4f}")
    save(result, "fd_matching.json")
    return result

def exp_impact(device):
    kappas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    impact_types = ["linear", "sqrt", "quadratic"]
    results = {}
    for itype in impact_types:
        log(f"\n  --- {itype} ---")
        for kap in kappas:
            gpu_reset()
            r = run_impact(kap, itype, 2000, device)
            key = f"{itype}_{kap}"
            results[key] = {**r, "kappa": kap, "impact_type": itype}
            s = "DIV" if r["diverged"] else "OK "
            y = f"{r['y0']:.4f}" if r["y0"] is not None else "NaN"
            lip = f"{r.get('lipschitz_z', 0) or 0:.3f}"
            log(f"    kappa={kap:5.2f}: [{s}] Y0={y} Lip={lip}")
    # Phase diagram
    log(f"\n  Phase diagram:")
    log(f"  {'kappa':>7}" + "".join(f"  {t:>10}" for t in impact_types))
    for kap in kappas:
        row = f"  {kap:7.2f}"
        for itype in impact_types:
            r = results[f"{itype}_{kap}"]
            row += f"  {'DIVERGED':>10}" if r["diverged"] else f"  {r['y0']:10.4f}"
        log(row)
    # Diagnostics
    log(f"\n  Diagnostics (linear):")
    log(f"  {'kappa':>7} {'Lip(Z)':>8} {'max|Z|':>8} {'Var(q_T)':>10}")
    for kap in kappas:
        r = results.get(f"linear_{kap}", {})
        lip = r.get("lipschitz_z")
        zm = r.get("z_max")
        pv = r.get("path_var_pre", {})
        lip_s = f"{lip:.4f}" if lip is not None else "---"
        zm_s = f"{zm:.4f}" if zm is not None else "---"
        vq_s = f"{pv.get('var_q_T', 0):.4f}" if pv else "---"
        log(f"  {kap:7.2f} {lip_s:>8} {zm_s:>8} {vq_s:>10}")
    save(results, "impact_phase_transition.json")
    return results

def exp_equilibrium(device):
    # Neural FP equilibrium
    r_eq = run_fp("two_stream", 15, 1500, 0.1, device=device)
    fp_h = r_eq["history"][-1]["h"]
    fp_y0 = r_eq["history"][-1]["y0"]
    fp_w2 = r_eq["history"][-1]["w2"]

    # FD at that h
    fd_eq = solve_fd_2d(fp_h)
    fd_nocomp = solve_fd_2d(1.0)
    log(f"  Neural FP: h={fp_h:.4f}, Y0={fp_y0:.4f}, W2={fp_w2:.4f}")
    log(f"  FD at h={fp_h:.4f}: V={fd_eq['V']:.4f}, spread={fd_eq['spread']:.4f}")
    log(f"  FD baseline (h=1): V={fd_nocomp['V']:.4f}")

    # Linearity test
    gpu_reset()
    _, model, bsde = train_mv("two_stream", 2000, device)
    model.eval()
    log(f"\n  Linearity test (Z_q ~ linear in q?):")
    for std_q in [0.1, 1.0, 3.0]:
        pop = generate_pop(256, std_q)
        p = torch.tensor(pop, dtype=torch.float64, device=device)
        le = model.law_encoder.encode(p)
        leb = le.unsqueeze(0)
        q_grid = np.linspace(-4, 4, 30)
        zqs = []
        with torch.no_grad():
            for qv in q_grid:
                agent = torch.tensor([[100.0, qv]], dtype=torch.float64, device=device)
                si = torch.cat([agent, leb], dim=1)
                z = model.subnet[0](si) / bsde.dim
                zqs.append(z[:, 1].item())
        zqs = np.array(zqs)
        coeffs = np.polyfit(q_grid, zqs, 1)
        pred = np.polyval(coeffs, q_grid)
        ss_res = np.sum((zqs - pred) ** 2)
        ss_tot = np.sum((zqs - np.mean(zqs)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        log(f"    std={std_q}: Z_q = {coeffs[0]:.4f}*q + {coeffs[1]:.4f}, R²={r2:.4f}")

    result = {"fp_h": fp_h, "fp_y0": fp_y0, "fp_w2": fp_w2,
              "fd_at_fp_h": fd_eq, "fd_baseline": fd_nocomp}
    save(result, "equilibrium_comparison.json")
    return result

def exp_fd_fp():
    log(f"  Running FD fictitious play (pure CPU, no neural network)...")
    r = fd_fictitious_play(phi=0.1, outer_iterations=20, damping=0.3, n_agents=2000)
    h_traj = [x["h"] for x in r["history"]]
    log(f"  h trajectory: {' '.join(f'{x:.3f}' for x in h_traj)}")
    log(f"  Final h={r['final_h']:.4f}, V={r['final_V']:.4f}, spread={r['final_spread']:.4f}")
    save(r, "fd_fictitious_play.json")
    return r

def exp_scale(device):
    results = {}
    for K in [1, 2, 5]:
        gpu_reset()
        log(f"\n  --- K={K} (d={2*K}) ---")
        torch.manual_seed(42); np.random.seed(42)
        if K == 1:
            _, model, bsde = train_mv("two_stream", 3000, device)
            h = eval_h(model, bsde, 1.0, device)
        else:
            cfg = Config.from_json("configs/lob_multiasset_k5.json")
            cfg.eqn.n_assets = K
            cfg.eqn.dim = 2 * K
            cfg.net.opt_config1.num_iterations = 5000 if K >= 5 else 3000
            cfg.net.logging_frequency = cfg.net.opt_config1.num_iterations
            cfg.net.verbose = False
            bsde = EQUATION_REGISTRY["contxiong_lob_multiasset"](cfg.eqn)
            slv = ContXiongLOBMVSolver(cfg, bsde, device=device)
            r = slv.train()
            slv.model.eval()
            try:
                pop = np.stack([np.full(256, 100.0)] * K +
                               [np.clip(np.random.normal(0, 1.0, 256), -10, 10)] * K, axis=1)
                p = torch.tensor(pop, dtype=torch.float64, device=device)
                le = slv.model.law_encoder.encode(p)
                with torch.no_grad():
                    h = bsde.compute_competitive_factor(le).item()
            except:
                h = None
            model = slv.model
        results[f"K={K}"] = {"dim": 2*K, "y0": model._last_z_max_overall if hasattr(model, '_last_z_max_overall') else 0, "h": h}
        log(f"  h={h:.4f if h else 'N/A'}")
    save(results, "scale_test.json")
    return results

def exp_film_fp(device):
    r = run_fp("film", 15, 1500, 0.1, device=device)
    w2s = [h["w2"] for h in r["history"]]
    hs = [h["h"] for h in r["history"]]
    log(f"  W2: {' '.join(f'{x:.4f}' for x in w2s)}")
    log(f"  h:  {' '.join(f'{x:.4f}' for x in hs)}")
    save({"history": r["history"], "converged": r["converged"]}, "film_fp.json")
    return r

def exp_common_noise(device):
    results = {}
    for sc in [0.0, 0.1, 0.2, 0.5, 1.0]:
        gpu_reset()
        torch.manual_seed(42); np.random.seed(42)
        cfg = Config.from_json("configs/lob_d2_mv.json")
        cfg.eqn.law_encoder_type = "moments"
        cfg.eqn.sigma_common = sc
        cfg.net.opt_config1.num_iterations = 3000
        cfg.net.logging_frequency = 3000
        cfg.net.verbose = False
        bsde = EQUATION_REGISTRY["contxiong_lob_common_noise"](cfg.eqn)
        slv = ContXiongLOBMVSolver(cfg, bsde, device=device)
        r = slv.train()
        slv.model.eval()
        h_n = eval_h(slv.model, bsde, 0.1, device)
        h_w = eval_h(slv.model, bsde, 3.0, device)
        results[f"sigma={sc}"] = {"y0": r["y0"], "loss": r["final_loss"],
                                   "h_narrow": h_n, "h_wide": h_w, "h_gap": h_n - h_w}
        diverged = np.isnan(r["y0"]) or abs(r["y0"]) > 10
        status = "DIV" if diverged else "OK"
        log(f"  sigma={sc:.1f}: [{status}] Y0={r['y0']:.4f}, h_gap={h_n-h_w:.4f}")
    save(results, "common_noise.json")
    return results

def exp_horizon(device):
    results = {}
    for T in [0.5, 1.0, 2.0, 5.0]:
        gpu_reset()
        torch.manual_seed(42); np.random.seed(42)
        cfg = Config.from_json("configs/lob_d2_mv.json")
        cfg.eqn.law_encoder_type = "moments"
        cfg.eqn.total_time = T
        cfg.net.opt_config1.num_iterations = 3000
        cfg.net.logging_frequency = 3000
        cfg.net.verbose = False
        bsde = EQUATION_REGISTRY["contxiong_lob_mv"](cfg.eqn)
        slv = ContXiongLOBMVSolver(cfg, bsde, device=device)
        r = slv.train()
        slv.model.eval()
        h_n = eval_h(slv.model, bsde, 0.1, device)
        h_w = eval_h(slv.model, bsde, 3.0, device)
        zm = slv.model._last_z_max_overall if hasattr(slv.model, "_last_z_max_overall") else 0
        diverged = np.isnan(r["y0"]) or abs(r["y0"]) > 10 or r["final_loss"] > 10
        results[f"T={T}"] = {"y0": r["y0"], "loss": r["final_loss"], "z_max": zm,
                              "h_narrow": h_n, "h_wide": h_w, "h_gap": h_n - h_w, "diverged": diverged}
        status = "DIV" if diverged else "OK"
        log(f"  T={T:.1f}: [{status}] Y0={r['y0']:.4f}, max|Z|={zm:.4f}, h_gap={h_n-h_w:.4f}")
    save(results, "horizon_sweep.json")
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

    r1  = run_experiment("1. FP two_stream (15×1500, alpha=0.1)", lambda: exp_fp_two_stream(device))
    r2  = run_experiment("2. FP film (15×1500, alpha=0.1)", lambda: exp_fp_film(device))
    r3  = run_experiment("3. Multi-seed FP (5 seeds)", lambda: exp_multi_seed(device))
    r4  = run_experiment("4. FD quantitative matching", lambda: exp_fd_matching(device))
    r5  = run_experiment("5. Impact phase transition", lambda: exp_impact(device))
    r6  = run_experiment("6. Equilibrium comparison + linearity", lambda: exp_equilibrium(device))
    r7  = run_experiment("7. FD fictitious play (ground truth)", lambda: exp_fd_fp())
    r8  = run_experiment("8. Scale test (K=1,2,5)", lambda: exp_scale(device))
    r9  = run_experiment("9. FiLM + FP comparison", lambda: exp_film_fp(device))
    r10 = run_experiment("10. Common noise sweep", lambda: exp_common_noise(device))
    r11 = run_experiment("11. Time horizon sweep", lambda: exp_horizon(device))

    # Compare neural FP vs FD FP
    if r1 and r7:
        neural_h = r1["history"][-1]["h"]
        fd_h = r7["final_h"]
        log(f"\n  EQUILIBRIUM COMPARISON:")
        log(f"    Neural FP h = {neural_h:.4f}")
        log(f"    FD FP h     = {fd_h:.4f}")
        log(f"    Match: {abs(neural_h - fd_h) < 0.1}")

    elapsed = (time.time() - start) / 60
    log(f"\n{'='*60}")
    log(f"ALL DONE in {elapsed:.0f} min ({elapsed/60:.1f} hours)")
    log(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*60}")

    # Summary of what completed
    names = ["FP_ts", "FP_film", "multi_seed", "FD_match", "impact",
             "equilibrium", "FD_FP", "scale", "FiLM_FP", "common_noise", "horizon"]
    results = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11]
    for n, r in zip(names, results):
        log(f"  {n}: {'OK' if r is not None else 'CRASHED'}")

    save({"completed": [n for n, r in zip(names, results) if r is not None],
          "crashed": [n for n, r in zip(names, results) if r is None],
          "elapsed_minutes": elapsed},
         "summary.json")


if __name__ == "__main__":
    main()

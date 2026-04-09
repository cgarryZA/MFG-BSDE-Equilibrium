#!/usr/bin/env python
"""
Resume overnight suite from where it crashed.
Experiments 1-3 completed. Remaining: 4 (multi-seed), 5 (FD), 6 (impact).

Uses alpha=0.1 (best from damping sweep).
"""

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

OUT = "results_overnight"
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "log.txt")


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


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
    return {"V": V[0, mid], "da": da[0, mid], "db": db[0, mid], "spread": da[0, mid] + db[0, mid]}


def solve_fd_full(h_val, phi=0.1, alpha=1.5, lambda_a=1.0, lambda_b=1.0,
                  r=0.1, T=1.0, H=5, Delta=1.0, N_t=200):
    """Full FD solver: returns V and optimal quotes at ALL (t, q) points."""
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


def simulate_population_fd(q_grid, da_grid, db_grid, h_val,
                           alpha=1.5, lambda_a=1.0, lambda_b=1.0,
                           sigma_s=0.3, T=1.0, N_t=200, n_agents=1000):
    """Simulate population under FD-optimal policy.

    Given the FD solution (quotes at each (t,q)), forward-simulate N agents
    using those quotes to get the terminal inventory distribution.
    """
    dt = T / N_t
    q = np.zeros(n_agents)  # start at q=0
    Delta = q_grid[1] - q_grid[0] if len(q_grid) > 1 else 1.0

    for t in range(N_t):
        # Map each agent's inventory to nearest grid point
        q_idx = np.clip(np.round((q - q_grid[0]) / Delta).astype(int), 0, len(q_grid) - 1)

        # Get quotes from FD solution
        d_a = da_grid[t, q_idx]
        d_b = db_grid[t, q_idx]

        # Execution rates
        f_a = lambda_a * h_val * np.exp(-alpha * np.clip(d_a, 0.001, 10.0))
        f_b = lambda_b * h_val * np.exp(-alpha * np.clip(d_b, 0.001, 10.0))

        # Inventory update (diffusion approximation)
        inv_drift = (f_b - f_a) * dt
        inv_diff = np.sqrt(np.maximum(f_b + f_a, 1e-8)) * np.random.normal(0, np.sqrt(dt), n_agents)
        q = np.clip(q + inv_drift + inv_diff, -10, 10)

    return q  # terminal inventories


def fd_fictitious_play(phi=0.1, alpha=1.5, lambda_a=1.0, lambda_b=1.0,
                       r=0.1, T=1.0, sigma_s=0.3, H=5, Delta=1.0, N_t=200,
                       n_agents=1000, outer_iterations=15, damping=0.3):
    """Fictitious play using FD as inner solver (ground truth).

    1. Start with h=1.0 (no competition)
    2. Solve FD to get optimal policy
    3. Simulate population under that policy
    4. Compute implied h from population spread statistics
    5. Damped update of h, repeat

    The 'implied h' comes from: wider population inventory dispersion →
    more aggressive competitors → lower effective execution probability.
    We use h = exp(-beta * std(q)) as the mean-field mapping.
    """
    h = 1.0  # start with no competition
    beta = 0.5  # sensitivity of h to population dispersion
    history = []

    for k in range(outer_iterations):
        # Solve single-agent problem at current h
        q_grid, V, da, db = solve_fd_full(h, phi, alpha, lambda_a, lambda_b, r, T, H, Delta, N_t)

        # Simulate population under this policy
        q_terminal = simulate_population_fd(q_grid, da, db, h, alpha, lambda_a, lambda_b, sigma_s, T, N_t, n_agents)
        q_std = float(np.std(q_terminal))

        # Implied h from population dispersion
        # More dispersed → more competition → lower h
        h_implied = float(np.exp(-beta * q_std))
        h_implied = max(0.01, min(h_implied, 1.0))

        # Damped update
        h_new = damping * h_implied + (1 - damping) * h
        w2 = abs(h_new - h)

        mid = len(q_grid) // 2
        history.append({
            "iteration": k + 1, "h": h, "h_implied": h_implied, "h_new": h_new,
            "q_std": q_std, "V_q0": float(V[0, mid]),
            "spread_q0": float(da[0, mid] + db[0, mid]), "w2_h": w2,
        })

        h = h_new

    return {"history": history, "final_h": h,
            "final_V": float(V[0, mid]), "final_spread": float(da[0, mid] + db[0, mid])}


def compute_lipschitz_z(model, bsde, device, n_points=50):
    model.eval()
    q_vals = np.linspace(-4, 4, n_points)
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


def compute_path_variance(bsde, n_sample=512):
    dw, x = bsde.sample(n_sample)
    return {
        "var_q_T": float(np.var(x[:, 1, -1])),
        "var_s_T": float(np.var(x[:, 0, -1])),
        "max_abs_q_T": float(np.max(np.abs(x[:, 1, -1]))),
    }


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
        path_var_post = compute_path_variance(bsde) if not diverged else None
        return {"y0": y0, "loss": loss, "z_max": zm, "diverged": diverged,
                "lipschitz_z": lip_z, "path_var_pre": path_var, "path_var_post": path_var_post}
    except Exception as e:
        return {"y0": None, "loss": None, "z_max": None, "diverged": True,
                "error": str(e), "path_var_pre": path_var}


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"\nRESUMING — Device: {device}")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()

    # ==============================================================
    # 4. MULTI-SEED FP (alpha=0.1, best from damping sweep)
    # ==============================================================
    log("\n" + "=" * 60)
    log("4. Multi-seed FP (5 seeds, alpha=0.1, 15 outer × 1500 inner)")
    log("=" * 60)
    seeds = []
    for s in range(5):
        log(f"  --- seed={s} ---")
        t0 = time.time()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        r = run_fp("two_stream", 15, 1500, 0.1, seed=s, device=device)
        w2s = [h["w2"] for h in r["history"]]
        seeds.append({"seed": s, "converged": r["converged"],
                      "final_w2": r["history"][-1]["w2"],
                      "final_y0": r["history"][-1]["y0"],
                      "final_h": r["history"][-1]["h"],
                      "w2s": w2s})
        log(f"  W2={w2s[-1]:.4f} Y0={r['history'][-1]['y0']:.4f} ({(time.time()-t0)/60:.0f}m)")
        log(f"  W2 trajectory: {' '.join(f'{x:.4f}' for x in w2s)}")
        save(seeds, "multi_seed.json")  # save after each seed

    w2_finals = [s["final_w2"] for s in seeds]
    y0_finals = [s["final_y0"] for s in seeds]
    log(f"\n  W2 mean={np.mean(w2_finals):.4f} +/- {np.std(w2_finals):.4f}")
    log(f"  Y0 mean={np.mean(y0_finals):.4f} +/- {np.std(y0_finals):.4f}")

    # ==============================================================
    # 5. FD MATCHING
    # ==============================================================
    log("\n" + "=" * 60)
    log("5. FD quantitative matching")
    log("=" * 60)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.manual_seed(42); np.random.seed(42)
    cfg = Config.from_json("configs/lob_d2_mv.json")
    cfg.eqn.law_encoder_type = "moments"
    cfg.eqn.subnet_type = "two_stream"
    cfg.eqn.phi = 0.1
    cfg.net.opt_config1.num_iterations = 3000
    cfg.net.logging_frequency = 3000
    cfg.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_mv"](cfg.eqn)
    slv = ContXiongLOBMVSolver(cfg, bsde, device=device)
    slv.train(); m = slv.model; m.eval()

    fd_match = {}
    log(f"  {'pop':>6} {'h':>6} {'NN_da':>7} {'FD_da':>7} {'NN_spr':>7} {'FD_spr':>7}")
    for std in [0.1, 1.0, 3.0]:
        h = eval_h(m, bsde, std, device)
        nn = eval_quotes(m, bsde, std, device)
        fd = solve_fd_2d(h)
        fd_match[f"std={std}"] = {"h": h, "nn": nn, "fd": fd}
        log(f"  {std:6.1f} {h:6.3f} {nn['da']:7.4f} {fd['da']:7.4f} {nn['spread']:7.4f} {fd['spread']:7.4f}")

    hs = [fd_match[f"std={s}"]["h"] for s in [0.1, 3.0]]
    nn_s = [fd_match[f"std={s}"]["nn"]["spread"] for s in [0.1, 3.0]]
    fd_s = [fd_match[f"std={s}"]["fd"]["spread"] for s in [0.1, 3.0]]
    dh = hs[0] - hs[1]
    nn_sens = (nn_s[0] - nn_s[1]) / max(dh, 1e-6)
    fd_sens = (fd_s[0] - fd_s[1]) / max(dh, 1e-6)
    log(f"  d(spread)/d(h): NN={nn_sens:.4f}  FD={fd_sens:.4f}  ratio={nn_sens/max(abs(fd_sens),1e-6):.1f}x")
    fd_match["sensitivity"] = {"nn": nn_sens, "fd": fd_sens}
    save(fd_match, "fd_matching.json")

    # ==============================================================
    # 6. IMPACT PHASE TRANSITION
    # ==============================================================
    log("\n" + "=" * 60)
    log("6. Non-linear impact phase transition")
    log("=" * 60)
    kappas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    impact_types = ["linear", "sqrt", "quadratic"]
    impact_results = {}

    for itype in impact_types:
        log(f"\n  --- {itype} ---")
        for kap in kappas:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            t0 = time.time()
            r = run_impact(kap, itype, 2000, device)
            key = f"{itype}_{kap}"
            impact_results[key] = {**r, "kappa": kap, "impact_type": itype}
            s = "DIV" if r["diverged"] else "OK "
            y = f"{r['y0']:.4f}" if r["y0"] is not None else "NaN"
            lip = f"{r.get('lipschitz_z', 0) or 0:.3f}"
            log(f"    kappa={kap:5.2f}: [{s}] Y0={y} Lip(Z)={lip} ({time.time()-t0:.0f}s)")

    # Phase diagram
    log(f"\n  Phase diagram (Y0):")
    log(f"  {'kappa':>7}" + "".join(f"  {t:>10}" for t in impact_types))
    for kap in kappas:
        row = f"  {kap:7.2f}"
        for itype in impact_types:
            r = impact_results[f"{itype}_{kap}"]
            row += f"  {'DIVERGED':>10}" if r["diverged"] else f"  {r['y0']:10.4f}"
        log(row)

    # Stability boundary
    for itype in impact_types:
        last = 0.0
        for kap in kappas:
            if not impact_results[f"{itype}_{kap}"]["diverged"]:
                last = kap
        log(f"  {itype}: stable up to kappa={last}")

    # Diagnostics table
    log(f"\n  Diagnostics (linear impact):")
    log(f"  {'kappa':>7} {'Lip(Z)':>8} {'max|Z|':>8} {'Var(q_T)':>10}")
    for kap in kappas:
        r = impact_results.get(f"linear_{kap}", {})
        lip = r.get("lipschitz_z")
        zm = r.get("z_max")
        pv = r.get("path_var_pre", {})
        lip_s = f"{lip:.4f}" if lip is not None else "---"
        zm_s = f"{zm:.4f}" if zm is not None else "---"
        vq_s = f"{pv.get('var_q_T', 0):.4f}" if pv else "---"
        log(f"  {kap:7.2f} {lip_s:>8} {zm_s:>8} {vq_s:>10}")

    save(impact_results, "impact_phase_transition.json")

    # ==============================================================
    # 7. EQUILIBRIUM COMPARISON (FP h vs FD comparative statics)
    # ==============================================================
    log("\n" + "=" * 60)
    log("7. Equilibrium comparison: FP learned h vs FD")
    log("=" * 60)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Run FP with best damping to get equilibrium h
    torch.manual_seed(42); np.random.seed(42)
    r_eq = run_fp("two_stream", 15, 1500, 0.1, device=device)
    # Get h at the FP fixed point from last iteration
    fp_h = r_eq["history"][-1]["h"] if r_eq["history"] else None
    fp_y0 = r_eq["history"][-1]["y0"] if r_eq["history"] else None
    fp_w2 = r_eq["history"][-1]["w2"] if r_eq["history"] else None

    # FD at that h value
    if fp_h:
        fd_eq = solve_fd_2d(fp_h)
        log(f"  FP equilibrium: h={fp_h:.4f}, Y0={fp_y0:.4f}, W2={fp_w2:.4f}")
        log(f"  FD at h={fp_h:.4f}: V={fd_eq['V']:.4f}, spread={fd_eq['spread']:.4f}")
        log(f"  Y0 comparison: FP={fp_y0:.4f} vs FD={fd_eq['V']:.4f}")

        # Also FD at h=1.0 (no competition baseline)
        fd_nocomp = solve_fd_2d(1.0)
        log(f"  FD baseline (h=1.0): V={fd_nocomp['V']:.4f}, spread={fd_nocomp['spread']:.4f}")
        log(f"  MV effect: FD value drops {fd_nocomp['V']:.4f} -> {fd_eq['V']:.4f} at learned h")

        # Linearity test: Cont-Xiong equilibrium implies Z_q ~ linear in q
        # Fit Z_q(q) = a + b*q and measure R^2
        log(f"\n  Linearity test (Cont-Xiong parametric form Z = xi0 + xi1*q):")
        for std_q_test in [0.1, 1.0, 3.0]:
            pop_test = generate_pop(256, std_q_test)
            p_test = torch.tensor(pop_test, dtype=torch.float64, device=device)
            # Need a trained model — use the FP's last solver
            # Re-train a quick model at this population
            torch.manual_seed(42); np.random.seed(42)
            cfg_lin = Config.from_json("configs/lob_d2_mv.json")
            cfg_lin.eqn.law_encoder_type = "moments"
            cfg_lin.net.opt_config1.num_iterations = 2000
            cfg_lin.net.logging_frequency = 2000
            cfg_lin.net.verbose = False
            bsde_lin = EQUATION_REGISTRY["contxiong_lob_mv"](cfg_lin.eqn)
            slv_lin = ContXiongLOBMVSolver(cfg_lin, bsde_lin, device=device)
            slv_lin.train()
            slv_lin.model.eval()
            le_test = slv_lin.model.law_encoder.encode(p_test)
            leb_test = le_test.unsqueeze(0)

            q_grid = np.linspace(-4, 4, 30)
            zq_vals = []
            with torch.no_grad():
                for qv in q_grid:
                    agent = torch.tensor([[100.0, qv]], dtype=torch.float64, device=device)
                    si = torch.cat([agent, leb_test], dim=1)
                    z = slv_lin.model.subnet[0](si) / bsde_lin.dim
                    zq_vals.append(z[:, 1].item())
            zq_arr = np.array(zq_vals)

            # Linear fit
            coeffs = np.polyfit(q_grid, zq_arr, 1)
            zq_pred = np.polyval(coeffs, q_grid)
            ss_res = np.sum((zq_arr - zq_pred) ** 2)
            ss_tot = np.sum((zq_arr - np.mean(zq_arr)) ** 2)
            r_squared = 1 - ss_res / max(ss_tot, 1e-10)
            log(f"    std={std_q_test}: Z_q = {coeffs[0]:.4f}*q + {coeffs[1]:.4f}, R^2={r_squared:.4f}")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        save({"fp_h": fp_h, "fp_y0": fp_y0, "fp_w2": fp_w2,
              "fd_at_fp_h": fd_eq, "fd_baseline": fd_nocomp,
              "w2_trajectory": [h["w2"] for h in r_eq["history"]],
              "h_trajectory": [h["h"] for h in r_eq["history"]]},
             "equilibrium_comparison.json")

    # ==============================================================
    # 7b. FD FICTITIOUS PLAY (ground truth equilibrium)
    # ==============================================================
    log("\n" + "=" * 60)
    log("7b. FD fictitious play (ground truth, no neural network)")
    log("=" * 60)
    t0 = time.time()
    fd_fp = fd_fictitious_play(phi=0.1, outer_iterations=20, damping=0.3, n_agents=2000)
    log(f"  {(time.time()-t0)/60:.1f} min")
    log(f"  Final h={fd_fp['final_h']:.4f}, V(0,0)={fd_fp['final_V']:.4f}, spread={fd_fp['final_spread']:.4f}")
    h_traj = [x["h"] for x in fd_fp["history"]]
    log(f"  h trajectory: {' '.join(f'{x:.4f}' for x in h_traj)}")

    # Compare neural FP vs FD FP
    if fp_h is not None:
        log(f"\n  Neural FP h = {fp_h:.4f}")
        log(f"  FD FP h     = {fd_fp['final_h']:.4f}")
        log(f"  Difference  = {abs(fp_h - fd_fp['final_h']):.4f}")
        match = abs(fp_h - fd_fp['final_h']) < 0.1
        log(f"  Match? {'YES' if match else 'NO'}")

    save(fd_fp, "fd_fictitious_play.json")

    # ==============================================================
    # 8. SCALE TEST (multi-asset K=1,2,5)
    # ==============================================================
    log("\n" + "=" * 60)
    log("8. Scale test: multi-asset K=1,2,5 (d=2,4,10)")
    log("=" * 60)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    scale_results = {}

    for K in [1, 2, 5]:
        log(f"\n  --- K={K} (d={2*K}) ---")
        torch.manual_seed(42); np.random.seed(42)
        t0 = time.time()

        if K == 1:
            # Use standard 2D model for K=1 baseline
            cfg = Config.from_json("configs/lob_d2_mv.json")
            cfg.eqn.law_encoder_type = "moments"
            cfg.net.opt_config1.num_iterations = 3000
            cfg.net.logging_frequency = 3000
            cfg.net.verbose = False
            bsde_k = EQUATION_REGISTRY["contxiong_lob_mv"](cfg.eqn)
        else:
            cfg = Config.from_json("configs/lob_multiasset_k5.json")
            cfg.eqn.n_assets = K
            cfg.eqn.dim = 2 * K
            cfg.net.opt_config1.num_iterations = 5000 if K >= 5 else 3000
            cfg.net.logging_frequency = cfg.net.opt_config1.num_iterations
            cfg.net.verbose = False
            bsde_k = EQUATION_REGISTRY["contxiong_lob_multiasset"](cfg.eqn)

        slv = ContXiongLOBMVSolver(cfg, bsde_k, device=device)
        r = slv.train()
        elapsed_k = (time.time() - t0) / 60

        # Evaluate h
        slv.model.eval()
        h_val = eval_h(slv.model, bsde_k, 1.0, device) if K == 1 else None
        if K > 1:
            try:
                pop = np.stack(
                    [np.full(256, 100.0)] * K +
                    [np.clip(np.random.normal(0, 1.0, 256), -10, 10)] * K,
                    axis=1
                )
                particles = torch.tensor(pop, dtype=torch.float64, device=device)
                le = slv.model.law_encoder.encode(particles)
                with torch.no_grad():
                    h_val = bsde_k.compute_competitive_factor(le).item()
            except:
                h_val = None

        scale_results[f"K={K}"] = {
            "dim": 2 * K, "y0": r["y0"], "loss": r["final_loss"],
            "h": h_val, "elapsed_min": elapsed_k,
            "n_iters": cfg.net.opt_config1.num_iterations,
        }
        y0_s = f"{r['y0']:.4f}" if not np.isnan(r['y0']) else "NaN"
        log(f"  Y0={y0_s}, loss={r['final_loss']:.4e}, h={h_val:.4f if h_val else 'N/A'} ({elapsed_k:.1f}m)")

    save(scale_results, "scale_test.json")
    log(f"\n  Scale summary:")
    for k_str, v in scale_results.items():
        log(f"    {k_str} (d={v['dim']}): Y0={v['y0']:.4f}, {v['elapsed_min']:.1f} min")

    # ==============================================================
    # 9. FiLM + FP COMPARISON
    # ==============================================================
    log("\n" + "=" * 60)
    log("9. FiLM + FP comparison (15 outer x 1500 inner, alpha=0.1)")
    log("=" * 60)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    t0 = time.time()
    r_film_fp = run_fp("film", 15, 1500, 0.1, device=device)
    w2s_ff = [h["w2"] for h in r_film_fp["history"]]
    h_ff = [h["h"] for h in r_film_fp["history"]]
    y0_ff = [h["y0"] for h in r_film_fp["history"]]
    log(f"  {(time.time()-t0)/60:.0f} min | converged={r_film_fp['converged']}")
    log(f"  W2: {' '.join(f'{x:.4f}' for x in w2s_ff)}")
    log(f"  h:  {' '.join(f'{x:.4f}' for x in h_ff)}")

    # Compare to two_stream FP (from experiment 7)
    if fp_w2 is not None:
        log(f"\n  Comparison at convergence:")
        log(f"    two_stream: W2={fp_w2:.4f}, h={fp_h:.4f}, Y0={fp_y0:.4f}")
        log(f"    film:       W2={w2s_ff[-1]:.4f}, h={h_ff[-1]:.4f}, Y0={y0_ff[-1]:.4f}")
        same_eq = abs(h_ff[-1] - fp_h) < 0.05
        log(f"    Same equilibrium? {'YES' if same_eq else 'NO'} (h diff={abs(h_ff[-1]-fp_h):.4f})")

    save({"film_fp": {"history": r_film_fp["history"], "converged": r_film_fp["converged"]},
          "comparison": {"ts_h": fp_h, "ts_w2": fp_w2, "film_h": h_ff[-1] if h_ff else None,
                         "film_w2": w2s_ff[-1] if w2s_ff else None}},
         "film_fp_comparison.json")

    # ==============================================================
    # 10. COMMON NOISE SWEEP
    # ==============================================================
    log("\n" + "=" * 60)
    log("10. Common noise sweep (sigma_common = 0, 0.1, 0.2, 0.5, 1.0)")
    log("=" * 60)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    cn_results = {}

    for sc in [0.0, 0.1, 0.2, 0.5, 1.0]:
        torch.manual_seed(42); np.random.seed(42)
        t0 = time.time()
        cfg = Config.from_json("configs/lob_d2_mv.json")
        cfg.eqn.eqn_name = "contxiong_lob_common_noise"
        cfg.eqn.law_encoder_type = "moments"
        cfg.eqn.sigma_common = sc
        cfg.net.opt_config1.num_iterations = 3000
        cfg.net.logging_frequency = 3000
        cfg.net.verbose = False
        bsde_cn = EQUATION_REGISTRY["contxiong_lob_common_noise"](cfg.eqn)
        slv_cn = ContXiongLOBMVSolver(cfg, bsde_cn, device=device)
        r_cn = slv_cn.train()
        slv_cn.model.eval()

        # Evaluate h at narrow/wide
        h_n = eval_h(slv_cn.model, bsde_cn, 0.1, device)
        h_w = eval_h(slv_cn.model, bsde_cn, 3.0, device)

        cn_results[f"sigma={sc}"] = {
            "y0": r_cn["y0"], "loss": r_cn["final_loss"],
            "h_narrow": h_n, "h_wide": h_w, "h_gap": h_n - h_w,
        }
        elapsed_cn = (time.time() - t0) / 60
        diverged = np.isnan(r_cn["y0"]) or abs(r_cn["y0"]) > 10
        status = "DIV" if diverged else "OK"
        log(f"  sigma_common={sc:.1f}: [{status}] Y0={r_cn['y0']:.4f}, h_gap={h_n-h_w:.4f} ({elapsed_cn:.1f}m)")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    save(cn_results, "common_noise_sweep.json")

    # ==============================================================
    # 11. TIME HORIZON SWEEP (Germain 2022 failure mode)
    # ==============================================================
    log("\n" + "=" * 60)
    log("11. Time horizon sweep (T = 0.5, 1.0, 2.0, 5.0)")
    log("=" * 60)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    horizon_results = {}

    for T_val in [0.5, 1.0, 2.0, 5.0]:
        torch.manual_seed(42); np.random.seed(42)
        t0 = time.time()
        cfg = Config.from_json("configs/lob_d2_mv.json")
        cfg.eqn.law_encoder_type = "moments"
        cfg.eqn.total_time = T_val
        cfg.net.opt_config1.num_iterations = 3000
        cfg.net.logging_frequency = 3000
        cfg.net.verbose = False
        bsde_t = EQUATION_REGISTRY["contxiong_lob_mv"](cfg.eqn)
        slv_t = ContXiongLOBMVSolver(cfg, bsde_t, device=device)
        r_t = slv_t.train()
        slv_t.model.eval()

        h_n = eval_h(slv_t.model, bsde_t, 0.1, device)
        h_w = eval_h(slv_t.model, bsde_t, 3.0, device)
        zm = slv_t.model._last_z_max_overall if hasattr(slv_t.model, "_last_z_max_overall") else 0

        diverged = np.isnan(r_t["y0"]) or abs(r_t["y0"]) > 10 or r_t["final_loss"] > 10
        horizon_results[f"T={T_val}"] = {
            "y0": r_t["y0"], "loss": r_t["final_loss"], "z_max": zm,
            "h_narrow": h_n, "h_wide": h_w, "h_gap": h_n - h_w,
            "diverged": diverged,
        }
        elapsed_t = (time.time() - t0) / 60
        status = "DIV" if diverged else "OK"
        log(f"  T={T_val:.1f}: [{status}] Y0={r_t['y0']:.4f}, max|Z|={zm:.4f}, h_gap={h_n-h_w:.4f} ({elapsed_t:.1f}m)")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    save(horizon_results, "horizon_sweep.json")

    # Find stability boundary
    for T_str, v in horizon_results.items():
        if v["diverged"]:
            log(f"  Horizon instability at {T_str}")
            break
    else:
        log(f"  Stable at all tested horizons")

    # ==============================================================
    # DONE
    # ==============================================================
    elapsed = (time.time() - start) / 60
    log(f"\n{'=' * 60}")
    log(f"ALL EXPERIMENTS COMPLETE in {elapsed:.0f} min ({elapsed/60:.1f} hours)")
    log(f"{'=' * 60}")

    w2m = np.mean(w2_finals)
    log(f"  Multi-seed W2: {w2m:.4f} +/- {np.std(w2_finals):.4f}")
    log(f"  FD sensitivity ratio: {nn_sens/max(abs(fd_sens),1e-6):.1f}x")
    log(f"  All results in {OUT}/")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(f"FATAL ERROR:\n{traceback.format_exc()}")
        raise

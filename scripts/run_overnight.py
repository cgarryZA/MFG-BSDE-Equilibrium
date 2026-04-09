#!/usr/bin/env python
"""
FULL OVERNIGHT SUITE — runs everything sequentially.
=====================================================
Estimated ~8-12 hours on RTX 3090.

1. FP two_stream (20 outer × 2000 inner, damping=0.5)
2. FP film (20 outer × 2000 inner, damping=0.5)
3. Damping sweep (alpha=0.1, 0.3, 1.0 × 15 outer × 1500 inner)
4. Multi-seed FP or extended FP (conditional on convergence)
5. FD quantitative matching
6. Non-linear impact phase transition (kappa × impact_type sweep)

All results saved incrementally to results_overnight/.
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
with open(LOG, "w") as f:
    f.write("")


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


def generate_pop(n, std_q):
    S = np.full(n, 100.0)
    q = np.clip(np.random.normal(0, std_q, n), -10, 10)
    return np.stack([S, q], axis=1)


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


def compute_lipschitz_z(model, bsde, device, n_points=50, eps=0.05):
    """Estimate Lipschitz constant of Z w.r.t. state (q) at t=0.

    Lip(Z) = max |Z(q1) - Z(q2)| / |q1 - q2| over a grid.
    High Lip(Z) → the Z-process is steep → Euler-Maruyama may be inaccurate.
    """
    model.eval()
    q_vals = np.linspace(-4, 4, n_points)
    # Use a fixed medium population for law embedding
    pop = generate_pop(256, 1.0)
    particles = torch.tensor(pop, dtype=torch.float64, device=device)
    law_embed = model.law_encoder.encode(particles)
    leb = law_embed.unsqueeze(0)

    zqs = []
    with torch.no_grad():
        for q in q_vals:
            a = torch.tensor([[100.0, q]], dtype=torch.float64, device=device)
            si = torch.cat([a, leb], dim=1)
            z = model.subnet[0](si) / bsde.dim
            zqs.append(z[:, 1].item())
    zqs = np.array(zqs)

    # Finite difference Lipschitz estimate
    dz = np.abs(np.diff(zqs))
    dq = np.abs(np.diff(q_vals))
    lip = np.max(dz / dq) if len(dq) > 0 else 0.0
    return float(lip)


def compute_path_variance(bsde, n_sample=512):
    """Variance of terminal state X_T across sampled paths.

    If Var(X_T) explodes, the Euler-Maruyama discretisation is unstable.
    """
    dw, x = bsde.sample(n_sample)
    # x is [batch, dim, T+1]
    q_terminal = x[:, 1, -1]  # terminal inventory
    s_terminal = x[:, 0, -1]  # terminal price
    return {
        "var_q_T": float(np.var(q_terminal)),
        "var_s_T": float(np.var(s_terminal)),
        "max_abs_q_T": float(np.max(np.abs(q_terminal))),
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

    # Diagnostic 1: path variance BEFORE training (from SDE sampling)
    path_var = compute_path_variance(bsde)

    solver = ContXiongLOBMVSolver(config, bsde, device=device)
    try:
        r = solver.train()
        y0 = r["y0"]; loss = r["final_loss"]
        zm = solver.model._last_z_max_overall if hasattr(solver.model, "_last_z_max_overall") else 0
        diverged = np.isnan(y0) or np.isnan(loss) or abs(y0) > 10 or loss > 10 or zm > 100

        # Diagnostic 2: Lipschitz constant of learned Z
        lip_z = compute_lipschitz_z(solver.model, bsde, device) if not diverged else None

        # Diagnostic 3: path variance AFTER training (re-sample to check)
        path_var_post = compute_path_variance(bsde) if not diverged else None

        return {
            "y0": y0, "loss": loss, "z_max": zm, "diverged": diverged,
            "lipschitz_z": lip_z,
            "path_var_pre": path_var,
            "path_var_post": path_var_post,
        }
    except Exception as e:
        return {
            "y0": None, "loss": None, "z_max": None, "diverged": True,
            "error": str(e), "path_var_pre": path_var,
        }


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()
    ALL = {}

    # ==============================================================
    # 1. FP TWO_STREAM
    # ==============================================================
    log("\n" + "=" * 60)
    log("1. FP two_stream (20 outer × 2000 inner, damping=0.5)")
    log("=" * 60)
    t0 = time.time()
    r1 = run_fp("two_stream", 20, 2000, 0.5, device=device)
    ALL["fp_two_stream"] = r1; save(r1, "fp_two_stream.json")
    w = [h["w2"] for h in r1["history"]]
    log(f"  {(time.time()-t0)/60:.0f} min | converged={r1['converged']} | W2: {' '.join(f'{x:.4f}' for x in w)}")

    # ==============================================================
    # 2. FP FILM
    # ==============================================================
    log("\n" + "=" * 60)
    log("2. FP film (20 outer × 2000 inner, damping=0.5)")
    log("=" * 60)
    t0 = time.time()
    r2 = run_fp("film", 20, 2000, 0.5, device=device)
    ALL["fp_film"] = r2; save(r2, "fp_film.json")
    w = [h["w2"] for h in r2["history"]]
    log(f"  {(time.time()-t0)/60:.0f} min | converged={r2['converged']} | W2: {' '.join(f'{x:.4f}' for x in w)}")

    # ==============================================================
    # 3. DAMPING SWEEP
    # ==============================================================
    log("\n" + "=" * 60)
    log("3. Damping sweep (alpha = 0.1, 0.3, 1.0)")
    log("=" * 60)
    damp = {}
    for alpha in [0.1, 0.3, 1.0]:
        t0 = time.time()
        r = run_fp("two_stream", 15, 1500, alpha, device=device)
        damp[f"{alpha}"] = r
        w = [h["w2"] for h in r["history"]]
        log(f"  alpha={alpha}: {(time.time()-t0)/60:.0f} min | W2 final={w[-1]:.4f}")
    ALL["damping"] = damp; save(damp, "damping_sweep.json")

    best_a = min(damp, key=lambda a: damp[a]["history"][-1]["w2"])
    log(f"  Best: alpha={best_a}")

    # ==============================================================
    # 4. MULTI-SEED or EXTENDED
    # ==============================================================
    min_w2 = min(
        r1["history"][-1]["w2"],
        r2["history"][-1]["w2"],
        min(damp[a]["history"][-1]["w2"] for a in damp),
    )
    log("\n" + "=" * 60)
    if min_w2 < 0.03:
        log(f"4. Multi-seed FP (5 seeds, alpha={best_a})")
        log("=" * 60)
        seeds = []
        for s in range(5):
            t0 = time.time()
            r = run_fp("two_stream", 15, 1500, float(best_a), seed=s, device=device)
            seeds.append({"seed": s, "final_w2": r["history"][-1]["w2"],
                          "final_y0": r["history"][-1]["y0"],
                          "w2s": [h["w2"] for h in r["history"]]})
            log(f"  seed={s}: W2={r['history'][-1]['w2']:.4f} Y0={r['history'][-1]['y0']:.4f} ({(time.time()-t0)/60:.0f}m)")
        ALL["multi_seed"] = seeds; save(seeds, "multi_seed.json")
    else:
        log(f"4. Extended FP (min W2={min_w2:.4f}, trying 30 outer × 2000 inner, alpha=0.1)")
        log("=" * 60)
        t0 = time.time()
        r_ext = run_fp("two_stream", 30, 2000, 0.1, device=device)
        ALL["fp_extended"] = r_ext; save(r_ext, "fp_extended.json")
        w = [h["w2"] for h in r_ext["history"]]
        log(f"  {(time.time()-t0)/60:.0f} min | W2: {' '.join(f'{x:.4f}' for x in w[-10:])}")

    # ==============================================================
    # 5. FD MATCHING
    # ==============================================================
    log("\n" + "=" * 60)
    log("5. FD quantitative matching")
    log("=" * 60)
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
    fd_match["sensitivity"] = {"nn": nn_sens, "fd": fd_sens, "ratio": nn_sens / max(abs(fd_sens), 1e-6)}
    ALL["fd_matching"] = fd_match; save(fd_match, "fd_matching.json")

    # ==============================================================
    # 6. NON-LINEAR IMPACT PHASE TRANSITION
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
            t0 = time.time()
            r = run_impact(kap, itype, 2000, device)
            key = f"{itype}_{kap}"
            impact_results[key] = {**r, "kappa": kap, "impact_type": itype}
            s = "DIV" if r["diverged"] else "OK "
            y = f"{r['y0']:.4f}" if r["y0"] is not None else "NaN"
            log(f"    kappa={kap:5.2f}: [{s}] Y0={y} ({time.time()-t0:.0f}s)")

    # Phase diagram summary
    log(f"\n  Phase diagram:")
    log(f"  {'kappa':>7}" + "".join(f"  {t:>10}" for t in impact_types))
    for kap in kappas:
        row = f"  {kap:7.2f}"
        for itype in impact_types:
            r = impact_results[f"{itype}_{kap}"]
            row += f"  {'DIVERGED':>10}" if r["diverged"] else f"  {r['y0']:10.4f}"
        log(row)

    for itype in impact_types:
        last = 0.0
        for kap in kappas:
            if not impact_results[f"{itype}_{kap}"]["diverged"]:
                last = kap
        log(f"  {itype}: stable up to kappa={last}")

    # Diagnostic summary: Lipschitz(Z), path variance, max|Z| vs kappa
    log(f"\n  Diagnostics (linear impact):")
    log(f"  {'kappa':>7} {'Lip(Z)':>8} {'max|Z|':>8} {'Var(q_T)':>10} {'Var(s_T)':>10}")
    for kap in kappas:
        r = impact_results.get(f"linear_{kap}", {})
        lip = r.get("lipschitz_z")
        zm = r.get("z_max")
        pv = r.get("path_var_pre", {})
        lip_s = f"{lip:.4f}" if lip is not None else "---"
        zm_s = f"{zm:.4f}" if zm is not None else "---"
        vq = f"{pv.get('var_q_T', 0):.4f}" if pv else "---"
        vs = f"{pv.get('var_s_T', 0):.4f}" if pv else "---"
        log(f"  {kap:7.2f} {lip_s:>8} {zm_s:>8} {vq:>10} {vs:>10}")

    ALL["impact"] = impact_results; save(impact_results, "impact_phase_transition.json")

    # ==============================================================
    # DONE
    # ==============================================================
    elapsed = (time.time() - start) / 60
    ALL["metadata"] = {"elapsed_minutes": elapsed, "device": str(device),
                       "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    save(ALL, "all_results.json")
    log(f"\n{'=' * 60}")
    log(f"ALL DONE in {elapsed:.0f} min ({elapsed/60:.1f} hours)")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(f"FATAL ERROR:\n{traceback.format_exc()}")
        raise

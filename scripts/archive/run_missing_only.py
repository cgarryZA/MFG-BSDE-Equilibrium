#!/usr/bin/env python
"""
Run ONLY the experiments we don't have yet.
Estimated ~6 hours total.

We already have (from previous runs):
- FP two_stream, FP film, damping sweep, multi-seed
- FD matching, impact phase transition (24 runs)
- FiLM ablation + interaction test

Missing:
1. FD grid convergence (CPU only, ~1 min)
2. FD fictitious play ground truth (CPU only, ~1 min)
3. Equilibrium comparison + linearity test (~30 min)
4. Scale test K=1,2,5 (~30 min)
5. FiLM + FP (~90 min)
6. Common noise sweep (~50 min)
7. Time horizon sweep (~40 min)
8. Harder impact stress test (~60 min)
9. FP with more agents (~90 min)
10. Jump BSDE vs diffusion (~20 min)
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

OUT = "results_missing"
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
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert(v) for v in obj]
    return obj

def save(data, name):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        json.dump(convert(data), f, indent=2)
    log(f"  Saved {p}")

def run_safe(name, func):
    log(f"\n{'='*60}")
    log(f"{name}")
    log(f"{'='*60}")
    gpu_reset()
    t0 = time.time()
    try:
        result = func()
        log(f"  Done in {(time.time()-t0)/60:.0f} min")
        return result
    except Exception:
        log(f"  CRASHED after {(time.time()-t0)/60:.0f} min")
        log(traceback.format_exc())
        gpu_reset()
        return None

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

def solve_fd(h_val, phi=0.1, alpha=1.5, r=0.1, T=1.0, H=5, Delta=1.0, N_t=200):
    dt = T / N_t
    q = np.arange(-H, H + Delta, Delta)
    nq = len(q); mid = nq // 2
    V = np.zeros((N_t + 1, nq)); da = np.zeros((N_t + 1, nq)); db = np.zeros((N_t + 1, nq))
    for j in range(nq): V[N_t, j] = -phi * q[j]**2
    for n in range(N_t-1, -1, -1):
        for j in range(nq):
            Vh = V[n+1, j]
            Vd = V[n+1, j-1] if j > 0 else -phi*(q[j]-Delta)**2
            Vu = V[n+1, j+1] if j < nq-1 else -phi*(q[j]+Delta)**2
            d_a = max(1/alpha-(Vd-Vh)/Delta, 0.001); d_b = max(1/alpha-(Vu-Vh)/Delta, 0.001)
            ra = h_val*np.exp(-alpha*d_a); rb = h_val*np.exp(-alpha*d_b)
            V[n,j] = Vh+dt*(ra*(d_a*Delta+Vd-Vh)+rb*(d_b*Delta+Vu-Vh)-phi*q[j]**2-r*Vh)
            da[n,j]=d_a; db[n,j]=d_b
    return {"V":V[0,mid],"da":da[0,mid],"db":db[0,mid],"spread":da[0,mid]+db[0,mid]}

def run_fp(subnet_type, outer, inner, damping, n_agents=256, seed=42, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    config = Config.from_json("configs/lob_d2_mv.json")
    config.eqn.law_encoder_type = "moments"
    config.eqn.subnet_type = subnet_type
    config.eqn.phi = 0.1
    config.net.opt_config1.num_iterations = inner
    config.net.logging_frequency = inner
    config.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_mv"](config.eqn)
    fp = FictitiousPlaySolver(config, bsde, device=device,
        outer_iterations=outer, inner_iterations=inner,
        w2_threshold=0.005, damping_alpha=damping, n_sim_agents=n_agents, warm_start=True)
    return fp.train()

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

def exp_fd_grid():
    """FD grid convergence: does spread -> 2/alpha as Delta -> 0?"""
    results = {}
    for Delta in [2.0, 1.0, 0.5, 0.25, 0.1]:
        fd = solve_fd(0.4, Delta=Delta, N_t=400)
        results[f"Delta={Delta}"] = fd
        log(f"  Delta={Delta:.2f}: spread={fd['spread']:.6f}")
    log(f"  2/alpha = {2/1.5:.6f}")
    save(results, "fd_grid_convergence.json")
    return results

def exp_fd_fp():
    """FD fictitious play: ground truth equilibrium h."""
    h = 1.0; beta = 0.5; history = []
    for k in range(20):
        q_grid = np.arange(-5, 6, 1.0); nq = len(q_grid); mid = nq//2
        _, V, da_g, db_g = _fd_full(h)
        # Simulate
        q_sim = np.zeros(2000); dt = 1.0/200
        for t in range(200):
            qi = np.clip(np.round(q_sim - q_grid[0]).astype(int), 0, nq-1)
            fa = h*np.exp(-1.5*np.clip(da_g[t,qi],0.001,10)); fb = h*np.exp(-1.5*np.clip(db_g[t,qi],0.001,10))
            q_sim = np.clip(q_sim+(fb-fa)*dt+np.sqrt(np.maximum(fb+fa,1e-8))*np.random.normal(0,np.sqrt(dt),2000),-10,10)
        h_imp = max(0.01, min(float(np.exp(-beta*np.std(q_sim))), 1.0))
        h_new = 0.3*h_imp + 0.7*h
        history.append({"iter":k+1,"h":h,"h_implied":h_imp,"h_new":h_new,"q_std":float(np.std(q_sim)),
                        "V":float(V[0,mid]),"spread":float(da_g[0,mid]+db_g[0,mid])})
        h = h_new
    log(f"  h trajectory: {' '.join(f'{x['h']:.3f}' for x in history)}")
    log(f"  Final h={h:.4f}")
    save({"history":history,"final_h":h}, "fd_fictitious_play.json")
    return {"history":history,"final_h":h}

def _fd_full(h_val, phi=0.1, alpha=1.5, r=0.1, T=1.0, H=5, Delta=1.0, N_t=200):
    dt=T/N_t; q=np.arange(-H,H+Delta,Delta); nq=len(q)
    V=np.zeros((N_t+1,nq)); da=np.zeros((N_t+1,nq)); db=np.zeros((N_t+1,nq))
    for j in range(nq): V[N_t,j]=-phi*q[j]**2
    for n in range(N_t-1,-1,-1):
        for j in range(nq):
            Vh=V[n+1,j]; Vd=V[n+1,j-1] if j>0 else -phi*(q[j]-Delta)**2; Vu=V[n+1,j+1] if j<nq-1 else -phi*(q[j]+Delta)**2
            d_a=max(1/alpha-(Vd-Vh)/Delta,0.001); d_b=max(1/alpha-(Vu-Vh)/Delta,0.001)
            ra=h_val*np.exp(-alpha*d_a); rb=h_val*np.exp(-alpha*d_b)
            V[n,j]=Vh+dt*(ra*(d_a*Delta+Vd-Vh)+rb*(d_b*Delta+Vu-Vh)-phi*q[j]**2-r*Vh)
            da[n,j]=d_a; db[n,j]=d_b
    return q,V,da,db

def exp_equilibrium(device):
    """Neural FP vs FD equilibrium + linearity test."""
    r_eq = run_fp("two_stream", 15, 1500, 0.1, device=device)
    fp_h = r_eq["history"][-1]["h"]; fp_y0 = r_eq["history"][-1]["y0"]; fp_w2 = r_eq["history"][-1]["w2"]
    fd = solve_fd(fp_h); fd_nc = solve_fd(1.0)
    log(f"  Neural FP: h={fp_h:.4f}, Y0={fp_y0:.4f}, W2={fp_w2:.4f}")
    log(f"  FD at h={fp_h:.4f}: V={fd['V']:.4f}, spread={fd['spread']:.4f}")
    log(f"  FD baseline (h=1): V={fd_nc['V']:.4f}")
    # Linearity test
    gpu_reset()
    _, model, bsde = train_mv("two_stream", 3000, device)
    model.eval()
    log(f"  Linearity test:")
    for std_q in [0.1, 1.0, 3.0]:
        pop = generate_pop(256, std_q)
        le = model.law_encoder.encode(torch.tensor(pop, dtype=torch.float64, device=device))
        q_grid = np.linspace(-4, 4, 30); zqs = []
        with torch.no_grad():
            for qv in q_grid:
                a = torch.tensor([[100.0, qv]], dtype=torch.float64, device=device)
                z = model.subnet[0](torch.cat([a, le.unsqueeze(0)], dim=1)) / bsde.dim
                zqs.append(z[:, 1].item())
        zqs = np.array(zqs); coeffs = np.polyfit(q_grid, zqs, 1)
        ss_res = np.sum((zqs - np.polyval(coeffs, q_grid))**2)
        ss_tot = np.sum((zqs - np.mean(zqs))**2)
        r2 = 1 - ss_res/max(ss_tot, 1e-10)
        log(f"    std={std_q}: Z = {coeffs[0]:.4f}*q + {coeffs[1]:.4f}, R²={r2:.4f}")
    save({"fp_h":fp_h,"fp_y0":fp_y0,"fp_w2":fp_w2,"fd":fd,"fd_baseline":fd_nc}, "equilibrium.json")
    return {"fp_h":fp_h}

def exp_scale(device):
    """Scale test: K=1,2,5 assets."""
    results = {}
    for K in [1, 2, 5]:
        gpu_reset(); torch.manual_seed(42); np.random.seed(42)
        log(f"  K={K} (d={2*K})...")
        t0 = time.time()
        if K == 1:
            r, model, bsde = train_mv("two_stream", 3000, device)
            y0 = r["y0"]; loss = r["final_loss"]
        else:
            cfg = Config.from_json("configs/lob_multiasset_k5.json")
            cfg.eqn.n_assets = K; cfg.eqn.dim = 2*K
            cfg.net.opt_config1.num_iterations = 3000 if K < 5 else 5000
            cfg.net.logging_frequency = cfg.net.opt_config1.num_iterations
            cfg.net.verbose = False
            bsde = EQUATION_REGISTRY["contxiong_lob_multiasset"](cfg.eqn)
            slv = ContXiongLOBMVSolver(cfg, bsde, device=device)
            r = slv.train(); y0 = r["y0"]; loss = r["final_loss"]
        results[f"K={K}"] = {"dim":2*K, "y0":y0, "loss":loss, "time_min":(time.time()-t0)/60}
        log(f"  K={K}: Y0={y0:.4f}, loss={loss:.4e}, {(time.time()-t0)/60:.0f}m")
    save(results, "scale_test.json")
    return results

def exp_film_fp(device):
    """FiLM + FP: does FiLM converge to different equilibrium?"""
    r = run_fp("film", 15, 1500, 0.1, device=device)
    w2s = [h["w2"] for h in r["history"]]; hs = [h["h"] for h in r["history"]]
    log(f"  W2 final={w2s[-1]:.4f}, h final={hs[-1]:.4f}")
    save({"history":r["history"],"converged":r["converged"]}, "film_fp.json")
    return r

def exp_common_noise(device):
    """Common noise sweep."""
    results = {}
    for sc in [0.0, 0.1, 0.2, 0.5, 1.0]:
        gpu_reset(); torch.manual_seed(42); np.random.seed(42)
        cfg = Config.from_json("configs/lob_d2_mv.json")
        cfg.eqn.law_encoder_type = "moments"; cfg.eqn.sigma_common = sc
        cfg.net.opt_config1.num_iterations = 3000; cfg.net.logging_frequency = 3000; cfg.net.verbose = False
        bsde = EQUATION_REGISTRY["contxiong_lob_common_noise"](cfg.eqn)
        slv = ContXiongLOBMVSolver(cfg, bsde, device=device)
        r = slv.train(); slv.model.eval()
        h_n = eval_h(slv.model, bsde, 0.1, device); h_w = eval_h(slv.model, bsde, 3.0, device)
        results[f"sigma={sc}"] = {"y0":r["y0"],"h_narrow":h_n,"h_wide":h_w,"h_gap":h_n-h_w}
        log(f"  sigma={sc:.1f}: Y0={r['y0']:.4f}, h_gap={h_n-h_w:.4f}")
    save(results, "common_noise.json")
    return results

def exp_horizon(device):
    """Time horizon sweep (Germain 2022 failure mode)."""
    results = {}
    for T in [0.5, 1.0, 2.0, 5.0, 10.0]:
        gpu_reset(); torch.manual_seed(42); np.random.seed(42)
        cfg = Config.from_json("configs/lob_d2_mv.json")
        cfg.eqn.law_encoder_type = "moments"; cfg.eqn.total_time = T
        cfg.net.opt_config1.num_iterations = 3000; cfg.net.logging_frequency = 3000; cfg.net.verbose = False
        bsde = EQUATION_REGISTRY["contxiong_lob_mv"](cfg.eqn)
        slv = ContXiongLOBMVSolver(cfg, bsde, device=device)
        r = slv.train(); slv.model.eval()
        zm = slv.model._last_z_max_overall if hasattr(slv.model,"_last_z_max_overall") else 0
        diverged = np.isnan(r["y0"]) or abs(r["y0"])>10 or r["final_loss"]>10
        results[f"T={T}"] = {"y0":r["y0"],"loss":r["final_loss"],"z_max":zm,"diverged":diverged}
        s = "DIV" if diverged else "OK"
        log(f"  T={T}: [{s}] Y0={r['y0']:.4f}, max|Z|={zm:.4f}")
    save(results, "horizon_sweep.json")
    return results

def exp_harder_impact(device):
    """Push impact harder to find actual phase transition."""
    results = {}
    log("  Extreme kappa (sqrt impact):")
    for kappa in [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]:
        gpu_reset(); torch.manual_seed(42); np.random.seed(42)
        cfg = Config.from_json("configs/lob_d2_mv_impact.json")
        cfg.eqn.kappa = kappa; cfg.eqn.impact_type = "sqrt"
        cfg.net.opt_config1.num_iterations = 2000; cfg.net.logging_frequency = 2000; cfg.net.verbose = False
        bsde = EQUATION_REGISTRY["contxiong_lob_impact"](cfg.eqn)
        slv = ContXiongLOBMVSolver(cfg, bsde, device=device)
        try:
            r = slv.train(); y0=r["y0"]; loss=r["final_loss"]
            zm = slv.model._last_z_max_overall if hasattr(slv.model,"_last_z_max_overall") else 0
            diverged = np.isnan(y0) or abs(y0)>10 or loss>10 or zm>100
        except: y0=None; loss=None; zm=None; diverged=True
        results[f"sqrt_{kappa}"] = {"y0":y0,"loss":loss,"z_max":zm,"diverged":diverged,"kappa":kappa}
        s="DIV" if diverged else "OK"
        y=f"{y0:.4f}" if y0 is not None else "NaN"
        log(f"    kappa={kappa:5.1f}: [{s}] Y0={y}")

    log("\n  Combined stress (sqrt, T=5):")
    for kappa in [0.0, 1.0, 5.0, 10.0, 20.0]:
        gpu_reset(); torch.manual_seed(42); np.random.seed(42)
        cfg = Config.from_json("configs/lob_d2_mv_impact.json")
        cfg.eqn.kappa=kappa; cfg.eqn.impact_type="sqrt"; cfg.eqn.total_time=5.0
        cfg.net.opt_config1.num_iterations=2000; cfg.net.logging_frequency=2000; cfg.net.verbose=False
        bsde = EQUATION_REGISTRY["contxiong_lob_impact"](cfg.eqn)
        slv = ContXiongLOBMVSolver(cfg, bsde, device=device)
        try:
            r=slv.train(); y0=r["y0"]; loss=r["final_loss"]
            zm=slv.model._last_z_max_overall if hasattr(slv.model,"_last_z_max_overall") else 0
            diverged=np.isnan(y0) or abs(y0)>10 or loss>10 or zm>100
        except: y0=None; loss=None; zm=None; diverged=True
        results[f"sqrt_T5_{kappa}"]={"y0":y0,"loss":loss,"z_max":zm,"diverged":diverged,"kappa":kappa,"T":5.0}
        s="DIV" if diverged else "OK"; y=f"{y0:.4f}" if y0 is not None else "NaN"
        log(f"    kappa={kappa:5.1f} T=5: [{s}] Y0={y}")
    save(results, "harder_impact.json")
    return results

def exp_fp_more_agents(device):
    """FP with more simulation agents."""
    results = {}
    for n in [256, 1024, 2048]:
        gpu_reset()
        log(f"  n_agents={n}...")
        r = run_fp("two_stream", 15, 1500, 0.1, n_agents=n, device=device)
        w2s = [h["w2"] for h in r["history"]]
        results[f"n={n}"] = {"final_w2":w2s[-1], "min_w2":min(w2s), "w2s":w2s}
        log(f"    final W2={w2s[-1]:.4f}, min={min(w2s):.4f}")
        save(results, "fp_more_agents.json")
    return results

def exp_jump_bsde(device):
    """Jump BSDE vs diffusion: does learning U_a, U_b fix the spread?"""
    results = {}
    # Jump BSDE (subnet outputs 4: Z_s, Z_q, U_a, U_b)
    log("  Training jump BSDE with learned U_a, U_b (5000 iter)...")
    torch.manual_seed(42); np.random.seed(42)
    cfg = Config.from_json("configs/lob_d2_mv_jump.json")
    cfg.net.opt_config1.num_iterations = 5000; cfg.net.logging_frequency = 1000; cfg.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_jump"](cfg.eqn)
    slv = ContXiongLOBMVSolver(cfg, bsde, device=device)
    r = slv.train(); slv.model.eval()
    log(f"  Jump: Y0={r['y0']:.4f}, loss={r['final_loss']:.4e}")

    pop = generate_pop(256, 1.0)
    a = torch.tensor([[100.0, 0.0]], dtype=torch.float64, device=device)
    le = slv.model.law_encoder.encode(torch.tensor(pop, dtype=torch.float64, device=device))
    with torch.no_grad():
        h = bsde.compute_competitive_factor(le).item()
        z_full = slv.model.subnet[0](torch.cat([a, le.unsqueeze(0)], dim=1)) / bsde.dim
        U_a = z_full[:, 2:3].item()  # V(q-1) - V(q)
        U_b = z_full[:, 3:4].item()  # V(q+1) - V(q)
        da_j = 1.0/bsde.alpha - U_a
        db_j = 1.0/bsde.alpha - U_b

    # Diffusion BSDE for comparison
    gpu_reset()
    log("  Training diffusion BSDE (5000 iter)...")
    torch.manual_seed(42); np.random.seed(42)
    _, m2, b2 = train_mv("two_stream", 5000, device)
    m2.eval()
    le2 = m2.law_encoder.encode(torch.tensor(pop, dtype=torch.float64, device=device))
    with torch.no_grad():
        z2 = m2.subnet[0](torch.cat([a, le2.unsqueeze(0)], dim=1)) / b2.dim
        da_d = (1.0/b2.alpha + z2[:, 1:2]/b2._sigma_q_equilibrium()).item()
        db_d = (1.0/b2.alpha - z2[:, 1:2]/b2._sigma_q_equilibrium()).item()

    fd = solve_fd(h)
    results = {
        "jump_spread": da_j+db_j, "jump_da": da_j, "jump_db": db_j,
        "jump_U_a": U_a, "jump_U_b": U_b,
        "diffusion_spread": da_d+db_d,
        "fd_spread": fd["spread"], "fd_da": fd["da"], "fd_V": fd["V"],
        "target_2_over_alpha": 2/1.5,
        "jump_y0": r["y0"], "h": h,
    }
    log(f"\n  Spread comparison at q=0:")
    log(f"    Jump BSDE:      {da_j+db_j:.4f}  (U_a={U_a:.4f}, U_b={U_b:.4f})")
    log(f"    Diffusion BSDE: {da_d+db_d:.4f}  (structurally = 2/alpha)")
    log(f"    FD (Delta=1):   {fd['spread']:.4f}  (ground truth)")
    log(f"    2/alpha:        {2/1.5:.4f}")
    gap_jump = abs((da_j+db_j) - fd["spread"])
    gap_diff = abs((da_d+db_d) - fd["spread"])
    log(f"    Gap from FD: jump={gap_jump:.4f}, diffusion={gap_diff:.4f}")
    save(results, "jump_bsde.json")
    return results


def exp_h_mapping(device):
    """Diagnose h mismatch: extract neural h(std) mapping, compare to FD FP formula."""
    log("  Training model to extract learned h mapping...")
    _, model, bsde = train_mv("two_stream", 3000, device)
    model.eval()
    results = {"neural_h": {}, "fd_formula_h": {}}
    log(f"  {'std':>6} {'neural_h':>10} {'exp(-0.5*std)':>14} {'ratio':>8}")
    for std_q in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        pop = generate_pop(256, std_q)
        particles = torch.tensor(pop, dtype=torch.float64, device=device)
        le = model.law_encoder.encode(particles)
        with torch.no_grad():
            h_nn = bsde.compute_competitive_factor(le).item()
        h_fd = float(np.exp(-0.5 * std_q))
        results["neural_h"][f"std={std_q}"] = h_nn
        results["fd_formula_h"][f"std={std_q}"] = h_fd
        ratio = h_nn / max(h_fd, 1e-6)
        log(f"  {std_q:6.1f} {h_nn:10.4f} {h_fd:14.4f} {ratio:8.2f}")

    # The FD FP used h=exp(-0.5*std) which is ARBITRARY.
    # Now run FD FP using the NEURAL h mapping instead.
    log(f"\n  Running FD FP with neural h mapping...")
    # Build interpolation: std -> neural_h
    stds_map = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    hs_map = [results["neural_h"][f"std={s}"] for s in stds_map]

    h = 1.0; history = []
    for k in range(20):
        q_grid, V, da_g, db_g = _fd_full(h)
        nq = len(q_grid); mid = nq // 2; dt = 1.0/200
        q_sim = np.zeros(2000)
        for t in range(200):
            qi = np.clip(np.round(q_sim - q_grid[0]).astype(int), 0, nq-1)
            fa = h*np.exp(-1.5*np.clip(da_g[t,qi],0.001,10))
            fb = h*np.exp(-1.5*np.clip(db_g[t,qi],0.001,10))
            q_sim = np.clip(q_sim+(fb-fa)*dt+np.sqrt(np.maximum(fb+fa,1e-8))*np.random.normal(0,np.sqrt(dt),2000),-10,10)
        q_std = float(np.std(q_sim))
        # Use neural h mapping (interpolate)
        h_implied = float(np.interp(q_std, stds_map, hs_map))
        h_new = 0.3 * h_implied + 0.7 * h
        history.append({"iter":k+1, "h":h, "h_implied":h_implied, "q_std":q_std, "V":float(V[0,mid])})
        h = h_new

    log(f"  FD FP with neural mapping:")
    log(f"  h: {' '.join(f'{x['h']:.3f}' for x in history)}")
    log(f"  Final h={h:.4f}")
    results["fd_fp_neural_mapping"] = {"history": history, "final_h": h}

    # Compare: FD FP with exp(-0.5*std) gave h=0.73, what does neural mapping give?
    log(f"\n  Comparison:")
    log(f"    FD FP with exp(-0.5*std): h=0.732 (from earlier run)")
    log(f"    FD FP with neural mapping: h={h:.4f}")
    log(f"    Neural FP directly: h~0.24 (from earlier run)")
    save(results, "h_mapping_diagnostic.json")
    return results


# ================================================================
def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()

    # Already completed (from results_missing/): FD grid convergence, FD FP
    # Already completed (from results_overnight/): FP ts, FP film, damping, multi-seed, FD match, impact 24

    # GPU experiments (each isolated with gpu_reset)
    r1 = run_safe("1. H mapping diagnostic", lambda: exp_h_mapping(device))
    r3 = run_safe("2. Equilibrium comparison + linearity", lambda: exp_equilibrium(device))
    r4 = run_safe("3. Scale test K=1,2,5", lambda: exp_scale(device))
    r5 = run_safe("4. FiLM + FP", lambda: exp_film_fp(device))
    r6 = run_safe("5. Common noise sweep", lambda: exp_common_noise(device))
    r7 = run_safe("6. Time horizon sweep", lambda: exp_horizon(device))
    r8 = run_safe("7. Harder impact", lambda: exp_harder_impact(device))
    r9 = run_safe("8. FP more agents", lambda: exp_fp_more_agents(device))
    r10 = run_safe("9. Jump BSDE (fixed: learns U_a, U_b separately)", lambda: exp_jump_bsde(device))

    elapsed = (time.time() - start) / 60
    log(f"\n{'='*60}")
    log(f"ALL DONE in {elapsed:.0f} min ({elapsed/60:.1f} hours)")
    log(f"{'='*60}")
    names = ["h_mapping","equilibrium","scale","film_fp","common_noise","horizon","harder_impact","fp_agents","jump_bsde"]
    results = [r1,r3,r4,r5,r6,r7,r8,r9,r10]
    for n, r in zip(names, results):
        log(f"  {n}: {'OK' if r else 'CRASHED'}")
    save({"completed":[n for n,r in zip(names,results) if r],
          "crashed":[n for n,r in zip(names,results) if not r]}, "summary.json")

if __name__ == "__main__":
    main()

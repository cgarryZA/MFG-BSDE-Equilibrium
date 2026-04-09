#!/usr/bin/env python
"""
Final run: only the experiments that never completed + longer jump BSDE.

1. Common noise sweep (~50 min)
2. Time horizon sweep (~40 min)
3. Harder impact stress test (~60 min)
4. Jump BSDE with 10000 iter (~30 min)

Total: ~3 hours. Should complete before CUDA crashes.
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
from solver import ContXiongLOBMVSolver

OUT = "results_final"
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
    time.sleep(2)  # let GPU cool

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
    log(name)
    log("=" * 60)
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

def exp_common_noise(device):
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
    results = {}
    for T in [0.5, 1.0, 2.0, 5.0, 10.0]:
        gpu_reset(); torch.manual_seed(42); np.random.seed(42)
        cfg = Config.from_json("configs/lob_d2_mv.json")
        cfg.eqn.law_encoder_type = "moments"; cfg.eqn.total_time = T
        cfg.net.opt_config1.num_iterations = 3000; cfg.net.logging_frequency = 3000; cfg.net.verbose = False
        bsde = EQUATION_REGISTRY["contxiong_lob_mv"](cfg.eqn)
        slv = ContXiongLOBMVSolver(cfg, bsde, device=device)
        r = slv.train(); slv.model.eval()
        zm = slv.model._last_z_max_overall if hasattr(slv.model, "_last_z_max_overall") else 0
        diverged = np.isnan(r["y0"]) or abs(r["y0"]) > 10 or r["final_loss"] > 10
        h_n = eval_h(slv.model, bsde, 0.1, device); h_w = eval_h(slv.model, bsde, 3.0, device)
        results[f"T={T}"] = {"y0":r["y0"],"loss":r["final_loss"],"z_max":zm,
                              "h_gap":h_n-h_w,"diverged":diverged}
        s = "DIV" if diverged else "OK"
        log(f"  T={T}: [{s}] Y0={r['y0']:.4f}, max|Z|={zm:.4f}, h_gap={h_n-h_w:.4f}")
        save(results, "horizon_sweep.json")
    return results

def exp_harder_impact(device):
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
        s="DIV" if diverged else "OK"; y=f"{y0:.4f}" if y0 is not None else "NaN"
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
        results[f"sqrt_T5_{kappa}"]={"y0":y0,"loss":loss,"z_max":zm,"diverged":diverged}
        s="DIV" if diverged else "OK"; y=f"{y0:.4f}" if y0 is not None else "NaN"
        log(f"    kappa={kappa:5.1f} T=5: [{s}] Y0={y}")
    save(results, "harder_impact.json")
    return results

def exp_jump_bsde_long(device):
    """Jump BSDE with 10000 iterations — needs more training for U_a, U_b."""
    log("  Training jump BSDE (10000 iter)...")
    torch.manual_seed(42); np.random.seed(42)
    cfg = Config.from_json("configs/lob_d2_mv_jump.json")
    cfg.net.opt_config1.num_iterations = 10000
    cfg.net.opt_config1.lr_boundaries = [4000, 8000]
    cfg.net.logging_frequency = 2000
    cfg.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_jump"](cfg.eqn)
    slv = ContXiongLOBMVSolver(cfg, bsde, device=device)
    r = slv.train(); slv.model.eval()
    log(f"  Y0={r['y0']:.4f}, loss={r['final_loss']:.4e}")

    pop = generate_pop(256, 1.0)
    a = torch.tensor([[100.0, 0.0]], dtype=torch.float64, device=device)
    le = slv.model.law_encoder.encode(torch.tensor(pop, dtype=torch.float64, device=device))
    with torch.no_grad():
        h = bsde.compute_competitive_factor(le).item()
        z_full = slv.model.subnet[0](torch.cat([a, le.unsqueeze(0)], dim=1)) / bsde.dim
        U_a = z_full[:, 2:3].item()
        U_b = z_full[:, 3:4].item()
        da_j = 1.0/bsde.alpha - U_a
        db_j = 1.0/bsde.alpha - U_b

    # Also check at multiple q values
    log(f"\n  Jump BSDE quotes at various q:")
    log(f"  {'q':>6} {'da':>8} {'db':>8} {'spread':>8} {'U_a':>8} {'U_b':>8}")
    for q_val in [-3, -2, -1, 0, 1, 2, 3]:
        with torch.no_grad():
            agent = torch.tensor([[100.0, float(q_val)]], dtype=torch.float64, device=device)
            z_f = slv.model.subnet[0](torch.cat([agent, le.unsqueeze(0)], dim=1)) / bsde.dim
            ua = z_f[:, 2:3].item(); ub = z_f[:, 3:4].item()
            d_a = 1.0/bsde.alpha - ua; d_b = 1.0/bsde.alpha - ub
        log(f"  {q_val:6d} {d_a:8.4f} {d_b:8.4f} {d_a+d_b:8.4f} {ua:8.4f} {ub:8.4f}")

    # Compare to FD
    fd = solve_fd(h)
    gpu_reset()

    # Diffusion comparison
    log(f"\n  Training diffusion BSDE (5000 iter) for comparison...")
    _, m2, b2 = train_mv("two_stream", 5000, device)
    m2.eval()
    le2 = m2.law_encoder.encode(torch.tensor(pop, dtype=torch.float64, device=device))
    with torch.no_grad():
        z2 = m2.subnet[0](torch.cat([a, le2.unsqueeze(0)], dim=1)) / b2.dim
        da_d = (1.0/b2.alpha + z2[:, 1:2]/b2._sigma_q_equilibrium()).item()
        db_d = (1.0/b2.alpha - z2[:, 1:2]/b2._sigma_q_equilibrium()).item()

    results = {
        "jump_spread": da_j+db_j, "jump_da": da_j, "jump_db": db_j,
        "jump_U_a": U_a, "jump_U_b": U_b,
        "diffusion_spread": da_d+db_d,
        "fd_spread": fd["spread"], "fd_da": fd["da"], "fd_V": fd["V"],
        "jump_y0": r["y0"], "jump_loss": r["final_loss"], "h": h,
        "n_iters": 10000,
    }
    log(f"\n  Spread comparison at q=0:")
    log(f"    Jump BSDE (10k iter): {da_j+db_j:.4f}  (U_a={U_a:.4f}, U_b={U_b:.4f})")
    log(f"    Diffusion BSDE:       {da_d+db_d:.4f}")
    log(f"    FD (Delta=1):         {fd['spread']:.4f}")
    log(f"    2/alpha:              {2/1.5:.4f}")
    save(results, "jump_bsde_long.json")
    return results


def exp_cx_model(device):
    """Cont-Xiong proper execution probabilities (no learned h).

    Train the CX model and compare:
    1. Does it produce different spreads from the base model?
    2. Do quotes depend on population (through the competitive mechanism)?
    3. Run FP with CX model — does it converge to a different equilibrium?
    """
    results = {}

    # Train CX model
    log("  Training CX model (5000 iter)...")
    torch.manual_seed(42); np.random.seed(42)
    cfg = Config.from_json("configs/lob_d2_cx.json")
    cfg.net.opt_config1.num_iterations = 5000
    cfg.net.logging_frequency = 5000
    cfg.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_cx"](cfg.eqn)
    slv = ContXiongLOBMVSolver(cfg, bsde, device=device)
    r = slv.train(); slv.model.eval()
    log(f"  CX model: Y0={r['y0']:.4f}, loss={r['final_loss']:.4e}")

    # Evaluate quotes at different population stds
    log(f"\n  CX model quotes by population:")
    log(f"  {'std':>6} {'da':>8} {'db':>8} {'spread':>8}")
    for std_q in [0.1, 1.0, 3.0]:
        pop = generate_pop(256, std_q)
        particles = torch.tensor(pop, dtype=torch.float64, device=device)
        le = slv.model.law_encoder.encode(particles)
        with torch.no_grad():
            agent = torch.tensor([[100.0, 0.0]], dtype=torch.float64, device=device)
            leb = le.unsqueeze(0)
            si = torch.cat([agent, leb], dim=1)
            z = slv.model.subnet[0](si) / bsde.dim
            sig = bsde._sigma_q_equilibrium()
            da = (1.0/bsde.alpha + z[:, 1:2]/sig).item()
            db = (1.0/bsde.alpha - z[:, 1:2]/sig).item()
        results[f"std={std_q}"] = {"da": da, "db": db, "spread": da + db}
        log(f"  {std_q:6.1f} {da:8.4f} {db:8.4f} {da+db:8.4f}")

    # Compare: does CX produce different spread sensitivity than base?
    gpu_reset()
    log(f"\n  Training base model for comparison (5000 iter)...")
    _, m_base, b_base = train_mv("two_stream", 5000, device)
    m_base.eval()
    for std_q in [0.1, 1.0, 3.0]:
        pop = generate_pop(256, std_q)
        le = m_base.law_encoder.encode(torch.tensor(pop, dtype=torch.float64, device=device))
        with torch.no_grad():
            agent = torch.tensor([[100.0, 0.0]], dtype=torch.float64, device=device)
            z = m_base.subnet[0](torch.cat([agent, le.unsqueeze(0)], dim=1)) / b_base.dim
            sig = b_base._sigma_q_equilibrium()
            da = (1.0/b_base.alpha + z[:, 1:2]/sig).item()
            db = (1.0/b_base.alpha - z[:, 1:2]/sig).item()
        results[f"base_std={std_q}"] = {"da": da, "db": db, "spread": da + db}

    log(f"\n  Spread comparison (CX vs base, at q=0):")
    log(f"  {'std':>6} {'CX_spread':>10} {'base_spread':>12}")
    for std_q in [0.1, 1.0, 3.0]:
        cx_s = results[f"std={std_q}"]["spread"]
        base_s = results[f"base_std={std_q}"]["spread"]
        log(f"  {std_q:6.1f} {cx_s:10.4f} {base_s:12.4f}")

    results["cx_y0"] = r["y0"]
    save(results, "cx_model.json")
    return results


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()

    r1 = run_safe("1. Common noise sweep", lambda: exp_common_noise(device))
    r2 = run_safe("2. Time horizon sweep", lambda: exp_horizon(device))
    r3 = run_safe("3. Harder impact stress test", lambda: exp_harder_impact(device))
    r4 = run_safe("4. Jump BSDE (10000 iter, longer training)", lambda: exp_jump_bsde_long(device))
    r5 = run_safe("5. Cont-Xiong proper execution (CX model)", lambda: exp_cx_model(device))

    elapsed = (time.time() - start) / 60
    log(f"\n{'='*60}")
    log(f"ALL DONE in {elapsed:.0f} min ({elapsed/60:.1f} hours)")
    log(f"{'='*60}")
    names = ["common_noise", "horizon", "harder_impact", "jump_bsde_long", "cx_model"]
    results = [r1, r2, r3, r4, r5]
    for n, r in zip(names, results):
        log(f"  {n}: {'OK' if r else 'CRASHED'}")
    save({"completed": [n for n,r in zip(names,results) if r],
          "crashed": [n for n,r in zip(names,results) if not r]}, "summary.json")

if __name__ == "__main__":
    main()

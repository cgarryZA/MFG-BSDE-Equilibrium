#!/usr/bin/env python
"""Run experiments 7 (h-only), 8 (generalisation), 9 (penalty sweep).
Merges with existing results from experiments 1-6."""

import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from registry import EQUATION_REGISTRY
import equations
from solver import ContXiongLOBMVModel, ContXiongLOBMVSolver


def generate_population(n_agents, mean_q, std_q, s_init=100.0):
    S = np.full(n_agents, s_init)
    q = np.random.normal(mean_q, std_q, n_agents)
    q = np.clip(q, -10, 10)
    return np.stack([S, q], axis=1)


def generate_special_population(n_agents, family, s_init=100.0):
    S = np.full(n_agents, s_init)
    if family == "uniform":
        q = np.random.uniform(-1, 1, n_agents)
    elif family == "bimodal":
        q = np.concatenate([np.random.normal(-2, 0.3, n_agents//2),
                            np.random.normal(2, 0.3, n_agents - n_agents//2)])
    elif family == "skewed":
        q = np.random.chisquare(2, n_agents) - 2
    elif family == "heavy_tail":
        q = np.random.standard_t(3, n_agents)
    else:
        raise ValueError(family)
    q = np.clip(q, -10, 10)
    return np.stack([S, q], axis=1)


def evaluate_at_pop(model, bsde, pop_np, q_eval=0.0, device="cpu"):
    model.eval()
    device = next(model.parameters()).device
    alpha = bsde.alpha
    particles = torch.tensor(pop_np, dtype=torch.float64, device=device)
    law_embed = model.law_encoder.encode(particles)
    with torch.no_grad():
        h = bsde.compute_competitive_factor(law_embed).item()
        agent = torch.tensor([[100.0, q_eval]], dtype=torch.float64, device=device)
        leb = law_embed.unsqueeze(0)
        si = torch.cat([agent, leb], dim=1)
        z = model.subnet[0](si) / bsde.dim
        zq = z[:, 1:2]
        sig = bsde._sigma_q_equilibrium()
        p = zq / sig
        da = (1.0/alpha + p).item()
        db = (1.0/alpha - p).item()
    return {"h": h, "z_q": z[:, 1].item(), "delta_a": da, "delta_b": db}


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


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    n_iters = 3000
    base_mv = "configs/lob_d2_mv.json"
    results = {}
    start = time.time()

    # ================================================================
    # 7. H-ONLY CONTROL
    # ================================================================
    print(f"\n{'='*60}")
    print(f"7. H-only control ({n_iters} iter)")
    print(f"{'='*60}")
    torch.manual_seed(42); np.random.seed(42)
    config = Config.from_json(base_mv)
    config.eqn.law_encoder_type = "moments"
    config.net.opt_config1.num_iterations = n_iters
    config.net.logging_frequency = n_iters
    config.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_mv"](config.eqn)
    solver = ContXiongLOBMVSolver(config, bsde, device=device)
    # Enable h-only mode
    solver.model.h_only_mode = True
    result = solver.train()
    print(f"  H-only: Y0={result['y0']:.4f}, loss={result['final_loss']:.4e}")

    honly_evals = {}
    solver.model.eval()
    law_dim = bsde.law_embed_dim
    for std_q in [0.1, 1.0, 3.0]:
        pop = generate_population(256, 0.0, std_q)
        particles = torch.tensor(pop, dtype=torch.float64, device=device)
        law_embed = solver.model.law_encoder.encode(particles)
        with torch.no_grad():
            h_val = bsde.compute_competitive_factor(law_embed).item()
            agent = torch.tensor([[100.0, 0.0]], dtype=torch.float64, device=device)
            zero_law = torch.zeros(1, law_dim, dtype=torch.float64, device=device)
            si = torch.cat([agent, zero_law], dim=1)
            z = solver.model.subnet[0](si) / bsde.dim
        honly_evals[f"std={std_q}"] = {"h": h_val, "z_q": z[:, 1].item()}
        print(f"    std={std_q}: h={h_val:.4f}, z_q={z[:, 1].item():.6f}")
    results["h_only"] = {"y0": result["y0"], "loss": result["final_loss"], "evaluations": honly_evals}

    # ================================================================
    # 8. GENERALISATION
    # ================================================================
    print(f"\n{'='*60}")
    print(f"8. Generalisation ({n_iters} iter)")
    print(f"{'='*60}")
    torch.manual_seed(42); np.random.seed(42)
    config = Config.from_json(base_mv)
    config.eqn.law_encoder_type = "moments"
    config.net.opt_config1.num_iterations = n_iters
    config.net.logging_frequency = n_iters
    config.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_mv"](config.eqn)
    solver = ContXiongLOBMVSolver(config, bsde, device=device)
    r = solver.train()
    print(f"  Base: Y0={r['y0']:.4f}")

    gen = {}
    for family in ["uniform", "bimodal", "skewed", "heavy_tail"]:
        pop = generate_special_population(256, family)
        ev = evaluate_at_pop(solver.model, bsde, pop, device=device)
        gen[family] = ev
        print(f"  {family}: h={ev['h']:.4f}, da={ev['delta_a']:.4f}")
    results["generalisation"] = gen

    # ================================================================
    # 9. PENALTY SWEEP
    # ================================================================
    print(f"\n{'='*60}")
    print(f"9. Penalty sweep ({n_iters} iter each)")
    print(f"{'='*60}")
    pen = {}
    for phi in [0.01, 0.05, 0.1, 0.5]:
        torch.manual_seed(42); np.random.seed(42)
        config = Config.from_json(base_mv)
        config.eqn.law_encoder_type = "moments"
        config.eqn.phi = phi
        config.net.opt_config1.num_iterations = n_iters
        config.net.logging_frequency = n_iters
        config.net.verbose = False
        bsde = EQUATION_REGISTRY["contxiong_lob_mv"](config.eqn)
        solver = ContXiongLOBMVSolver(config, bsde, device=device)
        r = solver.train()
        print(f"  phi={phi}: Y0={r['y0']:.4f}")
        pop_n = generate_population(256, 0.0, 0.1)
        pop_w = generate_population(256, 0.0, 3.0)
        en = evaluate_at_pop(solver.model, bsde, pop_n, device=device)
        ew = evaluate_at_pop(solver.model, bsde, pop_w, device=device)
        pen[f"phi={phi}"] = {
            "y0": r["y0"], "h_narrow": en["h"], "h_wide": ew["h"],
            "h_gap": en["h"] - ew["h"],
            "da_narrow": en["delta_a"], "da_wide": ew["delta_a"],
            "policy_gap": abs(en["delta_a"] - ew["delta_a"]),
        }
        print(f"    h_gap={en['h']-ew['h']:.4f}, policy_gap={abs(en['delta_a']-ew['delta_a']):.4f}")
    results["penalty_sweep"] = pen

    elapsed = time.time() - start
    results["metadata"] = {"elapsed_seconds": elapsed}

    out = "results_paper_final/remaining_results.json"
    os.makedirs("results_paper_final", exist_ok=True)
    with open(out, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nDone in {elapsed/60:.1f} min. Saved: {out}")


if __name__ == "__main__":
    main()

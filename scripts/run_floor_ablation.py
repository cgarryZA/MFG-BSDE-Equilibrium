#!/usr/bin/env python
"""
Floor ablation + clipped policy analysis.

1. Train MV model with h floor at 0.01 (lowered from previous 0.1)
2. Report h(narrow), h(medium), h(wide)
3. Report unclipped AND clipped quotes at multiple q values
4. Compare to no-coupling baseline

Usage:
    python scripts/run_floor_ablation.py --device cuda
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from registry import EQUATION_REGISTRY
import equations
from solver import ContXiongLOBMVModel, ContXiongLOBMVSolver, ContXiongLOBSolver


def generate_population(n_agents, mean_q, std_q, s_init=100.0):
    S = np.full(n_agents, s_init)
    q = np.random.normal(mean_q, std_q, n_agents)
    q = np.clip(q, -10, 10)
    return np.stack([S, q], axis=1)


def evaluate_full(model, bsde, std_q, n_agents=256, device="cpu"):
    """Evaluate h, unclipped quotes, and clipped quotes for a population."""
    model.eval()
    device = next(model.parameters()).device
    alpha = bsde.alpha

    pop = generate_population(n_agents, mean_q=0.0, std_q=std_q)
    particles = torch.tensor(pop, dtype=torch.float64, device=device)
    law_embed = model.law_encoder.encode(particles)

    # h
    with torch.no_grad():
        h = model.bsde.compute_competitive_factor(law_embed).item()

    # Evaluate quotes at multiple q values
    q_values = [0.0, 1.0, 2.0, -1.0, -2.0]
    results = {"h": h, "quotes": {}}

    for q_val in q_values:
        agent_state = torch.tensor([[100.0, q_val]], dtype=torch.float64, device=device)
        law_embed_batch = law_embed.unsqueeze(0)
        subnet_input = torch.cat([agent_state, law_embed_batch], dim=1)

        with torch.no_grad():
            z = model.subnet[0](subnet_input) / bsde.dim
            z_q = z[:, 1:2]

            sigma_q = bsde._sigma_q_equilibrium()
            p = z_q / sigma_q
            base_spread = 1.0 / alpha
            delta_a_raw = (base_spread + p).item()
            delta_b_raw = (base_spread - p).item()

            # Clipped: enforce non-negative spreads
            eps = 0.01
            delta_a_clip = max(eps, min(delta_a_raw, 10.0 / alpha))
            delta_b_clip = max(eps, min(delta_b_raw, 10.0 / alpha))

        results["quotes"][f"q={q_val}"] = {
            "delta_a_raw": delta_a_raw,
            "delta_b_raw": delta_b_raw,
            "delta_a_clip": delta_a_clip,
            "delta_b_clip": delta_b_clip,
            "z_q": z[:, 1].item(),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_iters", type=int, default=3000)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    print(f"Device: {device}")

    # ================================================================
    # Train MV model (floor = 0.01)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Training MV model with h floor = 0.01 ({args.n_iters} iter)")
    print(f"{'='*60}")

    config = Config.from_json("configs/lob_d2_mv.json")
    config.eqn.law_encoder_type = "moments"
    config.net.opt_config1.num_iterations = args.n_iters
    config.net.logging_frequency = args.n_iters // 5

    bsde = EQUATION_REGISTRY["contxiong_lob_mv"](config.eqn)
    solver = ContXiongLOBMVSolver(config, bsde, device=device)
    result = solver.train()
    model = solver.model
    print(f"Y0={result['y0']:.4f}, loss={result['final_loss']:.4e}")

    # ================================================================
    # Evaluate across population shapes
    # ================================================================
    print(f"\n{'='*60}")
    print("Law sensitivity with lowered floor (h_min=0.01)")
    print(f"{'='*60}")

    stds = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    all_evals = {}

    for std_q in stds:
        r = evaluate_full(model, bsde, std_q, device=device)
        all_evals[f"std={std_q}"] = r
        q0 = r["quotes"]["q=0.0"]
        print(f"\nstd={std_q:.1f}: h={r['h']:.4f}")
        print(f"  q=0 unclipped: delta_a={q0['delta_a_raw']:.4f}, delta_b={q0['delta_b_raw']:.4f}")
        print(f"  q=0 clipped:   delta_a={q0['delta_a_clip']:.4f}, delta_b={q0['delta_b_clip']:.4f}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: h values and quote comparison at q=0")
    print(f"{'='*60}")
    print(f"{'std_q':>6} {'h':>8} {'da_raw':>8} {'db_raw':>8} {'da_clip':>8} {'db_clip':>8} {'skew_raw':>10} {'skew_clip':>10}")
    print("-" * 78)
    for std_q in stds:
        r = all_evals[f"std={std_q}"]
        q0 = r["quotes"]["q=0.0"]
        skew_raw = q0["delta_a_raw"] - q0["delta_b_raw"]
        skew_clip = q0["delta_a_clip"] - q0["delta_b_clip"]
        print(f"{std_q:6.1f} {r['h']:8.4f} {q0['delta_a_raw']:8.4f} {q0['delta_b_raw']:8.4f} "
              f"{q0['delta_a_clip']:8.4f} {q0['delta_b_clip']:8.4f} {skew_raw:10.4f} {skew_clip:10.4f}")

    # h variation
    h_vals = [all_evals[f"std={s}"]["h"] for s in stds]
    print(f"\nh range: {min(h_vals):.4f} to {max(h_vals):.4f} ({max(h_vals)/max(min(h_vals), 1e-6):.1f}x)")

    # Does clipping eliminate the variation?
    da_clip_vals = [all_evals[f"std={s}"]["quotes"]["q=0.0"]["delta_a_clip"] for s in stds]
    da_raw_vals = [all_evals[f"std={s}"]["quotes"]["q=0.0"]["delta_a_raw"] for s in stds]
    print(f"delta_a raw range:     {min(da_raw_vals):.4f} to {max(da_raw_vals):.4f}")
    print(f"delta_a clipped range: {min(da_clip_vals):.4f} to {max(da_clip_vals):.4f}")

    # Also evaluate at q=2 (where skew is most interesting)
    print(f"\n{'='*60}")
    print("Quotes at q=2 (long inventory)")
    print(f"{'='*60}")
    print(f"{'std_q':>6} {'da_raw':>8} {'db_raw':>8} {'da_clip':>8} {'db_clip':>8} {'skew_raw':>10} {'skew_clip':>10}")
    print("-" * 72)
    for std_q in stds:
        q2 = all_evals[f"std={std_q}"]["quotes"]["q=2.0"]
        skew_raw = q2["delta_a_raw"] - q2["delta_b_raw"]
        skew_clip = q2["delta_a_clip"] - q2["delta_b_clip"]
        print(f"{std_q:6.1f} {q2['delta_a_raw']:8.4f} {q2['delta_b_raw']:8.4f} "
              f"{q2['delta_a_clip']:8.4f} {q2['delta_b_clip']:8.4f} {skew_raw:10.4f} {skew_clip:10.4f}")

    # ================================================================
    # No-coupling baseline for comparison
    # ================================================================
    print(f"\n{'='*60}")
    print("No-coupling baseline")
    print(f"{'='*60}")
    config_nc = Config.from_json("configs/lob_d2.json")
    config_nc.eqn.type = 1
    config_nc.net.opt_config1.num_iterations = args.n_iters
    config_nc.net.opt_config1.freq_update_drift = 9999999
    config_nc.net.logging_frequency = args.n_iters // 5
    config_nc.net.verbose = False
    bsde_nc = EQUATION_REGISTRY["contxiong_lob"](config_nc.eqn)
    solver_nc = ContXiongLOBSolver(config_nc, bsde_nc, device=device)
    r_nc = solver_nc.train()
    print(f"No-coupling Y0={r_nc['y0']:.4f}")

    # Save results
    os.makedirs("results_floor_ablation", exist_ok=True)

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

    save_data = {
        "mv_y0": result["y0"],
        "mv_loss": result["final_loss"],
        "no_coupling_y0": r_nc["y0"],
        "h_floor": 0.01,
        "evaluations": all_evals,
        "n_iters": args.n_iters,
    }
    with open("results_floor_ablation/results.json", "w") as f:
        json.dump(convert(save_data), f, indent=2)
    print("\nSaved results_floor_ablation/results.json")


if __name__ == "__main__":
    main()

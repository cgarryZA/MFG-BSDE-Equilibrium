#!/usr/bin/env python
"""
FiLM conditioning experiments:
1. Ablation: train two_stream vs film vs film_additive, compare h/Z_q/quotes
2. Interaction test: evaluate Z_q on (q, population_std) grid, measure cross-partial
3. H-only control: confirm generator pathway is still insufficient with FiLM

Usage:
    python scripts/run_film_experiments.py --device cuda
    python scripts/run_film_experiments.py --device cuda --quick
"""

import argparse
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
from solver import ContXiongLOBMVSolver


def generate_population(n_agents, mean_q, std_q, s_init=100.0):
    S = np.full(n_agents, s_init)
    q = np.random.normal(mean_q, std_q, n_agents)
    q = np.clip(q, -10, 10)
    return np.stack([S, q], axis=1)


def evaluate_at_q(model, bsde, law_embed, q_val, device):
    """Evaluate Z_q at a specific inventory q given a law embedding."""
    with torch.no_grad():
        agent = torch.tensor([[100.0, q_val]], dtype=torch.float64, device=device)
        leb = law_embed.unsqueeze(0)
        si = torch.cat([agent, leb], dim=1)
        z = model.subnet[0](si) / bsde.dim
    return z[:, 1].item()


def train_model(subnet_type, n_iters, device, h_only=False):
    """Train an MV model with given subnet type."""
    config = Config.from_json("configs/lob_d2_mv.json")
    config.eqn.law_encoder_type = "moments"
    config.eqn.subnet_type = subnet_type
    config.eqn.phi = 0.1
    config.net.opt_config1.num_iterations = n_iters
    config.net.logging_frequency = n_iters
    config.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_mv"](config.eqn)
    solver = ContXiongLOBMVSolver(config, bsde, device=device)
    if h_only:
        solver.model.h_only_mode = True
    result = solver.train()
    return result, solver.model, bsde


def evaluate_sensitivity(model, bsde, stds, device):
    """Evaluate h and Z_q at q=0 for multiple population stds."""
    results = {}
    for std_q in stds:
        pop = generate_population(256, 0.0, std_q)
        particles = torch.tensor(pop, dtype=torch.float64, device=device)
        law_embed = model.law_encoder.encode(particles)
        with torch.no_grad():
            h = bsde.compute_competitive_factor(law_embed).item()
        z_q = evaluate_at_q(model, bsde, law_embed, 0.0, device)
        results[f"std={std_q}"] = {"h": h, "z_q": z_q}
    return results


def interaction_test(model, bsde, device):
    """Evaluate Z_q on (q, std) grid and compute cross-partial derivative.

    For additive architecture: Z(q, law) = f(q) + g(law), so d2Z/(dq * d_std) = 0.
    For FiLM: Z(q, law) = gamma(law) * f(q) + beta(law), cross-partial can be nonzero.
    """
    q_values = np.linspace(-4, 4, 17)
    std_values = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]

    # Build Z_q grid: [len(q_values), len(std_values)]
    grid = np.zeros((len(q_values), len(std_values)))

    for j, std_q in enumerate(std_values):
        pop = generate_population(256, 0.0, std_q)
        particles = torch.tensor(pop, dtype=torch.float64, device=device)
        law_embed = model.law_encoder.encode(particles)
        for i, q in enumerate(q_values):
            grid[i, j] = evaluate_at_q(model, bsde, law_embed, q, device)

    # Compute cross-partial via finite differences: d2Z/(dq * d_std)
    # dZ/d_std at each q (across columns)
    # Then d/dq of that (across rows)
    dZ_dstd = np.diff(grid, axis=1)  # [n_q, n_std-1]
    d2Z_dq_dstd = np.diff(dZ_dstd, axis=0)  # [n_q-1, n_std-1]

    interaction_strength = float(np.mean(np.abs(d2Z_dq_dstd)))
    max_interaction = float(np.max(np.abs(d2Z_dq_dstd)))

    return {
        "grid": grid.tolist(),
        "q_values": q_values.tolist(),
        "std_values": std_values,
        "interaction_strength": interaction_strength,
        "max_interaction": max_interaction,
        "cross_partial": d2Z_dq_dstd.tolist(),
    }


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    n_iters = 500 if args.quick else 3000
    print(f"Device: {device}, iterations: {n_iters}")

    os.makedirs("results_film", exist_ok=True)
    all_results = {}
    stds = [0.1, 1.0, 3.0]
    start = time.time()

    # ================================================================
    # 1. Architecture ablation
    # ================================================================
    print(f"\n{'='*60}")
    print("1. Architecture ablation")
    print(f"{'='*60}")

    for subnet_type in ["two_stream", "film", "film_additive"]:
        print(f"\n--- {subnet_type} ---")
        torch.manual_seed(42); np.random.seed(42)
        result, model, bsde = train_model(subnet_type, n_iters, device)
        sens = evaluate_sensitivity(model, bsde, stds, device)

        print(f"  Y0={result['y0']:.4f}, loss={result['final_loss']:.4e}")
        for k, v in sens.items():
            print(f"  {k}: h={v['h']:.4f}, z_q={v['z_q']:.6f}")

        all_results[f"ablation_{subnet_type}"] = {
            "y0": result["y0"], "loss": result["final_loss"],
            "sensitivity": sens,
        }

    # ================================================================
    # 2. Interaction test (THE critical experiment)
    # ================================================================
    print(f"\n{'='*60}")
    print("2. Interaction test")
    print(f"{'='*60}")

    for subnet_type in ["two_stream", "film"]:
        print(f"\n--- {subnet_type} ---")
        torch.manual_seed(42); np.random.seed(42)
        _, model, bsde = train_model(subnet_type, n_iters, device)
        interaction = interaction_test(model, bsde, device)
        print(f"  Interaction strength: {interaction['interaction_strength']:.6f}")
        print(f"  Max interaction:      {interaction['max_interaction']:.6f}")
        all_results[f"interaction_{subnet_type}"] = interaction

    # Compare
    ts_int = all_results["interaction_two_stream"]["interaction_strength"]
    film_int = all_results["interaction_film"]["interaction_strength"]
    print(f"\n  Two-stream interaction: {ts_int:.6f}")
    print(f"  FiLM interaction:       {film_int:.6f}")
    print(f"  Ratio (FiLM/two_stream): {film_int/max(ts_int, 1e-10):.2f}x")

    # ================================================================
    # 3. H-only control
    # ================================================================
    print(f"\n{'='*60}")
    print("3. H-only control (expect Z_q=constant for all)")
    print(f"{'='*60}")

    for subnet_type in ["two_stream", "film"]:
        print(f"\n--- {subnet_type} h-only ---")
        torch.manual_seed(42); np.random.seed(42)
        result, model, bsde = train_model(subnet_type, n_iters, device, h_only=True)
        model.eval()
        law_dim = bsde.law_embed_dim

        honly_evals = {}
        for std_q in stds:
            pop = generate_population(256, 0.0, std_q)
            particles = torch.tensor(pop, dtype=torch.float64, device=device)
            law_embed = model.law_encoder.encode(particles)
            with torch.no_grad():
                h = bsde.compute_competitive_factor(law_embed).item()
                agent = torch.tensor([[100.0, 0.0]], dtype=torch.float64, device=device)
                zero_law = torch.zeros(1, law_dim, dtype=torch.float64, device=device)
                si = torch.cat([agent, zero_law], dim=1)
                z = model.subnet[0](si) / bsde.dim
            honly_evals[f"std={std_q}"] = {"h": h, "z_q": z[:, 1].item()}
            print(f"  std={std_q}: h={h:.4f}, z_q={z[:, 1].item():.6f}")

        all_results[f"honly_{subnet_type}"] = {
            "y0": result["y0"], "evaluations": honly_evals,
        }

    # ================================================================
    # Save
    # ================================================================
    elapsed = time.time() - start
    all_results["metadata"] = {
        "n_iters": n_iters, "elapsed_seconds": elapsed,
        "device": str(device), "quick": args.quick,
    }
    out = "results_film/film_experiments.json"
    with open(out, "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min. Saved: {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

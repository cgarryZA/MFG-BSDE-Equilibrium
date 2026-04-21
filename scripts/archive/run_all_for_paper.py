#!/usr/bin/env python
"""
Master experiment script: re-runs ALL paper experiments with h_floor=0.01.

Experiments:
1. Main sensitivity (h + quotes at 3 population shapes)
2. Encoder ablation (moments, quantiles, histogram, deepsets)
3. Multi-seed (5 seeds × moments encoder)
4. Placebo test (real vs shuffled vs random embeddings)
5. Disentanglement (cross-feed h vs subnet embeddings)
6. H-only control (law enters generator only, not subnet)
7. Generalisation (uniform, bimodal, skewed, heavy-tailed)
8. Model hierarchy (no-coupling vs MV-moments vs MV-adverse)

Usage:
    python scripts/run_all_for_paper.py --device cuda
    python scripts/run_all_for_paper.py --device cuda --quick  # 500 iters for testing
"""

import argparse
import json
import os
import sys
import time
import copy
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


def generate_special_population(n_agents, family, s_init=100.0):
    """Generate non-Gaussian populations for generalisation test."""
    S = np.full(n_agents, s_init)
    if family == "uniform":
        q = np.random.uniform(-1, 1, n_agents)
    elif family == "bimodal":
        q = np.concatenate([
            np.random.normal(-2, 0.3, n_agents // 2),
            np.random.normal(2, 0.3, n_agents - n_agents // 2),
        ])
    elif family == "skewed":
        q = np.random.chisquare(2, n_agents) - 2  # centered approx
    elif family == "heavy_tail":
        q = np.random.standard_t(3, n_agents)
    else:
        raise ValueError(f"Unknown family: {family}")
    q = np.clip(q, -10, 10)
    return np.stack([S, q], axis=1)


def evaluate_at_population(model, bsde, particles_np, q_eval=0.0, device="cpu"):
    """Evaluate h, Z, quotes for a given population at a given q."""
    model.eval()
    device = next(model.parameters()).device
    alpha = bsde.alpha

    particles = torch.tensor(particles_np, dtype=torch.float64, device=device)
    law_embed = model.law_encoder.encode(particles)

    with torch.no_grad():
        h = bsde.compute_competitive_factor(law_embed).item()

        agent_state = torch.tensor([[100.0, q_eval]], dtype=torch.float64, device=device)
        law_embed_batch = law_embed.unsqueeze(0)
        subnet_input = torch.cat([agent_state, law_embed_batch], dim=1)
        z = model.subnet[0](subnet_input) / bsde.dim
        z_q = z[:, 1:2]

        sigma_q = bsde._sigma_q_equilibrium()
        p = z_q / sigma_q
        base_spread = 1.0 / alpha
        delta_a = (base_spread + p).item()
        delta_b = (base_spread - p).item()

        # Execution rates
        eps_clip = 0.01
        da_clip = max(eps_clip, min(delta_a, 10.0 / alpha))
        db_clip = max(eps_clip, min(delta_b, 10.0 / alpha))
        nu_a = np.exp(-alpha * da_clip) * bsde.lambda_a * h
        nu_b = np.exp(-alpha * db_clip) * bsde.lambda_b * h

    return {
        "h": h, "z_q": z[:, 1].item(), "z_s": z[:, 0].item(),
        "delta_a": delta_a, "delta_b": delta_b,
        "delta_a_clip": da_clip, "delta_b_clip": db_clip,
        "nu_a": nu_a, "nu_b": nu_b,
        "law_embed": law_embed.detach().cpu().numpy().tolist(),
    }


def train_mv(config_path, overrides, device, label=""):
    config = Config.from_json(config_path)
    for key, val in overrides.items():
        parts = key.split(".")
        obj = config
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)
    config.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob_mv"](config.eqn)
    solver = ContXiongLOBMVSolver(config, bsde, device=device)
    result = solver.train()
    print(f"  {label}: Y0={result['y0']:.4f}, loss={result['final_loss']:.4e}")
    return result, solver.model, bsde, config


def train_base(config_path, overrides, device, label=""):
    config = Config.from_json(config_path)
    for key, val in overrides.items():
        parts = key.split(".")
        obj = config
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)
    config.net.verbose = False
    bsde = EQUATION_REGISTRY["contxiong_lob"](config.eqn)
    solver = ContXiongLOBSolver(config, bsde, device=device)
    result = solver.train()
    print(f"  {label}: Y0={result['y0']:.4f}, loss={result['final_loss']:.4e}")
    return result


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
    parser.add_argument("--out_dir", default="results_paper_final")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    print(f"Device: {device}")
    n_iters = 500 if args.quick else 3000
    n_iters_multi = 500 if args.quick else 2000
    os.makedirs(args.out_dir, exist_ok=True)
    base_mv = "configs/lob_d2_mv.json"
    base_lob = "configs/lob_d2.json"
    all_results = {}
    start = time.time()
    stds_main = [0.1, 1.0, 3.0]
    stds_full = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]

    # ================================================================
    # 1. MAIN SENSITIVITY (MomentEncoder)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"1. Main sensitivity (MomentEncoder, {n_iters} iter)")
    print(f"{'='*60}")
    torch.manual_seed(42); np.random.seed(42)
    result, model, bsde, config = train_mv(base_mv, {
        "eqn.law_encoder_type": "moments",
        "net.opt_config1.num_iterations": n_iters,
        "net.logging_frequency": n_iters,
    }, device, label="MomentEncoder")

    main_evals = {}
    for std_q in stds_full:
        pop = generate_population(256, 0.0, std_q)
        r = evaluate_at_population(model, bsde, pop, q_eval=0.0, device=device)
        main_evals[f"std={std_q}"] = r
        print(f"  std={std_q:.1f}: h={r['h']:.4f}, da={r['delta_a']:.4f}, db={r['delta_b']:.4f}")

    all_results["main_sensitivity"] = {
        "y0": result["y0"], "loss": result["final_loss"],
        "evaluations": main_evals,
    }

    # Save model weights for reuse
    torch.save({
        "model_state": model.state_dict(),
        "law_encoder_state": model.law_encoder.state_dict() if hasattr(model.law_encoder, 'state_dict') else None,
        "competitive_factor_state": bsde.competitive_factor_net.state_dict(),
    }, os.path.join(args.out_dir, "main_model.pt"))

    # ================================================================
    # 2. ENCODER ABLATION
    # ================================================================
    print(f"\n{'='*60}")
    print(f"2. Encoder ablation ({n_iters} iter each)")
    print(f"{'='*60}")
    encoder_results = {}
    for enc in ["moments", "quantiles", "histogram", "deepsets"]:
        torch.manual_seed(42); np.random.seed(42)
        r, m, b, c = train_mv(base_mv, {
            "eqn.law_encoder_type": enc,
            "net.opt_config1.num_iterations": n_iters,
            "net.logging_frequency": n_iters,
        }, device, label=enc)
        enc_evals = {}
        for std_q in stds_main:
            pop = generate_population(256, 0.0, std_q)
            ev = evaluate_at_population(m, b, pop, q_eval=0.0, device=device)
            enc_evals[f"std={std_q}"] = ev
            print(f"    std={std_q}: h={ev['h']:.4f}")
        encoder_results[enc] = {
            "y0": r["y0"], "loss": r["final_loss"],
            "evaluations": enc_evals,
        }
    all_results["encoder_ablation"] = encoder_results

    # ================================================================
    # 3. MULTI-SEED (5 seeds, moments)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"3. Multi-seed (5 seeds, {n_iters_multi} iter)")
    print(f"{'='*60}")
    seed_results = {"y0s": [], "losses": [], "h_narrow": [], "h_wide": [], "h_gaps": []}
    for seed in range(5):
        torch.manual_seed(seed); np.random.seed(seed)
        r, m, b, c = train_mv(base_mv, {
            "eqn.law_encoder_type": "moments",
            "net.opt_config1.num_iterations": n_iters_multi,
            "net.logging_frequency": n_iters_multi,
        }, device, label=f"seed={seed}")
        pop_n = generate_population(256, 0.0, 0.1)
        pop_w = generate_population(256, 0.0, 3.0)
        h_n = evaluate_at_population(m, b, pop_n, device=device)["h"]
        h_w = evaluate_at_population(m, b, pop_w, device=device)["h"]
        seed_results["y0s"].append(r["y0"])
        seed_results["losses"].append(r["final_loss"])
        seed_results["h_narrow"].append(h_n)
        seed_results["h_wide"].append(h_w)
        seed_results["h_gaps"].append(h_n - h_w)
        print(f"    h_narrow={h_n:.4f}, h_wide={h_w:.4f}, gap={h_n-h_w:.4f}")

    seed_results["y0_mean"] = float(np.mean(seed_results["y0s"]))
    seed_results["y0_std"] = float(np.std(seed_results["y0s"]))
    seed_results["h_gap_mean"] = float(np.mean(seed_results["h_gaps"]))
    seed_results["h_gap_std"] = float(np.std(seed_results["h_gaps"]))
    print(f"  Y0: {seed_results['y0_mean']:.4f} +/- {seed_results['y0_std']:.4f}")
    print(f"  h-gap: {seed_results['h_gap_mean']:.4f} +/- {seed_results['h_gap_std']:.4f}")
    all_results["multi_seed"] = seed_results

    # ================================================================
    # 4. NO-COUPLING BASELINE
    # ================================================================
    print(f"\n{'='*60}")
    print(f"4. No-coupling baseline ({n_iters} iter)")
    print(f"{'='*60}")
    torch.manual_seed(42); np.random.seed(42)
    r_nc = train_base(base_lob, {
        "eqn.type": 1,
        "net.opt_config1.num_iterations": n_iters,
        "net.opt_config1.freq_update_drift": 9999999,
        "net.logging_frequency": n_iters,
    }, device, label="no-coupling")
    all_results["no_coupling"] = {"y0": r_nc["y0"], "loss": r_nc["final_loss"]}

    # ================================================================
    # 5. PLACEBO TEST (using main model)
    # ================================================================
    print(f"\n{'='*60}")
    print("5. Placebo test")
    print(f"{'='*60}")
    # Reload main model
    ckpt = torch.load(os.path.join(args.out_dir, "main_model.pt"), weights_only=False, map_location=device)
    # Re-train a fresh model for placebo (need the full model object)
    torch.manual_seed(42); np.random.seed(42)
    _, pm, pb, pc = train_mv(base_mv, {
        "eqn.law_encoder_type": "moments",
        "net.opt_config1.num_iterations": n_iters,
        "net.logging_frequency": n_iters,
    }, device, label="placebo-base")

    placebo_results = {}
    for std_q in stds_main:
        pop = generate_population(256, 0.0, std_q)
        # Real embedding
        real = evaluate_at_population(pm, pb, pop, device=device)

        # Shuffled: randomly permute inventory assignments
        pop_shuf = pop.copy()
        np.random.shuffle(pop_shuf[:, 1])
        shuf = evaluate_at_population(pm, pb, pop_shuf, device=device)

        # Random: completely random particles
        pop_rand = generate_population(256, 0.0, 1.0)  # always std=1
        rand = evaluate_at_population(pm, pb, pop_rand, device=device)

        placebo_results[f"std={std_q}"] = {
            "real_h": real["h"], "shuffled_h": shuf["h"], "random_h": rand["h"],
        }
        print(f"  std={std_q}: real={real['h']:.4f}, shuffled={shuf['h']:.4f}, random={rand['h']:.4f}")
    all_results["placebo"] = placebo_results

    # ================================================================
    # 6. DISENTANGLEMENT (cross-feed experiment)
    # ================================================================
    print(f"\n{'='*60}")
    print("6. Pathway disentanglement")
    print(f"{'='*60}")
    pm.eval()
    pop_narrow = generate_population(256, 0.0, 0.1)
    pop_wide = generate_population(256, 0.0, 3.0)
    particles_n = torch.tensor(pop_narrow, dtype=torch.float64, device=device)
    particles_w = torch.tensor(pop_wide, dtype=torch.float64, device=device)
    embed_n = pm.law_encoder.encode(particles_n)
    embed_w = pm.law_encoder.encode(particles_w)

    disentangle = {}
    configs = [
        ("narrow_both", embed_n, embed_n),  # h=narrow, subnet=narrow
        ("wide_both", embed_w, embed_w),    # h=wide, subnet=wide
        ("h_narrow_sub_wide", embed_n, embed_w),  # h=narrow, subnet=wide
        ("h_wide_sub_narrow", embed_w, embed_n),  # h=wide, subnet=narrow
    ]

    for label, h_embed, sub_embed in configs:
        with torch.no_grad():
            h_val = pb.compute_competitive_factor(h_embed).item()
            agent_state = torch.tensor([[100.0, 0.0]], dtype=torch.float64, device=device)
            sub_batch = sub_embed.unsqueeze(0)
            subnet_input = torch.cat([agent_state, sub_batch], dim=1)
            z = pm.subnet[0](subnet_input) / pb.dim
            z_q = z[:, 1:2]
            sigma_q = pb._sigma_q_equilibrium()
            p = z_q / sigma_q
            delta_a = (1.0 / pb.alpha + p).item()
            delta_b = (1.0 / pb.alpha - p).item()

        disentangle[label] = {"h": h_val, "delta_a": delta_a, "delta_b": delta_b, "z_q": z[:, 1].item()}
        print(f"  {label}: h={h_val:.4f}, da={delta_a:.4f}, db={delta_b:.4f}")
    all_results["disentanglement"] = disentangle

    # ================================================================
    # 7. H-ONLY CONTROL (law enters generator only)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"7. H-only control ({n_iters} iter)")
    print(f"{'='*60}")
    # Train a model where subnet gets ZEROS for law embedding
    torch.manual_seed(42); np.random.seed(42)
    config_honly = Config.from_json(base_mv)
    config_honly.eqn.law_encoder_type = "moments"
    config_honly.net.opt_config1.num_iterations = n_iters
    config_honly.net.logging_frequency = n_iters
    config_honly.net.verbose = False
    bsde_honly = EQUATION_REGISTRY["contxiong_lob_mv"](config_honly.eqn)
    solver_honly = ContXiongLOBMVSolver(config_honly, bsde_honly, device=device)

    # Monkey-patch the model's forward to zero out law in subnet input
    original_forward = solver_honly.model.forward

    def h_only_forward(inputs):
        dw, x, mean_y_input = inputs
        dw = torch.as_tensor(dw, dtype=torch.float64, device=device)
        x = torch.as_tensor(x, dtype=torch.float64, device=device)
        loss_inter = torch.tensor(0.0, dtype=torch.float64, device=device)
        mean_y = []
        time_stamp = np.arange(0, config_honly.eqn.num_time_interval) * bsde_honly.delta_t
        batch_size = dw.shape[0]
        all_one = torch.ones(batch_size, 1, dtype=torch.float64, device=device)
        y = all_one * solver_honly.model.y_init
        z = all_one @ solver_honly.model.z_init
        mean_y.append(torch.mean(y))

        law_dim = bsde_honly.law_embed_dim

        for t in range(bsde_honly.num_time_interval - 1):
            # Compute law embedding for generator (h)
            particles_t = x[:, :, t]
            law_embed = solver_honly.model.law_encoder.encode(particles_t)
            if hasattr(bsde_honly, 'set_current_law_embed'):
                bsde_honly.set_current_law_embed(law_embed)

            # BSDE step (generator uses h from law_embed)
            y = (y - bsde_honly.delta_t * bsde_honly.f_tf(time_stamp[t], x[:, :, t], y, z)
                 + torch.sum(z * dw[:, :, t], dim=1, keepdim=True))
            mean_y.append(torch.mean(y))

            # Subnet: ZEROS for law embedding
            own_state = x[:, :, t + 1]
            zero_law = torch.zeros(batch_size, law_dim, dtype=torch.float64, device=device)
            subnet_input = torch.cat([own_state, zero_law], dim=1)
            z = solver_honly.model.subnet[t](subnet_input) / bsde_honly.dim

        # Terminal
        particles_terminal = x[:, :, -2]
        law_embed_terminal = solver_honly.model.law_encoder.encode(particles_terminal)
        if hasattr(bsde_honly, 'set_current_law_embed'):
            bsde_honly.set_current_law_embed(law_embed_terminal)
        y = (y - bsde_honly.delta_t * bsde_honly.f_tf(time_stamp[-1], x[:, :, -2], y, z)
             + torch.sum(z * dw[:, :, -1], dim=1, keepdim=True))
        mean_y.append(torch.mean(y))

        # Loss
        delta = y - bsde_honly.g_tf(bsde_honly.total_time, x[:, :, -1])
        loss = torch.mean(delta ** 2)
        mean_y_arr = torch.stack(mean_y)
        return y, mean_y_arr, loss_inter

    solver_honly.model.forward = h_only_forward
    result_honly = solver_honly.train()
    print(f"  H-only: Y0={result_honly['y0']:.4f}, loss={result_honly['final_loss']:.4e}")

    # Evaluate h-only at different populations
    honly_evals = {}
    solver_honly.model.eval()
    for std_q in stds_main:
        pop = generate_population(256, 0.0, std_q)
        particles = torch.tensor(pop, dtype=torch.float64, device=device)
        law_embed = solver_honly.model.law_encoder.encode(particles)
        with torch.no_grad():
            h_val = bsde_honly.compute_competitive_factor(law_embed).item()
            # Subnet with zeros (as trained)
            agent = torch.tensor([[100.0, 0.0]], dtype=torch.float64, device=device)
            zero_law = torch.zeros(1, law_dim, dtype=torch.float64, device=device)
            subnet_in = torch.cat([agent, zero_law], dim=1)
            z = solver_honly.model.subnet[0](subnet_in) / bsde_honly.dim
        honly_evals[f"std={std_q}"] = {"h": h_val, "z_q": z[:, 1].item()}
        print(f"    std={std_q}: h={h_val:.4f}, z_q={z[:, 1].item():.6f}")
    all_results["h_only"] = {
        "y0": result_honly["y0"], "loss": result_honly["final_loss"],
        "evaluations": honly_evals,
    }

    # ================================================================
    # 8. GENERALISATION (non-Gaussian populations)
    # ================================================================
    print(f"\n{'='*60}")
    print("8. Generalisation across population families")
    print(f"{'='*60}")
    # Use the main model (already trained)
    torch.manual_seed(42); np.random.seed(42)
    _, gm, gb, gc = train_mv(base_mv, {
        "eqn.law_encoder_type": "moments",
        "net.opt_config1.num_iterations": n_iters,
        "net.logging_frequency": n_iters,
    }, device, label="generalisation-base")

    gen_results = {}
    for family in ["uniform", "bimodal", "skewed", "heavy_tail"]:
        pop = generate_special_population(256, family)
        ev = evaluate_at_population(gm, gb, pop, device=device)
        gen_results[family] = {"h": ev["h"], "delta_a": ev["delta_a"], "delta_b": ev["delta_b"]}
        print(f"  {family}: h={ev['h']:.4f}, da={ev['delta_a']:.4f}")
    all_results["generalisation"] = gen_results

    # ================================================================
    # 9. PENALTY REGIME SWEEP
    # ================================================================
    print(f"\n{'='*60}")
    print(f"9. Penalty regime sweep")
    print(f"{'='*60}")
    penalty_results = {}
    for phi in [0.01, 0.05, 0.1, 0.5]:
        torch.manual_seed(42); np.random.seed(42)
        r, m, b, c = train_mv(base_mv, {
            "eqn.law_encoder_type": "moments",
            "eqn.phi": phi,
            "net.opt_config1.num_iterations": n_iters,
            "net.logging_frequency": n_iters,
        }, device, label=f"phi={phi}")
        pop_n = generate_population(256, 0.0, 0.1)
        pop_w = generate_population(256, 0.0, 3.0)
        ev_n = evaluate_at_population(m, b, pop_n, device=device)
        ev_w = evaluate_at_population(m, b, pop_w, device=device)
        penalty_results[f"phi={phi}"] = {
            "y0": r["y0"], "h_narrow": ev_n["h"], "h_wide": ev_w["h"],
            "h_gap": ev_n["h"] - ev_w["h"],
            "da_narrow": ev_n["delta_a"], "da_wide": ev_w["delta_a"],
            "policy_gap": abs(ev_n["delta_a"] - ev_w["delta_a"]),
        }
        print(f"    h_gap={ev_n['h']-ev_w['h']:.4f}, policy_gap={abs(ev_n['delta_a']-ev_w['delta_a']):.4f}")
    all_results["penalty_sweep"] = penalty_results

    # ================================================================
    # SAVE ALL RESULTS
    # ================================================================
    elapsed = time.time() - start
    all_results["metadata"] = {
        "device": str(device), "n_iters": n_iters, "n_iters_multi": n_iters_multi,
        "h_floor": 0.01, "elapsed_seconds": elapsed, "quick": args.quick,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

    out_path = os.path.join(args.out_dir, "all_results.json")
    with open(out_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed/60:.1f} min")
    print(f"Saved: {out_path}")
    print(f"{'='*60}")

    # Print summary for paper
    print("\n=== PAPER SUMMARY ===")
    ms = all_results["main_sensitivity"]["evaluations"]
    print(f"\nTable 1 (main): h varies {ms['std=0.1']['h']:.3f} → {ms['std=3.0']['h']:.3f} "
          f"({ms['std=0.1']['h']/max(ms['std=3.0']['h'],1e-6):.1f}x)")
    print(f"  No-coupling Y0={all_results['no_coupling']['y0']:.4f}")
    print(f"  MV Y0={all_results['main_sensitivity']['y0']:.4f}")

    ms2 = all_results["multi_seed"]
    print(f"\nMulti-seed: Y0={ms2['y0_mean']:.4f}±{ms2['y0_std']:.4f}, "
          f"h-gap={ms2['h_gap_mean']:.4f}±{ms2['h_gap_std']:.4f}")

    ho = all_results["h_only"]
    print(f"\nH-only: Y0={ho['y0']:.4f}")
    for k, v in ho["evaluations"].items():
        print(f"  {k}: h={v['h']:.4f}, z_q={v['z_q']:.6f}")


if __name__ == "__main__":
    main()

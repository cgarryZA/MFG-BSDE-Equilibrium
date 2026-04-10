#!/usr/bin/env python
"""
Overnight run on the corrected CX model.

1. Validated scale test N=1,2,5,10 (exact + neural, with 1/N fix)
2. Continuous inventory solver (N=2, compare to discrete ground truth)
3. N=2 full validation (exact vs neural single-pass vs neural FP)
4. Tacit collusion: full-info vs no-info training

~3-4 hours total.
"""

import gc, json, os, sys, time, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equations.contxiong_exact import ContXiongExact
from solver_cx import CXSolver, CXFictitiousPlay
from solver_cx_continuous import CXContinuousSolver
from scripts.cont_xiong_exact import fictitious_play as exact_fp

OUT = "results_cx_overnight"
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "log.txt")
with open(LOG, "w") as f: f.write("")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def convert(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert(v) for v in obj]
    return obj

def save(data, name):
    p = os.path.join(OUT, name)
    with open(p, "w") as f: json.dump(convert(data), f, indent=2)
    log(f"  Saved {p}")

def run_safe(name, func):
    log(f"\n{'='*60}")
    log(name)
    log("=" * 60)
    gpu_reset()
    t0 = time.time()
    try:
        r = func()
        log(f"  Done in {(time.time()-t0)/60:.0f} min")
        return r
    except Exception:
        log(f"  CRASHED after {(time.time()-t0)/60:.0f} min")
        log(traceback.format_exc())
        gpu_reset()
        return None

class CXConfig:
    lambda_a = 2.0; lambda_b = 2.0; discount_rate = 0.01
    Delta_q = 1.0; q_max = 5.0; phi = 0.005; N_agents = 2


# ================================================================
# EXPERIMENTS
# ================================================================

def exp_scale():
    """Scale test with corrected 1/N factor."""
    results = {}
    for N in [1, 2, 5, 10]:
        log(f"\n  N={N}:")
        # Exact
        ex = exact_fp(N=N, max_iter=50)
        mid = len(ex["q_grid"]) // 2
        ex_s = ex["delta_a"][mid] + ex["delta_b"][mid]
        log(f"    Exact: spread(0)={ex_s:.4f}, V(0)={ex['V'][mid]:.4f}")

        # Neural
        gpu_reset()
        cfg = CXConfig(); cfg.N_agents = N
        eqn = ContXiongExact(cfg)
        solver = CXSolver(eqn, device=device, lr=1e-3, n_iter=5000, verbose=False)
        r = solver.train()
        nn_s = r["delta_a"][mid] + r["delta_b"][mid]
        err = abs(nn_s - ex_s)
        log(f"    Neural: spread(0)={nn_s:.4f}, V(0)={r['V'][mid]:.4f}, err={err:.4f}")

        results[f"N={N}"] = {
            "N": N, "exact_spread": ex_s, "exact_V": ex["V"][mid],
            "nn_spread": nn_s, "nn_V": r["V"][mid], "error": err,
            "exact_da": ex["delta_a"], "exact_db": ex["delta_b"],
            "nn_da": r["delta_a"], "nn_db": r["delta_b"],
        }
        save(results, "scale_test.json")
    return results

def exp_continuous():
    """Continuous inventory solver."""
    log("  Training continuous solver (N=2, 15000 iter)...")
    solver = CXContinuousSolver(N=2, device=device, n_iter=15000, batch_size=64, lr=5e-4)
    r = solver.train()

    # Compare to exact at grid points
    exact = exact_fp(N=2, max_iter=50)
    mid = len(exact["q_grid"]) // 2
    log(f"\n  Comparison at grid points:")
    log(f"  {'q':>4} {'Ex_spr':>7} {'NN_spr':>7} {'err':>7}")
    max_err = 0
    for j, q in enumerate(exact["q_grid"]):
        ex_s = exact["delta_a"][j] + exact["delta_b"][j]
        nn_s = r["delta_a"][j] + r["delta_b"][j]
        err = abs(nn_s - ex_s)
        max_err = max(max_err, err)
        log(f"  {q:4.0f} {ex_s:7.4f} {nn_s:7.4f} {err:7.4f}")
    log(f"  Max error: {max_err:.4f}")

    r["exact_delta_a"] = exact["delta_a"]
    r["exact_delta_b"] = exact["delta_b"]
    save(r, "continuous_result.json")
    return r

def exp_validation():
    """Full N=2 validation: exact vs single-pass vs FP."""
    exact = exact_fp(N=2, max_iter=50)
    mid = len(exact["q_grid"]) // 2

    # Single-pass
    gpu_reset()
    cfg = CXConfig()
    eqn = ContXiongExact(cfg)
    solver = CXSolver(eqn, device=device, lr=1e-3, n_iter=8000, verbose=False)
    r_single = solver.train()

    # FP
    gpu_reset()
    eqn2 = ContXiongExact(cfg)
    fp = CXFictitiousPlay(eqn2, device=device, outer_iter=15, inner_iter=3000,
                           lr=1e-3, damping=0.5)
    r_fp = fp.train()

    ex_s = exact["delta_a"][mid] + exact["delta_b"][mid]
    nn_s = r_single["delta_a"][mid] + r_single["delta_b"][mid]
    fp_s = r_fp["final_delta_a"][mid] + r_fp["final_delta_b"][mid]

    log(f"  Exact:       spread(0)={ex_s:.4f}")
    log(f"  Single-pass: spread(0)={nn_s:.4f} (err={abs(nn_s-ex_s):.4f})")
    log(f"  Neural FP:   spread(0)={fp_s:.4f} (err={abs(fp_s-ex_s):.4f})")

    save({"exact_spread": ex_s, "single_spread": nn_s, "fp_spread": fp_s,
          "exact_da": exact["delta_a"], "exact_db": exact["delta_b"],
          "single_da": r_single["delta_a"], "fp_da": r_fp["final_delta_a"]},
         "validation.json")
    return {"exact": ex_s, "single": nn_s, "fp": fp_s}

def exp_collusion():
    """Tacit collusion: full-info vs no-info.

    Full-info: agent knows population average quotes (normal training).
    No-info: agent assumes a FIXED competitor level (monopolist assumption).
    The difference in equilibrium spreads measures tacit collusion.

    CX Section 6 shows RL agents converge above Nash. We test whether
    our BSDE solver does the same under limited information.
    """
    cfg = CXConfig()

    # Full-info: normal training (knows competitors' quotes)
    log("  Full-info solver (knows population)...")
    gpu_reset()
    eqn1 = ContXiongExact(cfg)
    solver1 = CXSolver(eqn1, device=device, lr=1e-3, n_iter=5000, verbose=False)
    r_full = solver1.train()
    mid = eqn1.mid
    full_spread = r_full["delta_a"][mid] + r_full["delta_b"][mid]
    log(f"    spread(0)={full_spread:.4f}")

    # No-info: agent doesn't know competitors' quotes
    # Assumes it's a monopolist (avg_competitor_quote = own quote)
    # This is "self-play" — the agent optimises against itself
    log("  No-info solver (assumes monopolist)...")
    gpu_reset()
    cfg_mono = CXConfig(); cfg_mono.N_agents = 1  # thinks it's alone
    eqn_mono = ContXiongExact(cfg_mono)
    solver_mono = CXSolver(eqn_mono, device=device, lr=1e-3, n_iter=5000, verbose=False)
    r_mono = solver_mono.train()
    mono_spread = r_mono["delta_a"][eqn_mono.mid] + r_mono["delta_b"][eqn_mono.mid]
    log(f"    spread(0)={mono_spread:.4f}")

    # Partial-info: knows N=2 but uses STALE population estimate
    # (doesn't update avg_da during training — uses initial guess)
    log("  Partial-info solver (stale population estimate)...")
    gpu_reset()
    eqn_partial = ContXiongExact(cfg)
    solver_partial = CXSolver(eqn_partial, device=device, lr=1e-3, n_iter=5000,
                               fixed_avg_da=0.9, fixed_avg_db=0.9, verbose=False)
    r_partial = solver_partial.train()
    partial_spread = r_partial["delta_a"][mid] + r_partial["delta_b"][mid]
    log(f"    spread(0)={partial_spread:.4f}")

    # Ground truth Nash
    exact = exact_fp(N=2, max_iter=50)
    nash_spread = exact["delta_a"][mid] + exact["delta_b"][mid]

    log(f"\n  Collusion comparison:")
    log(f"    Nash equilibrium:  {nash_spread:.4f}")
    log(f"    Full-info solver:  {full_spread:.4f}")
    log(f"    Partial-info:      {partial_spread:.4f}")
    log(f"    No-info (monopolist): {mono_spread:.4f}")
    log(f"    Collusion gap (partial - Nash): {partial_spread - nash_spread:.4f}")

    save({"nash": nash_spread, "full_info": full_spread,
          "partial_info": partial_spread, "no_info_mono": mono_spread,
          "collusion_gap": partial_spread - nash_spread},
         "collusion.json")
    return {"nash": nash_spread, "full": full_spread, "partial": partial_spread}


# ================================================================
def main():
    global device
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()

    r1 = run_safe("1. Scale test (N=1,2,5,10) with 1/N fix", exp_scale)
    r2 = run_safe("2. Continuous inventory solver (N=2)", exp_continuous)
    r3 = run_safe("3. Full N=2 validation (exact vs single vs FP)", exp_validation)
    r4 = run_safe("4. Tacit collusion detection", exp_collusion)

    elapsed = (time.time() - start) / 60
    log(f"\n{'='*60}")
    log(f"ALL DONE in {elapsed:.0f} min")
    log(f"{'='*60}")
    names = ["scale", "continuous", "validation", "collusion"]
    results = [r1, r2, r3, r4]
    for n, r in zip(names, results):
        log(f"  {n}: {'OK' if r else 'CRASHED'}")
    save({"elapsed_min": elapsed}, "summary.json")

if __name__ == "__main__":
    main()

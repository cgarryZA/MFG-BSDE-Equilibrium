#!/usr/bin/env python
"""
Overnight CX suite. Run after the MADDPG finishes (or independently).

1. Q=20 and Q=50 neural solver with bigger network (fix the 8% error)
2. Continuous inventory solver with corrected 1/N code
3. Multiple MADDPG rounds for statistics
4. Regenerate all figures with final data
"""

import gc, json, os, sys, time, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "results_cx_overnight2"
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
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    time.sleep(2)

def convert(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
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


def exp_q_scaling():
    """Q=20 and Q=50 with larger network and more iterations."""
    from equations.contxiong_exact import ContXiongExact
    from solver_cx import CXSolver
    from scripts.cont_xiong_exact import fictitious_play

    results = {}
    for Q in [20, 50]:
        log(f"  Q={Q}...")
        ex = fictitious_play(N=2, Q=Q, max_iter=100)
        mid = len(ex["q_grid"]) // 2
        ex_s = ex["delta_a"][mid] + ex["delta_b"][mid]
        log(f"    Exact: {ex_s:.4f}")

        gpu_reset()
        class Cfg:
            lambda_a=2.0; lambda_b=2.0; discount_rate=0.01
            Delta_q=1.0; q_max=float(Q); phi=0.005; N_agents=2
        eqn = ContXiongExact(Cfg())
        # More iterations + lower LR for larger Q
        solver = CXSolver(eqn, device=device, lr=3e-4, n_iter=20000, verbose=False)
        r = solver.train()
        nn_s = r["delta_a"][eqn.mid] + r["delta_b"][eqn.mid]
        err = abs(nn_s - ex_s)
        log(f"    Neural: {nn_s:.4f} (err={err:.4f}, {err/ex_s*100:.1f}%)")
        results[f"Q={Q}"] = {"exact": ex_s, "neural": nn_s, "error": err}
        save(results, "q_scaling.json")
    return results


def exp_continuous():
    """Continuous inventory with corrected code."""
    from solver_cx_continuous import CXContinuousSolver
    from scripts.cont_xiong_exact import fictitious_play

    ex = fictitious_play(N=2, Q=5, max_iter=50)
    mid = len(ex["q_grid"]) // 2

    gpu_reset()
    solver = CXContinuousSolver(N=2, device=device, n_iter=20000, batch_size=64, lr=3e-4)
    r = solver.train()

    log(f"  Continuous vs exact at grid points:")
    max_err = 0
    for j, q in enumerate(ex["q_grid"]):
        ex_s = ex["delta_a"][j] + ex["delta_b"][j]
        nn_s = r["delta_a"][j] + r["delta_b"][j]
        err = abs(nn_s - ex_s)
        max_err = max(max_err, err)
    log(f"  Max spread error: {max_err:.4f}")
    save(r, "continuous.json")
    return r


def exp_maddpg_rounds():
    """Multiple MADDPG rounds for statistics."""
    from solver_cx_multiagent import MADDPGTrainer
    from scripts.cont_xiong_exact import fictitious_play

    nash = fictitious_play(N=2, Q=5, max_iter=50)
    mid = len(nash["q_grid"]) // 2
    nash_spread = nash["delta_a"][mid] + nash["delta_b"][mid]

    all_spreads = []
    for round_idx in range(5):
        log(f"  Round {round_idx+1}/5...")
        gpu_reset()
        torch.manual_seed(round_idx); np.random.seed(round_idx)
        trainer = MADDPGTrainer(N=2, Q=5, device=device,
            lr_actor=1e-4, lr_critic=1e-3, tau=0.001,
            n_episodes=500, steps_per_episode=500, batch_size=32)
        r = trainer.train()
        spread = r["avg_final_spread"]
        all_spreads.append(spread)
        log(f"    Spread: {spread:.4f}")
        save({"spreads": all_spreads, "nash": nash_spread,
              "round": round_idx+1}, "maddpg_rounds.json")

    mean_s = np.mean(all_spreads)
    log(f"  Mean spread: {mean_s:.4f} +/- {np.std(all_spreads):.4f}")
    log(f"  Nash: {nash_spread:.4f}")
    log(f"  Above Nash? {'YES' if mean_s > nash_spread else 'NO'}")
    return {"spreads": all_spreads, "mean": mean_s, "nash": nash_spread}


def main():
    global device
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()

    r1 = run_safe("1. Q=20,50 scaling (bigger network)", exp_q_scaling)
    r2 = run_safe("2. Continuous inventory solver", exp_continuous)
    r3 = run_safe("3. MADDPG collusion (5 rounds)", exp_maddpg_rounds)

    elapsed = (time.time() - start) / 60
    log(f"\n{'='*60}")
    log(f"ALL DONE in {elapsed:.0f} min")
    log(f"{'='*60}")
    names = ["q_scaling", "continuous", "maddpg"]
    results = [r1, r2, r3]
    for n, r in zip(names, results):
        log(f"  {n}: {'OK' if r else 'CRASHED'}")
    save({"elapsed_min": elapsed}, "summary.json")


if __name__ == "__main__":
    main()

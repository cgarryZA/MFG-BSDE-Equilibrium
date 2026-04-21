#!/usr/bin/env python
"""
Tonight's run:
1. Q scaling with scaled network (Q=5,10,20,50)
2. 20 MADDPG rounds with episode-averaged spreads
"""

import gc, json, os, sys, time, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "results_tonight"
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
    """Q scaling with scaled network architecture."""
    from equations.contxiong_exact import ContXiongExact
    from solver_cx import CXSolver
    from scripts.cont_xiong_exact import fictitious_play

    results = {}
    for Q in [5, 10, 20, 50]:
        nq = int(2 * Q + 1)
        log(f"\n  Q={Q} ({nq} grid points)...")

        # Exact
        ex = fictitious_play(N=2, Q=Q, max_iter=100)
        mid = len(ex["q_grid"]) // 2
        ex_s = ex["delta_a"][mid] + ex["delta_b"][mid]
        log(f"    Exact: {ex_s:.4f}")

        # Neural — scale iterations with Q
        gpu_reset()
        class Cfg:
            lambda_a = 2.0; lambda_b = 2.0; discount_rate = 0.01
            Delta_q = 1.0; q_max = float(Q); phi = 0.005; N_agents = 2
        eqn = ContXiongExact(Cfg())

        # More iterations for larger Q, lower LR
        n_iter = {5: 5000, 10: 10000, 20: 20000, 50: 40000}[Q]
        lr = {5: 1e-3, 10: 5e-4, 20: 3e-4, 50: 2e-4}[Q]

        t0 = time.time()
        solver = CXSolver(eqn, device=device, lr=lr, n_iter=n_iter, verbose=False)
        r = solver.train()
        nn_t = time.time() - t0

        nn_s = r["delta_a"][eqn.mid] + r["delta_b"][eqn.mid]
        err = abs(nn_s - ex_s)
        log(f"    Neural: {nn_s:.4f} (err={err:.4f}, {err/ex_s*100:.1f}%, {nn_t:.0f}s)")

        results[f"Q={Q}"] = {
            "Q": Q, "nq": nq, "exact": ex_s, "neural": nn_s,
            "error": err, "pct_error": err/ex_s*100, "time": nn_t,
        }
        save(results, "q_scaling.json")
    return results


def exp_maddpg_20rounds():
    """20 MADDPG rounds, report episode-averaged spreads."""
    from solver_cx_multiagent import MADDPGTrainer
    from scripts.cont_xiong_exact import fictitious_play

    nash = fictitious_play(N=2, Q=5, max_iter=50)
    mid = len(nash["q_grid"]) // 2
    nash_spread = nash["delta_a"][mid] + nash["delta_b"][mid]

    all_final_spreads = []
    all_avg_spreads = []  # average of last 100 episodes

    for round_idx in range(20):
        log(f"  Round {round_idx+1}/20...")
        gpu_reset()
        torch.manual_seed(round_idx * 7 + 13)
        np.random.seed(round_idx * 7 + 13)

        trainer = MADDPGTrainer(N=2, Q=5, device=device,
            lr_actor=1e-4, lr_critic=1e-3, tau=0.001,
            n_episodes=500, steps_per_episode=500, batch_size=32)
        r = trainer.train()

        final_s = r["avg_final_spread"]
        # Average spread over last 5 logged episodes (every 50 episodes)
        last_spreads = [h["avg_spread"] for h in r["history"][-5:]]
        avg_s = np.mean(last_spreads) if last_spreads else final_s

        all_final_spreads.append(final_s)
        all_avg_spreads.append(avg_s)
        log(f"    Final: {final_s:.4f}, Avg(last 5 logs): {avg_s:.4f}")

        save({
            "final_spreads": all_final_spreads,
            "avg_spreads": all_avg_spreads,
            "nash": nash_spread,
            "round": round_idx + 1,
            "n_above_nash_final": sum(1 for s in all_final_spreads if s > nash_spread),
            "n_above_nash_avg": sum(1 for s in all_avg_spreads if s > nash_spread),
        }, "maddpg_20rounds.json")

    mean_final = np.mean(all_final_spreads)
    std_final = np.std(all_final_spreads)
    mean_avg = np.mean(all_avg_spreads)
    std_avg = np.std(all_avg_spreads)

    log(f"\n  MADDPG Summary (20 rounds):")
    log(f"    Nash spread:         {nash_spread:.4f}")
    log(f"    Final spread:        {mean_final:.4f} +/- {std_final:.4f}")
    log(f"    Avg(last) spread:    {mean_avg:.4f} +/- {std_avg:.4f}")
    log(f"    Above Nash (final):  {sum(1 for s in all_final_spreads if s > nash_spread)}/20")
    log(f"    Above Nash (avg):    {sum(1 for s in all_avg_spreads if s > nash_spread)}/20")

    # 95% CI
    from scipy import stats
    ci = stats.t.interval(0.95, df=len(all_avg_spreads)-1,
                          loc=mean_avg, scale=stats.sem(all_avg_spreads))
    log(f"    95% CI (avg):        [{ci[0]:.4f}, {ci[1]:.4f}]")
    log(f"    CI above Nash?       {'YES' if ci[0] > nash_spread else 'NO'}")

    return {
        "final_spreads": all_final_spreads,
        "avg_spreads": all_avg_spreads,
        "nash": nash_spread,
        "mean_final": mean_final, "std_final": std_final,
        "mean_avg": mean_avg, "std_avg": std_avg,
        "ci_95": list(ci),
    }


def main():
    global device
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()

    r1 = run_safe("1. Q scaling (scaled network)", exp_q_scaling)
    r2 = run_safe("2. MADDPG collusion (20 rounds)", exp_maddpg_20rounds)

    elapsed = (time.time() - start) / 60
    log(f"\n{'='*60}")
    log(f"ALL DONE in {elapsed:.0f} min ({elapsed/60:.1f} hours)")
    log(f"{'='*60}")
    for n, r in zip(["q_scaling", "maddpg"], [r1, r2]):
        log(f"  {n}: {'OK' if r else 'CRASHED'}")
    save({"elapsed_min": elapsed}, "summary.json")


if __name__ == "__main__":
    main()

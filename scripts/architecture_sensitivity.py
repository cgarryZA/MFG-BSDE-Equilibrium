#!/usr/bin/env python -u
"""Architecture sensitivity sweep for the neural Bellman solver.

Tests: hidden size x layers x activation x learning rate grid.
Patched solver (with boundary fix) so floor error should be machine precision
— this measures which architectures achieve it vs which plateau higher.

CPU-only. ~15-20 min.
"""

import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
from equations.contxiong_exact import ContXiongExact
from solver_cx import CXSolver, ValueNet
from scripts.cont_xiong_exact import fictitious_play

device = torch.device("cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def run_one(hidden, n_layers, activation, lr, n_iter=3000):
    """Train one config with given architecture. Uses boundary-fixed averaging."""
    cx = SimpleNamespace(lambda_a=2.0, lambda_b=2.0, discount_rate=0.01,
                        Delta_q=1.0, q_max=5.0, phi=0.005, N_agents=2)
    eqn = ContXiongExact(cx)

    # Patch: replace CXSolver with boundary-fixed averaging
    solver = CXSolver(eqn, device=device, lr=lr, n_iter=n_iter, verbose=False,
                     early_stopping=True, es_patience=300)

    # Override network for different architecture
    if activation == "tanh":
        act = nn.Tanh
    elif activation == "relu":
        act = nn.ReLU
    elif activation == "gelu":
        act = nn.GELU
    else:
        raise ValueError(activation)

    # Rebuild net
    layers = [nn.Linear(1, hidden, dtype=torch.float64), act()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden, dtype=torch.float64), act()]
    layers.append(nn.Linear(hidden, 1, dtype=torch.float64))
    solver.value_net = nn.Sequential(*layers).to(device)
    solver.optimizer = torch.optim.Adam(solver.value_net.parameters(), lr=lr)

    # Patch train() to use boundary-fixed averaging
    original_train = solver.train
    def patched_train():
        eqn = solver.eqn
        history = []
        avg_da = 0.75; avg_db = 0.75
        for step in range(solver.n_iter):
            V = solver.value_net(solver.q_norm).squeeze(1)
            V_np = V.detach().cpu().numpy()
            da_np, db_np = eqn.compute_optimal_quotes(V_np, avg_da, avg_db)
            # Boundary fix: include zeros in average
            avg_da = float(np.mean(da_np))
            avg_db = float(np.mean(db_np))
            da = torch.tensor(da_np, dtype=torch.float64)
            db = torch.tensor(db_np, dtype=torch.float64)
            res = eqn.bellman_residual(V, da, db, torch.tensor(avg_da), torch.tensor(avg_db))
            loss = torch.sum(res**2)
            solver.optimizer.zero_grad()
            loss.backward()
            solver.optimizer.step()
            if solver.es is not None and solver.es(loss.item()):
                break
        V_np = solver.value_net(solver.q_norm).squeeze(1).detach().cpu().numpy()
        da_np, db_np = eqn.compute_optimal_quotes(V_np, avg_da, avg_db)
        return {"V": V_np.tolist(), "delta_a": da_np.tolist(),
                "delta_b": db_np.tolist(), "avg_da": avg_da, "avg_db": avg_db}

    t0 = time.time()
    r = patched_train()
    elapsed = time.time() - t0
    mid = eqn.mid
    spread = r['delta_a'][mid] + r['delta_b'][mid]
    return spread, elapsed


if __name__ == "__main__":
    # Exact reference
    exact = fictitious_play(N=2, Q=5, Delta=1)
    exact_spread = exact['delta_a'][5] + exact['delta_b'][5]
    print(f"Exact spread: {exact_spread:.6f}", flush=True)

    configs = []
    for hidden in [32, 64, 128]:
        for n_layers in [2, 3]:
            for act in ["tanh", "relu", "gelu"]:
                for lr in [1e-3, 5e-4]:
                    configs.append({"hidden": hidden, "n_layers": n_layers,
                                   "activation": act, "lr": lr})

    print(f"\nRunning {len(configs)} architecture configs...", flush=True)
    print(f"{'hidden':>6s}  {'layers':>6s}  {'act':>5s}  {'lr':>8s}  {'spread':>10s}  {'error':>8s}  {'time':>6s}", flush=True)
    print("-" * 65, flush=True)

    results = []
    for cfg in configs:
        gc.collect()
        try:
            torch.manual_seed(0); np.random.seed(0)
            s, t = run_one(**cfg)
            err = abs(s - exact_spread) / exact_spread * 100
            print(f"{cfg['hidden']:6d}  {cfg['n_layers']:6d}  {cfg['activation']:>5s}  "
                  f"{cfg['lr']:8.0e}  {s:10.6f}  {err:7.4f}%  {t:5.0f}s", flush=True)
            results.append({**cfg, "spread": float(s), "error_pct": float(err),
                          "elapsed": float(t)})
        except Exception as e:
            print(f"  {cfg}: FAILED: {e}", flush=True)
            results.append({**cfg, "error": str(e)})

    # Summary
    valid = [r for r in results if "spread" in r]
    if valid:
        errs = [r["error_pct"] for r in valid]
        best = min(valid, key=lambda r: r["error_pct"])
        worst = max(valid, key=lambda r: r["error_pct"])
        print(f"\n{'='*60}")
        print(f"Best:  {best['hidden']}h, {best['n_layers']}L, {best['activation']}, "
              f"lr={best['lr']:.0e} -> {best['error_pct']:.4f}%")
        print(f"Worst: {worst['hidden']}h, {worst['n_layers']}L, {worst['activation']}, "
              f"lr={worst['lr']:.0e} -> {worst['error_pct']:.4f}%")
        print(f"Mean error: {np.mean(errs):.4f}%, Range: [{min(errs):.4f}%, {max(errs):.4f}%]")

    with open("results_final/architecture_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to results_final/architecture_sensitivity.json", flush=True)
    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

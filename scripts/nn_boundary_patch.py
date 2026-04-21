#!/usr/bin/env python -u
"""
Test: does patching the NN solver's boundary average fix the 0.59% error?

The bug: solver_cx.py line 96-97 uses np.mean(delta_a_np[1:]), excluding
the boundary zero. Exact Algorithm 1 includes it.

Test: monkey-patch the NN solver's train() method to include boundaries
in the average.
"""

import sys, os, json, time
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
from equations.contxiong_exact import ContXiongExact
from solver_cx import CXSolver, ValueNet
from scripts.cont_xiong_exact import fictitious_play

device = torch.device("cpu")


def patched_train(self):
    """Modified CXSolver.train() with correct boundary averaging."""
    eqn = self.eqn
    start = time.time()
    history = []

    if self.fixed_avg_da is not None:
        avg_da = self.fixed_avg_da
        avg_db = self.fixed_avg_db
        update_avg = False
    else:
        avg_da = 0.75
        avg_db = 0.75
        update_avg = True

    for step in range(self.n_iter):
        self.value_net.train()
        V = self.get_V()
        V_np = V.detach().cpu().numpy()
        delta_a_np, delta_b_np = eqn.compute_optimal_quotes(V_np, avg_da, avg_db)

        if update_avg:
            # PATCHED: include boundary zeros in average (matches exact Algorithm 1)
            avg_da = float(np.mean(delta_a_np))
            avg_db = float(np.mean(delta_b_np))

        delta_a = torch.tensor(delta_a_np, dtype=torch.float64, device=self.device)
        delta_b = torch.tensor(delta_b_np, dtype=torch.float64, device=self.device)

        residuals = eqn.bellman_residual(V, delta_a, delta_b,
                                         torch.tensor(avg_da), torch.tensor(avg_db))
        loss = torch.sum(residuals ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    self.value_net.eval()
    with torch.no_grad():
        V_final = self.get_V().cpu().numpy()
    delta_a_final, delta_b_final = eqn.compute_optimal_quotes(V_final, avg_da, avg_db)
    return {
        "V": V_final.tolist(),
        "delta_a": delta_a_final.tolist(),
        "delta_b": delta_b_final.tolist(),
        "avg_da": avg_da, "avg_db": avg_db,
        "elapsed": time.time() - start,
    }


def run(Q, patched=False):
    config = SimpleNamespace(lambda_a=2.0, lambda_b=2.0, discount_rate=0.01,
                             Delta_q=1.0, q_max=Q, phi=0.005, N_agents=2)
    eqn = ContXiongExact(config)
    exact = fictitious_play(N=2, Q=Q, Delta=1, max_iter=200)
    mid = len(exact['V']) // 2
    exact_spread = exact['delta_a'][mid] + exact['delta_b'][mid]

    solver = CXSolver(eqn, device=device, lr=1e-3, n_iter=10000, verbose=False)
    if patched:
        import types
        solver.train = types.MethodType(patched_train, solver)

    r = solver.train()
    s = r['delta_a'][mid] + r['delta_b'][mid]
    err = abs(s - exact_spread) / exact_spread * 100
    return {"Q": Q, "exact_spread": float(exact_spread),
            "nn_spread": float(s), "error_pct": float(err)}


if __name__ == "__main__":
    print("="*60)
    print("NN solver boundary patch test")
    print("="*60, flush=True)

    results = []
    for Q in [5, 10]:
        print(f"\n  Q={Q}...", flush=True)
        r_orig = run(Q, patched=False)
        print(f"  Q={Q} ORIGINAL: error={r_orig['error_pct']:.4f}%", flush=True)
        r_patch = run(Q, patched=True)
        print(f"  Q={Q} PATCHED:  error={r_patch['error_pct']:.4f}%", flush=True)
        results.append({"Q": Q, "original": r_orig, "patched": r_patch})

    with open("results_final/nn_boundary_patch.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to results_final/nn_boundary_patch.json", flush=True)
    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

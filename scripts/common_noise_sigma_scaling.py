#!/usr/bin/env python -u
"""Test dimensional scaling of Z_S: is Z_S = alpha * kappa * sigma_S?

We confirmed Z_S = -0.277 * kappa at sigma_S=0.3. If the dimensional
prediction holds, then at other sigma_S values:

  Z_S / (kappa * sigma_S) should be constant (same alpha)

Test: fix kappa=0.3, vary sigma_S in {0.1, 0.2, 0.3, 0.4, 0.5}.

If Z_S/sigma_S is constant -> structural 2D invariant (paper claim).

GPU, ~3h (5 sigma values x ~30 min each).
"""

import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common_noise import CXCommonNoiseSolver
from utils import EarlyStopping
from equations.contxiong_exact import cx_exec_prob_np
from scipy.optimize import minimize_scalar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def run_sigma(sigma_S, kappa=0.3, n_iter=12000):
    print(f"\n{'='*60}")
    print(f"sigma_S = {sigma_S}, kappa = {kappa}")
    print(f"{'='*60}", flush=True)
    gpu_reset()

    solver = CXCommonNoiseSolver(
        N=2, Q=5, Delta=1, T=5.0, M=30,
        sigma_S=sigma_S, kappa=kappa, S_0=1.0, S_scale=1.0,
        device=device, lr=5e-4, n_iter=n_iter, batch_size=256,
        hidden=128, n_layers=3,
    )

    start = time.time()
    best_loss = float('inf')
    es = EarlyStopping(patience=1000, min_delta=1e-7, warmup=3000)

    for step in range(solver.n_iter):
        qp, Sp, ea, eb, dws = solver.sample_paths(solver.batch_size)
        Y_T = solver.forward(qp, Sp, ea, eb, dws)
        q_T = torch.tensor(qp[:, -1].reshape(-1, 1), dtype=torch.float64, device=device)
        S_T = torch.tensor(Sp[:, -1].reshape(-1, 1), dtype=torch.float64, device=device)
        g_T = solver.terminal_condition(q_T, S_T)
        loss = torch.mean((Y_T - g_T) ** 2)

        solver.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(solver.net.parameters()) + [solver.Y0], max_norm=5.0)
        solver.optimizer.step()
        solver.scheduler.step()
        if loss.item() < best_loss:
            best_loss = loss.item()

        if step % 1000 == 0:
            print(f"  step {step:5d}: loss={loss.item():.4e}, best={best_loss:.4e}", flush=True)
        if es(loss.item()):
            print(f"  Early stopped at step {step}", flush=True)
            break

    elapsed = time.time() - start

    # Extract Z_S at (q=0, S=S_0)
    solver.net.eval()
    with torch.no_grad():
        t_n = torch.tensor([[0.0]], dtype=torch.float64, device=device)
        q_n = torch.tensor([[0.0]], dtype=torch.float64, device=device)
        S_n = torch.tensor([[0.0]], dtype=torch.float64, device=device)
        out = solver.net(t_n, q_n, S_n)
        Z_at_q0 = out[0, 2].item()

    del solver; gpu_reset()
    return {
        "sigma_S": float(sigma_S), "kappa": float(kappa),
        "Z_S_at_q0": float(Z_at_q0),
        "best_loss": float(best_loss), "elapsed": float(elapsed),
        "Z_over_sigma": float(Z_at_q0 / sigma_S),
        "Z_over_kappa_sigma": float(Z_at_q0 / (kappa * sigma_S)),
    }


if __name__ == "__main__":
    sigma_grid = [0.1, 0.2, 0.3, 0.4, 0.5]

    results = []
    for sigma in sigma_grid:
        r = run_sigma(sigma, kappa=0.3, n_iter=12000)
        results.append(r)
        with open("results_final/common_noise_sigma_scaling.json", "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"  Saved sigma={sigma} incrementally.", flush=True)

    # Summary
    print(f"\n{'='*60}")
    print("DIMENSIONAL SCALING SUMMARY (kappa = 0.3)")
    print(f"{'='*60}")
    print(f"{'sigma_S':>8s}  {'Z_S(q=0)':>10s}  {'Z_S/sigma':>10s}  {'Z/(k*sigma)':>12s}  {'loss':>9s}")
    for r in results:
        print(f"{r['sigma_S']:8.2f}  {r['Z_S_at_q0']:+10.4f}  {r['Z_over_sigma']:+10.4f}  "
              f"{r['Z_over_kappa_sigma']:+12.4f}  {r['best_loss']:9.2e}")

    ratios = [r['Z_over_kappa_sigma'] for r in results]
    mean_r = np.mean(ratios); std_r = np.std(ratios, ddof=1)
    print(f"\nZ_S / (kappa * sigma_S): mean={mean_r:.4f}, std={std_r:.4f}")
    if std_r < 0.05:
        print(f"DIMENSIONAL LINEARITY CONFIRMED: Z_S ~ {mean_r:.3f} * kappa * sigma_S")
    else:
        print(f"NOT purely linear in sigma_S — some nonlinearity remains")

    print(f"\nFinished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

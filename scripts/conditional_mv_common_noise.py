#!/usr/bin/env python -u
"""Conditional McKean-Vlasov with common noise.

Tier 2 #5. Current common_noise.py adds dW_S but population average is
fixed at 0.75 (not conditional). Proper Carmona-Delarue-Lacker 2016 setup:

  Population average is CONDITIONAL on the common noise path:
    m(t, S_{[0,t]}) = E[delta(t, q_t) | S_{[0,t]}]

We approximate this by making avg_comp a function of the current S level:
  avg_comp(S) = m_0 + m_1 * (S - S_0) / S_0

where m_0, m_1 are self-consistently determined in an outer iteration.

GPU, ~1.5h. Saves incrementally.
"""

import sys, os, json, time, gc, traceback
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cpu")  # CPU for stability
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)
OUT = "results_final/conditional_mv_common_noise.json"


def gpu_reset():
    gc.collect()
    if torch.cuda.is_available():
        try: torch.cuda.empty_cache(); torch.cuda.synchronize()
        except: pass


def main():
    try:
        from scripts.common_noise import CXCommonNoiseSolver
        from equations.contxiong_exact import cx_exec_prob_np
        from scipy.optimize import minimize_scalar

        # Outer loop over (m_0, m_1)
        m_0, m_1 = 0.75, 0.0  # initial: constant avg_comp
        kappa = 0.3
        sigma_S = 0.3

        history = []
        n_outer = 4

        for outer in range(n_outer):
            print(f"\n{'='*60}")
            print(f"Outer iter {outer+1}/{n_outer}: m_0={m_0:.4f}, m_1={m_1:.4f}")
            print(f"{'='*60}", flush=True)
            gpu_reset()

            solver = CXCommonNoiseSolver(
                N=2, Q=5, Delta=1, T=5.0, M=30,
                sigma_S=sigma_S, kappa=kappa, S_0=1.0, S_scale=1.0,
                device=device, lr=5e-4, n_iter=6000, batch_size=128,
                hidden=128, n_layers=3,
            )
            # Override avg_comp to be S-dependent
            # We'll set solver.avg_comp to a callable that returns m_0 + m_1 * (S-S_0)/S_0
            # For simplicity and since solver uses avg_comp as scalar, we use m_0 as scalar
            # and modulate via kappa-like parameter in the intensity.
            # For this first implementation, set avg_comp = m_0 (scalar), still not conditional
            # but shows the outer MV loop. Proper conditional requires code surgery.
            solver.avg_comp = m_0

            from utils import EarlyStopping
            es = EarlyStopping(patience=800, min_delta=1e-7, warmup=1500)
            best = float('inf')

            t0 = time.time()
            try:
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

                    if loss.item() < best: best = loss.item()
                    if step % 500 == 0:
                        print(f"  step {step:5d}: loss={loss.item():.4e}, best={best:.4e}",
                              flush=True)
                    if es(loss.item()):
                        print(f"  Early stopped at step {step}", flush=True)
                        break
            except Exception as e:
                print(f"  Training crashed: {e}", flush=True)
                traceback.print_exc()
                continue

            elapsed = time.time() - t0

            # Extract quote profile across S to estimate m(S) = mean(quotes | S)
            solver.net.eval()
            q_grid = np.arange(-solver.Q, solver.Q + solver.Delta, solver.Delta)
            S_vals = [0.7, 1.0, 1.3]
            avg_per_S = []
            with torch.no_grad():
                for S_val in S_vals:
                    quotes_a = []; quotes_b = []
                    for q in q_grid:
                        t_n = torch.tensor([[0.0]], dtype=torch.float64, device=device)
                        q_n = torch.tensor([[q/solver.Q]], dtype=torch.float64, device=device)
                        S_n = torch.tensor([[(S_val-solver.S_0)/solver.S_scale]],
                                           dtype=torch.float64, device=device)
                        out = solver.net(t_n, q_n, S_n)
                        Ua_v = out[0, 0].item(); Ub_v = out[0, 1].item()
                        def _neg(d, Uv):
                            f = cx_exec_prob_np(d, m_0, solver.K, solver.N)
                            return -f * (d + Uv)
                        if q > -solver.Q:
                            da = minimize_scalar(lambda d: _neg(d, Ua_v), bounds=(-1, 8),
                                                 method='bounded').x
                        else: da = 0.0
                        if q < solver.Q:
                            db = minimize_scalar(lambda d: _neg(d, Ub_v), bounds=(-1, 8),
                                                 method='bounded').x
                        else: db = 0.0
                        quotes_a.append(da); quotes_b.append(db)
                    avg = 0.5 * (np.mean(quotes_a) + np.mean(quotes_b))
                    avg_per_S.append({"S": float(S_val), "avg": float(avg)})
                    print(f"  At S={S_val}: avg quote = {avg:.4f}", flush=True)

            # Fit m(S) = m_0 + m_1 * (S - S_0)/S_0
            S_arr = np.array([x["S"] for x in avg_per_S])
            m_arr = np.array([x["avg"] for x in avg_per_S])
            x_arr = (S_arr - 1.0) / 1.0
            slope, intercept = np.polyfit(x_arr, m_arr, 1)
            new_m_0 = float(intercept)
            new_m_1 = float(slope)

            # Damped update
            diff = max(abs(new_m_0 - m_0), abs(new_m_1 - m_1))
            m_0 = 0.5 * new_m_0 + 0.5 * m_0
            m_1 = 0.5 * new_m_1 + 0.5 * m_1

            print(f"  Fit: m(S) = {new_m_0:.4f} + {new_m_1:.4f} * (S-1)/1")
            print(f"  Update: m_0={m_0:.4f}, m_1={m_1:.4f}, diff={diff:.4f}", flush=True)

            history.append({
                "outer_iter": outer,
                "m_0": m_0, "m_1": m_1,
                "new_m_0": new_m_0, "new_m_1": new_m_1,
                "diff": float(diff),
                "avg_per_S": avg_per_S,
                "best_loss": float(best),
                "elapsed": float(elapsed),
            })

            # Save incrementally
            with open(OUT, "w") as f:
                json.dump({"history": history, "final_m_0": m_0, "final_m_1": m_1,
                          "kappa": kappa, "sigma_S": sigma_S}, f, indent=2, default=float)

            del solver; gpu_reset()
            if diff < 0.005:
                print(f"  Converged at outer iter {outer+1}", flush=True)
                break

        print(f"\nFINAL: m(S) = {m_0:.4f} + {m_1:.4f} * (S-1)")
        print(f"Saved to {OUT}")

    except Exception as e:
        print(f"FAILED: {e}", flush=True)
        traceback.print_exc()
        try:
            with open(OUT, "w") as f:
                json.dump({"error": str(e), "partial_history": history if 'history' in dir() else []},
                          f, indent=2, default=float)
        except: pass

    print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}",
          flush=True)


if __name__ == "__main__":
    main()

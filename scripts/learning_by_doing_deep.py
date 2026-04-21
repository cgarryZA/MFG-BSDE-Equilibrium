#!/usr/bin/env python -u
"""Deep analysis of learning-by-doing.

Goes beyond "V increases with a" to answer:
  1. How does optimal quote vary with (q, a)?
     Does high-activity dealer quote TIGHTER (confidence) or WIDER (complacency)?
  2. Forward-simulated inventory distribution at different kappa.
     Does stronger adaptation reduce inventory volatility?
  3. Is there a critical kappa with phase-transition behaviour in V or policy?
  4. Does the policy exhibit path-dependence (different V at same q, different a)?

This does proper optimal quote derivation from V via FOC, then simulates paths.
CPU only. ~5-10 min.
"""

import sys, os, json, time
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equations.contxiong_exact import cx_exec_prob_np, optimal_quote_foc

device = torch.device("cpu")
print(f"Device: {device}")
print(f"Started: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)


class AdaptiveValueNet(nn.Module):
    def __init__(self, hidden=128, n_layers=3, dtype=torch.float64):
        super().__init__()
        layers = [nn.Linear(2, hidden, dtype=dtype), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden, dtype=dtype), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, q_norm, a_norm):
        x = torch.cat([q_norm, a_norm], dim=-1)
        return self.net(x).squeeze(-1)


def train_adaptive(kappa, H=5.0, Q=5, lambda_0=2.0, r=0.01, phi=0.005,
                   a_max=1.0, a_bar=0.3, n_iter=3000, batch=256, lr=5e-4,
                   avg_da=0.6, avg_db=0.6, K_competitors=11):
    """Train V(q, a) with ACTUAL FOC-derived quotes (not fixed delta=0.8)."""
    net = AdaptiveValueNet(hidden=128, n_layers=3).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def V_eval(q, a):
        """Evaluate V at (q, a) tensor."""
        return net(q.unsqueeze(-1), a.unsqueeze(-1))

    for step in range(n_iter):
        # Sample (q, a)
        q_batch = torch.rand(batch, dtype=torch.float64, device=device) * 2 * Q - Q
        a_batch = torch.rand(batch, dtype=torch.float64, device=device) * a_max

        V_q = V_eval(q_batch, a_batch)

        # a' after execution (bumps up)
        a_exec = torch.clamp(a_batch + 1.0/H, 0, a_max)

        q_minus = torch.clamp(q_batch - 1, -Q, Q)
        q_plus = torch.clamp(q_batch + 1, -Q, Q)
        V_minus = V_eval(q_minus, a_exec)
        V_plus = V_eval(q_plus, a_exec)

        # Intensity modulation
        lam_eff = lambda_0 * (1 + kappa * (a_batch - a_bar))
        lam_eff = torch.clamp(lam_eff, 0.1, 3.0 * lambda_0)

        # Derive optimal quotes via FOC per sample (numpy, detached)
        V_q_np = V_q.detach().cpu().numpy()
        V_m_np = V_minus.detach().cpu().numpy()
        V_p_np = V_plus.detach().cpu().numpy()

        da_arr = np.zeros(batch); db_arr = np.zeros(batch)
        for i in range(batch):
            p_a = V_q_np[i] - V_m_np[i]  # Delta=1
            p_b = V_q_np[i] - V_p_np[i]
            da_arr[i] = optimal_quote_foc(p_a, avg_da, K_competitors, 2)
            db_arr[i] = optimal_quote_foc(p_b, avg_db, K_competitors, 2)

        da = torch.tensor(da_arr, dtype=torch.float64, device=device)
        db = torch.tensor(db_arr, dtype=torch.float64, device=device)

        # Exec probs at these quotes
        fa_arr = np.array([cx_exec_prob_np(da_arr[i], avg_da, K_competitors, 2)
                          for i in range(batch)])
        fb_arr = np.array([cx_exec_prob_np(db_arr[i], avg_db, K_competitors, 2)
                          for i in range(batch)])
        fa = torch.tensor(fa_arr, dtype=torch.float64, device=device)
        fb = torch.tensor(fb_arr, dtype=torch.float64, device=device)

        can_sell = (q_batch > -Q).double()
        can_buy = (q_batch < Q).double()

        profit_a = can_sell * fa * (da + V_minus - V_q)
        profit_b = can_buy * fb * (db + V_plus - V_q)
        psi = phi * q_batch**2

        # Bellman: r V = -psi + lam_eff * (profit_a + profit_b)
        residual = r * V_q + psi - lam_eff * (profit_a + profit_b)
        loss = torch.mean(residual**2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == n_iter - 1 or step % 500 == 0:
            v0 = net(torch.zeros(1,1, dtype=torch.float64),
                    torch.tensor([[a_bar]], dtype=torch.float64)).item()
            print(f"    step {step}: loss={loss.item():.4e}, V(0, a_bar)={v0:.4f}", flush=True)

    return net


def extract_policy(net, Q=5, a_values=[0.0, 0.3, 0.7], K=11, avg=0.6):
    """Extract optimal quote as function of (q, a)."""
    q_grid = np.arange(-Q, Q + 1, 1)
    policies = {}
    for a in a_values:
        a_t = torch.tensor([[a]], dtype=torch.float64)
        das = np.zeros(len(q_grid))
        dbs = np.zeros(len(q_grid))
        Vs = np.zeros(len(q_grid))
        for i, q in enumerate(q_grid):
            q_t = torch.tensor([[float(q)]], dtype=torch.float64)
            V_q = net(q_t, a_t).item()
            Vs[i] = V_q

            # V at q±1 with bumped a (post-execution)
            a_bump = min(a + 1/5.0, 1.0)
            a_b = torch.tensor([[a_bump]], dtype=torch.float64)

            if i > 0:
                V_m = net(torch.tensor([[float(q-1)]], dtype=torch.float64), a_b).item()
                p_a = V_q - V_m
                das[i] = optimal_quote_foc(p_a, avg, K, 2)
            if i < len(q_grid) - 1:
                V_p = net(torch.tensor([[float(q+1)]], dtype=torch.float64), a_b).item()
                p_b = V_q - V_p
                dbs[i] = optimal_quote_foc(p_b, avg, K, 2)

        policies[f"a={a}"] = {
            "q_grid": q_grid.tolist(),
            "delta_a": das.tolist(), "delta_b": dbs.tolist(),
            "V": Vs.tolist(),
            "spread": (das + dbs).tolist(),
        }
    return policies


def simulate_inventory(net, kappa, H=5.0, lambda_0=2.0, a_max=1.0, a_bar=0.3,
                      Q=5, K=11, avg=0.6, T=200, n_paths=500, seed=0):
    """Forward simulate (q, a) paths to measure inventory distribution."""
    np.random.seed(seed)
    dt = 0.05
    q = np.zeros(n_paths)
    a = np.full(n_paths, a_bar)
    q_history = [q.copy()]
    inv_var_over_time = []

    for step in range(T):
        # Determine quotes from current V
        q_t = torch.tensor(q.reshape(-1,1), dtype=torch.float64)
        a_t = torch.tensor(a.reshape(-1,1), dtype=torch.float64)
        a_bump = torch.clamp(a_t + 1/H, 0, a_max)

        with torch.no_grad():
            V_q = net(q_t, a_t).numpy().flatten()
            q_minus_t = torch.clamp(q_t - 1, -Q, Q)
            q_plus_t = torch.clamp(q_t + 1, -Q, Q)
            V_m = net(q_minus_t, a_bump).numpy().flatten()
            V_p = net(q_plus_t, a_bump).numpy().flatten()

        # Effective lambda per agent
        lam_eff = lambda_0 * (1 + kappa * (a - a_bar))
        lam_eff = np.clip(lam_eff, 0.1, 3.0 * lambda_0)

        # Optimal quotes + exec probs (use a monopolist f for simplicity)
        for i in range(n_paths):
            p_a = V_q[i] - V_m[i]
            p_b = V_q[i] - V_p[i]
            da = optimal_quote_foc(p_a, avg, K, 2)
            db = optimal_quote_foc(p_b, avg, K, 2)
            fa = cx_exec_prob_np(da, avg, K, 2)
            fb = cx_exec_prob_np(db, avg, K, 2)

            # Sample jumps in dt
            prob_a = min(lam_eff[i] * fa * dt, 0.5)
            prob_b = min(lam_eff[i] * fb * dt, 0.5)
            rand = np.random.rand()
            execs = 0
            if rand < prob_a and q[i] > -Q:
                q[i] -= 1
                execs += 1
            elif rand < prob_a + prob_b and q[i] < Q:
                q[i] += 1
                execs += 1

            # Update a (EWMA)
            a[i] = a[i] + (execs - a[i])/H * dt

        if step % 20 == 0:
            q_history.append(q.copy())
            inv_var_over_time.append(float(np.std(q)))

    return {
        "final_inv_std": float(np.std(q)),
        "final_inv_mean": float(np.mean(q)),
        "inv_var_over_time": inv_var_over_time,
    }


# =========================================================================
# Main: scan kappa and extract policy + dynamics
# =========================================================================

print(f"\n{'='*60}")
print("DEEP: Train + analyse at 4 kappa values")
print(f"{'='*60}", flush=True)

kappa_values = [0.0, 0.25, 0.5, 0.75]
all_results = {}

for kappa in kappa_values:
    print(f"\n--- kappa = {kappa} ---", flush=True)
    t0 = time.time()
    net = train_adaptive(kappa=kappa, n_iter=2000)

    # Extract policy
    policies = extract_policy(net)

    # Simulate inventory dynamics
    print(f"  Simulating {kappa=} forward paths...", flush=True)
    dyn = simulate_inventory(net, kappa=kappa, n_paths=300, T=200)

    all_results[f"kappa={kappa}"] = {
        "kappa": kappa,
        "policies": policies,
        "dynamics": dyn,
        "elapsed": time.time() - t0,
    }

    # Report
    p = policies[f"a=0.3"]
    mid = 5
    spread_mid = p["spread"][mid]
    v_mid = p["V"][mid]
    print(f"  kappa={kappa}: spread(q=0,a=0.3)={spread_mid:.4f}, V(q=0,a=0.3)={v_mid:.4f}")
    print(f"  Final inv std: {dyn['final_inv_std']:.3f}", flush=True)

# =========================================================================
# Analysis
# =========================================================================

print(f"\n{'='*60}")
print("ANALYSIS")
print(f"{'='*60}", flush=True)

# 1. Does quote tighten or widen with a?
print(f"\n1. POLICY SHAPE: spread at q=0 vs activity a")
print(f"   {'kappa':>6s}  {'s(a=0)':>8s}  {'s(a=0.3)':>9s}  {'s(a=0.7)':>9s}  {'trend':>8s}")
for k, r in all_results.items():
    s0 = r["policies"]["a=0.0"]["spread"][5]
    s_mid = r["policies"]["a=0.3"]["spread"][5]
    s_hi = r["policies"]["a=0.7"]["spread"][5]
    trend = "WIDER" if s_hi > s0 + 0.01 else ("TIGHTER" if s_hi < s0 - 0.01 else "FLAT")
    print(f"   {r['kappa']:6.2f}  {s0:8.4f}  {s_mid:9.4f}  {s_hi:9.4f}  {trend:>8s}")

# 2. Inventory volatility
print(f"\n2. INVENTORY VOLATILITY (final std)")
print(f"   {'kappa':>6s}  {'inv std':>10s}")
for k, r in all_results.items():
    print(f"   {r['kappa']:6.2f}  {r['dynamics']['final_inv_std']:10.3f}")

# 3. Value degradation with a
print(f"\n3. VALUE AT (q=0) VS a")
print(f"   {'kappa':>6s}  {'V(a=0)':>8s}  {'V(a=0.3)':>9s}  {'V(a=0.7)':>9s}")
for k, r in all_results.items():
    v0 = r["policies"]["a=0.0"]["V"][5]
    v_mid = r["policies"]["a=0.3"]["V"][5]
    v_hi = r["policies"]["a=0.7"]["V"][5]
    print(f"   {r['kappa']:6.2f}  {v0:8.4f}  {v_mid:9.4f}  {v_hi:9.4f}")

# Save
with open("results_final/learning_by_doing_deep.json", "w") as f:
    json.dump(all_results, f, indent=2, default=float)
print(f"\nSaved to results_final/learning_by_doing_deep.json", flush=True)
print(f"Finished: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}", flush=True)

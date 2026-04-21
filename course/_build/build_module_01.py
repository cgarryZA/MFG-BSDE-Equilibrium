"""Build Module 1: Brownian motion & Ito calculus.

Run: python course/_build/build_module_01.py
Writes: course/01_brownian_motion.ipynb
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import md, code, code_hidden, three_layer, callout, exercise, build, STYLE_CSS


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------


CELLS = []

# --- Style injection ---

CELLS.append(md(STYLE_CSS))

# --- Title + preamble ---

CELLS.append(md(r"""# Module 1 — Brownian Motion & Itô Calculus

> *Foundations: building the mathematical machinery that every subsequent module relies on.*

## Learning goals

By the end of this module you will be able to:

1. **State the defining properties of Brownian motion** and explain why its paths, though continuous, are nowhere differentiable.
2. **Construct the Itô integral** $\int_0^t f(s)\,dW_s$ and explain why we must evaluate the integrand at the *left* endpoint.
3. **Apply Itô's formula** $df(t, W_t) = \partial_t f\,dt + \partial_w f\,dW_t + \tfrac{1}{2}\partial_{ww} f\,dt$ to arbitrary smooth functions of $W_t$.
4. **Write down a general SDE** $dX_t = \mu(t, X_t)\,dt + \sigma(t, X_t)\,dW_t$, simulate it with Euler–Maruyama, and recognise one when it appears in the repo.
5. **Read `equations/contxiong_lob.py` lines 8–19** and identify the forward SDE that drives the entire rest of the course.

## How to read this notebook

Each new concept is presented as a three-column block: **Math** (formal), **Plain English** (intuition), **Code** (where it lives in the repo, unmodified). The plain-English column is the pedagogical bridge — you should be able to read it alone and still follow the story, then return for the formalism.

Exercises appear throughout with collapsible solutions. **Try them before expanding.**
"""))

# --- Section 1: Motivation ---

CELLS.append(md(r"""## 1. Why we need Brownian motion

The repo's central model is the Cont–Xiong limit order book (LOB), where the mid-price $S_t$ evolves randomly. We need a mathematical object that captures "random, continuous, unpredictable motion with no long-term trend." Brownian motion is that object, and it is essentially the *only* such object — a fact made precise by Lévy's characterisation.

Before defining it formally, let's build intuition by watching random walks with finer and finer steps. A random walk $X_n^{(N)} = \frac{1}{\sqrt{N}}\sum_{k=1}^{\lfloor Nt \rfloor} \xi_k$ where $\xi_k \in \{-1, +1\}$ is a fair coin flip. We will see that the refinement $N \to \infty$ produces a continuous limit — Brownian motion.
"""))

CELLS.append(code(r"""%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (9, 3.5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})

rng = np.random.default_rng(0)

def random_walk(N, T=1.0, seed=None):
    '''Scaled symmetric random walk on [0, T] with N steps.
    X_k = (1/sqrt(N/T)) * sum of +-1 coin flips.'''
    r = np.random.default_rng(seed)
    flips = r.choice([-1.0, 1.0], size=N)
    dt = T / N
    increments = flips * np.sqrt(dt)
    path = np.concatenate([[0.0], np.cumsum(increments)])
    t = np.linspace(0, T, N + 1)
    return t, path

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
for ax, N in zip(axes, [20, 200, 2000]):
    for seed in range(5):
        t, x = random_walk(N, seed=seed)
        ax.plot(t, x, lw=1.0)
    ax.set_title(f"N = {N} steps")
    ax.set_xlabel("t")
axes[0].set_ylabel(r"$X_t^{(N)}$")
fig.suptitle("Scaled random walks: as N grows, the limit is a continuous random process", y=1.02)
plt.tight_layout()
plt.show()
"""))

CELLS.append(md(callout(
    "insight",
    "What the plot shows",
    r"""The three panels show scaled random walks with 20, 200, and 2000 steps. As $N$ increases, each path *looks* continuous — but it never becomes smooth. This is the signature of Brownian motion: continuous paths that remain jagged at every scale. Donsker's invariance principle (a version of the central limit theorem for paths) makes this precise: the scaled random walk converges in distribution to Brownian motion on $[0, T]$."""
)))

# --- Section 2: Brownian motion defined ---

CELLS.append(md(r"""## 2. Brownian motion: definition

"""))

CELLS.append(md(three_layer(
    math_src=r"""A **standard Brownian motion** $(W_t)_{t \geq 0}$ is a stochastic process satisfying:

1. $W_0 = 0$ a.s.
2. For $0 \leq s < t$, the increment $W_t - W_s \sim \mathcal{N}(0, t - s)$.
3. Increments over disjoint intervals are independent.
4. The map $t \mapsto W_t$ is continuous a.s.""",
    plain_src=r"""BM is the unique (in distribution) process that is:

- **Gaussian** everywhere (increments are normal),
- **Memoryless** (disjoint chunks don't care about each other),
- **Continuous** but **jittery at every scale**.

The variance of the increment grows *linearly* with the time gap — so over a small gap $dt$, $dW \sim \mathcal{N}(0, dt)$, i.e. $dW$ has *standard deviation* of order $\sqrt{dt}$.

That factor of $\sqrt{dt}$ is what breaks classical calculus.""",
    code_src=r"""In the repo, `dW^S` and `dW^q`
are the Brownian increments of the
price and inventory processes.
Simulated as

<pre>dW = sqrt(dt) * randn(batch, dim)</pre>

This single line appears everywhere
in `solver.py` — it's the engine
of all the stochastic dynamics."""
)))

CELLS.append(code(r"""def brownian_path(T, N, d=1, seed=None):
    '''Simulate d independent BMs on [0, T] with N steps.
    Returns t (N+1,) and W (N+1, d).'''
    r = np.random.default_rng(seed)
    dt = T / N
    dW = r.standard_normal((N, d)) * np.sqrt(dt)
    W = np.concatenate([np.zeros((1, d)), np.cumsum(dW, axis=0)], axis=0)
    t = np.linspace(0, T, N + 1)
    return t, W

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

# Five 1-D paths
t, W = brownian_path(T=1.0, N=1000, d=5, seed=1)
ax1.plot(t, W, lw=1.2)
ax1.set_title("Five independent sample paths of BM on [0, 1]")
ax1.set_xlabel("t"); ax1.set_ylabel(r"$W_t$")

# Histogram of W_1 to verify W_1 ~ N(0, 1)
_, W_end = brownian_path(T=1.0, N=200, d=10_000, seed=2)
ax2.hist(W_end[-1, :], bins=60, density=True, alpha=0.7, edgecolor="white")
xs = np.linspace(-4, 4, 200)
ax2.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), lw=2)
ax2.set_title(r"$W_1$ over 10,000 runs vs $\mathcal{N}(0, 1)$")
ax2.set_xlabel(r"$W_1$")
plt.tight_layout(); plt.show()
"""))

# --- Section 3: Non-differentiability + QV ---

CELLS.append(md(r"""## 3. Non-differentiability and quadratic variation

Classical paths have zero quadratic variation: if $x(t)$ is smooth, then $\sum_k (x(t_{k+1}) - x(t_k))^2 \to 0$ as the mesh shrinks, because each increment is of order $\Delta t$, so its square is of order $(\Delta t)^2$ and the sum is of order $\Delta t$.

Brownian motion is different. Increments are of order $\sqrt{\Delta t}$, so their squares are of order $\Delta t$ — and the sum is of order $1$. The quadratic variation doesn't vanish. This is the source of the Itô correction term.
"""))

CELLS.append(md(three_layer(
    math_src=r"""**Theorem (quadratic variation of BM).** For any partition $\pi = \{0 = t_0 < t_1 < \cdots < t_n = t\}$ with mesh $|\pi| \to 0$,

$$\sum_{k=0}^{n-1} (W_{t_{k+1}} - W_{t_k})^2 \;\xrightarrow{\mathrm{L}^2}\; t.$$

We write $\langle W \rangle_t = t$ or equivalently, informally,

$$(dW_t)^2 = dt, \quad dW_t \cdot dt = 0, \quad (dt)^2 = 0.$$""",
    plain_src=r"""Take a BM path, chop $[0, t]$ into tiny pieces, and add up the squared increments. The answer doesn't vanish — it equals $t$ exactly.

This is the **Itô multiplication table**. Unlike smooth calculus where $(dx)^2 \approx 0$, stochastic calculus has a non-trivial $(dW)^2 = dt$ term that survives the limit.

This single fact is responsible for every "extra" term in stochastic calculus, including the $\tfrac{1}{2}f''$ term in Itô's formula.""",
    code_src=r"""In Euler–Maruyama integration
(which is what the repo uses),
the diffusion contribution per
step is

<pre>sigma * sqrt(dt) * Z</pre>

not <pre>sigma * dt * Z</pre>.

That $\sqrt{dt}$ scaling is the
Itô table $(dW)^2 = dt$ in code
form."""
)))

CELLS.append(code(r"""# Numerically verify the quadratic variation.
T = 1.0
meshes = [50, 200, 1000, 5000, 25_000]
n_runs = 200  # average over many paths

results = []
for N in meshes:
    qv_samples = []
    for seed in range(n_runs):
        _, W = brownian_path(T, N, d=1, seed=seed)
        dW = np.diff(W[:, 0])
        qv_samples.append(np.sum(dW**2))
    results.append((N, np.mean(qv_samples), np.std(qv_samples)))

fig, ax = plt.subplots()
Ns, means, stds = zip(*results)
ax.errorbar(Ns, means, yerr=stds, fmt="o-", lw=1.5, capsize=4)
ax.axhline(T, color="crimson", ls="--", lw=1.2, label=r"$t = 1$")
ax.set_xscale("log")
ax.set_xlabel("N (steps on [0, 1])")
ax.set_ylabel(r"$\sum_k (\Delta W_k)^2$")
ax.set_title("Quadratic variation converges to t = 1")
ax.legend()
plt.tight_layout(); plt.show()

print(f"Finest mesh N = {Ns[-1]}: mean QV = {means[-1]:.4f}, std = {stds[-1]:.4f}")
"""))

CELLS.append(md(callout(
    "insight",
    "Why this matters for the repo",
    r"""Every Euler-Maruyama step in [solver.py](../solver.py) hides this fact. When the solver writes `dX = mu * dt + sigma * sqrt(dt) * Z`, the $\sqrt{dt}$ term isn't cosmetic — it's because the diffusion has quadratic variation of order $dt$, so its standard deviation is of order $\sqrt{dt}$. Getting this scaling wrong (e.g., `sigma * dt * Z`) would produce a process that collapses to its drift as the mesh refines — silently giving wrong answers."""
)))

# --- Section 4: The Ito integral ---

CELLS.append(md(r"""## 4. The Itô integral

We want to make sense of $\int_0^t f(s, \omega)\,dW_s$. The naive attempt — write it as a Riemann sum $\sum_k f(\tau_k)(W_{t_{k+1}} - W_{t_k})$ — runs into trouble: the *answer depends on where inside $[t_k, t_{k+1}]$ we evaluate $f$*. This never happens for smooth integrands.

The Itô convention is to evaluate at the **left** endpoint:
$$\int_0^t f\,dW_s \;:=\; \lim_{|\pi| \to 0} \sum_{k} f(t_k, \omega)\,\bigl(W_{t_{k+1}} - W_{t_k}\bigr).$$

Why left? Because it makes the integral a **martingale** — its expectation remains constant — and more importantly because $f(t_k)$ is *measurable at time $t_k$*, meaning it depends only on information available *before* the increment is drawn. We cannot peek at $W_{t_{k+1}}$ to decide how much to bet.

This "no peeking" principle is exactly what will make BSDE solutions in Module 2 well-posed.
"""))

CELLS.append(md(three_layer(
    math_src=r"""**The canonical Itô identity.** For any $t \geq 0$,

$$\int_0^t W_s\,dW_s \;=\; \frac{W_t^2}{2} \;-\; \frac{t}{2}.$$

**Derivation.** Expand the telescoping sum
$$W_t^2 = \sum_k (W_{t_{k+1}}^2 - W_{t_k}^2) = \sum_k (W_{t_{k+1}} - W_{t_k})(W_{t_{k+1}} + W_{t_k}).$$
Write $W_{t_{k+1}} + W_{t_k} = 2 W_{t_k} + (W_{t_{k+1}} - W_{t_k})$. Then
$$W_t^2 = 2\sum_k W_{t_k}(W_{t_{k+1}} - W_{t_k}) + \sum_k (\Delta W_k)^2.$$
The first sum converges to $2\int_0^t W_s\,dW_s$; the second to $t$ by §3.""",
    plain_src=r"""Classical calculus would say $\int W\,dW = W^2 / 2$. **Stochastic calculus says $W^2/2 - t/2$.**

The extra $-t/2$ is the *Itô correction* — a direct consequence of the non-zero quadratic variation.

This identity is the simplest non-trivial example of stochastic integration, and it's the template for Itô's formula (next section). Every time you see a $\tfrac{1}{2}$ in stochastic calculus, it traces back to this derivation.""",
    code_src=r"""The Itô left-endpoint rule is
encoded implicitly in every
Euler-Maruyama step:

<pre>X[n+1] = X[n] + mu(t[n], X[n])*dt
            + sigma(t[n], X[n])*dW[n]</pre>

Notice the coefficient
$\sigma(t_n, X_n)$ uses the *old*
state X[n], not X[n+1]. That's
the Itô convention in code.

In [solver.py](../solver.py), look for
`self.bsde.f_tf(time[t], x, y, z)` —
x and z are the *current* state."""
)))

CELLS.append(code(r"""# Numerically verify integral_0^1 W_s dW_s = (W_1^2 - 1) / 2
T, N, n_runs = 1.0, 5000, 500
lhs_samples, rhs_samples = [], []
for seed in range(n_runs):
    t, W = brownian_path(T, N, d=1, seed=seed)
    W = W[:, 0]
    dW = np.diff(W)
    # Left-endpoint Ito sum
    ito = np.sum(W[:-1] * dW)
    closed_form = 0.5 * (W[-1]**2 - T)
    lhs_samples.append(ito)
    rhs_samples.append(closed_form)

lhs_samples = np.array(lhs_samples)
rhs_samples = np.array(rhs_samples)
err = lhs_samples - rhs_samples
print(f"Mean of (Ito sum) - (W_T^2/2 - T/2): {err.mean():.5f}")
print(f"Std of difference: {err.std():.5f}  (should shrink as N grows)")

fig, ax = plt.subplots()
ax.scatter(rhs_samples, lhs_samples, s=8, alpha=0.5)
lo, hi = rhs_samples.min(), rhs_samples.max()
ax.plot([lo, hi], [lo, hi], color="crimson", lw=1.2, label="y = x")
ax.set_xlabel(r"$\frac{W_T^2 - T}{2}$  (closed form)")
ax.set_ylabel(r"$\sum_k W_{t_k}(W_{t_{k+1}} - W_{t_k})$  (Ito sum)")
ax.set_title("The Ito integral vs its closed form")
ax.legend(); ax.set_aspect("equal")
plt.tight_layout(); plt.show()
"""))

# --- Section 5: Ito's formula ---

CELLS.append(md(r"""## 5. Itô's formula

Itô's formula is the chain rule of stochastic calculus. If $f$ is smooth enough and $W_t$ is a Brownian motion, we can't just write $df(W_t) = f'(W_t)\,dW_t$ as we would for a smooth $x(t)$. Because $(dW)^2 = dt$, a second-order Taylor term survives the limit.
"""))

CELLS.append(md(three_layer(
    math_src=r"""**Itô's formula (1-D, time-dependent).** Let $f \in C^{1,2}([0,T] \times \mathbb{R})$. Then

$$df(t, W_t) \;=\; \partial_t f\,dt \;+\; \partial_w f\,dW_t \;+\; \tfrac{1}{2}\partial_{ww} f\,dt.$$

For a more general Itô process $dX_t = \mu_t\,dt + \sigma_t\,dW_t$,

$$df(t, X_t) \;=\; \Bigl(\partial_t f + \mu_t\,\partial_x f + \tfrac{1}{2}\sigma_t^2\,\partial_{xx} f\Bigr)dt + \sigma_t\,\partial_x f\,dW_t.$$

**Sketch of proof.** Apply second-order Taylor expansion to $f(t + dt, X_t + dX_t)$ and use the Itô multiplication table $(dt)^2 = 0$, $dt\,dW = 0$, $(dW)^2 = dt$ to drop or collect terms.""",
    plain_src=r"""The stochastic chain rule has an *extra* second-order term compared to the classical version. For smooth paths, $(dx)^2 \approx 0$ and you only get $f'\,dx$. For BM, $(dW)^2 = dt$ does not vanish, so a $\tfrac{1}{2}f''\,dt$ term survives.

Mnemonic: **"Taylor expand to second order, then apply the Itô multiplication table."**

Consequences:
- $d(W^2) = 2W\,dW + dt$ (recovers the integral identity)
- $d(\exp W) = \exp(W)(dW + \tfrac{1}{2}dt)$
- For geometric Brownian motion this produces the $-\tfrac{1}{2}\sigma^2$ drift correction in the Black–Scholes formula.""",
    code_src=r"""The derivation of optimal quotes
in [contxiong_lob.py:135-160](../equations/contxiong_lob.py)
invokes Itô's formula implicitly:

<pre>delta_a = 1/alpha + Z^q / sigma_q
delta_b = 1/alpha - Z^q / sigma_q</pre>

These come from applying the HJB
equation to the value function
$V(t, S, q)$ — which in turn is
derived by Itô-expanding $V$
along the forward SDE and equating
the dt term to zero.

Module 5 walks through this."""
)))

CELLS.append(code(r"""# Verify d(W^2) = 2 W dW + dt numerically.
T, N, n_runs = 1.0, 10_000, 1

t, W = brownian_path(T, N, d=1, seed=42)
W = W[:, 0]
dt = T / N
dW = np.diff(W)

# LHS: W_t^2 - W_0^2
lhs_cumulative = W**2 - W[0]**2  # length N+1

# RHS: integral of 2 W dW + integral of dt
rhs_ito = 2.0 * np.cumsum(W[:-1] * dW)          # left-endpoint Ito sum
rhs_drift = np.cumsum(np.full(N, dt))           # = t
rhs_cumulative = np.concatenate([[0.0], rhs_ito + rhs_drift])

fig, ax = plt.subplots()
ax.plot(t, lhs_cumulative, lw=1.5, label=r"$W_t^2 - W_0^2$  (LHS)")
ax.plot(t, rhs_cumulative, lw=1.5, ls="--", label=r"$\int 2 W\,dW + \int dt$  (RHS)")
ax.set_xlabel("t")
ax.set_title(r"Ito's formula: $d(W_t^2) = 2 W_t\,dW_t + dt$")
ax.legend()
plt.tight_layout(); plt.show()

max_err = np.max(np.abs(lhs_cumulative - rhs_cumulative))
print(f"Max pointwise error over [0, 1]: {max_err:.5f}  (goes to 0 as N grows)")
"""))

# --- Section 6: SDEs + Euler-Maruyama ---

CELLS.append(md(r"""## 6. Stochastic differential equations

A **stochastic differential equation (SDE)** combines a deterministic drift with a Brownian diffusion:

$$dX_t \;=\; \mu(t, X_t)\,dt \;+\; \sigma(t, X_t)\,dW_t, \qquad X_0 = x_0.$$

Under Lipschitz conditions on $\mu$ and $\sigma$ a unique strong solution exists. The simplest numerical scheme is Euler–Maruyama:
"""))

CELLS.append(md(three_layer(
    math_src=r"""**Euler–Maruyama.** Given a grid $0 = t_0 < t_1 < \cdots < t_N = T$ with step $\Delta t = t_{k+1} - t_k$, set
$$X_{t_{k+1}} \;=\; X_{t_k} + \mu(t_k, X_{t_k})\,\Delta t + \sigma(t_k, X_{t_k})\,\sqrt{\Delta t}\,Z_k,$$
where $Z_k \sim \mathcal{N}(0, 1)$ i.i.d.

Weak order of convergence: 1. Strong order: $1/2$.""",
    plain_src=r"""The simplest discretisation. Each step:

1. Advance by the **drift** $\mu \cdot \Delta t$.
2. Add a **diffusion kick** of standard deviation $\sigma \sqrt{\Delta t}$.

Note the square root in the diffusion — that's the Itô table again. Use $\Delta t$ (not $\sqrt{\Delta t}$) for the drift, $\sqrt{\Delta t}$ (not $\Delta t$) for the diffusion.

For the repo's BSDE solvers, the forward SDE is always discretised this way.""",
    code_src=r"""Every forward-pass in the repo
is Euler–Maruyama. In
[solver.py: ContXiongLOBModel](../solver.py),
each time step does:

<pre>x_new = x + drift(t, x)*dt
          + diffusion(t,x)*sqrt(dt)*Z
y_new = y - f_tf(t, x, y, z)*dt
          + z * sqrt(dt)*Z
z_new = subnet[t](x_new) / dim</pre>

The y-equation is the BSDE step
(Module 2); the x-equation is
exactly the EM scheme above."""
)))

CELLS.append(code(r"""def euler_maruyama(mu, sigma, X0, T, N, n_paths=1, seed=None):
    '''Generic 1-D Euler-Maruyama.
    mu, sigma: callables (t, X) -> scalar.
    Returns t (N+1,) and X (N+1, n_paths).'''
    r = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    X = np.empty((N + 1, n_paths))
    X[0, :] = X0
    for k in range(N):
        Z = r.standard_normal(n_paths)
        X[k + 1, :] = (
            X[k, :]
            + mu(t[k], X[k, :]) * dt
            + sigma(t[k], X[k, :]) * np.sqrt(dt) * Z
        )
    return t, X

# Geometric Brownian motion: dS = r S dt + sigma S dW
# Closed form: S_t = S_0 * exp((r - sigma^2/2) t + sigma W_t)
r_rate, vol, S0, T, N = 0.05, 0.2, 100.0, 1.0, 500

t, S = euler_maruyama(
    mu=lambda t, x: r_rate * x,
    sigma=lambda t, x: vol * x,
    X0=S0, T=T, N=N, n_paths=10, seed=7,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
ax1.plot(t, S, lw=1.1)
ax1.set_title(r"10 GBM paths: $dS = 0.05\,S\,dt + 0.2\,S\,dW$")
ax1.set_xlabel("t"); ax1.set_ylabel(r"$S_t$")

# Compare terminal distribution to the analytic log-normal
_, S_many = euler_maruyama(
    mu=lambda t, x: r_rate * x,
    sigma=lambda t, x: vol * x,
    X0=S0, T=T, N=N, n_paths=10_000, seed=8,
)
ax2.hist(S_many[-1, :], bins=80, density=True, alpha=0.7, edgecolor="white")
# Analytic log-normal density
s_grid = np.linspace(50, 200, 400)
m = np.log(S0) + (r_rate - 0.5 * vol**2) * T
s_sd = vol * np.sqrt(T)
analytic = np.exp(-(np.log(s_grid) - m)**2 / (2 * s_sd**2)) / (s_grid * s_sd * np.sqrt(2 * np.pi))
ax2.plot(s_grid, analytic, lw=2, color="crimson", label="log-normal")
ax2.set_xlabel(r"$S_T$"); ax2.set_title(r"$S_T$ distribution (EM vs analytic)")
ax2.legend()
plt.tight_layout(); plt.show()
"""))

# --- Section 7: Codebase connections ---

CELLS.append(md(r"""## 7. First encounter with the codebase

Everything we've built so far — Brownian motion, the Itô table, SDEs, Euler–Maruyama — shows up on *one page* of the repo. Look at the forward SDEs in [`equations/contxiong_lob.py`](../equations/contxiong_lob.py):
"""))

CELLS.append(md(three_layer(
    math_src=r"""**Price process (driftless):**
$$dS_t = \sigma_S\,dW_t^S, \qquad S_0 \text{ given}.$$

**Inventory process (diffusion approximation of order flow):**
$$dq_t = \mu_q(q_t, \delta_t)\,dt + \sigma_q(q_t, \delta_t)\,dW_t^q,$$

with
$$\mu_q = \lambda_b f_b(\delta_b) - \lambda_a f_a(\delta_a),$$
$$\sigma_q = \sqrt{\lambda_b f_b(\delta_b) + \lambda_a f_a(\delta_a)}.$$

The processes $W^S$ and $W^q$ are independent standard BMs.""",
    plain_src=r"""Two coupled SDEs:

1. **Price** — unbiased random walk scaled by volatility $\sigma_S$. No drift, because we're in the (normalised) reference frame of the mid-price.
2. **Inventory** — drift from the net order flow (buys minus sells), diffusion approximating the variance of a compound Poisson process.

The $\sqrt{\cdot}$ in $\sigma_q$ is the moment-matched diffusion: if order arrivals were truly Poisson with intensity $\lambda$, the variance of inventory over $dt$ would be $\lambda\,dt$, so the diffusion coefficient is $\sqrt{\lambda}$.

Module 3 will derive this rigorously as a BSDEJ reduction.""",
    code_src=r"""From
[equations/contxiong_lob.py:9-19](../equations/contxiong_lob.py)
(simplified):

<pre>def sample(self, num_sample):
  dW = sqrt(dt) * randn(n, 2)
  # price: dS = sigma_S * dW_S
  dS = sigma_s * dW[..., 0]
  # inventory: dq = mu_q dt
  #             + sigma_q sqrt(dt) Z
  fa = lambda_a * exp(-alpha*da)
  fb = lambda_b * exp(-alpha*db)
  mu_q = lambda_b*fb - lambda_a*fa
  sig_q = sqrt(lambda_b*fb
             + lambda_a*fa)
  dq = mu_q*dt + sig_q*dW[..., 1]</pre>

Every line here is an instance of
the machinery from this module."""
)))

CELLS.append(md(callout(
    "codebase",
    "Where to read next",
    r"""Open [equations/contxiong_lob.py](../equations/contxiong_lob.py) lines 8–130. You now have enough vocabulary to understand the `sample` method line-by-line: the forward SDE loop is identical to our `euler_maruyama` above, just with richer drift/diffusion. What you **won't** understand yet is the `f_tf` method (the BSDE generator) or the `g_tf` method (terminal condition). Those are Module 2's job."""
)))

# --- Exercises ---

CELLS.append(md(r"""## 8. Exercises

Try each before revealing the solution. They should take 10-30 minutes each.
"""))

CELLS.append(md(exercise(
    1,
    r"""**Quadratic variation at different mesh sizes.** Modify the quadratic variation simulation above to plot, on the same axes, the *distribution* (not just mean) of $\sum_k (\Delta W_k)^2$ over $n_{\text{runs}} = 1000$ replications for each of $N \in \{50, 500, 5000\}$. Verify both that the mean is close to $T = 1$ at all mesh sizes and that the variance shrinks as $N$ grows. What is the rate of shrinkage?""",
    r"""The variance of the quadratic variation estimator is $\mathrm{Var}\bigl[\sum_k (\Delta W_k)^2\bigr] = 2 T^2 / N$, so the standard deviation scales like $1/\sqrt{N}$. A log-log plot of std vs $N$ should show slope $-1/2$. The ratio of successive stds when $N$ quadruples should be roughly $2$. Runnable solution code is in the next cell (collapsed — click the arrow to reveal)."""
)))

CELLS.append(code_hidden(r"""# Solution — Exercise 1
fig, ax = plt.subplots()
stds = []
Ns = [50, 500, 5000]
for N in Ns:
    s = [np.sum(np.diff(brownian_path(1.0, N, seed=k)[1][:, 0]) ** 2)
         for k in range(1000)]
    ax.hist(s, bins=40, alpha=0.5, density=True, label=f"N={N}")
    stds.append(np.std(s))
ax.axvline(1.0, color="k", ls="--", label="t = 1")
ax.set_xlabel(r"$\sum_k (\Delta W_k)^2$")
ax.set_title("QV distribution concentrates around t = 1 as N grows")
ax.legend()
plt.tight_layout(); plt.show()

print("Std at each N:", [f"{s:.4f}" for s in stds])
print("Std ratios (expect ~sqrt(10) ~ 3.16 between successive):",
      [f"{stds[i] / stds[i+1]:.2f}" for i in range(len(stds) - 1)])
"""))

CELLS.append(md(exercise(
    2,
    r"""**Itô's formula by hand.** Using $df(t, W_t) = \partial_t f\,dt + \partial_w f\,dW_t + \tfrac{1}{2}\partial_{ww} f\,dt$, derive the SDE satisfied by

$$M_t \;=\; \exp\!\Bigl(\lambda W_t - \tfrac{1}{2}\lambda^2 t\Bigr), \qquad \lambda \in \mathbb{R}.$$

Show that $M_t$ is a martingale (its drift vanishes). Then verify numerically that $\mathbb{E}[M_t] = 1$ for all $t$ using 50,000 paths.""",
    r"""Let $f(t, w) = \exp(\lambda w - \tfrac{1}{2}\lambda^2 t)$. Then
$$\partial_t f = -\tfrac{1}{2}\lambda^2 f, \quad \partial_w f = \lambda f, \quad \partial_{ww} f = \lambda^2 f.$$
Substituting,
$$dM_t = \bigl(-\tfrac{1}{2}\lambda^2 f + \tfrac{1}{2}\lambda^2 f\bigr)dt + \lambda f\,dW_t = \lambda M_t\,dW_t.$$
The drift term is identically zero — so $M_t$ is a martingale. This is the **exponential martingale** that underpins Girsanov's theorem. Runnable numerical check in the next cell."""
)))

CELLS.append(code_hidden(r"""# Solution — Exercise 2
lam, T, N = 0.6, 1.0, 500
_, W = brownian_path(T, N, d=50_000, seed=11)
M_T = np.exp(lam * W[-1, :] - 0.5 * lam**2 * T)
print(f"E[M_T] = {M_T.mean():.4f}  (should be ~1.0)")
print(f"Std of M_T = {M_T.std():.4f}")

# Check that E[M_t] = 1 at every time slice, not just T
t_grid = np.linspace(0, T, N + 1)
M_full = np.exp(lam * W - 0.5 * lam**2 * t_grid[:, None])
fig, ax = plt.subplots()
ax.plot(t_grid, M_full.mean(axis=1), lw=1.5, label=r"Empirical $E[M_t]$")
ax.axhline(1.0, color="crimson", ls="--", label="1")
ax.set_xlabel("t"); ax.set_title("Exponential martingale: mean stays at 1")
ax.legend(); plt.tight_layout(); plt.show()
"""))

CELLS.append(md(exercise(
    3,
    r"""**Euler–Maruyama for the Cont–Xiong price SDE.** The repo's price SDE is $dS_t = \sigma_S\,dW_t^S$ with $\sigma_S = 0.3$ and $S_0 = 100$, $T = 1$. Simulate 1000 paths with $N = 200$ steps. Plot the distribution of $S_T$ and compare to the analytic density $\mathcal{N}(S_0, \sigma_S^2 T)$. Why is this *not* geometric Brownian motion?""",
    r"""The repo uses *arithmetic* BM for the price, so $S_t = S_0 + \sigma_S W_t \sim \mathcal{N}(S_0, \sigma_S^2 t)$. This differs from GBM (used in Black-Scholes) because the Cont–Xiong model operates on the mid-price in a short-horizon regime where the log-transform adds little. Negative prices are technically possible but astronomically unlikely over $T \leq 1$ with $\sigma_S = 0.3$, $S_0 = 100$. The mid-price can also genuinely be negative (e.g., negative rates, energy markets) so allowing it is a feature. Runnable check in the next cell."""
)))

CELLS.append(code_hidden(r"""# Solution — Exercise 3
sigma_S, S0, T, N = 0.3, 100.0, 1.0, 200
t, S = euler_maruyama(
    mu=lambda t, x: 0.0,
    sigma=lambda t, x: sigma_S,
    X0=S0, T=T, N=N, n_paths=1000, seed=9,
)
fig, ax = plt.subplots()
ax.hist(S[-1, :], bins=50, density=True, alpha=0.7, edgecolor="white")
xs = np.linspace(S[-1].min(), S[-1].max(), 200)
analytic = np.exp(-(xs - S0)**2 / (2 * sigma_S**2 * T)) / (sigma_S * np.sqrt(2 * np.pi * T))
ax.plot(xs, analytic, lw=2, color="crimson", label=r"$\mathcal{N}(S_0, \sigma_S^2 T)$")
ax.set_xlabel(r"$S_T$"); ax.set_title("Arithmetic BM price — EM vs analytic")
ax.legend(); plt.tight_layout(); plt.show()

print(f"Empirical mean S_T = {S[-1].mean():.3f}  (analytic: {S0})")
print(f"Empirical std  S_T = {S[-1].std():.3f}  (analytic: {sigma_S * np.sqrt(T):.3f})")
"""))

CELLS.append(md(exercise(
    4,
    r"""**Reading the codebase.** Open [`equations/contxiong_lob.py`](../equations/contxiong_lob.py) and locate the `sample` method. Identify (a) the line implementing the price increment, (b) the lines implementing the inventory drift and diffusion, and (c) the computation of the moment-matched diffusion $\sigma_q = \sqrt{\lambda_a f_a + \lambda_b f_b}$. Then answer: why is the diffusion evaluated at the *current* $(t_k, X_{t_k})$ rather than the next step?""",
    r"""(a) The price increment is the term proportional to the first column of `dW`, multiplied by `sigma_s`.
(b) The inventory drift is `lambda_b * f_b - lambda_a * f_a`, and the diffusion is the `np.sqrt(...)` term multiplying the second `dW` column.
(c) Because the Itô convention requires evaluating the integrand at the left endpoint — we use information from $t_k$, not $t_{k+1}$. Using $(t_{k+1}, X_{t_{k+1}})$ would (i) violate adaptedness (we'd be peeking at the future) and (ii) produce the Stratonovich integral, which has different calculus rules (no Itô correction)."""
)))

# --- Wrap-up ---

CELLS.append(md(r"""## 9. Takeaways

| Concept | Why it matters for the repo |
|---------|----------------------------|
| $(dW)^2 = dt$ | Every Euler step scales diffusion by $\sqrt{dt}$ |
| Left-endpoint Itô rule | BSDE generators evaluate at $(t_k, X_{t_k})$ |
| Itô's formula | The HJB derivation in Module 5 is one Itô expansion |
| SDE existence + EM scheme | Every forward pass in `solver.py` is an EM discretisation |
| Non-differentiability | BSDE control $Z_t$ is the *derivative* of the value function along the diffusion — a delicate object |

## What's next

[Module 2](02_bsdes.ipynb) introduces **Backward Stochastic Differential Equations** — the object the whole repo exists to solve. We'll derive the Y/Z decomposition, state the martingale representation theorem, and see how the BSDE generator $f$ becomes a running cost in stochastic control.

<div class="module-nav">
<span></span>
<span><strong>Next →</strong> Module 2: BSDEs</span>
</div>
"""))


# ---------------------------------------------------------------------------
# Assemble and write the notebook
# ---------------------------------------------------------------------------

build("01_brownian_motion.ipynb", CELLS)

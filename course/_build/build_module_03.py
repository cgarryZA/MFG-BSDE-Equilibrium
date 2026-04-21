"""Build Module 3: BSDEs with jumps."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import md, code, code_hidden, three_layer, callout, exercise, build, STYLE_CSS


CELLS = []
CELLS.append(md(STYLE_CSS))

CELLS.append(md(r"""# Module 3 — BSDEs with Jumps (BSDEJ)

> *The repo's inventory dynamics are driven by **discrete order arrivals** — a compound Poisson process. The diffusion approximation of Module 2 was a simplification. Now we do it properly.*

## Learning goals

1. **Define a Poisson random measure** and write the compensated martingale form.
2. **State the Itô formula for jump-diffusions** and identify the extra jump term.
3. **Write the BSDEJ canonical form** $dY_t = -f(t, X_t, Y_t, Z_t, U_t(\cdot))\,dt + Z_t\,dW_t + \int_E U_t(e)\,\tilde N(de, dt)$.
4. **Prove (in the simple symmetric case) that the diffusion approximation in [contxiong_lob.py:10](../equations/contxiong_lob.py) is moment-matched** to the true compound Poisson dynamics.
5. **Read [solver_cx_bsdej.py](../solver_cx_bsdej.py)** and identify where the jump integrand $U$ lives.

## Prereqs

Modules 1-2.
"""))

CELLS.append(md(r"""## 1. Why jumps?

A limit order book is a discrete, event-driven object: at random times, a market order arrives and consumes liquidity at your quote. This is **not** a diffusion — it's a counting process. The repo's *main* model uses a diffusion approximation (for numerical tractability), but the underlying reality is

$$dN_t^{a} \sim \text{Poisson}(\lambda_a f_a(\delta_a)\,dt), \qquad dN_t^{b} \sim \text{Poisson}(\lambda_b f_b(\delta_b)\,dt),$$

and the inventory evolves as
$$dq_t = -dN_t^{a} + dN_t^{b}.$$

Two reasons to treat jumps seriously:

1. **Correctness at low $\lambda$.** Diffusion approximation is accurate when arrivals are frequent. In sparse regimes, the Gaussian tail misrepresents the Poisson tail.
2. **Theoretical foundations.** The repo has a branch [solver_cx_bsdej.py](../solver_cx_bsdej.py) that solves the *true* jump BSDE — useful as a reference benchmark.
"""))

CELLS.append(md("## 2. Poisson random measures"))

CELLS.append(md(three_layer(
    math_src=r"""A **Poisson random measure** $N(dt, de)$ on $[0,T] \times E$ with intensity $\nu(de)\,dt$ is a random counting measure such that for disjoint Borel sets $A \subset E$ and time intervals $[s, t]$, $N([s, t], A)$ is Poisson with mean $(t - s)\,\nu(A)$.

The **compensated** random measure
$$\tilde N(dt, de) := N(dt, de) - \nu(de)\,dt$$
is a martingale: $\mathbb{E}[\tilde N([s, t], A)] = 0$.

For $\phi : E \to \mathbb{R}$,
$$\int_0^t \int_E \phi(e)\,\tilde N(ds, de)$$
is a martingale provided $\int_E \phi^2\,d\nu < \infty$.""",
    plain_src=r"""Think of $N$ as a rain of random dots on the $(t, e)$-plane, with density $\nu(de)\,dt$. The random variable $\int \phi\,dN$ adds up $\phi(e)$ over every dot that falls.

**Compensation** subtracts off the expected rate: $\tilde N = N - (\text{expected}N)$. What's left is a mean-zero martingale, ready to plug into Itô calculus.

For market-making, the "marks" $e$ are the size of each incoming order and $\nu(de) = \lambda(de)\,dt$ is the arrival intensity. In the simplest version, each order has unit size — so $E = \{+1, -1\}$ and $\lambda$ depends on side and quote.""",
    code_src=r"""A Poisson jump at rate
$\lambda$ is simulated as

<pre>dN = rng.poisson(lam * dt,
                 size=(N_steps,))</pre>

For the inventory process:

<pre>dq = -dN_a + dN_b</pre>

The compensated form subtracts
expected flow:

<pre>tilde_dN = dN - lam * dt</pre>

[solver_cx_bsdej.py](../solver_cx_bsdej.py)
uses this pattern."""
)))

CELLS.append(md("## 3. Itô formula with jumps"))

CELLS.append(md(callout(
    "theorem",
    "Itô formula for jump-diffusions",
    r"""Let $X_t$ solve
$$dX_t = \mu_t\,dt + \sigma_t\,dW_t + \int_E \gamma_t(e)\,\tilde N(dt, de),$$
and $f \in C^{1,2}([0, T] \times \mathbb{R})$. Then
$$df(t, X_t) = \Bigl(\partial_t f + \mu_t \partial_x f + \tfrac{1}{2}\sigma_t^2 \partial_{xx} f\Bigr) dt + \sigma_t \partial_x f\,dW_t$$
$$+ \int_E \bigl(f(t, X_{t-} + \gamma_t(e)) - f(t, X_{t-})\bigr)\,\tilde N(dt, de)$$
$$+ \int_E \bigl(f(t, X_{t-} + \gamma_t(e)) - f(t, X_{t-}) - \gamma_t(e)\,\partial_x f\bigr)\nu(de)\,dt.$$

The last line is the **compensator drift** — an extra deterministic term because Itô expansion in the jump direction produces non-trivial second-order terms that don't collapse."""
)))

CELLS.append(md(callout(
    "insight",
    "Two separate Taylor limits",
    r"""The continuous part of Itô's formula uses $(dW)^2 = dt$ to produce the $\tfrac{1}{2}f''$ correction. The jump part does **not** Taylor-expand — jumps are $O(1)$ in size, not infinitesimal. Instead you evaluate $f$ at the post-jump state and take a *difference*. The compensator just subtracts the expected value so the martingale part is clean."""
)))

CELLS.append(md("## 4. BSDE with jumps"))

CELLS.append(md(three_layer(
    math_src=r"""A **BSDEJ** is a triple $(Y, Z, U)$ with $Y$ scalar, $Z$ Brownian adjoint, $U(e)$ jump adjoint:

$$Y_t = g(X_T) + \int_t^T f(s, X_{s-}, Y_{s-}, Z_s, U_s(\cdot))\,ds$$
$$- \int_t^T Z_s\,dW_s - \int_t^T \int_E U_s(e)\,\tilde N(ds, de).$$

Existence and uniqueness (Tang–Li, Barles–Buckdahn–Pardoux) hold under Lipschitz assumptions on $f$ in $(y, z, u)$.""",
    plain_src=r"""A BSDE with three unknowns instead of two:

- $Y_t$ — the value (same as before).
- $Z_t$ — the Brownian delta (same as before).
- $U_t(e)$ — the *jump delta*. Tells you how much $Y$ changes in response to a jump of size $e$.

If $X$ has no jumps, $U \equiv 0$ and BSDEJ reduces to BSDE.

Pardoux–Peng extends naturally; the only new ingredient is that the martingale representation theorem is replaced by the **Jacod–Yor representation**, which states that every square-integrable martingale in a Brownian + Poisson filtration is a stochastic integral against $dW$ and $d\tilde N$.""",
    code_src=r"""In
[solver_cx_bsdej.py](../solver_cx_bsdej.py),
the BSDE loop includes
jump adjoints:

<pre>U_sample = subnet_U[t](x)
dN = poisson(lam * dt)

y = y - f(t,x,y,z,U)*dt
      + z * sqrt(dt)*Z
      + U * (dN - lam*dt)</pre>

The `U` network learns the
jump-adjoint function
$e \mapsto U_t(e)$."""
)))

CELLS.append(md(r"""## 5. The diffusion approximation — where the main repo lives

The **main** [contxiong_lob.py](../equations/contxiong_lob.py) model does **not** use BSDEJ — it uses a diffusion approximation of the compound Poisson inventory process. Let's derive it.
"""))

CELLS.append(md(three_layer(
    math_src=r"""Let $N_t^a, N_t^b$ be independent Poisson processes with intensities $\lambda_a f_a, \lambda_b f_b$. Inventory $q_t$ satisfies
$$dq_t = -dN_t^a + dN_t^b.$$

- $\mathbb{E}[dq_t] = (\lambda_b f_b - \lambda_a f_a)\,dt = \mu_q\,dt$.
- $\mathrm{Var}(dq_t) = (\lambda_a f_a + \lambda_b f_b)\,dt = \sigma_q^2\,dt$.

Replace the jump dynamics by a diffusion with the same first two moments:

$$dq_t \approx \mu_q\,dt + \sigma_q\,dW_t^q, \quad \sigma_q = \sqrt{\lambda_a f_a + \lambda_b f_b}.$$""",
    plain_src=r"""**Moment matching.** A compound Poisson process with unit jumps has:
- Drift = (upward rate) − (downward rate).
- Variance = (sum of rates).

(Because $\mathrm{Var}(N_t) = \lambda t$ for a Poisson process.)

The diffusion with the same drift and variance (to first order in $dt$) is the one above. It's a **Gaussian approximation** of the jump process — accurate when rates are high (many small jumps approximate Brownian motion, which is exactly Donsker's theorem from Module 1).

This is why the Cont–Xiong paper gets away with a diffusion: at the high-frequency regime it targets, the approximation error is tiny.""",
    code_src=r"""From
[contxiong_lob.py:9-19](../equations/contxiong_lob.py):

<pre>mu_q = lambda_b*f_b
       - lambda_a*f_a
sigma_q = sqrt(lambda_b*f_b
             + lambda_a*f_a)
dq = mu_q*dt
   + sigma_q*sqrt(dt)*Z</pre>

Exactly the moment-matched
diffusion. The jumps have been
smoothed into Gaussian
innovations — a clean
approximation that keeps the
model solvable by plain Deep
BSDE instead of Deep BSDEJ."""
)))

CELLS.append(md(r"""## 6. Numerical comparison: compound Poisson vs diffusion approximation

Let's verify the approximation holds when jump rates are high.
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

def simulate_cp(lambda_a, lambda_b, T, N, n_paths, seed=0):
    '''Exact compound Poisson inventory: dq = -dN_a + dN_b.'''
    rng = np.random.default_rng(seed)
    dt = T / N
    dN_a = rng.poisson(lambda_a * dt, size=(N, n_paths))
    dN_b = rng.poisson(lambda_b * dt, size=(N, n_paths))
    q = np.zeros((N + 1, n_paths))
    q[1:] = np.cumsum(dN_b - dN_a, axis=0)
    return q

def simulate_diffusion(lambda_a, lambda_b, T, N, n_paths, seed=0):
    '''Moment-matched Gaussian diffusion.'''
    rng = np.random.default_rng(seed)
    dt = T / N
    mu = lambda_b - lambda_a
    sigma = np.sqrt(lambda_b + lambda_a)
    dW = rng.standard_normal((N, n_paths)) * np.sqrt(dt)
    q = np.zeros((N + 1, n_paths))
    q[1:] = np.cumsum(mu * dt + sigma * dW, axis=0)
    return q

# High rate regime
T, N, n_paths = 1.0, 100, 20_000
lambda_a, lambda_b = 5.0, 5.0  # many arrivals

q_cp = simulate_cp(lambda_a, lambda_b, T, N, n_paths)
q_df = simulate_diffusion(lambda_a, lambda_b, T, N, n_paths)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
ax1.hist(q_cp[-1], bins=50, density=True, alpha=0.6, label="compound Poisson")
ax1.hist(q_df[-1], bins=50, density=True, alpha=0.6, label="diffusion approx")
ax1.set_xlabel(r"$q_T$"); ax1.set_title(f"$\\lambda = 5$ (high): approximation OK")
ax1.legend()

# Low rate regime — approximation fails
lambda_a, lambda_b = 0.5, 0.5
q_cp2 = simulate_cp(lambda_a, lambda_b, T, N, n_paths)
q_df2 = simulate_diffusion(lambda_a, lambda_b, T, N, n_paths)

ax2.hist(q_cp2[-1], bins=30, density=True, alpha=0.6, label="compound Poisson")
ax2.hist(q_df2[-1], bins=30, density=True, alpha=0.6, label="diffusion approx")
ax2.set_xlabel(r"$q_T$"); ax2.set_title(r"$\lambda = 0.5$ (low): approx fails in tails")
ax2.legend()
plt.tight_layout(); plt.show()

print(f"High rate:  Poisson mean={q_cp[-1].mean():.3f} std={q_cp[-1].std():.3f}")
print(f"            Diffusion mean={q_df[-1].mean():.3f} std={q_df[-1].std():.3f}")
print(f"Low rate:   Poisson mean={q_cp2[-1].mean():.3f} std={q_cp2[-1].std():.3f}")
print(f"            Diffusion mean={q_df2[-1].mean():.3f} std={q_df2[-1].std():.3f}")
"""))

CELLS.append(md(callout(
    "insight",
    "When does the approximation work?",
    r"""As $\lambda T \to \infty$ at fixed $\lambda T$ variance, the compound Poisson process converges to Brownian motion (a direct consequence of the CLT). For the repo's default $\lambda_a = \lambda_b = 1.0$ and $T = 1$, we're in the "OK-ish but not perfect" regime. [solver_cx_bsdej.py](../solver_cx_bsdej.py) exists partly to quantify the gap."""
)))

CELLS.append(md(r"""## 7. The BSDEJ generator in the repo

When we *do* keep the jumps, the generator picks up a third argument $U$ describing the jump-adjoint:
"""))

CELLS.append(md(three_layer(
    math_src=r"""With jumps on both sides (ask fills at size $-1$, bid fills at size $+1$) and execution probabilities $f_a, f_b$ depending on the controls $\delta_a, \delta_b$,

$$f(t, x, y, z, U_{-1}, U_{+1}) = -r y - \phi q^2$$
$$\;\;+ \lambda_a f_a(\delta_a)\,\bigl(\delta_a - U_{-1}\bigr) + \lambda_b f_b(\delta_b)\,\bigl(\delta_b - U_{+1}\bigr).$$

The "$-U$" term compensates the expected jump of $Y$ when the corresponding fill happens.""",
    plain_src=r"""Two changes from the diffusion generator:

1. $Z$ drops out of the quote FOC — replaced by $U_{+1} - U_{-1}$ (the *difference* in jump-adjoints), which plays the same role as $Z^q / \sigma_q$.
2. A "$-U$" term appears next to each profit term to compensate the expected jump change of $Y$.

Algebraically messier than the diffusion version, but conceptually the same: the BSDE accumulates expected profit while tracking how $Y$ responds to every source of randomness (now: diffusion **and** jumps).""",
    code_src=r"""In
[solver_cx_bsdej.py:60-95](../solver_cx_bsdej.py):

<pre>def f_tf(self, t, x, y, z,
         u_minus, u_plus):
    da = 1/alpha
       + (u_plus - u_minus)
    db = 1/alpha
       - (u_plus - u_minus)
    fa = exp(-alpha*da)
    fb = exp(-alpha*db)
    return (-r*y - phi*q**2
      + lam_a*fa*(da - u_minus)
      + lam_b*fb*(db - u_plus))</pre>"""
)))

CELLS.append(md(r"""## 8. Exercises"""))

CELLS.append(md(exercise(
    1,
    r"""**Verify moment matching.** For a compound Poisson process with arrival rate $\lambda$ and unit jumps, show that $\mathbb{E}[N_t] = \lambda t$ and $\mathrm{Var}(N_t) = \lambda t$. Then verify numerically for the two-sided inventory process at $\lambda_a = \lambda_b = 2.0$, $T = 1$. Does the empirical variance of $q_T$ match $\lambda_a + \lambda_b$?""",
    r"""$N_t$ is Poisson with parameter $\lambda t$, which has mean and variance both equal to $\lambda t$. For $q_t = -N_t^a + N_t^b$ (difference of *independent* Poissons), $\mathrm{Var}(q_t) = \lambda_a t + \lambda_b t$. Run the next cell to check empirically."""
)))

CELLS.append(code_hidden(r"""# Solution — Exercise 1
T, N, n_paths = 1.0, 200, 50_000
la, lb = 2.0, 2.0
q = simulate_cp(la, lb, T, N, n_paths)
print(f"Mean(q_T)   = {q[-1].mean():.4f}   (theory: {(lb - la)*T})")
print(f"Var(q_T)    = {q[-1].var():.4f}    (theory: {(la + lb)*T})")
"""))

CELLS.append(md(exercise(
    2,
    r"""**Itô on a jump-diffusion.** Let $q_t$ be the pure compound Poisson process above (no diffusion part). Compute $d(q_t^2)$ using the jump Itô formula. Identify the "quadratic variation" of this process — how does it differ from the Brownian case?""",
    r"""With no continuous part, Itô's formula reduces to the jump term:
$$d(q_t^2) = (q_{t-} + \Delta q)^2 - q_{t-}^2 \,dN = \bigl(2 q_{t-} \Delta q + (\Delta q)^2\bigr)\,dN.$$
Summing over jumps: $q_t^2 = 2\int_0^t q_{s-}\,dq_s + \sum_{s \leq t} (\Delta q_s)^2$. The last sum is the **quadratic variation of jumps**, which equals $N_t^a + N_t^b$ (since each jump is unit size). Unlike Brownian QV (which is deterministic $= t$), jump QV is itself a counting process."""
)))

CELLS.append(md(exercise(
    3,
    r"""**Run the repo's BSDEJ solver.** Open [`solver_cx_bsdej.py`](../solver_cx_bsdej.py) and identify (a) the `sample` method, (b) where jumps $dN_a, dN_b$ are drawn, (c) the `U`-subnet in the forward loop. How does the BSDEJ differ structurally from the diffusion BSDE in [`solver.py`](../solver.py)?""",
    r"""Structurally: there is an extra subnet per time step whose output represents $U_t(e)$ for the two possible jump sizes (since $E = \{-1, +1\}$, it's a 2-vector). The forward loop draws Poisson jumps directly (not Gaussian increments) and uses the compensated form $dN - \lambda\,dt$ in the backward update. Everything else — terminal condition, loss — is unchanged."""
)))

CELLS.append(md(r"""## 9. Takeaways

| Concept | Where it appears |
|---------|-----------------|
| Poisson random measure | Order arrivals in the true LOB model |
| Compensated $\tilde N = N - \nu$ | Martingale form inside BSDEJ |
| BSDEJ triple $(Y, Z, U)$ | Extra jump-adjoint process |
| Moment-matched diffusion | [contxiong_lob.py:10](../equations/contxiong_lob.py) — approximation used by main solver |
| BSDEJ solver | [solver_cx_bsdej.py](../solver_cx_bsdej.py) — finite-horizon with jumps |

## What's next

[Module 4](04_mckean_vlasov.ipynb) tackles the **population** dimension: what if the generator depends on the *law* of the state process, not just the current state? This is where the competitive factor $h(\mu_t)$ enters the repo.

<div class="module-nav">
<a href="02_bsdes.ipynb"><strong>← Prev</strong> Module 2: BSDEs</a>
<a href="04_mckean_vlasov.ipynb"><strong>Next →</strong> Module 4: McKean–Vlasov</a>
</div>
"""))


build("03_bsdes_with_jumps.ipynb", CELLS)

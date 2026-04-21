"""Build Module 5: Cont-Xiong LOB model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import md, code, code_hidden, three_layer, callout, exercise, build, STYLE_CSS


CELLS = []
CELLS.append(md(STYLE_CSS))

CELLS.append(md(r"""# Module 5 — The Cont–Xiong LOB Model

> *This is the model the entire repo is built around. We derive the value function, the HJB equation, and the explicit optimal-quote formulae used throughout `solver.py` and `equations/contxiong_lob*.py`.*

## Learning goals

1. **Describe the LOB microstructure** — mid-price, bid/ask, half-spread, fill intensity.
2. **Derive the value function HJB** for a market maker with quadratic inventory penalty.
3. **Solve the first-order condition** for the optimal quotes:
   $$\delta_a^* = \frac{1}{\alpha} + \frac{Z^q}{\sigma_q}, \quad \delta_b^* = \frac{1}{\alpha} - \frac{Z^q}{\sigma_q}.$$
4. **Interpret $Z^q$** as the marginal value of inventory: "one more unit of $q$ changes my future PnL by $Z^q / \sigma_q$ per unit $q$-shock."
5. **Extend to the MV coupled case** with competitive factor $h(\mu_t)$.
6. **Map every line of [`equations/contxiong_lob.py`](../equations/contxiong_lob.py)** to the corresponding piece of mathematics.

## Prereqs

Modules 1-4.
"""))

CELLS.append(md(r"""## 1. The microstructure setup

A **limit order book** (LOB) is a queue of outstanding buy and sell orders. The **best bid** is the highest price someone will pay; the **best ask** is the lowest price someone will accept. The difference is the **bid–ask spread**, and its midpoint is the **mid-price** $S_t$.

A **market maker** quotes both sides simultaneously:
- **Ask**: sell at $S_t + \delta_a$ (some distance above the mid).
- **Bid**: buy at $S_t - \delta_b$ (some distance below).

When a market order arrives, it trades against whichever side is closer. The further from the mid (the wider the spread) you quote, the **less likely** you are to be hit — but the more **profit per trade** you collect.

This is the core trade-off: fill probability vs margin. The model formalises this as a stochastic-optimal-control problem.
"""))

CELLS.append(md(three_layer(
    math_src=r"""**Execution intensity.** The probability per unit time that your ask quote is hit is
$$\lambda_a(\delta_a) = \lambda_a^0 \cdot f(\delta_a),$$
and likewise for the bid. The form used in the repo is
$$f(\delta) = \exp(-\alpha \delta).$$
$\alpha > 0$ measures how sensitive market-order flow is to your quote. Larger $\alpha$ = more aggressive competition from other market participants.

**Inventory dynamics (jump form).** With unit trade size,
$$dq_t = dN_t^b - dN_t^a,$$
where $N_t^{a,b}$ are Poisson with rates $\lambda_{a,b}(\delta_{a,b})$.""",
    plain_src=r"""The **exponential intensity** $f(\delta) = e^{-\alpha\delta}$ is a standard microstructure ansatz (Avellaneda–Stoikov, 2008). It encodes:

- If you quote at the mid ($\delta = 0$), fill rate is $\lambda_0$ — the "baseline" liquidity.
- Every unit you move away from mid cuts your fill rate by factor $e^{-\alpha}$.

This form is **log-linear**, which makes the FOC solvable in closed form (§3). If you used, say, a Hill function, you'd lose that.

**Inventory** tracks how many units of the asset you currently hold. Positive = long, negative = short. Each fill changes $q$ by $\pm 1$.""",
    code_src=r"""From
[contxiong_lob.py:115-124](../equations/contxiong_lob.py):

<pre>def _exec_prob_tf(self, delta):
    # exp(-alpha*delta)
    # clamped for stability
    return torch.exp(
        -self.alpha
        * torch.clamp(delta,
            -5/alpha, 10/alpha))</pre>

The clamp buys Lipschitz for
Pardoux–Peng (Module 2)."""
)))

CELLS.append(md(r"""## 2. The value function"""))

CELLS.append(md(three_layer(
    math_src=r"""Let $V(t, S, q)$ be the market-maker's value function at time $t$ given mid-price $S$ and inventory $q$. With terminal inventory penalty $g(q) = -\phi q^2$,

$$V(t, S, q) = \sup_{\delta_a, \delta_b} \mathbb{E}\!\left[e^{-r(T-t)} g(q_T) + \int_t^T e^{-r(s-t)}\bigl(\lambda_a f_a \delta_a + \lambda_b f_b \delta_b - \phi q_s^2\bigr)ds\right].$$

Integrand = **instantaneous expected profit**:
- $\lambda_a f_a \delta_a$ — expected ask revenue per unit time ($\lambda_a f_a$ = fill rate, $\delta_a$ = markup).
- $\lambda_b f_b \delta_b$ — expected bid revenue.
- $-\phi q^2$ — running inventory penalty.""",
    plain_src=r"""**$V(t, S, q)$** is the best expected discounted PnL from now until terminal, given you follow an optimal policy.

The supremum is taken over all admissible strategies $(\delta_a, \delta_b)$.

Under price-independence of the generator (which holds in the base Cont–Xiong model because the mid is driftless), $V$ actually **does not depend on $S$** — only on time and inventory. This is a big simplification: we only need to learn a 2-D value function (instead of 3-D).

In the BSDE formalism of Module 2: $Y_t = V(t, S_t, q_t)$, and $Z_t = \sigma \nabla V(t, S, q)$.""",
    code_src=r"""[contxiong_lob.py:302-348](../equations/contxiong_lob.py)
encodes this as the BSDE
generator:

<pre>f_tf = (-r * y          # discount
        - phi * q**2   # penalty
        + lambda_a * f_a * delta_a
        + lambda_b * f_b * delta_b)</pre>

With terminal
<pre>g_tf(T, x) = -phi * q_T**2</pre>"""
)))

CELLS.append(md(r"""## 3. HJB derivation

Apply Itô's formula to $V(t, S_t, q_t)$ along the (jump form) forward dynamics, take expectation, and sup over controls. In the diffusion-approximation regime (Module 3) we treat $q_t$ as a diffusion, which yields a cleaner PDE:
"""))

CELLS.append(md(callout(
    "theorem",
    "HJB equation (diffusion approximation)",
    r"""Under $dq = \mu_q\,dt + \sigma_q\,dW^q$ with $\mu_q, \sigma_q$ depending on the controls via the intensities,
$$\partial_t V + \tfrac{1}{2}\sigma_q^2\,\partial_{qq} V + \mu_q\,\partial_q V - r V - \phi q^2 + \sup_{\delta_a, \delta_b}\bigl\{\lambda_a f_a \delta_a + \lambda_b f_b \delta_b\bigr\} = 0,$$
$$V(T, S, q) = -\phi q^2.$$

Since $\mu_q, \sigma_q$ are functions of $(\delta_a, \delta_b)$, the Hamiltonian couples to the controls."""
)))

CELLS.append(md(three_layer(
    math_src=r"""**FOC for the ask quote.** Differentiating the Hamiltonian in $\delta_a$:

$$\frac{\partial}{\partial \delta_a}\Bigl[\lambda_a e^{-\alpha \delta_a} \delta_a + \text{coupling through } \mu_q, \sigma_q\Bigr] = 0.$$

After tedious algebra (which the paper [Cont–Xiong 2024] carries out carefully), and identifying $\partial_q V$ with $Z^q / \sigma_q$ along the forward trajectory, one obtains

$$\boxed{\delta_a^* = \frac{1}{\alpha} + \frac{Z^q}{\sigma_q}, \qquad \delta_b^* = \frac{1}{\alpha} - \frac{Z^q}{\sigma_q}.}$$""",
    plain_src=r"""**Interpretation of the FOC.**

- $\frac{1}{\alpha}$ is the classical **monopolist spread**: a greedy market maker with no inventory aversion would quote at this width. It comes from the profit-maximisation condition $\delta \cdot (1 - \alpha\delta) = 0 \Rightarrow \delta = 1/\alpha$ (ignoring the prefactor).

- $Z^q / \sigma_q$ is the **inventory shift**. It's positive when $\partial_q V < 0$ (being long hurts future value), which shifts the ask **outward** (harder to get hit) and the bid **inward** (easier to get hit) — a mechanism that pushes inventory back toward zero.

Think of it as: "the monopolist quote, plus an adjustment for how much I value *not* having this inventory."

The ask and bid formulae are symmetric: the monopolist term is unsigned, and the inventory shift appears with opposite signs.""",
    code_src=r"""From
[contxiong_lob.py:135-160](../equations/contxiong_lob.py):

<pre>p = z[..., 1:2] / sigma_q
delta_a = 1.0/alpha + p
delta_b = 1.0/alpha - p</pre>

The entire FOC fits in three
lines. **This is the heart of
the repo** — everything else
(networks, training loops,
encoders) is machinery to
compute $Z^q$ correctly."""
)))

CELLS.append(md(r"""## 4. Visualising the optimal quotes

Let's plot the optimal quote surface as a function of inventory, assuming an ansatz $V(t, q) = -A(t) q^2 + B(t)$ (which is exactly right in the infinite-horizon limit). Then $\partial_q V = -2 A(t) q$, and the ask/bid shifts are linear in $q$.
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

# Repo-default parameters
alpha, lam = 1.5, 1.0
phi = 0.01
A = 0.3  # a plausible 'value-function curvature'

q_grid = np.linspace(-5, 5, 200)

# Sigma_q at q = 0 (equilibrium diffusion coefficient)
# In the repo, sigma_q = sqrt(lam_a * fa + lam_b * fb).
# At symmetric equilibrium with fa = fb = exp(-1) (because alpha * delta = 1),
# sigma_q = sqrt(2 * lam * exp(-1))
sigma_q = np.sqrt(2 * lam * np.exp(-1))

# Adjoint Z^q = sigma_q * dV/dq = sigma_q * (-2 A q)
Z_q = sigma_q * (-2 * A * q_grid)

delta_a = 1 / alpha + Z_q / sigma_q
delta_b = 1 / alpha - Z_q / sigma_q

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

ax1.plot(q_grid, delta_a, lw=1.8, label=r"$\delta_a$ (ask)")
ax1.plot(q_grid, delta_b, lw=1.8, label=r"$\delta_b$ (bid)")
ax1.axhline(1/alpha, color="grey", ls="--", lw=1, label=r"monopolist $1/\alpha$")
ax1.set_xlabel("inventory q"); ax1.set_ylabel(r"optimal half-spread $\delta$")
ax1.set_title("Optimal quotes shift with inventory")
ax1.legend()

# Spread and skew
spread = delta_a + delta_b
skew = delta_a - delta_b
ax2.plot(q_grid, spread, lw=1.8, label="spread = δ_a + δ_b")
ax2.plot(q_grid, skew, lw=1.8, label="skew = δ_a − δ_b")
ax2.set_xlabel("q"); ax2.set_title("Spread (constant at 2/alpha) + skew (linear in q)")
ax2.legend(); plt.tight_layout(); plt.show()

print(f"Monopolist half-spread 1/alpha = {1/alpha:.3f}")
print(f"Equilibrium sigma_q            = {sigma_q:.3f}")
print(f"Skew slope = -2A               = {-2 * A:.3f}")
"""))

CELLS.append(md(callout(
    "insight",
    "Spread vs skew",
    r"""Notice the clean decomposition: the **spread** $\delta_a + \delta_b = 2/\alpha$ is constant in $q$ — you always quote the same total width. The **skew** $\delta_a - \delta_b$ is linear in $q$ with negative slope — when long, you quote ask tighter / bid wider (to unload); when short, the reverse. This is the inventory-driven skew, and it's what every market-maker model predicts in some form. [FINDINGS.md](../archive/FINDINGS.md) §5 reports how this skew *changes* with the competitive factor $h$ in the MV model."""
)))

CELLS.append(md(r"""## 5. The MV extension — competitive factor

Now we embed the single-agent problem in a population. The fill intensity of *your* quotes depends on what *everyone else* is doing: if everyone is quoting tight, you need to quote tighter to be competitive.

The repo models this multiplicatively:
$$f_{a}(\delta, \mu) = e^{-\alpha\delta} \cdot h(\mu), \qquad h(\mu) \in (0, 1].$$

$h = 1$ means no competition (you're alone). $h < 1$ means your fills are rarer because others are closer. $h$ depends on the population law $\mu$ through the encoder from Module 4:

$$h(\mu) = \sigma\!\left(\mathrm{MLP}(\Phi(\mu))\right) \cdot 0.99 + 0.01,$$

with $\sigma$ the logistic sigmoid and the $0.99 + 0.01$ clamp keeping $h$ in $[0.01, 1]$.
"""))

CELLS.append(md(three_layer(
    math_src=r"""**MV generator** (from [contxiong_lob_mv.py:96-124](../equations/contxiong_lob_mv.py)):
$$f(t, x, y, z, \Phi(\mu)) = -r y - \phi q^2 + \lambda_a f_a(\delta_a) h(\Phi(\mu))\,\delta_a + \lambda_b f_b(\delta_b) h(\Phi(\mu))\,\delta_b.$$

Everything else in the HJB derivation carries through. The FOC becomes:

$$\delta_a^*(q, \mu) = \frac{1}{\alpha} + \frac{Z^q}{\sigma_q(\mu)},$$

where $\sigma_q$ now depends on $h$ through the equilibrium variance.""",
    plain_src=r"""The **monopolist term** $1/\alpha$ is unchanged — it's determined purely by the exponential-intensity shape.

The **inventory shift** term scales as $Z^q / \sigma_q$. Since $\sigma_q \propto \sqrt{h}$, a lower $h$ (tighter competition) produces a *larger* inventory shift. Counter-intuitive at first, but the logic is: when you get hit less, each hit matters more, so the inventory-management component gets amplified.

**Finding in the repo**: after fixing all three failure modes (Module 6), $h$ varies by 4× across population shapes (0.442 narrow → 0.109 wide). That variation propagates to quotes measurably — $\delta_a$ at $q = 0$ drops from 0.696 (narrow population) to 0.334 (wide population), per [FINDINGS.md](../archive/FINDINGS.md) §5 and [paper.pdf](../paper.pdf).""",
    code_src=r"""The generator update in
[contxiong_lob_mv.py:109-110](../equations/contxiong_lob_mv.py):

<pre>h = comp_factor_net(law_embed)
fa = exp(-alpha*da) * h * lam_a
fb = exp(-alpha*db) * h * lam_b</pre>

`comp_factor_net` is a small
MLP: 16 → 1 with sigmoid,
clamped to $[0.01, 1]$."""
)))

CELLS.append(md(r"""## 6. Adverse selection (the 3-D extension)

A further extension (used in [contxiong_lob_adverse.py](../equations/contxiong_lob_adverse.py) and [contxiong_lob_mv_adverse.py](../equations/contxiong_lob_mv_adverse.py)) adds a third state variable — an EMA of recent price increments — that **biases execution probabilities**:

$$f_a(\delta, \text{signal}) \propto e^{-\alpha\delta} \cdot (1 + \eta \cdot \text{signal}), \qquad f_b(\delta, \text{signal}) \propto e^{-\alpha\delta} \cdot (1 - \eta \cdot \text{signal}),$$

with the factor clamped to $[0.1, 3.0]$ for stability. Intuition: if the price is drifting up, informed traders are more likely to *lift* asks (buy aggressively) — so your ask fills go up, but those fills are **adverse** (you sold just before a price rise).

This adds dimensionality (state is now 3-D: $(S, q, \text{signal})$) but no new mathematics — the HJB just has an extra $\partial^2 / \partial(\text{signal})^2$ term.
"""))

CELLS.append(md(r"""## 7. Numerical sanity check — monopolist special case

If we ignore $q$ (set $\phi = 0$ so inventory doesn't matter), the FOC collapses to $\delta^* = 1/\alpha$. Let's verify this matches a brute-force optimisation of expected per-unit-time profit.
"""))

CELLS.append(code(r"""# Expected instantaneous profit = lambda * f(delta) * delta = lambda * exp(-alpha*delta) * delta
# Optimising in delta: d/d_delta = lambda * exp(-alpha*delta) * (1 - alpha*delta) = 0 => delta = 1/alpha.

alpha, lam = 1.5, 1.0
deltas = np.linspace(0.01, 4.0, 200)
profit = lam * np.exp(-alpha * deltas) * deltas

fig, ax = plt.subplots()
ax.plot(deltas, profit, lw=1.8)
ax.axvline(1 / alpha, color="crimson", ls="--", lw=1.2, label=r"$1/\alpha$")
ax.set_xlabel(r"half-spread $\delta$"); ax.set_ylabel("expected profit rate")
ax.set_title("Monopolist FOC: peak at 1/alpha")
ax.legend(); plt.tight_layout(); plt.show()

opt_delta = deltas[np.argmax(profit)]
print(f"Empirical optimum: delta* = {opt_delta:.4f}")
print(f"Closed form:       1/alpha = {1/alpha:.4f}")
"""))

CELLS.append(md(r"""## 8. Map to the codebase

Putting everything together — the full derivation translates to the following files:

| Math concept | File:line |
|--------------|-----------|
| Forward SDE for $S, q$ | [contxiong_lob.py:8-19](../equations/contxiong_lob.py) |
| Execution intensity $f(\delta) = e^{-\alpha\delta}$ | [contxiong_lob.py:115-124](../equations/contxiong_lob.py) |
| FOC quotes $\delta_a = 1/\alpha \pm Z^q/\sigma_q$ | [contxiong_lob.py:135-160](../equations/contxiong_lob.py) |
| BSDE generator $f$ | [contxiong_lob.py:302-348](../equations/contxiong_lob.py) |
| Terminal $g = -\phi q^2$ | [contxiong_lob.py:367-370](../equations/contxiong_lob.py) |
| Quadratic penalty $\psi(q) = \phi q^2$ | [contxiong_lob.py:350-365](../equations/contxiong_lob.py) |
| MV generator with $h$ | [contxiong_lob_mv.py:96-124](../equations/contxiong_lob_mv.py) |
| Competitive factor MLP | [contxiong_lob_mv.py:23-43](../equations/contxiong_lob_mv.py) |
| Adverse selection factor | [contxiong_lob_adverse.py:48-68](../equations/contxiong_lob_adverse.py) |
| Full MV + adverse generator | [contxiong_lob_mv_adverse.py:54-80](../equations/contxiong_lob_mv_adverse.py) |

Read each file with Module 2's BSDE formalism + Module 4's MV formalism in mind. The models are thin wrappers around the generator we derived.
"""))

CELLS.append(md(r"""## 9. Exercises"""))

CELLS.append(md(exercise(
    1,
    r"""**Verify the FOC by hand.** Starting from the Hamiltonian $H(\delta_a, \delta_b) = \lambda_a e^{-\alpha \delta_a} \delta_a + \lambda_b e^{-\alpha \delta_b} \delta_b$ (ignoring the $\mu_q, \sigma_q$ dependence — just the profit terms), compute $\partial H / \partial \delta_a$ and set to zero. Solve for $\delta_a^*$. Then show how the inventory shift $Z^q/\sigma_q$ appears when you include the drift and diffusion dependence on $\delta_a$.""",
    r"""$\partial H / \partial \delta_a = \lambda_a e^{-\alpha\delta_a}(1 - \alpha \delta_a) = 0$, giving $\delta_a^* = 1/\alpha$ (monopolist).

Including the dependence through $\mu_q = \lambda_b f_b - \lambda_a f_a$ and $\sigma_q^2 = \lambda_a f_a + \lambda_b f_b$: the additional contribution from $\mu_q \partial_q V + \tfrac{1}{2}\sigma_q^2 \partial_{qq} V$ introduces terms proportional to $\partial_q V$. Identifying $\partial_q V = Z^q / \sigma_q$ (from the BSDE–PDE correspondence) and simplifying yields the $+ Z^q / \sigma_q$ correction. Full algebra is in Cont–Xiong (2024, §2) — several pages of careful bookkeeping."""
)))

CELLS.append(md(exercise(
    2,
    r"""**Spread and skew in the MV model.** With $h = 0.5$ (moderate competition) and the same $A, \phi$ as the plot above, redraw the spread and skew. Which changes, which doesn't?""",
    r"""$\sigma_q \propto \sqrt{h}$, so smaller $h$ means smaller $\sigma_q$, which means larger $Z^q / \sigma_q$ — the *skew slope* steepens. The *spread* $2/\alpha$ is unchanged because $\alpha$ doesn't depend on $h$. So in the MV model: competition affects how aggressively you mean-revert inventory, not your total width."""
)))

CELLS.append(code_hidden(r"""# Solution — Exercise 2
h_values = [1.0, 0.5, 0.2]
fig, ax = plt.subplots()
for h in h_values:
    sigma_q_h = np.sqrt(2 * lam * np.exp(-1) * h)
    Z_q_h = sigma_q_h * (-2 * A * q_grid)
    skew = Z_q_h / sigma_q_h * 2  # 2 * Z^q/sigma_q (since skew = 2*shift)
    ax.plot(q_grid, skew, lw=1.8, label=f"h = {h}")
ax.set_xlabel("q"); ax.set_ylabel("skew δ_a - δ_b")
ax.set_title("MV skew steepens as h falls (more competition)")
ax.legend(); plt.tight_layout(); plt.show()
"""))

CELLS.append(md(exercise(
    3,
    r"""**Read the full chain.** Open [`equations/contxiong_lob_mv_adverse.py`](../equations/contxiong_lob_mv_adverse.py) (the full model) and identify (a) the forward SDE with signal, (b) the combined execution factor (MV × adverse), (c) the generator. Match each to lines in the simpler `contxiong_lob.py` and `contxiong_lob_mv.py`, noting what changes and what stays the same.""",
    r"""Structure: inherits the signal-augmented forward SDE from `contxiong_lob_adverse.py`, the law-encoder plumbing from `contxiong_lob_mv.py`, and combines both execution factors multiplicatively: `f_a = exp(-alpha*delta_a) * adv_a(signal) * h(law) * lam_a`. The generator is structurally the same — only the execution factor formula changes. Terminal is the same $-\phi q_T^2$."""
)))

CELLS.append(md(r"""## 10. Takeaways

| Concept | Role in the repo |
|---------|-----------------|
| $f(\delta) = e^{-\alpha\delta}$ | Exponential fill intensity — standard Avellaneda–Stoikov |
| HJB with quadratic penalty | Produces linear-in-$q$ optimal skew |
| $\delta_a^* = 1/\alpha + Z^q/\sigma_q$ | The formula the BSDE solver ultimately learns |
| $h(\mu)$ | Competitive factor $\in (0.01, 1]$, learned from law embedding |
| 2-D vs 3-D state | Adverse-selection extension adds a signal coordinate |

## What's next

[Module 6](06_deep_bsde_numerics.ipynb) is where the rubber meets the road: the **deep BSDE solver** of Han–Jentzen–E, and the *three silent failure modes* the repo diagnoses (generator bypass, BatchNorm erasure, DeepSets collapse). We'll reproduce the 4× variation in $h$ and see why it was previously invisible.

<div class="module-nav">
<a href="04_mckean_vlasov.ipynb"><strong>← Prev</strong> Module 4: McKean–Vlasov</a>
<a href="06_deep_bsde_numerics.ipynb"><strong>Next →</strong> Module 6: Deep BSDE Numerics</a>
</div>
"""))


build("05_cont_xiong_lob.ipynb", CELLS)

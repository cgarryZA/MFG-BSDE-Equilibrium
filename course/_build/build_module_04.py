"""Build Module 4: McKean-Vlasov and mean-field coupling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import md, code, code_hidden, three_layer, callout, exercise, build, STYLE_CSS


CELLS = []
CELLS.append(md(STYLE_CSS))

CELLS.append(md(r"""# Module 4 — McKean–Vlasov and Mean-Field Coupling

> *Until now a single market maker has been alone in a static world. Now: a population of identical market makers compete, and each one's payoff depends on what **everyone else** is doing.*

## Learning goals

1. **Construct the mean-field limit** of an interacting particle system via propagation of chaos.
2. **Define a McKean–Vlasov SDE** whose coefficients depend on $\mu_t = \text{Law}(X_t)$.
3. **State the MV-BSDE** generator and explain why it is the natural object for the competitive LOB problem.
4. **Compare four finite-dimensional representations** of $\mu_t$ (moments, quantiles, histogram, DeepSets) that the repo uses.
5. **Read [`equations/law_encoders.py`](../equations/law_encoders.py)** and implement a new encoder.

## Prereqs

Modules 1-3.
"""))

CELLS.append(md(r"""## 1. Interacting particles

Consider $N$ market makers, each with their own inventory $q^{(i)}_t$. If every one's execution rate depends not only on their own quote but also on the **population average** of quotes — "the competition" — then the $N$ dynamics are coupled.

In symmetric form, for $i = 1, \ldots, N$:

$$dq_t^{(i)} = \mu\bigl(q_t^{(i)}, \mu_t^N\bigr)\,dt + \sigma\bigl(q_t^{(i)}, \mu_t^N\bigr)\,dW_t^{(i)}, \qquad \mu_t^N := \frac{1}{N}\sum_{j=1}^N \delta_{q_t^{(j)}}.$$

Each agent's drift and diffusion depend on the **empirical distribution** $\mu_t^N$ of the whole population. As $N \to \infty$, $\mu_t^N$ concentrates on the deterministic limit $\mu_t = \text{Law}(q_t^\infty)$, and each agent's dynamics decouple — this is **propagation of chaos**.
"""))

CELLS.append(md(callout(
    "theorem",
    "Propagation of chaos (Sznitman, 1991)",
    r"""Suppose the coefficients $\mu(x, \mu), \sigma(x, \mu)$ are jointly Lipschitz in $(x, \mu)$ (with $\mu$ metrised by Wasserstein-$2$). Then as $N \to \infty$ the particles become asymptotically independent, each following the **McKean–Vlasov SDE**
$$dX_t^\infty = \mu(X_t^\infty, \mu_t)\,dt + \sigma(X_t^\infty, \mu_t)\,dW_t, \qquad \mu_t = \text{Law}(X_t^\infty).$$

Quantitatively, for $k$ fixed particles,
$$\sup_{t \leq T} \mathbb{E}\!\left[\bigl\|(q^{(1)}_t, \ldots, q^{(k)}_t) - (X^{\infty,1}_t, \ldots, X^{\infty,k}_t)\bigr\|_2^2\right] \lesssim \frac{1}{N}.$$"""
)))

CELLS.append(md(r"""## 2. The MV-SDE and fixed-point structure"""))

CELLS.append(md(three_layer(
    math_src=r"""A **McKean–Vlasov SDE** takes the form
$$dX_t = \mu(t, X_t, \mathrm{Law}(X_t))\,dt + \sigma(t, X_t, \mathrm{Law}(X_t))\,dW_t.$$

This is a **fixed-point** problem: the solution's law appears in its own coefficients. Existence/uniqueness under Lipschitz assumptions follows from a Picard iteration in Wasserstein distance.

Sznitman's **propagation of chaos** provides a practical sampling recipe:
1. Simulate $N$ interacting particles with $\mu_t^N$ used in place of $\mu_t$.
2. For large $N$, trajectories approximate i.i.d. samples from $\mu_t$.""",
    plain_src=r"""Every agent responds to "what everyone is doing", but what everyone is doing is determined by every agent's response — a classic circular dependence, resolved by a fixed point.

**Simulation trick.** In the repo, we don't need to solve the fixed-point analytically. We just run the forward SDE with $N$ particles (typically $N = 256$ or so) and treat the empirical $\mu_t^N$ as a proxy for $\mu_t$. The law encoders (§4) compute a fixed-dimensional summary of $\mu_t^N$.""",
    code_src=r"""In [solver.py](../solver.py)
the MV solver passes all
particles through each forward
step, computes $\mu_t^N$, and
feeds a summary of it into
the generator and subnet:

<pre>particles = X[:, :, t]
law = encoder.encode(particles)
# law is now a fixed-dim vector
z = subnet(x_own, law)</pre>"""
)))

CELLS.append(md(r"""## 3. The MV-BSDE"""))

CELLS.append(md(three_layer(
    math_src=r"""Combining Modules 2 and this module, a **McKean–Vlasov BSDE** is a system

$$-dY_t = f(t, X_t, Y_t, Z_t, \mu_t)\,dt - Z_t\,dW_t, \qquad Y_T = g(X_T, \mu_T),$$

where $\mu_t$ is the law of $X_t$ (itself possibly an MV-SDE solution).

The value function $v(t, x, \mu)$ satisfies an **infinite-dimensional HJB** on the space of probability measures — the "Master equation" in mean-field-game language.""",
    plain_src=r"""The generator $f$ has an extra argument: the law $\mu_t$ of the whole population. Everything else is the Module 2 story.

For the repo, $f$ depends on $\mu_t$ through a scalar "competitive factor":
$$f(t, x, y, z, \mu) = \text{base\_generator}(t, x, y, z) \cdot h(\mu) + \text{const.}$$

More competition (wider inventory spread in $\mu$) lowers $h$, dampens your fill rates, and shifts your optimal quotes wider.""",
    code_src=r"""[contxiong_lob_mv.py:96-124](../equations/contxiong_lob_mv.py):

<pre>def f_tf(self, t, x, y, z,
         law_embed):
    h = comp_factor_net(law_embed)
    fa = exp(-alpha*da) * h * lam_a
    fb = exp(-alpha*db) * h * lam_b
    return (-r*y - phi*q**2
            + fa*da + fb*db)</pre>

The scalar $h \in (0.01, 1]$ is
the learnable competition
discount."""
)))

CELLS.append(md(r"""## 4. Four law representations

We cannot feed an infinite-dimensional object $\mu_t$ into a neural network directly. We need to summarise $\mu_t$ as a **fixed-dimensional embedding** $\Phi(\mu_t) \in \mathbb{R}^d$. The repo implements four such encoders with a common interface:

| Encoder | Features | dim | Learnable |
|--------|---------|-----|-----------|
| Moments | $\mathbb{E}[q], \mathrm{Var}[q], \mathrm{Skew}[q], \mathbb{E}\|q\|, \max\|q\|, \sigma_q$ | 6 | No |
| Quantiles | $\mathbb{E}[q], Q_{0.1}, Q_{0.25}, Q_{0.5}, Q_{0.75}, Q_{0.9}$ | 6 | No |
| Histogram | Soft Gaussian bins over $[-q_{\max}, q_{\max}]$ | 20 | No |
| DeepSets | $\rho\bigl(\frac{1}{N}\sum_i \psi(x_i)\bigr)$ | 16 | Yes |

All four are in [equations/law_encoders.py](../equations/law_encoders.py).
"""))

CELLS.append(md(three_layer(
    math_src=r"""**DeepSets** is the only *learned* encoder:
$$\Phi(\mu) = \rho\!\left(\frac{1}{N}\sum_{i=1}^N \psi(x_i)\right),$$
with $\psi : \mathbb{R}^{d_x} \to \mathbb{R}^{d_h}$, $\rho : \mathbb{R}^{d_h} \to \mathbb{R}^{d_{\text{embed}}}$ both MLPs. Permutation-invariance is baked in by the mean pooling.

Zaheer et al. (2017) proved that every permutation-invariant function on multisets of bounded size is of this form — a universal approximation result.""",
    plain_src=r"""**Universal** in theory. In practice, mean pooling is a severe information bottleneck.

**DeepSets collapse (Finding 2 in [FINDINGS.md](../archive/FINDINGS.md)):** when inputs are symmetric around zero (e.g., inventories from a symmetric population), a randomly-initialised $\psi$ has roughly symmetric positive/negative contributions that **cancel in the mean**. The resulting embedding barely moves even when the population variance changes by a factor of 50.

This is why the other three encoders (with explicit variance features) outperform DeepSets in the repo's ablations — a clean example of a principled architecture failing for mundane symmetry reasons.""",
    code_src=r"""[equations/law_encoders.py:105-144](../equations/law_encoders.py):

<pre>class DeepSetsEncoder:
  def __init__(self, …):
    self.psi = MLP([2, 32, 32])
    self.rho = MLP([32, 32, 16])
  def encode(self, particles):
    h = self.psi(particles)  # N×32
    pooled = h.mean(dim=0)    # 32
    return self.rho(pooled)   # 16</pre>"""
)))

CELLS.append(md(r"""## 5. Visualising the collapse

Let's construct two populations that have very different variances but the same mean, and see what each encoder gives.
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

# Two populations, same mean (0), very different std.
N = 256
pop_narrow = rng.normal(0.0, 0.1, size=N)
pop_wide   = rng.normal(0.0, 3.0, size=N)

# -- Moment encoder (fixed, variance-sensitive) --
def moment_encode(q):
    sd = q.std()
    return np.array([
        q.mean(),
        q.var(),
        ((q - q.mean()) ** 3 / max(sd, 1e-8) ** 3).mean(),  # skew
        np.abs(q).mean(),
        np.abs(q).max(),
        sd,
    ])

# -- Quantile encoder --
def quantile_encode(q):
    return np.array([
        q.mean(),
        np.quantile(q, 0.1),
        np.quantile(q, 0.25),
        np.quantile(q, 0.5),
        np.quantile(q, 0.75),
        np.quantile(q, 0.9),
    ])

# -- Histogram encoder (soft Gaussian bins) --
def histogram_encode(q, n_bins=20, q_max=5.0):
    centres = np.linspace(-q_max, q_max, n_bins)
    sigma = 0.5 * (2 * q_max / n_bins)
    kernel = np.exp(-0.5 * ((q[:, None] - centres[None, :]) / sigma) ** 2)
    hist = kernel.mean(axis=0)
    return hist / (hist.sum() + 1e-8)

# -- DeepSets (random untrained psi, rho) --
# Use random ReLU nets as a fair model of a freshly-initialised encoder.
def deepsets_encode(q, embed_dim=16, seed=0):
    r = np.random.default_rng(seed)
    W1 = r.standard_normal((1, 32)) / np.sqrt(1)   # psi layer 1
    b1 = r.standard_normal((32,))
    W2 = r.standard_normal((32, 32)) / np.sqrt(32) # psi layer 2
    b2 = r.standard_normal((32,))
    Wr = r.standard_normal((32, embed_dim)) / np.sqrt(32)  # rho
    h1 = np.maximum(q[:, None] @ W1 + b1, 0)
    h2 = np.maximum(h1 @ W2 + b2, 0)
    pooled = h2.mean(axis=0)
    return np.maximum(pooled @ Wr, 0)

# Compute embedding distances between narrow and wide populations.
for name, enc in [("Moments", moment_encode),
                  ("Quantiles", quantile_encode),
                  ("Histogram", histogram_encode),
                  ("DeepSets (untrained)", deepsets_encode)]:
    phi_n = enc(pop_narrow)
    phi_w = enc(pop_wide)
    dist = np.linalg.norm(phi_n - phi_w)
    cos_sim = np.dot(phi_n, phi_w) / (np.linalg.norm(phi_n) * np.linalg.norm(phi_w) + 1e-12)
    print(f"{name:25s}  ||Phi_narrow - Phi_wide|| = {dist:7.4f}   cos = {cos_sim:+.4f}")
"""))

CELLS.append(md(callout(
    "insight",
    "What the numbers mean",
    r"""The moment and quantile encoders produce **large** embedding distances between narrow and wide populations — because they have explicit variance features. The DeepSets encoder (with random weights) produces a **tiny** distance and cosine similarity near 1 — the two populations look nearly identical to it. This is the failure mode reported in [FINDINGS.md](../archive/FINDINGS.md) Finding 2. Training DeepSets through a task gradient can partially fix this, but only if the loss landscape actually pushes $\psi$ away from symmetric initialisations. In the repo's experiments it typically doesn't."""
)))

CELLS.append(md(r"""## 6. Visualising the encoders"""))

CELLS.append(code(r"""# Plot histogram encoder vs quantile locations for both populations.
n_bins, q_max = 20, 5.0
centres = np.linspace(-q_max, q_max, n_bins)

h_n = histogram_encode(pop_narrow, n_bins, q_max)
h_w = histogram_encode(pop_wide, n_bins, q_max)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
ax1.bar(centres - 0.1, h_n, width=0.2, label="narrow pop (std=0.1)", alpha=0.7)
ax1.bar(centres + 0.1, h_w, width=0.2, label="wide pop (std=3.0)", alpha=0.7)
ax1.set_title("Histogram encoder: two populations"); ax1.set_xlabel("q")
ax1.legend()

# Quantile encoder visualisation
qs = [0.1, 0.25, 0.5, 0.75, 0.9]
ax2.scatter(qs, [np.quantile(pop_narrow, q) for q in qs], label="narrow", s=60)
ax2.scatter(qs, [np.quantile(pop_wide, q) for q in qs], label="wide", s=60)
ax2.set_xlabel("quantile level"); ax2.set_ylabel("q value")
ax2.set_title("Quantile encoder: two populations")
ax2.legend(); plt.tight_layout(); plt.show()
"""))

CELLS.append(md(r"""## 7. How the repo uses the encoder

In [`solver.py:ContXiongLOBMVModel`](../solver.py) (around lines 1549-1595), the encoder is called once per time step on the full particle cloud, and the resulting embedding is (a) broadcast across all particles and (b) used as an *input* to the subnet that computes $Z_t$ **and** the competitive factor network that computes $h$.

This dual usage — law in the subnet **and** in the generator — is what [FINDINGS.md](../archive/FINDINGS.md) Finding 1 refers to as the "two-pathway" structure. Module 6 shows what happens if you remove one pathway.
"""))

CELLS.append(md(r"""## 8. Exercises"""))

CELLS.append(md(exercise(
    1,
    r"""**Propagation of chaos numerically.** Simulate $N$ interacting diffusions $dq_i = -\frac{1}{N}\sum_j (q_i - q_j)\,dt + dW_i$ (mean-reverting to the empirical mean). Show that for $N = 2, 10, 100, 1000$ the marginal distribution of $q_1$ at $t = 1$ converges as $N$ grows. This is propagation of chaos in action.""",
    r"""The MV limit is $dX^\infty = -(X^\infty - \mathbb{E}[X^\infty])\,dt + dW$, which (since $\mathbb{E}[X^\infty_t] = 0$ by symmetry) is just $dX = -X\,dt + dW$ — an Ornstein-Uhlenbeck process. The $N$-particle system should approach this distribution as $N$ grows."""
)))

CELLS.append(code_hidden(r"""# Solution — Exercise 1
T, N_time, n_particles_list = 1.0, 200, [2, 10, 100, 1000]
dt = T / N_time
fig, ax = plt.subplots()
for N in n_particles_list:
    rng = np.random.default_rng(N)
    q = rng.standard_normal(N) * 0.1  # small initial spread
    for _ in range(N_time):
        drift = -(q - q.mean())
        q = q + drift * dt + rng.standard_normal(N) * np.sqrt(dt)
    # Run many independent realisations to get a distribution for q_1
    samples = []
    for seed in range(2000):
        r2 = np.random.default_rng(seed)
        qq = r2.standard_normal(N) * 0.1
        for _ in range(N_time):
            drift = -(qq - qq.mean())
            qq = qq + drift * dt + r2.standard_normal(N) * np.sqrt(dt)
        samples.append(qq[0])
    ax.hist(samples, bins=50, density=True, alpha=0.4, label=f"N={N}")
# OU stationary density N(0, 1/2)
xs = np.linspace(-3, 3, 200)
ax.plot(xs, np.exp(-xs**2) / np.sqrt(np.pi), "k--", lw=1.5, label=r"MV limit (OU stationary)")
ax.set_title(r"Marginal of $q_1$ at $t=1$ for interacting population"); ax.legend()
plt.tight_layout(); plt.show()
"""))

CELLS.append(md(exercise(
    2,
    r"""**Write a new encoder.** Implement a `VarianceOnlyEncoder` that returns only $[\mathrm{Var}(q)]$ (a 1-D embedding). Test it against the narrow/wide populations above. What do you predict about the repo's market-maker behaviour if you swapped in this encoder?""",
    r"""Variance alone distinguishes narrow from wide perfectly (distance = $|0.01 - 9| = 8.99$). But with only one feature, you lose any ability to respond to mean drift, skewness, or multi-modal shapes. In the repo, this encoder would correctly identify "more variance → more competition" but would fail to handle asymmetric inventories (e.g., if the population is skewed long, you'd want to quote a lower ask regardless of variance)."""
)))

CELLS.append(code_hidden(r"""# Solution — Exercise 2
def variance_only_encode(q):
    return np.array([q.var()])

for name, enc in [("VarianceOnly", variance_only_encode),
                  ("Moments", moment_encode)]:
    phi_n = enc(pop_narrow)
    phi_w = enc(pop_wide)
    print(f"{name:14s} phi(narrow) = {phi_n},  phi(wide) = {phi_w},  dist = {np.linalg.norm(phi_n - phi_w):.4f}")
"""))

CELLS.append(md(exercise(
    3,
    r"""**Diagnosing DeepSets collapse.** For the DeepSets encoder with random weights, plot the histogram of $\psi(q_i)$ values over `pop_narrow` vs `pop_wide` along a single output coordinate. Explain why the **mean** pooled representation can be identical even when the per-element $\psi(q_i)$ distribution differs substantially.""",
    r"""Even if $\psi$ outputs are highly variable, their *mean* loses all information about variance. Two populations whose per-element outputs have the same mean (but very different spreads) will pool to the same embedding. Sum-pooling (instead of mean-pooling) would partially fix this when $N$ varies; but for fixed-$N$ populations with symmetric random $\psi$, sum and mean are both vulnerable to the cancellation effect."""
)))

CELLS.append(md(r"""## 9. Takeaways

| Concept | Role in the repo |
|---------|-----------------|
| Propagation of chaos | Justifies using $\mu_t^N$ in place of $\mu_t$ during simulation |
| McKean–Vlasov fixed point | Solved implicitly by simulating the $N$-particle system end-to-end |
| Law encoder $\Phi(\mu)$ | Fixed-dim summary; 4 variants in [law_encoders.py](../equations/law_encoders.py) |
| DeepSets collapse | [FINDINGS.md](../archive/FINDINGS.md) Finding 2 — symmetric inputs + mean pool = info loss |
| $h(\Phi(\mu))$ | Competitive factor learned in MV models; varies 4× across populations after fix |

## What's next

[Module 5](05_cont_xiong_lob.ipynb) derives the **Cont–Xiong LOB model** in full — from the microstructure story to the HJB FOC that produces the explicit quote formulae used throughout the repo. This is where everything so far pays off.

<div class="module-nav">
<a href="03_bsdes_with_jumps.ipynb"><strong>← Prev</strong> Module 3: BSDEJ</a>
<a href="05_cont_xiong_lob.ipynb"><strong>Next →</strong> Module 5: Cont–Xiong LOB</a>
</div>
"""))


build("04_mckean_vlasov.ipynb", CELLS)

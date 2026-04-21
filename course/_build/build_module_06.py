"""Build Module 6: Deep BSDE numerics and the repo's architectural findings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import md, code, code_hidden, three_layer, callout, exercise, build, STYLE_CSS


CELLS = []
CELLS.append(md(STYLE_CSS))

CELLS.append(md(r"""# Module 6 — Deep BSDE Numerics

> *How do we actually **compute** the solution of a BSDE? We replace the regression basis of Module 2 with a neural network. This is the Han–Jentzen–E algorithm, and it's also where the repo's most surprising findings live.*

## Learning goals

1. **State the Han–Jentzen–E (2017, 2018) algorithm** — forward simulation of $X$, neural network for $Y_0$ and $Z_t$.
2. **Describe the loss function** — terminal mismatch $\mathbb{E}\|Y_T - g(X_T)\|^2$.
3. **Diagnose the BatchNorm erasure failure mode** and the two-stream fix.
4. **Diagnose the DeepSets representation collapse** and understand why moment/quantile encoders don't suffer from it.
5. **Read [FINDINGS.md](../archive/FINDINGS.md)** and [paper.pdf](../paper.pdf), and explain the 4× variation in $h$ after the fixes.
6. **Follow the pathway analysis**: generator channel vs direct subnet conditioning.

## Prereqs

Modules 1-5.
"""))

CELLS.append(md(r"""## 1. The Han–Jentzen–E algorithm

Han, Jentzen & E (2017–2018) made the following observation: a BSDE has an *adapted* solution $(Y, Z)$ where $Z_t$ is a function of $X_t$ (in the Markovian case). So parameterise $Z_t$ as a neural network and $Y_0$ as a scalar. Simulate the forward process, roll the BSDE forward using Euler, and minimise the terminal mismatch.
"""))

CELLS.append(md(three_layer(
    math_src=r"""**Algorithm (HJE).**

**Parameters**: $\theta_0 = Y_0 \in \mathbb{R}$, plus weights $\theta$ of a neural network $Z_\theta : [0, T] \times \mathbb{R}^d \to \mathbb{R}^d$.

**Forward pass**:
$$X_{t_{k+1}} = X_{t_k} + \mu(t_k, X_{t_k})\,\Delta t + \sigma(t_k, X_{t_k})\,\Delta W_k,$$
$$Y_{t_{k+1}} = Y_{t_k} - f(t_k, X_{t_k}, Y_{t_k}, Z_{t_k})\,\Delta t + Z_{t_k}^\top \Delta W_k,$$
with $Z_{t_k} = Z_\theta(t_k, X_{t_k})$.

**Loss**:
$$\mathcal{L}(\theta_0, \theta) = \mathbb{E}\|Y_{t_N} - g(X_{t_N})\|^2.$$

Minimise by SGD / Adam.""",
    plain_src=r"""**Three ideas in one.**

1. **Forward simulation.** Use Module 1's Euler–Maruyama to produce many $X$ paths.
2. **Neural ansatz for $Z$.** At each discretisation time $t_k$, a small NN maps $X_{t_k}$ to $Z_{t_k}$.
3. **Terminal-mismatch loss.** The only supervisory signal is that $Y_T$ should equal $g(X_T)$. You push *every* intermediate $Z_{t_k}$ through this single bottleneck.

It works because BSDE theory says there's a unique $(Y, Z)$ that makes this identity hold pathwise. If your network gets $Z$ wrong anywhere, $Y_T$ won't match $g(X_T)$ — so gradient descent discovers the right $Z$.

**Convergence** (Beck–E–Jentzen 2019, Han–Long 2020): under Lipschitz + boundedness, the scheme converges to the true solution as $\Delta t \to 0$ and network width $\to \infty$.""",
    code_src=r"""In [solver.py](../solver.py) the
forward pass at
`ContXiongLOBModel.forward`
looks like:

<pre>y = self.y_init           # Y_0
z = self.z_init           # Z_0
x = self.x_init
for k in range(num_time):
    dW = randn() * sqrt(dt)
    # Forward SDE
    x = x + mu(t,x)*dt
          + sigma(t,x)*dW
    # BSDE Euler
    y = y - f(t,x,y,z)*dt
          + z @ dW
    # Update Z with NN
    z = subnet[k](x) / dim

loss = (y - g(T, x)).pow(2).mean()</pre>"""
)))

CELLS.append(md(r"""## 2. The naive implementation on a linear BSDE

Let's build a tiny torch-free version to demystify the algorithm. We'll solve the same linear BSDE from Module 2 (call option, Feynman–Kac) using HJE with a fixed-basis regression "network" — just to show the structure.
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

# Linear BSDE from Module 2: GBM underlying, call payoff, f(y) = -r y.
# We parametrise Z_t = beta_t^T phi(X_t) with polynomial basis.
# Y_0 is also a parameter. Train by gradient descent on terminal mismatch.

def basis(x, S0):
    z = (x - S0) / S0
    return np.stack([np.ones_like(z), z, z**2, z**3], axis=-1)  # (..., 4)

S0, K, T, r_rate, sigma_bs = 100.0, 100.0, 1.0, 0.05, 0.20
N, n_paths = 50, 10_000
dt = T / N

rng = np.random.default_rng(11)
dW_ = rng.standard_normal((N, n_paths)) * np.sqrt(dt)
logS = np.zeros((N + 1, n_paths))
logS[0] = np.log(S0)
for k in range(N):
    logS[k + 1] = logS[k] + (r_rate - 0.5 * sigma_bs**2) * dt + sigma_bs * dW_[k]
S_paths = np.exp(logS)

# Parameters: Y_0 (scalar), beta_t (4 coeffs per time step)
def loss_and_run(Y0, betas):
    Y = np.full(n_paths, Y0)
    for k in range(N):
        phi_x = basis(S_paths[k], S0)
        Z = phi_x @ betas[k]   # (n_paths,)
        # BSDE Euler: Y_{k+1} = Y_k - f(Y_k) dt + Z dW_k
        Y = Y - (-r_rate * Y) * dt + Z * dW_[k]
    payoff = np.maximum(S_paths[-1] - K, 0.0)
    residual = Y - payoff
    return 0.5 * np.mean(residual**2), Y, residual

# Initialise and descend
Y0 = 5.0
betas = np.zeros((N, 4))
lr = 0.02

losses = []
for step in range(400):
    # Finite differences for gradient (ok at this scale; real solver uses autograd)
    base_loss, _, residual = loss_and_run(Y0, betas)
    losses.append(base_loss)
    # Gradient w.r.t. Y0: dL/dY0 = mean(residual * dY_T/dY0)
    # dY_T/dY0 = product of (1 + r dt)^N ≈ exp(r T). Compute exactly:
    grow = (1 + r_rate * dt) ** N
    grad_Y0 = np.mean(residual * grow)
    Y0 -= lr * grad_Y0
    # Gradient w.r.t. betas[k]: dY_T/dbetas[k] = phi(X_k) * dW_k * product_{j>k}(1 + r dt)
    for k in range(N):
        phi_xk = basis(S_paths[k], S0)           # (n_paths, 4)
        grow_after = (1 + r_rate * dt) ** (N - k - 1)
        dY_T_dbeta = phi_xk * (dW_[k] * grow_after)[:, None]
        grad_beta = np.mean(residual[:, None] * dY_T_dbeta, axis=0)
        betas[k] -= lr * grad_beta

# Verify: Y_0 close to Black-Scholes
from math import log, sqrt, erf
d1 = (log(S0/K) + (r_rate + 0.5*sigma_bs**2)*T) / (sigma_bs*sqrt(T))
d2 = d1 - sigma_bs*sqrt(T)
Nf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
bs = S0*Nf(d1) - K*np.exp(-r_rate*T)*Nf(d2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
ax1.plot(losses, lw=1.5)
ax1.set_yscale("log"); ax1.set_xlabel("gradient step")
ax1.set_ylabel("terminal-mismatch loss")
ax1.set_title("Deep-BSDE-style descent on a linear call BSDE")

ax2.axvline(bs, color="crimson", ls="--", label=f"Black-Scholes ({bs:.2f})")
ax2.axvline(Y0, color="navy", lw=1.8, label=f"learned Y_0 = {Y0:.2f}")
ax2.set_xlim(bs - 1, bs + 1); ax2.set_title("Final Y_0 converges to BS price")
ax2.legend(); plt.tight_layout(); plt.show()

print(f"Black-Scholes : {bs:.4f}")
print(f"Learned Y_0    : {Y0:.4f}")
print(f"Difference     : {abs(Y0 - bs):.4f}")
"""))

CELLS.append(md(callout(
    "insight",
    "What just happened",
    r"""We solved a BSDE by **backpropagating through the Euler scheme**. The "network" here is trivial (polynomial basis, one linear layer per time step), but the structure is identical to the repo's full deep BSDE solver. Scale up the basis to a multi-layer MLP, replace finite differences with autograd, and you have [solver.py](../solver.py)."""
)))

# --- Failure modes ---

CELLS.append(md(r"""## 3. Three failure modes of mean-field coupling

The paper ([paper.pdf](../paper.pdf) §3) identifies **three independent failures** that each silently suppress the mean-field signal. Any single one is sufficient to produce the "MV has no effect" non-result that the early runs reported.

1. **Generator bypass** (implementation error). The BSDE generator `f_tf` inherited from the non-MV base class and silently used the old scalar moment proxy instead of the law embedding $\Phi(\mu_t)$. The optimizer had no gradient path from the loss back to the encoder. Fix: override `f_tf` in the MV classes and connect the law embedding via a small `CompetitiveFactorNet`.
2. **BatchNorm erasure** (architectural). Because $\Phi(\mu_t)$ is broadcast identically across the batch, its batch variance is zero — BatchNorm zeros it out. §3.1 below.
3. **DeepSets representation collapse** (architectural). Mean pooling of a randomly initialised $\psi$ under symmetric inputs cancels variance information. §3.2 below.

This module focuses on the two architectural failures (2 and 3) because they generalise beyond this specific codebase — any deep BSDE solver with broadcast conditioning features is vulnerable to the same traps. Failure 1 was a code bug specific to this project.
"""))

CELLS.append(md(r"""### 3.1 Failure mode 2 — BatchNorm erasure

The naive MV-BSDE solver (prior to the fix) used BatchNorm on the combined `[state, law_embedding]` input. BatchNorm subtracts the per-batch mean and divides by the per-batch standard deviation:

$$\text{BN}(x) = \gamma \cdot \frac{x - \mathbb{E}_{\text{batch}}[x]}{\sqrt{\mathrm{Var}_{\text{batch}}(x) + \epsilon}} + \beta.$$

If a feature is **constant across the batch**, its batch variance is zero — so BatchNorm divides by zero. In practice the $\epsilon$ regularisation prevents a NaN, but the output is essentially zero: the feature is **erased**.
"""))

CELLS.append(md(three_layer(
    math_src=r"""When the population $\mu_t$ is used across the whole batch, the embedding $\Phi(\mu_t)$ is **broadcast identically to all $N$ particles**:
$$\text{batch input} = \bigl[(X^{(1)}_t, \Phi(\mu_t)), \ldots, (X^{(N)}_t, \Phi(\mu_t))\bigr].$$
Only the $X$ coordinates vary across the batch; the $\Phi$ coordinates are identical. BatchNorm sees $\mathrm{Var}_{\text{batch}}(\Phi) = 0$ and silently zeros them out.""",
    plain_src=r"""**The subtle bug.** When you feed the whole population through the network simultaneously, the law embedding is the same value for every particle in the batch. BN computes the cross-batch standard deviation and finds **zero variance** in those coordinates. It then produces output ≈ 0 for those coordinates.

From the network's perspective, the law embedding doesn't exist. All the machinery of law encoders, `CompetitiveFactorNet`, etc. is computing signals that are silently zeroed before reaching any weights. The training loop still converges — but to a solution that completely ignores the population.

This was responsible for the earlier "MV doesn't matter" results that the repo overturned.""",
    code_src=r"""The fix is **two-stream**
architecture
([solver.py:MeanFieldSubNet, lines 31-82](../solver.py)):

<pre>state_stream = BN(state) >
               Dense > BN > ReLU
law_stream   = Dense > ReLU
             (NO BN)
output = state_stream + law_stream</pre>

BN is removed from the law
path. The state path keeps BN
for its usual benefits
(conditioning the optimisation)
but the law path passes through
unchanged."""
)))

CELLS.append(md(r"""### 3.1.1 Numerical illustration of BatchNorm erasure"""))

CELLS.append(code(r"""# A toy BatchNorm showing the erasure. Two features:
#  - feature 1: varies across batch (state-like)
#  - feature 2: constant across batch (broadcast, law-like)

batch_size = 256
state = np.random.default_rng(0).standard_normal((batch_size, 1))   # varying
law_broadcast = np.full((batch_size, 1), 0.7)                        # constant

x = np.concatenate([state, law_broadcast], axis=1)  # (batch, 2)

# Naive BatchNorm (no learnable gamma/beta, just the normalisation)
epsilon = 1e-5
mean = x.mean(axis=0, keepdims=True)
var  = x.var(axis=0, keepdims=True)
x_bn = (x - mean) / np.sqrt(var + epsilon)

print("Original feature means :", x.mean(axis=0))
print("Original feature stds  :", x.std(axis=0))
print("Post-BN feature means  :", x_bn.mean(axis=0))
print("Post-BN feature stds   :", x_bn.std(axis=0))

print("\nLaw-feature sample after BN (all should be ~0):")
print(x_bn[:8, 1])
"""))

CELLS.append(md(callout(
    "warning",
    "Every value is zero",
    r"""The second column — which originally held the broadcast law embedding 0.7 everywhere — has been entirely zeroed out by BatchNorm. Since all values are identical, the batch variance is 0 and the normalised feature is 0 (up to the $\epsilon$ regulariser). Any downstream network receiving this feature sees only noise from $\epsilon$ — effectively **no signal**."""
)))

CELLS.append(md(r"""## 5. Failure mode 3 — DeepSets representation collapse

A different, subtler failure mode in the DeepSets encoder (Module 4 §4). The encoder is
$$\Phi(\mu) = \rho\Bigl(\tfrac{1}{N}\sum_i \psi(x_i)\Bigr).$$

When the per-element outputs $\psi(x_i)$ are approximately symmetric around the origin (because $\psi$ is initialised randomly with a symmetric distribution), the **mean pooling** operation causes positive and negative contributions to **cancel**.

From [FINDINGS.md](../archive/FINDINGS.md) Finding 2:

| Encoder | Embedding distance (std 0.1 vs 5.0) | Cosine similarity |
|---------|---------------------------------------|--------------------|
| DeepSets (mean pool) | **0.003** | **1.000** |
| Moments | 24.98 | varies |
| Quantiles | 9.84 | varies |

Two populations with std differing by 50× produce **cosine similarity 1** in DeepSets. The encoder is effectively blind.
"""))

CELLS.append(md(three_layer(
    math_src=r"""**Why moments succeed where DeepSets fails.**

The variance feature $\mathbb{E}[(q - \bar q)^2]$ is **invariant** to the sign structure of individual $q_i$ — it picks up spread directly. Quantiles similarly reflect distribution shape.

DeepSets *could* learn to compute variance — theoretically, a $\psi$ of the form $\psi(q) = (q, q^2)$ combined with $\rho(\mu, m) = m - \mu^2$ computes $\mathrm{Var}$. But random initialisation doesn't place $\psi$ anywhere near this, and the symmetry of gradients makes it hard to escape the collapsed regime.""",
    plain_src=r"""**Moments and quantiles are better because they're *not* permutation-invariant via mean pooling.** They extract higher-order information directly — variance is an average of *squared* differences, quantiles are rank statistics — both of which evade the cancellation issue.

**Is DeepSets always bad?** No: for asymmetric populations (e.g., all positive inventories), the cancellation problem doesn't occur. The failure is specific to the symmetric, zero-centred case that happens to be the repo's canonical test scenario.

**Practical lesson.** "Universal approximation" theorems say a family of models **can** fit any function. They don't say gradient descent **will** find it from a random init. The mean-pooled DeepSets is a case where the inductive bias actively works against the task.""",
    code_src=r"""[law_encoders.py:124-144](../equations/law_encoders.py):

<pre>class DeepSetsEncoder:
    def encode(self, particles):
        h = self.psi(particles)  # (N, 32)
        pooled = h.mean(0)        # (32,)
        return self.rho(pooled)</pre>

The same `.mean(0)` line is
where variance information
vanishes."""
)))

CELLS.append(md(r"""## 6. The results after the fixes

After fixing all three failure modes — (i) overriding `f_tf` in the MV classes so the generator actually uses the law embedding (an implementation bug known as **generator bypass**, [paper.pdf](../paper.pdf) §3.1), (ii) adopting the two-stream architecture to sidestep BatchNorm, and (iii) swapping DeepSets for a variance-aware encoder — the repo reports:

- **4×** variation in $h$ across population shapes (0.442 → 0.109, [paper.pdf](../paper.pdf) §3 / [FINDINGS.md](../archive/FINDINGS.md) §3).
- **Measurable quote shifts**: $\delta_a$ at $q = 0$ goes from 0.696 (narrow pop) to 0.334 (wide pop) — ~2× range ([FINDINGS.md](../archive/FINDINGS.md) §5).
- **Distribution sensitivity is not encoder-specific** (Moments, Quantiles both work) provided the encoder has explicit variance features.

But a further finding: even with $h$ correctly varying, **policy variation requires direct subnet conditioning**, not just generator-channel coupling.
"""))

CELLS.append(md(three_layer(
    math_src=r"""**Two pathways** for law influence:

1. **Generator channel**: $\Phi(\mu) \to h(\Phi) \to f(t, x, y, z, h) \to $ BSDE dynamics.
2. **Subnet channel**: $\Phi(\mu) \to$ input to $Z$-network $\to Z_t \to$ quotes $\delta = 1/\alpha + Z^q/\sigma_q$.

Pathway 1 affects $Y$ dynamics. Pathway 2 affects controls directly.

**h-only experiment**: zero out the law from the subnet input, keeping only pathway 1. Result: $h$ still varies 4× correctly, but controls are identical across populations. The generator pathway alone does *not* propagate distribution information to the policy.""",
    plain_src=r"""**Why this matters.**

In classical BSDE theory, the generator determines the solution. So naively, if $f$ depends on $\mu$, the policy should too — via $Z_t = \sigma \nabla V(t, x, \mu)$.

But in the *discretised + learned* setting, the network only sees what you explicitly feed it. If $\mu$ is only plumbed through the generator (a scalar multiplicative factor), the solver has no architectural mechanism to make $Z(x)$ depend on $\mu$. It learns a single $Z$ that averages across populations — losing the differentiation.

**Implication for practice**: when designing deep BSDE solvers for mean-field problems, you need direct conditioning of every network on $\mu$. Relying on the BSDE dynamics to propagate the signal is insufficient.""",
    code_src=r"""In [solver.py:1588-1594](../solver.py)
the `h_only_mode` flag controls
this:

<pre>if self.h_only_mode:
    subnet_input = own_state  # no law
else:
    subnet_input = cat(
        [own_state, law_embed])
z = subnet[t](subnet_input)</pre>

Toggle the flag → you turn
pathway 2 on and off."""
)))

CELLS.append(md(r"""## 7. The two-stream architecture in detail

Let's look at the fix concretely. The `MeanFieldSubNet` in [solver.py:31-82](../solver.py) has two parallel paths and adds their outputs.
"""))

CELLS.append(md(three_layer(
    math_src=r"""**State stream**:
$$x_{\text{state}} \to \text{BN} \to \text{Dense} \to \text{BN} \to \text{ReLU} \to \text{Dense}.$$
BN is used for its usual role — whitening the state coordinates for optimiser conditioning.

**Law stream**:
$$x_{\text{law}} \to \text{Dense} \to \text{ReLU} \to \text{Dense}.$$
No BN. The law embedding is broadcast-identical across batch — applying BN would zero it out.

**Output**:
$$z = \text{state\_stream}(x_{\text{state}}) + \text{law\_stream}(x_{\text{law}}).$$

The sum is an inductive bias: law shifts $Z$ additively. A multiplicative alternative (**FiLM**, [solver.py:85-144](../solver.py)) is also provided in the repo.""",
    plain_src=r"""**The additive split works because** law and state live in *different subspaces*. Summing them is fine — they don't need to be compared element-wise.

**FiLM** (Feature-wise Linear Modulation) is an alternative where the law stream generates a scale $\gamma$ and shift $\beta$ that multiplicatively modulate the state stream's hidden activations:
$$h \leftarrow \gamma(\text{law}) \odot h + \beta(\text{law}).$$
This permits richer cross-partial interactions $\partial^2 Z / (\partial q \cdot \partial\Phi) \neq 0$, which the additive form cannot express.

The repo experiments with both; both fix the erasure bug.""",
    code_src=r"""[solver.py:31-82](../solver.py):

<pre>class MeanFieldSubNet:
    def __init__(self, cfg):
        self.bn_in = BN(state_dim)
        self.w_state = Dense(state_dim, h)
        self.bn_h = BN(h)
        self.w_state2 = Dense(h, d_out)
        # law path (no BN)
        self.w_law = Dense(law_dim, h)
        self.w_law2 = Dense(h, d_out)
    def forward(self, x_state, x_law):
        s = self.bn_in(x_state)
        s = self.bn_h(self.w_state(s))
        s = self.w_state2(relu(s))
        l = self.w_law2(
            relu(self.w_law(x_law)))
        return s + l</pre>"""
)))

CELLS.append(md(r"""## 8. Summary of findings

From [FINDINGS.md](../archive/FINDINGS.md) §§1–7:

| # | Finding | Significance |
|---|---------|--------------|
| 1 | **Generator bypass**: `f_tf` didn't use the law embedding (implementation bug) | Baseline for all MV claims; fix: override `f_tf` in MV classes |
| 2 | **BatchNorm erasure**: broadcast law features zeroed at every BN layer | Pre-fix "MV has no effect" results were invalid |
| 3 | **DeepSets collapse**: symmetric inputs + mean pooling cancel variance | Moments/Quantiles recommended for MV work |
| 4 | $h(\Phi)$ varies 4× across populations after fix | MV coupling is learnable — given the right architecture |
| 5 | Constant competition discount ≠ distribution sensitivity | DeepSets learns *that* there's competition, not *what shape* |
| 6 | Controls vary dramatically across populations (δ_a: 0.696 → 0.334) | The variation reaches the quotes, not just the value |
| 7 | QuantileEncoder matches MomentEncoder | Insensitivity not encoder-specific — needs variance features |
| 8 | MV coupling active at all $\phi$ values; breaks at $\phi = 0.5$ | Large penalties destabilise the solver |

Add to this the pathway analysis: direct subnet conditioning is needed for policy differentiation, not just generator coupling.
"""))

CELLS.append(md(r"""## 9. Exercises"""))

CELLS.append(md(exercise(
    1,
    r"""**Train the toy HJE solver for a non-linear BSDE.** Extend the linear example from §2 to handle $f(y, z) = -r y + \tfrac{1}{2} z^2$ (quadratic in $z$, which is not a linear BSDE). Re-solve the call option. How does the quadratic term change the answer?""",
    r"""Quadratic-in-$z$ generators correspond to utility-indifference pricing (entropic risk measures). You'll get a price *above* Black–Scholes (more conservative). Implement by updating the Euler step: $Y_{k+1} = Y_k - (-r Y_k + \tfrac{1}{2} Z_k^2) dt + Z_k dW_k$. The regression is unchanged."""
)))

CELLS.append(md(exercise(
    2,
    r"""**Reproduce the BatchNorm erasure.** Using the pattern in §4, construct a three-feature batch where two features vary and one is constant. Pass it through a BatchNorm layer from torch (or numpy). Confirm the constant feature is erased. Then compute what happens if you add a tiny amount of noise (std $10^{-6}$) to the "constant" feature — is the signal recovered?""",
    r"""Adding noise of std $10^{-6}$ makes the batch variance $\approx 10^{-12}$, which with $\epsilon = 10^{-5}$ still gets dominated by the regulariser — so the signal is *not* recovered. BN is fundamentally incompatible with broadcast features; you have to architect around it (two-stream or no-BN on law path)."""
)))

CELLS.append(md(exercise(
    3,
    r"""**Compare Moments vs DeepSets empirically.** Using the plotting code from Module 4, run ten different random initialisations of the DeepSets encoder and report the mean and std of the embedding distance between narrow and wide populations. How often does DeepSets recover a meaningful distance by chance?""",
    r"""In 10 random seeds, you'll typically see distances well below 1 across the board. The symmetry cancellation is robust — the embedding distance depends only weakly on the random init. In contrast, the Moments encoder is deterministic (no learnable weights) and always gives ~25. The next cell runs this experiment."""
)))

CELLS.append(code_hidden(r"""# Solution — Exercise 3
# Self-contained: define encoders + populations inline so this notebook runs standalone.
rng = np.random.default_rng(0)
N = 256
pop_narrow = rng.normal(0.0, 0.1, size=N)
pop_wide = rng.normal(0.0, 3.0, size=N)

def deepsets_encode(q, embed_dim=16, seed=0):
    r = np.random.default_rng(seed)
    W1 = r.standard_normal((1, 32)) / np.sqrt(1)
    b1 = r.standard_normal((32,))
    W2 = r.standard_normal((32, 32)) / np.sqrt(32)
    b2 = r.standard_normal((32,))
    Wr = r.standard_normal((32, embed_dim)) / np.sqrt(32)
    h1 = np.maximum(q[:, None] @ W1 + b1, 0)
    h2 = np.maximum(h1 @ W2 + b2, 0)
    pooled = h2.mean(axis=0)
    return np.maximum(pooled @ Wr, 0)

def moment_encode(q):
    sd = q.std()
    return np.array([q.mean(), q.var(), 0, np.abs(q).mean(), np.abs(q).max(), sd])

distances = []
for seed in range(10):
    phi_n = deepsets_encode(pop_narrow, seed=seed)
    phi_w = deepsets_encode(pop_wide, seed=seed)
    distances.append(np.linalg.norm(phi_n - phi_w))
print(f"DeepSets embedding distance across 10 seeds:")
print(f"  mean = {np.mean(distances):.4f}")
print(f"  std  = {np.std(distances):.4f}")
print(f"  max  = {np.max(distances):.4f}")
print(f"Moments (deterministic) distance = {np.linalg.norm(moment_encode(pop_narrow) - moment_encode(pop_wide)):.4f}")
"""))

CELLS.append(md(r"""## 10. Takeaways

| Concept | Where it lives |
|---------|----------------|
| HJE forward pass + terminal loss | [solver.py](../solver.py) throughout |
| BatchNorm erasure fix | Two-stream [MeanFieldSubNet, solver.py:31-82](../solver.py) |
| DeepSets collapse | Avoid mean-pooling with symmetric inputs; use Moments/Quantiles |
| Two-pathway principle | Law must enter *both* the generator *and* the subnet for policy variation |
| Empirical validation | [paper.pdf](../paper.pdf) §3 documents the 4× and 2× variation |

## What's next

[Module 7](07_multi_agent_nash.ipynb) steps up one more level: what if the population of market makers is not a passive mean field but a *strategic* collection of agents, each training their own policy? We get **multi-agent RL**, the MADDPG algorithm, and the repo's finding of **tacit collusion** — self-interested agents converging above the single-shot Nash.

<div class="module-nav">
<a href="05_cont_xiong_lob.ipynb"><strong>← Prev</strong> Module 5: Cont–Xiong LOB</a>
<a href="07_multi_agent_nash.ipynb"><strong>Next →</strong> Module 7: Multi-Agent Nash</a>
</div>
"""))


build("06_deep_bsde_numerics.ipynb", CELLS)

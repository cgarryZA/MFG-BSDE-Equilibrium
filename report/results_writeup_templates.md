# Paper-quality writeup templates (fill in with actual numbers)

## 1. Adverse selection section (DONE - fill numbers)

We study the CX dealer market with a fraction α of arriving RFQs being informed
— these trades, conditional on execution, impose a per-trade adverse cost θ on
the dealer (a stylised representation of post-trade price drift against the dealer).
The remaining 1−α are uninformed ("liquidity") flow. The dealer's optimal quote
strategy accounts for this mixture via the FOC:

  δ* = argmax f(δ, avg) · (δ − p − αθ)

where p is the marginal inventory value and αθ is the expected adverse cost.

### Finding 1: Linear equilibrium spread response

Across α ∈ [0, 0.8] at θ = 0.3, the equilibrium spread satisfies

  spread(α) = 1.513 + 0.513 α + O(α²)

with the quadratic term [SSE 1e-9 vs 1e-5 for linear] confirming linearity to
machine precision. We identify this slope as the equilibrium pass-through
coefficient.

### Finding 2: Incomplete pass-through (~85%)

The naive dealer formula — which simply adds αθ to the effective marginal
inventory cost on each side — predicts a spread increase of 2αθ per unit of α,
i.e. a slope of 0.6 at θ = 0.3. Observed slope is 0.513, corresponding to a
stable pass-through coefficient of **0.855 ± 0.007** across all α values tested.

We interpret this as follows: dealers cannot fully internalise adverse selection
costs because widening quotes reduces execution probability in equilibrium. This
creates an endogenous execution-probability feedback that partially compensates
for informed flow.

### Finding 3: Cross-side contagion

When adverse selection is asymmetric — only the ask side is contaminated with
informed flow (α_a > 0, α_b = 0) — both ask and bid quotes widen. At α_a = 0.6,
θ = 0.3:

  δ_a(q=0): 0.758 → 0.848   (+0.090, direct effect)
  δ_b(q=0): 0.758 → 0.819   (+0.061, indirect effect via equilibrium coupling)

The bid-side widening is 68% of the ask-side widening despite the bid having
no adverse cost. We attribute this to the population-averaging structure of the
execution probability: informed flow on the ask side raises the equilibrium
average ask quote, which via the mean-field coupling shifts the bid-side FOC.

### Finding 4: Inventory vs information substitution

The ratio of spread at |q|=3 to spread at q=0 measures inventory sensitivity:

  α = 0.0: ratio 1.021
  α = 0.8: ratio 1.013

As α rises, the quote profile flattens across inventory. We interpret this as a
regime shift from inventory-dominated to information-dominated quoting: when
informed flow is a substantial fraction of order flow, the dealer's primary
concern shifts from inventory risk to adverse selection risk, and the inventory
state becomes less policy-relevant.

### Value degradation

Dealer value V(q=0) drops from 16.18 at α=0 to 11.43 at α=0.8, a loss of
**29.4%**. The value function is approximately linear in α (quadratic coefficient
0.90 vs linear 6.65), consistent with the linear equilibrium spread response.

---

## 2. Learning-by-doing section (TEMPLATE — fill when deep results finish)

We introduce a state variable a_t ∈ [0, a_max] representing an exponentially
weighted moving average of the dealer's recent execution activity with half-life H:

  da_t/dt = (𝟙{execution at t} − a_t) / H

The dealer's execution intensity is modulated by this activity state:

  λ_eff(a) = λ_0 · (1 + κ (a − ā))

where κ ≥ 0 is the adaptation strength and ā is a reference level.
This breaks Markovity in inventory alone — the state space becomes (q, a) with
a continuous dimension, rendering the exact Algorithm 1 inapplicable.

### Finding 1 (TBD): Policy shape vs activity

[INSERT from deep results: does spread widen or tighten with a?]

### Finding 2 (TBD): Endogenous franchise value

[INSERT: V(q=0, high a) − V(q=0, low a) as function of κ. How does this gap
scale? Linear? Sub-linear? What's the economic scale relative to baseline V?]

### Finding 3 (TBD): Inventory dynamics

[INSERT: forward-simulated inventory std at different κ. Does adaptation
reduce or amplify inventory volatility?]

### Finding 4 (TBD): Phase transition

[INSERT: scan κ densely; look for discontinuity or regime change.]

---

## Overall narrative (for the Part III chapter)

### Proposed thesis statement

> Deep BSDE methods enable the study of dealer equilibria in regimes that
> render the classical dynamic programming approach of Cont and Xiong (2024)
> inapplicable: when state dimensions are high, when state variables are
> continuous (as with activity EWMAs), when non-Markov memory is present,
> or when randomness is shared across agents (common noise).
> We document four such extensions and characterise their equilibrium
> properties empirically.

### The four mechanisms, framed as a coherent programme

| Model | Relaxes | What breaks exact | What we show |
|-------|---------|-------------------|--------------|
| Adverse selection | Common flow | State now (q, s) where s is info state | Linear response, 85% pass-through, cross-side contagion |
| Learning-by-doing | Markov in q alone | State is (q, a), a continuous | Franchise value, regime shift (TBD) |
| Common noise | Independence | State is (q, S) with shared W_S | Z_S ≠ 0 emerges (TBD) |
| Non-stationary φ(t) | Time homogeneity | Bellman time-dependent | ... (TBD) |

### One-paragraph summary for the intro

> The Cont-Xiong (2024) dealer market-making model admits an exact fictitious-play
> algorithm for its Nash equilibrium, but only within a restricted regime: discrete
> inventory, single asset, stationary parameters, symmetric agents, and independent
> Poisson flow. This paper uses deep BSDE methods to study dealer equilibria in
> four regimes that violate these assumptions — adverse selection, endogenous
> activity, common price noise, and time-varying risk aversion. We validate the
> method against exact benchmarks where available and document the economic
> mechanisms that emerge when dynamic programming breaks down.

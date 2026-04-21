# Key Findings for Preprint 2

## Finding 1: Two critical bugs in MV coupling (FIXED)

**Bug 1: Generator didn't use law embedding**
- `f_tf` inherited from base class, used old scalar moment proxy
- Law embedding Φ(μ_t) entered subnet inputs but NOT the BSDE generator
- Training dynamics didn't depend on population → network learned to ignore Φ
- Fix: override f_tf with CompetitiveFactorNet(Φ) → h ∈ (0.1, 1)

**Bug 2: BatchNorm killed broadcast law embedding**
- Φ(μ_t) is identical across all agents in batch → zero cross-batch variance
- BatchNorm normalises by (x - mean) / sqrt(var + eps) → zero when var = 0
- Even after layer mixing, law contribution remains constant-like → BN erases it at every layer
- Fix: Two-stream MeanFieldSubNet — BN on state path only, no BN on law path

**All previous "MV has no effect" results were INVALID** — testing a doubly-dead architecture.

## Finding 2: DeepSets representation collapse under symmetric inputs

**Setup:** Two populations with same mean(q) = 0, different variance (std 0.1 vs 5.0)

| Encoder | Embedding distance | Cosine similarity |
|---------|-------------------|-------------------|
| DeepSets (mean pool) | 0.003 | 1.000 |
| MomentEncoder | 24.98 | varies |
| QuantileEncoder | 9.84 | varies |

**Root cause:** DeepSets computes Φ = (1/N) Σ ψ(q_i). With symmetric ψ (random init) and zero-mean distributions, positive and negative contributions cancel under mean pooling. Variance information is structurally lost.

**Implication:** "Permutation-invariant encoders based on mean pooling may fail to capture higher-order distributional properties under symmetric inputs unless explicitly trained to do so."

This is NOT a bug — it's a structural representation limitation.

## Finding 3: MV coupling IS learnable with correct encoder

With MomentEncoder (explicit mean, var, skew features):

| Population std(q) | h(Φ) | Interpretation |
|-------------------|-------|----------------|
| 0.1 (narrow) | 0.442 | Low competition → high fill |
| 1.0 (medium) | 0.331 | Moderate competition |
| 5.0 (wide) | 0.109 | High competition → low fill |

**4x variation** in competitive factor. Model learned: wider inventory distribution → more aggressive competitors → lower fill rate. Economically sensible.

Proves: the model CAN learn law-dependent competition, but only if the encoder provides variance-sensitive features.

## Finding 4: Constant competition discount vs distribution sensitivity

With corrected coupling (both fixes):
- No coupling: Y₀ = 0.380
- MV DeepSets: Y₀ = 0.113 (large shift, but constant h across populations)
- MV Moments: Y₀ = 0.097 (large shift AND h varies with population)

The DeepSets model learned "competition exists" (h ≈ 0.43 < 1.0) but NOT "population shape matters" (h constant across distributions). The MomentEncoder model learned both.

## Emerging dissertation narrative

1. Naive MV implementation shows no effect (initial, due to bugs)
2. After fixing coupling + architecture → MV affects value (constant discount)
3. But DeepSets shows no distribution sensitivity (representation collapse)
4. Handcrafted encoders recover full discrimination (4x h variation)
5. Representation choice critically determines whether MF effects are learnable

This is a layered, diagnostic result — exactly what the dissertation needs.

## Finding 5: Controls change dramatically with population (MomentEncoder)

After fixing both bugs and using MomentEncoder:

At q=0, quotes vary by 2x across population shapes:
- Narrow (std=0.1): δ_a=0.696, δ_b=0.638, ν_a=0.160, ν_b=0.175
- Wide (std=3.0):   δ_a=0.334, δ_b=0.999, ν_a=0.065, ν_b=0.024

At q=2 (long inventory), quote skew flips:
- Narrow: skew = +0.142 (slightly wider ask)
- Wide:   skew = -1.622 (extremely aggressive ask)

This is genuine distribution-dependent optimal control:
- Wider population → fiercer competition → narrower ask, wider bid
- The market maker adapts strategy based on population inventory distribution
- Not just a level shift — the entire quoting strategy changes shape

## Finding 6: Encoder robustness confirmed

QuantileEncoder produces the same qualitative pattern as MomentEncoder:
- h(narrow) = 0.449, h(medium) = 0.345, h(wide) = 0.125
- 3.6x variation (vs 4x for MomentEncoder)
- Distribution sensitivity is NOT encoder-specific

## Finding 7: MV coupling active at ALL tested phi values

| phi  | h(narrow) | h(wide) | Gap   |
|------|-----------|---------|-------|
| 0.01 | 0.353     | 0.100   | 0.253 |
| 0.05 | 0.453     | 0.100   | 0.353 |
| 0.10 | 0.479     | 0.100   | 0.379 |
| 0.50 | 0.169     | 0.100   | 0.069 |

The earlier "no effect at phi=0.01" was entirely due to bugs (f_tf + BN).
With corrected architecture, MV coupling is measurable even at phi=0.01.
At phi=0.5, the model struggles (Y0 negative) suggesting instability.

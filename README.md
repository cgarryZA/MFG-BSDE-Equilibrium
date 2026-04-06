# DeepMVBSDEJ: Architectural Conditions for Mean-Field Deep BSDE Solvers

Investigates when and why distribution-dependent mean-field coupling becomes learnable in deep BSDE solvers for market-making. Extends the [DeepBSDE-LOB](https://github.com/cgarryZA/DeepBSDE-LOB) solver with law encoders, two-stream architecture, and controlled pathway experiments.

**Paper:** "When Does Mean-Field Structure Matter? Distribution Dependence, Representation Collapse, and Architectural Constraints in Deep BSDE Market-Making"

## Key Findings

1. **Two silent failure modes** suppress mean-field effects: BatchNorm erasing broadcast features, and mean-pooled DeepSets collapsing distributional variance.
2. **After correction**, the competitive factor h varies 17x across population shapes and quotes shift measurably.
3. **However**, all policy variation comes from direct subnet conditioning — the BSDE generator pathway alone does not propagate the signal to controls (h-only model produces identical Z despite 17x h variation).
4. Learning a correct economic signal is **not sufficient** for it to affect behaviour through the BSDE dynamics.

## Model Hierarchy

| Model | State | MF Coupling | Key Feature |
|-------|-------|------------|-------------|
| `contxiong_lob` | 2D (S, q) | None | Diffusion surrogate baseline |
| `contxiong_lob_mv` | 2D (S, q) | Law encoder | Distribution-dependent coupling |
| `contxiong_lob_adverse` | 3D (S, q, signal) | None | Price-dependent execution |
| `contxiong_lob_mv_adverse` | 3D (S, q, signal) | Law encoder | **Full model** |

## Law Encoders

Four population distribution representations, all with the same interface:

| Encoder | Features | Learnable | embed_dim |
|---------|----------|-----------|-----------|
| Moments | mean, var, skew, mean\|q\|, max\|q\|, std | No | 6 |
| Quantiles | mean + 5 quantiles | No | 6 |
| Histogram | Soft Gaussian bins | No | 20 |
| DeepSets | Permutation-invariant NN | **Yes** | 16 |

## Installation

```bash
git clone https://github.com/cgarryZA/DeepMVBSDEJ.git
cd DeepMVBSDEJ
pip install torch numpy matplotlib
```

## Quick Start

```bash
# Base model (diffusion surrogate)
python main.py --config configs/lob_d2.json --exp_name base --device auto

# MV model with DeepSets encoder
python main.py --config configs/lob_d2_mv.json --exp_name mv_deepsets --device auto

# Adverse selection (3D, genuinely multi-dimensional)
python main.py --config configs/lob_d3_adverse.json --exp_name adverse --device auto

# Full model: MV + adverse selection
python main.py --config configs/lob_d3_mv_adverse.json --exp_name full --device auto
```

## Experiments

```bash
# Reproduce ALL paper results (main sensitivity, encoder ablation,
# multi-seed, placebo, disentanglement, h-only, generalisation, penalty sweep)
python scripts/run_all_for_paper.py --device cuda

# Quick version (~30 min instead of ~2.5h)
python scripts/run_all_for_paper.py --device cuda --quick

# Individual experiments
python scripts/run_mv_experiments.py --device cuda          # Encoder ablation
python scripts/law_sensitivity_test.py --train --device cuda # Law sensitivity
python scripts/stability_frontier.py --device cuda           # Phase diagram
python scripts/finite_difference_adverse.py --eta 0.5        # 3D FD baseline
python scripts/run_floor_ablation.py --device cuda           # Floor ablation + clipping
```

## Repository Structure

```
DeepMVBSDEJ/
├── main.py                              # Training entry point
├── solver.py                            # Models + solvers (MeanFieldSubNet, ContXiongLOBMVModel)
├── config.py                            # Configuration
├── registry.py                          # Equation registration
├── equations/
│   ├── base.py                          # Abstract base
│   ├── law_encoders.py                  # 4 law encoder classes + registry
│   ├── contxiong_lob.py                 # Diffusion surrogate (base)
│   ├── contxiong_lob_mv.py              # + MV coupling (CompetitiveFactorNet)
│   ├── contxiong_lob_adverse.py         # + adverse selection (3D)
│   └── contxiong_lob_mv_adverse.py      # + both (full model)
├── configs/                             # JSON experiment configs
├── scripts/
│   ├── run_all_for_paper.py             # Master script: ALL paper experiments
│   ├── run_mv_experiments.py            # Encoder ablation + baselines
│   ├── run_floor_ablation.py            # Floor ablation + clipping analysis
│   ├── law_sensitivity_test.py          # Law sensitivity test
│   ├── stability_frontier.py            # Phase diagram sweep
│   ├── plot_stability.py                # Phase diagram plots
│   └── finite_difference_adverse.py     # 3D FD baseline
├── results_paper_final/                 # Reproducible results for paper
└── report/preprint/                     # LaTeX source
```

## Citation

```bibtex
@misc{garry2026deepmvbsdej,
  author       = {Christian Garry},
  title        = {{DeepMVBSDEJ}: {McKean-Vlasov} Jump-{BSDE} Solver for
                  Market-Making in Limit Order Books},
  year         = {2026},
  howpublished = {\url{https://github.com/cgarryZA/DeepMVBSDEJ}},
}
```

## References

- Cont, R. & Xiong, W. (2024). Dynamics of market making algorithms in dealer markets. *Math. Finance*, 34:467-521.
- Han, J., Hu, R. & Long, J. (2022). Learning high-dimensional McKean-Vlasov FBSDEs. *SIAM J. Numer. Anal.*, 60(4):2208-2232.
- Han, J., Jentzen, A. & E, W. (2018). Solving high-dimensional PDEs using deep learning. *PNAS*, 115(34):8505-8510.
- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quant. Finance*, 8(3):217-224.

## License

MIT

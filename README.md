# DeepMVBSDEJ: McKean-Vlasov Jump-BSDE Solver for Market-Making

Distribution-dependent mean-field coupling and adverse selection for optimal market-making in limit order books. Extends the [DeepBSDE-LOB](https://github.com/cgarryZA/DeepBSDE-LOB) preprint with genuine McKean-Vlasov structure.

## What's New (vs DeepBSDE-LOB)

| Feature | DeepBSDE-LOB (preprint 1) | This repo |
|---------|--------------------------|-----------|
| Mean-field coupling | Moment proxy (inactive) | **4 law encoders** (moments, quantiles, histogram, DeepSets) |
| Price dependence | Economically inert (1D reduction) | **Adverse selection** breaks reduction → genuinely 3D |
| State dimension | 2 (S, q) effectively 1D | 3 (S, q, signal) genuinely multi-dimensional |
| Stability analysis | Preliminary table | **Full phase diagram** (phi × eta × horizon × coupling) |

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
# MV encoder ablation + baselines
python scripts/run_mv_experiments.py --quick --device cuda

# Law sensitivity test (does distribution shape affect policy?)
python scripts/law_sensitivity_test.py --train --device cuda

# Stability frontier (phase diagram)
python scripts/stability_frontier.py --quick --device cuda

# 3D finite-difference baseline for adverse selection
python scripts/finite_difference_adverse.py --eta 0.5
```

## Repository Structure

```
DeepMVBSDEJ/
├── main.py                              # Training entry point
├── solver.py                            # All model + solver classes
├── config.py                            # Configuration
├── registry.py                          # Equation registration
├── equations/
│   ├── base.py                          # Abstract base
│   ├── law_encoders.py                  # 4 law encoder classes
│   ├── contxiong_lob.py                 # Diffusion surrogate (base)
│   ├── contxiong_lob_mv.py              # + MV coupling
│   ├── contxiong_lob_adverse.py         # + adverse selection (3D)
│   └── contxiong_lob_mv_adverse.py      # + both (full model)
├── configs/                             # JSON experiment configs
├── scripts/
│   ├── run_mv_experiments.py            # Encoder ablation + baselines
│   ├── law_sensitivity_test.py          # Critical MV validation
│   ├── stability_frontier.py            # Phase diagram sweep
│   ├── plot_stability.py                # Phase diagram plots
│   └── finite_difference_adverse.py     # 3D FD baseline
└── plots/                               # Generated figures
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

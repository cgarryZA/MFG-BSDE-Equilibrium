# Deep MV-BSDE Solver for Limit Order Book Market-Making

A PyTorch implementation of the Deep Mean-Field BSDE method applied to optimal market-making in limit order books, based on the Cont-Xiong (2024) dealer market model.

The solver recovers optimal bid-ask quotes directly from the learned gradient process $Z_t$ of the value function, using the Han-Hu-Long (2022) fictitious play architecture for McKean-Vlasov FBSDEs. Both a continuous inventory relaxation (Option A) and an exact jump-diffusion formulation (Option B / FBSDEJ) are implemented.

## Key Results

| Metric | Value |
|--------|-------|
| Optimal spread | $2/\alpha = 1.333$ (exact A-S recovery) |
| Final loss | $3.3 \times 10^{-5}$ (5000 iterations) |
| Value function $Y_0$ | 0.456 (matches finite-horizon theory 0.467) |
| max $\|Z_t\|$ | 0.07 (stable, no gradient explosion) |

## Mathematical Formulation

### Forward SDE (continuous relaxation)

$$dS_t = \sigma \, dW_t^S$$

$$dq_t = \bigl(\lambda^b f_b(\delta_t^b, \mu_t) - \lambda^a f_a(\delta_t^a, \mu_t)\bigr) dt + \sqrt{\lambda^b f_b + \lambda^a f_a} \, dW_t^q$$

where $f(\delta) = e^{-\alpha\delta} \cdot h(\mu_t)$ is the execution probability with mean-field competitive factor.

### Optimal Quotes (from HJB first-order condition)

$$\delta_t^{a*} = \frac{1}{\alpha} + Z_t^q, \qquad \delta_t^{b*} = \frac{1}{\alpha} - Z_t^q$$

The neural network learns $Z_t^q$ (the inventory gradient of the value function), and optimal quotes emerge directly from this gradient --- recovering the Avellaneda-Stoikov (2008) result as a special case.

### BSDE Generator

$$f(t,x,y,z) = -ry - \psi(q) + \lambda^a f_a(\delta^{a*})\delta^{a*} + \lambda^b f_b(\delta^{b*})\delta^{b*}$$

### Inventory Penalties (configurable)

| Type | Formula | Monotonicity |
|------|---------|-------------|
| Quadratic | $\psi(q) = \phi q^2$ | Preserved |
| Cubic | $\psi(q) = \phi q^2 + \frac{\phi}{3}\|q\|^3$ | Broken |
| Exponential | $\psi(q) = \phi(e^{\gamma\|q\|} - 1)$ | Severely broken |

## Installation

```bash
git clone https://github.com/cgarryZA/DeepBSDE.git
cd DeepBSDE
pip install torch numpy matplotlib
```

For GPU support (recommended for stress tests):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Quick Start

**Train the continuous LOB solver:**
```bash
python main.py --config configs/lob_d2.json --exp_name lob_demo --log_dir ./logs --device auto
```

**Generate thesis-quality plots:**
```bash
python plot_lob.py --config configs/lob_d2.json \
    --result logs/lob_demo_result.txt \
    --weights logs/lob_demo_model.pt \
    --out_dir plots/demo
```

**Run the full experiment suite** (A-S benchmark, mean-field, N-particle scaling, phi stress tests):
```bash
python run_experiments.py           # full suite (~1 hour)
python run_experiments.py --quick   # debug mode (~10 min)
```

**Find the solver's breaking point** (binary search for gradient explosion):
```bash
python find_breaking_point.py --param gamma --lo 0.1 --hi 5.0 --penalty exponential
```

## Repository Structure

```
DeepBSDE/
├── main.py                          # Entry point
├── solver.py                        # All model + solver classes
├── config.py                        # Configuration dataclasses
├── registry.py                      # Equation registration decorator
├── equations/
│   ├── base.py                      # Abstract base class
│   ├── sinebm.py                    # Sine-BM benchmark (Han-Hu-Long 2022)
│   ├── flocking.py                  # Cucker-Smale MFG
│   ├── contxiong_lob.py             # Cont-Xiong LOB (Option A: continuous)
│   └── contxiong_lob_jump.py        # Cont-Xiong LOB (Option B: FBSDEJ)
├── configs/
│   ├── lob_d2.json                  # Main LOB config (Type 3 mean-field)
│   ├── lob_d2_no_competition.json   # A-S benchmark (Type 1, no mean-field)
│   └── lob_d2_jump.json             # Jump-diffusion config
├── plot_lob.py                      # Visualization suite (9 plot types)
├── run_experiments.py               # Full experiment runner
└── find_breaking_point.py           # Automated stability threshold finder
```

## Configuration

Key parameters in `configs/lob_d2.json`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_s` | 0.3 | Mid-price volatility |
| `lambda_a`, `lambda_b` | 1.0 | Order arrival rates |
| `alpha` | 1.5 | Execution probability decay |
| `phi` | 0.01 | Inventory penalty coefficient |
| `discount_rate` | 0.1 | Discount rate $r$ |
| `penalty_type` | `"quadratic"` | `"quadratic"`, `"cubic"`, or `"exponential"` |
| `type` | 3 | 1 = no coupling, 3 = mean-field fictitious play |
| `num_time_interval` | 50 | Euler-Maruyama time steps |

## Plots Generated

| Plot | Description |
|------|-------------|
| `convergence.png` | Loss and $Y_0$ vs training step |
| `z_max_evolution.png` | max $\|Z_t\|$ over training (Lipschitz diagnostic) |
| `quoting_strategy.png` | Optimal spread vs inventory |
| `spread_heatmap.png` | Spread surface over (time, inventory) |
| `value_surface_3d.png` | 3D value function $V(t, q)$ |
| `z_gradient_surface_3d.png` | 3D gradient surface $Z_t^q(t, q)$ |
| `sample_paths.png` | Price and inventory trajectories |
| `inventory_distribution.png` | Terminal inventory distribution |
| `value_function.png` | $V(q)$ cross-section with theory comparison |

## Citation

```bibtex
@misc{garry2026deepbsdelob,
  author       = {Christian Garry},
  title        = {Deep Mean-Field {BSDE} Methods for Optimal Market-Making
                  in Limit Order Books},
  year         = {2026},
  howpublished = {\url{https://github.com/cgarryZA/DeepBSDE}},
}
```

## References

- Cont, R. & Xiong, W. (2024). Dynamics of market making algorithms in dealer markets. *Mathematical Finance*, 34:467--521.
- Han, J., Hu, R. & Long, J. (2022). Learning high-dimensional McKean-Vlasov forward-backward SDEs. *SIAM J. Numer. Anal.*
- Han, J., Jentzen, A. & E, W. (2018). Solving high-dimensional PDEs using deep learning. *PNAS*, 115(34):8505--8510.
- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3):217--224.
- Carmona, R. & Delarue, F. (2018). *Probabilistic Theory of Mean Field Games with Applications I & II*. Springer.
- Pardoux, E. & Peng, S. (1990). Adapted solution of a backward stochastic differential equation. *Systems & Control Letters*.

## License

MIT

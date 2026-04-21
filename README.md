# MFG-BSDE-Equilibrium

Deep BSDE solvers and multi-agent reinforcement learning for the Cont-Xiong (2024)
dealer market-making model. MSc dissertation repo.

## Overview

Validates deep BSDE methods against a game with exact benchmarks (Cont-Xiong
Algorithm 1) and extends to:
- **BSDEJ** with jumps (Wang et al. 2023 deep BSDE for jump processes)
- **Continuous inventory** (diffusion approximation with Z ≠ 0)
- **Multi-asset** (curse of dimensionality regime)
- **Heterogeneous agents** (asymmetric risk aversion)
- **Common noise** (systematic price shocks → conditional MV-BSDE)
- **Tacit collusion** (decentralised MADDPG at N=2 and N=5)

## Structure

```
.
├── solver_cx.py                     Neural Bellman solver (discrete inventory, Z=0)
├── solver_cx_bsdej.py               BSDEJ solver — per-timestep networks
├── solver_cx_bsdej_shared.py        BSDEJ solver — shared weights + warm-start (main)
├── solver_cx_bsde_diffusion.py      Continuous inventory deep BSDE (Z ≠ 0)
├── solver_cx_multiagent.py          MADDPG trainer (CX Section 6)
├── solver_cx_multiasset.py          Multi-asset extension (K assets)
├── equations/contxiong_exact.py     CX model definition
├── scripts/
│   ├── cont_xiong_exact.py          Exact Algorithm 1 (linear algebra ground truth)
│   ├── cont_xiong_pareto.py         Pareto optimum (collusion upper bound)
│   ├── heterogeneous_agents.py      Asymmetric phi game
│   ├── common_noise.py              CX with dW_S price shock
│   ├── q_scaling_direct_v.py        Tabular V via L-BFGS
│   ├── nonstationary_phi.py         Time-varying risk aversion
│   ├── bsdej_n_scaling.py           BSDEJ at N=2,5,10,20,50
│   ├── compensated_martingale_ablation.py  Wang et al. fix ablation
│   ├── maddpg_analysis.py           Post-hoc statistics on MADDPG runs
│   └── generate_paper_figures.py    Figure pipeline
├── tests/                           Sanity test suite (pytest)
├── report/                          LaTeX sources
├── cluster/                         SLURM scripts for Hamilton
├── results_final/                   Validated results (keep)
├── results_cluster/                 MADDPG cluster output
└── archive/                         Old runners and stale results
```

## Running tests

Always run tests before kicking off long experiments:

```bash
./run_tests.sh          # fast tests (~1 min)
./run_tests.sh --slow   # full suite with training accuracy checks (~5 min)
./run_tests.sh -k bsdej # only BSDEJ tests
```

## Running experiments

```bash
# Exact solver (ground truth, fast)
python scripts/cont_xiong_exact.py

# Neural Bellman at Q=5 (standard)
python main.py --config configs/cx_bellman_q5.json

# BSDEJ with warm-start (main deep BSDE result)
python solver_cx_bsdej_shared.py

# Extensions
python scripts/heterogeneous_agents.py
python scripts/common_noise.py
python scripts/q_scaling_direct_v.py
```

## Key results

| Result | Value | Source |
|--------|-------|--------|
| Neural Bellman spread error at Q=5 | 0.59% | `solver_cx.py` |
| Direct-V spread error at Q=5 (with boundary fix) | 0.0001% | `scripts/q_scaling_direct_v.py` |
| BSDEJ shared + warmstart error | 2.6% | `solver_cx_bsdej_shared.py` |
| Compensated jump martingale fix | 264% → 2.7% | `scripts/compensated_martingale_ablation.py` |
| Mean-field N→∞ rate | O(1/√N) confirmed to N=5000 | `scripts/cont_xiong_exact.py` |
| MADDPG tacit collusion (20 seeds, N=2) | 15/20 above Nash, p=0.002 | `results_cluster/` |
| MADDPG N=5 collusion | 5/5 above Nash | `results_final/maddpg_N5.json` |

## Requirements

```
python >= 3.10
torch >= 2.0
numpy, scipy
pytest (for tests)
```

See `environment.yml` for the full conda spec.

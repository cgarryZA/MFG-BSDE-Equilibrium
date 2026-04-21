# DeepMVBSDEJ — A Mathematical Course

A math-first Jupyter course that builds from Brownian motion up to multi-agent Nash equilibria in limit-order-book market making, using the [DeepMVBSDEJ](../README.md) codebase as the running example.

## Audience

Intended for readers with undergraduate probability and calculus who want to understand — at a graduate-thesis level of rigour — how the repo works, **and more importantly, the mathematics underneath it**. The code is explained thoroughly, but each concept is introduced through formal mathematics first.

## Format

Every concept is presented as a **three-layer translation**:

| Layer | What it is | Why it's there |
|-------|-----------|----------------|
| 🔷 **Math** | Formal notation, definitions, theorems | The ground truth |
| 📝 **Plain English** | Intuition that bridges math and code | Teaches *what's happening* |
| 💻 **Code** | Exact unmodified snippet from the repo | Shows *where it lives* |

The plain-English layer does the heavy lifting of translating between vocabularies. Exercises throughout have collapsible solutions.

## Module arc

| # | Module | Key math | Core files it unlocks |
|---|--------|----------|----------------------|
| 1 | [Brownian motion & Itô calculus](01_brownian_motion.ipynb) | SDEs, Itô formula, Euler-Maruyama | `equations/contxiong_lob.py:9-10` |
| 2 | BSDEs | Y/Z decomposition, martingale representation, existence | `equations/base.py`, generator `f` |
| 3 | BSDEs with jumps (BSDEJ) | Compensated Poisson, generator with $U$-term | `solver_cx_bsdej.py` |
| 4 | McKean-Vlasov & mean-field coupling | Law dependence, propagation of chaos | `equations/law_encoders.py` |
| 5 | Cont-Xiong LOB model | Execution intensity, HJB FOC, quote derivation | `equations/contxiong_lob*.py`, `solver.py` |
| 6 | Deep BSDE numerics | Han–Jentzen–E, BatchNorm erasure, DeepSets collapse | `solver.py`, `archive/FINDINGS.md` |
| 7 | Multi-agent Nash | MADDPG, Nash vs Pareto, tacit collusion | `solver_cx_multiagent.py` |

## How to run

```bash
cd course
jupyter lab 01_brownian_motion.ipynb
```

Each notebook is self-contained. Running order is recommended but not enforced — you can dip in anywhere if you already know the prerequisites.

## Rebuilding

The notebooks are generated from Python builders in `_build/` for maintainability. To regenerate Module 1:

```bash
python course/_build/build_module_01.py
```

Edits made directly in Jupyter will be **lost** on rebuild. For permanent changes, edit the builder.

## Dependencies

`numpy`, `matplotlib`, `scipy`, `jupyter`. The `torch` dependency (from the main repo) is only needed for modules 6–7 where we actually run solver code.

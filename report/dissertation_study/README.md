# Dissertation Study Draft

A **long-form explanatory draft** of the dissertation *Numerical Analysis of Mean-Field BSDEs: Deep Learning Solutions for Interacting Agents in Limit Order Books*.

## What this is

- A **study companion** to the actual dissertation. Where a submitted dissertation is terse, this draft spells out every derivation step, gives intuition before and after each theorem, and cross-references the repo code.
- Modelled on the pedagogical style of [`course/`](../../course/) — three-layer thinking (math / plain English / code), lots of derivation steps, no jumps in reasoning.
- **Not** the submission draft. Think of it as lecture notes I can later compress into the actual dissertation.

## Structure

| File | Corresponds to |
|------|---------------|
| `main.tex` | Wrapper — preamble, packages, `\input` each chapter |
| `ch01_introduction.tex` | Chapter 1: Motivation, contributions, outline |
| `ch02_mathematical_framework.tex` | Chapter 2: BSDEs, BSDEJ, nonlinear Feynman–Kac, mean-field, CX model |
| `ch03_deep_bsde_methods.tex` | Chapter 3: Han–Jentzen–E, MV-FBSDE, jump extensions, architecture |
| `ch04_neural_bellman_solver.tex` | Chapter 4: stationary solver, validation, Q-scaling, continuous inventory |
| `ch05_nash_and_collusion.tex` | Chapter 5: Nash, Pareto, MADDPG, tacit collusion |
| `ch06_conclusion.tex` | Chapter 6: summary, limitations, future work |
| `appendices.tex` | Proofs, implementation details |

## Building

```bash
cd report/dissertation_study
pdflatex main.tex
pdflatex main.tex   # second pass for refs
```

Requires a standard TeX distribution (TeX Live, MiKTeX). Uses only core packages (amsmath, amsthm, amssymb, hyperref, graphicx, booktabs).

## Relation to other documents

- [`../dissertation_outline.md`](../dissertation_outline.md) — the chapter skeleton this draft fleshes out.
- [`../preprint/main.tex`](../preprint/main.tex) — the submitted companion paper on architectural diagnostics (Chapter 3 material, compressed).
- [`../bsde_connection.tex`](../bsde_connection.tex) — the note establishing that the CX model is a BSDEJ; incorporated into Chapter 2.
- [`../../course/`](../../course/) — the seven-module Jupyter course; the same story in interactive form.

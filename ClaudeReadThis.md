# ClaudeReadThis — Handoff notes

**Date frozen:** 2026-04-21. Pick this up from any machine with the repo cloned.

This file records what was decided, what was built, what we still need to do, and — critically — the framing we committed to for the dissertation and positioning. Read this first when you resume.

---

## 1. Project context (not always obvious from the code)

- **Dissertation title:** *Numerical Analysis of Mean-Field BSDEs: Deep Learning Solutions for Interacting Agents in Limit Order Books*
- **Programme:** Durham MSc (Mathematical Sciences)
- **Deadline:** September 2026 (~4–5 months of runway from this note)
- **Intended leverage:** Jane Street quant-research application, Oxford financial-maths PhD application
- **Reader framing implied by those goals:** not just an external examiner — also quant-research screeners and a PhD admissions panel who will scan for one defensible isolated contribution + citable preprint

---

## 2. The framing we committed to (read twice)

ChatGPT's review nudged toward retitling everything around "failure modes of deep BSDE." **We rejected that.** The lit review already commits to a different novelty claim and we stick to it.

### The one-sentence thesis

> **"This work provides the first validated application of deep BSDE methods to the Cont–Xiong model, and shows that doing so requires resolving previously unrecognised architectural failures in mean-field settings."**

If a reviewer pushes back on *"previously unrecognised"* (because BatchNorm-on-broadcast-features is noted in conditional-GAN literature, and DeepSets mean-pool limits are known in permutation-invariance work), the safe fallback is *"not previously surfaced in mean-field BSDE solvers"* or *"previously undiagnosed in the deep-BSDE literature."* Same meaning, tighter scope.

### The three-contribution shape

The novelty is the **combination**. The failure modes are **what made the combination non-trivial**, not a competing contribution. The three enriching elements:

1. **BSDEJ formulation of Cont–Xiong.** CX isn't obviously a BSDE until the compensated-jump derivation identifies $U^a, U^b$ with the discrete gradient of $V$. Theoretical bridge — see [report/bsde_connection.tex](report/bsde_connection.tex) and dissertation study Ch 2 §2.7.5.
2. **Architectural findings.** Three failure modes that the naive deep BSDE solver hits silently in MV settings: generator bypass, BatchNorm erasure of broadcast features, DeepSets symmetric-pooling collapse. These are what made the combination genuinely research rather than engineering.
3. **Validation against Algorithm 1.** Algorithm 1 of Cont–Xiong gives exact Nash; we hit it to 0.79% spread / 1.28% V at $N=2$, <1% at $N \in \{5, 10\}$. This proves the combination is non-vacuous.

**Positioning for different audiences:**

| Audience | Lead with | Mention as |
|---|---|---|
| Examiner / dissertation | The combination (matches lit review gap) | Failure modes as Ch 3 scaffolding |
| arXiv preprint | Failure modes (sharp isolated claim) | Positioned as *diagnostic emerged from* CX application |
| Jane Street / PhD cover letter | The combination | "I identified structural issues while building X" |

---

## 3. Repo state we're handing off

### Recent history
- `d3f5d11` (mine) added the 7-module Jupyter course + 2 regenerated plots from committed data
- `51a101e` (mine) fixed the dissertation-study bibliography (all 59 cites now resolve)
- `8f579a7` (mine) created the 49-page dissertation study draft
- `f94e478` (yours, from another machine) reorganised the repo: moved `FINDINGS.md` → `archive/`, moved `results_cx_*` → `archive/old_results/`, updated main README, added tests/, many new solvers and scripts

**The main README is authoritative.** FINDINGS.md (now at `archive/FINDINGS.md`) is an interim log; trust the README key-results table over it.

### Authoritative numbers (from main README)

| Result | Value | Source |
|---|---|---|
| Neural Bellman spread error at Q=5 | **0.59%** | `solver_cx.py` |
| Direct-V spread error (boundary-patched) at Q=5 | **0.0001%** | `scripts/q_scaling_direct_v.py` |
| BSDEJ shared + warmstart error | **2.6%** | `solver_cx_bsdej_shared.py` |
| Compensated-martingale fix | 264% → 2.7% | `scripts/compensated_martingale_ablation.py` |
| Mean-field N→∞ rate | O(1/√N) confirmed to N=5000 | `scripts/bsdej_n_scaling.py` |
| MADDPG tacit collusion (N=2, 20 seeds) | **15/20 above Nash, p=0.002** | `results_cluster/` |
| MADDPG N=5 collusion | **5/5 above Nash** | `results_final/maddpg_N5.json` |

### Benchmark spreads (q=0, from `archive/old_results/results_cx_exact/`)

| Regime | Total spread | Half-spread per side |
|---|---:|---:|
| Nash (N=2) | **1.478** | 0.739 |
| Monopolist (N=1) | **1.593** | 0.797 |
| Pareto (N=2 cartel) | **1.636** | 0.818 |

**Ordering: Nash < Monopolist < Pareto.** This matters — they're three distinct values, not two.

### What I built / what's in the repo now

| Artifact | Path | Purpose |
|---|---|---|
| Jupyter course (7 modules) | [course/](course/) | Math-first walkthrough of the whole story, from Brownian motion → multi-agent Nash |
| Course builders | [course/_build/](course/_build/) | Python generators; edit content and rebuild |
| Dissertation study draft | [report/dissertation_study/main.pdf](report/dissertation_study/main.pdf) | 51-page explanatory long-form draft (NOT the submission dissertation) |
| Dissertation study sources | [report/dissertation_study/](report/dissertation_study/) | `main.tex` + `chNN_*.tex` + `references.bib` |
| Regenerated validation plots | [plots_cx/regenerated/](plots_cx/regenerated/) | Nash/Pareto spreads + neural-validation, built from committed JSON |
| Companion preprint | [paper.pdf](paper.pdf) | Architectural diagnostics paper (from April 3 — content still accurate, but the repo has evolved past it) |
| Full dissertation outline | [report/dissertation_outline.md](report/dissertation_outline.md) | The chapter plan the dissertation study expands on |
| Theoretical connection note | [report/bsde_connection.tex](report/bsde_connection.tex) | The formal CX ↔ BSDEJ derivation |
| Lit review revised sections | [report/lit_review_revised_sections.tex](report/lit_review_revised_sections.tex) | **Sections 4–5 only** — Sections 1–3 are NOT in this repo |

**Important: the "dissertation study draft" is not the submission dissertation.** It's a deliberately verbose explanatory companion — full derivations, intuition paragraphs, no cuts. The actual dissertation (when written) should be the same content compressed 30–40%.

---

## 4. Issues to fix before committing to anything

These are concrete, specific, and independent of the framing debate. Every one is a real claim–evidence gap.

### 4.1 Pareto ≠ Monopolist (critical)

My dissertation study Ch 5 conflated the two. The lit review correctly treats them as distinct; the data supports three values (1.478 / 1.593 / 1.636). Fix:

- [ ] Update [report/dissertation_study/ch05_nash_and_collusion.tex](report/dissertation_study/ch05_nash_and_collusion.tex) to show three distinct rows in the benchmark table
- [ ] Regenerate [plots_cx/regenerated/nash_pareto_spreads.png](plots_cx/regenerated/nash_pareto_spreads.png) to include the Pareto bar (currently shows only Monopolist + Nash N=2 + Nash N=5; Pareto N=2 is in `archive/old_results/results_cx_exact/pareto_N2.json` with `spread_q0 = 1.636`)
- [ ] If the dissertation-outline.md or preprint also conflates, fix those too

### 4.2 Info-structure experiment (lit-review risk)

The lit review (revised Section 4) claims a contribution: *"the information structure is varied from full information (converges to Nash) to no information (converges to monopolist level)."* I couldn't find a script that runs this sweep. Options:

- [ ] **Option A:** Run the experiment (short MADDPG sweep varying the observation mask). 2–4 days of work if the infrastructure in `solver_cx_multiagent.py` supports masked observations; longer if not.
- [ ] **Option B:** Demote to Open Problems / Future Work. Remove the contribution claim from the lit review revised Section 4 and the dissertation.

Do one of these before committing the lit review. An examiner who asks "show me the full-info-converges-to-Nash plot" will need either a plot or a clean answer.

### 4.3 "Algorithm 0" reference

Lit review revised Section 4 says Cont–Xiong have an *"Algorithm 0"* for the Pareto optimum. I'm not confident that label appears in the paper. Before committing:

- [ ] Open Cont–Xiong (2024) and confirm the Pareto algorithm is labelled Algorithm 0, or rephrase as "the Pareto-optimum solver derived from joint-welfare maximisation"

### 4.4 Lit review Sections 1–3

The revised file only contains Sections 4–5. Sections 1–3 ("BSDE theory, numerical methods, MFG/MV-FBSDE") are said to "remain as-is" but I couldn't find them in this repo or the three other repos I checked (`Deep-Learning-Solutions-for-Interacting-Agents-in-Limit-Order-Books`, `DeepBSDE-LOB`, `DeepMVBSDEJ`, `Dissertation`). Probably on Overleaf or a different machine.

- [ ] Locate Sections 1–3 (Overleaf? OneDrive? supervisor-shared folder?)
- [ ] Bring them into `report/lit_review/` and wire up a `main.tex` so the whole thing compiles to one PDF
- [ ] Verify Section 3 (MFG / MV-FBSDE) mentions Han–Hu–Long (2022) fictitious play, since you rely on it

### 4.5 Citation verification

Several recent citations in the revised lit review I couldn't independently verify. Before the bibliography is frozen, check:

- [ ] `wang2023` — does a "deep learning methods for high-dimensional fully nonlinear PIDEs and FBSDEJs" paper exist with that attribution?
- [ ] `lu2024` — multi-agent jump-diffusion relative investment
- [ ] `deng2025` — Bertrand oligopoly RL collusion study
- [ ] `wang2025` — MARL market making with adaptive incentive control
- [ ] `hu2024survey` — ML for stochastic control and games
- [ ] `carmona2018a` — conditional distribution / common noise reference

If any are misattributed, fix before submission.

---

## 5. Lit review assessment (from 2026-04-21 review)

**Committable with the fixes above.** Scope is appropriate for a strong Masters — on the upper end (four contributions), which is fine if each is modest.

### Verdict on each contribution

| Contribution | Masters-appropriate? | Notes |
|---|---|---|
| (C1) CX as BSDEJ | ✅ | Clean theorem-level observation, derivation fits in a chapter |
| (C2) Validated solver | ✅ **Core contribution** | Strongest piece; lead with this |
| (C3) Pareto + Nash ordering | ⚠️ | Thin on its own — frame as part of (C2)'s validation scaffolding |
| (C4) MADDPG tacit collusion | ✅ | Reproduction with statistical rigour (the 15/20, p=0.002) |

If the examiner asks "what's novel", lean hardest on (C2) with (C1) as its theoretical foundation. (C4) is a careful reproduction; (C3) is scaffolding.

### What lit-review readers at target audiences see

- **Examiner:** gap statement ("no work applies deep BSDE to CX") is defensible, four contributions are logically ordered, open-problems section matches actual WIP. Clean.
- **Quant-research screener / PhD panel:** One citable claim, a validated benchmark, a statistical significance test, recent-currency citations. Strong but not overhyped.

---

## 6. Plan for next ~4 months

Revised after the framing discussion: extract the preprint FIRST, then wrap the dissertation around it. Rationale: a dated arXiv preprint is a tangible deliverable you can cite in Jane Street / Oxford applications; a single-examiner dissertation is not.

### Phase 1 — Lock the thesis + fix claim gaps (now → mid-May)

- [ ] Lock the one-sentence thesis (§2 above)
- [ ] Fix the Pareto/Monopolist conflation in Ch 5 and anywhere else
- [ ] Decide info-structure: run or demote
- [ ] Verify "Algorithm 0" reference
- [ ] Locate lit-review Sections 1–3 and consolidate
- [ ] Verify citations

### Phase 2 — Preprint extraction (mid-May → June)

Goal: a standalone arXiv submission built around the three-failure-modes findings, framed as a diagnostic study that emerged from the broader CX application (not as a pure ML paper that pretends the CX context doesn't exist).

- [ ] Start from [paper.pdf](paper.pdf) source ([report/preprint/main.tex](report/preprint/main.tex)) as the base
- [ ] Tighten: 4× h variation, 2× quote shift, sign-flip of skew as the headline figure
- [ ] Promote the three failure modes to propositions with statements and short proofs (not narrative prose)
- [ ] Run one controlled architecture-sweep experiment and produce an encoder-comparison flagship figure (sharpens existing Finding 5 — don't invent new results)
- [ ] Post to arXiv
- [ ] Keep the file path [paper.pdf](paper.pdf) updated in the repo so both versions stay consistent

### Phase 3 — Dissertation structure (July)

- [ ] Rebuild dissertation around: problem → method → failure modes → fix → validated results
- [ ] Integrate the preprint as Chapter 3 (failure modes) — don't duplicate, cite
- [ ] Cut 30–40% of exposition from the study draft when transplanting into the submission
- [ ] Finalise Chapter 5 (Nash + tacit collusion) with the statistical test written up properly

### Phase 4 — Polish + second output (August → early Sept)

- [ ] Finalise dissertation
- [ ] Optional: second preprint on the CX solver itself (C2, the validated neural Bellman). Frame as the applied paper that the failure-modes preprint cites.
- [ ] Prepare one-page summary for Jane Street / Oxford application packets

---

## 7. Opinions I formed along the way

Keeping these so I don't forget them when resuming:

- **ChatGPT's review** is ~80% directionally correct once you factor in the career framing. The 20% that's wrong: (a) it reviewed the *study draft* as the submission (different documents), (b) it advised retitling around failure modes (which the lit review already rejected), (c) it proposed a "killer table" that already exists as FINDINGS.md §3–§5 (surface, don't invent).
- **"First validated application"** is the right phrasing because validation against Algorithm 1 is the hard part. Anyone can claim "first application"; "first *validated* application" is anchored to the benchmark.
- **"Previously unrecognised"** is defensible in the deep-BSDE context but softer to say "not previously surfaced in mean-field BSDE solvers" if challenged.
- **Don't split prematurely.** The preprint pulls from the dissertation's architectural chapter. Write the shared machinery once, then extract.
- **Finding 5 is the flagship.** δ_a at q=0 goes from 0.696 (narrow population) to 0.334 (wide) — that's a 2× shift that visibly propagates through a four-line FOC. Make it a figure.
- **The repo has more experimental branches than the dissertation uses.** `solver_cx_bsde_diffusion.py`, `heterogeneous_agents.py`, `common_noise.py`, `multiasset_K5_CoD.py` are exploration. Don't let them creep into the scope unless one produces a result clean enough to include.

---

## 8. Quick-reference pointers

```
MFG-BSDE-Equilibrium/
├── README.md                                     authoritative summary + key results
├── paper.pdf                                     companion preprint (architectural diagnostics)
├── ClaudeReadThis.md                             ← you are here
├── course/                                       7-module Jupyter course
│   ├── README.md
│   ├── 01_brownian_motion.ipynb ... 07_multi_agent_nash.ipynb
│   └── _build/                                   notebook generators
├── report/
│   ├── dissertation_study/                       51-page explanatory LaTeX draft
│   │   ├── main.pdf                              ← read this for the long version
│   │   ├── main.tex, ch01-06_*.tex, appendices.tex
│   │   └── references.bib
│   ├── dissertation_outline.md                   chapter plan
│   ├── preprint/main.tex                         source of paper.pdf (architectural)
│   ├── bsde_connection.tex                       CX ↔ BSDEJ derivation
│   └── lit_review_revised_sections.tex           sections 4–5 only (find 1–3)
├── solver_cx.py                                  neural Bellman
├── solver_cx_bsdej_shared.py                     BSDEJ with warm-start (main deep BSDE)
├── solver_cx_multiagent.py                       MADDPG
├── equations/contxiong_exact.py                  exact CX fixed-point
├── scripts/
│   ├── cont_xiong_exact.py                       Algorithm 1 (Nash)
│   ├── cont_xiong_pareto.py                      Pareto solver
│   ├── compensated_martingale_ablation.py        264% → 2.7% fix ablation
│   ├── maddpg_analysis.py                        post-hoc stats
│   └── q_scaling_direct_v.py                     0.0001% direct-V solver
├── cluster/                                      SLURM scripts for Hamilton (MADDPG 20-round)
├── tests/                                        pytest suite — run before long experiments
├── archive/
│   ├── FINDINGS.md                               interim findings log (superseded by README)
│   └── old_results/                              committed result JSONs (incl. nash_N2, pareto_N2)
├── plots/                                        MV-BSDE plots (encoder comparison etc.)
└── plots_cx/
    ├── *.png, *.gif                              MADDPG + CX animations
    └── regenerated/                              plots rebuilt from committed data
```

### How to rebuild the dissertation study PDF

```bash
cd report/dissertation_study
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### How to rebuild the Jupyter course

```bash
for n in 01 02 03 04 05 06 07; do python course/_build/build_module_${n}.py; done
```

### How to regenerate the two validation plots

Python snippet in the earlier conversation — reads `archive/old_results/results_cx_exact/*.json` and `archive/old_results/results_cx_validation/validation.json`. Re-runnable on any machine.

---

## 9. First thing to do when resuming

1. Read §2 (the framing) again — this is the decision that changes everything downstream.
2. Run through the Phase 1 checklist in §6.
3. Decide whether to invest 2–4 days running the info-structure experiment or demote it.
4. Start the preprint extraction once Phase 1 is done — don't wait for the full dissertation.

Good luck.

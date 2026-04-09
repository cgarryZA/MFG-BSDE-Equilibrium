#!/usr/bin/env python
"""
Run AFTER the overnight FP suite completes.
Covers items 6-7 from the roadmap:
  6. Non-linear market impact phase transition
  7. Document stability boundary

Chain this after run_overnight.py:
  python scripts/run_overnight.py && python scripts/run_after_overnight.py

Or just run standalone if overnight is already done.

Estimated: ~2 hours on RTX 3090 (24 training runs × ~5 min each)
"""

import subprocess
import sys
import time
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Started: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = (time.time() - t0) / 60
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{status}] {label} — {elapsed:.1f} min\n")
    return result.returncode


def main():
    start = time.time()
    print(f"Starting post-overnight experiments at {time.strftime('%H:%M:%S')}")

    # Phase transition: kappa × impact_type sweep
    run(
        f"{sys.executable} scripts/run_impact_phase_transition.py --device cuda --n_iters 2000",
        "Phase transition: non-linear price impact (24 runs)"
    )

    total = (time.time() - start) / 60
    print(f"\n{'='*60}")
    print(f"  ALL DONE — {total:.1f} min")
    print(f"  Results: results_impact/phase_transition.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

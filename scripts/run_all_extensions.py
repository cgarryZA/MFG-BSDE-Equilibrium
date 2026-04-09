#!/usr/bin/env python
"""
Run ALL extension experiments sequentially:
1. FiLM ablation + interaction test (~80 min)
2. Fictitious play with two_stream (~30 min)
3. Fictitious play with film (~30 min)

Total: ~2.5 hours. Set it running and go to the gym.
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
    print(f"Starting all extension experiments at {time.strftime('%H:%M:%S')}")
    print(f"Estimated completion: ~2.5 hours\n")

    # 1. FiLM experiments (ablation + interaction + h-only)
    run(
        f"{sys.executable} scripts/run_film_experiments.py --device cuda",
        "FiLM ablation + interaction test (3000 iter)"
    )

    # 2. FP with two_stream (10 outer × 2000 inner)
    run(
        f"{sys.executable} scripts/run_fictitious_play.py --device cuda --quick --subnet_type two_stream --outer 10 --inner 1000",
        "Fictitious play — two_stream (10 outer × 1000 inner)"
    )

    # 3. FP with film (10 outer × 2000 inner)
    run(
        f"{sys.executable} scripts/run_fictitious_play.py --device cuda --quick --subnet_type film --outer 10 --inner 1000",
        "Fictitious play — film (10 outer × 1000 inner)"
    )

    total = (time.time() - start) / 60
    print(f"\n{'='*60}")
    print(f"  ALL DONE — {total:.1f} min total")
    print(f"  Finished: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    print(f"\nResults saved to:")
    print(f"  results_film/film_experiments.json")
    print(f"  results_fp/fp_two_stream.json")
    print(f"  results_fp/fp_film.json")


if __name__ == "__main__":
    main()

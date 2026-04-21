#!/usr/bin/env python
"""Wait for full suite to finish, then run fixes."""
import os, sys, time, subprocess
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SENTINEL = "results_full_suite/summary.json"
print(f"[{time.strftime('%H:%M:%S')}] Waiting for full suite (watching {SENTINEL})...")

while not os.path.exists(SENTINEL):
    time.sleep(30)

print(f"[{time.strftime('%H:%M:%S')}] Full suite done. Starting fixes...")
time.sleep(10)  # brief pause to let GPU release

result = subprocess.run([sys.executable, "scripts/run_fixes.py"])
print(f"[{time.strftime('%H:%M:%S')}] Fixes done. Exit: {result.returncode}")

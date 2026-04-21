#!/usr/bin/env python
"""
Wait for overnight suite to finish, then run impact experiments.
Checks every 60 seconds for the all_overnight_results.json file.
"""

import os
import sys
import time
import subprocess

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SENTINEL = "results_overnight/all_overnight_results.json"

print(f"[{time.strftime('%H:%M:%S')}] Waiting for overnight suite to finish...")
print(f"  Watching for: {SENTINEL}")

while not os.path.exists(SENTINEL):
    time.sleep(60)

print(f"\n[{time.strftime('%H:%M:%S')}] Overnight suite complete! Starting impact experiments...")

result = subprocess.run([sys.executable, "scripts/run_after_overnight.py"])

print(f"\n[{time.strftime('%H:%M:%S')}] All done. Exit code: {result.returncode}")

#!/usr/bin/env python
"""Wait for current MADDPG to finish, then run overnight suite."""
import os, sys, time, subprocess
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Wait for GPU to be free (no other python using >500MB)
import psutil
print(f"[{time.strftime('%H:%M:%S')}] Waiting for GPU to be free...")
while True:
    python_procs = [p for p in psutil.process_iter(['name', 'memory_info'])
                    if p.info['name'] and 'python' in p.info['name'].lower()
                    and p.pid != os.getpid()
                    and p.info['memory_info'] and p.info['memory_info'].rss > 500_000_000]
    if not python_procs:
        break
    time.sleep(30)

print(f"[{time.strftime('%H:%M:%S')}] GPU free. Starting overnight suite...")
time.sleep(5)
result = subprocess.run([sys.executable, "scripts/run_cx_overnight.py"])
print(f"[{time.strftime('%H:%M:%S')}] Done. Exit: {result.returncode}")

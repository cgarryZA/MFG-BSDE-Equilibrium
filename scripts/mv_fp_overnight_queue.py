#!/usr/bin/env python -u
"""Overnight queue runner for MV-FP BSDEJ variants.

Runs v4, v5, v6, v7, v8 sequentially as subprocesses.  Continues on
failure (each variant is independent), captures per-variant log,
saves status to results_final/mv_fp_overnight_status.json after each.

Variants:
  v4 — L2 penalty to warmstart weights
  v5 — re-warmstart per iter toward current avg_comp's Bellman
  v6 — FOC consistency penalty
  v7 — short inner (1500) + low damping (0.15)
  v8 — low LR (1e-4) + tight grad clip (1.0)

Run: python -u scripts/mv_fp_overnight_queue.py
"""

import sys, os, json, time, subprocess, gc
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

STATUS_FILE = "results_final/mv_fp_overnight_status.json"

VARIANTS = [
    ("v7", "scripts/mv_fp_bsdej_v7.py", 90),   # shortest first
    ("v4", "scripts/mv_fp_bsdej_v4.py", 180),  # L2 penalty - highest a priori prob
    ("v8", "scripts/mv_fp_bsdej_v8.py", 180),  # low LR + tight clip
    ("v6", "scripts/mv_fp_bsdej_v6.py", 200),  # FOC consistency
    ("v5", "scripts/mv_fp_bsdej_v5.py", 220),  # re-warmstart (slowest)
]


def run_one(name, script, timeout_sec):
    log_path = f"results_final/log_mv_fp_{name}.txt"
    print(f"\n{'#'*70}")
    print(f"### {name}  ({script})")
    print(f"### Started: {time.strftime('%Y-%m-%d %H:%M:%S')}, "
          f"timeout={timeout_sec/60:.0f}min")
    print(f"{'#'*70}", flush=True)
    t0 = time.time()
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                [sys.executable, "-u", script],
                stdout=logf, stderr=subprocess.STDOUT,
                timeout=timeout_sec * 60,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
        elapsed = time.time() - t0
        success = (proc.returncode == 0)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        success = False
        print(f"  TIMEOUT after {elapsed:.0f}s", flush=True)
    except Exception as e:
        elapsed = time.time() - t0
        success = False
        print(f"  UNEXPECTED ERROR: {e}", flush=True)

    tail = ""
    try:
        with open(log_path) as f:
            lines = f.readlines()
            tail = "".join(lines[-12:])
    except Exception:
        pass
    gc.collect()

    # Read the per-variant result JSON if present
    json_path = f"results_final/mv_fp_bsdej_{name}_N2.json"
    result_summary = {}
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                d = json.load(f)
            result_summary = {
                "final_spread": d.get("final_spread"),
                "final_error_pct": d.get("final_error_pct"),
                "final_avg_comp": d.get("final_avg_comp"),
                "n_outer_done": len(d.get("history", [])),
                "best_iter_spread": min(
                    (abs(h.get("spread_q0", 999) - d["nash_spread"]) / d["nash_spread"] * 100
                     for h in d.get("history", [])), default=None),
            }
        except Exception as e:
            result_summary = {"parse_error": str(e)}

    status = "OK" if success else "FAILED"
    print(f"  {status} in {elapsed:.0f}s", flush=True)
    return {
        "name": name, "script": script, "success": success,
        "elapsed": elapsed, "tail": tail, "result": result_summary,
    }


def main():
    os.makedirs("results_final", exist_ok=True)
    print(f"\n{'='*70}")
    print(f"OVERNIGHT QUEUE: {len(VARIANTS)} variants")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}", flush=True)

    results = []
    for name, script, timeout in VARIANTS:
        info = run_one(name, script, timeout)
        results.append(info)
        with open(STATUS_FILE, "w") as f:
            json.dump({"results": results,
                       "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, indent=2)

    print(f"\n{'='*70}")
    print(f"OVERNIGHT QUEUE COMPLETE")
    print(f"{'='*70}")
    for r in results:
        mark = "[OK]" if r["success"] else "[FAIL]"
        res = r["result"]
        spread = res.get("final_spread", "?")
        err = res.get("final_error_pct", "?")
        best = res.get("best_iter_spread", "?")
        print(f"  {mark} {r['name']:<4s}  {r['elapsed']:6.0f}s  "
              f"final_err={err}%  best_iter_err={best}%")
    print("", flush=True)


if __name__ == "__main__":
    main()

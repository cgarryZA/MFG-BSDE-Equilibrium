#!/usr/bin/env python -u
"""Option B orchestrator — runs all Tier 1/2 experiments robustly.

Runs each script in a subprocess, captures output+errors, continues on
failure, GPU-resets between runs, and restarts the pipeline once if any
script failed.

Saves per-script status to results_final/option_b_status.json after each
completion so you can monitor progress.
"""

import sys, os, json, time, subprocess, gc
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

STATUS_FILE = "results_final/option_b_status.json"

# Scripts ordered fastest-first (so results land early)
JOBS = [
    # name, script, estimated duration (min)
    ("monotonicity",    "scripts/monotonicity_check.py",          5),
    ("master_eq",       "scripts/master_equation_verification.py", 10),
    ("multiasset_CoD",  "scripts/multiasset_K5_CoD.py",           90),
    ("mv_fp_bsdej",     "scripts/mv_fp_bsdej.py",                180),
    ("conditional_mv",  "scripts/conditional_mv_common_noise.py", 120),
]


def gpu_reset_external():
    """Best-effort GPU cleanup (between subprocesses)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


def run_one(name, script, timeout_sec):
    """Run a single script as subprocess. Returns (success, elapsed, output_tail)."""
    log_path = f"results_final/log_option_b_{name}.txt"
    print(f"\n{'#'*70}")
    print(f"### {name}  ({script})")
    print(f"### Started: {time.strftime('%Y-%m-%d %H:%M:%S')}, timeout={timeout_sec/60:.0f}min")
    print(f"{'#'*70}", flush=True)

    t0 = time.time()
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                [sys.executable, "-u", script],
                stdout=logf, stderr=subprocess.STDOUT,
                timeout=timeout_sec,
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

    # Always GPU reset after each job
    gpu_reset_external()

    # Tail the log
    tail = ""
    try:
        with open(log_path) as f:
            lines = f.readlines()
            tail = "".join(lines[-10:])
    except Exception:
        pass

    status = "OK" if success else "FAILED"
    print(f"  {status} in {elapsed:.0f}s", flush=True)
    return success, elapsed, tail


def main():
    os.makedirs("results_final", exist_ok=True)

    # Pipeline: run all jobs, track failures, retry once if any failed
    MAX_ATTEMPTS = 2
    all_status = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n{'='*70}")
        print(f"PIPELINE ATTEMPT {attempt}/{MAX_ATTEMPTS}")
        print(f"{'='*70}", flush=True)

        attempt_status = []
        any_failed = False

        for name, script, est_min in JOBS:
            # Check if already completed (file exists and non-empty)
            result_file = None
            for candidate in [
                f"results_final/{name}.json",
                f"results_final/{name.replace('mv_', 'mv_')}.json",
            ]:
                if os.path.exists(candidate) and os.path.getsize(candidate) > 10:
                    result_file = candidate
                    break

            # For this first version, we always run (don't skip) but catch failures.
            success, elapsed, tail = run_one(name, script, est_min * 60 + 300)
            attempt_status.append({
                "attempt": attempt, "name": name, "script": script,
                "success": success, "elapsed": elapsed,
                "output_tail": tail,
            })
            if not success:
                any_failed = True

            # Save after each job
            with open(STATUS_FILE, "w") as f:
                json.dump({"all_status": all_status + attempt_status}, f, indent=2)

        all_status.extend(attempt_status)

        # Summary for this attempt
        ok = sum(1 for s in attempt_status if s["success"])
        total = len(attempt_status)
        print(f"\n{'='*70}")
        print(f"ATTEMPT {attempt} SUMMARY: {ok}/{total} succeeded")
        print(f"{'='*70}")
        for s in attempt_status:
            mark = "[OK]" if s["success"] else "[FAIL]"
            print(f"  {mark} {s['name']:<22s}  {s['elapsed']:6.0f}s")
        print("", flush=True)

        if not any_failed:
            print("All jobs passed. Pipeline complete.", flush=True)
            break
        elif attempt < MAX_ATTEMPTS:
            print(f"Some jobs failed. Retrying the whole pipeline (attempt {attempt+1}/{MAX_ATTEMPTS})...",
                  flush=True)
            gpu_reset_external()
            time.sleep(10)  # brief pause before retry
        else:
            print(f"Pipeline reached {MAX_ATTEMPTS} attempts. See individual logs.", flush=True)

    with open(STATUS_FILE, "w") as f:
        json.dump({"all_status": all_status, "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, indent=2)
    print(f"\nFinal status saved to {STATUS_FILE}", flush=True)


if __name__ == "__main__":
    main()

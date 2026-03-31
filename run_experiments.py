#!/usr/bin/env python
"""
Run all LOB experiments for the dissertation.

Experiments:
1. A-S benchmark (Type 1, no competition) — validates against known analytical solution
2. Full mean-field (Type 3) — the main result
3. N-particle scaling — tests 1/sqrt(N) convergence of mean-field approximation

Usage:
    python run_experiments.py [--quick]  # --quick for 500 iterations (debugging)
"""

import argparse
import json
import os
import copy
import subprocess
import sys


def make_config(base_path, overrides, out_path):
    """Create a modified config file from a base config."""
    with open(base_path) as f:
        config = json.load(f)

    for key_path, value in overrides.items():
        keys = key_path.split(".")
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    return out_path


def run(config_path, exp_name, log_dir="./logs"):
    """Run the solver and return the result file path."""
    cmd = [
        sys.executable, "main.py",
        "--config", config_path,
        "--exp_name", exp_name,
        "--log_dir", log_dir,
    ]
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")
    subprocess.run(cmd, check=True)
    return os.path.join(log_dir, f"{exp_name}_result.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Short runs for debugging")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs("configs/generated", exist_ok=True)

    num_iters = 500 if args.quick else 5000
    base_config = "configs/lob_d2.json"

    # ----------------------------------------------------------------
    # Experiment 1: Avellaneda-Stoikov benchmark (no competition)
    # ----------------------------------------------------------------
    as_config = make_config(base_config, {
        "eqn_config.type": 1,
        "eqn_config.couple_coeff": 0.0,
        "net_config.opt_config1.num_iterations": min(num_iters, 3000),
        "net_config.opt_config1.freq_update_drift": 9999999,
    }, "configs/generated/lob_as_benchmark.json")
    run(as_config, "lob_as_benchmark")

    # ----------------------------------------------------------------
    # Experiment 2: Full mean-field (Type 3)
    # ----------------------------------------------------------------
    mf_config = make_config(base_config, {
        "eqn_config.type": 3,
        "net_config.opt_config1.num_iterations": num_iters,
    }, "configs/generated/lob_meanfield.json")
    run(mf_config, "lob_meanfield")

    # ----------------------------------------------------------------
    # Experiment 3: N-particle scaling (vary N_simu)
    # ----------------------------------------------------------------
    for n_simu in [100, 500, 2000]:
        particle_config = make_config(base_config, {
            "eqn_config.type": 3,
            "eqn_config.N_simu": n_simu,
            "eqn_config.N_learn": n_simu,
            "net_config.opt_config1.num_iterations": num_iters,
        }, f"configs/generated/lob_particles_{n_simu}.json")
        run(particle_config, f"lob_particles_{n_simu}")

    # ----------------------------------------------------------------
    # Experiment 4: Stress test — crank up inventory penalty
    # ----------------------------------------------------------------
    for phi in [0.001, 0.01, 0.1, 1.0]:
        phi_config = make_config(base_config, {
            "eqn_config.type": 1,
            "eqn_config.phi": phi,
            "eqn_config.couple_coeff": 0.0,
            "net_config.opt_config1.num_iterations": min(num_iters, 3000),
            "net_config.opt_config1.freq_update_drift": 9999999,
        }, f"configs/generated/lob_phi_{phi}.json")
        run(phi_config, f"lob_phi_{phi}")

    print("\n" + "=" * 60)
    print("All experiments complete. Results in ./logs/")
    print("Run: python plot_lob.py --result logs/<exp>_result.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()

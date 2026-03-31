# main.py
#
# Entry point for the Deep MV-BSDE solver.

import argparse
import json
import logging
import os
import time
import numpy as np
import torch
from pathlib import Path

from config import Config
from registry import EQUATION_REGISTRY
import equations  # triggers @register_equation decorators
from solver import SineBMSolver, SineBMDBDPSolver, FlockSolver, ContXiongLOBSolver


def main():
    parser = argparse.ArgumentParser(description="Deep MV-BSDE PyTorch Solver")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--exp_name", type=str, default="test", help="Experiment name")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto' (use GPU if available)")
    parser.add_argument("--num_threads", type=int, default=0,
                        help="CPU threads for PyTorch (0 = use all cores)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)-6s %(message)s")

    # CPU threading — use all cores on the 5950x
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(args.num_threads)
    else:
        num_cores = os.cpu_count()
        torch.set_num_threads(num_cores)
        torch.set_num_interop_threads(min(num_cores, 4))
    logging.info("PyTorch using %d threads" % torch.get_num_threads())

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logging.info("Using device: %s" % device)

    # Load config
    config = Config.from_json(args.config)

    # Set default dtype
    if config.net.dtype == "float64":
        torch.set_default_dtype(torch.float64)

    # Create output directory
    os.makedirs(args.log_dir, exist_ok=True)
    path_prefix = os.path.join(args.log_dir, args.exp_name)

    # Instantiate equation via registry
    if config.eqn.eqn_name not in EQUATION_REGISTRY:
        raise ValueError(
            f"Equation '{config.eqn.eqn_name}' not found. "
            f"Available: {list(EQUATION_REGISTRY.keys())}"
        )
    bsde = EQUATION_REGISTRY[config.eqn.eqn_name](config.eqn)

    # Select solver based on equation + loss type
    logging.info("Begin to solve %s" % config.eqn.eqn_name)
    logging.info("Experiment name: %s" % args.exp_name)

    if config.eqn.eqn_name == "sinebm":
        if config.net.loss_type == "DBDPiter":
            solver = SineBMDBDPSolver(config, bsde)
        else:
            solver = SineBMSolver(config, bsde)
    elif config.eqn.eqn_name == "flocking":
        solver = FlockSolver(config, bsde)
    elif config.eqn.eqn_name == "contxiong_lob":
        solver = ContXiongLOBSolver(config, bsde, device=device)
        solver._save_path = "{}_model.pt".format(path_prefix)
    else:
        raise ValueError(f"No solver for equation '{config.eqn.eqn_name}'")

    # Train
    result = solver.train()

    # Save results
    if config.eqn.eqn_name == "sinebm":
        np.savetxt(
            "{}_result.txt".format(path_prefix),
            result["history"],
            fmt=["%d", "%.5e", "%.5e", "%.5e", "%d"],
            delimiter=",",
            header="step,loss_function,Y0_init,err_mean_y,elapsed_time",
            comments="",
        )
    elif config.eqn.eqn_name == "flocking":
        np.savetxt(
            "{}_result.txt".format(path_prefix),
            result["history"],
            fmt=["%d", "%.5e", "%.5e", "%d"],
            delimiter=",",
            header="step,loss_function,err_Y2_init,elapsed_time",
            comments="",
        )
    elif config.eqn.eqn_name == "contxiong_lob":
        np.savetxt(
            "{}_result.txt".format(path_prefix),
            result["history"],
            fmt=["%d", "%.5e", "%.5e", "%d"],
            delimiter=",",
            header="step,loss_function,Y0_init,elapsed_time",
            comments="",
        )


if __name__ == "__main__":
    main()

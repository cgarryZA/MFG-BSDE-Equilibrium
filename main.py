# main.py
#
# Entry point for the Deep MV-BSDE solver.

import argparse
import logging
import os
import time
import numpy as np
import torch

from config import Config
from registry import EQUATION_REGISTRY
import equations  # triggers @register_equation decorators
from solver import ContXiongLOBSolver, ContXiongLOBMVSolver


def main():
    parser = argparse.ArgumentParser(description="Deep MV-BSDE Solver")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--exp_name", type=str, default="test", help="Experiment name")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--num_threads", type=int, default=0,
                        help="CPU threads (0 = all cores)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-6s %(message)s")

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(args.num_threads)
    else:
        num_cores = os.cpu_count()
        torch.set_num_threads(num_cores)
        torch.set_num_interop_threads(min(num_cores, 4))
    logging.info("PyTorch using %d threads" % torch.get_num_threads())

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logging.info("Using device: %s" % device)

    config = Config.from_json(args.config)
    if config.net.dtype == "float64":
        torch.set_default_dtype(torch.float64)

    os.makedirs(args.log_dir, exist_ok=True)
    path_prefix = os.path.join(args.log_dir, args.exp_name)

    if config.eqn.eqn_name not in EQUATION_REGISTRY:
        raise ValueError(
            f"Equation '{config.eqn.eqn_name}' not found. "
            f"Available: {list(EQUATION_REGISTRY.keys())}"
        )
    bsde = EQUATION_REGISTRY[config.eqn.eqn_name](config.eqn)

    logging.info("Equation: %s (dim=%d)" % (config.eqn.eqn_name, bsde.dim))
    logging.info("Experiment: %s" % args.exp_name)

    # Select solver: MV models use MV solver, others use base solver
    if "mv" in config.eqn.eqn_name:
        solver = ContXiongLOBMVSolver(config, bsde, device=device)
    else:
        solver = ContXiongLOBSolver(config, bsde, device=device)
    solver._save_path = "{}_model.pt".format(path_prefix)

    result = solver.train()

    # Save training history
    fmt = ["%d", "%.5e", "%.5e"]
    header = "step,loss_function,Y0_init"
    if result["history"].shape[1] > 3:
        fmt.append("%.5e")
        header += ",z_max"
    fmt.append("%d")
    header += ",elapsed_time"
    np.savetxt(
        "{}_result.txt".format(path_prefix),
        result["history"],
        fmt=fmt, delimiter=",", header=header, comments="",
    )


if __name__ == "__main__":
    main()

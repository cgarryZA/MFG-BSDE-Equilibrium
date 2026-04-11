#!/bin/bash
# Run this ONCE on Hamilton to set up the environment

# Create conda env
conda env create -f cluster/environment.yml

# Or if you already have a conda env:
# conda activate deepbsde
# pip install torch numpy scipy matplotlib

echo "Setup done. To submit:"
echo "  sbatch cluster/submit_maddpg.sh"
echo ""
echo "To collect results after all jobs finish:"
echo "  python cluster/collect_results.py"

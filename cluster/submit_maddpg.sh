#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=16G
#SBATCH -p tpg-gpu-small
#SBATCH -t 0-03:00:00
#SBATCH --job-name=maddpg
#SBATCH --array=0-19
#SBATCH -o logs/maddpg_%a_%j.out
#SBATCH -e logs/maddpg_%a_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=szbc46@durham.ac.uk

source /etc/profile
module purge
module load cuda/12.3-cudnn8.9

source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepbsde

mkdir -p logs results_cluster

cd ~/MFG-BSDE-Equilibrium

echo "=== MADDPG seed $SLURM_ARRAY_TASK_ID at $(date) ==="
stdbuf -oL python cluster/run_maddpg_single.py \
    --seed $SLURM_ARRAY_TASK_ID \
    --outdir results_cluster \
    --episodes 500 \
    --steps 500 \
    2>&1 | tee logs/maddpg_${SLURM_ARRAY_TASK_ID}.log

echo "Exit code: $?"
echo "=== Done at $(date) ==="
conda deactivate

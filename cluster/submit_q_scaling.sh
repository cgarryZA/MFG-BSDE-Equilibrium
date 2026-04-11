#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=16G
#SBATCH -p tpg-gpu-small
#SBATCH -t 0-04:00:00
#SBATCH --job-name=q_scaling
#SBATCH -o logs/q_scaling_%j.out
#SBATCH -e logs/q_scaling_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=szbc46@durham.ac.uk

source /etc/profile
module purge
module load cuda/12.3-cudnn8.9

source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepbsde

mkdir -p logs results_cluster

cd ~/MFG-BSDE-Equilibrium

echo "=== Q scaling at $(date) ==="
stdbuf -oL python -c "
import sys; sys.path.insert(0, '.')
import gc, json, numpy as np, torch, time
torch.set_default_dtype(torch.float64)
from equations.contxiong_exact import ContXiongExact
from solver_cx import CXSolver
from scripts.cont_xiong_exact import fictitious_play
device = torch.device('cuda')
results = {}
for Q in [5, 10, 20, 50]:
    nq = int(2*Q+1)
    print(f'Q={Q} ({nq} points)')
    ex = fictitious_play(N=2, Q=Q, max_iter=100)
    mid = len(ex['q_grid'])//2
    ex_s = ex['delta_a'][mid]+ex['delta_b'][mid]
    gc.collect(); torch.cuda.empty_cache()
    class Cfg:
        lambda_a=2.0; lambda_b=2.0; discount_rate=0.01
        Delta_q=1.0; q_max=float(Q); phi=0.005; N_agents=2
    eqn = ContXiongExact(Cfg())
    n_iter = {5:5000, 10:10000, 20:20000, 50:40000}[Q]
    lr = {5:1e-3, 10:5e-4, 20:3e-4, 50:2e-4}[Q]
    t0 = time.time()
    solver = CXSolver(eqn, device=device, lr=lr, n_iter=n_iter, verbose=True)
    r = solver.train()
    nn_s = r['delta_a'][eqn.mid]+r['delta_b'][eqn.mid]
    err = abs(nn_s-ex_s)
    print(f'  Exact:{ex_s:.4f} Neural:{nn_s:.4f} Err:{err:.4f} ({err/ex_s*100:.1f}%) {time.time()-t0:.0f}s')
    results[f'Q={Q}'] = {'Q':Q,'exact':ex_s,'neural':nn_s,'error':err}
    with open('results_cluster/q_scaling.json','w') as f:
        json.dump(results, f, indent=2, default=float)
print('Done')
" 2>&1 | tee logs/q_scaling.log

echo "Exit code: $?"
echo "=== Done at $(date) ==="
conda deactivate

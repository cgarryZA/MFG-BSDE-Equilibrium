#!/bin/bash
cd D:/DeepBSDE
mkdir -p results_final

echo "=== 1/5: Direct-V Q-scaling FIXED (~20min) ==="
python -u scripts/q_scaling_direct_v.py 2>&1 | tee results_final/log_direct_v.txt

echo ""
echo "=== 2/5: Non-stationary phi(t) (~2.5h) ==="
python -u scripts/nonstationary_phi.py 2>&1 | tee results_final/log_nonstat_phi.txt

echo ""
echo "=== 3/5: BSDEJ N-scaling (~2.5h) ==="
python -u scripts/bsdej_n_scaling.py 2>&1 | tee results_final/log_bsdej_n.txt

echo ""
echo "=== 4/5: Diffusion Q-scaling (~7.5h) ==="
python -u scripts/q_scaling_diffusion.py 2>&1 | tee results_final/log_diff_q.txt

echo ""
echo "=== 5/5: BSDEJ improvement (~11h) ==="
python -u run_bsdej_improve.py 2>&1 | tee results_final/log_bsdej_improve.txt

echo ""
echo "ALL DONE: $(date)"

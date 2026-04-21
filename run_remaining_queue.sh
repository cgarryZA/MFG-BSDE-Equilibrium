#!/bin/bash
# Remaining experiments, ordered fastest first
cd D:/DeepBSDE
mkdir -p results_final

echo "========================================"
echo "QUEUE START: $(date)"
echo "========================================"

echo ""
echo "=== 1/5: Multi-asset K=3 (~20-30 min) ==="
python -u scripts/multiasset_K3.py 2>&1 | tee results_final/log_multiasset_K3.txt

echo ""
echo "=== 2/5: Non-stationary phi(t) (~1h) ==="
python -u scripts/nonstationary_phi.py 2>&1 | tee results_final/log_nonstat_phi.txt

echo ""
echo "=== 3/5: BSDEJ N-scaling (~2.5h) ==="
python -u scripts/bsdej_n_scaling.py 2>&1 | tee results_final/log_bsdej_n.txt

echo ""
echo "=== 4/5: Common noise (~2h) ==="
python -u scripts/common_noise.py 2>&1 | tee results_final/log_common_noise.txt

echo ""
echo "=== SKIPPED 5/5: Diffusion Q-scaling (7.5h, not core per narrative focus) ==="

echo ""
echo "========================================"
echo "QUEUE DONE: $(date)"
echo "========================================"

#!/bin/bash
cd D:/DeepBSDE
mkdir -p results_final

echo "========================================"
echo "OVERNIGHT ROBUSTNESS QUEUE START: $(date)"
echo "========================================"

echo ""
echo "=== 1/5: AS extreme alpha (~2 min CPU) ==="
python -u scripts/as_extreme_alpha.py 2>&1 | tee results_final/log_as_extreme.txt

echo ""
echo "=== 2/5: AS robustness N x lambda (~30 min CPU) ==="
python -u scripts/as_robustness_N_lambda.py 2>&1 | tee results_final/log_as_robust.txt

echo ""
echo "=== 3/5: LBD seed variance (~2h CPU) ==="
python -u scripts/lbd_seed_variance.py 2>&1 | tee results_final/log_lbd_variance.txt

echo ""
echo "=== 4/5: Non-stationary alt profiles (~2h GPU) ==="
python -u scripts/nonstationary_alt_profiles.py 2>&1 | tee results_final/log_nonstat_alt.txt

echo ""
echo "=== 5/5: Common noise sigma scaling (~3h GPU) ==="
python -u scripts/common_noise_sigma_scaling.py 2>&1 | tee results_final/log_cn_sigma.txt

echo ""
echo "========================================"
echo "QUEUE DONE: $(date)"
echo "========================================"

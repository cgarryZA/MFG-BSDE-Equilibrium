#!/bin/bash
echo "Waiting for direct-V to finish..."
while true; do
  if ! wmic process where "CommandLine like '%q_scaling_direct_v%'" get ProcessId 2>/dev/null | grep -q '[0-9]'; then
    if ! wmic process where "CommandLine like '%wait_and_direct_v%'" get ProcessId 2>/dev/null | grep -q '[0-9]'; then
      echo "Direct-V done. Starting diffusion Q-scaling..."
      cd D:/DeepBSDE
      python -u scripts/q_scaling_diffusion.py
      exit 0
    fi
  fi
  sleep 120
done

#!/bin/bash
echo "Waiting for Q-scaling to finish..."
while true; do
  if ! wmic process where "CommandLine like '%q_scaling%'" get ProcessId 2>/dev/null | grep -q '[0-9]'; then
    # Check if any heavy python process is still running (the original run)
    if ! wmic process where "CommandLine like '%bsde_diffusion%'" get ProcessId 2>/dev/null | grep -q '[0-9]'; then
      if ! wmic process where "CommandLine like '%Q-scaling robust%'" get ProcessId 2>/dev/null | grep -q '[0-9]'; then
        echo "GPU free! Starting BSDEJ improvement run..."
        cd D:/DeepBSDE
        python -u run_bsdej_improve.py
        exit 0
      fi
    fi
  fi
  sleep 120
done

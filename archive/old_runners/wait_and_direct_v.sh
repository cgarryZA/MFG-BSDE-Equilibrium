#!/bin/bash
echo "Waiting for BSDEJ improvement to finish..."
while true; do
  if ! wmic process where "CommandLine like '%bsdej_improve%'" get ProcessId 2>/dev/null | grep -q '[0-9]'; then
    echo "BSDEJ improvement done. Starting direct-V script..."
    cd D:/DeepBSDE
    python -u scripts/q_scaling_direct_v.py
    exit 0
  fi
  sleep 60
done

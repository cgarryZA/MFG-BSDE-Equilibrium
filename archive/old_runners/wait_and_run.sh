#!/bin/bash
# Wait for run_everything.py to finish, then start overnight_final.py
echo "Waiting for run_everything.py to finish..."
while true; do
  if ! wmic process where "CommandLine like '%run_everything%'" get ProcessId 2>/dev/null | grep -q '[0-9]'; then
    echo "run_everything.py done! Starting overnight_final.py..."
    cd D:/DeepBSDE
    python -u run_overnight_final.py
    exit 0
  fi
  sleep 120
done

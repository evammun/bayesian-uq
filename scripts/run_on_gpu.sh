#!/bin/bash
# ============================================================================
# Main experiment runner — called by autorun.sh (backgrounded) or manually.
#
# Manual usage:
#   cd /workspace/bayesian-uq && bash scripts/run_on_gpu.sh
# ============================================================================

PROJECT="/workspace/bayesian-uq"
cd "$PROJECT" || exit 1

echo ""
echo "============================================"
echo "[$(date)] run_on_gpu.sh starting"
echo "============================================"

# Step 1: Ensure deps + model (idempotent — fast if already done)
echo ""
echo "[$(date)] Running vastai_setup.sh..."
bash scripts/vastai_setup.sh
SETUP_EXIT=$?

if [ $SETUP_EXIT -ne 0 ]; then
    echo "[$(date)] WARNING: setup exited with code $SETUP_EXIT"
    echo "[$(date)] Attempting experiments anyway..."
fi

# Step 2: Run all experiments (auto-resume skips completed ones)
echo ""
echo "[$(date)] Running experiments..."
bash scripts/run_all_pilots.sh
EXP_EXIT=$?

if [ $EXP_EXIT -ne 0 ]; then
    echo "[$(date)] WARNING: experiments exited with code $EXP_EXIT"
fi

echo ""
echo "[$(date)] run_on_gpu.sh finished (setup=$SETUP_EXIT, experiments=$EXP_EXIT)"

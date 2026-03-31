#!/bin/bash
# ============================================================================
# vast.ai on-start entrypoint
# ============================================================================
# Paste this as the vast.ai template on-start command:
#
#   bash /workspace/bayesian-uq/scripts/autorun.sh
#
# RULES FOR ON-START SCRIPTS (learned the hard way):
#   1. MUST exit quickly — vast.ai runs this synchronously during container
#      init. If it blocks, the container never starts and SSH is unreachable.
#   2. MUST exit 0 even if everything is broken — non-zero kills the container.
#   3. MUST handle missing files — on first boot, project may not be deployed yet.
#
# The real work (setup + experiments) runs in the background via nohup.
# ============================================================================

LOG="/workspace/autorun.log"
PROJECT="/workspace/bayesian-uq"

echo "[$(date)] on-start triggered" >> "$LOG" 2>/dev/null

# If project files aren't deployed yet, exit cleanly — user will scp them
# and run manually. On future boots (after interruption), files will be there.
if [ ! -f "$PROJECT/scripts/run_on_gpu.sh" ]; then
    echo "[$(date)] Project files not found — waiting for deploy" >> "$LOG" 2>/dev/null
    exit 0
fi

# Launch real work in background. MUST use nohup so it survives init.
nohup bash "$PROJECT/scripts/run_on_gpu.sh" >> "$LOG" 2>&1 &
disown

echo "[$(date)] Backgrounded run_on_gpu.sh (PID $!)" >> "$LOG" 2>/dev/null

# Exit immediately so container init completes and SSH becomes available
exit 0

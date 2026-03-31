#!/bin/bash
# ============================================================================
# Fetch results from vast.ai instance (one-shot or polling)
# ============================================================================
# Usage:
#   One-shot:  bash scripts/fetch_results.sh <ssh_host> <ssh_port>
#   Polling:   bash scripts/fetch_results.sh <ssh_host> <ssh_port> --poll [interval_seconds]
#
# Examples:
#   bash scripts/fetch_results.sh root@ssh5.vast.ai 12345
#   bash scripts/fetch_results.sh root@ssh5.vast.ai 12345 --poll 300
#
# Results are saved to results/ in the project directory.
# Existing files are overwritten only if the remote copy is newer.
# ============================================================================

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash scripts/fetch_results.sh <ssh_host> <ssh_port> [--poll interval_sec]"
    echo "Example: bash scripts/fetch_results.sh root@ssh5.vast.ai 12345 --poll 300"
    exit 1
fi

SSH_HOST="$1"
SSH_PORT="$2"
POLL_MODE=false
POLL_INTERVAL=300  # default 5 minutes

if [ "$3" = "--poll" ]; then
    POLL_MODE=true
    if [ -n "$4" ]; then
        POLL_INTERVAL="$4"
    fi
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_RESULTS="$PROJECT_ROOT/results"
REMOTE_DIR="/workspace/bayesian-uq/results"

mkdir -p "$LOCAL_RESULTS"

do_fetch() {
    local timestamp
    timestamp=$(date '+%H:%M:%S')
    echo "[$timestamp] Fetching results from $SSH_HOST:$SSH_PORT..."

    # rsync: only copy files that are newer on remote, show progress
    if rsync -avz --update --progress \
        -e "ssh -p $SSH_PORT" \
        "$SSH_HOST:$REMOTE_DIR/" "$LOCAL_RESULTS/" 2>/dev/null; then

        local count
        count=$(ls "$LOCAL_RESULTS"/*.json 2>/dev/null | wc -l)
        echo "[$timestamp] Done. $count result files in $LOCAL_RESULTS/"

        # Quick summary: how many questions completed per file
        if [ "$count" -gt 0 ]; then
            echo ""
            python3 -c "
import json, os, sys
results_dir = '$LOCAL_RESULTS'
for f in sorted(os.listdir(results_dir)):
    if not f.endswith('.json'):
        continue
    try:
        with open(os.path.join(results_dir, f)) as fh:
            data = json.load(fh)
        n = len(data.get('question_results', []))
        name = data.get('config', {}).get('run_name', f)
        status = 'DONE' if n >= 4609 else f'{n}/4609'
        print(f'  {name:50s} {status}')
    except:
        print(f'  {f:50s} (parse error)')
" 2>/dev/null || true
            echo ""
        fi
    else
        echo "[$timestamp] rsync failed (instance may be paused/stopped)"
    fi
}

if [ "$POLL_MODE" = true ]; then
    echo "============================================"
    echo "  Polling results every ${POLL_INTERVAL}s"
    echo "  Press Ctrl+C to stop"
    echo "============================================"
    echo ""
    while true; do
        do_fetch
        echo "--- Next fetch in ${POLL_INTERVAL}s ---"
        echo ""
        sleep "$POLL_INTERVAL"
    done
else
    do_fetch
fi

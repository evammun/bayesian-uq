#!/bin/bash
# ============================================================================
# Run all experiment configs with automatic resume support
# ============================================================================
# Usage:
#   bash scripts/run_all_pilots.sh
#
# For each config, this script:
#   1. Checks if a COMPLETE result file exists → skips if done
#   2. Checks if ANY result file exists (even corrupted) → resumes from it
#   3. Otherwise → starts fresh
#
# Designed for interruptible vast.ai instances: when you get paused and
# resume, just run this script again. It picks up where it left off.
#
# Key robustness feature: uses byte-counting (grep -c) instead of JSON
# parsing to count results. A file truncated mid-write by an interruption
# will still be detected as partial and resumed, not ignored.
# ============================================================================

# NOTE: no set -e — if one experiment fails, continue with the rest

# --- Lock file to prevent duplicate instances ---
LOCKFILE="/tmp/run_all_pilots.lock"
if [ -f "$LOCKFILE" ]; then
    OLD_PID=$(cat "$LOCKFILE" 2>/dev/null)
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[$(date)] Another instance already running (PID $OLD_PID). Exiting."
        exit 0
    else
        echo "[$(date)] Stale lock file (PID $OLD_PID dead). Removing."
        rm -f "$LOCKFILE"
    fi
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

CONFIGS=(
    "experiments/configs/quality_direct_noshuffle_sufficient.yaml"
    "experiments/configs/quality_direct_noshuffle_insufficient.yaml"
    "experiments/configs/quality_direct_shuffle_sufficient.yaml"
    "experiments/configs/quality_direct_shuffle_insufficient.yaml"
    "experiments/configs/quality_cot_noshuffle_sufficient.yaml"
    "experiments/configs/quality_cot_noshuffle_insufficient.yaml"
    "experiments/configs/quality_cot_shuffle_sufficient.yaml"
    "experiments/configs/quality_cot_shuffle_insufficient.yaml"
)

# Total questions in QuALITY dataset
TOTAL_QUESTIONS=4609

echo "============================================"
echo "  Running ${#CONFIGS[@]} experiments"
echo "  (with automatic resume support)"
echo "============================================"
echo ""

COMPLETED=0
RESUMED=0
STARTED=0
FAILED=0

for config in "${CONFIGS[@]}"; do
    run_name=$(grep '^run_name:' "$config" | awk '{print $2}' | tr -d '\r')

    # Find ALL result files for this run_name, newest first
    # (there may be multiple from interrupted runs)
    result_files=$(ls -t "$RESULTS_DIR"/${run_name}_*.json 2>/dev/null)

    if [ -n "$result_files" ]; then
        # Find the file with the MOST results (not necessarily newest —
        # a fresh restart creates a newer but emptier file)
        best_file=""
        best_count=0

        for rf in $result_files; do
            # Fast byte-count: grep for "question_id" occurrences
            n=$(grep -c '"question_id"' "$rf" 2>/dev/null || echo 0)
            if [ "$n" -gt "$best_count" ]; then
                best_count=$n
                best_file=$rf
            fi
        done

        if [ "$best_count" -ge "$TOTAL_QUESTIONS" ]; then
            echo "[DONE]   $run_name — $best_count questions complete ($(basename $best_file))"
            COMPLETED=$((COMPLETED + 1))

            # Clean up smaller fragment files from interrupted runs
            for rf in $result_files; do
                if [ "$rf" != "$best_file" ]; then
                    echo "         Removing fragment: $(basename $rf)"
                    rm -f "$rf"
                fi
            done
            continue

        elif [ "$best_count" -gt "0" ]; then
            echo "[RESUME] $run_name — $best_count/$TOTAL_QUESTIONS done ($(basename $best_file))"
            RESUMED=$((RESUMED + 1))

            # Clean up smaller fragment files before resuming
            for rf in $result_files; do
                if [ "$rf" != "$best_file" ]; then
                    echo "         Removing fragment: $(basename $rf)"
                    rm -f "$rf"
                fi
            done

            if python3 experiments/run_experiment.py \
                --config "$config" \
                --resume "$best_file"; then
                echo "[DONE]   $run_name — completed after resume"
            else
                echo "[FAIL]   $run_name — failed during resume"
                FAILED=$((FAILED + 1))
            fi
            continue
        fi
    fi

    # No result file found (or all empty) — start fresh
    echo "[START]  $run_name"
    STARTED=$((STARTED + 1))

    if python3 experiments/run_experiment.py --config "$config"; then
        echo "[DONE]   $run_name — completed"
    else
        echo "[FAIL]   $run_name — failed"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
echo "  Summary"
echo "  Already complete: $COMPLETED"
echo "  Resumed:          $RESUMED"
echo "  Started fresh:    $STARTED"
echo "  Failed:           $FAILED"
echo "============================================"

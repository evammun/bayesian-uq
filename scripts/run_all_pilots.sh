#!/bin/bash
# ============================================================================
# Run all pilot experiment configs with automatic resume support
# ============================================================================
# Usage:
#   bash scripts/run_all_pilots.sh
#
# For each config, this script:
#   1. Checks if a COMPLETE result file exists (has "question_results" with
#      the expected number of entries) → skips if done
#   2. Checks if a PARTIAL result file exists → resumes from it
#   3. Otherwise → starts fresh
#
# Designed for interruptible vast.ai instances: when you get paused and
# resume, just run this script again. It picks up where it left off.
# ============================================================================

# NOTE: no set -e — if one experiment fails, continue with the rest

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
    run_name=$(grep '^run_name:' "$config" | awk '{print $2}')

    # Find result files matching this run_name
    # Sort by modification time (newest first) to find the latest
    latest_result=$(ls -t "$RESULTS_DIR"/${run_name}_*.json 2>/dev/null | head -1)

    if [ -n "$latest_result" ]; then
        # Count how many question_results are in the file
        n_results=$(python3 -c "
import json, sys
try:
    with open('$latest_result') as f:
        data = json.load(f)
    print(len(data.get('question_results', [])))
except:
    print(0)
")

        if [ "$n_results" -ge "$TOTAL_QUESTIONS" ]; then
            echo "[DONE]   $run_name — $n_results questions complete"
            COMPLETED=$((COMPLETED + 1))
            continue
        elif [ "$n_results" -gt "0" ]; then
            echo "[RESUME] $run_name — $n_results/$TOTAL_QUESTIONS done, resuming..."
            RESUMED=$((RESUMED + 1))

            if python3 experiments/run_experiment.py \
                --config "$config" \
                --resume "$latest_result"; then
                echo "[DONE]   $run_name — completed after resume"
            else
                echo "[FAIL]   $run_name — failed during resume"
                FAILED=$((FAILED + 1))
            fi
            continue
        fi
    fi

    # No result file found — start fresh
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

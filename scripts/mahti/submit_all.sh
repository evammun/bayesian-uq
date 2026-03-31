#!/bin/bash
# ============================================================================
# Submit all Mahti experiments as separate SLURM jobs
# ============================================================================
# Usage: bash scripts/mahti/submit_all.sh [--dry-run]
#
# Checks each config for completion status before submitting.
# Sets wall time based on experiment type:
#   - noshuffle (1 query/q): 2h for direct, 6h for cot
#   - shuffle (10 queries/q): 20h for direct, 36h for cot
#
# Use --dry-run to see what would be submitted without actually doing it.
# ============================================================================

PROJECT=/scratch/project_2018384/bayesian-uq
RESULTS=$PROJECT/results
CONFIGS_DIR=$PROJECT/experiments/configs/mahti
TOTAL=4609
DRY_RUN=false

if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "[DRY RUN — nothing will be submitted]"
    echo ""
fi

# Ensure logs directory exists
mkdir -p /scratch/project_2018384/logs
mkdir -p "$RESULTS"

echo "============================================"
echo "  Mahti Experiment Submission"
echo "============================================"
echo ""

SUBMITTED=0
SKIPPED=0
DONE=0

for config in "$CONFIGS_DIR"/quality_*.yaml; do
    [ -f "$config" ] || continue

    run_name=$(grep '^run_name:' "$config" | awk '{print $2}' | tr -d '\r')
    prompt_mode=$(grep '^prompt_mode:' "$config" | awk '{print $2}' | tr -d '\r')
    shuffle=$(grep '^shuffle_options:' "$config" | awk '{print $2}' | tr -d '\r')
    num_perm=$(grep '^num_permutations:' "$config" | awk '{print $2}' | tr -d '\r')

    # Check if already complete
    best=$(ls -S "$RESULTS"/${run_name}_*.json 2>/dev/null | head -1)
    if [ -n "$best" ]; then
        n=$(grep -c '"question_id"' "$best" 2>/dev/null || echo 0)
        if [ "$n" -ge "$TOTAL" ]; then
            echo "[DONE]   $run_name ($n questions)"
            DONE=$((DONE + 1))
            continue
        fi
        echo "[PARTIAL] $run_name ($n/$TOTAL — will resume)"
    fi

    # Set wall time based on experiment type
    if [ "$shuffle" = "true" ]; then
        if [ "$prompt_mode" = "cot" ]; then
            WALL_TIME="36:00:00"  # cot + shuffle: ~30h
        else
            WALL_TIME="20:00:00"  # direct + shuffle: ~15h
        fi
    else
        if [ "$prompt_mode" = "cot" ]; then
            WALL_TIME="06:00:00"  # cot + noshuffle: ~4h
        else
            WALL_TIME="02:00:00"  # direct + noshuffle: ~1.5h
        fi
    fi

    # Estimate BU cost
    hours=$(echo "$WALL_TIME" | awk -F: '{print $1 + $2/60}')
    bu_est=$(echo "$hours * 100" | bc 2>/dev/null || echo "?")

    JOB_NAME="uq-${run_name}"

    if [ "$DRY_RUN" = true ]; then
        echo "[WOULD]  $run_name (wall=$WALL_TIME, ~${bu_est} BU max)"
    else
        echo "[SUBMIT] $run_name (wall=$WALL_TIME, ~${bu_est} BU max)"
        sbatch --job-name="$JOB_NAME" \
               --time="$WALL_TIME" \
               --export=CONFIG="$config" \
               "$PROJECT/scripts/mahti/slurm_single.sh"
    fi
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================"
echo "  Summary"
echo "  Already complete: $DONE"
echo "  Submitted:        $SUBMITTED"
echo "  Total configs:    $((DONE + SUBMITTED + SKIPPED))"
echo "============================================"
echo ""
echo "Monitor with: squeue --me"
echo "Logs at:      /scratch/project_2018384/logs/"

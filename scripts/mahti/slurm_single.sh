#!/bin/bash
# ============================================================================
# SLURM batch script for a single experiment on CSC Mahti
# ============================================================================
# Usage: sbatch --export=CONFIG=experiments/configs/mahti/quality_cot_noshuffle_sufficient.yaml scripts/mahti/slurm_single.sh
# Or:    sbatch --time=20:00:00 --export=CONFIG=... scripts/mahti/slurm_single.sh
#
# The CONFIG env var must be set (via --export) to the YAML config path
# relative to the project root.
# ============================================================================

#SBATCH --job-name=uq-inference
#SBATCH --account=project_2018384
#SBATCH --partition=gpusmall
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1,nvme:100
#SBATCH --output=/scratch/project_2018384/logs/%x_%j.out
#SBATCH --error=/scratch/project_2018384/logs/%x_%j.err

# --- Environment ---
module load gcc cuda python-data
source /projappl/project_2018384/llama-env/bin/activate

PROJECT=/scratch/project_2018384/bayesian-uq
cd "$PROJECT" || exit 1

# --- Validate CONFIG is set ---
if [ -z "$CONFIG" ]; then
    echo "ERROR: CONFIG env var not set. Use: sbatch --export=CONFIG=path/to/config.yaml ..."
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

RUN_NAME=$(grep '^run_name:' "$CONFIG" | awk '{print $2}' | tr -d '\r')
echo "============================================"
echo "  Experiment: $RUN_NAME"
echo "  Config:     $CONFIG"
echo "  Node:       $(hostname)"
echo "  GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Started:    $(date -u)"
echo "============================================"
echo ""

# --- Copy model to fast local NVMe ---
MODEL_SRC=/scratch/project_2018384/models/qwen3-8b-q4_k_m.gguf
MODEL_DST=$LOCAL_SCRATCH/qwen3-8b-q4_k_m.gguf

echo "Copying model to LOCAL_SCRATCH..."
cp "$MODEL_SRC" "$MODEL_DST"
if [ ! -f "$MODEL_DST" ]; then
    echo "ERROR: Model copy failed"
    exit 1
fi
echo "  Model: $(ls -lh $MODEL_DST | awk '{print $5}')"
export UQ_MODEL_PATH="$MODEL_DST"

# --- Auto-resume logic ---
RESULT_DIR="$PROJECT/results"
mkdir -p "$RESULT_DIR"
EXISTING=$(ls -S "$RESULT_DIR"/${RUN_NAME}_*.json 2>/dev/null | head -1)

if [ -n "$EXISTING" ]; then
    N=$(grep -c '"question_id"' "$EXISTING" 2>/dev/null || echo 0)
    TOTAL=4609
    if [ "$N" -ge "$TOTAL" ]; then
        echo "Already complete: $EXISTING ($N questions). Nothing to do."
        exit 0
    fi
    echo "Resuming from $EXISTING ($N/$TOTAL questions done)"
    echo ""
    srun python3 experiments/run_experiment.py --config "$CONFIG" --resume "$EXISTING"
else
    echo "Starting fresh: $RUN_NAME"
    echo ""
    srun python3 experiments/run_experiment.py --config "$CONFIG"
fi

EXIT_CODE=$?
echo ""
echo "============================================"
echo "  Finished: $(date -u)"
echo "  Exit code: $EXIT_CODE"
echo "============================================"

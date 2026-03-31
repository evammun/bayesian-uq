#!/bin/bash
# ============================================================================
# One-time environment setup for CSC Mahti
# ============================================================================
# Run this on the Mahti LOGIN NODE (needs internet for pip/wget).
#
# Usage: bash scripts/mahti/setup_env.sh
#
# What it does:
#   1. Creates Python venv on /projappl/ (persistent)
#   2. Installs llama-cpp-python with CUDA support
#   3. Installs remaining Python deps
#   4. Downloads GGUF model to /scratch/
#   5. Creates required directories
#
# Idempotent — safe to run multiple times.
# ============================================================================

set -e  # stop on first error — this is setup, not runtime

PROJECT_ID="project_2018384"
PROJAPPL="/projappl/$PROJECT_ID"
SCRATCH="/scratch/$PROJECT_ID"
VENV="$PROJAPPL/llama-env"
MODEL_DIR="$SCRATCH/models"
MODEL_FILE="$MODEL_DIR/qwen3-8b-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/qwen3-8b-q4_k_m.gguf"

echo "============================================"
echo "  CSC Mahti — Environment Setup"
echo "============================================"
echo ""

# --- 0. Load modules ---
echo "[0/5] Loading modules..."
module load gcc cuda python-data
echo "  gcc:  $(gcc --version | head -1)"
echo "  nvcc: $(nvcc --version | tail -1)"
echo "  python: $(python3 --version)"
echo ""

# --- 1. Create directories ---
echo "[1/5] Creating directories..."
mkdir -p "$SCRATCH/bayesian-uq/results"
mkdir -p "$SCRATCH/logs"
mkdir -p "$MODEL_DIR"
echo "  OK"
echo ""

# --- 2. Python venv ---
echo "[2/5] Python venv..."
if [ -f "$VENV/bin/activate" ]; then
    echo "  Venv already exists: $VENV"
else
    echo "  Creating venv with --system-site-packages..."
    python3 -m venv --system-site-packages "$VENV"
    echo "  Created: $VENV"
fi
source "$VENV/bin/activate"
echo "  Activated: $(which python3)"
echo ""

# --- 3. Install llama-cpp-python ---
echo "[3/5] Installing llama-cpp-python..."
if python3 -c "import llama_cpp; print(f'  Already installed: {llama_cpp.__version__}')" 2>/dev/null; then
    echo "  (skipping)"
else
    echo "  Trying pre-built CUDA wheel..."
    if pip install "llama-cpp-python>=0.3.16" \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 2>/dev/null; then
        echo "  Wheel install succeeded"
    else
        echo "  Wheel failed — building from source (this takes 3-5 min)..."
        CMAKE_ARGS="-DGGML_CUDA=on" \
        CUDA_ARCHITECTURES="80" \
        CUDACXX="$(which nvcc)" \
        pip install llama-cpp-python --no-binary llama-cpp-python --verbose
    fi
fi

# Verify
python3 -c "import llama_cpp; print(f'  llama-cpp-python {llama_cpp.__version__}')" || {
    echo "  ERROR: llama-cpp-python installation failed"
    exit 1
}
echo ""

# --- 4. Install other deps ---
echo "[4/5] Installing Python dependencies..."
pip install pydantic>=2.0 pyyaml orjson tqdm 2>/dev/null
python3 -c "
import pydantic, numpy, scipy, yaml, orjson
print('  pydantic', pydantic.__version__)
print('  numpy', numpy.__version__)
print('  All deps OK')
" || {
    echo "  ERROR: Missing dependencies"
    exit 1
}

# Clean up __pycache__ to save file count quota
find "$VENV" -name '__pycache__' -exec rm -rf {} + 2>/dev/null
echo ""

# --- 5. Download model ---
echo "[5/5] Checking model..."
if [ -f "$MODEL_FILE" ]; then
    SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "  Model already exists: $MODEL_FILE ($SIZE)"
else
    echo "  Downloading model (~5.2 GB)..."
    TEMP="$MODEL_FILE.downloading"
    rm -f "$TEMP"
    wget --progress=dot:giga -O "$TEMP" "$MODEL_URL" || {
        echo "  ERROR: Download failed"
        rm -f "$TEMP"
        exit 1
    }
    mv "$TEMP" "$MODEL_FILE"
    SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "  Downloaded: $MODEL_FILE ($SIZE)"
fi
echo ""

# --- Done ---
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Upload code: rsync -azP ... evamar@mahti.csc.fi:$SCRATCH/bayesian-uq/"
echo "    2. Smoke test:  sinteractive --account $PROJECT_ID --time 0:15:00 --gres=gpu:a100:1,nvme:100"
echo "    3. Submit jobs:  bash scripts/mahti/submit_all.sh --dry-run"
echo "============================================"

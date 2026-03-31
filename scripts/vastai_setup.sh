#!/bin/bash
# ============================================================================
# vast.ai GPU instance setup for Pre-Action UQ experiments
# ============================================================================
# Idempotent — safe to run multiple times (skips what's already done).
# On a fresh instance: ~3 min (model download is the bottleneck).
# On a resumed instance: ~5 sec (everything already present).
#
# MUST NOT use set -e — individual failures should not kill the whole script.
# Called by autorun.sh (backgrounded via nohup) and also manually.
# ============================================================================

MODEL_URL="https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
MODEL_FILE="/workspace/models/qwen3-8b-q4_k_m.gguf"
MIN_MODEL_SIZE=1000000000  # 1GB — real file is ~5.2GB, catches 0-byte/corrupt downloads

echo "============================================"
echo "  Pre-Action UQ — vast.ai setup"
echo "============================================"
echo ""

# ------------------------------------------------------------------
# 1. Python dependencies
# ------------------------------------------------------------------
echo "[1/3] Checking Python dependencies..."

# llama-cpp-python with CUDA wheels (the critical one)
python3 -c "import llama_cpp; print(f'  llama-cpp-python {llama_cpp.__version__}')" 2>/dev/null || {
    echo "  Installing llama-cpp-python (CUDA)..."
    pip install "llama-cpp-python>=0.3.16" \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 || \
    pip install "llama-cpp-python>=0.3.16" \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu123
}

# Other deps
pip install pydantic numpy scipy pyyaml orjson tqdm 2>/dev/null

python3 -c "import pydantic, numpy, scipy, yaml; print('  Core deps OK')" || {
    echo "  ERROR: Missing core dependencies"
    return 1 2>/dev/null || exit 1
}
echo ""

# ------------------------------------------------------------------
# 2. GPU and model
# ------------------------------------------------------------------
echo "[2/3] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
else
    echo "  WARNING: nvidia-smi not found"
fi

echo ""
echo "Checking model..."
NEED_DOWNLOAD=false

if [ -f "$MODEL_FILE" ]; then
    FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -lt "$MIN_MODEL_SIZE" ]; then
        echo "  Model file exists but is too small (${FILE_SIZE} bytes) — re-downloading"
        rm -f "$MODEL_FILE"
        NEED_DOWNLOAD=true
    else
        SIZE_HR=$(du -h "$MODEL_FILE" | cut -f1)
        echo "  Model OK: $MODEL_FILE ($SIZE_HR)"
    fi
else
    NEED_DOWNLOAD=true
fi

if [ "$NEED_DOWNLOAD" = true ]; then
    echo "  Downloading model (~5.2GB)..."
    mkdir -p /workspace/models

    # Download to temp file first — if it fails, we don't leave a 0-byte file
    TEMP_FILE="${MODEL_FILE}.downloading"
    rm -f "$TEMP_FILE"

    if wget --progress=dot:giga -O "$TEMP_FILE" "$MODEL_URL"; then
        TEMP_SIZE=$(stat -c%s "$TEMP_FILE" 2>/dev/null || stat -f%z "$TEMP_FILE" 2>/dev/null || echo 0)
        if [ "$TEMP_SIZE" -lt "$MIN_MODEL_SIZE" ]; then
            echo "  ERROR: Downloaded file too small (${TEMP_SIZE} bytes). URL may be wrong."
            rm -f "$TEMP_FILE"
            return 1 2>/dev/null || exit 1
        fi
        mv "$TEMP_FILE" "$MODEL_FILE"
        SIZE_HR=$(du -h "$MODEL_FILE" | cut -f1)
        echo "  Downloaded: $MODEL_FILE ($SIZE_HR)"
    else
        echo "  ERROR: wget failed. Check URL: $MODEL_URL"
        rm -f "$TEMP_FILE"
        return 1 2>/dev/null || exit 1
    fi
fi
echo ""

# ------------------------------------------------------------------
# 3. Smoke test
# ------------------------------------------------------------------
echo "[3/3] Smoke test — loading model and running one inference..."
python3 -c "
import sys
sys.path.insert(0, 'src')
from pre_action_uq.inference import LlamaCppClient

print('Loading model...')
client = LlamaCppClient('$MODEL_FILE', n_ctx=12288, verbose=False)
print(f'  Loaded in {client.load_time:.1f}s')

result = client.generate_with_logprobs(
    'What is 2+2?\n\nA) 3\nB) 4\nC) 5\nD) 6\n\nAnswer:'
)
print(f'  Inference OK — top token: {result[\"response_text\"]}')
top4 = [(lp['token'], f'{lp[\"logprob\"]:.2f}') for lp in result['logprobs'][0]['top_logprobs'][:4]]
print(f'  Top logprobs: {top4}')
" || {
    echo "  WARNING: Smoke test failed — check inference.py compatibility"
}

echo ""
echo "============================================"
echo "  Setup complete."
echo "============================================"

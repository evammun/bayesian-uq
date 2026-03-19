# llama-cpp-python Prototype

This is an isolated prototype for testing direct GPU inference via llama-cpp-python as a replacement for Ollama's HTTP API.

## Motivation

Ollama uses an HTTP API to communicate with local models, which adds overhead (~75% of GPU time wasted on I/O). This prototype tests whether direct inference via llama-cpp-python can reduce latency and improve GPU utilization.

## Files

- **llama_client.py**: A self-contained client that mirrors `OllamaClient`'s interface but uses llama-cpp-python directly
  - Supports three modes: `direct` (single-token), `cot` (chain-of-thought), `cot_structured`
  - Extracts logprobs in the same format as Ollama for compatibility
  - No streaming (llama-cpp-python limitation)

- **test_pilot.py**: Quick test with 4 questions to verify the client works
  - Loads from the project's `data/questions.json`
  - Prints question, model answer, probabilities, and correctness

- **benchmark.py**: Compare Ollama and llama-cpp-python on N questions
  - Measures wall time per query
  - Compares logprob values and answers
  - Prints speedup comparison

## Installation

First install llama-cpp-python:

```bash
pip install llama-cpp-python
```

On some systems (especially with older CUDA/ROCm), you may need to install from source:

```bash
pip install llama-cpp-python --no-binary llama-cpp-python
```

Check the [llama-cpp-python repo](https://github.com/abetlen/llama-cpp-python) for GPU-specific instructions.

## Finding the Model File

The Ollama model is typically stored in one of these locations:

**Windows:**
```
C:\Users\<username>\.ollama\models\blobs\sha256-<hash>
```

**macOS/Linux:**
```
~/.ollama/models/blobs/sha256-<hash>
```

To find the exact path, check which model is loaded in Ollama and locate its blob:

```bash
ls ~/.ollama/models/blobs/
```

The blob filename is a long hex hash. It's the GGUF file you need.

Alternatively, if the Ollama directory is not accessible from the sandbox, you'll need to:
1. Download the model to a known location
2. Pass that path to the prototype scripts

For Qwen3 8B Q4_K_M:
```bash
# Using ollama CLI
ollama pull qwen3:8b-q4_K_M

# Then find the file (it's symlinked in ~/.ollama/models/blobs/)
ls -lh ~/.ollama/models/blobs/ | grep qwen
```

## Running the Tests

### Quick Pilot Test (4 questions)

```bash
python test_pilot.py \
  --model-path /path/to/qwen3-8b-q4_K_M.gguf \
  --prompt-mode direct
```

Or test both modes:

```bash
python test_pilot.py \
  --model-path /path/to/qwen3-8b-q4_K_M.gguf \
  --prompt-mode both
```

Output shows:
- Question text and choices
- Model's predicted answer
- Probability distribution (softmax of logprobs)
- Whether it matches the ground truth
- Wall time per query

### Benchmark (Compare Ollama vs. llama-cpp-python)

```bash
python benchmark.py \
  --model-path /path/to/qwen3-8b-q4_K_M.gguf \
  --n 10 \
  --prompt-mode direct
```

Requires Ollama running and the OllamaClient available from the project.

Output shows:
- Per-question wall time for both backends
- Speedup ratio (Ollama time / llama-cpp-python time)
- Summary statistics (avg, min, max, total)
- Answer agreement between backends

## Data Format

Both scripts expect the same question format as the main project:

```json
{
  "question_id": "mmlu_redux_...",
  "question_text": "...",
  "choices": ["A text", "B text", "C text", "D text"],
  "correct_answer": 1,
  "subject": "abstract_algebra",
  "variant": "valid",
  ...
}
```

Questions are filtered to those where:
- `variant` is `"valid"` or `null`
- `correct_answer` is not `null`

## Implementation Notes

### Logprobs Format Conversion

llama-cpp-python returns logprobs in a different format than Ollama:

**llama-cpp-python:**
```python
output["choices"][0]["logprobs"]["top_logprobs"] = [
    {" B": -0.12, " A": -3.21, ...},  # position 0
    {" C": -0.05, " B": -2.1, ...},   # position 1
]
```

**Ollama (what we use internally):**
```python
[
    {"top_logprobs": [{"token": " B", "logprob": -0.12}, ...]},  # pos 0
    {"top_logprobs": [{"token": " C", "logprob": -0.05}, ...]},  # pos 1
]
```

The `_convert_logprobs()` method handles this conversion automatically.

### Token Matching

Answer tokens may be tokenized differently across models and between Ollama and llama-cpp-python:
- Token could be "A", " A", "a", " a", etc.
- Code normalizes to the canonical letter via `_token_to_letter()`
- If a letter appears in both forms (e.g., "A" and " A"), the one with higher logprob is kept

### CoT Two-Pass

For chain-of-thought modes:
1. **Pass 1**: Generate reasoning with `max_tokens=300`, stop at `"\nAnswer:"`
2. **Pass 2**: Strip the answer from context, then complete the answer token with `max_tokens=1` to get clean logprobs

This ensures we get a proper probability distribution instead of a near-certain token that's already been committed.

### GPU Acceleration

The client uses `n_gpu_layers=-1` by default, which offloads all model layers to GPU. Adjust this if you run out of VRAM:
- `-1`: All layers (requires most VRAM)
- `40`: Typical for 8B models on 8GB VRAM
- `20`: Conservative (some layers stay on CPU)
- `0`: CPU-only (slow)

## Known Limitations

1. **No streaming**: llama-cpp-python doesn't stream completions like Ollama does. All tokens are generated in one pass.
2. **No think mode**: Hidden reasoning with extended context is not implemented yet.
3. **Model loading time**: First call includes model compilation (~10-20s). Subsequent calls are cached.
4. **Context window**: Fixed context size (no dynamic batching like Ollama's context slots).

## Performance Expectations

On NVIDIA RTX 3070 (8GB VRAM) with Qwen3 8B Q4:
- **Direct mode**: ~0.5-1.0s per query (mostly CUDA kernel launch overhead)
- **CoT mode**: ~2-5s per query (reasoning generation + answer extraction)
- **Ollama direct mode**: ~3-5s per query (HTTP overhead dominates)
- **Expected speedup**: 3-5x faster for direct mode, 1-2x for CoT

## Debugging

If a query fails:

1. **Model not loading**: Check the path exists and is readable:
   ```bash
   ls -lh /path/to/model.gguf
   ```

2. **CUDA/GPU errors**: Check CUDA/cuBLAS installation:
   ```bash
   python -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
   ```

3. **Memory errors**: Reduce `n_gpu_layers` or increase swap:
   ```bash
   nvidia-smi  # check free memory
   ```

4. **Answer tokens missing**: Check model tokenization — some models may tokenize differently. The code falls back to scanning top_logprobs for any answer letter.

## Next Steps

Once this prototype is validated:
1. Integrate into the main pipeline as an optional backend
2. Profile both backends on full experiment (5330 questions)
3. Measure actual throughput and VRAM usage
4. Consider caching compiled models or using batching
5. Explore quantization further (int8 on VRAM-constrained systems)

# Prototype Manifest

## Created Files

### Core Implementation

**`llama_client.py`** (19 KB)
- Self-contained LlamaCppClient class
- Interface mirrors OllamaClient from the project
- Supports: direct, cot, cot_structured modes
- Key methods:
  - `send_query()` → tuple[str, list[dict], str, dict | None]
  - `_direct_completion()` → single-token extraction with logprobs
  - `_cot_reasoning()` → reasoning generation with stop sequence
  - `_convert_logprobs()` → format conversion from llama-cpp-python to Ollama format
- Utility functions:
  - `extract_answer_logprobs()` → extract answer probabilities
  - `generate_permutation()` → answer shuffling
  - `_token_to_letter()` → token normalization (handles " A", "A", etc.)
  - `_find_last_answer_token_logprobs()` → CoT last-token extraction

**`__init__.py`** (330 bytes)
- Package exports for clean imports
- Exports: LlamaCppClient, extract_answer_logprobs, generate_permutation, constants

### Test & Benchmark Scripts

**`test_pilot.py`** (7.7 KB)
- Loads 4 questions from project's `data/questions.json`
- Runs them through LlamaCppClient in one or both modes
- Prints: question, model answer, probability distribution, correctness, wall time
- Usage:
  ```bash
  python test_pilot.py --model-path /path/to/model.gguf --prompt-mode direct
  python test_pilot.py --model-path /path/to/model.gguf --prompt-mode both
  ```
- Accepts: `--model-path`, `--prompt-mode {direct|cot|cot_structured|both}`, `--num-questions`

**`benchmark.py`** (9.9 KB)
- Compares Ollama and llama-cpp-python on N questions
- Measures:
  - Wall time per query for both backends
  - Logprob values (should be similar)
  - Predicted answers (should match)
  - Speedup ratio and overall improvement
- Imports OllamaClient from project (graceful fallback if unavailable)
- Usage:
  ```bash
  python benchmark.py --model-path /path/to/model.gguf --n 10 --prompt-mode direct
  ```
- Accepts: `--model-path`, `--n {number}`, `--prompt-mode`

### Documentation

**`README.md`** (6.7 KB)
- Full documentation covering:
  - Motivation (75% overhead from Ollama HTTP API)
  - Installation (including GPU-specific instructions)
  - Finding the model file (Windows, macOS, Linux paths)
  - Running tests (pilot and benchmark)
  - Data format and question filtering
  - Implementation details (logprobs format, token matching, CoT two-pass, GPU acceleration)
  - Known limitations and performance expectations
  - Debugging guide
  - Next steps for integration

**`QUICKSTART.md`** (1.5 KB)
- Fast-track guide for getting started
- 5 steps: install, find model, run pilot, run benchmark, troubleshoot
- Typical expected outputs
- Quick reference for common issues

**`MANIFEST.md`** (this file)
- Inventory of all created files
- Purpose of each file
- Key code structure and exports
- Integration points with the main project

## Design Decisions

### Isolation
- All code in `experiments/llama_cpp_prototype/` is completely self-contained
- Can be tested independently without touching `src/` or main pipeline
- No changes to existing code needed
- Can be deleted without affecting project

### Compatibility
- LlamaCppClient mirrors OllamaClient's `send_query()` interface exactly
- Same return signature: `tuple[str, list[dict], str, dict | None]`
- Reuses same `extract_answer_logprobs()` logic (adapted for llama-cpp format)
- Same permutation and token matching code

### Format Conversion
- llama-cpp-python returns logprobs as: `{" B": -0.12, " A": -3.21, ...}`
- Ollama returns: `{"top_logprobs": [{"token": " B", "logprob": -0.12}, ...]}`
- `_convert_logprobs()` handles the transformation
- All downstream code sees Ollama-compatible format

### CoT Two-Pass
- Pass 1: Generate reasoning with stop sequence
- Pass 2: Extract answer token logprobs via single-token completion
- Ensures proper probability distribution (not near-certain committed token)
- Code validates Pass 2 success, falls back to Pass 1 if needed

### GPU Acceleration
- Uses `n_gpu_layers=-1` by default (offload all layers)
- Users can adjust for VRAM constraints
- Falls back gracefully if GPU unavailable (will be slow on CPU)

## Integration Checklist

When ready to integrate into main pipeline:

- [ ] Validate against full 5330 questions (speed + accuracy)
- [ ] Measure VRAM usage under load
- [ ] Compare logprob values precisely (tolerance bounds)
- [ ] Profile CPU/GPU utilization
- [ ] Create `src/bayesian_uq/query_llamacpp.py` as backend option
- [ ] Add to ExperimentConfig as `backend: "ollama" | "llamacpp"`
- [ ] Update pipeline.py to select backend based on config
- [ ] Add tests for backend swapping
- [ ] Document VRAM/GPU requirements
- [ ] Consider model caching for warm starts

## File Locations

```
experiments/llama_cpp_prototype/
├── __init__.py           (exports)
├── llama_client.py       (main implementation)
├── test_pilot.py         (4-question test)
├── benchmark.py          (speed comparison)
├── README.md             (full docs)
├── QUICKSTART.md         (fast track)
└── MANIFEST.md           (this file)
```

## Dependencies

**Required:**
- llama-cpp-python >= 0.2.0
- (GPU-optional: CUDA/cuBLAS for GPU acceleration)

**For benchmark.py:**
- bayesian_uq project (for OllamaClient, optional fallback if unavailable)
- Ollama running (for comparison, optional)

## Performance Targets

Expected on NVIDIA RTX 3070 with Qwen3 8B Q4:
- Direct mode: 0.5-1.0s/query (vs 3-5s with Ollama) = 3-5x speedup
- CoT mode: 2-5s/query (vs 4-8s with Ollama) = 1-2x speedup
- VRAM: ~6-7GB (vs full GPU memory for Ollama context slots)

## Known Limitations & TODOs

- [ ] No streaming (llama-cpp-python limitation)
- [ ] No think mode (can be added, not in MVP)
- [ ] No context slot management (fixed n_ctx per client)
- [ ] No batching (one query at a time)
- [ ] Model loading time ~10-20s first call (cached after)
- [ ] No answer shuffling support yet (always identity permutation in CoT)

## Testing Done

- [x] Python syntax check (all files)
- [x] Import validation (llama_client, dependencies)
- [x] Code structure review (matches OllamaClient interface)
- [ ] End-to-end test with actual model (requires model file)
- [ ] Speed comparison (requires Ollama + model file)
- [ ] Full experiment validation (5330 questions)

## Next Phase

Once model file is accessible:

1. Run test_pilot.py to validate basic functionality
2. Run benchmark.py to quantify speedup
3. Profile on full experiment dataset
4. Finalize integration points and decision criteria

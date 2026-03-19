# Quick Start

## 1. Install llama-cpp-python

```bash
pip install llama-cpp-python
```

On systems with NVIDIA GPU, this should auto-detect CUDA. If not:

```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --no-binary llama-cpp-python
```

## 2. Find your model file

The Qwen3 8B Q4_K_M model that Ollama uses is stored in:

```bash
# Find the model blob (long hash filename)
ls ~/.ollama/models/blobs/ | head

# It will be something like:
# sha256-3a4f5b8c9d2e1f0a...

# The full path is:
~/.ollama/models/blobs/sha256-3a4f5b8c9d2e1f0a...
```

Or if you're on Windows:
```
C:\Users\<username>\.ollama\models\blobs\sha256-<hash>
```

## 3. Run the 4-question pilot

Test that everything works:

```bash
python test_pilot.py \
  --model-path /path/to/qwen3-model.gguf \
  --prompt-mode direct
```

Expected output:
```
[INFO] Model loaded in 12.34s
[INFO] Loading questions from .../data/questions.json
[INFO] Loaded 4 valid questions

======================================================================
MODE: direct
======================================================================

[Q1] mmlu_redux_abstract_algebra_0000
Subject: abstract_algebra
Question: Determine whether the polynomial...
Choices:
  A) Yes, with p=2.
  B) Yes, with p=3.
  C) Yes, with p=5.
  D) No.
Correct answer: B
  Time: 0.85s
  Model answer: B (canonical: 1)
  Correct: True
  Probs: [A: 0.012 | B: 0.876 | C: 0.098 | D: 0.014]
  Response: " B"
...
```

## 4. Run a quick benchmark (if Ollama is running)

Compare speed with Ollama:

```bash
# Make sure Ollama is running in another terminal
python benchmark.py \
  --model-path /path/to/qwen3-model.gguf \
  --n 5 \
  --prompt-mode direct
```

This will show speedup like:
```
Speedup (Ollama / llama-cpp-python): 3.5x
Improvement: 71% faster with llama-cpp-python
```

## 5. Troubleshooting

**Model not found:**
```bash
# Check the path exists
ls -lh /path/to/model.gguf
```

**CUDA not working:**
```bash
# Test CUDA from Python
python -c "from llama_cpp import Llama; print(Llama.supported_backends())"
```

**Out of VRAM:**
- Reduce `-n_gpu_layers` from `-1` (all) to `30` or `20`
- Or run on CPU with `n_gpu_layers=0` (very slow, but diagnostic)

## Next Steps

1. **Integrate into pipeline**: Once validated, move the client into the main pipeline
2. **Full experiment**: Run on all 5330 questions
3. **Profile VRAM**: Monitor memory usage during full runs
4. **Compare accuracy**: Ensure llama-cpp-python matches Ollama's answers

## Files

- `llama_client.py` — Main client (self-contained, mirrors OllamaClient)
- `test_pilot.py` — 4-question test
- `benchmark.py` — Speed comparison with Ollama
- `README.md` — Full documentation
- `__init__.py` — Package exports
- `QUICKSTART.md` — This file

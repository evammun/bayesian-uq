# Architecture Brainstorm — Replacing Ollama

**Created:** 2026-03-19
**Context:** Ollama's HTTP-per-request overhead wastes 75%+ of GPU time on rented hardware. This document evaluates alternatives for future experiment runs (Experiment 2, other models, temperature sensitivity).

---

## Why Ollama Was Chosen (And Why It's Now The Bottleneck)

Ollama was the right choice for getting started: one-command install, simple API, manages model downloads, handles quantisation natively, works on Windows. For the initial direct-mode experiments (single token, 200ms per query), the HTTP overhead was negligible.

The problem emerged with CoT and think mode: each query now requires 2 HTTP round-trips (two-pass pipeline), each with ~0.5-1s of overhead from Python → HTTP → Go scheduler → llama.cpp → Go scheduler → HTTP → Python. On the RTX 5090, we measured 0.94s of actual GPU work inside 6s of wall time — 84% overhead. The GPU sits at 6-30% utilisation regardless of how many experiments we run.

Ollama's architecture (Go HTTP server wrapping llama.cpp) adds overhead at every layer: HTTP serialization, JSON parsing, request routing, Go channel synchronisation, and linear request queuing. None of this can be configured away.

---

## The Options

### Option 1: llama-cpp-python (In-Process — Recommended)

**What it is:** Python bindings for llama.cpp. The model runs directly in your Python process. No HTTP, no server, no serialisation.

**Why it's the best fit:**

- **Zero HTTP overhead.** Direct function calls. The measured 5s of overhead per request disappears entirely.
- **GGUF native.** Uses the exact same model files Ollama already downloaded. Point it at `~/.ollama/models/blobs/sha256-<hash>` or re-download the GGUF from HuggingFace.
- **Logprobs supported.** `llm.create_completion(prompt="...", logprobs=20)` returns top-N logprobs per token position. Same data we're already extracting.
- **Simple API.** Direct mode becomes:
  ```python
  from llama_cpp import Llama
  llm = Llama(model_path="qwen3-8b-q4_k_m.gguf", n_ctx=2048)
  out = llm.create_completion(
      prompt=f"{question}\n\n{choices}\n\nAnswer:",
      max_tokens=1,
      logprobs=20,
  )
  # out["choices"][0]["logprobs"]["top_logprobs"][0] → {"A": -0.12, "B": -3.21, ...}
  ```
- **CoT two-pass becomes trivial:**
  ```python
  # Pass 1: generate reasoning
  out1 = llm.create_completion(prompt=prompt, max_tokens=200, stop=["\nAnswer:"])
  reasoning = out1["choices"][0]["text"]

  # Pass 2: extract answer logprobs
  out2 = llm.create_completion(
      prompt=f"{prompt}{reasoning}\nAnswer:",
      max_tokens=1,
      logprobs=20,
  )
  ```
  No endpoint switching, no model reloads, no HTTP. Both calls use the same loaded model in memory.

**Estimated speedup:** llama.cpp is ~27% faster than Ollama in benchmarks, plus eliminating HTTP overhead. For our CoT two-pass queries: ~6s → ~1.5s per query. For an 11-query question: ~66s → ~17s. For 5,330 questions: ~98 hours → ~25 hours.

**Limitations:**
- No continuous batching (single inference at a time per model instance)
- No built-in model management (have to point at GGUF files manually)
- Threading needs care (llama.cpp has its own thread pool for inference)

**Code changes required:**
- Replace `OllamaClient` with a `LlamaCppClient` that wraps llama-cpp-python
- `send_query` stays the same interface but calls `llm.create_completion()` instead of `requests.post()`
- `extract_answer_logprobs` needs minor adaptation (logprobs format is slightly different: dict of token→logprob instead of list of {token, logprob} dicts)
- Remove all HTTP-related retry logic (no network errors possible)
- Pipeline, config, analysis: untouched

**Effort estimate:** 2-3 hours for a working prototype. The OllamaClient is ~300 lines; a LlamaCppClient would be ~150.

---

### Option 2: llama-server (llama.cpp's HTTP server)

**What it is:** llama.cpp's built-in HTTP server, which is what Ollama wraps. Running it directly eliminates Ollama's Go layer.

**Advantages:**
- Continuous batching (`--cont-batching` flag) — processes multiple requests in one GPU forward pass. This is what would actually saturate the 5090.
- OpenAI-compatible API (mostly)
- GGUF native
- Logprobs via `n_probs` parameter

**Disadvantages:**
- Still HTTP-based (though lower overhead than Ollama)
- Logprobs API uses `n_probs` not `top_logprobs` — needs adaptation
- Less well-documented than Ollama
- Need to find and point at the raw GGUF file

**When to use it:** If we need to run many concurrent experiments on a rented GPU and want continuous batching. The HTTP overhead per request is lower than Ollama but still present.

**Code changes:** Similar to Option 1 but keep the HTTP client architecture, just change the API format slightly.

---

### Option 3: vLLM

**What it is:** Production-grade inference engine with PagedAttention and continuous batching.

**Why NOT for us:**
- GGUF support is poor — vLLM is designed for full-precision or GPTQ/AWQ quantised models
- Measured 93 tok/s with 958ms TTFT on GGUF — slower than native llama.cpp
- Overkill for single-model, single-GPU experiments
- Complex setup

**When it would make sense:** If we scaled to multiple models, multiple GPUs, or production deployment. Not for research experiments on consumer hardware.

---

### Option 4: SGLang

**What it is:** Research-oriented serving framework with continuous batching and advanced scheduling.

**Status:** Supports GGUF and logprobs, but has known issues with empty top_logprobs on some models. More production-grade than Ollama but heavier than llama-server.

**Verdict:** Consider if we need advanced features (constrained decoding, parallel function calling). For our use case, it's more complex than needed.

---

## Recommendation

**For the laptop (RTX 3070, 8GB VRAM):** Switch to **llama-cpp-python**. The in-process approach eliminates all HTTP overhead, which is the dominant bottleneck. Direct mode queries would go from ~200ms to ~150ms. CoT two-pass from ~10s to ~2s. Think mode from ~120s to ~60-80s. The simplicity of direct function calls also simplifies the code.

**For rented GPUs (5090, A100, etc.):** Use **llama-server** with continuous batching. Multiple experiments can share a single server, and continuous batching actually saturates the GPU. This is the configuration that would push utilisation from 25% to 70%+.

**The migration path:**
1. Create a `LlamaCppClient` that implements the same `send_query()` interface as `OllamaClient`
2. Make the choice between them configurable (env var or config file)
3. Keep `OllamaClient` working for backward compatibility
4. Test on the 4-question pilot, verify logprobs match
5. Run Experiment 2 with the new backend

**What to keep from Ollama:** The model files. Ollama downloads GGUF blobs to `~/.ollama/models/`. We can either point llama-cpp-python at those files directly, or download the same GGUF from HuggingFace.

---

## Implementation Sketch

```python
# New file: src/bayesian_uq/llama_client.py

from llama_cpp import Llama
from .config import ANSWER_LETTERS

class LlamaCppClient:
    """Direct llama.cpp inference via Python bindings. No HTTP overhead."""

    def __init__(self, model_path: str, n_ctx: int = 2048, temperature: float = 0.7):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,  # offload everything to GPU
            logits_all=True,  # needed for logprobs
            verbose=False,
        )
        self.temperature = temperature

    def send_query(self, question_text, choices, answer_permutation):
        """Same interface as OllamaClient.send_query()."""
        # Build prompt (same as current code)
        # ...

        # Direct mode: single token completion
        out = self.llm.create_completion(
            prompt=prompt_text,
            max_tokens=1,
            temperature=self.temperature,
            logprobs=20,
        )

        # Extract logprobs
        top_lp = out["choices"][0]["logprobs"]["top_logprobs"][0]
        # top_lp is {"A": -0.12, " B": -3.21, "C": -5.43, ...}

        # Convert to our format and return
        # ...
```

The key insight: `send_query()` keeps the same interface. Everything above it (pipeline, analysis, dashboard) stays untouched. We're only replacing the bottom layer.

---

## Timeline

- **Now:** Let current Ollama experiments finish. Analyse results.
- **Before Experiment 2:** Implement LlamaCppClient (2-3 hours).
- **Validate:** Run the 4-question pilot, compare logprobs between Ollama and llama-cpp-python. They should match (same llama.cpp engine underneath).
- **Run Experiment 2:** With the new backend. Expect 3-4× speedup on rented GPU.

---

## Why We Picked Ollama In The First Place

It was the right call at the time:
- One-command install on Windows (`winget install ollama`)
- Model management (`ollama pull qwen3:8b-q4_K_M`)
- Simple API that worked for V1 (JSON schema) and V2 (logprobs)
- Think mode support via `think: true`
- Eva was new to local LLM deployment — Ollama has the gentlest learning curve

The overhead only became a problem when we moved from 1 query/question (direct mode, 200ms) to 22 queries/question (CoT two-pass × 11 paraphrases, 66s). At 200ms/query, HTTP overhead is 20% of wall time. At 22 queries × 2 passes × 0.5s overhead each, it's 84%. The architecture that was fine for prototyping became the bottleneck for production experiments.

This is normal engineering — you don't optimise the inference layer before you know what queries you'll be running.

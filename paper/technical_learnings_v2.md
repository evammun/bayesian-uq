# Technical Learnings from v2 MMLU Experiments

Hard-won lessons from the prior project. Reference this before building the new pipeline.

---

## Inference Backend

### Ollama limitations (why we're switching to llama-cpp-python)
- Ollama wraps llama.cpp in a Go HTTP server → adds ~0.5-1s overhead per request
- On rented GPU (RTX 5090), measured 0.94s GPU work inside 6s wall time — 84% overhead
- No way to configure away the overhead — it's architectural (HTTP serialization, Go channel sync, linear request queuing)
- For CoT two-pass: ~6s → ~1.5s per query by switching to llama-cpp-python (in-process, zero HTTP)

### llama-cpp-python is the right choice for us
- Direct Python bindings to llama.cpp. Model runs in-process. No HTTP, no server.
- Uses same GGUF files Ollama downloads (~/.ollama/models/blobs/ or download from HuggingFace)
- Logprobs: `llm.create_completion(prompt="...", logprobs=20)` → top-N per token
- Two-pass CoT trivial: pass 1 generates reasoning with stop sequence, pass 2 extracts logprobs
- ~27% faster than Ollama even ignoring HTTP overhead
- Limitations: no continuous batching, no built-in model management, threading needs care

### llama-server (alternative if we need batching)
- llama.cpp's built-in HTTP server — less overhead than Ollama, supports continuous batching
- Useful for rented GPU with concurrent experiments
- Logprobs via `n_probs` parameter (different API format from Ollama)

### vLLM: skip it
- GGUF support poor, slower than native llama.cpp on GGUF, overkill for our setup

---

## Logprob Extraction

### Structured output kills the signal
- JSON schema enforcement (`{"answer": "B"}`) → logprobs at answer position spike to 99.99%
- The scaffolding tokens absorb all uncertainty before the decision token
- **Fix:** Use text completion, end prompt with `Answer:`, set max_tokens=1, request top_logprobs=20

### CoT scaffolding absorption (same problem, weaker)
- After writing reasoning, answer-token logprobs spike to ~90-99%
- **Fix:** Two-pass pipeline:
  - Pass 1: generate reasoning (with stop sequence at `\nAnswer:`)
  - Pass 2: new completion with reasoning prepended, extract logprobs at answer position
  - This recovers meaningful spread in logprob distributions

### Use /api/generate (not /api/chat) or raw completion
- /api/chat applies chat templates that add system/assistant/user framing
- For controlled logprob extraction, raw completion is cleaner
- With llama-cpp-python this becomes `llm.create_completion()` — no chat template issue

### Top-20 logprobs contain rich signal
- Only 4 slots are answer letters; other ~16 are what the model *wanted* to say (reasoning starters, hedging, formatting)
- "Answer coverage" = probability mass on answer letters vs total top-20 mass — novel signal
- If top-1 token isn't an answer letter, model wanted to explain rather than answer

---

## Speed & Performance

### Windows localhost bug
- `localhost` resolves via IPv6 on Windows → 2s delay per HTTP request to Ollama
- **Fix:** Use `127.0.0.1` directly. Saved ~2s per request (2.2s → 0.19s for direct mode)
- Not relevant if using llama-cpp-python (no HTTP), but worth remembering for any local HTTP services

### Ollama parallel workers
- `OLLAMA_NUM_PARALLEL=3` for concurrent request processing
- Only helps for short queries (direct mode, ~200ms each). GPU already busy during CoT.

### Smart query scheduling
- no-para + no-shuffle = 1 query (identical prompt → identical logprobs, repetition pointless)
- Don't waste compute on conditions that produce duplicate data

### CoT verbosity control
- Qwen 8B ignores adjective-based brevity ("briefly", "concise")
- **Fix:** Structured ✓/✗ format with one-shot example → responses 1500→300 chars, 17s→8.8s per question
- System message "You are a concise exam grader..." helps as behavioral anchor

### Always save incrementally
- Long-running scripts: write results after every batch, not just at the end
- Design for resumability — check for existing results before re-running

---

## Paraphrase Generation

### Anthropic API (Claude Sonnet) for offline generation
- 10 paraphrases per question, ~$5 for 5,330 questions
- Quality: conservative (synonym substitution, sentence restructuring). More aggressive reframing can crack questions that conservative paraphrases can't — worth exploring as a variable.
- Validate: coverage check (all questions have N paraphrases), duplicate detection, original text matching

### Temperature for logprobs
- Logprobs reflect the post-temperature distribution — higher T = softer spread
- T=0.7 used in v2. T=0.0 gives raw model distribution. Effect on AUROC untested.
- For the new project, decide upfront and document.

---

## Model-Specific Notes (Qwen 3/3.5 8B)

- Qwen 3 8B Q4_K_M: ~76% on MMLU-Redux (vs 86.4% published FP16 with thinking). Gap from quantisation (~5pp) + no thinking (~5pp).
- Qwen 3.5: thinking enabled by default, can't soft-toggle mid-conversation. Disable at server level: `--chat-template-kwargs '{"enable_thinking": false}'`
- Think mode produces `<think>...</think>` before response. Local inference gives logprobs on thinking tokens too (cloud APIs block this).
- CoT accuracy *lower* than direct (68.6% vs 75.5%) — reasoning-induced error in small models.

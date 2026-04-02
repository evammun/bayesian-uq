# Pre-Action UQ — Running Notes & Decision Log

**Authors:** Eva Martin, Professor Luigi
**Working title:** TBD
**Started:** March 2026 (pivot from MMLU-based UQ)
**Status:** Brainstorming / design phase. No experiments yet.

---

## 0. Prior Work Summary (v2 MMLU Project, March 4-19 2026)

Full archive in `v2_mmlu_archive/`. Quick summary of what was built, what we found, and what transfers.

### What we built
- **V1 (sampling-based):** Dirichlet posterior with adaptive stopping over repeated model queries. Paraphrase + shuffle variations. 5-8 hours per condition. Demonstrated convergence speed as diagnostic signal (correct answers converge 3× faster).
- **V2 (logprob-based):** Single-query logprob extraction per paraphrase. 100× faster than V1. Full [P(A),P(B),P(C),P(D)] distribution from first output token, normalised over answer letters. 11 queries per question (1 original + 10 paraphrases).
- **Signal taxonomy:** ~25 uncertainty signals across 5 tiers — within-prompt (MSP, entropy, answer coverage), across-prompt (agreement, epistemic/aleatoric, confidence variance, rank stability), position sensitivity, 2D confidence×consistency space, cross-condition signals.
- **Infrastructure:** Ollama-based pipeline, YAML experiment configs, Streamlit monitoring dashboard, Anthropic API paraphrase generation.

### Key findings
- **Fragile confidence:** High single-prompt confidence + low cross-paraphrase consistency = dangerous failure mode, invisible to any single metric.
- **Scaffolding absorption:** JSON schema and CoT reasoning absorb uncertainty into scaffolding tokens before the answer token, spiking logprobs to ~100%. Fix: two-pass pipeline (reasoning in pass 1, logprob extraction in pass 2).
- **Answer coverage (novel):** Off-label probability mass on non-answer tokens — model "leaking" probability to reasoning/hedging tokens. Uniquely accessible from local models with top-N logprobs.
- **CoT accuracy drop:** 68.6% vs 75.5% direct — reasoning-induced error in small models. Known phenomenon (CCoT, Renze & Guven 2024).
- **Accuracy flat across conditions** (~76-77%). AUROC modest. Results methodologically interesting but not compelling as standalone paper.

### What transfers to new project
- Paraphrase generation pipeline (Anthropic API, 10 per question)
- Logprob extraction + normalisation logic
- Signal computation framework (adapted for non-MCQ setting)
- Two-pass pipeline pattern (reasoning → logprob extraction)
- 2D uncertainty space (confidence × consistency)
- Technical learnings — see `technical_learnings_v2.md`

---

## March 25, 2026 — The Pivot

### Why we moved on from MMLU
- v2 results (archived `v2_mmlu_archive/`): 10 conditions, 5,330 questions, ~25 uncertainty signals
- Interesting methodological findings (fragile confidence, scaffolding absorption, answer coverage) but AUROC modest, accuracy flat across conditions (~76-77%)
- Contributions were about the *measurement instrument*, not a real problem. Nobody deploys Qwen 8B for MMLU in production.

### The new direction
- Pre-action UQ: before a local LLM acts on query + RAG context, verify it understood the situation
- Key reframe: output-side UQ ("is the answer right?") → input-side UQ ("did the model understand what it received?")
- Three uncertainty types: query comprehension, context sufficiency, action selection

### Lit review completed
- Full review in `lit_review_pre_action_uq.md`
- **Gap confirmed:** Nobody probes input-processing uncertainty pre-generation. Closest: SPUQ (output consistency), S2AF (post-hoc), CoCA (training-based)
- **RAG paradox** (Google Sufficient Context, ICLR 2025): retrieval *suppresses* abstention — bad context → more confident hallucination
- **AbstentionBench** (Meta, NeurIPS 2025): reasoning fine-tuning degrades abstention by ~24%. Validates our scaffolding absorption finding at scale.
- **Datasets exist:** MuSiQue-Full (answerable/unanswerable pairs), SQuAD 2.0 (is_impossible), BFCL (tool-calling), MetaTool, AbstentionBench

### What transfers from v2
- Paraphrase-based logprob extraction pipeline, full signal taxonomy, 2D uncertainty space
- Tech stack (Ollama/llama-cpp-python, Qwen 8B, YAML configs)
- Scaffolding absorption finding (now externally validated)

### Project restructured
- All v2 work → `v2_mmlu_archive/`
- Fresh structure: `src/pre_action_uq/`, `data/`, `results/`, `paper/`, `experiments/configs/`

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-25 | Pivot to pre-action UQ | MMLU results not compelling; new direction = real unsolved problem |
| 2026-03-25 | Archive v2, fresh structure | Clean slate for new direction |
| 2026-03-25 | BFCL for tool-calling component (RQ3) | Irrelevance detection + missing params = proceed/clarify/escalate labels |
| 2026-03-25 | Reconsidered MuSiQue — not a great fit | Tests multi-hop reasoning chains, not comprehension of input. Artificial questions, extractive format, paraphrase signal muddied by reasoning path variation. See `dataset_evaluation.md` |
| 2026-03-25 | RGB as starting dataset for RQ2 | Negative rejection testbed = exactly our scenario. Noise robustness gradient gives continuous signal. Tested on Qwen-7B. Small but diagnostic. |
| 2026-03-25 | RQ2 (context sufficiency) is the load-bearing RQ | Cleanest operationalisation, strongest motivation (RAG paradox), most direct pipeline transfer. RQ1/RQ3 are real but less developed — test separately. |
| 2026-03-30 | **RGB dropped, replaced by QuALITY** | See March 30 notes below. |

---

## March 30, 2026 — Dataset Change: RGB → QuALITY

### Why RGB is out
After actually examining the data (300 refine, 100 integration, 100 counterfactual), fundamental problems:
- **All queries are simple factoid lookups:** "Who won X?", "How much did Y cost?", "When was Z?" Answers are short extractive spans (names, numbers, dates). Tests information *retrieval*, not *comprehension*.
- **Too small:** 500 total questions, only 100 with counterfactual condition. Not enough for meaningful signal analysis.
- **Not our claim:** We claim to measure comprehension uncertainty. A dataset where comprehension doesn't matter undermines the entire paper. A reviewer would immediately flag this.
- **Free-form answers = no clean logprob extraction point.** Would need to bolt on MCQ probes or sufficiency probes — adding complexity without adding scientific value.

RGB data deleted from `data/`. No trace remaining.

### Why QuALITY (Pang et al., NAACL 2022)
- **~6,000 MCQ questions across ~260 articles**, 4 options per question, ~5K-token passages from Project Gutenberg (fiction + non-fiction). CC BY 4.0.
- **Questions genuinely require comprehension.** Written by people who read the full article. Hard subset = questions speed-readers got wrong but careful readers got right. Literally a comprehension test by construction.
- **MCQ format maps directly to v2 pipeline.** A/B/C/D logprob extraction, same uncertainty signals, same compute_signals.py, same dashboard. Minimal port effort.
- **~5K token passages = realistic RAG context length.** Fits in Qwen 8B context window.
- **We construct our own context conditions** (see brainstorm.md §4.1 for details):
  1. Sufficient: correct passage + question
  2. Insufficient: wrong passage (same genre, different article)
  3. Partial: truncated passage (answer section removed)
  4. Counterfactual: passage with key facts modified
- **Hard subset gives us a natural difficulty axis.** Easy questions (speed-readers got right) vs hard questions (only careful readers got right). Prediction: uncertainty signals should be more discriminative on the hard subset.

### What this means for the pipeline
- MCQ format → reuse v2 logprob extraction almost unchanged
- Context conditions are constructed, not dataset-provided → cleaner experimental control
- Paraphrase the *question* across conditions (same as v2), extract logprobs over A/B/C/D
- Think vs no-think as an additional factor (scaffolding absorption in comprehension setting)

### Dataset-to-RQ mapping (updated)
- **RQ2 (context sufficiency):** QuALITY with constructed context conditions (sufficient/insufficient/partial/counterfactual)
- **RQ1 (query comprehension):** QuALITY paraphrase consistency (free — same experiment, different analysis)
- **RQ3 (action selection):** BFCL (deferred for now)

---

## March 30, 2026 (cont.) — Pipeline Port & Inference Layer

### llama-cpp-python installation (Windows)
- **Official CUDA wheels are broken on Windows** for versions ≥0.3.5 (GitHub issue #1967). Wheels listed in the index 404 on download.
- **Solution:** JamePeng's community fork ([github.com/JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python/releases)) provides working pre-built CUDA wheels. Installed `llama_cpp_python-0.3.33+cu124.basic` (711MB wheel, ggml-cuda.dll = 1.3GB). cu124 wheels are backward-compatible with CUDA 12.7.
- **On Linux rentals:** Official index works fine. `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124`
- **CVE-2026-33298:** Integer overflow in llama.cpp, fixed in b7824, requires ≥0.3.16. We're on 0.3.33.

### Critical fix: `logits_all=False` (low-level logit extraction)
The biggest technical discovery of the session. `create_completion(logprobs=N)` in llama-cpp-python requires `logits_all=True`, which allocates a logits buffer of `n_ctx × vocab_size × 4 bytes`. For Qwen3 (vocab ~152K) at n_ctx=8192, that's **~4.7 GB** — OOM on an 8GB RTX 3070 alongside the 5.2GB model.

**The insight:** llama.cpp always computes logits for the last evaluated position (it must, to sample the next token). We only need logprobs at that one position (the answer token). By bypassing `create_completion(logprobs=N)` and going lower-level:
1. `model.tokenize(prompt)` → token list
2. `model.eval(tokens)` → forward pass, `logits_all=False`
3. `llama_cpp.llama_get_logits(ctx)` → pointer to last position's logits
4. Log-softmax → top-k extraction (top-20 + force-include A/B/C/D)

Logits buffer drops from ~4.7GB to ~600KB. Problem gone.

**Key details:**
- `llama_get_logits()` (without `_ith`) is the correct call — returns the last evaluated position
- `llama_get_logits_ith(idx)` returns NULL with `logits_all=False` (only the last position is stored)
- Must copy logits to numpy array before any further model operations (pointer reuse)

### KV cache: `memory_clear(True)` not `reset()`
`Llama.reset()` only sets `n_tokens=0` — it does NOT clear the KV cache. Subsequent `eval()` calls fail with batch decode errors because the internal state is inconsistent. The fix: `model._ctx.memory_clear(True)` properly clears the KV cache. `llama_kv_cache_clear` doesn't exist in this version of llama-cpp-python (0.3.33).

### `n_batch` must be explicit
Without setting `n_batch`, llama-cpp-python uses a default (512 or 2048). Long QuALITY prompts (6000-7000 tokens) can hit batch errors during chunked evaluation. Fix: set `n_batch=n_ctx` so the full prompt processes in one pass.

### Chat template is essential for logprob extraction
Initially, `generate_with_logprobs()` passed the raw prompt to the model without wrapping in Qwen3's chat template. Result: top token was "GSM" with `<think>` at p=0.23. Answer letters A/B/C/D were at logprob -20 to -25 (effectively zero probability). The model was in free-completion mode, not answering mode.

**Fix:** Wrap in `<|im_start|>system\n/no_think\n...<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\nAnswer:` — the "Answer:" is placed as **assistant prefill**, not at the end of the user message. This means the model's first token in the assistant turn is the answer letter. After this fix: top tokens are A/B/C/D with proper probability distributions (p=0.80-0.98 for confident answers).

### Threading removed
`ThreadPoolExecutor` in the pipeline was pointless — every `generate_with_logprobs()` call acquires a `threading.Lock` (necessary because `eval()` + logit extraction isn't thread-safe on a single GPU context). Threads just queued up sequentially with extra overhead. Removed parallel execution path entirely. Lock kept as safety net.

### Model discovery via Ollama manifests
`find_model_path()` now parses Ollama manifests at `~/.ollama/models/manifests/registry.ollama.ai/library/{model}/{tag}` to find the correct blob, rather than just picking the largest file. Supports `model_name` in config for testing different models.

### Context window limits
Some QuALITY articles exceed n_ctx=8192 when combined with question + options. Added assertion that raises `ValueError` with a clear message rather than silently truncating. On the 3070, 8192 is the sweet spot. On rented 5090s, can increase significantly (Qwen3 training context is 40,960).

### Insufficient context: early signal
Smoke test: sufficient context → correct (B, p=0.80). Insufficient context (swapped article) → wrong answer (C, p=0.84). Model is confident but wrong — exactly the scenario our uncertainty signals should detect. N=1, but encouraging.

### Dataset merged
`quality_train.jsonl` (300 articles, 2523 questions) + `quality_dev.jsonl` (230 articles, 2086 questions) → `quality_all.jsonl` (530 articles, 4609 questions). Zero article_id overlap. Config default and all YAML configs updated.

---

## Decision Log (updated)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-25 | Pivot to pre-action UQ | MMLU results not compelling; new direction = real unsolved problem |
| 2026-03-25 | Archive v2, fresh structure | Clean slate for new direction |
| 2026-03-25 | BFCL for tool-calling component (RQ3) | Irrelevance detection + missing params = proceed/clarify/escalate labels |
| 2026-03-25 | Reconsidered MuSiQue — not a great fit | Tests multi-hop reasoning chains, not comprehension of input |
| 2026-03-25 | RGB as starting dataset for RQ2 | Negative rejection testbed = exactly our scenario |
| 2026-03-25 | RQ2 (context sufficiency) is the load-bearing RQ | Cleanest operationalisation, strongest motivation |
| 2026-03-30 | RGB dropped, replaced by QuALITY | See March 30 dataset notes above |
| 2026-03-30 | `logits_all=False` + low-level extraction | Saves ~4.7GB VRAM. Required to run on 8GB GPUs |
| 2026-03-30 | Chat template with "Answer:" as assistant prefill | Without it, model produces garbage logprobs (top token "GSM") |
| 2026-03-30 | `memory_clear(True)` for KV cache reset | `reset()` doesn't actually clear the cache — causes batch decode failures |
| 2026-03-30 | Remove threading from pipeline | Lock makes parallelism fake. GPU is the bottleneck, not CPU |

---

## March 31, 2026 — Full Experiment Runs + CSC Mahti

### Experiment design finalised
- **Factorial:** prompt_mode (direct/cot) x shuffle (on/off) x context (sufficient/insufficient) = 8 conditions
- `num_paraphrases` renamed to `num_permutations` for clarity. Shuffle=on → 10 permutations per question. Shuffle=off → 1 (identical prompt = identical logprobs, no point repeating)
- `n_ctx` bumped from 8192 to 12288 on vast.ai (42% of articles overflowed at 8192, 0% at 12288). 32768 on Mahti A100 (plenty of headroom)

### Running experiments on two platforms simultaneously
- **vast.ai RTX 5090** ($0.50-0.80/hr): Direct mode experiments. 3 of 4 direct conditions complete (~24MB results each). Shuffle+sufficient still running (~2000/4609 done)
- **CSC Mahti A100 40GB** (university allocation, 1000 BU): CoT experiments. Two noshuffle CoT jobs running in parallel on separate A100 nodes

### CoT two-pass: answer leaking fixed
Initial CoT runs showed all logprobs at exactly 1.000 — the model was writing "The correct answer is C" in its reasoning, so Pass 2 just confirmed what the reasoning already committed to. This is the scaffolding absorption problem from v2.

**Fix:** Added v2's anti-leak instruction to the CoT prompt: `"BE CONCISE. 3-4 bullet points of reasoning only -- do NOT name the answer letter in your reasoning."` After fix, reasoning traces are clean bullet points without answer letters, and Pass 1/Pass 2 answers can now disagree (the interesting case).

### CoT Pass 1 answer capture
`generate_cot()` now lets the model generate freely (no stop sequence), parses the answer letter from the last `"Answer: X"` pattern, then strips everything from "Answer:" onwards before feeding reasoning to Pass 2. This gives us two signals per CoT question: what the model freely chose (Pass 1) and the pre-commitment probability distribution (Pass 2 logprobs).

### Interruptible GPU resume system (vast.ai)
Built a resume system for vast.ai interruptible instances. Key bugs found and fixed:
- **CRLF in YAML configs** caused `run_name` to include `\r`, breaking all file matching (script always started fresh instead of resuming)
- **No PID lock file** meant autorun.sh launched duplicate instances on every resume, creating fragment result files
- **`json.load` on truncated files** crashed the resume logic. Fixed with partial JSON recovery
- **Fix:** `grep -c` byte-counting for completion detection, PID lock at `/tmp/`, CRLF stripping via `tr -d '\r'`

### CSC Mahti deployment
Ported pipeline to the university supercomputer. Key differences from vast.ai: SLURM batch jobs instead of long-running containers, no internet on compute nodes, model copied to NVMe per-job.

Issues hit (all solved):
1. `module` command not available in SLURM batch scripts — need `source /appl/profile/zz-csc-env.sh` first
2. llama-cpp-python CPU-only wheel installed by default (CUDA 11.5 has no pre-built wheel). Had to build from source on a compute node (login node fails because `libcuda.so` only exists on GPU nodes)
3. YAML files with Windows em-dash byte (0x97) crash Python's YAML parser on Linux
4. `srun` doesn't inherit the bash environment (modules, venv) — removed, just call `python3` directly

### Dashboard rebuilt
New Streamlit dashboard with: progress+timing per run, per-run distribution histograms (2 per row), dynamic accuracy table with grouped headers (Overall/Easy/Hard x Accuracy/Confidence), calibration reliability diagrams, effect analysis with matched-pair comparisons, question explorer with per-run results and CoT reasoning traces.

---

## Decision Log (updated)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-25 | Pivot to pre-action UQ | MMLU results not compelling; new direction = real unsolved problem |
| 2026-03-25 | Archive v2, fresh structure | Clean slate for new direction |
| 2026-03-25 | BFCL for tool-calling component (RQ3) | Irrelevance detection + missing params = proceed/clarify/escalate labels |
| 2026-03-25 | Reconsidered MuSiQue — not a great fit | Tests multi-hop reasoning chains, not comprehension of input |
| 2026-03-25 | RGB as starting dataset for RQ2 | Negative rejection testbed = exactly our scenario |
| 2026-03-25 | RQ2 (context sufficiency) is the load-bearing RQ | Cleanest operationalisation, strongest motivation |
| 2026-03-30 | RGB dropped, replaced by QuALITY | See March 30 dataset notes above |
| 2026-03-30 | `logits_all=False` + low-level extraction | Saves ~4.7GB VRAM. Required to run on 8GB GPUs |
| 2026-03-30 | Chat template with "Answer:" as assistant prefill | Without it, model produces garbage logprobs (top token "GSM") |
| 2026-03-30 | `memory_clear(True)` for KV cache reset | `reset()` doesn't actually clear the cache — causes batch decode failures |
| 2026-03-30 | Remove threading from pipeline | Lock makes parallelism fake. GPU is the bottleneck, not CPU |
| 2026-03-31 | 8 factorial configs: direct/cot x shuffle/noshuffle x suf/insuf | Full experimental design covering prompt mode, answer variation, context quality |
| 2026-03-31 | n_ctx=12288 (vast.ai) / 32768 (Mahti) | 42% overflow at 8192. 12288 covers all articles. 32768 for CoT headroom on A100 |
| 2026-03-31 | CoT anti-leak prompt instruction | Model was writing "The correct answer is C" in reasoning, contaminating Pass 2 |
| 2026-03-31 | Dual-platform execution (vast.ai + Mahti) | Direct experiments on rented GPU, CoT on university cluster. Run in parallel |

---

## April 2, 2026 — Dedup Fix, Insufficient Dropped, CoT Shuffle Running

### Duplicate results in CoT noshuffle runs (diagnosed + fixed)
Both CoT noshuffle result files (sufficient: 9178 entries, insufficient: 9168) had nearly 2x the expected 4609 entries. Investigation:
- Data shows two separate inference runs interleaved in alternating 10-entry blocks (matching incremental save interval)
- First run saved 4569/4559 entries, then died. Second run resumed, carried over results, but reprocessed all questions instead of just the remaining ~40
- Confirmed by timestamps: filename from run 1 (20:18 UTC), JSON timestamp from run 2 (20:22 UTC), ~4 min gap = ~40 questions at 6s/q
- Both copies are valid inferences (different CoT reasoning, 5.5% answer disagreement) — not byte-identical dupes
- Exact filtering failure mechanism unclear (code looks correct, IDs match). Likely related to early SLURM script issues (`srun` environment inheritance, first deployment bugs)
- **Fix:** Three layers of duplicate prevention now in place (pre-loop filter, belt-and-suspenders skip, per-question set update). Data deduped: kept first occurrence of each question_id.

### RNG seeding bug on resume (fixed)
`_make_question_rng(seed, idx)` used loop position from the FILTERED list. On resume, remaining questions got different RNG seeds → different shuffle permutations. Fixed by building a stable index map from question_unique_id → original position BEFORE filtering. Critical for the cot_shuffle run which will need multiple resume cycles.

### Insufficient context configs removed
Dropped all 8 insufficient configs (4 regular + 4 Mahti). Sufficient-only going forward — insufficient data already collected is enough as supporting evidence, not the main experiment.

### CoT shuffle sufficient: running on Mahti
- Job 6213699 submitted with `--time=36:00:00`
- Config verified: cot, shuffle=true, sufficient, 10 permutations, n_ctx=32768
- Early results (50 questions): 80% accuracy, 97.6% Pass 1/2 agreement, no duplicates, clean reasoning traces
- Estimated ~77h total compute, will need 2-3 submissions with auto-resume

### Replaced two-pass anti-leak with single-pass natural reasoning
Old architecture: CoT anti-leak prompt ("do NOT name the answer letter") + separate Pass 2 with stripped reasoning. New: model reasons freely, logprobs extracted at the answer position conditioned on FULL reasoning output. Same method for CoT (visible reasoning) and think mode (internal `<think>` blocks). Simpler, more natural, consistent across modes.

### Think mode: scaffolding absorption confirmed
- 85.5% accuracy (vs 75% direct) — think improves answers
- But 802/820 questions at >99% confidence. Logprobs nearly always 1.0 — completely uninformative for UQ
- The think block always commits ("So the answer is C") before the answer token, collapsing the distribution
- **Key exception:** 2/830 Pass 1/2 disagreements (`61405_DPAPHI73_8`, `60507_QOO9BH2K_10`). Both cases show genuine deliberative uncertainty in the think block ("not 100% sure", flip-flopping between options). The uncertainty survived into the logprobs (0.89/0.11 and 0.65/0.35). Worth revisiting with full results — rare cases where think mode logprobs ARE informative may correspond to detectable hedging patterns.

### CoT natural: conciseness preserves uncertainty
- The "3-4 bullet points" instruction naturally prevents scaffolding absorption — model doesn't have space to commit before "Answer:"
- CoT logprobs show real variance: mean 0.938, min 0.437 (vs think mean 0.997)
- 96.5% Pass 1/2 agreement — 3.5% disagreements are temperature sampling cases
- The conciseness constraint is the actual anti-leak mechanism, not the explicit instruction we dropped

### Dashboard fixes
- Progress bar: clamp pct to [0, 1.0] (was crashing on pre-dedup files with count > total)
- CoT explorer: Pass 1/2 comparison now in canonical space (was comparing display vs canonical letter)
- Effects tab: pairwise mode comparisons (CoT vs direct, think vs direct, think vs CoT) + MSP delta
- Think mode shows as its own mode in all tables (not "direct + think=true")
- All accuracy tables sorted by descending accuracy

---

## Decision Log (updated)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-02 | Drop insufficient context configs | Enough data collected, not our main experiment |
| 2026-04-02 | Single sbatch instead of submit_all.sh | Prevents accidental double-submission that may have caused duplicates |
| 2026-04-02 | 3-layer duplicate prevention | Belt-and-suspenders: pre-filter + in-loop skip + per-question set tracking |
| 2026-04-02 | Stable RNG seeding via original index map | Ensures shuffle permutations are reproducible across resume cycles |
| 2026-04-02 | Replace two-pass anti-leak with single-pass natural reasoning | Anti-leak was artificial; conciseness instruction naturally prevents absorption. Consistent extraction across CoT and think modes |
| 2026-04-02 | Think mode as separate experimental condition | Tests Qwen3 internal reasoning vs visible CoT vs direct. Finding: improves accuracy but destroys logprob uncertainty signal |

---

## Next Steps
### Completed
- [x] Dataset: QuALITY (530 articles, 4609 questions, MCQ format)
- [x] Inference layer: llama-cpp-python, low-level logit extraction, chat template
- [x] Pipeline: experiment runner, incremental save, resume, CoT two-pass
- [x] Dashboard: progress, distributions, accuracy, effects, question explorer
- [x] Direct noshuffle sufficient + insufficient — DONE (vast.ai)
- [x] Direct shuffle sufficient — DONE (vast.ai)
- [x] Direct shuffle insufficient — DONE (vast.ai, 820q partial)
- [x] CoT noshuffle sufficient + insufficient — DONE (Mahti, deduped)
- [x] Mahti deployment, smoke tests, SLURM scripts

### In progress
- [ ] **CoT shuffle sufficient — Mahti** (50/4609, ~77h total, 2-3 resume cycles)

### Remaining
- [ ] Compute signals on completed results, validate discriminative power
- [ ] Key question: does sufficient vs insufficient show different uncertainty profiles?
- [ ] Build partial (C3) and counterfactual (C4) context conditions

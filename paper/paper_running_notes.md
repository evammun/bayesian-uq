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

## April 13, 2026 — Paraphrase Experiments + Completed Runs

### Completed experiments
- **CoT shuffle sufficient** — DONE (4609q, 78.6% acc, MSP 0.904)
- **Think noshuffle sufficient** — DONE (4609q, 80.3% acc, MSP 0.997)
- **Think shuffle sufficient** — in progress (1130/4609, resuming)
- **Direct noshuffle paraphrase** — in progress (40/4609, early results)
- **Direct shuffle paraphrase** — queued

### Paraphrase generation: Qwen vs Sonnet
Generated two banks: Sonnet API ($19, 3A+3B+4C) and Qwen on Mahti (free, 3A+7B). Key findings from 100-question Sonnet test:
- **Category C (angle shifts) is broken** — 54.5% accuracy vs 70% baseline. Counterfactuals and negation change the question, not just the phrasing.
- **A+B are safe** — 68% accuracy (close to baseline). Meaning preserved.
- Qwen paraphrases are formulaic but safe. Cat A = "According to the passage, [same question]". Cat B = basic synonym swaps.

### Paraphrase results confirmed (n=190)
Qwen paraphrases track baseline: 82.1% vs 81.6%, 95.3% agreement, epistemic 0.056. Shuffle on same questions: 87.4%, epistemic 0.136. Paraphrasing adds almost no diagnostic value — shuffle is 2-3x more effective at surfacing uncertainty.

### Dashboard: signal battery + epistemic decomposition
Built Signal Battery tab with 9 signals across 5 tables:
- **Single-query:** MSP, 2nd Gap, Coverage. Finding: 2nd Gap beats MSP as discriminator.
- **Multi-query:** Epistemic, Conf Var, Agree-Conf. Finding: epistemic strongest, think shuffle +0.32.
- **Cross-mode disagreement:** CoT vs direct disagree on 14% of questions → 44.9% accuracy on those. Strongest single signal found.
- **Position loyalty:** incorrect answers are more position-dependent (+0.09 to +0.15 delta).
- **Reasoning signals:** think trace length +1976 chars when wrong (2x longer). CoT P1/P2 disagreement 5.3x higher on incorrect.

Added epistemic/aleatoric decomposition (mutual information) to replace simple agreement as the primary multi-query uncertainty measure.

---

## Decision Log (updated)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-02 | Drop insufficient context configs | Enough data collected, not our main experiment |
| 2026-04-02 | Single sbatch instead of submit_all.sh | Prevents accidental double-submission that may have caused duplicates |
| 2026-04-02 | 3-layer duplicate prevention | Belt-and-suspenders: pre-filter + in-loop skip + per-question set tracking |
| 2026-04-02 | Stable RNG seeding via original index map | Ensures shuffle permutations are reproducible across resume cycles |
| 2026-04-13 | Epistemic decomposition as primary UQ metric | Mutual information (total - aleatoric entropy) discriminates correct/incorrect better than simple agreement |
| 2026-04-13 | Signal battery dashboard tab | 9 signals across single-query, multi-query, cross-mode, position, and reasoning categories |
| 2026-04-13 | Paraphrase adds minimal value | n=190: paraphrase agreement 95%, epistemic 0.056 vs shuffle 0.136. Shuffle disrupts position bias; paraphrase doesn't change cognitive path |
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

---

## April 14-16, 2026 — Final Experiment Runs, All Paraphrase/Shuffle Jobs

### Completed experiments (all 4609q)
- **direct noshuffle paraphrase** — DONE (Mahti, Qwen paraphrase bank)
- **direct shuffle paraphrase** — DONE
- **cot shuffle paraphrase** — DONE
- **think shuffle** — 4440/4609 as of Apr 16, final 169q running (job 6335112, 6h wall)

### Job babysitting notes
- Mahti gpusmall = 36h max wall; gpumedium similar. Long jobs (cot+think with shuffle) need 2-3 resume cycles.
- 3-layer dedup + stable RNG (from Apr 2 fix) held up across all resumes — no duplicate entries in final files.
- Direct paraphrase jobs finished in 2h resubmits comfortably (~130q remaining each).
- Think shuffle hits wall time repeatedly — slowest mode (~80s/q with shuffle), needs multiple resubmissions.

### Config details
- All Mahti configs use `paraphrases_file: data/paraphrases_qwen.json` (free, 3A+7B format)
- Sonnet paraphrase bank abandoned (Cat C angle shifts broken) — kept only for comparison
- Config files live in `experiments/configs/mahti/`, one YAML per run_name

### Outstanding work (post-data-collection)
- Full AUROC table — every signal × every condition
- Selective prediction curves (AUARC) — practical metric for routing
- Easy vs hard subset breakdown (QuALITY provides this split)
- N-query sensitivity (N=2,3,5,7,10 from existing 10-permutation data)
- Think trace content analysis (hedging phrase counting)
- Calibration reliability diagrams (ECE per condition)
- Adaptive escalation Pareto frontier (see brainstorm §8)

---

## Decision Log (updated)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-14 | Qwen paraphrases (3A+7B) over Sonnet (3A+3B+4C) | Free, safe, no Cat C question-altering issues. Qwen formulaic but semantically stable. |
| 2026-04-16 | Mahti wall-time strategy: 36h for shuffle modes, 2-6h for tail resubmits | Matches gpusmall limit; avoids wasted allocation on small tails |
| 2026-04-23 | Temperature scaling T=3.0 for Bayesian posterior | Raw per-query logprobs have ECE=0.186 — posterior concentrates in 1-2 samples without scaling. T=3.0 gives ECE=0.012. Note: mean probs (dashboard reliability diagram) are better calibrated; per-query probs (what posterior uses) are the overconfident ones. |
| 2026-04-23 | Joint (T, τ) optimization, not calibration-only | T=3.0 is ECE-optimal but not deployment-optimal. Real objective: total_cost = avg_N + cap_rate × esc_cost. Grid search finds Pareto frontier. |
| 2026-04-23 | Claude Code workflow: Opus for architecture, Sonnet for implementation | Sonnet as capable intern — trust but verify. Have it build tests. Opus for judgment calls, synthesis, high-blast-radius decisions. |

---

## April 23, 2026 — Adaptive Sampling Framework (Bayesian Posterior)

### Context
Luigi showed results from his dataset: Bayesian adaptive sampling with posterior stopping. Instead of running all N shuffle permutations, stop when posterior exceeds τ, escalate capped questions to expensive mode. We retroactively tested this on our existing 10-permutation direct shuffle data.

### Method: Bayesian posterior over MCQ answers
- Prior: uniform P(k) = 1/4 over 4 options
- Each shuffle permutation gives a full probability vector from logprobs
- Posterior update: P(k | samples 1..N) ∝ prior(k) × ∏ₙ pₖ⁽ⁿ⁾ (multiply likelihoods, normalise)
- No Dirichlet needed — we have full probability vectors, just multiply and normalise in log-space
- Stop when max posterior > τ; if N_max reached without crossing τ → "capped" → escalate

### The overconfidence problem
- Raw per-query logprobs: MSP ≈ 0.91, ECE = 0.186
- Posterior concentrates exponentially on 4-option MCQ → MSP=0.70 crosses τ=0.95 in 2-3 consistent samples
- Without calibration: cap_rate ≈ 0% at all τ, no escalation ever triggers, framework is useless
- **Important distinction:** mean probs across permutations (dashboard reliability diagram) are reasonably calibrated. Per-query probs (what the posterior multiplies) are severely overconfident. Both facts are true simultaneously.

### Temperature scaling
- Calibration: new_probs = softmax(log(old_probs) / T), T > 1 deflates confidence
- Grid search over T: optimal T=3.0 (ECE: 0.186 → 0.012, NLL also improves)
- Dashboard default updated to T=3.0
- At T=3.0, N_max=5, τ=0.95: avg_N=2.76, acc=76.4%, cap_rate=15.1%, acc+think=78.8%

### Joint optimization insight
- T=3.0 optimises calibration (ECE), not deployment cost
- Real objective: total_cost = avg_N + cap_rate × escalation_cost
- Higher T → slower convergence → higher avg_N, fewer caps → less escalation cost but more base cost
- Lower T → faster convergence → lower avg_N, more caps (some wrong) → more escalation cost
- Added joint (T × τ) grid search to dashboard: 9 temperatures × 6 τ values, Pareto frontier of accuracy vs total cost
- Escalation cost configurable via slider (CoT ≈ 10× base, think ≈ 24× base)

### Signal-augmented stopping (designed, not yet implemented)
Beyond pure posterior, additional veto signals for the stopping decision:
- **2nd Gap** < threshold → don't stop even if posterior is high (runner-up too close)
- **Argmax flips** — if the leading answer changed during the N samples (15.1% of questions), don't stop
- **Epistemic uncertainty** (mutual information across permutations) > threshold → keep sampling
- **Confidence variance** across permutations > threshold → unstable, keep sampling
- These address the posterior's blind spot: consistency ≠ correctness. Model can be consistently wrong.
- Tier 2 (future): cross-mode disagreement, think trace hedging phrase count

### Dashboard additions (Tab 7: Adaptive Sampling)
- Two methods compared side by side: **Product** (multiply likelihoods) and **Sum** (Dirichlet pseudo-counts)
- Product: Bayesian posterior update, max posterior as stopping criterion, requires temperature scaling (T=3.0)
- Sum: accumulate prob vectors as pseudo-counts, Bonferroni exceedance probability as stopping criterion, no temperature needed
- Exceedance function uses `scipy.special.betainc` — Bonferroni lower bound, conservative but fast (3 beta function calls per step vs MC sampling)
- Side-by-side HTML tables, combined accuracy-vs-compute chart (teal=Product, rose=Sum)
- Rich "How it works" expanders for each method explaining the math, tradeoffs, and implementation
- **Joint Optimization section**: Product sweeps T×τ, Sum sweeps τ only, both on same Pareto frontier chart with escalation cost slider
- Pareto tables side by side for both methods

### Luigi's Bayesian library comparison (April 23) — now integrated
- Luigi uses **Dirichlet Sum** and **Dirichlet MLE** (Minka's fixed-point). We now have all four: Product, Sum, MLE, Composite.
- Luigi uses **exceedance probability** (P(leader is true mode)) not max posterior. We use exceedance for Sum + MLE (copula), max posterior for Product, vetoed posterior for Composite.
- Key insight: Product is theoretically correct but breaks on overconfident logprobs (need temperature). Sum/MLE sidestep this because evidence accumulates linearly/via distribution fitting.
- **All Luigi items now integrated:** MLE ✓, copula exceedance ✓, augmented escalation ✓, regularisation ✓.
- Paper comparison: Product (with temp) vs Sum (no temp) vs MLE (no temp) vs Composite, Pareto frontier in dashboard.

### Next steps (adaptive) — updated Apr 23 evening
- [x] ~~Implement signal-augmented stopping~~ — done: Composite method (gap + flip veto)
- [x] ~~Implement Dirichlet MLE~~ — done: Minka 2000, batch-vectorised (1.5s for 18K fits)
- [x] ~~Implement Luigi-style escalation~~ — done: augment α with esc prob vector, both CoT + think
- [x] ~~Per-query cost instrumentation~~ — done: inference_time_s, prompt_tokens, output_tokens. Validated in Gemma 4 test run.
- [x] ~~Multi-model: Gemma 4~~ — 100-question test passed, 72% accuracy. Ready for full runs.
- [ ] Multi-model: Qwen 3.5 — blocked on llama.cpp recurrent memory bug. Options: wait for upstream fix, try different quant, or use transformers+bitsandbytes
- [ ] Launch full Gemma 4 runs (all conditions) on Mahti
- [ ] Determine optimal (T, τ) for specific deployment scenarios (cheap-and-fast vs max-accuracy)

### Claude Code workflow note
- **Opus** for: architecture decisions, judgment calls, synthesis, reviewing Sonnet's work, anything where getting it wrong costs time
- **Sonnet** for: implementation, boilerplate, data exploration, running tests — treat as a capable intern nearing end of internship. Don't hand-hold, do verify. Have it write tests for its own code.
- Pattern: Opus designs and specifies → Sonnet implements → Opus reviews → ship

---

## April 23, 2026 (afternoon) — Luigi's Methods + Multi-Model Prep

### Integrated Luigi's Bayesian library improvements
Reviewed Luigi's full `bayesian.py` (1321 lines) and exceedance approximation doc. Three changes integrated into dashboard:

**1. Damped Gaussian-copula exceedance** — replaces Bonferroni bound for Sum and MLE methods.
- Exact pairwise Beta marginals + first-order Gaussian copula correction with K-dependent damping
- Damping: `d(K) = 0.637 + 0.206 × exp(-0.587 × (K-3))`, calibrated against MC ground truth
- K=4 validation: errors ±0.002-0.004 vs MC (Bonferroni was -0.028 to -0.382)
- The K=4 moderate case is striking: Bonferroni gives 0.062 (useless), copula gives 0.440 (near-exact 0.444)
- From Luigi's derivation: the first-order truncation benefits from error cancellation — higher-order terms converge toward the wrong (MVN) target

**2. MLE regularisation** — uniform pseudo-observation + label smoothing.
- `suff_stats = (log_p_sum + prior_strength × log(1/K)) / (N + prior_strength)`, prior_strength=1.0
- Label smoothing: `p_smooth = ε/K + (1-ε)p`, ε=10⁻³ — prevents log(0) without distorting high-N fits
- Luigi's design from his `dirichlet_mle()`: standard Bayesian regularisation

**3. Luigi's trigamma/inverse-digamma** — pure numpy, replaces scipy.special.polygamma.
- `_trigamma()`: recurrence + asymptotic series (Abramowitz & Stegun 6.4.12)
- `_inverse_digamma()`: Minka 2003 initialisation + 8 Newton iterations
- Used in both per-question and batch MLE paths

### 2nd Gap replaced by MLE (earlier this session)
- Luigi called 2nd Gap a "hack" — not wrong, the gap signal is captured by the other methods
- MLE is the formally correct approach: fits the Dirichlet concentration from observed variation
- Batch-vectorised: 18,436 MLE fits in 1.5s (was 60s+ per-question loop)

### Per-query cost instrumentation added
- Added `inference_time_s`, `prompt_tokens`, `output_tokens` to `QueryResult` (config.py)
- Wrapped inference calls with `time.time()` in inference.py (both direct and two-pass paths)
- Token counts extracted from existing llama-cpp-python internals (already computed, just not saved)
- Zero overhead — fields are Optional with None defaults for backward compat
- Not yet collected — will populate on next experiment runs (Qwen 3.5 9B, Gemma 4)

### Multi-model expansion: Gemma 4 working, Qwen 3.5 blocked
- **Gemma 4 E4B (Q4_K_M):** 100-question test run PASSED. 72% accuracy, all logprobs valid, mean 0.96s/question on A100.
- **Qwen 3.5 9B (Q4_K_M):** BLOCKED by llama.cpp bug. `llama-memory-recurrent.cpp:544` assertion failure — Qwen 3.5's hybrid recurrent attention architecture not properly supported. Crashes after 1-2 questions. Tested on both llama-cpp-python 0.3.19 and 0.3.36 (JamePeng fork). Pipeline code is fine (the 2 questions it processed have valid logprobs and timing).
- Chat templates implemented: Qwen 3 (ChatML + `/no_think`), Qwen 3.5 (ChatML + plain-text "answer directly"), Gemma 4 (`<start_of_turn>`/`<end_of_turn>`)
- Model family auto-detection from GGUF filename stem
- SLURM script made model-generic via `MODEL_FILE` env var
- **Bug fix:** `completed_at` timestamp was never written because incremental writer skips when no new results to append. Added `finalize()` method.
- **Gemma 4 observation:** Very overconfident — mean max-prob confidence 0.961 with 72% accuracy. Even wrong answers have 0.894 mean confidence. Expected for single-query direct mode, but more extreme than Qwen 3. Good signal for the paper: different models have different calibration profiles under the same pipeline.
- **llama-cpp-python upgraded to 0.3.36** (JamePeng fork) on Mahti. Built from source with CUDA 80 (A100). Fixes Gemma 4 support but not Qwen 3.5.

### Dashboard multi-model refactor
- **Model always visible:** All 3 models shown simultaneously, no model dropdown. Condition is the filtered dimension.
- **Model colors:** Qwen 3 = teal, Gemma 4 = rose, Qwen 3.5 = amber. Consistent across all charts.
- **Labels:** `Qwen 3 · direct · noshuffle` (removed "sufficient" — vestigial from dropped insufficient experiments)
- **Tab 1 Progress:** Split into In Progress / Completed sections. ETA now computed from actual `inference_time_s` per query (reads file tail), not dashboard refresh deltas.
- **Tab 3 Condition Comparison:** Model is first column in accuracy table. Calibration diagram: color = model, linestyle = condition.
- **Tab 5 Effect Analysis:** Model column in summary table. Violin plot colored by model. Matched pairs within model.
- **Tab 6 Signal Battery:** Model column in all signal tables.
- **Deleted:** All insufficient result files (local + Mahti), insufficient configs, hide_insufficient toggle.

### Full Gemma 4 runs launched on Mahti
- 4 conditions: direct × {noshuffle, shuffle, noshuffle+para, shuffle+para}
- All sufficient-only (no insufficient)
- Jobs: 6383926 (noshuffle ~75 min), 6383927/28/29 (10-perm, ~12h each)
- Early validation at 720 questions: 81.5% accuracy, all logprobs valid, mean 1.03s/q

### Decision Log (updated)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-23 | Damped Gaussian-copula exceedance replaces Bonferroni | RMSE 0.005 vs 0.04 for K=4. Luigi's derivation + validation. Conservative Bonferroni was giving misleading low exceedance values. |
| 2026-04-23 | MLE regularisation with prior_strength=1.0 | Stabilises low-N fits. Luigi's approach: uniform pseudo-observation in sufficient statistics. |
| 2026-04-23 | 2nd Gap dropped, replaced by Dirichlet MLE | Gap is a heuristic captured by other methods. MLE is the principled approach — fits concentration from observed variation. |
| 2026-04-23 | Augmented escalation (not replace) | α_aug = α_cap + p_esc preserves accumulated evidence. Luigi's design. Applied to both CoT and think. |
| 2026-04-23 | Keep pipeline as-is for multi-model, retroactive adaptive | Always better to have more data. Adaptive stopping computed retroactively from full permutation data. |
| 2026-04-23 | Per-query timing instrumentation | Need cost baselines for each query to compare adaptive approaches. inference_time_s + token counts. |
| 2026-04-23 | llama-cpp-python 0.3.36 (JamePeng fork) on Mahti | PyPI 0.3.20 too old for Qwen 3.5 and Gemma 4 architectures. JamePeng fork built from source with CUDA. |
| 2026-04-23 | Qwen 3.5 fixed — `llama_memory_clear` workaround | `llama_memory_seq_rm` crashes on recurrent cells. `llama_memory_clear` (full wipe) works without latency penalty. 20/20 test, 90% accuracy. |
| 2026-04-23 | Gemma 4 E4B validated as second model | 72% accuracy, valid logprobs, fast inference (0.96s/q). Very overconfident — good calibration contrast with Qwen 3. |

### Qwen 3.5 fix and launch (same session, later)
- **Root cause:** `llama_memory_seq_rm` iterates recurrent cells and asserts `cell.has_seq_id(seq_id)`, which fails for Qwen 3.5's hybrid DeltaNet architecture.
- **Fix 1 (failed):** Skip `llama_memory_seq_rm`, just reset `n_tokens = 0`. No crash, but 19/20 queries failed — stale recurrent state leaked between questions.
- **Fix 2 (success):** Use `llama_memory_clear(mem, True)` instead — wipes all memory (KV + recurrent) without per-cell seq_id checks. Model-family-gated: only Qwen 3.5 uses this path; other models keep `llama_memory_seq_rm`.
- **Validation:** 20/20 questions processed, 90% accuracy, valid logprobs. No latency impact.
- **Full runs launched on Mahti:** Same 4 conditions as Gemma 4 (direct × {noshuffle, shuffle, noshuffle+para, shuffle+para}). Jobs 6384215-6384218. Pending behind Gemma 4 runs.
- **Gemma 4 progress at time of launch:** noshuffle at 2539/4609 (~55%), shuffle/para runs at ~224/4609 (~5%)

---

## April 23, 2026 (evening) — CoT/Think Modes, Bug Fixes, Dashboard Polish

### Dashboard improvements
- **Calibration reliability diagram split by prompt mode:** Replaced single unreadable chart with 3 separate charts (direct/cot/think). Each 380px, own color scheme per model.
- **Accuracy table heatmaps:** Column-wise min-max blue shading (rgba(42,100,160,alpha)). Model-colored left borders (teal=Qwen3, rose=Gemma4, amber=Qwen3.5). Bold highlighting for best values.
- **Progress tab compaction:** Metric boxes padding/font reduced (22px→16px values, 11px→10px labels) to fit 8 concurrent runs on screen.

### Qwen 3.5 think leak in CoT mode (critical fix)
Qwen 3.5 was generating `<think>` blocks even in CoT mode — 8372-char think blocks in some responses. Template-level suppression fix: empty `<think>\n\n</think>\n\n` block as assistant turn prefix (matches official Jinja template `enable_thinking=false` behavior). Also added CoT-mode think-stripping: Pass 2 eval uses `visible_output` only (strips any spontaneous think blocks). Validated: 0/20 think blocks after fix. Synced to Mahti via scp.

### Incremental writer last-batch data loss (fix)
Gemma 4 direct noshuffle saved only 4600/4609 questions. Root cause: incremental writer saves every 10 questions, `finalize()` only patches `completed_at` without flushing remaining results. Fix: added `writer.write()` before `writer.finalize()` in pipeline.py. Synced to Mahti, then resumed the run — all 4609 questions confirmed.

### CoT + Think production runs launched
4 new conditions submitted on Mahti (no shuffle, no paraphrase, sufficient-only):
- Gemma 4 CoT (job 6384486), Gemma 4 Think (6384487)
- Qwen 3.5 CoT (6384489), Qwen 3.5 Think (6384490)
Think configs use `think: true, prompt_mode: direct`. CoT configs use `think: false, prompt_mode: cot`. All n_ctx=12288.

### Test validation summary
All 4 conditions tested on 20 questions before production launch:
- Gemma 4 CoT: 75%, pass1+pass2 logprobs captured
- Gemma 4 Think: 85%, thinking_trace present in all questions
- Qwen 3.5 CoT (post-fix): 90%, 0 think blocks (was leaking before fix)
- Qwen 3.5 Think: 75%, thinking traces valid

### Measured escalation cost multipliers (preliminary)
From `inference_time_s` in result files (Gemma 4, A100):
- **Direct:** 0.96s/query (n=4609)
- **CoT:** 2.27s/query → **2.4× direct** (n=80)
- **Think:** 6.82s/query → **7.1× direct** (n=20)
- Qwen 3.5 direct: 1.61s/query (n=1840), CoT/Think not yet available
- **TODO:** Update dashboard `ESC_COSTS` with final estimates from full runs (all models, all questions). Current values are Gemma 4 only, small-n for CoT/Think.

Dashboard updated: replaced escalation cost slider with selectbox using measured values. Base run dropdown now allows any direct run (was filtering to shuffle-only).

### Think n_ctx fix
Gemma 4 and Qwen 3.5 think configs had `n_ctx: 12288` — too tight for two-pass think (worst-case Pass 2 ~11.9K tokens). Bumped to `n_ctx: 32768` to match Qwen 3 think setup. Cancelled partial runs (~100q Gemma, ~0q Qwen 3.5), deleted results, resubmitted fresh (jobs 6384556, 6384557).

### Job status at session end
- 7 prior runs still running (Gemma 4 shuffle/para, Qwen 3.5 all 4 direct conditions)
- 4 new CoT/Think runs (2 CoT running, 2 Think resubmitted with n_ctx fix)
- Gemma 4 direct noshuffle: COMPLETE (4609/4609 after resume)


# CLAUDE.md — Project Context for Claude Code

## Project Overview

This is a research project developing a framework for uncertainty quantification (UQ) in LLM outputs. The framework combines paraphrase-based input diversification with logprob extraction to build probability distributions over answer choices without requiring model internals beyond token logprobs. It is being developed as an academic paper and potential foundation for an AI consulting offering.

**Authors:** Eva Martin (lead researcher) and Professor Luigi (supervisor, Bayesian inference and computational neuroscience background).

## Architecture History

**v1 (Dirichlet sampling)** — the original design treated the LLM as a black box returning categorical votes. Each query returned a single answer choice (via JSON schema enforcement), which incremented a Dirichlet posterior. Exceedance probability was computed via Monte Carlo (Gamma sampling trick) and used for adaptive stopping: stop early when confident, keep querying when uncertain. This worked but had a fundamental limitation: with temperature=0 or structured output, the same prompt always gives the same answer, so repeating identical prompts adds no information. The v1 code is fully preserved in `v1_sampling_archive/` (including its own dashboard).

**v2 (logprob extraction)** — the current architecture. Instead of treating each query as a categorical vote, we extract the full logprob distribution over answer tokens from each query. This gives us a rich probability vector per query rather than a single vote. Since logprobs are deterministic per prompt, the signal comes entirely from varying the input (paraphrases + answer reordering) and observing how the distribution shifts. No Dirichlet, no Monte Carlo, no adaptive stopping — every paraphrase is queried exactly once with a fixed budget.

## Core Concept (v2)

- Query LLMs with paraphrased versions of the same question (+ answer reordering for multiple-choice)
- Extract logprobs for each answer token (A/B/C/D) from each query
- Convert logprobs to normalised probability distributions per query
- Aggregate across queries: mean of per-query probability vectors → final distribution
- Final answer = argmax(mean_probs), not majority vote — accounts for confidence magnitude
- Answer shuffling on every query integrates out position bias
- Every paraphrase is queried exactly once (fixed budget, no adaptive stopping)
- Uncertainty metrics (entropy, JSD, epistemic uncertainty) computed post-hoc in analysis.py

## Two Experiments

**Experiment 1 (Standard benchmarks — MMLU Redux 2.0):** Run framework on 5,330 verified-correct MMLU questions. Factorial design: 3 prompt modes × 2 shuffle settings × 2 paraphrase settings = 12 conditions. Compare accuracy, AUROC (does confidence separate correct from incorrect?), and uncertainty metrics across conditions.

**Experiment 2 (Broken-premise detection):** Novel contribution. Paired dataset of MMLU questions, each with a matched broken-premise variant. The broken version modifies one element to invalidate the premise while keeping answer options identical. Six break types: existence violations, contradictory setups, category errors, temporal violations, mathematical impossibilities, entity-framing mismatches. Hypothesis: broken-premise questions show higher entropy, lower agreement, and lower confidence than matched valid questions.

## Tech Stack

- **Python** with Pydantic for data models, NumPy/SciPy for analysis
- **Ollama** for local model inference via HTTP API
- **Primary model:** Qwen 3 8B Q4 (`qwen3:8b-q4_K_M`)
- **Streamlit** dashboard for live monitoring and post-hoc analysis
- **YAML** configs for experiment definitions

## Project Structure

```
src/bayesian_uq/          Core library
  config.py                Pydantic data models (QueryResult, ExperimentConfig, etc.)
  query.py                 OllamaClient — logprob extraction, token matching, API calls
  pipeline.py              Experiment runner — parallel/sequential query loop, incremental save
  analysis.py              Post-hoc metrics — entropy, JSD, epistemic uncertainty

dashboard/
  app.py                   Streamlit dashboard (5 tabs: Progress, Distributions, Comparison, Effects, Explorer)

experiments/
  configs/                 YAML experiment configs (100q pilot + full 5330q, 12 conditions each)
  run_experiment.py        CLI entry point for running experiments

data/
  questions.json           Master question database (MMLU Redux + broken-premise pairs)
  paraphrases.json         Pre-generated paraphrases (5,330 questions × 10 paraphrases)
  mmlu/                    Raw MMLU Redux 2.0 source data
  broken_pairs/            Broken-premise paired dataset

results/                   Experiment output JSON files (git-ignored, can be 100MB+)
paper/                     LaTeX/manuscript drafts
v1_sampling_archive/       Archived v1 Dirichlet sampling code
```

## How It Works Mechanically (v2)

1. Load a question (4 options: A, B, C, D) and its pre-generated paraphrases
2. For each query variant (original + up to 10 paraphrases):
   - Shuffle the answer order (if shuffle enabled) to control for position bias
   - Query the model via Ollama API, extracting logprobs for answer tokens
   - Direct mode: `/api/generate` with `raw: true` (completion-style, no chat template)
   - CoT modes: `/api/chat` with streaming (needs chat template for reasoning)
   - Match tokens " A"/"A" (space-prefixed from generate endpoint) to answer letters
   - Convert raw logprobs to normalised probability vector [P(A), P(B), P(C), P(D)]
   - Map back to canonical answer ordering via the permutation
3. Aggregate: mean of all per-query probability vectors → `mean_probs`
4. Final answer: `argmax(mean_probs)` — better than majority vote because it weights by confidence
5. Save results incrementally to JSON after each question

**Query budget:** With paraphrases off + shuffle off = 1 query per question. With either on = 11 queries (1 original + 10 paraphrases). Direct mode uses 3 parallel workers; CoT modes run sequentially.

## Key Design Decisions

- Raw logprobs stored unmodified alongside normalised `canonical_probs` for full audit trail
- Token matching handles both "B" and " B" (space-prefixed from generate endpoint)
- Missing answer tokens (not in top 20 logprobs) get `exp(-30)` floor before normalisation
- `answer_permutation` maps display positions to canonical indices, stored per query for reproducibility
- Paraphrases generated offline by Claude API, filtered by embedding similarity and lexical distance
- Results saved as detailed JSON so analysis can be re-run without re-querying models
- Use `127.0.0.1` not `localhost` in Ollama URL — Windows IPv6 DNS adds ~2s per request

## Experimental Design

3 prompt modes × 2 shuffle × 2 paraphrase = **12 conditions**:

| Prompt Mode    | Description |
|----------------|-------------|
| `direct`       | Completion-style via `/api/generate` with `raw: true` |
| `cot`          | Chain-of-thought via `/api/chat`, free-form reasoning then answer |
| `cot_structured` | CoT with structured output format |

Each condition has a YAML config in `experiments/configs/`. Naming convention: `exp1_{scale}_{prompt}_{shuffle}_{para}.yaml` (e.g. `exp1_full_direct_shuffle_nopara.yaml`).

## Dashboard

Streamlit dashboard (`dashboard/app.py`) with 5 tabs:

1. **Progress** — per-run status cards, accuracy, timing estimates
2. **Probability Distributions** — per-query confidence histograms, agreement distributions, mean prob analysis
3. **Condition Comparison** — accuracy heatmap by subject category, key metrics table
4. **Effect Analysis** — matched-pair main effects (accuracy Δ, AUROC Δ, confidence Δ, agreement Δ), effect consistency bar charts, interaction spotlight
5. **Question Explorer** — drill into individual questions, per-query probability breakdowns

Auto-refresh (30s) available for monitoring live experiments. Result files can be 100MB+; the sidebar uses lightweight 4KB config extraction to avoid re-parsing full files on every refresh.

## Coding Style

- Clear, readable Python. Prioritise simplicity over cleverness.
- Type hints on function signatures.
- Docstrings on all public functions.
- Keep core library (src/) clean and modular. Experiment scripts can be scrappier.
- When in doubt, ask — don't assume.

## Important Context

- Eva has strong Python skills (MSc in Data Science & AI) but is newer to Bayesian statistics — prefer clear variable names and comments over terse mathematical notation in code.
- The project targets local models on a laptop with an NVIDIA RTX 3070 (8GB VRAM). Performance and memory constraints matter.
- The primary baseline paper is Adaptive-Consistency (Aggarwal et al., EMNLP 2023) — we extend their approach with paraphrase-based input diversification and logprob extraction.
- This is a real research project aimed at publication. Code quality matters because experiments need to be reproducible.

# Experiment Context

## What we're investigating

LLMs give single answers with no confidence measure. When you ask a question, you don't know if the model is certain, guessing, or confidently wrong. We're building a framework that wraps around any LLM and produces calibrated uncertainty estimates by combining logprob extraction with input diversification (paraphrases + answer reordering).

The key insight: if you ask the same question multiple ways and the model's logprob distributions stay consistent, it's probably right. If the distributions shift — different answer gets highest probability depending on phrasing — the model is uncertain, and that inconsistency is a useful signal.

## Two experiments

**Experiment 1 — Standard benchmarks (MMLU Redux 2.0):** Run the framework on 5,330 verified-correct multiple-choice questions across 12 factorial conditions. Measure: does it get the right answer? Does confidence (max mean prob) separate correct from incorrect answers (AUROC)? How do the experimental variables (prompt mode, shuffling, paraphrasing) affect accuracy and uncertainty calibration?

**Experiment 2 — Broken-premise detection (novel contribution):** Paired questions where each pair has a valid MMLU question and a matched broken-premise variant. Example:

- Valid: "What is the primary function of the mitochondria in a cell?" → Answer: Energy production
- Broken: "What is the primary function of the mitochondria in a Monocercomonoides cell?" → Monocercomonoides has no mitochondria. No answer is correct, but the model is forced to pick one.

The hypothesis: on broken-premise questions, the probability distributions will be less consistent across paraphrases (higher entropy, lower agreement, lower confidence) than on matched valid questions. This inconsistency is invisible from a single query but visible when you aggregate across diverse inputs.

Key methodological point (from Luigi): we only analyse broken-premise results for pairs where the model confidently gets the valid version RIGHT. This rules out the alternative explanation that the model is just confused because it doesn't know the topic.

## Technical constraints

- Local models via Ollama on an NVIDIA RTX 3070 (8GB VRAM)
- Primary test model: Qwen 3 8B Q4 quantisation
- Logprob extraction via Ollama API (`top_logprobs: 20`)
- Paraphrases pre-generated offline by Claude API, stored and reused
- All results stored as detailed JSON logs so analysis can be re-run without re-querying models

## Performance and Runtime Considerations

- **Think about runtime before writing loops.** If a task involves 5,000+ items and an API call or model query per item, estimate the total wall-clock time before starting. If it's over 10 minutes, consider parallelisation.
- **Use parallel workers for I/O-bound tasks** like API calls, model queries, and file downloads. Python's `concurrent.futures.ThreadPoolExecutor` for API calls, `ProcessPoolExecutor` for CPU-bound work. Default to 4-8 workers for API calls (respect rate limits).
- **Batch sensibly.** When batching items for API calls, keep batches small enough that one failure doesn't lose much progress (10-20 items), but large enough to be efficient. Never put 200+ items in a single batch.
- **Always save incrementally.** Any long-running script that produces output should write results after every batch or iteration, not just at the end. Design for resumability — check what's already done before starting, skip completed work.
- **Estimate and display progress.** Long-running scripts should print: items completed, items remaining, elapsed time, and estimated time remaining. Use tqdm or manual progress logging.
- **Set timeouts when appropriate.** API calls and model queries should have explicit timeouts. If something is taking 10x longer than expected, it's stuck — don't wait, retry or skip.